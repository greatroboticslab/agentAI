"""The taxonomy authority (judge J-taxon, contract §4.3; lever L12, §8.5): names to
taxa, with every authority response cached and hashed.

The authority is named by the domain config (taxonomy.authority); the kind
implemented here is "gbif_backbone" (the GBIF backbone taxonomy, api.gbif.org
v1, checked 2026-09-28):

  * scientific: GET match_url?name=<query>&strict=true. Accepted when the
    match type is EXACT and the rank is genus or below. The accepted name of
    a synonym is read from the match's own classification (GBIF puts the
    accepted species and genus there), so no second call is needed.
  * vernacular: GET search_url?q=<query>&qField=VERNACULAR&limit=20 (plus the
    config's search_params, e.g. the backbone datasetKey). It is informative
    when a result lies in the kingdom of the targets; it never maps a class
    (a common name alone is ambiguous).
  * overrides (config taxonomy.overrides, project policy, R4) win over both
    and are recorded.
  * version: GET version_url; the backbone's "modified" (else "pubDate").

The cache (funnel/taxonomy_cache.json, format funnel-taxonomy-cache/1) holds
the responses verbatim with the sha256 of each body as received, and one
"resolved" entry per name key. The cluster has no network in jobs, so the
cache is built on the lab (fetch --what taxonomy) and the Resolver is offline
by default: a name the cache lacks is refused with a message naming L12, never
guessed.

Nothing here names a domain: the kingdom, the targets, the "not" lists and the
overrides all come from the config.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import (FunnelError, TaxonomyError, canonical_json, header, read_json, utc,
               write_json_atomic)
from .names import key

AUTHORITY_KINDS = ("gbif_backbone",)
LINEAGE = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
GENUS_OR_BELOW = frozenset({"GENUS", "SUBGENUS", "SECTION", "SUBSECTION", "SERIES", "SUBSERIES",
                            "INFRAGENERIC_NAME", "SPECIES", "SUBSPECIES", "VARIETY", "SUBVARIETY",
                            "FORM", "SUBFORM", "INFRASPECIFIC_NAME", "CULTIVAR", "CULTIVAR_GROUP"})
INFRASPECIFIC = frozenset({"SUBSPECIES", "VARIETY", "SUBVARIETY", "FORM", "SUBFORM", "INFRASPECIFIC_NAME",
                           "CULTIVAR", "CULTIVAR_GROUP"})
MISS_MESSAGE = ("no resolution for %d name(s) in taxonomy_cache.json: run fetch --what taxonomy (lever L12)"
                " [%s]")


def overrides_sha256(overrides):
    return hashlib.sha256(canonical_json(overrides or {}).encode("utf-8")).hexdigest()


def query_variants(name, rules=None):
    """The query strings tried for a name, in order: separators (config
    names.query_rules.separators, default "_" and "-") become spaces, then the
    same without a leading numeric token when one is followed by words
    ("0 dock" -> "dock"). Deterministic; empty names give []."""
    rules = rules or {}
    sep = rules.get("separators") or r"[_\-]+"
    base = re.sub(r"\s+", " ", re.sub(sep, " ", str(name or ""))).strip()
    out = [base] if base else []
    if rules.get("drop_leading_numeric_token", True):
        parts = base.split(" ")
        if len(parts) > 1 and parts[0].isdigit():
            stripped = " ".join(parts[1:])
            if stripped and stripped not in out:
                out.append(stripped)
    for q in list(out):
        parts = q.split(" ")
        # "Genus sp." / "Genus spp." name the genus (a nomenclature convention)
        if len(parts) > 1 and parts[-1].rstrip(".").lower() in ("sp", "spp"):
            g = " ".join(parts[:-1])
            if g not in out:
                out.append(g)
    return out


def _lineage(body):
    return {k: (body.get(k) if isinstance(body.get(k), str) else None) for k in LINEAGE}


def _accepted(rank, lineage, canonical):
    r = (rank or "").upper()
    if r == "SPECIES" or r in INFRASPECIFIC:
        return lineage.get("species") or canonical
    if r == "GENUS":
        return lineage.get("genus") or canonical
    return canonical


def entry_from_match(name, body, via):
    rank = body.get("rank")
    lin = _lineage(body)
    canonical = body.get("canonicalName")
    return {"name": name, "taxon_key": body.get("usageKey"), "canonical": canonical,
            "rank": rank.lower() if isinstance(rank, str) else None, "status": body.get("status"),
            "accepted": _accepted(rank, lin, canonical),
            "accepted_key": body.get("acceptedUsageKey") or body.get("usageKey"),
            "lineage": lin, "match_type": body.get("matchType"), "via": via, "vernacular_candidates": []}


def entry_from_search(name, results):
    first = results[0]
    rank = first.get("rank")
    lin = _lineage(first)
    canonical = first.get("canonicalName")
    cands = []
    for r in results:
        c = r.get("canonicalName")
        if isinstance(c, str) and c and c not in cands:
            cands.append(c)
    return {"name": name, "taxon_key": first.get("key"), "canonical": canonical,
            "rank": rank.lower() if isinstance(rank, str) else None, "status": first.get("taxonomicStatus"),
            "accepted": _accepted(rank, lin, canonical),
            "accepted_key": first.get("acceptedKey") or first.get("nubKey") or first.get("key"),
            "lineage": lin, "match_type": "VERNACULAR", "via": "vernacular", "vernacular_candidates": cands}


def empty_entry(name):
    return {"name": name, "taxon_key": None, "canonical": None, "rank": None, "status": None,
            "accepted": None, "accepted_key": None, "lineage": {k: None for k in LINEAGE},
            "match_type": "NONE", "via": "none", "vernacular_candidates": []}


def _authority(domain):
    tx = domain.section("taxonomy")
    auth = dict(tx.get("authority") or {})
    if auth.get("kind") not in AUTHORITY_KINDS:
        raise TaxonomyError("domain %s: taxonomy authority kind %r is not one of %s"
                            % (domain.name, auth.get("kind"), AUTHORITY_KINDS))
    for k in ("match_url", "search_url", "version_url"):
        if not isinstance(auth.get(k), str) or not auth[k]:
            raise TaxonomyError("domain %s: taxonomy.authority.%s is required" % (domain.name, k))
    return tx, auth


def domain_taxa(domain):
    """Every taxon the config names: targets, their "not" lists, attractors and
    the independent-truth taxa (the cache resolves all of them)."""
    out = []
    for t in domain.targets:
        if t.get("taxon"):
            out.append(t["taxon"])
        out.extend(t.get("not", []) or [])
    for a in domain.attractors:
        out.append(a["taxon"])
    kt7 = (domain.raw.get("known_truth") or {}).get("kt7") or {}
    out.extend(kt7.get("taxa", []) or [])
    for ov in ((domain.raw.get("taxonomy") or {}).get("overrides") or {}).values():
        out.append(ov["taxon"])
    for res in ((domain.raw.get("sources") or {}).get("card_resolvers") or {}).values():
        for row in ((res or {}).get("class_table") or {}).values():
            if isinstance(row, dict) and row.get("taxon"):
                out.append(row["taxon"])
        for tx in ((res or {}).get("names") or {}).values():
            if tx:
                out.append(tx)
    seen, uniq = set(), []
    for t in out:
        if t and key(t) not in seen:
            seen.add(key(t))
            uniq.append(t)
    return uniq


class Resolver(object):
    """Names to cached taxon entries. Offline unless allow_network and a
    transport(url, params) -> (status, bytes, headers) are given."""

    def __init__(self, cache, domain, transport=None, allow_network=False):
        self.domain = domain
        self.tx, self.auth = _authority(domain)
        self.overrides = {key(k): v for k, v in (self.tx.get("overrides") or {}).items()}
        self.informative_via = tuple(self.tx.get("informative_via") or ("scientific", "vernacular"))
        self.mappable_via = tuple(self.tx.get("mappable_via") or ("scientific", "override"))
        self.rules = (domain.raw.get("names") or {}).get("query_rules") or {}
        self.cache = cache if cache is not None else {}
        self.cache.setdefault("responses", {})
        self.cache.setdefault("resolved", {})
        self.transport = transport
        self.allow_network = bool(allow_network and transport is not None)
        self.misses = []
        self._kingdom = None
        self._kingdom_busy = False

    # ---------------------------------------------------------- authority
    def _get(self, kind, query):
        rk = "%s:%s" % (kind, query)
        rec = self.cache["responses"].get(rk)
        if rec is not None:
            if rec.get("status") != 200 or not isinstance(rec.get("body"), dict):
                raise TaxonomyError("cached %s response for %r has status %s" % (kind, query, rec.get("status")))
            return rec["body"]
        if not self.allow_network:
            raise TaxonomyError(MISS_MESSAGE % (1, "%s %r" % (kind, query)))
        if kind == "match":
            url, params = self.auth["match_url"], dict(self.auth.get("match_params") or {"strict": "true"})
            params["name"] = query
        elif kind == "vernacular":
            url = self.auth["search_url"]
            params = dict(self.auth.get("search_params") or {"qField": "VERNACULAR", "limit": "20"})
            params["q"] = query
        elif kind == "match_in_kingdom":
            # "<kingdom>|<name>": the same strict match, restricted to one kingdom
            kingdom, _, name = query.partition("|")
            url, params = self.auth["match_url"], dict(self.auth.get("match_params") or {"strict": "true"})
            params.update(name=name, kingdom=kingdom)
        elif kind == "version":
            url, params = self.auth["version_url"], {}
        else:
            raise TaxonomyError("unknown lookup kind %r" % kind)
        status, data, _headers = self.transport(url, params)
        sha = hashlib.sha256(data or b"").hexdigest()
        if status != 200:
            raise TaxonomyError("%s %s answered %s" % (url, params, status))
        try:
            body = json.loads((data or b"").decode("utf-8"))
        except ValueError as e:
            raise TaxonomyError("%s %s is not JSON (%s)" % (url, params, e))
        if not isinstance(body, dict):
            raise TaxonomyError("%s %s: the body is not a JSON object" % (url, params))
        self.cache["responses"][rk] = {"url": url, "params": params, "status": status, "sha256": sha,
                                       "body": body, "fetched_utc": utc()}
        return body

    def _scientific(self, name, disambiguate=False):
        homonyms = []
        for q in query_variants(name, self.rules):
            body = self._get("match", q)
            if body.get("matchType") == "EXACT" and str(body.get("rank") or "").upper() in GENUS_OR_BELOW:
                return entry_from_match(name, body, "scientific")
            if body.get("matchType") == "NONE" and "multiple equal matches" in str(body.get("note") or "").lower():
                homonyms.append(q)
        if disambiguate and homonyms and not self._kingdom_busy:
            # a name the configuration gives as a taxon (a genus such as one that is also
            # an animal genus) is read in the targets' kingdom: accepted when the answer
            # is that very name, at genus rank or below. Source class names never take
            # this path (a homonym there stays unresolved scientifically).
            kingdom = self.kingdom()
            for q in homonyms:
                body = self._get("match_in_kingdom", "%s|%s" % (kingdom, q))
                if (body.get("kingdom") == kingdom and str(body.get("rank") or "").upper() in GENUS_OR_BELOW
                        and (body.get("matchType") == "EXACT"
                             or (body.get("matchType") == "HIGHERRANK" and key(body.get("canonicalName")) == key(q)))):
                    return entry_from_match(name, body, "scientific")
        return None

    def _vernacular(self, name):
        kingdom = self.kingdom()
        for q in query_variants(name, self.rules):
            body = self._get("vernacular", q)
            res = [r for r in (body.get("results") or []) if isinstance(r, dict) and r.get("kingdom") == kingdom]
            if res:
                return entry_from_search(name, res)
        return None

    def _compute(self, name, scientific_only=False):
        k = key(name)
        ov = self.overrides.get(k)
        if ov is not None:
            e = self._scientific(ov["taxon"], disambiguate=True)
            if e is None:
                raise TaxonomyError("override %r -> %r does not resolve scientifically at the authority"
                                    % (name, ov["taxon"]))
            e = dict(e, name=name, via="override", canonical=ov["taxon"], match_type="OVERRIDE")
            return e
        e = self._scientific(name, disambiguate=scientific_only)
        if e is None and not scientific_only:
            e = self._vernacular(name)
        return e if e is not None else empty_entry(name)

    # ------------------------------------------------------------ public
    def kingdom(self):
        """The kingdom of the targets (every target taxon must resolve, in one kingdom)."""
        if self._kingdom is None:
            ks = set()
            self._kingdom_busy = True           # the targets themselves resolve without a kingdom
            try:
                for t in self.domain.targets:
                    e = self.resolve(t["taxon"], scientific_only=True)
                    if e is None or e.get("via") == "none" or not (e.get("lineage") or {}).get("kingdom"):
                        raise TaxonomyError("target taxon %r does not resolve at the authority" % t["taxon"])
                    ks.add(e["lineage"]["kingdom"])
            finally:
                self._kingdom_busy = False
            if len(ks) != 1:
                raise TaxonomyError("the target taxa lie in more than one kingdom: %s" % sorted(ks))
            self._kingdom = ks.pop()
        return self._kingdom

    def resolve(self, name, scientific_only=False):
        """The cached "resolved" entry of a name (via "none" when nothing
        resolves), or None for an empty name. Offline, a name the cache lacks
        raises TaxonomyError naming lever L12."""
        k = key(name)
        if not k:
            return None
        e = self.cache["resolved"].get(k)
        if e is not None:
            return e
        if not self.allow_network:
            self.misses.append(name)
            raise TaxonomyError(MISS_MESSAGE % (1, name))
        e = self._compute(name, scientific_only=scientific_only)
        self.cache["resolved"][k] = e
        return e

    def target_entry(self, target):
        e = self.resolve(target["taxon"], scientific_only=True)
        if e is None or e.get("via") == "none":
            raise TaxonomyError("target taxon %r does not resolve" % target.get("taxon"))
        return e

    def _not_entries(self, target):
        out = []
        for t in target.get("not", []) or []:
            e = self.resolve(t, scientific_only=True)
            if e is None or e.get("via") == "none":
                raise TaxonomyError("'not' taxon %r of %s does not resolve" % (t, target.get("name")))
            out.append(e)
        return out

    def is_target_synonym(self, res, target):
        """res is the target taxon (accepted name or key equal), or lies below it
        (a descendant at the target's rank: a species of a genus-rank target,
        an infraspecific name of a species-rank target)."""
        if not res or res.get("via") not in self.mappable_via:
            return False
        t = self.target_entry(target)
        if res.get("accepted_key") is not None and res.get("accepted_key") == t.get("accepted_key"):
            return True
        if res.get("accepted") and res.get("accepted") == t.get("accepted"):
            return True
        lin = res.get("lineage") or {}
        if target.get("rank") == "genus":
            return bool(lin.get("genus")) and lin.get("genus") == t.get("accepted")
        rr = (res.get("rank") or "").upper()
        return rr in INFRASPECIFIC and bool(lin.get("species")) and lin.get("species") == t.get("accepted")

    def is_relative(self, res, target):
        """Not the target, but in its genus or in its "not" list. A vernacular
        resolution counts when any of its candidates does (conservative: a
        relative is never relabelled)."""
        if not res or res.get("via") == "none":
            return False
        if self.is_target_synonym(res, target):
            return False
        t = self.target_entry(target)
        tgenus = (t.get("lineage") or {}).get("genus") or str(target.get("genus") or "")
        nots = self._not_entries(target)
        not_names = set()
        not_genera = set()
        for e in nots:
            if (e.get("rank") or "") == "genus":
                not_genera.add(e.get("accepted"))
            else:
                not_names.add(e.get("accepted"))
        names = [res.get("accepted")]
        genera = [(res.get("lineage") or {}).get("genus")]
        if res.get("via") == "vernacular":
            for c in res.get("vernacular_candidates") or []:
                names.append(c)
                genera.append(c.split(" ")[0])
        for n, g in zip(names, genera):
            if g and g == tgenus:
                return True
            if n and n in not_names:
                return True
            if g and g in not_genera:
                return True
        return False

    def lineage_string(self, taxon):
        """"Kingdom Phylum Class Order Family Genus [species]" of a taxon the
        cache resolves (the zero-shot prompt format)."""
        e = self.resolve(taxon, scientific_only=True)
        if e is None or e.get("via") == "none":
            raise TaxonomyError("taxon %r does not resolve; no lineage" % taxon)
        lin = e.get("lineage") or {}
        parts = [lin.get(k) for k in ("kingdom", "phylum", "class", "order", "family")]
        rank = (e.get("rank") or "").upper()
        if rank == "GENUS":
            parts.append(lin.get("genus") or e.get("accepted"))
        else:
            parts.append(lin.get("species") or e.get("accepted"))
        return " ".join(p for p in parts if p)

    def check_complete(self, names):
        """Refuse (TaxonomyError naming L12) when any name lacks a cache entry."""
        missing = sorted({n for n in names if key(n) and key(n) not in self.cache["resolved"]})
        if missing:
            raise TaxonomyError(MISS_MESSAGE % (len(missing), ", ".join(repr(m) for m in missing[:10])))


def load_cache(path, domain):
    """The cache content, after checking its format, its authority (kind and
    URLs as the config names them) and the sha256 of the overrides it was
    built with."""
    try:
        cache = read_json(path)
    except FunnelError as e:
        raise TaxonomyError("%s; run fetch --what taxonomy (lever L12)" % e)
    if cache.get("format") != "funnel-taxonomy-cache/1":
        raise TaxonomyError("%s is not a funnel-taxonomy-cache/1" % path)
    _tx, auth = _authority(domain)
    got = cache.get("authority") or {}
    for k in ("kind", "match_url", "search_url"):
        if got.get(k) != auth.get(k):
            raise TaxonomyError("%s was built against authority %s=%r, the config names %r"
                                % (path, k, got.get(k), auth.get(k)))
    want = overrides_sha256((domain.raw.get("taxonomy") or {}).get("overrides") or {})
    if (cache.get("overrides") or {}).get("sha256") != want:
        raise TaxonomyError("%s was built with other overrides than the config's; rerun fetch --what "
                            "taxonomy (lever L12)" % path)
    for rk, rec in (cache.get("responses") or {}).items():
        body = rec.get("body")
        if not isinstance(body, dict) or not isinstance(rec.get("sha256"), str):
            raise TaxonomyError("%s: response %s has no body or sha256" % (path, rk))
    return cache


def build_cache(names, domain, transport, out_path, prereg=None, inputs=None, testing=False):
    """Lab only (lever L12): resolve every name plus every taxon the config
    names through the authority, and write funnel/taxonomy_cache.json. Returns
    the cache content. prereg: the loaded prereg (the header needs it)."""
    if transport is None:
        raise TaxonomyError("build_cache needs a transport (lab network)")
    if prereg is None:
        raise TaxonomyError("build_cache needs the prereg for the output header")
    _tx, auth = _authority(domain)
    r = Resolver({"responses": {}, "resolved": {}}, domain, transport=transport, allow_network=True)
    version = r._get("version", "")
    backbone_version = version.get("modified") or version.get("pubDate")
    if not backbone_version:
        raise TaxonomyError("the authority's version record has neither modified nor pubDate")
    r.kingdom()
    unresolved = [t for t in domain_taxa(domain) if (r.resolve(t, scientific_only=True) or {}).get("via") == "none"]
    if unresolved:
        # a taxon the configuration names must resolve: the known-truth taxa, the
        # zero-shot prompts' lineages and the card maps are read through it
        raise TaxonomyError("the authority resolves no taxon for the config's %s" % unresolved)
    uniq = []
    for n in names:
        if key(n) and key(n) not in {key(u) for u in uniq}:
            uniq.append(n)
    for n in sorted(uniq, key=key):
        r.resolve(n)
    overrides = (domain.raw.get("taxonomy") or {}).get("overrides") or {}
    from . import taxonomy as _self
    from . import names as _names
    out = header("taxonomy_cache", domain, prereg, inputs or {}, modules=(_self, _names), testing=testing)
    out.update({
        "authority": {"kind": auth["kind"], "match_url": auth["match_url"], "search_url": auth["search_url"],
                      "version_url": auth["version_url"], "backbone_version": backbone_version,
                      "version_fetched_utc": r.cache["responses"]["version:"]["fetched_utc"]},
        "overrides": {"sha256": overrides_sha256(overrides), "entries": overrides},
        "kingdom": r.kingdom(),
        "responses": dict(sorted(r.cache["responses"].items())),
        "resolved": dict(sorted(r.cache["resolved"].items())),
        "names_requested": len(uniq),
    })
    write_json_atomic(out_path, out)
    return out
