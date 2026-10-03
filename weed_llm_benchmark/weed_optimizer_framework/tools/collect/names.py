"""Class names to taxa for the collector (docs/CONTINUOUS_LOOP.md §3.2 "Class
map", §6.2 lever L26).

The resolver is funnel.taxonomy.Resolver used as a library over two caches:
  * the funnel's offline taxonomy cache (INC_DIR/funnel/taxonomy_cache.json,
    format funnel-taxonomy-cache/1), read only and checked by
    funnel.taxonomy.load_cache against the funnel domain config;
  * the collector's own names layer (INC_DIR/intake/names/names_cache.json,
    format collect-names-cache/1), written only by lever L26 (`collect
    names`), which holds the authority's answers for names the funnel cache
    lacks. The funnel's entries win where both hold a name.
Offline (every job), a name neither holds is a miss: the caller records it as
pending (lever L26), never guesses. `collect names` runs on the lab (it needs
the network), resolves a source's pending names through the authority the
funnel domain config names, and appends them to the layer; it never writes
the funnel's directory.

A name's status is funnel.names.status_v2; its target is the target the
resolver calls it a synonym of (Resolver.is_target_synonym). Nothing here
names a domain.
"""
from __future__ import annotations

from pathlib import Path

from . import (FORMATS, CollectError, Refusal, header, in_slurm, inc_dir, intake_dir, intake_lock, read_json,
               sha256_file, utc, write_json_atomic)

LAYER_FORMAT = FORMATS["names_cache"]


def cache_paths(cfg, inc=None):
    tx = cfg.raw["taxonomy"]
    base = inc_dir(inc)
    return base / tx["funnel_cache"], base / tx["names_layer"]


def _authority_record(cfg):
    from ..funnel import taxonomy as T
    _tx, auth = T._authority(cfg.funnel)
    return {"kind": auth["kind"], "match_url": auth["match_url"], "search_url": auth["search_url"],
            "version_url": auth["version_url"]}


def _overrides_sha(cfg):
    from ..funnel import taxonomy as T
    return T.overrides_sha256((cfg.funnel.raw.get("taxonomy") or {}).get("overrides") or {})


def load_layer(path, cfg):
    """The names layer, after checking its format, authority and overrides
    against the funnel domain config (as funnel.taxonomy.load_cache does for
    the funnel's cache)."""
    doc = read_json(path, "names layer")
    if doc.get("format") != LAYER_FORMAT:
        raise CollectError("%s is not a %s" % (path, LAYER_FORMAT))
    want = _authority_record(cfg)
    got = doc.get("authority") or {}
    for k in ("kind", "match_url", "search_url"):
        if got.get(k) != want[k]:
            raise CollectError("%s was built against authority %s=%r, the funnel config names %r"
                               % (path, k, got.get(k), want[k]))
    if (doc.get("overrides") or {}).get("sha256") != _overrides_sha(cfg):
        raise CollectError("%s was built with other overrides than the funnel config's; rerun collect names "
                           "(lever L26)" % path)
    for rk, rec in (doc.get("responses") or {}).items():
        if not isinstance(rec.get("body"), dict) or not isinstance(rec.get("sha256"), str):
            raise CollectError("%s: response %s has no body or sha256" % (path, rk))
    return doc


class Names(object):
    """A resolver over the funnel cache plus the names layer, with the
    status and target of a name."""

    def __init__(self, cfg, resolver, provenance, funnel_keys):
        self.cfg = cfg
        self.resolver = resolver
        self.provenance = provenance
        self._funnel_responses, self._funnel_resolved = funnel_keys
        self.misses = []

    # ---- lookups
    def status(self, name, joined_target=None):
        """funnel.names.status_v2 of a source class name ({"status", "via",
        "taxon", "rank"}) plus "target" (a target name or None) and
        "ambiguous"; None when the caches lack the name (a miss, recorded).
        joined_target: the target the project's alias table joins the name
        to (the funnel's "target" status, via "join"), or None."""
        from ..funnel import TaxonomyError
        from ..funnel import names as N
        try:
            st = N.status_v2(name, bool(joined_target), self.cfg.funnel, self.resolver)
        except TaxonomyError:
            self.misses.append(name)
            return None
        st = dict(st, target=None, ambiguous=False)
        if st["status"] == "target":
            st["target"] = joined_target
        elif st["status"] == "target_synonym":
            res = self.resolver.resolve(name)
            hits = [t["name"] for t in self.cfg.targets if self.resolver.is_target_synonym(res, t)]
            if len(hits) == 1:
                st["target"] = hits[0]
            else:
                st["ambiguous"] = True
        return st

    def knows(self, name):
        from ..funnel.names import key
        return key(name) in self.resolver.cache["resolved"]

    def target_accepted(self):
        """{target name: accepted taxon} for the targets the caches resolve."""
        from ..funnel import TaxonomyError
        out = {}
        for t in self.cfg.targets:
            try:
                out[t["name"]] = self.resolver.target_entry(t).get("accepted")
            except TaxonomyError:
                continue
        return out

    def vernacular_names(self):
        """{target name: [vernacular query strings]}: the names the authority's
        vernacular search answered with the target itself (the taxonomy
        cache's GBIF vernacular answers, §7.1)."""
        acc = {v: k for k, v in self.target_accepted().items() if v}
        genus_targets = {t["taxon"]: t["name"] for t in self.cfg.targets if t.get("rank") == "genus"}
        out = {}
        for e in self.resolver.cache["resolved"].values():
            if not isinstance(e, dict) or e.get("via") != "vernacular":
                continue
            tname = acc.get(e.get("accepted"))
            if tname is None:
                g = ((e.get("lineage") or {}).get("genus"))
                tname = genus_targets.get(g)
            if tname and e.get("name"):
                out.setdefault(tname, [])
                if e["name"] not in out[tname]:
                    out[tname].append(e["name"])
        return {k: sorted(v, key=str.lower) for k, v in out.items()}

    # ---- the layer
    def new_entries(self):
        """(responses, resolved) the funnel cache does not hold."""
        c = self.resolver.cache
        resp = {k: v for k, v in c["responses"].items() if k not in self._funnel_responses}
        res = {k: v for k, v in c["resolved"].items() if k not in self._funnel_resolved}
        return resp, res

    def save_layer(self, path, testing=False, inputs=None):
        resp, res = self.new_entries()
        doc = header("names_cache", self.cfg, inputs=inputs or {}, testing=testing)
        doc.update({"authority": _authority_record(self.cfg),
                    "overrides": {"sha256": _overrides_sha(self.cfg)},
                    "base": {"funnel_cache": self.provenance.get("funnel_cache")},
                    "responses": dict(sorted(resp.items())), "resolved": dict(sorted(res.items()))})
        sha = write_json_atomic(path, doc)
        return {"path": str(path), "sha256": sha, "responses": len(resp), "resolved": len(res)}


def load(cfg, inc=None, allow_network=False, transport=None):
    """Names over the funnel cache (when present) and the names layer (when
    present). Offline unless allow_network and a transport are given."""
    from ..funnel import FunnelError
    from ..funnel import taxonomy as T
    fpath, lpath = cache_paths(cfg, inc)
    cache = {"responses": {}, "resolved": {}}
    prov = {"funnel_cache": None, "names_layer": None}
    fr, fs = set(), set()
    if fpath.is_file():
        try:
            fc = T.load_cache(fpath, cfg.funnel)
        except FunnelError as e:
            raise CollectError("the funnel taxonomy cache %s does not load: %s" % (fpath, e))
        cache["responses"].update(fc.get("responses") or {})
        cache["resolved"].update(fc.get("resolved") or {})
        fr, fs = set(cache["responses"]), set(cache["resolved"])
        prov["funnel_cache"] = {"path": str(fpath), "sha256": sha256_file(fpath)}
    if lpath.is_file():
        layer = load_layer(lpath, cfg)
        for k, v in (layer.get("responses") or {}).items():
            cache["responses"].setdefault(k, v)
        for k, v in (layer.get("resolved") or {}).items():
            cache["resolved"].setdefault(k, v)
        prov["names_layer"] = {"path": str(lpath), "sha256": sha256_file(lpath)}
    try:
        r = T.Resolver(cache, cfg.funnel, transport=transport, allow_network=allow_network)
    except FunnelError as e:
        raise CollectError("the taxonomy authority of the funnel config is unusable: %s" % e)
    return Names(cfg, r, prov, (fr, fs))


# ------------------------------------------------------------------ lever L26
def source_names(cfg, source_id, inc=None, candidates=None):
    """Every name `collect names` resolves for a source: the class names the
    intake recorded as pending (intake/work/<source>/pending_names.json), the
    candidate's declared classes and hints (candidates.json), the taxa of its
    card table, and the binomials of EPPO codes among them."""
    names = []
    work = intake_dir(inc) / "work" / _safe(source_id) / "pending_names.json"
    if work.is_file():
        doc = read_json(work, "pending names")
        names.extend(doc.get("names") or [])
    for c in (candidates or []):
        if c.get("source_id") != source_id:
            continue
        for cl in c.get("classes") or []:
            names.append(cl.get("name"))
            names.extend(cl.get("hints") or [])
    card = cfg.card_table(source_id)
    if card is not None:
        for v in list(card[0]["by_id"].values()) + list(card[0]["by_name"].values()):
            if v.get("taxon"):
                names.append(v["taxon"])
    from .classmap import eppo_prefix_binomial
    for n in list(names):
        b = cfg.eppo_binomial(n) or eppo_prefix_binomial(cfg, n)
        if b:
            names.append(b)
    out, seen = [], set()
    from ..funnel.names import key
    for n in names:
        if isinstance(n, str) and key(n) and key(n) not in seen:
            seen.add(key(n))
            out.append(n)
    return out


def _safe(s):
    from . import safe_name
    return safe_name(s)


def run_names(cfg, source_id, out_dir=None, inc=None, transport=None, candidates=None, testing=False):
    """Lever L26 (lab, R0): resolve a source's names through the authority and
    append them to the names layer. Writes <out>/names_cache.json and
    <out>/names_<source>.json. Refuses inside a Slurm job (compute nodes have
    no network) unless testing."""
    from ..funnel import TaxonomyError
    if in_slurm() and not testing:
        raise Refusal("needs_network", "collect names needs the authority's network; it runs on the lab, not "
                      "in a Slurm job", action="refuse")
    if transport is None:
        from ..funnel.fetch import default_transport
        transport = default_transport
    _f, layer_path = cache_paths(cfg, inc)
    out_dir = Path(out_dir) if out_dir else layer_path.parent
    with intake_lock(inc, what="collect names"):
        nm = load(cfg, inc, allow_network=True, transport=transport)
        wanted = source_names(cfg, source_id, inc=inc, candidates=candidates)
        rows, errors = [], []
        try:
            nm.resolver.kingdom()
        except TaxonomyError as e:
            raise Refusal("authority_unusable", "the target taxa do not resolve at the authority: %s" % e,
                          action="hold", risk="R3")
        for n in wanted:
            st = nm.status(n)                    # an authority failure is a miss, recorded per name
            if st is None:
                errors.append({"name": n, "error": "unresolved"})
                continue
            rows.append({"name": n, "status": st["status"], "via": st["via"], "taxon": st["taxon"],
                         "target": st["target"], "ambiguous": st["ambiguous"]})
        layer = nm.save_layer(out_dir / layer_path.name, testing=testing)
        rep = header("names", cfg, inputs={"names_layer": {"path": layer["path"], "sha256": layer["sha256"]}},
                     testing=testing)
        rep.update({"source": source_id, "names": rows, "errors": errors, "resolved_utc": utc(),
                    "layer_is_configured_path": (out_dir / layer_path.name) == layer_path})
        write_json_atomic(out_dir / ("names_%s.json" % _safe(source_id)), rep)
    return {"status": "done" if not errors else "partial", "source": source_id, "names": len(rows),
            "errors": len(errors), "layer": layer}
