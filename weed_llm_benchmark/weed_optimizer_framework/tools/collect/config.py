"""The collector's domain config (collect/domains/<domain>.json, format
collect-domain/1; docs/CONTINUOUS_LOOP.md §3.1 "Inputs", §6.5, §7).

The config holds everything domain-specific the collector reads:
  targets            a pointer to the funnel domain config by path and sha256
                     (never a copy): the target classes, their taxa, the
                     reject class id and the taxonomy authority come from it;
  class_space        the intake-only id of an unmapped class;
  taxonomy           where the funnel's offline taxonomy cache and the
                     collector's own names layer (lever L26) live;
  alias_table        a module attribute holding the project's name aliases;
  search_terms, priority, query_grammar
                     the query grammar of discovery (§7.1);
  providers          which providers exist, their endpoints (where the code
                     holds none), credentials, placement and search template;
  download, budgets, estimates
                     timeouts scaled to size, byte and attempt caps, yield
                     floors (set by a person, §7.4) and ranking priors;
  licence_policy     P6 (§2.5): licence ids to classes;
  prefilter          the pre-download filter's word lists and rules (§7.2);
  class_map          which name status becomes which class id (§3.2);
  lab_groups         declared lab groups and which are evaluation labs;
  copy_scan          the files whose passed calibration means the copy scan
                     is available (P9);
  eppo               the offline EPPO code table, pinned by path and sha256;
  card_class_tables  owner-pinned class tables for sources whose names need one;
  known_items        the owner's D-C list, each item with its decision stamp
                     (decided_by) and its source id, a recall audit of
                     discovery and fetchable after a recorded miss;
  licence_overrides  a person's decision on a source whose licence is
                     unresolved, by exact source id: {id, research_only,
                     decided_by (human:<id>), decided_utc, reason}; it lets
                     the source through the licence hold (never a refused
                     licence) and its rows carry research_only as decided
                     (false only for an id the policy reads as permissive);
  placement          the providers kept on the lab whatever the network probe
                     finds (lab_only).

The config is a governance file (§6.5 [review]): the platform never writes
it. load() refuses a config whose pointer or pins no longer hash as recorded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from pathlib import Path

from . import DOMAINS_DIR, FORMATS, TOOLS_DIR, ConfigError, sha256_file, sha256_json
from . import licence as LIC

SCHEMA = FORMATS["domain"]
TOP_KEYS = ("format", "domain", "about", "targets", "class_space", "taxonomy", "alias_table", "search_terms",
            "priority", "query_grammar", "providers", "download", "licence_policy", "budgets", "estimates",
            "prefilter", "class_map", "lab_groups", "copy_scan", "hold", "eppo", "card_class_tables",
            "known_items", "known_items_why", "guard", "normalise", "licence_overrides", "placement")
REQUIRED = ("format", "domain", "targets", "class_space", "taxonomy", "query_grammar", "providers", "download",
            "licence_policy", "budgets", "estimates", "prefilter", "class_map", "lab_groups", "known_items")
PROVIDER_KINDS = ("zenodo", "mendeley", "huggingface", "kaggle", "github", "roboflow", "weedai", "mediatum")
LICENCE_CLASSES = ("permissive", "research_only", "refused", "unresolved")
MAP_TARGETS = ("target", "other", "unmapped")
TEMPLATE_KINDS = ("text", "class", "taxon", "eppo")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def resolve_path(p):
    """A config-relative path: absolute as given; otherwise relative to the
    tools package directory (collect/domains/x.json, funnel/domains/x.json)."""
    p = Path(str(p))
    if p.is_absolute():
        return p
    if p.exists() and not str(p).startswith(("collect/", "funnel/")):
        return p.resolve()
    return TOOLS_DIR / p


def config_path(name_or_path):
    """A config path from a path or a domain name (collect/domains/<name>.json)."""
    s = str(name_or_path)
    if s.endswith(".json") or "/" in s:
        return resolve_path(s)
    if not _NAME_RE.match(s):
        raise ConfigError("%r is neither a domain name nor a config path" % s)
    return DOMAINS_DIR / ("%s.json" % s)


def validate(raw):
    """Every problem of a collect-domain/1 config that can be seen without
    reading other files (empty = valid)."""
    p = []
    if not isinstance(raw, dict):
        return ["the config is not a JSON object"]
    for k in sorted(raw):
        if k not in TOP_KEYS:
            p.append("unknown top-level key %r" % k)
    for k in REQUIRED:
        if k not in raw:
            p.append("missing top-level key %r" % k)
    if raw.get("format") != SCHEMA:
        p.append("format %r is not %s" % (raw.get("format"), SCHEMA))
    if not isinstance(raw.get("domain"), str) or not _NAME_RE.match(raw.get("domain") or ""):
        p.append("domain %r must match [a-z][a-z0-9_]*" % raw.get("domain"))
    tg = raw.get("targets")
    if not isinstance(tg, dict) or not isinstance(tg.get("path"), str) or not isinstance(tg.get("sha256"), str):
        p.append("targets: {path, sha256} of the funnel domain config is required (a pointer, never a copy)")
    cs = raw.get("class_space")
    if not isinstance(cs, dict) or not isinstance(cs.get("unmapped_id"), int):
        p.append("class_space.unmapped_id: an integer is required")
    tx = raw.get("taxonomy")
    if not isinstance(tx, dict) or not isinstance(tx.get("funnel_cache"), str) or not isinstance(
            tx.get("names_layer"), str):
        p.append("taxonomy: funnel_cache and names_layer (paths under INC_DIR) are required")
    qg = raw.get("query_grammar")
    if isinstance(qg, dict):
        tpl = qg.get("templates")
        if not isinstance(tpl, dict) or not tpl:
            p.append("query_grammar.templates: an object of template lists")
        else:
            for k, v in tpl.items():
                if k not in TEMPLATE_KINDS:
                    p.append("query_grammar.templates.%s: kind not in %s" % (k, TEMPLATE_KINDS))
                if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                    p.append("query_grammar.templates.%s: a list of non-empty strings" % k)
        for k in ("max_queries_per_provider", "max_results_per_query", "max_describe_per_provider"):
            if not isinstance(qg.get(k), int) or qg.get(k) <= 0:
                p.append("query_grammar.%s: a positive integer" % k)
    else:
        p.append("query_grammar: an object")
    prov = raw.get("providers")
    if not isinstance(prov, dict) or not prov:
        p.append("providers: an object of provider sections")
    else:
        for name, sec in prov.items():
            if not isinstance(sec, dict):
                p.append("providers.%s: an object" % name)
                continue
            if sec.get("kind") not in PROVIDER_KINDS:
                p.append("providers.%s: kind %r not in %s" % (name, sec.get("kind"), PROVIDER_KINDS))
            if sec.get("template") is not None and sec.get("template") not in TEMPLATE_KINDS:
                p.append("providers.%s: template %r not in %s" % (name, sec.get("template"), TEMPLATE_KINDS))
            if sec.get("kind") in ("weedai", "mediatum") and not isinstance(sec.get("base_url"), str):
                p.append("providers.%s: base_url is required for kind %s" % (name, sec.get("kind")))
    dl = raw.get("download")
    if not isinstance(dl, dict) or not all(_is_num(dl.get(k)) and dl.get(k) > 0
                                           for k in ("base_timeout_s", "min_rate_bytes_per_s", "read_timeout_s",
                                                     "attempts")):
        p.append("download: base_timeout_s, min_rate_bytes_per_s, read_timeout_s, attempts (positive numbers)")
    lp = raw.get("licence_policy")
    if isinstance(lp, dict):
        seen = {}
        for cls in LICENCE_CLASSES:
            ids = lp.get(cls)
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                p.append("licence_policy.%s: a list of licence ids" % cls)
                continue
            for x in ids:
                if x in seen and seen[x] != cls:
                    p.append("licence_policy: %r is in both %s and %s" % (x, seen[x], cls))
                seen[x] = cls
    else:
        p.append("licence_policy: an object")
    bu = raw.get("budgets")
    if isinstance(bu, dict):
        for k in ("bytes_per_source", "bytes_envelope", "attempts_per_source"):
            if not _is_num(bu.get(k)) or bu.get(k) <= 0:
                p.append("budgets.%s: a positive number" % k)
        # no daily byte cap unless one is declared (amendment 2026-10-04,
        # docs/CONTINUOUS_LOOP.md 6.6): null or absent is none
        bd = bu.get("bytes_daily")
        if bd is not None and (not _is_num(bd) or bd <= 0):
            p.append("budgets.bytes_daily: null (no daily byte cap) or a positive number")
        im = bu.get("intake_max_images")
        if im is not None and not (isinstance(im, int) and not isinstance(im, bool) and im > 0):
            p.append("budgets.intake_max_images: null or a positive integer")
        ms = bu.get("intake_max_seconds")
        if ms is not None and not (_is_num(ms) and ms > 0):
            p.append("budgets.intake_max_seconds: null or a positive number")
        for k in ("floor_gb", "floor_su"):
            v = bu.get(k)
            if v is not None and not (isinstance(v, dict) and _is_num(v.get("value")) and v.get("set_by")):
                p.append("budgets.%s: null, or {value, set_by} written by a person (a bare number is a "
                         "placeholder and is refused, §7.4)" % k)
    else:
        p.append("budgets: an object")
    es = raw.get("estimates")
    if not isinstance(es, dict) or not all(_is_num(es.get(k)) and es.get(k) >= 0 for k in
                                           ("boxes_per_image", "unknown_images", "unknown_bytes",
                                            "text_target_share", "deficit_bonus")):
        p.append("estimates: boxes_per_image, unknown_images, unknown_bytes, text_target_share, deficit_bonus")
    pf = raw.get("prefilter")
    if isinstance(pf, dict):
        for k in ("reject_tokens", "legacy_labels", "box_words", "image_level_words"):
            if not isinstance(pf.get(k), list) or not all(isinstance(x, str) and x for x in pf.get(k)):
                p.append("prefilter.%s: a list of non-empty strings" % k)
        if not isinstance(pf.get("copy_candidate_min"), int) or pf.get("copy_candidate_min") < 1:
            p.append("prefilter.copy_candidate_min: a positive integer")
        for rx in pf.get("never_fetch_title_regex") or []:
            try:
                re.compile(rx)
            except (re.error, TypeError):
                p.append("prefilter.never_fetch_title_regex: %r does not compile" % (rx,))
    else:
        p.append("prefilter: an object")
    cm = raw.get("class_map")
    sm = (cm or {}).get("status_map") if isinstance(cm, dict) else None
    if not isinstance(sm, dict) or sm.get("default") not in MAP_TARGETS:
        p.append("class_map.status_map: an object with a default in %s" % (MAP_TARGETS,))
    else:
        for k, v in sm.items():
            vals = list(v.values()) if isinstance(v, dict) else [v]
            if any(x not in MAP_TARGETS for x in vals):
                p.append("class_map.status_map.%s: values must be in %s" % (k, MAP_TARGETS))
    lg = raw.get("lab_groups")
    if not isinstance(lg, dict):
        p.append("lab_groups: an object")
    else:
        for g, sec in lg.items():
            if not isinstance(sec, dict) or not isinstance(sec.get("evaluation"), bool):
                p.append("lab_groups.%s: {evaluation: bool, members, refs}" % g)
    ki = raw.get("known_items")
    if not isinstance(ki, list):
        p.append("known_items: a list of items, each with its decided_by stamp")
    else:
        ids, handles = set(), set()
        for i, it in enumerate(ki):
            if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not _ID_RE.match(it.get("id") or ""):
                p.append("known_items[%d]: id required (the source id the item has everywhere)" % i)
                continue
            if it["id"] in ids:
                p.append("known_items: duplicate id %r" % it["id"])
            ids.add(it["id"])
            if it.get("name") is not None:
                if it["name"] in handles or it["name"] in ids:
                    p.append("known_items: duplicate name %r" % it["name"])
                handles.add(it["name"])
            if not isinstance(it.get("decided_by"), str) or not it.get("decided_by"):
                p.append("known item %s: decided_by is required (the owner's decision stamp)" % it["id"])
            if not isinstance(it.get("match"), list) or not it["match"]:
                p.append("known item %s: a non-empty match list (the recall audit needs it)" % it["id"])
            if it.get("fetchable", True) and (not it.get("provider") or not it.get("ref")):
                p.append("known item %s: a fetchable item needs provider and ref" % it["id"])
            if it.get("provider") and isinstance(prov, dict) and it["provider"] not in prov:
                p.append("known item %s: provider %r is not configured" % (it["id"], it.get("provider")))
    pl = raw.get("placement")
    if pl is not None:
        lo = pl.get("lab_only") if isinstance(pl, dict) else None
        if not isinstance(lo, list) or any(x not in (prov or {}) for x in lo):
            p.append("placement.lab_only: a list of configured provider names")
    ep = raw.get("eppo")
    if ep is not None and (not isinstance(ep, dict) or not isinstance(ep.get("path"), str)
                           or not isinstance(ep.get("sha256"), str)):
        p.append("eppo: {path, sha256}")
    ct = raw.get("card_class_tables") or {}
    if not isinstance(ct, dict):
        p.append("card_class_tables: an object")
    else:
        for src, t in ct.items():
            if not isinstance(t, dict) or not (isinstance(t.get("by_id"), dict) or isinstance(t.get("by_name"), dict)):
                p.append("card_class_tables.%s: by_id and/or by_name tables" % src)
            elif not t.get("table_source") or not t.get("pinned_by"):
                p.append("card_class_tables.%s: table_source and pinned_by are required" % src)
    lo = raw.get("licence_overrides")
    if lo is not None and not isinstance(lo, dict):
        p.append("licence_overrides: an object of a person's decisions keyed by source id")
    else:
        for src, ov in (lo or {}).items():
            if not _ID_RE.match(src):
                p.append("licence_overrides: %r is not a source id" % src)
            if not isinstance(ov, dict):
                p.append("licence_overrides.%s: {id, research_only, decided_by, decided_utc, reason}" % src)
                continue
            if not isinstance(ov.get("id"), str) or not ov["id"].strip():
                p.append("licence_overrides.%s: id (the licence the person records) is required" % src)
            if not isinstance(ov.get("research_only"), bool):
                p.append("licence_overrides.%s: research_only must be true or false" % src)
            elif ov["research_only"] is False and isinstance(ov.get("id"), str) and ov["id"].strip():
                # fail closed (P6): an override lifts research_only only for a licence the policy reads as permissive
                pol = {c: v for c, v in lp.items() if isinstance(v, list)} if isinstance(lp, dict) else {}
                cls = LIC.classify(LIC.canonical(ov["id"]), pol)
                if cls != "permissive":
                    p.append("licence_overrides.%s: research_only false needs a licence the policy reads as "
                             "permissive (%r is %s)" % (src, ov["id"], cls))
            if not isinstance(ov.get("decided_by"), str) or not ov["decided_by"].startswith("human:"):
                p.append("licence_overrides.%s: decided_by must be a person (human:<id>)" % src)
            for k in ("decided_utc", "reason"):
                if not isinstance(ov.get(k), str) or not ov[k].strip():
                    p.append("licence_overrides.%s: %s is required" % (src, k))
    return p


class CollectConfig(object):
    """A validated collector config and the files it pins."""

    def __init__(self, raw, path, sha256, funnel_domain, eppo=None, eppo_record=None):
        self.raw = raw
        self.path = Path(path)
        self.sha256 = sha256
        self.name = raw["domain"]
        self.funnel = funnel_domain
        self.eppo = dict(eppo or {})
        self.eppo_record = eppo_record
        self._eppo_rev = None

    def __repr__(self):
        return "CollectConfig(%r, %s)" % (self.name, self.sha256[:12])

    def record(self):
        return {"path": str(self.path), "sha256": self.sha256,
                "targets": {"path": str(self.funnel.path), "sha256": self.funnel.sha256},
                "eppo": self.eppo_record}

    def section(self, key, default=None):
        v = self.raw.get(key)
        return copy.deepcopy(v) if v is not None else copy.deepcopy(default)

    # ---- class space
    @property
    def targets(self):
        return self.funnel.targets

    @property
    def target_names(self):
        return self.funnel.target_names

    @property
    def other_id(self):
        return int(self.funnel.other["id"])

    @property
    def unmapped_id(self):
        return int(self.raw["class_space"]["unmapped_id"])

    def target(self, name_or_id):
        return self.funnel.target(name_or_id)

    # ---- providers
    def providers(self, enabled_only=True):
        out = {}
        for name, sec in sorted((self.raw.get("providers") or {}).items()):
            if enabled_only and sec.get("enabled") is False:
                continue
            out[name] = copy.deepcopy(sec)
        return out

    def provider(self, name):
        sec = (self.raw.get("providers") or {}).get(name)
        if sec is None:
            raise ConfigError("provider %r is not in the config %s" % (name, self.path.name))
        return copy.deepcopy(sec)

    # ---- known items
    def known_items(self):
        return copy.deepcopy(self.raw["known_items"])

    def known_decided_by(self):
        """The owner's decision stamp of the known items (one string when they
        share it, else the sorted list)."""
        stamps = sorted({it["decided_by"] for it in self.raw["known_items"]})
        return stamps[0] if len(stamps) == 1 else stamps

    def known_item(self, key):
        """The known item whose id (its source id) or name (a short handle) is key."""
        for it in self.raw["known_items"]:
            if key is not None and key in (it["id"], it.get("name")):
                return copy.deepcopy(it)
        return None

    def known_item_for(self, source_id=None, provider=None, ref=None, title=None, text=None):
        """The known item a candidate is (by id or name, or by one of the item's
        match rules), else None."""
        for it in self.raw["known_items"]:
            if source_id and source_id in (it["id"], it.get("name")):
                return copy.deepcopy(it)
            if match_any(it.get("match") or [], provider, ref, title, text):
                return copy.deepcopy(it)
        return None

    # ---- placement
    def lab_only(self, provider):
        """True when the config keeps a provider on the lab whatever the probe says."""
        return provider in ((self.raw.get("placement") or {}).get("lab_only") or [])

    # ---- lab groups
    def lab_groups(self):
        """{group: {"evaluation", "members", "refs"}}: the collector's lab
        groups plus the funnel domain's (read only), whose groups are the
        labs of the evaluation splits' origin when the collector config marks
        them so."""
        out = {}
        for g, members in (self.funnel.lab_groups() or {}).items():
            out[g] = {"evaluation": False, "members": list(members), "refs": []}
        for g, sec in (self.raw.get("lab_groups") or {}).items():
            cur = out.setdefault(g, {"evaluation": False, "members": [], "refs": []})
            cur["evaluation"] = bool(sec.get("evaluation"))
            cur["members"] = sorted(set(cur["members"]) | set(sec.get("members") or []))
            cur["refs"] = list(cur["refs"]) + list(sec.get("refs") or [])
        return out

    def evaluation_labs(self):
        return sorted(g for g, s in self.lab_groups().items() if s["evaluation"])

    def lab_group_of(self, source_id=None, provider=None, ref=None, title=None, text=None):
        """(group, "declared") when the config names the source's lab, else (None, None)."""
        for g, sec in sorted(self.lab_groups().items()):
            if source_id and source_id in sec["members"]:
                return g, "declared"
            if match_any(sec["refs"], provider, ref, title, text):
                return g, "declared"
        return None, None

    # ---- tables
    def card_table(self, source_id):
        """(table, sha256, origin) of the class table for a source: the
        collector config's pinned table, else the funnel domain's card
        resolver table (read only), else None. table: {"by_id": {src id:
        {name, taxon}}, "by_name": {name key: {taxon}}}."""
        from ..funnel.names import key as name_key
        t = (self.raw.get("card_class_tables") or {}).get(source_id)
        if t is not None:
            by_id = {str(k): dict(v) for k, v in (t.get("by_id") or {}).items()}
            by_name = {name_key(k): dict(v) for k, v in (t.get("by_name") or {}).items()}
            tab = {"by_id": by_id, "by_name": by_name}
            return tab, sha256_json(t), "collect_config"
        res = ((self.funnel.raw.get("sources") or {}).get("card_resolvers") or {}).get(source_id) or {}
        ct = res.get("class_table")
        names = res.get("names")
        if isinstance(ct, dict) or isinstance(names, dict):
            by_id = {str(k): dict(v) for k, v in (ct or {}).items() if isinstance(v, dict)}
            by_name = {name_key(k): {"taxon": v} for k, v in (names or {}).items() if v}
            for v in by_id.values():
                if v.get("name"):
                    by_name.setdefault(name_key(v["name"]), {"taxon": v.get("taxon")})
            return {"by_id": by_id, "by_name": by_name}, sha256_json({"class_table": ct, "names": names}), \
                "funnel_config"
        return None

    def eppo_binomial(self, code):
        if not isinstance(code, str):
            return None
        return self.eppo.get(code.strip())

    def eppo_codes_of(self, binomial):
        """EPPO codes whose binomial is `binomial` or lies in genus `binomial`."""
        if self._eppo_rev is None:
            rev = {}
            for c, b in self.eppo.items():
                rev.setdefault(b.lower(), []).append(c)
            self._eppo_rev = rev
        b = str(binomial or "").lower().strip()
        out = list(self._eppo_rev.get(b, []))
        if " " not in b:
            for name, codes in self._eppo_rev.items():
                if name.split(" ")[0] == b and name != b:
                    out.extend(codes)
        return sorted(set(out))

    # ---- misc sections
    def budgets(self):
        return copy.deepcopy(self.raw["budgets"])

    def yield_floors(self):
        """{"floor_gb": value | None, "floor_su": value | None}: None until a
        person sets it from the first wave (§7.4)."""
        bu = self.raw["budgets"]
        return {k: (bu.get(k) or {}).get("value") if bu.get(k) else None for k in ("floor_gb", "floor_su")}

    def intake_max_images(self):
        """The most images with a box one intake batch takes (budgets), or
        None: no cap."""
        v = self.raw["budgets"].get("intake_max_images")
        return int(v) if v else None

    def intake_max_seconds(self):
        """The wall clock after which an intake defers the images it has not
        judged yet (budgets), or None: no limit."""
        v = self.raw["budgets"].get("intake_max_seconds")
        return float(v) if v else None

    def hold_deadline_days(self):
        return int(((self.raw.get("hold") or {}).get("deadline_days")) or 21)


def match_any(rules, provider=None, ref=None, title=None, text=None):
    """True when one rule matches: {"provider"?, "ref"?, "ref_regex"?,
    "title_regex"?, "text_regex"?} (text: the record's title, description and
    keywords); every key a rule gives must match."""
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        ok = True
        if "provider" in r:
            ok = ok and provider == r["provider"]
        if "ref" in r:
            ok = ok and ref is not None and str(ref).lower() == str(r["ref"]).lower()
        if "ref_regex" in r:
            ok = ok and ref is not None and re.search(r["ref_regex"], str(ref)) is not None
        if "title_regex" in r:
            ok = ok and title is not None and re.search(r["title_regex"], str(title)) is not None
        if "text_regex" in r:
            ok = ok and text is not None and re.search(r["text_regex"], str(text)) is not None
        if ok and any(k in r for k in ("ref", "ref_regex", "title_regex", "text_regex")):
            return True
    return False


def licence_override(cfg, source_id):
    """A person's licence decision for a source (licence_overrides, by its
    exact source id), else None: the one reading the pre-check, plan and
    intake share. It lets an unresolved licence through, never a refused one."""
    ov = (cfg.raw.get("licence_overrides") or {}).get(source_id) if source_id is not None else None
    return copy.deepcopy(ov) if ov else None


def record_text(c):
    """The text a text_regex rule reads: title, description and keywords."""
    return " ".join([str(c.get("title") or ""), str(c.get("description") or "")]
                    + [str(k) for k in (c.get("keywords") or [])])


def _load_eppo(raw, cfg_path):
    ep = raw.get("eppo")
    if not ep:
        return {}, None
    p = resolve_path(ep["path"])
    if not p.is_file():
        raise ConfigError("the EPPO table %s the config pins is missing" % p)
    got = sha256_file(p)
    if got != ep["sha256"]:
        raise ConfigError("the EPPO table %s hashes to %s, the config pins %s: a changed table needs a new pin "
                          "by a person (§10 item 5)" % (p, got[:12], ep["sha256"][:12]))
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != FORMATS["eppo"] or not isinstance(doc.get("codes"), dict):
        raise ConfigError("%s is not a %s table" % (p, FORMATS["eppo"]))
    for c, b in doc["codes"].items():
        if not re.match(r"^[0-9A-Z]{5,6}$", c) or not isinstance(b, str) or not b.strip():
            raise ConfigError("%s: code %r -> %r is not an EPPO code and a name" % (p, c, b))
    return dict(doc["codes"]), {"path": str(p), "sha256": got, "version": doc.get("version"),
                                "verified": doc.get("verified")}


def load(name_or_path):
    """A CollectConfig from a path or domain name. Refuses (ConfigError) on any
    validation problem, a stale pointer to the funnel domain or a stale EPPO pin."""
    if isinstance(name_or_path, CollectConfig):
        return name_or_path
    path = config_path(name_or_path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ConfigError("cannot read the collector config %s (%s)" % (path, e))
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise ConfigError("%s is not JSON (%s)" % (path, e))
    problems = validate(raw)
    if problems:
        raise ConfigError("%s: %d problem(s): %s" % (path, len(problems), "; ".join(problems)))
    from ..funnel import DomainError
    from ..funnel import domain as FD
    tp = resolve_path(raw["targets"]["path"])
    if not tp.is_file():
        raise ConfigError("the funnel domain config %s the collector points to is missing" % tp)
    got = sha256_file(tp)
    if got != raw["targets"]["sha256"]:
        raise ConfigError("the funnel domain config %s hashes to %s, the collector config records %s: the "
                          "targets changed; re-pin the pointer (a governance change, §6.5)"
                          % (tp, got[:12], raw["targets"]["sha256"][:12]))
    try:
        fdom = FD.load(str(tp))
    except DomainError as e:
        raise ConfigError("the funnel domain config does not load: %s" % e)
    if fdom.name != raw["domain"]:
        raise ConfigError("the collector config is for domain %r, the funnel config for %r" % (raw["domain"], fdom.name))
    ids = set(fdom.target_ids) | {int(fdom.other["id"])}
    if int(raw["class_space"]["unmapped_id"]) in ids:
        raise ConfigError("class_space.unmapped_id %s collides with a class id of the funnel domain"
                          % raw["class_space"]["unmapped_id"])
    names = set(fdom.target_names)
    for k in list((raw.get("search_terms") or {}).keys()) + list((raw.get("priority") or {}).keys()):
        if k not in names:
            raise ConfigError("search_terms/priority name %r is not a target of the funnel domain" % k)
    pd = (raw["prefilter"].get("presumed_derivative") or {})
    for k in pd.get("declares_any") or []:
        if k not in names:
            raise ConfigError("prefilter.presumed_derivative.declares_any: %r is not a target" % k)
    eppo, eppo_rec = _load_eppo(raw, path)
    return CollectConfig(raw, path, hashlib.sha256(data).hexdigest(), fdom, eppo, eppo_rec)


def alias_table(cfg):
    """{name key: target name} from the module attribute the config names
    (alias_table.module / .attr), or {} when it names none. Keys are name
    keys (letters and digits, lower case); values must be target names."""
    spec = cfg.raw.get("alias_table")
    if not spec:
        return {}
    try:
        mod = importlib.import_module(spec["module"])
        table = getattr(mod, spec["attr"])
    except (ImportError, AttributeError, KeyError) as e:
        raise ConfigError("alias_table %r cannot be read (%s)" % (spec, e))
    names = set(cfg.target_names)
    out = {}
    for k, v in dict(table).items():
        if v in names:
            out[str(k)] = v
    return out
