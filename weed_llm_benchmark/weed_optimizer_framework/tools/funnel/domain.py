"""The domain config (domains/<domain>.json, format funnel-domain/1) and the
pre-registration (prereg_v1.json, format funnel-prereg/1).

Runner §3 pins the schema; contract §8.8 says everything domain-specific sits
in the config, so the engine reads class names, stages, exams, lab groups and
known-truth sets from here and never spells them.

validate(raw) returns every problem it finds; load() raises DomainError
listing them. The checks that encode a contract rule (a guard stage marked
recoverable, a claimed known-truth set qualifying the reference labeller, an
exam both deciding and not) are refusals, not warnings.

The prereg is loaded with its contract hash checked. Its core sha256 (the
canonical JSON without "amendments") is what every artifact compares, so the
sample-lock amendment does not make earlier outputs stale. append_amendment
is the only writer of the prereg file, and it can change nothing but the
amendments list.

Standard library only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from ..inc import common as C
from . import (FUN_DIR, DomainError, PreregError, _atomic_write_bytes, canonical_json)
from .ledger import ROLES as STAGE_ROLES, UNITS as STAGE_UNITS

SCHEMA_VERSION = "funnel-domain/1"
PREREG_FORMAT = "funnel-prereg/1"
DOMAINS_DIR = FUN_DIR / "domains"

TOP_KEYS = ("format", "domain", "adapter", "domain_terms", "classes", "attractors", "names",
            "taxonomy", "stages", "exams", "sources", "known_truth", "identity_checks", "judges",
            "reference_labeller", "sampling", "leak", "recovery")
REQUIRED_KEYS = ("format", "domain", "adapter", "classes", "stages", "exams", "sources",
                 "known_truth")
RANKS = ("species", "genus")
JUDGE_KINDS = ("step1_probe", "zero_shot", "knn", "rl")
# Provider kinds a fetch spec may name (fetch.KINDS) and labeller backend
# kinds; a grep term equal to one of them would flag the engine code that
# dispatches on it, so domain_terms may not hold them.
PROVIDER_KINDS = ("http", "archive", "inat_observations", "roboflow_classes", "gbif_backbone",
                  "external", "ollama")
KT_SPECIAL_KEYS = ("qualify_rl_on", "forbidden", "kt7")
TAIL_ANSWERS = ("other", "non_object", "invalid", "unsure")
PAIR_ANSWERS = ("same", "consecutive", "different", "unsure")
LEVELS = ("species", "genus", "plant")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _strs(v):
    return isinstance(v, list) and all(isinstance(x, str) and x for x in v)


def _kt_ids(raw):
    kt = raw.get("known_truth")
    if not isinstance(kt, dict):
        return []
    return sorted(k for k in kt if k not in KT_SPECIAL_KEYS)


def _provider_kinds(raw):
    kinds = set(PROVIDER_KINDS)
    auth = ((raw.get("taxonomy") or {}).get("authority") or {})
    if isinstance(auth.get("kind"), str):
        kinds.add(auth["kind"])
    kt7 = ((raw.get("known_truth") or {}).get("kt7") or {})
    prov = kt7.get("provider") if isinstance(kt7, dict) else None
    if isinstance(prov, dict) and isinstance(prov.get("kind"), str):
        kinds.add(prov["kind"])
    for res in ((raw.get("sources") or {}).get("card_resolvers") or {}).values():
        if not isinstance(res, dict):
            continue
        specs = list(res.get("fetch") or [])
        if isinstance(res.get("upstream_annotations"), dict):
            specs.append(res["upstream_annotations"])
        for s in specs:
            if isinstance(s, dict) and isinstance(s.get("kind"), str):
                kinds.add(s["kind"])
    for b in ((raw.get("reference_labeller") or {}).get("backends") or {}).values():
        if isinstance(b, dict) and isinstance(b.get("kind"), str):
            kinds.add(b["kind"])
    return {k.lower() for k in kinds}


def _check_stages(stages, problems):
    if not isinstance(stages, list) or not stages:
        problems.append("stages: a non-empty list is required")
        return
    seen = []
    for i, st in enumerate(stages):
        where = "stages[%d]" % i
        if not isinstance(st, dict):
            problems.append("%s: not an object" % where)
            continue
        sid = st.get("id")
        if not isinstance(sid, str) or not sid:
            problems.append("%s: id must be a non-empty string" % where)
            continue
        where = "stage %s" % sid
        if sid in seen:
            problems.append("%s: duplicate id" % where)
        for key in ("name", "filter"):
            if key in st and not isinstance(st[key], str):
                problems.append("%s: %s must be a string" % (where, key))
        if st.get("role") not in STAGE_ROLES:
            problems.append("%s: role %r not in %s" % (where, st.get("role"), STAGE_ROLES))
        if st.get("unit") not in STAGE_UNITS:
            problems.append("%s: unit %r not in %s" % (where, st.get("unit"), STAGE_UNITS))
        if not isinstance(st.get("guard"), bool):
            problems.append("%s: guard must be true or false" % where)
        rec = st.get("recoverable")
        if not (isinstance(rec, bool) or (isinstance(rec, str) and rec)):
            problems.append("%s: recoverable must be a bool or a non-empty string" % where)
        if st.get("guard") is True and rec is not False:  # funnel-mutation: M5
            problems.append("%s: a guard stage is never recoverable (recoverable=%r)" % (where, rec))
        deps = st.get("depends_on", [])
        if not isinstance(deps, list):
            problems.append("%s: depends_on must be a list" % where)
        else:
            for d in deps:
                if d not in seen:
                    problems.append("%s: depends_on %r does not name an earlier stage" % (where, d))
        seen.append(sid)


def _check_classes(raw, problems):
    cl = raw.get("classes")
    if not isinstance(cl, dict):
        problems.append("classes: an object is required")
        return [], None
    targets = cl.get("targets")
    if not isinstance(targets, list) or not targets:
        problems.append("classes.targets: a non-empty list is required")
        targets = []
    names = []
    for i, t in enumerate(targets):
        if not isinstance(t, dict):
            problems.append("classes.targets[%d]: not an object" % i)
            continue
        if t.get("id") != i or not _is_int(t.get("id")):
            problems.append("classes.targets[%d]: id %r is not %d (ids must be 0..n-1 in order)"
                            % (i, t.get("id"), i))
        nm = t.get("name")
        if not isinstance(nm, str) or not nm:
            problems.append("classes.targets[%d]: name required" % i)
        elif nm in names:
            problems.append("classes.targets[%d]: duplicate name %r" % (i, nm))
        names.append(nm)
        if "rank" in t and t["rank"] not in RANKS:
            problems.append("classes.targets[%d]: rank %r not in %s" % (i, t["rank"], RANKS))
        if "taxon" in t and not isinstance(t["taxon"], str):
            problems.append("classes.targets[%d]: taxon must be a string" % i)
        if "not" in t and not isinstance(t["not"], list):
            problems.append("classes.targets[%d]: not must be a list" % i)
    for i, t in enumerate(targets):
        if isinstance(t, dict):
            for s in t.get("siblings", []) or []:
                if s not in names:
                    problems.append("classes.targets[%d]: sibling %r is not a target" % (i, s))
    other = cl.get("other")
    if not isinstance(other, dict) or not _is_int(other.get("id")) or not isinstance(other.get("name"), str):
        problems.append("classes.other: {id: int, name: str} required")
        other = None
    else:
        if other["id"] in range(len(targets)):
            problems.append("classes.other: id %d collides with a target id" % other["id"])
        if other["name"] in names:
            problems.append("classes.other: name %r is a target name" % other["name"])
    if "genus_answer_unsure_for" in cl and not _strs(cl["genus_answer_unsure_for"]):
        problems.append("classes.genus_answer_unsure_for: a list of names")
    if "small_class_train_boxes_below" in cl and not _is_int(cl["small_class_train_boxes_below"]):
        problems.append("classes.small_class_train_boxes_below: an integer")
    return names, other


def _check_known_truth(raw, problems):
    kt = raw.get("known_truth")
    if not isinstance(kt, dict):
        problems.append("known_truth: an object is required")
        return []
    ids = _kt_ids(raw)
    for k in ids:
        e = kt[k]
        where = "known_truth.%s" % k
        if not isinstance(e, dict):
            problems.append("%s: not an object" % where)
            continue
        if not isinstance(e.get("independent"), bool):
            problems.append("%s: independent must be true or false" % where)
        cb = e.get("claimed_by")
        if cb is not None and not isinstance(cb, str):
            problems.append("%s: claimed_by must be a hypothesis id or null" % where)
        if cb and e.get("independent") is True:
            problems.append("%s: a set whose labels a hypothesis tests (claimed_by %s) cannot be "
                            "independent" % (where, cb))
        for key in ("allowed_uses", "never_qualifies", "sources"):
            if key in e and not isinstance(e[key], list):
                problems.append("%s: %s must be a list" % (where, key))
    q = kt.get("qualify_rl_on")
    if not isinstance(q, list):
        problems.append("known_truth.qualify_rl_on: a list of known-truth ids is required")
    else:
        for k in q:
            if k not in ids:
                problems.append("known_truth.qualify_rl_on: %r is not a known-truth set" % k)
            elif isinstance(kt[k], dict) and kt[k].get("claimed_by"):
                problems.append("known_truth.qualify_rl_on: %s holds claimed labels (claimed_by %s); "
                                "a claimed set never qualifies the reference labeller"
                                % (k, kt[k]["claimed_by"]))
    if "forbidden" in kt and not _strs(kt["forbidden"]):
        problems.append("known_truth.forbidden: a list of split names")
    return ids


def _check_exams(raw, problems):
    ex = raw.get("exams")
    if not isinstance(ex, dict):
        problems.append("exams: an object is required")
        return
    dec = ex.get("decision")
    if not isinstance(dec, str) or not dec:
        problems.append("exams.decision: a split name is required")
    nd = ex.get("non_decision")
    extra = ex.get("extra_non_decision", [])
    if not isinstance(nd, list) or not all(isinstance(x, str) and x for x in nd):
        problems.append("exams.non_decision: a list of split names is required")
        nd = []
    if not isinstance(extra, list) or not all(isinstance(x, str) and x for x in extra):
        problems.append("exams.extra_non_decision: a list of split names")
        extra = []
    if dec in nd or dec in extra:
        problems.append("exams: %r is both the decision exam and a non-decision exam" % dec)
    dup = sorted(set(nd) & set(extra))
    if dup or len(set(nd)) != len(nd) or len(set(extra)) != len(extra):
        problems.append("exams: a non-decision exam is listed twice %s" % dup)


def _check_sources(raw, problems):
    src = raw.get("sources")
    if not isinstance(src, dict):
        problems.append("sources: an object is required")
        return
    if not isinstance(src.get("reference"), str) or not src.get("reference"):
        problems.append("sources.reference: the reference source name is required")
    lg = src.get("lab_groups", {})
    if not isinstance(lg, dict):
        problems.append("sources.lab_groups: an object of lists")
        return
    owner = {}
    for g in sorted(lg):
        if not _strs(lg[g]):
            problems.append("sources.lab_groups.%s: a list of source names" % g)
            continue
        for s in lg[g]:
            if s in owner and owner[s] != g:
                problems.append("sources.lab_groups: %r is in both %s and %s" % (s, owner[s], g))
            owner[s] = g
    for key in ("authoritative", "licences", "not_recoverable", "capture_stem_regex",
                "card_image_counts", "card_resolvers"):
        if key in src and not isinstance(src[key], dict):
            problems.append("sources.%s: an object" % key)
    for slug, rx in (src.get("capture_stem_regex") or {}).items():
        try:
            re.compile(rx)
        except (re.error, TypeError):
            problems.append("sources.capture_stem_regex.%s: not a regular expression" % slug)


def _check_reference_labeller(raw, kt_ids, problems):
    rl = raw.get("reference_labeller")
    if rl is None:
        return
    if not isinstance(rl, dict):
        problems.append("reference_labeller: an object")
        return
    tail = rl.get("options_tail")
    if tail is not None and (not _strs(tail) or len(tail) != len(TAIL_ANSWERS)):
        problems.append("reference_labeller.options_tail: %d texts, in the order %s"
                        % (len(TAIL_ANSWERS), TAIL_ANSWERS))
    po = rl.get("pair_options")
    if po is not None and (not _strs(po) or len(po) != len(PAIR_ANSWERS)):
        problems.append("reference_labeller.pair_options: %d texts, in the order %s"
                        % (len(PAIR_ANSWERS), PAIR_ANSWERS))
    for b in rl.get("boards", []) or []:
        if not isinstance(b, dict):
            problems.append("reference_labeller.boards: not an object")
            continue
        for key in ("targets_from", "attractors_from"):
            if b.get(key) not in kt_ids:
                problems.append("reference_labeller.boards %s: %s %r is not a known-truth set"
                                % (b.get("id"), key, b.get(key)))
    for name, be in (rl.get("backends") or {}).items():
        if not isinstance(be, dict) or not isinstance(be.get("family"), str):
            problems.append("reference_labeller.backends.%s: kind, model and family required" % name)
    sheet = rl.get("sheet")
    if sheet is not None:
        if not isinstance(sheet, dict) or not all(_is_int(sheet.get(k)) for k in ("items", "sentinels")):
            problems.append("reference_labeller.sheet: items and sentinels (integers) required")
        elif not 0 < sheet["sentinels"] < sheet["items"]:
            problems.append("reference_labeller.sheet: 0 < sentinels < items")


def validate(raw):
    """Every problem of a funnel-domain/1 config, as sentences (empty = valid)."""
    problems = []
    if not isinstance(raw, dict):
        return ["the config is not a JSON object"]
    for k in sorted(raw):
        if k not in TOP_KEYS:
            problems.append("unknown top-level key %r" % k)
    for k in REQUIRED_KEYS:
        if k not in raw:
            problems.append("missing top-level key %r" % k)
    if raw.get("format") != SCHEMA_VERSION:
        problems.append("format %r is not %s" % (raw.get("format"), SCHEMA_VERSION))
    if not isinstance(raw.get("domain"), str) or not _NAME_RE.match(raw.get("domain") or ""):
        problems.append("domain %r must match [a-z][a-z0-9_]*" % raw.get("domain"))
    if not isinstance(raw.get("adapter"), str) or not _NAME_RE.match(raw.get("adapter") or ""):
        problems.append("adapter %r must name a module under adapters/" % raw.get("adapter"))
    names, other = _check_classes(raw, problems)
    atts = raw.get("attractors", [])
    if not isinstance(atts, list):
        problems.append("attractors: a list")
        atts = []
    aids = []
    for i, a in enumerate(atts):
        if not isinstance(a, dict) or not isinstance(a.get("id"), str) or not a.get("id"):
            problems.append("attractors[%d]: id required" % i)
            continue
        if a["id"] in aids:
            problems.append("attractors[%d]: duplicate id %r" % (i, a["id"]))
        aids.append(a["id"])
        if a.get("rank") not in RANKS:
            problems.append("attractor %s: rank %r not in %s" % (a["id"], a.get("rank"), RANKS))
        if not isinstance(a.get("taxon"), str) or not isinstance(a.get("option"), str):
            problems.append("attractor %s: taxon and option text required" % a["id"])
        for t in a.get("confused_with", []) or []:
            if t not in names:
                problems.append("attractor %s: confused_with %r is not a target" % (a["id"], t))
    _check_stages(raw.get("stages"), problems)
    _check_exams(raw, problems)
    _check_sources(raw, problems)
    kt_ids = _check_known_truth(raw, problems)
    terms = raw.get("domain_terms", [])
    if not _strs(terms):
        problems.append("domain_terms: a list of non-empty strings")
    else:
        kinds = _provider_kinds(raw)
        for t in terms:
            if t.lower() in kinds:
                problems.append("domain_terms: %r is an authority or provider kind, which the engine "
                                "must be able to name" % t)
    judges = raw.get("judges")
    if judges is not None:
        panel = judges.get("panel", []) if isinstance(judges, dict) else None
        if not isinstance(panel, list):
            problems.append("judges.panel: a list")
        else:
            jids = []
            for j in panel:
                if not isinstance(j, dict) or not isinstance(j.get("id"), str):
                    problems.append("judges.panel: an entry without an id")
                    continue
                if j["id"] in jids:
                    problems.append("judges.panel: duplicate id %r" % j["id"])
                jids.append(j["id"])
                if j.get("kind") not in JUDGE_KINDS:
                    problems.append("judges.panel %s: unknown kind %r (known: %s)"
                                    % (j["id"], j.get("kind"), JUDGE_KINDS))
    for i, ic in enumerate(raw.get("identity_checks", []) or []):
        if not isinstance(ic, dict):
            problems.append("identity_checks[%d]: not an object" % i)
            continue
        if ic.get("class") not in names:
            problems.append("identity_checks[%d]: class %r is not a target" % (i, ic.get("class")))
        if ic.get("from") not in kt_ids:
            problems.append("identity_checks[%d]: from %r is not a known-truth set" % (i, ic.get("from")))
        if not _is_int(ic.get("n")) or ic["n"] <= 0:
            problems.append("identity_checks[%d]: n must be a positive integer" % i)
    _check_reference_labeller(raw, kt_ids, problems)
    names_cfg = raw.get("names")
    if names_cfg is not None:
        fr = names_cfg.get("frames") if isinstance(names_cfg, dict) else None
        if not isinstance(fr, dict) or any(not isinstance(fr.get(k), list)
                                           for k in ("noinfo", "named", "excluded")):
            problems.append("names.frames: noinfo, named and excluded lists required")
        else:
            flat = fr["noinfo"] + fr["named"] + fr["excluded"]
            if len(flat) != len(set(flat)):
                problems.append("names.frames: a status is in more than one frame")
    return problems


# ------------------------------------------------------------------ Domain
class Domain(object):
    """A validated domain config. Everything the engine may know about a domain
    is read through these accessors."""

    def __init__(self, raw, path, sha256):
        self.raw = raw
        self.path = Path(path)
        self.sha256 = sha256
        self.name = raw["domain"]
        self.adapter = raw["adapter"]

    def __repr__(self):
        return "Domain(%r, %s)" % (self.name, self.sha256[:12])

    # classes
    @property
    def targets(self):
        return list(self.raw["classes"]["targets"])

    @property
    def other(self):
        return dict(self.raw["classes"]["other"])

    @property
    def class_names(self):
        return [t["name"] for t in self.raw["classes"]["targets"]] + [self.raw["classes"]["other"]["name"]]

    @property
    def target_ids(self):
        return [t["id"] for t in self.raw["classes"]["targets"]]

    @property
    def target_names(self):
        return [t["name"] for t in self.raw["classes"]["targets"]]

    def target(self, name_or_id):
        for t in self.raw["classes"]["targets"]:
            if name_or_id == t["name"] or (_is_int(name_or_id) and name_or_id == t["id"]):
                return dict(t)
        raise DomainError("%r is not a target class of domain %s" % (name_or_id, self.name))

    def class_name(self, cid):
        cid = int(cid)
        other = self.raw["classes"]["other"]
        if cid == other["id"]:
            return other["name"]
        if 0 <= cid < len(self.raw["classes"]["targets"]):
            return self.raw["classes"]["targets"][cid]["name"]
        raise DomainError("class id %r is not in domain %s" % (cid, self.name))

    def class_id(self, name):
        other = self.raw["classes"]["other"]
        if name == other["name"]:
            return other["id"]
        return self.target(name)["id"]

    def is_target(self, cid):
        return cid is not None and _is_int(cid) and 0 <= cid < len(self.raw["classes"]["targets"])

    @property
    def attractors(self):
        return list(self.raw.get("attractors", []))

    def genus_unsure(self):
        return list(self.raw["classes"].get("genus_answer_unsure_for", []))

    def small_class_threshold(self):
        v = self.raw["classes"].get("small_class_train_boxes_below")
        if v is None:
            raise DomainError("classes.small_class_train_boxes_below is not set in %s" % self.name)
        return int(v)

    # labeller options
    def _rl(self):
        rl = self.raw.get("reference_labeller")
        if not isinstance(rl, dict):
            raise DomainError("domain %s has no reference_labeller section" % self.name)
        return rl

    def options(self):
        """The blind multiple-choice list (runner §4.10): the targets in config
        order, then the attractors, then options_tail; numbered from 1."""
        tail = self._rl().get("options_tail")
        if not tail:
            raise DomainError("reference_labeller.options_tail is not set in %s" % self.name)
        other_id = self.raw["classes"]["other"]["id"]
        out = []
        for t in self.raw["classes"]["targets"]:
            rank = t.get("rank", "species")
            text = t.get("option")
            if not text:
                if not t.get("taxon"):
                    raise DomainError("target %s has neither an option text nor a taxon" % t["name"])
                label = t.get("common") or t["name"]
                text = ("%s (%s spp.)" % (label, t["taxon"]) if rank == "genus"
                        else "%s (%s)" % (label, t["taxon"]))
            out.append({"n": len(out) + 1, "text": text, "kind": "target", "class": t["id"],
                        "attractor": None, "rank": rank, "taxon": t.get("taxon"), "answer": t["name"]})
        for a in self.raw.get("attractors", []):
            out.append({"n": len(out) + 1, "text": a["option"], "kind": "attractor", "class": other_id,
                        "attractor": a["id"], "rank": a["rank"], "taxon": a["taxon"], "answer": "other"})
        for text, ans in zip(tail, TAIL_ANSWERS):
            out.append({"n": len(out) + 1, "text": text, "kind": "tail",
                        "class": other_id if ans == "other" else None, "attractor": None,
                        "rank": None, "taxon": None, "answer": ans})
        return out

    def pair_options(self):
        po = self._rl().get("pair_options")
        if not po:
            raise DomainError("reference_labeller.pair_options is not set in %s" % self.name)
        return [{"n": i + 1, "text": t, "answer": a} for i, (t, a) in enumerate(zip(po, PAIR_ANSWERS))]

    # stages
    @property
    def stages(self):
        return list(self.raw["stages"])

    def stage(self, stage_id):
        for s in self.raw["stages"]:
            if s["id"] == stage_id:
                return dict(s)
        raise DomainError("no stage %r in domain %s" % (stage_id, self.name))

    def stages_with_role(self, role):
        return [s["id"] for s in self.raw["stages"] if s.get("role") == role]

    def recoverable_stages(self):
        return [s["id"] for s in self.raw["stages"] if s.get("recoverable") is True]

    # sources and labs
    def lab_groups(self):
        return {g: list(v) for g, v in (self.raw["sources"].get("lab_groups") or {}).items()}

    def lab_of(self, source):
        for g, members in sorted((self.raw["sources"].get("lab_groups") or {}).items()):
            if source in members:
                return g
        return "src:%s" % source

    @property
    def reference_source(self):
        return self.raw["sources"]["reference"]

    def reference_lab(self):
        return self.lab_of(self.raw["sources"]["reference"])

    def authoritative_sources(self):
        return sorted((self.raw["sources"].get("authoritative") or {}).keys())

    def not_recoverable_sources(self):
        return dict(self.raw["sources"].get("not_recoverable") or {})

    # known truth
    def kt_ids(self):
        return _kt_ids(self.raw)

    def kt(self, kt_id):
        if kt_id in KT_SPECIAL_KEYS or kt_id not in self.raw["known_truth"]:
            raise DomainError("no known-truth set %r in domain %s" % (kt_id, self.name))
        return dict(self.raw["known_truth"][kt_id])

    def qualify_rl_on(self):
        return list(self.raw["known_truth"].get("qualify_rl_on", []))

    def claimed_sets(self, hypothesis=None):
        """KT ids whose labels a hypothesis tests (all claimed sets, or those of one hypothesis)."""
        out = []
        for k in self.kt_ids():
            cb = self.raw["known_truth"][k].get("claimed_by")
            if cb and (hypothesis is None or cb == hypothesis):
                out.append(k)
        return out

    def independent_sets(self):
        return [k for k in self.kt_ids() if self.raw["known_truth"][k].get("independent") is True]

    def forbidden_splits(self):
        return list(self.raw["known_truth"].get("forbidden", []))

    # exams
    def exam_splits(self):
        ex = self.raw["exams"]
        return {"decision": ex["decision"], "non_decision": tuple(ex.get("non_decision", [])),
                "extra_non_decision": tuple(ex.get("extra_non_decision", []))}

    def non_dev_exams(self):
        ex = self.raw["exams"]
        return tuple(ex.get("non_decision", [])) + tuple(ex.get("extra_non_decision", []))

    # config sections the engine reads as a whole
    def section(self, key, required=True):
        v = self.raw.get(key)
        if v is None and required:
            raise DomainError("domain %s has no %r section" % (self.name, key))
        return copy.deepcopy(v)

    def terms(self):
        """Grep terms for the domain-free test (runner §7.5), from this config
        only: substrings (domain_terms and every source slug the config names)
        and whole tokens (the domain name, class names, taxon words of five or
        more letters, non-decision exam names, lab group names). The fixed
        cross-domain list lives in the test, not in engine code."""
        raw = self.raw
        sub = set(t.lower() for t in raw.get("domain_terms", []))
        src = raw.get("sources", {})
        slugs = set()
        for members in (src.get("lab_groups") or {}).values():
            slugs.update(members)
        for key in ("authoritative", "licences", "not_recoverable", "capture_stem_regex",
                    "card_image_counts", "card_resolvers"):
            slugs.update((src.get(key) or {}).keys())
        for k in self.kt_ids():
            slugs.update(raw["known_truth"][k].get("sources", []) or [])
        slugs.update(((raw.get("sampling") or {}).get("G5_allocation") or {}).keys())
        for pair in ((raw.get("leak") or {}).get("negative_source_pairs") or []):
            slugs.update(pair)
        # Not domain terms: a known-truth set's pseudo-source named after the
        # set (the runner's file vocabulary, e.g. <set>/ and crops_<set>.csv)
        # and the reference source, whose name is part of pinned field names
        # (census train_core_boxes_per_class, the prereg's screening keys).
        skip = {k.lower() for k in self.kt_ids()} | {str(src.get("reference") or "").lower()}
        sub.update(s.lower() for s in slugs if s and s.lower() not in skip)
        tok = {raw["domain"].lower()}
        for n in self.class_names:
            tok.add(re.sub(r"[^a-z0-9]", "", n.lower()))
        taxa = []
        for t in raw["classes"]["targets"]:
            taxa.append(t.get("taxon") or "")
            taxa.extend(t.get("not", []) or [])
        for a in raw.get("attractors", []) or []:
            taxa.append(a.get("taxon") or "")
        kt7 = raw["known_truth"].get("kt7") or {}
        if isinstance(kt7, dict):
            taxa.extend(kt7.get("taxa", []) or [])
        for tx in taxa:
            for w in _WORD_RE.findall(str(tx).lower()):
                if len(w) >= 5:
                    tok.add(w)
        for e in self.non_dev_exams():
            if e not in ("dev", "test"):
                tok.add(re.sub(r"[^a-z0-9]", "", e.lower()))
        for g in (src.get("lab_groups") or {}):
            tok.add(re.sub(r"[^a-z0-9]", "", g.lower()))
        tok.discard("")
        return {"substring": sorted(sub), "token": sorted(tok)}


def _read_bytes(path, err):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as e:
        raise err("cannot read %s (%s)" % (path, e))


def resolve_path(name_or_path):
    """A config path from a path or a name: $FUNNEL_DOMAINS_DIR/<name>.json
    when that variable is set (tests with synthetic domains), else
    domains/<name>.json."""
    s = str(name_or_path)
    if s.endswith(".json") or os.sep in s or "/" in s:
        return Path(s)
    if not _NAME_RE.match(s):
        raise DomainError("%r is neither a domain name nor a config path" % s)
    base = os.environ.get("FUNNEL_DOMAINS_DIR")
    return (Path(base) if base else DOMAINS_DIR) / ("%s.json" % s)


def load(name_or_path):
    """A Domain from a name ("x" -> domains/x.json) or a path. DomainError lists
    every validation problem."""
    if isinstance(name_or_path, Domain):
        return name_or_path
    path = resolve_path(name_or_path)
    data = _read_bytes(path, DomainError)
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise DomainError("%s is not JSON (%s)" % (path, e))
    problems = validate(raw)
    if problems:
        raise DomainError("%s: %d problem(s): %s" % (path, len(problems), "; ".join(problems)))
    return Domain(raw, path, hashlib.sha256(data).hexdigest())


# ------------------------------------------------------------------ prereg
def prereg_core_sha256(obj):
    """sha256 of the canonical JSON of the prereg without its amendments."""
    core = {k: v for k, v in obj.items() if k != "amendments"}
    return hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()


def contract_path(prereg, prereg_path=None):
    """The contract file a prereg names: $FUNNEL_CONTRACT when set, else REPO /
    the prereg's contract path when that file exists, else the first ancestor
    of the prereg file's directory that holds the contract path (the lab keeps
    its package tree under ~/weed_llm_benchmark and the docs beside it, so its
    REPO is not the git root). The caller checks the file's sha256 against the
    prereg, so a search cannot substitute another contract."""
    raw = prereg.raw if isinstance(prereg, Prereg) else prereg
    env = os.environ.get("FUNNEL_CONTRACT")
    if env:
        return Path(env)
    try:
        rel = raw["contract"]["path"]
    except (KeyError, TypeError):
        raise PreregError("the prereg names no contract path")
    first = C.REPO / rel
    if first.is_file():
        return first
    if prereg_path is None and isinstance(prereg, Prereg):
        prereg_path = prereg.path
    if prereg_path is not None:
        for anc in Path(prereg_path).resolve().parents:
            if (anc / rel).is_file():
                return anc / rel
    return first


class Prereg(object):
    def __init__(self, raw, path, sha256, contract_file, contract_sha256):
        self.raw = raw
        self.path = Path(path)
        self.sha256 = sha256
        self.core_sha256 = prereg_core_sha256(raw)
        self.domain_name = raw["domain"]
        self.contract_path = Path(contract_file)
        self.contract_sha256 = contract_sha256

    def __repr__(self):
        return "Prereg(%s, core %s)" % (self.path.name, self.core_sha256[:12])

    @property
    def groups(self):
        """{group: (planned, minimum)} from sampling.groups."""
        g = (self.raw.get("sampling") or {}).get("groups")
        if not isinstance(g, dict):
            raise PreregError("prereg has no sampling.groups")
        out = {}
        for k, v in g.items():
            if not isinstance(v, list) or len(v) != 2 or not _is_int(v[0]):
                raise PreregError("sampling.groups.%s must be [planned, minimum|null]" % k)
            out[k] = (int(v[0]), None if v[1] is None else int(v[1]))
        return out

    @property
    def amendments(self):
        return list(self.raw.get("amendments", []))

    @property
    def sample_lock(self):
        locks = [a for a in self.raw.get("amendments", []) if a.get("kind") == "sample_lock"]
        if len(locks) > 1:
            raise PreregError("%s holds %d sample locks" % (self.path, len(locks)))
        return dict(locks[0]) if locks else None

    def record(self):
        return {"path": str(self.path), "sha256": self.sha256, "core_sha256": self.core_sha256}

    def hypothesis(self, hid):
        h = (self.raw.get("hypotheses") or {}).get(hid)
        if h is None:
            raise PreregError("the prereg has no hypothesis %r" % hid)
        return copy.deepcopy(h)


def check_prereg_domain(prereg, domain):
    """The prereg and the config must name the same domain."""
    name = domain.name if isinstance(domain, Domain) else str(domain)
    if prereg.domain_name != name:
        raise DomainError("the prereg is for domain %r, the config for %r" % (prereg.domain_name, name))


def load_prereg(path, domain=None):
    """A Prereg with its format and its contract's sha256 checked. With a
    domain (a Domain, a name or a config path), the two must name the same
    domain."""
    path = Path(path)
    data = _read_bytes(path, PreregError)
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise PreregError("%s is not JSON (%s)" % (path, e))
    if not isinstance(raw, dict) or raw.get("format") != PREREG_FORMAT:
        raise PreregError("%s: format %r is not %s" % (path, raw.get("format") if isinstance(raw, dict)
                                                         else None, PREREG_FORMAT))
    if not isinstance(raw.get("domain"), str) or not _NAME_RE.match(raw["domain"]):
        raise PreregError("%s: domain %r" % (path, raw.get("domain")))
    if not isinstance(raw.get("amendments", []), list):
        raise PreregError("%s: amendments must be a list" % path)
    want = (raw.get("contract") or {}).get("sha256")
    cpath = contract_path(raw, prereg_path=path)
    cdata = _read_bytes(cpath, PreregError)
    got = hashlib.sha256(cdata).hexdigest()
    if got != want:
        raise PreregError("contract %s hashes to %s, the prereg records %s"
                          % (cpath, got[:12], str(want)[:12]))
    pre = Prereg(raw, path, hashlib.sha256(data).hexdigest(), cpath, got)
    if domain is not None:
        check_prereg_domain(pre, load(domain) if not isinstance(domain, Domain) else domain)
    return pre


def load_pair(prereg_path):
    """(Prereg, Domain): the prereg and the config it names, checked together."""
    pre = load_prereg(prereg_path)
    dom = load(pre.domain_name)
    check_prereg_domain(pre, dom)
    return pre, dom


def append_amendment(path, amendment):
    """Append one dated amendment to the prereg file, atomically. Refuses when
    the amendment is malformed or repeats an id, when the file no longer loads
    (contract changed), when an amendment names a prereg core other than the
    file's (the file was edited outside its amendments), or when the rewrite
    would change anything but the amendments list. Returns the new raw prereg."""
    path = Path(path)
    pre = load_prereg(path)
    if not isinstance(amendment, dict):
        raise PreregError("an amendment is a JSON object")
    for key in ("id", "kind", "date"):
        if not isinstance(amendment.get(key), str) or not amendment.get(key):
            raise PreregError("an amendment needs %r" % key)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", amendment["date"]):
        raise PreregError("amendment date %r is not YYYY-MM-DD" % amendment["date"])
    if any(a.get("id") == amendment["id"] for a in pre.amendments):
        raise PreregError("amendment id %r already exists" % amendment["id"])
    named = amendment.get("prereg_core_sha256")
    if named is not None and named != pre.core_sha256:
        raise PreregError("the amendment names prereg core %s but the file's core is %s: the prereg was "
                          "edited outside its amendments" % (str(named)[:12], pre.core_sha256[:12]))
    if amendment["kind"] == "sample_lock" and pre.sample_lock is not None:
        raise PreregError("the prereg already holds a sample lock (%s)" % pre.sample_lock.get("id"))
    try:
        canonical_json(amendment)
    except Exception as e:
        raise PreregError("amendment is not JSON (%s)" % e)
    new = copy.deepcopy(pre.raw)
    new["amendments"] = list(pre.amendments) + [copy.deepcopy(amendment)]
    if prereg_core_sha256(new) != pre.core_sha256:
        raise PreregError("refusing an amendment that changes the prereg outside its amendments")
    if {k: v for k, v in new.items() if k != "amendments"} != \
            {k: v for k, v in pre.raw.items() if k != "amendments"}:
        raise PreregError("refusing an amendment that changes the prereg outside its amendments")
    text = json.dumps(new, indent=1, ensure_ascii=False)
    _atomic_write_bytes(path, text.encode("utf-8"))
    return new


def next_amendment_id(prereg):
    return "A%d" % (len(prereg.amendments) + 1)
