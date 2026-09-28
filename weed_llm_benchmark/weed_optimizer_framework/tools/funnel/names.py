"""Name status v2 (contract §3.1 S7b, runner §4.4): how informative each source
class name is, decided through the taxonomy authority and the config's word
lists instead of hand lists alone.

A source class name is informative when it resolves to a taxon (taxonomy.py,
lever L12) and uninformative when it does not. The config's word lists
(domains/<domain>.json "names") catch the names an authority cannot judge:
empty and numeric names, object and state names, role names ("crop") and the
generic words. The rules run in RULE_ORDER; the first that applies decides:

    target          the adapter's join gave the class a target label
    no_name         empty key
    numeric         names.numeric_regex matches the whole key
    state           a state word inside the key, or a state key
    non_object      an object word inside the key, or an object key
    role            the key is a role name
    generic         a generic key, or names.generic_regex matches the whole key
    target_synonym  a scientific or override resolution to a target taxon at the
                    target's rank, or to a descendant of it
    target_related  a resolution to a taxon in a target's genus or in a target's
                    "not" list, or the key holds a related token (and is not an
                    allowed key)
    taxon_resolved  any other scientific or vernacular resolution
    unresolvable    nothing above

Frames (names.frames in the config) group the statuses for the strata:
noinfo, named and excluded; "target" is its own frame.

The v1 status (the Step 1 verifier's own rule) is computed by the adapter;
this module never names v1's status strings. Nothing here names a domain.
"""
from __future__ import annotations

import collections
import re

from . import NamesError

RULE_ORDER = ("target", "no_name", "numeric", "state", "non_object", "role", "generic",
              "target_synonym", "target_related", "taxon_resolved", "unresolvable")
STATUSES = RULE_ORDER
FRAMES = ("noinfo", "named", "excluded")
VIAS = ("join", "pattern", "scientific", "vernacular", "override", "none")


def key(name):
    """Lower case, letters and digits only: 'Field Pea' and 'field_pea' agree."""
    return re.sub(r"[^a-z0-9]", "", str(name if name is not None else "").lower())


class _Patterns(object):
    """The config's names section, compiled once per domain."""

    def __init__(self, domain):
        cfg = domain.section("names")
        need = ("numeric_regex", "generic_keys", "generic_regex", "non_object_words", "state_words",
                "role_names", "related_tokens", "related_allowed_keys", "frames")
        missing = [k for k in need if k not in cfg]
        if missing:
            raise NamesError("domain %s: names section lacks %s" % (domain.name, missing))
        try:
            self.numeric = re.compile(cfg["numeric_regex"])
            self.generic_re = re.compile(cfg["generic_regex"])
        except re.error as e:
            raise NamesError("domain %s: a names regex does not compile (%s)" % (domain.name, e))
        self.generic_keys = frozenset(key(w) for w in cfg["generic_keys"])
        self.non_object_words = tuple(key(w) for w in cfg["non_object_words"] if key(w))
        self.non_object_keys = frozenset(key(w) for w in cfg.get("non_object_keys", []))
        self.state_words = tuple(key(w) for w in cfg["state_words"] if key(w))
        self.state_keys = frozenset(key(w) for w in cfg.get("state_keys", []))
        self.role_names = frozenset(key(w) for w in cfg["role_names"])
        self.related_tokens = tuple(key(w) for w in cfg["related_tokens"] if key(w))
        self.related_allowed = frozenset(key(w) for w in cfg["related_allowed_keys"])
        frames = cfg["frames"]
        self.frame = {}
        for fr in FRAMES:
            for s in frames.get(fr, []):
                if s not in STATUSES:
                    raise NamesError("domain %s: names.frames.%s holds unknown status %r" % (domain.name, fr, s))
                if s in self.frame:
                    raise NamesError("domain %s: status %r is in two frames" % (domain.name, s))
                self.frame[s] = fr
        lost = [s for s in STATUSES if s != "target" and s not in self.frame]
        if lost:
            raise NamesError("domain %s: names.frames leaves statuses %s in no frame" % (domain.name, lost))


_CACHE = {}


def _patterns(domain):
    k = (getattr(domain, "path", None), getattr(domain, "sha256", None))
    p = _CACHE.get(k)
    if p is None:
        p = _CACHE[k] = _Patterns(domain)
    return p


def frame_of(status, domain):
    """"noinfo" | "named" | "excluded" | "target" for a v2 status."""
    if status == "target":
        return "target"
    fr = _patterns(domain).frame.get(status)
    if fr is None:
        raise NamesError("status %r is in no frame of domain %s" % (status, domain.name))
    return fr


def _pattern_status(k, P):
    if not k:
        return "no_name"
    if P.numeric.fullmatch(k):
        return "numeric"
    if k in P.state_keys or any(w in k for w in P.state_words):
        return "state"
    if k in P.non_object_keys or any(w in k for w in P.non_object_words):
        return "non_object"
    if k in P.role_names:
        return "role"
    if k in P.generic_keys or P.generic_re.fullmatch(k):
        return "generic"
    return None


def status_v2(name, joined_target, domain, resolver):
    """{"status", "via", "taxon", "rank"} of one source class name (module
    docstring). The resolver is consulted only when no pattern rule applies;
    an offline resolver without the name raises TaxonomyError (lever L12)."""
    if joined_target:
        return {"status": "target", "via": "join", "taxon": None, "rank": None}
    P = _patterns(domain)
    k = key(name)
    st = _pattern_status(k, P)
    if st is not None:
        return {"status": st, "via": "none" if st == "no_name" else "pattern", "taxon": None, "rank": None}
    res = resolver.resolve(name)
    via = (res or {}).get("via") or "none"
    taxon = (res or {}).get("accepted") or (res or {}).get("canonical")
    rank = (res or {}).get("rank")
    targets = domain.targets
    if res is not None and via in resolver.mappable_via:
        if any(resolver.is_target_synonym(res, t) for t in targets):
            return {"status": "target_synonym", "via": via, "taxon": taxon, "rank": rank}
    if res is not None and via != "none" and any(resolver.is_relative(res, t) for t in targets):
        return {"status": "target_related", "via": via, "taxon": taxon, "rank": rank}
    if k not in P.related_allowed and any(t in k for t in P.related_tokens):
        return {"status": "target_related", "via": via if via != "none" else "pattern",
                "taxon": taxon if via != "none" else None, "rank": rank if via != "none" else None}
    # a resolution that may map a class (a scientific one, or a project override) is
    # informative too: an override that names a non-target taxon resolves the name
    if res is not None and via != "none" and (via in resolver.informative_via or via in resolver.mappable_via):
        return {"status": "taxon_resolved", "via": via, "taxon": taxon, "rank": rank}
    return {"status": "unresolvable", "via": "none", "taxon": None, "rank": None}


def build_name_status(rows, domain, resolver, contract_check=None):
    """The content of name_status_v2.json (without the header, which the
    caller adds with its inputs).

    rows: [{"source", "src_id", "name", "joined_target", "status_v1", "boxes",
    "conflicts"}], one per source class. contract_check: {label: {"status":
    status, "keys": [name key] (optional), "contract": number | None}}; each
    label gets the boxes of the rows with that v2 status (and, with keys, one
    of those keys), next to the contract's number, recorded and not
    asserted."""
    names, by_status = [], collections.OrderedDict((s, {"names": 0, "boxes": 0, "conflicts": 0})
                                                   for s in STATUSES)
    seen = set()
    for r in rows:
        for f in ("source", "src_id", "name", "joined_target", "status_v1", "boxes", "conflicts"):
            if f not in r:
                raise NamesError("name status row lacks %r: %r" % (f, r))
        ident = (r["source"], str(r["src_id"]))
        if ident in seen:
            raise NamesError("name status rows repeat class %s|%s" % ident)
        seen.add(ident)
        s = status_v2(r["name"], bool(r["joined_target"]), domain, resolver)
        names.append({"source": r["source"], "src_id": str(r["src_id"]), "name": r["name"],
                      "key": key(r["name"]), "status_v1": r["status_v1"], "status_v2": s["status"],
                      "via": s["via"], "taxon": s["taxon"], "rank": s["rank"],
                      "boxes": int(r["boxes"]), "conflicts": int(r["conflicts"])})
        b = by_status[s["status"]]
        b["names"] += 1
        b["boxes"] += int(r["boxes"])
        b["conflicts"] += int(r["conflicts"])
    names.sort(key=lambda n: (n["source"], n["src_id"]))
    P = _patterns(domain)
    frames = {fr: [s for s in STATUSES if P.frame.get(s) == fr] for fr in FRAMES}
    out = {"frozen": True, "rule_order": list(RULE_ORDER), "names": names,
           "by_status": dict(by_status), "frames": frames,
           "by_frame": {fr: {"boxes": sum(by_status[s]["boxes"] for s in frames[fr]),
                             "conflicts": sum(by_status[s]["conflicts"] for s in frames[fr])}
                        for fr in FRAMES}}
    if contract_check:
        cc = {}
        for label in sorted(contract_check):
            spec = contract_check[label]
            want = spec.get("status")
            if want not in STATUSES:
                raise NamesError("contract_check %s: unknown status %r" % (label, want))
            keys = set(key(k) for k in spec.get("keys", [])) if spec.get("keys") is not None else None
            cc[label] = sum(n["boxes"] for n in names
                            if n["status_v2"] == want and (keys is None or n["key"] in keys))
        cc["contract"] = {label: contract_check[label].get("contract") for label in sorted(contract_check)}
        out["contract_check"] = cc
    return out


def status_table(name_status):
    """{(source, src_id): v2 status} of a name_status_v2.json content."""
    return {(n["source"], n["src_id"]): n["status_v2"] for n in name_status["names"]}
