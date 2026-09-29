"""Class list -> class id, with provenance (docs/CONTINUOUS_LOOP.md §3.2 "Class
map").

For each source class (id, name, hints) the names tried, in order:
  card   the source's card class table (config card_class_tables, else the
         funnel domain's card resolver, read only), by class id then by name:
         its taxon decides, whatever the name says (a card is how a numeric or
         unresolvable name becomes a target, FUNNEL_AUDIT H3a);
  eppo   the binomial of an EPPO code (the config's pinned EPPO table);
  hint   a taxon the annotation format carries (e.g. a category's "taxon"
         field, or "<role>: <taxon> (<qualifier>)" names);
  name   the class name itself.
Each is judged by funnel.names.status_v2 through the offline resolver
(names.Names), with the project's alias table as the join (status "target",
via "join"). The first informative answer decides (target, target_synonym,
taxon_resolved, target_related, role); a name the caches lack before any
informative answer makes the class pending (lever L26 resolves it, and intake
refuses until then). The status becomes a class id through the config's
class_map.status_map: "target" -> the target's id (0..n-1), "other" -> the
reject class id, "unmapped" -> class_space.unmapped_id (a class id that exists
only in intake labels, never in a training label; its boxes are kept so that
per-box admission can mask them).

The provenance records the sha256 of the card table, the funnel taxonomy
cache, the names layer, the EPPO table, the status map and the funnel domain
config, and the status of each name.
"""
from __future__ import annotations

import re

from . import sha256_json

INFORMATIVE = ("target", "target_synonym", "taxon_resolved", "target_related", "role")
SPECIES_RANKS = ("species", "subspecies", "variety", "subvariety", "form", "subform", "infraspecific_name",
                 "cultivar", "cultivar_group")
_ROLE_TAXON = re.compile(r"^\s*[A-Za-z_ -]{1,20}:\s*(?P<taxon>[^()]+?)\s*(\(.*\))?\s*$")


def role_taxon_hint(name):
    """"<role>: <taxon> (<qualifier>)" -> "<taxon>" (a WeedCOCO-style category
    name), else None."""
    m = _ROLE_TAXON.match(str(name or ""))
    if not m:
        return None
    t = m.group("taxon").strip()
    return t or None


def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def queries_for(cl, cfg, card):
    """[(basis, name)] in precedence order (module docstring)."""
    out = []
    name = cl.get("name")
    if card is not None:
        tab = card[0]
        e = None
        if cl.get("id") is not None:
            e = tab["by_id"].get(str(cl["id"]))
        if e is None:
            e = tab["by_name"].get(_key(name))
        if e is not None and e.get("taxon"):
            out.append(("card", e["taxon"]))
    b = cfg.eppo_binomial(name)
    if b:
        out.append(("eppo", b))
    hints = list(cl.get("hints") or [])
    rt = role_taxon_hint(name)
    if rt and rt not in hints:
        hints.append(rt)
    for h in hints:
        out.append(("hint", cfg.eppo_binomial(h) or h))
    out.append(("name", name))
    return out


def inc_of(st, cfg, targets_by_name):
    """(class id, kind, target name or None) of a status record; kind is
    "target", "other" or "unmapped"."""
    cm = cfg.raw["class_map"]
    sm = cm["status_map"]
    v = sm.get(st["status"], sm["default"])
    if isinstance(v, dict):
        v = v.get(st.get("via"), v.get("default", sm["default"]))
    if (st["status"] == "target_related" and cm.get("related_needs_species_rank")
            and str(st.get("rank") or "").lower() not in SPECIES_RANKS):
        v = "unmapped"          # a target's genus (or a relative named above species rank) may be the target
    if v == "target":
        if st.get("target") and not st.get("ambiguous") and st["target"] in targets_by_name:
            return int(targets_by_name[st["target"]]["id"]), "target", st["target"]
        v = "unmapped"
    if v == "other":
        return cfg.other_id, "other", None
    return cfg.unmapped_id, "unmapped", None


def reject_token(name, cfg):
    """The first reject word (prefilter.reject_tokens) that is a whole token of
    the class name, else None: a short reject word rejects a class of that
    name but never a longer name that merely starts with its letters."""
    toks = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name or "")).lower()))
    for w in cfg.raw["prefilter"]["reject_tokens"]:
        if w.lower() in toks:
            return w
    return None


def map_class(cl, cfg, names, targets, card=None):
    """The class-map entry of one source class (module docstring)."""
    by_name = {t["name"]: t for t in cfg.targets}
    tried, chosen, fallback, pending = [], None, None, None
    for basis, q in queries_for(cl, cfg, card):
        st = names.status(q, targets.alias_target(q))
        tried.append({"basis": basis, "query": q, "status": None if st is None else st["status"]})
        if st is None:
            pending = q
            break
        if basis == "card" or st["status"] in INFORMATIVE:
            chosen = (basis, q, st)
            break
        if fallback is None:
            fallback = (basis, q, st)
    rt = reject_token(cl.get("name"), cfg)
    base = {"src_id": cl.get("id"), "name": cl.get("name"), "hints": list(cl.get("hints") or []),
            "boxes": cl.get("boxes"), "reject_token": rt, "tried": tried}
    if chosen is None and pending is not None:
        return dict(base, pending=True, pending_name=pending, inc_id=cfg.unmapped_id, cls="unmapped",
                    status=None, via=None, taxon=None, target=None, basis=None, query=None)
    if chosen is None:
        chosen = fallback
    basis, q, st = chosen
    # a reject word never outvotes the authority: it is recorded, and the prefilter
    # uses it only for a candidate with no target class
    inc_id, cls, tname = inc_of(st, cfg, by_name)
    return dict(base, pending=False, pending_name=None, inc_id=inc_id, cls=cls, status=st["status"],
                via=st.get("via"), taxon=st.get("taxon"), target=tname, ambiguous=bool(st.get("ambiguous")),
                basis=basis, query=q)


def build(source_id, classes, cfg, names, targets):
    """The class map of a source: {"classes": [entry], "by_src": {src id: class
    id}, "pending": [names], "provenance": {...}}."""
    card = cfg.card_table(source_id)
    entries = [map_class(cl, cfg, names, targets, card) for cl in classes]
    pending = sorted({e["pending_name"] for e in entries if e["pending"]})
    prov = {
        "card": None if card is None else {"origin": card[2], "sha256": card[1]},
        "taxonomy_cache": names.provenance.get("funnel_cache"),
        "names_layer": names.provenance.get("names_layer"),
        "eppo": cfg.eppo_record,
        "status_map_sha256": sha256_json(cfg.raw["class_map"]["status_map"]),
        "funnel_domain": {"path": str(cfg.funnel.path), "sha256": cfg.funnel.sha256},
        "alias_table": cfg.raw.get("alias_table"),
        "statuses": {str(e["src_id"]): e["status"] for e in entries},
    }
    return {"classes": entries, "by_src": {e["src_id"]: e["inc_id"] for e in entries},
            "pending": pending, "provenance": prov,
            "target_ids": sorted({e["inc_id"] for e in entries if e["cls"] == "target"})}
