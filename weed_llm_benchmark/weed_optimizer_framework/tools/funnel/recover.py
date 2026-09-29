"""Recovery (lever L13): an overlay of recovered labels and masked copies,
never an edit (contract docs/FUNNEL_AUDIT.md §7, §9.2, DEC-7; runner
docs/FUNNEL_AUDIT_RUNNER.md §4.18, §5.4.4).

    python -m <package>.tools.funnel recover --prereg PATH --audit A --maps M --policy R-A,R-C,R-T,R-V,R-J
    python -m <package>.tools.funnel recover --arms --realloop EXP_DIR --base BASE --prereg PATH

Everything is written under INC_DIR/step1_r1/ (content-addressed where a file
is named by its content):
  labels_overlay/<source>/<key>.<sha16>.txt   recovered labels (INC format)
  images_masked/<source>/<key>.<sha16>.png    EXIF-transposed, lossless copies
                                              whose masked boxes (pixel box
                                              rounded outward) hold the image's
                                              mean RGB
  recovered_pool.jsonl                        one row per recovered image
  domain_dev.jsonl, domain_dev.json           the H10d hold-out
  recovery.json                               gates, maps, guards, counts, licences
  arms/*.jsonl, arms/arms.json                the realloop_v2 panel arms (--arms)

Source labels and the step-1 labels are never written; every step-1 label
read is re-hashed after the write (source_labels_unchanged).

Gates are read from the audit entry of the quantity they bound, in
estimate.py's event vocabulary, never from another estimate of the stratum:
the box gate of R-A and R-V from "label" (source label right, box valid), of
R-J from "pred" (the step-1 probe's predicted class right, box valid), a
class unit's class gate from "purity:<class>" and R-C's box gate from
"label:<class>" (answered as <class>, box valid), <class> being the unit's
card map or scientific synonym (unit_classes). A stratum without that entry
fails the gate and says which estimate is missing.

Policies (one per image, in the order VETO > AUTH > CLASS > JUDGE):
  R-V  images holding a verified target box the image rule vetoed: verified
       and other_ok boxes are kept, unknown, conflict and failed boxes (and
       small target boxes) are masked; box gate on the box's G2v stratum.
  R-A  non-admitted images of the authoritative sources: target boxes keep
       the source label when their G2 stratum passes the box gate, H1-pre
       passed and H1 is supported (and, for a class the config's identity
       checks put before R-A, the check is recorded as passed: a missing
       record masks the class); other target boxes are masked.
       Named non-target boxes stay the other class whatever the step-1 probe
       says (the sibling guard); a no-information other box the probe calls
       a target is masked.
  R-C  images of a source with an accepted class map (card + geometry,
       H3a supported, class and box gates on the class unit): mapped boxes
       take the mapped class, every other id stays the other class.
  R-T  classes whose name resolves scientifically to a target (target
       synonym), class gate on the class unit.
  R-J  other-class boxes that every qualified judge allowed for the box
       calls the same target, that target being the step-1 probe's predicted
       class (the event the box gate measured), H2a supported, box gate on the
       box's stratum, and not sibling-guarded (a named class, or a taxon that
       is a relative of that target); in an H4 (other_ok) stratum the voters
       must also be qualified against the independent set's attractors. Any
       other box that any judge with crop scores calls a target is masked.
  R-F  refetched images, only with the Step 1 guard-chain record
       (refetch/chain_v1.json), then as R-C: H3a supported and every target
       box's class the class of an accepted card map whose gates passed.

Preconditions: the audit is valid and did not stop; the copy detector's
record (domain.leak_record: leak_v2.json when it exists; once an amendment of
the prereg requires detector version 2 (contract §14 A2), leak_v2.json and
never leak_v1.json, whose quarantine is the superseded first reading) records
a calibrated detector and an H6(b) result without an incident.

Never recoverable: guard stages, small boxes, sources the config marks not
recoverable, sources H6(a) links to an evaluation set (the whole source), the
H10d hold-out, base B's images. A genus answer in a genus the class policy
leaves unsure is masked.

Guard: every recovered row passes the never-train index on the unmasked
image (its pool dHash) and on the masked copy (the file's dHash), and the H6
copy detector on both. A hit raises NeverTrainHit after recovery.json is
written with status "refused" (a complete record already there is replaced
only with --force). A rerun with other inputs and no --force is refused
before recovered_pool.jsonl, domain_dev.* or recovery.json is written.

Nothing here names a domain.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import select as S
from ..inc import verify as V
from . import (FUNNEL_DIR, R1_DIR, NeverTrainHit, RecoverError, StaleInput,
               _atomic_write_bytes, canonical_json, check_records, file_record, header, json_text, read_csv,
               read_json, rng, sha256_bytes, strip_volatile, write_json_atomic)
from . import qualify as Q

POLICIES = ("R-A", "R-C", "R-T", "R-V", "R-J", "R-F")
POOL_OF = {"R-V": "VETO", "R-A": "AUTH", "R-C": "CLASS", "R-T": "CLASS", "R-J": "JUDGE", "R-F": "FETCH"}
PRECEDENCE = ("R-V", "R-A", "R-C", "R-T", "R-J", "R-F")
GROUP_POLICY = {"G2v": "R-V", "G2": "R-A", "G3": "R-C", "G1": "R-J", "G4": "R-J"}
EXTRA_KEYS = ("pool", "policy", "strata", "unmasked_image", "unmasked_sha256", "masked_boxes", "ctl_label",
              "ctl_label_sha256", "provenance_group", "lab", "licence", "near_dup3", "dhash_unmasked",
              "dhash_masked")
ARM_STEPS = {"CLASS_ctl": "REC-CLASS-1", "JUDGE_ctl": "REC-JUDGE-1"}
CHAIN_FILE = Path("refetch") / "chain_v1.json"
CHAIN_CHECKS = ("never_train", "core_copy", "exact_dup", "h6", "embedded")


def log(msg):
    print("[funnel.recover] %s" % msg, flush=True)


# ------------------------------------------------------------------- masks
def mask(image_path, boxes, out_dir, key=None):
    """A lossless PNG copy of the EXIF-transposed image whose boxes (normalised
    (cx, cy, w, h) or (cls, cx, cy, w, h)), each rounded outward to whole
    pixels, are filled with the image's mean RGB (rounded). Named
    <key>.<sha16>.png under out_dir, written once. Returns (path, sha256)."""
    import io
    import numpy as np
    from PIL import Image, ImageOps
    try:
        with Image.open(image_path) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
    except Exception as e:
        raise RecoverError("cannot open %s to mask it (%s: %s)" % (image_path, type(e).__name__, e))
    a = np.asarray(im).copy()
    H, W = a.shape[:2]
    mean = np.rint(a.reshape(-1, 3).astype(np.float64).mean(axis=0)).astype(np.uint8)
    for b in boxes:
        cx, cy, w, h = [float(v) for v in (b[1:5] if len(b) == 5 else b[:4])]
        x0 = max(0, int(math.floor((cx - w / 2.0) * W)))
        x1 = min(W, int(math.ceil((cx + w / 2.0) * W)))
        y0 = max(0, int(math.floor((cy - h / 2.0) * H)))
        y1 = min(H, int(math.ceil((cy + h / 2.0) * H)))
        if x1 > x0 and y1 > y0:
            a[y0:y1, x0:x1] = mean
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    sha = sha256_bytes(data)
    name = "%s.%s.png" % (key or Path(image_path).stem, sha[:16])
    path = Path(out_dir) / name
    if not path.is_file() or file_record(path)["sha256"] != sha:
        _atomic_write_bytes(path, data)
    return path, sha


# ------------------------------------------------------------------- gates
def _stratum_part(stratum, key):
    for seg in str(stratum).split("/")[1:]:
        k, _, v = seg.partition("=")
        if k == key:
            return v
    return None


def _est_lb(e):
    if not isinstance(e, dict):
        return None, None
    if "lb" in e:
        return e.get("estimate"), e.get("lb")
    iv = e.get("interval")
    return e.get("estimate"), (iv[0] if isinstance(iv, (list, tuple)) and iv else None)


def _gate_numbers(entry):
    """(point, lb, ub, why) of an audit stratum, the most conservative over
    the stated estimate and both assignments of unsure answers."""
    iv = entry.get("interval") or [None, None]
    points = [entry.get("estimate")]
    lbs = [iv[0] if iv else None]
    for k in ("as_no", "as_yes"):
        p, lb = _est_lb((entry.get("unsure") or {}).get(k))
        points.append(p)
        lbs.append(lb)
    points = [p for p in points if p is not None]
    lbs = [x for x in lbs if x is not None]
    if not points or not lbs:
        return None, None, (iv[1] if iv else None), "no estimate or interval"
    return min(points), min(lbs), (iv[1] if len(iv) > 1 else None), None


def _hyp(audit, hid):
    return ((audit.get("hypotheses") or {}).get(hid) or {}).get("verdict")


def _recovery_thresholds(prereg):
    g = prereg.raw.get("recovery_gates") or {}
    box = g.get("box") or {}
    for k in ("precision_lb_min", "precision_point_min"):
        if k not in box:
            raise RecoverError("prereg recovery_gates.box.%s is missing" % k)
    if "class_purity_lb_min" not in g:
        raise RecoverError("prereg recovery_gates.class_purity_lb_min is missing")
    return {"lb": float(box["precision_lb_min"]), "point": float(box["precision_point_min"]),
            "class_lb": float(g["class_purity_lb_min"])}


def _policy_of_stratum(group, stratum, domain):
    pol = GROUP_POLICY.get(group)
    if pol is None:
        return None
    if group in ("G1", "G4"):
        return "R-J" if _stratum_part(stratum, "frame") == "noinfo" else None
    if group == "G2":
        src = _stratum_part(stratum, "source")
        return "R-A" if src in domain.authoritative_sources() else None
    if group == "G3":
        unit = _stratum_part(stratum, "unit") or ""
        src = unit[2:].split("|")[0] if unit[:2] in ("c:", "k:") else None
        return "R-C" if src in Q._card_resolved_sources(domain) else "R-T"
    return pol


# The audit event each gate is read from, in estimate.py's event vocabulary. A gate is never read from an
# estimate of another quantity (the audit's per-stratum "target" share says nothing about whether the
# source label, or the predicted class, is the right one).
BOX_EVENT = {"R-A": "label", "R-V": "label", "R-J": "pred"}   # label (or J1's prediction) right, box valid
CLASS_EVENT = "purity:%s"          # a class unit's class gate: the share answered as <class>
CLASS_BOX_EVENT = "label:%s"       # a card-mapped class unit's box gate: answered as <class>, box valid


def unit_classes(class_maps, name_status, domain):
    """{"c:<source>|<src_id>": class name} each class unit's gates are read
    for: a card-resolved source's proposed map (when its proposals agree on
    one class), any other source's scientific synonym. A unit with no class,
    or with two, is gated for none and fails its class gate."""
    card = set(Q._card_resolved_sources(domain))
    cand = collections.defaultdict(set)
    for p in (class_maps or {}).get("proposals") or []:
        if p.get("map_to") and p.get("source") in card:
            cand["c:%s|%s" % (p.get("source"), p.get("src_id"))].add(p["map_to"])
    for n in (name_status or {}).get("names") or []:
        if (n.get("status_v2") == "target_synonym" and n.get("via") in ("scientific", "override")
                and n.get("source") not in card):
            t = _target_of_taxon(n.get("taxon"), domain)
            if t is not None:
                cand["c:%s|%s" % (n["source"], n["src_id"])].add(domain.class_name(t))
    return {u: next(iter(v)) for u, v in sorted(cand.items()) if len(v) == 1}


def gates(audit, rl_qual, relation, leak, domain, prereg=None, thresholds=None, unit_class=None):
    """{stratum id: gate record} for every audit stratum a policy uses:
    {"policy", "level", "n_labelled", "precision": {"event", "estimate", "lb",
    "ub", "rogan_gladen"}, "box_gate", "class_gate", "class",
    "class_precision", "hypotheses", "passed", "why"}.

    Each gate is read from the audit entry of its own event (BOX_EVENT,
    CLASS_EVENT, CLASS_BOX_EVENT); a stratum whose audit holds no such entry
    fails that gate and says which estimate is missing. unit_class maps a
    class unit to the class its gates are for (unit_classes)."""
    if thresholds is None and prereg is None:
        raise RecoverError("gates needs the prereg (its recovery_gates) or the thresholds")
    th = thresholds or _recovery_thresholds(prereg)
    h1_pre = (relation or {}).get("h1_pre") or {}
    unit_class = unit_class or {}
    by, group_of = collections.defaultdict(dict), {}
    for e in audit.get("strata") or []:
        sid, ev = e.get("stratum"), e.get("event")
        if ev in by[sid]:
            raise RecoverError("the audit holds two %r estimates of stratum %s" % (ev, sid))
        by[sid][ev] = e
        group_of[sid] = e.get("group")
    out = {}
    for stratum in sorted(by):
        group = group_of[stratum]
        pol = _policy_of_stratum(group, stratum, domain)
        if pol is None:
            continue
        events = by[stratum]
        why = []
        cls = None
        if group == "G3":
            cls = unit_class.get(_stratum_part(stratum, "unit"))
            if cls is None:
                why.append("no card map or scientific synonym names one class for this unit")
            box_ev = (CLASS_BOX_EVENT % cls) if (cls and pol == "R-C") else None
            class_ev = (CLASS_EVENT % cls) if cls else None
        else:
            box_ev, class_ev = BOX_EVENT[pol], None

        def judged(ev, lb_min, point_min, what):
            e = events.get(ev)
            if e is None:
                why.append("%s: the audit holds no %r estimate for this stratum (it holds %s)"
                           % (what, ev, sorted(events)))
                return False, None
            point, lb, ub, w0 = _gate_numbers(e)
            rg = e.get("rogan_gladen") or {}
            bad = [w0] if w0 else []
            if not rg.get("applied"):
                bad.append("%s: %r is not Rogan-Gladen corrected" % (what, ev))
            if rg.get("flag"):
                bad.append("%s: Rogan-Gladen flag %s" % (what, rg["flag"]))
            ok = bool(not bad and lb >= lb_min and (point_min is None or point >= point_min))
            if not bad and not ok:
                bad.append("%s: %r lb %.3f / point %.3f below %.2f / %s"
                           % (what, ev, lb, point, lb_min, "-" if point_min is None else "%.2f" % point_min))
            why.extend(bad)
            return ok, {"event": ev, "estimate": point, "lb": lb, "ub": ub, "rogan_gladen": rg,
                        "level": e.get("level"), "n_labelled": e.get("n_labelled")}
        box_ok, box_rec = judged(box_ev, th["lb"], th["point"], "box gate") if box_ev else (None, None)
        class_gate, class_rec = (judged(class_ev, th["class_lb"], None, "class gate") if class_ev
                                 else ((False, None) if group == "G3" else (None, None)))
        used = box_rec or class_rec or {}
        hyps = {}
        ok_h = True
        if pol == "R-A":
            src = _stratum_part(stratum, "source")
            pre = (h1_pre.get(src) or {}).get("pass")
            hyps = {"H1-pre": "pass" if pre else "fail", "H1": _hyp(audit, "H1")}
            if not pre:
                why.append("H1-pre did not pass for %s" % src)
                ok_h = False
            if hyps["H1"] != "supported":
                why.append("H1 is %s" % hyps["H1"])
                ok_h = False
        elif pol == "R-C":
            hyps = {"H3a": _hyp(audit, "H3a")}
            if hyps["H3a"] != "supported":
                why.append("H3a is %s" % hyps["H3a"])
                ok_h = False
        elif pol == "R-J":
            hyps = {"H2a": _hyp(audit, "H2a"), "H7": _hyp(audit, "H7")}
            if hyps["H2a"] != "supported":
                why.append("H2a is %s" % hyps["H2a"])
                ok_h = False
        if pol == "R-C":
            passed = bool(class_gate) and bool(box_ok) and ok_h
        elif pol == "R-T":
            passed = bool(class_gate) and ok_h
        else:
            passed = bool(box_ok) and ok_h
        out[stratum] = {"policy": pol, "group": group, "level": used.get("level"),
                        "n_labelled": used.get("n_labelled"), "event": used.get("event"),
                        "precision": {k: used.get(k) for k in ("event", "estimate", "lb", "ub", "rogan_gladen")},
                        "box_gate": box_ok, "class_gate": class_gate, "class": cls,
                        "class_precision": class_rec, "hypotheses": hyps, "passed": passed,
                        "why": "; ".join(why) or "passed"}
    return out


def level_ok(gate, cls, domain):
    """The level a box gate was measured at must be species, or genus for a
    class the policy defines at genus rank."""
    lv = gate.get("level")
    if lv == "species":
        return True
    if lv == "genus" and domain.is_target(cls):
        return domain.target(int(cls)).get("rank") == "genus"
    return False


# --------------------------------------------------------------- the guard
def sibling_guarded(status, taxon, target, domain):
    """True when an other-class box may not become `target`: its class is
    named (the named or excluded frame of the name-status rule) or its taxon
    is a relative of the target (same genus, or in the target's `not`
    list) -- the sibling guard (contract §7, L12)."""
    frames = (domain.raw.get("names") or {}).get("frames") or {}
    named = set(frames.get("named") or []) | set(frames.get("excluded") or [])
    if status in named or is_relative(taxon, target, domain):  # funnel-mutation: M4
        return True
    return False


def is_relative(taxon, target, domain, resolver=None):
    if taxon in (None, ""):
        return False
    t = domain.target(target)
    if resolver is not None:
        res = resolver.resolve(taxon)
        if res is not None:
            return bool(resolver.is_relative(res, t))
    g = str(taxon).split()[0]
    tg = str(t.get("taxon") or "").split()[0]
    return bool(g and g == tg) or taxon in (t.get("not") or [])


# ----------------------------------------------------------------- inputs
def _iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


LEDGER_FIELDS = ("id", "source", "key", "box", "crop_id", "label", "src_id", "src_name", "name_status_v2",
                 "lab", "near_dup3", "provenance", "pred", "path", "kt")


def load_ledger(path, sources=None):
    """{image key: [box rows]} (box rows only, the fields recovery reads),
    each image's boxes in box order."""
    out = collections.defaultdict(list)
    for r in _iter_jsonl(path):
        if r.get("unit") != "box":
            continue
        if sources is not None and r.get("source") not in sources:
            continue
        out[r["key"]].append({k: r.get(k) for k in LEDGER_FIELDS})
    for k in out:
        out[k].sort(key=lambda r: int(r["box"]))
    return dict(out)


def frames_lookup(funnel_dir):
    """{unit id: [(group, stratum)]} from frames_v1.json and its csvs."""
    fj = Path(funnel_dir) / "frames_v1.json"
    frames = read_json(fj)
    out = collections.defaultdict(list)
    for g, info in sorted((frames.get("groups") or {}).items()):
        rec = info.get("file") or {}
        p = Path(rec.get("path") or (Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)))
        if not p.is_file():
            p = Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)
        if rec.get("sha256") and file_record(p)["sha256"] != rec["sha256"]:
            raise StaleInput("frame file %s does not hash to what frames_v1.json records" % p)
        _h, rows = read_csv(p)
        for r in rows:
            out[r["unit_id"]].append((g, r["stratum"]))
    return dict(out)


def _genus_unsure_units(funnel_dir, domain):
    """Units whose gold answer is a genus answer in a genus the class policy
    leaves unsure (masked in any recovered image)."""
    p = Path(funnel_dir) / "gold_v1.csv"
    if not p.is_file():
        return set()
    _h, rows = read_csv(p)
    unsure = set(domain.genus_unsure())
    out = set()
    for r in rows:
        if r.get("answer_level") == "genus" and str(r.get("answer_taxon") or "").split()[:1] and \
                str(r.get("answer_taxon")).split()[0] in unsure:
            out.add(r.get("unit_id"))
    return out


def leak_reading(funnel_dir, prereg):
    """(path, record) of the copy detector's record recovery reads
    (domain.leak_record). Once an amendment requires a later detector
    version, its record must exist and state that version: leak_v1.json's
    quarantine is never used in its place (contract §14 A2)."""
    from . import domain as DM
    path, version, why = DM.leak_record(funnel_dir, prereg)
    if path is None:
        raise RecoverError("no copy detector record to recover under: %s; nothing is recovered (contract §10 F4)"
                           % why)
    doc = read_json(path)
    got = int(doc.get("detector_version") or 1)
    if got != version:
        raise RecoverError("%s states copy detector version %d, the prereg requires %d; nothing is recovered"
                           % (path, got, version))
    return path, doc


def _quarantined(leak):
    q = set(((leak or {}).get("h6a") or {}).get("quarantine") or [])
    for name, sc in ((leak or {}).get("scans") or {}).items():
        if str(name).startswith("source:") and (sc or {}).get("copy_found"):
            q.add(str(name)[len("source:"):])
    return q


def _provenance_group(source, leak, domain):
    groups = ((leak or {}).get("h6c") or {}).get("groups") or {}
    for g, members in sorted(groups.items()):
        if source in members:
            return g
    return domain.lab_of(source)


def _accepted_maps(class_maps, gates_, h3a, domain):
    """[(source, src_id, map_to, accepted, why)] for every proposal."""
    out = []
    for p in (class_maps or {}).get("proposals") or []:
        why = []
        unit = "c:%s|%s" % (p.get("source"), p.get("src_id"))
        g = next((v for s, v in gates_.items() if _stratum_part(s, "unit") == unit), None)
        if p.get("status") != "proposed":
            why.append("status %s" % p.get("status"))
        if p.get("via") != "card+geometry":
            why.append("via %s (a map needs the card and the geometry match)" % p.get("via"))
        if not p.get("map_to"):
            why.append("maps to no class")
        if h3a != "supported":
            why.append("H3a is %s" % h3a)
        if g is None:
            why.append("no audit stratum for %s" % unit)
        elif g.get("class") != p.get("map_to"):
            why.append("the unit's gates are for %s, not %s" % (g.get("class"), p.get("map_to")))
        elif not g.get("class_gate"):
            why.append("class gate: %s" % g.get("why"))
        acc = not why
        out.append({"source": p.get("source"), "src_id": str(p.get("src_id")), "map_to": p.get("map_to"),
                    "accepted": acc, "why": "; ".join(why) or "card and geometry agree; class gate passed"})
    return out


# ------------------------------------------------------------------- plan
def _judge_label_space(funnel_dir, judge):
    """The label names of a judge's crop score file (its meta "labels")."""
    import numpy as np
    p = Path(funnel_dir) / "judges" / ("%s__crops.npz" % judge)
    if not p.is_file():
        raise RecoverError("judge %s has no crop score file %s" % (judge, p))
    with np.load(p, allow_pickle=False) as z:
        return list(json.loads(str(z["meta"])).get("labels") or [])


def fetched_licences(funnel_dir):
    """{source: licence text} from the cards L11a fetched and hashed
    (cards/index.json): the first entry of a source that answered 200 with a
    licence, recorded with its URL and date. The config's sources.licences
    wins over it (plan). {} without the index."""
    p = Path(funnel_dir) / "cards" / "index.json"
    if not p.is_file():
        return {}
    try:
        idx = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise RecoverError("cards/index.json is unreadable (%s)" % e)
    out = {}
    for src, ents in sorted((idx.get("cards") or {}).items()):
        for e in ents or []:
            if e.get("status") == 200 and e.get("licence"):
                out[src] = "%s (fetched by L11a from %s, %s)" % (e["licence"], e.get("url"), e.get("fetched_utc"))
                break
    return out


def mask_judges_of(label_spaces, domain):
    """The judges whose calls can mask a box (contract §7 R-J, "any judge calls a
    target"): those whose label space holds at least one non-target option. A
    judge that can only answer with a target (a kNN bank of target crops) names a
    target for every box, so its answer is not a call; it may still vote where
    it is qualified (runner decision F1-R2)."""
    return sorted(j for j, labels in label_spaces.items()
                  if any(lab not in domain.target_names for lab in labels))


def _judge_calls(funnel_dir, judge, crop_ids):
    """{crop_id: label name} of a judge's crop score file for the crops asked."""
    import numpy as np
    p = Path(funnel_dir) / "judges" / ("%s__crops.npz" % judge)
    if not p.is_file():
        raise RecoverError("judge %s has no crop score file %s" % (judge, p))
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        labels = list(meta.get("labels") or [])
        idx = z["unit_index"].astype(np.int64)
        top = z["top"].astype(np.int64)
    want = set(int(c) for c in crop_ids)
    out = {}
    for i, t in zip(idx.tolist(), top.tolist()):
        if i in want:
            out[i] = labels[t] if 0 <= t < len(labels) else None
    return out


def _stratum_of(unit, group, frames_of):
    for g, s in frames_of.get(unit, []):
        if g == group:
            return s
    return None


def _box_status(r):
    p = r.get("path") or {}
    return p.get("S6"), p.get("S8"), p.get("S9"), p.get("S10")


def plan(policies, ledger_rows, gates_, class_maps, name_status, resolver, judge_scores, leak, domain,
         frames_of=None, labels=None, pool_rows=None, base_keys=(), audit=None, rl_qual=None, jq=None,
         materials=None, genus_unsure_units=(), chain=None, mask_judges=None, fetched_licences=None):
    """([image plan], refusals, accepted class maps). ledger_rows maps an
    image key to its box rows (box order); labels maps a key to the step-1
    label boxes; judge_scores maps a judge to {crop_id: label name}."""
    frames_of = frames_of or {}
    pool_rows = pool_rows or {}
    labels = labels or {}
    base_keys = set(base_keys)
    other = domain.other["id"]
    frames_cfg = (domain.raw.get("names") or {}).get("frames") or {}
    noinfo = set(frames_cfg.get("noinfo") or [])
    quarantined = _quarantined(leak)
    not_rec = domain.not_recoverable_sources()
    licences = dict(fetched_licences or {})
    licences.update((domain.raw.get("sources") or {}).get("licences") or {})   # the config's record wins
    auth = set(domain.authoritative_sources())
    h3a = _hyp(audit or {}, "H3a")
    maps = _accepted_maps(class_maps, gates_, h3a, domain)
    map_to = {(m["source"], m["src_id"]): domain.class_id(m["map_to"]) for m in maps if m["accepted"]}
    identity = (rl_qual or {}).get("identity") or {}
    # classes whose identity check the config puts before R-A: recovered only when the check passed
    # (a missing record is a check not done, never a pass)
    id_before_auth = {c["class"] for c in domain.raw.get("identity_checks") or [] if "R-A" in (c.get("before") or [])}
    synonyms = {}
    for n in (name_status or {}).get("names") or []:
        if n.get("status_v2") == "target_synonym" and n.get("via") in ("scientific", "override"):
            t = _target_of_taxon(n.get("taxon"), domain)
            if t is not None:
                synonyms[(n["source"], str(n["src_id"]))] = t
    taxon_of = {(n["source"], str(n["src_id"])): n.get("taxon") for n in (name_status or {}).get("names") or []}
    refusals = []
    refused_sources = set()
    plans = []
    for key in sorted(ledger_rows):
        rows = ledger_rows[key]
        src = rows[0]["source"]
        if key in base_keys:
            continue
        if src in quarantined:  # funnel-mutation: M9
            if src not in refused_sources:
                refusals.append({"source": src, "why": "H6(a) links the source to an evaluation set; it is "
                                 "excluded from recovery as a whole"})
                refused_sources.add(src)
            continue
        if src in not_rec:
            continue
        pr = pool_rows.get(key)
        if pr is None:
            raise RecoverError("image %s is in the ledger but not in the pool" % key)
        lab_boxes = labels.get(key)
        if lab_boxes is None or len(lab_boxes) != len(rows) or any(
                int(b[0]) != int(r["label"]) for b, r in zip(lab_boxes, rows)):
            raise RecoverError("image %s: the step-1 label does not match the ledger's boxes" % key)
        p0 = rows[0].get("path") or {}
        if p0.get("S4", "pass") != "pass" or p0.get("S5", "pass") != "pass":
            raise RecoverError("image %s failed a guard stage and cannot be in the pool" % key)
        choice = None
        for pol in PRECEDENCE:
            if pol not in policies or pol == "R-F":
                continue
            res = _apply(pol, key, src, rows, lab_boxes, gates_, frames_of, domain, other, auth, noinfo,
                         map_to, synonyms, taxon_of, identity, judge_scores, jq, materials, genus_unsure_units,
                         resolver, id_before_auth=id_before_auth, mask_judges=mask_judges)
            if res is not None:
                choice = (pol, res)
                break
        if choice is None:
            continue
        if src not in licences:
            if src not in refused_sources:
                refusals.append({"source": src, "why": "no licence recorded for the source (DEC-7)"})
                refused_sources.add(src)
            continue
        pol, (kept, masked, strata) = choice
        plans.append({"key": key, "source": src, "pool": POOL_OF[pol], "policy": pol, "strata": sorted(set(strata)),
                      "boxes": kept, "masked": masked, "row": pr, "lab": rows[0].get("lab") or domain.lab_of(src),
                      "near_dup3": rows[0].get("near_dup3"), "licence": licences[src],
                      "provenance_group": _provenance_group(src, leak, domain)})
    if "R-F" in policies:
        plans.extend(_plan_refetch(chain, domain, leak, licences, refusals, maps=maps, gates_=gates_, h3a=h3a))
    return plans, refusals, maps


def _target_of_taxon(taxon, domain):
    if not taxon:
        return None
    for t in domain.targets:
        if t.get("taxon") == taxon:
            return t["id"]
        if t.get("rank") == "genus" and str(taxon).split()[0] == t.get("taxon"):
            return t["id"]
    return None


def _apply(pol, key, src, rows, lab_boxes, gates_, frames_of, domain, other, auth, noinfo, map_to, synonyms,
           taxon_of, identity, judge_scores, jq, materials, genus_unsure, resolver, id_before_auth=(),
           mask_judges=None):
    """(kept boxes, masked boxes, strata) of an image under one policy, or
    None when the policy does not apply or recovers nothing."""
    kept, masked, strata = [], [], []
    recovered = 0

    def mask_box(b):
        masked.append(list(b))

    for r, b in zip(rows, lab_boxes):
        s6, s8, s9, s10 = _box_status(r)
        lbl = int(r["label"])
        unit = r["id"]
        tgt = domain.is_target(lbl)
        unsure = unit in genus_unsure
        if pol == "R-V":
            if not any(_box_status(x)[1] == "verified" and _box_status(x)[3] != "admitted" for x in rows):
                return None
            if tgt and s8 == "verified" and not unsure:
                st = _stratum_of(unit, "G2v", frames_of)
                g = gates_.get(st) if st else None
                if g and g["passed"] and level_ok(g, lbl, domain):
                    kept.append(list(b))
                    strata.append(st)
                    recovered += 1
                else:
                    mask_box(b)
            elif not tgt and (s9 == "other_ok" or s6 == "small"):
                kept.append(list(b))
            else:
                mask_box(b)
        elif pol == "R-A":
            if src not in auth or s10 == "admitted":
                return None
            if tgt:
                st = _stratum_of(unit, "G2", frames_of)
                g = gates_.get(st) if st else None
                name = domain.class_name(lbl)
                id_ok = name not in id_before_auth or bool((identity.get(name) or {}).get("pass"))
                if (s6 != "small" and g and g["passed"] and level_ok(g, lbl, domain) and id_ok and not unsure):
                    kept.append(list(b))
                    strata.append(st)
                    recovered += 1
                else:
                    mask_box(b)
            else:
                called = r.get("pred") is not None and domain.is_target(int(r["pred"])) and s9 == "conflict"
                if called and not sibling_guarded(r.get("name_status_v2"), taxon_of.get((src, str(r.get("src_id")))),
                                                  int(r["pred"]), domain):
                    mask_box(b)
                else:
                    kept.append(list(b))
        elif pol in ("R-C", "R-T"):
            cls_key = (src, str(r.get("src_id")))
            new = map_to.get(cls_key) if pol == "R-C" else synonyms.get(cls_key)
            st = _stratum_of(unit, "G3", frames_of) or ("G3/unit=c:%s|%s" % cls_key)
            g = gates_.get(st)
            if new is not None and not (g and g.get("class") == domain.class_name(new)):
                new = None                                # the unit's gates were read for another class
            if pol == "R-T" and new is not None and not g.get("class_gate"):
                new = None
            if new is not None and not tgt:
                if s6 == "small" or unsure or g is None or not g["passed"] or not level_ok(g, new, domain):
                    mask_box(b)
                else:
                    kept.append([new] + list(b[1:]))
                    strata.append(st)
                    recovered += 1
            elif tgt:
                if s8 == "verified":
                    kept.append(list(b))
                else:
                    mask_box(b)
            else:
                kept.append(list(b))
        elif pol == "R-J":
            if tgt:
                if s8 == "verified":
                    kept.append(list(b))
                else:
                    mask_box(b)
                continue
            if s6 == "small" or r.get("crop_id") in (None, ""):
                kept.append(list(b))
                continue
            status = r.get("name_status_v2")
            frame_group = "G1" if s9 == "conflict" else "G4"
            st = _stratum_of(unit, frame_group, frames_of)
            box_keys = {k: r.get(k) for k in Q.KEYS}
            cid = int(r["crop_id"])
            judges_ok = Q.qualified_judges("other_noinfo", box_keys, jq, materials) if jq else []
            if frame_group == "G4" and judges_ok:
                # an other_ok box lies in an H4 stratum: its voters must also be qualified against the
                # independent set's attractors (contract §4.3)
                named_ok = {j for j, _s in Q.qualified_judges("other_named", box_keys, jq, materials)}
                judges_ok = [(j, s) for j, s in judges_ok if j in named_ok]
            calls = [(j, (judge_scores.get(j) or {}).get(cid)) for j, _s in judges_ok]
            named_t = [c for _j, c in calls if c in domain.target_names]
            # masking listens to every judge with crop scores that can answer a non-target, qualified or
            # not (contract §7 R-J; mask_judges_of)
            maskers = judge_scores if mask_judges is None else [j for j in judge_scores if j in mask_judges]
            any_target = any((judge_scores.get(j) or {}).get(cid) in domain.target_names for j in maskers) \
                or (r.get("pred") is not None and domain.is_target(int(r["pred"])) and s9 == "conflict")
            unanimous = (calls and len(named_t) == len(calls) and len(set(named_t)) == 1)
            if unanimous:
                t = domain.class_id(named_t[0])
                g = gates_.get(st) if st else None
                guarded = sibling_guarded(status, taxon_of.get((src, str(r.get("src_id")))), t, domain)
                # the box gate measures the step-1 probe's predicted class ("pred"): it covers a relabel
                # to that class only
                as_pred = r.get("pred") not in (None, "") and int(r["pred"]) == t
                if (as_pred and not guarded and status in noinfo and g and g["passed"] and level_ok(g, t, domain)
                        and not unsure):
                    kept.append([t] + list(b[1:]))
                    strata.append(st)
                    recovered += 1
                    continue
            if any_target:
                mask_box(b)
            else:
                kept.append(list(b))
    if recovered == 0:
        return None
    return kept, masked, strata


def _plan_refetch(chain, domain, leak, licences, refusals, maps=(), gates_=None, h3a=None):
    """R-F: refetched images, only after the Step 1 guard chain, then "as
    R-C" (contract §7): H3a supported, and every target box's class the class
    of an accepted card map of the source whose unit passed its gates."""
    if chain is None:
        raise RecoverError("R-F needs the Step 1 guard-chain record %s (never-train, reference copy, exact_dup, "
                           "H6, embedding) before any refetched image is recovered" % CHAIN_FILE)
    out = []
    gates_ = gates_ or {}
    passed_classes = collections.defaultdict(set)
    for m in maps or []:
        g = next((v for s, v in gates_.items() if _stratum_part(s, "unit") == "c:%s|%s" % (m["source"], m["src_id"])),
                 None)
        if m["accepted"] and g is not None and g.get("passed"):
            passed_classes[m["source"]].add(domain.class_id(m["map_to"]))
    quarantined = _quarantined(leak)
    for r in chain.get("images") or []:
        bad = [c for c in CHAIN_CHECKS if (r.get("checks") or {}).get(c) is not True]
        if bad:
            refusals.append({"key": r.get("key"), "why": "refetched image failed %s" % bad})
            continue
        src = r["source"]
        if src in quarantined or src in domain.not_recoverable_sources():
            refusals.append({"key": r.get("key"), "why": "source %s is not recoverable" % src})
            continue
        if h3a != "supported":
            refusals.append({"key": r.get("key"), "why": "R-F is recovered as R-C, and H3a is %s" % h3a})
            continue
        classes = {int(b[0]) for b in r.get("label_boxes") or [] if domain.is_target(int(b[0]))}
        unmapped = sorted(classes - passed_classes.get(src, set()))
        if not classes or unmapped:
            refusals.append({"key": r.get("key"), "why": "target classes %s of the refetched image have no accepted "
                             "card map of %s whose gates passed" % (unmapped or "none", src)})
            continue
        if src not in licences:
            refusals.append({"key": r.get("key"), "why": "no licence for %s" % src})
            continue
        if file_record(r["image"])["sha256"] != r["sha256"]:
            raise StaleInput("refetched image %s changed after the chain record" % r["image"])
        row = {"image": r["image"], "sha256": r["sha256"], "label": None, "label_sha256": None,
               "session": "", "key": r["key"], "source": src}
        out.append({"key": r["key"], "source": src, "pool": "FETCH", "policy": "R-F",
                    "strata": list(r.get("strata") or []), "boxes": [list(b) for b in r["label_boxes"]],
                    "masked": [list(b) for b in r.get("masked_boxes") or []], "row": row,
                    "lab": domain.lab_of(src), "near_dup3": r.get("near_dup3"), "licence": licences[src],
                    "provenance_group": _provenance_group(src, leak, domain), "dhash": r.get("dhash")})
    return out


# ----------------------------------------------------------------- writing
def write_overlay(plans, out_dir, adapter=None, dhash_of=None):
    """labels_overlay/ and images_masked/ for every plan; returns the
    recovered_pool rows (sorted by key)."""
    out_dir = Path(out_dir)
    rows = []
    for p in plans:
        pr = p["row"]
        text = V._yolo_text([tuple(b) for b in p["boxes"]])
        lpath, lsha = V._label_file(out_dir / "labels_overlay", p["source"], p["key"], text)
        V._write_label(lpath, text)
        if C.sha256_file(lpath) != lsha:
            raise RecoverError("overlay label %s does not hold what its name says" % lpath)
        image, isha, dh_m = pr["image"], pr["sha256"], None
        if p["masked"]:
            mpath, isha = mask(pr["image"], p["masked"], out_dir / "images_masked" / V._sanitise(p["source"]),
                               key=p["key"])
            image = str(mpath)
            dh_m = C.dhash(mpath)
        dh_u = p.get("dhash")
        if dh_u is None and dhash_of is not None:
            dh_u = dhash_of.get(p["key"])
        rows.append({"image": image, "label": str(lpath), "sha256": isha, "label_sha256": lsha,
                     "source": p["source"], "session": pr.get("session", ""), "key": p["key"],
                     "pool": p["pool"], "policy": p["policy"], "strata": p["strata"],
                     "unmasked_image": pr["image"], "unmasked_sha256": pr["sha256"],
                     "masked_boxes": p["masked"], "ctl_label": pr.get("label"),
                     "ctl_label_sha256": pr.get("label_sha256"), "provenance_group": p["provenance_group"],
                     "lab": p["lab"], "licence": p["licence"], "near_dup3": p["near_dup3"],
                     "dhash_unmasked": dh_u, "dhash_masked": dh_m})
    rows.sort(key=lambda r: r["key"])
    return rows


def _default_detector(adapter, funnel_dir, leak_doc, domain):
    """leak.detect against the evaluation index, with the calibration of the
    copy detector's record (leak_record) and the evaluation descriptors it
    records (the file never leaves the cluster)."""
    from . import leak as L
    from . import embed as EM
    if domain is None:
        raise RecoverError("the H6 detector needs the domain config (its feature extractor)")
    embedder = EM.default_embedder(domain)
    rec = (leak_doc or {}).get("eval_descriptors") or {}
    name = Path(rec["path"]).name if rec.get("path") else L.EVAL_DESC
    idx = L.eval_index(adapter, embedder, Path(funnel_dir) / name)

    def detect(images):
        return L.detect(images, idx, leak_doc)
    return detect


def guard(rows, adapter, leak_calibration, detector=None, funnel_dir=None, domain=None):
    """The never-train index on every unmasked image (its pool dHash) and
    every masked copy (the file's dHash), and the H6 detector on both.
    Returns the guard record; raises NeverTrainHit on any hit."""
    G = adapter.never_train_guard()
    if detector is None:
        detector = _default_detector(adapter, funnel_dir or FUNNEL_DIR, leak_calibration, domain)
    nt_hits, unhashable = [], []
    n_u = n_m = 0
    for r in rows:
        n_u += 1
        h = r.get("dhash_unmasked")
        hits_u, unh_u = G.check([r["unmasked_image"]], hash_fn=lambda _p, h=h: h)
        if hits_u or unh_u:  # funnel-mutation: M8
            nt_hits.extend({"key": r["key"], "which": "unmasked", "hit": list(x)} for x in hits_u)
            unhashable.extend({"key": r["key"], "which": "unmasked"} for _ in unh_u)
        if r["image"] != r["unmasked_image"]:
            n_m += 1
            hits_m, unh_m = G.check([r["image"]])
            nt_hits.extend({"key": r["key"], "which": "masked", "hit": list(x)} for x in hits_m)
            unhashable.extend({"key": r["key"], "which": "masked"} for _ in unh_m)
    imgs_u = [{"key": r["key"], "path": r["unmasked_image"]} for r in rows]
    imgs_m = [{"key": r["key"], "path": r["image"]} for r in rows if r["image"] != r["unmasked_image"]]
    copies = list(detector(imgs_u) or []) + list(detector(imgs_m) or []) if rows else []
    rec = {"never_train": {"unmasked_checked": n_u, "masked_checked": n_m, "hits": len(nt_hits),
                           "unhashable": len(unhashable)},
           "h6": {"unmasked_checked": len(imgs_u), "masked_checked": len(imgs_m), "copies": len(copies)}}
    if nt_hits or unhashable or copies:
        rec["listed"] = {"never_train": nt_hits[:50], "unhashable": unhashable[:50],
                         "copies": [dict(c) for c in copies[:50]]}
    return rec


def domain_dev(sources, ledger_rows, domain, seed_prefix="funnel/v1/h10d"):
    """The H10d hold-out: per source, a block of whole groups (capture-stem
    groups when the config declares a stem regex for the source, else 3-bit
    near-duplicate groups) of at least min_group_images images, starting at
    a seeded group. ledger_rows: [{"key", "source", "near_dup3", "image",
    "base": bool}]; a group holding a base image is never held out (the base
    would train on the domain dev's near-duplicates), and base images are
    never domain-dev rows."""
    cfg = ((domain.raw.get("recovery") or {}).get("domain_dev") or {})
    if "min_group_images" not in cfg:
        raise RecoverError("recovery.domain_dev.min_group_images is not set in the domain config")
    need = int(cfg["min_group_images"])
    stems = (domain.raw.get("sources") or {}).get("capture_stem_regex") or {}
    by_src = collections.defaultdict(list)
    for r in ledger_rows:
        if r["source"] in sources:
            by_src[r["source"]].append(r)
    out = {"min_group_images": need, "sources": {}, "keys": [], "key_group": {}}
    for src in sorted(sources):
        imgs = sorted(by_src.get(src, []), key=lambda r: r["key"])
        seed_text = "%s/%s" % (seed_prefix, src)
        rx = re.compile(stems[src]) if src in stems else None
        groups = collections.OrderedDict()
        with_base = set()
        for r in imgs:
            if rx is not None:
                m = rx.search(Path(r.get("image") or r["key"]).stem)
                gid = "stem:%s" % ((m.group(1) if m and m.groups() else m.group(0)) if m else r["key"])
            else:
                gid = r.get("near_dup3") or ("n:%s" % r["key"])
            if r.get("base"):
                with_base.add(gid)
                continue
            groups.setdefault(gid, []).append(r["key"])
        gids = sorted((g for g in groups if g not in with_base), key=lambda g: groups[g][0])
        imgs = [r for r in imgs if not r.get("base")]
        if len(imgs) < 2 * need or not gids:
            out["sources"][src] = {"groups": [], "images": 0, "seed_text": seed_text,
                                   "why": "%d image(s): fewer than twice min_group_images" % len(imgs)}
            continue
        start = int(rng(seed_text).integers(0, len(gids)))
        chosen, keys = [], []
        for i in range(len(gids)):
            g = gids[(start + i) % len(gids)]
            chosen.append(g)
            keys.extend(groups[g])
            if len(keys) >= need:
                break
        out["sources"][src] = {"groups": chosen, "images": len(keys), "seed_text": seed_text,
                               "grouping": "capture_stem" if rx is not None else "near_dup3"}
        out["keys"].extend(keys)
        for g in chosen:
            for k in groups[g]:
                out["key_group"][k] = "%s|%s" % (src, g)
    out["keys"] = sorted(out["keys"])
    return out


# --------------------------------------------------------------------- run
def _read_opt(path):
    p = Path(path)
    return read_json(p) if p.is_file() else None


def run(prereg, domain, funnel_dir=None, out_dir=None, policies=("R-A", "R-C", "R-T", "R-V", "R-J"),
        adapter=None, audit_path=None, maps_path=None, detector=None, resolver=None, testing=False,
        force=False):
    """recovery.json and the overlay under out_dir (INC_DIR/step1_r1)."""
    from . import adapters as A
    prereg, domain = Q._load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    out_dir = Path(out_dir or R1_DIR)
    policies = tuple(policies)
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        raise RecoverError("unknown policies %s (known: %s)" % (bad, POLICIES))
    if adapter is None:
        adapter = A.load(domain.adapter)
    audit_path = Path(audit_path or funnel_dir / "audit_v1.json")
    audit = read_json(audit_path)
    if audit.get("format") != "funnel-audit/1":
        raise RecoverError("%s is not a funnel-audit/1 file" % audit_path)
    if audit.get("stop"):
        raise RecoverError("the audit stopped the campaign (%s); nothing is recovered" % audit["stop"])
    if not audit.get("valid"):
        raise RecoverError("the audit is invalid (calibration overlap %s); nothing may cite it"
                           % audit.get("calibration_overlap"))
    check_records(audit.get("inputs"))
    th = _recovery_thresholds(prereg)
    rl_qual = read_json(funnel_dir / "rl_qualification.json")
    relation = _read_opt(funnel_dir / "relation_geometry_v1.json")
    maps_path = Path(maps_path or funnel_dir / "class_maps.json")
    class_maps = _read_opt(maps_path)
    leak_path, leak = leak_reading(funnel_dir, prereg)
    if ((leak.get("calibration") or {}).get("ok")) is not True:
        raise RecoverError("the copy detector in %s did not meet its calibration: H6(a) quarantines nothing that "
                           "can be trusted, so nothing is recovered (contract §10 F4)" % leak_path)
    h6b = leak.get("h6b")
    if not isinstance(h6b, dict) or h6b.get("incident") is not False or h6b.get("base_copy"):
        raise RecoverError("%s: H6(b) %s; a copy of an evaluation image in base B or an increment is an R4 incident "
                           "resolved before F9 (contract §10 stop rules)"
                           % (leak_path, "is not recorded" if not isinstance(h6b, dict) else "found a copy"))
    if "R-A" in policies and relation is None:
        raise RecoverError("R-A needs relation_geometry_v1.json (H1-pre)")
    if "R-C" in policies and class_maps is None:
        raise RecoverError("R-C needs class_maps.json")
    ns_path = funnel_dir / "name_status_v2.json"
    name_status = read_json(ns_path)
    jq = materials = None
    if "R-J" in policies:
        jq = read_json(funnel_dir / Q.JUDGE_FILE)
        materials = Q.judge_materials(funnel_dir)
    chain = _read_opt(funnel_dir / CHAIN_FILE) if "R-F" in policies else None
    inputs = {"audit": file_record(audit_path), "rl_qualification": file_record(funnel_dir / "rl_qualification.json"),
              "leak": file_record(leak_path), "name_status_v2": file_record(ns_path),
              "frames": file_record(funnel_dir / "frames_v1.json"),
              "ledger": file_record(funnel_dir / "ledger.jsonl")}
    if relation is not None:
        inputs["relation_geometry"] = file_record(funnel_dir / "relation_geometry_v1.json")
    if class_maps is not None:
        inputs["class_maps"] = file_record(maps_path)
    if jq is not None:
        inputs["judge_qualification"] = file_record(funnel_dir / Q.JUDGE_FILE)
    if (funnel_dir / "gold_v1.csv").is_file():
        inputs["gold"] = file_record(funnel_dir / "gold_v1.csv")
    if chain is not None:
        inputs["refetch_chain"] = file_record(funnel_dir / CHAIN_FILE)
    gates_ = gates(audit, rl_qual, relation, leak, domain, thresholds=th,
                   unit_class=unit_classes(class_maps, name_status, domain))
    frames_of = frames_lookup(funnel_dir)
    ledger = load_ledger(funnel_dir / "ledger.jsonl")
    pool = {r["key"]: r for r in adapter.pool_rows()}
    base_keys = {r["key"] for r in adapter.base_rows()}
    labels = adapter.label_rows(sorted(ledger))
    label_sha_before = {k: pool[k]["label_sha256"] for k in ledger if k in pool}
    judge_scores = {}
    if "R-J" in policies and jq:
        # every judge with crop scores: the qualified ones vote, and any of them calling a target masks a box
        crop_ids = [int(r["crop_id"]) for rows in ledger.values() for r in rows if r.get("crop_id") not in (None, "")]
        for j in sorted(jq.get("judges", {})):
            if jq["judges"][j].get("kind") in ("step1_probe", "rl"):
                continue
            judge_scores[j] = _judge_calls(funnel_dir, j, crop_ids)
    mask_judges = mask_judges_of({j: _judge_label_space(funnel_dir, j) for j in judge_scores}, domain)
    plans, refusals, maps = plan(policies, ledger, gates_, class_maps, name_status, resolver, judge_scores, leak,
                                 domain, frames_of=frames_of, labels=labels, pool_rows=pool, base_keys=base_keys,
                                 audit=audit, rl_qual=rl_qual, jq=jq, materials=materials,
                                 genus_unsure_units=_genus_unsure_units(funnel_dir, domain), chain=chain,
                                 mask_judges=mask_judges, fetched_licences=fetched_licences(funnel_dir))
    rec_sources = sorted({p["source"] for p in plans})
    img_rows = [{"key": k, "source": rows[0]["source"], "near_dup3": rows[0].get("near_dup3"),
                 "image": pool[k]["image"], "base": k in base_keys} for k, rows in ledger.items() if k in pool]
    dd = domain_dev(rec_sources, img_rows, domain)
    held = set(dd["keys"])
    plans = [p for p in plans if p["key"] not in held]
    try:
        dh = S.read_pool_dhash(V.POOL_META, [p["key"] for p in plans if p["policy"] != "R-F"]) if plans else {}
    except (S.SelectError, OSError) as e:
        raise RecoverError("pool dHashes: %s" % e)
    rows = write_overlay(plans, out_dir, adapter, dhash_of=dh)
    doc_common = {"refusals": refusals, "gates": gates_,
                  "class_maps": maps, "identity_checks": rl_qual.get("identity") or {},
                  "quarantined_sources": sorted(_quarantined(leak)), "policies": list(policies),
                  "mask_judges": mask_judges}
    grec = guard(rows, adapter, leak, detector=detector, funnel_dir=funnel_dir, domain=domain)
    rj = out_dir / "recovery.json"
    old = _read_opt(rj)
    # a complete record is replaced only with --force; nothing below is written before that is settled
    keep_old = old is not None and old.get("status") == "complete" and not force
    if grec["never_train"]["hits"] or grec["never_train"]["unhashable"] or grec["h6"]["copies"]:
        msg = ("recovered images hit the never-train index (%d, %d unhashable) or the H6 detector (%d): F9 stops"
               % (grec["never_train"]["hits"], grec["never_train"]["unhashable"], grec["h6"]["copies"]))
        if keep_old:
            raise NeverTrainHit(msg + "; the complete %s already there is kept (rerun with --force to record the "
                                "refusal)" % rj)
        doc = dict(header("recovery", domain, prereg, inputs, modules=(sys.modules[__name__],), testing=testing),
                   status="refused", guards=grec, **doc_common)
        write_json_atomic(rj, doc)
        raise NeverTrainHit(msg)
    changed = []
    for k, want in sorted(label_sha_before.items()):
        p = pool[k]["label"]
        if not Path(p).is_file() or C.sha256_file(p) != want:
            changed.append(k)
    if changed:
        raise RecoverError("%d step-1 label(s) changed during recovery, e.g. %s" % (len(changed), changed[:3]))
    rp_bytes = _jsonl_bytes(rows)
    rp_sha = sha256_bytes(rp_bytes)
    dd_rows = [{**{k: pool[key][k] for k in C.MANIFEST_KEYS}, "group": dd["key_group"][key]}
               for key in dd["keys"]]
    dd_bytes = _jsonl_bytes(dd_rows)
    dd_sha = sha256_bytes(dd_bytes)
    dd_doc = dict(header("domain_dev", domain, prereg, {"ledger": inputs["ledger"]},
                         seeds={s: v["seed_text"] for s, v in dd["sources"].items()},
                         modules=(sys.modules[__name__],), testing=testing),
                  min_group_images=dd["min_group_images"], sources=dd["sources"],
                  rows={"path": str(out_dir / "domain_dev.jsonl"), "sha256": dd_sha, "images": len(dd_rows)})
    dd_doc_bytes = json_text(dd_doc).encode("utf-8")
    counts = {}
    for r in rows:
        c = counts.setdefault(r["pool"], {"images": 0, "boxes_by_class": collections.Counter(), "masked_boxes": 0})
        c["images"] += 1
        c["masked_boxes"] += len(r["masked_boxes"])
        for b in C.read_yolo(r["label"]):
            c["boxes_by_class"][domain.class_name(b[0])] += 1
    for c in counts.values():
        c["boxes_by_class"] = dict(sorted(c["boxes_by_class"].items()))
    prov = collections.defaultdict(set)
    for r in rows:
        prov[r["provenance_group"]].add(r["source"])
    doc = dict(header("recovery", domain, prereg, inputs, modules=(sys.modules[__name__],), testing=testing),
               status="complete", guards=grec, counts=counts,
               licences={r["source"]: r["licence"] for r in rows},
               provenance_groups={g: sorted(v) for g, v in sorted(prov.items())},
               source_labels_unchanged={"checked": len(label_sha_before), "changed": 0},
               recovered_pool={"path": str(out_dir / "recovered_pool.jsonl"), "sha256": rp_sha, "images": len(rows)},
               domain_dev={"path": str(out_dir / "domain_dev.json"), "sha256": sha256_bytes(dd_doc_bytes),
                           "rows_sha256": dd_sha},
               **doc_common)
    if keep_old:
        if _identity(old) == _identity(doc):
            return old
        raise RecoverError("%s exists and was made from other inputs; rerun with --force (nothing was written)" % rj)
    _atomic_write_bytes(out_dir / "recovered_pool.jsonl", rp_bytes)
    _atomic_write_bytes(out_dir / "domain_dev.jsonl", dd_bytes)
    _atomic_write_bytes(out_dir / "domain_dev.json", dd_doc_bytes)
    write_json_atomic(rj, doc)
    log("recover: %d image(s) %s; refusals %d; guard %s"
        % (len(rows), {k: v["images"] for k, v in counts.items()}, len(refusals), grec))
    return doc


def _jsonl_bytes(rows):
    """The bytes write_jsonl_atomic writes for rows."""
    return "".join(canonical_json(r) + "\n" for r in rows).encode("utf-8")


def _identity(doc):
    """A recovery.json without its volatile keys and without the sha256 of
    domain_dev.json (whose header carries a build time); the domain-dev rows'
    sha256 stays in."""
    d = strip_volatile(doc)
    if isinstance(d.get("domain_dev"), dict):
        d["domain_dev"] = {k: v for k, v in d["domain_dev"].items() if k != "sha256"}
    return canonical_json(d)


# -------------------------------------------------------------------- arms
def _select_u(rows, cap, seed_text):
    """A seeded draw of `cap` rows stratified by pool (largest remainder,
    ties by pool name); every row when there are no more than cap."""
    if len(rows) <= cap:
        return sorted(rows, key=lambda r: r["key"])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["pool"]].append(r)
    total = len(rows)
    quota = {p: cap * len(v) / float(total) for p, v in by.items()}
    alloc = {p: int(math.floor(q)) for p, q in quota.items()}
    rest = cap - sum(alloc.values())
    for p in sorted(by, key=lambda p: (-(quota[p] - alloc[p]), p))[:rest]:
        alloc[p] += 1
    out = []
    for p in sorted(by):
        v = sorted(by[p], key=lambda r: r["key"])
        perm = rng("%s/%s" % (seed_text, p)).permutation(len(v))
        out.extend(v[int(i)] for i in perm[:alloc[p]])
    return sorted(out, key=lambda r: r["key"])


def _ctl(r):
    if not r.get("ctl_label") or not r.get("ctl_label_sha256"):
        raise RecoverError("recovered row %s (pool %s) has no join label: it cannot enter a control arm"
                           % (r.get("key"), r.get("pool")))
    return {"image": r["unmasked_image"], "label": r["ctl_label"], "sha256": r["unmasked_sha256"],
            "label_sha256": r["ctl_label_sha256"], "source": r["source"], "session": r.get("session", ""),
            "key": r["key"]}


def _man(r):
    return {k: r[k] for k in C.MANIFEST_KEYS}


def arms(realloop_exp_dir, out_dir, base_manifest, adapter=None, prereg=None, domain=None, testing=False):
    """arms/U.jsonl, U_ctl.jsonl, CLASS_ctl.jsonl, JUDGE_ctl.jsonl and
    arms/arms.json (contract §9.2)."""
    out_dir = Path(out_dir or R1_DIR)
    exp_dir = Path(realloop_exp_dir)
    rec_path = out_dir / "recovery.json"
    rec = read_json(rec_path)
    if prereg is None:
        prereg = (rec.get("prereg") or {}).get("path")
        if not prereg:
            raise RecoverError("recovery.json names no prereg; pass it")
    prereg, domain = Q._load(prereg, domain)
    if (rec.get("prereg") or {}).get("core_sha256") not in (None, prereg.core_sha256):
        raise StaleInput("recovery.json was made under another prereg core")
    if rec.get("status") != "complete":
        raise RecoverError("recovery.json is %r, not complete" % rec.get("status"))
    rp = out_dir / "recovered_pool.jsonl"
    if file_record(rp)["sha256"] != (rec.get("recovered_pool") or {}).get("sha256"):
        raise StaleInput("%s does not hash to what recovery.json records" % rp)
    exp_path = exp_dir / "exp.json"
    exp = read_json(exp_path)
    ov = ((exp.get("step1") or {}).get("increment_sources") or {})
    ov = ov.get("overlay") if isinstance(ov, dict) else None
    if not ov or not ov.get("recovery_sha256"):
        raise RecoverError("%s is not a recovered-mode experiment (it records no step1 overlay)" % exp_path)
    if ov["recovery_sha256"] != file_record(rec_path)["sha256"]:
        raise StaleInput("%s was built from another recovery.json" % exp_path)
    base = C.read_manifest(base_manifest)
    recovered = C.read_manifest(rp)
    by_key = {r["key"]: r for r in recovered}
    for r in recovered:
        for k in ("image", "label"):
            if not Path(r[k]).is_file():
                raise StaleInput("recovered row %s: %s %s is missing" % (r["key"], k, r[k]))
        if C.sha256_file(r["label"]) != r["label_sha256"]:
            raise StaleInput("recovered row %s: the overlay label changed" % r["key"])
    base_keys = {r["key"] for r in base}
    clash = sorted(base_keys & set(by_key))
    if clash:
        raise RecoverError("recovered rows share keys with base B: %s" % clash[:3])
    seed_text = "funnel/v1/arms/U"
    u_rows = _select_u(recovered, len(base), seed_text)
    arms_rows = {"U": [_man(r) for r in u_rows], "U_ctl": [_ctl(r) for r in u_rows]}
    missing = {}
    for arm, step_name in ARM_STEPS.items():
        step = next((s for s in exp.get("steps") or [] if s.get("name") == step_name), None)
        if step is None:
            missing[arm] = "the experiment has no step %s" % step_name
            continue
        mp = Path(step["manifest"])
        if not mp.is_file():
            mp = exp_dir / "manifests" / Path(step["manifest"]).name
        if not mp.is_file() or not step.get("manifest_sha256") or C.sha256_file(mp) != step["manifest_sha256"]:
            raise StaleInput("step %s manifest %s is missing, unrecorded or changed" % (step_name, mp))
        keys = [r["key"] for r in C.read_manifest(mp)]
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise RecoverError("step %s holds images that are not recovered rows: %s" % (step_name, unknown[:3]))
        arms_rows[arm] = [_ctl(by_key[k]) for k in keys]
    arms_dir = out_dir / "arms"
    out = {}
    for arm, rows in sorted(arms_rows.items()):
        path = arms_dir / ("%s.jsonl" % arm)
        sha = C.write_manifest(path, list(base) + rows)
        by_pool = collections.Counter(by_key[r["key"]]["pool"] for r in rows)
        out[arm] = {"path": str(path), "sha256": sha, "images": len(base) + len(rows), "from_base": len(base),
                    "from_recovered": len(rows), "by_pool": dict(sorted(by_pool.items())),
                    "seed_text": seed_text if arm in ("U", "U_ctl") else None}
    doc = dict(header("arms", domain, prereg, {"recovery": rec_path, "recovered_pool": rp, "base": base_manifest,
                                               "exp": exp_path},
                      seeds={"U": seed_text}, modules=(sys.modules[__name__],), testing=testing),
               realloop={"exp": exp.get("exp") or exp_dir.name, "exp_sha256": file_record(exp_path)["sha256"]},
               arms=out, not_built=missing, cap=len(base))
    write_json_atomic(arms_dir / "arms.json", doc)
    log("arms: %s; not built %s" % ({a: v["images"] for a, v in out.items()}, missing))
    return doc
