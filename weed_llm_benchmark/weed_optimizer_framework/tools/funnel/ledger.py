"""The funnel ledger (contract §8.2, runner §4.3) and the per-unit ledger rows
(runner §4.2).

funnel-ledger/1 lists every filter stage with what it took in, kept and
discarded, what it depends on, whether it is a guard, whether its discards are
recoverable, the known truth it was calibrated on and, once an audit exists,
its measured false-negative rate. The generic diagnoses (D17-D19) read only
this file, the claims and the loop reports, so they run on any domain: the
runner's extensions (derivation, fingerprint, name_status_version, role,
guard, kept_by_label, discarded_by_label, reject_class) exist for that.

ledger.jsonl holds one row per unit (a box of a pool image, or an image
dropped before the pool) with its outcome at every stage and its discard path.
Its stage vocabulary belongs to the adapter that writes it; validate_unit_row
checks the structure and, when the adapter passes one, the vocabulary.

Standard library only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from . import (LedgerError, _atomic_write_bytes, canonical_json, file_record, json_text,
               read_json)

FORMAT = "funnel-ledger/1"
ROLES = ("discovery", "cap", "registry", "read", "dedup", "guard", "size", "join", "name_status",
         "reject_class_sample", "target_check", "other_check", "image_rule", "selection",
         "evidence", "relevance", "gate")
UNITS = ("box", "image", "source", "class", "dataset", "increment")
DERIVATIONS = ("summaries", "census")
NAME_STATUS_VERSIONS = ("v1", "v2")
KINDS = ("named", "numeric", "none", "generic")
STAGE_KEYS = ("id", "filter", "version", "unit", "role", "depends_on", "recoverable", "guard",
              "in", "kept", "discarded", "kept_by_label", "discarded_by_label", "by_source",
              "by_label_pred", "calibration", "audit")
STAGE_REQUIRED = ("id", "filter", "version", "unit", "role", "depends_on", "recoverable", "guard")
TOP_KEYS = ("format", "domain", "inputs", "derivation", "fingerprint", "name_status_version",
            "target_classes", "other_class", "stages", "label_spaces", "domain_scores",
            "reject_class")
UNIT_ROW_KEYS = {
    "box": ("id", "unit", "source", "key", "box", "crop_id", "label", "src_id", "src_name", "wh_px",
            "name_status_v1", "name_status_v2", "lab", "near_dup3", "provenance", "pred", "p", "cos",
            "p_target_max", "p_other", "second", "p_second", "fail", "blockers", "path",
            "failed_stages", "first_cause", "sole_cause", "kt"),
    "image": ("id", "unit", "source", "rel", "dhash", "path", "near", "twin_of", "failed_stages",
              "first_cause", "sole_cause"),
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def fingerprint(identity_inputs):
    """sha256 of the canonical JSON of {name: sha256} of the adapter's identity
    inputs; the same in both derivations, recorded by the audit (D17 (A))."""
    if not isinstance(identity_inputs, dict) or not identity_inputs:
        raise LedgerError("fingerprint needs a non-empty {name: sha256}")
    norm = {}
    for k in sorted(identity_inputs):
        v = identity_inputs[k]
        if isinstance(v, dict):
            v = v.get("sha256")
        if not isinstance(v, str) or not _SHA_RE.match(v):
            raise LedgerError("identity input %s is not a sha256" % k)
        norm[k] = v
    return hashlib.sha256(canonical_json(norm).encode("utf-8")).hexdigest()


def new(domain, inputs, derivation, name_status_version, identity_inputs, prereg=None, modules=(),
        testing=False):
    """An empty funnel-ledger/1 for `domain` (a domain.Domain or a name).

    inputs: {name: path | {"path","sha256","bytes"} | sha256}. The [§8.2]
    field "inputs" keeps {name: sha256}; "input_records" keeps the path
    records for the freshness check. With a prereg the header keys of runner
    §1.2 are added."""
    if isinstance(domain, (str, Path)):
        from . import domain as D
        domain = D.load(domain)
    if derivation not in DERIVATIONS:
        raise LedgerError("derivation %r not in %s" % (derivation, DERIVATIONS))
    if name_status_version not in NAME_STATUS_VERSIONS:
        raise LedgerError("name_status_version %r not in %s" % (name_status_version, NAME_STATUS_VERSIONS))
    shas, records = {}, {}
    for k in sorted(inputs or {}):
        v = inputs[k]
        if isinstance(v, str) and _SHA_RE.match(v):
            shas[k] = v
        elif isinstance(v, dict):
            if "sha256" not in v:
                raise LedgerError("input record %s lacks sha256" % k)
            shas[k] = v["sha256"]
            if v.get("path"):
                records[k] = dict(v)
        else:
            rec = file_record(v)
            shas[k], records[k] = rec["sha256"], rec
    out = {}
    if prereg is not None:
        from . import header
        out.update(header("ledger", domain, prereg, records, modules=modules, testing=testing))
    out.update({
        "format": FORMAT,
        "domain": domain.name,
        "inputs": shas,
        "input_records": records,
        "derivation": derivation,
        "fingerprint": fingerprint(identity_inputs),
        "identity_inputs": {k: (v["sha256"] if isinstance(v, dict) else v)
                            for k, v in sorted(identity_inputs.items())},
        "name_status_version": name_status_version,
        "target_classes": list(domain.target_names),
        "other_class": domain.other["id"],
        "stages": [],
        "label_spaces": {},
        "domain_scores": {},
        "reject_class": None,
    })
    return out


def _stage_problems(st, seen):
    where = "stage %s" % st.get("id")
    probs = []
    for k in STAGE_REQUIRED:
        if k not in st:
            probs.append("%s: missing %r" % (where, k))
    for k in st:
        if k not in STAGE_KEYS:
            probs.append("%s: unknown key %r" % (where, k))
    if not isinstance(st.get("id"), str) or not st.get("id"):
        probs.append("stage without an id")
        return probs
    if st["id"] in seen:
        probs.append("%s: duplicate id" % where)
    if st.get("role") not in ROLES:
        probs.append("%s: role %r not in %s" % (where, st.get("role"), ROLES))
    if st.get("unit") not in UNITS:
        probs.append("%s: unit %r not in %s" % (where, st.get("unit"), UNITS))
    if not isinstance(st.get("guard"), bool):
        probs.append("%s: guard must be a bool" % where)
    rec = st.get("recoverable")
    if not (isinstance(rec, bool) or (isinstance(rec, str) and rec)):
        probs.append("%s: recoverable must be a bool or a non-empty string" % where)
    if st.get("guard") is True and rec is not False:
        probs.append("%s: a guard stage is recoverable (%r)" % (where, rec))
    deps = st.get("depends_on")
    if not isinstance(deps, list):
        probs.append("%s: depends_on must be a list" % where)
    else:
        for d in deps:
            if d not in seen:
                probs.append("%s: depends_on %r is not an earlier stage" % (where, d))
    for k in ("in", "kept"):
        if st.get(k) is not None and not _int(st.get(k)):
            probs.append("%s: %s must be an integer or null" % (where, k))
    disc = st.get("discarded", {})
    if disc is None:
        disc = {}
    if not isinstance(disc, dict) or not all(_int(v) for v in disc.values()):
        probs.append("%s: discarded must be {reason: int}" % where)
        disc = {}
    if _int(st.get("in")) and _int(st.get("kept")) and st.get("in") != st["kept"] + sum(disc.values()):
        probs.append("%s: in %d != kept %d + discarded %d" % (where, st["in"], st["kept"], sum(disc.values())))
    kbl = st.get("kept_by_label")
    if kbl is not None:
        if not isinstance(kbl, dict) or not all(_int(v) for v in kbl.values()):
            probs.append("%s: kept_by_label must be {class: int}" % where)
        elif _int(st.get("kept")) and sum(kbl.values()) != st["kept"]:
            probs.append("%s: kept_by_label sums to %d, kept is %d" % (where, sum(kbl.values()), st["kept"]))
    dbl = st.get("discarded_by_label")
    if dbl is not None:
        if not isinstance(dbl, dict) or not all(isinstance(v, dict) and all(_int(x) for x in v.values())
                                                for v in dbl.values()):
            probs.append("%s: discarded_by_label must be {class: {reason: int}}" % where)
    bs = st.get("by_source")
    if bs is not None and not isinstance(bs, dict):
        probs.append("%s: by_source must be an object" % where)
    blp = st.get("by_label_pred")
    if blp is not None:
        if not isinstance(blp, dict) or not all(_int(v) and isinstance(k, str) and "|" in k
                                                for k, v in blp.items()):
            probs.append("%s: by_label_pred must be {\"<label>|<pred>\": int}" % where)
    cal = st.get("calibration")
    if cal is not None:
        if not isinstance(cal, dict) or not isinstance(cal.get("known_truth_sets"), list):
            probs.append("%s: calibration must hold known_truth_sets" % where)
        else:
            for kts in cal["known_truth_sets"]:
                ds = kts.get("domain_score") if isinstance(kts, dict) else None
                if not isinstance(kts, dict) or "id" not in kts or not (
                        isinstance(ds, list) and len(ds) == 2 and all(_num(x) for x in ds)):
                    probs.append("%s: a known-truth set needs id and domain_score [lo, hi]" % where)
    aud = st.get("audit")
    if aud is not None:
        if not isinstance(aud, dict) or not isinstance(aud.get("sha256"), str) or "fn_rate" not in aud:
            probs.append("%s: audit must be null or {sha256, fn_rate}" % where)
    return probs


def add_stage(ledger, stage):
    """Append a stage; the optional keys default to null or {}. Refuses an
    unknown key, role or unit, a duplicate id and a depends_on that is not an
    earlier stage."""
    st = {"in": None, "kept": None, "discarded": {}, "kept_by_label": None,
          "discarded_by_label": None, "by_source": {}, "by_label_pred": {}, "calibration": None,
          "audit": None}
    st.update(copy.deepcopy(stage))
    seen = [s["id"] for s in ledger["stages"]]
    probs = _stage_problems(st, seen)
    if probs:
        raise LedgerError("; ".join(probs))
    ledger["stages"].append(st)
    return st


def validate(ledger):
    """Every problem of a funnel-ledger/1 (empty = valid)."""
    if not isinstance(ledger, dict):
        return ["the ledger is not a JSON object"]
    probs = []
    for k in TOP_KEYS:
        if k not in ledger:
            probs.append("missing key %r" % k)
    if ledger.get("format") != FORMAT:
        probs.append("format %r is not %s" % (ledger.get("format"), FORMAT))
    if not isinstance(ledger.get("domain"), str) or not ledger.get("domain"):
        probs.append("domain must be a name")
    inputs = ledger.get("inputs")
    if not isinstance(inputs, dict) or not all(isinstance(v, str) and _SHA_RE.match(v)
                                               for v in inputs.values()):
        probs.append("inputs must be {name: sha256}")
    if ledger.get("derivation") not in DERIVATIONS:
        probs.append("derivation %r not in %s" % (ledger.get("derivation"), DERIVATIONS))
    if not isinstance(ledger.get("fingerprint"), str) or not _SHA_RE.match(ledger.get("fingerprint") or ""):
        probs.append("fingerprint must be a sha256")
    ident = ledger.get("identity_inputs")
    if ident is not None and isinstance(ledger.get("fingerprint"), str):
        try:
            if fingerprint(ident) != ledger["fingerprint"]:
                probs.append("fingerprint does not match identity_inputs")
        except LedgerError as e:
            probs.append(str(e))
    if ledger.get("name_status_version") not in NAME_STATUS_VERSIONS:
        probs.append("name_status_version %r not in %s" % (ledger.get("name_status_version"),
                                                          NAME_STATUS_VERSIONS))
    tc = ledger.get("target_classes")
    if not isinstance(tc, list) or not tc or not all(isinstance(x, str) for x in tc):
        probs.append("target_classes must be a non-empty list of names")
    if not _int(ledger.get("other_class")):
        probs.append("other_class must be a class id")
    stages = ledger.get("stages")
    if not isinstance(stages, list):
        probs.append("stages must be a list")
        stages = []
    seen = []
    for st in stages:
        if not isinstance(st, dict):
            probs.append("a stage is not an object")
            continue
        for k in STAGE_KEYS:
            if k not in st:
                probs.append("stage %s: missing %r" % (st.get("id"), k))
        probs.extend(_stage_problems(st, seen))
        if isinstance(st.get("id"), str):
            seen.append(st["id"])
    ls = ledger.get("label_spaces")
    if not isinstance(ls, dict):
        probs.append("label_spaces must be an object")
    else:
        for src, sp in sorted(ls.items()):
            if not isinstance(sp, dict) or not _int(sp.get("boxes")) or not _int(sp.get("classes")):
                probs.append("label_spaces.%s: classes and boxes (integers) required" % src)
                continue
            kinds = sp.get("kinds")
            if not isinstance(kinds, dict) or set(kinds) != set(KINDS) or not all(_int(v) for v in kinds.values()):
                probs.append("label_spaces.%s: kinds must be {%s: int}" % (src, ", ".join(KINDS)))
            elif sum(kinds.values()) != sp["boxes"]:
                probs.append("label_spaces.%s: kinds sum to %d, boxes is %d"
                             % (src, sum(kinds.values()), sp["boxes"]))
    dsc = ledger.get("domain_scores")
    if not isinstance(dsc, dict):
        probs.append("domain_scores must be an object")
    else:
        for k, v in dsc.items():
            if k == "reference":
                if not isinstance(v, dict) or not all(_num(v.get(q)) for q in ("q05", "q50")):
                    probs.append("domain_scores.reference needs q05 and q50")
            elif v is not None and not _num(v):
                probs.append("domain_scores.%s must be a number" % k)
    rc = ledger.get("reject_class")
    if rc is not None and (not isinstance(rc, dict) or not isinstance(rc.get("stage"), str)):
        probs.append("reject_class must be null or name its stage")
    elif rc is not None and rc["stage"] not in seen:
        probs.append("reject_class.stage %r is not a stage" % rc["stage"])
    return probs


def load(path):
    """A validated ledger; LedgerError lists every problem."""
    try:
        led = read_json(path)
    except Exception as e:
        raise LedgerError(str(e))
    probs = validate(led)
    if probs:
        raise LedgerError("%s: %d problem(s): %s" % (path, len(probs), "; ".join(probs)))
    return led


def write(path, ledger):
    """Validate, then write atomically. Returns the file's sha256."""
    probs = validate(ledger)
    if probs:
        raise LedgerError("refusing to write an invalid ledger: %s" % "; ".join(probs))
    return _atomic_write_bytes(path, json_text(ledger).encode("utf-8"))


def stage_by_role(ledger, role):
    if role not in ROLES:
        raise LedgerError("role %r not in %s" % (role, ROLES))
    return [s for s in ledger["stages"] if s.get("role") == role]


def stage_index(ledger):
    return {s["id"]: s for s in ledger["stages"]}


def unaudited_dependencies(ledger):
    """(stage, dependency) pairs where a stage depends on a filter whose
    discards are recoverable and whose false-negative rate has not been
    measured (audit null): a keep/discard criterion derived from an unaudited
    filter chains that filter's errors (contract §8.1 item 6, signal S5)."""
    idx = stage_index(ledger)
    out = []
    for s in ledger["stages"]:
        for d in s.get("depends_on", []):
            dep = idx.get(d)
            if dep is not None and dep.get("recoverable") is True and dep.get("audit") is None:
                out.append((s["id"], d))
    return out


def recoverable_stages(ledger):
    return [s["id"] for s in ledger["stages"] if s.get("recoverable") is True]


def attach_audit(ledger, audit_path):
    """A new ledger whose stages carry the audit's sha256 and FN rate. An
    invalid audit (calibration overlap) is refused: nothing may cite it."""
    audit_path = Path(audit_path)
    try:
        audit = read_json(audit_path)
    except Exception as e:
        raise LedgerError(str(e))
    if audit.get("format") != "funnel-audit/1":
        raise LedgerError("%s is not a funnel-audit/1" % audit_path)
    if audit.get("valid") is not True:
        raise LedgerError("%s is not valid (calibration overlap %s); nothing may cite it"
                          % (audit_path, audit.get("calibration_overlap")))
    if audit.get("ledger_fingerprint") != ledger.get("fingerprint"):
        raise LedgerError("the audit was made on ledger %s, this ledger is %s"
                          % (str(audit.get("ledger_fingerprint"))[:12], str(ledger.get("fingerprint"))[:12]))
    sha = file_record(audit_path)["sha256"]
    out = copy.deepcopy(ledger)
    stages = audit.get("stages") or {}
    for st in out["stages"]:
        a = stages.get(st["id"])
        if a is None or a.get("fn_rate") is None:
            continue
        fr = a["fn_rate"]
        st["audit"] = {"sha256": sha, "fn_rate": {"estimate": fr.get("estimate"),
                                                  "interval": fr.get("interval"), "n": fr.get("n")}}
    return out


# --------------------------------------------------------------- unit rows
def validate_unit_row(row, stage_order=None, vocab=None):
    """Problems of one ledger.jsonl row (runner §4.2).

    stage_order: the ledger's stage ids in order; failed_stages must follow it
    and every path key must be one of them. vocab: {stage id: allowed values
    or None (any non-empty string)}, the adapter's path vocabulary; a path key
    outside it is refused."""
    if not isinstance(row, dict):
        return ["a unit row is not an object"]
    unit = row.get("unit")
    if unit not in UNIT_ROW_KEYS:
        return ["unit %r not in %s" % (unit, tuple(UNIT_ROW_KEYS))]
    probs = []
    for k in UNIT_ROW_KEYS[unit]:
        if k not in row:
            probs.append("missing %r" % k)
    for k in row:
        if k not in UNIT_ROW_KEYS[unit]:
            probs.append("unknown key %r" % k)
    if probs:
        return probs
    rid = row["id"]
    if unit == "box":
        if not isinstance(row["key"], str) or not _int(row["box"]) or row["box"] < 0:
            probs.append("key must be a string and box a non-negative integer")
        elif rid != "b:%s#%d" % (row["key"], row["box"]):
            probs.append("id %r is not b:<key>#<box>" % rid)
        if row["crop_id"] is not None and not _int(row["crop_id"]):
            probs.append("crop_id must be an integer or null")
        if not _int(row["label"]):
            probs.append("label must be a class id")
        for k in ("pred", "second"):
            if row[k] is not None and not _int(row[k]):
                probs.append("%s must be a class id or null" % k)
        for k in ("p", "cos", "p_target_max", "p_other", "p_second"):
            if row[k] is not None and not _num(row[k]):
                probs.append("%s must be a number or null" % k)
        wh = row["wh_px"]
        if not (isinstance(wh, list) and len(wh) == 2 and all(_int(x) for x in wh)):
            probs.append("wh_px must be [w, h] integers")
        for k in ("src_id", "src_name", "name_status_v1", "name_status_v2", "lab", "near_dup3",
                  "provenance", "source"):
            if not isinstance(row[k], str):
                probs.append("%s must be a string" % k)
        if row["fail"] is not None and not isinstance(row["fail"], str):
            probs.append("fail must be a code or null")
        if row["blockers"] is not None and not (isinstance(row["blockers"], list)
                                                and all(isinstance(b, str) for b in row["blockers"])):
            probs.append("blockers must be a list of codes or null")
        if not isinstance(row["kt"], list) or not all(isinstance(k, str) for k in row["kt"]):
            probs.append("kt must be a list of known-truth ids")
    else:
        if not isinstance(row["source"], str) or not isinstance(row["rel"], str):
            probs.append("source and rel must be strings")
        elif rid != "d:%s|%s" % (row["source"], row["rel"]):
            probs.append("id %r is not d:<source>|<rel>" % rid)
        if row["dhash"] is not None and not _int(row["dhash"]):
            probs.append("dhash must be an integer or null")
        near = row["near"]
        if near is not None and not (isinstance(near, dict) and {"split", "eval", "bits"} <= set(near)):
            probs.append("near must be null or {split, eval, bits}")
        if row["twin_of"] is not None and not isinstance(row["twin_of"], str):
            probs.append("twin_of must be a pool key or null")
    path = row["path"]
    if not isinstance(path, dict) or not path or not all(isinstance(k, str) and isinstance(v, str) and v
                                                        for k, v in path.items()):
        probs.append("path must be {stage id: outcome}")
        path = {}
    if vocab is not None:
        for k, v in path.items():
            if k not in vocab:
                probs.append("path stage %r is not in the adapter's vocabulary" % k)
            elif vocab[k] is not None and v not in vocab[k]:
                probs.append("path %s = %r not in %s" % (k, v, tuple(vocab[k])))
    fs = row["failed_stages"]
    if not isinstance(fs, list) or not all(isinstance(s, str) for s in fs):
        probs.append("failed_stages must be a list of stage ids")
        fs = []
    if len(set(fs)) != len(fs):
        probs.append("failed_stages repeats a stage")
    if stage_order is not None:
        order = {s: i for i, s in enumerate(stage_order)}
        for s in list(fs) + list(path):
            if s not in order:
                probs.append("stage %r is not a ledger stage" % s)
        idx = [order[s] for s in fs if s in order]
        if idx != sorted(idx):
            probs.append("failed_stages %s are not in stage order" % fs)
    want_first = fs[0] if fs else None
    want_sole = fs[0] if len(fs) == 1 else None
    if row["first_cause"] != want_first:
        probs.append("first_cause %r, want %r" % (row["first_cause"], want_first))
    if row["sole_cause"] != want_sole:
        probs.append("sole_cause %r, want %r" % (row["sole_cause"], want_sole))
    return probs


def iter_units(path, unit=None):
    """Stream ledger.jsonl rows (optionally one unit kind) without loading the
    file whole. A malformed line raises LedgerError naming it."""
    path = Path(path)
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        raise LedgerError("cannot read %s (%s)" % (path, e))
    with fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise LedgerError("%s line %d is not JSON (%s)" % (path, i, e))
            if unit is None or row.get("unit") == unit:
                yield row
