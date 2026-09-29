"""The INC Step 1 adapter (runner §5.2.2): Step 1's files as the funnel engine's
ledger, census, stratum raw material, known truth and disjointness keys.

Step 1 (inc/verify.py, inc/select.py) is read through its own path constants
and readers and never edited. Two derivations of the funnel ledger:

  ledger_from_summaries   the three Step 1 summaries, calibration.json and
                          census_v0.json only (lab or cluster login node): the
                          stage table of contract §3 with name status v1, so
                          D17 and D19 can fire before any census job runs (F2).
  census                  the cluster job F3: every box of every pool image
                          and every image dropped before the pool, with its
                          discard path (ledger.jsonl), the per-class census
                          (census_v1.json), name status v2 (name_status_v2.json,
                          through the taxonomy cache), the guard pairs
                          (guard_pairs_v1.csv) and the census-derived ledger.

The census re-derives the verifier's scores (P over the 13 classes and the
prototype cosines) from the fitted probe and the Step 1 embeddings, applies
verify.verdicts, and must reproduce pool_verdicts.npz (verdict and argmax) for
every pool crop, or it refuses. The images dropped at S2-S5 are listed again
by verify's own directory reads (_resolve_dir, _layout, one listdir per
directory) and hash cache, and their per-slug counts must reproduce
pool_summary.json. Every count the contract's F3 acceptance names is checked
against the Step 1 summaries (reconciliation); a failed check writes the
census with reconciliation.ok false and then refuses.

This adapter is the domain's: it may name the domain's classes, sources and
files. Everything it writes carries the funnel header (inputs with sha256).
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from ...inc import common as C
from ...inc import verify as V
from ... import cwd12_species as SP
from .. import (AdapterError, FunnelError, SampleLocked, StaleInput, TaxonomyError, file_record, header,
                read_json, write_csv_atomic, write_json_atomic)
from .. import domain as D
from .. import ledger as L
from .. import names as N
from .. import taxonomy as T

OTHER = C.OTHER_PLANT
NT = C.OTHER_PLANT                     # number of target classes (ids 0..11)
CHUNK = 50000
ADAPTER_STAGES = ("S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S7b", "S8b", "S8", "S9",
                  "S10", "S11", "S12", "S13", "S14")
STAGE_ROLES = {"S-1": "discovery", "S0": "cap", "S1": "registry", "S2": "read", "S3": "dedup",
               "S4": "guard", "S5": "guard", "S6": "size", "S7": "join", "S7b": "name_status",
               "S8b": "reject_class_sample", "S8": "target_check", "S9": "other_check",
               "S10": "image_rule", "S11": "selection", "S12": "evidence", "S13": "relevance",
               "S14": "gate"}
FAIL_CODES = ("argmax_other_confident", "argmax_target_confident", "argmax_wrong", "p_below_tau",
              "cos_below_sigma", "failed")
BLOCKERS = ("other_called_target", "species_conflict", "species_unknown", "small_species_box", "failed")
S2_REASONS = ("no_label", "unreadable_label", "no_boxes", "unmapped_class", "unhashable",
              "calibration_only_not_train_core")
READ_REASONS = ("no_label", "unreadable_label", "no_boxes", "unmapped_class")
SELECT_BASE = ("selected", "refilled")
PATH_VOCAB = {
    "S2": ("pass",) + S2_REASONS,
    "S3": ("kept", "exact_dup", "n/a"),
    "S4": ("pass", "near_eval", "n/a"),
    "S5": ("pass", "cwd12_copy", "n/a"),
    "S6": ("embedded", "small", "no_size"),
    "S7": ("target", "other"),
    "S8": ("verified", "conflict", "unknown", "failed", "n/a"),
    "S9": ("other_ok", "conflict", "failed", "n/a"),
    "S10": ("admitted", "conflict", "unknown"),
    "S11": None,
    "S12": ("evidenced", "not_evidenced", "n/a"),
}
V1_NAMED = "named_other"
MIN_EVIDENCE = 1                        # select.MIN_EVIDENCE (the increment evidence criterion)


def log(msg):
    print("[funnel.inc_step1] %s" % msg, flush=True)


# ============================================================= name status v1
def name_status_v1(name):
    """verify.other_name_status of a source class name, with numeric names split
    out of "generic" (name_key(name).isdigit(), which reproduces census_v0's
    55,657 numeric boxes). Strings: target, target_related, no_name, numeric,
    generic, not_a_plant, named_other."""
    s = V.other_name_status(name or "")
    if s is None:
        return V1_NAMED
    if s == "generic" and SP.name_key(name or "").isdigit():
        return "numeric"
    return {"cwd12_species": "target", "cwd12_related": "target_related"}.get(s, s)


def v1_kind(status):
    """The funnel-ledger/1 label-space kind of a v1 status."""
    return {"no_name": "none", "numeric": "numeric", "generic": "generic"}.get(status, "named")


def v2_kind(status, domain):
    if status == "no_name":
        return "none"
    if status == "numeric":
        return "numeric"
    if status in ("generic", "unresolvable"):
        return "generic"
    return "named"


# ============================================================= paths and inputs
def _step1(step1_dir=None):
    if step1_dir is not None and Path(step1_dir).resolve() != Path(V.STEP1).resolve():
        raise AdapterError("the census reads Step 1 through verify's own paths (%s), not %s"
                           % (V.STEP1, step1_dir))
    return Path(V.STEP1)


def _funnel_dir(out_dir=None):
    from .. import FUNNEL_DIR
    return Path(out_dir) if out_dir is not None else Path(FUNNEL_DIR)


def crop_table():
    """step1/crops.csv as verify.Crops, after verify.check_fresh (StaleInput when a
    file crops.csv was made from changed since)."""
    try:
        crops = V.Crops()
        V.check_fresh(crops)
    except V.VerifyError as e:
        raise StaleInput(str(e))
    return crops


def _read_jsonl(path):
    try:
        return C.read_manifest(path)
    except OSError as e:
        raise AdapterError("cannot read %s (%s)" % (path, e))


def _read_json(path):
    try:
        return read_json(path)
    except FunnelError as e:
        raise AdapterError(str(e))


def _sha(path):
    return C.sha256_file(path)


# ============================================================= summaries ledger
def verifier_fit_info_projection(step1_dir=None):
    """{"derived_from": record, "other_sample": {...}}: the part of
    step1/verifier/fit_info.json the lab may read (the OtherPlant training
    sample's sources and name counts), shipped as step1/verifier_fit_info.json."""
    step1 = Path(step1_dir) if step1_dir is not None else Path(V.STEP1)
    path = step1 / "verifier" / "fit_info.json"
    fi = _read_json(path)
    os_ = fi.get("other_sample")
    if not isinstance(os_, dict):
        raise AdapterError("%s has no other_sample record" % path)
    keep = ("requested", "taken", "seed", "sources", "eligible_names", "excluded_names_by_reason")
    return {"format": "funnel-verifier-fit-info-projection/1", "source": "derived",
            "derived_from": {"path": str(path), "sha256": _sha(path)},
            "other_sample": {k: os_.get(k) for k in keep}}


def _load_summaries(step1_dir):
    step1 = Path(step1_dir)
    out = {}
    for name in ("pool_summary", "admit_summary", "select_summary", "calibration"):
        p = step1 / ("%s.json" % name)
        out[name] = _read_json(p)
        out[name + "_path"] = p
    return out


def _identity(s):
    return {k: _sha(s[k + "_path"]) for k in ("admit_summary", "pool_summary", "select_summary", "calibration")}


def rows_v0_view(rows):
    """census_v0's rows (one per source, source class name and label) from
    census rows at any finer grain (census_v1 rows are per source class id):
    counts summed, mean_p the pred-weighted mean, rounded to 3 decimals."""
    acc = collections.OrderedDict()
    for r in rows:
        k = (r["source"], r["src_name"], r["label"])
        a = acc.get(k)
        if a is None:
            a = acc[k] = {"source": r["source"], "src_name": r["src_name"], "label": r["label"], "n": 0,
                          "verdict": collections.Counter(), "pred_all": collections.Counter(),
                          "pred_confident": collections.Counter(), "_psum": collections.Counter()}
        a["n"] += int(r["n"])
        a["verdict"].update(r["verdict"])
        a["pred_all"].update(r["pred_all"])
        a["pred_confident"].update(r["pred_confident"])
        for c, m in (r.get("mean_p") or {}).items():
            a["_psum"][c] += float(m) * int(r["pred_all"].get(c, 0))
    out = []
    for a in acc.values():
        mp = {c: round(a["_psum"][c] / a["pred_all"][c], 3) for c in a["pred_all"] if a["pred_all"][c]}
        out.append({"source": a["source"], "src_name": a["src_name"], "label": a["label"], "n": a["n"],
                    "verdict": dict(a["verdict"]), "pred_all": dict(a["pred_all"]),
                    "pred_confident": dict(a["pred_confident"]), "mean_p": mp})
    return out


def aggregate_rows(sources, src_ids, src_names, labels, verdicts, preds, ps):
    """Census rows (source, src_id, src_name, label, n, verdict, pred_all,
    pred_confident, mean_p) from per-crop arrays. verdicts are verdict names;
    preds class ids (argmax; -1 for failed crops, counted under "failed");
    ps the argmax probability. pred_confident counts the argmax over the crops
    whose verdict is verified or conflict (the confident calls). mean_p is the
    mean argmax probability per argmax class, rounded to 3 decimals."""
    acc = collections.OrderedDict()
    for s, sid, nm, lab, v, j, p in zip(sources, src_ids, src_names, labels, verdicts, preds, ps):
        k = (s, str(sid), int(lab))
        a = acc.get(k)
        if a is None:
            a = acc[k] = {"source": s, "src_id": str(sid), "src_name": nm, "label": C.CLASS_NAMES[int(lab)],
                          "n": 0, "verdict": collections.Counter(), "pred_all": collections.Counter(),
                          "pred_confident": collections.Counter(), "_psum": collections.Counter()}
        elif a["src_name"] != nm:
            raise AdapterError("source class %s|%s carries two names (%r, %r)" % (s, sid, a["src_name"], nm))
        a["n"] += 1
        a["verdict"][str(v)] += 1
        j = int(j)
        cname = C.CLASS_NAMES[j] if 0 <= j < C.NC else "failed"
        a["pred_all"][cname] += 1
        if v in (V.VERIFIED, V.CONFLICT):
            a["pred_confident"][cname] += 1
        if 0 <= j < C.NC and np.isfinite(p):
            a["_psum"][cname] += float(p)
    rows = []
    for a in acc.values():
        mp = {c: round(a["_psum"][c] / a["pred_all"][c], 3) for c in a["pred_all"]
              if c != "failed" and a["pred_all"][c]}
        rows.append({"source": a["source"], "src_id": a["src_id"], "src_name": a["src_name"],
                     "label": a["label"], "n": a["n"], "verdict": dict(sorted(a["verdict"].items())),
                     "pred_all": dict(sorted(a["pred_all"].items())),
                     "pred_confident": dict(sorted(a["pred_confident"].items())), "mean_p": mp})
    rows.sort(key=lambda r: (r["source"], r["src_id"], r["label"]))
    return rows


def rows_from_step1_files(crops_path, verdicts_path, src_id_of=None):
    """Census rows from a crops.csv and a pool_verdicts.npz in verify's formats
    (verify.CROP_FIELDS; arrays crop_id, verdict, pred, p, cosine and meta with
    crops_sha256 and verdict_codes). src_id_of(source, src_name) gives the
    source class id (the census reads it from pool_meta.jsonl); by default the
    name stands for its id. The census computes its rows with the same
    aggregate_rows."""
    # streamed (a million rows as verify.Crops lists would not fit a test);
    # the same checks: verify's columns, crop_id 0..n-1 in order
    sha = C.sha256_file(crops_path)
    cols = {"source": [], "src_name": [], "label": [], "set": []}
    with open(crops_path, newline="") as fh:
        rd = csv.reader(fh)
        if tuple(next(rd)) != V.CROP_FIELDS:
            raise AdapterError("%s: not verify's CROP_FIELDS" % crops_path)
        ix = {f: V.CROP_FIELDS.index(f) for f in ("crop_id", "set", "source", "src_name", "label")}
        for n, row in enumerate(rd):
            if int(row[ix["crop_id"]]) != n:
                raise AdapterError("%s: crop_id is not 0..n-1 in order" % crops_path)
            for f in ("source", "src_name", "set"):
                cols[f].append(row[ix[f]])
            cols["label"].append(int(row[ix["label"]]))
    with np.load(verdicts_path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        cid, verdict, pred, p = d["crop_id"], d["verdict"], d["pred"], d["p"]
    if meta.get("crops_sha256") != sha:
        raise AdapterError("%s was made from another crops.csv" % verdicts_path)
    codes = list(meta["verdict_codes"])
    idx = np.asarray(cid, dtype=np.int64)
    if any(cols["set"][i] != "pool" for i in idx):
        raise AdapterError("%s names a crop that is not a pool crop" % verdicts_path)
    src = [cols["source"][i] for i in idx]
    nms = [cols["src_name"][i] for i in idx]
    sids = [(src_id_of(s, n) if src_id_of else n) for s, n in zip(src, nms)]
    labels = np.asarray([cols["label"][i] for i in idx], dtype=np.int64)
    return aggregate_rows(src, sids, nms, labels, [codes[int(x)] for x in verdict], pred, p)


def _kinds_from_rows(rows, status_of, kind_of):
    """label_spaces {src: {classes, kinds, boxes}} over embedded boxes."""
    out = {}
    for r in rows:
        sp = out.setdefault(r["source"], {"classes": set(), "kinds": collections.Counter(), "boxes": 0})
        sp["classes"].add(r.get("src_id", r["src_name"]))
        sp["kinds"][kind_of(status_of(r))] += int(r["n"])
        sp["boxes"] += int(r["n"])
    return {s: {"classes": len(v["classes"]),
                "kinds": {k: int(v["kinds"].get(k, 0)) for k in L.KINDS}, "boxes": v["boxes"]}
            for s, v in sorted(out.items())}


def _check_domain_stages(domain):
    have = [(s["id"], s["role"]) for s in domain.stages]
    want = [(s, STAGE_ROLES[s]) for s in ADAPTER_STAGES]
    if have != want:
        raise AdapterError("domain %s names stages %s; the Step 1 adapter fills %s" % (domain.name, have, want))


def _cfg_stage(domain, sid):
    s = domain.stage(sid)
    return {"id": sid, "filter": s.get("filter") or s.get("name") or sid, "version": "step1",
            "unit": s["unit"], "role": s["role"], "depends_on": list(s.get("depends_on", [])),
            "recoverable": s["recoverable"], "guard": bool(s["guard"])}


def _calibration_sets(cal, select_summary):
    """The known-truth sets the verifier was calibrated on (calibration.json):
    the cwd12 copies per slug. Their images are copies of reference images, so
    their domain score spans the reference range [q05, 1]."""
    q05, _q50 = select_summary["retrieval"]["train_core_image_score_q05_q50"]
    per = ((cal.get("cwd12_copies") or {}).get("current_join") or {}).get("per_slug") or {}
    sets = []
    for slug in sorted(per):
        sets.append({"id": "copies:%s" % slug, "sha256": None, "sources": [slug],
                     "domain_score": [float(q05), 1.0],
                     "domain_basis": "copies of reference images (within the train_core copy radius)"})
    sets.append({"id": "swaps:train_core", "sha256": None, "sources": ["train_core"],
                 "domain_score": [float(q05), 1.0], "domain_basis": "held-out reference images"})
    return sets


def _build_stages(domain, parts):
    """The funnel-ledger/1 stages of Step 1 from counts both derivations share.

    parts: pool_summary, admit_summary, select_summary, calibration (raw),
    calibration_sha256, rows (census rows with label, verdict, pred_confident,
    source), eligible_other (S8b in, or None), taken_other (S8b kept, or None)."""
    ps, ad, ss, cal = parts["pool_summary"], parts["admit_summary"], parts["select_summary"], parts["calibration"]
    rows = parts["rows"]
    stages = []
    per_slug = ps["per_slug"]
    # S-1
    st = _cfg_stage(domain, "S-1")
    stages.append(st)
    # S0 cap: sources whose card lists more images than the harvest listed
    st = _cfg_stage(domain, "S0")
    cards = (domain.raw["sources"].get("card_image_counts") or {})
    bs, tin, tkept = {}, 0, 0
    for slug, n in sorted(cards.items()):
        listed = int((per_slug.get(slug) or {}).get("images_listed", 0))
        if slug not in per_slug:
            continue
        bs[slug] = {"in": int(n), "kept": listed, "discarded": {"cap": max(0, int(n) - listed)}}
        tin += int(n)
        tkept += listed
    st.update({"in": tin, "kept": tkept, "discarded": {"cap": tin - tkept}, "by_source": bs})
    stages.append(st)
    # S1 registry
    st = _cfg_stage(domain, "S1")
    st.update({"in": int(ps["slugs_total"]), "kept": int(ps["slugs_used"]),
               "discarded": {k: int(v) for k, v in sorted(ps["skipped_counts"].items())}})
    stages.append(st)
    # image flow in verify's order: read -> near_eval (S4) -> copy (S5) -> exact_dup (S3)
    listed = sum(int(v["images_listed"]) for v in per_slug.values())
    drop = collections.Counter()
    for slug, v in sorted(per_slug.items()):
        drop.update(v.get("dropped") or {})
    s2d = {r: int(drop.get(r, 0)) for r in S2_REASONS if drop.get(r, 0)}
    after2 = listed - sum(s2d.values())
    ne = int(drop.get("near_eval", 0))
    cp = int(drop.get("cwd12_copy", 0))
    ex = int(drop.get("exact_dup", 0))
    unknown = set(drop) - set(S2_REASONS) - {"near_eval", "cwd12_copy", "exact_dup"}
    if unknown:
        raise AdapterError("pool_summary.json drops images for reasons the adapter does not know: %s"
                           % sorted(unknown))
    def by_src(reasons):
        out = {}
        for slug, v in sorted(per_slug.items()):
            d = {r: int(n) for r, n in (v.get("dropped") or {}).items() if r in reasons}
            if d:
                out[slug] = {"discarded": d}
        return out
    st = _cfg_stage(domain, "S2")
    st.update({"in": listed, "kept": after2, "discarded": s2d, "by_source": by_src(S2_REASONS)})
    s2 = st
    st3 = _cfg_stage(domain, "S3")
    st4 = _cfg_stage(domain, "S4")
    st5 = _cfg_stage(domain, "S5")
    st4.update({"in": after2, "kept": after2 - ne, "discarded": {"near_eval": ne}, "by_source": by_src({"near_eval"})})
    st5.update({"in": after2 - ne, "kept": after2 - ne - cp, "discarded": {"cwd12_copy": cp},
                "by_source": by_src({"cwd12_copy"})})
    st3.update({"in": after2 - ne - cp, "kept": after2 - ne - cp - ex, "discarded": {"exact_dup": ex},
                "by_source": by_src({"exact_dup"})})
    if st3["kept"] != int(ps["images"]):
        raise AdapterError("image flow does not reconcile: %d listed - drops = %d, pool_summary images %d"
                           % (listed, st3["kept"], ps["images"]))
    stages.extend([s2, st3, st4, st5])
    # S6 size
    small = int(ad["boxes_small_not_embedded"])
    pool_boxes = int(ps["boxes"])
    embedded = sum(int(r["n"]) for r in rows)
    st = _cfg_stage(domain, "S6")
    st.update({"in": pool_boxes, "kept": pool_boxes - small, "discarded": {"small": small}})
    if pool_boxes - small != embedded:
        raise AdapterError("pool boxes %d - small %d != embedded crops %d" % (pool_boxes, small, embedded))
    stages.append(st)
    # S7 join: every box keeps a label; kept_by_label is the joined class of every pool box
    st = _cfg_stage(domain, "S7")
    kbl = {n: int(ps["boxes_per_class"].get(n, 0)) for n in C.CLASS_NAMES}
    st.update({"in": pool_boxes, "kept": sum(kbl.values()), "discarded": {}, "kept_by_label": kbl})
    if sum(kbl.values()) != pool_boxes:
        raise AdapterError("pool_summary boxes_per_class sums to %d, boxes %d" % (sum(kbl.values()), pool_boxes))
    stages.append(st)
    # S7b name status: a classification of the class names
    st = _cfg_stage(domain, "S7b")
    ncls = len({(r["source"], r.get("src_id", r["src_name"])) for r in rows})
    st.update({"in": ncls, "kept": ncls, "discarded": {}})
    stages.append(st)
    # S8b reject class sample
    st = _cfg_stage(domain, "S8b")
    st.update({"in": parts.get("eligible_other"), "kept": parts.get("taken_other"), "discarded": {}})
    if st["in"] is not None and st["kept"] is not None:
        st["discarded"] = {"not_drawn": int(st["in"]) - int(st["kept"])}
    stages.append(st)
    # S8 / S9 from the census rows
    tg = [r for r in rows if r["label"] != C.CLASS_NAMES[OTHER]]
    ot = [r for r in rows if r["label"] == C.CLASS_NAMES[OTHER]]
    def vsum(rs, v):
        return sum(int(r["verdict"].get(v, 0)) for r in rs)
    kept_by_label = collections.Counter()
    disc_by_label = collections.defaultdict(collections.Counter)
    by_source8, by_source9 = {}, {}
    blp8, blp9 = collections.Counter(), collections.Counter()
    for r in tg:
        kept_by_label[r["label"]] += int(r["verdict"].get(V.VERIFIED, 0))
        for v in (V.CONFLICT, V.UNKNOWN, V.FAILED):
            if r["verdict"].get(v):
                disc_by_label[r["label"]][v] += int(r["verdict"][v])
        b = by_source8.setdefault(r["source"], {"in": 0, "kept": 0, "discarded": collections.Counter()})
        b["in"] += int(r["n"])
        b["kept"] += int(r["verdict"].get(V.VERIFIED, 0))
        for v in (V.CONFLICT, V.UNKNOWN, V.FAILED):
            if r["verdict"].get(v):
                b["discarded"][v] += int(r["verdict"][v])
        for c, n in r["pred_confident"].items():
            if c != r["label"]:
                blp8["%s|%s" % (r["label"], c)] += int(n)
    for r in ot:
        b = by_source9.setdefault(r["source"], {"in": 0, "kept": 0, "discarded": collections.Counter()})
        b["in"] += int(r["n"])
        b["kept"] += int(r["verdict"].get(V.OTHER_OK, 0))
        for v in (V.CONFLICT, V.FAILED):
            if r["verdict"].get(v):
                b["discarded"][v] += int(r["verdict"][v])
        for c, n in r["pred_confident"].items():
            blp9["%s|%s" % (r["label"], c)] += int(n)
    st = _cfg_stage(domain, "S8")
    disc8 = {v: vsum(tg, v) for v in (V.CONFLICT, V.UNKNOWN, V.FAILED) if vsum(tg, v)}
    cur = ((cal.get("cwd12_copies") or {}).get("current_join") or {}).get("overall") or {}
    st.update({"in": sum(int(r["n"]) for r in tg), "kept": vsum(tg, V.VERIFIED), "discarded": disc8,
               "kept_by_label": {n: int(kept_by_label.get(n, 0)) for n in C.CLASS_NAMES[:NT]},
               "discarded_by_label": {k: dict(v) for k, v in sorted(disc_by_label.items())},
               "by_source": {s: {"in": b["in"], "kept": b["kept"], "discarded": dict(b["discarded"])}
                             for s, b in sorted(by_source8.items())},
               "by_label_pred": dict(sorted(blp8.items())),
               "calibration": {"known_truth_sets": [dict(k, sha256=parts["calibration_sha256"])
                                                    for k in _calibration_sets(cal, ss)],
                               "precision": cur.get("verified_precision"),
                               "recall": cur.get("verified_recall_on_correct"),
                               "domains_covered": ["reference"]}})
    stages.append(st)
    st = _cfg_stage(domain, "S9")
    disc9 = {v: vsum(ot, v) for v in (V.CONFLICT, V.FAILED) if vsum(ot, v)}
    st.update({"in": sum(int(r["n"]) for r in ot), "kept": vsum(ot, V.OTHER_OK), "discarded": disc9,
               "by_source": {s: {"in": b["in"], "kept": b["kept"], "discarded": dict(b["discarded"])}
                             for s, b in sorted(by_source9.items())},
               "by_label_pred": dict(sorted(blp9.items())),
               "calibration": {"known_truth_sets": [dict(k, sha256=parts["calibration_sha256"])
                                                    for k in _calibration_sets(cal, ss)],
                               "precision": None, "recall": None, "domains_covered": ["reference"]}})
    stages.append(st)
    # S10 image rule
    im = ad["images"]
    st = _cfg_stage(domain, "S10")
    st.update({"in": sum(int(v) for v in im.values()), "kept": int(im.get(V.ADMITTED, 0)),
               "discarded": {k: int(v) for k, v in sorted(im.items()) if k != V.ADMITTED},
               "by_source": {s: {"in": sum(int(x) for x in (d.get("images") or {}).values()),
                                 "kept": int((d.get("images") or {}).get(V.ADMITTED, 0)),
                                 "discarded": {k: int(x) for k, x in (d.get("images") or {}).items()
                                               if k != V.ADMITTED}}
                             for s, d in sorted((ad.get("per_slug") or {}).items())}})
    stages.append(st)
    # S11 selection (routing): selected, the rest by status
    sz = ss["sizes"]
    st = _cfg_stage(domain, "S11")
    d11 = {k: int(sz.get(k, 0)) for k in ("no_evidence", "below_gate", "no_feature") if sz.get(k)}
    if sz.get("dropped_other_budget"):
        d11["dropped_other_budget"] = int(sz["dropped_other_budget"])
    if sz.get("dropped_by_cap"):
        d11["dropped_by_cap"] = int(sz["dropped_by_cap"])
    rest = int(sz["verified"]) - int(sz["selected"]) - int(sz.get("refilled", 0)) - sum(d11.values())
    if rest:
        d11["pool"] = rest
    st.update({"in": int(sz["verified"]), "kept": int(sz["selected"]) + int(sz.get("refilled", 0)),
               "discarded": d11})
    stages.append(st)
    # S12 evidence criterion over the increment pool's sources
    inc_src = (ss.get("sources") or {}).get("increment_pool") or {}
    ver_by = {s: int(((d or {}).get("boxes") or {}).get(V.VERIFIED, 0)) for s, d in (ad.get("per_slug") or {}).items()}
    ev = {s for s in inc_src if ver_by.get(s, 0) >= MIN_EVIDENCE}
    st = _cfg_stage(domain, "S12")
    st.update({"in": len(inc_src), "kept": len(ev), "discarded": {"not_evidenced": len(inc_src) - len(ev)},
               "by_source": {s: {"in": int(n), "kept": int(n) if s in ev else 0,
                                 "discarded": {} if s in ev else {"not_evidenced": int(n)}}
                             for s, n in sorted(inc_src.items())}})
    stages.append(st)
    for sid in ("S13", "S14"):
        stages.append(_cfg_stage(domain, sid))
    return stages


def _domain_scores(select_summary):
    r = select_summary["retrieval"]
    q05, q50 = r["train_core_image_score_q05_q50"]
    out = {s: float(v["median"]) for s, v in sorted((r.get("source_evidence") or {}).items())
           if v.get("median") is not None}
    out["reference"] = {"q05": float(q05), "q50": float(q50)}
    return out


def _reject_class_from_projection(proj):
    if not proj:
        return None
    os_ = proj.get("other_sample") or {}
    names = os_.get("eligible_names") or {}
    tot = sum(int(v) for v in names.values())
    top = None
    if names and tot:
        n, c = sorted(names.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0]
        top = {"cell": n, "n": int(c), "share": round(int(c) / tot, 6), "of": tot,
               "basis": "eligible names (fit_info eligible_names; name, not source)"}
    return {"stage": "S8b", "eligible_top_cell": top, "drawn_top_cell": None,
            "sample_sources": {k: int(v) for k, v in sorted((os_.get("sources") or {}).items())} or None}


def ledger_from_summaries(domain, step1_dir, census_v0, out_path, fit_info_projection=None, prereg=None,
                          testing=False):
    """funnel_ledger.json (derivation "summaries", name status v1) from
    step1/{pool,admit,select}_summary.json, calibration.json and census_v0.json.
    The reject-class record comes from fit_info_projection when given, else
    from step1/verifier_fit_info.json (the projection the lab holds), else
    from step1/verifier/fit_info.json (the cluster's own record), else it is
    null. Refuses to replace a census-derived ledger. Returns the ledger."""
    domain = D.load(domain)
    _check_domain_stages(domain)
    out_path = Path(out_path)
    if out_path.exists():
        try:
            old = read_json(out_path)
        except FunnelError:
            old = None
        if isinstance(old, dict) and old.get("derivation") == "census":
            raise AdapterError("%s is census-derived; a summaries ledger never replaces it" % out_path)
    s = _load_summaries(step1_dir)
    v0_path = Path(census_v0)
    rows = _read_json(v0_path)
    if not isinstance(rows, list) or not rows:
        raise AdapterError("%s is not a census_v0 row list" % v0_path)
    proj, proj_rec = None, None
    if fit_info_projection is None:
        shipped = Path(step1_dir) / "verifier_fit_info.json"          # the lab's shipped projection
        fitted = Path(step1_dir) / "verifier" / "fit_info.json"        # the cluster's own record
        if shipped.is_file():
            fit_info_projection = shipped
        elif fitted.is_file():
            proj_rec = file_record(fitted)
            proj = verifier_fit_info_projection(step1_dir)
    if fit_info_projection is not None:
        if isinstance(fit_info_projection, (str, Path)):
            proj_rec = file_record(fit_info_projection)
            proj = _read_json(fit_info_projection)
        else:
            proj = fit_info_projection
    eligible = sum(int(r["n"]) for r in rows if r["label"] == C.CLASS_NAMES[OTHER]
                   and name_status_v1(r["src_name"]) == V1_NAMED)
    taken = int(((proj or {}).get("other_sample") or {}).get("taken")) if proj else None
    parts = dict(s, rows=rows, calibration_sha256=_sha(s["calibration_path"]),
                 eligible_other=eligible, taken_other=taken)
    inputs = {k: file_record(s[k + "_path"]) for k in ("admit_summary", "pool_summary", "select_summary",
                                                       "calibration")}
    inputs["census_v0"] = file_record(v0_path)
    if proj_rec:
        inputs["verifier_fit_info"] = proj_rec
    _check_totals_v0(rows, s["admit_summary"])
    led = L.new(domain, inputs, "summaries", "v1", _identity(s), prereg=prereg,
                modules=(_self_module(),) if prereg is not None else (), testing=testing)
    for st in _build_stages(domain, parts):
        L.add_stage(led, st)
    led["label_spaces"] = _kinds_from_rows(rows, lambda r: name_status_v1(r["src_name"]), v1_kind)
    led["domain_scores"] = _domain_scores(s["select_summary"])
    led["reject_class"] = _reject_class_from_projection(proj)
    L.write(out_path, led)
    return led


def _check_totals_v0(rows, admit):
    got = collections.Counter()
    for r in rows:
        got.update(r["verdict"])
    want = {k: int(v) for k, v in admit["boxes"].items() if k not in (V.SMALL,)}
    got = {k: int(v) for k, v in got.items()}
    if {k: v for k, v in got.items() if v} != {k: v for k, v in want.items() if v}:
        raise AdapterError("census_v0 verdict totals %s do not reconcile with admit_summary.json %s" % (got, want))


def _self_module():
    import sys
    return sys.modules[__name__]


# ============================================================= contract numbers
_NUM = r"([0-9][0-9,]*)"


def _int(s):
    return int(s.replace(",", ""))


def contract_numbers(contract_path):
    """The Step 1 counts the contract states in §1-§3 (read from the contract
    file, never typed), as {name: int}. Used to record contract checks and by
    the tests; a phrase the file does not hold raises AdapterError."""
    text = Path(contract_path).read_text(encoding="utf-8")
    N_ = _NUM
    pats = [
        (r"BroWeed and NarWeed %s boxes; \"crop\" %s; non-plant objects %s; disease or state names %s"
         % (N_, N_, N_, N_), ("s7b_broweed_narweed", "s7b_crop", "s7b_non_object", "s7b_state")),
        (r"The lost boxes sit in %s images" % N_, ("veto_images",)),
        (r"an OtherPlant box called a species: %s;\s*- a species conflict: %s;\s*- an unknown species box: %s\."
         % (N_, N_, N_), ("veto_other_called_target", "veto_species_conflict", "veto_species_unknown")),
        (r"The remaining %s are most likely" % N_, ("veto_remaining",)),
        (r"no_boxes %s \(rf_tuf %s; csgo %s\)" % (N_, N_, N_), ("s2_no_boxes", "s2_no_boxes_rf_tuf",
                                                             "s2_no_boxes_csgo")),
        (r"the alphabetically first slug is kept \| %s \|" % N_, ("s3_exact_dup",)),
        (r"within 6 dHash bits of dev, test or an exam \| %s \(test %s; dev %s; ood23 %s; ood22 %s; imageweeds %s\)"
         % (N_, N_, N_, N_, N_, N_), ("s4_near_eval", "s4_test", "s4_dev", "s4_ood23", "s4_ood22",
                                      "s4_imageweeds")),
        (r"within 6 bits of a train_core image \| %s \|" % N_, ("s5_cwd12_copy",)),
        (r"never embedded \| %s \|" % N_, ("s6_small",)),
        (r"Of %s embedded OtherPlant boxes, by name status \(`verify.other_name_status`\): named other plant %s; "
         r"generic %s; no name %s; numeric %s; target-related %s" % (N_, N_, N_, N_, N_, N_),
         ("s7_other_embedded", "s7_named_other", "s7_generic", "s7_no_name", "s7_numeric", "s7_target_related")),
        (r"\| Of %s boxes: \*\*verified %s\*\*; unknown %s \(" % (N_, N_, N_),
         ("s8_in", "s8_verified", "s8_unknown")),
        (r"; conflict %s\. Of the conflicts, %s are confidently OtherPlant" % (N_, N_),
         ("s8_conflict", "s8_conflict_other")),
        (r"and %s another target \(%s → PalmerAmaranth\)" % (N_, N_), ("s8_conflict_target",
                                                                     "s8_conflict_target_palmer")),
        (r"\| %s conflicts\. By predicted class" % N_, ("s9_conflict",)),
        (r"By name status: no name %s; numeric %s; named other %s; target-related %s; generic %s\."
         % (N_, N_, N_, N_, N_), ("s9_no_name", "s9_numeric", "s9_named_other", "s9_target_related",
                                  "s9_generic")),
        (r"Images: conflict %s; unknown %s\. Verified target boxes admitted: %s of %s\. \*\*%s were lost"
         % (N_, N_, N_, N_, N_), ("s10_img_conflict", "s10_img_unknown", "s10_admitted_target",
                                  "s10_verified", "s10_lost")),
        (r"Of %s admitted images: no_evidence %s; below_gate %s; no_feature %s; OtherPlant budget %s; "
         r"selected %s\." % (N_, N_, N_, N_, N_, N_),
         ("s11_admitted_images", "s11_no_evidence", "s11_below_gate", "s11_no_feature", "s11_other_budget",
          "s11_selected")),
        (r"increment pool \(%s\)" % N_, ("s11_increment_pool",)),
        (r"\| %s of %s sources; %s of %s pool images \|" % (N_, N_, N_, N_),
         ("s12_not_evidenced", "s12_sources", "s12_images_out", "s12_pool_images")),
        (r"Target-labelled boxes in the pool: %s → embedded: %s → verified: %s → admitted: %s → in base B: %s"
         % (N_, N_, N_, N_, N_), ("pool_target_boxes", "embedded_target_boxes", "funnel_verified",
                                  "funnel_admitted", "base_target_boxes")),
        (r"\| %s of %s slugs: never_train %s, quarantined %s, unknown_layout %s, user_flag_garbage %s\."
         % (N_, N_, N_, N_, N_, N_), ("s1_skipped", "s1_slugs", "s1_never_train", "s1_quarantined",
                                      "s1_unknown_layout", "s1_user_flag")),
        (r"Step 1 harvested %s images and %s boxes from %s sources" % (N_, N_, N_),
         ("harvest_images", "harvest_boxes", "harvest_sources")),
        (r"\| %s / %s = \*\*0\.404\*\*" % (N_, N_), ("s2_uninformative_v1", "embedded_crops")),
    ]
    out = {}
    for pat, names in pats:
        m = re.search(pat, text)
        if m is None:
            raise AdapterError("the contract at %s does not state %s" % (contract_path, ", ".join(names)))
        for name, g in zip(names, m.groups()):
            out[name] = _int(g)
    return out


def stage_counts(step1_dir, census_v0):
    """The Step 1 quantities contract §3 states, re-derived from the Step 1
    summaries and census_v0 (names as in contract_numbers)."""
    s = _load_summaries(step1_dir)
    rows = _read_json(census_v0)
    ps, ad, ss = s["pool_summary"], s["admit_summary"], s["select_summary"]
    drop = collections.Counter()
    ne_split = collections.Counter()
    for v in ps["per_slug"].values():
        drop.update(v.get("dropped") or {})
        ne_split.update(v.get("near_eval_by_split") or {})
    other = [r for r in rows if r["label"] == C.CLASS_NAMES[OTHER]]
    tg = [r for r in rows if r["label"] != C.CLASS_NAMES[OTHER]]
    v1 = collections.Counter()
    v1c = collections.Counter()
    for r in other:
        st = name_status_v1(r["src_name"])
        v1[st] += int(r["n"])
        v1c[st] += int(r["verdict"].get(V.CONFLICT, 0))
    s8_other = sum(int(r["pred_confident"].get(C.CLASS_NAMES[OTHER], 0)) for r in tg)
    s8_conf = sum(int(r["verdict"].get(V.CONFLICT, 0)) for r in tg)
    target_names = C.CLASS_NAMES[:NT]
    admitted_t = sum(int(ad["admitted_boxes_per_class"].get(n, 0)) for n in target_names)
    verified = int(ad["boxes"][V.VERIFIED])
    inc_src = (ss.get("sources") or {}).get("increment_pool") or {}
    ver_by = {sl: int(((d or {}).get("boxes") or {}).get(V.VERIFIED, 0)) for sl, d in ad["per_slug"].items()}
    ev = [sl for sl in inc_src if ver_by.get(sl, 0) >= MIN_EVIDENCE]
    sz = ss["sizes"]
    base_t = sum(int(ss["boxes"]["base_B"].get(n, 0)) - int(ss["boxes"]["train_core"].get(n, 0))
                 for n in target_names)
    embedded = sum(int(r["n"]) for r in rows)
    pool_t = sum(int(ps["boxes_per_class"].get(n, 0)) for n in target_names)
    kinds = collections.Counter()
    for r in rows:
        kinds[v1_kind(name_status_v1(r["src_name"]))] += int(r["n"])
    def slug_drop(prefix, reason):
        hits = [v for sl, v in ps["per_slug"].items() if sl.startswith(prefix)]
        if len(hits) != 1:
            raise AdapterError("%d slugs start with %r" % (len(hits), prefix))
        return int((hits[0].get("dropped") or {}).get(reason, 0))
    to_palmer = sum(int(r["pred_confident"].get("PalmerAmaranth", 0)) for r in tg
                    if r["label"] != "PalmerAmaranth")
    return {
        "s7b_crop": sum(int(r["n"]) for r in rows if SP.name_key(r["src_name"]) == "crop"),
        "s2_no_boxes": int(drop["no_boxes"]),
        "s2_no_boxes_rf_tuf": slug_drop("rf_tuf__", "no_boxes"),
        "s2_no_boxes_csgo": slug_drop("fvossel__csgo", "no_boxes"),
        "s8_conflict_target_palmer": to_palmer,
        "s10_verified": verified,
        "s12_pool_images": sum(int(n) for n in inc_src.values()),
        "funnel_verified": verified,
        "funnel_admitted": admitted_t,
        "s1_slugs": int(ps["slugs_total"]),
        "harvest_sources": int(ps["slugs_used"]),
        "s3_exact_dup": int(drop["exact_dup"]),
        "s4_near_eval": int(drop["near_eval"]),
        "s4_test": int(ne_split["test"]), "s4_dev": int(ne_split["dev"]), "s4_ood23": int(ne_split["ood23"]),
        "s4_ood22": int(ne_split["ood22"]), "s4_imageweeds": int(ne_split["imageweeds"]),
        "s5_cwd12_copy": int(drop["cwd12_copy"]),
        "s6_small": int(ad["boxes_small_not_embedded"]),
        "s7_other_embedded": sum(int(r["n"]) for r in other),
        "s7_named_other": v1[V1_NAMED], "s7_generic": v1["generic"], "s7_no_name": v1["no_name"],
        "s7_numeric": v1["numeric"], "s7_target_related": v1["target_related"],
        "s8_in": sum(int(r["n"]) for r in tg),
        "s8_verified": sum(int(r["verdict"].get(V.VERIFIED, 0)) for r in tg),
        "s8_unknown": sum(int(r["verdict"].get(V.UNKNOWN, 0)) for r in tg),
        "s8_conflict": s8_conf, "s8_conflict_other": s8_other, "s8_conflict_target": s8_conf - s8_other,
        "s9_conflict": sum(int(r["verdict"].get(V.CONFLICT, 0)) for r in other),
        "s9_no_name": v1c["no_name"], "s9_numeric": v1c["numeric"], "s9_named_other": v1c[V1_NAMED],
        "s9_target_related": v1c["target_related"], "s9_generic": v1c["generic"],
        "s10_img_conflict": int(ad["images"].get(V.CONFLICT, 0)),
        "s10_img_unknown": int(ad["images"].get(V.UNKNOWN, 0)),
        "s10_admitted_target": admitted_t, "s10_lost": verified - admitted_t,
        "s11_admitted_images": int(sz["verified"]), "s11_no_evidence": int(sz["no_evidence"]),
        "s11_below_gate": int(sz["below_gate"]), "s11_no_feature": int(sz["no_feature"]),
        "s11_other_budget": int(sz["dropped_other_budget"]), "s11_selected": int(sz["selected"]),
        "s11_increment_pool": int(sz["increment_pool"]),
        "s12_not_evidenced": len(inc_src) - len(ev), "s12_sources": len(inc_src),
        "s12_images_out": sum(int(n) for sl, n in inc_src.items() if sl not in ev),
        "pool_target_boxes": pool_t, "embedded_target_boxes": sum(int(r["n"]) for r in tg),
        "base_target_boxes": base_t,
        "s1_skipped": sum(int(v) for v in ps["skipped_counts"].values()),
        "s1_never_train": int(ps["skipped_counts"].get("never_train", 0)),
        "s1_quarantined": int(ps["skipped_counts"].get("quarantined", 0)),
        "s1_unknown_layout": int(ps["skipped_counts"].get("unknown_layout", 0)),
        "s1_user_flag": int(ps["skipped_counts"].get("user_flag_garbage", 0)),
        "harvest_images": int(ps["images"]), "harvest_boxes": int(ps["boxes"]),
        "s2_uninformative_v1": kinds["none"] + kinds["numeric"] + kinds["generic"],
        "embedded_crops": embedded,
    }


# ============================================================= disjointness keys
class _Keys(object):
    """near_dup3 groups over train_core, the copies, the pool and the KT7 photos
    (select.dup_groups at 3 bits; group id n:<first member key>), provenance
    and lab of every unit, built once per process."""

    def __init__(self, domain, funnel_dir=None):
        from ...inc import select as SEL
        self.domain = domain
        step1 = Path(V.STEP1)
        self.meta = {}
        for m in _read_jsonl(V.POOL_META):
            self.meta[m["key"]] = m
        # the pool image of each key: provenance by capture stem reads its file name,
        # exactly as the census ledger rows do
        self.image = {r["key"]: r["image"] for r in _read_jsonl(V.POOL)} if V.POOL.exists() else {}
        self.copies = {r["key"]: r for r in _read_jsonl(V.COPIES)} if V.COPIES.exists() else {}
        # a pre-pool image that is a reference copy (its d:/p: id names slug|rel) shares
        # its twin's provenance: (source, image path tail as rel) -> train_core key
        self.copy_twin = {}
        for r in self.copies.values():
            parts = str(r["image"]).replace(os.sep, "/").split("/")
            for n in (2, 3):
                k = (r["source"], "/".join(parts[-n:]))
                self.copy_twin[k] = r["train_core_key"] if k not in self.copy_twin else None
        core = C.read_manifest(C.manifest_path("train_core"))
        self.core = {r["key"]: r for r in core}
        probe = _read_json(step1 / "cache" / "train_core_probe.json") if (step1 / "cache" / "train_core_probe.json").exists() else {}
        self.kt7 = _kt7_items(domain, funnel_dir)
        hashes, keys = [], []
        for k, r in sorted(self.core.items()):
            ent = probe.get(r["image"])
            if ent and ent[2] is not None:
                keys.append(k)
                hashes.append(int(ent[2]))
        for k, r in sorted(self.copies.items()):
            if r.get("dhash") is not None:
                keys.append(k)
                hashes.append(int(r["dhash"]))
        for k, m in sorted(self.meta.items()):
            if m.get("dhash") is not None:
                keys.append(k)
                hashes.append(int(m["dhash"]))
        for it in self.kt7:
            if it.get("dhash") is not None:
                keys.append(it["id"])
                hashes.append(int(it["dhash"]))
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        keys = [keys[i] for i in order]
        hashes = [hashes[i] for i in order]
        self.group = {}
        if keys:
            g = SEL.dup_groups(hashes, bits=3)
            first = {}
            for i, gi in enumerate(g.tolist()):
                first.setdefault(gi, keys[i])
                self.group[keys[i]] = "n:%s" % first[gi]
        from ...near_dup import NearHashIndex
        self.index = NearHashIndex()
        for k, h in zip(keys, hashes):
            self.index.add(int(h), k, max_bits=3)
        self.stem_rx = {s: re.compile(rx) for s, rx in
                        (domain.raw["sources"].get("capture_stem_regex") or {}).items()}

    def near(self, key, h=None):
        g = self.group.get(key)
        if g is not None:
            return g
        if h is not None:
            m = self.index.find(int(h))
            if m is not None:
                return self.group[m[0]]
        return "n:%s" % key

    def provenance_prepool(self, slug, rel):
        """A pre-pool image's provenance (runner §1.3): its twin's for a reference
        copy (prov:<train_core key>), else its own (prov:d:<slug>|<rel>)."""
        k = (slug, str(rel))
        if k in self.copy_twin:
            if self.copy_twin[k] is None:
                raise AdapterError("d:%s|%s names two reference copies; its twin is ambiguous" % (slug, rel))
            return "prov:%s" % self.copy_twin[k]
        return "prov:d:%s|%s" % (slug, rel)

    def provenance_pool(self, key, source, image):
        rx = self.stem_rx.get(source)
        if rx is not None:
            stem = os.path.splitext(os.path.basename(image or ""))[0]
            m = rx.search(stem)
            if m:
                return "prov:%s|%s" % (source, m.group(1) if m.groups() else m.group(0))
        return "prov:%s" % key


_KEYS_CACHE = {}


def _keys(domain, funnel_dir=None):
    k = (domain.sha256, str(funnel_dir), V.POOL_META.stat().st_mtime_ns if V.POOL_META.exists() else 0)
    obj = _KEYS_CACHE.get(k)
    if obj is None:
        _KEYS_CACHE.clear()
        obj = _KEYS_CACHE[k] = _Keys(domain, funnel_dir)
    return obj


def _kt7_items(domain, funnel_dir):
    """The KT7 rows (kt7/kt7_items.jsonl) with each photo's dHash, when fetched."""
    fd = _funnel_dir(funnel_dir)
    p = fd / "kt7" / "kt7_items.jsonl"
    if not p.exists():
        return []
    rows = _read_jsonl(p)
    crop_of = {}
    cp = fd / "kt7" / "crops_kt7.csv"
    if cp.exists():
        with open(cp, newline="") as fh:
            rd = csv.reader(fh)
            if tuple(next(rd)) != V.CROP_FIELDS:
                raise AdapterError("%s: not verify's CROP_FIELDS" % cp)
            for rec in rd:
                crop_of[rec[2]] = int(rec[0])
    for r in rows:
        photo = fd / "kt7" / "photos" / r.get("file", "")
        r["dhash"] = C.dhash(photo) if r.get("file") and photo.exists() else None
        r["crop_id"] = crop_of.get(r["id"])
    return rows


def unit_keys(unit_ids, domain=None, funnel_dir=None):
    """{unit_id: {"source", "lab", "near_dup3", "provenance"}} for the unit ids of
    runner §1.3. Units spanning many images (classes, clusters, sources) have
    near_dup3 and provenance None."""
    domain = D.load(domain or "weed")
    K = _keys(domain, funnel_dir)
    out = {}
    for uid in unit_ids:
        kind, _, rest = uid.partition(":")
        if kind in ("b", "i"):
            key = rest.split("#")[0] if kind == "b" else rest
            m = K.meta.get(key)
            if m is None:
                raise AdapterError("unit %s: no pool image %s" % (uid, key))
            if m["source"] in K.stem_rx and key not in K.image:
                raise AdapterError("unit %s: pool.jsonl lists no image for %s (its capture stem is needed)"
                                   % (uid, key))
            out[uid] = {"source": m["source"], "lab": domain.lab_of(m["source"]),
                        "near_dup3": K.near(key),
                        "provenance": K.provenance_pool(key, m["source"], K.image.get(key))}
        elif kind == "t1":
            key = rest.split("#")[0]
            if key not in K.core:
                raise AdapterError("unit %s: no train_core image %s" % (uid, key))
            src = domain.reference_source
            out[uid] = {"source": src, "lab": domain.lab_of(src), "near_dup3": K.near(key),
                        "provenance": "prov:%s" % key}
        elif kind == "t2":
            key = rest.split("#")[0]
            r = K.copies.get(key)
            if r is None:
                raise AdapterError("unit %s: no copy %s" % (uid, key))
            out[uid] = {"source": r["source"], "lab": domain.lab_of(r["source"]), "near_dup3": K.near(key),
                        "provenance": "prov:%s" % r["train_core_key"]}
        elif kind == "t7":
            obs = rest.split("/")[0]
            out[uid] = {"source": "kt7", "lab": domain.lab_of("kt7"), "near_dup3": K.near(uid),
                        "provenance": "prov:kt7|%s" % obs}
        elif kind in ("d", "p"):
            parts = rest.split("|")
            slug, rel = parts[0], parts[1]
            h = _cached_dhash(slug, rel)
            out[uid] = {"source": slug, "lab": domain.lab_of(slug), "near_dup3": K.near("d:%s|%s" % (slug, rel), h),
                        "provenance": K.provenance_prepool(slug, rel)}
        elif kind in ("c", "k", "s"):
            slug = rest.split("|")[0]
            out[uid] = {"source": slug, "lab": domain.lab_of(slug), "near_dup3": None, "provenance": None}
        else:
            raise AdapterError("unit id %r has no known kind" % uid)
    return out


_DHASH_CACHES = {}


def _slug_cache(slug):
    c = _DHASH_CACHES.get(slug)
    if c is None:
        p = Path(V.CACHE_DIR) / "dhash" / ("%s.json" % V._sanitise(slug))
        try:
            with open(p) as fh:
                c = json.load(fh)
        except (OSError, ValueError):
            c = {}
        _DHASH_CACHES[slug] = c
    return c


def _cached_dhash(slug, rel):
    ent = _slug_cache(slug).get(rel)
    return int(ent[2]) if ent and ent[2] is not None else None


# ============================================================= simple readers
def pool_rows(sources=None):
    rows = _read_jsonl(V.POOL)
    if sources is not None:
        want = set(sources)
        rows = [r for r in rows if r["source"] in want]
    return rows


def base_rows():
    return _read_jsonl(Path(V.STEP1) / "base_B.jsonl")


def increment_rows(exp):
    """{step manifest stem: rows} of an experiment's increment manifests
    (INC_DIR/<exp>/manifests/*.jsonl)."""
    d = Path(exp) if os.sep in str(exp) else C.INC_DIR / str(exp)
    md = d / "manifests"
    if not md.is_dir():
        raise AdapterError("%s has no manifests/ directory" % d)
    return {p.stem: _read_jsonl(p) for p in sorted(md.iterdir()) if p.suffix == ".jsonl"}


# A capture session in the INC manifests' "session" field: the date and the
# camera of a capture, "<YYYYMMDD>_<camera>[_<operator>]" (the reference
# dataset's and its lab's own captures, e.g. dev's sessions); any other value
# (empty, a stem group) names no capture. The same pattern as
# inc2.embed_calibration's CAPTURE_RE and DATE_RE.
CAPTURE_SESSION_RE = re.compile(r"^[0-9]{8}_[A-Za-z0-9]")
CAPTURE_DATE_RE = re.compile(r"^([0-9]{8})_")


def capture_session(row):
    """(capture session, capture date) of a manifest row, each None when the
    row names no capture: the engine's session key for the copy detector's
    per-image calibration (leak detector version 2, contract §14 A2), which
    leaves out a negative's own session and date. An optional adapter
    function: the engine names neither the field nor the pattern."""
    s = str((row or {}).get("session") or "")
    if not CAPTURE_SESSION_RE.match(s):
        return None, None
    m = CAPTURE_DATE_RE.match(s)
    return s, (m.group(1) if m else None)


def eval_rows():
    """{split: manifest rows} for every evaluation split (C.EVAL_SPLITS)."""
    out = {}
    for split in C.EVAL_SPLITS:
        p = C.manifest_path(split)
        if not p.exists():
            raise AdapterError("evaluation manifest %s is missing" % p)
        out[split] = _read_jsonl(p)
    return out


def never_train_guard():
    try:
        return C.NeverTrainGuard.load()
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise AdapterError("never-train index: %s" % e)


def text_encoder():
    """The zero-shot text tower of the Step 1 embedder (inc/relevance.py)."""
    from ...inc import relevance as R
    return R.BioclipTextEncoder()


def bioclip_embedder():
    return V.BioclipEmbedder()


def step1_features():
    """(X float16 [crops, dim], info): the Step 1 crop embeddings, complete and current."""
    crops = crop_table()
    try:
        return V.load_embeddings(crops)
    except V.VerifyError as e:
        raise StaleInput(str(e))


def j1_scores(X, verifier=None):
    """(P [n, 13], cos [n, 13]) of the fitted Step 1 probe on raw features X
    (NaN rows stay NaN), in chunks."""
    ver = verifier or V.Verifier.load()
    X = np.asarray(X)
    n = len(X)
    P = np.full((n, C.NC), np.nan)
    COS = np.full((n, C.NC), np.nan)
    for a in range(0, n, CHUNK):
        Xa = np.asarray(X[a:a + CHUNK], dtype=np.float32)
        ok = np.isfinite(Xa).all(axis=1)
        if ok.any():
            Pa, Ca = ver.scores(V._norm(Xa[ok]))
            P[a + np.flatnonzero(ok)] = Pa
            COS[a + np.flatnonzero(ok)] = Ca
    return P, COS


def label_rows(keys):
    """{pool key: [(cls, cx, cy, w, h)]} from the step1 label files, each
    checked against its label_sha256 in pool.jsonl."""
    want = set(keys)
    out = {}
    for r in _read_jsonl(V.POOL):
        if r["key"] in want:
            if C.sha256_file(r["label"]) != r["label_sha256"]:
                raise StaleInput("step1 label %s does not hash to its label_sha256" % r["label"])
            out[r["key"]] = C.read_yolo(r["label"])
    missing = sorted(want - set(out))
    if missing:
        raise AdapterError("%d key(s) are not pool images, e.g. %s" % (len(missing), missing[:3]))
    return out


def guard_pairs(out_path, ledger_path=None, domain=None):
    """guard_pairs_v1.csv from ledger.jsonl's pre-pool rows: every near_eval and
    cwd12_copy pair and every exact_dup twin, with both sides' sha256 and the
    pool side's disjointness keys (lab, near_dup3, provenance: a reference
    copy carries its twin's provenance, runner §1.3), which the G5 frame reads."""
    out_path = Path(out_path)
    ledger_path = Path(ledger_path) if ledger_path else out_path.parent / "ledger.jsonl"
    dom = D.load(domain or "weed")
    K = _keys(dom, out_path.parent)
    evals = {}
    # the evaluation manifests the never-train index was built from: a missing one
    # refuses (eval_rows), never a pair without its evaluation side
    er = eval_rows()
    for split, rows in er.items():
        for r in rows:
            evals[(split, r["key"])] = r
            evals[(split, r["image"])] = r
    core = {r["key"]: r for r in C.read_manifest(C.manifest_path("train_core"))}
    pool = {r["key"]: r for r in _read_jsonl(V.POOL)}
    rows = []
    for u in L.iter_units(ledger_path, unit="image"):
        slug, rel = u["source"], u["rel"]
        ent = _slug_cache(slug).get(rel) or [None, None, None, None, 0, 0]
        sha_a = ent[3]
        near = u.get("near")
        keys = {"lab": dom.lab_of(slug), "near_dup3": K.near("d:%s|%s" % (slug, rel), u["dhash"]),
                "provenance": K.provenance_prepool(slug, rel)}
        if u["path"].get("S3") == "exact_dup":
            kept = pool.get(u["twin_of"])
            sha_b = kept["sha256"] if kept else None
            rows.append({"pair_id": "p:%s|%s|dup|%s" % (slug, rel, u["twin_of"]), "kind": "exact_dup",
                         "source": slug, "rel": rel, "dhash": u["dhash"], "split": "dup",
                         "eval_key": u["twin_of"], "eval_image": kept["image"] if kept else "",
                         "bits": 0, "sha_a": sha_a, "sha_b": sha_b,
                         "sha_differs": int(bool(sha_a and sha_b and sha_a != sha_b)), **keys})
        elif near is not None:
            kind = "near_eval" if u["path"].get("S4") == "near_eval" else "cwd12_copy"
            if kind == "near_eval":
                er_ = evals.get((near["split"], near["eval"]))
                if er_ is None:
                    raise AdapterError("%s|%s: its near_eval hit %s/%s is in no evaluation manifest (the "
                                       "never-train index and the manifests disagree)"
                                       % (slug, rel, near["split"], near["eval"]))
                ekey, eimg, sha_b = er_["key"], er_["image"], er_["sha256"]
            else:
                cr = core.get(near["eval"])
                if cr is None:
                    raise AdapterError("%s|%s: its train_core twin %s is not in the train_core manifest"
                                       % (slug, rel, near["eval"]))
                ekey, eimg, sha_b = near["eval"], cr["image"], cr["sha256"]
            rows.append({"pair_id": "p:%s|%s|%s|%s" % (slug, rel, near["split"], ekey), "kind": kind,
                         "source": slug, "rel": rel, "dhash": u["dhash"], "split": near["split"],
                         "eval_key": ekey, "eval_image": eimg, "bits": int(near["bits"]),
                         "sha_a": sha_a, "sha_b": sha_b,
                         "sha_differs": int(bool(sha_a and sha_b and sha_a != sha_b)), **keys})
    rows.sort(key=lambda r: r["pair_id"])
    header_ = ("pair_id", "kind", "source", "rel", "dhash", "split", "eval_key", "eval_image", "bits",
               "sha_a", "sha_b", "sha_differs", "lab", "near_dup3", "provenance")
    sha = write_csv_atomic(out_path, header_, rows)
    return {"path": str(out_path), "sha256": sha, "rows": len(rows),
            "by_kind": dict(collections.Counter(r["kind"] for r in rows))}


# ============================================================= census (F3)
def _load_pair(prereg, domain):
    pre = prereg if hasattr(prereg, "core_sha256") else D.load_prereg(prereg)
    dom = D.load(domain if domain is not None else pre.domain_name)
    D.check_prereg_domain(pre, dom)
    return pre, dom


def _census_inputs(step1, taxonomy_cache, known_items):
    """{logical name: record} of every file the census reads."""
    rec = {}
    names = {"pool": V.POOL, "pool_meta": V.POOL_META, "pool_summary": V.POOL_SUMMARY, "copies": V.COPIES,
             "crops": V.CROPS, "crops_skipped": V.CROPS_SKIPPED, "crops_info": V.CROPS_INFO,
             "probe": V.VERIFIER_DIR / "probe.joblib", "verifier_npz": V.VERIFIER_DIR / "verifier.npz",
             "thresholds": V.VERIFIER_DIR / "thresholds.json", "fit_info": V.VERIFIER_DIR / "fit_info.json",
             "calibration": V.CALIBRATION, "verified": V.VERIFIED_MANIFEST, "conflicts": V.CONFLICTS,
             "admit_summary": V.ADMIT_SUMMARY, "pool_verdicts": V.POOL_VERDICTS,
             "select_summary": step1 / "select_summary.json", "base_B": step1 / "base_B.jsonl",
             "increment_pool": step1 / "increment_pool.jsonl", "select_clusters": step1 / "select_clusters.csv",
             "train_core_probe": step1 / "cache" / "train_core_probe.json",
             "registry": V.REGISTRY, "never_train_index": C.NEVER_TRAIN_INDEX,
             "train_core_manifest": C.manifest_path("train_core"), "taxonomy_cache": Path(taxonomy_cache)}
    if V.FLAGS.exists():
        names["flags"] = V.FLAGS
    if (step1 / "base_selected.jsonl").exists():
        names["base_selected"] = step1 / "base_selected.jsonl"
    if known_items is not None:
        names["known_items"] = Path(known_items)
    for k, p in names.items():
        if not Path(p).is_file():
            raise AdapterError("census input %s is missing (%s)" % (k, p))
        rec[k] = file_record(p)
    emb = Path(V.EMB_DIR)
    for name in sorted(os.listdir(emb)) if emb.is_dir() else []:
        if re.fullmatch(r"emb_s\d{3}_of_\d{3}\.npz", name):
            rec["emb:%s" % name] = file_record(emb / name)
    dd = Path(V.CACHE_DIR) / "dhash"
    for name in sorted(os.listdir(dd)) if dd.is_dir() else []:
        if name.endswith(".json"):
            rec["dhash_cache:%s" % name[:-5]] = file_record(dd / name)
    return rec


def _j1_pool(ver, X, pidx):
    n = len(pidx)
    P = np.full((n, C.NC), np.nan)
    COS = np.full((n, C.NC), np.nan)
    for a in range(0, n, CHUNK):
        ids = pidx[a:a + CHUNK]
        Xa = np.asarray(X[ids], dtype=np.float32)
        ok = np.isfinite(Xa).all(axis=1)
        if ok.any():
            Pa, Ca = ver.scores(V._norm(Xa[ok]))
            P[a + np.flatnonzero(ok)] = Pa
            COS[a + np.flatnonzero(ok)] = Ca
    return P, COS


def _rederive(crops, ver, X):
    """Per pool crop (in crops.where("pool") order): verdict, argmax, p, cos and
    the full P / cos vectors, as verify admit computes them (the OtherPlant
    training crops by their out-of-fold arrays), checked against
    pool_verdicts.npz."""
    pidx = crops.where("pool").astype(np.int64)
    labels = crops.label[pidx]
    P, COS = _j1_pool(ver, X, pidx)
    n = len(pidx)
    xfin = np.isfinite(P).all(axis=1)          # the probe answers every finite feature row
    v = np.full(n, V.FAILED, dtype=object)
    j = np.full(n, -1, dtype=np.int64)
    pj = np.full(n, np.nan)
    cj = np.full(n, np.nan)
    fin = np.isfinite(P).all(axis=1)
    if fin.any():
        v[fin], j[fin], pj[fin], cj[fin] = V.verdicts(labels[fin], P[fin], COS[fin], ver.tau_p, ver.sigma)
    pos = {int(c): k for k, c in enumerate(pidx)}
    other_ids = np.asarray(ver.arrays.get("other_ids", np.zeros(0, np.int64)), dtype=np.int64)
    oof = np.zeros(n, dtype=bool)
    if len(other_ids):
        oP, oC, oU = (ver.arrays.get("other_oof_P"), ver.arrays.get("other_oof_cos"),
                      ver.arrays.get("other_oof_usable"))
        if oP is None or oC is None or oU is None or len(oP) != len(other_ids):
            raise AdapterError("the verifier has no out-of-fold arrays for its OtherPlant training crops")
        rows = [pos.get(int(c)) for c in other_ids]
        if any(r is None for r in rows):
            raise AdapterError("an OtherPlant training crop is not a pool crop")
        rows = np.array(rows, dtype=np.int64)
        vo, jo, pjo, cjo = V.verdicts(np.full(len(rows), OTHER), oP, oC, ver.tau_p, ver.sigma)
        vo[~np.asarray(oU).astype(bool)] = V.UNKNOWN
        v[rows], j[rows], pj[rows], cj[rows] = vo, jo, pjo, cjo
        P[rows] = oP
        COS[rows] = oC
        oof[rows] = True
    with np.load(V.POOL_VERDICTS, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        cid, vcode, pred, pnpz = d["crop_id"], d["verdict"], d["pred"], d["p"]
    if meta.get("crops_sha256") != crops.sha:
        raise StaleInput("pool_verdicts.npz was made from another crops.csv")
    if list(meta.get("verdict_codes") or []) != list(V.VERDICT_CODES):
        raise AdapterError("pool_verdicts.npz verdict codes %s are not verify's %s"
                           % (meta.get("verdict_codes"), V.VERDICT_CODES))
    if not np.array_equal(np.asarray(cid, dtype=np.int64), pidx):
        raise AdapterError("pool_verdicts.npz crop ids are not the pool crops of crops.csv in order")
    mine = np.array([V.VERDICT_CODES.index(x) for x in v], dtype=np.int8)
    bad_v = int((mine != np.asarray(vcode)).sum())
    bad_j = int((j != np.asarray(pred, dtype=np.int64)).sum())
    if bad_v or bad_j:
        ex = np.flatnonzero((mine != np.asarray(vcode)) | (j != np.asarray(pred, dtype=np.int64)))[:5]
        raise AdapterError("re-derived verdicts disagree with pool_verdicts.npz: %d verdict(s), %d argmax(es) "
                           "differ, e.g. crop ids %s" % (bad_v, bad_j, pidx[ex].tolist()))
    both = np.isfinite(pj) & np.isfinite(np.asarray(pnpz, dtype=np.float64))
    dp = float(np.max(np.abs(pj[both] - np.asarray(pnpz, dtype=np.float64)[both]))) if both.any() else 0.0
    finite = np.isfinite(P).all(axis=1)
    Pz = np.where(np.isfinite(P), P, -1.0)
    top2 = np.argsort(-Pz, axis=1, kind="stable")[:, :2]
    second = np.where(finite, top2[:, 1], -1)
    p_second = np.where(finite, P[np.arange(n), np.maximum(second, 0)], np.nan)
    p_tmax = np.where(finite, np.max(Pz[:, :NT], axis=1), np.nan)
    p_other = np.where(finite, P[:, OTHER], np.nan)
    return {"pidx": pidx, "labels": labels, "P": P, "COS": COS, "v": v, "j": j, "pj": pj, "cj": cj,
            "second": second, "p_second": p_second, "p_tmax": p_tmax, "p_other": p_other, "oof": oof,
            "xfin": xfin,
            "check": {"crops": n, "verdict_mismatch": bad_v, "argmax_mismatch": bad_j,
                      "max_abs_p_diff": round(dp, 9), "oof_crops": int(oof.sum())}}


def _fail_code(label, v, j, p, cos_j, tau, sigma):
    """Why a target-labelled box was not verified (runner §4.1; first rule that applies)."""
    if v == V.VERIFIED:
        return None
    if v == V.FAILED or j < 0:
        return "failed"
    conf = p >= tau[j] and (j == OTHER or cos_j >= sigma[j])
    if j == OTHER and p >= tau[OTHER]:
        return "argmax_other_confident"
    if j != label and j < OTHER and conf:
        return "argmax_target_confident"
    if j != label:
        return "argmax_wrong"
    if not p >= tau[label]:              # a class without a threshold (NaN) is never confident
        return "p_below_tau"
    return "cos_below_sigma"


def _blocker(label, verdict):
    if verdict == V.FAILED:
        return "failed"
    if label == OTHER:
        return "other_called_target" if verdict == V.CONFLICT else None
    return {V.CONFLICT: "species_conflict", V.UNKNOWN: "species_unknown",
            V.SMALL: "small_species_box"}.get(verdict)


def _r6(x):
    if x is None:
        return None
    x = float(x)
    return round(x, 6) if np.isfinite(x) else None


def _class_rows_from_joins(pool_summary):
    """{(slug, src_id): (name, joined_target)} for every class of every used slug."""
    out = {}
    for slug, st in sorted(pool_summary["per_slug"].items()):
        for sid, (name, inc) in (st.get("join") or {}).items():
            out[(slug, str(sid))] = ("" if name is None else str(name), inc != C.CLASS_NAMES[OTHER])
    return out


def _named_taxa(domain, resolver):
    """accepted name -> config taxon, for attractor and "not" species; accepted
    genus -> config taxon for genus-rank attractors."""
    sp, ge = {}, {}
    for a in domain.attractors:
        e = resolver.resolve(a["taxon"], scientific_only=True)
        acc = (e or {}).get("accepted") or a["taxon"]
        (ge if a["rank"] == "genus" else sp)[acc] = a["taxon"]
    for t in domain.targets:
        for nt in t.get("not", []) or []:
            e = resolver.resolve(nt, scientific_only=True)
            sp[(e or {}).get("accepted") or nt] = nt
    return sp, ge


def _card_taxon(domain, source, name):
    """The taxon a source's card gives a class, matched by class NAME (never by
    id: which id a card row means is what H1-pre and H3a test)."""
    res = (domain.raw["sources"].get("card_resolvers") or {}).get(source) or {}
    k = N.key(name)
    if not k:
        return None
    for row in (res.get("class_table") or {}).values():
        if isinstance(row, dict) and N.key(row.get("name")) == k:
            return row.get("taxon")
    return (res.get("names") or {}).get(k)


def kt5_taxa(domain, resolver, name_status):
    """{(source, src_id): attractor or "not" taxon} of the OtherPlant classes
    whose card taxon, or v2 resolution, is one (known-truth set KT5)."""
    sp, ge = _named_taxa(domain, resolver)
    out = {}
    for n in name_status["names"]:
        if n["status_v2"] == "target":
            continue
        tx = _card_taxon(domain, n["source"], n["name"])
        acc = None
        if tx:
            e = resolver.resolve(tx, scientific_only=True)
            acc = (e or {}).get("accepted") or tx
        elif n.get("via") in ("scientific", "override", "vernacular") and n.get("taxon"):
            acc = n["taxon"]
        if not acc:
            continue
        if acc in sp:
            out[(n["source"], n["src_id"])] = sp[acc]
        elif acc.split(" ")[0] in ge:
            out[(n["source"], n["src_id"])] = ge[acc.split(" ")[0]]
    return out


def _replay_prepool(domain, pool_rows_, meta, copies):
    """The images verify pool listed but did not keep (S2-S5), in verify's own
    order and by its own readers and hash cache. Returns (d rows, per-slug
    counts, h5a raw twins, kept images per slug, skipped slugs)."""
    from ... import mega_trainer as MT
    from ...near_dup import NearHashIndex
    registry = V._load_registry()
    flags = V._load_flags()
    guard = never_train_guard()
    core = sorted(C.read_manifest(C.manifest_path("train_core")), key=lambda r: r["key"])
    probe_path = Path(V.CORE_PROBE)
    core_cache = _read_json(probe_path) if probe_path.exists() else {}
    core_index = NearHashIndex()
    for r in core:
        ent = core_cache.get(r["image"])
        if ent and ent[2] is not None:
            core_index.add(int(ent[2]), r["key"], max_bits=V.CORE_COPY_BITS)
    calib_only = set(MT.CWD12_COPY_ID_MAPS)
    pool_by_image = {r["image"]: r["key"] for r in pool_rows_}
    pool_key_by_dhash = {}
    for m in meta.values():
        pool_key_by_dhash.setdefault(int(m["dhash"]), m["key"])
    drows, counts, twins, kept = [], collections.defaultdict(collections.Counter), [], collections.Counter()
    near_split = collections.defaultdict(collections.Counter)
    skipped = collections.defaultdict(list)
    for slug in sorted(registry):
        info = registry[slug]
        reason = V._skip_reason(slug, info, flags, MT)
        root = parts = None
        if reason is None:
            root = V._resolve_dir(info)
            if root is None:
                reason = "missing_dir"
        if reason is None:
            parts = V._layout(root)
            if parts is None:
                reason = "unknown_layout"
        if reason is not None:
            skipped[reason].append(slug)
            continue
        to_inc, _names, wildcard = V.class_join(slug, info)
        cache = _slug_cache(slug)
        for split, idir, ldir in parts:
            labels = set(os.listdir(ldir))
            for name in sorted(os.listdir(idir)):
                if not V._is_image(name):
                    continue
                stem = os.path.splitext(name)[0]
                rel = "%s/images/%s" % (split, name) if split else "images/%s" % name
                image = str(Path(idir) / name)

                def drop(why, stage, near=None, twin=None, h=None, inc=None, src=None):
                    counts[slug][why] += 1
                    path = {"S2": "pass", "S3": "n/a", "S4": "n/a", "S5": "n/a"}
                    if stage == "S2":
                        path["S2"] = why
                    elif stage == "S4":
                        path["S4"] = "near_eval"
                    elif stage == "S5":
                        path.update(S4="pass", S5="cwd12_copy")
                    else:
                        path.update(S4="pass", S5="pass", S3="exact_dup")
                    drows.append({"id": "d:%s|%s" % (slug, rel), "unit": "image", "source": slug, "rel": rel,
                                  "dhash": None if h is None else int(h), "path": path, "near": near,
                                  "twin_of": twin, "failed_stages": [stage], "first_cause": stage,
                                  "sole_cause": stage})
                    if twin is not None:
                        twins.append({"id": "d:%s|%s" % (slug, rel), "source": slug, "twin_of": twin,
                                      "inc": inc, "src": src})

                if stem + ".txt" not in labels:
                    drop("no_label", "S2")
                    continue
                try:
                    src, _bad, _clipped = V.read_source_label(Path(ldir) / (stem + ".txt"))
                except (OSError, UnicodeDecodeError):
                    drop("unreadable_label", "S2")
                    continue
                if not src:
                    drop("no_boxes", "S2")
                    continue
                if wildcard:
                    inc = [OTHER] * len(src)
                else:
                    inc = [to_inc.get(b[0]) for b in src]
                    if any(c is None for c in inc):
                        drop("unmapped_class", "S2")
                        continue
                ent = cache.get(rel)
                h = int(ent[2]) if ent and ent[2] is not None else None
                if h is None or not ent[4] or not ent[5]:
                    drop("unhashable", "S2", h=h)
                    continue
                try:
                    stt = os.stat(image)
                except OSError:
                    raise StaleInput("%s is gone since verify pool listed it" % image)
                if stt.st_size != ent[0] or stt.st_mtime_ns != ent[1]:
                    raise StaleInput("%s changed since verify pool hashed it" % image)
                hits, _ = guard.check([image], hash_fn=lambda _p, _h=h: _h)
                if hits:
                    near_split[slug][hits[0][1]] += 1
                    drop("near_eval", "S4", near={"split": hits[0][1], "eval": str(hits[0][2]),
                                                  "bits": int(hits[0][3])}, h=h)
                    continue
                m = core_index.find(h)
                if m is None and slug in calib_only:
                    drop("calibration_only_not_train_core", "S2", h=h)
                    continue
                if m is not None:
                    drop("cwd12_copy", "S5", near={"split": "train_core", "eval": m[0], "bits": int(m[1])}, h=h)
                    continue
                if image in pool_by_image:
                    kept[slug] += 1
                    continue
                twin = pool_key_by_dhash.get(h)
                if twin is None:
                    raise AdapterError("%s|%s: verify kept neither it nor an image with its dHash; the pool was "
                                       "not made from this registry and hash cache" % (slug, rel))
                drop("exact_dup", "S3", twin=twin, h=h, inc=inc,
                     src=[int(b[0]) if not wildcard else "*" for b in src])
    drows.sort(key=lambda r: r["id"])
    return drows, counts, twins, kept, near_split, skipped


def census(prereg, domain, out_dir, taxonomy_cache, known_items=None, step1_dir=None, force=False,
           testing=False):
    """F3 (lever L10 census): census_v1.json, ledger.jsonl, funnel_ledger.json
    (derivation census), name_status_v2.json and guard_pairs_v1.csv under
    out_dir. Returns the census record. See the module docstring."""
    t0 = time.time()
    pre, dom = _load_pair(prereg, domain)
    _check_domain_stages(dom)
    step1 = _step1(step1_dir)
    out_dir = _funnel_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    census_path = out_dir / "census_v1.json"
    tax_path = Path(taxonomy_cache)
    if not tax_path.is_file():
        raise TaxonomyError("no taxonomy cache at %s: run fetch --what taxonomy (lever L12)" % tax_path)
    inputs = _census_inputs(step1, tax_path, known_items)
    code = {"adapter": _self_module(), "names": N, "taxonomy": T, "ledger": L}
    if census_path.exists():
        old = read_json(census_path)
        same = (old.get("inputs") == inputs and (old.get("domain_config") or {}).get("sha256") == dom.sha256
                and (old.get("prereg") or {}).get("core_sha256") == pre.core_sha256
                and old.get("code") == header("census", dom, pre, {}, modules=tuple(code.values()))["code"])
        if same and (old.get("reconciliation") or {}).get("ok"):
            log("census_v1.json is current (same inputs); nothing to do")
            return old
    if pre.sample_lock is not None:
        # the lock fixes name_status_v2.json (and the frames drawn from the
        # census): no census run may rewrite it, even when census_v1.json is gone
        raise SampleLocked("the sample is locked (%s); the census outputs are not rewritten"
                           % pre.sample_lock.get("id"))
    if census_path.exists() and not force:
        raise AdapterError("%s exists and was made from other inputs; rerun with --force" % census_path)
    # --- the taxonomy cache and name status v2 need no Step 1 pixels: refuse early
    cache = T.load_cache(tax_path, dom)
    resolver = T.Resolver(cache, dom)
    ps = _read_json(V.POOL_SUMMARY)
    joins = _class_rows_from_joins(ps)
    resolver.check_complete([nm for (nm, _t) in joins.values()])
    # --- Step 1: crops, verifier, embeddings, verdicts
    crops = crop_table()
    try:
        ver, X, emb = V._load_fitted(crops)
    except V.VerifyError as e:
        raise StaleInput(str(e))
    R = _rederive(crops, ver, X)
    log("re-derived %d pool verdicts: %s" % (len(R["pidx"]), R["check"]))
    tau, sigma = np.asarray(ver.tau_p, dtype=np.float64), np.asarray(ver.sigma, dtype=np.float64)
    pool_list = C.read_manifest(V.POOL)
    pool = {r["key"]: r for r in pool_list}
    meta = {m["key"]: m for m in _read_jsonl(V.POOL_META)}
    if set(pool) != set(meta):
        raise AdapterError("pool.jsonl and pool_meta.jsonl hold different keys")
    ad = _read_json(V.ADMIT_SUMMARY)
    ss = _read_json(step1 / "select_summary.json")
    cal = _read_json(V.CALIBRATION)
    copies = _read_jsonl(V.COPIES) if V.COPIES.exists() else []
    admitted_keys = {r["key"] for r in C.read_manifest(V.VERIFIED_MANIFEST)}
    base_sel = step1 / "base_selected.jsonl"
    core_keys = {r["key"] for r in C.read_manifest(C.manifest_path("train_core"))}
    base_keys = ({r["key"] for r in C.read_manifest(base_sel)} if base_sel.exists()
                 else {r["key"] for r in C.read_manifest(step1 / "base_B.jsonl")} - core_keys)
    inc_keys = {r["key"] for r in C.read_manifest(step1 / "increment_pool.jsonl")}
    from ...inc import select as SEL
    clusters = SEL.read_clusters(step1 / "select_clusters.csv")
    ver_by_slug = {s: int(((d or {}).get("boxes") or {}).get(V.VERIFIED, 0))
                   for s, d in (ad.get("per_slug") or {}).items()}
    evidenced = {s for s, n in ver_by_slug.items() if n >= MIN_EVIDENCE}
    pos_of = {(crops.key[i], int(crops.box[i])): n for n, i in enumerate(R["pidx"])}
    skipped_boxes = V.read_skipped("pool")
    # --- pass 1: box verdicts, image verdicts, class box counts
    box_v, img_v = {}, {}
    cls_boxes, cls_conf = collections.Counter(), collections.Counter()
    wildcard = {slug for slug, st in ps["per_slug"].items() if "*" in (st.get("join") or {})}
    for key in sorted(meta):
        m = meta[key]
        labels_ = [int(b[0]) for b in m["boxes"]]
        vs = []
        for b, lab in enumerate(labels_):
            n = pos_of.get((key, b))
            if n is not None:
                x = R["v"][n]
            else:
                why = skipped_boxes.get((key, b))
                if why is None:
                    raise AdapterError("pool box %s/%d has no crop row and was not left out by verify crops"
                                       % (key, b))
                x = V.SMALL if why == "small" else V.FAILED
            vs.append(x)
            sid = "*" if m["source"] in wildcard else str(m["src"][b][0])
            if n is not None:
                cls_boxes[(m["source"], sid)] += 1
                cls_conf[(m["source"], sid)] += int(x == V.CONFLICT)
        box_v[key] = vs
        iv = V.image_verdict(labels_, vs)
        img_v[key] = iv
        if (iv == V.ADMITTED) != (key in admitted_keys):
            raise AdapterError("image %s: the re-derived image verdict %s disagrees with verified.jsonl" % (key, iv))
    # --- name status v2
    cn = contract_numbers(pre.contract_path)
    ns_rows = []
    for (slug, sid), (nm, tgt) in sorted(joins.items()):
        ns_rows.append({"source": slug, "src_id": sid, "name": nm, "joined_target": tgt,
                        "status_v1": name_status_v1(nm), "boxes": int(cls_boxes.get((slug, sid), 0)),
                        "conflicts": int(cls_conf.get((slug, sid), 0))})
    extra = sorted(set(cls_boxes) - set(joins))
    if extra:
        raise AdapterError("pool boxes of classes pool_summary.json does not join: %s" % extra[:5])
    ns = N.build_name_status(ns_rows, dom, resolver, contract_check={
        "broweed_narweed_unresolvable_boxes": {"status": "unresolvable", "keys": ["broweed", "narweed"],
                                               "contract": cn["s7b_broweed_narweed"]},
        "crop_role_boxes": {"status": "role", "keys": ["crop"], "contract": cn["s7b_crop"]},
        "non_object_boxes": {"status": "non_object", "contract": cn["s7b_non_object"]},
        "state_boxes": {"status": "state", "contract": cn["s7b_state"]}})
    ns_status = N.status_table(ns)
    ns_by = {(n["source"], n["src_id"]): n for n in ns["names"]}
    ns_path = out_dir / "name_status_v2.json"
    ns_doc = header("name_status", dom, pre, {"taxonomy_cache": inputs["taxonomy_cache"],
                                               "pool_summary": inputs["pool_summary"],
                                               "pool_meta": inputs["pool_meta"], "crops": inputs["crops"],
                                               "pool_verdicts": inputs["pool_verdicts"]},
                    modules=(_self_module(), N, T), testing=testing)
    ns_doc.update(ns)
    ns_doc["taxonomy_cache"] = {"path": str(tax_path), "sha256": inputs["taxonomy_cache"]["sha256"]}
    ns_sha = write_json_atomic(ns_path, ns_doc)
    # --- known-truth membership (census time: KT4 and KT5; KT6 needs H3a)
    auth = set(dom.authoritative_sources())
    k5 = kt5_taxa(dom, resolver, ns)
    K = _keys(dom, out_dir)
    # --- pass 2: ledger rows (boxes), streamed
    led_path = out_dir / "ledger.jsonl"
    stage_order = [s["id"] for s in dom.stages]
    tmp = led_path.with_name(led_path.name + ".tmp")
    import hashlib
    hsh = hashlib.sha256()
    problems = []
    vetoes = []
    fail_rows = collections.defaultdict(collections.Counter)
    oac = collections.defaultdict(lambda: {"n": 0, "p": 0.0, "second": collections.Counter(), "p2": 0.0})
    per_class_vs = collections.defaultdict(lambda: {"verified": 0, "admitted": 0})
    totals_box = collections.Counter()
    admitted_target = 0
    small_total = 0
    n_box_rows = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        def emit(row):
            pr = L.validate_unit_row(row, stage_order, PATH_VOCAB)
            if pr and len(problems) < 20:
                problems.append("%s: %s" % (row["id"], "; ".join(pr)))
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            fh.write(line)
            hsh.update(line.encode("utf-8"))
        for key in sorted(meta):
            m = meta[key]
            src = m["source"]
            labels_ = [int(b[0]) for b in m["boxes"]]
            vs = box_v[key]
            iv = img_v[key]
            nd3 = K.near(key)
            prov = K.provenance_pool(key, src, pool[key]["image"])
            lab = dom.lab_of(src)
            in_base = key in base_keys
            if in_base and iv != V.ADMITTED:
                raise AdapterError("base image %s is not admitted" % key)
            if iv == V.ADMITTED:
                if in_base:
                    s11 = "base"
                else:
                    if key not in inc_keys:
                        raise AdapterError("admitted image %s is in neither base B nor the increment pool" % key)
                    st = (clusters.get(key) or {}).get("status")
                    if st is None:
                        raise AdapterError("admitted image %s has no select_clusters.csv row" % key)
                    s11 = "increment_pool" if st == "pool" else st
                    if st in SELECT_BASE:
                        raise AdapterError("image %s is selected in select_clusters.csv but not in base B" % key)
            else:
                s11 = "n/a"
            s12 = "n/a" if in_base else ("evidenced" if src in evidenced else "not_evidenced")
            for b in sorted(range(len(labels_)), key=lambda x: str(x)):
                lab_b = labels_[b]
                n = pos_of.get((key, b))
                x = vs[b]
                sid = "*" if src in wildcard else str(m["src"][b][0])
                nm = "" if src in wildcard else str(m["src"][b][1])
                ne = ns_by[(src, sid)]
                _cls, _cx, _cy, w, h = m["boxes"][b]
                small = x == V.SMALL
                no_size = n is None and x == V.FAILED
                failed = []
                if small or no_size:
                    failed.append("S6")
                small_total += int(small)
                is_t = lab_b < OTHER
                if n is not None:
                    totals_box[x] += 1
                s8 = (x if x in (V.VERIFIED, V.CONFLICT, V.UNKNOWN, V.FAILED) else "n/a") if is_t else "n/a"
                s9 = (x if x in (V.OTHER_OK, V.CONFLICT, V.FAILED) else "n/a") if not is_t else "n/a"
                if is_t and n is not None and x != V.VERIFIED:
                    failed.append("S8")
                if (not is_t) and n is not None and x in (V.CONFLICT, V.FAILED):
                    failed.append("S9")
                blockers = None
                if iv != V.ADMITTED:
                    bl = set()
                    for o, (lo, xo) in enumerate(zip(labels_, vs)):
                        if o != b:
                            c = _blocker(lo, xo)
                            if c:
                                bl.add(c)
                    blockers = [c for c in BLOCKERS if c in bl]
                    if blockers:
                        failed.append("S10")
                if s12 == "not_evidenced":
                    failed.append("S12")
                fail = None
                if is_t and n is not None:
                    fail = _fail_code(lab_b, x, int(R["j"][n]), float(R["pj"][n]) if np.isfinite(R["pj"][n]) else -1.0,
                                      float(R["cj"][n]) if np.isfinite(R["cj"][n]) else -1.0, tau, sigma)
                    if fail:
                        fail_rows[(src, sid, lab_b)][fail] += 1
                    if x == V.CONFLICT and int(R["j"][n]) == OTHER:
                        a = oac[(src, sid, lab_b)]
                        a["n"] += 1
                        a["p"] += float(R["pj"][n])
                        a["second"][C.CLASS_NAMES[int(R["second"][n])]] += 1
                        a["p2"] += float(R["p_second"][n])
                if is_t and x == V.VERIFIED:
                    per_class_vs[C.CLASS_NAMES[lab_b]]["verified"] += 1
                    if iv == V.ADMITTED:
                        per_class_vs[C.CLASS_NAMES[lab_b]]["admitted"] += 1
                    else:
                        vetoes.append({"key": key, "box": b, "label": C.CLASS_NAMES[lab_b], "blockers": blockers or []})
                if is_t and iv == V.ADMITTED:
                    admitted_target += 1
                kt = []
                if n is not None:
                    # known truth is judged as crops: a box that was never embedded
                    # (small, no size) has no crop and is in no set (KT4 = the
                    # contract's 4,639 embedded boxes)
                    if is_t and src in auth:
                        kt.append("KT4")
                    if (not is_t) and (src, sid) in k5:
                        kt.append("KT5")
                failed = [s for s in stage_order if s in set(failed)]
                row = {"id": "b:%s#%d" % (key, b), "unit": "box", "source": src, "key": key, "box": b,
                       "crop_id": int(R["pidx"][n]) if n is not None else None, "label": lab_b,
                       "src_id": sid, "src_name": nm, "wh_px": [int(round(w * m["W"])), int(round(h * m["H"]))],
                       "name_status_v1": ne["status_v1"], "name_status_v2": ne["status_v2"], "lab": lab,
                       "near_dup3": nd3, "provenance": prov,
                       "pred": int(R["j"][n]) if n is not None and R["j"][n] >= 0 else None,
                       "p": _r6(R["pj"][n]) if n is not None else None,
                       "cos": _r6(R["cj"][n]) if n is not None else None,
                       "p_target_max": _r6(R["p_tmax"][n]) if n is not None else None,
                       "p_other": _r6(R["p_other"][n]) if n is not None else None,
                       "second": int(R["second"][n]) if n is not None and R["second"][n] >= 0 else None,
                       "p_second": _r6(R["p_second"][n]) if n is not None else None,
                       "fail": fail, "blockers": blockers,
                       "path": {"S3": "kept", "S4": "pass", "S5": "pass",
                                "S6": "small" if small else ("no_size" if no_size else "embedded"),
                                "S7": "target" if is_t else "other", "S8": s8, "S9": s9, "S10": iv,
                                "S11": s11, "S12": s12},
                       "failed_stages": failed, "first_cause": failed[0] if failed else None,
                       "sole_cause": failed[0] if len(failed) == 1 else None, "kt": kt}
                emit(row)
                n_box_rows += 1
        # --- pre-pool images (S2-S5)
        drows, dcounts, twins, kept_by_slug, near_split, skipped_slugs = _replay_prepool(dom, pool_list, meta, copies)
        for r in drows:
            emit(r)
    if problems:
        os.unlink(tmp)
        raise AdapterError("ledger rows fail validation: %s" % " | ".join(problems[:5]))
    os.replace(tmp, led_path)
    led_sha = hsh.hexdigest()
    # --- census rows
    src_ids = []
    names_ = []
    sources_ = []
    for i in R["pidx"]:
        key = crops.key[i]
        b = int(crops.box[i])
        m = meta[key]
        s = m["source"]
        sources_.append(s)
        if s in wildcard:
            src_ids.append("*")
            names_.append("")
        else:
            src_ids.append(str(m["src"][b][0]))
            names_.append(str(m["src"][b][1]))
            if names_[-1] != crops.src_name[i]:
                raise AdapterError("crop %d: crops.csv names %r, pool_meta.jsonl %r" % (i, crops.src_name[i], names_[-1]))
    rows = aggregate_rows(sources_, src_ids, names_, R["labels"], R["v"], R["j"], R["pj"])
    for r in rows:
        lab_id = C.CLASS_NAMES.index(r["label"])
        ne = ns_by[(r["source"], r["src_id"])]
        r["name_status_v1"] = ne["status_v1"]
        r["name_status_v2"] = ne["status_v2"]
        r["lab"] = dom.lab_of(r["source"])
        kt = []
        if lab_id < OTHER and r["source"] in auth:
            kt.append("KT4")
        if lab_id == OTHER and (r["source"], r["src_id"]) in k5:
            kt.append("KT5")
        r["kt"] = kt
        if lab_id < OTHER:
            r["fail_modes"] = dict(sorted(fail_rows.get((r["source"], r["src_id"], lab_id), {}).items()))
            a = oac.get((r["source"], r["src_id"], lab_id))
            r["other_argmax_conflicts"] = ({"n": a["n"], "mean_p_other": round(a["p"] / a["n"], 6),
                                            "tau_other": _r6(tau[OTHER]), "second": dict(sorted(a["second"].items())),
                                            "mean_p_second": round(a["p2"] / a["n"], 6)} if a else None)
        else:
            r["fail_modes"] = None
            r["other_argmax_conflicts"] = None
    # --- veto
    by_blocker = collections.Counter(v["blockers"][0] for v in vetoes if v["blockers"])
    combos = collections.Counter("+".join(v["blockers"]) for v in vetoes if v["blockers"])
    veto = {"lost_boxes": len(vetoes), "images": len({v["key"] for v in vetoes}),
            "by_blocker": {k: int(by_blocker.get(k, 0)) for k in BLOCKERS},
            "combinations": dict(sorted(combos.items())),
            "without_blocker": sum(1 for v in vetoes if not v["blockers"]),
            "per_class": {k: dict(v) for k, v in sorted(per_class_vs.items())},
            "contract_check": {"images": {"got": len({v["key"] for v in vetoes}), "contract": cn["veto_images"]},
                               "by_blocker": {"got": {k: int(by_blocker.get(k, 0)) for k in BLOCKERS},
                                              "contract": {"other_called_target": cn["veto_other_called_target"],
                                                           "species_conflict": cn["veto_species_conflict"],
                                                           "species_unknown": cn["veto_species_unknown"],
                                                           "remaining": cn["veto_remaining"]}},
                               "recorded_not_asserted": True}}
    # --- S8b reject class
    fi = _read_json(V.VERIFIER_DIR / "fit_info.json")
    elig = collections.Counter()
    status_cache = {}
    for n_, i in enumerate(R["pidx"]):
        if int(crops.label[i]) != OTHER or not R["xfin"][n_]:
            continue
        nm = crops.src_name[i]
        if nm not in status_cache:
            status_cache[nm] = V.other_name_status(nm)
        if status_cache[nm] is None:
            elig["%s|%s" % (crops.source[i], nm)] += 1
    other_ids = np.asarray(ver.arrays.get("other_ids", np.zeros(0, np.int64)), dtype=np.int64)
    drawn = collections.Counter("%s|%s" % (crops.source[i], crops.src_name[i]) for i in other_ids)
    drawn_src = collections.Counter(crops.source[i] for i in other_ids)
    s8_discard_src = {r["source"] for r in rows if r["label"] != C.CLASS_NAMES[OTHER]
                      and (r["verdict"].get(V.CONFLICT) or r["verdict"].get(V.UNKNOWN))}

    def top(cnt):
        if not cnt:
            return None
        c, n = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return {"cell": c, "n": int(n), "share": round(n / sum(cnt.values()), 6), "of": int(sum(cnt.values()))}
    s8b = {"eligible": dict(sorted(elig.items())), "drawn": dict(sorted(drawn.items())),
           "drawn_by_source": dict(sorted(drawn_src.items())), "eligible_top_cell": top(elig),
           "drawn_top_cell": top(drawn),
           "rejected_positive_sources_in_drawn": sorted(set(drawn_src) & s8_discard_src),
           "fit_info_taken": int(((fi.get("other_sample") or {}).get("taken")) or 0)}
    # --- H5a
    twins_imgs = twins_boxes = more_inf = 0
    examples = []
    for t in twins:
        kept_lab = collections.Counter(int(b[0]) for b in meta[t["twin_of"]]["boxes"])
        drop_lab = collections.Counter(int(c) for c in t["inc"])
        extra_t = sum(max(0, drop_lab[k] - kept_lab[k]) for k in range(NT))
        if extra_t:
            twins_imgs += 1
            twins_boxes += extra_t
        km = meta[t["twin_of"]]
        kslug = km["source"]
        kinf = sum(1 for b in range(len(km["boxes"]))
                   if N.frame_of(ns_status[(kslug, "*" if kslug in wildcard else str(km["src"][b][0]))], dom)
                   in ("named", "target"))
        dinf = sum(1 for sid in t["src"]
                   if N.frame_of(ns_status[(t["source"], "*" if sid == "*" else str(sid))], dom) in ("named", "target"))
        if dinf > kinf:
            more_inf += 1
        if (extra_t or dinf > kinf) and len(examples) < 20:
            examples.append({"id": t["id"], "twin_of": t["twin_of"], "extra_target_boxes": extra_t,
                             "informative_boxes": {"dropped": dinf, "kept": kinf}})
    h5a = {"twins": len(twins),
           "dropped_twins_with_target_box_kept_lacks": {"images": twins_imgs, "boxes": twins_boxes},
           "dropped_twins_more_informative_names": {"images": more_inf},
           "rule": "boxes of a target class the dropped twin holds beyond the kept twin's count; a twin is "
                   "more informative when more of its boxes carry a name in the named or target frame (v2)",
           "examples": examples}
    # --- S1 skipped slugs
    registry = V._load_registry()
    s1 = {}
    for reason, slugs in sorted((ps.get("skipped") or {}).items()):
        for s_ in slugs:
            ent = registry.get(s_)
            s1[s_] = {"reason": reason,
                      "registry_entry": ({k: v for k, v in ent.items() if k != "class_names"}
                                         if isinstance(ent, dict) else None)}
    # --- H12
    h12 = None
    if known_items is not None:
        ki = _read_json(known_items)
        if ki.get("format") != "funnel-known-items/1":
            raise AdapterError("%s is not a funnel-known-items/1" % known_items)
        slugs_all = sorted(registry)
        items = []
        for it in ki.get("items", []):
            pats = [re.compile(p) for p in it.get("slug_patterns", [])]
            hit = sorted(s_ for s_ in slugs_all if any(p.search(s_) for p in pats))
            items.append({"name": it["name"], "present": bool(hit), "slugs": hit})
        h12 = {"known_items": {"path": str(known_items), "sha256": inputs["known_items"]["sha256"]},
               "items": items, "present": sum(1 for i in items if i["present"]), "total": len(items)}
    # --- totals and reconciliation
    tcore = collections.Counter(C.CLASS_NAMES[int(crops.label[i])] for i in crops.where("core"))
    img_tot = collections.Counter(img_v.values())
    checks = []

    def chk(name, got, want, src):
        checks.append({"name": name, "got": got, "want": want, "want_from": src, "ok": got == want})
    chk("embedded_crops", int(len(R["pidx"])), sum(int(v) for k, v in ad["boxes"].items() if k != V.SMALL),
        "admit_summary.json boxes (without small)")
    for vname in (V.VERIFIED, V.CONFLICT, V.UNKNOWN, V.OTHER_OK):
        chk("verdict_%s" % vname, int(totals_box.get(vname, 0)), int(ad["boxes"].get(vname, 0)),
            "admit_summary.json boxes.%s" % vname)
    want_adm = sum(int(ad["admitted_boxes_per_class"].get(n, 0)) for n in C.CLASS_NAMES[:NT])
    chk("admitted_target_boxes", admitted_target, want_adm, "admit_summary.json admitted_boxes_per_class")
    chk("small_boxes", small_total, int(ad["boxes_small_not_embedded"]), "admit_summary.json boxes_small_not_embedded")
    chk("pool_images", len(meta), int(ps["images"]), "pool_summary.json images")
    chk("pool_boxes", sum(len(m["boxes"]) for m in meta.values()), int(ps["boxes"]), "pool_summary.json boxes")
    for k in (V.ADMITTED, V.CONFLICT, V.UNKNOWN):
        chk("images_%s" % k, int(img_tot.get(k, 0)), int(ad["images"].get(k, 0)), "admit_summary.json images")
    chk("veto_lost_boxes", len(vetoes), int(ad["boxes"][V.VERIFIED]) - want_adm,
        "admit_summary.json verified - admitted target boxes")
    chk("veto_boxes_without_blocker", veto["without_blocker"], 0, "every lost box has a blocker")
    chk("base_selected", sum(1 for k in meta if k in base_keys), int(ss["sizes"]["selected"]) +
        int(ss["sizes"].get("refilled", 0)), "select_summary.json sizes.selected")
    chk("increment_pool", len(inc_keys), int(ss["sizes"]["increment_pool"]), "select_summary.json sizes.increment_pool")
    for slug, st in sorted(ps["per_slug"].items()):
        want = {k: int(v) for k, v in (st.get("dropped") or {}).items()}
        got = {k: int(v) for k, v in dcounts.get(slug, {}).items()}
        chk("dropped:%s" % slug, got, want, "pool_summary.json per_slug.dropped")
        chk("kept:%s" % slug, int(kept_by_slug.get(slug, 0)), int(st["kept"]), "pool_summary.json per_slug.kept")
        chk("near_eval_by_split:%s" % slug, dict(near_split.get(slug, {})),
            {k: int(v) for k, v in (st.get("near_eval_by_split") or {}).items()},
            "pool_summary.json per_slug.near_eval_by_split")
    chk("skipped_slugs", {k: sorted(v) for k, v in skipped_slugs.items()},
        {k: sorted(v) for k, v in (ps.get("skipped") or {}).items()}, "pool_summary.json skipped")
    chk("s8b_taken", int(len(other_ids)), s8b["fit_info_taken"], "fit_info.json other_sample.taken")
    ok = all(c["ok"] for c in checks)
    # --- guard pairs and the census-derived ledger
    gp = guard_pairs(out_dir / "guard_pairs_v1.csv", led_path, domain=dom)
    kt_counts = {}
    census_doc = header("census", dom, pre, inputs, seeds={}, modules=tuple(code.values()), testing=testing)
    census_doc.update({
        "rows": rows,
        "totals": {"embedded_crops": int(len(R["pidx"])),
                   "verdicts": {k: int(v) for k, v in sorted(totals_box.items())},
                   "admitted_target_boxes": admitted_target, "small_boxes": small_total,
                   "pool_images": len(meta), "pool_boxes": sum(len(m["boxes"]) for m in meta.values()),
                   "images": {k: int(v) for k, v in sorted(img_tot.items())}},
        "reconciliation": {"ok": ok, "checks": checks},
        "rederivation": R["check"],
        "veto": veto, "s1_skipped": s1, "s8b_reject_class": s8b, "h5a": h5a,
        "train_core_boxes_per_class": {n: int(tcore.get(n, 0)) for n in C.CLASS_NAMES[:NT]},
        "h12": h12,
        "name_status_v2": {"path": str(ns_path), "sha256": ns_sha},
        "ledger_jsonl": {"path": str(led_path), "sha256": led_sha, "box_rows": n_box_rows,
                         "image_rows": len(drows)},
        "guard_pairs": gp,
        "known_truth_counts": kt_counts,
        "seconds": round(time.time() - t0, 1),
    })
    if ok:
        led = _census_ledger(dom, pre, inputs, ps, ad, ss, cal, rows, ns_status, s8b, elig, other_ids, testing)
        fl_path = out_dir / "funnel_ledger.json"
        if fl_path.exists():
            try:
                old = read_json(fl_path)
            except FunnelError:
                old = None
            if (isinstance(old, dict) and old.get("derivation") == "census"
                    and old.get("fingerprint") != led["fingerprint"] and not force):
                raise AdapterError("%s is census-derived from other Step 1 files; rerun with --force" % fl_path)
        L.write(fl_path, led)
        census_doc["funnel_ledger"] = {"path": str(fl_path), "fingerprint": led["fingerprint"]}
        try:
            kt = known_truth(dom, out_dir)
            census_doc["known_truth_counts"] = {k: len(v) for k, v in sorted(kt.items())}
        except (AdapterError, TaxonomyError) as e:
            census_doc["known_truth_counts"] = {"error": str(e)}
    write_json_atomic(census_path, census_doc)
    if not ok:
        bad = [c for c in checks if not c["ok"]]
        raise AdapterError("census does not reconcile with the Step 1 summaries (%d check(s)): %s"
                           % (len(bad), "; ".join("%s got %s want %s" % (c["name"], c["got"], c["want"])
                                                  for c in bad[:5])))
    log("census: %d box rows, %d pre-pool rows, %d census rows (%.0fs)"
        % (n_box_rows, len(drows), len(rows), time.time() - t0))
    return census_doc


def _census_ledger(dom, pre, inputs, ps, ad, ss, cal, rows, ns_status, s8b, elig, other_ids, testing):
    identity = {k: inputs[k]["sha256"] for k in ("admit_summary", "pool_summary", "select_summary")}
    identity["calibration"] = inputs["calibration"]["sha256"]
    parts = {"pool_summary": ps, "admit_summary": ad, "select_summary": ss, "calibration": cal, "rows": rows,
             "calibration_sha256": inputs["calibration"]["sha256"],
             "eligible_other": int(sum(elig.values())), "taken_other": int(len(other_ids))}
    led_inputs = {k: inputs[k] for k in ("admit_summary", "pool_summary", "select_summary", "calibration",
                                         "crops", "pool_verdicts", "taxonomy_cache")}
    led = L.new(dom, led_inputs, "census", "v2", identity, prereg=pre, modules=(_self_module(), N, T),
                testing=testing)
    for st in _build_stages(dom, parts):
        L.add_stage(led, st)
    led["label_spaces"] = _kinds_from_rows(rows, lambda r: ns_status[(r["source"], r["src_id"])],
                                           lambda s: v2_kind(s, dom))
    led["domain_scores"] = _domain_scores(ss)
    led["reject_class"] = {"stage": "S8b", "eligible_top_cell": s8b["eligible_top_cell"],
                           "drawn_top_cell": s8b["drawn_top_cell"],
                           "sample_sources": dict(s8b["drawn_by_source"]) or None}
    return led


# ============================================================= known truth
def known_truth(domain, funnel_dir):
    """{KT id: [item]} (runner §4.6) from the Step 1 files, the census outputs
    (ledger.jsonl, name_status_v2.json), relation_geometry_v1.json (KT6, only
    after the H3a exact part passes) and the fetched KT7 photos."""
    dom = D.load(domain)
    fd = _funnel_dir(funnel_dir)
    K = _keys(dom, fd)
    other_id = dom.other["id"]
    out = {k: [] for k in dom.kt_ids()}

    def item(uid, kt, crop_id, crop_set, truth, taxon, kind, claimed, source, key_, prov, session=None,
             role=None):
        return {"id": uid, "kt": kt, "crop_id": crop_id, "crop_set": crop_set, "truth": truth,
                "truth_taxon": taxon, "truth_kind": kind, "claimed": claimed, "source": source,
                "lab": dom.lab_of(source), "near_dup3": K.near(key_), "provenance": prov,
                "session": session, "role": role}
    taxon_of = {t["id"]: t["taxon"] for t in dom.targets}
    crops = crop_table()
    # KT1: every train_core crop
    ref = dom.reference_source
    for i in crops.where("core"):
        key_ = crops.key[i]
        lab = int(crops.label[i])
        out.setdefault("KT1", []).append(item("t1:%s#%d" % (key_, int(crops.box[i])), "KT1", int(i), "core", lab,
                                              taxon_of.get(lab), "target", False, ref, key_, "prov:%s" % key_,
                                              session=crops.group[i]))
    # KT2 / KT3: copies box-matched to their train_core twin
    kt3_sources = set(dom.kt("KT3").get("sources", [])) if "KT3" in out else set()
    copy_crop = {(crops.key[i], int(crops.box[i])): int(i) for i in crops.where("copy")}
    for r in (_read_jsonl(V.COPIES) if V.COPIES.exists() else []):
        mine = C.read_yolo(r["label"])
        ref_boxes = C.read_yolo(r["train_core_label"])
        kt = "KT3" if r["source"] in kt3_sources else "KT2"
        for b, box in enumerate(mine):
            t = V._match_box(box, ref_boxes)
            if t is None:
                continue
            lab = int(t[0])
            out.setdefault(kt, []).append(item("t2:%s#%d" % (r["key"], b), kt, copy_crop.get((r["key"], b)), "copy",
                                               lab, taxon_of.get(lab), "target", False, r["source"], r["key"],
                                               "prov:%s" % r["train_core_key"],
                                               session=r.get("train_core_session")))
    # KT4 / KT5 from the census ledger
    led = fd / "ledger.jsonl"
    nsp = fd / "name_status_v2.json"
    if not led.exists() or not nsp.exists():
        raise AdapterError("known truth KT4/KT5 needs the census (ledger.jsonl, name_status_v2.json in %s)" % fd)
    ns = read_json(nsp)
    tc = (ns.get("taxonomy_cache") or {}).get("path")
    if not tc:
        # KT5's taxa and KT6's card taxa are read through the resolver: never an empty guess
        raise AdapterError("%s names no taxonomy cache; known truth KT5/KT6 needs it" % nsp)
    resolver = T.Resolver(T.load_cache(tc, dom), dom)
    k5 = kt5_taxa(dom, resolver, ns)
    kt6_ids = _kt6_ids(dom, fd, resolver)
    kt6_src = set(dom.kt("KT6").get("sources", [])) if "KT6" in out else set()
    kt6_items = []
    for row in L.iter_units(led, unit="box"):
        if "KT4" in row["kt"]:
            out.setdefault("KT4", []).append(item(row["id"], "KT4", row["crop_id"], "pool", row["label"],
                                                  taxon_of.get(row["label"]), "target", True, row["source"],
                                                  row["key"], row["provenance"]))
        if "KT5" in row["kt"]:
            tx5 = k5.get((row["source"], row["src_id"]))
            if tx5 is None:
                raise AdapterError("ledger row %s is in KT5 but its class %s|%s resolves to no attractor or "
                                   "'not' taxon now (the census and the config or cache disagree; rerun census)"
                                   % (row["id"], row["source"], row["src_id"]))
            out.setdefault("KT5", []).append(item(row["id"], "KT5", row["crop_id"], "pool", other_id,
                                                  tx5, "attractor", True,
                                                  row["source"], row["key"], row["provenance"]))
        if kt6_ids and row["source"] in kt6_src and row["src_id"] in kt6_ids:
            tid = kt6_ids[row["src_id"]]
            kt6_items.append(item(row["id"], "KT6", row["crop_id"], "pool", tid, taxon_of.get(tid), "target",
                                  True, row["source"], row["key"], row["provenance"]))
    if kt6_items:
        groups = sorted({it["near_dup3"] for it in kt6_items})
        g = np.random.default_rng(C.stable_int("funnel/v1/kt6/split")).permutation(len(groups))
        cal_groups = {groups[i] for i in g[:(len(groups) + 1) // 2]}
        for it in kt6_items:
            it["role"] = "calibration" if it["near_dup3"] in cal_groups else "estimation"
        out["KT6"] = kt6_items
    # KT7
    for it in _kt7_items(dom, fd):
        crop_id = it.get("crop_id")
        att = it.get("attractor")
        tgt = it.get("target")
        truth = dom.class_id(tgt) if tgt else other_id
        out.setdefault("KT7", []).append({
            "id": it["id"], "kt": "KT7", "crop_id": crop_id, "crop_set": "kt7", "truth": truth,
            "truth_taxon": it.get("taxon"), "truth_kind": "target" if tgt else ("attractor" if att else "other"),
            "claimed": False, "source": "kt7", "lab": dom.lab_of("kt7"), "near_dup3": K.near(it["id"]),
            "provenance": "prov:kt7|%s" % it.get("observation_id"), "session": None, "role": it.get("role")})
    for k in out:
        out[k].sort(key=lambda x: x["id"])
    return out


def _kt6_ids(dom, fd, resolver):
    """{AgML class id (str): target id} for the card-resolved classes, only when
    relation_geometry_v1.json records that the H3a exact part passed."""
    p = fd / "relation_geometry_v1.json"
    if not p.exists() or resolver is None:
        return {}
    geo = read_json(p)
    if not (geo.get("h3a_exact") or {}).get("pass"):
        return {}
    srcs = dom.kt("KT6").get("sources", [])
    out = {}
    from .. import relation as REL
    for slug in srcs:
        mt = (geo.get("matches") or {}).get(slug)
        res = (dom.raw["sources"].get("card_resolvers") or {}).get(slug) or {}
        table = res.get("class_table") or {}
        if not mt or not table:
            continue
        if not REL._card_names_ok(mt):
            # the card's ids are not shown to be the upstream's: no card truth from it
            continue
        amap = REL.alignments(len(table)).get(mt.get("chosen"))
        if amap is None:
            continue
        for uid, row in table.items():
            e = resolver.resolve(row["taxon"], scientific_only=True)
            for t in dom.targets:
                if e is not None and resolver.is_target_synonym(e, t):
                    agml = amap.get(int(uid))
                    if agml is not None:
                        out[str(agml)] = t["id"]
    return out


# ============================================================= relation raw material
def source_labels(slug):
    """{pool key: [(source class id, cx, cy, w, h, W, H)]} of one source's pool
    images: the source's own class ids (pool_meta.jsonl "src"), not the joined
    INC ids, with the image size verify recorded (geometry match, H3a, H1-pre)."""
    out = {}
    for m in _read_jsonl(V.POOL_META):
        if m["source"] != slug:
            continue
        rows = []
        for b, box in enumerate(m["boxes"]):
            sid = m["src"][b][0] if b < len(m.get("src") or []) else None
            if sid is None:
                raise AdapterError("pool image %s box %d has no source class id" % (m["key"], b))
            rows.append((int(sid), float(box[1]), float(box[2]), float(box[3]), float(box[4]), int(m["W"]), int(m["H"])))
        out[m["key"]] = rows
    if not out:
        raise AdapterError("no pool image of source %s" % slug)
    return out


def relation_units(domain, sources):
    """Class units of the reference-image copies (cwd12_copies.jsonl) of the
    given sources: {"c:<slug>|<src_id>": {"source", "src_id", "boxes",
    "crop_ids", "truth", "join", "old_join"}}. truth is the majority class of
    the box-matched train_core twins; join the current joined class; old_join
    the pre-v3.60.0 join (verify.old_join), "deleted" when that join dropped
    the box. Only boxes matched to a twin count."""
    D.load(domain)
    want = set(sources)
    crops = crop_table()
    copy_crop = {(crops.key[i], int(crops.box[i])): int(i) for i in crops.where("copy")}
    acc = {}
    for r in (_read_jsonl(V.COPIES) if V.COPIES.exists() else []):
        if r["source"] not in want:
            continue
        mine = C.read_yolo(r["label"])
        ref = C.read_yolo(r["train_core_label"])
        old = r.get("old_join") or [None] * len(mine)
        src = r.get("src") or [[None, ""]] * len(mine)
        for b, box in enumerate(mine):
            t = V._match_box(box, ref)
            if t is None:
                continue
            sid = str(src[b][0]) if src[b][0] is not None else "*"
            u = "c:%s|%s" % (r["source"], sid)
            a = acc.setdefault(u, {"source": r["source"], "src_id": sid, "crop_ids": [], "truth": collections.Counter(),
                                   "join": collections.Counter(), "old_join": collections.Counter()})
            a["truth"][C.CLASS_NAMES[int(t[0])]] += 1
            a["join"][C.CLASS_NAMES[int(box[0])]] += 1
            a["old_join"]["deleted" if old[b] is None else C.CLASS_NAMES[int(old[b])]] += 1
            c = copy_crop.get((r["key"], b))
            if c is not None:
                a["crop_ids"].append(c)
    out = {}
    for u, a in sorted(acc.items()):
        out[u] = {"source": a["source"], "src_id": a["src_id"], "boxes": len(a["crop_ids"]),
                  "crop_ids": sorted(a["crop_ids"]), "truth": a["truth"].most_common(1)[0][0],
                  "truth_purity": round(a["truth"].most_common(1)[0][1] / sum(a["truth"].values()), 6),
                  "join": a["join"].most_common(1)[0][0], "old_join": a["old_join"].most_common(1)[0][0]}
    return out


def pool_class_units(domain, funnel_dir):
    """{"c:<slug>|<src_id>": {"source", "src_id", "boxes", "crop_ids", "join",
    "status_v2"}} over the embedded pool boxes of ledger.jsonl."""
    fd = _funnel_dir(funnel_dir)
    led = fd / "ledger.jsonl"
    if not led.exists():
        raise AdapterError("no ledger.jsonl in %s: run census first" % fd)
    out = {}
    for row in L.iter_units(led, unit="box"):
        if row["crop_id"] is None:
            continue
        u = "c:%s|%s" % (row["source"], row["src_id"])
        a = out.setdefault(u, {"source": row["source"], "src_id": row["src_id"], "boxes": 0, "crop_ids": [],
                               "join": C.CLASS_NAMES[int(row["label"])], "status_v2": row["name_status_v2"]})
        a["boxes"] += 1
        a["crop_ids"].append(int(row["crop_id"]))
    return out
