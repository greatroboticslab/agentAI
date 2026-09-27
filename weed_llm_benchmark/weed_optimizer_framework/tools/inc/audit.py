"""INC attribution item 4: the BioCLIP-2 label audit. Contract:
docs/INCREMENTAL_PROTOCOL.md, "Gate", attribution item 4.

    python -m weed_optimizer_framework.tools.inc.audit --trusted TRUSTED.jsonl \\
        --audit NAME=MANIFEST.jsonl [NAME=MANIFEST.jsonl ...] --out OUT.json [--nshards N]

What share of a manifest's box labels do the pixels contradict? A 12-species
probe is fitted on the trusted manifest's boxes only, and every box of each
audited manifest is judged against the label that manifest's own label file
gives it (a planted-corruption copy is judged on its corrupted labels).

Inputs are inc/verify.py's Step 1 outputs, read through its own path constants
and readers: crops.csv (one row per box of train_core, the cwd12 copies and the
pool, with the box geometry; verify.Crops), crops_skipped.csv, crops_info.json
(verify.check_fresh: refused when crops.csv or anything it was made from
changed since `verify crops`) and the BioCLIP-2 embedding shards
emb/emb_sNNN_of_NNN.npz (verify.load_embeddings). No image is opened and
nothing is embedded: a box the Step 1 crops do not hold is not judged.

Box lookup. A manifest row's image is looked up among the images crops.csv
was cut from (the train_core manifest, cwd12_copies.jsonl, pool.jsonl): by its
path, or, when the path is not there, by its sha256 (the same bytes under
another path; counted as images_matched_by_sha256). An image found by path
under another sha256 changed after verify read it; its boxes are refused
(image_changed). Box k of the row's label file (common.read_yolo order, the
order `verify crops` numbered the boxes in) is the crop row (image, k), used
only when the row's (cx, cy, w, h) all lie within GEOM_TOL (1e-4) of the
label's. Otherwise the box is not judged and is reported as no_embedding, with
its reason: geometry_mismatch, failed (a NaN feature row), small / no_size
(left out by `verify crops`), no_crop_row, image_not_in_crops, image_changed.
A box is judged on the crop of exactly its image region, or not at all.

The probe: verify.fit_verifier on the trusted manifest's species boxes (ids
0-11), labelled by the trusted manifest's own label files. It is verify's
probe, prototypes and threshold rule: logistic regression on L2-normalised
features; per-class joint thresholds at 95 % per-species recall from
5-fold GroupKFold, grouped by capture session (a row without a session is
grouped by its source). Only species boxes train it: the trusted set's
OtherPlant boxes are counted and left out, and the pool's OtherPlant sample
that `verify fit` adds is not (it could hold the very images audited here), so
this probe never predicts OtherPlant.

Verdicts: verify.verdicts (Verifier.judge_features), the rule fit, calibrate
and admit share. A species label is verified (the probe confidently says it),
a conflict (the probe confidently says another species) or unknown; an
OtherPlant label is other_ok, or a conflict when the probe confidently calls a
cwd12 species. A species the trusted boxes do not contain is one the probe
cannot say: a box with that label is label_unseen and is not judged.

An audited image that is also in the trusted manifest is refused
(in_trusted), because the probe was fitted on its label: the same path, the
same sha256, or a Step 1 image of the same photograph. A cwd12 copy and its
train_core twin are the same photograph (cwd12_copies.jsonl records the
twin, within 6 dHash bits), in either direction. Other near-duplicates are
not matched; verify pool keeps pool images off train_core's 6-bit
neighbourhood (they become copies), so a train_core trusted set such as P0
has none in the pool. Audited rows that share a capture session with trusted
rows are counted (burst neighbours of trusted photographs are judged
optimistically); the pilot's sessions are disjoint by construction.

Rates, per audited manifest, over judged boxes:
  * the conflict rate among species-labelled boxes (the label-noise estimate),
    the verified and unknown rates, with a Wilson 95 % interval;
  * per species (of the label given), per source, and the conflict pairs
    (label -> the probe's call);
  * OtherPlant-labelled boxes: the rate the probe confidently calls a cwd12
    species (with no OtherPlant class, this also counts non-cwd12 plants that
    look like one);
  * known truth: a box whose crop is a train_core crop carries its train_core
    label in crops.csv, so for those boxes the true label error rate and the
    audit's conflict recall and false-conflict rate are reported (for a
    Bswap-style copy of train_core, the planted swaps).
Baseline, on held-out parts of the trusted set (as verify calibrate (b)):
outer GroupKFold (5) over the trusted groups; each fold's verifier, thresholds
included, is fitted on the other folds and judges the held-out fold
  * with the true labels: the baseline false-conflict rate f (and verified /
    unknown rates);
  * with every held-out box given one seeded wrong species (uniform over the
    other 11, as the pilot's Bswap draws): the conflict recall r on wrong
    labels;
  * both per species (f_k over true species k with its own label, r_k over
    true species k given a wrong one).
Boxes whose true (or wrong) label the fold's probe never saw are left out.
Per-species rates differ widely (a species close to another is flagged more
often), and an audited set's species mix need not be the trusted set's, so
each audited set is compared with the baseline for its own mix
(baseline_for_mix): f_mix = sum w_k f_k and r_mix = sum w_k r_k, w_k the
share of the set's judged species-labelled boxes labelled k. f_mix's 95 %
interval: percentiles of seeded draws of sum w_k f_k, each f_k from its
Jeffreys posterior, so a species with few trusted boxes widens it. A species
without a baseline entry is listed and its boxes are left out of the
comparison. noise_estimate_corrected = (c - f_mix) / (r_mix - f_mix),
clipped to [0, 1], c the set's conflict rate over those boxes, is the share
of wrong labels that gives conflict rate c when wrong labels are caught at
rate r and right ones flagged at rate f. It assumes the audited set's labels
behave like held-out trusted ones (same domain) and its wrong labels are
uniform swaps; the label mix stands in for the true species of the wrong
labels (exact when every label is right, approximate for a set both skewed
and noisy). For harvested sources neither f nor r is measured on their own
domain, so there it is indicative only. above_baseline: c's Wilson interval
lies wholly above f_mix's interval. The same two numbers against the
baseline pooled over the trusted set's own mix are reported as context only
(noise_estimate_pooled_baseline, above_pooled_baseline).

Outputs: OUT.json, OUT.md (tables) and OUT_boxes.csv (one row per audited
box: status, reason, the probe's call, crop id). Deterministic: the same
inputs give the same files, apart from built_utc and seconds. --out must end
in .json; no output may lie under the Step 1 directory or be an input, and an
existing file is overwritten only when it is an earlier audit's output.

First use, the pilot's post-hoc item 4 (the driver does not run it; exp.json
"attribution_scope" lists it as not run):
    trusted = INC_DIR/pilot_v1/manifests/P0.jsonl
    audited = I1..I5, Bswap, Breal of INC_DIR/pilot_v1/manifests/
Every pilot image path is an absolute cluster path. P0 and I1..I5 are
train_core rows (their images and label files are train_core's, so every box
is its own core crop). Bswap's rows are train_core images with the corrupted
label copies under INC_DIR/pilot_v1/labels/Bswap/: the geometry of every line
is unchanged, only classes differ, and the known-truth block measures the
audit against the planted swaps. Breal's rows are images of the two Breal
slugs with the pilot's join labels; verify's pool holds them (its crops exist)
when verify pool read the same files, else they are reported as
image_not_in_crops. A Breal box verify clipped to the frame, or whose index
moved because verify dropped a degenerate box before it, is a
geometry_mismatch, not a guess.
    sbatch run_inc_audit.sh --trusted $INC/pilot_v1/manifests/P0.jsonl \\
        --audit I1=$M/I1.jsonl ... Bswap=$M/Bswap.jsonl Breal=$M/Breal.jsonl \\
        --out $INC/pilot_v1/audit/label_audit.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import verify as V

GEOM_TOL = 1e-4               # crop row vs label geometry, normalised units
N_SPECIES = V.N_SPECIES
OTHER = C.OTHER_PLANT
N_FOLDS = V.N_FOLDS
SWAP_SEED = "inc/audit/swap/%d"
MIN_SEPARATION = 0.10         # r - f below this: no corrected estimate
MIX_SEED = "inc/audit/mix"    # the draws of a species mix's baseline interval
MIX_DRAWS = 20000
Z95 = 1.959964
NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
EXAMPLES = 20
WHAT = "INC attribution item 4, the BioCLIP-2 label audit (docs/INCREMENTAL_PROTOCOL.md, Gate)"
MD_TITLE = "# Label audit (INC attribution item 4)"

# statuses of an audited box beyond verify's verdicts
NO_EMB = "no_embedding"
IN_TRUSTED = "in_trusted"
LABEL_UNSEEN = "label_unseen"
STATUSES = (V.VERIFIED, V.CONFLICT, V.UNKNOWN, V.OTHER_OK, LABEL_UNSEEN, IN_TRUSTED, NO_EMB)
JUDGED = (V.VERIFIED, V.CONFLICT, V.UNKNOWN, V.OTHER_OK)
# why a box has no embedding to judge
IMAGE_NOT_IN_CROPS, IMAGE_CHANGED, NO_CROP_ROW, GEOMETRY_MISMATCH = (
    "image_not_in_crops", "image_changed", "no_crop_row", "geometry_mismatch")
REASONS = (IMAGE_NOT_IN_CROPS, IMAGE_CHANGED, NO_CROP_ROW, GEOMETRY_MISMATCH, "small", "no_size",
           V.FAILED)
SETS = ("core", "copy", "pool")         # verify.SETS, in lookup preference order
BOX_FIELDS = ("manifest", "key", "image", "source", "session", "box", "label", "label_name",
              "status", "reason", "pred", "pred_name", "p", "cosine", "crop_id", "crop_set",
              "matched_by", "train_core_label", "geom_delta")


class AuditError(RuntimeError):
    """A condition under which the audit must not produce its outputs."""


def log(msg):
    print("[inc.audit] %s" % msg, flush=True)


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rate(k, n):
    return round(float(k) / n, 4) if n else None


def _wilson(k, n, z=Z95):
    """Wilson score 95 % interval of k successes in n, or None."""
    if not n:
        return None
    p = float(k) / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _clean(obj):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


# ------------------------------------------------------------------ inputs
def parse_audits(specs):
    """[(name, Path)] from NAME=MANIFEST.jsonl strings; names unique and path-safe."""
    out, seen = [], set()
    for s in specs or ():
        name, sep, path = str(s).partition("=")
        if not sep or not name or not path:
            raise AuditError("--audit takes NAME=MANIFEST.jsonl, got %r" % s)
        if not NAME_RE.fullmatch(name):
            raise AuditError("audit name %r: use letters, digits, '_', '.', '-'" % name)
        if name in seen:
            raise AuditError("audit name %r given twice" % name)
        seen.add(name)
        out.append((name, Path(path)))
    if not out:
        raise AuditError("nothing to audit: give at least one --audit NAME=MANIFEST.jsonl")
    return out


def read_rows(path, what):
    """[(row, boxes)] of a manifest, sorted by key. Every label file must hash
    to the manifest's label_sha256 (the audit reads the labels the manifest
    names, not a file changed since) and hold INC ids only."""
    path = Path(path)
    if not path.is_file():
        raise AuditError("%s manifest %s not found" % (what, path))
    rows = C.read_manifest(path)
    if not rows:
        raise AuditError("%s manifest %s is empty" % (what, path))
    out = []
    for r in sorted(rows, key=lambda r: r.get("key", "")):
        missing = [k for k in C.MANIFEST_KEYS if k not in r]
        if missing:
            raise AuditError("%s: row %s lacks %s" % (path, r.get("key"), missing))
        try:
            lsha = C.sha256_file(r["label"])
        except OSError as e:
            raise AuditError("%s: label of %s unreadable: %s" % (path, r["key"], e))
        if lsha != r["label_sha256"]:
            raise AuditError("%s: label %s does not hash to the manifest's label_sha256 (%s != %s): "
                             "it changed after the manifest was written"
                             % (path, r["label"], lsha[:12], str(r["label_sha256"])[:12]))
        boxes = C.read_yolo(r["label"])
        bad = [b[0] for b in boxes if not 0 <= b[0] < C.NC]
        if bad:
            raise AuditError("%s: label %s holds class %d, outside the INC class space 0..%d"
                             % (path, r["label"], bad[0], C.NC - 1))
        out.append((r, boxes))
    return out


def _source_files():
    """The manifests crops.csv was cut from, per verify crops set."""
    return (("core", C.manifest_path("train_core")), ("copy", Path(V.COPIES)),
            ("pool", Path(V.POOL)))


class CropIndex:
    """Where the Step 1 crop of each manifest box is, or why there is none.

    Only the images the manifests name are indexed (by path, and by sha256
    for rows whose path is not among the Step 1 images). An entry is
    (set, key, sha256, image, photograph): the photograph is the train_core
    image a cwd12 copy was matched to (cwd12_copies.jsonl train_core_image,
    within 6 dHash bits), else the image itself."""

    def __init__(self, crops, X, manifests):
        images = {r["image"] for rows in manifests for r, _b in rows}
        shas = {r["sha256"] for rows in manifests for r, _b in rows}
        self.crops, self.X = crops, X
        self.by_image, self.by_sha = {}, collections.defaultdict(list)
        for set_, path in _source_files():
            if not path.is_file():
                raise AuditError("%s not found: the Step 1 inputs are incomplete" % path)
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    img, sha = r["image"], r.get("sha256")
                    if img not in images and sha not in shas:
                        continue
                    photo = img
                    if set_ == "copy":
                        photo = r.get("train_core_image")
                        if not photo:
                            raise AuditError("%s: copy %s has no train_core_image" % (path, r["key"]))
                    ent = (set_, r["key"], sha, img, photo)
                    self.by_image.setdefault(img, ent)
                    self.by_sha[sha].append(ent)
        for sha in self.by_sha:
            self.by_sha[sha].sort(key=lambda e: (SETS.index(e[0]), e[3], e[1]))
        need = {e[3] for e in self.by_image.values()}
        need.update(e[3] for lst in self.by_sha.values() for e in lst)
        self.row_of = collections.defaultdict(list)
        for i, img in enumerate(crops.image):
            if img in need:
                self.row_of[(img, int(crops.box[i]))].append(i)
        self.skipped = {s: V.read_skipped(s) for s in SETS}
        self._finite = {}

    def image(self, row):
        """(entry, how): entry = (set, key, sha256, image, photograph) of the
        Step 1 image this row's boxes are cut from, how = 'path' | 'sha256';
        or (None, why)."""
        e = self.by_image.get(row["image"])
        if e is not None:
            if e[2] != row["sha256"]:
                return None, IMAGE_CHANGED
            return e, "path"
        cands = self.by_sha.get(row["sha256"])
        if cands:
            return cands[0], "sha256"
        return None, IMAGE_NOT_IN_CROPS

    def finite(self, i):
        v = self._finite.get(i)
        if v is None:
            v = self._finite[i] = bool(np.isfinite(np.asarray(self.X[i], dtype=np.float32)).all())
        return v

    def box(self, entry, k, box):
        """(crop id, reason or None, geometry delta) of box k of an image."""
        ids = self.row_of.get((entry[3], k), [])
        if ids:
            cr = self.crops
            deltas = [max(abs(cr.cx[i] - box[1]), abs(cr.cy[i] - box[2]),
                          abs(cr.w[i] - box[3]), abs(cr.h[i] - box[4])) for i in ids]
            ok = [i for i, d in zip(ids, deltas) if d <= GEOM_TOL]
            if not ok:
                j = int(np.argmin(deltas))
                return ids[j], GEOMETRY_MISMATCH, float(deltas[j])
            i = ok[0]
            d = float(deltas[ids.index(i)])
            return (i, None, d) if self.finite(i) else (i, V.FAILED, d)
        why = self.skipped.get(entry[0], {}).get((entry[1], k))
        if why is not None:
            return -1, why, None
        return -1, NO_CROP_ROW, None


def trusted_ids(index, rows):
    """(images, sha256s, photographs) of the trusted rows [(row, boxes)]:
    the photograph of a row's Step 1 image (CropIndex), which for a cwd12
    copy is its train_core twin."""
    photos = set()
    for r, _b in rows:
        e = index.image(r)[0]
        if e is not None:
            photos.update((e[3], e[4]))
    return ({r["image"] for r, _b in rows}, {r["sha256"] for r, _b in rows}, photos)


def in_trusted(index, row, trusted):
    """True when row's image is one of the trusted manifest's: the same path,
    the same sha256, or a Step 1 image of the same photograph (the same
    image, a cwd12 copy of a trusted train_core image, or the train_core
    twin of a trusted copy). trusted = trusted_ids(...)."""
    if trusted is None:
        return False
    entry, _how = index.image(row)
    return (row["image"] in trusted[0] or row["sha256"] in trusted[1]
            or (entry is not None and (entry[3] in trusted[2] or entry[4] in trusted[2])))


def resolve(index, rows, trusted=None):
    """One record per box of rows [(row, boxes)]; the boxes of an image of the
    trusted manifest (in_trusted) are refused."""
    recs = []
    for r, boxes in rows:
        entry, how = index.image(r)
        refused = IN_TRUSTED if in_trusted(index, r, trusted) else None
        for k, b in enumerate(boxes):
            rec = {"key": r["key"], "image": r["image"], "source": r["source"],
                   "session": r.get("session") or "", "box": k, "label": int(b[0]),
                   "status": None, "reason": None, "crop_id": -1, "crop_set": "",
                   "matched_by": how if entry is not None else "", "geom_delta": None,
                   "pred": -1, "p": None, "cosine": None, "train_core_label": None}
            if refused:
                rec["status"] = refused
            elif entry is None:
                rec["status"], rec["reason"] = NO_EMB, how
            else:
                i, why, d = index.box(entry, k, b)
                rec["geom_delta"] = d
                if i >= 0:
                    rec["crop_id"], rec["crop_set"] = int(i), V.SETS[int(index.crops.set[i])]
                if why is not None:
                    rec["status"], rec["reason"] = NO_EMB, why
                elif rec["crop_set"] == "core":
                    rec["train_core_label"] = int(index.crops.label[i])
            recs.append(rec)
    return recs


def _groups(recs):
    """CV group of each record: its capture session, else its source."""
    return np.array([("session:" + r["session"]) if r["session"] else ("source:" + r["source"])
                     for r in recs], dtype=object)


# --------------------------------------------------------------- summaries
def species_block(statuses):
    """Rates over species-labelled boxes with a verdict."""
    c = collections.Counter(statuses)
    n = c[V.VERIFIED] + c[V.CONFLICT] + c[V.UNKNOWN]
    return {"boxes_judged": n, "verified": c[V.VERIFIED], "conflict": c[V.CONFLICT],
            "unknown": c[V.UNKNOWN], "conflict_rate": _rate(c[V.CONFLICT], n),
            "conflict_rate_ci95": _wilson(c[V.CONFLICT], n),
            "verified_rate": _rate(c[V.VERIFIED], n), "unknown_rate": _rate(c[V.UNKNOWN], n)}


def other_block(statuses):
    c = collections.Counter(statuses)
    n = c[V.OTHER_OK] + c[V.CONFLICT]
    return {"boxes_judged": n, "other_ok": c[V.OTHER_OK], "conflict": c[V.CONFLICT],
            "conflict_rate": _rate(c[V.CONFLICT], n), "conflict_rate_ci95": _wilson(c[V.CONFLICT], n)}


def corrected_noise(c, f, r):
    """(c - f) / (r - f) clipped to [0, 1]; None when r and f are too close."""
    if c is None or f is None or r is None or r - f < MIN_SEPARATION:
        return None
    return round(min(1.0, max(0.0, (c - f) / (r - f))), 4)


SPECIES_VERDICTS = (V.VERIFIED, V.CONFLICT, V.UNKNOWN)


def _has_baseline(b):
    return (b is not None and b.get("boxes_judged") and b.get("conflict_rate") is not None
            and b.get("conflict_recall_on_wrong") is not None)


def mix_interval(counts, weights):
    """95 % interval of f = sum_k w_k k_k / n_k, the rates independent
    binomial with counts [(k_k, n_k)]: the 2.5 and 97.5 percentiles of
    MIX_DRAWS seeded draws with each rate from its Jeffreys posterior
    Beta(k + 0.5, n - k + 0.5), widened to contain f itself (the prior lifts
    the lower percentile above 0 when no species has a conflict). For one
    species it is that rate's Jeffreys interval; a species with few baseline
    boxes widens it."""
    rng = np.random.default_rng(C.stable_int(MIX_SEED))
    tot = np.zeros(MIX_DRAWS)
    f = 0.0
    for (k, n), w in zip(counts, weights):
        tot += w * rng.beta(k + 0.5, n - k + 0.5, size=MIX_DRAWS)
        f += w * float(k) / n
    lo, hi = np.percentile(tot, [2.5, 97.5])
    return [round(min(float(lo), f), 4), round(max(float(hi), f), 4)]


def mix_baseline(recs, baseline):
    """The baseline for this set's own species mix, and the set's conflict
    rate over the boxes it covers.

    Per species k, baseline_cv.per_species holds the held-out false-conflict
    rate f_k (true labels) and the conflict recall r_k (true species k given
    one wrong label). Both are weighted by w_k, the share of this set's judged
    species-labelled boxes labelled k: f_mix = sum w_k f_k, r_mix = sum
    w_k r_k; f_mix's 95 % interval is mix_interval. When every label is
    right the label mix is the species mix, so f_mix is the conflict rate
    expected of this set's correct labels. r_k is indexed by the true
    species; for wrong labels the label mix only stands in for their true
    species, so the corrected estimate is approximate when a set is both
    skewed and noisy. A species without a baseline entry
    (the trusted boxes hold it in one group only, so no fold held it out) is
    listed and its boxes are left out of the weights and of the conflict rate
    compared with them. Returns (block, conflicts, boxes) over the boxes
    weighted."""
    bps = baseline["per_species"]
    judged = [r for r in recs if r["label"] < OTHER and r["status"] in SPECIES_VERDICTS]
    n_by = collections.Counter(r["label"] for r in judged)
    used = {k: n for k, n in n_by.items() if _has_baseline(bps.get(C.CLASS_NAMES[k]))}
    cov = [r for r in judged if r["label"] in used]
    n = len(cov)
    k_conf = sum(1 for r in cov if r["status"] == V.CONFLICT)
    blk = {"boxes": n, "conflict": k_conf, "conflict_rate": _rate(k_conf, n),
           "conflict_rate_ci95": _wilson(k_conf, n),
           "species_without_baseline": {C.CLASS_NAMES[k]: int(n_by[k]) for k in sorted(n_by)
                                        if k not in used},
           "false_conflict_rate": None, "false_conflict_rate_ci95": None,
           "conflict_recall_on_wrong": None, "per_species": {},
           "rule": "f and r of each species (baseline_cv.per_species) weighted by this set's judged "
                   "species-labelled boxes per label; the interval of f: 2.5/97.5 percentiles of %d "
                   "seeded draws, each species' f from its Jeffreys Beta(k + 0.5, n - k + 0.5), "
                   "widened to contain f" % MIX_DRAWS}
    if not n:
        return blk, k_conf, n
    f = r = 0.0
    counts, weights = [], []
    for k in sorted(used):
        b = bps[C.CLASS_NAMES[k]]
        w = used[k] / float(n)
        f += w * b["conflict_rate"]
        r += w * b["conflict_recall_on_wrong"]
        counts.append((b["conflict"], b["boxes_judged"]))
        weights.append(w)
        blk["per_species"][C.CLASS_NAMES[k]] = {
            "boxes": int(used[k]), "weight": round(w, 4), "false_conflict_rate": b["conflict_rate"],
            "conflict_recall_on_wrong": b["conflict_recall_on_wrong"],
            "baseline_boxes": b["boxes_judged"], "baseline_wrong_boxes": b.get("wrong_boxes_judged")}
    blk.update({"false_conflict_rate": round(f, 4),
                "false_conflict_rate_ci95": mix_interval(counts, weights),
                "conflict_recall_on_wrong": round(r, 4)})
    return blk, k_conf, n


def summarise(recs, baseline):
    """The per-manifest block of OUT.json."""
    sp = [r for r in recs if r["label"] < OTHER]
    ot = [r for r in recs if r["label"] == OTHER]
    status = collections.Counter(r["status"] for r in recs)
    reasons = collections.Counter(r["reason"] for r in recs if r["status"] == NO_EMB)
    species = species_block([r["status"] for r in sp])
    # the noise estimate and the test against the baseline, for this set's species mix
    mix, k_mix, n_mix = mix_baseline(recs, baseline)
    species["noise_estimate_corrected"] = corrected_noise(
        float(k_mix) / n_mix if n_mix else None, mix["false_conflict_rate"],
        mix["conflict_recall_on_wrong"])
    ci, bci = mix["conflict_rate_ci95"], mix["false_conflict_rate_ci95"]
    species["above_baseline"] = bool(ci and bci and ci[0] > bci[1])
    species["baseline_for_mix"] = mix
    # context only: the baseline pooled over the trusted set's own mix
    n_sp = species["boxes_judged"]
    pooled = baseline["species"]
    species["noise_estimate_pooled_baseline"] = corrected_noise(
        float(species["conflict"]) / n_sp if n_sp else None, pooled["conflict_rate"],
        baseline["swapped"]["conflict_recall_on_wrong"])
    ci, bci = species["conflict_rate_ci95"], pooled["conflict_rate_ci95"]
    species["above_pooled_baseline"] = bool(ci and bci and ci[0] > bci[1])
    per_species = {}
    for k in range(C.NC):
        mine = [r for r in recs if r["label"] == k]
        if not mine:
            continue
        blk = (species_block if k < OTHER else other_block)([r["status"] for r in mine])
        blk["boxes"] = len(mine)
        blk["not_judged"] = sum(1 for r in mine if r["status"] not in JUDGED)
        per_species[C.CLASS_NAMES[k]] = blk
    per_source = {}
    for s in sorted({r["source"] for r in recs}):
        mine = [r for r in recs if r["source"] == s]
        per_source[s] = {"boxes": len(mine),
                         "species": species_block([r["status"] for r in mine if r["label"] < OTHER]),
                         "otherplant": other_block([r["status"] for r in mine if r["label"] == OTHER])}
    pairs = collections.Counter("%s -> %s" % (C.CLASS_NAMES[r["label"]], C.CLASS_NAMES[r["pred"]])
                                for r in recs if r["status"] == V.CONFLICT)
    known = [r for r in recs if r["train_core_label"] is not None and r["status"] in JUDGED]
    truth = None
    if known:
        correct = np.array([r["label"] == r["train_core_label"] for r in known], dtype=bool)
        truth = V._metrics(correct, np.array([r["status"] for r in known], dtype=object))
        truth["label_error_rate"] = _rate(int((~correct).sum()), len(correct))
        truth["note"] = ("boxes judged on a train_core crop; truth = that crop's train_core label "
                         "(crops.csv)")
    mism = [r for r in recs if r["reason"] == GEOMETRY_MISMATCH]
    return {"boxes": len(recs), "boxes_by_status": {s: int(status.get(s, 0)) for s in STATUSES},
            "no_embedding_reasons": {k: int(reasons.get(k, 0)) for k in REASONS if reasons.get(k)},
            "species": species, "otherplant": other_block([r["status"] for r in ot]),
            "per_species": per_species, "per_source": per_source,
            "conflict_pairs": dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))),
            "known_truth": truth,
            "geometry_mismatch_examples": [
                {"key": r["key"], "box": r["box"], "delta": round(r["geom_delta"], 6)}
                for r in mism[:EXAMPLES]]}


# ------------------------------------------------------------ the verifier
def fit_trusted(Xc, yc, gc):
    try:
        return V.fit_verifier(Xc, yc, gc, seed=0)
    except V.VerifyError as e:
        raise AuditError("fitting the probe on the trusted boxes: %s" % e)


def cv_baseline(Xc, yc, gc, n_folds=N_FOLDS):
    """Held-out trusted boxes judged by verifiers fitted, thresholds included,
    on the other folds: with their true labels (the baseline false-conflict
    rate) and with one seeded wrong species each (the conflict recall on
    wrong labels)."""
    try:
        outer = V._assign_folds(gc, n_folds, 0, random_ok=False)
    except V.VerifyError as e:
        raise AuditError("baseline CV over the trusted groups: %s" % e)
    nf = int(outer.max()) + 1
    st_true = np.full(len(yc), None, dtype=object)
    st_wrong = np.full(len(yc), None, dtype=object)
    folds = []
    for f in range(nf):
        tr, te = outer != f, outer == f
        if len(np.unique(gc[tr])) < 2:
            raise AuditError("baseline fold %d: its training part holds fewer than 2 groups; "
                             "the trusted set needs at least 3 capture sessions or sources" % f)
        try:
            ver = V.fit_verifier(Xc[tr], yc[tr], gc[tr], seed=f)
        except V.VerifyError as e:
            raise AuditError("baseline fold %d: %s" % (f, e))
        te_idx = np.flatnonzero(te)
        y = yc[te]
        rng = np.random.default_rng(C.stable_int(SWAP_SEED % f))
        wrong = (y + rng.integers(1, N_SPECIES, size=len(y))) % N_SPECIES
        v_true = ver.judge(y, Xc[te])[0]
        v_wrong = ver.judge(wrong, Xc[te])[0]
        seen = np.isin(y, ver.probe.classes_)
        seen_w = seen & np.isin(wrong, ver.probe.classes_)
        st_true[te_idx[seen]] = v_true[seen]
        st_wrong[te_idx[seen_w]] = v_wrong[seen_w]
        b = species_block(v_true[seen].tolist())
        w = species_block(v_wrong[seen_w].tolist())
        folds.append({"fold": f, "groups": int(len(np.unique(gc[te]))), "boxes": int(te.sum()),
                      "unseen_label": int((~seen).sum()), "false_conflict_rate": b["conflict_rate"],
                      "conflict_recall_on_wrong": w["conflict_rate"]})
        log("baseline fold %d/%d: %d held-out boxes, false conflict %s, conflict recall on a wrong "
            "species %s" % (f + 1, nf, int(te.sum()), b["conflict_rate"], w["conflict_rate"]))
    judged = np.array([s is not None for s in st_true])
    species = species_block([s for s in st_true if s is not None])
    wrong = species_block([s for s in st_wrong if s is not None])
    per_species = {}
    for k in range(N_SPECIES):
        m = (yc == k) & judged
        if m.any():
            blk = species_block(st_true[m].tolist())
            mw = (yc == k) & np.array([s is not None for s in st_wrong])
            bw = species_block(st_wrong[mw].tolist())
            blk["wrong_boxes_judged"] = bw["boxes_judged"]
            blk["conflict_recall_on_wrong"] = bw["conflict_rate"]
            per_species[C.CLASS_NAMES[k]] = blk
    return {"folds": nf, "per_fold": folds,
            "species": dict(species, false_conflict_rate=species["conflict_rate"]),
            "swapped": {"boxes_judged": wrong["boxes_judged"],
                        "conflict_recall_on_wrong": wrong["conflict_rate"],
                        "verified_on_wrong": wrong["verified_rate"],
                        "rule": "each held-out box given one wrong species, uniform over the other 11, "
                                "seeded by stable_int('%s' %% fold)" % SWAP_SEED},
            "unseen_label_boxes": int((~judged).sum()), "per_species": per_species}


def judge(ver, index, recs):
    """Fill the verdicts of the records that have a crop to judge."""
    todo = [r for r in recs if r["status"] is None]
    if not todo:
        return
    labels = np.array([r["label"] for r in todo], dtype=np.int64)
    ids = np.array([r["crop_id"] for r in todo], dtype=np.int64)
    v, j, pj, cj = ver.judge_features(labels, index.X[ids])
    seen = set(int(c) for c in ver.probe.classes_)
    for r, vv, jj, pp, cc in zip(todo, v, j, pj, cj):
        r["pred"], r["p"] = int(jj), float(pp)
        r["cosine"] = float(cc) if np.isfinite(cc) else None
        r["status"] = LABEL_UNSEEN if (r["label"] < OTHER and r["label"] not in seen) else str(vv)


# ------------------------------------------------------------------ the run
def _paths(out):
    out = Path(out)
    return out, out.with_suffix(".md"), out.with_name(out.stem + "_boxes.csv")


def _first_line(path):
    with open(path, errors="replace") as fh:
        return fh.readline().rstrip("\r\n")


def _is_audit_output(path, kind):
    """True when the existing file is one this module wrote (an earlier run)."""
    try:
        if kind == "json":
            with open(path) as fh:
                obj = json.load(fh)
            return isinstance(obj, dict) and obj.get("what") == WHAT
        return _first_line(path) == (MD_TITLE if kind == "md" else ",".join(BOX_FIELDS))
    except (OSError, ValueError):
        return False


def check_out(out, inputs):
    """Refuse an --out whose three files could overwrite anything but an
    earlier audit's outputs: --out must end in .json; no output may lie under
    the Step 1 directory (crops.csv, its inputs, the shards, the verifier) or
    be one of `inputs` (the manifests, their label files, the files crops.csv
    was made from); an existing file is overwritten only when it is an
    earlier audit output."""
    out_json, out_md, out_csv = _paths(out)
    if out_json.suffix != ".json":
        raise AuditError("--out %s: must end in .json (OUT.md and OUT_boxes.csv go beside it)" % out)
    step1 = Path(V.STEP1).resolve()
    inputs = {str(Path(p).resolve()) for p in inputs}
    for p, kind in ((out_json, "json"), (out_md, "md"), (out_csv, "csv")):
        rp = p.resolve()
        if rp == step1 or step1 in rp.parents:
            raise AuditError("--out: %s lies under the Step 1 directory %s; the audit never writes "
                             "there" % (p, step1))
        if str(rp) in inputs:
            raise AuditError("--out: %s is an input of this audit; it would overwrite it" % p)
        if p.exists() and not (p.is_file() and _is_audit_output(p, kind)):
            raise AuditError("--out: %s exists and is not an earlier audit output; it would overwrite "
                             "it" % p)


def run_audit(trusted_path, audits, out, nshards=None):
    """Audit each (name, manifest) against a probe fitted on trusted_path's
    boxes; write OUT.json, OUT.md and OUT_boxes.csv. Returns the result."""
    t0 = time.time()
    trusted_path = Path(trusted_path)
    out_json, out_md, out_csv = _paths(out)
    manifests = [trusted_path] + [Path(p) for _n, p in audits]
    check_out(out, manifests)
    t_rows = read_rows(trusted_path, "trusted")
    a_rows = [(name, Path(p), read_rows(p, name)) for name, p in audits]
    check_out(out, manifests + [r["label"] for rows in [t_rows] + [x for _n, _p, x in a_rows]
                                for r, _b in rows]
              + [C.manifest_path("train_core"), V.COPIES, V.POOL, V.POOL_META])
    log("trusted %s: %d images; audited %s"
        % (trusted_path, len(t_rows), {n: len(r) for n, _p, r in a_rows}))
    try:
        crops = V.Crops()
        fresh = V.check_fresh(crops)
        X, emb = V.load_embeddings(crops, nshards)
    except V.VerifyError as e:
        raise AuditError("Step 1 inputs: %s" % e)
    log("crops.csv: %d crops; embeddings %s" % (crops.n, emb))
    index = CropIndex(crops, X, [t_rows] + [r for _n, _p, r in a_rows])

    # the probe: trusted species boxes only, their own labels
    t_recs = resolve(index, t_rows)
    use = [r for r in t_recs if r["status"] is None and r["label"] < OTHER]
    if not use:
        raise AuditError("no trusted species box has a Step 1 embedding")
    Xc = V._norm(X[np.array([r["crop_id"] for r in use], dtype=np.int64)])
    yc = np.array([r["label"] for r in use], dtype=np.int64)
    gc = _groups(use)
    log("probe: %d trusted species boxes from %d groups; no OtherPlant class, by design (verify's "
        "OtherPlant sample could hold the audited images)" % (len(use), len(np.unique(gc))))
    ver = fit_trusted(Xc, yc, gc)
    baseline = cv_baseline(Xc, yc, gc)
    classes = set(int(c) for c in ver.probe.classes_)
    t_reasons = collections.Counter(r["reason"] for r in t_recs if r["status"] == NO_EMB)
    t_sessions = {r.get("session") for r, _b in t_rows if r.get("session")}
    group_kind = collections.Counter(g.split(":", 1)[0] for g in gc)
    trusted = {
        "manifest": str(trusted_path), "manifest_sha256": C.sha256_file(trusted_path),
        "images": len(t_rows), "boxes": len(t_recs),
        "species_boxes_used": len(use),
        "used_per_species": {C.CLASS_NAMES[k]: int((yc == k).sum()) for k in range(N_SPECIES)},
        "otherplant_boxes_left_out": sum(1 for r in t_recs if r["label"] == OTHER),
        "no_embedding": sum(t_reasons.values()),
        "no_embedding_reasons": {k: int(t_reasons.get(k, 0)) for k in REASONS if t_reasons.get(k)},
        "images_matched_by_sha256": len({r["key"] for r in t_recs if r["matched_by"] == "sha256"}),
        "groups": int(len(np.unique(gc))),
        "grouped_by": {"session": int(group_kind.get("session", 0)),
                       "source": int(group_kind.get("source", 0))},
    }
    pc = ver.info.get("per_class") or {}
    probe = {k: ver.info.get(k) for k in ("folds", "recall_target", "max_cond_pass", "threshold_rule",
                                          "n_species_crops", "cv_top1_species",
                                          "cv_recall_species_min", "cv_recall_species_mean")}
    probe.update({
        "classes": [C.CLASS_NAMES[k] for k in sorted(classes)],
        "species_not_in_trusted": [C.CLASS_NAMES[k] for k in range(N_SPECIES) if k not in classes],
        "species_never_verified": [C.CLASS_NAMES[k] for k in range(N_SPECIES)
                                   if k in classes and not np.isfinite(ver.tau_p[k])],
        "thresholds": {C.CLASS_NAMES[k]: {"tau_p": V._f(ver.tau_p[k]), "sigma": V._f(ver.sigma[k])}
                       for k in range(C.NC)},
        "per_class": {n: pc[n] for n in C.CLASS_NAMES if n in pc},
        "otherplant_class": False,
        "note": "verify.fit_verifier on the trusted species boxes only (no OtherPlant sample): the "
                "probe never predicts OtherPlant"})
    if probe["species_not_in_trusted"]:
        log("WARNING: the trusted boxes hold no %s: a box labelled so is label_unseen"
            % probe["species_not_in_trusted"])

    results, box_rows = {}, []
    t_ids = trusted_ids(index, t_rows)
    for name, path, rows in a_rows:
        recs = resolve(index, rows, trusted=t_ids)
        judge(ver, index, recs)
        blk = summarise(recs, baseline)
        refused = {r["key"] for r, _b in rows if in_trusted(index, r, t_ids)}
        shared = sorted({r.get("session") for r, _b in rows if r.get("session") in t_sessions})
        how = collections.Counter()
        for r, _b in rows:
            e, h = index.image(r)
            how[h if e is not None else "none"] += 1
        blk.update({
            "manifest": str(path), "manifest_sha256": C.sha256_file(path), "images": len(rows),
            "sources": dict(sorted(collections.Counter(r["source"] for r, _b in rows).items())),
            "boxes_per_class": {C.CLASS_NAMES[k]: sum(1 for r in recs if r["label"] == k)
                                for k in range(C.NC)},
            "images_refused_in_trusted": len(refused),
            "images_matched_by_sha256": int(how.get("sha256", 0)),
            "images_without_step1_image": int(how.get("none", 0)),
            "sessions_shared_with_trusted": shared,
            "rows_in_shared_sessions": sum(1 for r, _b in rows if r.get("session") in t_sessions)})
        results[name] = blk
        sp = blk["species"]
        mx = sp["baseline_for_mix"]
        log("%s: %d images, %d boxes; species boxes judged %d, conflict rate %s %s, verified %s; "
            "not judged %s" % (name, len(rows), len(recs), sp["boxes_judged"], sp["conflict_rate"],
                               sp["conflict_rate_ci95"], sp["verified_rate"],
                               {s: n for s, n in blk["boxes_by_status"].items()
                                if s not in JUDGED and n} or "none"))
        log("%s: baseline for its species mix f %s %s, r %s; noise estimate %s, above baseline %s "
            "(pooled baseline: %s, %s)" % (name, mx["false_conflict_rate"], mx["false_conflict_rate_ci95"],
                                           mx["conflict_recall_on_wrong"], sp["noise_estimate_corrected"],
                                           sp["above_baseline"], sp["noise_estimate_pooled_baseline"],
                                           sp["above_pooled_baseline"]))
        if mx["species_without_baseline"]:
            log("WARNING: %s: species without a baseline entry, left out of the noise estimate: %s"
                % (name, mx["species_without_baseline"]))
        if refused:
            log("WARNING: %s: %d image(s) are in the trusted manifest; refused" % (name, len(refused)))
        if shared:
            log("WARNING: %s: %d row(s) share a capture session with the trusted rows"
                % (name, blk["rows_in_shared_sessions"]))
        if blk["no_embedding_reasons"].get(GEOMETRY_MISMATCH):
            log("WARNING: %s: %d box(es) do not match their crop row's geometry; not judged"
                % (name, blk["no_embedding_reasons"][GEOMETRY_MISMATCH]))
        for r in recs:
            box_rows.append((name, r))

    import sklearn
    result = _clean({
        "built_utc": _utc(), "seconds": round(time.time() - t0, 1),
        "what": WHAT,
        "trusted": trusted, "probe": probe, "baseline_cv": baseline, "audits": results,
        "audit_order": list(results),
        "inputs": {"crops": {"path": str(crops.path), "sha256": crops.sha, "rows": crops.n},
                   "crops_made_from": fresh.get("inputs"), "embeddings": emb, "geom_tol": GEOM_TOL,
                   "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}},
        "outputs": {"json": str(out_json), "md": str(out_md), "boxes_csv": str(out_csv)},
        "notes": [
            "conflict_rate: conflicts / species-labelled boxes with a verdict (verified, conflict, "
            "unknown); the label-noise estimate. Boxes not judged (no_embedding, in_trusted, "
            "label_unseen) are counted, never guessed.",
            "baseline_cv: held-out trusted boxes judged by a verifier fitted, thresholds included, on "
            "the other folds (grouped by capture session, else source): the false-conflict rate f on "
            "true labels, and the conflict recall r under one uniform wrong species per box.",
            "baseline_for_mix: f and r of each species (baseline_cv.per_species) weighted by the "
            "audited set's judged species-labelled boxes per label; f's 95 % interval from seeded "
            "draws of each species' f from its Jeffreys posterior. Species without a baseline entry "
            "are listed and left out, with their boxes.",
            "noise_estimate_corrected = (c - f) / (r - f) with f and r of baseline_for_mix, clipped "
            "to [0, 1]; it assumes in-domain labels and uniform swaps, so it is indicative only for "
            "another domain.",
            "above_baseline: the conflict rate's Wilson 95 % interval lies wholly above "
            "baseline_for_mix's interval of f.",
            "noise_estimate_pooled_baseline, above_pooled_baseline: the same against the baseline "
            "pooled over the trusted set's own species mix; context only, since per-species f and r "
            "differ and an audited set's mix need not be the trusted set's.",
            "known_truth: boxes judged on a train_core crop, whose true label is the crop's "
            "train_core label."]})
    _write_boxes(out_csv, box_rows)
    _write_md(out_md, result)
    V._write_json(out_json, result)
    log("wrote %s, %s, %s (%.0fs)" % (out_json, out_md, out_csv, time.time() - t0))
    return result


# ----------------------------------------------------------------- outputs
def _fmt(x, digits=3):
    return "-" if x is None else ("%.*f" % (digits, x))


def _ci(ci):
    return "" if not ci else " [%.3f, %.3f]" % tuple(ci)


def _write_boxes(path, box_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(BOX_FIELDS)
        for name, r in box_rows:
            p = r["p"]
            wr.writerow([name, r["key"], r["image"], r["source"], r["session"], r["box"], r["label"],
                         C.CLASS_NAMES[r["label"]], r["status"], r["reason"] or "",
                         r["pred"] if r["pred"] >= 0 else "",
                         C.CLASS_NAMES[r["pred"]] if r["pred"] >= 0 else "",
                         "" if p is None or not np.isfinite(p) else "%.4f" % p,
                         "" if r["cosine"] is None else "%.4f" % r["cosine"],
                         r["crop_id"] if r["crop_id"] >= 0 else "", r["crop_set"], r["matched_by"],
                         "" if r["train_core_label"] is None else r["train_core_label"],
                         "" if r["geom_delta"] is None else "%.6f" % r["geom_delta"]])
    os.replace(tmp, path)


def _write_md(path, res):
    t, pr, bl = res["trusted"], res["probe"], res["baseline_cv"]
    L = [MD_TITLE, "",
         "Trusted: `%s` (%d images, %d species boxes used from %d groups: %d by session, %d by "
         "source). Probe: verify's 12-species logistic regression on BioCLIP-2 crop features, "
         "thresholds at %s per-species recall (%s-fold grouped CV); CV top-1 on species %s."
         % (t["manifest"], t["images"], t["species_boxes_used"], t["groups"],
            t["grouped_by"]["session"], t["grouped_by"]["source"], pr.get("recall_target"),
            pr.get("folds"), _fmt(pr.get("cv_top1_species"))), ""]
    if pr["species_not_in_trusted"]:
        L += ["Species absent from the trusted boxes (label_unseen): %s."
              % ", ".join(pr["species_not_in_trusted"]), ""]
    if pr["species_never_verified"]:
        L += ["Species with too few calibration crops (never verified): %s."
              % ", ".join(pr["species_never_verified"]), ""]
    L += ["The noise estimate and 'above baseline' use the baseline for each set's own species mix: "
          "the held-out false-conflict rate f (95 % CI) and the conflict recall r on a wrong species "
          "of each species, weighted by the set's judged species-labelled boxes per label. The "
          "pooled columns use the baseline over the trusted set's own mix, for context.", "",
          "| Set | Images | Species boxes judged | Conflict rate [95 % CI] | Verified | Unknown | "
          "Baseline f for its mix [95 % CI] | r for its mix | Noise estimate (corrected) | "
          "Above baseline | Pooled: estimate, above | OtherPlant conflict (n) | Not judged |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    s = bl["species"]
    L.append("| trusted, held-out CV (baseline) | %d | %d | %s%s | %s | %s | - | %s | - | - | - | - | %d |"
             % (t["images"], s["boxes_judged"], _fmt(s["conflict_rate"]), _ci(s["conflict_rate_ci95"]),
                _fmt(s["verified_rate"]), _fmt(s["unknown_rate"]),
                _fmt(bl["swapped"]["conflict_recall_on_wrong"]), bl["unseen_label_boxes"]))
    for name in res["audit_order"]:
        a = res["audits"][name]
        sp, ot = a["species"], a["otherplant"]
        mx = sp["baseline_for_mix"]
        nj = sum(n for st, n in a["boxes_by_status"].items() if st not in JUDGED)
        L.append("| %s | %d | %d | %s%s | %s | %s | %s%s | %s | %s | %s | %s, %s | %s (%d) | %d |"
                 % (name, a["images"], sp["boxes_judged"], _fmt(sp["conflict_rate"]),
                    _ci(sp["conflict_rate_ci95"]), _fmt(sp["verified_rate"]), _fmt(sp["unknown_rate"]),
                    _fmt(mx["false_conflict_rate"]), _ci(mx["false_conflict_rate_ci95"]),
                    _fmt(mx["conflict_recall_on_wrong"]), _fmt(sp["noise_estimate_corrected"]),
                    "yes" if sp["above_baseline"] else "no", _fmt(sp["noise_estimate_pooled_baseline"]),
                    "yes" if sp["above_pooled_baseline"] else "no",
                    _fmt(ot["conflict_rate"]), ot["boxes_judged"], nj))
    L += ["", "Baseline conflict recall on a wrong species (held-out trusted boxes, one uniform wrong "
          "species each): %s over %d boxes." % (_fmt(bl["swapped"]["conflict_recall_on_wrong"]),
                                                bl["swapped"]["boxes_judged"]), ""]
    nob = [(n, res["audits"][n]["species"]["baseline_for_mix"]["species_without_baseline"])
           for n in res["audit_order"]]
    nob = [(n, x) for n, x in nob if x]
    if nob:
        L += ["Species without a baseline entry, left out of the noise estimate (boxes): %s."
              % "; ".join("%s: %s" % (n, ", ".join("%s %d" % kv for kv in x.items())) for n, x in nob),
              ""]
    kt = [(n, res["audits"][n]["known_truth"]) for n in res["audit_order"]
          if res["audits"][n]["known_truth"]]
    if kt:
        L += ["## Known truth (boxes judged on a train_core crop)", "",
              "| Set | Boxes | True label error rate | Conflict recall on wrong labels | "
              "False conflict on right labels | Verified precision |", "|---|---|---|---|---|---|"]
        for n, k in kt:
            L.append("| %s | %d | %s | %s | %s | %s |"
                     % (n, k["judged"], _fmt(k["label_error_rate"]), _fmt(k["conflict_recall_on_wrong"]),
                        _fmt(k["false_conflict_rate_on_correct"]), _fmt(k["verified_precision"])))
        L.append("")
    names = list(res["audit_order"])
    L += ["## Conflict rate per species (of the label given; boxes judged)", "",
          "| Species | baseline f (n) | baseline r | " + " | ".join(names) + " |",
          "|---|---|---|" + "---|" * len(names)]
    for k in range(C.NC):
        cn = C.CLASS_NAMES[k]
        b = bl["per_species"].get(cn)
        cells = (["%s (%d)" % (_fmt(b["conflict_rate"]), b["boxes_judged"]),
                  _fmt(b.get("conflict_recall_on_wrong"))] if b else ["-", "-"])
        for n in names:
            x = res["audits"][n]["per_species"].get(cn)
            cells.append("%s (%d)" % (_fmt(x["conflict_rate"]), x["boxes_judged"]) if x else "-")
        if any(c != "-" for c in cells):
            L.append("| %s | %s |" % (cn, " | ".join(cells)))
    L += ["", "## Boxes not judged", "",
          "| Set | in_trusted | label_unseen | no_embedding | reasons |", "|---|---|---|---|---|"]
    for name in res["audit_order"]:
        a = res["audits"][name]
        st = a["boxes_by_status"]
        L.append("| %s | %d | %d | %d | %s |"
                 % (name, st[IN_TRUSTED], st[LABEL_UNSEEN], st[NO_EMB],
                    ", ".join("%s %d" % kv for kv in a["no_embedding_reasons"].items()) or "-"))
    L += ["", "## Notes", ""] + ["- " + n for n in res["notes"]]
    L += ["", "Per-box verdicts: `%s` (beside this file)." % Path(res["outputs"]["boxes_csv"]).name, ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\n".join(L))
    os.replace(tmp, path)


# --------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.audit",
                                 description="INC attribution item 4: BioCLIP-2 label audit of "
                                             "manifests against a probe fitted on a trusted one")
    ap.add_argument("--trusted", required=True, help="manifest whose boxes (and labels) fit the probe")
    ap.add_argument("--audit", required=True, nargs="+", action="extend", metavar="NAME=MANIFEST",
                    help="manifests to audit, each judged on its own label files")
    ap.add_argument("--out", required=True, help="OUT.json; OUT.md and OUT_boxes.csv go beside it")
    ap.add_argument("--nshards", type=int, default=None,
                    help="the embedding shard set to read (default: the complete, current one)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        run_audit(a.trusted, parse_audits(a.audit), a.out, nshards=a.nshards)
    except AuditError as e:
        log("FAILED: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
