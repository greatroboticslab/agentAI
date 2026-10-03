"""Base v3 for experiment E1: every admitted weed box as INC class 12
(docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-03): E1, weed-box base v3
(pre-registered)").

    python -m weed_optimizer_framework.tools.inc2.base3 build --stream SID [--procs N] [--testing]
    python -m weed_optimizer_framework.tools.inc2.base3 count --stream SID --out DIR [--procs N]

build (a GPU job, run_inc2_build.sh, lever L23V) writes INC_DIR/splits/v3/;
count (the login node, read-only) runs the same selection on stored hashes
and writes one report to --out, nothing under INC_DIR.

Why. The 12-class cwd12 test score (0.8786, YOLO11m at 640 on base_v2's
6,811 images) is capped by box quality: class-agnostic it is 0.8901, and
five larger or newer detectors moved it by nothing. Data has only ever been
added in hundreds of images. E1 asks whether a much larger, cleaned,
leak-checked weed-box training set improves box quality. Every weed box of
every admitted source is labelled class 12 (OtherPlant) in the existing
13-class INC space, so the models are one-class weed detectors that every
inc2 check, the pinned driver and the locked scorer accept; their
class-agnostic dev mAP50-95 compares directly with b_v2_m640's.

Inputs (all recorded by sha256 in summary.json):
  * the pre-registered config base3_v2.json beside this module (CONFIG):
    the allow-list of sources, each class name's role (weed or drop), the
    families and provider tiers, each source's file-name capture group
    (group_regex, registry and intake sources alike), every rule's numbers;
  * splits v2's base_v2.jsonl (by LOCK v2's sha256) and its provenance
    (dHash and 8 variants of each row);
  * the dataset registry (REPO/results/framework/dataset_registry.json):
    each listed slug's local_path, images paired with labels as
    mega_trainer pairs them (images/ -> labels/, then the same directory,
    then a labels/ directory above), class names from data.yaml, else the
    registry's class_names; a slug whose registry path lies under the
    intake directory is never read through the registry;
  * the committed intake batches of the listed intake sources
    (INC_DIR/intake/<batch>/manifest.jsonl): their label ids are INC ids
    (0-13), every one a weed box here. A row with intake holds is admitted
    only when Step 1's queue (inc2.stream.QueueView, the queue's one
    reader) holds it with no hold left;
  * the stream's source quarantines (--stream SID: inc2.stream's ledger
    fold, its head recorded): a quarantined source's rows never enter arm B.
    The quarantine is an input, never a constant here, and it is applied
    after the holdout (below), so test v1 does not depend on it. The same
    fold gives the rows the stream has trained on or may still pool: every
    pool's rows and every row of an increment in flight (cut, suspect, or
    any status that has not released its rows), a masked row also by its
    original's path and sha256. The build reads the fold again once its
    selection is made (a segment the stream cut while the job ran: marked,
    and the selection made again) and once more, holding the stream's
    lease, after the test lists are written (a listed row now in a pool or
    in flight refuses the build);
  * the evaluation groups' images (config "evaluation_groups": test v1's
    groups and the OOD-dev groups, every slug excluded): their 8-variant
    dHashes from the v1 Step 1 pool, else hashed from the registry's
    local_path (build and count alike);
  * every test list an earlier build wrote (INC_DIR/splits/*/test_v1/
    *.jsonl, whatever its directory is now called) and its companions
    (splits/*/test_v1_companions/*.jsonl): never trained.

The rules (pre-registered; the config's "rules"). base_v2 rows are exempt
from every box, image and source rule: arm A is base_v2 whole.
  * Classes: a box of a 'weed' class becomes class 12, a box of a 'drop'
    class is removed (background); a class id with no name, or a name the
    config does not list, refuses the image (unmapped_class). A polygon
    line becomes its bounding box; coordinates are clipped to [0, 1].
  * Box side: sqrt(w x h) in pixels after the image's long side is scaled
    to 640 (one definition for every rule).
  * Source admission, over its weed boxes as delivered: median side >= 32
    px, at most 25 % under 16 px, at most 10 % covering > 80 % of their
    image, median weed boxes per image <= 25. A failing source is excluded
    whole (convention).
  * Per box: side < 8 px -> removed and its rectangle filled with the
    image's mean colour (inc2.mask.masked_array; kept boxes' pixels are
    never filled).
  * Per image, dropped when: no weed box; a weed box covers > 90 % of it;
    its shorter side < 320 px; the masked area > 50 %.
  * Dedupe (never 4-6 bits): the same original bytes; or dHash within 3
    bits under one of the 8 flips and rotations AND >= 80 % of the larger
    box set matched one-to-one at IoU >= 0.8 after that transform; or an
    identical label layout (rounded to 0.01, >= 3 boxes) whose dHashes lie
    within 10 bits under one of the 8 variants. One row kept per
    copy group: base_v2 rows always (an external copy of one is dropped),
    else the lowest tier, then the most pixels, then the slug, then the key.
  * Leak guard (build): GuardV2 and the index cross-check on the exact
    files written (inc2.train.guard_verdicts), the L-5 and L-8 lists on the
    original bytes, the calibrated embedding copy detector
    (GuardV2.check_embed, LOCK v2's calibration) on every external row, and
    D28-v2 per source (inc2.eval_hits pair cosines of its dHash hits, the
    embedding binomial rule, the base-copy share; thresholds from
    inc_autopilot/stream_thresholds.json): a refused row is dropped, a new
    row that copies base_v2 (base_copy) is dropped, a source that leaks is
    excluded whole. A base_v2 row the guard refuses for a never-train reason
    leaves both arms (expected none). count weighs stored hashes of the
    originals and reports the embedding check and the pair cosines as
    pending (dHash hits unweighed).
  * Evaluation-group guard: a candidate row within 6 dHash bits (GuardV2's
    never-train radius) under the 8 variants, in both directions, of any
    image of an evaluation group is dropped (near_eval_group): never in arm
    B, never in test v1. summary.json reports the rows within 3 and within
    6 bits per group and source (base_v2's counted, never dropped).
  * Earlier test lists: a row an earlier build's test list holds (the same
    original or written bytes, the same key, a shared capture relation, or
    within 6 dHash bits under the 8 variants) never trains.
  * Main-test holdout v1, computed before the quarantine is applied (every
    step up to it ignores the quarantine, so test v1 is the same whichever
    sources are quarantined): capture groups join images within 6 dHash
    bits under the 8 variants (both directions), their copy edges, the same
    Roboflow export stem (across a family), the intake capture group and a
    source's group_regex session (a registry source's on its file names, an
    intake source's on its rows' original file names: SIU's video; an
    intake row its group_regex does not match is dropped, group_unmatched).
    A group is named by its own content (the
    sha256 of its members' sorted sha256s), never by row positions. From
    each included external source (quarantined or not), whole groups it
    owns are held out in the order stable_int("inc2/base3/test_v1/<source>/
    <group digest>") -- a group holding a row of an earlier test list first
    -- until ~15 % of its kept rows (min 30, max 400 images) are held; a
    source's count includes its rows held in other sources' groups, and a
    group is taken only while every source with rows in it stays at or
    under 400. A group touching base_v2 or a row of the stream's pools (an
    image a current model trained on) or of an increment in flight (one a
    pool may still take) -- also through a capture relation it shares with
    such a row that a rule dropped before the grouping --, or larger than
    400 images, is never held out (summary.json counts each source's rows
    in such groups by reason). Held-out rows never train and are listed in
    splits/v3/test_v1/<source>.jsonl (with their capture relations) for a
    later never-train index v3 and for every later build; every other row
    sharing a capture relation with a held-out row (dropped before the
    grouping, or a duplicate in a held-out group) is listed in
    splits/v3/test_v1_companions/<source>.jsonl: never trained, and never
    cut by the stream (Step 1's queue reads both, step1_stream.test_v1_rows).
  * The quarantine: a quarantined source's rows that are not held out
    leave arm B (quarantined); a copy group whose kept row is quarantined
    keeps its best copy from a source that is not.
  * Family cap (dock): after the quarantine, D <= floor(0.35 / 0.65 x
    non-dock images), enforced by dropping whole capture groups in the order
    stable_int("inc2/base3/test_v1/cap/dock/<group digest>").

Materialisation (build). An image whose long side exceeds 640, or that has a
masked box, is written as a lossless PNG of exactly the pixels Ultralytics
trains on: its own BaseDataset.load_image (imread, the long side resized to
640 with INTER_LINEAR) applied to the original, then the mask. Ultralytics'
RAM cache holds those pixels, so a cached run of the original and a run of
the PNG train on the same arrays, and a 30K-image base needs no 12 MP
decode per sample. Every written PNG is read back through load_image and
must equal them. Other images are used as they are. PNGs live under
splits/v3/images/<sha[:2]>/<sha256>.png, labels (class 12 lines) under
splits/v3/labels/<sha[:2]>/<sha256>.txt, both content-addressed and
written once; base_v2 rows are materialised the same way in both arms, so
arm A's rows are byte-identical inside arm B.

Outputs (INC_DIR/splits/v3/; a manifest is never overwritten with other
bytes, and summary.json is written last):
  base_v2_weed.jsonl     arm A (inc.common manifest rows)
  base_v3_weed.jsonl     arm B (A plus the admitted external rows)
  provenance_v1.jsonl    per row: original path and sha256, the file
                         trained on, labels, masks, dHash and variants,
                         group, licence
  test_v1/<source>.jsonl the held-out rows of each source (original and
                         written file, dHash and variants)
  test_v1_companions/<source>.jsonl
                         rows sharing a held-out row's capture relation,
                         never test rows (never trained, never cut)
  dropped_v1.jsonl       every row not admitted, with its reason
  summary.json           status complete; the arms (manifest path, sha256,
                         images, boxes, distinct photos); per source: seen,
                         admitted, boxes, masked, dropped by reason,
                         deduped, held out (holdout_v1), capped, guard,
                         leak verdict; the inputs' sha256s; no key named
                         after a non-dev split (the snapshot ships it)
inc2.baseline build trains cold_budget on exactly the two manifests a
complete summary records (by sha256), and refuses any other build on a
manifest under splits/v3 or one any base v3 summary records (v3_claim);
inc2.train runs cold_budget only for such an E1 arm (e1_problems).
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..inc import common as C
from ..inc.splits import sanitise
from . import common as C2

FORMAT = "inc2-base3-summary/1"
COUNT_FORMAT = "inc2-base3-count/1"
CONFIG_FORMAT = "inc2-base3-config/1"
CONFIG = Path(__file__).resolve().with_name("base3_v2.json")
VERSION = "v3"
ARM_A, ARM_B = "base_v2_weed", "base_v3_weed"
ARMS = {"A": ARM_A, "B": ARM_B}
SUMMARY = "summary.json"
PROVENANCE = "provenance_v1.jsonl"
DROPPED = "dropped_v1.jsonl"
HOLDOUT_DIR = "test_v1"
# rows that share a capture relation with a held-out row but are not test rows themselves (dropped before the
# grouping, or a held group's duplicates): never trained, never cut (prior_test_lists reads them with the test lists)
COMPANION_DIR = "test_v1_companions"
STREAM_LEASE_WAIT_S = 1800              # build: how long the last pool check waits for the stream's lease
IMGSZ = 640
WEED_ID = 12
VARIANTS = ("id", "hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "transverse")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
RF_STEM_RE = re.compile(r"^(?P<stem>.+?)_(?:jpe?g|png|bmp)\.rf\.[0-9a-f]{16,64}$", re.I)
BASE_KIND, INTAKE_KIND, REGISTRY_KIND = "base", "intake", "registry"
# GuardV2 reasons of a dHash hit on an evaluation image (inc2.eval_hits.DHASH_HIT_REASONS)
DHASH_HIT_REASONS = ("near_eval_v2", "near_eval_variant")
COUNT_HASH_MAX_BYTES = 1500000          # count: an image without stored hashes is hashed only below this size


class Base3Error(RuntimeError):
    """A condition under which base v3 must not be built."""


def log(msg):
    print("[inc2.base3] %s" % msg, flush=True)


def out_dir():
    return Path(C.INC_DIR) / "splits" / VERSION


def registry_path():
    return Path(C.REPO) / "results" / "framework" / "dataset_registry.json"


def thresholds_path():
    return Path(__file__).resolve().parents[1] / "inc_autopilot" / "stream_thresholds.json"


def _sha_file(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def _sha_text(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def round_half_up(x):
    return int(math.floor(float(x) + 0.5))


def _median(xs):
    xs = sorted(xs)
    k = len(xs)
    if not k:
        return None
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


# ------------------------------------------------------------------- config
def load_config(path=None):
    """(config, sha256) of the pre-registered config; refuses a malformed one."""
    path = Path(path or CONFIG)
    try:
        with open(path) as fh:
            conf = json.load(fh)
    except (OSError, ValueError) as e:
        raise Base3Error("cannot read the base v3 config %s: %s" % (path, e))
    if conf.get("format") != CONFIG_FORMAT:
        raise Base3Error("%s is not a %s config" % (path, CONFIG_FORMAT))
    if int(conf.get("class_id", -1)) != WEED_ID:
        raise Base3Error("the config's class id %r is not %d" % (conf.get("class_id"), WEED_ID))
    for k in ("rules", "sources", "intake", "excluded", "base_v2"):
        if not isinstance(conf.get(k), dict):
            raise Base3Error("%s has no %r block" % (path, k))
    both = sorted(set(conf["sources"]) & set(conf["excluded"]))
    if both:
        raise Base3Error("%s lists %s as included and excluded" % (path, both))
    fams = conf["rules"].get("families") or {}
    for s, e in conf["sources"].items():
        cls = e.get("classes")
        if not isinstance(cls, dict) or not cls or set(cls.values()) - {"weed", "drop"}:
            raise Base3Error("source %s: classes must map names to 'weed' or 'drop'" % s)
        if "weed" not in cls.values():
            raise Base3Error("source %s has no weed class" % s)
        if e.get("family") is not None and e["family"] not in fams:
            raise Base3Error("source %s: family %r has no rule" % (s, e["family"]))
        _group_regex(s, e)
    for s, e in conf["intake"].items():
        if e.get("classes") != "all_weed":
            raise Base3Error("intake source %s: only 'all_weed' is supported" % s)
        if e.get("family") is not None and e["family"] not in fams:
            raise Base3Error("intake source %s: family %r has no rule" % (s, e["family"]))
        _group_regex(s, e)
    dd = conf["rules"].get("dedupe") or {}
    if not isinstance(dd.get("layout_max_bits"), int) or dd["layout_max_bits"] < 0:
        raise Base3Error("%s: rules.dedupe.layout_max_bits must be a whole number of bits" % path)
    eg = conf.get("evaluation_groups")
    if not isinstance(eg, dict) or not isinstance(eg.get("groups"), dict) or not eg["groups"] \
            or not isinstance(eg.get("bits"), int):
        raise Base3Error("%s has no evaluation_groups block (bits and groups)" % path)
    for g, slugs in eg["groups"].items():
        if not isinstance(slugs, list) or not slugs:
            raise Base3Error("evaluation group %s lists no slug" % g)
        inside = sorted(set(slugs) - set(conf["excluded"]))
        if inside:
            raise Base3Error("evaluation group %s lists %s, which the config does not exclude" % (g, inside))
    return conf, C.sha256_file(path)


def _group_regex(source, entry):
    """The compiled group_regex of a config entry (None without one); a
    pattern that does not compile or captures no group refuses the config."""
    if not entry.get("group_regex"):
        return None
    try:
        rx = re.compile(entry["group_regex"])
    except re.error as e:
        raise Base3Error("source %s: group_regex %r does not compile (%s)" % (source, entry["group_regex"], e))
    if rx.groups < 1:
        raise Base3Error("source %s: group_regex %r captures no group" % (source, entry["group_regex"]))
    return rx


# ------------------------------------------------------------------ geometry
def parse_label(path):
    """([(cls, cx, cy, w, h)], polygons, problems) of a YOLO label file: a
    polygon line becomes its bounding box; every box is clipped to [0, 1];
    a box of zero area after clipping is left out (counted as a problem
    only when its line is malformed)."""
    boxes, problems, polys, degenerate = [], [], 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        return [], 0, ["unreadable: %s" % e]
    for i, ln in enumerate(text.splitlines(), 1):
        t = ln.split()
        if not t:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            problems.append("line %d is not numeric" % i)
            continue
        if not all(math.isfinite(x) for x in v) or v[0] != int(v[0]) or v[0] < 0:
            problems.append("line %d has a bad class or value" % i)
            continue
        c = int(v[0])
        if len(v) == 5:
            cx, cy, w, h = v[1:]
            x0, x1, y0, y1 = cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0
        elif len(v) >= 7 and (len(v) - 1) % 2 == 0:
            xs, ys = v[1::2], v[2::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            polys += 1
        else:
            problems.append("line %d has %d fields" % (i, len(v)))
            continue
        x0, x1 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))
        y0, y1 = max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))
        if x1 <= x0 or y1 <= y0:
            degenerate += 1
            continue
        boxes.append((c, (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0))
    return boxes, polys, problems


def side_px(box, W, H):
    """sqrt(w x h) of a normalised box (cx, cy, w, h) in pixels at 640 (long side)."""
    s = IMGSZ / float(max(W, H))
    return math.sqrt(max(0.0, box[2] * W * s) * max(0.0, box[3] * H * s))


def transform_box(box, variant):
    """A normalised box (cx, cy, w, h) under one of funnel.leak's 8 variants
    (the transform whose dHash dhash_variants gives under that name)."""
    cx, cy, w, h = box
    return {"id": (cx, cy, w, h), "hflip": (1 - cx, cy, w, h), "vflip": (cx, 1 - cy, w, h),
            "rot180": (1 - cx, 1 - cy, w, h), "transpose": (cy, cx, h, w), "rot90": (cy, 1 - cx, h, w),
            "rot270": (1 - cy, cx, h, w), "transverse": (1 - cy, 1 - cx, h, w)}[variant]


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def layout_match(a, b, variant, iou_min, share):
    """True when, after transforming a's boxes by variant, at least share of
    max(|a|, |b|) boxes pair one-to-one (greedy by IoU) at IoU >= iou_min."""
    ta = [transform_box(x, variant) for x in a]
    n = max(len(ta), len(b))
    if n == 0:
        return False
    pairs = sorted(((box_iou(x, y), i, j) for i, x in enumerate(ta) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, m = set(), set(), 0
    for u, i, j in pairs:
        if u < iou_min:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        m += 1
    return m >= share * n - 1e-12


def layout_key(boxes, nd=2):
    return tuple(sorted(tuple(round(float(v), nd) for v in b) for b in boxes))


# ------------------------------------------------------------- pair search
def _popcount(x):
    import numpy as np
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    m1, m2, m4, h01 = (np.uint64(0x5555555555555555), np.uint64(0x3333333333333333),
                       np.uint64(0x0F0F0F0F0F0F0F0F), np.uint64(0x0101010101010101))
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int64)


def _blocks(nbits):
    n = nbits + 1
    widths = [64 // n + (1 if i < 64 % n else 0) for i in range(n)]
    out, shift = [], 0
    for w in widths:
        out.append((shift, (1 << w) - 1))
        shift += w
    return out


def _reduce_pairs(Q, T, D, V, n):
    """The arrays with one entry per (Q, T): the fewest bits, then the lowest variant."""
    import numpy as np
    if not len(Q):
        return Q, T, D, V
    key = Q.astype(np.int64) * int(n) + T.astype(np.int64)
    order = np.lexsort((V, D, key))
    key = key[order]
    first = np.r_[True, key[1:] != key[:-1]]
    sel = order[first]
    return Q[sel], T[sel], D[sel], V[sel]


def _pair_search(HQ, okQ, HT, bits, same):
    """near_pairs' search: every (q, t) with popcount(HQ[q, v] ^ HT[t, 0])
    <= bits for some variant v (v 0 only when okQ[q] is False), one entry
    per pair (the fewest bits, then the lowest variant); with same (HQ is
    HT) the pairs q == t are left out. Pigeonhole blocks: two 64-bit hashes
    within `bits` agree exactly on one of bits + 1 blocks."""
    import numpy as np
    HQ = np.asarray(HQ, dtype=np.uint64).reshape(-1, 8)
    HT = np.asarray(HT, dtype=np.uint64).reshape(-1, 8)
    ok = np.asarray(okQ, dtype=bool).reshape(-1)
    nq, nt = len(HQ), len(HT)
    z = np.zeros(0, dtype=np.int64)
    acc = (z, z, z, z)
    if nq == 0 or nt == 0 or (same and nq < 2):
        return acc
    for shift, mask in _blocks(bits):
        sh, mk = np.uint64(shift), np.uint64(mask)
        kid = (HT[:, 0] >> sh) & mk
        order = np.argsort(kid, kind="stable")
        ks = kid[order]
        parts = [[acc[0]], [acc[1]], [acc[2]], [acc[3]]]
        for v in range(8):
            q = np.arange(nq) if v == 0 else np.flatnonzero(ok)
            if not len(q):
                continue
            kq = (HQ[q, v] >> sh) & mk
            lo = np.searchsorted(ks, kq, side="left")
            hi = np.searchsorted(ks, kq, side="right")
            has = np.flatnonzero(hi > lo)
            if not len(has):
                continue
            keys = kq[has]
            oq = np.argsort(keys, kind="stable")
            hs, keys = has[oq], keys[oq]
            starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            ends = np.r_[starts[1:], len(keys)]
            for a, b in zip(starts.tolist(), ends.tolist()):
                qi = q[hs[a:b]]
                ti = order[lo[hs[a]]:hi[hs[a]]]
                for c0 in range(0, len(qi), 1024):
                    qc = qi[c0:c0 + 1024]
                    d = _popcount(HQ[qc, v][:, None] ^ HT[ti, 0][None, :])
                    rr, cc = np.nonzero(d <= bits)
                    if len(rr):
                        parts[0].append(qc[rr].astype(np.int64))
                        parts[1].append(ti[cc].astype(np.int64))
                        parts[2].append(d[rr, cc].astype(np.int64))
                        parts[3].append(np.full(len(rr), v, dtype=np.int64))
        Q, T, D, V = (np.concatenate(p) for p in parts)
        if same:
            keep = Q != T
            Q, T, D, V = Q[keep], T[keep], D[keep], V[keep]
        acc = _reduce_pairs(Q, T, D, V, nt)
    return acc


def near_pairs(H, var_ok, bits):
    """(Q, T, D, V) int64 arrays, one entry per ordered pair q != t of rows
    with popcount(H[q, v] ^ H[t, 0]) <= bits: D the fewest bits, V the
    variant index (VARIANTS order) giving them, the lowest on a tie. H:
    [N, 8] uint64, column 0 the dHash; a row with var_ok False is queried by
    its dHash only."""
    return _pair_search(H, var_ok, H, bits, True)


def near_pairs_between(HQ, okQ, HT, bits):
    """near_pairs from one set (HQ, queried under its variants) to another
    (HT, by its dHash): (Q, T, D, V) with Q indexing HQ and T indexing HT."""
    return _pair_search(HQ, okQ, HT, bits, False)


def cross_nearest(HA, okA, HB, okB, bits):
    """(dist, arg): for every row of A the fewest bits to a row of B under
    the 8 variants in both directions (A's variants against B's dHash, B's
    variants against A's dHash), bits + 1 when none is within `bits`, and
    that row of B (-1 when none)."""
    import numpy as np
    na = len(np.asarray(HA).reshape(-1, 8))
    best = np.full(na, bits + 1, dtype=np.int64)
    arg = np.full(na, -1, dtype=np.int64)
    if na == 0 or len(np.asarray(HB).reshape(-1, 8)) == 0:
        return best, arg
    Q1, T1, D1, _V1 = near_pairs_between(HA, okA, HB, bits)
    T2, Q2, D2, _V2 = near_pairs_between(HB, okB, HA, bits)
    a = np.r_[Q1, Q2].astype(np.int64)
    b = np.r_[T1, T2].astype(np.int64)
    d = np.r_[D1, D2].astype(np.int64)
    if len(a):
        order = np.lexsort((b, d, a))
        a, b, d = a[order], b[order], d[order]
        first = np.r_[True, a[1:] != a[:-1]]
        best[a[first]] = d[first]
        arg[a[first]] = b[first]
    return best, arg


def hash_matrix(rows):
    """(H [N, 8] uint64, var_ok [N], has [N]) of rows: the 8 variants when
    known, else the dHash alone (var_ok False); has False without a hash."""
    import numpy as np
    H = np.zeros((len(rows), 8), dtype=np.uint64)
    ok = np.zeros(len(rows), dtype=bool)
    has = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        if r.get("variants") is not None:
            H[i] = np.asarray([int(x) for x in r["variants"]], dtype=np.uint64)
            ok[i] = has[i] = True
        elif r.get("dhash") is not None:
            H[i, 0] = np.uint64(int(r["dhash"]))
            has[i] = True
    return H, ok, has


def hash_dist(a, b):
    """The fewest dHash bits between two rows under the 8 variants in both
    directions (the dHashes alone when neither has variants); None when
    either has no hash."""
    if a.get("dhash") is None or b.get("dhash") is None:
        return None
    da, db = int(a["dhash"]), int(b["dhash"])
    best = bin(da ^ db).count("1")
    for v in (a.get("variants") or ())[1:]:
        best = min(best, bin(int(v) ^ db).count("1"))
    for v in (b.get("variants") or ())[1:]:
        best = min(best, bin(int(v) ^ da).count("1"))
    return best


def _components(n, Q, T):
    """Connected-component labels of n nodes under the undirected edges (Q, T)."""
    import numpy as np
    if not len(Q):
        return np.arange(n)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = coo_matrix((np.ones(len(Q), dtype=np.int8), (Q, T)), shape=(n, n)).tocsr()
    return connected_components(m, directed=False)[1]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb

    def groups(self, members):
        out = collections.defaultdict(list)
        for m in members:
            out[self.find(m)].append(m)
        return out


def hash_table(rows):
    """(unique hash rows UH [U, 8], their var_ok, row -> unique index (-1:
    no hash), {unique index: [rows]}). Rows with identical hashes (all 8, or
    the dHash alone when the variants are unknown) share one entry."""
    import numpy as np
    keyrows, uidx = {}, []
    for i, r in enumerate(rows):
        if r["variants"] is not None:
            k = (tuple(int(x) for x in r["variants"]), True)
        elif r["dhash"] is not None:
            k = ((int(r["dhash"]),) + (0,) * 7, False)
        else:
            uidx.append(-1)
            continue
        if k not in keyrows:
            keyrows[k] = len(keyrows)
        uidx.append(keyrows[k])
    keys = list(keyrows)
    UH = np.asarray([k[0] for k in keys], dtype=np.uint64).reshape(-1, 8)
    Uok = np.asarray([k[1] for k in keys], dtype=bool)
    members = collections.defaultdict(list)
    for i, u in enumerate(uidx):
        if u >= 0:
            members[u].append(i)
    return UH, Uok, np.asarray(uidx, dtype=np.int64), members


def hash_components(rows, bits):
    """A component label per row: rows within `bits` dHash bits under the 8
    variants (both directions) share one; a row without a hash is alone."""
    import numpy as np
    UH, Uok, uidx, _m = hash_table(rows)
    lab_u = _components(len(UH), *near_pairs(UH, Uok, bits)[:2]) if len(UH) else np.zeros(0, dtype=np.int64)
    out = np.empty(len(rows), dtype=np.int64)
    base = int(lab_u.max()) + 1 if len(lab_u) else 0
    for i, u in enumerate(uidx.tolist()):
        out[i] = lab_u[u] if u >= 0 else base + i
    return out


# ------------------------------------------------------------- enumeration
def _names_of(entry, root):
    """The class names of a registry dataset: data.yaml's, else the registry's."""
    for cand in ("data.yaml", "dataset.yaml"):
        p = Path(root) / cand
        if p.is_file():
            try:
                import yaml
                with open(p) as fh:
                    d = yaml.safe_load(fh) or {}
            except Exception as e:  # noqa: BLE001 - an unreadable yaml falls back to the registry, recorded
                log("WARNING: %s unreadable (%s); the registry's class names are used" % (p, e))
                break
            names = d.get("names")
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=lambda x: int(x))], "yaml:%s" % cand
            if isinstance(names, list):
                return [str(x) for x in names], "yaml:%s" % cand
    names = entry.get("class_names")
    if isinstance(names, str):
        try:
            names = ast.literal_eval(names)
        except (ValueError, SyntaxError):
            names = None
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names], "registry"
    return None, "none"


def _label_for(img, root, files=None):
    """mega_trainer._find_label_for_image's pairing (images/ -> labels/, the
    same directory, then a labels/ directory above, up to the dataset root).
    files: the set of every file path under root (one walk), so the pairing
    makes no system call per image on Lustre."""
    img = Path(img)
    have = (lambda q: str(q) in files) if files is not None else (lambda q: q.exists())
    if "images" in img.parts:
        lab = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if have(lab):
            return lab
    same = img.with_suffix(".txt")
    if have(same):
        return same
    for parent in img.parents:
        cand = parent / "labels" / (img.stem + ".txt")
        if have(cand):
            return cand
        if parent == Path(root):
            break
    return None


def _list_files(root):
    """(images, the set of every file path) under root, one walk."""
    out, every = [], set()
    for d, dirs, files in os.walk(str(root)):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__MACOSX")
        for f in sorted(files):
            every.add(os.path.join(d, f))
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(Path(d) / f)
    return out, every


def _header_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - an unreadable image is dropped as unreadable
        return None


def _stem_of(path):
    m = RF_STEM_RE.match(Path(path).stem)
    return m.group("stem") if m else None


def _new_row(source, kind, conf_src, image, label, key, family=None, tier=3):
    return {"source": source, "kind": kind, "family": family, "tier": int(tier), "image": str(image),
            "label": None if label is None else str(label), "key": key, "W": None, "H": None, "bytes": None,
            "boxes": [], "mask": [], "n_raw": 0, "n_weed": 0, "n_drop": 0, "polygons": 0, "session": "",
            "capture": None, "stem": _stem_of(image), "group_key": None, "sha256": None, "dhash": None,
            "variants": None, "drop": None, "licence": None, "research_only": None, "train_image": None,
            "train_sha256": None, "masked_area": 0.0, "guard": None}


def base_rows(conf, lock=None, production=True):
    """base_v2's rows (LOCK v2's manifest, by sha256), every box class 12;
    with their dHash, variants and licence from base_v2_provenance.jsonl."""
    d = C2.SPLITS_DIR
    path, prov = d / "base_v2.jsonl", d / "base_v2_provenance.jsonl"
    lock = lock or C2.read_lock_v2()
    want = (lock.get("manifests") or {}).get("base_v2")
    got = _sha_file(path)
    if not want or got != want:
        raise Base3Error("%s does not hash as LOCK v2 records base_v2 (%s, %s)" % (path, str(got)[:12],
                                                                                  str(want)[:12]))
    pinfo = {}
    psha = _sha_file(prov)
    if psha is not None and (lock.get("provenance_sha256") in (None, psha)):
        for r in C.read_manifest(prov):
            pinfo[r.get("key")] = r
    elif production:
        raise Base3Error("%s is missing or does not hash as LOCK v2 records it" % prov)
    out = []
    tier = int((conf.get("base_v2") or {}).get("tier", 0))
    for r in C.read_manifest(path):
        row = _new_row(r["source"], BASE_KIND, None, r["image"], r["label"], r["key"], tier=tier)
        row["sha256"], row["session"] = r["sha256"], r.get("session") or ""
        row["label_sha256"] = r["label_sha256"]
        p = pinfo.get(r["key"]) or {}
        if p.get("dhash") is not None:
            row["dhash"] = int(p["dhash"])
        if isinstance(p.get("variants"), list) and len(p["variants"]) == 8:
            row["variants"] = [int(x) for x in p["variants"]]
        row["licence"], row["research_only"] = p.get("licence"), p.get("research_only")
        boxes = C.read_yolo(r["label"])
        row["n_raw"] = row["n_weed"] = len(boxes)
        row["boxes"] = [tuple(float(x) for x in b[1:5]) for b in boxes]
        out.append(row)
    return out, {"path": str(path), "sha256": got, "provenance_sha256": psha, "rows": len(out)}


def intake_rows(conf, holds_view=None):
    """The listed intake sources' committed batches (INC_DIR/intake/<batch>/
    manifest.jsonl), every label id a weed box. A row with intake holds is
    admitted only when Step 1's queue holds it with none left. A source with
    a group_regex sets each row's file-name session as a registry source's
    rows have it (group_key '<source>|<group 1>'), matched on the row's
    original file name (its 'rel' in the source, else its image's name): the
    intake's own capture group may be the single image (SIU's frames), and
    the session joins every frame of a video into one capture group. A row
    whose name the pattern does not match cannot be placed with its video,
    in test v1 or in arm B, so it is dropped (group_unmatched); each batch's
    record counts the matched and unmatched rows per source."""
    idir = Path(C.INC_DIR) / "intake"
    rxs = {s: _group_regex(s, e) for s, e in conf["intake"].items()}
    out, rec = [], {}
    try:
        batches = sorted(p for p in os.listdir(idir) if (idir / p / "manifest.jsonl").is_file())
    except OSError:
        batches = []
    for b in batches:
        mp = idir / b / "manifest.jsonl"
        rows = C.read_manifest(mp)
        srcs = {r.get("source") for r in rows}
        mine = [r for r in rows if r.get("source") in conf["intake"]]
        if not mine:
            continue
        rec[b] = {"manifest_sha256": C.sha256_file(mp), "rows": len(mine), "sources": sorted(s for s in srcs if s)}
        grx = {}
        for r in mine:
            src = r["source"]
            row = _new_row(src, INTAKE_KIND, conf["intake"][src], r["image"], r["label"], r["key"],
                           family=conf["intake"][src].get("family"), tier=conf["intake"][src].get("tier", 1))
            row["sha256"], row["session"] = r.get("sha256"), str(r.get("session") or r.get("capture_group") or "")
            row["capture"] = "%s|%s" % (src, r.get("capture_group")) if r.get("capture_group") else None
            unmatched = False
            if rxs.get(src) is not None:
                m = rxs[src].search(os.path.basename(str(r.get("rel") or r["image"])))
                row["group_key"] = "%s|%s" % (src, m.group(1)) if m else None
                unmatched = m is None
                g = grx.setdefault(src, {"pattern": rxs[src].pattern, "matched": 0, "unmatched": 0})
                g["unmatched" if unmatched else "matched"] += 1
            row["W"], row["H"] = r.get("width"), r.get("height")
            row["dhash"] = int(r["dhash"]) if r.get("dhash") is not None else None
            row["licence"], row["research_only"] = r.get("licence"), r.get("research_only")
            row["batch"] = b
            holds = [h for h in (r.get("holds") or []) if h]
            if holds:
                # released only when Step 1's queue holds the row with no hold left
                left = None if holds_view is None else holds_view.get(r["key"])
                row["intake_holds"] = holds
                if left is None or left:
                    row["drop"] = "intake_hold"
            if unmatched and row["drop"] is None:
                row["drop"] = "group_unmatched"     # its video is unknown: neither test v1 nor arm B can take it
            out.append(row)
        if grx:
            rec[b]["group_regex"] = grx
    return out, rec


def registry_rows(conf, registry, sources=None):
    """The listed registry slugs' images, paired with their labels; (rows,
    per-source records). A slug under the intake directory is refused here."""
    ents = (registry or {}).get("datasets", registry) or {}
    idir = str(Path(C.INC_DIR) / "intake")
    out, rec = [], {}
    for slug in sorted(sources if sources is not None else conf["sources"]):
        sc = conf["sources"][slug]
        e = ents.get(slug)
        if not isinstance(e, dict):
            rec[slug] = {"error": "not in the registry"}
            continue
        root = e.get("local_path")
        if not root or not os.path.isdir(root):
            rec[slug] = {"error": "local_path %r is not a directory" % root}
            continue
        if str(Path(root).resolve()).startswith(idir):
            rec[slug] = {"error": "its registry path is an intake directory: read through the intake only"}
            continue
        names, basis = _names_of(e, root)
        unknown = sorted(set(names or []) - set(sc["classes"]))
        rec[slug] = {"local_path": root, "names": names, "names_basis": basis, "unknown_names": unknown,
                     "registry_status": e.get("status"),
                     "licence": (e.get("provenance") or {}).get("license") if isinstance(e.get("provenance"), dict)
                     else None}
        if names is None or unknown:
            rec[slug]["error"] = ("no class names" if names is None else
                                  "class names %s are not in the config" % unknown)
            continue
        cache = e.get("dhash_cache") or {}
        if isinstance(cache, str):
            try:
                cache = ast.literal_eval(cache)
            except (ValueError, SyntaxError):
                cache = {}
        t0 = time.time()
        imgs, every = _list_files(root)
        rec[slug]["images"] = len(imgs)
        log("registry %s: %d images listed in %.0fs" % (slug, len(imgs), time.time() - t0))
        rx = _group_regex(slug, sc)
        if rx is not None:
            rec[slug]["group_regex"] = {"pattern": rx.pattern, "matched": 0, "unmatched": 0}
        seen_keys = set()
        for p in imgs:
            rel = str(p.relative_to(root))
            key = sanitise("%s__%s" % (slug, os.path.splitext(rel)[0]))
            if key in seen_keys:
                key = "%s_%s" % (key, hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8])
            seen_keys.add(key)
            row = _new_row(slug, REGISTRY_KIND, sc, p, _label_for(p, root, every), key, family=sc.get("family"),
                           tier=sc.get("tier", 3))
            row["names"] = names
            row["rel"] = rel
            if rel in cache and cache[rel] is not None:
                row["dhash"] = int(cache[rel])
            if rx is not None:
                m = rx.search(p.name)
                row["group_key"] = "%s|%s" % (slug, m.group(1)) if m else None
                rec[slug]["group_regex"]["matched" if m else "unmatched"] += 1
            row["licence"] = rec[slug]["licence"]
            out.append(row)
    return out, rec


# ------------------------------------------------------------------- rules
def map_boxes(row, conf):
    """The row's weed boxes (class 12) from its label file; sets n_raw,
    n_weed, n_drop, polygons, or drop when the label cannot be read or a
    class is not mapped."""
    if row["kind"] == BASE_KIND:
        return
    if not row["label"] or not os.path.isfile(row["label"]):
        row["drop"] = row["drop"] or "no_label"
        return
    boxes, polys, probs = parse_label(row["label"])
    row["polygons"] = polys
    row["n_raw"] = len(boxes)
    if probs:
        row["drop"] = row["drop"] or "label_error"
        return
    weed = []
    if row["kind"] == INTAKE_KIND:
        weed = [b[1:] for b in boxes]
    else:
        classes = conf["sources"][row["source"]]["classes"]
        names = row.get("names") or []
        for b in boxes:
            c = b[0]
            role = classes.get(names[c]) if c < len(names) else None
            if role is None:
                row["drop"] = row["drop"] or "unmapped_class"
                return
            if role == "weed":
                weed.append(b[1:])
            else:
                row["n_drop"] += 1
    row["boxes"] = weed
    row["n_weed"] = len(weed)


def source_convention(rows, rules):
    """The source rule's statistics over its weed boxes as delivered (images
    with a weed box and a known size) and whether it passes."""
    sides, big, per_image = [], 0, []
    for r in rows:
        if not r["boxes"] or not r["W"] or not r["H"]:
            continue
        per_image.append(len(r["boxes"]))
        for b in r["boxes"]:
            sides.append(side_px(b, r["W"], r["H"]))
            if b[2] * b[3] > rules["big_area"]:
                big += 1
    n = len(sides)
    med = _median(sides)
    small = sum(1 for s in sides if s < rules["small_side_px"])
    out = {"boxes": n, "images": len(per_image), "median_side_px": None if med is None else round(med, 2),
           "small_share": round(small / float(n), 4) if n else None,
           "big_share": round(big / float(n), 4) if n else None,
           "median_boxes_per_image": _median(per_image)}
    fails = []
    if not n:
        fails.append("no weed box")
    else:
        if med < rules["median_side_min_px"]:
            fails.append("median side %.1f px < %s" % (med, rules["median_side_min_px"]))
        if out["small_share"] > rules["small_share_max"]:
            fails.append("%.1f %% of boxes under %s px > %.0f %%" % (100 * out["small_share"], rules["small_side_px"],
                                                                      100 * rules["small_share_max"]))
        if out["big_share"] > rules["big_share_max"]:
            fails.append("%.1f %% of boxes over %.0f %% of the image > %.0f %%"
                         % (100 * out["big_share"], 100 * rules["big_area"], 100 * rules["big_share_max"]))
        if out["median_boxes_per_image"] > rules["median_boxes_max"]:
            fails.append("median %s boxes per image > %s" % (out["median_boxes_per_image"], rules["median_boxes_max"]))
    out["passes"], out["fails"] = not fails, fails
    return out


def image_rules(row, rules):
    """The per-box and per-image rules on one external row: masked boxes and
    the drop reason, if any."""
    if row["kind"] == BASE_KIND or row["drop"]:
        return
    if not row["W"] or not row["H"]:
        row["drop"] = "unreadable"
        return
    if not row["boxes"]:
        row["drop"] = "no_weed_box"
        return
    ir, br = rules["image"], rules["box"]
    if min(row["W"], row["H"]) < ir["min_short_side_px"]:
        row["drop"] = "short_side"
        return
    if any(b[2] * b[3] > ir["drop_box_area"] for b in row["boxes"]):
        row["drop"] = "box_over_90"
        return
    keep = [b for b in row["boxes"] if side_px(b, row["W"], row["H"]) >= br["mask_side_px"]]
    mask = [b for b in row["boxes"] if side_px(b, row["W"], row["H"]) < br["mask_side_px"]]
    row["boxes"], row["mask"] = keep, mask
    if mask:
        row["masked_area"] = masked_area(mask, keep, row["W"], row["H"])
    if not keep:
        row["drop"] = "no_weed_box_after_mask"
    elif row["masked_area"] > ir["max_masked_area"]:
        row["drop"] = "masked_over_50"


def size_640(W, H):
    """(w, h) of an image after Ultralytics' load_image at 640 (rect mode)."""
    r = IMGSZ / float(max(W, H))
    if r == 1:
        return int(W), int(H)
    return min(int(math.ceil(W * r)), IMGSZ), min(int(math.ceil(H * r)), IMGSZ)


def masked_area(mask, keep, W, H):
    """The share of the image inc2.mask.masked_array fills at 640 (masked
    rectangles minus kept ones, inc2.mask.pixel_rect's rounding), from the
    boxes alone."""
    import numpy as np
    from . import mask as MK
    w, h = size_640(W, H)
    fill = np.zeros((h, w), dtype=bool)
    kept = np.zeros((h, w), dtype=bool)
    for boxes, arr in ((mask, fill), (keep, kept)):
        for b in boxes:
            x0, y0, x1, y1 = MK.pixel_rect(b, w, h)
            if x1 > x0 and y1 > y0:
                arr[y0:y1, x0:x1] = True
    return round(float((fill & ~kept).sum()) / float(w * h), 6)


# ------------------------------------------------------------------- hashes
def pool_hashes(inc_dir=None):
    """({image path: [8 variants]}, {image path: sha256}, record) of the v1
    Step 1 pool (funnel/emb_dinov2_images_pool.npz and step1/pool.jsonl):
    stored hashes, empty when the files are missing."""
    inc = Path(inc_dir or C.INC_DIR)
    npz, pool = inc / "funnel" / "emb_dinov2_images_pool.npz", inc / "step1" / "pool.jsonl"
    by_path, sha_by_path = {}, {}
    if not (npz.is_file() and pool.is_file()):
        return by_path, sha_by_path, {"pool_npz": None, "hashed": 0}
    import numpy as np
    path_of = {}
    with open(pool) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                path_of[r["key"]] = os.path.normpath(r["image"])
                sha_by_path[os.path.normpath(r["image"])] = r.get("sha256")
    with np.load(str(npz), allow_pickle=False) as z:
        keys, H = z["keys"], z["H"]
        ok = z["hash_ok"] if "hash_ok" in z.files else np.ones(len(keys), bool)
    for k, h, o in zip(keys.tolist(), H, ok):
        q = path_of.get(str(k))
        if q is not None and bool(o):
            by_path[q] = [int(x) for x in h]
    return by_path, sha_by_path, {"pool_npz": str(npz), "pool_manifest": str(pool), "hashed": len(by_path)}


class StoredHashes:
    """count's hashes: what is stored (the v1 Step 1 pool's 8 variants and
    sha256 by image path, the registry's dHash cache, intake and base_v2
    records); an image with none is hashed when it is small, else its
    variants stay unknown (pending). The original's sha256 is read for every
    row that has none (the holdout ranks groups by their members' bytes)."""

    def __init__(self, inc_dir=None, compute_max_bytes=COUNT_HASH_MAX_BYTES):
        self.by_path, self.sha_by_path, self.record = pool_hashes(inc_dir)
        self.compute_max_bytes = compute_max_bytes

    def fill(self, row):
        key = os.path.normpath(row["image"])
        vs = self.by_path.get(key)
        if vs is not None:
            row["dhash"], row["variants"] = int(vs[0]), list(vs)
        if row["sha256"] is None:
            row["sha256"] = self.sha_by_path.get(key) or _sha_file(row["image"])
        try:
            size = os.path.getsize(row["image"])
        except OSError:
            size = None
        row["bytes"] = size
        if row["variants"] is None:
            if size is not None and size <= self.compute_max_bytes:
                from . import guard as G
                h, v = G.image_hashes(row["image"])
                if h is not None and v is not None:
                    row["dhash"] = int(h)
                    row["variants"] = [int(v[k]) for k in VARIANTS]
                    row["hashed_here"] = True


# ------------------------------------------------------ evaluation groups
def eval_group_images(conf, registry):
    """{group: {slug: [image paths]}} of the config's evaluation groups (the
    test v1 groups and the OOD-dev groups, every slug excluded), and
    {slug: why not read} (not in the registry, no directory)."""
    ents = (registry or {}).get("datasets", registry) or {}
    out, missing = {}, {}
    for g, slugs in sorted(((conf.get("evaluation_groups") or {}).get("groups") or {}).items()):
        out[g] = {}
        for slug in slugs:
            e = ents.get(slug)
            root = e.get("local_path") if isinstance(e, dict) else None
            if not root or not os.path.isdir(root):
                missing[slug] = "not in the registry" if not isinstance(e, dict) else "local_path %r is not a " \
                                                                                     "directory" % root
                continue
            out[g][slug] = [str(x) for x in _list_files(root)[0]]
    return out, missing


def eval_group_hashes(conf, registry, stored=None, compute=True, max_bytes=None, procs=8):
    """({group: (H [n, 8] uint64, var_ok [n])}, record): every image of each
    evaluation group, its 8 variants from the Step 1 pool's stored hashes
    when held there, else hashed here (every image when max_bytes is None,
    as build and count call it)."""
    import numpy as np
    from . import guard as G
    imgs, missing = eval_group_images(conf, registry)
    by_path = stored if stored is not None else pool_hashes()[0]
    out, rec = {}, {"groups": {}, "slugs_not_read": missing}
    for g, slugs in imgs.items():
        H, per = [], {}
        for slug, paths in sorted(slugs.items()):
            have = [by_path.get(os.path.normpath(p)) for p in paths]
            todo = [p for p, h in zip(paths, have) if h is None]
            if todo and compute:
                def one(p):
                    try:
                        if max_bytes is not None and os.path.getsize(p) > max_bytes:
                            return None
                    except OSError:
                        return None
                    h, v = G.image_hashes(p)
                    return None if h is None or v is None else [int(h)] + [int(v[k]) for k in VARIANTS[1:]]
                with ThreadPoolExecutor(max_workers=max(1, int(procs))) as ex:
                    got = dict(zip(todo, ex.map(one, todo)))
            else:
                got = {}
            vals = [h if h is not None else got.get(p) for p, h in zip(paths, have)]
            H.extend(v for v in vals if v is not None)
            per[slug] = {"images": len(paths), "stored": sum(1 for h in have if h is not None),
                         "hashed_here": sum(1 for p in todo if got.get(p) is not None),
                         "unhashed": sum(1 for v in vals if v is None)}
        arr = np.asarray(H, dtype=np.uint64).reshape(-1, 8)
        out[g] = (arr, np.ones(len(arr), dtype=bool))
        rec["groups"][g] = {"hashes": int(len(arr)), "slugs": per}
    return out, rec


def eval_guard(rows, groups, bits):
    """Every candidate row (drop None) against each evaluation group's
    hashes: row["eval_group"] = {group: bits} for each group within `bits`
    under the 8 variants (both directions); an external row within `bits` of
    any group is dropped (near_eval_group: never in arm B, never in test
    v1). Returns the record (per group and source: rows within 3 bits and
    within `bits`; base_v2 rows, which stay, counted apart)."""
    import numpy as np
    cand = [r for r in rows if r["drop"] is None]
    H, ok, has = hash_matrix(cand)
    idx = np.flatnonzero(has)
    rec = {"bits": bits, "judged": int(len(idx)), "unjudged": int(len(cand) - len(idx)), "groups": {}}
    for r in cand:
        r.pop("eval_group", None)
    for g, (HG, okG) in sorted(groups.items()):
        d, _a = cross_nearest(H[idx], ok[idx], HG, okG, bits)
        by = collections.defaultdict(lambda: {"within_3": 0, "within": 0})
        for k, dist in zip(idx.tolist(), d.tolist()):
            if dist <= bits:
                r = cand[k]
                r.setdefault("eval_group", {})[g] = int(dist)
                s = r["source"] if r["kind"] != BASE_KIND else "base_v2"
                by[s]["within"] += 1
                by[s]["within_3"] += 1 if dist <= 3 else 0
        rec["groups"][g] = {"hashes": int(len(HG)), "by_source": {s: dict(v) for s, v in sorted(by.items())}}
    n = 0
    for r in cand:
        if r.get("eval_group") and r["kind"] != BASE_KIND:
            r["drop"] = "near_eval_group"
            n += 1
    rec["dropped"] = n
    return rec


def merge_eval_records(hashed, guarded):
    """eval_group_hashes' record (each group's hashes, per slug) and
    eval_guard's (each group's rows within range, per source) as one."""
    out = dict(guarded, slugs_not_read=hashed.get("slugs_not_read"), groups={})
    for g in sorted(set(hashed.get("groups") or {}) | set(guarded.get("groups") or {})):
        out["groups"][g] = dict((hashed.get("groups") or {}).get(g) or {}, **((guarded.get("groups") or {}).get(g) or {}))
    return out


# ------------------------------------------------------ earlier test lists
def prior_test_lists(splits_dir=None):
    """(rows, record) of every test list an earlier base build wrote
    (<splits>/*/test_v1/*.jsonl, whatever the directory is now called) and
    of its companions (<splits>/*/test_v1_companions/*.jsonl: rows sharing a
    capture relation with a held-out row, never test rows themselves): the
    rows never train in a later build (mark_prior, select), and Step 1's
    queue never offers them to the cutter (step1_stream.test_v1_rows)."""
    root = Path(splits_dir or (Path(C.INC_DIR) / "splits"))
    rows, files = [], []
    for f in sorted(list(root.glob("*/%s/*.jsonl" % HOLDOUT_DIR)) + list(root.glob("*/%s/*.jsonl" % COMPANION_DIR))):
        try:
            got = C.read_manifest(f)
        except Exception as e:  # noqa: BLE001 - an unreadable test list cannot be honoured: refuse
            raise Base3Error("the earlier test list %s cannot be read (%s): its rows' status is unknown" % (f, e))
        rows.extend(got)
        files.append({"file": str(f), "sha256": C.sha256_file(f), "rows": len(got)})
    return rows, {"files": files, "rows": len(rows)}


def _load_640(path):
    """(BGR array, (h0, w0)) of an image exactly as Ultralytics' own
    BaseDataset.load_image gives it (rect mode at 640: its imread, the long
    side resized with its interpolation), called on a stand-in dataset."""
    import cv2
    from ultralytics.data.base import BaseDataset
    stub = types.SimpleNamespace(ims=[None], im_files=[str(path)], npy_files=[Path(str(path) + ".base3-none.npy")],
                                 imgsz=IMGSZ, cv2_flag=cv2.IMREAD_COLOR, augment=False, prefix="", cache=None,
                                 buffer=[], max_buffer_length=0, im_hw0=[None], im_hw=[None])
    im, hw0, _hw = BaseDataset.load_image(stub, 0)
    return im, hw0


def _store_bytes(path, data):
    path = Path(path)
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
            return False
        raise Base3Error("%s exists with other bytes: content-addressed files are written once" % path)
    C2.write_bytes_atomic(path, data)
    return True


def materialise(row, root):
    """build: the file the row trains on (module docstring) and its hashes;
    sets train_image, train_sha256, sha256 (original), dhash, variants."""
    import cv2
    from . import guard as G
    from . import mask as MK
    sha = C.sha256_file(row["image"])
    if row["sha256"] is not None and row["sha256"] != sha:
        row["drop"] = "sha_mismatch"
        return row
    row["sha256"] = sha
    W, H = int(row["W"]), int(row["H"])
    if max(W, H) > IMGSZ or row["mask"]:
        im, _hw0 = _load_640(row["image"])
        if im is None:
            row["drop"] = "unreadable"
            return row
        if row["mask"]:
            im, _region, _kept, _mean = MK.masked_array(im, row["mask"], row["boxes"])
        ok, buf = cv2.imencode(".png", im)
        if not ok:
            raise Base3Error("PNG encoding failed for %s" % row["image"])
        data = buf.tobytes()
        psha = hashlib.sha256(data).hexdigest()
        dst = Path(root) / "images" / psha[:2] / ("%s.png" % psha)
        row["png_written"] = _store_bytes(dst, data)
        back, _ = _load_640(dst)
        if back is None or back.shape != im.shape or not (back == im).all():
            raise Base3Error("%s does not read back as the pixels it was written from" % dst)
        row["train_image"], row["train_sha256"] = str(dst), psha
    else:
        row["train_image"], row["train_sha256"] = row["image"], sha
    h, v = G.image_hashes(row["train_image"])
    if h is None or v is None:
        row["drop"] = "unhashable"
        return row
    row["dhash"], row["variants"] = int(h), [int(v[k]) for k in VARIANTS]
    return row


# -------------------------------------------------------------------- guard
def load_thresholds():
    """D28's thresholds from inc_autopilot/stream_thresholds.json (one source of truth)."""
    p = thresholds_path()
    with open(p) as fh:
        th = json.load(fh)
    d = th.get("D28") or {}
    out = {}
    for k in ("source_alpha", "confirm_cos", "p_confirmed", "base_copy_share"):
        v = d.get(k)
        out[k] = float(v["value"] if isinstance(v, dict) else v)
    return out, C.sha256_file(p)


def leak_verdict(n, dh, cos, emb, p_false, copy_cos, base_copies, th):
    """D28-v2 on one source (diagnose_stream.d28's judge plus its base-copy
    rule): eval_hits.verdict on its dHash hits, embed_calibration.
    source_verdict on its embedding hits, base copies >= base_copy_share."""
    from . import embed_calibration as EC
    from . import eval_hits as EH
    dv = EH.verdict(n, dh, cos, th["confirm_cos"], copy_cos, th["p_confirmed"], th["source_alpha"])
    ev = EC.source_verdict(n, emb, 0, 0, p_false, alpha=th["source_alpha"])
    share = (base_copies / float(n)) if n else 0.0
    why = list(dv["why"]) + list(ev["why"])
    if share >= th["base_copy_share"] - 1e-12 and base_copies:
        why.append("%.1f %% base copies >= %.0f %%" % (100 * share, 100 * th["base_copy_share"]))
    return {"images": n, "dhash": dv, "embed": ev, "base_copy_share": round(share, 4), "flagged": bool(why),
            "why": why, "describe": EH.describe(dv)}


def run_guard(rows, production=True, guards=None):
    """GuardV2 and the index cross-check over every row's file (its stored
    dHash and variants), the L-5 / L-8 lists over its original sha256:
    sets row["guard"] = {"reasons", "refused", "crosscheck", "match"}. Rows
    without variants are 'pending' (count) and are not judged."""
    from . import train as T
    guards = guards or T.load_v2_guards(production=production)
    judged = [r for r in rows if r["variants"] is not None and r["dhash"] is not None]
    items = [{"key": "u%d" % r["uid"], "image": "u%d" % r["uid"], "sha256": r["sha256"] or ""} for r in judged]
    dh = {"u%d" % r["uid"]: int(r["dhash"]) for r in judged}
    var = {"u%d" % r["uid"]: dict(zip(VARIANTS, r["variants"])) for r in judged}
    per, rec, _ref, _cc = T.guard_verdicts(items, dh, production=production, guards=guards, variants=var)
    for r in judged:
        v = per["u%d" % r["uid"]]
        r["guard"] = {"reasons": list(v["reasons"]), "refused": bool(v["refused"]),
                      "refused_by": [x[1] for x in v["refused"]], "crosscheck": v["crosscheck"] is not None,
                      "match": v.get("match")}
    for r in rows:
        if r["guard"] is None:
            r["guard"] = {"pending": True, "reasons": [], "refused": False, "refused_by": [], "crosscheck": False}
    return guards, rec


def embed_scanner(lock_path=None, production=True, embedder=None, procs=1, cache_dir=None):
    """(EmbedScanner or None, record): LOCK v2's calibration and an
    evaluation index made by its embedder (the splits v2 cache when it is
    current for these rows, else one under splits/v3/leak). A production
    build without a calibration refuses."""
    from . import embed_calibration as EC
    from . import guard as G
    lock_path = Path(lock_path or C2.LOCK_PATH)
    cal, why = EC.locked(lock_path, production=production)
    if cal is None:
        return None, {"checked": False, "why": why}
    if embedder is None:
        from ..funnel import embed as FE
        model, _, pooling = str(cal["embedder"]).rpartition(":")
        embedder = FE.LazyEmbedder(model, pooling or "cls")
    rows = G.v2_eval_rows(lock_path)
    index, used = None, None
    caches = (lock_path.parent / "leak" / "eval_desc_v2.npz", Path(cache_dir or out_dir() / "leak") / "eval_desc.npz")
    for cache in caches:
        try:
            index = G.eval_index(rows, embedder, cache, procs=procs)
            used = str(cache)
            break
        except Exception as e:  # noqa: BLE001 - a cache made from other rows is never overwritten: the next one
            log("evaluation descriptors not taken from %s (%s: %s)" % (cache, type(e).__name__, str(e)[:200]))
    if index is None:
        raise Base3Error("the evaluation index cannot be built: the embedding check cannot run (fail closed)")
    sc = G.EmbedScanner(index, cal)
    return sc, dict(sc.record(), checked=True, eval_cache=used, cos_threshold=cal["cos_threshold"],
                    p_false=cal.get("p_false"), calibration_file=cal["file"])


def embed_and_weigh(rows, scanner, guards, procs=1, desc_path=None):
    """The embedding check on every external row that passed GuardV2, and the
    pair cosine of every dHash hit (inc2.eval_hits): sets row["embed"] and
    row["pair_cos"]."""
    from . import eval_hits as EH
    guard = guards[1]
    ext = [r for r in rows if r["kind"] != BASE_KIND and not r["drop"] and not r["guard"]["refused"]]
    items = [{"key": "u%d" % r["uid"], "image": r["train_image"], "sha256": r["train_sha256"]} for r in ext]
    got = guard.check_embed(items, scanner, desc_path=desc_path, procs=procs) if items else {}
    for r in ext:
        v = got.get("u%d" % r["uid"])
        r["embed"] = v[0] if v else "pass"
    hits = [r for r in rows if r["kind"] != BASE_KIND and any(x in DHASH_HIT_REASONS for x in r["guard"]["reasons"])]
    index = scanner.index
    pos = {(str(s), str(k)): j for j, (s, k) in enumerate(zip(index.split, index.eval_key))}

    def eval_desc(split, key):
        j = pos.get((split, key))
        return None if j is None else index.Xn[j]
    its = []
    for r in hits:
        m = r["guard"].get("match") if isinstance(r["guard"].get("match"), dict) else {}
        also = EH.eval_matches(guard, dict(zip(VARIANTS, r["variants"])))
        its.append({"key": "u%d" % r["uid"], "source": r["source"], "image": r["train_image"],
                    "split": m.get("split"), "eval_key": m.get("key"), "also": also})
    scored = EH.pair_cosines(its, index.embedder, eval_desc=eval_desc, procs=procs) if its else {}
    for r in hits:
        r["pair_cos"] = (scored.get("u%d" % r["uid"]) or {}).get("pair_cos")
    return len(items), len(its)


# ---------------------------------------------------------------- selection
def _rank(r):
    """Which copy is kept: base_v2 first, then the lowest provider tier, the
    most pixels, the slug and the key."""
    return (0 if r["kind"] == BASE_KIND else 1, int(r["tier"]), -int((r["W"] or 0) * (r["H"] or 0)),
            r["source"], r["key"])


def _exact_key(r):
    """The original bytes' sha256; count's stand-in without one: the dHash,
    the size and the file length together."""
    if r["sha256"]:
        return "sha:%s" % r["sha256"]
    if r["dhash"] is not None and r["bytes"]:
        return "proxy:%s:%sx%s:%s" % (r["dhash"], r["W"], r["H"], r["bytes"])
    return None


def _identity(r):
    """A row's content identity: its original's sha256 (its key without one)."""
    return ("sha:%s" % r["sha256"]) if r.get("sha256") else ("key:%s" % r["key"])


def copy_components(live, dd):
    """UnionFind over the rows: exact copies, near copies (dHash within
    near_bits under a variant AND the layout matches after it) and identical
    layouts (rounded to layout_round, at least layout_min_boxes boxes) whose
    dHashes lie within layout_max_bits under a variant (module docstring).
    Returns (uf, record)."""
    n = len(live)
    uf = UnionFind(n)
    exact = {}
    for i, r in enumerate(live):
        k = _exact_key(r)
        if k is not None:
            if k in exact:
                uf.union(exact[k], i)
            else:
                exact[k] = i
    UH, Uok, uidx, members = hash_table(live)
    # rows sharing every hash: compare each with the earlier ones (variant id)
    taken = 0
    for u, rs in members.items():
        for a in range(1, len(rs)):
            for b in range(a):
                i, j = rs[a], rs[b]
                if uf.find(i) != uf.find(j) and layout_match(live[i]["boxes"], live[j]["boxes"], "id",
                                                              dd["layout_iou"], dd["layout_share"]):
                    uf.union(i, j)
                    taken += 1
    if len(UH):
        Q, T, _D, V = near_pairs(UH, Uok, dd["near_bits"])
        for q, t, v in zip(Q.tolist(), T.tolist(), V.tolist()):
            for i in members[q]:
                for j in members[t]:
                    if uf.find(i) != uf.find(j) and layout_match(live[i]["boxes"], live[j]["boxes"], VARIANTS[v],
                                                                  dd["layout_iou"], dd["layout_share"]):
                        uf.union(i, j)
                        taken += 1
    nd = 2 if abs(float(dd["layout_round"]) - 0.01) < 1e-12 else 4
    lay = collections.defaultdict(list)
    lay_taken, lay_far = 0, 0
    for i, r in enumerate(live):
        if len(r["boxes"]) >= dd["layout_min_boxes"]:
            k = layout_key(r["boxes"], nd)
            for j in lay[k]:
                d = hash_dist(r, live[j])
                if d is None or d > dd["layout_max_bits"]:
                    lay_far += 1
                    continue
                if uf.find(i) != uf.find(j):
                    uf.union(i, j)
                    lay_taken += 1
            lay[k].append(i)
    return uf, {"near_layout_edges": taken, "layout_edges": lay_taken, "layout_pairs_too_far": lay_far}


def capture_keys(r):
    """The capture relations of a row besides its hash: its export stem (per
    family, else per source), its intake capture group, its file-name session."""
    ks = []
    if r.get("stem"):
        ks.append(("stem", r.get("family") or r["source"], r["stem"]))
    if r.get("capture"):
        ks.append(("capture", r["capture"]))
    if r.get("group_key"):
        ks.append(("session", r["group_key"]))
    return ks


def capture_groups(live, copies, bits):
    """({row: group name}, {group name: digest}) (module docstring): 6-bit
    hash components, copy components, the same export stem (per family, else
    per source), the intake capture group and the group_regex session. A
    group is named by its members' content (the sha256 of their sorted
    identities), never by row positions, so a group keeps its name and its
    place in the holdout's order whatever else is read."""
    n = len(live)
    ug = UnionFind(n)
    for comp in copies:
        for i in comp[1:]:
            ug.union(comp[0], i)
    first = {}
    for i, (r, h) in enumerate(zip(live, hash_components(live, bits).tolist())):
        for k in [("hash", h)] + capture_keys(r):
            if k in first:
                ug.union(first[k], i)
            else:
                first[k] = i
    out, digest = {}, {}
    for _root, mem in ug.groups(range(n)).items():
        dg = hashlib.sha256("\n".join(sorted(_identity(live[i]) for i in mem)).encode("utf-8")).hexdigest()
        name = "g%s" % dg[:16]
        digest[name] = dg
        for i in mem:
            out[i] = name
    return out, digest


def mark_prior(rows, prior, bits):
    """prior_test on every candidate row that a test list of an earlier build
    holds (inc2.base3 test_v1/*.jsonl under any splits directory): the same
    original or written bytes, the same key, a shared capture relation, or a
    dHash within `bits` under the 8 variants (both directions). Such a row
    never trains (select). Returns the number marked."""
    cand = [r for r in rows if r["drop"] is None]
    for r in cand:
        r.pop("prior_test", None)
    if not prior or not cand:
        return 0
    shas, keys, caps = set(), set(), set()
    for p in prior:
        for k in ("original_sha256", "sha256"):
            if p.get(k):
                shas.add(str(p[k]))
        if p.get("key"):
            keys.add(str(p["key"]))
        for c in p.get("capture_keys") or ():
            caps.add(tuple(c))
    HP, okP, hasP = hash_matrix([{"dhash": p.get("dhash"), "variants": p.get("variants")} for p in prior])
    HR, okR, hasR = hash_matrix(cand)
    import numpy as np
    near = np.full(len(cand), bits + 1, dtype=np.int64)
    if hasP.any() and hasR.any():
        ir, ip = np.flatnonzero(hasR), np.flatnonzero(hasP)
        d, _a = cross_nearest(HR[ir], okR[ir], HP[ip], okP[ip], bits)
        near[ir] = d
    n = 0
    for i, r in enumerate(cand):
        why = None
        if (r.get("sha256") and r["sha256"] in shas) or (r.get("train_sha256") and r["train_sha256"] in shas):
            why = "bytes"
        elif r["key"] in keys:
            why = "key"
        elif any(tuple(k) in caps for k in capture_keys(r)):
            why = "capture"
        elif near[i] <= bits:
            why = "dhash_%d" % int(near[i])
        if why:
            r["prior_test"] = why
            n += 1
    return n


def select(rows, conf, quarantine=(), seed_text=None):
    """Dedupe, capture groups, the main-test holdout, the quarantine and the
    family caps over the rows still standing (drop None). Every step up to
    the holdout ignores the quarantine, so test v1 is the same whichever
    sources are quarantined; a quarantined source's rows then stay out of
    arm B (its held rows stay in test v1). A group is a pool's group when a
    member is in a pool or in flight, or shares a capture relation with a
    pool row a rule dropped before the grouping (`rows` holds every row).
    Sets drop (duplicate, dup_of_base, holdout_v1, prior_test_v1,
    quarantined, family_cap), dup_of and group; returns the record."""
    rules = conf["rules"]
    dd, ho = rules["dedupe"], rules["holdout"]
    seed_text = seed_text or ho["seed_text"]
    q = set(quarantine)
    live = [r for r in rows if r["drop"] is None]
    t0 = time.time()
    uf, crec = copy_components(live, dd)
    comps = list(uf.groups(range(len(live))).values())
    n_dup = 0
    for comp in comps:
        if len(comp) < 2:
            continue
        base = [i for i in comp if live[i]["kind"] == BASE_KIND]
        best = min(comp, key=lambda i: _rank(live[i]))
        keep = set(base) if base else {best}
        for i in comp:
            if i not in keep:
                live[i]["drop"] = "dup_of_base" if base else "duplicate"
                live[i]["dup_of"] = live[best]["key"]
                n_dup += 1
    gid, digest = capture_groups(live, comps, ho["group_bits"])
    members = collections.defaultdict(list)
    for i, g in gid.items():
        live[i]["group"] = g
        members[g].append(i)

    def ext(i):
        return live[i]["kind"] != BASE_KIND and live[i]["drop"] is None
    # a pool row that a rule dropped before the grouping still shares its capture relations (its video, its
    # export stem): a group holding another frame of it is a pool's group too, never held out
    pool_caps = {k for r in rows if r.get("in_pool") and r["kind"] != BASE_KIND and r["drop"] is not None
                 for k in capture_keys(r)}
    owner, eligible, why_not, rows_of, prior_in = {}, {}, {}, {}, {}
    for g, mem in members.items():
        cnt = collections.Counter(live[i]["source"] for i in mem if ext(i))
        rows_of[g] = cnt
        if cnt:
            tier = {live[i]["source"]: live[i]["tier"] for i in mem}
            owner[g] = sorted(cnt, key=lambda s: (-cnt[s], tier[s], s))[0]
        touch = any(live[i]["kind"] == BASE_KIND or live[i].get("in_pool") for i in mem) or (
            bool(pool_caps) and any(k in pool_caps for i in mem for k in capture_keys(live[i])))
        why_not[g] = "base_or_pool" if touch else ("over_max_group" if len(mem) > ho["max_group"] else None)
        eligible[g] = why_not[g] is None
        prior_in[g] = any(live[i].get("prior_test") for i in mem)
    kept_by = collections.Counter(live[i]["source"] for i in range(len(live)) if ext(i))
    held, held_n, holdout = set(), collections.Counter(), {}
    for s in sorted(kept_by):
        nk = kept_by[s]
        target = min(max(round_half_up(ho["share"] * nk), ho["min_images"]), ho["max_images"])
        # an order built from each group's own content: a group earlier test lists hold comes first
        cand = sorted((g for g, o in owner.items() if o == s and eligible[g]),
                      key=lambda g: (0 if prior_in[g] else 1, C.stable_int("%s/%s/%s" % (seed_text, s, digest[g])), g))
        took, skipped = [], 0
        for g in cand:
            if held_n[s] >= target and not prior_in[g]:
                break                  # a group an earlier test list holds is held again whatever the target
            add = rows_of[g]
            if any(held_n[t] + add[t] > ho["max_images"] for t in add):
                skipped += 1           # it would take some source past max_images: never held
                continue
            took.append(g)
            held.add(g)
            held_n.update(add)
        inel = collections.Counter()
        for g, cnt in rows_of.items():
            if cnt.get(s) and not eligible[g]:
                inel[why_not[g]] += cnt[s]
        holdout[s] = {"kept_rows": nk, "target": target, "eligible_groups": len(cand), "groups": len(took),
                      "skipped_over_max": skipped, "ineligible_rows": dict(sorted(inel.items()))}
    n_held = 0
    for g in held:
        for i in members[g]:
            if ext(i):
                live[i]["drop"] = "holdout_v1"
                live[i]["holdout_owner"] = owner.get(g)
                n_held += 1
    for s, v in holdout.items():
        v["images"] = held_n[s]
        v["quarantined_source"] = s in q
    n_prior = 0
    for i, r in enumerate(live):
        if ext(i) and r.get("prior_test"):
            r["drop"] = "prior_test_v1"            # an earlier build's test list holds it: never trained
            n_prior += 1
    # the quarantine, after the holdout: a quarantined source's rows stay out of arm B; a copy group whose kept
    # row is quarantined keeps its best copy from a source that is not
    n_q, promoted = 0, 0
    for comp in comps:
        quar = [i for i in comp if ext(i) and live[i]["source"] in q]
        for i in quar:
            live[i]["drop"] = "quarantined"
            n_q += 1
        if quar and len(comp) > 1 and not any(ext(i) for i in comp):
            alt = [i for i in comp if live[i]["drop"] == "duplicate" and live[i]["source"] not in q
                   and not live[i].get("prior_test")]
            if alt:
                m = min(alt, key=lambda i: _rank(live[i]))
                live[m]["drop"] = None
                live[m].pop("dup_of", None)
                for i in comp:
                    if live[i]["drop"] == "duplicate":
                        live[i]["dup_of"] = live[m]["key"]
                promoted += 1
    caps = {}
    for fam, fr in sorted((rules.get("families") or {}).items()):
        share = float(fr["cap_share"])
        inb = [i for i, r in enumerate(live) if r["drop"] is None]
        fam_rows = [i for i in inb if live[i]["family"] == fam]
        non = len(inb) - len(fam_rows)
        cap = int(math.floor(share / (1.0 - share) * non + 1e-9))
        rec = {"share": share, "before": len(fam_rows), "non_family": non, "cap": cap, "dropped": 0}
        if len(fam_rows) > cap:
            fg = collections.defaultdict(list)
            for i in fam_rows:
                fg[gid[i]].append(i)
            order = sorted(fg, key=lambda g: (C.stable_int("%s/cap/%s/%s" % (seed_text, fam, digest[g])), g))
            cur = len(fam_rows)
            for g in order:
                if cur <= cap:
                    break
                for i in fg[g]:
                    live[i]["drop"] = "family_cap"
                    cur -= 1
                    rec["dropped"] += 1
        rec["after"] = sum(1 for i in fam_rows if live[i]["drop"] is None)
        caps[fam] = rec
    big = sorted(((len(m), g) for g, m in members.items()), reverse=True)[:5]
    return dict({"rows": len(live), "duplicates": n_dup, "groups": len(members), "held_groups": len(held),
                 "held_images": n_held, "prior_test_dropped": n_prior, "quarantined": n_q,
                 "promoted_copies": promoted, "quarantine": sorted(q), "holdout": holdout, "family_caps": caps,
                 "largest_groups": [{"group": g, "rows": n, "sources": dict(collections.Counter(
                     live[i]["source"] if live[i]["kind"] != BASE_KIND else "base_v2" for i in members[g])),
                     "eligible": eligible[g], "why_not": why_not[g]} for n, g in big],
                 "seconds": round(time.time() - t0, 1)}, **crec)


def distinct_photos(rows):
    """Photos, not files: rows joined by the same export stem (per family,
    else per source) or a dHash within 3 bits under the 8 variants."""
    n = len(rows)
    uf = UnionFind(n)
    first = {}
    for i, (r, h) in enumerate(zip(rows, hash_components(rows, 3).tolist() if n else [])):
        ks = [("hash", h)]
        if r["stem"]:
            ks.append(("stem", r["family"] or r["source"], r["stem"]))
        for k in ks:
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    return len(uf.groups(range(n)))


# ------------------------------------------------------------------- driver
def released_status(status):
    """True for an increment status (inc2.stream's fold) whose rows have
    left the stream's way to a pool: quarantined ('data'), returned to the
    queue (a return disposition, a bisect's verdict) or released (a stale
    base, withdrawn). A returned or released row reaches a pool again only
    through a new cut, and the cutter never cuts a test v1 row
    (step1_stream.test_v1_rows). Every other status -- in_segment (cut, its
    segment not yet committed), suspect (accepted, then rolled back), or any
    status this list does not know -- may still put its rows in a pool."""
    from . import stream as S
    s = str(status)
    return s in (S.DATA, "stale", "withdrawn") + tuple(S.RETURN_DISPOSITIONS) or s.startswith("bisect_")


def quarantined_sources(sid):
    """({source: record}, ledger record, (pool keys, pool image sha256s,
    pool original paths)) of the stream: its quarantined sources, and the
    rows of every pool it has trained on and of every increment in flight,
    through inc2.stream's ledger fold (fail closed: no stream, no build).
    Such a row never enters the main-test holdout (select). An increment in
    flight is one whose status has not released its rows (released_status):
    a segment cut before this build trains on rows no accepted pool holds
    yet, and its commit may accept them into the next pool, so they count
    as pool rows exactly as an accepted pool's do. A masked row trains on a
    masked copy, whose sha256 is not its original's: for every masked row
    of an accepted or in-flight increment, its original's path
    (unmasked_image) and that file's sha256 count too, so the candidate row
    of the original is marked (a Step 1 key need not equal this builder's
    key). The record counts them (pool_rows; in_flight: per increment its
    status and rows, and the rows in all; masked_originals: hashed, and the
    files that could not be read, matched by path only)."""
    from . import stream as S
    try:
        st = S.Stream(sid, quiet=True)
        f = st.load()
        keys, shas, paths = set(), set(), set()
        for name in list(f.pools):
            if not (f.pools[name] or {}).get("path"):
                continue
            for r in f.pool_rows(name):
                keys.add(str(r.get("key")))
                shas.add(str(r.get("sha256")))
        n_pool = len(keys)
        flight, fkeys = {}, set()
        masked = {"rows": 0}
        hashed = {}
        for inc, rec in sorted(f.increments.items()):
            accepted = rec.get("status") == S.ACCEPTED
            if not accepted and released_status(rec.get("status")):
                continue
            rows = f.inc_rows(inc)
            if not accepted:                 # an accepted increment's keys are its pool's (above)
                flight[inc] = {"status": rec.get("status"), "segment": rec.get("segment"), "rows": len(rows)}
            for k, r in rows.items():
                if not accepted:
                    fkeys.add(str(k))
                    keys.add(str(k))
                    if r.get("sha256"):
                        shas.add(str(r["sha256"]))
                orig = r.get("unmasked_image")
                if orig and str(orig) != str(r.get("image")):
                    masked["rows"] += 1
                    paths.add(str(orig))
                    if str(orig) not in hashed:
                        hashed[str(orig)] = _sha_file(orig)
                    if hashed[str(orig)]:
                        shas.add(hashed[str(orig)])
        bad = sorted(p for p, v in hashed.items() if not v)
        masked.update(hashed=sum(1 for v in hashed.values() if v), unreadable=len(bad), unreadable_first=bad[:20])
    except Exception as e:  # noqa: BLE001 - an unreadable ledger is no quarantine record: refuse
        raise Base3Error("the stream %s's ledger, pools or increments cannot be read (%s: %s): the quarantine "
                         "and the rows it trains on are unknown" % (sid, type(e).__name__, e))
    return dict(f.q_sources), {"sid": sid, "ledger": str(st.p.ledger), "head_sha256": f.head,
                               "events": f.events, "quarantined_sources": sorted(f.q_sources),
                               "pools": sorted(f.pools), "pool_rows": n_pool,
                               "in_flight": {"increments": flight, "rows": len(fkeys)},
                               "masked_originals": masked}, (keys, shas, paths)


def queue_holds():
    """({key: holds left}, record) of Step 1's queue (inc2.stream.QueueView);
    (None, record) when it cannot be read: rows with intake holds then stay out."""
    from . import stream as S
    try:
        q = S.QueueView().load()
    except Exception as e:  # noqa: BLE001 - recorded; intake rows with holds are not admitted
        return None, {"read": False, "why": "%s: %s" % (type(e).__name__, str(e)[:300])}
    out = {k: S.holds_of(r) for k, r in q.rows.items()}
    return out, {"read": True, "rows": len(out), "queue_sha256": q.sha256, "events_sha256": q.events_sha256,
                 "reader": q.reader}


def gather(conf, sid, registry=None, testing=False):
    """Every candidate row with the rules applied: (rows, inputs record,
    per-source records, quarantine). The stream's quarantine is returned,
    never applied here: select applies it after the holdout, so every step
    before it is the same whichever sources are quarantined."""
    q_sources, q_rec, pool = quarantined_sources(sid)
    holds, h_rec = queue_holds()
    reg_path = registry_path()
    if registry is None:
        try:
            with open(reg_path) as fh:
                registry = json.load(fh)
        except (OSError, ValueError) as e:
            raise Base3Error("cannot read the dataset registry %s: %s" % (reg_path, e))
        reg_rec = {"path": str(reg_path), "sha256": _sha_file(reg_path)}
    else:
        reg_rec = {"path": None, "sha256": _sha_text(registry), "in_memory": True}
    lock = C2.read_lock_v2()
    if lock.get("testing") and not testing:
        raise Base3Error("LOCK v2 was written by a testing build")
    brows, b_rec = base_rows(conf, lock=lock, production=not testing)
    irows, i_rec = intake_rows(conf, holds_view=holds)
    rrows, r_rec = registry_rows(conf, registry)
    rows = brows + irows + rrows
    for u, r in enumerate(rows):
        r["uid"] = u
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda r: map_boxes(r, conf), rows))
    log("labels: %d rows read in %.0fs" % (len(rows), time.time() - t0))
    t0 = time.time()
    need = [r for r in rows if r["W"] is None and (r["kind"] == BASE_KIND or not r["drop"])]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r, sz in zip(need, ex.map(lambda r: _header_size(r["image"]), need)):
            if sz:
                r["W"], r["H"] = int(sz[0]), int(sz[1])
    log("image headers: %d read in %.0fs" % (len(need), time.time() - t0))
    if any(not r["W"] for r in rows if r["kind"] == BASE_KIND):
        raise Base3Error("a base_v2 image cannot be read: arm A would not be base_v2 whole")
    conv = {}
    by_src = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != BASE_KIND:
            by_src[r["source"]].append(r)
    for s, rs in sorted(by_src.items()):
        conv[s] = source_convention([r for r in rs if r["drop"] is None], conf["rules"]["source"])
        if not conv[s]["passes"]:
            for r in rs:
                r["drop"] = r["drop"] or "convention"
    for r in rows:
        image_rules(r, conf["rules"])
    mark_pool(rows, pool)
    inputs = {"registry": reg_rec, "base_v2": b_rec, "intake": i_rec,
              "stream": q_rec, "step1_queue": h_rec, "lock_v2": {"path": str(C2.LOCK_PATH),
                                                                  "sha256": _sha_file(C2.LOCK_PATH)}}
    not_read = {s: v["error"] for s, v in sorted(r_rec.items()) if v.get("error")}
    return rows, inputs, {"registry": r_rec, "convention": conv, "pool": pool,
                          "registry_doc": registry, "not_read": not_read}, q_sources


def mark_pool(rows, pool):
    """in_pool on every row the stream has trained on or may still pool (its
    pools: base_v2 and every accepted increment; every increment in flight,
    quarantined_sources), by key, by the original's sha256 or by the
    original's path (a masked row's unmasked_image): such a row never enters
    the main-test holdout, nor does any row sharing a capture relation with
    it, whatever its source (select). Called again once the originals'
    sha256s are known. Returns the number of candidate rows marked besides
    base_v2's."""
    keys, shas = pool[0], pool[1]
    paths = pool[2] if len(pool) > 2 else ()
    n = 0
    for r in rows:
        r["in_pool"] = r["kind"] == BASE_KIND or r["key"] in keys or (r["sha256"] or "") in shas \
            or str(r["image"]) in paths
        n += 1 if r["in_pool"] and r["kind"] != BASE_KIND else 0
    return n


def _selection_state(rows):
    """What select sets on each row (drop, dup_of), to select again."""
    return [(r["drop"], "dup_of" in r, r.get("dup_of")) for r in rows]


def _restore_selection(rows, state):
    for r, (drop, had, dup_of) in zip(rows, state):
        r["drop"] = drop
        if had:
            r["dup_of"] = dup_of
        else:
            r.pop("dup_of", None)
        r.pop("group", None)
        r.pop("holdout_owner", None)


def _pool_changed(a, b):
    return any(set(x) != set(y) for x, y in zip(tuple(a) + ((),) * (3 - len(a)), tuple(b) + ((),) * (3 - len(b))))


def recheck_pool(sid, rows, pool, conf, quarantine, state, sel, events_before=None):
    """build: the stream's fold read again once the selection is made, before
    any output is written. gather read it when the job started, and the
    stream may have cut a segment since (the job runs for hours): when its
    pools or in-flight rows changed, the rows are marked again and selected
    again from the state before the first selection (`state`). Returns
    (selection record, pool, record)."""
    _q, now, pool2 = quarantined_sources(sid)
    rec = {"head_sha256": now["head_sha256"], "events_before": events_before, "events": now["events"],
           "in_flight": now["in_flight"], "quarantined_sources_now": now["quarantined_sources"],
           "changed": _pool_changed(pool, pool2)}
    if rec["changed"]:
        rec["pool_marked_rows"] = mark_pool(rows, pool2)
        _restore_selection(rows, state)
        sel = select(rows, conf, quarantine=quarantine)
        rec["reselected"] = True
        log("the stream's pools or in-flight rows changed while the job ran (ledger events %s -> %s): selected "
            "again" % (events_before, now["events"]))
    return sel, (pool2 if rec["changed"] else pool), rec


def companion_rows(rows):
    """{source: [row]}: every row that is not a test row but shares a capture
    relation (export stem, intake capture group, file-name session: SIU's
    video) with a held-out row, or lies in a held-out group (its duplicates):
    rows a rule dropped before the grouping (an intake hold, a box or image
    rule, an evaluation group) are in no group, and Step 1's queue may still
    offer them to the stream's cutter, which matches a test list by bytes,
    key and dHash only. Listed beside the test lists, they never train and
    are never cut (prior_test_lists)."""
    held = [r for r in rows if r["drop"] == "holdout_v1"]
    caps = {k for r in held for k in capture_keys(r)}
    groups = {r.get("group") for r in held if r.get("group")}
    out = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == BASE_KIND or r["drop"] in (None, "holdout_v1", "prior_test_v1"):
            continue
        if (r.get("group") and r["group"] in groups) or any(k in caps for k in capture_keys(r)):
            out[r["source"]].append(r)
    return out


def final_pool_check(sid, rows, companions, wait_s=None):
    """build, after the test lists and their companions are written: the
    stream's fold read once more while this job holds the stream's lease
    (inc2.stream's writers hold it to cut and build a segment). A cut that
    began before the lists existed has written its ledger lines by then; a
    cut after it reads them (step1_stream.test_v1_rows) and never takes a
    listed row. Raises Base3Error when a listed row, or a row sharing a
    capture relation with a held-out row, is now in a pool or in flight:
    summary.json is then never written. Returns the record."""
    from ..inc import driver as D
    from . import stream as S
    lease = D.Lease(S.StreamPaths(sid).lease)
    t0, wait_s = time.time(), STREAM_LEASE_WAIT_S if wait_s is None else wait_s
    while not lease.acquire():
        if time.time() - t0 >= wait_s:
            raise Base3Error("the stream %s's lease was held for %.0f s: the last check of the rows it trains on "
                             "could not run (fail closed; the test lists stay as never-train)" % (sid, wait_s))
        time.sleep(10)
    try:
        _q, now, (keys, shas, paths) = quarantined_sources(sid)
    finally:
        lease.release()

    def pooled(r):
        return r["key"] in keys or (r["sha256"] or "") in shas or str(r["image"]) in paths
    held = [r for r in rows if r["drop"] == "holdout_v1"]
    caps = {k for r in held for k in capture_keys(r)}
    listed = held + [r for rs in companions.values() for r in rs]
    hits = sorted({r["key"] for r in listed if pooled(r)} | {
        r["key"] for r in rows if r["kind"] != BASE_KIND and pooled(r) and any(k in caps for k in capture_keys(r))})
    rec = {"head_sha256": now["head_sha256"], "events": now["events"], "in_flight": now["in_flight"],
           "lease_wait_s": round(time.time() - t0, 1), "conflicts": len(hits)}
    if hits:
        raise Base3Error("%d row(s) of test v1 or of its companions, or sharing a held-out row's capture relation, "
                         "are now in a pool of the stream %s or in an increment in flight (cut while this job ran; "
                         "e.g. %s): summary.json is not written; the test lists stay as never-train; move %s aside "
                         "and build again" % (len(hits), sid, ", ".join(hits[:5]), out_dir()))
    return rec


def per_source(rows, recs, quarantine, lifted=()):
    out = {}
    for r in rows:
        s = r["source"] if r["kind"] != BASE_KIND else "base_v2"
        v = out.setdefault(s, {"kind": r["kind"], "family": r["family"], "tier": r["tier"], "seen": 0, "labelled": 0,
                               "with_weed_box": 0, "admitted": 0, "in_B": 0, "boxes_B": 0, "masked_boxes_B": 0,
                               "polygons": 0, "dropped": collections.Counter(), "guard": collections.Counter(),
                               "eval_groups": collections.Counter(), "holdout_v1": 0, "quarantined": False})
        v["seen"] += 1
        v["labelled"] += 1 if (r["label"] or r["kind"] == BASE_KIND) else 0
        v["with_weed_box"] += 1 if (r["n_weed"] or r["boxes"]) else 0
        v["polygons"] += int(r["polygons"] or 0)
        for x in (r.get("guard") or {}).get("reasons") or []:
            v["guard"][x] += 1
        for g in r.get("eval_group") or {}:
            v["eval_groups"][g] += 1
        if r["drop"] is None:
            v["in_B"] += 1
            v["boxes_B"] += len(r["boxes"])
            v["masked_boxes_B"] += len(r["mask"])
        elif r["drop"] == "holdout_v1":
            v["holdout_v1"] += 1
        else:
            v["dropped"][r["drop"]] += 1
    for s, v in out.items():
        v["dropped"] = dict(sorted(v["dropped"].items()))
        v["guard"] = dict(sorted(v["guard"].items()))
        v["eval_groups"] = dict(sorted(v["eval_groups"].items()))
        v["quarantined"] = s in quarantine and s not in set(lifted)
        v["convention"] = recs["convention"].get(s)
        if s in recs["registry"]:
            v["registry"] = {k: recs["registry"][s].get(k) for k in ("names_basis", "error", "registry_status",
                                                                   "licence", "group_regex")}
    return out


def arm_rows(rows):
    a = [r for r in rows if r["kind"] == BASE_KIND and r["drop"] is None]
    b = [r for r in rows if r["drop"] is None]
    return a, b


def _guard_drops(rows):
    """Apply the guard's verdicts: never-train refusals and cross-check hits
    drop a row; base_copy drops a new row; base rows are never dropped as
    base copies."""
    for r in rows:
        g = r.get("guard") or {}
        if r["drop"] or g.get("pending"):
            continue
        if g.get("refused") or g.get("crosscheck"):
            r["drop"] = "guard:%s" % ((g.get("refused_by") or ["crosscheck"])[0])
        elif r["kind"] != BASE_KIND and "base_copy" in (g.get("reasons") or []):
            r["drop"] = "base_copy"


def apply_leak_drops(rows, leak):
    """The embedding check's and D28-v2's verdicts (build): a row the
    embedding detector refuses (near_eval_embed, or unhashable: fail closed)
    is dropped; every other row of a source D28-v2 flags is dropped
    (source_leak)."""
    for r in rows:
        if r["drop"]:
            continue
        if r.get("embed") in ("near_eval_embed", "unhashable"):
            r["drop"] = "embed:%s" % r["embed"]
        elif r["kind"] != BASE_KIND and (leak.get(r["source"]) or {}).get("flagged"):
            r["drop"] = "source_leak"


def held_keys(rows):
    return sorted(r["key"] for r in rows if r["drop"] == "holdout_v1")


def count(sid, out, conf_path=None, registry=None, testing=False, hashes=None):
    """count (module docstring): the selection on stored hashes, read-only;
    the report goes to `out` (a directory outside INC_DIR)."""
    import copy as _copy
    t0 = time.time()
    conf, csha = load_config(conf_path)
    out = Path(out)
    if str(out.resolve()).startswith(str(Path(C.INC_DIR).resolve())):
        raise Base3Error("count writes outside INC_DIR only (%s)" % out)
    rows, inputs, recs, quarantine = gather(conf, sid, registry=registry, testing=testing)
    hashes = hashes or StoredHashes()
    th0 = time.time()
    live = [r for r in rows if r["drop"] is None]
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(hashes.fill, live))
    log("hashes: %d rows (%d hashed here) in %.0fs" % (len(live), sum(1 for r in live if r.get("hashed_here")),
                                                       time.time() - th0))
    inputs["stream"]["pool_marked_rows"] = mark_pool(rows, recs["pool"])
    guards, grec = run_guard(live, production=not testing)
    th, th_sha = load_thresholds()
    _guard_drops(rows)
    egs, eg_rec = eval_group_hashes(conf, recs["registry_doc"], stored=hashes.by_path)
    eg_rec = merge_eval_records(eg_rec, eval_guard(rows, egs, conf["evaluation_groups"]["bits"]))
    prior, prior_rec = prior_test_lists()
    prior_rec["marked"] = mark_prior(rows, prior, conf["rules"]["holdout"]["group_bits"])
    lifted = sorted(s for s in quarantine if s in conf["sources"] or s in conf["intake"])
    scen, held = {}, {}
    for name, lift in (("as_is", ()), ("lifted", tuple(lifted))):
        rs = _copy.deepcopy(rows)
        sel = select(rs, conf, quarantine=set(quarantine) - set(lift))
        ps = per_source(rs, recs, quarantine, set(lift))
        a, b = arm_rows(rs)
        held[name] = held_keys(rs)
        dhash_hits = {}
        for r in rs:
            reasons = (r.get("guard") or {}).get("reasons") or []
            if r["kind"] != BASE_KIND and any(x in DHASH_HIT_REASONS for x in reasons):
                dhash_hits[r["source"]] = dhash_hits.get(r["source"], 0) + 1
        scen[name] = {"lifted": list(lift), "selection": sel, "per_source": ps,
                      "arms": {"A": {"images": len(a), "boxes": sum(len(r["boxes"]) for r in a)},
                               "B": {"images": len(b), "boxes": sum(len(r["boxes"]) for r in b),
                                     "distinct_photos": distinct_photos(b)}},
                      "holdout_v1": {"images": len(held[name]),
                                     "sha256_of_keys": C.sha256_text("\n".join(held[name])), "keys": held[name]},
                      "dhash_hits_unweighed": dhash_hits,
                      "pending": {"embedding_check": "not run (count; the build job runs GuardV2.check_embed)",
                                  "pair_cosines": "not run (count): %d source(s) with dHash hits would be judged by "
                                                  "D28-v2 in the build" % len(dhash_hits),
                                  "guard_unjudged_rows": sum(1 for r in rs if (r.get("guard") or {}).get("pending")
                                                             and r["drop"] is None),
                                  "png_hashes": "hashes are the originals'; the build hashes the files it writes"}}
        log("%s: arm A %d images, arm B %d images (%d distinct photos), held out %d"
            % (name, len(a), len(b), scen[name]["arms"]["B"]["distinct_photos"], sel["held_images"]))
    rep = {"format": COUNT_FORMAT, "built_utc": C2.utc(), "config": {"path": str(conf_path or CONFIG), "sha256": csha},
           "inputs": inputs, "hashes": hashes.record, "guard": {k: grec.get(k) for k in ("reasons", "checked",
                                                                                        "refused", "crosscheck_hits")},
           "evaluation_groups": eg_rec, "prior_test": prior_rec, "sources_not_read": recs["not_read"],
           "holdout_same_in_every_scenario": len({tuple(v) for v in held.values()}) == 1,
           "thresholds_sha256": th_sha, "scenarios": scen, "seconds": round(time.time() - t0, 1),
           "note": "read-only rehearsal on stored hashes of the originals; the embedding check and the D28-v2 pair "
                   "cosines are pending (the build job runs them on the files it writes)"}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "e1_base3_count.json"
    C2.write_json_atomic(p, json.loads(json.dumps(rep, default=str)))
    log("count report: %s (%.0fs)" % (p, time.time() - t0))
    return rep


def _label_bytes(boxes):
    return "".join("%d %.6f %.6f %.6f %.6f\n" % ((WEED_ID,) + tuple(b)) for b in boxes).encode("utf-8")


def _write_label(root, boxes):
    data = _label_bytes(boxes)
    sha = hashlib.sha256(data).hexdigest()
    p = Path(root) / "labels" / sha[:2] / ("%s.txt" % sha)
    _store_bytes(p, data)
    return str(p), sha


def _write_manifest_once(path, rows):
    """inc.common.write_manifest's bytes, refusing to replace a file that holds
    other bytes (a manifest is never overwritten)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="base3_") as tmp:
        tp = Path(tmp) / "m.jsonl"
        sha = C.write_manifest(tp, rows)
        data = tp.read_bytes()
    if Path(path).is_file():
        if C.sha256_file(path) != sha:
            raise Base3Error("%s exists with other bytes: a manifest is never overwritten (move it aside first)" % path)
        return sha
    C2.write_bytes_atomic(path, data)
    return sha


def build(sid, conf_path=None, registry=None, testing=False, procs=5, embedder=None, scanner=None, loader=None):
    """build (module docstring). Returns the summary."""
    from ..inc import pilot as P
    t0 = time.time()
    try:
        testing = P._check_testing(testing)
    except P.PilotError as e:
        raise Base3Error(str(e))
    conf, csha = load_config(conf_path)
    root = out_dir()
    if (root / SUMMARY).is_file():
        raise Base3Error("%s exists: base v3 is built once (a rebuild is a new version)" % (root / SUMMARY))
    for d in (HOLDOUT_DIR, COMPANION_DIR):
        if (root / d).exists():
            raise Base3Error("%s holds the test lists of a build that did not finish: move %s aside inside %s first "
                             "(its test lists are then read as never-train)" % (root / d, root, root.parent))
    prior, prior_rec = prior_test_lists()
    rows, inputs, recs, quarantine = gather(conf, sid, registry=registry, testing=bool(testing))
    todo = [r for r in rows if not r["drop"]]
    log("materialising %d rows (%d workers)" % (len(todo), procs))
    tm = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(procs))) as ex:
        list(ex.map(lambda r: materialise(r, root), todo))
    written = sum(1 for r in todo if r.get("png_written"))
    log("materialised in %.0fs: %d PNG(s) written, %d used as they are" % (
        time.time() - tm, written, sum(1 for r in todo if r.get("train_image") == r["image"])))
    inputs["stream"]["pool_marked_rows"] = mark_pool(rows, recs["pool"])
    live = [r for r in rows if not r["drop"]]
    guards, grec = run_guard(live, production=not testing)
    if scanner is None:
        scanner, erec = embed_scanner(production=not testing, embedder=embedder, procs=procs)
    else:
        erec = dict(scanner.record(), checked=True, injected=True)
    th, th_sha = load_thresholds()
    leak = {}
    if scanner is not None:
        n_emb, n_hits = embed_and_weigh(live, scanner, guards, procs=procs,
                                        desc_path=root / "leak" / "scan_desc.npz")
        erec.update(scanned=n_emb, dhash_hits_weighed=n_hits)
    elif not testing:
        raise Base3Error("no embedding calibration: the embedding check cannot run (fail closed)")
    else:
        erec.update(note="testing LOCK without a calibration: no embedding check, dHash hits unweighed")
    by_src = collections.defaultdict(list)
    for r in live:
        if r["kind"] != BASE_KIND:
            by_src[r["source"]].append(r)
    copy_cos = (erec or {}).get("cos_threshold")
    for s, rs in sorted(by_src.items()):
        hits = [r for r in rs if any(x in DHASH_HIT_REASONS for x in r["guard"]["reasons"])]
        cos = None if scanner is None else [r.get("pair_cos") for r in hits if r.get("pair_cos") is not None]
        emb = sum(1 for r in rs if r.get("embed") == "near_eval_embed")
        bc = sum(1 for r in rs if "base_copy" in r["guard"]["reasons"])
        leak[s] = leak_verdict(len(rs), len(hits), cos, emb, erec.get("p_false"), copy_cos, bc, th)
    _guard_drops(rows)
    apply_leak_drops(rows, leak)
    te = time.time()
    egs, eg_rec = eval_group_hashes(conf, recs["registry_doc"], procs=max(1, int(procs)))
    eg_rec = merge_eval_records(eg_rec, eval_guard(rows, egs, conf["evaluation_groups"]["bits"]))
    log("evaluation groups: %s hashes, %d rows dropped within %d bits, in %.0fs"
        % ({g: v["hashes"] for g, v in eg_rec["groups"].items()}, eg_rec["dropped"], eg_rec["bits"],
           time.time() - te))
    prior_rec["marked"] = mark_prior(rows, prior, conf["rules"]["holdout"]["group_bits"])
    state = _selection_state(rows)
    sel = select(rows, conf, quarantine=quarantine)
    # the stream may have cut a segment since the job started (gather read its fold then): read it again, and
    # select again over the rows it now trains on or may pool, before anything is written
    sel, recs["pool"], inputs["stream"]["recheck"] = recheck_pool(
        sid, rows, recs["pool"], conf, quarantine, state, sel, events_before=inputs["stream"].get("events"))
    a, b = arm_rows(rows)
    if len(a) == 0 or len(b) <= len(a):
        raise Base3Error("arm A has %d rows and arm B %d: nothing to compare" % (len(a), len(b)))
    # ---- outputs
    for r in rows:
        if r["drop"] in (None, "holdout_v1"):
            boxes = r["boxes"] if r["kind"] != BASE_KIND else r["boxes"]
            r["v3_label"], r["v3_label_sha256"] = _write_label(root, boxes)
            if r["kind"] == BASE_KIND and r["train_image"] is None:
                raise Base3Error("base row %s has no materialised image" % r["key"])

    def man(r):
        return {"image": r["train_image"], "label": r["v3_label"], "sha256": r["train_sha256"],
                "label_sha256": r["v3_label_sha256"], "source": r["source"], "session": r["session"] or "",
                "key": r["key"]}
    sha_a = _write_manifest_once(root / ("%s.jsonl" % ARM_A), [man(r) for r in a])
    sha_b = _write_manifest_once(root / ("%s.jsonl" % ARM_B), [man(r) for r in b])
    prov = []
    for r in sorted(rows, key=lambda r: r["key"]):
        if r["drop"] not in (None, "holdout_v1"):
            continue
        prov.append({"key": r["key"], "source": r["source"], "kind": r["kind"], "family": r["family"],
                     "arms": (["A", "B"] if r["kind"] == BASE_KIND else ["B"]) if r["drop"] is None else [],
                     "holdout_v1": r["drop"] == "holdout_v1", "original_image": r["image"],
                     "original_sha256": r["sha256"], "original_label": r["label"], "image": r["train_image"],
                     "sha256": r["train_sha256"], "label": r["v3_label"], "label_sha256": r["v3_label_sha256"],
                     "boxes": len(r["boxes"]), "masked_boxes": len(r["mask"]), "masked_area": r["masked_area"],
                     "dhash": r["dhash"], "variants": r["variants"], "group": r.get("group"),
                     "capture_keys": [list(k) for k in capture_keys(r)],
                     "quarantined_source": r["source"] in quarantine, "prior_test": r.get("prior_test"),
                     "licence": r["licence"], "research_only": r["research_only"]})
    prov_sha = C2.write_jsonl_atomic(root / PROVENANCE, prov)
    hold_files = {}
    for s in sorted({r["source"] for r in rows if r["drop"] == "holdout_v1"}):
        hrows = [p for p in prov if p["source"] == s and p["holdout_v1"]]
        hold_files[s] = {"file": "%s/%s.jsonl" % (HOLDOUT_DIR, s),
                         "sha256": C2.write_jsonl_atomic(root / HOLDOUT_DIR / ("%s.jsonl" % s), hrows),
                         "images": len(hrows), "quarantined_source": s in quarantine}
    comp = companion_rows(rows)
    comp_files = {}
    for s, rs in sorted(comp.items()):
        crows = [{"key": r["key"], "source": r["source"], "kind": r["kind"], "companion": True, "reason": r["drop"],
                  "original_image": r["image"], "original_sha256": r["sha256"], "sha256": r.get("train_sha256"),
                  "dhash": r["dhash"], "variants": r["variants"], "group": r.get("group"),
                  "capture_keys": [list(k) for k in capture_keys(r)]} for r in sorted(rs, key=lambda r: r["key"])]
        comp_files[s] = {"file": "%s/%s.jsonl" % (COMPANION_DIR, s), "rows": len(crows),
                         "sha256": C2.write_jsonl_atomic(root / COMPANION_DIR / ("%s.jsonl" % s), crows)}
    # the lists exist now, so a later cut never takes a listed row; one that began before them is in the ledger
    # once this job holds the stream's lease
    inputs["stream"]["final_check"] = final_pool_check(sid, rows, comp)
    drop_rows = [{"key": r["key"], "source": r["source"], "image": r["image"], "reason": r["drop"],
                  "dup_of": r.get("dup_of"), "group": r.get("group"), "eval_group": r.get("eval_group"),
                  "prior_test": r.get("prior_test")} for r in sorted(rows, key=lambda r: r["key"])
                 if r["drop"] not in (None, "holdout_v1")]
    drop_sha = C2.write_jsonl_atomic(root / DROPPED, drop_rows)
    used = {r["train_image"] for r in rows if r["drop"] in (None, "holdout_v1") and r.get("train_image")}
    removed = 0
    for r in rows:
        if r.get("png_written") and r["train_image"] not in used and Path(r["train_image"]).is_file():
            os.unlink(r["train_image"])
            removed += 1
    ps = per_source(rows, recs, quarantine)
    for s, v in ps.items():
        if s in leak:
            v["leak"] = {k: leak[s][k] for k in ("flagged", "why", "describe", "base_copy_share")}
    tw = time.time()
    lr = loader_rate([man(r) for r in b], **(loader or {}))
    wt = dict(walltime(lr["ms_per_image"], conf["rules"]), loader=lr)
    log("arm B's loader: %.2f ms per image (%d workers) -> a base run projected at %.2f h (line %.1f h) in %.0fs"
        % (lr["ms_per_image"], lr["workers"], wt["projected_base_run_h"], wt["line_h"], time.time() - tw))
    status = "over_walltime" if wt["over_line"] else "complete"
    summary = {
        "format": FORMAT, "status": status, "version": VERSION, "built_utc": C2.utc(), "testing": bool(testing),
        "decided_by": conf.get("decided_by"), "config": {"path": str(conf_path or CONFIG), "sha256": csha},
        "class_id": WEED_ID,
        "arms": {"A": {"name": ARM_A, "manifest": str(root / ("%s.jsonl" % ARM_A)), "sha256": sha_a,
                       "images": len(a), "boxes": sum(len(r["boxes"]) for r in a)},
                 "B": {"name": ARM_B, "manifest": str(root / ("%s.jsonl" % ARM_B)), "sha256": sha_b,
                       "images": len(b), "boxes": sum(len(r["boxes"]) for r in b),
                       "distinct_photos": distinct_photos(b)}},
        "files": {"provenance": {"file": PROVENANCE, "sha256": prov_sha, "rows": len(prov)},
                  "dropped": {"file": DROPPED, "sha256": drop_sha, "rows": len(drop_rows)},
                  "pngs_written": written, "pngs_removed_unused": removed},
        "holdout_v1": {"dir": HOLDOUT_DIR, "seed_text": conf["rules"]["holdout"]["seed_text"],
                       "per_source": hold_files, "images": sum(v["images"] for v in hold_files.values()),
                       "companions": {"dir": COMPANION_DIR, "per_source": comp_files,
                                      "rows": sum(v["rows"] for v in comp_files.values())}},
        "quarantine": sorted(quarantine), "sources_not_read": recs["not_read"], "evaluation_groups": eg_rec,
        "prior_test": prior_rec,
        "selection": sel, "per_source": ps, "guard": {k: grec.get(k) for k in ("reasons", "checked", "refused",
                                                                               "crosscheck_hits", "lock_sha256",
                                                                               "index_sha256")},
        "embedding": {k: v for k, v in (erec or {}).items() if k not in ("eval_descriptors",)},
        "leak_thresholds": dict(th, sha256=th_sha), "inputs": inputs, "walltime": wt,
        "seconds": round(time.time() - t0, 1)}
    summary = json.loads(json.dumps(summary, default=str))
    C2.write_json_atomic(root / SUMMARY, summary)
    if status != "complete":
        raise Base3Error("arm B's base run is projected at %.2f h, over the %.1f h line (loader %.2f ms per image): "
                         "splits v3 is written with status %s, so no E1 arm builds on it; a person decides"
                         % (wt["projected_base_run_h"], wt["line_h"], lr["ms_per_image"], status))
    log("base v3 built: arm A %d images, arm B %d images (%d distinct photos); held out %d; %s"
        % (len(a), len(b), summary["arms"]["B"]["distinct_photos"], summary["holdout_v1"]["images"], root / SUMMARY))
    return summary


# ---------------------------------------------------------------- walltime
def loader_rate(rows, sample=3200, batches=60, warm=5, batch=32, workers=5, seed_text="inc2/base3/loader"):
    """ms per image of Ultralytics' own training data pipeline (its YOLODataset
    with the default augmentation: mosaic, perspective, HSV, flips; no
    cache) over a seeded sample of an arm's rows, through its DataLoader with
    `workers` processes, as the base runs read them: the rate the loader can
    feed a GPU at. Returns {"ms_per_image", "images", "seconds", ...}."""
    import tempfile
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    rng = np.random.default_rng(C.stable_int(seed_text))
    pick = [rows[i] for i in sorted(rng.choice(len(rows), size=min(sample, len(rows)), replace=False).tolist())]
    with tempfile.TemporaryDirectory(prefix="base3_loader_") as tmp:
        C.materialise(pick, Path(tmp) / "data")
        cfg = get_cfg(overrides={"imgsz": IMGSZ, "cache": False, "workers": workers, "batch": batch, "task": "detect"})
        data = {"names": {i: n for i, n in enumerate(C.CLASS_NAMES)}, "nc": C.NC, "channels": 3}
        ds = build_yolo_dataset(cfg, str(Path(tmp) / "data" / "images"), batch, data, mode="train", rect=False,
                                stride=32)
        dl = build_dataloader(ds, batch, workers, shuffle=True)
        it = iter(dl)
        t0, n = None, 0
        for i in range(warm + batches):
            b = next(it)
            if i == warm - 1:
                t0 = time.time()
            elif i >= warm:
                n += int(len(b["im_file"]))
        secs = time.time() - (t0 or time.time())
    return {"ms_per_image": round(1000.0 * secs / max(1, n), 3), "images": n, "seconds": round(secs, 2),
            "sample": len(pick), "workers": workers, "batch": batch,
            "basis": "Ultralytics build_yolo_dataset (mode train, default augmentation, no cache) and its "
                     "DataLoader, %d batches after %d warm-up batches" % (batches, warm)}


def walltime(rate_ms, rules):
    """The projected base run of cold_budget against the cold limit: max(the
    loader's rate, m640's measured GPU rate) x the budget, plus the fixed
    overhead; over share x limit (D26's line) the arm cannot run."""
    from . import recipes as RC
    w = rules["walltime"]
    per = max(float(rate_ms), float(w["gpu_ms_per_image_epoch"]))
    h = per * RC.BUDGET_IMAGE_EPOCHS / 3.6e6 + float(w["fixed_hours"])
    line = float(w["share"]) * float(w["limit_hours"])
    return {"loader_ms_per_image": float(rate_ms), "gpu_ms_per_image_epoch": float(w["gpu_ms_per_image_epoch"]),
            "projected_base_run_h": round(h, 3), "line_h": line, "over_line": h > line, "rule": w.get("why")}


# ------------------------------------------------------------- the summary
def read_summary(root=None):
    """The complete summary.json of splits v3, or None."""
    p = Path(root or out_dir()) / SUMMARY
    try:
        with open(p) as fh:
            s = json.load(fh)
    except (OSError, ValueError):
        return None
    return s if isinstance(s, dict) and s.get("format") == FORMAT and s.get("status") == "complete" else None


def v3_claim(manifest, manifest_sha):
    """Why a manifest belongs to a base v3 build, or None: it lies under
    splits/v3, or a base v3 summary.json under any splits directory records
    its sha256 as an arm (whatever that summary's status). inc2.baseline
    trains such a manifest only as an E1 arm (e1_arm_of)."""
    try:
        if Path(os.path.abspath(str(manifest))).resolve().is_relative_to(out_dir().resolve()):
            return "it lies under %s" % out_dir()
    except (OSError, ValueError):
        pass
    for sp in sorted((Path(C.INC_DIR) / "splits").glob("*/%s" % SUMMARY)):
        try:
            with open(sp) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(s, dict) or s.get("format") != FORMAT:
            continue
        for k, a in sorted((s.get("arms") or {}).items()):
            if isinstance(a, dict) and a.get("sha256") == manifest_sha:
                return "%s records it as arm %s (status %s)" % (sp, k, s.get("status"))
    return None


def e1_arm_of(manifest_sha, root=None):
    """('A' or 'B', summary) when manifest_sha is one of the two E1 manifests
    a complete splits v3 summary records (and the manifest still hashes so),
    else (None, summary or None)."""
    s = read_summary(root)
    if s is None:
        return None, None
    for k, a in (s.get("arms") or {}).items():
        if a.get("sha256") == manifest_sha and _sha_file(a.get("manifest") or "") == manifest_sha:
            return k, s
    return None, s


# --------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Base v3 for E1: every admitted weed box as class 12.")
    ap.add_argument("command", choices=("build", "count"))
    ap.add_argument("--stream", required=True, help="the stream whose source quarantines apply")
    ap.add_argument("--out", default=None, help="count: the report directory (outside INC_DIR)")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--testing", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.command == "build":
            build(a.stream, testing=a.testing, procs=a.procs)
        else:
            if not a.out:
                raise Base3Error("count needs --out")
            count(a.stream, a.out, testing=a.testing)
    except Base3Error as e:
        print("[inc2.base3] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
