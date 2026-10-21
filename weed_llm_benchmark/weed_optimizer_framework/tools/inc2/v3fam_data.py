"""V3F data builders (lever L23H): the training manifests of D1, D4, D5b and
(if D0-S triggers it) BCLEAN, the D5b probe exam in its own root, the
near-copy doses and their catch index, and the both-direction never-train
check on every new manifest (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11):
V3F, the v3 experiment family D0-D9", V3F.3.7, V3F.4, V3F.5 D1/D4/D5b,
V3F.6.1).

    python -m weed_optimizer_framework.tools.inc2.v3fam_data build [--procs N]
    python -m weed_optimizer_framework.tools.inc2.v3fam_data lists [--base-b PATH] [--base-v2 PATH]
                                                                    (read-only: the key lists and the probe draw)

What is built (INC_DIR/splits/v3fam/, every manifest written once, summary.json
last; format inc2-v3fam-data/1, which inc2.base3.v3_claim never claims):

  arm      file                parent (rows byte-identical)   label form   recipe (V3F.3.8)
  D4F      d4_f.jsonl          base B (base_v3_weed)          one-class    cold_budget
  D4N      d4_n.jsonl          base B                          one-class    cold_budget
  D4R15    d4_r15.jsonl        base B                          one-class    cold_budget
  D1B1     d1_b1.jsonl         base B                          one-class    cold_budget
  D1A1     d1_a1.jsonl         base A one-class (base_v2_weed) one-class    cold_budget
  D1A12    d1_a12.jsonl        base_v2 (LOCK v2)               12-class     cold (100 epochs)
  D5B0     d5b_0.jsonl         base_v2 minus the probe         12-class     cold (100 epochs)
  D5B25    d5b_25.jsonl        ... plus 119 near copies        12-class     cold
  D5B100   d5b_100.jsonl       ... plus 476 near copies        12-class     cold
  D0BCLEAN d0_bclean.jsonl     base B minus D0-S's copies      one-class    cold_budget (only if triggered)

The setting map (V3F.4, decision 10): each base B row's setting by its exact
`source`, and for rf_tuf__weed-3434e by the file-name family of the original
name taken from the key (strip the source prefix -- train_core__, tsw22__,
tsw23__ for base_v2's parts, else "<source>__" --, a leading
train_images_/valid_images_/test_images_ and Roboflow's _<ext>_rf_<32 hex>
suffix), its rules tried in order: starts cwd_weed_ (Indoor; CWD30), starts
mh_ (Field), contains DSC_ (Field). A source the map does not list, or an
rf_tuf row of no family, refuses the build. F = the Field rows; N = base B
minus SIU (zenodo_15808623); R15 = the |F| rows of base B with the smallest
sha256("inc2/v3fam/d4/r15|" + key) (hex, then key). A key list's sha256 is
sha256 of the sorted keys joined by "\n"; before D0-S and the never-train
check each must equal the amendment's (REGISTERED: F cf108c70...144d, N
bd4861e0...15d7, R15 14eb7da8...f987, B' c2bb53a9...73119), and base B must
be 18d24ead...899a, or the build refuses.

D1 (decision 7): A' = base_v2 minus every tsw22/tsw23 row (3,697) and minus
the 26 train_core rows of session 20220129_CanonEOS4000D_EO (3,088 rows,
one-class from base_v2_weed.jsonl and 12-class from base_v2.jsonl, the same
keys); B' = base B minus the same rows (40,762). The season guard checks
every A' and B' file against ood22 and ood23 and drops every hit, base_v2
rows included, from A' and B' alike.

D5b (decision 11, Revision 26): the probe exam is whole train_core sessions
of at most PROBE_SESSION_MAX (100) images in ascending
stable_int("inc2/v3fam/d5b/probe/<session>"), a session taken while the probe
stays <= 500 images, stopping at the first total >= 400 (10 sessions, 476
images, keys 14604a97...8c90). It lives in its own root
(INC_DIR/v3fam/v1/probe_root: splits/v1/probe.jsonl -- base_v2's rows, bytes
unchanged --, splits/v1/LOCK.json with inc/scorer.py's sha256, and
exams/v1/probe: image symlinks and verified label copies), the probe exam only.
T_d = (base_v2 minus the probe) plus one near copy of each probe image in the
first round(d x 476) of the probe ranked by stable_int("inc2/v3fam/d5b/copy/
<key>") (so the doses nest), the family by rank mod 4: jpeg75 (Ultralytics'
load_image at 640, JPEG quality 75), crop90 (the centre 0.9 of the image as
Ultralytics reads it, then load_image; a box keeping < 50 % of its area or
under 2 px dropped), letterbox640 (load_image, grey 124 padding to 640 x 640,
centred), hflip (load_image, flipped). Each copy is a lossless PNG at long side
640 (content-addressed under splits/v3fam/d5b/, written once, read back
through load_image), passes the production guard and the both-direction
check, or is dropped and counted. The catch index (v3fam/v1/d5b/catch/
record.json) says per copy whether GuardV2 (one direction), the
both-direction dHash, the calibrated DINOv2 detector and D0-S's pipeline would
have matched it to its original were the probe never-train.

The never-train check (V3F.3.7, decision 16), on every new manifest (and,
recorded only, D8's train_core and D6's base_v2):
  1 inc2.train.check_manifest and guard_rows on the written file (fail
    closed); no row's key or sha256 in a test v1 list or its companions
    (inc2.base3.prior_test_lists);
  2 the 8-variant dHash within 6 bits in both directions (inc2.base3's pair
    search, as zoo.py uses it) against dev, test and ImageWeeds -- and ood22
    and ood23 for D1 --, on stored hashes: base v3's provenance for base B's
    files, base_v2's provenance for base_v2's original files, the zoo's exam
    hash cache (_zoo/v1/hashcache, checked against LOCK v1's manifests) for
    dev, test and ImageWeeds, the base_v2 copies' provenance hashes for ood22
    and ood23 (an ood image without a copy is hashed from its file); 100
    seeded rows of each source of each store recomputed must equal them.
    No dev, test or ImageWeeds image file is opened;
  3 the calibrated DINOv2 copy detector (LOCK v2's, facebook/dinov2-base:cls,
    cosine 0.946384; dev, test and ImageWeeds from LOCK v2's stored
    descriptors only) on every file no build scan has judged against that
    exam: base_v2's train_core and NDSU files and the copies against dev,
    test and ImageWeeds (base v3's scan judged base B's external rows, the v2
    build's scan the tsw rows); every A' and B' file against ood22 and ood23;
  4 every hit is dropped and counted, except: the one row of P0's record
    (KNOWN_FALSE_MATCH: sha256 36b99d86...cacb6, E1-B manifest line 12,882,
    against the test image f75aeb5c...7ae only), kept in every manifest
    derived from base B and recorded as known_false_match with P0's
    evidence; and base_v2's rows against dev, test and ImageWeeds, whose hits
    are recorded with the pair and kept. The season guard has no exception;
  5 D0-S's confirmed copies (inc2.v3fam_sweep.removal_keys) leave every
    family training set before any manifest is written, the list's sha256
    recorded; if F or R15 lost rows, R15 is redrawn by the same ranking over
    the rows of base B still eligible, taking F's final count.
Every recipe is recomputed from the arm's final count (cold_budget's record).

Inputs (load_inputs; all recorded by sha256): splits/v3/summary.json's arms
(base B, base A one-class) and provenance_v1.jsonl; LOCK v2's base_v2.jsonl,
train_core.jsonl and base_v2_provenance.jsonl; LOCK v1's ood22 and ood23
manifests; the zoo's exam hash cache; LOCK v2's calibration and stored
evaluation descriptors; D0-S's complete record. Nothing under splits/v1,
splits/v2, splits/v3, _zoo or twostage is written.

What the platform registration needs (not in this module; V3F.6.2-6.4):
run_inc2_build.sh verb "inc2.v3fam_data build" (provenance and lock
v3fam_data; HF_HUB_OFFLINE; this module and inc2/v3fam_sweep.py in the drift
list); lever L23H (MAINT, estimator v3fam_data 2.0 GPU-h, cap 4), proposed
only once inc2.v3fam_sweep.complete(); evidence.ALLOWED gains
splits/v3fam/summary.json; inc2.baseline build --v3fam ARM reads
summary["arms"][ARM] ("manifest", "sha256") and refuses any other
splits/v3fam manifest; inc2.train's v3fam_problems checks the summary's
sha256 and that cold_budget names one of BUDGET_ARMS; recipes' v3fam set takes
"n_images" from summary["arms"][ARM]["images"].

Deviation 4 (2026-10-20; docs/prereg/V3F_dev4_2026-10-20.md). The DINOv2
scans run once per distinct file and are keyed by the file's sha256: a
manifest key is not unique among the files scanned (6,805 of base_v2's 6,811
keys name base v3's re-written PNG in base B and the original file in
base_v2), and funnel.embed refuses a repeated key, which failed both builds
of 2026-10-12. Every hit names the file it is of (file_sha256) and counts for
its key whichever of the key's files it is of, so no arm's rule changes. A
build started over a failed build's partial outputs (splits/v3fam/d5b's
copies, v3fam/v1/leak/ood_desc.npz) writes beside them: the copies are
content-addressed and written once (a copy made again is the same file), the
season index's cache is reused only when its identity is this build's (else
funnel.leak refuses rather than overwrite it); every file present at the
start is listed in summary.json's "preexisting", must still hold its bytes at
the end (else the build refuses before the summary), and none is removed;
a cut-off attempt's transient files (funnel.embed's resume parts
<name>.partNNNNN.npz beside a scan's descriptor file, which the resumed scan
reuses and then removes, and temp files) are listed apart and need not
remain.

A refusal raises DataRefused (a V3famRefused); the CLI prints
"[inc2.v3fam_data] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from ..inc import common as C
from . import base3 as B3
from . import common as C2
from . import v3fam as F
from . import v3fam_sweep as SW

# ------------------------------------------------------------------ constants (pre-registered)
FORMAT = "inc2-v3fam-data/1"
FORMAT_PROBE_LOCK = "inc2-v3fam-probe-lock/1"
FORMAT_CATCH = "inc2-v3fam-d5b-catch/1"
SPLITS_NAME = "v3fam"
SUMMARY = "summary.json"
BASE_B_SHA256 = SW.BASE_B_SHA256              # 18d24ead...899a (E1-B's manifest)
BASE_B_ROWS = 44485
KEY_PREFIX = {"cottonweeddet12/train": "train_core__", "3seasonweeddet10/data2022": "tsw22__",
              "3seasonweeddet10/data2023": "tsw23__"}
SPLIT_PREFIX_RE = re.compile(r"^(train|valid|test)_images_")
RF_SUFFIX_RE = re.compile(r"_(jpg|jpeg|png|JPG|JPEG|PNG)_rf_[0-9a-f]{32}$")
RF_TUF = "rf_tuf__weed-3434e"
RF_TUF_RULES = (("starts", "cwd_weed_", "Indoor"), ("starts", "mh_", "Field"), ("contains", "DSC_", "Field"))
SETTINGS = ("Field", "Turf", "Indoor")
SETTING_OF_SOURCE = {
    "cottonweeddet12/train": "Field", "3seasonweeddet10/data2022": "Field", "3seasonweeddet10/data2023": "Field",
    "project_agml__weed_crop_detection": "Field", "project_agml__imageweeds_aerial_weed_detection": "Field",
    "project_agml__mh_weed16_weed_detection": "Field", "rf_school-5ult5__weed-6a90d": "Field",
    "gh_tehreemnoor__yolov5-weed-detection-model": "Field", "kg_yuzhenlu__cottonweeddet3": "Field",
    "rf_kinjj__weed-avnag": "Field",
    "rf_vizerion__weed-detection-5ell6": "Turf", "rf_unitec-qdvgo__weed-detection-in-grass": "Turf",
    "rf_computer-vision-for-traffic-management-9ot3y__weed-detection-7tviy": "Turf",
    "rf_itmo-mp0nn__grass-detection-4": "Turf", "rf_cotton-weed-detect__grass-weed-cotton-detect": "Turf",
    "rf_weed-detection-o9sfb__weed-detection-in-a-field": "Turf",
    "zenodo_15808623": "Indoor", "mediatum_1717366": "Indoor", "rf_new-workspace-zbm50__weeds-qftz4": "Indoor",
    RF_TUF: None}                                # by family (RF_TUF_RULES)
SETTING_COUNTS = {"Field": 15432, "Turf": 5910, "Indoor": 23143}
SIU = "zenodo_15808623"
R15_SEED_TEXT = "inc2/v3fam/d4/r15|"
R15_COMPOSITION = {"Field": 5402, "Turf": 2077, "Indoor": 7953, "SIU": 5393}
TSW_SOURCES = ("3seasonweeddet10/data2022", "3seasonweeddet10/data2023")
TRAIN_CORE_SOURCE = "cottonweeddet12/train"
NDSU_SOURCES = SW.NDSU_SOURCES
TSW_ROWS = 3697
SEASON_SESSION = "20220129_CanonEOS4000D_EO"
SEASON_SESSION_ROWS = 26
A_PRIME_ROWS = 3088
# list sha256 = sha256("\n".join(sorted(keys))), before D0-S and the never-train check (V3F.4, F4)
REGISTERED = {
    "F": (15432, "cf108c703d63e1a383322e1570bad349d09055bf19bab50859cb3d5a7383144d"),
    "N": (28953, "bd4861e0b42bd097bbe476c67ac00be97300ea41e3abf8374a840fe9930b15d7"),
    "R15": (15432, "14eb7da803a2df15b4ab8738dad67294cb74dbf255a01377805d77f62c48f987"),
    "BPRIME": (40762, "c2bb53a96bc5b6a4422b46e63464968cd61f674a6181e09de709182a3fe73119")}
KNOWN_FALSE_MATCH = {
    "key": "rf_school-5ult5__weed-6a90d__train_images_img_01665_png_jpg_rf_8a338581db11f56bc5f636c7bd25227a",
    "sha256": "36b99d86cbc9f29682005467dc3ec2626aa4c91734618344253df6c3622cacb6",
    "exam": "test", "exam_sha256": "f75aeb5c804e224b79c9ca07395124e7e39b6fa5f4b23433779f3e4976dac7ae",
    "exam_image": "cottonweeddet12 valid/20210903_iPhone11Pro_BT_148.jpg", "max_bits": 6, "manifest_line": 12882,
    "evidence": ("P0_AUDIT.md section 1 (sha256 c2a645ab...bc3ca), verify.json pairs[0] (8f0399bc...4108f): 6 dHash "
                 "bits in one direction only (the training image's id against the test image's rot90 variant; 23 "
                 "bits the other way), DINOv2-base CLS cosine 0.428 (best of 8 orientations 0.507; threshold "
                 "0.946384), the test image ranks 587th of 1,977; two different photographs on zoo_hits_montage.jpg, "
                 "checked visually by Claude on 2026-10-11; kept in every manifest derived from base B (V3F.3.7 "
                 "item 3)")}
# D5b (Revision 26)
PROBE_SEED_TEXT = "inc2/v3fam/d5b/probe/%s"
PROBE_SESSION_MAX = 100
PROBE_MAX = 500
PROBE_MIN = 400
PROBE_SESSIONS = 10
PROBE_IMAGES = 476
PROBE_KEYS_SHA256 = "14604a97f5e86b690fddb4e22e41f28ed56d41ad75dacfb3d502356da4958c90"
PROBE_BOXES = {"Waterhemp": 215, "MorningGlory": 138, "Purslane": 26, "SpottedSpurge": 64, "Carpetweed": 99,
               "Ragweed": 54, "Eclipta": 68, "PricklySida": 28, "PalmerAmaranth": 56, "Sicklepod": 0,
               "Goosegrass": 16, "CutleafGroundcherry": 2, "OtherPlant": 0}
COPY_SEED_TEXT = "inc2/v3fam/d5b/copy/%s"
COPY_FAMILIES = ("jpeg75", "crop90", "letterbox640", "hflip")      # by rank mod 4
COPY_KEY = "v3fam_d5b_copy__%s"
COPY_SOURCE = "v3fam/d5b_copy"
COPY_MIN_AREA = 0.5
COPY_MIN_PX = 2.0
COPY_JPEG_QUALITY = 75
COPY_GREY = 124
IMGSZ = 640
DOSES = (("D5B0", 0.0), ("D5B25", 0.25), ("D5B100", 1.0))
DOSE_ROWS = {"D5B0": 6335, "D5B25": 6454, "D5B100": 6811}
# the never-train check
BITS = 6
HASH_CHECK = 100
HASH_SEED_TEXT = "inc2/v3fam/data/hashcheck/%s/%s"
EXAMS_EVAL = ("dev", "test", "imageweeds")
EXAMS_SEASON = ("ood22", "ood23")
ZOO_VERSION = "v1"
# arms: file, parent, label form, recipe
ARMS = {
    "D4F": {"file": "d4_f.jsonl", "parent": "base_b", "form": "one-class", "recipe": "cold_budget"},
    "D4N": {"file": "d4_n.jsonl", "parent": "base_b", "form": "one-class", "recipe": "cold_budget"},
    "D4R15": {"file": "d4_r15.jsonl", "parent": "base_b", "form": "one-class", "recipe": "cold_budget"},
    "D1B1": {"file": "d1_b1.jsonl", "parent": "base_b", "form": "one-class", "recipe": "cold_budget"},
    "D1A1": {"file": "d1_a1.jsonl", "parent": "base_a1", "form": "one-class", "recipe": "cold_budget"},
    "D1A12": {"file": "d1_a12.jsonl", "parent": "base_v2", "form": "12-class", "recipe": "cold"},
    "D5B0": {"file": "d5b_0.jsonl", "parent": "base_v2", "form": "12-class", "recipe": "cold"},
    "D5B25": {"file": "d5b_25.jsonl", "parent": "base_v2", "form": "12-class", "recipe": "cold"},
    "D5B100": {"file": "d5b_100.jsonl", "parent": "base_v2", "form": "12-class", "recipe": "cold"},
    "D0BCLEAN": {"file": "d0_bclean.jsonl", "parent": "base_b", "form": "one-class", "recipe": "cold_budget"}}
BUDGET_ARMS = tuple(a for a, v in ARMS.items() if v["recipe"] == "cold_budget")
SEASON_ARMS = ("D1A1", "D1A12", "D1B1")
BUDGET_ARM = "m640"


class DataRefused(F.V3famRefused):
    pass


def log(msg):
    print("[inc2.v3fam_data] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths and small helpers
def out_dir():
    return C.INC_DIR / "splits" / SPLITS_NAME


def probe_root():
    return F.root_of("probe")


def catch_dir():
    return F.root() / "d5b" / "catch"


def list_sha256(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def manifest_line(row):
    """A manifest line exactly as inc.common.write_manifest writes it."""
    return (json.dumps({k: row[k] for k in C.MANIFEST_KEYS}, sort_keys=True) + "\n").encode("utf-8")


def parse_lines(data):
    """[(key, row, line bytes)] of a manifest's bytes, in order."""
    out = []
    for line in bytes(data).splitlines(keepends=True):
        if line.strip():
            r = json.loads(line)
            out.append((r["key"], r, line))
    return out


def read_parent(path, want_sha=None, what="manifest"):
    """{"path", "sha256", "bytes", "lines", "rows" {key: row}} of a parent manifest (by its sha256 when given)."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if want_sha is not None and sha != want_sha:
        raise DataRefused("%s %s hashes to %s, %s expected" % (what, path, sha[:12], str(want_sha)[:12]))
    lines = parse_lines(data)
    keys = [k for k, _r, _l in lines]
    if len(set(keys)) != len(keys) or keys != sorted(keys):
        raise DataRefused("%s %s: keys are not unique and sorted (a manifest inc.common wrote)" % (what, path))
    return {"path": str(path), "sha256": sha, "lines": lines, "rows": {k: r for k, r, _l in lines}}


def filter_parent(parent, keep, extra_lines=()):
    """(bytes, kept keys): the parent's lines whose key is in `keep`, byte for byte, merged in key order with
    extra (key, line) entries (D5b's copies). Refuses a kept key the parent lacks."""
    keep = set(keep)
    missing = sorted(keep - set(parent["rows"]))
    if missing:
        raise DataRefused("%d key(s) to keep are not rows of %s (e.g. %s)" % (len(missing), parent["path"],
                                                                             missing[:3]))
    items = [(k, line) for k, _r, line in parent["lines"] if k in keep] + list(extra_lines)
    items.sort(key=lambda t: t[0])
    ks = [k for k, _l in items]
    if len(set(ks)) != len(ks):
        raise DataRefused("an added row repeats a parent key")
    return b"".join(line for _k, line in items), ks


def write_once_bytes(path, data):
    """A manifest is written once: an existing file must hold exactly these bytes (base3's rule)."""
    path = Path(path)
    sha = hashlib.sha256(data).hexdigest()
    if path.is_file():
        if C.sha256_file(path) != sha:
            raise DataRefused("%s exists with other bytes: a family manifest is never overwritten" % path)
        return sha
    C2.write_bytes_atomic(path, data)
    return sha


# ------------------------------------------------------------------ the setting map (V3F.4)
def orig_name(row):
    """The original file name of a base B row, from its key (families.py orig())."""
    k, s = row["key"], row["source"]
    pre = KEY_PREFIX.get(s, s + "__")
    if not k.startswith(pre):
        raise DataRefused("key %r does not start with its source's prefix %r" % (k, pre))
    k = SPLIT_PREFIX_RE.sub("", k[len(pre):])
    return RF_SUFFIX_RE.sub("", k)


def setting_of(row):
    """(setting, family) of a base B row by the setting map; refuses a source the map does not list and an
    rf_tuf row of none of its three families."""
    s = row["source"]
    if s not in SETTING_OF_SOURCE:
        raise DataRefused("source %r is not in the setting map (V3F.4): a 21st source refuses the build" % s)
    if s != RF_TUF:
        return SETTING_OF_SOURCE[s], "all"
    o = orig_name(row)
    for how, text, setting in RF_TUF_RULES:
        if (how == "starts" and o.startswith(text)) or (how == "contains" and text in o):
            return setting, text.strip("_")
    raise DataRefused("rf_tuf row %s (original %r) matches none of its family rules (cwd_weed_, mh_, DSC_)"
                      % (row["key"], o))


def r15_rank(key):
    return hashlib.sha256((R15_SEED_TEXT + key).encode("utf-8")).hexdigest()


def r15_draw(keys, n):
    """The n keys with the smallest sha256(R15_SEED_TEXT + key) (hex order, then key)."""
    return sorted(sorted(keys, key=lambda k: (r15_rank(k), k))[:int(n)])


def d4_lists(base_rows):
    """{"F", "N", "R15"} key lists of base B (V3F.4), with the setting counts and R15's composition."""
    setting = {}
    fam = collections.Counter()
    for k, r in base_rows.items():
        st, fm = setting_of(r)
        setting[k] = st
        if r["source"] == RF_TUF:
            fam[fm] += 1
    f = sorted(k for k, st in setting.items() if st == "Field")
    n = sorted(k for k, r in base_rows.items() if r["source"] != SIU)
    r15 = r15_draw(list(base_rows), len(f))
    comp = collections.Counter(setting[k] for k in r15)
    comp["SIU"] = sum(1 for k in r15 if base_rows[k]["source"] == SIU)
    return {"F": f, "N": n, "R15": r15}, {"settings": dict(collections.Counter(setting.values())),
                                          "rf_tuf_families": dict(fam), "r15_composition": dict(comp)}, setting


def season_drop_keys(rows):
    """D1's removals from base_v2's rows: every tsw22/tsw23 row and the train_core rows of SEASON_SESSION
    (by its session field, which must agree with the key)."""
    tsw = sorted(k for k, r in rows.items() if r["source"] in TSW_SOURCES)
    ses = []
    for k, r in rows.items():
        if r["source"] != TRAIN_CORE_SOURCE:
            continue
        by_key = k.startswith("%s%s_" % (KEY_PREFIX[TRAIN_CORE_SOURCE], SEASON_SESSION))
        by_field = (r.get("session") or "") == SEASON_SESSION
        if by_key != by_field:
            raise DataRefused("train_core row %s: its session field %r and its key disagree about %s"
                              % (k, r.get("session"), SEASON_SESSION))
        if by_field:
            ses.append(k)
    return tsw, sorted(ses)


def check_registered(lists, base_sha, registered=REGISTERED, base_want=BASE_B_SHA256):
    """Refuse unless base B and every pre-registered key list are the amendment's: {name: record}."""
    if base_want is not None and base_sha != base_want:
        raise DataRefused("base B's manifest is %s, the amendment's %s: the key lists cannot be reproduced"
                          % (base_sha[:12], base_want[:12]))
    out = {}
    for name, (n, sha) in sorted(registered.items()):
        keys = lists[name]
        got = list_sha256(keys)
        out[name] = {"rows": len(keys), "sha256": got, "registered_rows": n, "registered_sha256": sha}
        if len(keys) != n or got != sha:
            raise DataRefused("key list %s: %d rows, sha256 %s; the amendment registers %d rows, %s (V3F.4): the "
                              "builder refuses rather than build another list" % (name, len(keys), got[:12], n,
                                                                                sha[:12]))
    return out


def key_lists(base_b, base_v2, registered=REGISTERED, base_want=BASE_B_SHA256, production=True):
    """Every pre-guard key list (F, N, R15, B', A') from the parents, checked against `registered` (the
    amendment's in production) and, in production, against the amendment's counts: (lists, record, setting)."""
    lists, rec, setting = d4_lists(base_b["rows"])
    tsw, ses = season_drop_keys(base_v2["rows"])
    drop = set(tsw) | set(ses)
    b_extra = sorted(set(base_b["rows"]) - set(base_v2["rows"]))
    if set(base_v2["rows"]) - set(base_b["rows"]):
        raise DataRefused("base B does not hold every base_v2 key: it is not E1-B's manifest")
    lists["BPRIME"] = sorted(k for k in base_b["rows"] if k not in drop)
    lists["APRIME"] = sorted(k for k in base_v2["rows"] if k not in drop)
    rec.update(tsw_rows=len(tsw), season_session_rows=len(ses), external_rows=len(b_extra))
    if registered is not None:
        rec["registered"] = check_registered(lists, base_b["sha256"], registered, base_want)
    if production:
        if len(tsw) != TSW_ROWS or len(ses) != SEASON_SESSION_ROWS or len(lists["APRIME"]) != A_PRIME_ROWS:
            raise DataRefused("A' removes %d tsw and %d session rows and holds %d rows; the amendment's %d, %d and "
                              "%d" % (len(tsw), len(ses), len(lists["APRIME"]), TSW_ROWS, SEASON_SESSION_ROWS,
                                      A_PRIME_ROWS))
        if rec["settings"] != SETTING_COUNTS:
            raise DataRefused("settings %s, the amendment's %s" % (rec["settings"], SETTING_COUNTS))
    return lists, rec, setting


# ------------------------------------------------------------------ D5b: the probe draw
def probe_draw(base_v2_rows):
    """(chosen sessions [(session, images)], probe keys) of the D5b draw (module docstring)."""
    by = collections.defaultdict(list)
    for k, r in base_v2_rows.items():
        if r["source"] == TRAIN_CORE_SOURCE:
            if not r.get("session"):
                raise DataRefused("train_core row %s has no session: whole sessions cannot be drawn" % k)
            by[r["session"]].append(k)
    order = sorted(by, key=lambda s: (C.stable_int(PROBE_SEED_TEXT % s), s))
    chosen, tot = [], 0
    for s in order:
        if tot >= PROBE_MIN:
            break
        if len(by[s]) > PROBE_SESSION_MAX:
            continue
        if tot + len(by[s]) <= PROBE_MAX:
            chosen.append((s, len(by[s])))
            tot += len(by[s])
    keys = sorted(k for s, _n in chosen for k in by[s])
    return chosen, keys


def check_probe(chosen, keys, production=True):
    rec = {"sessions": [[s, n] for s, n in chosen], "images": len(keys), "keys_sha256": list_sha256(keys)}
    if production and (len(chosen) != PROBE_SESSIONS or len(keys) != PROBE_IMAGES
                       or rec["keys_sha256"] != PROBE_KEYS_SHA256):
        raise DataRefused("the probe draw gives %d sessions, %d images, keys %s; the amendment's %d, %d, %s"
                          % (len(chosen), len(keys), rec["keys_sha256"][:12], PROBE_SESSIONS, PROBE_IMAGES,
                             PROBE_KEYS_SHA256[:12]))
    return rec


def copy_plan(probe_keys):
    """[(key, rank, family)] of the probe in copy rank order (stable_int(COPY_SEED_TEXT % key), then key)."""
    order = sorted(probe_keys, key=lambda k: (C.stable_int(COPY_SEED_TEXT % k), k))
    return [(k, i, COPY_FAMILIES[i % len(COPY_FAMILIES)]) for i, k in enumerate(order)]


def dose_keys(plan, d):
    """The probe keys copied at dose d: the first round(d x |probe|) of the plan (the doses nest)."""
    n = int(B3.round_half_up(float(d) * len(plan)))
    return [k for k, _i, _f in plan[:n]]


# ------------------------------------------------------------------ D5b: the near copies
def _ul_imread(path):
    from ultralytics.utils.patches import imread
    import cv2
    im = imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise DataRefused("cannot read %s" % path)
    return im


def transform_boxes(boxes, family, W, H, crop=None, out_wh=None, pad=None):
    """YOLO boxes [(cls, cx, cy, w, h)] (normalised to the W x H source) under a copy family: crop90 (crop =
    (x0, y0, x1, y1) in source pixels; a box keeping < COPY_MIN_AREA of its area, or under COPY_MIN_PX in the
    copy's out_wh pixels, is dropped), letterbox640 (out_wh = (w, h) of the resized image, pad = (left, top)),
    hflip, jpeg75 (unchanged). Returns (boxes, dropped)."""
    out, dropped = [], 0
    for c, cx, cy, bw, bh in boxes:
        if family == "jpeg75":
            out.append((c, cx, cy, bw, bh))
        elif family == "hflip":
            out.append((c, 1.0 - cx, cy, bw, bh))
        elif family == "letterbox640":
            w, h = out_wh
            left, top = pad
            out.append((c, (cx * w + left) / IMGSZ, (cy * h + top) / IMGSZ, bw * w / IMGSZ, bh * h / IMGSZ))
        elif family == "crop90":
            x0, y0, x1, y1 = crop
            a0, b0 = (cx - bw / 2) * W, (cy - bh / 2) * H
            a1, b1 = (cx + bw / 2) * W, (cy + bh / 2) * H
            area = max(0.0, a1 - a0) * max(0.0, b1 - b0)
            ca0, cb0, ca1, cb1 = max(a0, x0), max(b0, y0), min(a1, x1), min(b1, y1)
            carea = max(0.0, ca1 - ca0) * max(0.0, cb1 - cb0)
            cw, ch = float(x1 - x0), float(y1 - y0)
            ow, oh = out_wh
            if area <= 0 or carea < COPY_MIN_AREA * area or (ca1 - ca0) * ow / cw < COPY_MIN_PX \
                    or (cb1 - cb0) * oh / ch < COPY_MIN_PX:
                dropped += 1
                continue
            out.append((c, ((ca0 + ca1) / 2 - x0) / cw, ((cb0 + cb1) / 2 - y0) / ch, (ca1 - ca0) / cw,
                        (cb1 - cb0) / ch))
        else:
            raise DataRefused("no copy family %r (%s)" % (family, list(COPY_FAMILIES)))
    return [(int(c),) + tuple(min(1.0, max(0.0, float(v))) for v in b) for c, *b in out], dropped


def copy_array(image, family):
    """(BGR array the copy is written from, its geometry): every family at long side 640 as Ultralytics'
    load_image gives it (inc2.base3's materialisation)."""
    import numpy as np
    import cv2
    if family == "crop90":
        full = _ul_imread(image)
        H, W = full.shape[:2]
        w, h = int(round(0.9 * W)), int(round(0.9 * H))
        x0, y0 = (W - w) // 2, (H - h) // 2
        with tempfile.TemporaryDirectory(prefix="v3fam_crop_") as tmp:
            p = Path(tmp) / "crop.png"
            ok, buf = cv2.imencode(".png", full[y0:y0 + h, x0:x0 + w])
            if not ok:
                raise DataRefused("PNG encoding failed for %s" % image)
            p.write_bytes(buf.tobytes())
            im, _hw0 = B3._load_640(p)
        return im, {"W": W, "H": H, "crop": (x0, y0, x0 + w, y0 + h), "out_wh": (im.shape[1], im.shape[0])}
    base, (h0, w0) = B3._load_640(image)
    if base is None:
        raise DataRefused("cannot read %s" % image)
    h, w = base.shape[:2]
    geo = {"W": w0, "H": h0, "out_wh": (w, h)}
    if family == "jpeg75":
        ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), COPY_JPEG_QUALITY])
        if not ok:
            raise DataRefused("JPEG encoding failed for %s" % image)
        return cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR), geo
    if family == "hflip":
        return np.ascontiguousarray(base[:, ::-1]), geo
    if family == "letterbox640":
        top, left = (IMGSZ - h) // 2, (IMGSZ - w) // 2
        out = np.full((IMGSZ, IMGSZ, 3), COPY_GREY, dtype=np.uint8)
        out[top:top + h, left:left + w] = base
        return out, dict(geo, pad=(left, top))
    raise DataRefused("no copy family %r" % family)


def make_copy(row, family, root):
    """One near copy of a probe row (module docstring): its PNG and label written once, content-addressed under
    root/images and root/labels; returns the manifest row and its record."""
    import cv2
    from . import guard as G
    im, geo = copy_array(row["image"], family)
    boxes = C.read_yolo(row["label"])
    nb, dropped = transform_boxes(boxes, family, geo["W"], geo["H"], crop=geo.get("crop"), out_wh=geo["out_wh"],
                                  pad=geo.get("pad"))
    ok, buf = cv2.imencode(".png", im)
    if not ok:
        raise DataRefused("PNG encoding failed for the copy of %s" % row["key"])
    data = buf.tobytes()
    sha = hashlib.sha256(data).hexdigest()
    dst = Path(root) / "images" / sha[:2] / ("%s.png" % sha)
    B3._store_bytes(dst, data)
    back, _ = B3._load_640(dst)
    if back is None or back.shape != im.shape or not (back == im).all():
        raise DataRefused("%s does not read back through load_image as the pixels it was written from" % dst)
    text = "".join("%d %.6f %.6f %.6f %.6f\n" % b for b in nb).encode("utf-8")
    lsha = hashlib.sha256(text).hexdigest()
    lab = Path(root) / "labels" / lsha[:2] / ("%s.txt" % lsha)
    B3._store_bytes(lab, text)
    h, v = G.image_hashes(str(dst))
    mrow = {"image": str(dst), "label": str(lab), "sha256": sha, "label_sha256": lsha, "source": COPY_SOURCE,
            "session": row.get("session") or "", "key": COPY_KEY % row["key"]}
    rec = {"key": mrow["key"], "of": row["key"], "family": family, "boxes": len(nb), "boxes_dropped": dropped,
           "size": [int(im.shape[1]), int(im.shape[0])], "sha256": sha,
           "dhash": None if h is None else int(h),
           "variants": None if v is None else [int(v[k]) for k in G.VARIANTS]}
    return mrow, rec


# ------------------------------------------------------------------ the never-train check (V3F.3.7)
def near_hits(HA, okA, HE, okE, bits=BITS):
    """{(a, e): fewest bits} of every pair within `bits` under the 8 variants in both directions (A's variants
    against E's dHash, E's variants against A's dHash): inc2.base3's pair search, as zoo.py uses it."""
    import numpy as np
    out = {}
    if not len(HA) or not len(HE):
        return out
    Q1, T1, D1, _v = B3.near_pairs_between(HA, okA, HE, int(bits))
    T2, Q2, D2, _w = B3.near_pairs_between(HE, okE, HA, int(bits))
    for a, e, d in zip(np.r_[Q1, Q2].tolist(), np.r_[T1, T2].tolist(), np.r_[D1, D2].tolist()):
        k = (int(a), int(e))
        if k not in out or d < out[k]:
            out[k] = int(d)
    return out


def hash_matrix(entries):
    """(H [n, 8] uint64, ok [n]) of [{"dhash", "variants"}]; refuses an entry without its 8 variants."""
    import numpy as np
    H = np.zeros((len(entries), 8), dtype=np.uint64)
    for i, e in enumerate(entries):
        v = e.get("variants")
        if e.get("dhash") is None or not isinstance(v, (list, tuple)) or len(v) != 8 or int(v[0]) != int(e["dhash"]):
            raise DataRefused("%s has no stored dHash with its 8 variants: a file the check cannot compare is one it "
                              "cannot clear" % e.get("key", e.get("image")))
        H[i] = np.asarray([int(x) for x in v], dtype=np.uint64)
    return H, np.ones(len(entries), dtype=bool)


def hash_check(rows, store, what, n=HASH_CHECK, hasher=None):
    """V3F.3.7 item 2: for each source of the rows, n rows ranked by stable_int(HASH_SEED_TEXT % (what, key))
    recomputed (inc2.guard.image_hashes on the row's file) must equal the store's dHash and 8 variants."""
    from . import guard as G
    hasher = hasher or G.image_hashes
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("source") or ""].append(r)
    rec, bad = {}, []
    for s, rs in sorted(by.items()):
        pick = sorted(rs, key=lambda r: (C.stable_int(HASH_SEED_TEXT % (what, r["key"])), r["key"]))[:int(n)]
        for r in pick:
            e = store[r["key"]]
            h, v = hasher(r["image"])
            vals = None if v is None else [int(v[k]) for k in G.VARIANTS]
            if h is None or int(h) != int(e["dhash"]) or vals != [int(x) for x in e["variants"]]:
                bad.append([what, s, r["key"]])
        rec[s] = len(pick)
    if bad:
        raise DataRefused("%d stored hash(es) of %s differ from the file recomputed (e.g. %s): the stored hashes "
                          "cannot be used" % (len(bad), what, bad[:3]))
    return {"store": what, "per_source": rec, "checked": sum(rec.values()), "equal": True}


def file_entries(arm_rows, store):
    """[{"key", "image", "sha256", "dhash", "variants"}] of rows' files (the manifest's image and sha256 must be
    the store's for that key)."""
    out = []
    for k, r in arm_rows.items():
        e = store.get(k)
        if e is None or e.get("image") != r["image"] or e.get("sha256") != r["sha256"]:
            raise DataRefused("no stored hashes for %s's file %s (the store records %s)" % (k, r["image"],
                                                                                          (e or {}).get("image")))
        out.append({"key": k, "image": r["image"], "sha256": r["sha256"], "dhash": e["dhash"],
                    "variants": e["variants"]})
    return out


def dhash_check(entries, exams):
    """Each file against each exam's hashes: {key: [{"exam", "exam_key", "exam_sha256", "bits", "how",
    "file_sha256"}]}. A key can name two files (a base_v2 key's base B file is base v3's re-written PNG, its base_v2
    file the original), so each hit names the file it is of (deviation 4)."""
    hits = collections.defaultdict(list)
    if not entries:
        return hits
    H, ok = hash_matrix(entries)
    for exam, ex in sorted(exams.items()):
        for (a, e), d in sorted(near_hits(H, ok, ex["H"], ex["ok"]).items()):
            hits[entries[a]["key"]].append({"exam": exam, "exam_key": ex["keys"][e], "exam_sha256": ex["sha256"][e],
                                            "bits": d, "how": "dhash", "file_sha256": entries[a].get("sha256")})
    return hits


def embed_check(entries, scanner, desc_path=None, procs=1):
    """The calibrated DINOv2 detector on files: {key: [{"exam", "exam_key", "cos", "bits", "how"}]}, keyed by the
    entries' "key" (the scanner's row key, which must be unique: funnel.embed refuses a repeated one); an entry's
    "row_key" (the manifest key, when the scan is keyed otherwise) rides along for the refusal's message and is
    ignored by inc2.guard.EmbedScanner. A file it cannot describe refuses (fail closed)."""
    hits = collections.defaultdict(list)
    if not entries:
        return hits
    rows = [dict({"key": e["key"], "image": e["image"], "sha256": e["sha256"]},
                 **({"row_key": e["row_key"]} if e.get("row_key") is not None else {})) for e in entries]
    copies, unscannable = scanner.scan(rows, desc_path=desc_path, procs=procs)
    if unscannable:
        named = {r["key"]: r.get("row_key") or r["key"] for r in rows}
        raise DataRefused("%d file(s) the DINOv2 detector cannot describe (e.g. %s): fail closed"
                          % (len(unscannable), sorted(named.get(k, k) for k in unscannable)[:3]))
    for k, es in copies.items():
        for e in es:
            hits[k].append({"exam": e.get("eval_split") or e.get("split"), "exam_key": e.get("eval_key"),
                            "cos": e.get("cos"), "bits": e.get("bits"), "how": "embed"})
    return hits


def is_known_false_match(key, row, hits):
    """True when every hit of the row is P0's pair: the row of KNOWN_FALSE_MATCH against its test image."""
    k = KNOWN_FALSE_MATCH
    return (key == k["key"] and row.get("sha256") == k["sha256"] and bool(hits)
            and all(h["exam"] == k["exam"] and h.get("exam_sha256") == k["exam_sha256"] and h["how"] == "dhash"
                    and h["bits"] <= k["max_bits"] for h in hits))


def judge(rows, hits, base_v2_keys, season=False):
    """V3F.3.7 item 3 on one set of rows: (dropped {key: hits}, kept {key: {"why", "hits"}}). A hit drops its
    row except the known false match (dev/test/ImageWeeds check only) and base_v2's rows against dev, test and
    ImageWeeds; the season guard (ood22, ood23) drops every hit."""
    dropped, kept = {}, {}
    for k in sorted(hits):
        hs = hits[k]
        if k not in rows or not hs:
            continue
        if season:
            dropped[k] = hs
        elif is_known_false_match(k, rows[k], hs):
            kept[k] = {"why": "known_false_match", "hits": hs, "evidence": KNOWN_FALSE_MATCH["evidence"]}
        elif k in base_v2_keys:
            kept[k] = {"why": "base_v2_row", "hits": hs}
        else:
            dropped[k] = hs
    return dropped, kept


def merge_hits(*dicts):
    out = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out[k].extend(v)
    return out


def prior_rows():
    """{key}, {sha256} of every earlier test v1 list and its companions (inc2.base3.prior_test_lists)."""
    rows, rec = B3.prior_test_lists()
    keys = {str(r.get("key")) for r in rows if r.get("key")}
    shas = {str(s) for r in rows for s in (r.get("sha256"), r.get("original_sha256")) if s}
    return keys, shas, rec


def check_prior(arm, data, keys, shas):
    hit = [r["key"] for _k, r, _l in parse_lines(data) if r["key"] in keys or r["sha256"] in shas]
    if hit:
        raise DataRefused("%s holds %d row(s) of an earlier test v1 list or its companions (e.g. %s): never trained"
                          % (arm, len(hit), hit[:3]))


def train_checks(path, production=True):
    """V3F.3.7 item 1 on a written manifest: inc2.train.check_manifest and guard_rows (fail closed)."""
    from . import train as T
    try:
        rows, dh, rec = T.check_manifest(path)
        grec = T.guard_rows(rows, dh, production=production)
    except T.RunError as e:
        raise DataRefused("%s fails inc2.train's checks: %s" % (path, e))
    return {"check_manifest": {k: rec[k] for k in ("train_manifest_sha256", "n_train_images", "n_train_boxes")},
            "guard": {k: grec.get(k) for k in ("checked", "refused", "crosscheck_hits", "reasons", "lock_sha256",
                                                "index_sha256")}}


# ------------------------------------------------------------------ inputs (production)
def _zoo_exam_hashes(version=ZOO_VERSION):
    """{exam: {"keys", "sha256", "H", "ok"}} of dev, test and ImageWeeds from the zoo's exam hash cache, each
    checked against LOCK v1's manifest (the same keys and sha256s, in key order): no exam image is opened."""
    import numpy as np
    from . import zoo as Z
    meta, arrays = Z.load_exam_hashes(version)
    zd = Z.zoo_dir(version)
    if meta.get("npz_sha256") != C.sha256_file(zd / "hashcache" / "exams.npz"):
        raise DataRefused("the zoo's exams.npz does not hash as its exams.json records")
    lock = C.read_lock()
    out = {}
    for exam in EXAMS_EVAL:
        C.verify_manifest_against_lock(exam, lock)
        man = sorted(C.read_manifest(C.manifest_path(exam)), key=lambda r: r["key"])
        em = meta["exams"][exam]
        if list(em.get("keys") or []) != [r["key"] for r in man] or \
                list(em.get("sha256") or []) != [r["sha256"] for r in man]:
            raise DataRefused("the zoo's %s hashes are not of LOCK v1's %s manifest (keys or sha256s differ)"
                              % (exam, exam))
        has = np.asarray(arrays["exam_%s__has" % exam], dtype=bool)
        if not has.all():
            raise DataRefused("the zoo's cache leaves %d %s image(s) unhashed: fail closed" % (int((~has).sum()), exam))
        out[exam] = {"keys": [r["key"] for r in man], "sha256": [r["sha256"] for r in man],
                     "H": np.asarray(arrays["exam_%s__H" % exam], dtype=np.uint64),
                     "ok": np.asarray(arrays["exam_%s__ok" % exam], dtype=bool)}
    return out, {"exams_json_sha256": C.sha256_file(zd / "hashcache" / "exams.json"),
                 "exams_npz_sha256": meta.get("npz_sha256")}


def _ood_exam_hashes(v2prov):
    """{ood22, ood23: {"keys", "sha256", "H", "ok", "image" (the file hashed)}} from base_v2's provenance (the
    tsw copies, by sha256); an image without a base_v2 copy is hashed from its exam file."""
    from . import guard as G
    by_sha = {}
    for e in v2prov.values():
        by_sha.setdefault(e["sha256"], e)
    lock = C.read_lock()
    out, rec = {}, {}
    for exam in EXAMS_SEASON:
        C.verify_manifest_against_lock(exam, lock)
        man = sorted(C.read_manifest(C.manifest_path(exam)), key=lambda r: r["key"])
        ents, here = [], []
        for r in man:
            e = by_sha.get(r["sha256"])
            if e is not None:
                ents.append({"key": r["key"], "dhash": e["dhash"], "variants": e["variants"], "image": e["image"]})
            else:
                h, v = G.image_hashes(r["image"])
                if h is None or v is None:
                    raise DataRefused("%s image %s cannot be hashed: fail closed" % (exam, r["key"]))
                ents.append({"key": r["key"], "dhash": int(h), "variants": [int(v[k]) for k in G.VARIANTS],
                             "image": r["image"]})
                here.append(r["key"])
        H, ok = hash_matrix(ents)
        out[exam] = {"keys": [r["key"] for r in man], "sha256": [r["sha256"] for r in man], "H": H, "ok": ok,
                     "image": [e["image"] for e in ents]}
        rec[exam] = {"images": len(man), "from_base_v2_copies": len(man) - len(here), "hashed_here": here}
    return out, rec


def _provenance(path, want_sha, what):
    """{key: {"image", "sha256", "dhash", "variants", "source"}} of a provenance file (by its sha256)."""
    got = C.sha256_file(path)
    if got != want_sha:
        raise DataRefused("%s %s hashes to %s, %s recorded" % (what, path, got[:12], str(want_sha)[:12]))
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = {"key": r["key"], "image": r["image"], "sha256": r["sha256"], "dhash": r.get("dhash"),
                                 "variants": r.get("variants"), "source": r.get("source"),
                                 "arms": r.get("arms")}
    return out


def eval_scanner(embedder=None):
    """The calibrated DINOv2 detector against dev, test and ImageWeeds (LOCK v2's calibration and evaluation
    index) from LOCK v2's stored descriptors only: a cache that is not current refuses rather than open an
    evaluation image."""
    from ..funnel import embed as FE
    from ..funnel import leak as L
    from . import embed_calibration as EC
    from . import guard as G
    cal, why = EC.locked(C2.LOCK_PATH, production=True)
    if cal is None:
        raise DataRefused("LOCK v2 records no calibration (%s): the DINOv2 check cannot run (fail closed)" % why)
    model, _, pooling = str(cal["embedder"]).rpartition(":")
    emb = embedder or FE.LazyEmbedder(model, pooling)
    rows = G.v2_eval_rows(C2.LOCK_PATH)
    cache = C2.SPLITS_DIR / "leak" / "eval_desc_v2.npz"
    items = [{"key": "%s|%s" % (s, r["key"]), "image": r["image"], "sha256": r.get("sha256")}
             for s in sorted(rows) for r in sorted(rows[s], key=lambda r: r["key"])]
    if not FE.image_file_current(cache, FE.image_ident(items, emb, L.prepare_hashed, len(L.VARIANTS))):
        raise DataRefused("%s is not LOCK v2's current evaluation descriptors: the check refuses rather than open "
                          "an evaluation image" % cache)
    return G.EmbedScanner(G.eval_index(rows, emb, cache), cal)


def ood_scanner(ood, embedder=None, cache=None):
    """The calibrated DINOv2 detector against ood22 and ood23 (D1's season guard), its index made from the files
    the ood hashes came from (base_v2's tsw copies; the exam file without a copy)."""
    from ..funnel import embed as FE
    from . import embed_calibration as EC
    from . import guard as G
    cal, why = EC.locked(C2.LOCK_PATH, production=True)
    if cal is None:
        raise DataRefused("LOCK v2 records no calibration (%s): the season guard cannot run" % why)
    model, _, pooling = str(cal["embedder"]).rpartition(":")
    emb = embedder or FE.LazyEmbedder(model, pooling)
    rows = {e: [{"key": k, "image": p, "sha256": s} for k, p, s in zip(ood[e]["keys"], ood[e]["image"],
                                                                         ood[e]["sha256"])] for e in EXAMS_SEASON}
    cache = Path(cache or (F.root() / "leak" / "ood_desc.npz"))
    return G.EmbedScanner(G.eval_index(rows, emb, cache), cal)


def load_inputs(production=True, procs=5, embedder=None):
    """Every input of the build (module docstring), checked and recorded."""
    summ = B3.read_summary()
    if summ is None:
        raise DataRefused("no complete splits/v3/summary.json: base B is not built")
    arms = summ["arms"]
    base_b = read_parent(arms["B"]["manifest"], arms["B"]["sha256"], "base B")
    base_a1 = read_parent(arms["A"]["manifest"], arms["A"]["sha256"], "base A (one-class)")
    lock2 = C2.read_lock_v2()
    base_v2 = read_parent(C2.v2_manifest_path("base_v2"), lock2["manifests"]["base_v2"], "base_v2")
    train_core = read_parent(C2.v2_manifest_path("train_core"), lock2["manifests"]["train_core"], "train_core")
    pf = summ["files"]["provenance"]
    hash_b = _provenance(B3.out_dir() / pf["file"], pf["sha256"], "base v3 provenance")
    hash_v2 = _provenance(C2.BASE_PROVENANCE, lock2.get("provenance_sha256"), "base_v2 provenance")
    exams, zrec = _zoo_exam_hashes()
    ood, orec = _ood_exam_hashes(hash_v2)
    checks = [hash_check([dict(r, key=k) for k, r in base_b["rows"].items()], hash_b, "base_v3_provenance"),
              hash_check([dict(r, key=k) for k, r in base_v2["rows"].items()], hash_v2, "base_v2_provenance")]
    ood_rows = [{"key": k, "image": p, "source": e} for e in EXAMS_SEASON
                for k, p in zip(ood[e]["keys"], ood[e]["image"])]
    ood_store = {k: {"dhash": int(ood[e]["H"][i][0]), "variants": [int(x) for x in ood[e]["H"][i]]}
                 for e in EXAMS_SEASON for i, k in enumerate(ood[e]["keys"])}
    checks.append(hash_check(ood_rows, ood_store, "ood_base_v2_copies"))
    keys, shas, prec = prior_rows()
    sweep = SW.read_record()
    if sweep is None:
        raise DataRefused("D0-S's record is not complete: L23H writes no manifest before it (V3F.5 D0-S)")
    scanners = {"eval": eval_scanner(embedder), "ood": ood_scanner(ood, embedder)}
    return {"base_b": base_b, "base_a1": base_a1, "base_v2": base_v2, "train_core": train_core,
            "hash_b": hash_b, "hash_v2": hash_v2, "exams": dict(exams, **ood), "scanners": scanners,
            "sweep": sweep, "prior": (keys, shas), "production": bool(production), "registered": REGISTERED,
            "base_want": BASE_B_SHA256,
            "record": {"splits_v3_summary_sha256": C.sha256_file(B3.out_dir() / B3.SUMMARY),
                       "lock_v2_sha256": C.sha256_file(C2.LOCK_PATH), "lock_v1_sha256": C.sha256_file(C.LOCK_PATH),
                       "zoo_hashcache": zrec, "ood_hashes": orec, "hash_checks": checks, "prior_test": prec,
                       "sweep_record_sha256": C.sha256_file(SW.sweep_dir() / "record.json")}}


# ------------------------------------------------------------------ the probe root
def probe_tags(base_v2, probe_keys):
    """{tag: images} of the probe images' EXIF orientations; refuses one a family pass on the probe does not read
    (twostage.OK_ORIENTATIONS: none, 0, 1), before any copy is made in that frame."""
    from . import twostage as TS
    tags = collections.Counter()
    for k in probe_keys:
        ori, _wh = TS.header(base_v2["rows"][k]["image"])
        tags[str(ori)] += 1
        if ori not in TS.OK_ORIENTATIONS:
            raise DataRefused("probe image %s carries EXIF orientation %s, which a pass on the probe refuses" % (k, ori))
    return dict(tags)


def write_probe_root(base_v2, probe_keys, root=None):
    """The probe exam in its own root (Z1's pattern, the probe exam only): splits/v1/probe.jsonl (base_v2's rows,
    bytes unchanged), splits/v1/LOCK.json (inc/scorer.py's sha256), exams/v1/probe (symlinks, verified label
    copies). Every probe image's EXIF orientation must be one a family pass reads on the probe."""
    from ..inc import scorer as S
    from ..inc import splits as SP
    root = Path(root or probe_root())
    data, keys = filter_parent(base_v2, probe_keys)
    rows = [r for _k, r, _l in parse_lines(data)]
    tags = probe_tags(base_v2, probe_keys)
    sha = write_once_bytes(root / "splits" / "v1" / "probe.jsonl", data)
    lock = {"format": FORMAT_PROBE_LOCK, "manifests": {"probe": sha}, "splits_version": "v1",
            "scorer_sha256": C.sha256_file(Path(S.__file__).resolve()), "keys_sha256": list_sha256(keys),
            "base_v2_sha256": base_v2["sha256"], "decided_by": F.DECIDED_BY}
    lp = root / "splits" / "v1" / "LOCK.json"
    old = F._read_json(lp)
    if isinstance(old, dict):
        if {k: old.get(k) for k in lock} != lock:
            raise DataRefused("%s exists with other contents: the probe's LOCK is written once" % lp)
    else:
        F.write_once(lp, dict(lock, created_utc=F._utc()))
    ed = root / "exams" / "v1" / "probe"
    if not ed.exists():
        C.materialise(rows, ed)
        for n in os.listdir(ed / "labels"):
            os.chmod(ed / "labels" / n, 0o444)
        os.chmod(ed / "data.yaml", 0o444)
    probs = SP.exam_problems("probe", rows, ed)
    if probs:
        raise DataRefused("the probe exam %s has problems: %s" % (ed, probs[:3]))
    counts = C.class_counts(rows)
    return {"root": str(root), "manifest_sha256": sha, "lock_sha256": C.sha256_file(lp), "images": len(rows),
            "boxes": counts, "exif_tags": tags}


# ------------------------------------------------------------------ the catch index (descriptive)
def catch_index(probe_rows, copies, scanner_factory=None, embedder=None, procs=1, testing=False):
    """Per copy, whether each guard would have matched it to its original were the probe never-train: GuardV2
    (one direction), the both-direction dHash, the calibrated DINOv2 detector and D0-S's pipeline; written once
    under v3fam/v1/d5b/catch/. Descriptive: nothing is dropped by it."""
    import numpy as np
    from . import guard as G
    cd = catch_dir()
    if (cd / "record.json").exists():
        return F._read_json(cd / "record.json")
    orig = sorted(probe_rows, key=lambda r: r["key"])
    oh = [dict(zip(("dhash", "variants"), (lambda hv: (hv[0], None if hv[1] is None else
                                                       [int(hv[1][k]) for k in G.VARIANTS]))(G.image_hashes(r["image"]))),
               key=r["key"]) for r in orig]
    guard = G.GuardV2([(e["dhash"], "probe", e["key"]) for e in oh])
    Ho, oko = hash_matrix(oh)
    cents = [{"key": c["key"], "dhash": c["dhash"], "variants": c["variants"]} for c in copies]
    Hc, okc = hash_matrix(cents) if cents else (np.zeros((0, 8), np.uint64), np.zeros(0, bool))
    both = near_hits(Hc, okc, Ho, oko)
    idx = {r["key"]: i for i, r in enumerate(orig)}
    per = []
    for i, c in enumerate(copies):
        reason, match = guard.check(c["dhash"], c["variants"])
        per.append({"key": c["key"], "of": c["of"], "family": c["family"],
                    "guardv2": bool(reason in ("near_eval_v2", "near_eval_variant")
                                    and (match or {}).get("key") == c["of"]),
                    "dhash_both": (i, idx[c["of"]]) in both})
    if scanner_factory is not None:
        scanner = scanner_factory({"probe": [{"key": r["key"], "image": r["image"], "sha256": r["sha256"]}
                                             for r in orig]})
        found, unscannable = scanner.scan([{"key": c["key"], "image": c["image"], "sha256": c["sha256"]}
                                           for c in copies], desc_path=cd / "copy_desc.npz", procs=procs)
        if unscannable:
            raise DataRefused("%d copies the DINOv2 detector cannot describe" % len(unscannable))
        for p in per:
            p["dinov2"] = any(e.get("eval_key") == p["of"] for e in found.get(p["key"]) or [])
    if embedder is not None:
        q = [{"key": r["key"], "image": r["image"], "sha256": r["sha256"]} for r in orig]
        rr = [{"key": c["key"], "image": c["image"], "sha256": c["sha256"]} for c in copies]
        res = SW.pipeline(q, rr, embedder, work_dir=cd / "sweep_desc", procs=procs)
        ver = {(p["query"], p["ref"]) for p in res["pairs"] if p["verified"]}
        for p in per:
            p["d0s"] = (p["of"], p["key"]) in ver
    fam = {}
    for f in COPY_FAMILIES:
        ps = [p for p in per if p["family"] == f]
        fam[f] = {"copies": len(ps)}
        for g in ("guardv2", "dhash_both", "dinov2", "d0s"):
            vals = [p[g] for p in ps if g in p]
            fam[f][g] = None if not vals else round(sum(vals) / float(len(vals)), 4)
    rec = {"format": FORMAT_CATCH, "testing": bool(testing), "copies": per, "per_family": fam,
           "note": "descriptive: whether each guard would have matched each copy to its original were the probe "
                   "never-train (no training, nothing dropped)", "decided_by": F.DECIDED_BY,
           "created_utc": F._utc()}
    F.write_once(cd / "record.json", rec)
    return rec


# ------------------------------------------------------------------ the build
def _summary_path():
    return out_dir() / SUMMARY


def _budget(n):
    from . import recipes as RC
    return RC.budget_record(BUDGET_ARM, n)


def _file_hits(files, dh, emb_by_sha, scanned):
    """{key: hits} of files: each file's own dHash hits (dh's hits that name its sha256) plus the DINOv2 hits of its
    file when it was scanned, every file of a key added to that key's hits (a key can name two files: a base_v2
    key's base B file is base v3's re-written PNG, its base_v2 file the original; deviation 4). A hit drops or keeps
    its key whichever of the key's files it is of (V3F.3.7 item 3)."""
    out = {}
    for f in files:
        hs = [h for h in dh.get(f["key"], []) if h.get("file_sha256", f["sha256"]) == f["sha256"]]
        if f["sha256"] in scanned:
            hs += [dict(h, file_sha256=f["sha256"]) for h in emb_by_sha.get(f["sha256"], [])]
        if hs:
            out.setdefault(f["key"], []).extend(hs)
    return out


def _scan_unique(files, scanner, desc_path, procs):
    """The DINOv2 detector once per distinct file (by sha256): ({sha256: hits}, the scanned sha256s). The scanner's
    rows are keyed by the file's sha256, unique by construction, and each file's hits are its own: the manifest key
    is not unique among the files (6,805 of base_v2's 6,811 keys name base v3's re-written PNG in base B and the
    original file in base_v2, 3,108 of them in the first scan, which made funnel.embed refuse 'duplicate keys' in
    both data builds of 2026-10-12; deviation 4)."""
    uniq = {}
    for f in files:
        uniq.setdefault(f["sha256"], f)
    got = embed_check([{"key": sha, "row_key": f["key"], "image": f["image"], "sha256": sha}
                       for sha, f in uniq.items()], scanner, desc_path=desc_path, procs=procs)
    return {sha: list(got.get(sha, [])) for sha in uniq}, set(uniq)


# A build's own transient files, which a build cut off (a node lost, scancel, a requeue, ENOSPC) leaves behind and the
# next attempt reuses or replaces: funnel.embed's resume parts beside a scan's descriptor file
# (<name>.partNNNNN.npz, reused and then removed by the scan that resumes them) and the temp files of an atomic write
# (<name>.tmp, .<name>.<pid>.<ns>.tmp). They are listed, never required to remain (deviation 4).
TRANSIENT_RE = re.compile(r"(?:\.part\d{5}\.npz|\.tmp)\Z")


def _preexisting():
    """{path: {"bytes", "sha256"[, "transient"]}} of every file already under splits/v3fam/ and v3fam/v1/leak/ when a
    build starts: a failed build's partial outputs (the copies under splits/v3fam/d5b/, written content-addressed and
    once; leak/ood_desc.npz, the season index's cache, reused only when its identity is the one this build would
    make, else funnel.leak refuses rather than overwrite it), and its transient files (TRANSIENT_RE: a cut-off
    scan's resume parts, which this build's scan reuses and removes, and temp files), marked so. The build writes
    beside them and deletes none of the outputs (deviation 4)."""
    out = {}
    for d in (out_dir(), F.root() / "leak"):
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    out[str(p)] = dict({"bytes": p.stat().st_size, "sha256": C.sha256_file(p)},
                                       **({"transient": True} if TRANSIENT_RE.search(p.name) else {}))
    return out


def _preexisting_record(pre, copy_rows):
    """What became of the files present when the build started: each output must still hold its bytes (a changed or
    missing one refuses before the summary is written); the copy files the build made again (the same bytes, so the
    same content-addressed path) and the rest, listed, none removed. The transient files (a cut-off scan's resume
    parts, temp files) are listed apart with whether they remain: the scan that resumes from its parts removes them
    when it completes, so they are not required to remain."""
    outs = {p: e for p, e in pre.items() if not e.get("transient")}
    changed = sorted(p for p, e in outs.items() if not Path(p).is_file() or C.sha256_file(p) != e["sha256"])
    if changed:
        raise DataRefused("%d file(s) present before the build changed or went missing during it (e.g. %s): a "
                          "build writes beside earlier outputs and never over them" % (len(changed), changed[:3]))
    used = {str(r[k]) for r in copy_rows.values() for k in ("image", "label")}
    listing = [[p, outs[p]["bytes"], outs[p]["sha256"]] for p in sorted(outs)]
    return {"files": len(outs), "unchanged": True,
            "copy_files_made_again": sum(1 for p in outs if p in used),
            "not_used": sorted(p for p in outs if p not in used),
            "listing_sha256": hashlib.sha256(json.dumps(listing).encode("utf-8")).hexdigest(),
            "transient": [[p, pre[p]["bytes"], pre[p]["sha256"], Path(p).is_file()]
                          for p in sorted(pre) if pre[p].get("transient")],
            "note": "files under splits/v3fam/ and v3fam/v1/leak/ before this build (a failed build's partial "
                    "outputs): kept as they were; copies made again are the same files; transient: [path, bytes, "
                    "sha256, still present] of a cut-off scan's resume parts and temp files, reused or replaced by "
                    "this build (deviation 4)"}


def build(inputs=None, procs=5, embedder=None, testing=False, catch=True):
    """L23H (module docstring), the E-series trees unchanged around it (V3F.3.7). Returns summary.json's
    content."""
    with F.e_trees_unchanged():
        return _build(inputs=inputs, procs=procs, embedder=embedder, testing=testing, catch=catch)


def _build(inputs=None, procs=5, embedder=None, testing=False, catch=True):
    t0 = time.time()
    if _summary_path().exists():
        raise DataRefused("%s exists: the family's manifests are built once" % _summary_path())
    F.check_frozen()
    ts = SW.require_timestamp()
    earlier = _preexisting()                 # before anything is written (load_inputs may write ood_desc.npz)
    if inputs is None:
        if testing:
            raise DataRefused("a testing build is given its inputs")
        inputs = load_inputs(production=True, procs=procs, embedder=embedder)
    prod = bool(inputs["production"])
    if not prod and not testing:
        raise DataRefused("inputs that are not production's build only in testing")
    base_b, base_a1, base_v2 = inputs["base_b"], inputs["base_a1"], inputs["base_v2"]
    if prod and (inputs.get("registered") != REGISTERED or inputs.get("base_want") != BASE_B_SHA256):
        raise DataRefused("a production build checks the amendment's key lists and base B")
    lists, lrec, _setting = key_lists(base_b, base_v2, inputs.get("registered"), inputs.get("base_want"),
                                      production=prod)
    v2keys = set(base_v2["rows"])
    if set(base_a1["rows"]) != v2keys:
        raise DataRefused("base A (one-class) and base_v2 do not hold the same keys")
    # ---- D0-S's removals, before any manifest
    rm_keys, rm_sha = SW.removal_keys(inputs["sweep"])
    rm = set(rm_keys)
    unknown = sorted(rm - set(base_b["rows"]))
    if unknown:
        raise DataRefused("D0-S's removal list names %d key(s) that are not base B rows (e.g. %s)"
                          % (len(unknown), unknown[:3]))
    trig = inputs["sweep"]["trigger"]
    # ---- the probe (its EXIF orientations first) and its near copies
    chosen, probe_keys = probe_draw(base_v2["rows"])
    prec = check_probe(chosen, probe_keys, production=prod)
    probe_tags(base_v2, probe_keys)
    plan = copy_plan(probe_keys)
    copy_rows, copy_recs = {}, {}
    for k, _i, fam in plan:
        mrow, crec = make_copy(base_v2["rows"][k], fam, out_dir() / "d5b")
        copy_rows[mrow["key"]], copy_recs[mrow["key"]] = mrow, crec
    # ---- the never-train check: dHash in both directions on stored hashes, then the DINOv2 detector
    ex = inputs["exams"]
    ev, se = {e: ex[e] for e in EXAMS_EVAL}, {e: ex[e] for e in EXAMS_SEASON}
    b_files = file_entries(base_b["rows"], inputs["hash_b"])
    v2_files = file_entries(base_v2["rows"], inputs["hash_v2"])
    tc_files = file_entries(inputs["train_core"]["rows"], inputs["hash_v2"])
    c_files = [{"key": k, "image": r["image"], "sha256": r["sha256"], "dhash": copy_recs[k]["dhash"],
                "variants": copy_recs[k]["variants"]} for k, r in sorted(copy_rows.items())]
    sc = inputs["scanners"]
    ld = F.root() / "leak"
    # DINOv2 against dev/test/ImageWeeds on the files no build scan judged: base_v2's train_core and NDSU (both
    # their base B and their base_v2 files) and the copies
    unjudged = {k for k, r in base_v2["rows"].items() if r["source"] not in TSW_SOURCES}
    emb_ev, scanned_ev = _scan_unique([f for f in b_files + v2_files if f["key"] in unjudged] + c_files, sc["eval"],
                                      ld / "eval_scan_desc.npz", procs)
    hits_b = _file_hits(b_files, dhash_check(b_files, ev), emb_ev, scanned_ev)
    hits_v2 = _file_hits(v2_files, dhash_check(v2_files, ev), emb_ev, scanned_ev)
    hits_c = _file_hits(c_files, dhash_check(c_files, ev), emb_ev, scanned_ev)
    # the season guard: every A' and B' file (both label forms) against ood22 and ood23, dHash and DINOv2
    bp, ap = set(lists["BPRIME"]), set(lists["APRIME"])
    s_files = [f for f in b_files if f["key"] in bp] + [f for f in v2_files if f["key"] in ap]
    emb_s, scanned_s = _scan_unique(s_files, sc["ood"], ld / "ood_scan_desc.npz", procs)
    s_hits = collections.defaultdict(list)
    for k, hs in _file_hits(s_files, dhash_check(s_files, se), emb_s, scanned_s).items():
        s_hits[k].extend(hs)
    # ---- judge (V3F.3.7 item 3)
    drop_b, kept_b = judge(base_b["rows"], hits_b, v2keys)
    drop_v2, kept_v2 = judge(base_v2["rows"], hits_v2, v2keys)
    drop_c, _kc = judge(copy_rows, hits_c, set())
    drop_s, _ks = judge(base_b["rows"], s_hits, v2keys, season=True)
    if drop_v2:
        raise DataRefused("base_v2 rows dropped against dev/test/ImageWeeds (%s): they are recorded and kept"
                          % sorted(drop_v2)[:3])
    kfm = KNOWN_FALSE_MATCH["key"]
    elig = set(base_b["rows"]) - set(drop_b) - rm
    # ---- the arms' final key lists
    pre = {"D4F": lists["F"], "D4N": lists["N"], "D4R15": lists["R15"], "D1B1": lists["BPRIME"],
           "D1A1": lists["APRIME"], "D1A12": lists["APRIME"]}
    keys = {"D4F": sorted(set(lists["F"]) & elig), "D4N": sorted(set(lists["N"]) & elig),
            "D4R15": sorted(set(lists["R15"]) & elig), "D1B1": sorted(set(lists["BPRIME"]) & elig - set(drop_s)),
            "D1A1": sorted(set(lists["APRIME"]) - rm - set(drop_s)),
            "D1A12": sorted(set(lists["APRIME"]) - rm - set(drop_s))}
    redrawn = len(keys["D4F"]) != len(lists["F"]) or len(keys["D4R15"]) != len(lists["R15"])
    if redrawn:
        keys["D4R15"] = r15_draw(sorted(elig), len(keys["D4F"]))
    t0_pre = sorted(v2keys - set(probe_keys))
    dose_copies = {}
    for arm, d in DOSES:
        pre[arm] = t0_pre
        keys[arm] = sorted(set(t0_pre) - rm)
        dose_copies[arm] = [c for c in (COPY_KEY % k for k in dose_keys(plan, d)) if c not in drop_c]
    if trig.get("bclean"):
        pre["D0BCLEAN"] = sorted(base_b["rows"])
        keys["D0BCLEAN"] = sorted(elig)
    # ---- every manifest's bytes (rows byte-identical to the parent's) and the earlier test lists, before any
    # file is written; then each is written once and inc2.train's checks run on the file
    pkeys, pshas = inputs["prior"]
    datas = {}
    for arm in [a for a in ARMS if a in keys]:
        parent = inputs[ARMS[arm]["parent"]]
        datas[arm] = filter_parent(parent, keys[arm], [(c, manifest_line(copy_rows[c]))
                                                       for c in dose_copies.get(arm, [])])
        check_prior(arm, datas[arm][0], pkeys, pshas)
    arms = {}
    for arm, (data, final) in datas.items():
        spec = ARMS[arm]
        parent = inputs[spec["parent"]]
        path = out_dir() / spec["file"]
        sha = write_once_bytes(path, data)
        rows = [r for _k, r, _l in parse_lines(data)]
        p = set(pre[arm])
        a = {"manifest": str(path), "file": spec["file"], "sha256": sha, "images": len(rows),
             "boxes": sum(len(C.read_yolo(r["label"])) for r in rows), "form": spec["form"],
             "parent": {"name": spec["parent"], "manifest": parent["path"], "sha256": parent["sha256"]},
             "keys_sha256": list_sha256(final), "keys_before_checks": {"rows": len(p), "sha256": list_sha256(p)},
             "removed": {"d0s": len(p & rm), "nevertrain": len(p & set(drop_b)),
                         "season": len(p & set(drop_s)) if arm in SEASON_ARMS else 0},
             "known_false_match": kfm in final,
             "recipe": _budget(len(rows)) if arm in BUDGET_ARMS else {"recipe_name": "cold", "epochs": 100}}
        if arm in dose_copies:
            listed = [COPY_KEY % k for k in dose_keys(plan, dict(DOSES)[arm])]
            a["copies"] = {"listed": len(listed), "kept": len(dose_copies[arm]),
                           "dropped": sorted(set(listed) & set(drop_c)), "expected_images": DOSE_ROWS[arm]}
        arms[arm] = a
    for a in arms.values():
        a["train_checks"] = train_checks(a["manifest"], production=prod)
    # ---- the probe root and the catch index
    proot = write_probe_root(base_v2, probe_keys)
    if prod and proot["boxes"] != PROBE_BOXES:
        raise DataRefused("the probe's GT boxes %s are not the amendment's %s" % (proot["boxes"], PROBE_BOXES))
    crec = None
    if catch:
        factory = sc.get("catch_factory") or ((lambda rows: _probe_scanner(rows, embedder)) if prod else None)
        cemb = embedder or (_sweep_embedder() if prod else None)
        crec = catch_index([base_v2["rows"][k] for k in probe_keys],
                           [dict(copy_recs[k], image=copy_rows[k]["image"]) for k in sorted(copy_rows)],
                           scanner_factory=factory, embedder=cemb, procs=procs, testing=testing)
    summary = {
        "format": FORMAT, "status": "complete", "testing": bool(testing), "decided_by": F.DECIDED_BY,
        "timestamp_sha256": ts, "code": dict(F.code_shas(), v3fam_data=C.sha256_file(Path(__file__).resolve()),
                                             v3fam_sweep=C.sha256_file(Path(SW.__file__).resolve())),
        "inputs": dict({k: {"path": inputs[k]["path"], "sha256": inputs[k]["sha256"]}
                        for k in ("base_b", "base_a1", "base_v2", "train_core")}, **(inputs.get("record") or {})),
        "setting_map": dict(lrec, rule="V3F.4: exact source; rf_tuf by family (cwd_weed_ Indoor, mh_ Field, DSC_ "
                                       "Field, in that order)"),
        "key_lists_before_checks": {n: {"rows": len(v), "sha256": list_sha256(v)} for n, v in lists.items()},
        "d0s": {"record_sha256": (inputs.get("record") or {}).get("sweep_record_sha256"), "removal_keys": rm_keys,
                "removal_sha256": rm_sha, "trigger": trig, "d6_staging_removals": sorted(rm & v2keys)},
        "nevertrain": {"bits": BITS, "exams": list(EXAMS_EVAL), "season_exams": list(EXAMS_SEASON),
                       "base_b_dropped": drop_b, "base_b_kept": kept_b, "base_v2_kept": kept_v2,
                       "copies_dropped": drop_c, "season_dropped": drop_s,
                       "embed_scanned": {"eval": len(scanned_ev), "season": len(scanned_s)},
                       "known_false_match": dict(KNOWN_FALSE_MATCH, kept=kfm in kept_b,
                                                 hits=(kept_b.get(kfm) or {}).get("hits"),
                                                 in_arms=sorted(a for a, v in arms.items() if v["known_false_match"]))},
        "r15": {"redrawn": redrawn, "rows": len(keys["D4R15"]), "f_rows": len(keys["D4F"])},
        "probe": dict(prec, root=proot),
        "copies": {"plan": [[k, i, f] for k, i, f in plan], "records": [copy_recs[k] for k in sorted(copy_recs)],
                   "dropped": sorted(drop_c)},
        "catch": None if crec is None else {"file": str(catch_dir() / "record.json"),
                                            "sha256": C.sha256_file(catch_dir() / "record.json"),
                                            "per_family": crec["per_family"]},
        "recorded_only": {"D8_train_core": _record_only(tc_files, ev, emb_ev, scanned_ev),
                          "D6_base_v2": _record_only(v2_files, ev, emb_ev, scanned_ev)},
        "preexisting": _preexisting_record(earlier, copy_rows),
        "arms": arms, "seconds": round(time.time() - t0, 1), "built_utc": F._utc()}
    summary = json.loads(json.dumps(summary, default=str))
    F.write_once(_summary_path(), summary)
    log("family data built: %s; %s" % (", ".join("%s %d" % (a, v["images"]) for a, v in arms.items()),
                                       _summary_path()))
    return summary


def _record_only(files, ev, emb_by_sha, scanned):
    """D8's train_core and D6's base_v2: the same check, every hit recorded and kept (all base_v2 rows)."""
    return {"rows": len(files), "hits": _file_hits(files, dhash_check(files, ev), emb_by_sha, scanned),
            "kept": True, "why": "base_v2 rows against dev, test and ImageWeeds are recorded and kept (V3F.3.7 item 3)"}


def _sweep_embedder():
    from ..funnel import embed as FE
    model, _, pooling = SW.EMBEDDER.rpartition(":")
    return FE.LazyEmbedder(model, pooling)


def _probe_scanner(rows_by_split, embedder=None):
    from ..funnel import embed as FE
    from . import embed_calibration as EC
    from . import guard as G
    cal, _why = EC.locked(C2.LOCK_PATH, production=True)
    model, _, pooling = str(cal["embedder"]).rpartition(":")
    emb = embedder or FE.LazyEmbedder(model, pooling)
    return G.EmbedScanner(G.eval_index(rows_by_split, emb, catch_dir() / "probe_desc.npz"), cal)


def read_summary():
    """splits/v3fam/summary.json when complete, else None."""
    d = F._read_json(_summary_path())
    return d if isinstance(d, dict) and d.get("format") == FORMAT and d.get("status") == "complete" else None


def arm_manifest(arm):
    """(path, sha256, images) of a built arm, from the complete summary (what inc2.baseline build --v3fam reads);
    refuses an arm the summary does not hold or a file that no longer hashes so."""
    s = read_summary()
    if s is None:
        raise DataRefused("no complete %s" % _summary_path())
    a = (s.get("arms") or {}).get(arm)
    if not isinstance(a, dict):
        raise DataRefused("the family summary holds no arm %s" % arm)
    if C.sha256_file(a["manifest"]) != a["sha256"]:
        raise DataRefused("%s no longer hashes as the family summary records" % a["manifest"])
    return a["manifest"], a["sha256"], a["images"]


# ------------------------------------------------------------------ lists (read-only)
def lists(base_b_path=None, base_v2_path=None, production=True):
    """The pre-guard key lists, A' and B', and the probe draw from the given manifests (default: the summary's
    and LOCK v2's), checked against the amendment's values; writes nothing (V3F.8 item 3's rehearsal)."""
    if base_b_path is None:
        summ = B3.read_summary()
        if summ is None:
            raise DataRefused("no complete splits/v3/summary.json")
        base_b_path = summ["arms"]["B"]["manifest"]
    if base_v2_path is None:
        base_v2_path = C2.v2_manifest_path("base_v2")
    bb, bv = read_parent(base_b_path, what="base B"), read_parent(base_v2_path, what="base_v2")
    ls, rec, _s = key_lists(bb, bv, REGISTERED if production else None, BASE_B_SHA256 if production else None,
                            production=production)
    chosen, pk = probe_draw(bv["rows"])
    prec = check_probe(chosen, pk, production=production)
    plan = copy_plan(pk)
    return {"base_b": {"path": bb["path"], "sha256": bb["sha256"], "rows": len(bb["rows"])},
            "base_v2": {"path": bv["path"], "sha256": bv["sha256"], "rows": len(bv["rows"])},
            "lists": {n: {"rows": len(v), "sha256": list_sha256(v)} for n, v in ls.items()}, "record": rec,
            "probe": prec, "doses": {a: {"copies": len(dose_keys(plan, d)), "images": len(bv["rows"]) - len(pk)
                                         + len(dose_keys(plan, d))} for a, d in DOSES},
            "copy_families": dict(collections.Counter(f for _k, _i, f in plan)),
            "known_false_match_in": {n: KNOWN_FALSE_MATCH["key"] in set(v) for n, v in ls.items()}}


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F data builders (L23H): D1, D4, D5b and BCLEAN's manifests.")
    ap.add_argument("command", choices=("build", "lists"))
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--base-b", default=None)
    ap.add_argument("--base-v2", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        if a.command == "build":
            build(procs=a.procs)
        else:
            print(json.dumps(lists(a.base_b, a.base_v2), indent=1, sort_keys=True))
    except (F.V3famRefused, C2.Inc2Error, B3.Base3Error, ValueError) as e:
        print("[inc2.v3fam_data] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
