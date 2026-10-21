"""V3F: the v3 experiment family's shared core (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered
as one family)", kept byte for byte as docs/prereg/V3F_2026-10-11.md).

    python -m weed_optimizer_framework.tools.inc2.v3fam equivalence --item dev_identity
    python -m weed_optimizer_framework.tools.inc2.v3fam equivalence --item two_stage --e3-arm M|A|B
    python -m weed_optimizer_framework.tools.inc2.v3fam equivalence --item identity --exam E --arm X
    python -m weed_optimizer_framework.tools.inc2.v3fam equivalence --item injection
    python -m weed_optimizer_framework.tools.inc2.v3fam exif-preflight
    python -m weed_optimizer_framework.tools.inc2.v3fam exif-verdict --frame yes|no --read-by owner|delegate
    python -m weed_optimizer_framework.tools.inc2.v3fam frozen
    python -m weed_optimizer_framework.tools.inc2.v3fam score --arm X          (L23J; SCORE_ARMS, the arm's module)
    python -m weed_optimizer_framework.tools.inc2.v3fam verdict [--cap-refused ITEM,...] [--ticker-failed ITEM,...]
                                                                                    (L23L; inc2/v3fam_verdict.py)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-read --arm X --read-by owner|delegate   (a person's)
    python -m weed_optimizer_framework.tools.inc2.v3fam score-test --arm X     (the job test-read prepares)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-report --arm X
    python -m weed_optimizer_framework.tools.inc2.v3fam reanalyse-test --read-by owner|delegate     (a person's job)
    python -m weed_optimizer_framework.tools.inc2.v3fam spend                  (the family's GPU-h, read-only)
    python -m weed_optimizer_framework.tools.inc2.v3fam _pass --spec SPEC.json      (internal: another root)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

What this module is. Every score of every comparison of the family (the
comparators E3-M, E3-A, E3-B and b_v2_m640 included) is taken by one locked
scorer call with the family validator installed (_family_pass), and every
decision is computed by the statistics below. The arm modules
(v3fam_data, v3fam_embed, v3fam_wbf, v3fam_detr, v3fam_sweep) and the per-arm
verbs build on it. The E3-frozen files (FROZEN_SHA256: inc2/twostage.py,
inc/scorer.py, inc2/scorer_sidecar.py, inc2/scorer_native.py, inc/verify.py,
semisup_labeler.py) are imported as libraries and never edited; every
family verb refuses once one of them hashes otherwise (check_frozen).

The family validator (family_validator_class) is a subclass of E3's own
validator class (twostage.e3_validator_class(cfg)[0], itself built on the
scorer sidecar's capturing validator, built on the locked scorer's). Its
modes (MODES):
  identity      E3's: the detector's predictions after NMS emitted unchanged
  two_stage     E3's: stage-1 boxes cut, embedded, classified, top 3 at q x p
  gt            E3's: the classifier on the ground-truth crops (accuracy)
  stage1        a one-class detector's box set: its predictions emitted
                unchanged, its stage-1 rows (original pixels) and its
                class-collapsed matches recorded, with the target-species
                agnostic AP with OtherPlant boxes as ignore regions (P7) and
                its symmetric OtherPlant reading
  inject_boxes  D3: a given box set per image (original pixels, score q)
                replaces stage 1; then E3's naming and emission
  inject_rows   D6: given rows (original pixels, conf, class) replace the
                predictions after NMS; emitted as identity emits
  oracle        O5: a stage-1 box with IoU >= 0.5 to a GT box is emitted once
                with that GT class at q; every other box takes E3's top 3
  view640       D2's BioCLIP-2@640: E3's two-stage pass with the crops cut by
                the arm's cut function (cfg["cut"], _e3_cut's contract)
Emission, the limits (conf 0.001, max_det 300), the geometry (the per-axis
inverse letterbox, twostage.to_original), the ground-truth round trip
(1e-3) and the size check are E3's. Every pass also reads each image's EXIF
orientation itself: none, 0 and 1 (twostage.OK_ORIENTATIONS) on every exam,
and 3, 6 and 8 (twostage.FIT_ORIENTATIONS) on ood22 and ood23 only once the
frame pre-flight has passed (exif_preflight_passed). E3's crop check refuses
3, 6 and 8 because its size check cannot see a 180-degree turn; on ood22 and
ood23 after the pre-flight the family's frame adapter (_FrameWorkers) hands
that check a tag it accepts and the family records the true tag
(stats.orientations; E3's own counter is kept as stats.e3_orientations).

Exams and who may read them (EXAM_READERS, V3F.3.1): dev, imageweeds, ood22,
ood23 under the main INC_DIR; ooddev_v1 (the zoo's root) and probe (D5b's
root) only in a subprocess with INC_DIR at their root (family_pass_in_root,
verb _pass); test only with a token score-test holds (test_read_token). A
pass under another arm, exam or root refuses before anything is scored.

The reproductions (V3F.3.2, decision 17), each written once under
v3fam/v1/equivalence/ with the code that ran (code_shas), and required by
every family score (require_equivalence):
  dev_identity.json      b_v2_m640 s0-s2 identity = their recorded protocol dev scores
  two_stage_<M|A|B>.json E3-M/A/B's stage-1 weights + E3's pinned classifier = E3's
                         recorded dev files; each stage-1 box set = the stored
                         dev.stage1.npz (rows, xyxy 1e-3 px, q 1e-4) or refused
  identity_<exam>.json   b_v2_m640 s0 identity = the plain locked score in the same job
  injection.json         b_v2_m640 s0's own dev rows, taken to original pixels and
                         injected, = its recorded protocol dev score
Within REPRO_TOL (1e-6) a reproduction passes; within DEVIATION_TOL (0.002)
it is recorded as a deviation and the job continues (both sides of every
comparison still share the path); beyond it refuses, writing nothing.

The statistics (V3F.3.3, V3F.3.4), on each run's tie-broken per-image arrays
(scorer_sidecar.tie_break, species_arrays, class_ap, resample_indices):
  statistic "species"   the mean AP50-95 over the species with a GT box in the
                        resample (as inc2.baseline.native_bootstrap; a subset by
                        species=); "agnostic": inc.scorer.collapsed_ap on arrays
                        {keys, tp, conf, pred_img, n_gt} (P7's ignore-region arrays,
                        an agnostic reading, as inc2.baseline.agnostic_bootstrap)
  image_bootstrap       1,000 resamples of the exam's images, one draw for every
                        run, seed text inc2/v3fam/<cid>/image; reproduces
                        native_bootstrap's SE exactly and returns every resample
  session_bootstrap     1,000 resamples of the sessions (image multiplicity = its
                        session's), seed text inc2/v3fam/<cid>/session; the
                        key -> session rule of V3F.3.1 (session_of)
  selection_correct     P2, P3, P5: E3's choice (M, A, B; ties M then A then B)
                        redone in each image resample; D_sel, SE_sel, share of B
  decide                the E-rule (D > 2 x pooled sd and D > SE; P4's one system:
                        D > 2 sd(E3-B), seed term 2 sd(E3-B)^2 / 3), combined z,
                        the 90 % interval and the outcome classes with SESOI 0.005
  comparison            all of the above for one pre-declared comparison id
  holm_overlay          Holm per exam (HOLM_SETS; not-run primaries keep their place)
  challenger_gate       C = 14 registered challengers, bar Phi^-1(1 - 0.05/14)
  read_order/check_read the sealed-test read order and cap (<= 5, challengers <= 3,
                        2 once BCLEAN takes a read), read_by values
  family_gpuh/check_cap the 260 GPU-h family cap (running items at their estimate)

Records written once (write_once, commit_pass): v3fam/v1/equivalence/*,
v3fam/v1/exif/{preflight.json, montage.jpg, verdict.json},
v3fam/v1/cap/<item>.json (not run (family cap)), and each arm-seed's
v3fam/v1/arms/<X>/s<k>/<exam>.{json, images.npz, stage1.npz, rows.npz,
p7.npz}. Every family job records the sha256 tree of twostage/e3_v1 (but
sensitivity.json) and of capacity/e{1,2,3}_* before and after it runs and
fails if either changed (e_trees_unchanged).

Public API (for the arm modules and verbs):
  constants   SEEDS, RESAMPLES, SESOI, ALPHA, Z90, MODES, EXAMS, EXAM_READERS,
              SCORE_ARMS, BUILD_ARMS, PRIMARIES, HOLM_SETS, SELECTION_PRIMARIES,
              SESSION_PRIMARIES, ONE_SYSTEM_PRIMARIES, CHALLENGERS, C_CHALLENGERS,
              CHALLENGER_BAR, READ_CAP, READ_BY, DECIDED_BY_ENV, FAMILY_CAP_GPUH,
              FROZEN_SHA256, DECIDED_BY, PREREG_SHA256, PROBE_SPECIES, REPRO_TOL,
              DEVIATION_TOL, STAGE1_XYXY_TOL, STAGE1_Q_TOL
  passes      _family_pass(weights, exam, cfg, arm, ...) -> {result, arrays, stage1,
              rows, p7, stage1_agnostic, stats, gt, seconds, validator};
              family_pass_in_root(root, weights, exam, cfg, arm, ...); root_of(exam);
              test_read_token(); inject_from_rows(rows); stage1_compare(new, stored)
  records     family_doc(...), commit_pass(dir, exam, doc, ps, variant=""),
              write_once(path, data), arm_dir(arm, seed), code_shas(), check_frozen(),
              e_trees(), e_trees_unchanged()
  equivalence equivalence_dev_identity, equivalence_two_stage, equivalence_identity,
              equivalence_injection, require_equivalence(names)
  exif        exif_frame_check(path), exif_preflight(rows=None), exif_verdict(frame,
              read_by), exif_preflight_passed()
  statistics  session_of(exam, key, field), sessions_for(exam, rows), image_counts,
              session_counts, run_values, image_bootstrap, session_bootstrap,
              selection_correct, decide, outcome, comparison, statistic_value,
              subset_arrays, subset_mean, ignore_region_matches, p7_value,
              holm_overlay, challenger_gate, read_order, check_read, read_by_value,
              dev_session_stratum, family_gpuh, check_cap

A refusal raises V3famRefused (V3famCapRefused for the family cap); the CLI
prints "[inc2.v3fam] ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import common as C2
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import twostage as TS

# ------------------------------------------------------------------ constants (pre-registered)
NAME = "v1"
PREREG_FILE = "docs/prereg/V3F_2026-10-11.md"
PREREG_SHA256 = "5826660c92a875bcccf12fd8e7d221f49209653aa4d2cba34388170f05b52f7f"
PREREG_COMMIT = "55cc2d97cd2147d01644c1e193098b73aeba0a06"
# V3F.3.9: the Revisions made before any family number (Revisions 29-31, 2026-10-11), each section kept byte for byte
# and time-stamped (V3F.12) before the next family step; v3fam/v1/timestamp.json names their stamps
REVISION_FILE = "docs/prereg/V3F_rev29-31_2026-10-11.md"
REVISION_SHA256 = "36f4637898156ace639112f4605fbfe1192ecacad67b81ed47b1a6465bc0f814"
DECIDED_BY = ("docs/CONTINUOUS_LOOP.md, Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered "
              "as one family); commit 55cc2d9; %s sha256 %s; Revisions 29-31: %s sha256 %s"
              % (PREREG_FILE, PREREG_SHA256, REVISION_FILE, REVISION_SHA256))
# V3F.1: the E3-frozen files, as twostage/e3_v1/equivalence.json -> code records them
FROZEN_SHA256 = {
    "twostage": "8fe2edc0e0d2a00d07182c3746407281a112c9e50c76662bc9f48a558b328ddc",
    "scorer": "18c008371cd2c021104d034d06b837564016f4509e8a969c2a3f850b09dd2de4",
    "scorer_sidecar": "de7aacda903780b32944866be0b4daa5c6b1f76e2c86965f7a99d85f443d5e52",
    "scorer_native": "ba9eeb928a2b5a0756ae8ac0144b54df2fdfb3570fd011f66b4a8022039d6b7d",
    "verify": "602eba3c935f375179d6f922dce28bbf89006054e5bcec5894f0c9cc7b87aa10",
    "semisup_labeler": "03b28c50699b30a32720e383ff2b713a2638926650007acb92a367d1f450458c"}

SEEDS = (0, 1, 2)
RESAMPLES = 1000
SESOI = 0.005                                # mAP50-95, one number for every exam (decision 4)
ALPHA = 0.05                                 # one-sided
Z90 = statistics.NormalDist().inv_cdf(0.95)  # the 90 % interval's half-width in sigma (the text's 1.645, unrounded)
REFERENCE = TS.REFERENCE                     # b_v2_m640
IMGSZ = TS.IMGSZ
TOP_K = TS.TOP_K
TIE_TOL = TS.TIE_TOL

MODES = ("identity", "two_stage", "gt", "stage1", "inject_boxes", "inject_rows", "oracle", "view640")
E3_MODE = {"identity": "identity", "two_stage": "two_stage", "gt": "gt", "stage1": "identity",
           "inject_rows": "identity", "inject_boxes": "two_stage", "oracle": "two_stage", "view640": "two_stage"}
CLASSIFY_MODES = tuple(m for m in MODES if E3_MODE[m] in ("two_stage", "gt"))
INJECT_MODES = ("inject_boxes", "inject_rows")
ROWS_MODES = ("identity", "stage1", "inject_rows")         # the emitted rows recorded in original pixels
ROOT_MODES = ("identity", "stage1", "inject_rows")         # what a pass in another root may take (no embedder)

# ------------------------------------------------------------------ arms and exams (V3F.3.1, V3F.6.3)
SCORE_ARMS = ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
              "D3", "D9", "D1", "D4F", "D4N", "D4R15", "D5B", "D6", "D8", "BCLEAN")
BUILD_ARMS = ("D0BCLEAN", "D8M", "D8S", "D4F", "D4N", "D4R15", "D1A1", "D1B1", "D1A12", "D5B0", "D5B25", "D5B100")
EXAMS = ("dev", "imageweeds", "ood22", "ood23", "ooddev_v1", "probe", "test")
ROOT_EXAMS = ("ooddev_v1", "probe")          # scored only in their own root (a subprocess with INC_DIR there)
OOD_EXAMS = ("ood22", "ood23")               # the only exams that may read EXIF tags 3, 6 and 8
EXAM_READERS = {
    "dev": ("E3RS", "D1", "D2", "D3", "D4", "D6", "D8", "D9", "BCLEAN"),
    "imageweeds": ("E3RS", "D1", "D3", "D4", "D6", "D8", "BCLEAN"),
    "ood22": ("D1",),
    "ood23": ("D1",),
    "ooddev_v1": ("E3RS", "D1", "D3", "D4", "D6", "D8"),
    "probe": ("D5B",),
    "test": ()}                              # test: only with score-test's token (test_read_token)
ZOO_VERSION = "v1"                           # ooddev_v1 is Z1's exam, in Z1's root
PROBE_SPECIES = ("Waterhemp", "MorningGlory", "SpottedSpurge", "Carpetweed", "Ragweed", "Eclipta", "PalmerAmaranth")

# ------------------------------------------------------------------ the decision (V3F.3.3, V3F.3.4)
PRIMARIES = {
    "P1": {"exam": "dev", "x": "E3-B boxes + BioCLIP-2", "y": "E3-B boxes + LAION-2B CLIP ViT-L/14", "se": "image"},
    "P2": {"exam": "dev", "x": "E3-B", "y": "TS-F", "se": "selection"},
    "P3": {"exam": "dev", "x": "E3-B", "y": "TS-N", "se": "selection"},
    "P4": {"exam": "dev", "x": "WBF of E3-B's three stage-1 seeds", "y": "E3-B", "se": "image (one system)"},
    "P5": {"exam": "dev", "x": "E3-B", "y": "RF-DETR-L 12-class", "se": "selection"},
    "P6": {"exam": "ood22", "x": "TS-B'", "y": "A'12", "se": "max(image, session)"},
    "P7": {"exam": "ood23", "x": "B'1", "y": "A'1", "se": "max(image, session)",
           "statistic": "target-species agnostic AP, OtherPlant boxes as ignore regions"},
    "P8": {"exam": "probe", "x": "dose 1.0", "y": "dose 0", "se": "max(image, session)"},
    "P9": {"exam": "probe", "x": "dose 0.25", "y": "dose 0", "se": "max(image, session)"}}
HOLM_SETS = {"dev": ("P1", "P2", "P3", "P4", "P5"), "ood22": ("P6",), "ood23": ("P7",), "probe": ("P8", "P9")}
SELECTION_PRIMARIES = ("P2", "P3", "P5")
SESSION_PRIMARIES = ("P6", "P7", "P8", "P9")
ONE_SYSTEM_PRIMARIES = ("P4",)
SELECTION_ARMS = TS.ARM_ORDER                # M, A, B: E3's tie order
SELECTED = "B"                               # E3's choice, the X of P2, P3 and P5
# Tier C: fourteen registered challengers, counted whatever is run (R13; decisions 3 and 24)
CHALLENGERS = ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
               "D3", "D4F", "D4N", "D4R15", "D6")
C_CHALLENGERS = 14
CHALLENGER_BAR = statistics.NormalDist().inv_cdf(1.0 - ALPHA / C_CHALLENGERS)     # 2.690110
# Sealed-test reads (V3F.3.6)
READ_CAP = 5
CHALLENGER_READS = 3
CHALLENGER_READS_BCLEAN = 2
REFERENCE_READS = ("D6", "D8M")
BCLEAN_READ = "BCLEAN"
READ_BY = {"owner": "Harry He (owner)", "delegate": "Claude (delegate under the 2026-09-30 owner grant)"}
DECIDED_BY_ENV = {"delegate": "assistant:claude-delegate"}    # the owner's: the owner's account, as recorded
# The family cap (decision 25)
FAMILY_CAP_GPUH = 260.0
CAP_EPS = 1e-6                               # GPU-h: sums of decimal estimates compared without float noise
NOT_RUN_CAP = "not run (family cap)"

# ------------------------------------------------------------------ reproductions and checks (V3F.3.2)
REPRO_TOL = 1e-6
DEVIATION_TOL = SC.MAX_SCORE_DIFF            # 0.002
STAGE1_XYXY_TOL = 1e-3                       # px
STAGE1_Q_TOL = 1e-4
EQUIV_ITEMS = ("dev_identity", "two_stage", "identity", "injection")
EXIF_TAGS = (3, 6, 8, 0)                     # one base_v2 copy each
EXIF_MAD_TOL = 2.0                           # grey levels, mean absolute difference
EXIF_SEED_TEXT = "inc2/v3fam/exif/%s"
IOU_IGNORE = 0.5                             # P7: a non-TP row this close to an OtherPlant box is ignored
ORACLE_IOU = 0.5
STAMP = "V3F-"
SCORE_STAMPS = ("manifest_sha256", "key_order_sha256", "n_images", "locked_scorer_sha256", "ultralytics_version")
COMPARE_STAMPS = ("exam", "manifest_sha256", "key_order_sha256", "n_images", "weights_sha256", "n_gt")

FMT_EQUIV = "inc2-v3fam-equivalence/1"
FMT_SCORE = "inc2-v3fam-score/1"
FMT_EXIF = "inc2-v3fam-exif-preflight/1"
FMT_EXIF_VERDICT = "inc2-v3fam-exif-verdict/1"
FMT_CAP = "inc2-v3fam-cap/1"
FMT_PASS_SPEC = "inc2-v3fam-pass-spec/1"

_TEST_TOKENS = []                            # score-test's tokens: the only callers that may score test
_IN_ROOT = {}                                # set by the _pass verb in another root's subprocess


class V3famRefused(RuntimeError):
    """What was asked is not the family's pre-registered measurement."""


class V3famCapRefused(V3famRefused):
    """The item would take the family past its GPU-h cap: not run (family cap)."""


def log(msg):
    print("[inc2.v3fam] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths, files, code
def root():
    return C.INC_DIR / "v3fam" / NAME


def equiv_dir():
    return root() / "equivalence"


def exif_dir():
    return root() / "exif"


def cap_dir():
    return root() / "cap"


def arm_dir(arm, seed):
    return root() / "arms" / arm / ("s%d" % int(seed))


def root_of(exam, inc_dir=None):
    """The root an exam is scored in: the zoo's (ooddev_v1), the probe's (probe), else INC_DIR."""
    base = Path(inc_dir) if inc_dir else C.INC_DIR
    if exam == "ooddev_v1":
        return base / "_zoo" / ZOO_VERSION / "root"
    if exam == "probe":
        return base / "v3fam" / NAME / "probe_root"
    return base


def file_names(exam, variant=""):
    """(json, images npz, stage-1 npz, rows npz, p7 npz) names of a family score."""
    stem = "%s%s" % (exam, ".%s" % variant if variant else "")
    return tuple("%s.%s" % (stem, x) for x in ("json", "images.npz", "stage1.npz", "rows.npz", "p7.npz"))


def _utc():
    return TS._utc()


def _read_json(path):
    return TS._read_json(path)


def _sha(path):
    return TS._sha(path)


def write_once(path, data):
    """Write bytes or a JSON object to path only if it does not exist (twostage._write_once)."""
    try:
        return TS._write_once(path, data)
    except TS.E3Refused as e:
        raise V3famRefused(str(e))


def code_shas():
    """The six E3-frozen files' sha256s and this module's: the code a family pass runs."""
    out = TS._code_shas()
    out["v3fam"] = C.sha256_file(Path(__file__).resolve())
    return out


def check_frozen():
    """Refuse unless the six E3-frozen files hash as V3F.1 records."""
    now = TS._code_shas()
    bad = sorted(k for k in FROZEN_SHA256 if now.get(k) != FROZEN_SHA256[k])
    if bad or set(now) != set(FROZEN_SHA256):
        raise V3famRefused("the E3-frozen file(s) %s no longer hash as V3F.1 records: no family verb runs"
                           % (bad or sorted(set(now) ^ set(FROZEN_SHA256))))
    return now


def e_trees():
    """{relative path: sha256} of twostage/e3_v1 (but sensitivity.json, D9a's one permitted write) and of
    capacity/e1_*, e2_*, e3_* (V3F.3.7, write isolation)."""
    out = {}
    tw = C.INC_DIR / "twostage" / TS.E3_NAME
    if tw.is_dir():
        for p in sorted(tw.rglob("*")):
            if p.is_file() and not p.is_symlink() and p != tw / "sensitivity.json":
                out[str(p.relative_to(C.INC_DIR))] = C.sha256_file(p)
    cap = C.INC_DIR / "capacity"
    if cap.is_dir():
        for p in sorted(cap.iterdir()):
            if p.is_file() and re.match(r"e[123]_", p.name):
                out[str(p.relative_to(C.INC_DIR))] = C.sha256_file(p)
    return out


@contextlib.contextmanager
def e_trees_unchanged():
    """The E-series trees before and after the block: a family job fails if either changed."""
    before = e_trees()
    yield before
    after = e_trees()
    if after != before:
        diff = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise V3famRefused("the E-series records changed while this family job ran (%s): the family writes only "
                           "under v3fam" % diff[:5])


# ------------------------------------------------------------------ geometry (E3's, plus its exact inverse)
def to_letterbox(xyxy, ratio_pad):
    """Original-pixel xyxy -> the validator's letterboxed frame (the forward map twostage.to_original inverts,
    per axis, unclipped). float64 [n, 4]."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    (gh, gw), (px, py) = ratio_pad
    gh, gw, px, py = float(gh), float(gw), float(px), float(py)
    out = np.empty_like(a)
    out[:, [0, 2]] = a[:, [0, 2]] * gw + px
    out[:, [1, 3]] = a[:, [1, 3]] * gh + py
    return out


def from_letterbox(xyxy, ratio_pad):
    """Letterboxed xyxy -> original pixels, unclipped (twostage.to_original without the clip): the exact inverse
    of to_letterbox, so a captured row re-injected lands on its own coordinates."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    (gh, gw), (px, py) = ratio_pad
    gh, gw, px, py = float(gh), float(gw), float(px), float(py)
    if gh <= 0 or gw <= 0:
        raise RuntimeError("a non-positive letterbox gain %r" % (ratio_pad,))
    out = np.empty_like(a)
    out[:, [0, 2]] = (a[:, [0, 2]] - px) / gw
    out[:, [1, 3]] = (a[:, [1, 3]] - py) / gh
    return out


def _ratio_pad(pb):
    rp = pb["ratio_pad"]
    try:
        (gh, gw), (px, py) = rp
        return (float(gh), float(gw)), (float(px), float(py))
    except (TypeError, ValueError) as e:
        raise RuntimeError("ratio_pad %r is not ((gain_h, gain_w), (pad_x, pad_y)) (%s)" % (rp, e))


# ------------------------------------------------------------------ P7: ignore regions (V3F.3.2)
def ignore_region_matches(gt_boxes, gt_cls, boxes, match):
    """The target-species agnostic reading of one image's stage-1 boxes, with OtherPlant boxes as ignore regions
    (P7, decision 7), and its symmetric OtherPlant reading. gt_boxes [M, 4], gt_cls [M] and boxes [N, 4] are
    torch tensors in one frame; match(iou [m, N]) -> bool [N, 10] is Ultralytics' match_predictions class-collapsed
    (as twostage's stage-1 helper calls it). A box that is not a true positive at IoU 0.50 and overlaps a box of
    the other kind at IoU >= 0.5 is ignored at every threshold. Returns numpy {tp_t, ign_t, n_t, tp_o, ign_o, n_o}."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import box_iou
    n = int(len(boxes))
    cls = np.rint(gt_cls.float().cpu().numpy().reshape(-1)).astype(np.int64) if len(gt_cls) else np.zeros(0, np.int64)
    is_t = torch.as_tensor(cls != C.OTHER_PLANT, device=gt_boxes.device)
    out = {"n_t": int((cls != C.OTHER_PLANT).sum()), "n_o": int((cls == C.OTHER_PLANT).sum())}

    def read(gsel, osel, tag):
        tp = np.zeros((n, SC.NIOU), dtype=bool)
        near = np.zeros(n, dtype=bool)
        if n and bool(gsel.any()):
            tp = np.asarray(match(box_iou(gt_boxes[gsel], boxes)), dtype=bool).reshape(n, -1)
        if n and bool(osel.any()):
            near = (box_iou(gt_boxes[osel], boxes).max(0).values >= IOU_IGNORE).cpu().numpy().reshape(-1)
        out["tp_" + tag] = tp
        out["ign_" + tag] = ~tp[:, 0] & near if n else np.zeros(0, dtype=bool)
    read(is_t, ~is_t, "t")
    read(~is_t, is_t, "o")
    return out


def default_match():
    """Ultralytics' match_predictions at its ten IoU thresholds, class-collapsed, outside a validator (tests)."""
    import types
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    ns = types.SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))

    def match(iou):
        n = iou.shape[1]
        return DetectionValidator.match_predictions(ns, torch.zeros(n), torch.zeros(iou.shape[0]), iou).cpu().numpy()
    return match


def p7_value(tp, conf, ign, n_gt):
    """(AP50-95, AP50) of the rows not ignored against n_gt boxes (inc.scorer.collapsed_ap)."""
    import numpy as np
    keep = ~np.asarray(ign, dtype=bool).reshape(-1)
    return S.collapsed_ap(np.asarray(tp, dtype=bool)[keep], np.asarray(conf, dtype=np.float64).reshape(-1)[keep],
                          int(n_gt))


# ------------------------------------------------------------------ the frame adapter (EXIF tags 3, 6, 8)
class _FrameWorkers(object):
    """The workers E3's crop loop maps over (verify._Workers' imap). For view640 the cut function is the arm's
    (cfg["cut"]) instead of twostage._e3_cut. On ood22 and ood23 after the frame pre-flight an image tagged 3, 6
    or 8 (cut in the exif_transpose frame, which the pre-flight showed Ultralytics' loader reads too) is handed to
    E3's orientation check as untagged; the family's own header read records its true tag."""

    def __init__(self, inner, cut=None, extra_tags=()):
        self.inner, self.cut, self.extra = inner, cut, tuple(extra_tags)
        self.passed = collections.Counter()

    def imap(self, fn, items, chunksize=8):
        f = self.cut if (self.cut is not None and fn is TS._e3_cut) else fn
        for res in self.inner.imap(f, items, chunksize):
            image, ids, arrays, err, hd = res
            if hd is not None and hd[0] in self.extra:
                self.passed[str(hd[0])] += 1
                hd = (None, hd[1])
            yield image, ids, arrays, err, hd


# ------------------------------------------------------------------ the family validator
def family_validator_class(cfg):
    """(family validator class, the scorer's own validator). cfg: mode (MODES), nms locked | agnostic, and as
    E3's: W, b, classes, prior, embedder, workers (two-stage modes), top_k, max_det; inject {key: {xyxy, conf,
    cls}} (inject modes; original pixels); cut (view640: a picklable _e3_cut-like function); orientations (the
    tags this pass reads; default twostage.OK_ORIENTATIONS). Built on twostage.e3_validator_class(cfg)[0]."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import box_iou

    mode, nms = cfg.get("mode"), cfg.get("nms", "locked")
    if mode not in MODES or nms not in ("locked", "agnostic"):
        raise V3famRefused("family validator mode %r nms %r (modes %s)" % (mode, nms, list(MODES)))
    allowed = tuple(cfg.get("orientations") or TS.OK_ORIENTATIONS)
    if any(t not in TS.FIT_ORIENTATIONS for t in allowed):
        raise V3famRefused("orientations %s: a pass reads at most %s" % (allowed, TS.FIT_ORIENTATIONS))
    extra = tuple(t for t in allowed if t not in TS.OK_ORIENTATIONS)
    e3cfg = dict(cfg, mode=E3_MODE[mode], nms=nms)
    frame = None
    if mode in CLASSIFY_MODES:
        if cfg.get("workers") is None:
            raise V3famRefused("a %s pass needs the crop workers" % mode)
        if mode == "view640" and not callable(cfg.get("cut")):
            raise V3famRefused("a view640 pass needs the arm's cut function (cfg['cut'])")
        frame = _FrameWorkers(cfg["workers"], cut=cfg.get("cut") if mode == "view640" else None, extra_tags=extra)
        e3cfg["workers"] = frame
    inject = cfg.get("inject") if mode in INJECT_MODES else None
    if mode in INJECT_MODES and not isinstance(inject, dict):
        raise V3famRefused("an %s pass needs inject {key: rows}" % mode)
    E3V, base = TS.e3_validator_class(e3cfg)

    class FamilyValidator(E3V):
        _v3fam_layer = True

        def inc_reset(self):
            super().inc_reset()
            self.f_tags = collections.Counter()
            self.f_seen = []
            self.f_rows, self.f_s1, self.f_p7 = {}, {}, {}
            self.f_p7_order = []
            self.f_stats = collections.Counter()
            self.f_frame = frame

        # -------------------------------------------------------- per image, before E3's stage-1 helper
        def e3_stage1(self, pred, pb):
            key = Path(pb["im_file"]).stem
            self.f_header(pb, key)
            if mode in INJECT_MODES:
                pred = self.f_injected(pred, pb, key)
            it = super().e3_stage1(pred, pb)
            self.f_seen.append(key)
            if mode in ("stage1", "inject_rows"):
                h0, w0 = int(pb["ori_shape"][0]), int(pb["ori_shape"][1])
                self.e3_gt_check(pb, key, w0, h0)
            if mode in ROWS_MODES:
                p = it["pred"]
                self.f_rows[key] = {"xyxy": from_letterbox(p["bboxes"].float().cpu().numpy(), _ratio_pad(pb)),
                                    "conf": p["conf"].float().cpu().numpy().reshape(-1).astype(np.float32),
                                    "cls": p["cls"].float().cpu().numpy().reshape(-1).astype(np.float32)}
            if mode == "stage1":
                n = len(it["q"])
                self.f_s1[key] = {"xyxy": TS.to_original(it["s1_boxes"].float().cpu().numpy(), pb["ratio_pad"],
                                                         pb["ori_shape"]) if n else np.zeros((0, 4)),
                                  "q": np.asarray(it["q"], dtype=np.float64)}
                zeros = torch.zeros(n, device=it["s1_boxes"].device)
                r = ignore_region_matches(
                    pb["bboxes"], pb["cls"], it["s1_boxes"],
                    lambda iou: self.match_predictions(zeros, torch.zeros(iou.shape[0], device=iou.device),
                                                       iou).cpu().numpy())
                r["q"] = np.asarray(it["q"], dtype=np.float64)
                self.f_p7[key] = r
                self.f_p7_order.append(key)
                self.f_stats["ignored_target_rows"] += int(r["ign_t"].sum())
                self.f_stats["ignored_otherplant_rows"] += int(r["ign_o"].sum())
            return it

        def f_header(self, pb, key):
            try:
                ori, (w, h) = TS.header(pb["im_file"])
            except Exception as e:             # noqa: BLE001 - every exam image was hashed: a defect, refused
                raise V3famRefused("cannot read the header of %s: %s" % (pb["im_file"], e))
            self.f_tags[str(ori)] += 1
            if ori not in allowed:
                raise V3famRefused("%s has EXIF orientation %r: this pass reads %s (tags 3, 6 and 8 only on ood22 "
                                   "and ood23 after the frame pre-flight)" % (key, ori, list(allowed)))
            h0, w0 = int(pb["ori_shape"][0]), int(pb["ori_shape"][1])
            if (w, h) != (w0, h0):
                raise V3famRefused("%s is %dx%d after EXIF transposition, Ultralytics read it as %dx%d (ori_shape)"
                                   % (key, w, h, w0, h0))

        def f_injected(self, pred, pb, key):
            rows = inject.get(key)
            if rows is None:
                raise V3famRefused("no injected rows for exam image %s" % key)
            xy = np.asarray(rows.get("xyxy"), dtype=np.float64).reshape(-1, 4)
            conf = np.asarray(rows.get("conf"), dtype=np.float32).reshape(-1)
            if len(conf) != len(xy) or not (np.isfinite(xy).all() and np.isfinite(conf).all()):
                raise V3famRefused("the injected rows of %s are not n boxes with n finite scores" % key)
            if mode == "inject_rows":
                cls = np.asarray(rows.get("cls"), dtype=np.float64).reshape(-1)
                if len(cls) != len(xy) or not np.all((cls == np.rint(cls)) & (cls >= 0) & (cls < C.NC)):
                    raise V3famRefused("the injected rows of %s carry classes outside 0..%d" % (key, C.NC - 1))
            else:
                cls = np.zeros(len(xy), dtype=np.float64)
            b0 = pred["bboxes"]
            lb = to_letterbox(xy, _ratio_pad(pb))
            self.f_stats["injected_rows"] += int(len(xy))
            return {"bboxes": torch.as_tensor(lb, device=b0.device).to(b0.dtype).reshape(-1, 4),
                    "conf": torch.as_tensor(conf, dtype=torch.float32, device=b0.device),
                    "cls": torch.as_tensor(cls, device=b0.device).to(pred["cls"].dtype),
                    "extra": b0.new_zeros((len(xy), 0))}

        # -------------------------------------------------------- after E3's naming
        def e3_classify(self, items):
            super().e3_classify(items)
            if mode == "oracle":
                self.f_oracle(items)

        def f_oracle(self, items):
            """O5: a stage-1 box with IoU >= 0.5 to a GT box (the most overlapping; ties to the lower GT index)
            takes that GT class with p 1, so E3's emission emits it once at q."""
            for it in items:
                pb, n = it["pb"], len(it["q"])
                gt = np.full(n, -1, dtype=np.int64)
                if n and len(pb["cls"]):
                    iou = box_iou(pb["bboxes"], it["s1_boxes"]).float().cpu().numpy()      # [M, n]
                    best, arg = iou.max(axis=0), iou.argmax(axis=0)
                    gc = np.rint(pb["cls"].float().cpu().numpy().reshape(-1)).astype(np.int64)
                    hit = best >= ORACLE_IOU
                    gt[hit] = gc[arg[hit]]
                    P = it["P"]
                    for j in np.flatnonzero(hit):
                        P[j] = 0.0
                        P[j, gt[j]] = 1.0
                    top = np.argsort(-P, axis=1, kind="stable")[:, :TS.TOP_K]
                    it["rec"]["top_cls"] = top.astype(np.int16)
                    it["rec"]["top_p"] = np.take_along_axis(P, top, axis=1).astype(np.float32)
                    self.f_stats["oracle_named"] += int(hit.sum())
                it["rec"]["oracle_gt"] = gt

    layers = [k for k in FamilyValidator.__mro__ if k.__dict__.get("_v3fam_layer")]
    if len(layers) != 1 or FamilyValidator.__mro__.count(base) != 1 or E3V not in FamilyValidator.__mro__:
        raise RuntimeError("the family validator holds %d family layers over E3's" % len(layers))
    return FamilyValidator, base


# ------------------------------------------------------------------ the family pass
def experiment_of(arm):
    """The experiment an arm verb belongs to (EXAM_READERS' names)."""
    if arm not in SCORE_ARMS:
        raise V3famRefused("arm %r is not one of the family's score arms %s" % (arm, list(SCORE_ARMS)))
    for p in ("D2", "D4"):
        if arm.startswith(p):
            return p
    return arm


def check_exam_allowed(arm, exam, test_token=None):
    """V3F.3.1: refuse an exam the arm may not read (test only with score-test's token)."""
    if exam not in EXAMS:
        raise V3famRefused("exam %r is not one of the family's %s" % (exam, list(EXAMS)))
    exp = experiment_of(arm)
    if exam == "test":
        if not any(test_token is t for t in _TEST_TOKENS):
            raise V3famRefused("test is read only by score-test, after the family verdict and a read prepared by "
                               "test-read (V3F.3.6)")
        return True
    if exp not in EXAM_READERS[exam]:
        raise V3famRefused("%s (%s) does not read %s: its readers are %s (V3F.3.1)" % (arm, exp, exam,
                                                                                     list(EXAM_READERS[exam])))
    return True


@contextlib.contextmanager
def test_read_token():
    """score-test's token: the only way a family pass reads the sealed test (V3F.3.6)."""
    tok = object()
    _TEST_TOKENS.append(tok)
    try:
        yield tok
    finally:
        _TEST_TOKENS.remove(tok)


def orientations_for(exam):
    """The EXIF tags a pass on exam reads: none, 0, 1; and 3, 6, 8 on ood22 and ood23 after the pre-flight."""
    if exam in OOD_EXAMS and exif_preflight_passed():
        return tuple(TS.FIT_ORIENTATIONS)
    return tuple(TS.OK_ORIENTATIONS)


def _flat_rows(rec, keys, fields):
    """{keys, img, <fields>} of per-image row records in exam key order."""
    import numpy as np
    parts = collections.defaultdict(list)
    for i, k in enumerate(keys):
        r = rec.get(k)
        if r is None:
            raise V3famRefused("the pass recorded nothing for exam image %s" % k)
        n = len(r[fields[0]])
        parts["img"].append(np.full(n, i, dtype=np.int64))
        for f in fields:
            parts[f].append(np.asarray(r[f]))
    out = {"keys": np.asarray(list(keys)), "img": np.concatenate(parts["img"]) if keys else np.zeros(0, np.int64)}
    for f in fields:
        out[f] = np.concatenate(parts[f], 0) if keys else np.zeros(0)
    return out


def _p7_arrays(v, keys):
    """P7's per-image arrays in exam key order (agnostic arrays for run_values: keys, tp, conf, pred_img, n_gt,
    with the ignored rows left out), for the target reading (t) and the OtherPlant one (o); and the full-set
    values in the validator's order (the order the scorer's own agnostic AP concatenates in)."""
    import numpy as np
    out = {}
    for tag in ("t", "o"):
        tps, confs, img, ngt = [], [], [], []
        for i, k in enumerate(keys):
            r = v.f_p7[k]
            keep = ~r["ign_" + tag]
            tps.append(r["tp_" + tag][keep])
            confs.append(r["q"][keep])
            img.append(np.full(int(keep.sum()), i, dtype=np.int64))
            ngt.append(r["n_" + tag])
        out[tag] = {"keys": np.asarray(list(keys)),
                    "tp": np.concatenate(tps, 0) if tps else np.zeros((0, SC.NIOU), bool),
                    "conf": np.concatenate(confs) if confs else np.zeros(0),
                    "pred_img": np.concatenate(img) if img else np.zeros(0, np.int64),
                    "n_gt": np.asarray(ngt, dtype=np.int64)}
    full = {}
    for tag in ("t", "o"):
        tp = [v.f_p7[k]["tp_" + tag] for k in v.f_p7_order]
        q = [v.f_p7[k]["q"] for k in v.f_p7_order]
        ig = [v.f_p7[k]["ign_" + tag] for k in v.f_p7_order]
        n = sum(int(v.f_p7[k]["n_" + tag]) for k in v.f_p7_order)
        tp = np.concatenate(tp, 0) if tp else np.zeros((0, SC.NIOU), bool)
        q = np.concatenate(q) if q else np.zeros(0)
        ig = np.concatenate(ig) if ig else np.zeros(0, bool)
        full[tag] = {"ignore": p7_value(tp, q, ig, n), "all": p7_value(tp, q, np.zeros(len(q), bool), n),
                     "n_gt": n, "ignored_rows": int(ig.sum())}
    return out, full


def _check_inject(cfg, exam):
    rows = cfg.get("inject")
    if not isinstance(rows, dict):
        raise V3famRefused("an %s pass needs inject {key: rows}" % cfg.get("mode"))
    keys = [r["key"] for r in C.read_manifest(C.manifest_path(exam))]
    missing, extra = sorted(set(keys) - set(rows)), sorted(set(rows) - set(keys))
    if missing or extra:
        raise V3famRefused("the injected rows cover %d of %s's %d images (missing %s, extra %s): every exam image "
                           "has its rows, possibly none" % (len(set(keys) & set(rows)), exam, len(keys), missing[:3],
                                                            extra[:3]))


def _family_pass(weights, exam, cfg, arm, lock_check=True, batch=S.BATCH, device=None, test_token=None):
    """One locked-scorer pass of `weights` on `exam` with the family validator installed (module docstring), for
    arm `arm` (the exam allow-list). Returns {result, arrays (exam key order), stage1, rows, p7, p7_full,
    stage1_agnostic (AP50-95, AP50), stats, gt, seconds, validator}."""
    import numpy as np
    check_frozen()
    check_exam_allowed(arm, exam, test_token)
    if exam in ROOT_EXAMS and _IN_ROOT.get("exam") != exam:
        raise V3famRefused("%s is scored only in its own root, in a subprocess with INC_DIR there "
                           "(family_pass_in_root)" % exam)
    if _IN_ROOT and _IN_ROOT.get("exam") != exam:
        raise V3famRefused("this process scores %s in its root, not %s" % (_IN_ROOT.get("exam"), exam))
    mode = cfg.get("mode")
    if mode not in MODES:
        raise V3famRefused("mode %r is not one of %s" % (mode, list(MODES)))
    if _IN_ROOT and mode not in ROOT_MODES:
        raise V3famRefused("a pass in another root takes %s only, not %s" % (list(ROOT_MODES), mode))
    if mode in CLASSIFY_MODES and cfg.get("embedder") is None:
        raise V3famRefused("a %s pass needs the embedder" % mode)
    if mode in INJECT_MODES:
        _check_inject(cfg, exam)
    cfg = dict(cfg, orientations=orientations_for(exam))
    cls, base = family_validator_class(cfg)
    prev = S._VALIDATOR
    del SC._CAPTURE[:]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="inc2_v3fam_") as tmp:
        S._VALIDATOR = cls
        try:
            res = S.score(weights, exam, Path(tmp) / "score.json", lock_check=lock_check, imgsz=IMGSZ, batch=batch,
                          device=device)
        except S.ScorerRefused as e:
            raise V3famRefused("the locked scorer refuses: %s" % e)
        except TS.E3Refused as e:
            raise V3famRefused("E3's path refuses: %s" % e)
        finally:
            S._VALIDATOR = prev if prev is not None else base
    caps = [x for x in SC._CAPTURE if isinstance(x, cls)]
    del SC._CAPTURE[:]
    if not caps:
        raise V3famRefused("the scorer ran no family validator")
    v = caps[-1]
    res = {k: val for k, val in res.items() if k != "out"}
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path(exam)))
    if C.sha256_text("\n".join(keys)) != res["key_order_sha256"]:
        raise V3famRefused("the exam's key order does not hash to the score's key_order_sha256")
    if sorted(v.f_seen) != keys:
        raise V3famRefused("the family validator saw %d of the exam's %d images" % (len(set(v.f_seen)), len(keys)))
    try:
        recompute, restricted = SC.check_capture(res["per_class"], v.sc_images)
        arrays = SC.flatten(v.sc_images, keys)
    except SC.SidecarError as e:
        raise V3famRefused("the captured per-image arrays do not reproduce the score: %s" % e)
    if v.e3_s1_ngt != int(res["n_boxes"]):
        raise V3famRefused("the stage-1 helper counted %d GT boxes, the scorer %d" % (v.e3_s1_ngt, res["n_boxes"]))
    tp = np.concatenate(v.e3_s1_tp, 0) if v.e3_s1_tp else np.zeros((0, SC.NIOU), dtype=bool)
    q = np.concatenate(v.e3_s1_q, 0) if v.e3_s1_q else np.zeros(0)
    s1 = S.collapsed_ap(tp, q, v.e3_s1_ngt)
    stats = {k: int(val) for k, val in sorted(v.e3_stats.items())}
    stats.update({k: int(val) for k, val in sorted(v.f_stats.items())})
    stats.update(orientations=dict(sorted(v.f_tags.items())), e3_orientations=dict(sorted(v.e3_orientations.items())),
                 frame_adapted=dict(sorted(v.f_frame.passed.items())) if v.f_frame is not None else {},
                 orientations_read=[str(t) for t in cfg["orientations"]], gt_roundtrip_max_diff=v.e3_gt_maxdiff,
                 embed_seconds=round(v.e3_embed_s, 3), cut_seconds=round(v.e3_cut_s, 3),
                 full_recompute_max_abs_diff=recompute, species_restricted_max_abs_diff=restricted)
    out = {"result": res, "arrays": arrays, "stage1_agnostic": s1, "stats": stats, "gt": list(v.e3_gt),
           "seconds": round(time.time() - t0, 3), "validator": v, "mode": mode, "exam": exam, "arm": arm,
           "stage1": None, "rows": None, "p7": None, "p7_full": None}
    if mode in CLASSIFY_MODES and mode != "gt":
        out["stage1"] = TS._stage1_arrays(v, keys)
        if mode == "oracle":
            out["stage1"]["oracle_gt"] = np.concatenate([v.e3_images[k]["oracle_gt"] for k in keys]) if keys \
                else np.zeros(0, np.int64)
    elif mode == "stage1":
        out["stage1"] = _flat_rows(v.f_s1, keys, ("xyxy", "q"))
    if mode in ROWS_MODES:
        out["rows"] = _flat_rows(v.f_rows, keys, ("xyxy", "conf", "cls"))
    if mode == "stage1":
        out["p7"], out["p7_full"] = _p7_arrays(v, keys)
    return out


def inject_from_rows(rows):
    """{key: {xyxy, conf, cls}} from a rows record (keys, img, xyxy, conf, cls; img ascending, each image's rows
    in their order), every key present."""
    import numpy as np
    img = np.asarray(rows["img"], dtype=np.int64)
    if len(img) and np.any(np.diff(img) < 0):
        raise V3famRefused("a rows record is grouped by image in key order (img ascending)")
    cuts = np.searchsorted(img, np.arange(len(rows["keys"]) + 1))
    out = {}
    for i, k in enumerate(rows["keys"]):
        a, b = int(cuts[i]), int(cuts[i + 1])
        out[str(k)] = {f: np.asarray(rows[f])[a:b] for f in ("xyxy", "conf", "cls") if f in rows}
    return out


# ------------------------------------------------------------------ another root (a subprocess)
def _root_env(root_):
    env = dict(os.environ)
    env["INC_DIR"] = str(root_)
    env["INC_ZOO_SOURCE"] = str(C.INC_DIR)
    pkg = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def family_pass_in_root(weights, exam, cfg, arm, out_dir, plain=False, lock_check=True, batch=S.BATCH, device=None,
                        root_dir=None):
    """_family_pass on an exam of another root (ooddev_v1, probe) in a subprocess with INC_DIR at that root (as
    Z1 scores its root); the scorer reads the root through a temporary view and writes nothing in it. cfg takes
    ROOT_MODES only (no embedder); an inject_rows cfg's rows are passed as a file. plain: the plain locked score
    of the same weights too, in the same subprocess (V3F.3.2 item 3). out_dir: where the child writes its
    result (outside the root). Returns _family_pass's dict (no validator) and "plain"."""
    import numpy as np
    if exam not in ROOT_EXAMS:
        raise V3famRefused("%s is not scored in another root" % exam)
    if cfg.get("mode") not in ROOT_MODES:
        raise V3famRefused("a pass in another root takes %s only" % list(ROOT_MODES))
    experiment_of(arm)
    if experiment_of(arm) not in EXAM_READERS[exam]:
        raise V3famRefused("%s does not read %s (V3F.3.1)" % (arm, exam))
    root_ = Path(root_dir) if root_dir else root_of(exam)
    if not (root_ / "splits" / "v1" / "LOCK.json").is_file():
        raise V3famRefused("the root %s holds no splits/v1/LOCK.json" % root_)
    out_dir = Path(out_dir)
    if out_dir.resolve() == root_.resolve() or root_.resolve() in out_dir.resolve().parents:
        raise V3famRefused("the child writes outside the root it scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"format": FMT_PASS_SPEC, "root": str(root_.resolve()), "weights": str(Path(weights).resolve()),
            "exam": exam, "arm": arm, "mode": cfg["mode"], "nms": cfg.get("nms", "locked"), "plain": bool(plain),
            "lock_check": bool(lock_check), "batch": int(batch), "device": device, "out": str(out_dir.resolve())}
    if cfg["mode"] == "inject_rows":
        inj = cfg.get("inject") or {}
        keys = sorted(inj)
        flat = {"keys": np.asarray(keys), "img": np.concatenate([np.full(len(inj[k]["conf"]), i, np.int64)
                                                                for i, k in enumerate(keys)]) if keys else
                np.zeros(0, np.int64)}
        for f in ("xyxy", "conf", "cls"):
            flat[f] = np.concatenate([np.asarray(inj[k][f]) for k in keys], 0) if keys else np.zeros(0)
        spec["inject"] = str(out_dir.resolve() / "inject.npz")
        SC.save_npz(spec["inject"], flat)
    sp = out_dir / "spec.json"
    sp.write_text(TS._dump(spec))
    p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam", "_pass", "--spec",
                        str(sp)], env=_root_env(root_), capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        raise V3famRefused("the pass of %s in %s exited %d: %s" % (exam, root_, p.returncode, " | ".join(tail)))
    got = _read_json(out_dir / "result.json")
    if not isinstance(got, dict):
        raise V3famRefused("the pass in %s wrote no result" % root_)
    for name in ("arrays", "stage1", "rows"):
        f = out_dir / ("%s.npz" % name)
        got[name] = SC.load_npz(f) if f.is_file() else None
    p7 = {}
    for tag in ("t", "o"):
        f = out_dir / ("p7_%s.npz" % tag)
        if f.is_file():
            p7[tag] = SC.load_npz(f)
    got["p7"] = p7 or None
    got["stage1_agnostic"] = tuple(got.get("stage1_agnostic") or ())
    return got


def _run_pass_spec(spec_path, batch=None, device=None, lock_check=None):
    """The _pass verb: the child side of family_pass_in_root."""
    spec = _read_json(spec_path)
    if not isinstance(spec, dict) or spec.get("format") != FMT_PASS_SPEC:
        raise V3famRefused("%s is not a family pass spec" % spec_path)
    if C.INC_DIR.resolve() != Path(spec["root"]).resolve():
        raise V3famRefused("INC_DIR is %s, the spec's root %s" % (C.INC_DIR, spec["root"]))
    if spec["exam"] not in ROOT_EXAMS:
        raise V3famRefused("%s is not a root exam" % spec["exam"])
    _IN_ROOT["exam"] = spec["exam"]
    out = Path(spec["out"])
    cfg = {"mode": spec["mode"], "nms": spec.get("nms", "locked")}
    if spec["mode"] == "inject_rows":
        cfg["inject"] = inject_from_rows(SC.load_npz(spec["inject"]))
    kw = {"lock_check": spec["lock_check"] if lock_check is None else lock_check,
          "batch": spec["batch"] if batch is None else batch, "device": spec.get("device") if device is None
          else device}
    ps = _family_pass(spec["weights"], spec["exam"], cfg, spec["arm"], **kw)
    doc = {"result": ps["result"], "stage1_agnostic": list(ps["stage1_agnostic"]), "stats": ps["stats"],
           "p7_full": ps["p7_full"], "mode": ps["mode"], "exam": ps["exam"], "arm": ps["arm"],
           "seconds": ps["seconds"], "code": code_shas(), "root": spec["root"]}
    if spec.get("plain"):
        with tempfile.TemporaryDirectory(prefix="inc2_v3fam_plain_") as tmp:
            pr = S.score(spec["weights"], spec["exam"], Path(tmp) / "plain.json", **kw)
        doc["plain"] = {k: v for k, v in pr.items() if k != "out"}
    SC.save_npz(out / "arrays.npz", ps["arrays"])
    for name in ("stage1", "rows"):
        if ps[name] is not None:
            SC.save_npz(out / ("%s.npz" % name), ps[name])
    for tag, a in (ps["p7"] or {}).items():
        SC.save_npz(out / ("p7_%s.npz" % tag), a)
    TS._write_json(out / "result.json", doc)
    return doc


# ------------------------------------------------------------------ score files (written once)
def _locked(scorer_sha):
    return TS._locked(scorer_sha)


def stamp(locked_sha, test):
    return "%s%s%s" % (S.TEST_PREFIX if test else "", STAMP, locked_sha)


def family_doc(arm, seed, exam, ps, cfg, variant="", extra=None, classifier=None, equivalence=None):
    """A family score document: the locked scorer's result, stamped V3F- (never a protocol score), with the
    mode, the stage-1 and stage-2 records, the checks, the equivalence records it rests on (require_equivalence's
    result; a deviation among them is listed under family_deviations) and the code. classifier: the pin's
    sha256s of the classifier a two-stage mode named the boxes with."""
    res = ps["result"]
    locked = _locked(res["scorer_sha256"])
    test = not res.get("production")
    doc = dict(res)
    doc.update({"format": FMT_SCORE, "family": NAME, "arm": arm, "seed": None if seed is None else int(seed),
                "exam": exam, "variant": variant or None, "scorer_sha256": stamp(locked, test),
                "locked_scorer_sha256": locked, "production": False, "v3fam_production": not test, "imgsz": IMGSZ,
                "settings": dict(res["settings"], mode=ps["mode"], nms=cfg.get("nms", "locked"),
                                 agnostic_nms=cfg.get("nms") == "agnostic", top_k=int(cfg.get("top_k") or TOP_K)),
                "stage1": {"n_boxes": int(ps["stats"].get("stage1_boxes", 0)),
                           "agnostic_map50_95": ps["stage1_agnostic"][0], "agnostic_map50": ps["stage1_agnostic"][1]},
                "stage2": {"embedder": getattr(cfg.get("embedder"), "name", None),
                           "n_crops": int(ps["stats"].get("crops", 0)), "classifier": classifier}
                if ps["mode"] in CLASSIFY_MODES else None,
                "equivalence": equivalence,
                "family_deviations": sorted(n for n, v in (equivalence or {}).items() if v.get("deviation")),
                "p7": ps.get("p7_full"),
                "checks": {k: ps["stats"].get(k) for k in ("orientations", "e3_orientations", "frame_adapted",
                                                           "gt_roundtrip_max_diff", "full_recompute_max_abs_diff",
                                                           "species_restricted_max_abs_diff")},
                "stats": ps["stats"], "decided_by": DECIDED_BY, "code": code_shas(), "created_utc": _utc(),
                "seconds": ps.get("seconds")})
    if extra:
        doc.update(extra)
    return doc


def commit_pass(d, exam, doc, ps, variant=""):
    """Write a family score once: its npz files (stage-1 records keep per-box features on dev only; a side file
    no JSON names yet, a crashed attempt's, is replaced, as twostage._commit_file does), then the JSON
    (scorer_native's commit: the JSON is hard-linked only if absent). Returns the JSON's path."""
    import numpy as np
    d = Path(d)
    jn, nn, sn, rn, pn = file_names(exam, variant)
    js = d / jn
    if js.exists() or js.is_symlink():
        raise V3famRefused("%s exists: a family score is written once" % js)
    d.mkdir(parents=True, exist_ok=True)
    side = []
    if ps.get("stage1") is not None:
        s1 = dict(ps["stage1"])
        if exam != "dev":
            s1.pop("feat", None)
        side.append(("stage1_images", d / sn, s1))
    if ps.get("rows") is not None:
        side.append(("rows", d / rn, ps["rows"]))
    if ps.get("p7") is not None:
        flat = {}
        for tag, a in ps["p7"].items():
            flat.update(("%s_%s" % (tag, k), np.asarray(v)) for k, v in a.items())
        side.append(("p7", d / pn, flat))
    for name, path, arrays in side:
        doc[name] = {"name": path.name, "sha256": SC.save_npz(path, arrays)}
    staged = (d / nn).with_name(".%s.%d.%d.staged" % (nn, os.getpid(), time.monotonic_ns()))
    try:
        npz_sha = SC.save_npz(staged, ps["arrays"])
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise
    doc["images"] = {"name": nn, "sha256": npz_sha, "n_images": int(len(ps["arrays"]["keys"])),
                     "n_predictions": int(len(ps["arrays"]["conf"])), "n_targets": int(len(ps["arrays"]["target_cls"]))}
    try:
        SN._commit(js, d / nn, staged, doc)
    except SN.NativeRefused as e:
        raise V3famRefused(str(e))
    return js


# ------------------------------------------------------------------ the reproductions (V3F.3.2)
def repro_status(diff):
    """passed (<= 1e-6), deviation (<= 0.002) or refused (beyond, or not finite)."""
    if diff is None or not math.isfinite(float(diff)):
        return "refused"
    if diff <= REPRO_TOL:
        return "passed"
    if diff <= DEVIATION_TOL:
        return "deviation"
    return "refused"


def compare_with_recorded(new, rec, what):
    """Refuse unless new and rec are scores of the same weights on the same exam by the same locked scorer and
    cover the same classes; returns {max_abs_diff, map50_95, species_map50_95, per_class: {c: diff}}."""
    if not isinstance(rec, dict):
        raise V3famRefused("%s: no recorded score to reproduce" % what)
    for k in COMPARE_STAMPS:
        if rec.get(k) != new.get(k):
            raise V3famRefused("%s: the family pass has %s %r, the recorded score %r" % (what, k, str(new.get(k))[:40],
                                                                                       str(rec.get(k))[:40]))
    ls_new = _locked(new.get("locked_scorer_sha256") or new.get("scorer_sha256"))
    ls_rec = rec.get("locked_scorer_sha256") or _locked(rec.get("scorer_sha256"))
    if ls_rec and ls_rec.startswith(TS.STAMP):
        ls_rec = ls_rec[len(TS.STAMP):]
    if ls_new != ls_rec:
        raise V3famRefused("%s: the recorded score was taken by another scorer (%s, not %s)" % (what, str(ls_rec)[:12],
                                                                                              str(ls_new)[:12]))
    if set(rec.get("per_class") or {}) != set(new.get("per_class") or {}):
        raise V3famRefused("%s: per_class covers %s in the family pass, %s in the record" % (
            what, sorted(new.get("per_class") or {}), sorted(rec.get("per_class") or {})))
    per = {c: abs(float(new["per_class"][c]) - float(rec["per_class"][c])) for c in rec["per_class"]}
    out = {"map50_95": abs(float(new["map50_95"]) - float(rec["map50_95"])), "per_class": per}
    if rec.get("species_map50_95") is not None or new.get("species_map50_95") is not None:
        out["species_map50_95"] = abs(float(new["species_map50_95"]) - float(rec["species_map50_95"]))
    if rec.get("agnostic_map50_95") is not None:
        out["agnostic_map50_95"] = abs(float(new["agnostic_map50_95"]) - float(rec["agnostic_map50_95"]))
    vals = [out["map50_95"]] + list(per.values()) + [out[k] for k in ("species_map50_95",) if k in out]
    out["max_abs_diff"] = max(vals) if vals else 0.0
    return out


def stage1_compare(new, stored):
    """A stage-1 box set against a stored dev.stage1.npz (V3F.3.2 item 2): the same rows (keys, image index),
    xyxy within 1e-3 px and q within 1e-4, or refused. Returns the record."""
    import numpy as np
    if [str(k) for k in new["keys"]] != [str(k) for k in stored["keys"]]:
        raise V3famRefused("the stage-1 record is in another key order than the stored one")
    if not np.array_equal(np.asarray(new["img"]), np.asarray(stored["img"])):
        raise V3famRefused("the stage-1 box set has %d rows, the stored one %d (or another image's)"
                           % (len(new["img"]), len(stored["img"])))
    dx = float(np.abs(np.asarray(new["xyxy"], np.float64) - np.asarray(stored["xyxy"], np.float64)).max()) \
        if len(new["img"]) else 0.0
    dq = float(np.abs(np.asarray(new["q"], np.float64) - np.asarray(stored["q"], np.float64)).max()) \
        if len(new["img"]) else 0.0
    rec = {"rows": int(len(new["img"])), "xyxy_max_abs_diff_px": dx, "q_max_abs_diff": dq,
           "xyxy_tol_px": STAGE1_XYXY_TOL, "q_tol": STAGE1_Q_TOL}
    if "top_cls" in new and "top_cls" in stored:
        rec["top_cls_equal"] = bool(np.array_equal(np.asarray(new["top_cls"]), np.asarray(stored["top_cls"])))
    if not dx <= STAGE1_XYXY_TOL or not dq <= STAGE1_Q_TOL:
        raise V3famRefused("the stage-1 box set does not reproduce the stored one (xyxy %.3g px, q %.3g; tolerances "
                           "%g px, %g)" % (dx, dq, STAGE1_XYXY_TOL, STAGE1_Q_TOL))
    return rec


def _equiv_doc(item, per, status, extra=None):
    import torch
    import ultralytics
    doc = {"format": FMT_EQUIV, "item": item, "status": status, "per": per, "tolerance": REPRO_TOL,
           "deviation_band": DEVIATION_TOL, "code": code_shas(), "decided_by": DECIDED_BY,
           "versions": dict(TS._versions(), ultralytics=ultralytics.__version__, torch=torch.__version__),
           "testing": S.testing(), "created_utc": _utc()}
    doc.update(extra or {})
    return doc


def _status_of(diffs, what):
    st = [repro_status(d) for d in diffs]
    if "refused" in st:
        raise V3famRefused("%s: a reproduction differs by more than %g (%s): no family score is taken"
                           % (what, DEVIATION_TOL, ["%.3g" % d for d in diffs]))
    return "deviation" if "deviation" in st else "passed"


def equivalence_dev_identity(lock_check=True, batch=S.BATCH, device=None):
    """Item 1: b_v2_m640 seeds 0-2 through the family identity pass reproduce their recorded protocol dev scores
    (mAP, species mean and every class) to 1e-6. Writes equivalence/dev_identity.json once."""
    out = equiv_dir() / "dev_identity.json"
    if out.exists():
        raise V3famRefused("%s exists: written once" % out)
    testing = TS._testing_ok()
    per, diffs = {}, []
    for s in SEEDS:
        src = TS._source("M", s, testing_ok=testing)
        ps = _family_pass(src["weights"], "dev", {"mode": "identity", "nms": "locked"}, "E3RS",
                          lock_check=lock_check, batch=batch, device=device)
        cmp_ = compare_with_recorded(ps["result"], src["recorded_doc"], "b_v2_m640 seed %d dev" % s)
        per[str(s)] = {"weights_sha256": src["weights_sha256"], "recorded": {k: src["recorded"][k] for k in (
            "file", "sha256", "species_map50_95", "agnostic_map50_95", "production")}, "diff": cmp_,
                       "status": repro_status(cmp_["max_abs_diff"])}
        diffs.append(cmp_["max_abs_diff"])
    status = _status_of(diffs, "dev identity")
    doc = _equiv_doc("dev_identity", per, status, {"exam": "dev", "reference": REFERENCE, "seeds": list(SEEDS)})
    write_once(out, doc)
    log("equivalence dev identity: %s (max diff %.3g)" % (status, max(diffs)))
    return doc


def equivalence_two_stage(e3_arm, embedder, workers=None, lock_check=True, batch=S.BATCH, device=None):
    """Item 2 for one E3 arm (M, A or B): its stage-1 weights with E3's pinned classifier through the family
    two-stage pass reproduce E3's recorded dev files to 1e-6, and each stage-1 box set the stored dev.stage1.npz.
    Writes equivalence/two_stage_<arm>.json once. Returns {record, passes: {seed: pass}} (E3RS keeps the passes
    as the family-path arrays)."""
    if e3_arm not in TS.ARMS:
        raise V3famRefused("E3 arm %r is not one of %s" % (e3_arm, list(TS.ARM_ORDER)))
    out = equiv_dir() / ("two_stage_%s.json" % e3_arm)
    if out.exists():
        raise V3famRefused("%s exists: written once" % out)
    testing = TS._testing_ok()
    ccfg, _crec, shas = TS._clf_cfg()
    per, diffs, passes = {}, [], {}
    with TS._workers(workers) as wk:
        for s in SEEDS:
            src = TS._source(e3_arm, s, testing_ok=testing)
            d = TS.arm_dir(e3_arm, s)
            jn, _nn, sn = TS.file_names("dev")
            rec = _read_json(d / jn)
            if not isinstance(rec, dict) or rec.get("format") != TS.FMT_SCORE:
                raise V3famRefused("E3-%s seed %d has no recorded dev file" % (e3_arm, s))
            if (rec.get("stage1") or {}).get("weights_sha256") != src["weights_sha256"]:
                raise V3famRefused("E3-%s seed %d's file names other stage-1 weights" % (e3_arm, s))
            st2 = rec.get("stage2") or {}
            if st2.get("classifier_npz_sha256") != shas["npz_sha256"]:
                raise V3famRefused("E3-%s seed %d was scored with another classifier than the pinned one" % (e3_arm, s))
            if _sha(d / sn) != (rec.get("stage1_images") or {}).get("sha256"):
                raise V3famRefused("E3-%s seed %d's dev.stage1.npz does not hash as its file records" % (e3_arm, s))
            stored = SC.load_npz(d / sn)
            cfg = dict(ccfg, mode="two_stage", nms=TS.NMS[e3_arm], embedder=embedder, workers=wk, top_k=TS.TOP_K)
            ps = _family_pass(src["weights"], "dev", cfg, "E3RS", lock_check=lock_check, batch=batch, device=device)
            cmp_ = compare_with_recorded(ps["result"], rec, "E3-%s seed %d dev" % (e3_arm, s))
            s1 = stage1_compare(ps["stage1"], stored)
            per[str(s)] = {"weights_sha256": src["weights_sha256"], "recorded": {
                "file": "twostage/%s/arms/%s/s%d/%s" % (TS.E3_NAME, e3_arm, s, jn), "sha256": _sha(d / jn),
                "species_map50_95": rec.get("species_map50_95")}, "diff": cmp_, "stage1": s1,
                "status": repro_status(cmp_["max_abs_diff"])}
            diffs.append(cmp_["max_abs_diff"])
            passes[s] = ps
    status = _status_of(diffs, "E3-%s two-stage" % e3_arm)
    doc = _equiv_doc("two_stage", per, status, {"exam": "dev", "e3_arm": e3_arm, "nms": TS.NMS[e3_arm],
                                                "classifier": shas, "embedder": getattr(embedder, "name", None)})
    write_once(out, doc)
    log("equivalence two-stage E3-%s: %s (max diff %.3g)" % (e3_arm, status, max(diffs)))
    return {"record": doc, "passes": passes}


def equivalence_identity(exam, arm, lock_check=True, batch=S.BATCH, device=None, scratch=None):
    """Item 3: b_v2_m640 seed 0's identity output on exam equals the plain locked score of the same weights on the
    same exam, taken in the same job (for ooddev_v1 and probe: in one subprocess in their root). The plain score
    is recorded here only. Writes equivalence/identity_<exam>.json once."""
    if exam in ("dev", "test"):
        raise V3famRefused("item 3 covers the exams other than dev (item 1) and never test")
    out = equiv_dir() / ("identity_%s.json" % exam)
    if out.exists():
        raise V3famRefused("%s exists: written once" % out)
    src = TS._source("M", SEEDS[0], testing_ok=TS._testing_ok())
    if exam in ROOT_EXAMS:
        with tempfile.TemporaryDirectory(prefix="inc2_v3fam_eq_", dir=scratch) as tmp:
            got = family_pass_in_root(src["weights"], exam, {"mode": "identity", "nms": "locked"}, arm, tmp,
                                      plain=True, lock_check=lock_check, batch=batch, device=device)
        res, plain = got["result"], got["plain"]
    else:
        check_exam_allowed(arm, exam)
        ps = _family_pass(src["weights"], exam, {"mode": "identity", "nms": "locked"}, arm, lock_check=lock_check,
                          batch=batch, device=device)
        with tempfile.TemporaryDirectory(prefix="inc2_v3fam_plain_") as tmp:
            plain = S.score(src["weights"], exam, Path(tmp) / "plain.json", lock_check=lock_check, imgsz=IMGSZ,
                            batch=batch, device=device)
        plain = {k: v for k, v in plain.items() if k != "out"}
        res = ps["result"]
    cmp_ = compare_with_recorded(res, plain, "b_v2_m640 seed 0 %s" % exam)
    status = _status_of([cmp_["max_abs_diff"]], "%s identity" % exam)
    per = {"0": {"weights_sha256": src["weights_sha256"], "diff": cmp_, "status": status,
                 "plain": {k: plain.get(k) for k in ("map50_95", "species_map50_95", "agnostic_map50_95",
                                                      "per_class", "production", "scorer_sha256", "n_images")}}}
    doc = _equiv_doc("identity", per, status, {"exam": exam, "arm": arm, "reference": REFERENCE,
                                               "note": "the plain scores are recorded here only and never compared "
                                                       "with anything"})
    write_once(out, doc)
    log("equivalence identity on %s: %s (diff %.3g)" % (exam, status, cmp_["max_abs_diff"]))
    return doc


def equivalence_injection(lock_check=True, batch=S.BATCH, device=None):
    """Item 4: b_v2_m640 seed 0's own dev predictions, taken to original pixels (the exact inverse of the
    letterbox) and injected (inject_rows), reproduce its recorded protocol dev score to 1e-6. Writes
    equivalence/injection.json once."""
    out = equiv_dir() / "injection.json"
    if out.exists():
        raise V3famRefused("%s exists: written once" % out)
    src = TS._source("M", SEEDS[0], testing_ok=TS._testing_ok())
    ident = _family_pass(src["weights"], "dev", {"mode": "identity", "nms": "locked"}, "E3RS", lock_check=lock_check,
                         batch=batch, device=device)
    inj = inject_from_rows(ident["rows"])
    ps = _family_pass(src["weights"], "dev", {"mode": "inject_rows", "nms": "locked", "inject": inj}, "E3RS",
                      lock_check=lock_check, batch=batch, device=device)
    cmp_ = compare_with_recorded(ps["result"], src["recorded_doc"], "b_v2_m640 seed 0 injected dev")
    vs_ident = compare_with_recorded(ps["result"], ident["result"], "injected vs identity")
    status = _status_of([cmp_["max_abs_diff"]], "injection")
    per = {"0": {"weights_sha256": src["weights_sha256"], "diff": cmp_, "vs_identity": vs_ident, "status": status,
                 "rows": int(len(ident["rows"]["conf"])), "injected_rows": int(ps["stats"].get("injected_rows", 0)),
                 "gt_roundtrip_max_diff": ps["stats"].get("gt_roundtrip_max_diff")}}
    doc = _equiv_doc("injection", per, status, {"exam": "dev", "reference": REFERENCE})
    write_once(out, doc)
    log("equivalence injection: %s (diff %.3g)" % (status, cmp_["max_abs_diff"]))
    return doc


def equivalence_names(exam=None, e3_arms=()):
    """The equivalence records a family score on exam rests on."""
    names = ["dev_identity.json", "injection.json"]
    names += ["two_stage_%s.json" % a for a in e3_arms]
    if exam not in (None, "dev", "test"):
        names.append("identity_%s.json" % exam)
    return names


def require_equivalence(names):
    """{name: {sha256, status, deviation}} of the named equivalence records, refusing a missing or refused one or
    one taken by other code than this (V3F.3.2: before any family score)."""
    check_frozen()
    now = code_shas()
    out = {}
    for n in names:
        p = equiv_dir() / n
        d = _read_json(p)
        if not isinstance(d, dict) or d.get("format") != FMT_EQUIV or d.get("status") not in ("passed", "deviation"):
            raise V3famRefused("%s is missing or did not pass: no family score is taken without it" % p)
        if d.get("code") != now:
            diff = sorted(k for k in set(now) | set(d.get("code") or {}) if now.get(k) != (d.get("code") or {}).get(k))
            raise V3famRefused("the code changed since %s ran (%s): it is moved aside and rerun, as a deviation"
                               % (n, ", ".join(diff)))
        out[n] = {"sha256": _sha(p), "status": d["status"], "deviation": d["status"] == "deviation"}
    return out


# ------------------------------------------------------------------ the EXIF frame pre-flight (V3F.3.2)
def exif_frame_check(path):
    """{tag, loader_shape, transposed_shape, mad, ok} of one image: the array Ultralytics' validator loader reads
    (ultralytics.utils.patches.imread, BaseDataset.load_image's call) against PIL's exif_transpose, within a mean
    absolute difference of EXIF_MAD_TOL grey levels, the shape the validator records as ori_shape equal to the
    transposed one."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from ultralytics.utils.patches import imread
    tag = TS.header(path)[0]
    a = imread(str(path), flags=cv2.IMREAD_COLOR)
    if a is None:
        return {"tag": tag, "ok": False, "why": "the loader read nothing"}
    a = np.ascontiguousarray(a[..., ::-1])
    with Image.open(path) as im0:
        b = np.asarray(ImageOps.exif_transpose(im0).convert("RGB"))
    rec = {"tag": tag, "loader_shape": list(a.shape[:2]), "transposed_shape": list(b.shape[:2])}
    if a.shape != b.shape:
        return dict(rec, ok=False, mad=None, why="the loader's frame is not exif_transpose's")
    mad = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    return dict(rec, mad=mad, ok=bool(mad <= EXIF_MAD_TOL), mad_tol=EXIF_MAD_TOL)


def _loader_rgb(path):
    import cv2
    import numpy as np
    from ultralytics.utils.patches import imread
    a = imread(str(path), flags=cv2.IMREAD_COLOR)
    return np.ascontiguousarray(a[..., ::-1])


def exif_preflight(rows=None, tile=512):
    """V3F.3.2's frame pre-flight, on one base_v2 copy each of tags 3, 6, 8 and 0 (the first of each tag in the
    order of stable_int('inc2/v3fam/exif/<key>')): the numeric frame check, and a montage of the GT boxes drawn on
    the loader's array for the visual check (exif_verdict). Writes exif/preflight.json and montage.jpg once."""
    import io
    from PIL import Image, ImageDraw
    out = exif_dir() / "preflight.json"
    if out.exists():
        raise V3famRefused("%s exists: written once" % out)
    if rows is None:
        try:
            msha = C2.verify_manifest_against_lock_v2("base_v2")
        except C2.Inc2Error as e:
            raise V3famRefused("LOCK v2's base_v2: %s" % e)
        rows = C.read_manifest(C2.v2_manifest_path("base_v2"))
    else:
        msha = None
    ranked = sorted(rows, key=lambda r: (C.stable_int(EXIF_SEED_TEXT % r["key"]), r["key"]))
    picks = {}
    for r in ranked:
        tag = TS.header(r["image"])[0]
        if tag in EXIF_TAGS and tag not in picks:
            picks[tag] = r
        if len(picks) == len(EXIF_TAGS):
            break
    missing = [t for t in EXIF_TAGS if t not in picks]
    if missing:
        raise V3famRefused("base_v2 holds no image with EXIF tag(s) %s: the pre-flight needs one of each" % missing)
    checks, tiles = {}, []
    with tempfile.TemporaryDirectory(prefix="inc2_v3fam_exif_") as tmp:
        for t in EXIF_TAGS:
            r = picks[t]
            ck = exif_frame_check(r["image"])
            boxes = TS._parse_verified(r["label"], r["label_sha256"], tmp)
            ck.update(key=r["key"], sha256=r.get("sha256"), boxes=len(boxes))
            checks[str(t)] = ck
            a = _loader_rgb(r["image"])
            im = Image.fromarray(a)
            dr = ImageDraw.Draw(im)
            H, W = a.shape[:2]
            lw = max(2, int(round(max(W, H) / 300.0)))
            for (_c, cx, cy, w, h) in boxes:
                dr.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
                             outline=(255, 0, 0), width=lw)
            sc = tile / float(max(W, H))
            tiles.append(im.resize((max(1, int(round(W * sc))), max(1, int(round(H * sc)))), Image.BILINEAR))
    mont = Image.new("RGB", (sum(t.size[0] for t in tiles) + 8 * (len(tiles) - 1), max(t.size[1] for t in tiles)),
                     (255, 255, 255))
    x = 0
    for t in tiles:
        mont.paste(t, (x, 0))
        x += t.size[0] + 8
    buf = io.BytesIO()
    mont.save(buf, format="JPEG", quality=90)
    msha_img = write_once(exif_dir() / "montage.jpg", buf.getvalue())
    numeric = all(c["ok"] for c in checks.values())
    doc = {"format": FMT_EXIF, "tags": list(EXIF_TAGS), "checks": checks, "numeric_passed": numeric,
           "base_v2_sha256": msha, "montage": {"name": "montage.jpg", "sha256": msha_img,
                                               "order": [str(t) for t in EXIF_TAGS]},
           "visual": "pending: a named viewer checks that the GT boxes lie on the plants (exif-verdict)",
           "decided_by": DECIDED_BY, "code": code_shas(), "created_utc": _utc()}
    write_once(out, doc)
    log("EXIF frame pre-flight: numeric %s (%s); visual check pending" % (
        "passed" if numeric else "FAILED", {k: c.get("mad") for k, c in checks.items()}))
    return doc


def exif_verdict(frame, read_by):
    """The visual half of the pre-flight: yes when the GT boxes on the montage lie on the plants; written once by
    a named viewer, naming the pre-flight record by sha256."""
    if frame not in ("yes", "no"):
        raise V3famRefused("--frame yes|no")
    pre = exif_dir() / "preflight.json"
    d = _read_json(pre)
    if not isinstance(d, dict) or d.get("format") != FMT_EXIF:
        raise V3famRefused("no pre-flight record at %s" % pre)
    if _sha(exif_dir() / "montage.jpg") != (d.get("montage") or {}).get("sha256"):
        raise V3famRefused("the montage does not hash as the pre-flight records")
    doc = {"format": FMT_EXIF_VERDICT, "frame": frame, "read_by": read_by_value(read_by),
           "preflight_sha256": _sha(pre), "montage_sha256": d["montage"]["sha256"], "written_utc": _utc()}
    write_once(exif_dir() / "verdict.json", doc)
    return doc


def exif_preflight_passed():
    """True when the pre-flight's numeric checks passed and its visual verdict, naming that record, is yes."""
    pre = exif_dir() / "preflight.json"
    d, v = _read_json(pre), _read_json(exif_dir() / "verdict.json")
    return (isinstance(d, dict) and d.get("format") == FMT_EXIF and d.get("numeric_passed") is True
            and isinstance(v, dict) and v.get("format") == FMT_EXIF_VERDICT and v.get("frame") == "yes"
            and v.get("preflight_sha256") == _sha(pre))


# ------------------------------------------------------------------ sessions (V3F.3.1)
_DIGITS_RE = re.compile(r"\A(.+)_[0-9]+\Z")
_IOS_RE = re.compile(r"\A([0-9]{8})_[0-9]{9}_iOS\Z")


def session_of(exam, key, field=""):
    """A key's session: dev, test and probe their manifest's session field; ood22 and ood23 by the key rule
    (strip the exam prefix; <YYYYMMDD>_<9 digits>_iOS -> <YYYYMMDD>_iOS; else everything before a trailing
    _<digits>; any other key refuses)."""
    if exam in ("dev", "test", "probe"):
        if not field:
            raise V3famRefused("%s key %s has no session field" % (exam, key))
        return str(field)
    if exam in OOD_EXAMS:
        pre = "%s__" % exam
        if not str(key).startswith(pre):
            raise V3famRefused("%s key %r does not start with %s" % (exam, key, pre))
        return key_session(str(key)[len(pre):])
    raise V3famRefused("no session rule for %s (sessions are drawn on dev, test, probe, ood22 and ood23)" % exam)


def key_session(rest):
    m = _IOS_RE.match(rest)
    if m:
        return "%s_iOS" % m.group(1)
    m = _DIGITS_RE.match(rest)
    if m:
        return m.group(1)
    raise V3famRefused("key %r has no session by the rule (<..>_<digits> or <YYYYMMDD>_<9 digits>_iOS)" % rest)


def sessions_for(exam, rows, keys=None):
    """[session of each key] in keys' order (default: the manifest rows sorted by key, the arrays' order)."""
    by = {r["key"]: r for r in rows}
    keys = sorted(by) if keys is None else [str(k) for k in keys]
    return [session_of(exam, k, by[k].get("session") or "") for k in keys]


def dev_session_stratum(test_rows, dev_rows):
    """The test keys whose session is one of dev's (V3F.3.6: 329 of 1,977 on 2026-10-11), sorted."""
    ds = {r.get("session") for r in dev_rows if r.get("session")}
    return sorted(r["key"] for r in test_rows if r.get("session") in ds)


# ------------------------------------------------------------------ the statistics (V3F.3.3)
def seed_text(cid, kind):
    if kind not in ("image", "session"):
        raise V3famRefused("a bootstrap is drawn over images or sessions, not %r" % kind)
    return "inc2/v3fam/%s/%s" % (cid, kind)


def image_counts(n, seed_text_, resamples=RESAMPLES):
    """[resamples, n] image multiplicities of the image bootstrap (scorer_sidecar.resample_indices)."""
    import numpy as np
    if n < 2:
        raise V3famRefused("an image bootstrap needs at least 2 exam images, got %d" % n)
    return np.stack([np.bincount(row, minlength=n) for row in SC.resample_indices(n, resamples, seed_text_)])


def session_counts(sessions, seed_text_, resamples=RESAMPLES):
    """[resamples, n] image multiplicities of the session bootstrap: the sessions (sorted) drawn with replacement
    by numpy.random.default_rng(stable_int(seed text)), each image counted as often as its session was drawn."""
    import numpy as np
    uniq = sorted(set(sessions))
    if len(uniq) < 2:
        raise V3famRefused("a session bootstrap needs at least 2 sessions, got %d" % len(uniq))
    pos = {s: i for i, s in enumerate(uniq)}
    idx = np.asarray([pos[s] for s in sessions], dtype=np.int64)
    rng = np.random.default_rng(C.stable_int(seed_text_))
    draws = rng.integers(0, len(uniq), size=(resamples, len(uniq)))
    per = np.stack([np.bincount(row, minlength=len(uniq)) for row in draws])
    return per[:, idx]


def _same_exam(runs, statistic):
    import numpy as np
    keys = [str(k) for k in runs[0]["keys"]]
    for a in runs[1:]:
        if [str(k) for k in a["keys"]] != keys:
            raise V3famRefused("the runs' per-image arrays are not in one key order")
        if statistic == "species":
            if not (np.array_equal(a["target_img"], runs[0]["target_img"])
                    and np.array_equal(a["target_cls"], runs[0]["target_cls"])):
                raise V3famRefused("the runs' GT boxes differ: they were not scored on one exam")
        elif not np.array_equal(a["n_gt"], runs[0]["n_gt"]):
            raise V3famRefused("the runs' GT counts differ: they were not scored on one exam")
    return keys


def run_values(runs, counts, statistic="species", species=SC.SPECIES, ap_fn=None):
    """[len(counts), len(runs)] float64: the statistic of each run under each row of image multiplicities, on
    the run's tie-broken arrays (NaN where the row holds no GT box). species: the mean over the species with a GT
    box in the row (inc2.baseline.native_bootstrap's twelve); agnostic: inc.scorer.collapsed_ap."""
    import numpy as np
    ap_fn = ap_fn or SC._ap_per_class()
    if statistic not in ("species", "agnostic"):
        raise V3famRefused("statistic %r" % statistic)
    counts = np.asarray(counts)
    V_ = np.full((len(counts), len(runs)), np.nan)
    for j, a in enumerate(runs):
        tb = SC.tie_break(a)
        if statistic == "species":
            per = []
            for name in species:
                s_id = C.CLASS_NAMES.index(name)
                tp, conf, pimg, gt = SC.species_arrays(tb, s_id)
                per.append((s_id, tp, conf, pimg, gt))
            for r in range(len(counts)):
                m = counts[r]
                aps = []
                for s_id, tp, conf, pimg, gt in per:
                    n_gt = int((m * gt).sum())
                    if n_gt <= 0:
                        continue
                    aps.append(SC.class_ap(np.repeat(tp, m[pimg], axis=0), np.repeat(conf, m[pimg]), n_gt, s_id,
                                           ap_fn))
                if aps:
                    V_[r, j] = statistics.fmean(aps)
        else:
            gt = np.asarray(tb["n_gt"], dtype=np.int64)
            tp, conf, pimg = tb["tp"], tb["conf"], tb["pred_img"]
            for r in range(len(counts)):
                m = counts[r]
                n_gt = int((m * gt).sum())
                if n_gt <= 0:
                    continue
                rep = m[pimg]
                V_[r, j] = S.collapsed_ap(np.repeat(tp, rep, axis=0), np.repeat(conf, rep), n_gt)[0]
    return V_


def _diffs(Vx, Vy):
    """Per resample: mean over X's runs - mean over Y's (statistics.fmean, as native_bootstrap), skipping a
    resample without a GT box. Returns (diffs, the resamples kept)."""
    import numpy as np
    out, kept = [], []
    for r in range(len(Vx)):
        x, y = Vx[r], Vy[r]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        out.append(statistics.fmean(float(v) for v in x) - statistics.fmean(float(v) for v in y))
        kept.append(r)
    return out, kept


def _sd(v):
    import numpy as np
    return float(np.std(np.asarray(v, dtype=np.float64), ddof=1)) if len(v) >= 2 else None


def _bootstrap(x_runs, y_runs, counts, statistic, species, ap_fn):
    _same_exam(list(x_runs) + list(y_runs), statistic)
    Vx = run_values(x_runs, counts, statistic, species, ap_fn)
    Vy = run_values(y_runs, counts, statistic, species, ap_fn)
    d, kept = _diffs(Vx, Vy)
    return {"se": _sd(d), "n_valid": len(d), "diffs": d, "kept": kept, "Vx": Vx, "Vy": Vy}


def image_bootstrap(x_runs, y_runs, seed_text_, resamples=RESAMPLES, statistic="species", species=SC.SPECIES,
                    ap_fn=None):
    """The paired image bootstrap of D = mean(X runs) - mean(Y runs): one draw of the exam's images for every run
    (V3F.3.3). Reproduces inc2.baseline.native_bootstrap's se exactly on the same arrays and seed text (species)
    and agnostic_bootstrap's (agnostic). Returns {se, n_valid, diffs, kept, Vx, Vy}."""
    if not x_runs or not y_runs:
        raise V3famRefused("the bootstrap needs runs on both sides")
    n = len(x_runs[0]["keys"])
    return _bootstrap(x_runs, y_runs, image_counts(n, seed_text_, resamples), statistic, species, ap_fn)


def session_bootstrap(x_runs, y_runs, sessions, seed_text_, resamples=RESAMPLES, statistic="species",
                      species=SC.SPECIES, ap_fn=None):
    """The session-cluster bootstrap (decision 6): sessions drawn with replacement, every image of a drawn session
    taken with its session's multiplicity; the same statistic. sessions: one per image, in the arrays' order."""
    if not x_runs or not y_runs:
        raise V3famRefused("the bootstrap needs runs on both sides")
    if len(sessions) != len(x_runs[0]["keys"]):
        raise V3famRefused("%d sessions for %d images" % (len(sessions), len(x_runs[0]["keys"])))
    return _bootstrap(x_runs, y_runs, session_counts(sessions, seed_text_, resamples), statistic, species, ap_fn)


def statistic_value(arrays, statistic="species", species=SC.SPECIES, ap_fn=None):
    """The full-set statistic of one run on its tie-broken arrays (every image once)."""
    import numpy as np
    n = len(arrays["keys"])
    return float(run_values([arrays], np.ones((1, n), dtype=np.int64), statistic, species, ap_fn)[0, 0])


def subset_arrays(arrays, keys):
    """Per-image arrays restricted to keys (in the arrays' order), images re-indexed: dev minus H (D0-S), the
    329-image stratum. Works on species arrays (target_*) and agnostic ones (n_gt)."""
    import numpy as np
    want = set(str(k) for k in keys)
    allk = [str(k) for k in arrays["keys"]]
    missing = want - set(allk)
    if missing:
        raise V3famRefused("%d keys are not exam images (e.g. %s)" % (len(missing), sorted(missing)[:3]))
    sel = np.asarray([k in want for k in allk], dtype=bool)
    new = -np.ones(len(allk), dtype=np.int64)
    new[sel] = np.arange(int(sel.sum()))
    out = {"keys": np.asarray([k for k in allk if k in want])}
    p = sel[np.asarray(arrays["pred_img"], dtype=np.int64)] if len(arrays["pred_img"]) else np.zeros(0, bool)
    out["pred_img"] = new[np.asarray(arrays["pred_img"], dtype=np.int64)[p]]
    for f in ("tp", "conf", "pred_cls"):
        if f in arrays:
            out[f] = np.asarray(arrays[f])[p]
    if "target_img" in arrays:
        t = sel[np.asarray(arrays["target_img"], dtype=np.int64)] if len(arrays["target_img"]) else np.zeros(0, bool)
        out["target_img"] = new[np.asarray(arrays["target_img"], dtype=np.int64)[t]]
        out["target_cls"] = np.asarray(arrays["target_cls"])[t]
    if "n_gt" in arrays:
        out["n_gt"] = np.asarray(arrays["n_gt"])[sel]
    return out


def subset_mean(per_class, species):
    """The mean of per_class over the species of a fixed list that have a GT box (e.g. D5b's seven)."""
    vals = [float(per_class[s]) for s in species if s in (per_class or {})]
    return statistics.fmean(vals) if vals else None


def _mean_sd(xs):
    xs = [float(x) for x in xs]
    return (statistics.fmean(xs) if xs else None, statistics.stdev(xs) if len(xs) >= 2 else None)


def outcome(qualifies, d, sigma):
    """(class, flags, interval) of V3F.3.3: qualifies (flag below SESOI when the 90 % interval lies inside
    (-SESOI, SESOI)); equivalent (does not qualify, interval inside); inconclusive (flag reversed when the whole
    interval lies below 0)."""
    lo, hi = d - Z90 * sigma, d + Z90 * sigma
    inside = -SESOI < lo and hi < SESOI
    if qualifies:
        return "qualifies", (["below SESOI"] if inside else []), [lo, hi]
    if inside:
        return "equivalent", [], [lo, hi]
    return "inconclusive", (["reversed"] if hi < 0 else []), [lo, hi]


def decide(d, se, x_values, y_values, one_system=False):
    """The E-rule, combined z and the outcome class of one comparison (V3F.3.3). x_values, y_values: the per-seed
    values (three each; one_system: X is one system, its condition D > 2 x sd(Y) and its seed term 2 sd(Y)^2 / 3)."""
    _mx, sx = _mean_sd(x_values)
    _my, sy = _mean_sd(y_values)
    if one_system:
        if len(x_values) != 1 or len(y_values) != len(SEEDS):
            raise V3famRefused("a one-system comparison has one X value and %d Y values" % len(SEEDS))
        two_sd = 2.0 * sy
        seed_term = 2.0 * sy ** 2 / len(SEEDS)
        pooled = None
    else:
        if len(x_values) != len(SEEDS) or len(y_values) != len(SEEDS):
            raise V3famRefused("a comparison has %d seeds per side, not %d and %d" % (len(SEEDS), len(x_values),
                                                                                    len(y_values)))
        pooled = math.sqrt((sx ** 2 + sy ** 2) / 2.0)
        two_sd = 2.0 * pooled
        seed_term = (sx ** 2 + sy ** 2) / len(SEEDS)
    if se is None or not math.isfinite(se):
        raise V3famRefused("the comparison has no finite SE")
    sigma = math.sqrt(se ** 2 + seed_term)
    z = d / sigma if sigma > 0 else (math.inf if d > 0 else -math.inf if d < 0 else 0.0)
    conds = {"above_2_sd": d > two_sd, "above_se": d > se}
    q = all(conds.values())
    cls, flags, iv = outcome(q, d, sigma)
    return {"diff": d, "pooled_sd": pooled, "two_sd": two_sd, "se": se, "seed_term": seed_term, "sigma": sigma,
            "z": z, "conditions": conds, "qualifies": q, "outcome": cls, "flags": flags, "interval_90": iv,
            "one_system": bool(one_system)}


def selection_correct(sel_runs, sel_values, y_runs, counts, d_plain, statistic="species", species=SC.SPECIES,
                      ap_fn=None, Vy=None, VB=None):
    """Decision 5: E3's choice redone in each image resample r among E3-M, E3-A, E3-B (the arm with the largest
    resample mean over seeds; ties M, then A, then B; the qualification conditions are not re-evaluated); T_r =
    S*_c(r) - S*_Y; D_sel = D - mean_r[T_r - (S_c(r) - S_Y)] with S the full-set statistic on the same tie-broken
    arrays; SE_sel = sd_r(T_r). sel_runs {M, A, B: [arrays]}, sel_values the per-seed score values (reported)."""
    import numpy as np
    for a in SELECTION_ARMS:
        if a not in sel_runs:
            raise V3famRefused("the selection correction needs E3-%s's family-path dev arrays" % a)
    _same_exam([x for a in SELECTION_ARMS for x in sel_runs[a]] + list(y_runs), statistic)
    n = len(y_runs[0]["keys"])
    ones = np.ones((1, n), dtype=np.int64)
    Vs = {a: (VB if (a == SELECTED and VB is not None) else run_values(sel_runs[a], counts, statistic, species, ap_fn))
          for a in SELECTION_ARMS}
    Vy = Vy if Vy is not None else run_values(y_runs, counts, statistic, species, ap_fn)
    full = {a: statistics.fmean(float(v) for v in run_values(sel_runs[a], ones, statistic, species, ap_fn)[0])
            for a in SELECTION_ARMS}
    full_y = statistics.fmean(float(v) for v in run_values(y_runs, ones, statistic, species, ap_fn)[0])
    T, bias, chosen = [], [], []
    for r in range(len(counts)):
        rows = [Vs[a][r] for a in SELECTION_ARMS] + [Vy[r]]
        if any(np.isnan(x).any() for x in rows):
            continue
        s = {a: statistics.fmean(float(v) for v in Vs[a][r]) for a in SELECTION_ARMS}
        best = max(s.values())
        c = next(a for a in SELECTION_ARMS if s[a] >= best - TIE_TOL)
        sy = statistics.fmean(float(v) for v in Vy[r])
        T.append(s[c] - sy)
        bias.append((s[c] - sy) - (full[c] - full_y))
        chosen.append(c)
    if len(T) < 2:
        raise V3famRefused("the selection correction kept %d resamples" % len(T))
    b = statistics.fmean(bias)
    return {"diff_sel": d_plain - b, "se_sel": _sd(T), "bias": b, "n_valid": len(T),
            "share_chosen": {a: chosen.count(a) / float(len(chosen)) for a in SELECTION_ARMS},
            "full_set": dict(full, Y=full_y), "T": T, "chosen": chosen,
            "score_values": {a: list(sel_values.get(a) or []) for a in SELECTION_ARMS}}


def _same_runs(a_runs, b_runs):
    """True when two lists of per-image arrays hold the same arrays, run by run."""
    import numpy as np
    if len(a_runs) != len(b_runs):
        return False
    for a, b in zip(a_runs, b_runs):
        if set(a) != set(b) or any(not np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a):
            return False
    return True


def comparison(cid, x_values, y_values, x_runs, y_runs, statistic="species", species=SC.SPECIES, sessions=None,
               se_rule="image", one_system=False, selection=None, keep=None, resamples=RESAMPLES, ap_fn=None,
               paired=False, image_seed_text=None):
    """One pre-declared comparison "X over Y" (V3F.3.3). x_values, y_values: per-seed full-set values (the score
    files' species_map50_95, a subset mean or P7's value); x_runs, y_runs: their per-image arrays (one exam, key
    order). se_rule image | max (P6-P9: the larger of the image and session SEs) | selection (P2, P3, P5:
    selection = {"runs": {M, A, B: [arrays]}, "values": {M, A, B: [...]}}, X being E3-B). sessions: one per image
    (the session SE, reported beside on dev; deciding with se_rule max). keep: the images the comparison is
    decided on (D0-S's dev minus H): every array is restricted and the full-set values recomputed from them.
    image_seed_text: a seed text the amendment fixes for this comparison (BCLEAN's rule: inc2/v3fam/bclean/
    species_se), in place of seed_text(cid, "image")."""
    if se_rule not in ("image", "max", "selection"):
        raise V3famRefused("se_rule %r" % se_rule)
    if keep is not None:
        allk = [str(k) for k in x_runs[0]["keys"]]
        keepl = [k for k in allk if k in set(str(k_) for k_ in keep)]
        if sessions is not None:
            sessions = [s for k, s in zip(allk, sessions) if k in set(keepl)]
        x_runs = [subset_arrays(a, keepl) for a in x_runs]
        y_runs = [subset_arrays(a, keepl) for a in y_runs]
        x_values = [statistic_value(a, statistic, species, ap_fn) for a in x_runs]
        y_values = [statistic_value(a, statistic, species, ap_fn) for a in y_runs]
        if selection:
            selection = {"runs": {a: [subset_arrays(r, keepl) for r in rs] for a, rs in selection["runs"].items()},
                         "values": {a: [statistic_value(subset_arrays(r, keepl), statistic, species, ap_fn)
                                        for r in rs] for a, rs in selection["runs"].items()}}
    d = statistics.fmean(float(v) for v in x_values) - statistics.fmean(float(v) for v in y_values)
    n = len(x_runs[0]["keys"])
    st_i = image_seed_text or seed_text(cid, "image")
    counts = image_counts(n, st_i, resamples)
    img = _bootstrap(x_runs, y_runs, counts, statistic, species, ap_fn)
    rec = {"id": cid, "statistic": statistic, "species": list(species) if statistic == "species" else None,
           "x_values": [float(v) for v in x_values], "y_values": [float(v) for v in y_values], "diff": d,
           "image": {"seed_text": st_i, "seed": C.stable_int(st_i), "resamples": int(resamples), "se": img["se"],
                     "n_valid": img["n_valid"]},
           "session": None, "selection": None, "se_rule": se_rule, "n_images": n,
           "restricted_to": None if keep is None else {"n_images": n, "keys_sha256": C.sha256_text(
               "\n".join(str(k) for k in x_runs[0]["keys"]))}}
    if sessions is not None:
        st_s = seed_text(cid, "session")
        ses = session_bootstrap(x_runs, y_runs, sessions, st_s, resamples, statistic, species, ap_fn)
        rec["session"] = {"seed_text": st_s, "seed": C.stable_int(st_s), "resamples": int(resamples),
                          "se": ses["se"], "n_valid": ses["n_valid"], "n_sessions": len(set(sessions)),
                          "deciding": se_rule == "max"}
    elif se_rule == "max":
        raise V3famRefused("%s decides on the larger of the image and session SEs: it needs the sessions" % cid)
    d_used, se_used = d, img["se"]
    if se_rule == "max":
        se_used = max(img["se"], rec["session"]["se"])
    if se_rule == "selection":
        if not selection:
            raise V3famRefused("%s is selection-corrected: it needs E3-M's, E3-A's and E3-B's arrays" % cid)
        if not _same_runs(selection["runs"].get(SELECTED) or [], x_runs):
            raise V3famRefused("%s: X of a selection-corrected comparison is E3-%s, whose arrays these are not"
                               % (cid, SELECTED))
        sc_ = selection_correct(selection["runs"], selection.get("values") or {}, y_runs, counts, d, statistic,
                                species, ap_fn, Vy=img["Vy"], VB=img["Vx"])
        rec["selection"] = sc_
        rec["plain"] = {"diff": d, "se": img["se"], "share_B_chosen": sc_["share_chosen"][SELECTED]}
        d_used, se_used = sc_["diff_sel"], sc_["se_sel"]
    rec["decision"] = decide(d_used, se_used, x_values, y_values, one_system=one_system)
    if paired and len(x_values) == len(y_values):
        rec["paired_per_seed"] = [float(a) - float(b) for a, b in zip(x_values, y_values)]
    return rec


# ------------------------------------------------------------------ Holm, Tier C, reads (V3F.3.4, V3F.3.6)
def holm_overlay(primaries):
    """{id: {"status": decided | not run | failed, "qualifies", "z"}} -> per primary its Holm rank, bar, survival
    and family-confirmed (qualifies and its z survives Holm's step-down within its exam; a primary not run keeps
    its place in m and is never confirmed)."""
    out = {}
    for exam, ids in HOLM_SETS.items():
        m = len(ids)

        def zkey(p):
            r = primaries.get(p) or {}
            z = r.get("z") if r.get("status") == "decided" else None
            return -(z if z is not None and math.isfinite(z) else -math.inf), ids.index(p)
        order = sorted(ids, key=zkey)
        alive = True
        for i, p in enumerate(order):
            r = primaries.get(p) or {}
            bar = statistics.NormalDist().inv_cdf(1.0 - ALPHA / (m - i))
            z = r.get("z") if r.get("status") == "decided" else None
            survives = alive and z is not None and z > bar
            alive = survives
            out[p] = {"exam": exam, "m": m, "rank": i + 1, "bar": bar, "z": z, "survives": survives,
                      "status": r.get("status") or "not run",
                      "family_confirmed": bool(survives and r.get("qualifies") and r.get("status") == "decided")}
    return out


def challenger_gate(results):
    """Tier C: {arm: {"status", "qualifies", "z", "diff"}} for the fourteen registered challengers (a missing one
    counts as not run). An arm opens a read when it qualifies over E3-B and its z exceeds Phi^-1(1 - 0.05/14);
    C is fixed at 14 whatever is run. Returns {C, bar, arms, opening (largest D first)}."""
    if len(CHALLENGERS) != C_CHALLENGERS:
        raise RuntimeError("the registry holds %d challengers, not %d" % (len(CHALLENGERS), C_CHALLENGERS))
    unknown = sorted(set(results) - set(CHALLENGERS))
    if unknown:
        raise V3famRefused("%s are not registered challengers" % unknown)
    arms = {}
    for a in CHALLENGERS:
        r = results.get(a) or {}
        st = r.get("status") or "not run"
        z = r.get("z")
        opens = bool(st == "decided" and r.get("qualifies") and z is not None and z > CHALLENGER_BAR)
        arms[a] = {"status": st, "qualifies": bool(r.get("qualifies")) if st == "decided" else None, "z": z,
                   "diff": r.get("diff"), "opens_read": opens}
    opening = sorted((a for a in CHALLENGERS if arms[a]["opens_read"]),
                     key=lambda a: (-float(arms[a]["diff"]), CHALLENGERS.index(a)))
    return {"C": C_CHALLENGERS, "bar": CHALLENGER_BAR, "arms": arms, "opening": opening}


def read_order(gate, bclean_triggered=False, reference_available=REFERENCE_READS):
    """The sealed-test reads in the order they are taken (V3F.3.6, decision 25): BCLEAN (if D0-S triggered it),
    the reference reads D6 then D8-m (those whose arm exists), then the challengers that open a read, largest D
    first, at most 3 (2 once BCLEAN takes a read); at most READ_CAP in all. A challenger whose arm has a
    reference read (D6) takes no challenger place, whether or not its reference read is in the order: D6's read
    is 'a reference read ... never a candidate' (V3F.3.6), so the challenger places go to the other arms that open
    a read. Every arm is read at most once."""
    order = []
    if bclean_triggered:
        order.append({"arm": BCLEAN_READ, "kind": "bclean"})
    order += [{"arm": a, "kind": "reference"} for a in REFERENCE_READS if a in reference_available]
    k = CHALLENGER_READS_BCLEAN if bclean_triggered else CHALLENGER_READS
    order += [{"arm": a, "kind": "challenger"} for a in [x for x in gate["opening"] if x not in REFERENCE_READS][:k]]
    names = [o["arm"] for o in order]
    if len(set(names)) != len(names):
        raise RuntimeError("the read order names an arm twice: %s" % names)
    if len(order) > READ_CAP:
        raise RuntimeError("the read order holds %d reads, over the cap of %d" % (len(order), READ_CAP))
    return order


def check_read(arm, order, done):
    """Refuse a read of arm unless it is in the order, has not been taken, and every read before it was taken or
    recorded not run (done: {arm: "read" | NOT_RUN_CAP | "not run"})."""
    names = [o["arm"] for o in order]
    if arm not in names:
        raise V3famRefused("%s does not open a sealed-test read (the reads are %s)" % (arm, names or "none"))
    if arm in done:
        raise V3famRefused("%s's read was %s already: a read is taken once" % (arm, done[arm]))
    i = names.index(arm)
    before = [a for a in names[:i] if a not in done]
    if before:
        raise V3famRefused("the reads are taken in order: %s first" % before)
    if sum(1 for v in done.values() if v == "read") >= READ_CAP:
        raise V3famRefused("the family's %d sealed-test reads are taken" % READ_CAP)
    return {"arm": arm, "position": i + 1, "kind": order[i]["kind"]}


def read_by_value(who):
    """The read_by field: exactly the owner's or the delegate's name (decision 20)."""
    if who not in READ_BY:
        raise V3famRefused("--read-by owner|delegate, not %r" % (who,))
    return READ_BY[who]


# ------------------------------------------------------------------ the family cap (decision 25)
def family_gpuh(items):
    """The family's GPU-h: each item at its estimate until every job it started has ended, then its settled
    V100 GPU-h (items: [{"item", "state": running | ended, "estimate", "settled"}])."""
    total = 0.0
    for it in items:
        st, est, got = it.get("state"), it.get("estimate"), it.get("settled")
        if st == "ended" and got is not None:
            total += float(got)
        elif st in ("running", "ended", "proposed"):
            if est is None:
                raise V3famRefused("item %s has no estimate" % it.get("item"))
            total += float(est)
        else:
            raise V3famRefused("item %s state %r" % (it.get("item"), st))
    return total


def check_cap(items, item, estimate, record=True):
    """Refuse (V3famCapRefused, "not run (family cap)") an item whose estimate would take the family past
    FAMILY_CAP_GPUH, recording it once under v3fam/v1/cap/; an item recorded so is never retried. Returns the
    spend before and after it."""
    p = cap_dir() / ("%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(item)))
    if p.exists():
        raise V3famCapRefused("%s was recorded %s: it is never retried" % (item, NOT_RUN_CAP))
    spent = family_gpuh(items)
    after = spent + float(estimate)
    if after > FAMILY_CAP_GPUH + CAP_EPS:
        if record:
            write_once(p, {"format": FMT_CAP, "item": item, "status": NOT_RUN_CAP, "spent_gpuh": spent,
                           "estimate_gpuh": float(estimate), "cap_gpuh": FAMILY_CAP_GPUH, "written_utc": _utc()})
        raise V3famCapRefused("%s (%.2f GPU-h) would take the family from %.2f past its cap of %.0f GPU-h: %s"
                              % (item, float(estimate), spent, FAMILY_CAP_GPUH, NOT_RUN_CAP))
    return {"spent_gpuh": spent, "after_gpuh": after, "cap_gpuh": FAMILY_CAP_GPUH}


# ------------------------------------------------------------------ CLI
# ------------------------------------------------------------------ the score verb (L23J)
def score_module(arm):
    """The module whose score_arm (or score) takes arm: the system arms (E3RS, D3, D6, D8), D2's nine and D9, the
    trained arms (D1, D4F, D4N, D4R15, D5B, BCLEAN)."""
    if arm not in SCORE_ARMS:
        raise V3famRefused("--arm %r is not one of the family's score arms %s" % (arm, list(SCORE_ARMS)))
    if arm in ("E3RS", "D3", "D6", "D8"):
        from . import v3fam_systems as M
        return M, M.score_arm
    if arm.startswith("D2") or arm == "D9":
        from . import v3fam_embed as M
        return M, M.score
    from . import v3fam_trained as M
    return M, M.score_arm


def score(arm, **kw):
    """inc2.v3fam score --arm X (L23J): the arm's scoring job, in its module."""
    _m, fn = score_module(arm)
    return fn(arm, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F: the v3 family's shared core (reproductions, EXIF pre-flight, "
                                             "the frozen check, a pass in another root).")
    ap.add_argument("command", choices=("equivalence", "exif-preflight", "exif-verdict", "frozen", "_pass", "score",
                                        "verdict", "test-read", "score-test", "test-report", "reanalyse-test",
                                        "spend"))
    ap.add_argument("--item", default=None, choices=EQUIV_ITEMS)
    ap.add_argument("--e3-arm", default=None)
    ap.add_argument("--exam", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--read-by", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--cap-refused", default=None,
                    help="verdict: the family items the ticker recorded 'not run (family cap)', comma-separated")
    ap.add_argument("--ticker-failed", default=None,
                    help="verdict: the score, build and D6 items the ticker recorded failed after their own second "
                         "failure, comma-separated (deviation 4)")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    kw = {"lock_check": not a.no_lock_check_for_tests, "batch": a.batch or S.BATCH, "device": a.device}
    try:
        if a.command == "frozen":
            print(json.dumps(check_frozen(), indent=1, sort_keys=True))
        elif a.command == "_pass":
            if not a.spec:
                raise V3famRefused("_pass needs --spec")
            _run_pass_spec(a.spec, batch=a.batch, device=a.device,
                           lock_check=False if a.no_lock_check_for_tests else None)
        elif a.command == "exif-preflight":
            exif_preflight()
        elif a.command == "exif-verdict":
            exif_verdict(a.frame, a.read_by)
        elif a.command == "score":
            score(a.arm, **kw)
        elif a.command in ("verdict", "test-read", "score-test", "test-report", "reanalyse-test", "spend"):
            from . import v3fam_verdict as VR
            if a.command == "verdict":
                VR.verdict(cap_refused=[x for x in (a.cap_refused or "").split(",") if x],
                           ticker_failed=[x for x in (a.ticker_failed or "").split(",") if x])
            elif a.command == "spend":
                print(json.dumps(VR.spend(), indent=1, sort_keys=True))
            elif a.command == "reanalyse-test":
                VR.reanalyse_test(a.read_by)
            elif not a.arm:
                raise V3famRefused("%s needs --arm" % a.command)
            elif a.command == "test-read":
                rec = VR.test_read(a.arm, a.read_by)
                print(" ".join(rec["argv"]))
            elif a.command == "score-test":
                VR.score_test(a.arm, **kw)
            else:
                VR.test_report(a.arm)
        elif a.item == "dev_identity":
            equivalence_dev_identity(**kw)
        elif a.item == "two_stage":
            equivalence_two_stage(a.e3_arm, V.BioclipEmbedder(), **kw)
        elif a.item == "identity":
            if not a.exam or not a.arm:
                raise V3famRefused("--item identity needs --exam and --arm")
            equivalence_identity(a.exam, a.arm, **kw)
        elif a.item == "injection":
            equivalence_injection(**kw)
        else:
            raise V3famRefused("equivalence needs --item %s" % "|".join(EQUIV_ITEMS))
    except (V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError,
            C2.Inc2Error, ValueError) as e:
        print("[inc2.v3fam] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
