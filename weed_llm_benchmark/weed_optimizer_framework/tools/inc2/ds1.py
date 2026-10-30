"""DS1: scaling verified training data at the weed level (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-29): DS1,
scaling verified training data at the weed level (pre-registered)", kept byte for byte as
docs/prereg/DS1_2026-10-29.md). The family's core: its constants, records, the DS1 pass on E3's validator class, the
reproductions, the statistics and decisions, the power pre-flight, the re-analysis of the stream's gate records
(RA), the label-noise statistic (LN), the MT-W check rule, the sealed reads and the family cap.

    python -m weed_optimizer_framework.tools.inc2.ds1 score --arm ARM           (L33; one arm's passes)
    python -m weed_optimizer_framework.tools.inc2.ds1 verdict --phase 1|2 [--cap-refused I,..] [--ticker-failed I,..]
                                                                                 (L34)
    python -m weed_optimizer_framework.tools.inc2.ds1 ra [--report PATH ...]    (the delegate's, no GPU: DS.8.1)
    python -m weed_optimizer_framework.tools.inc2.ds1 ln                        (after the LN sheet is recorded)
    python -m weed_optimizer_framework.tools.inc2.ds1 test-read --arm X --exam mtw|cwd12 --read-by owner|delegate
    python -m weed_optimizer_framework.tools.inc2.ds1 score-test --arm X --exam mtw|cwd12   (the job test-read prints)
    python -m weed_optimizer_framework.tools.inc2.ds1 test-report --arm X --exam mtw|cwd12
    python -m weed_optimizer_framework.tools.inc2.ds1 timestamp --proof FILE --stamp-proof FILE
                                    --revision-proof FILE --revision-stamp-proof FILE   (the delegate's; per Revision)
    python -m weed_optimizer_framework.tools.inc2.ds1 timestamp-revision --revision N --proof FILE --stamp-proof FILE
                                    (the delegate's: a Revision made after timestamp.json was written)
    python -m weed_optimizer_framework.tools.inc2.ds1 spend                     (read-only: the family's GPU-h)
    python -m weed_optimizer_framework.tools.inc2.ds1 _pass --spec SPEC.json    (internal: a pass in an MD root)

What it rests on. inc2/v3fam.py and the six E3-frozen files are imported and never edited (DS.7.1): DS1's pass
(ds1_pass) installs a validator built on inc2.v3fam.family_validator_class (itself built on E3's validator class) with
one layer of its own (_ds1_layer) that keeps, per image, the class-collapsed matches of the stage-1 box set against
every ground-truth box: the weed arrays (keys, tp, conf, pred_img, n_gt), which hold the stage-1 helper's records
row for row and, put back in the validator's image order, give the pass's stage-1 agnostic AP (inc.scorer.collapsed_ap)
exactly (weed_check; Revision 2, item 1: equal confidences rank by input order, so the check is never made across two
orders). The statistics are V3F's array
functions (inc2.v3fam.image_counts, session_counts, run_values, decide) under DS1's own seed texts
(inc2/ds1/<comparison id>/image | group | session).

Exams (DS.4) and where they are scored:
  dev        cwd12 dev (LOCK v1): two-stage species mAP50-95 (stage-1 boxes and E3's pinned classifier, the
             V3F validator) and the stage-1 agnostic AP; image SE (session SE reported); N1, H1 and Tier S
  md_s0      MD-S0 in its own root (splits/ds1/md_v1/root, a subprocess with INC_DIR there, as Z1 scores its root):
             weed AP50-95, the class-collapsed AP of the stage-1 boxes; SE = max(image, capture-group bootstrap);
             decides P1-P3
  mdn_<s>    MD-N of each new source, in the same root: coverage only
  ood23      target-species agnostic AP with OtherPlant boxes as ignore regions (V3F P7's statistic), under deviation
             6's loader; SE = max(image, session); decides P4 (S arms only); the 1,771-image reading beside it
  ood22      agnostic, reported (S arms only); imageweeds agnostic, reported
  mtw, cwd12 sealed reads only (test-read, score-test; DS.7.5)

Records (each written once):
  ds1/v1/timestamp.json                 the prereg's sha256 and its proofs (the delegate's, DS.12 step 0a)
  ds1/v1/timestamp_rev<n>.json          a later Revision's sha256 and proofs (timestamp_revision; Revision 2 on)
  ds1/v1/equivalence/<item>.json        the three reproductions (DS.7.1): e1a_dev, e3a_dev, b1_ood23, md_identity
  ds1/v1/arms/<ARM>/s<k>/<exam>.{json, images.npz, stage1.npz, rows.npz, p7.npz, weed.npz}
  ds1/v1/reported/<ARM>.json            the non-deciding passes' files and statuses
  capacity/ds1_score_<arm>.json         the platform's record of a score job (dev and MD only)
  capacity/ds1_power.json               the power pre-flight (DS.7.6), before the first training item
  capacity/ds1_ra_gate.{json,md}        RA (DS.8.1)
  capacity/ds1_ln.json                  LN's statistic (DS.8.2), from the recorded sheet
  capacity/ds1_v1.json, ds1_v2.json     the phase verdicts (dev and MD only); ds1_v{1,2}_external.json the rest
  ds1/v1/reads/<n>_<arm>_<exam>/read.json, capacity/ds1_test_<arm>_<exam>.{json,md}
Every DS1 job records the sha256 tree of twostage/e3_v1 and capacity/{e*,v3fam}_* before and after it runs and fails
if either changed (trees_unchanged); it writes only under ds1/, splits/ds1/ and capacity/ds1_*.

A refusal raises Ds1Refused (Ds1CapRefused for the family cap); the CLI prints "[inc2.ds1] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
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
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F

# ------------------------------------------------------------------ the registration (DS.12 step 0a)
NAME = "v1"
PREREG_FILE = "docs/prereg/DS1_2026-10-29.md"
PREREG_SHA256 = "3b63933bbb134327a0491fc65d2457434663665674e2aa3464adaf5291d49699"
PREREG_COMMIT = "b60b8ee4763867f03b37f09a35ae808b630f5d28"
STAMP_COMMIT = "972b3eb92997c2235ebe52d17f0d37866ed74959"
STAMP_FILE = "docs/prereg/DS1_2026-10-29.stamp.txt"
# Revisions made before any DS1 number (docs/CONTINUOUS_LOOP.md, "Revision (2026-10-29): DS1 Revision 1" and "...
# DS1 Revision 2"), each kept byte for byte under docs/prereg/ and time-stamped as the amendment was (V3F.12); no DS1
# job runs before ds1/v1/timestamp.json names each with its proofs, or (a Revision made after it) its own record
# ds1/v1/timestamp_rev<n>.json does
REVISIONS = ({"n": 1, "file": "docs/prereg/DS1_rev1_2026-10-29.md",
              "sha256": "543ca520485f46bea0beee74758c608450d75b5860e8f4465b04cbd3fee531b7",
              "commit": "eb0835695680b4cefa9c3c0cdd94e3970425eaad",
              "stamp_file": "docs/prereg/DS1_rev1_2026-10-29.stamp.txt"},
             # Revision 2 (2026-10-29), before any DS1 training or score; named by ds1/v1/timestamp_rev2.json. Its
             # second text (the first, b9c73500...bb13 of commit f022a0b, never deployed, is kept under
             # docs/prereg/superseded/DS1_rev2_first_text/)
             {"n": 2, "file": "docs/prereg/DS1_rev2_2026-10-29.md",
              "sha256": "5970fd759df8e291f645caff1d3da4ee540972da33f86c0a7248c44ff8760fc4",
              "commit": "9ca7e4d321bf11dc4abba35ed920fb27ecb9d8bc",
              "stamp_file": "docs/prereg/DS1_rev2_2026-10-29.stamp.txt"})
DELEGATE = "the delegate under the 2026-09-30 owner grant"
OWNER = "Harry He (owner)"
DECIDED_BY = ("docs/CONTINUOUS_LOOP.md, Amendment (2026-10-29): DS1, scaling verified training data at the weed level "
              "(pre-registered); commit %s; %s sha256 %s" % (PREREG_COMMIT[:7], PREREG_FILE, PREREG_SHA256))
READ_BY = {"owner": OWNER, "delegate": DELEGATE}

# ------------------------------------------------------------------ statistics (DS.7.1, V3F's)
SEEDS = F.SEEDS
RESAMPLES = F.RESAMPLES
SESOI = F.SESOI                               # 0.005 on every exam
ALPHA = F.ALPHA                               # one-sided 5 %
Z90 = F.Z90
Z_POWER = statistics.NormalDist().inv_cdf(0.8)                    # 0.8416: power 0.8
SEED_TEXT = "inc2/ds1/%s/%s"                                      # <comparison id>/<image | group | session>
CLUSTER_KINDS = ("group", "session")
# Bonferroni within each exam, one-sided (DS.0 item 8): P1-P3 on MD-S0 (m = 3), P4 on ood23 (m = 1)
EXAM_M = {"md_s0": 3, "ood23": 1}
THRESHOLD = {e: statistics.NormalDist().inv_cdf(1.0 - ALPHA / m) for e, m in EXAM_M.items()}   # 2.128, 1.645
PRIMARIES = {
    "P1": {"exam": "md_s0", "x": "S1C", "y": "S0", "phase": 1, "se": "max(image, group)",
           "question": "Do the in-hand pools, quality-capped, place better weed boxes on held-out scenes of S0's own "
                       "sources, at equal compute?"},
    "P2": {"exam": "md_s0", "x": "S1U", "y": "S1C", "phase": 1, "se": "max(image, group)",
           "question": "Do more thinned greenhouse frames help on top of them, at equal compute?"},
    "P3": {"exam": "md_s0", "x": "S2X", "y": "S0X", "phase": 2, "se": "max(image, group)",
           "question": "Does the phase-2 build beat base B' at the pair's compute?"},
    "P4": {"exam": "ood23", "x": "S2X", "y": "S0X", "phase": 2, "se": "max(image, session)",
           "question": "Does the scaled build transfer to a held-out season and region? (V3F P7's statistic)"}}
N1_MARGIN = 0.005                             # non-inferiority on cwd12 dev species, outside the correction
H1_Z = -statistics.NormalDist().inv_cdf(1.0 - ALPHA)               # -1.645: the harm check
# DS.0 item 17: the substitution of S1u for S2 when S2's manifest is not frozen by this date; reads by READS_BY
S2_FREEZE_BY = "2026-12-05"
READS_BY = "2026-12-20"
SU_END = "2026-12-31"
SUBSTITUTE = {"S2": "S1U", "S2X": "S1UX"}
# ood23's 13 images of disputed season (DS.4): P4 is decided on all 1,784, the same statistic on 1,771 beside it
OOD23_DISPUTED = tuple("ood23__20230810_HTRC_iPhone13_BD_%d" % i for i in range(26, 39))
OOD23_IMAGES = 1784
# V3F P7's sigma, derived from its interval in V3F-R (D +0.0423, z 2.61; DS.7.6): ood23's MDE
P7_SIGMA = 0.0423 / 2.61
MDE_MAX = 0.02                                # MD-S0 is enlarged before its freeze when its MDE is above this
CLEAR_090 = 0.9000

# ------------------------------------------------------------------ arms (DS.6) and exams (DS.4)
BUILD_ARMS = ("KD", "KT", "KF", "S0", "S1CR", "S1C", "S1U", "S2")          # L32: inc2.baseline build --ds1 at 1.2 M
LONG_ARMS = ("S0X", "S2X", "S1UX")                                         # L32X: inc2.ds1_long at the pair's compute
LONG_PARENT = {"S0X": "S0", "S2X": "S2", "S1UX": "S1U"}                     # the manifest each long arm trains
SCORE_ARMS = ("POWER", "K0") + BUILD_ARMS + LONG_ARMS                       # L33
EXP_OF = {"K0": "e1_a_m640", "KD": "ds1_kd_m640", "KT": "ds1_kt_m640", "KF": "ds1_kf_m640", "S0": "ds1_s0_m640",
          "S1CR": "ds1_s1cr_m640", "S1C": "ds1_s1c_m640", "S1U": "ds1_s1u_m640", "S2": "ds1_s2_m640"}
POWER_EXPS = {"E1A": "e1_a_m640", "BV2": "b_v2_m640", "E1B": "e1_b_m640"}
B1_EXP = "v3_d1_b1_m640"                      # D1's B'1, whose ood23 statistic is reproduced (DS.7.1)
B1_OOD23 = (0.2562, 0.2517, 0.2598)           # B'1's ood23 statistic per seed, as V3F-R records it (4 decimals)
K_ARMS = ("KD", "KT", "KF")
S_ARMS = ("S0", "S1CR", "S1C", "S1U", "S2") + LONG_ARMS
SEASON_ARMS = ("S0", "S1C", "S1U", "S2") + LONG_ARMS                       # season-free: ood22 and ood23 are exams
ARM_EXAMS = {"POWER": ("md_s0", "dev"), "K0": ("dev", "md_s0", "imageweeds"),
             "KD": ("dev", "md_s0", "imageweeds"), "KT": ("dev", "md_s0", "imageweeds"),
             "KF": ("dev", "md_s0", "imageweeds"), "S1CR": ("md_s0", "dev")}
for _a in SEASON_ARMS:
    ARM_EXAMS[_a] = ("md_s0", "mdn", "dev", "ood23", "ood22", "imageweeds")
DECIDING_EXAMS = ("md_s0", "dev", "ood23")
PLATFORM_EXAMS = ("dev", "md_s0", "mdn")      # what capacity/ds1_score_<arm>.json and the verdicts' records name
NMS = TS.NMS["B"]                             # one-class detectors' boxes, taken as E3-B's (locked NMS)
NMS_REF = "B"

# ------------------------------------------------------------------ compute (DS.6, DS.11)
BUDGET_IMAGE_EPOCHS = 1200000
PAIR_IMAGE_EPOCHS = 3600000
PAIR_FALLBACK_IMAGE_EPOCHS = 2400000
COLD_WALLTIME_H = 8.0
LONG_WALLTIME_H = 24.0
RESUBMIT_WALLTIME_H = 36.0
WALLTIME_SHARE = 0.8                          # a run is projected within 0.8 x its walltime
FAMILY_CAP_GPUH = 260.0
PHASE_RESERVE_GPUH = 85.8                     # phase 2 and reads; phase-1 items may not take the family past 174.2
PHASE1_CEILING_GPUH = FAMILY_CAP_GPUH - PHASE_RESERVE_GPUH
CAP_EPS = F.CAP_EPS
NOT_RUN_CAP = "not run (family cap)"
# cut last when the cap binds (DS.11): the challenger read, S2 at 1.2 M, S1c-R
CUT_ORDER = ("read:challenger", "build:S2", "build:S1CR")
# ------------------------------------------------------------------ storage (DS.0 item 21)
PHASE1_BYTES_GB = 60.0
GIB = 1.073741824                             # GB per GiB
V3F_FLOOR = {"share": 0.03, "margin_gib": 30.0}         # 257.8 GB free on the 6.836 TiB quota
DATA_FLOOR = {"share": 0.03, "margin_gb": 200.0}        # lane DATA's: 425.6 GB free

# ------------------------------------------------------------------ reads (DS.7.5)
READ_CAP = 4
READ_KINDS = ("reference", "candidate", "challenger", "cwd12")

# ------------------------------------------------------------------ reproductions (DS.7.1)
REPRO_TOL = F.REPRO_TOL                       # 1e-6
DEVIATION_TOL = F.DEVIATION_TOL               # 0.002
REPRO_ITEMS = ("e1a_dev", "e3a_dev", "b1_ood23", "md_identity")
# The weed arrays against the pass's stage-1 agnostic AP (Revision 2, item 1; E1's rule, inc2.scorer_agnostic):
# ap_per_class ranks equal confidences by an unstable sort, so the AP of tied arrays depends on their order and on the
# CPU's sort kernel. The identity is checked in the validator's own image order (the same rows in the same order:
# exact); the tie-broken arrays in exam key order (what the statistics read) only within the ties' reach.
WEED_RECOMPUTE_TOL = 1e-9                     # scorer_agnostic.MAX_RECOMPUTE_DIFF
WEED_TIE_TOL = 0.02                           # scorer_agnostic.MAX_TIE_DIFF

FMT_SCORE = "inc2-ds1-score/1"
FMT_SCORE_RECORD = "inc2-ds1-score-record/1"
FMT_EQUIV = "inc2-ds1-equivalence/1"
FMT_POWER = "inc2-ds1-power/1"
FMT_RA = "inc2-ds1-ra/1"
FMT_LN = "inc2-ds1-ln/1"
FMT_VERDICT = "inc2-ds1-verdict/1"
FMT_EXTERNAL = "inc2-ds1-external/1"
FMT_READ = "inc2-ds1-read/1"
FMT_TEST = "inc2-ds1-test/1"
FMT_TIMESTAMP = "inc2-ds1-timestamp/1"
FMT_REV_TIMESTAMP = "inc2-ds1-revision-timestamp/1"
FMT_PASS_SPEC = "inc2-ds1-pass-spec/1"
FMT_REPORTED = "inc2-ds1-reported/1"
STAMP = "DS1-"

_TEST_TOKENS = []                             # score-test's tokens: the only callers that may read a sealed exam
_IN_ROOT = {}                                 # set by the _pass verb in an MD root's subprocess


class Ds1Refused(RuntimeError):
    """What was asked is not DS1's pre-registered measurement."""


class Ds1CapRefused(Ds1Refused):
    """The item would take the family past its GPU-h cap or the phase reserve: not run (family cap)."""


def log(msg):
    print("[inc2.ds1] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths and records
def root():
    return C.INC_DIR / "ds1" / NAME


def splits_dir():
    return C.INC_DIR / "splits" / "ds1"


def capacity(name):
    return C.INC_DIR / "capacity" / name


def equiv_dir():
    return root() / "equivalence"


def arm_dir(arm, seed):
    return root() / "arms" / arm / ("s%d" % int(seed))


def md_root(exam="md_s0"):
    """The frozen scoring root of an MD exam (Z1's pattern: splits/v1/<exam>.jsonl, splits/v1/LOCK.json,
    exams/v1/<exam>/; each exam its own root and LOCK), from its frozen.json."""
    d = _read_json(splits_dir() / "md_v1" / exam / "frozen.json")
    if not isinstance(d, dict) or not d.get("root"):
        raise Ds1Refused("%s is not frozen" % exam)
    return Path(d["root"])


def timestamp_path():
    return root() / "timestamp.json"


def _utc():
    return TS._utc()


def _read_json(path):
    return TS._read_json(path)


def _sha(path):
    return TS._sha(path)


def write_once(path, data):
    """Write bytes or a JSON object once (twostage._write_once)."""
    try:
        return TS._write_once(path, data)
    except TS.E3Refused as e:
        raise Ds1Refused(str(e))


def write_json(path, obj):
    TS._write_json(path, obj)


def code_shas():
    """The six E3-frozen files', inc2/v3fam.py's and this module's sha256s: the code a DS1 pass runs."""
    out = F.code_shas()
    out["ds1"] = C.sha256_file(Path(__file__).resolve())
    return out


def check_frozen():
    try:
        return F.check_frozen()
    except F.V3famRefused as e:
        raise Ds1Refused(str(e))


def today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ the time stamp (DS.9.3)
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"


def ots_digest(path):
    """The sha256 an OpenTimestamps proof stamps (its header: the magic, version 1, the sha256 op 0x08 and the 32-byte
    digest), or None when the file is not such a proof."""
    try:
        with open(str(path), "rb") as fh:
            b = fh.read(len(OTS_MAGIC) + 34)
    except OSError:
        return None
    if len(b) < len(OTS_MAGIC) + 34 or not b.startswith(OTS_MAGIC) or b[len(OTS_MAGIC):len(OTS_MAGIC) + 2] != b"\x01\x08":
        return None
    return b[len(OTS_MAGIC) + 2:].hex()


def _proof_pair(proof, stamp_proof, sha256, commit, what):
    """The checked proof pair of one stamped text: the proof stamps the text's sha256, the stamp proof the stamp file
    beside it, which names the text's sha256 and its commit."""
    for p in (proof, stamp_proof):
        if not Path(p).is_file():
            raise Ds1Refused("%s does not exist" % p)
    if ots_digest(proof) != sha256:
        raise Ds1Refused("%s does not stamp %s's sha256 %s (it stamps %s)" % (proof, what, sha256[:12],
                                                                             str(ots_digest(proof))[:12]))
    stamp = Path(str(stamp_proof)[:-len(".ots")]) if str(stamp_proof).endswith(".ots") else None
    if stamp is None or not stamp.is_file() or ots_digest(stamp_proof) != C.sha256_file(stamp):
        raise Ds1Refused("%s does not stamp the stamp file beside it (%s)" % (stamp_proof, stamp))
    text = stamp.read_text(encoding="utf-8", errors="replace")
    if sha256 not in text or commit not in text:
        raise Ds1Refused("%s does not name %s's sha256 and commit" % (stamp, what))
    return [{"file": Path(proof).name, "sha256": C.sha256_file(proof), "stamps": ots_digest(proof)},
            {"file": Path(stamp_proof).name, "sha256": C.sha256_file(stamp_proof), "stamps": ots_digest(stamp_proof),
             "stamp_file_sha256": C.sha256_file(stamp)}]


def revision_stamp_path(n, ts_path=None):
    """ds1/v1/timestamp_rev<n>.json beside the time-stamp record: a Revision made after the record was written (it is
    written once) is named by its own record (timestamp_revision)."""
    return Path(ts_path or timestamp_path()).parent / ("timestamp_rev%d.json" % int(n))


def _two_proofs(r):
    return len([p for p in (r or {}).get("proofs") or [] if isinstance(p, dict) and p.get("sha256")]) == 2


def revision_named(rv, d, ts_path=None):
    """True when Revision rv (an entry of REVISIONS) is named by its sha256 with two proofs: in the time-stamp record d,
    or (a Revision made after it) in its own record beside it (revision_stamp_path)."""
    got = {r.get("sha256"): r for r in (d or {}).get("revisions") or [] if isinstance(r, dict)} \
        if isinstance(d, dict) else {}
    if rv["sha256"] in got and _two_proofs(got[rv["sha256"]]):
        return True
    sep = _read_json(revision_stamp_path(rv["n"], ts_path))
    return bool(isinstance(sep, dict) and sep.get("format") == FMT_REV_TIMESTAMP and sep.get("sha256") == rv["sha256"]
                and sep.get("prereg_sha256") == PREREG_SHA256 and _two_proofs(sep))


def revisions_ok(d, ts_path=None):
    """True when every Revision (REVISIONS) is named (revision_named)."""
    return all(revision_named(rv, d, ts_path) for rv in REVISIONS)


def timestamp_missing(path=None):
    """What the time stamp still lacks, for the ticker's card (it names only these steps, each Revision by its own
    number): {"record": True when ds1/v1/timestamp.json names the prereg's sha256 with two proofs (it is written
    once), "revisions": [the number of each Revision named neither by it nor by its own record]}."""
    p = Path(path or timestamp_path())
    d = _read_json(p)
    rec = bool(isinstance(d, dict) and d.get("format") == FMT_TIMESTAMP and d.get("sha256") == PREREG_SHA256
               and _two_proofs(d))
    return {"record": rec, "revisions": [int(rv["n"]) for rv in REVISIONS if not revision_named(rv, d, p)]}


def timestamp_record(proof, stamp_proof, sha256=None, revisions=None):
    """ds1/v1/timestamp.json (the delegate's, DS.12 step 0a): the prereg's sha256 and its two proofs' names and
    sha256s, and each Revision's (REVISIONS: revisions [(proof, stamp proof)] in their order), written once. The
    proofs are the committed .ots files (docs/prereg/DS1_2026-10-29.md.ots and its stamp file's): the first must
    stamp the prereg's sha256, the second the stamp file beside it, which names the prereg's sha256 and commit
    (V3F.12's procedure); a Revision's likewise. The record names what each proof stamps, which `ots upgrade` never
    changes (an upgraded proof's own sha256 differs from the one recorded)."""
    revisions = list(revisions or [])
    if len(revisions) != len(REVISIONS):
        raise Ds1Refused("the time stamp names %d Revision proof pair(s); DS1 has %d Revision(s): %s" % (
            len(revisions), len(REVISIONS), [r["file"] for r in REVISIONS]))
    rev_docs = []
    for rv, (rp, rsp) in zip(REVISIONS, revisions):
        rev_docs.append({"file": rv["file"], "sha256": rv["sha256"], "commit": rv["commit"],
                         "proofs": _proof_pair(rp, rsp, rv["sha256"], rv["commit"], rv["file"])})
    for p in (proof, stamp_proof):
        if not Path(p).is_file():
            raise Ds1Refused("%s does not exist" % p)
    if ots_digest(proof) != PREREG_SHA256:
        raise Ds1Refused("%s does not stamp the prereg's sha256 %s (it stamps %s)" % (
            proof, PREREG_SHA256[:12], str(ots_digest(proof))[:12]))
    stamp = Path(str(stamp_proof)[:-len(".ots")]) if str(stamp_proof).endswith(".ots") else None
    if stamp is None or not stamp.is_file() or ots_digest(stamp_proof) != C.sha256_file(stamp):
        raise Ds1Refused("%s does not stamp the stamp file beside it (%s)" % (stamp_proof, stamp))
    text = stamp.read_text(encoding="utf-8", errors="replace")
    if PREREG_SHA256 not in text or PREREG_COMMIT not in text:
        raise Ds1Refused("%s does not name the prereg's sha256 and commit" % stamp)
    doc = {"format": FMT_TIMESTAMP, "prereg": PREREG_FILE, "sha256": sha256 or PREREG_SHA256,
           "commit": PREREG_COMMIT, "stamp_commit": STAMP_COMMIT,
           "proofs": [{"file": Path(proof).name, "sha256": C.sha256_file(proof), "stamps": ots_digest(proof)},
                      {"file": Path(stamp_proof).name, "sha256": C.sha256_file(stamp_proof),
                       "stamps": ots_digest(stamp_proof), "stamp_file_sha256": C.sha256_file(stamp)}],
           "revisions": rev_docs, "written_utc": _utc()}
    if doc["sha256"] != PREREG_SHA256:
        raise Ds1Refused("the time stamp names %s, the prereg is %s" % (doc["sha256"][:12], PREREG_SHA256[:12]))
    write_once(timestamp_path(), doc)
    return doc


def timestamp_revision(n, proof, stamp_proof):
    """ds1/v1/timestamp_rev<n>.json (the delegate's, written once): Revision n (REVISIONS) made after the time-stamp
    record was written, named by its sha256 and commit with its two proofs (the first stamps the Revision's byte copy,
    the second the stamp file beside it, which names the Revision's sha256 and commit; V3F.12's procedure), and the
    record it follows. Until it exists no DS1 job runs and the gate stays closed (timestamp_ok)."""
    rv = next((r for r in REVISIONS if int(r["n"]) == int(n)), None)
    if rv is None:
        raise Ds1Refused("DS1 has no Revision %s: its Revisions are %s" % (n, [r["n"] for r in REVISIONS]))
    d = _read_json(timestamp_path())
    if not (isinstance(d, dict) and d.get("format") == FMT_TIMESTAMP and d.get("sha256") == PREREG_SHA256):
        raise Ds1Refused("%s does not name the prereg: the amendment's time stamp comes first" % timestamp_path())
    if any(isinstance(r, dict) and r.get("sha256") == rv["sha256"] for r in d.get("revisions") or []):
        raise Ds1Refused("Revision %s is named by %s already" % (n, timestamp_path()))
    doc = {"format": FMT_REV_TIMESTAMP, "revision": int(n), "file": rv["file"], "sha256": rv["sha256"],
           "commit": rv["commit"], "prereg_sha256": PREREG_SHA256, "follows": {
               "file": timestamp_path().name, "sha256": C.sha256_file(timestamp_path())},
           "proofs": _proof_pair(proof, stamp_proof, rv["sha256"], rv["commit"], rv["file"]), "written_utc": _utc()}
    write_once(revision_stamp_path(n), doc)
    return doc


def timestamp_ok(path=None):
    """True when ds1/v1/timestamp.json names this text's sha256 and two proofs (DS.9.3's first condition)."""
    d = _read_json(path or timestamp_path())
    return bool(isinstance(d, dict) and d.get("format") == FMT_TIMESTAMP and d.get("sha256") == PREREG_SHA256
                and len([p for p in d.get("proofs") or [] if isinstance(p, dict) and p.get("sha256")]) == 2
                and revisions_ok(d, path or timestamp_path()))


def require_timestamp():
    if not timestamp_ok():
        raise Ds1Refused("%s does not name the prereg's sha256 %s and its proofs: no DS1 job runs before the "
                         "registration's time stamp (DS.9.3)" % (timestamp_path(), PREREG_SHA256[:12]))
    return C.sha256_file(timestamp_path())


# ------------------------------------------------------------------ write isolation (DS.9.1)
def trees():
    """{relative path: sha256} of twostage/e3_v1 and of capacity/e*_* and capacity/v3fam_* (DS.9.1)."""
    out = {}
    tw = C.INC_DIR / "twostage" / TS.E3_NAME
    if tw.is_dir():
        for p in sorted(tw.rglob("*")):
            if p.is_file() and not p.is_symlink():
                out[str(p.relative_to(C.INC_DIR))] = C.sha256_file(p)
    cap = C.INC_DIR / "capacity"
    if cap.is_dir():
        for p in sorted(cap.iterdir()):
            if p.is_file() and (re.match(r"e[0-9]", p.name) or p.name.startswith("v3fam")):
                out[str(p.relative_to(C.INC_DIR))] = C.sha256_file(p)
    return out


@contextlib.contextmanager
def trees_unchanged():
    """The E-series and V3F trees before and after the block: a DS1 job fails if either changed."""
    before = trees()
    yield before
    after = trees()
    if after != before:
        diff = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise Ds1Refused("the E-series or V3F records changed while this DS1 job ran (%s): DS1 writes only under "
                         "ds1/, splits/ds1/ and capacity/ds1_*" % diff[:5])


# ------------------------------------------------------------------ statistics (DS.7.1)
def seed_text(cid, kind):
    if kind not in ("image",) + CLUSTER_KINDS:
        raise Ds1Refused("a DS1 bootstrap is drawn over images, groups or sessions, not %r" % kind)
    return SEED_TEXT % (cid, kind)


def subset_runs(runs, keep):
    """Every run's arrays restricted to the images of `keep` (inc2.v3fam.subset_arrays)."""
    return [F.subset_arrays(a, keep) for a in runs]


def comparison(cid, x_values, y_values, x_runs, y_runs, statistic="agnostic", clusters=None, cluster_kind=None,
               se_rule="image", keep=None, resamples=None, ap_fn=None, species=SC.SPECIES):
    """One pre-declared DS1 comparison "X over Y" (DS.7.1): D = mean(X) - mean(Y) over the seeds' full-set values;
    the paired image bootstrap (seed text inc2/ds1/<cid>/image) and, with clusters (one label per image in the
    arrays' order), the cluster bootstrap (inc2/ds1/<cid>/group on MD, /session on ood23); SE = max(image,
    cluster) when se_rule is 'max', else the image SE (the cluster SE reported); the E-rule, combined z, the 90 %
    interval and the outcome class (inc2.v3fam.decide). keep: the images the comparison is decided on (ood23's
    1,771): every array is restricted and the full-set values recomputed."""
    resamples = RESAMPLES if resamples is None else int(resamples)
    if se_rule not in ("image", "max"):
        raise Ds1Refused("se_rule %r" % se_rule)
    if clusters is not None and cluster_kind not in CLUSTER_KINDS:
        raise Ds1Refused("a cluster bootstrap names its kind: %s" % list(CLUSTER_KINDS))
    if se_rule == "max" and clusters is None:
        raise Ds1Refused("%s decides on the larger of the image and %s SEs: it needs the clusters"
                         % (cid, cluster_kind or "cluster"))
    if keep is not None:
        allk = [str(k) for k in x_runs[0]["keys"]]
        ks = set(str(k) for k in keep)
        if clusters is not None:
            clusters = [c for k, c in zip(allk, clusters) if k in ks]
        x_runs, y_runs = subset_runs(x_runs, ks), subset_runs(y_runs, ks)
        x_values = [F.statistic_value(a, statistic, species, ap_fn) for a in x_runs]
        y_values = [F.statistic_value(a, statistic, species, ap_fn) for a in y_runs]
    if len(x_values) != len(SEEDS) or len(y_values) != len(SEEDS):
        raise Ds1Refused("%s: %d and %d seeds, not %d per side" % (cid, len(x_values), len(y_values), len(SEEDS)))
    d = statistics.fmean(float(v) for v in x_values) - statistics.fmean(float(v) for v in y_values)
    n = len(x_runs[0]["keys"])
    st_i = seed_text(cid, "image")
    img = F._bootstrap(x_runs, y_runs, F.image_counts(n, st_i, resamples), statistic, species, ap_fn)
    rec = {"id": cid, "statistic": statistic, "x_values": [float(v) for v in x_values],
           "y_values": [float(v) for v in y_values], "diff": d, "n_images": n, "se_rule": se_rule,
           "image": {"seed_text": st_i, "seed": C.stable_int(st_i), "resamples": int(resamples), "se": img["se"],
                     "n_valid": img["n_valid"]}, "cluster": None,
           "restricted_to": None if keep is None else {"n_images": n, "keys_sha256": C.sha256_text(
               "\n".join(str(k) for k in x_runs[0]["keys"]))}}
    if clusters is not None:
        if len(clusters) != n:
            raise Ds1Refused("%s: %d cluster labels for %d images" % (cid, len(clusters), n))
        st_c = seed_text(cid, cluster_kind)
        cl = F._bootstrap(x_runs, y_runs, F.session_counts(list(clusters), st_c, resamples), statistic, species,
                          ap_fn)
        rec["cluster"] = {"kind": cluster_kind, "seed_text": st_c, "seed": C.stable_int(st_c),
                          "resamples": int(resamples), "se": cl["se"], "n_valid": cl["n_valid"],
                          "n_clusters": len(set(clusters)), "deciding": se_rule == "max"}
    se = img["se"]
    if se_rule == "max":
        se = max(img["se"], rec["cluster"]["se"])
    rec["decision"] = F.decide(d, se, x_values, y_values)
    return rec


def confirmed(rec, exam):
    """DS.7.2: a primary is confirmed when it qualifies under the E-rule and its z passes its exam's Bonferroni
    threshold (P1-P3 on MD-S0: z > 2.128; P4 on ood23: z > 1.645)."""
    d = (rec or {}).get("decision") or {}
    z = d.get("z")
    return bool(d.get("qualifies") and z is not None and math.isfinite(z) and z > THRESHOLD[exam])


def primary_reading(pid, rec):
    """{status, qualifies, z, threshold, confirmed, reading} of a decided primary (DS.7.2): confirmed; 'qualifies,
    not confirmed' (exploratory); equivalent; inconclusive (reversed)."""
    exam = PRIMARIES[pid]["exam"]
    d = (rec or {}).get("decision") or {}
    conf = confirmed(rec, exam)
    if conf:
        reading = "confirmed"
    elif d.get("qualifies"):
        reading = "qualifies, not confirmed"
    else:
        reading = d.get("outcome")
    return {"status": "decided", "qualifies": bool(d.get("qualifies")), "z": d.get("z"), "threshold": THRESHOLD[exam],
            "confirmed": conf, "reading": reading, "flags": list(d.get("flags") or []),
            "interval_90": d.get("interval_90"), "diff": d.get("diff")}


def n1_reading(rec):
    """N1 (DS.7.2): S2X minus S0X on cwd12 dev species through the two-stage system is 'non-inferior' when the lower
    end of its 90 % interval is above -0.005; outside the correction."""
    lo = ((rec or {}).get("decision") or {}).get("interval_90", [None])[0]
    return {"non_inferior": bool(lo is not None and lo > -N1_MARGIN), "lower_90": lo, "margin": N1_MARGIN}


def h1_reading(rec):
    """H1 (DS.7.2): z < -1.645 is reported as 'the scaled build lowered cwd12 dev', whatever N1 says."""
    z = ((rec or {}).get("decision") or {}).get("z")
    return {"fires": bool(z is not None and z < H1_Z), "z": z, "bar": H1_Z}


def substitution_due(on_date, s2_frozen, s2x_walltime=False):
    """DS.0 item 17 and DS.7.2: S1u replaces S2 (S1UX replaces S2X) in P3, P4, N1 and H1 when S2's manifest is not
    frozen by 2026-12-05, decided by the date alone, without a further decision; and when S2x's smoke projects over
    28.8 h at the pair's compute, so S2x is recorded 'not run (walltime)' (DS.6: "the substitution of DS.7.2
    applies")."""
    return bool(s2x_walltime) or ((not s2_frozen) and str(on_date) > S2_FREEZE_BY)


def phase2_arms(substituted):
    """(the 1.2 M arm, the long arm) of phase 2: S2 and S2X, or S1U and S1UX under the substitution."""
    return ("S1U", "S1UX") if substituted else ("S2", "S2X")


def mde(sigma, z_alpha, z_power=Z_POWER):
    """The minimum detectable effect at one-sided alpha (z_alpha) and power 0.8: (z_alpha + 0.842) sigma."""
    return (float(z_alpha) + float(z_power)) * float(sigma)


def sigma_of(rec):
    return ((rec or {}).get("decision") or {}).get("sigma")


def n1_power(sigma, true_diff=0.0):
    """The power of N1 (lower 90 % end above -0.005) for a true difference: Phi((0.005 + true) / sigma - 1.645)."""
    return statistics.NormalDist().cdf((N1_MARGIN + float(true_diff)) / float(sigma) - Z90)


# ------------------------------------------------------------------ the outcome sentences (DS.7.4)
def sentence(pid, rec, counts=None, k=None, epochs_share=None):
    """The fixed sentence of a primary (DS.7.4) for its outcome: confirmed, equivalent or inconclusive."""
    r = primary_reading(pid, rec)
    d = r["diff"] if r["diff"] is not None else float("nan")
    lo, hi = (r["interval_90"] or [float("nan"), float("nan")])
    counts = counts or {}
    what = {"P1": "Adding the in-hand recovered images (%s images, %s distinct photos, %s capture groups)" % (
                counts.get("images", "N"), counts.get("photos", "P"), counts.get("groups", "G")),
            "P2": "More thinned greenhouse video frames, at equal compute, which gave every other image %s %% fewer "
                  "epochs" % (("%.0f" % epochs_share) if epochs_share is not None else "X"),
            "P3": "The scaled build, at %s times base B's compute" % (k if k is not None else "k")}
    if pid == "P4":
        caveat = (" (ood23: 13 of its images are of disputed season and one is the photograph ood22 also holds; the "
                  "same statistic on the other 1,771 images is reported beside it)")
        if r["confirmed"]:
            return "On a held-out season and region (ood23), the scaled build raised target-species agnostic AP by " \
                   "%.4f%s." % (d, caveat)
        if r["reading"] == "equivalent":
            return "On a held-out season and region (ood23), the scaled build changed target-species agnostic AP by " \
                   "less than +-0.005%s." % caveat
        return "On a held-out season and region (ood23), we cannot tell whether the scaled build changed " \
               "target-species agnostic AP (D %.4f, 90 %% interval [%.4f, %.4f])%s." % (d, lo, hi, caveat)
    head = what[pid]
    where = "weed AP on held-out scenes of base B's own sources"
    tail = " at equal compute" if pid in ("P1",) else ""
    if r["confirmed"]:
        return "%s raised %s by %.4f%s." % (head, where, d, tail)
    if r["reading"] == "equivalent":
        return "%s changed %s by less than +-0.005%s." % (head, where, tail)
    return "%s: we cannot tell whether it changed %s (D %.4f, 90 %% interval [%.4f, %.4f])%s." % (
        head, where, d, lo, hi, tail)


def n1_sentence(rec, power=None):
    r = n1_reading(rec)
    iv = ((rec or {}).get("decision") or {}).get("interval_90") or [float("nan"), float("nan")]
    if r["non_inferior"]:
        return "On cwd12, the scaled build is non-inferior to base B' within 0.005 (90 %% interval [%.4f, %.4f])." % (
            iv[0], iv[1])
    return "On cwd12, non-inferiority of the scaled build to base B' within 0.005 was not shown (90 %% interval " \
           "[%.4f, %.4f]; power about %s)." % (iv[0], iv[1], "%.2f" % power if power is not None else "?")


# ------------------------------------------------------------------ the 0.90 wording (DS.7.5)
def reaches_clears(values, se):
    """An arm reaches 0.90 when its 3-seed mean weed AP50-95 on MT-W is >= 0.9000; it clears 0.90 when the lower end
    of its 90 % interval, mean - 1.645 sqrt(SE^2 + sd^2 / 3) with SE = max(image, capture-group SE of the 3-seed
    mean), is >= 0.9000."""
    vals = [float(v) for v in values]
    if len(vals) != len(SEEDS):
        raise Ds1Refused("the 0.90 reading takes %d seeds, not %d" % (len(SEEDS), len(vals)))
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals)
    lo = m - Z90 * math.sqrt(float(se) ** 2 + sd ** 2 / len(vals))
    return {"mean": m, "sd": sd, "se": float(se), "lower_90": lo, "reaches": m >= CLEAR_090,
            "clears": lo >= CLEAR_090, "gap": CLEAR_090 - m}


# ------------------------------------------------------------------ the sealed reads (DS.7.5)
def read_order(v1, v2, substituted=False):
    """The sealed reads in their order, from the decided verdicts (records or None): the reference read (S0X on
    MT-W), the candidate read (S2X, or S1UX under the substitution, on MT-W), then only if P1 or P2 is confirmed the
    arm of the confirmed primary with the larger D (S1C for P1, S1U for P2) on MT-W, then S2X (S1UX) on the cwd12
    sealed test after verdict v2. Each read after the verdict that decides its arm; at most 4."""
    cand = phase2_arms(substituted)[1]
    order = []
    if not (isinstance(v2, dict) and v2.get("status") == "decided"):
        # no sealed read before verdict v2: DS.12 step 12 takes them in one order after it (S0x, S2x, the confirmed
        # 1.2 M arm, the cwd12 test), so no MT-W number exists while phase 2's sources and S2 are still chosen
        return order
    order.append({"arm": "S0X", "exam": "mtw", "kind": "reference"})
    order.append({"arm": cand, "exam": "mtw", "kind": "candidate"})
    if isinstance(v1, dict) and v1.get("status") == "decided":
        conf = []
        for pid, arm in (("P1", "S1C"), ("P2", "S1U")):
            p = (v1.get("primaries") or {}).get(pid) or {}
            if p.get("confirmed"):
                conf.append((float(p.get("diff") or 0.0), pid, arm))
        if conf:
            conf.sort(key=lambda t: (-t[0], t[1]))
            order.append({"arm": conf[0][2], "exam": "mtw", "kind": "challenger", "primary": conf[0][1]})
    if isinstance(v2, dict) and v2.get("status") == "decided":
        order.append({"arm": cand, "exam": "cwd12", "kind": "cwd12"})
    if len(order) > READ_CAP:
        raise RuntimeError("the read order holds %d reads, over the cap of %d" % (len(order), READ_CAP))
    return order


def check_read(arm, exam, order, done):
    """Refuse a read unless it is in the order, has not been taken, and every read before it in the order was taken
    or recorded not run (done: {(arm, exam): 'read' | 'not run ...'}); at most READ_CAP reads in all."""
    names = [(o["arm"], o["exam"]) for o in order]
    if (arm, exam) not in names:
        raise Ds1Refused("%s on %s is not a DS1 sealed read now (the reads are %s)" % (
            arm, exam, ["%s:%s" % n for n in names] or "none"))
    if (arm, exam) in done:
        raise Ds1Refused("%s on %s was %s already: a read is taken once" % (arm, exam, done[(arm, exam)]))
    i = names.index((arm, exam))
    before = ["%s:%s" % n for n in names[:i] if n not in done]
    if before:
        raise Ds1Refused("the reads are taken in DS.7.5's order: %s first" % before)
    if sum(1 for v in done.values() if v == "read") >= READ_CAP:
        raise Ds1Refused("DS1's %d sealed reads are taken" % READ_CAP)
    return {"arm": arm, "exam": exam, "position": i + 1, "kind": order[i]["kind"]}


def read_by_value(who):
    if who not in READ_BY:
        raise Ds1Refused("--read-by owner|delegate, not %r" % (who,))
    return READ_BY[who]


# ------------------------------------------------------------------ the family cap and the phase reserve (DS.11)
def cap_check(spent, estimate, phase, v2_done=False, substituted=False):
    """(ok, why): the item's estimate within the family cap (260 GPU-h), and, for a phase-1 item until verdict v2 or
    the date substitution, within 260 - 85.8 = 174.2 GPU-h (the phase reserve). spent: the family's GPU-h (running
    items at their estimates, ended ones settled, every attempt counted)."""
    after = float(spent) + float(estimate)
    if after > FAMILY_CAP_GPUH + CAP_EPS:
        return False, "%.2f GPU-h would take the family from %.2f past its cap of %.0f GPU-h" % (
            float(estimate), float(spent), FAMILY_CAP_GPUH)
    if int(phase) == 1 and not v2_done and not substituted and after > PHASE1_CEILING_GPUH + CAP_EPS:
        return False, "%.2f GPU-h would take the family from %.2f past %.1f GPU-h, the phase reserve's line (260 - " \
                      "85.8 kept for phase 2 and the reads)" % (float(estimate), float(spent), PHASE1_CEILING_GPUH)
    return True, ""


def floors(quota_gb):
    """{v3f, data}: the free GB each floor needs on /ocean (DS.0 item 21): V3F's D27 3 % + 30 GiB (257.8 GB on the
    6.836 TiB quota) and lane DATA's 3 % + 200 GB (425.6 GB)."""
    q = float(quota_gb)
    return {"v3f": V3F_FLOOR["share"] * q + V3F_FLOOR["margin_gib"] * GIB,
            "data": DATA_FLOOR["share"] * q + DATA_FLOOR["margin_gb"]}


def storage_short(free_gb, quota_gb, item_gb, phase1_bytes_gb=0.0, phase1=True):
    """GB short ('' when none) of a byte-adding DS1 item (DS.0 item 21): a phase-1 item whose bytes stay within
    phase 1's 60 GB keeps V3F's floor free after it; any later byte-adding item keeps lane DATA's floor too; every
    other item keeps V3F's floor."""
    fl = floors(quota_gb)
    left = float(free_gb) - float(item_gb or 0.0)
    within = phase1 and float(phase1_bytes_gb) + float(item_gb or 0.0) <= PHASE1_BYTES_GB + 1e-9
    need = fl["v3f"] if (within or not item_gb) else max(fl["v3f"], fl["data"])
    if left < need:
        return "storage short by %.1f GB (%.1f GB free - %.1f GB for the item, %.1f GB needed: %s)" % (
            need - left, float(free_gb), float(item_gb or 0.0), need,
            "V3F's floor" if need == fl["v3f"] else "lane DATA's floor")
    return ""


# ------------------------------------------------------------------ RA: the stream's gate records (DS.8.1)
def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def ra_arm(x_mean, x_sd, y_mean, y_sd, n=len(SEEDS)):
    """The seed-only statistics of a 3-v-3 difference X - Y (no per-image arrays were kept): D, the sds, the pooled
    sd, the seed-only SE sqrt((sd_X^2 + sd_Y^2) / n) and z (overstating |z|), the MDE at one-sided 5 % and power 0.8,
    the 90 % interval, and the outcome class against SESOI from the seed term only (inc2.v3fam.decide with SE 0)."""
    x_mean, x_sd, y_mean, y_sd = _f(x_mean), _f(x_sd), _f(y_mean), _f(y_sd)
    if None in (x_mean, x_sd, y_mean, y_sd):
        return {"status": "missing"}
    d = x_mean - y_mean
    se = math.sqrt((x_sd ** 2 + y_sd ** 2) / float(n))
    z = d / se if se > 0 else (math.inf if d > 0 else -math.inf if d < 0 else 0.0)
    pooled = math.sqrt((x_sd ** 2 + y_sd ** 2) / 2.0)
    q = d > 2.0 * pooled and d > 0.0
    cls, flags, iv = F.outcome(q, d, se)
    return {"status": "computed", "diff": d, "x_mean": x_mean, "x_sd": x_sd, "y_mean": y_mean, "y_sd": y_sd,
            "pooled_sd": pooled, "seed_se": se, "seed_z": z, "mde": mde(se, Z90), "interval_90": iv,
            "outcome": cls, "flags": flags + ["seed term only: overstates |z|"]}


def ra_reading(chain, attribution, blame):
    """DS.8.1's reading (descriptive; written after R6 was read)."""
    out = []
    if blame == "data" and chain.get("status") == "computed":
        lo, hi = chain["interval_90"]
        if lo <= 0.0 <= hi:
            out.append("not evidence of harm at the chain's resolution")
        else:
            out.append("a resolved 12-class loss, %s SESOI" % ("above" if abs(chain["diff"]) >= SESOI else "below"))
    ag = _f((attribution or {}).get("agnostic_delta"))
    if ag is not None and ag >= 0.0 and (attribution or {}).get("class_vs_loc") == "labels" \
            and chain.get("status") == "computed" and chain["diff"] < 0:
        out.append("naming, not placement (cause C)")
    return out


def ra_step(step, segment):
    """One recorded gate step of a stream segment's report (steps[] of <exp>/report.json)."""
    out = {"segment": segment, "step": step.get("step"), "tag": step.get("tag"), "chains": {}}
    for name, ch in sorted((step.get("chains") or {}).items()):
        att = ch.get("attribution") or {}
        chain = ra_arm(ch.get("cand_mean"), ch.get("cand_sd"), ch.get("null_mean"), ch.get("null_sd"))
        blame = att.get("blame") or ("hold" if ch.get("verdict") == "HOLD" else None)
        if chain.get("status") == "computed":
            chain["shared_null_shift"] = {"by": chain["y_sd"], "diff_if_null_up": chain["diff"] - chain["y_sd"],
                                          "diff_if_null_down": chain["diff"] + chain["y_sd"]}
        out["chains"][name] = {"chain": chain, "verdict": ch.get("verdict"), "disposition": blame,
                               "agnostic_delta": _f(att.get("agnostic_delta")), "map_delta": _f(att.get("map_delta")),
                               "class_vs_loc": att.get("class_vs_loc"), "p_data": _f(ch.get("p_data")),
                               "reading": ra_reading(chain, att, blame), "sources": ch.get("sources"),
                               "d_images": ch.get("d_images")}
    tr = step.get("truth") or {}
    truth = ra_arm(tr.get("with_mean"), tr.get("with_sd"), tr.get("without_mean"), tr.get("without_sd"))
    if truth.get("status") == "computed":
        truth["shared_control_shift"] = {"by": truth["y_sd"], "diff_if_control_up": truth["diff"] - truth["y_sd"],
                                         "diff_if_control_down": truth["diff"] + truth["y_sd"]}
    out["truth"] = dict(truth, verdict=tr.get("verdict"), species_failed=tr.get("species_failed"))
    return out


# RA's segments (DS.8.1, R6): the stream's segments since its fork at row 1364 (s001's inc0001-inc0004, s002's
# inc0005, and any later one); never an experiment whose name merely ends in _s<3 digits> (b_v2_s640) and never a
# segment of the stream before the fork (weed_stream_v1_s001, weed_stream_v1_s002)
RA_SEGMENT_RE = re.compile(r"weed_stream_v1_fork1364_s[0-9]{3}\Z")


def ra_reports(inc_dir=None):
    """The default reports of `ra`: <INC>/<segment>/report.json of every segment RA_SEGMENT_RE names, sorted."""
    base = Path(inc_dir) if inc_dir else C.INC_DIR
    return sorted(str(p / "report.json") for p in base.iterdir() if p.is_dir() and RA_SEGMENT_RE.match(p.name)
                  and (p / "report.json").is_file()) if base.is_dir() else []


def ra(report_paths, out_dir=None, write=True):
    """capacity/ds1_ra_gate.{json,md} (DS.8.1): every stream step with a recorded gate in the named segment reports,
    each report recorded by its sha256. The first DS1 step after the stamp; no GPU; descriptive."""
    steps, files = [], []
    for p in report_paths:
        p = Path(p)
        doc = _read_json(p)
        if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
            raise Ds1Refused("%s is not a stream segment's report (no steps)" % p)
        files.append({"file": str(p), "sha256": C.sha256_file(p), "exp": doc.get("exp")})
        for st in doc["steps"]:
            if isinstance(st, dict) and (st.get("chains") or st.get("truth")):
                steps.append(ra_step(st, doc.get("exp")))
    rec = {"format": FMT_RA, "status": "complete", "descriptive": True, "decided_by": DECIDED_BY,
           "reading_rule": "written after R6 was read: descriptive (DS.8.1)", "reports": files, "steps": steps,
           "sesoi": SESOI, "written_utc": _utc()}
    if write:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        write_once(d / "ds1_ra_gate.json", rec)
        write_once(d / "ds1_ra_gate.md", ra_md(rec).encode("utf-8"))
    return rec


def ra_md(rec):
    L = ["# DS1 RA: the stream's gate records re-analysed (descriptive)", "",
         "Seed-only statistics (no per-image arrays were kept): |z| is overstated. SESOI %.3f." % SESOI, "",
         "| Segment | Step | Chain D | seed SE | z | MDE | 90 % interval | Outcome | agnostic D | class_vs_loc | "
         "Reading | Truth D | Truth MDE |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rec["steps"]:
        for name, c in sorted(s["chains"].items()):
            ch, tr = c["chain"], s["truth"]

            def g(x, k, fmt="%.4f"):
                v = x.get(k)
                return fmt % v if isinstance(v, (int, float)) and math.isfinite(v) else "-"
            iv = ch.get("interval_90") or [None, None]
            L.append("| %s | %s | %s | %s | %s | %s | [%s, %s] | %s | %s | %s | %s | %s | %s |" % (
                s["segment"], s["step"], g(ch, "diff"), g(ch, "seed_se"), g(ch, "seed_z", "%.2f"), g(ch, "mde"),
                "%.4f" % iv[0] if iv[0] is not None else "-", "%.4f" % iv[1] if iv[1] is not None else "-",
                ch.get("outcome") or "-", "%.4f" % c["agnostic_delta"] if c["agnostic_delta"] is not None else "-",
                c["class_vs_loc"] or "-", "; ".join(c["reading"]) or "-", g(tr, "diff"), g(tr, "mde")))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ LN: a person's per-box label sheet (DS.8.2)
LN_GROUPS = {"siu_increments": 200, "inc0005": 100, "train_core": 250}
LN_MARKS = ("correct", "not_plant", "loose", "two_plants")
LN_SUPPORT = 0.03


def ln_error_share(images):
    """(wrong + missing) / (boxes + missing) over images [{"marks": [...], "missing": n}]."""
    wrong = sum(sum(1 for m in im.get("marks") or [] if m != "correct") for im in images)
    boxes = sum(len(im.get("marks") or []) for im in images)
    miss = sum(int(im.get("missing") or 0) for im in images)
    den = boxes + miss
    return (wrong + miss) / float(den) if den else None


def ln_statistic(by_group, resamples=RESAMPLES, seed="inc2/ds1/ln/bootstrap"):
    """LN (DS.8.2): per group the per-box error share with a 95 % image-bootstrap interval, and for each increment
    group minus train_core the bootstrap interval of the difference; the reading: label noise supported when the
    difference's lower end is above 0.03, not supported when its upper end is below 0.03, else unresolved."""
    import numpy as np
    rng = np.random.default_rng(C.stable_int(seed))
    out = {"groups": {}, "differences": {}}
    draws = {}
    for g, ims in sorted(by_group.items()):
        ims = list(ims)
        if not ims:
            out["groups"][g] = {"images": 0, "share": None}
            continue
        idx = rng.integers(0, len(ims), size=(int(resamples), len(ims)))
        vals = np.array([ln_error_share([ims[i] for i in row]) or 0.0 for row in idx])
        draws[g] = vals
        out["groups"][g] = {"images": len(ims), "share": ln_error_share(ims),
                            "interval_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    ref = draws.get("train_core")
    for g in sorted(draws):
        if g == "train_core" or ref is None:
            continue
        diff = draws[g] - ref
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        reading = "supported" if lo > LN_SUPPORT else ("not supported" if hi < LN_SUPPORT else "unresolved")
        out["differences"][g] = {"diff": (out["groups"][g]["share"] or 0.0) - (out["groups"]["train_core"]["share"]
                                                                              or 0.0),
                                 "interval_95": [lo, hi], "reading": reading, "threshold": LN_SUPPORT}
    return out


# ------------------------------------------------------------------ the MT-W check (DS.10 item 4)
MTW_SAMPLE = 60
MTW_MAX_ERRORS = 1


def wilson_upper(errors, n, z=1.959963984540054):
    """The 95 % Wilson upper bound of an error share (1 in 60: 0.089; 2 in 60: 0.114)."""
    if n <= 0:
        return 1.0
    p = errors / float(n)
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (centre + half) / den


def mtw_source_verdict(sample_images, total_images, errors, corrected=False, whole=None):
    """One source's MT-W check (DS.10 item 4): with a sample of 60 a source passes at most 1 image in error (95 %
    Wilson upper bound 0.089; 2 give 0.114, over 0.10); a source whose sample is all its MT-W images (fewer than 60)
    is checked whole and passes once its errors are corrected; a failing source is checked image by image or leaves
    MT-W, and its rows stay never-train."""
    whole = (int(sample_images) >= int(total_images)) if whole is None else bool(whole)
    if whole and int(total_images) < MTW_SAMPLE:
        ok = int(errors) == 0 or bool(corrected)
        return {"verdict": "pass" if ok else "fail", "rule": "checked whole (fewer than %d images): passes once its "
                "errors are corrected" % MTW_SAMPLE, "errors": int(errors), "corrected": bool(corrected),
                "images": int(total_images)}
    if int(sample_images) < MTW_SAMPLE:
        raise Ds1Refused("a source with %d MT-W images is checked on %d seeded images, not %d" % (
            int(total_images), MTW_SAMPLE, int(sample_images)))
    ok = int(errors) <= MTW_MAX_ERRORS
    return {"verdict": "pass" if ok else "fail", "rule": "at most %d image in error of %d" % (MTW_MAX_ERRORS,
                                                                                             MTW_SAMPLE),
            "errors": int(errors), "wilson_upper_95": wilson_upper(int(errors), int(sample_images)),
            "images": int(total_images), "sample": int(sample_images)}


# ------------------------------------------------------------------ the DS1 pass (DS.7.1)
EXAMS_MAIN = ("dev", "imageweeds", "ood22", "ood23")      # under the main INC_DIR
ROOT_EXAM_RE = re.compile(r"(md_s0|mdn_[a-z0-9_]{1,60}|mtw_v2)\Z")
SEALED_ROOT_EXAMS = ("test_v1", "evalgroups_v1", "mtw_v2")      # read only by score-test, through a prepared read
MTW_SIU_SOURCE = "zenodo_15808623"                              # test v1's in-plant stratum (DS.0 item 5)


def exam_allowed(arm, exam, test_token=None):
    """DS.4: an arm reads the exams its row of DS.6 names; mtw_v2, test v1's and the cwd12 sealed test only with
    score-test's token; md_s0 and mdn_<source> only in MD's own root."""
    if exam in ("test", "test_v1", "evalgroups_v1", "mtw_v2"):
        if not any(test_token is t for t in _TEST_TOKENS):
            raise Ds1Refused("%s is read only by score-test, after the verdict that decides the arm and a read prepared "
                             "by test-read (DS.7.5)" % exam)
        return True
    allowed = ARM_EXAMS.get(arm)
    if allowed is None:
        raise Ds1Refused("arm %r is not one of DS1's score arms %s" % (arm, list(SCORE_ARMS)))
    base = "mdn" if exam.startswith("mdn_") else exam
    if base not in allowed:
        raise Ds1Refused("%s does not read %s: its exams are %s (DS.6)" % (arm, exam, list(allowed)))
    return True


@contextlib.contextmanager
def test_token():
    tok = object()
    _TEST_TOKENS.append(tok)
    try:
        yield tok
    finally:
        _TEST_TOKENS.remove(tok)


def ds1_validator_class(cfg):
    """(DS1's validator class, the scorer's own validator): inc2.v3fam.family_validator_class(cfg) with one DS1
    layer that keeps, per image, the stage-1 box set's class-collapsed matches against every ground-truth box (the
    weed arrays)."""
    import numpy as np
    FV, base = F.family_validator_class(cfg)

    class Ds1Validator(FV):
        _ds1_layer = True

        def inc_reset(self):
            super().inc_reset()
            self.d_weed = {}
            self.d_order = []                 # the validator's image order (Revision 2, item 1)

        def e3_stage1(self, pred, pb):
            n0 = len(self.e3_s1_tp)
            it = super().e3_stage1(pred, pb)
            if len(self.e3_s1_tp) != n0 + 1:
                raise RuntimeError("E3's stage-1 helper kept %d records for one image" % (len(self.e3_s1_tp) - n0))
            self.d_weed[it["key"]] = {"tp": np.asarray(self.e3_s1_tp[-1], dtype=bool),
                                      "q": np.asarray(self.e3_s1_q[-1], dtype=np.float64),
                                      "n_gt": int(len(pb["cls"]))}
            self.d_order.append(it["key"])
            return it

    layers = [k for k in Ds1Validator.__mro__ if k.__dict__.get("_ds1_layer")]
    if len(layers) != 1 or FV not in Ds1Validator.__mro__:
        raise RuntimeError("the DS1 validator holds %d DS1 layers" % len(layers))
    return Ds1Validator, base


def weed_arrays(v, keys):
    """{keys, tp, conf, pred_img, n_gt}: the class-collapsed arrays of the stage-1 boxes in exam key order (what
    inc2.v3fam.run_values reads as 'agnostic')."""
    import numpy as np
    tps, confs, img, ngt = [], [], [], []
    for i, k in enumerate(keys):
        r = v.d_weed.get(k)
        if r is None:
            raise Ds1Refused("the DS1 pass kept no stage-1 record for exam image %s" % k)
        tps.append(r["tp"].reshape(-1, SC.NIOU) if len(r["tp"]) else np.zeros((0, SC.NIOU), bool))
        confs.append(r["q"])
        img.append(np.full(len(r["q"]), i, dtype=np.int64))
        ngt.append(r["n_gt"])
    return {"keys": np.asarray(list(keys)), "tp": np.concatenate(tps, 0) if tps else np.zeros((0, SC.NIOU), bool),
            "conf": np.concatenate(confs) if confs else np.zeros(0), "pred_img": np.concatenate(img) if img
            else np.zeros(0, np.int64), "n_gt": np.asarray(ngt, dtype=np.int64)}


def weed_values(weed):
    """(AP50-95, AP50) of the weed arrays as DS1's statistics read them: inc.scorer.collapsed_ap on the tie-broken
    arrays in exam key order (inc2.scorer_sidecar.tie_break; inc2.v3fam.statistic_value's 'agnostic'), one value
    whatever the machine's sort."""
    tb = SC.tie_break(weed)
    return S.collapsed_ap(tb["tp"], tb["conf"], int(tb["n_gt"].sum()))


def weed_check(v, keys, weed, s1):
    """The weed arrays are the stage-1 helper's own records (Revision 2, item 1): the validator saw each exam image
    once, in the order it recorded; the arrays, put back in that order, hold every record row for row
    (tp and conf equal) and the same ground-truth count, and give the pass's stage-1 agnostic AP (AP50-95 and AP50)
    within WEED_RECOMPUTE_TOL; their tie-broken AP in exam key order is within WEED_TIE_TOL of it. Returns the
    checks' record; Ds1Refused otherwise. (Before Revision 2 the key-order arrays were compared with the
    validator-order AP within 1e-9, which ties in the confidences break: 0.856888007 against 0.856857855 for E1-A
    seed 0 on dev, POWER jobs 50015220 and 50015542.)"""
    import numpy as np
    order = [str(k) for k in getattr(v, "d_order", None) or []]
    keys = [str(k) for k in keys]
    if len(order) != len(set(order)) or sorted(order) != sorted(keys) or len(order) != len(v.e3_s1_tp) \
            or len(order) != len(v.e3_s1_q):
        raise Ds1Refused("the DS1 layer recorded %d image(s) (%d distinct) in the validator's order, E3's stage-1 "
                         "helper %d, the exam %d" % (len(order), len(set(order)), len(v.e3_s1_tp), len(keys)))
    if [str(k) for k in weed["keys"]] != keys:
        raise Ds1Refused("the weed arrays are not in the exam's key order")
    at = {k: i for i, k in enumerate(keys)}
    pimg = np.asarray(weed["pred_img"], dtype=np.int64)
    if len(pimg) and (np.any(np.diff(pimg) < 0) or pimg[0] < 0 or pimg[-1] >= len(keys)):
        raise Ds1Refused("the weed arrays' rows are not grouped by exam image in key order")
    lo = np.searchsorted(pimg, np.arange(len(keys)), side="left")
    hi = np.searchsorted(pimg, np.arange(len(keys)), side="right")
    parts_tp, parts_q = [], []
    for j, k in enumerate(order):
        rows = np.arange(lo[at[k]], hi[at[k]])
        tp_w = np.asarray(weed["tp"], dtype=bool)[rows].reshape(-1, SC.NIOU)
        q_w = np.asarray(weed["conf"], dtype=np.float64)[rows]
        tp_e = np.asarray(v.e3_s1_tp[j], dtype=bool).reshape(-1, SC.NIOU)
        q_e = np.asarray(v.e3_s1_q[j], dtype=np.float64).reshape(-1)
        if tp_w.shape != tp_e.shape or not np.array_equal(tp_w, tp_e) or not np.array_equal(q_w, q_e):
            raise Ds1Refused("the weed arrays of exam image %s are not the stage-1 helper's record of it (%d rows "
                             "against %d)" % (k, len(q_w), len(q_e)))
        parts_tp.append(tp_w)
        parts_q.append(q_w)
    n_gt = int(np.asarray(weed["n_gt"], dtype=np.int64).sum())
    if n_gt != int(v.e3_s1_ngt):
        raise Ds1Refused("the weed arrays count %d ground-truth boxes, the stage-1 helper %d" % (n_gt, v.e3_s1_ngt))
    tp = np.concatenate(parts_tp, 0) if parts_tp else np.zeros((0, SC.NIOU), dtype=bool)
    q = np.concatenate(parts_q) if parts_q else np.zeros(0)
    cap = S.collapsed_ap(tp, q, int(v.e3_s1_ngt))
    rd = max(abs(float(cap[0]) - float(s1[0])), abs(float(cap[1]) - float(s1[1])))
    if rd > WEED_RECOMPUTE_TOL:
        raise Ds1Refused("the weed arrays in the validator's order give AP50-95 %.9f, the pass's stage-1 agnostic AP "
                         "%.9f" % (cap[0], s1[0]))
    tb = weed_values(weed)
    td = abs(float(tb[0]) - float(s1[0]))
    if td > WEED_TIE_TOL:
        raise Ds1Refused("the tie-broken weed arrays give AP50-95 %.9f, %.6f from the pass's stage-1 agnostic AP "
                         "(> %s: not the score's arrays)" % (tb[0], td, WEED_TIE_TOL))
    return {"weed_recompute_capture_order": rd, "weed_recompute_tol": WEED_RECOMPUTE_TOL,
            "weed_tie_broken_map50_95": float(tb[0]), "weed_tie_broken_vs_stage1": td, "weed_tie_tol": WEED_TIE_TOL}


def ds1_pass(weights, exam, cfg, arm, lock_check=True, batch=S.BATCH, device=None, test_token_=None):
    """One locked-scorer pass of weights on exam with DS1's validator installed (inc2.v3fam._family_pass's steps,
    DS1's exam allow-list). Returns {result, arrays, weed, stage1, rows, p7, p7_full, stage1_agnostic, stats, gt,
    seconds, mode, exam, arm}; the weed arrays are the stage-1 helper's records and give stage1_agnostic in the
    validator's order (weed_check), or the pass refuses."""
    import numpy as np
    check_frozen()
    if (ROOT_EXAM_RE.match(exam) or exam in SEALED_ROOT_EXAMS) and _IN_ROOT.get("exam") != exam:
        raise Ds1Refused("%s is scored only in its own root, in a subprocess with INC_DIR there" % exam)
    if exam in SEALED_ROOT_EXAMS and not _IN_ROOT.get("read"):
        raise Ds1Refused("%s is read only by score-test, through a read prepared by test-read (DS.7.5)" % exam)
    if _IN_ROOT and _IN_ROOT.get("exam") != exam:
        raise Ds1Refused("this process scores %s in its root, not %s" % (_IN_ROOT.get("exam"), exam))
    if exam in SEALED_ROOT_EXAMS:
        if arm not in SCORE_ARMS:
            raise Ds1Refused("arm %r is not one of DS1's" % arm)
    else:
        exam_allowed(arm, exam, test_token_)
    mode = cfg.get("mode")
    if mode not in ("identity", "stage1", "two_stage"):
        raise Ds1Refused("a DS1 pass is identity, stage1 or two_stage, not %r" % mode)
    if _IN_ROOT and mode == "two_stage":
        raise Ds1Refused("a pass in MD's root takes identity or stage1 only")
    if mode == "two_stage" and cfg.get("embedder") is None:
        raise Ds1Refused("a two_stage pass needs the embedder")
    cfg = dict(cfg, orientations=F.orientations_for(exam) if exam in F.OOD_EXAMS else tuple(TS.OK_ORIENTATIONS))
    cls, base = ds1_validator_class(cfg)
    prev = S._VALIDATOR
    del SC._CAPTURE[:]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="inc2_ds1_") as tmp:
        S._VALIDATOR = cls
        try:
            res = S.score(weights, exam, Path(tmp) / "score.json", lock_check=lock_check, imgsz=F.IMGSZ, batch=batch,
                          device=device)
        except S.ScorerRefused as e:
            raise Ds1Refused("the locked scorer refuses: %s" % e)
        except TS.E3Refused as e:
            raise Ds1Refused("E3's path refuses: %s" % e)
        except F.V3famRefused as e:
            raise Ds1Refused("the V3F validator refuses: %s" % e)
        finally:
            S._VALIDATOR = prev if prev is not None else base
    caps = [x for x in SC._CAPTURE if isinstance(x, cls)]
    del SC._CAPTURE[:]
    if not caps:
        raise Ds1Refused("the scorer ran no DS1 validator")
    v = caps[-1]
    res = {k: val for k, val in res.items() if k != "out"}
    keys = sorted(r["key"] for r in C.read_manifest(C.manifest_path(exam)))
    if C.sha256_text("\n".join(keys)) != res["key_order_sha256"]:
        raise Ds1Refused("the exam's key order does not hash to the score's key_order_sha256")
    if sorted(v.f_seen) != keys or sorted(v.d_weed) != keys:
        raise Ds1Refused("the DS1 validator saw %d of the exam's %d images" % (len(set(v.f_seen)), len(keys)))
    try:
        recompute, restricted = SC.check_capture(res["per_class"], v.sc_images)
        arrays = SC.flatten(v.sc_images, keys)
    except SC.SidecarError as e:
        raise Ds1Refused("the captured per-image arrays do not reproduce the score: %s" % e)
    tp = np.concatenate(v.e3_s1_tp, 0) if v.e3_s1_tp else np.zeros((0, SC.NIOU), dtype=bool)
    q = np.concatenate(v.e3_s1_q, 0) if v.e3_s1_q else np.zeros(0)
    s1 = S.collapsed_ap(tp, q, v.e3_s1_ngt)
    weed = weed_arrays(v, keys)
    wchk = weed_check(v, keys, weed, s1)
    stats = {k: int(val) for k, val in sorted(v.e3_stats.items())}
    stats.update({k: int(val) for k, val in sorted(v.f_stats.items())})
    stats.update(orientations=dict(sorted(v.f_tags.items())), gt_roundtrip_max_diff=v.e3_gt_maxdiff,
                 full_recompute_max_abs_diff=recompute, species_restricted_max_abs_diff=restricted,
                 orientations_read=[str(t) for t in cfg["orientations"]])
    stats.update(wchk)
    out = {"result": res, "arrays": arrays, "weed": weed, "stage1_agnostic": s1, "stats": stats, "gt": list(v.e3_gt),
           "seconds": round(time.time() - t0, 3), "mode": mode, "exam": exam, "arm": arm, "stage1": None,
           "rows": None, "p7": None, "p7_full": None}
    if mode == "two_stage":
        out["stage1"] = TS._stage1_arrays(v, keys)
    elif mode == "stage1":
        out["stage1"] = F._flat_rows(v.f_s1, keys, ("xyxy", "q"))
        out["p7"], out["p7_full"] = F._p7_arrays(v, keys)
    if mode in ("identity", "stage1"):
        out["rows"] = F._flat_rows(v.f_rows, keys, ("xyxy", "conf", "cls"))
    return out


def _root_env(root_):
    env = dict(os.environ)
    env["INC_DIR"] = str(root_)
    env["INC_ZOO_SOURCE"] = str(C.INC_DIR)
    pkg = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def pass_in_root(weights, exam, mode, arm, out_dir, root_dir=None, plain=False, lock_check=True, batch=S.BATCH,
                 device=None, read_json=None):
    """ds1_pass on an exam of MD's root (md_s0, mdn_<source>) in a subprocess with INC_DIR there (Z1's way); the
    child writes its result outside the root. plain: the plain locked score of the same weights in the same
    subprocess (the identity reproduction on MD). Returns ds1_pass's dict (no validator) and 'plain'."""
    sealed = exam in SEALED_ROOT_EXAMS
    if not (ROOT_EXAM_RE.match(exam) or sealed):
        raise Ds1Refused("%s is not an exam of another root" % exam)
    if mode not in ("identity", "stage1"):
        raise Ds1Refused("a pass in another root takes identity or stage1 only")
    if sealed:
        rd = _read_json(read_json) if read_json else None
        if not isinstance(rd, dict) or rd.get("format") != FMT_READ or rd.get("arm") != arm:
            raise Ds1Refused("%s is read only through a read test-read prepared for %s" % (exam, arm))
    else:
        exam_allowed(arm, exam)
    root_ = Path(root_dir) if root_dir else md_root(exam)
    if not (root_ / "splits" / "v1" / "LOCK.json").is_file():
        raise Ds1Refused("MD's root %s holds no splits/v1/LOCK.json" % root_)
    out_dir = Path(out_dir)
    if out_dir.resolve() == root_.resolve() or root_.resolve() in out_dir.resolve().parents:
        raise Ds1Refused("the child writes outside the root it scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"format": FMT_PASS_SPEC, "root": str(root_.resolve()), "weights": str(Path(weights).resolve()),
            "exam": exam, "arm": arm, "mode": mode, "nms": NMS if mode == "stage1" else "locked", "plain": bool(plain),
            "lock_check": bool(lock_check), "batch": int(batch), "device": device, "out": str(out_dir.resolve()),
            "read_json": str(Path(read_json).resolve()) if read_json else None}
    sp = out_dir / "spec.json"
    sp.write_text(TS._dump(spec))
    p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.ds1", "_pass", "--spec", str(sp)],
                       env=_root_env(root_), capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-3:]
        raise Ds1Refused("the pass of %s in %s exited %d: %s" % (exam, root_, p.returncode, " | ".join(tail)))
    got = _read_json(out_dir / "result.json")
    if not isinstance(got, dict):
        raise Ds1Refused("the pass in %s wrote no result" % root_)
    for name in ("arrays", "stage1", "rows", "weed"):
        f = out_dir / ("%s.npz" % name)
        got[name] = SC.load_npz(f) if f.is_file() else None
    p7 = {t: SC.load_npz(out_dir / ("p7_%s.npz" % t)) for t in ("t", "o") if (out_dir / ("p7_%s.npz" % t)).is_file()}
    got["p7"] = p7 or None
    got["stage1_agnostic"] = tuple(got.get("stage1_agnostic") or ())
    return got


def _run_pass_spec(spec_path):
    """The _pass verb: the child side of pass_in_root."""
    spec = _read_json(spec_path)
    if not isinstance(spec, dict) or spec.get("format") != FMT_PASS_SPEC:
        raise Ds1Refused("%s is not a DS1 pass spec" % spec_path)
    if C.INC_DIR.resolve() != Path(spec["root"]).resolve():
        raise Ds1Refused("INC_DIR is %s, the spec's root %s" % (C.INC_DIR, spec["root"]))
    if not (ROOT_EXAM_RE.match(spec["exam"]) or spec["exam"] in SEALED_ROOT_EXAMS):
        raise Ds1Refused("%s is not an exam of another root" % spec["exam"])
    _IN_ROOT["exam"] = spec["exam"]
    if spec["exam"] in SEALED_ROOT_EXAMS:
        rd = _read_json(spec.get("read_json")) if spec.get("read_json") else None
        if not isinstance(rd, dict) or rd.get("format") != FMT_READ or rd.get("arm") != spec["arm"]:
            raise Ds1Refused("the sealed exam %s is read only through a prepared read" % spec["exam"])
        _IN_ROOT["read"] = spec["read_json"]
    out = Path(spec["out"])
    kw = {"lock_check": spec["lock_check"], "batch": spec["batch"], "device": spec.get("device")}
    ps = ds1_pass(spec["weights"], spec["exam"], {"mode": spec["mode"], "nms": spec.get("nms", "locked")},
                  spec["arm"], **kw)
    doc = {"result": ps["result"], "stage1_agnostic": list(ps["stage1_agnostic"]), "stats": ps["stats"],
           "p7_full": ps["p7_full"], "mode": ps["mode"], "exam": ps["exam"], "arm": ps["arm"],
           "seconds": ps["seconds"], "code": code_shas(), "root": spec["root"]}
    if spec.get("plain"):
        with tempfile.TemporaryDirectory(prefix="inc2_ds1_plain_") as tmp:
            pr = S.score(spec["weights"], spec["exam"], Path(tmp) / "plain.json", **kw)
        doc["plain"] = {k: v for k, v in pr.items() if k != "out"}
    for name in ("arrays", "stage1", "rows", "weed"):
        if ps.get(name) is not None:
            SC.save_npz(out / ("%s.npz" % name), ps[name])
    for tag, a in (ps["p7"] or {}).items():
        SC.save_npz(out / ("p7_%s.npz" % tag), a)
    TS._write_json(out / "result.json", doc)
    return doc


# ------------------------------------------------------------------ committed scores
def file_names(exam):
    """(json, images npz, stage-1 npz, rows npz, p7 npz) as inc2.v3fam names them, and the weed npz."""
    return F.file_names(exam) + ("%s.weed.npz" % exam,)


def score_doc(arm, seed, exam, ps, cfg, source=None, extra=None, classifier=None, equivalence=None):
    """A DS1 score document: the locked scorer's result stamped DS1- (never a protocol score), the mode, the
    stage-1 and weed readings, the equivalence records it rests on, the code."""
    res = ps["result"]
    locked = TS._locked(res["scorer_sha256"])
    test = not res.get("production")
    w = ps.get("weed")
    w_ap = weed_values(w) if w is not None else (None, None)        # tie-broken, key order (Revision 2, item 1)
    doc = dict(res)
    doc.update({"format": FMT_SCORE, "family": "ds1", "arm": arm, "seed": None if seed is None else int(seed),
                "exam": exam, "scorer_sha256": "%s%s%s" % (S.TEST_PREFIX if test else "", STAMP, locked),
                "locked_scorer_sha256": locked, "production": False, "ds1_production": not test, "imgsz": F.IMGSZ,
                "settings": dict(res["settings"], mode=ps["mode"], nms=cfg.get("nms", "locked")),
                "stage1": {"n_boxes": int(ps["stats"].get("stage1_boxes", 0)),
                           "agnostic_map50_95": ps["stage1_agnostic"][0], "agnostic_map50": ps["stage1_agnostic"][1]},
                "weed_map50_95": w_ap[0], "weed_map50": w_ap[1], "p7_full": ps.get("p7_full"),
                "classifier": classifier, "source": source, "equivalence": equivalence,
                "stats": ps["stats"], "decided_by": DECIDED_BY, "code": code_shas(), "created_utc": _utc(),
                "seconds": ps.get("seconds")})
    if extra:
        doc.update(extra)
    return doc


def commit(arm, seed, exam, ps, cfg, source=None, extra=None, classifier=None, equivalence=None, d=None):
    """Write one DS1 score once under ds1/v1/arms/<ARM>/s<k>/ (inc2.v3fam.commit_pass: its npz side files, then the
    JSON, hard-linked only if absent), the weed npz first. Returns its entry."""
    d = Path(d) if d else arm_dir(arm, seed)
    if (d / file_names(exam)[0]).exists():
        raise Ds1Refused("%s exists: a DS1 score is written once" % (d / file_names(exam)[0]))
    d.mkdir(parents=True, exist_ok=True)
    doc = score_doc(arm, seed, exam, ps, cfg, source=source, extra=extra, classifier=classifier,
                    equivalence=equivalence)
    if ps.get("weed") is not None:
        wn = file_names(exam)[5]
        doc["weed"] = {"name": wn, "sha256": SC.save_npz(d / wn, ps["weed"])}
    try:
        js = F.commit_pass(d, exam, doc, ps)
    except F.V3famRefused as e:
        raise Ds1Refused(str(e))
    return entry(js, "written")


def entry(js, status):
    d = _read_json(js)
    if not isinstance(d, dict) or d.get("format") != FMT_SCORE:
        raise Ds1Refused("%s is not a DS1 score" % js)
    out = {"file": str(Path(js).relative_to(root())), "sha256": _sha(js), "status": status, "arm": d.get("arm"),
           "seed": d.get("seed"), "exam": d.get("exam")}
    if d.get("exam") == "dev" or ROOT_EXAM_RE.match(str(d.get("exam"))):
        out.update(species_map50_95=d.get("species_map50_95"), weed_map50_95=d.get("weed_map50_95"),
                   stage1_agnostic_map50_95=(d.get("stage1") or {}).get("agnostic_map50_95"))
    return out


def existing(arm, seed, exam):
    js = arm_dir(arm, seed) / file_names(exam)[0]
    return entry(js, "kept") if js.exists() else None


def load_runs(arm, exam, kind, seeds=SEEDS, base=None):
    """[(doc, arrays)] of committed DS1 scores: kind 'species' (the scorer's per-image arrays), 'weed' (the
    class-collapsed stage-1 arrays) or 'p7t' (P7's target reading with OtherPlant ignore regions); every file
    checked against the sha256 its JSON records."""
    out = []
    for s in seeds:
        d = (Path(base) if base else root() / "arms" / arm) / ("s%d" % int(s))
        jn, nn = file_names(exam)[:2]
        doc = _read_json(d / jn)
        if not isinstance(doc, dict) or doc.get("format") != FMT_SCORE:
            raise Ds1Refused("no DS1 score at %s" % (d / jn))
        if kind == "species":
            if _sha(d / nn) != (doc.get("images") or {}).get("sha256"):
                raise Ds1Refused("%s does not hash as its score records" % (d / nn))
            out.append((doc, SC.load_npz(d / nn)))
        elif kind == "weed":
            rec = doc.get("weed") or {}
            if not rec.get("name") or _sha(d / rec["name"]) != rec.get("sha256"):
                raise Ds1Refused("%s's weed arrays are missing or do not hash as recorded" % (d / jn))
            out.append((doc, SC.load_npz(d / rec["name"])))
        elif kind == "p7t":
            rec = doc.get("p7") if isinstance(doc.get("p7"), dict) and "name" in doc.get("p7") else None
            side = rec or {}
            if not side.get("name") or _sha(d / side["name"]) != side.get("sha256"):
                raise Ds1Refused("%s's p7 arrays are missing or do not hash as recorded" % (d / jn))
            flat = SC.load_npz(d / side["name"])
            out.append((doc, {k[2:]: v for k, v in flat.items() if k.startswith("t_")}))
        else:
            raise Ds1Refused("load_runs kind %r" % kind)
    return out


def score_record(arm, files, equivalence=None, reported=None, extra=None, before=None):
    """capacity/ds1_score_<arm>.json: dev and MD file names (relative to ds1/v1), sha256s, statuses and counts; the
    reproductions it rests on; the reported passes' counts only (ood22, ood23, ImageWeeds in ds1/v1/reported)."""
    rec = {"format": FMT_SCORE_RECORD, "family": "ds1", "arm": arm, "status": "complete",
           "files": [f for f in files if f.get("exam") == "dev" or ROOT_EXAM_RE.match(str(f.get("exam")))],
           "equivalence": {k: {"sha256": v.get("sha256"), "status": v.get("status")}
                           for k, v in sorted((equivalence or {}).items())},
           "reported": reported or {"status": "not run"},
           "trees": None if before is None else {"files": len(before), "sha256": C.sha256_text(json.dumps(
               before, sort_keys=True))},
           "decided_by": DECIDED_BY, "code": code_shas(), "written_utc": _utc()}
    rec.update(extra or {})
    write_json(capacity("ds1_score_%s.json" % arm.lower()), rec)
    return rec


def write_reported(arm, passes):
    write_json(root() / "reported" / ("%s.json" % arm), {"format": FMT_REPORTED, "arm": arm, "passes": passes,
                                                         "written_utc": _utc()})
    return {"status": "complete", "passes": len(passes),
            "failed": sum(1 for p in passes if p.get("status") == "failed")}


# ------------------------------------------------------------------ the reproductions (DS.7.1)
def repro_status(diff):
    return F.repro_status(diff)


def equiv_record(item, per, extra=None):
    """ds1/v1/equivalence/<item>.json written once: passed (all within 1e-6), deviation (within 0.002, recorded and
    the job continues) or refused (beyond: nothing written, the job refuses)."""
    sts = [repro_status(p.get("diff")) for p in per.values()]
    status = "refused" if "refused" in sts else ("deviation" if "deviation" in sts else "passed")
    if status == "refused":
        raise Ds1Refused("the reproduction %s refuses: %s" % (item, {k: v.get("diff") for k, v in per.items()}))
    doc = {"format": FMT_EQUIV, "item": item, "status": status, "per": per, "code": code_shas(),
           "tolerances": {"passed": REPRO_TOL, "deviation": DEVIATION_TOL}, "created_utc": _utc()}
    doc.update(extra or {})
    write_once(equiv_dir() / ("%s.json" % item), doc)
    return doc


def require_reproductions(items=REPRO_ITEMS):
    """{item: {sha256, status}} of the reproductions, each passed or a recorded deviation and taken by this code;
    no DS1 score is taken without them (DS.7.1)."""
    now = code_shas()
    out = {}
    for it in items:
        p = equiv_dir() / ("%s.json" % it)
        d = _read_json(p)
        if not isinstance(d, dict) or d.get("format") != FMT_EQUIV or d.get("status") not in ("passed", "deviation"):
            raise Ds1Refused("%s is missing or did not pass: no DS1 score is taken without it (DS.7.1)" % p)
        if d.get("code") != now:
            raise Ds1Refused("the code changed since %s was taken: it is moved aside and taken again, as a recorded "
                             "deviation" % p)
        out[it] = {"sha256": _sha(p), "status": d["status"]}
    return out


# ------------------------------------------------------------------ sources of weights
def build_source(arm, seed):
    """What seed k of a DS1 arm scores: a 1.2 M arm's base run (inc2.v3fam_systems.run_source on its experiment) or a
    long arm's exported final EMA weights (inc2.ds1_long's train record)."""
    from . import v3fam_systems as SYS
    if arm in LONG_ARMS:
        from . import ds1_long as LG
        return LG.source(arm, seed)
    exp = EXP_OF.get(arm)
    if exp is None:
        raise Ds1Refused("%s has no experiment" % arm)
    try:
        return SYS.run_source(exp, seed, testing_ok=TS._testing_ok())
    except F.V3famRefused as e:
        raise Ds1Refused(str(e))


def source_record(src):
    return {k: src[k] for k in ("exp", "arm", "seed", "run_id", "weights_sha256", "base_run_json_sha256",
                                "train_json_sha256", "recorded", "research_only") if k in src}


def md_exams():
    """The frozen MD exams: md_s0 first, then every frozen mdn_<source> of a source still in DS1 (a source rejected,
    role C or quarantined whole since its freeze leaves MD: ds1_data.sources_out; splits/ds1/md_v1/<exam>/
    frozen.json)."""
    from . import ds1_data as DD
    d = splits_dir() / "md_v1"
    gone = DD.sources_out() if d.is_dir() else set()
    got = sorted(p.parent.name for p in d.glob("*/frozen.json")
                 if int((_read_json(p) or {}).get("images") or 0) > 0
                 and (_read_json(p) or {}).get("source") not in gone) if d.is_dir() else []
    return (["md_s0"] if "md_s0" in got else []) + [e for e in got if e != "md_s0"]


def md_groups(exam):
    """{key: capture group} of an MD exam (splits/ds1/md_v1/groups.jsonl, written with its freeze)."""
    out = {}
    p = splits_dir() / "md_v1" / "groups.jsonl"
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("exam") == exam:
                    out[str(r["key"])] = str(r["group"])
    return out


def md_frozen(exam="md_s0"):
    d = _read_json(splits_dir() / "md_v1" / exam / "frozen.json")
    return isinstance(d, dict) and bool(d.get("frozen")) and (Path(d["root"]) / "splits" / "v1" / "LOCK.json").is_file()


# ------------------------------------------------------------------ scoring (L33)
def _kw(lock_check, batch, device):
    return {"lock_check": lock_check, "batch": batch, "device": device}


def _md_pass(src, exam, arm, kw):
    with tempfile.TemporaryDirectory(prefix="inc2_ds1_root_") as tmp:
        return pass_in_root(src["weights"], exam, "stage1", arm, Path(tmp) / "out", root_dir=md_root(exam), **kw)


def reproductions(embedder=None, workers=None, kw=None, seeds=SEEDS):
    """The three reproductions (DS.7.1), each taken once if missing: E1-A's three stage-1 dev records and E3-A's
    dev species records (E1-A's seeds through DS1's two-stage pass with E3's pinned classifier); B'1's ood23
    statistic (D1B1's seeds in stage1 mode under deviation 6's loader); an identity pass on MD equal to a plain
    inc.scorer.score on its root."""
    from ..inc import verify as V
    from . import v3fam_loader as LD
    from . import v3fam_systems as SYS
    kw = kw or {}
    done = {}
    if not (equiv_dir() / "e1a_dev.json").exists() or not (equiv_dir() / "e3a_dev.json").exists():
        ccfg, _r, shas = TS._clf_cfg()
        per1, per3 = {}, {}
        with TS._workers(workers) as wk:
            for s in seeds:
                src = TS._source("A", s, testing_ok=TS._testing_ok())
                emb = embedder or V.BioclipEmbedder()
                cfg = SYS._two_stage_cfg(ccfg, "A", emb, wk)
                ps = ds1_pass(src["weights"], "dev", cfg, "POWER", **kw)
                # E3-A's own dev file (twostage/e3_v1/arms/A/s<k>/dev.json): E1-A's boxes named by E3's classifier
                e3p = TS.arm_dir("A", s) / TS.file_names(TS.EXAM)[0]
                e3doc = _read_json(e3p) or {}
                want1 = (e3doc.get("stage1") or {}).get("agnostic_map50_95")
                want3 = e3doc.get("species_map50_95")
                per1[str(s)] = {"diff": None if want1 is None else abs(float(ps["stage1_agnostic"][0]) - float(want1)),
                                "new": ps["stage1_agnostic"][0], "recorded": want1, "recorded_file": str(e3p),
                                "recorded_sha256": _sha(e3p) if e3p.is_file() else None,
                                "run_recorded_agnostic": (src.get("recorded") or {}).get("agnostic_map50_95")}
                per3[str(s)] = {"diff": None if want3 is None else abs(float(ps["result"]["species_map50_95"]) -
                                                                       float(want3)),
                                "new": ps["result"]["species_map50_95"], "recorded": want3,
                                "recorded_file": str(e3p)}
        if not (equiv_dir() / "e1a_dev.json").exists():
            done["e1a_dev"] = equiv_record("e1a_dev", per1, {"what": "E1-A's stage-1 dev records"})
        if not (equiv_dir() / "e3a_dev.json").exists():
            done["e3a_dev"] = equiv_record("e3a_dev", per3, {"what": "E3-A's dev species records",
                                                             "classifier": shas})
    if not (equiv_dir() / "b1_ood23.json").exists():
        shapes = LD.preflight()
        per = {}
        for s in seeds:
            src = SYS.run_source(B1_EXP, s, testing_ok=TS._testing_ok())
            with LD.deciding("ood23") as obs:
                ps = ds1_pass(src["weights"], "ood23", {"mode": "stage1", "nms": NMS}, "S0", **kw)
            if obs is not None:
                obs.check(LD.exam_keys("ood23"), LD.changed_of(shapes, "ood23"), 1, "B'1 seed %d" % s)
            got = F.statistic_value(ps["p7"]["t"], "agnostic")
            # V3F's own D1B1 ood23 file (v3fam/v1/arms/D1B1/s<k>/ood23.json and its p7 arrays), taken under the same
            # loader fix: the statistic from its target reading's arrays, as V3F's verdict reads it
            want = v3f_p7_value(F.arm_dir("D1B1", s))
            if round(want, 4) != B1_OOD23[int(s)]:
                raise Ds1Refused("V3F's recorded ood23 statistic of B'1 seed %d is %.6f, not the registered %.4f"
                                 % (s, want, B1_OOD23[int(s)]))
            per[str(s)] = {"diff": abs(got - want), "new": got, "recorded": want, "registered_4dp": B1_OOD23[int(s)]}
        done["b1_ood23"] = equiv_record("b1_ood23", per, {"what": "B'1's ood23 statistic under deviation 6's loader"})
    if not (equiv_dir() / "md_identity.json").exists():
        src = TS._source("A", seeds[0], testing_ok=TS._testing_ok())
        draft = md_s0_draft()
        if draft is None:
            raise Ds1Refused("no MD-S0 draft: the identity reproduction on MD waits for L30's md_s0 step")
        with tempfile.TemporaryDirectory(prefix="inc2_ds1_root_") as tmp:
            ps = pass_in_root(src["weights"], "md_s0", "identity", "POWER", Path(tmp) / "out", plain=True,
                              root_dir=draft / "root", **kw)
        plain = ps.get("plain") or {}
        cmp_ = F.compare_with_recorded(ps["result"], plain, "MD identity")
        done["md_identity"] = equiv_record("md_identity", {"0": {"diff": cmp_["max_abs_diff"], "compare": cmp_}},
                                           {"what": "an identity pass on MD = a plain inc.scorer.score on its root"})
    return done


def _score_exam(arm, src, s, exam, kw, ccfg=None, shas=None, embedder=None, wk=None, shapes=None):
    """One pass of one arm-seed on one exam, committed (the kept file on a rerun)."""
    from ..inc import verify as V
    from . import v3fam_loader as LD
    from . import v3fam_systems as SYS
    got = existing(arm, s, exam)
    if got is not None:
        return got
    srec = source_record(src)
    if ROOT_EXAM_RE.match(exam):
        ps = _md_pass(src, exam, arm, kw)
        return commit(arm, s, exam, ps, {"mode": "stage1", "nms": NMS}, source=srec,
                      extra={"statistic": "weed AP50-95: the class-collapsed AP of the stage-1 boxes"})
    if exam == "dev":
        emb = embedder or V.BioclipEmbedder()
        cfg = SYS._two_stage_cfg(ccfg, NMS_REF, emb, wk)
        ps = ds1_pass(src["weights"], "dev", cfg, arm, **kw)
        return commit(arm, s, exam, ps, cfg, source=srec, classifier=shas,
                      extra={"statistic": "two-stage species mAP50-95 with E3's pinned classifier; the stage-1 "
                                          "agnostic AP beside"})
    cfg = {"mode": "stage1", "nms": NMS}
    x = {}
    with LD.deciding(exam) as obs:
        ps = ds1_pass(src["weights"], exam, cfg, arm, **kw)
    if obs is not None:
        x["loader_shape_fix"] = obs.check(LD.exam_keys(exam), LD.changed_of(shapes, exam), 1, "%s seed %d" % (arm, s))
    return commit(arm, s, exam, ps, cfg, source=srec, extra=x)


def score_arm(arm, embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None):
    """inc2.ds1 score --arm X (L33): every exam of the arm's row (DS.6), each seed's pass written once; dev and MD
    in capacity/ds1_score_<arm>.json, the other exams in ds1/v1/reported/<ARM>.json. POWER takes the reproductions
    first and writes capacity/ds1_power.json (DS.7.6)."""
    from . import v3fam_loader as LD
    if arm not in SCORE_ARMS:
        raise Ds1Refused("--arm %r is not one of DS1's score arms %s" % (arm, list(SCORE_ARMS)))
    check_frozen()
    require_timestamp()
    kw = _kw(lock_check, batch, device)
    t0 = time.time()
    with trees_unchanged() as before:
        if arm == "POWER":
            reproductions(embedder=embedder, workers=workers, kw=kw)
            rec = power(embedder=embedder, workers=workers, kw=kw, before=before)
            write_json(capacity("ds1_score_power.json"), {
                "format": FMT_SCORE_RECORD, "family": "ds1", "arm": "POWER", "status": "complete",
                "draft": rec.get("draft"), "enlarge": bool((rec.get("md_s0") or {}).get("enlarge")),
                "power_final": capacity("ds1_power.json").exists(), "decided_by": DECIDED_BY, "written_utc": _utc()})
            return rec
        eqs = require_reproductions()
        if not md_frozen("md_s0"):
            raise Ds1Refused("MD-S0 is not frozen: no arm is scored on it before its LOCK (DS.5.4)")
        exams = []
        for e in ARM_EXAMS[arm]:
            exams += md_exams()[1:] if e == "mdn" else [e]      # md_exams()[0] is md_s0
        ccfg, _r, shas = TS._clf_cfg()
        shapes = LD.preflight() if any(e in F.OOD_EXAMS for e in exams) else None
        files, passes = [], []
        with TS._workers(workers) as wk:
            for s in SEEDS:
                src = build_source(arm, s)
                for exam in exams:
                    deciding = exam in DECIDING_EXAMS or ROOT_EXAM_RE.match(exam)
                    if deciding:
                        files.append(_score_exam(arm, src, s, exam, kw, ccfg, shas, embedder, wk, shapes))
                    else:
                        try:
                            got = _score_exam(arm, src, s, exam, kw, ccfg, shas, embedder, wk, shapes)
                            passes.append({"seed": s, "exam": exam, "file": got["file"], "sha256": got["sha256"],
                                           "status": got["status"]})
                        except Exception as ex:          # noqa: BLE001 - a reported pass never fails the job
                            passes.append({"seed": s, "exam": exam, "status": "failed",
                                           "error": "%s: %s" % (type(ex).__name__, str(ex)[:400])})
        rec = score_record(arm, files, eqs, reported=write_reported(arm, passes), before=before,
                           extra={"exams": exams, "seconds": round(time.time() - t0, 1)})
    log("%s: %d deciding files, %d reported passes, %.0fs" % (arm, len(files), len(passes), time.time() - t0))
    return rec


def md_s0_draft():
    """The MD-S0 draft the power pre-flight reads: the frozen one once MD-S0 is frozen, else the latest drawn."""
    from . import ds1_data as DD
    fz = DD.frozen("md_s0")
    if fz:
        return DD.exam_dir("md_s0") / fz["draft"]
    drafts = sorted(DD.exam_dir("md_s0").glob("draft_r[0-9][0-9][0-9]")) if DD.exam_dir("md_s0").is_dir() else []
    return drafts[-1] if drafts else None


def power(embedder=None, workers=None, kw=None, before=None, write=True):
    """The power pre-flight (DS.7.6) on the latest MD-S0 draft, before the first training item: MD-S0's sigma_c from
    E1-A against b_v2_m640 (both trained on base_v2 only) with the capture-group bootstrap, its MDE (2.128 + 0.842)
    sigma_c and whether MD-S0 must be enlarged (MDE over 0.02); N1's power from E1-B against E1-A through the
    two-stage path on dev; ood23's MDE from V3F P7's sigma. ds1/v1/power/<draft>.json for every draft, and
    capacity/ds1_power.json (written once) for the draft that is not enlarged."""
    from ..inc import verify as V
    from . import v3fam_systems as SYS
    kw = kw or {}
    draft = md_s0_draft()
    if draft is None:
        raise Ds1Refused("no MD-S0 draft: L30's md_s0 step draws it first (DS.12 step 1)")
    groups = {}
    for line in (draft / "groups.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            groups[str(g["key"])] = str(g["group"])
    srcs = {k: {s: SYS.run_source(exp, s, testing_ok=TS._testing_ok()) for s in SEEDS} for k, exp in
            POWER_EXPS.items()}
    md_runs, dev_runs = {}, {}
    ccfg, _r, shas = TS._clf_cfg()
    pd = root() / "power" / draft.name
    with TS._workers(workers) as wk:
        for k in ("E1A", "BV2"):
            for s in SEEDS:
                d = pd / k / ("s%d" % s)
                if not (d / file_names("md_s0")[0]).exists():
                    with tempfile.TemporaryDirectory(prefix="inc2_ds1_root_") as tmp:
                        ps = pass_in_root(srcs[k][s]["weights"], "md_s0", "stage1", "POWER", Path(tmp) / "out",
                                          root_dir=draft / "root", **kw)
                    commit("POWER", s, "md_s0", ps, {"mode": "stage1", "nms": NMS}, source=source_record(srcs[k][s]),
                           d=d, extra={"power_system": k, "draft": draft.name})
            md_runs[k] = load_runs("POWER", "md_s0", "weed", base=pd / k)
        for k in ("E1A", "E1B"):
            for s in SEEDS:
                d = root() / "power" / ("%s_dev" % k) / ("s%d" % s)
                if not (d / file_names("dev")[0]).exists():
                    emb = embedder or V.BioclipEmbedder()
                    cfg = SYS._two_stage_cfg(ccfg, NMS_REF, emb, wk)
                    ps = ds1_pass(srcs[k][s]["weights"], "dev", cfg, "POWER", **kw)
                    commit("POWER", s, "dev", ps, cfg, source=source_record(srcs[k][s]), d=d, classifier=shas,
                           extra={"power_system": k})
            dev_runs[k] = load_runs("POWER", "dev", "species", base=root() / "power" / ("%s_dev" % k))
    rec = power_record(md_runs, dev_runs, groups, write=False)
    rec["draft"] = draft.name
    if write:
        write_once(root() / "power" / ("%s.json" % draft.name), rec)
        if not (rec.get("md_s0") or {}).get("enlarge"):
            write_once(capacity("ds1_power.json"), rec)
    return rec


def power_record(md_runs, dev_runs, groups, write=True):
    """capacity/ds1_power.json from the committed runs (md_runs {E1A, BV2: [(doc, weed arrays)]}, dev_runs {E1A,
    E1B: [(doc, species arrays)]}, groups {key: capture group})."""
    md = None
    if md_runs:
        xr, yr = [a for _d, a in md_runs["E1A"]], [a for _d, a in md_runs["BV2"]]
        keys = [str(k) for k in xr[0]["keys"]]
        missing = [k for k in keys if k not in groups]
        if missing:
            raise Ds1Refused("%d MD-S0 image(s) have no capture group (e.g. %s)" % (len(missing), missing[:3]))
        xv = [F.statistic_value(a, "agnostic") for a in xr]
        yv = [F.statistic_value(a, "agnostic") for a in yr]
        c = comparison("power/md_s0", xv, yv, xr, yr, "agnostic", [groups[k] for k in keys], "group", "max")
        sig = c["decision"]["sigma"]
        md = {"comparison": c, "sigma_c": sig, "mde": mde(sig, THRESHOLD["md_s0"]), "mde_max": MDE_MAX,
              "enlarge": mde(sig, THRESHOLD["md_s0"]) > MDE_MAX, "n_images": c["n_images"],
              "n_groups": len(set(groups[k] for k in keys))}
    dev = None
    if dev_runs:
        xr, yr = [a for _d, a in dev_runs["E1B"]], [a for _d, a in dev_runs["E1A"]]
        xv = [float(d_["species_map50_95"]) for d_, _a in dev_runs["E1B"]]
        yv = [float(d_["species_map50_95"]) for d_, _a in dev_runs["E1A"]]
        c = comparison("power/n1", xv, yv, xr, yr, "species")
        sig = c["decision"]["sigma"]
        dev = {"comparison": c, "sigma_c": sig, "n1_power_at_0": n1_power(sig), "mde_one_sided": mde(sig, Z90),
               "registered_range": [0.0085, 0.020]}
    rec = {"format": FMT_POWER, "status": "complete", "decided_by": DECIDED_BY, "md_s0": md, "dev": dev,
           "ood23": {"sigma_c": P7_SIGMA, "mde": mde(P7_SIGMA, THRESHOLD["ood23"]),
                     "source": "V3F-R P7 (D +0.0423, z 2.61): sigma = D / z (DS.7.6)"},
           "thresholds": THRESHOLD, "written_utc": _utc()}
    if write:
        write_once(capacity("ds1_power.json"), rec)
    return rec


# ------------------------------------------------------------------ the verdicts (L34)
def _values(runs, statistic):
    """Per-seed full-set values: the score files' species mAP50-95 (species), else the class-collapsed AP of the
    run's tie-broken arrays (weed, P7's target reading), as the bootstrap reads them."""
    if statistic == "species":
        return [float(d["species_map50_95"]) for d, _a in runs]
    return [F.statistic_value(a, "agnostic") for _d, a in runs]


def v3f_p7_value(d, exam="ood23"):
    """The P7 statistic (target-species agnostic AP, OtherPlant boxes as ignore regions) of a V3F score directory's
    committed p7 arrays, each file checked against its JSON's sha256."""
    doc = _read_json(Path(d) / F.file_names(exam)[0])
    side = (doc or {}).get("p7") if isinstance(doc, dict) else None
    if not isinstance(side, dict) or not side.get("name") or _sha(Path(d) / side["name"]) != side.get("sha256"):
        raise Ds1Refused("no recorded %s p7 arrays under %s (or they do not hash as recorded)" % (exam, d))
    flat = SC.load_npz(Path(d) / side["name"])
    return F.statistic_value({k[2:]: v for k, v in flat.items() if k.startswith("t_")}, "agnostic")


def decide_pair(cid, x_arm, y_arm, exam, loader=None, keep=None):
    """One comparison of DS1 between two arms' committed scores on an exam: MD (weed arrays, SE = max(image,
    capture group)), ood23 (P7's target reading, SE = max(image, session)), dev (species, image SE, session SE
    reported). loader(arm, exam, kind) -> [(doc, arrays)] (default load_runs)."""
    loader = loader or (lambda a, e, k: load_runs(a, e, k))
    if ROOT_EXAM_RE.match(exam):
        xr, yr = loader(x_arm, exam, "weed"), loader(y_arm, exam, "weed")
        keys = [str(k) for k in yr[0][1]["keys"]]
        g = md_groups(exam)
        return comparison(cid, _values(xr, "weed"), _values(yr, "weed"), [a for _d, a in xr], [a for _d, a in yr],
                          "agnostic", [g.get(k, k) for k in keys], "group", "max", keep=keep)
    if exam == "ood23":
        xr, yr = loader(x_arm, exam, "p7t"), loader(y_arm, exam, "p7t")
        keys = [str(k) for k in yr[0][1]["keys"]]
        ses = [F.session_of("ood23", k) for k in keys]
        return comparison(cid, _values(xr, "p7"), _values(yr, "p7"), [a for _d, a in xr], [a for _d, a in yr],
                          "agnostic", ses, "session", "max", keep=keep)
    if exam == "dev":
        xr, yr = loader(x_arm, exam, "species"), loader(y_arm, exam, "species")
        rows = C.read_manifest(C.manifest_path("dev"))
        keys = [str(k) for k in yr[0][1]["keys"]]
        ses = F.sessions_for("dev", rows, keys)
        return comparison(cid, _values(xr, "species"), _values(yr, "species"), [a for _d, a in xr],
                          [a for _d, a in yr], "species", ses, "session", "image", keep=keep)
    if exam in ("ood22", "imageweeds"):
        xr, yr = loader(x_arm, exam, "weed"), loader(y_arm, exam, "weed")
        return comparison(cid, _values(xr, "weed"), _values(yr, "weed"), [a for _d, a in xr], [a for _d, a in yr],
                          "agnostic", keep=keep)
    raise Ds1Refused("no DS1 comparison on %s" % exam)


def _primary(pid, arms_state, loader, x=None, y=None):
    p = PRIMARIES[pid]
    x, y = x or p["x"], y or p["y"]
    for a in (x, y):
        st = arms_state.get(a)
        if st != "done":
            return {"id": pid, "status": "not run", "why": "%s %s" % (a, st or "has no scores"), "x": x, "y": y}
    rec = decide_pair("%s/%s" % (pid, p["exam"]), x, y, p["exam"], loader)
    out = dict(primary_reading(pid, rec), id=pid, x=x, y=y, exam=p["exam"], comparison=rec)
    return out


def tier_s(pairs, arms_state, loader, mdes=None):
    """Tier S (DS.7.3): each pair (id, x, y, exam) decided by the same rules, with its MDE printed; never a claim."""
    out = {}
    for cid, x, y, exam in pairs:
        if arms_state.get(x) != "done" or arms_state.get(y) != "done":
            out[cid] = {"status": "not run", "x": x, "y": y, "exam": exam}
            continue
        try:
            rec = decide_pair(cid, x, y, exam, loader)
            sig = rec["decision"]["sigma"]
            out[cid] = {"status": "decided", "x": x, "y": y, "exam": exam, "comparison": rec,
                        "mde": mde(sig, Z90), "tier": "S (described, never decided)"}
        except Exception as e:                       # noqa: BLE001 - a Tier S comparison never fails the verdict
            out[cid] = {"status": "failed", "x": x, "y": y, "exam": exam, "error": "%s: %s" % (type(e).__name__,
                                                                                            str(e)[:300])}
    return out


TIER_S_PHASE1 = [("KT_over_KD/dev", "KT", "KD", "dev"), ("KF_over_KT/dev", "KF", "KT", "dev"),
                 ("KT_over_KD/md_s0", "KT", "KD", "md_s0"), ("KF_over_KT/md_s0", "KF", "KT", "md_s0"),
                 ("KD_over_K0/dev", "KD", "K0", "dev"), ("KT_over_K0/dev", "KT", "K0", "dev"),
                 ("KF_over_K0/dev", "KF", "K0", "dev"),
                 ("S1C_over_S1CR/md_s0", "S1C", "S1CR", "md_s0"), ("S1CR_over_S0/md_s0", "S1CR", "S0", "md_s0"),
                 ("S1C_over_S1CR/dev", "S1C", "S1CR", "dev"), ("S1CR_over_S0/dev", "S1CR", "S0", "dev"),
                 ("S1U_over_S0/md_s0", "S1U", "S0", "md_s0"), ("S1U_over_S0/dev", "S1U", "S0", "dev")]
TIER_S_PHASE1_EXT = [("S1U_over_S0/ood23", "S1U", "S0", "ood23"), ("S1C_over_S0/ood22", "S1C", "S0", "ood22"),
                     ("S1U_over_S0/ood22", "S1U", "S0", "ood22"), ("S0_over_K0/imageweeds", "S0", "K0", "imageweeds")]


def tier_s_phase2(arm12, armx):
    return [("%s_over_S0/md_s0" % arm12, arm12, "S0", "md_s0"), ("%s_over_S0/dev" % arm12, arm12, "S0", "dev"),
            ("%s_over_S1C/md_s0" % arm12, arm12, "S1C", "md_s0"), ("%s_over_%s/md_s0" % (armx, arm12), armx, arm12,
                                                                    "md_s0"),
            ("S0X_over_S0/md_s0", "S0X", "S0", "md_s0"), ("S0X_over_S0/dev", "S0X", "S0", "dev")]


VERDICT_ARMS = {"1": ("K0", "KD", "KT", "KF", "S0", "S1CR", "S1C", "S1U"), "2": ("S0X", "S2", "S2X"),
                "2_substituted": ("S0X", "S1UX")}


def _item_arm(item):
    """The arm an item names (build:S1C, score:K0, long:S2X:train -> S2X)."""
    return item.split(":")[1]


def verdict(phase, arms_state, loader=None, cap_refused=(), ticker_failed=(), on_date=None, s2_frozen=True,
            write=True, s2x_walltime=False):
    """L34: verdict v1 (P1, P2; Tier S of the phase-1 arms) or v2 (P3, P4, N1, H1; Tier S), each primary decided
    once, when its own arms' records are complete or failed. arms_state {ARM: done | failed | not run (...) |
    missing}. Refused while an arm of the phase is 'missing' (no score record and not recorded failed or not run): a
    primary is decided once, when its own records are complete (DS.7.2), never 'not run' because a verdict ran early.
    capacity/ds1_v<phase>.json (dev and MD only) and capacity/ds1_v<phase>_external.json (ood23, ood22, ImageWeeds),
    each written once."""
    loader = loader or (lambda a, e, k: load_runs(a, e, k))
    st = dict(arms_state)
    for it in list(cap_refused or ()):
        if it.startswith(("build:", "score:", "long:")):
            st[_item_arm(it)] = NOT_RUN_CAP
    for it in list(ticker_failed or ()):
        if it.startswith(("build:", "score:", "long:")):
            st[_item_arm(it)] = "failed"
    sub = substitution_due(on_date or today(), s2_frozen, s2x_walltime) if int(phase) == 2 else False
    need = VERDICT_ARMS["2_substituted" if sub else str(int(phase))]
    missing = [a for a in need if st.get(a) in (None, "missing")]
    if missing:
        raise Ds1Refused("verdict v%d waits for %s: no score record, and not recorded failed or not run (DS.7.2: "
                         "each primary is decided once, when its own records are complete)" % (int(phase), missing))
    platform, external = {}, {}
    if int(phase) == 1:
        prims = {pid: _primary(pid, st, loader) for pid in ("P1", "P2")}
        platform["primaries"] = prims
        platform["tier_s"] = tier_s(TIER_S_PHASE1, st, loader)
        external["tier_s"] = tier_s(TIER_S_PHASE1_EXT, st, loader)
        external["note"] = "ood23, ood22 and ImageWeeds readings (S arms on the season exams: season-free)"
    else:
        arm12, armx = phase2_arms(sub)
        prims = {"P3": _primary("P3", st, loader, x=armx, y="S0X")}
        p4 = _primary("P4", st, loader, x=armx, y="S0X")
        if p4.get("status") == "decided":
            keep = [k for k in loader("S0X", "ood23", "p7t")[0][1]["keys"] if str(k) not in OOD23_DISPUTED]
            p4["sensitivity_1771"] = decide_pair("P4/ood23_1771", armx, "S0X", "ood23", loader, keep=keep)
        n1 = h1 = None
        if st.get(armx) == "done" and st.get("S0X") == "done":
            rec = decide_pair("N1/dev", armx, "S0X", "dev", loader)
            n1 = dict(n1_reading(rec), comparison=rec)
            h1 = h1_reading(rec)
        platform.update(primaries=prims, n1=n1, h1=h1, substitution={"applied": sub, "arms": [arm12, armx],
                                                                     "s2x_walltime": bool(s2x_walltime),
                                                                     "rule": "DS.0 item 17, DS.6, DS.7.2"})
        platform["tier_s"] = tier_s(tier_s_phase2(arm12, armx), st, loader)
        external["primaries"] = {"P4": p4}
        external["tier_s"] = tier_s([("%s_over_S0/ood23" % arm12, arm12, "S0", "ood23"),
                                     ("%s_over_S0X/ood22" % armx, armx, "S0X", "ood22")], st, loader)
    doc = {"format": FMT_VERDICT, "status": "decided", "phase": int(phase), "exam": "md_s0 and dev",
           "arms": st, "thresholds": {"md_s0": THRESHOLD["md_s0"]}, "decided_by": DECIDED_BY, "code": code_shas(),
           "written_utc": _utc()}
    doc.update(platform)
    ext = {"format": FMT_EXTERNAL, "phase": int(phase), "status": "decided", "never_admitted": True,
           "thresholds": THRESHOLD, "decided_by": DECIDED_BY, "written_utc": _utc()}
    ext.update(external)
    if int(phase) == 2:
        doc["reads"] = {"order": read_order(_read_json(capacity("ds1_v1.json")), doc,
                                            platform["substitution"]["applied"]), "cap": READ_CAP}
    if write:
        write_once(capacity("ds1_v%d_external.json" % int(phase)), json.loads(json.dumps(ext, default=str)))
        write_once(capacity("ds1_v%d.json" % int(phase)), json.loads(json.dumps(doc, default=str)))
    return doc, ext


def arms_state_from_records():
    """{ARM: done | failed | not run (...) | missing} from capacity/ds1_score_<arm>.json, KD's and KT's not-run record
    (splits/ds1/arms/k_not_run.json: the K supply, or the K source in no manifest) and the long path's smokes (a long
    arm recorded 'not run (walltime)', DS.6); the ticker's argv adds the items it recorded failed or capped."""
    from . import ds1_long as LG
    out = {}
    for a in SCORE_ARMS:
        r = _read_json(capacity("ds1_score_%s.json" % a.lower()))
        out[a] = "done" if isinstance(r, dict) and r.get("status") == "complete" else (
            "not run (%s)" % r.get("reason") if isinstance(r, dict) and r.get("status") == "not run" else "missing")
    knr = _read_json(splits_dir() / "arms" / "k_not_run.json")
    if isinstance(knr, dict):
        for a in knr.get("arms") or ():
            if out.get(a) == "missing":
                out[a] = str(knr.get("status") or "not run")
    for a in LONG_ARMS:
        sm = _read_json(LG.arm_dir(a) / "smoke.json")
        if isinstance(sm, dict) and str(sm.get("status") or "").startswith("not run") and out.get(a) == "missing":
            out[a] = str(sm["status"])
    return out


def s2x_walltime_not_run():
    """True when S2x's smoke recorded it 'not run (walltime)' (DS.6): the substitution of DS.7.2 applies."""
    from . import ds1_long as LG
    sm = _read_json(LG.arm_dir("S2X") / "smoke.json")
    return isinstance(sm, dict) and str(sm.get("status") or "").startswith("not run")


# ------------------------------------------------------------------ sealed reads (DS.7.5)
def reads_dir():
    return root() / "reads"


def reads_done():
    out = {}
    d = reads_dir()
    if d.is_dir():
        for p in sorted(d.glob("*/read.json")):
            r = _read_json(p)
            if isinstance(r, dict) and r.get("format") == FMT_READ:
                out[(r["arm"], r["exam"])] = "read"
    return out


def test_read(arm, exam, who, v1=None, v2=None):
    """A person's or the delegate's sealed read (DS.7.5): checked against the read order the verdicts fix, the cap
    and read_by; writes ds1/v1/reads/<n>_<arm>_<exam>/read.json once and returns the sbatch argv the named person
    submits (score-test)."""
    v1 = v1 if v1 is not None else _read_json(capacity("ds1_v1.json"))
    v2 = v2 if v2 is not None else _read_json(capacity("ds1_v2.json"))
    sub = bool(((v2 or {}).get("substitution") or {}).get("applied"))
    order = read_order(v1, v2, sub)
    chk = check_read(arm, exam, order, reads_done())
    by = read_by_value(who)
    d = reads_dir() / ("%d_%s_%s" % (chk["position"], arm, exam))
    doc = {"format": FMT_READ, "arm": arm, "exam": exam, "kind": chk["kind"], "position": chk["position"],
           "read_by": by, "order": order, "verdicts": {"v1": _sha(capacity("ds1_v1.json")),
                                                       "v2": _sha(capacity("ds1_v2.json"))},
           "written_utc": _utc()}
    write_once(d / "read.json", doc)
    argv = ["sbatch", "--export=INCAP_DECIDED_BY", "-p", "GPU-shared", "--job-name=inc_build_ds1test_%s_%s" % (
        arm.lower(), exam), "run_inc2_build.sh", "inc2.ds1", "score-test", "--arm", arm, "--exam", exam]
    return doc, argv


def read_record(arm, exam):
    """(path, record) of the prepared read of arm on exam, or refused."""
    for p in sorted(reads_dir().glob("*_%s_%s/read.json" % (arm, exam))) if reads_dir().is_dir() else []:
        r = _read_json(p)
        if isinstance(r, dict) and r.get("format") == FMT_READ and r.get("arm") == arm and r.get("exam") == exam:
            return p, r
    raise Ds1Refused("no read of %s on %s was prepared (inc2.ds1 test-read)" % (arm, exam))


def test_dir(arm, exam):
    return root() / "test" / ("%s_%s" % (arm, exam))


def mtw_roots():
    """[(exam, root, stratum)] of MT-W's sealed parts: test v1 in Z1's sealed root (its SIU stratum apart), MT-W v1.1
    when the person's check corrected test v1 labels, and each frozen MT-W v2 source root."""
    from . import ds1_data as DD
    from . import zoo as Z
    out = [("test_v1", Z.root_dir("v1"), "test_v1")]
    v11 = DD.out_dir() / "mtw_v1_1" / "root"
    if (v11 / "splits" / "v1" / "LOCK.json").is_file():
        out.append(("mtw_v2", v11, "test_v1.1"))
    gone = DD.sources_out() if DD.out_dir().is_dir() else set()
    for p in sorted(DD.out_dir().glob("mtw_v2/mtw_*/frozen.json")) if DD.out_dir().is_dir() else []:
        d = _read_json(p) or {}
        if d.get("source") in gone:
            continue
        out.append(("mtw_v2", Path(d["root"]), "mtw_v2:%s" % d.get("source")))
    return out


def mtw_groups():
    """{key: capture group} of MT-W's images (DS.7.5's capture-group SE): test v1's rows (base3's holdout lists,
    their group), each frozen MT-W v2 source's rows (their draw's group)."""
    from . import base3 as B3
    from . import ds1_data as DD
    out = {}
    for f in sorted((B3.out_dir() / B3.HOLDOUT_DIR).glob("*.jsonl")) if (B3.out_dir() / B3.HOLDOUT_DIR).is_dir() \
            else []:
        for r in DD.read_jsonl(f):
            if r.get("group") is not None:
                out[str(r["key"])] = "v1:%s" % r["group"]
    for p in sorted(DD.out_dir().glob("mtw_v2/mtw_*/draft/rows.jsonl")) if DD.out_dir().is_dir() else []:
        for r in DD.read_jsonl(p):
            if r.get("group") is not None:
                out[str(r["key"])] = "v2:%s" % r["group"]
    return out


def score_test(arm, exam, lock_check=True, batch=S.BATCH, device=None, workers=None, embedder=None):
    """The sealed read a person submitted (DS.7.5): every seed of the arm on MT-W (test v1 in Z1's sealed root, MT-W
    v1.1 if corrected, every frozen MT-W v2 source root; stage-1 weed arrays) and evalgroups_v1 in the same job, or
    on the cwd12 sealed test (two-stage species with E3's pinned classifier). Written once under ds1/v1/test/."""
    from ..inc import verify as V
    from . import v3fam_systems as SYS
    from . import zoo as Z
    rp, rd = read_record(arm, exam)
    d = test_dir(arm, exam)
    if (d / "done.json").exists():
        raise Ds1Refused("%s on %s was scored already: a read is taken once" % (arm, exam))
    check_frozen()
    kw = _kw(lock_check, batch, device)
    files = []

    def kept(dd, ex):
        # a part committed by an earlier submission of this same read (a node failed after it): kept, never scored
        # twice, so a re-submission completes the read instead of refusing at that file
        js = Path(dd) / file_names(ex)[0]
        return entry(js, "kept") if js.exists() else None
    with trees_unchanged():
        for s in SEEDS:
            src = build_source(arm, s)
            if exam == "cwd12":
                got = kept(d / ("s%d" % s), "test")
                if got is not None:
                    files.append(got)
                    continue
                ccfg, _r, shas = TS._clf_cfg()
                with TS._workers(workers) as wk, test_token() as tok:
                    emb = embedder or V.BioclipEmbedder()
                    cfg = SYS._two_stage_cfg(ccfg, NMS_REF, emb, wk)
                    ps = ds1_pass(src["weights"], "test", cfg, arm, test_token_=tok, **kw)
                files.append(commit(arm, s, "test", ps, cfg, source=source_record(src), classifier=shas,
                                    d=d / ("s%d" % s), extra={"read": str(rp)}))
                continue
            parts = mtw_roots() + [("evalgroups_v1", Z.root_dir("v1"), "evalgroups_v1")]
            for ex, root_, stratum in parts:
                got = kept(d / ("s%d" % s) / re.sub(r"[^A-Za-z0-9_.-]", "_", stratum), ex)
                if got is not None:
                    files.append(got)
                    continue
                with tempfile.TemporaryDirectory(prefix="inc2_ds1_sealed_") as tmp:
                    ps = pass_in_root(src["weights"], ex, "stage1", arm, Path(tmp) / "out", root_dir=root_,
                                      read_json=rp, **kw)
                dd = d / ("s%d" % s) / re.sub(r"[^A-Za-z0-9_.-]", "_", stratum)
                files.append(commit(arm, s, ex, ps, {"mode": "stage1", "nms": NMS}, source=source_record(src), d=dd,
                                    extra={"read": str(rp), "stratum": stratum, "root": str(root_)}))
        write_once(d / "done.json", {"format": FMT_TEST, "arm": arm, "exam": exam, "read": str(rp),
                                     "read_sha256": _sha(rp), "files": files, "written_utc": _utc()})
    return files


def mean_se(runs, groups=None, cid="mtw", resamples=None):
    """(image SE, max(image, capture-group) SE) of the 3-seed mean of a class-collapsed AP (DS.7.5's 0.90 interval):
    the sd over resamples of the seeds' mean, the images (or the groups) drawn with replacement, one draw for every
    seed (seed texts inc2/ds1/<cid>/image and /group)."""
    import numpy as np
    resamples = RESAMPLES if resamples is None else int(resamples)
    n = len(runs[0]["keys"])

    def sd_of(counts):
        V = F.run_values(runs, counts, "agnostic")
        m = [float(np.mean(V[r])) for r in range(len(V)) if not np.isnan(V[r]).any()]
        return float(np.std(m, ddof=1)) if len(m) >= 2 else float("nan")
    se_img = sd_of(F.image_counts(n, seed_text(cid, "image"), resamples))
    se = se_img
    if groups:
        keys = [str(k) for k in runs[0]["keys"]]
        se = max(se_img, sd_of(F.session_counts([groups.get(k, k) for k in keys], seed_text(cid, "group"),
                                                resamples)))
    return se_img, se


def merge_weed(arrs):
    """One set of weed arrays over several exams' images (their keys distinct): MT-W's pooled reading."""
    import numpy as np
    keys, tp, conf, img, ngt = [], [], [], [], []
    off = 0
    for a in arrs:
        keys += [str(k) for k in a["keys"]]
        tp.append(np.asarray(a["tp"]))
        conf.append(np.asarray(a["conf"]))
        img.append(np.asarray(a["pred_img"], dtype=np.int64) + off)
        ngt.append(np.asarray(a["n_gt"], dtype=np.int64))
        off += len(a["keys"])
    if len(set(keys)) != len(keys):
        raise Ds1Refused("MT-W's parts share an image key")
    return {"keys": np.asarray(keys), "tp": np.concatenate(tp, 0), "conf": np.concatenate(conf),
            "pred_img": np.concatenate(img), "n_gt": np.concatenate(ngt)}


def test_report(arm, exam, groups=None, write=True):
    """capacity/ds1_test_<arm>_<exam>.{json,md} (never admitted): MT-W's weed AP50-95 per seed, its 3-seed mean, the
    0.90 reading (reaches: mean >= 0.9000; clears: the lower end of mean - 1.645 sqrt(SE^2 + sd^2/3) >= 0.9000 with
    SE = max(image, capture-group SE of the 3-seed mean)), the strata (test v1, v1.1, MT-W v2 per source, the SIU
    in-plant stratum apart), evalgroups_v1 beside; or the cwd12 sealed test's species mAP50-95 (descriptive)."""
    import numpy as np
    d = test_dir(arm, exam)
    done = _read_json(d / "done.json")
    if not isinstance(done, dict):
        raise Ds1Refused("%s on %s has no completed score-test" % (arm, exam))
    out = {"format": FMT_TEST, "arm": arm, "exam": exam, "read": done.get("read"), "decided_by": DECIDED_BY,
           "never_admitted": True, "written_utc": _utc()}
    if exam == "cwd12":
        runs = load_runs(arm, "test", "species", base=d)
        vals = [float(doc["species_map50_95"]) for doc, _a in runs]
        out.update(species_map50_95=vals, mean=statistics.fmean(vals), gap_to_090=CLEAR_090 - statistics.fmean(vals),
                   descriptive=True)
    else:
        per_seed, strata = [], collections.defaultdict(list)
        siu = None
        for s in SEEDS:
            arr, evg = {}, None
            for sd in sorted((d / ("s%d" % s)).iterdir()):
                if not sd.is_dir():
                    continue
                for jn in sd.glob("*.json"):
                    doc = _read_json(jn)
                    if not isinstance(doc, dict) or doc.get("format") != FMT_SCORE:
                        continue
                    w = SC.load_npz(sd / doc["weed"]["name"])
                    st = doc.get("stratum")
                    if st == "evalgroups_v1":
                        evg = F.statistic_value(w, "agnostic")
                        continue
                    if st == "test_v1":
                        man = C.read_manifest(Path(doc["root"]) / "splits" / "v1" / "test_v1.jsonl")
                        siu_keys = {r["key"] for r in man if r.get("source") == MTW_SIU_SOURCE}
                        if siu_keys:
                            siu = siu or []
                            siu.append(F.statistic_value(F.subset_arrays(w, sorted(siu_keys)), "agnostic"))
                        w = F.subset_arrays(w, [k for k in w["keys"] if str(k) not in siu_keys])
                    arr[st] = w
                    strata[st].append(F.statistic_value(w, "agnostic"))
            # the headline pools test v1's checked labels (v1.1) when the check corrected any, else test v1, with
            # every MT-W v2 source; the other test v1 reading is a stratum beside it (DS.4)
            v1 = "test_v1.1" if "test_v1.1" in arr else "test_v1"
            parts = [arr[v1]] + [a for k, a in sorted(arr.items()) if k.startswith("mtw_v2:")]
            pooled = merge_weed(parts)
            per_seed.append({"pooled": pooled, "value": F.statistic_value(pooled, "agnostic"), "evalgroups_v1": evg,
                             "test_v1_part": v1})
        vals = [p["value"] for p in per_seed]
        if groups is None:
            groups = mtw_groups()
        se_img, se = mean_se([p["pooled"] for p in per_seed], groups, "mtw/%s" % arm)
        out.update(values=vals, mean=statistics.fmean(vals), reading=reaches_clears(vals, se), se_image=se_img,
                   se=se, se_rule="max(image, capture-group) of the 3-seed mean (DS.7.5)",
                   groups=len(set(groups.values())) if groups else 0,
                   strata={k: v for k, v in strata.items()}, siu_in_plant_stratum=siu,
                   evalgroups_v1=[p["evalgroups_v1"] for p in per_seed], headline="weed AP50-95 on MT-W")
    if write:
        write_once(capacity("ds1_test_%s_%s.json" % (arm.lower(), exam)), json.loads(json.dumps(out, default=str)))
        md = ["# DS1 sealed read: %s on %s" % (arm, exam), "", "Never admitted to the platform's evidence.", ""]
        if exam == "mtw":
            r = out["reading"]
            md.append("Weed AP50-95 on MT-W: %s (mean %.4f); reaches 0.90: %s; clears 0.90: %s (lower 90 %% end "
                      "%.4f; SE %.4f)" % (", ".join("%.4f" % v for v in out["values"]), out["mean"], r["reaches"],
                                          r["clears"], r["lower_90"], out["se"]))
        else:
            md.append("cwd12 sealed test, two-stage species mAP50-95: %s (mean %.4f; gap to 0.90 %.4f; descriptive)" % (
                ", ".join("%.4f" % v for v in out["species_map50_95"]), out["mean"], out["gap_to_090"]))
        write_once(capacity("ds1_test_%s_%s.md" % (arm.lower(), exam)), ("\n".join(md) + "\n").encode("utf-8"))
    return out


def spend():
    """The family's GPU-h as the cluster's own records give it (read-only; the ticker's count is the binding one)."""
    out = {"score": {}, "long": {}}
    for a in SCORE_ARMS:
        r = _read_json(capacity("ds1_score_%s.json" % a.lower()))
        if isinstance(r, dict):
            out["score"][a] = r.get("seconds")
    return out


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="DS1's core: scores (L33), verdicts (L34), RA, LN, reads.")
    ap.add_argument("command", choices=("score", "verdict", "ra", "ln", "test-read", "score-test", "test-report",
                                        "timestamp", "timestamp-revision", "spend", "_pass"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--phase", type=int, choices=(1, 2), default=None)
    ap.add_argument("--exam", choices=("mtw", "cwd12"), default=None)
    ap.add_argument("--read-by", default=None)
    ap.add_argument("--report", action="append", default=None)
    ap.add_argument("--cap-refused", default="")
    ap.add_argument("--ticker-failed", default="")
    ap.add_argument("--proof", default=None)
    ap.add_argument("--stamp-proof", default=None)
    ap.add_argument("--revision-proof", action="append", default=None)
    ap.add_argument("--revision-stamp-proof", action="append", default=None)
    ap.add_argument("--revision", type=int, default=None, help="timestamp-revision: the Revision's number")
    ap.add_argument("--spec", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        if a.command == "score":
            score_arm(a.arm)
        elif a.command == "verdict":
            if a.phase is None:
                raise Ds1Refused("verdict needs --phase 1|2")
            require_timestamp()
            with trees_unchanged():
                fr = _read_json(splits_dir() / "summary.json") or {}
                verdict(a.phase, arms_state_from_records(), cap_refused=[x for x in a.cap_refused.split(",") if x],
                        ticker_failed=[x for x in a.ticker_failed.split(",") if x],
                        s2_frozen=bool(((fr.get("arms") or {}).get("S2") or {}).get("sha256")),
                        s2x_walltime=s2x_walltime_not_run())
        elif a.command == "ra":
            ra(a.report or ra_reports())
        elif a.command == "ln":
            from . import ds1_sheets as SH
            rec = ln_statistic(SH.ln_marks())
            write_once(capacity("ds1_ln.json"), dict(rec, format=FMT_LN, status="complete", decided_by=DECIDED_BY,
                                                     written_utc=_utc()))
        elif a.command == "test-read":
            doc, argv_ = test_read(a.arm, a.exam, a.read_by)
            print(json.dumps({"read": doc, "submit": argv_}, indent=1))
        elif a.command == "score-test":
            score_test(a.arm, a.exam)
        elif a.command == "test-report":
            print(json.dumps(test_report(a.arm, a.exam), indent=1, default=str)[:4000])
        elif a.command == "timestamp":
            rp_, rsp_ = a.revision_proof or [], a.revision_stamp_proof or []
            if len(rp_) != len(rsp_):
                raise Ds1Refused("--revision-proof and --revision-stamp-proof come in pairs")
            print(json.dumps(timestamp_record(a.proof, a.stamp_proof, revisions=list(zip(rp_, rsp_))), indent=1))
        elif a.command == "timestamp-revision":
            print(json.dumps(timestamp_revision(a.revision, a.proof, a.stamp_proof), indent=1))
        elif a.command == "spend":
            print(json.dumps(spend(), indent=1, sort_keys=True))
        else:
            _run_pass_spec(a.spec)
    except (Ds1Refused, F.V3famRefused, TS.E3Refused, ValueError) as e:
        print("[inc2.ds1] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
