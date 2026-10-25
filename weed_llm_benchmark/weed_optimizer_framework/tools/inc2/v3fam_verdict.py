"""V3F's verdict and its sealed-test reads (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered
as one family)", V3F.3.3-V3F.3.6, V3F.5, V3F.6.1).

    python -m weed_optimizer_framework.tools.inc2.v3fam_verdict verdict [--cap-refused ITEM,...] [--ticker-failed ITEM,...]
                                                                                    (L23L; a job; deviation 5)
    python -m weed_optimizer_framework.tools.inc2.v3fam verdict [--cap-refused ITEM,...]   (the same verb, a person's)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-read --arm X --read-by owner|delegate  (login node)
    sbatch run_inc2_build.sh inc2.v3fam score-test --arm X        (the person test-read names; INCAP_DECIDED_BY)
    python -m weed_optimizer_framework.tools.inc2.v3fam test-report --arm X
    sbatch run_inc2_build.sh inc2.v3fam reanalyse-test --read-by owner|delegate   (a person's; INCAP_DECIDED_BY)
    python -m weed_optimizer_framework.tools.inc2.v3fam spend      (the family's GPU-h as the cluster's records give it)

The verdict (L23L, once every arm's record is complete or failed). Every
side of every comparison is a set of committed family scores
(inc2.v3fam_systems.family_runs: each JSON and its per-image arrays checked
against the sha256 it records), so both sides went through the one family
path (decision 17). Each pre-declared comparison (COMPARISONS) is computed
by inc2.v3fam.comparison: D (D_sel for P2, P3, P5), 2 pooled sd, the image
SE, the session SE (dev's 8 sessions reported beside; deciding as the larger
of the two for P6-P9), z, the 90 % interval and the outcome class; a side
whose arm was not run, failed or was refused at the family cap is "not run"
with its reason, and keeps its place (Holm's m, C = 14). Then Holm per exam
(inc2.v3fam.holm_overlay), the Tier C gate (challenger_gate: plain D over
E3-B, z > Phi^-1(1 - 0.05/14)) and the order of the sealed-test reads
(read_order: BCLEAN if D0-S triggered it, the reference reads D6 then D8-m
when their arms were scored, the challengers that open a read; D6 never takes
a challenger place). D3's Tier C comparison is P4's record (V3F.5 D3: "the
same comparison"), never a second bootstrap. When D0-S found a dev image
with a true copy, every dev comparison (P1-P5, Tier C, Tier S) is decided on
dev minus H with its full-dev numbers beside, and E3's recorded dev
comparisons (E3-M, E3-A, E3-B over the reference under E3's seed text, and
E3-B over E3-A under its attribution text) are re-read on dev minus H from
E3RS's arrays, reported only (V3F.5 D0-S). Tier S also holds the readings
V3F.5 names that need their own arrays: the stage-1 agnostic AP of D3's
fused boxes against E3-B's seeds and of D4's detectors against E1-A's and
E1-B's (each committed stage-1 box set matched class-collapsed to dev's
ground truth, its full-set value checked against the one its score records),
D1's species subsets (ood23's three species with >= 100 GT boxes, ood22's
eight), and D5b's inflation per copy family and its 12-class mean. A
comparison a side of which lost training rows to D0-S's removals carries
that declaration (V3F.3.7 item 4). Arms the ticker refused at the family
cap reach the verdict in its argv (--cap-refused) and are recorded "not run
(family cap)" under v3fam/v1/cap before anything is decided.

Records (V3F.3.5): capacity/v3fam_v1.json, the platform's (dev only: P1-P5,
Holm on dev, Tier C, the read order, Tier S on dev, the dev pass count),
written once and last; capacity/v3fam_v1_external.json (P6-P9, Holm on
ood22, ood23 and the probe, Tier S on those exams); capacity/v3fam_ext_v1.json
(ImageWeeds agnostic and Ragweed AP, ooddev_v1 agnostic, per arm, reported);
capacity/v3fam_v1_report.{json,md} (the family table). The platform's
evidence admits only the first.

Sealed-test reads (V3F.3.6). test-read checks the verdict, that the arm is
next in the read order (check_read; a read refused at the family cap is
recorded "not run (family cap)" and not retried), the family cap
(check_cap over the family's GPU-h as the cluster's records give it: spend
items, a read prepared but not yet submitted at its estimate), pins the
verdict, the arm's weights and classifier by sha256, and
writes v3fam/v1/test/<X>/read.json once with read_by exactly "Harry He
(owner)" or "Claude (delegate under the 2026-09-30 owner grant)" and the one
argv. score-test refuses unless INCAP_DECIDED_BY names that person (the
owner's account, human:..., or assistant:claude-delegate for the delegate),
then reproduces b_v2_m640's three recorded test scores through the family
identity path (1e-6, 0.002 as a recorded deviation, else refused before any
family test score), then scores the arm once (written once; a second run
refuses). test-report states the gap to 0.90, the stratum of the test images
from dev's sessions (dev_session_stratum), read_by and the kind of read
(capacity/v3fam_test_<x>.{json,md}). reanalyse-test (no inference), the last
of the reads (decision 25: every read of the order taken or recorded not run
first; its own job's running attempt not counted twice against the cap),
reads E3's stored test arrays: O4's stratum for E3-A, E3-M, E3-B and the
reference, and test minus H when D0-S found test copies
(capacity/v3fam_retest.json); it runs inside e_trees_unchanged.

Deviation 4 (2026-10-20; docs/prereg/V3F_dev4_2026-10-20.md). The verdict
refuses unless every arm's record is complete or failed (verdict_ready: the
family's data summary complete, and no arm that never started: no score or
cap record, no attempt in the provenance of its score job, its builds or
D6's jobs, and no failure of one of those the ticker recorded after its
second failure, --ticker-failed, written once under v3fam/v1/failed). The
verdict of 2026-10-13 (capacity/v3fam_v1.json, sha256
PARTIAL_VERDICT_SHA256) was decided while seven arms had never started (D1,
D4F, D4N, D4R15, D5B, D6, D8; BCLEAN, also recorded missing, was not needed:
D0-S did not trigger it) and stays as it is; when that file is the one in
place, the verb writes the family's continuation verdict (CONTINUATION:
capacity/v3fam_v1_cont.json, its external, ext and report files) by the
same computation, once, after checking that it reproduces every comparison
the 2026-10-13 records decided (to 1e-9, the same outcome; a primary
family-confirmed then stays so), and the reads follow the continuation
(family_verdict_path): before it, every read refuses.

Deviation 5 (2026-10-24; docs/prereg/V3F_dev5_2026-10-24.md). Deviation 4
changed inc2/v3fam.py (its CLI only: the --ticker-failed option and the verdict
call it feeds), so every reproduction record of 2026-10-12, which pins that
file's sha256 (inc2.v3fam.code_shas), refused every score job of D1, D4F,
D4N, D4R15, D5B and D8 on 2026-10-21; the ticker counted those refusals as
the arms' failures, and the continuation of 2026-10-22 (capacity/
v3fam_v1_cont.json, sha256 CONT_SHA256) recorded the six arms 'failed twice'
and D6 'missing'. inc2/v3fam.py is restored byte for byte (sha256
3d776dae...) and the verdict's CLI lives here (main: python -m
...inc2.v3fam_verdict verdict), outside code_shas. The continuation of
2026-10-22 stays as it is. When it is the one in place, the verb writes the
family's second continuation (CONTINUATION_2: capacity/v3fam_v1_cont2.json,
its external, ext and report files) by the same computation, once, after
checking that it reproduces every comparison the 2026-10-13 records and the
2026-10-22 records decided (continuation_check over both: to 1e-9, the same
outcome; every primary either family-confirmed stays so); it writes nothing
under an earlier verdict's names, and the reads follow it. It refuses while
any of the six arms has no record since deviation 5 reopened it (DEV5_ARMS:
complete, not run, refused at the family cap, or failed after the reopening
as the ticker records it); the attempts of 2026-10-21 do not count. The
ticker's failure records of 2026-10-22 for the six arms (inc2.v3fam_supersede,
by sha256) are moved aside before the verdict runs and are never read as
failures. D6 with no score reports its smoke's own 'not run (walltime)'
(V3F.5 D6, V3F.3.4) and BCLEAN 'not needed' without D0-S's trigger, never
'missing'.

A refusal raises inc2.v3fam.V3famRefused (V3famCapRefused at the family cap).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import recipes as RC
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_supersede as SUP
from . import v3fam_systems as SYS

# ------------------------------------------------------------------ constants (pre-registered)
VERDICT = "v3fam_v1.json"                    # under capacity/: the platform's record (dev only)
EXTERNAL = "v3fam_v1_external.json"
EXT = "v3fam_ext_v1.json"
REPORT = "v3fam_v1_report"
RETEST = "v3fam_retest.json"
TEST_RECORD = "v3fam_test_%s"                # .json / .md, lower-case arm
FMT_VERDICT = "inc2-v3fam-verdict/1"
FMT_EXTERNAL = "inc2-v3fam-external/1"
FMT_EXT = "inc2-v3fam-ext/1"
FMT_REPORT = "inc2-v3fam-report/1"
FMT_READ = "inc2-v3fam-test-read/1"
FMT_TEST_REPORT = "inc2-v3fam-test-report/1"
FMT_RETEST = "inc2-v3fam-retest/1"
TARGET = 0.90
READ_GPUH = 0.5                              # one test read (V3F.7 step 12)
RETEST_GPUH = 0.1
BCLEAN_SEED_TEXT = "inc2/v3fam/bclean/species_se"     # BCLEAN's rule against b_v2_m640 (V3F.5 D0-S)
DECIDED_BY_OWNER_PREFIX = "human:"
DELEGATE_ACCOUNT = F.DECIDED_BY_ENV["delegate"]
SUBMITTER_ENV = "INCAP_DECIDED_BY"           # the read's submitter, named in read.json's sbatch --export
REF_TEST_DIR = "REF"

# The sides (V3F.5): {name: (arms/<dir>, variant, one-system ens or None, the score arm whose record it is)}
SIDES = {"E3M": ("E3M", "", None, "E3RS"), "E3A": ("E3A", "", None, "E3RS"), "E3B": ("E3B", "", None, "E3RS"),
         "REF": ("REF", "", None, "E3RS"), "ORACLE": ("E3B", SYS.ORACLE_VARIANT, None, "E3RS"),
         "D3": ("D3", "", "ens", "D3"), "D6": ("D6", "", None, "D6"), "D8S": ("D8S", "", None, "D8"),
         "D8M": ("D8M", "", None, "D8"), "BCLEAN": ("BCLEAN", "", None, "BCLEAN"),
         "D1A1": ("D1A1", "", None, "D1"), "D1B1": ("D1B1", "", None, "D1"), "D1A12": ("D1A12", "", None, "D1"),
         "TSA": ("TSA", "", None, "D1"), "TSB": ("TSB", "", None, "D1"),
         "D5B0": ("D5B0", "", None, "D5B"), "D5B25": ("D5B25", "", None, "D5B"), "D5B100": ("D5B100", "", None, "D5B")}
for _a in ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
           "D4F", "D4N", "D4R15"):
    SIDES[_a] = (_a, "", None, _a)
LABELS = {"E3M": "E3-M", "E3A": "E3-A", "E3B": "E3-B", "REF": "b_v2_m640 (YOLO11m reference)",
          "ORACLE": "E3-B's boxes, perfect naming (O5)", "D2LAION": "LAION-2B CLIP ViT-L/14",
          "D2CLIP": "OpenAI CLIP ViT-L/14", "D2DINOL": "DINOv2-L", "D2DINOB": "DINOv2-B", "D2SIGLIP": "SigLIP2",
          "D2RN50": "ResNet-50", "D2BC1": "BioCLIP-1", "D2BC2640": "BioCLIP-2@640", "D2FT": "FT-ResNet-50",
          "D3": "WBF of E3-B's three stage-1 seeds", "D4F": "TS-F", "D4N": "TS-N", "D4R15": "TS-R15",
          "D6": "RF-DETR-L 12-class", "D8S": "D8-s (YOLO11s train_core)", "D8M": "D8-m (YOLO11m train_core)",
          "BCLEAN": "BCLEAN", "D1A1": "A'1", "D1B1": "B'1", "D1A12": "A'12", "TSA": "TS-A'", "TSB": "TS-B'",
          "D5B0": "dose 0", "D5B25": "dose 0.25", "D5B100": "dose 1.0"}
# the pre-declared comparisons: (id, exam, X, Y, se rule, statistic, tier); statistic species | p7 | probe
COMPARISONS = (
    ("P1", "dev", "E3B", "D2LAION", "image", "species", "P"),
    ("P2", "dev", "E3B", "D4F", "selection", "species", "P"),
    ("P3", "dev", "E3B", "D4N", "selection", "species", "P"),
    ("P4", "dev", "D3", "E3B", "one_system", "species", "P"),
    ("P5", "dev", "E3B", "D6", "selection", "species", "P"),
    ("P6", "ood22", "TSB", "D1A12", "max", "species", "P"),
    ("P7", "ood23", "D1B1", "D1A1", "max", "p7", "P"),
    ("P8", "probe", "D5B100", "D5B0", "max", "probe", "P"),
    ("P9", "probe", "D5B25", "D5B0", "max", "probe", "P"))
CHALLENGER_SIDES = {a: a for a in F.CHALLENGERS}
TIER_S = (
    ("S_bc2_over_bc2640", "dev", "E3B", "D2BC2640", "image"),
    ("S_ft_over_rn50", "dev", "D2FT", "D2RN50", "image"),
    ("S_bc2_over_clip", "dev", "E3B", "D2CLIP", "image"),
    ("S_laion_over_clip", "dev", "D2LAION", "D2CLIP", "image"),
    ("S_dinol_over_clip", "dev", "D2DINOL", "D2CLIP", "image"),
    ("S_bc1_over_clip", "dev", "D2BC1", "D2CLIP", "image"),
    ("S_tsf_over_tsr15", "dev", "D4F", "D4R15", "image"),
    ("S_e3b_over_tsr15", "dev", "E3B", "D4R15", "image"),
    ("S_tsf_over_e3a", "dev", "D4F", "E3A", "image"),
    ("S_e3a_over_d6", "dev", "E3A", "D6", "image"),
    ("S_d6_over_ref", "dev", "D6", "REF", "image"),
    ("S_ref_over_d8m", "dev", "REF", "D8M", "image"),
    ("S_d8m_over_d8s", "dev", "D8M", "D8S", "image"),
    ("S_bclean_over_ref", "dev", "BCLEAN", "REF", "image"),
    ("S_bclean_over_e3b", "dev", "BCLEAN", "E3B", "image"),
    ("S_tsa_over_a12_ood22", "ood22", "TSA", "D1A12", "max"),
    ("S_tsb_over_tsa_ood22", "ood22", "TSB", "TSA", "max"))
# The other comparisons V3F.5 names, each on arrays of its own (Tier S, reported): (id, exam, X, Y, se rule,
# statistic, species). statistic: stage1 (the committed stage-1 box sets' class-collapsed AP50-95 on dev,
# stage1_runs), subset (the species mean over the list given), species (the 12-class mean), copy:<family> (P8's
# statistic on the probe images whose dose-1.0 copy is of that family, V3F.5 D5b)
OOD23_SPECIES_100 = ("Purslane", "PalmerAmaranth", "Ragweed")              # ood23's species with >= 100 GT boxes
OOD22_SPECIES_100 = ("Waterhemp", "MorningGlory", "Purslane", "SpottedSpurge", "Carpetweed", "Ragweed",
                     "PalmerAmaranth", "Goosegrass")                         # ood22's eight (Eclipta's 33 left out)
COPY_FAMILIES = ("jpeg75", "crop90", "letterbox640", "hflip")              # inc2.v3fam_data.COPY_FAMILIES
TIER_S_ARRAYS = (
    ("S_d3_stage1_over_e3b", "dev", "D3", "E3B", "one_system", "stage1", None),
    ("S_s1_tsf_over_e1a", "dev", "D4F", "E3A", "image", "stage1", None),
    ("S_s1_tsf_over_e1b", "dev", "D4F", "E3B", "image", "stage1", None),
    ("S_s1_tsn_over_e1a", "dev", "D4N", "E3A", "image", "stage1", None),
    ("S_s1_tsn_over_e1b", "dev", "D4N", "E3B", "image", "stage1", None),
    ("S_s1_tsr15_over_e1a", "dev", "D4R15", "E3A", "image", "stage1", None),
    ("S_s1_tsr15_over_e1b", "dev", "D4R15", "E3B", "image", "stage1", None),
    ("S_tsb_over_a12_ood22_8sp", "ood22", "TSB", "D1A12", "max", "subset", OOD22_SPECIES_100),
    ("S_tsa_over_a12_ood22_8sp", "ood22", "TSA", "D1A12", "max", "subset", OOD22_SPECIES_100),
    ("S_tsb_over_tsa_ood22_8sp", "ood22", "TSB", "TSA", "max", "subset", OOD22_SPECIES_100),
    ("S_tsb_over_a12_ood23_3sp", "ood23", "TSB", "D1A12", "max", "subset", OOD23_SPECIES_100),
    ("S_tsa_over_a12_ood23_3sp", "ood23", "TSA", "D1A12", "max", "subset", OOD23_SPECIES_100),
    ("S_tsb_over_tsa_ood23_3sp", "ood23", "TSB", "TSA", "max", "subset", OOD23_SPECIES_100),
    ("S_p8_12class", "probe", "D5B100", "D5B0", "max", "species", None),
    ("S_p9_12class", "probe", "D5B25", "D5B0", "max", "species", None)) + tuple(
    ("S_d5b_copy_%s" % f, "probe", "D5B100", "D5B0", "max", "copy:%s" % f, None) for f in COPY_FAMILIES)
# V3F.5 D3: "Tier C: the same comparison" as P4 -- one record (one seed text), so one comparison cannot both
# qualify as P4 and open no read as C_D3
SAME_COMPARISON = {"C_D3": "P4"}
# V3F.5 D0-S's dev branch: E3's recorded dev comparisons, re-read on dev minus H from E3RS's arrays (reported only),
# under E3's own seed texts (twostage.SPECIES_SEED_TEXT, ATTR_SEED_TEXT)
E3_REREADS = (("E3_M_over_ref", "E3M", "REF", TS.SPECIES_SEED_TEXT),
              ("E3_A_over_ref", "E3A", "REF", TS.SPECIES_SEED_TEXT),
              ("E3_B_over_ref", "E3B", "REF", TS.SPECIES_SEED_TEXT),
              ("E3_B_over_A", "E3B", "E3A", TS.ATTR_SEED_TEXT))
STAGE1_RECOMPUTE_TOL = F.DEVIATION_TOL      # a recomputed stage-1 value against the one its score records
# The sides whose training set is the family's own, built after D0-S's removals (V3F.3.7 item 4; D6's staging,
# Revision 29): from base B (any removed row it held) or from base_v2 (only the removed rows that are base_v2's); D8's
# train_core holds none of D0-S's reference rows
TRAINED_SIDES_B = ("D4F", "D4N", "D4R15", "BCLEAN", "D1B1", "TSB")
TRAINED_SIDES_V2 = ("D1A1", "D1A12", "TSA", "D5B0", "D5B25", "D5B100", "D6")
# E3's classifier, D2's frozen-backbone fits and D9's refits are on exactly E3's crops (V3F.5 D2, D9)
E3_CROP_SIDES = ("E3M", "E3A", "E3B", "ORACLE", "D3", "D4F", "D4N", "D4R15", "BCLEAN", "D2LAION", "D2CLIP",
                 "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT")
# The ticker's family items (stream.py _v3fam_key) a --cap-refused list may name, and the score arms each holds
CAP_ITEM_RE = re.compile(r"(data|verdict|score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))\Z")
SCORE_DEPS = {"BCLEAN": ("build:D0BCLEAN",), "D8": ("build:D8M", "build:D8S"), "D4F": ("build:D4F",),
              "D4N": ("build:D4N",), "D4R15": ("build:D4R15",), "D1": ("build:D1A1", "build:D1B1", "build:D1A12"),
              "D5B": ("build:D5B0", "build:D5B25", "build:D5B100"), "D6": ("detr:smoke", "detr:train")}
PAIRED = ("P1",) + tuple("C_%s" % a for a in F.CHALLENGERS if a.startswith("D2")) + (
    "S_bc2_over_bc2640", "S_ft_over_rn50", "S_bc2_over_clip", "S_laion_over_clip", "S_dinol_over_clip",
    "S_bc1_over_clip")
REPORTED_EXAMS = ("imageweeds", SYS.OODDEV)
FAMILY_ESTIMATES = {"v3fam_data": 2.0, "v3fam_sweep": 1.5, "v3fam_v1": 0.5, "v3fam_detr_smoke": 0.5,
                    "v3fam_detr_s": 24.0, "v3fam_retest": RETEST_GPUH, "v3famtest_": READ_GPUH}
SCORE_ESTIMATE_GPUH = 1.5                    # a family score job without a price of its own here
PERSON_JOBS = ("v3fam_sweep", "v3famtest_", "v3fam_retest")
# Deviation 4 (2026-10-20, docs/prereg/V3F_dev4_2026-10-20.md). The verdict of 2026-10-13 (job 49107774,
# capacity/v3fam_v1.json with this sha256) was decided while seven arms had never started (D1, D4F, D4N, D4R15, D5B,
# D6, D8; its table also lists BCLEAN missing, which D0-S had not triggered, so it was not needed): the family's data
# build had failed twice on a defect (funnel.embed's 'duplicate keys'), and the ticker took its dependants as ended.
# That record stays as it is. The family's verdict over every arm is its continuation, decided by the same code (the
# same comparisons, Holm sets, Tier C gate and read order) once every arm's record is complete or failed (V3F.7 step
# 11), written once under its own names; it must reproduce every comparison the record of 2026-10-13 decided, or it
# refuses. The sealed-test reads follow the continuation.
PARTIAL_VERDICT_SHA256 = "304bad59f15487bfd3965d2f1e9384658daac58c5971494118ecaad4fcfaa7d9"
PARTIAL_EXTERNAL_SHA256 = "81cd1811d9cbd1870d4b6ecae922320ae9ca7e446e2a9a809afff7db3e3a60f5"   # its external record
CONTINUATION = {"verdict": "v3fam_v1_cont.json", "external": "v3fam_v1_cont_external.json",
                "ext": "v3fam_ext_v1_cont.json", "report": "v3fam_v1_cont_report"}
DEVIATION_4 = "docs/prereg/V3F_dev4_2026-10-20.md"
# Deviation 5 (2026-10-24, docs/prereg/V3F_dev5_2026-10-24.md; module docstring). The continuation of 2026-10-22
# (job 49571261, capacity/v3fam_v1_cont.json with this sha256, its external record that one) stays as it is; the
# family's verdict over every arm is its second continuation, written once under CONTINUATION_2, after reproducing
# every comparison either earlier record decided, and only once each arm deviation 5 reopened has a record since.
CONT_SHA256 = "f1a4d60d2d2ef72a682a56be059d4b39defeb909c8a317be4c52c006e211a860"
CONT_EXTERNAL_SHA256 = "d30e04d027ccd23718a32a58c16fa15169b21de721456f88fe2a78670f07e692"
CONTINUATION_2 = {"verdict": "v3fam_v1_cont2.json", "external": "v3fam_v1_cont2_external.json",
                  "ext": "v3fam_ext_v1_cont2.json", "report": "v3fam_v1_cont2_report"}
DEVIATION_5 = SUP.DEVIATION_5
DEV5_ARMS = ("D8", "D4F", "D4N", "D4R15", "D1", "D5B")      # the score arms deviation 5 reopened
# the ticker's failure records of 2026-10-22 for those arms (inc2.v3fam_supersede moves them aside): by sha256
DEV5_SUPERSEDED_FAILED = {"score:%s" % rel.rsplit("score_", 1)[1][:-len(".json")]: sha
                          for rel, sha in SUP.superseded_sha256s("v3fam/v1/failed/").items()}
D6_SMOKE = ("d6", "smoke", "smoke.json")     # under v3fam/v1: D6's smoke record (inc2.v3fam_detr)
REPRO_FIELDS = ("diff", "z", "interval_90", "outcome", "qualifies")     # a decision the continuation reproduces
CONT_REPRO_TOL = 1e-9
# The items the ticker recorded failed after their own second failure (stream state), passed in the verdict's argv
# (--ticker-failed) so the verb counts the arms they end as ended, as the ticker does (diagnose_stream.
# v3fam_verdict_ready): a job can fail before it writes a provenance attempt (cancelled while pending, NODE_FAIL at
# launch, an environment that does not activate). Never the data build (its failure does not end the arms it builds,
# deviation 4) nor the verdict.
FAILED_ITEM_RE = re.compile(r"(score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))\Z")
SCORED_AFTER_ALL = "E3RS"                    # every other score arm is scored after E3RS (the domain's score_after)


def log(msg):
    print("[inc2.v3fam] %s" % msg, flush=True)


def cap_path(name):
    return C.INC_DIR / "capacity" / name


def verdict_path():
    return cap_path(VERDICT)


def is_partial(path=None):
    """True when capacity/v3fam_v1.json is the verdict of 2026-10-13, decided while seven arms had never started
    (deviation 4)."""
    vp = path or verdict_path()
    return vp.is_file() and F._sha(vp) == PARTIAL_VERDICT_SHA256


def is_first_continuation(path=None):
    """True when capacity/v3fam_v1_cont.json is the continuation of 2026-10-22 (deviation 5)."""
    cp = path or cap_path(CONTINUATION["verdict"])
    return cp.is_file() and F._sha(cp) == CONT_SHA256


def second_continuation_due():
    """True when the verdict of 2026-10-13 and the continuation of 2026-10-22 are both in place: the family's
    verdict over every arm is then the second continuation (deviation 5)."""
    return is_partial() and is_first_continuation()


def family_verdict_path():
    """The family's verdict the sealed-test reads follow: capacity/v3fam_v1.json, or its continuation when that
    file is the verdict of 2026-10-13 (deviation 4), written or not; its second continuation when the continuation
    in place is the one of 2026-10-22 (deviation 5), written or not."""
    if not is_partial():
        return verdict_path()
    return cap_path(CONTINUATION_2["verdict"]) if is_first_continuation() else cap_path(CONTINUATION["verdict"])


def test_dir(arm):
    return F.root() / "test" / arm


# ------------------------------------------------------------------ the sides
def score_record(arm):
    """The platform's record of a score arm (capacity/v3fam_score_<arm>.json), or None."""
    d = F._read_json(SYS.score_record_path(arm))
    return d if isinstance(d, dict) else None


def cap_file(item):
    return F.cap_dir() / ("%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(item)))


def cap_refused(item):
    return cap_file(item).exists()


cap_refused_item = cap_refused                # the same check where a parameter shadows the name


def record_ticker_refusals(items):
    """The items the ticker recorded 'not run (family cap)' (its stream state; stream.py _v3fam_cap_ok), passed in
    the verdict's argv, each written once under v3fam/v1/cap/ as check_cap writes its own (one already there is
    kept), so arm_status reads every cap refusal from the cluster's records (V3F.3.4, V3F.3.5). Returns the
    names."""
    out = []
    for it in items or []:
        it = str(it).strip()
        if not it:
            continue
        if not CAP_ITEM_RE.match(it):
            raise F.V3famRefused("--cap-refused names %r, not a family item (data, score:<ARM>, build:<ARM>, "
                                 "detr:smoke|train, verdict)" % it)
        p = cap_file(it)
        if not p.exists():
            F.write_once(p, {"format": F.FMT_CAP, "item": it, "status": F.NOT_RUN_CAP, "cap_gpuh": F.FAMILY_CAP_GPUH,
                             "source": "the ticker's refusal (stream state), passed in the verdict's argv",
                             "written_utc": F._utc()})
        out.append(it)
    return out


def failed_dir():
    return F.root() / "failed"


def failed_file(item):
    return failed_dir() / ("%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(item)))


def ending_items(arm):
    """The family items whose failure ends the arm (its record stays missing): its own score job, what it scores (a
    build, D6's jobs) and E3RS, which every other arm is scored after."""
    return ("score:%s" % arm,) + tuple(SCORE_DEPS.get(arm, ())) + (
        ("score:%s" % SCORED_AFTER_ALL,) if arm != SCORED_AFTER_ALL else ())


def _failed_items(items):
    """The --ticker-failed items, checked: a family item other than the data build and the verdict."""
    out = []
    for it in items or []:
        it = str(it).strip()
        if not it:
            continue
        if not FAILED_ITEM_RE.match(it):
            raise F.V3famRefused("--ticker-failed names %r, not a score, build or D6 item (score:<ARM>, build:<ARM>, "
                                 "detr:smoke|train); the data build's failure never ends the arms it builds "
                                 "(deviation 4)" % it)
        out.append(it)
    return out


def record_ticker_failures(items):
    """The items the ticker recorded failed after their own second failure (stream state), passed in the verdict's
    argv (--ticker-failed), each written once under v3fam/v1/failed/ (one already there is kept), so arm_status
    reports the arms they end as failed from the cluster's records (V3F.3.4, V3F.3.5: 'a job that failed twice').
    Returns the names."""
    out = []
    for it in _failed_items(items):
        p = failed_file(it)
        if not p.exists():
            F.write_once(p, {"format": "inc2-v3fam-failed/1", "item": it, "status": "failed",
                             "reason": "a job that failed twice (V3F.3.4, V3F.3.5)",
                             "source": "the ticker's record (stream state): failed after its own second failure, "
                                       "passed in the verdict's argv (deviation 4)", "written_utc": F._utc()})
        out.append(it)
    return out


def superseded_failure(item, path=None):
    """True for a failure record of 2026-10-22 deviation 5 supersedes (DEV5_SUPERSEDED_FAILED, by sha256): the
    platform's refusal of the arm's score jobs, not the arm's failure."""
    p = path or failed_file(item)
    return item in DEV5_SUPERSEDED_FAILED and p.is_file() and F._sha(p) == DEV5_SUPERSEDED_FAILED[item]


def failure_recorded(item):
    return failed_file(item).exists() and not superseded_failure(item)


def d6_smoke_status():
    """D6's smoke record's status (inc2.v3fam_detr writes v3fam/v1/d6/smoke/smoke.json), or None."""
    d = F._read_json(F.root().joinpath(*D6_SMOKE))
    return d.get("status") if isinstance(d, dict) else None


def supersede():
    """inc2.v3fam_supersede.apply on this INC_DIR (the records a recorded deviation supersedes moved aside byte for
    byte, each named in its folder's manifest), as the score jobs of v3fam_systems' and v3fam_trained's arms do
    first; refused as a family refusal."""
    return SYS.supersede()


def _write_once_json(path, obj):
    F.write_once(path, obj)


def set_aside_partial(names):
    """The second continuation's own records (CONTINUATION_2: external, ext, report .json/.md) that an attempt cut
    off before its verdict file left behind, moved byte for byte (a rename on the same file system: never
    overwritten, never deleted) into capacity/<verdict stem>_partial_<utc>/, with a MANIFEST.json naming each by
    sha256, so this attempt writes every record once. Only while the verdict file itself does not exist; it is
    never moved. Returns [{name, sha256, to}]."""
    if cap_path(names["verdict"]).exists():
        raise F.V3famRefused("%s exists: the family's second continuation verdict is written once (deviation 5)"
                             % cap_path(names["verdict"]))
    left = [n for n in [names["external"], names["ext"], names["report"] + ".json", names["report"] + ".md"]
            if cap_path(n).exists()]
    if not left:
        return []
    stem = names["verdict"][:-len(".json")]
    folder = cap_path("%s_partial_%s" % (stem, F._utc().replace(":", "").replace("-", "")[:15]))
    k = 1
    while folder.exists():
        k += 1
        folder = cap_path("%s_partial_%s_%d" % (stem, F._utc().replace(":", "").replace("-", "")[:15], k))
    folder.mkdir(parents=True)
    out = []
    for n in left:
        sha = F._sha(cap_path(n))
        os.rename(str(cap_path(n)), str(folder / n))
        out.append({"name": n, "sha256": sha, "to": "%s/%s" % (folder.name, n)})
        log("second continuation: %s, left by an attempt cut off before its verdict file, moved aside to %s "
            "(sha256 %s)" % (n, folder.name, sha[:12]))
    F.write_once(folder / "MANIFEST.json", {"format": "inc2-v3fam-partial-set-aside/1", "records": out,
                                            "why": "written by an attempt of the second continuation cut off before "
                                                   "its verdict file (deviation 5): moved aside, never overwritten",
                                            "job": os.environ.get("SLURM_JOB_ID") or None, "moved_utc": F._utc()})
    return out


def arm_status(arm):
    """complete | not run (<why>) | failed | missing, from the arm's score record, the family cap's records (the
    arm's own score item, or an item it depends on: a build it scores, D6's RF-DETR jobs, the data build) and the
    ticker's failure records (ending_items: an arm with no score record whose score job, build, D6 job or E3RS
    failed twice is 'failed (<item> failed twice)')."""
    deps = SCORE_DEPS.get(arm, ())
    if cap_refused("score_%s" % arm) or cap_refused(arm) or any(cap_refused(d) for d in deps) \
            or (deps and cap_refused("data")):
        return F.NOT_RUN_CAP
    r = score_record(arm)
    if r is None:
        ended = [i for i in ending_items(arm) if failure_recorded(i)]
        if ended:
            return "failed (%s failed twice)" % ", ".join(ended)
        if arm == "D6" and str(d6_smoke_status() or "").startswith("not run"):
            return str(d6_smoke_status())         # V3F.5 D6: the walltime refusal is 'not run' with its reason
        if arm == "BCLEAN" and not d0s_trigger().get("bclean"):
            return "not needed"                   # V3F.5 D0-S: BCLEAN only when D0-S triggered it
        return "missing"
    st = r.get("status")
    if st == "complete":
        return "complete"
    if st == "not run":
        return "not run (%s)" % (r.get("reason") or "recorded")
    return str(st or "failed")


def side(name, exam):
    """{"status": "ok", "docs", "runs", "species"} of a side's committed scores on exam, or {"status", "why"}."""
    dirname, variant, ens, arm = SIDES[name]
    try:
        got = SYS.family_runs(dirname, exam, variant, ens=ens)
    except F.V3famRefused as e:
        st = arm_status(arm)
        return {"status": "not run" if st != "complete" else "failed", "arm": arm,
                "why": "%s: %s" % (st, str(e)[:300])}
    docs = [d for d, _a in got]
    for d in docs:
        if d.get("exam") != exam or (d.get("variant") or "") != (variant or ""):
            return {"status": "failed", "arm": arm, "why": "%s's file is not a %s score" % (name, exam)}
    return {"status": "ok", "arm": arm, "docs": docs, "runs": [a for _d, a in got],
            "species": [d.get("species_map50_95") for d in docs]}


def p7_runs(name):
    """P7's per-image arrays of a stage1-mode side on ood23 (the target reading, ignored rows left out: the p7 npz
    the score JSON names) and its per-seed values, the collapsed AP50-95 of those arrays (V3F.3.2)."""
    dirname, _v, _e, arm = SIDES[name]
    runs = []
    for s in F.SEEDS:
        _doc, z = SYS.family_side(dirname, s, "ood23", "p7")
        runs.append({"keys": z["t_keys"], "tp": z["t_tp"], "conf": z["t_conf"], "pred_img": z["t_pred_img"],
                     "n_gt": z["t_n_gt"]})
    return runs, [F.statistic_value(a, "agnostic") for a in runs]


_GT = {}


def exam_gt(exam):
    """{key: [m, 4] float64} of every ground-truth box of exam's images in original pixels (the label's normalised
    boxes times the image's EXIF-transposed size, the frame the family pass's geometry returns boxes in; a label
    row repeated exactly is read once, as Ultralytics' label check reads it), each label checked against its
    manifest's label_sha256. Read once per process."""
    import numpy as np
    if exam in _GT:
        return _GT[exam]
    rows = SYS.exam_rows(exam)
    sizes = SYS.image_sizes(rows)
    out = {}
    for r in rows:
        lp = r.get("label")
        if not lp or not os.path.isfile(str(lp)):
            raise F.V3famRefused("%s's label %s is missing: its stage-1 boxes cannot be matched" % (r["key"], lp))
        if r.get("label_sha256") and C.sha256_file(lp) != r["label_sha256"]:
            raise F.V3famRefused("the label of %s does not hash as its manifest records" % r["key"])
        w, h = sizes[r["key"]]
        rows_ = [tuple(x) for x in C.read_yolo(lp)]
        rows_ = [x for i, x in enumerate(rows_) if x not in rows_[:i]]
        b = [((cx - bw / 2.0) * w, (cy - bh / 2.0) * h, (cx + bw / 2.0) * w, (cy + bh / 2.0) * h)
             for _c, cx, cy, bw, bh in rows_]
        out[r["key"]] = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    _GT[exam] = out
    return out


def stage1_runs(name, exam="dev"):
    """A side's committed stage-1 box sets on exam (its stage1.npz: one_per_box's boxes in original pixels and q)
    as agnostic arrays {keys, tp, conf, pred_img, n_gt}, matched class-collapsed to the exam's ground-truth boxes at
    IoU 0.50:0.95 by Ultralytics' match_predictions (inc2.v3fam.default_match, as twostage's stage-1 helper
    matches them); D3's fused set is one run. Each run's full-set value must equal the stage-1 agnostic AP50-95
    its score records within STAGE1_RECOMPUTE_TOL, or the comparison is refused. Returns (runs, values,
    {recorded, max_abs_diff})."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import box_iou
    dirname, variant, ens, _arm = SIDES[name]
    gt = exam_gt(exam)
    match = F.default_match()
    runs, vals, recorded, diffs = [], [], [], []
    for s in ([None] if ens else F.SEEDS):
        doc, z = SYS.family_side(dirname, s, exam, "stage1_images", variant=variant, ens=ens)
        keys = [str(k) for k in z["keys"]]
        if sorted(keys) != sorted(gt):
            raise F.V3famRefused("%s's stage-1 record does not hold %s's images" % (name, exam))
        img = np.asarray(z["img"], dtype=np.int64).reshape(-1)
        xyxy = np.asarray(z["xyxy"], dtype=np.float64).reshape(-1, 4)
        q = np.asarray(z["q"], dtype=np.float64).reshape(-1)
        tps, confs, pimg, ngt = [], [], [], []
        for i, k in enumerate(keys):
            sel = img == i
            b, g = xyxy[sel], gt[k]
            if len(b) and len(g):
                tp = np.asarray(match(box_iou(torch.as_tensor(g), torch.as_tensor(b))), dtype=bool).reshape(len(b), -1)
            else:
                tp = np.zeros((len(b), SC.NIOU), dtype=bool)
            tps.append(tp)
            confs.append(q[sel])
            pimg.append(np.full(len(b), i, dtype=np.int64))
            ngt.append(len(g))
        a = {"keys": np.asarray(keys), "tp": np.concatenate(tps, 0) if tps else np.zeros((0, SC.NIOU), dtype=bool),
             "conf": np.concatenate(confs) if confs else np.zeros(0),
             "pred_img": np.concatenate(pimg) if pimg else np.zeros(0, dtype=np.int64),
             "n_gt": np.asarray(ngt, dtype=np.int64)}
        if doc.get("n_boxes") is not None and int(a["n_gt"].sum()) != int(doc["n_boxes"]):
            raise F.V3famRefused("%s seed %s: %s's labels hold %d boxes, the score counted %d"
                                 % (name, s, exam, int(a["n_gt"].sum()), int(doc["n_boxes"])))
        v = F.statistic_value(a, "agnostic")
        rec = (doc.get("stage1") or {}).get("agnostic_map50_95")
        if rec is None or not abs(v - float(rec)) <= STAGE1_RECOMPUTE_TOL:
            raise F.V3famRefused("%s seed %s: the stage-1 boxes matched again give %.6f, its score records %s (more "
                                 "than %g apart)" % (name, s, v, rec, STAGE1_RECOMPUTE_TOL))
        runs.append(a)
        vals.append(v)
        recorded.append(float(rec))
        diffs.append(abs(v - float(rec)))
    return runs, vals, {"recorded": recorded, "max_abs_diff": max(diffs)}


def copy_keys(family):
    """The probe keys whose dose-1.0 copy is of copy family `family` and was kept (the family data summary's copy
    plan, its dropped copies left out; V3F.5 D5b's per-family inflation)."""
    from . import v3fam_data as VD
    s = VD.read_summary()
    if s is None:
        raise F.V3famRefused("no complete family data summary: the probe's copy families are unknown")
    cp = s.get("copies") or {}
    dropped = set(cp.get("dropped") or [])
    keys = sorted(k for k, _i, f in cp.get("plan") or [] if f == family and (VD.COPY_KEY % k) not in dropped)
    if not keys:
        raise F.V3famRefused("no kept dose-1.0 copy of family %s" % family)
    return keys


def sessions(exam):
    """One session per image in the arrays' key order (V3F.3.1's rule; the probe in its own root)."""
    rows = SYS.exam_rows(exam)
    return F.sessions_for(exam, rows)


def d0s_trigger():
    """D0-S's trigger from its complete record ({} when there is none)."""
    from . import v3fam_sweep as SW
    rec = SW.read_record()
    return (rec or {}).get("trigger") or {}


def dev_keep(trig):
    """The dev keys a dev comparison is decided on: every dev key but those with a true copy (D0-S's dev branch),
    or None when no dev image has one."""
    h = [str(k)[len("dev/"):] for k in trig.get("h_dev") or []]
    if not h:
        return None
    return [r["key"] for r in SYS.exam_rows("dev") if r["key"] not in set(h)]


def d0s_removals():
    """What D0-S's confirmed copies took out of the family's training sets (V3F.3.7 item 4), from the family data
    summary: {rows, sha256, base_v2_rows}, or None when nothing was removed."""
    from . import v3fam_data as VD
    d = (VD.read_summary() or {}).get("d0s") or {}
    keys = list(d.get("removal_keys") or [])
    if not keys:
        return None
    return {"rows": len(keys), "sha256": d.get("removal_sha256"),
            "base_v2_rows": len(d.get("d6_staging_removals") or [])}


def declared(xname, yname, rm):
    """The training-row difference D0-S's removals make between a comparison's sides, declared beside it (V3F.3.7
    item 4), or None: the sides whose training set is the family's own lost the rows; the others were not retrained;
    and, when base_v2 rows are among them, E3's crops (E3's classifier, D2's fits) keep their crops (Revision 31)."""
    if not rm:
        return None
    sides = [n for n in (xname, yname) if n]
    lost = [n for n in sides if n in TRAINED_SIDES_B or (n in TRAINED_SIDES_V2 and rm["base_v2_rows"])]
    mine = [LABELS.get(n, n) for n in lost]
    other = [LABELS.get(n, n) for n in sides if n not in lost]
    crops = rm["base_v2_rows"] and any(n in E3_CROP_SIDES for n in sides)
    if not mine and not crops:
        return None
    txt = "D0-S's confirmed copies: %d base B row(s), %d of them base_v2's (removal list %s)" % (
        rm["rows"], rm["base_v2_rows"], str(rm["sha256"])[:16])
    if mine:
        txt += "; they left %s's training set (a base_v2-built side loses the base_v2 rows only)" % " and ".join(mine)
        if other:
            txt += "; %s was not retrained or held none of them" % " and ".join(other)
    if crops:
        txt += ("; the stage-2 classifiers here are fitted on exactly E3's crops, which include the crops of those %d "
                "base_v2 row(s) (V3F.5 D2, D9; Revision 31)" % rm["base_v2_rows"])
    return txt


# ------------------------------------------------------------------ one comparison
def compare(cid, exam, xname, yname, rule, statistic="species", selection=None, keep=None, seed_text=None,
            species=None, subset=None):
    """One pre-declared comparison on committed family scores (module docstring); {"id", "status": decided |
    not run, ...} with inc2.v3fam.comparison's record when decided. statistic: species (the 12-class mean), probe
    (D5b's seven species), subset (the mean over `species`), p7 (P7's ignore-region arrays), stage1 (stage1_runs).
    keep: dev minus H (dev comparisons only); subset: the images a pre-declared subset comparison is computed on
    (D5b's copy families)."""
    base = {"id": cid, "exam": exam, "x": xname, "y": yname, "x_label": LABELS.get(xname),
            "y_label": LABELS.get(yname), "se_rule": rule}
    s1 = None
    try:
        if statistic == "p7":
            xr, xv = p7_runs(xname)
            yr, yv = p7_runs(yname)
            stat, species_ = "agnostic", SC.SPECIES
        elif statistic == "stage1":
            xr, xv, xc = stage1_runs(xname, exam)
            yr, yv, yc = stage1_runs(yname, exam)
            stat, species_ = "agnostic", SC.SPECIES
            s1 = {"x": xc, "y": yc}
        else:
            xs, ys = side(xname, exam), side(yname, exam)
            for nm, sd in ((xname, xs), (yname, ys)):
                if sd["status"] != "ok":
                    return dict(base, status="not run", why="%s: %s" % (nm, sd.get("why")))
            xr, yr = xs["runs"], ys["runs"]
            stat = "species"
            species_ = F.PROBE_SPECIES if statistic == "probe" else tuple(species) if statistic == "subset" \
                else SC.SPECIES
            if species_ != SC.SPECIES:
                xv = [F.statistic_value(a, "species", species_) for a in xr]
                yv = [F.statistic_value(a, "species", species_) for a in yr]
            else:
                xv, yv = xs["species"], ys["species"]
        ses = sessions(exam) if exam in ("dev", "ood22", "ood23", "probe") else None
        sel = None
        if rule == "selection":
            sel = {"runs": {}, "values": {}}
            for a in F.SELECTION_ARMS:
                sd = side("E3%s" % a, exam)
                if sd["status"] != "ok":
                    return dict(base, status="not run", why="the selection correction needs E3-%s: %s"
                                % (a, sd.get("why")))
                sel["runs"][a], sel["values"][a] = sd["runs"], sd["species"]
        restrict = keep if exam == "dev" else None
        if subset is not None:
            restrict = list(subset) if restrict is None else [k for k in subset if k in set(restrict)]
        rec = F.comparison(cid, xv, yv, xr, yr, statistic=stat, species=species_, sessions=ses,
                           se_rule={"one_system": "image"}.get(rule, rule), one_system=rule == "one_system",
                           selection=sel, keep=restrict, paired=cid in PAIRED, image_seed_text=seed_text)
    except F.V3famRefused as e:
        return dict(base, status="not run", why=str(e)[:400])
    for k in ("x_values", "y_values"):
        rec[k] = [float(v) for v in rec[k]]
    if rule == "one_system":
        rec["x_minus_each_y"] = [rec["x_values"][0] - y for y in rec["y_values"]]
    if s1 is not None:
        rec["stage1_recompute"] = s1
    if subset is not None:
        rec["subset"] = {"n_images": rec["n_images"], "of": statistic}
    sel_rec = rec.get("selection")
    if sel_rec:
        rec["selection"] = {k: v for k, v in sel_rec.items() if k not in ("T", "chosen")}
    return dict(base, status="decided", **rec)


def summary(rec):
    """What Holm, the gate and the report read of a comparison."""
    if rec.get("status") != "decided":
        return {"status": rec.get("status") or "not run", "why": rec.get("why")}
    d = rec["decision"]
    return {"status": "decided", "qualifies": d["qualifies"], "z": d["z"], "diff": d["diff"],
            "outcome": d["outcome"], "flags": d["flags"], "interval_90": d["interval_90"]}


# ------------------------------------------------------------------ the verdict (L23L)
def comparison_specs():
    """Every pre-declared comparison of the family as (id, exam, X, Y, rule, statistic, species, seed text): the
    primaries, Tier C (C_D3 is P4's record, SAME_COMPARISON), Tier S."""
    out = [(cid, exam, x, y, rule, stat, None, None) for cid, exam, x, y, rule, stat, _t in COMPARISONS]
    out += [("C_%s" % a, "dev", CHALLENGER_SIDES[a], "E3B", "image", "species", None, None) for a in F.CHALLENGERS
            if "C_%s" % a not in SAME_COMPARISON]
    out += [(cid, exam, x, y, rule, "species", None, BCLEAN_SEED_TEXT if cid == "S_bclean_over_ref" else None)
            for cid, exam, x, y, rule in TIER_S]
    out += [(cid, exam, x, y, rule, stat, sp, None) for cid, exam, x, y, rule, stat, sp in TIER_S_ARRAYS]
    return out


def run_spec(spec, keep):
    cid, exam, x, y, rule, stat, sp, st = spec
    subset = None
    if stat.startswith("copy:"):
        try:
            subset = copy_keys(stat[len("copy:"):])
        except F.V3famRefused as e:
            return {"id": cid, "exam": exam, "x": x, "y": y, "x_label": LABELS.get(x), "y_label": LABELS.get(y),
                    "se_rule": rule, "status": "not run", "why": str(e)[:400]}
        stat = "probe"
    return compare(cid, exam, x, y, rule, stat, keep=keep, seed_text=st, species=sp, subset=subset)


def decide_family(keep=None):
    """Every pre-declared comparison of the family on its committed scores; Holm, the gate, the read order. On
    D0-S's dev branch also every dev comparison's full-dev numbers and E3's recorded dev comparisons re-read on
    dev minus H (reported only)."""
    trig = d0s_trigger()
    keep = dev_keep(trig) if keep is None else keep
    rm = d0s_removals()
    specs = comparison_specs()
    out = {}
    for spec in specs:
        out[spec[0]] = run_spec(spec, keep)
    for cid, src in SAME_COMPARISON.items():
        out[cid] = dict(out[src], id=cid, same_as=src)
    for cid, r in out.items():
        note = declared(r.get("x"), r.get("y"), rm)
        if note:
            r["declared"] = note
    holm = F.holm_overlay({p: summary(out[p]) for p in F.PRIMARIES})
    gate = F.challenger_gate({a: summary(out["C_%s" % a]) for a in F.CHALLENGERS})
    ref_avail = tuple(a for a in F.REFERENCE_READS if arm_status("D8" if a == "D8M" else a) == "complete")
    bclean = bool(trig.get("bclean")) and arm_status("BCLEAN") == "complete"
    order = F.read_order(gate, bclean_triggered=bclean, reference_available=ref_avail)
    full, e3 = None, None
    if keep is not None:
        full = {spec[0]: summary(run_spec(spec, None)) for spec in specs if spec[1] == "dev"}
        for cid, src in SAME_COMPARISON.items():
            full[cid] = dict(full[src], same_as=src)
        e3 = {"dev_minus_h": {}, "full_dev": {}}
        for cid, x, y, st in E3_REREADS:
            e3["dev_minus_h"][cid] = _strip(compare(cid, "dev", x, y, "image", keep=keep, seed_text=st))
            e3["full_dev"][cid] = summary(compare(cid, "dev", x, y, "image", seed_text=st))
    return {"comparisons": out, "holm": holm, "gate": gate, "order": order, "trigger": trig,
            "decided_on": "dev minus H" if keep is not None else "dev",
            "dev_minus_h": None if keep is None else {"n_images": len(keep),
                                                     "keys_sha256": C.sha256_text("\n".join(sorted(keep)))},
            "full_dev_beside": full, "e3_reread": e3, "d0s_removals": rm}


def dev_passes():
    """Every dev pass of the family (V3F.3.5's printed count), by where it is recorded: {total, by_kind}.
    family_scores: the committed family scores on dev (arms/**: E3RS with O5, D2, D3, D4, D6, D8, D9b, BCLEAN,
    D1's dev readings); d2_gt: D2's ground-truth passes (d2/<KEY>/dev_gt.json, D2FT's dev_gt.s<k>.json); d9a: E3's
    five fold classifiers on seed 0 of each E3 arm (twostage/e3_v1/sensitivity.json); reproductions: dev
    identity's passes and injection's two (identity, injected); E3's two-stage reproductions are E3RS's dev
    scores, counted once. builds_protocol: the pinned driver's own protocol dev scores of the family's build runs
    (each base run's and each final's; D5b's are never read by the family)."""
    by = {"family_scores": 0, "d2_gt": 0, "d9a": 0, "reproductions": 0, "builds_protocol": 0}
    arms = F.root() / "arms"
    if arms.is_dir():
        for p in sorted(arms.rglob("dev*.json")):
            d = F._read_json(p)
            if isinstance(d, dict) and d.get("format") == F.FMT_SCORE and d.get("exam") == "dev":
                by["family_scores"] += 1
    d2 = F.root() / "d2"
    if d2.is_dir():
        by["d2_gt"] = sum(1 for p in d2.glob("*/dev_gt*.json") if isinstance(F._read_json(p), dict))
    sens = F._read_json(TS.root() / "sensitivity.json")
    if isinstance(sens, dict):
        by["d9a"] = sum(len((a or {}).get("values") or []) for a in (sens.get("arms") or {}).values())
    for name, per_pass in (("dev_identity.json", 1), ("injection.json", 2)):
        d = F._read_json(F.equiv_dir() / name)
        if isinstance(d, dict):
            by["reproductions"] += per_pass * len(d.get("per") or {})
    for exp in RC.V3FAM_EXPS.values():
        runs = C.INC_DIR / exp / "runs"
        if runs.is_dir():
            by["builds_protocol"] += sum(1 for p in runs.glob("*/scores/dev.json") if p.is_file())
    return {"total": sum(by.values()), "by_kind": by}


def _strip(rec):
    """A comparison record without its large members (the per-resample values)."""
    return {k: v for k, v in rec.items() if k not in ("Vx", "Vy", "diffs", "kept")}


def ext_record():
    """ImageWeeds (agnostic, Ragweed AP) and ooddev_v1 (agnostic) of every side that has them, reported."""
    out = {}
    for name in sorted(SIDES):
        for exam in REPORTED_EXAMS:
            sd = side(name, exam)
            if sd["status"] != "ok":
                continue
            vals = {"agnostic": [d.get("agnostic_map50_95") for d in sd["docs"]],
                    "stage1_agnostic": [(d.get("stage1") or {}).get("agnostic_map50_95") for d in sd["docs"]]}
            if exam == "imageweeds":
                vals["ragweed_ap50_95"] = [(d.get("per_class") or {}).get("Ragweed") for d in sd["docs"]]
            row = {}
            for k, v in vals.items():
                v = [float(x) for x in v if x is not None]
                if v:
                    m, s = F._mean_sd(v)
                    row[k] = {"values": v, "mean": m, "sd": s}
            out.setdefault(name, {})[exam] = row
    return out


def _table():
    """The family table: every side, every exam it holds, every seed (species, agnostic, stage-1 agnostic; on ood23
    a stage1-mode side's P7 readings: the target reading with and without the ignore regions, OtherPlant's)."""
    rows = []
    for name in sorted(SIDES):
        for exam in ("dev", "ood22", "ood23", "probe") + REPORTED_EXAMS:
            sd = side(name, exam)
            if sd["status"] != "ok":
                continue
            sp = [d.get("species_map50_95") for d in sd["docs"]]
            row = {"side": name, "label": LABELS.get(name), "exam": exam, "seeds": [d.get("seed") for d in sd["docs"]],
                   "species": sp, "species_mean": F._mean_sd([x for x in sp if x is not None])[0],
                   "agnostic": [d.get("agnostic_map50_95") for d in sd["docs"]],
                   "stage1_agnostic": [(d.get("stage1") or {}).get("agnostic_map50_95") for d in sd["docs"]]}
            if any(d.get("p7") for d in sd["docs"]):
                row["p7"] = [d.get("p7") for d in sd["docs"]]
            rows.append(row)
    return rows


def report_row(v):
    """A comparison as the family table shows it (V3F.3.5): D, and D_sel with the plain D, SE and the share of
    resamples in which E3-B is chosen beside it (P2, P3, P5); 2 sd; both SEs; z; the interval; the outcome."""
    out = dict(summary(v), id=v.get("id"), exam=v.get("exam"), x=v.get("x"), y=v.get("y"), x_label=v.get("x_label"),
               y_label=v.get("y_label"), se_rule=v.get("se_rule"), statistic=v.get("statistic"),
               decision=v.get("decision"), image=v.get("image"), session=v.get("session"), why=v.get("why"))
    for k in ("x_values", "y_values", "plain", "paired_per_seed", "x_minus_each_y", "same_as", "declared",
              "stage1_recompute", "subset", "restricted_to", "species"):
        if v.get(k) is not None:
            out[k] = v[k]
    sel = v.get("selection")
    if sel:
        out["selection"] = {k: sel.get(k) for k in ("diff_sel", "se_sel", "bias", "n_valid", "share_chosen")}
    return out


def _md(rep):
    def f(x):
        return "-" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))

    def head():
        return ["| id | exam | X | Y | D | D_sel | 2 sd | SE image | SE session | SE_sel | z | 90 % interval | outcome "
                "| confirmed / opens |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(cid, r, extra=""):
        if r.get("status") != "decided":
            return "| %s | %s | %s | %s | not run: %s |  |  |  |  |  |  |  |  | %s |" % (
                cid, r.get("exam"), r.get("x"), r.get("y"), str(r.get("why"))[:80], extra)
        d = r["decision"]
        plain = r.get("plain")
        sel = r.get("selection") or {}
        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | [%s, %s] | %s%s | %s |" % (
            cid, r["exam"], r["x"], r["y"], f(plain["diff"] if plain else d["diff"]), f(d["diff"] if plain else None),
            f(d["two_sd"]), f((r.get("image") or {}).get("se")), f((r.get("session") or {}).get("se")),
            f(sel.get("se_sel")), f(d["z"]), f(d["interval_90"][0]), f(d["interval_90"][1]), d["outcome"],
            (" (%s)" % ", ".join(d["flags"])) if d["flags"] else "", extra)
    dp = rep["dev_passes"]
    lines = ["# V3F family report (%s)" % rep["generated_utc"], "",
             "Decided on: %s. Dev passes of the family: %d (%s)." % (
                 rep["decided_on"], dp, ", ".join("%s %d" % kv for kv in sorted((rep.get("dev_passes_by_kind")
                                                                                 or {}).items()))), ""] + head()
    for cid, r in rep["comparisons"].items():
        extra = ""
        if cid in rep["holm"]:
            extra = "family-confirmed" if rep["holm"][cid]["family_confirmed"] else "not confirmed"
        if cid.startswith("C_"):
            g = rep["gate"]["arms"].get(cid[2:]) or {}
            extra = "opens a read" if g.get("opens_read") else "no read"
            if g.get("opens_read") and cid[2:] in F.REFERENCE_READS:
                extra += " (its read is the reference read)"
        if r.get("same_as"):
            extra += " (= %s)" % r["same_as"]
        lines.append(row(cid, r, extra))
    sel_lines = ["%s: E3-B chosen in %.1f %% of the image resamples; plain D %s, plain image SE %s; D_sel %s, SE_sel %s"
                 % (cid, 100.0 * float((r.get("plain") or {}).get("share_B_chosen") or 0.0),
                    f((r.get("plain") or {}).get("diff")), f((r.get("plain") or {}).get("se")),
                    f(r["decision"]["diff"]), f((r.get("selection") or {}).get("se_sel")))
                 for cid, r in rep["comparisons"].items() if r.get("status") == "decided" and r.get("plain")]
    if sel_lines:
        lines += ["", "Selection-corrected (V3F.3.3):"] + ["- %s" % x for x in sel_lines]
    decl = ["%s: %s" % (cid, r["declared"]) for cid, r in rep["comparisons"].items() if r.get("declared")]
    if decl:
        lines += ["", "Declared beside (V3F.3.7 item 4):"] + ["- %s" % x for x in decl]
    if rep.get("full_dev_beside"):
        lines += ["", "Full dev beside (decided on dev minus H):", "",
                  "| id | D | z | 90 % interval | outcome |", "|---|---|---|---|---|"]
        for cid, r in rep["full_dev_beside"].items():
            if r.get("status") == "decided":
                lines.append("| %s | %s | %s | [%s, %s] | %s |" % (cid, f(r["diff"]), f(r["z"]),
                                                                    f(r["interval_90"][0]), f(r["interval_90"][1]),
                                                                    r["outcome"]))
            else:
                lines.append("| %s | not run: %s |  |  |  |" % (cid, str(r.get("why"))[:80]))
    if rep.get("e3_reread"):
        lines += ["", "E3's recorded dev comparisons re-read on dev minus H (reported only):", ""] + head()
        for cid, r in rep["e3_reread"]["dev_minus_h"].items():
            lines.append(row(cid, r, "reported only"))
    lines += ["", "Reads in order: %s." % (", ".join("%s (%s)" % (o["arm"], o["kind"]) for o in rep["order"])
                                         or "none"), "",
              "| side | exam | species (seeds) | mean | agnostic | stage-1 agnostic |", "|---|---|---|---|---|---|"]
    for t in rep["table"]:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (t["label"], t["exam"], ", ".join(f(x) for x in t["species"]),
                                                          f(t["species_mean"]), ", ".join(f(x) for x in t["agnostic"]),
                                                          ", ".join(f(x) for x in t["stage1_agnostic"])))
    return "\n".join(lines) + "\n"


def _attempted(name):
    """True when the provenance of the family job `name` holds at least one attempt (the cluster's own record that
    the job ran)."""
    rec = F._read_json(C.INC_DIR / "_campaign" / "provenance" / ("%s.json" % name))
    return isinstance(rec, dict) and bool(rec.get("attempts"))


def arm_started(arm):
    """True when the arm, or what it scores, ran at least once: an attempt in the provenance of its score job
    (v3fam_score_<arm>), of a build it scores (the build's experiment name) or of D6's jobs (v3fam_detr_smoke,
    v3fam_detr_s<k>)."""
    names = ["v3fam_score_%s" % arm.lower()]
    for d in SCORE_DEPS.get(arm, ()):
        if d.startswith("build:"):
            names.append(RC.V3FAM_EXPS[d[len("build:"):]])
        elif d == "detr:smoke":
            names.append("v3fam_detr_smoke")
        elif d == "detr:train":
            names += ["v3fam_detr_s%d" % s for s in F.SEEDS]
    return any(_attempted(n) for n in names)


def verdict_ready(trig=None, cap_refused=(), failed=()):
    """V3F.7 step 11, 'L23L verdict, once every arm's record is complete or failed', on the cluster's records:
    (ready, why). The family's data build complete (splits/v3fam/summary.json: every arm of steps 8-10 trains on
    or after its manifests), and no arm that never started: an arm with no record (score record or family-cap
    record) and no attempt of its own or of what it scores never ran, so it is neither complete nor failed. BCLEAN
    without D0-S's trigger is not needed. The verdict of 2026-10-13 was decided with seven arms of that kind
    (deviation 4; BCLEAN, untriggered, was not needed); this guard refuses it. The ticker proposes L23L only once
    every item is terminal (diagnose_stream.v3fam_verdict_ready); an arm that started and failed, or was refused, is
    decided 'not run' with its reason (V3F.3.5). An item the ticker refused at the family cap (cap_refused, the
    verdict's --cap-refused, or a cap record already written) is a recorded 'not run (family cap)', the data
    build's too.
    An item the ticker recorded failed after its own second failure (failed, the verdict's --ticker-failed, or a
    failure record already written) ends the arms it ends (ending_items), as the ticker counts them: a job can fail
    before its provenance attempt is written. The data build's failure ends nothing: its summary is needed.
    Deviation 5: while the continuation of 2026-10-22 is the one in place (the second continuation is due), each
    arm deviation 5 reopened (DEV5_ARMS) needs a record written since (complete, or a failure or family-cap record
    the ticker gave after the reopening): the score jobs of 2026-10-21, refused by the platform's own stale
    records, do not count as its start, and their failure records (superseded by sha256) are not read."""
    from . import v3fam_data as VD
    trig = d0s_trigger() if trig is None else trig
    refused = {str(x).strip() for x in cap_refused or () if str(x).strip()}
    gone = set(_failed_items(failed))
    second = second_continuation_due()
    why = []
    if VD.read_summary() is None and "data" not in refused and not cap_refused_item("data"):
        why.append("the family's data build has no complete %s" % VD._summary_path())
    never, again = [], []
    for a in F.SCORE_ARMS:
        st = arm_status(a)
        if st != "missing":
            continue
        if a == "BCLEAN" and not trig.get("bclean"):
            continue
        if refused & ({"score:%s" % a, "data"} | set(SCORE_DEPS.get(a, ()))):
            continue
        if gone & set(ending_items(a)):
            continue
        if second and a in DEV5_ARMS:
            # deviation 5: reopened after its score jobs of 2026-10-21 were refused by the platform's stale records;
            # those attempts do not start it again: it needs a record (or the ticker's failure) since the reopening
            again.append(a)
            continue
        if not arm_started(a):
            never.append(a)
    if never:
        why.append("never started: %s" % ", ".join(never))
    if again:
        why.append("not scored since deviation 5 reopened them (%s): %s" % (DEVIATION_5, ", ".join(again)))
    return (not why), "; ".join(why)


def _reproduces(old, new):
    """[(id, field, old, new)] where a comparison the old record decided is not reproduced by the new one."""
    bad = []
    for cid, o in sorted((old or {}).items()):
        if not isinstance(o, dict) or o.get("status") != "decided":
            continue
        n = (new or {}).get(cid) or {}
        if n.get("status") != "decided":
            bad.append((cid, "status", "decided", n.get("status")))
            continue
        od, nd = o.get("decision") or {}, n.get("decision") or {}
        for k in REPRO_FIELDS:
            a, b = od.get(k), nd.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                if abs(float(a) - float(b)) > CONT_REPRO_TOL:
                    bad.append((cid, k, a, b))
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(
                    isinstance(x, (int, float)) and isinstance(y, (int, float)) for x, y in zip(a, b)):
                if any(abs(float(x) - float(y)) > CONT_REPRO_TOL for x, y in zip(a, b)):
                    bad.append((cid, k, a, b))
            elif a != b:
                bad.append((cid, k, a, b))
    return bad


def _decided(rec):
    """{id: comparison} of a verdict record: its primaries, Tier S and Tier C comparisons."""
    rec = rec or {}
    out = dict(rec.get("primaries") or {}, **(rec.get("tier_s") or {}))
    out.update({"C_%s" % a: v for a, v in ((rec.get("challengers") or {}).get("comparisons") or {}).items()})
    return out


def _prior_records():
    """The earlier verdict records a continuation must reproduce: [(label, verdict path, external path)] -- the
    verdict of 2026-10-13, and when the continuation of 2026-10-22 is in place (deviation 5), that one too."""
    out = [("2026-10-13", verdict_path(), cap_path(EXTERNAL))]
    if is_first_continuation():
        out.append(("2026-10-22", cap_path(CONTINUATION["verdict"]), cap_path(CONTINUATION["external"])))
    return out


def continuation_check(doc, ext):
    """What a continuation carries of the earlier verdicts (the verdict of 2026-10-13; and, for the second
    continuation, the continuation of 2026-10-22, deviation 5): their names and sha256s, and that every comparison
    any of them decided (dev: P1-P5, Tier C, Tier S; external: P6-P9 and Tier S off dev) is reproduced to 1e-9 with
    the same outcome, and every primary any of them family-confirmed stays confirmed (Holm's step-down over the same
    sets can only keep it). Refuses otherwise."""
    priors = _prior_records()
    nc = _decided(doc)
    bad, lost, by_rec, kept = [], [], {}, set()
    for label, vp, ep in priors:
        old = F._read_json(vp) or {}
        olde = F._read_json(ep) if ep.is_file() else None
        oc = _decided(old)
        bad += [(label,) + x for x in _reproduces(oc, nc)]
        if isinstance(olde, dict):
            bad += [(label,) + x for x in _reproduces(dict(olde.get("primaries") or {}, **(olde.get("tier_s") or {})),
                                                      dict(ext.get("primaries") or {}, **(ext.get("tier_s") or {})))]
        conf = sorted(p for p, h in (old.get("holm") or {}).items() if (h or {}).get("family_confirmed"))
        kept |= set(conf)
        lost += ["%s (%s)" % (p, label) for p in conf
                 if not ((doc.get("holm") or {}).get(p) or {}).get("family_confirmed")]
        by_rec[label] = sorted(k for k, v in oc.items() if isinstance(v, dict) and v.get("status") == "decided")
    if bad or lost:
        raise F.V3famRefused("the continuation does not reproduce the verdict of %s (%s): %s%s" % (
            " and the continuation of ".join(lb for lb, _v, _e in priors), VERDICT,
            ["%s %s %s: %r -> %r" % x for x in bad[:4]], "; no longer family-confirmed: %s" % lost if lost else ""))
    first = {"verdict": VERDICT, "sha256": PARTIAL_VERDICT_SHA256, "external": EXTERNAL,
             "external_sha256": F._sha(cap_path(EXTERNAL)) if cap_path(EXTERNAL).is_file() else None,
             "external_as_read_on_2026_10_18": (F._sha(cap_path(EXTERNAL)) == PARTIAL_EXTERNAL_SHA256)
             if cap_path(EXTERNAL).is_file() else False,
             "deviation": DEVIATION_4, "reproduced": by_rec["2026-10-13"], "tolerance": CONT_REPRO_TOL,
             "family_confirmed_kept": sorted(kept),
             "note": "the verdict of 2026-10-13 stays as it is; this record is the family's verdict over every arm, "
                     "by the same rules (deviation 4)"}
    if len(priors) == 1:
        return first
    cx = cap_path(CONTINUATION["external"])
    return dict(first, continuation=CONTINUATION["verdict"], continuation_sha256=CONT_SHA256,
                continuation_external=CONTINUATION["external"],
                continuation_external_sha256=F._sha(cx) if cx.is_file() else None,
                continuation_external_as_read_on_2026_10_24=(F._sha(cx) == CONT_EXTERNAL_SHA256) if cx.is_file()
                else False, deviation=DEVIATION_5, deviations=[DEVIATION_4, DEVIATION_5],
                reproduced=sorted(set(by_rec["2026-10-13"]) | set(by_rec["2026-10-22"])), reproduced_by_record=by_rec,
                note="the verdict of 2026-10-13 and its continuation of 2026-10-22 stay as they are; this record is "
                     "the family's verdict over every arm, by the same rules (deviation 5)")


def verdict(cap_refused=None, ticker_failed=None):
    """L23L (module docstring): the family's comparisons on its committed scores; the non-dev records first, the
    platform's dev record (capacity/v3fam_v1.json) once and last. cap_refused: the items the ticker recorded
    'not run (family cap)' (--cap-refused), written under v3fam/v1/cap first; ticker_failed: the items it recorded
    failed after their own second failure (--ticker-failed), written under v3fam/v1/failed first. Refused unless
    every arm's record is complete or failed (verdict_ready). When capacity/v3fam_v1.json is the verdict of
    2026-10-13 (deviation 4), the same computation writes the continuation's records (CONTINUATION) instead, once,
    after checking that it reproduces every comparison that verdict decided."""
    F.check_frozen()
    supersede()                              # deviation 5: the records it supersedes moved aside first
    vp = verdict_path()
    cont, second = False, False
    if vp.exists():
        if not is_partial(vp):
            raise F.V3famRefused("%s exists: the family's verdict is written once" % vp)
        cont = True
        cp = cap_path(CONTINUATION["verdict"])
        if cp.exists():
            if not is_first_continuation(cp):
                raise F.V3famRefused("%s exists: the family's continuation verdict is written once (deviation 4)" % cp)
            second = True
            c2 = cap_path(CONTINUATION_2["verdict"])
            if c2.exists():
                raise F.V3famRefused("%s exists: the family's second continuation verdict is written once "
                                     "(deviation 5)" % c2)
    names = dict(CONTINUATION_2) if second else dict(CONTINUATION) if cont else {
        "verdict": VERDICT, "external": EXTERNAL, "ext": EXT, "report": REPORT}
    if second:
        # never a name of the verdict of 2026-10-13 or of its continuation: those records stay as they are
        earlier = {VERDICT, EXTERNAL, EXT, REPORT + ".json", REPORT + ".md"} | {
            CONTINUATION[k] for k in ("verdict", "external", "ext")} | {CONTINUATION["report"] + x for x in
                                                                        (".json", ".md")}
        mine = {names[k] for k in ("verdict", "external", "ext")} | {names["report"] + x for x in (".json", ".md")}
        if earlier & mine:
            raise F.V3famRefused("the second continuation would write over %s" % sorted(earlier & mine))
    ok, why = verdict_ready(cap_refused=cap_refused, failed=ticker_failed)
    if not ok:
        raise F.V3famRefused("the verdict waits until every arm's record is complete or failed (V3F.7 step 11; "
                             "deviation 4): %s" % why)
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        ticker = record_ticker_refusals(cap_refused)
        ticker_f = record_ticker_failures(ticker_failed)
        arms = {a: arm_status(a) for a in F.SCORE_ARMS}
        fam = decide_family()
        comps = {k: _strip(v) for k, v in fam["comparisons"].items()}
        holm = fam["holm"]
        dev_ids = [c[0] for c in COMPARISONS if c[1] == "dev"]
        ext_ids = [c[0] for c in COMPARISONS if c[1] != "dev"]
        s_ids = [c[0] for c in TIER_S] + [c[0] for c in TIER_S_ARRAYS]
        s_exam = dict([(c[0], c[1]) for c in TIER_S] + [(c[0], c[1]) for c in TIER_S_ARRAYS])
        passes = dev_passes()
        common = {"family": F.NAME, "decided_by": F.DECIDED_BY, "code": dict(F.code_shas(),
                                                                          v3fam_verdict=C.sha256_file(__file__)),
                  "e_trees": {"files": len(before), "sha256": C.sha256_text(json.dumps(before, sort_keys=True))},
                  "cap_refused": sorted(p.stem for p in F.cap_dir().glob("*.json")) if F.cap_dir().is_dir() else [],
                  "cap_refused_by_ticker": ticker, "failed_by_ticker": ticker_f, "written_utc": F._utc()}
        ext = dict(common, format=FMT_EXTERNAL, status="decided",
                   primaries={p: comps[p] for p in ext_ids}, holm={p: holm[p] for p in ext_ids},
                   tier_s={c: comps[c] for c in s_ids if s_exam[c] != "dev"},
                   note="P6-P9 and the non-dev Tier S readings: never on the platform's evidence (V3F.3.5)")
        dev_doc = {"primaries": {p: comps[p] for p in dev_ids}, "holm": {p: holm[p] for p in dev_ids},
                   "tier_s": {c: comps[c] for c in s_ids if s_exam[c] == "dev"},
                   "challengers": {"comparisons": {a: comps["C_%s" % a] for a in F.CHALLENGERS}}}
        continues = continuation_check(dev_doc, ext) if cont else None   # before anything is written
        if continues:
            common["continues"] = continues
            ext["continues"] = continues
        if second:
            # deviation 5: every record of the second continuation is written once (never over a file); one an
            # attempt cut off before its verdict file left (a full disk, a lost node) is moved aside first
            aside = set_aside_partial(names)
            if aside:
                common["partial_set_aside"] = aside
                ext["partial_set_aside"] = aside
        put = _write_once_json if second else TS._write_json
        put(cap_path(names["external"]), ext)
        put(cap_path(names["ext"]), dict(common, format=FMT_EXT, arms=ext_record(),
                                         note="ImageWeeds Ragweed AP50-95 (common ragweed, 2,010 boxes; other "
                                              "ImageWeeds species scored only as OtherPlant) and agnostic; "
                                              "ooddev_v1 agnostic; reported only"))
        rep = dict(common, format=FMT_REPORT, decided_on=fam["decided_on"], dev_passes=passes["total"],
                   dev_passes_by_kind=passes["by_kind"],
                   comparisons={k: report_row(v) for k, v in comps.items()},
                   full_dev_beside=fam["full_dev_beside"], e3_reread=fam["e3_reread"],
                   d0s_removals=fam["d0s_removals"], holm=holm, gate=fam["gate"], order=fam["order"], arms=arms,
                   table=_table(), generated_utc=F._utc())
        put(cap_path(names["report"] + ".json"), rep)
        if second:
            F.write_once(cap_path(names["report"] + ".md"), _md(rep).encode("utf-8"))
        else:
            cap_path(names["report"] + ".md").write_text(_md(rep))
        doc = dict(common, format=FMT_VERDICT, status="decided", exam="dev", decided_on=fam["decided_on"],
                   dev_minus_h=fam["dev_minus_h"], full_dev_beside=fam["full_dev_beside"],
                   e3_reread=fam["e3_reread"], d0s_removals=fam["d0s_removals"],
                   primaries={p: comps[p] for p in dev_ids}, holm={p: holm[p] for p in dev_ids},
                   challengers=dict(fam["gate"], comparisons={a: comps["C_%s" % a] for a in F.CHALLENGERS}),
                   reads={"order": fam["order"], "cap": F.READ_CAP,
                          "bclean_triggered": any(o["kind"] == "bclean" for o in fam["order"])},
                   tier_s={c: comps[c] for c in s_ids if s_exam[c] == "dev"},
                   arms=arms, dev_passes=passes["total"], dev_passes_by_kind=passes["by_kind"],
                   sesoi=F.SESOI, rule="V3F.3.3: D > 2 pooled sd and D > SE; combined z; 90 % interval; outcome "
                                       "classes; Holm per exam; Tier C bar Phi^-1(1 - 0.05/14)",
                   note="dev only; the external primaries in capacity/%s; nothing switches" % names["external"])
        F.write_once(cap_path(names["verdict"]), doc)
    log("%s: %s; reads %s (%.0fs)" % ("second continuation verdict (deviation 5)" if second else
                                      "continuation verdict (deviation 4)" if cont else "verdict", ", ".join(
        "%s %s" % (p, (doc["primaries"].get(p) or {}).get("decision", {}).get(
            "outcome", doc["primaries"].get(p, {}).get("status"))) for p in dev_ids),
        [o["arm"] for o in fam["order"]], time.time() - t0))
    return doc


def read_verdict():
    """The family's verdict the reads follow (family_verdict_path): decided, or refused. While capacity/v3fam_v1.json
    is the verdict of 2026-10-13, decided while seven arms had never started, every read waits for its continuation
    (deviation 4)."""
    vp = family_verdict_path()
    d = F._read_json(vp)
    if not isinstance(d, dict) or d.get("format") != FMT_VERDICT or d.get("status") != "decided":
        extra = (" (capacity/%s is the verdict of 2026-10-13, decided while seven arms had never started: the reads "
                 "follow its continuation, deviation 4)" % VERDICT) if is_partial() else ""
        if second_continuation_due():
            extra = (" (capacity/%s, the continuation of 2026-10-22, recorded six arms failed whose score jobs the "
                     "platform's stale reproduction records had refused: the reads follow its second continuation, "
                     "deviation 5)" % CONTINUATION["verdict"])
        raise F.V3famRefused("the family's verdict (%s) is not decided: no sealed-test read before it (V3F.3.6)%s"
                             % (vp, extra))
    return d


# ------------------------------------------------------------------ the family's GPU-h (decision 25)
def _hours(a, b):
    try:
        t0 = datetime.datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.datetime.strptime(b, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None
    return max(0.0, (t1 - t0).total_seconds() / 3600.0)


def _estimate(name):
    for k, v in FAMILY_ESTIMATES.items():
        if name == k or (k.endswith("_") and name.startswith(k)) or (k == "v3fam_detr_s" and name.startswith(k)):
            return float(v)
    return SCORE_ESTIMATE_GPUH


def family_job(name):
    return name.startswith("v3fam") or name in RC.V3FAM_EXPS.values()


def spend_items(names=None, exclude_job=None):
    """The family's items as the cluster's records give them (inc2.v3fam.family_gpuh's input): every attempt of
    every family job's provenance (one V100 per job: finished wall time, or its estimate while running; every
    attempt, so a retry adds to the first), every sealed-test read prepared by test-read whose score-test job has
    not started (its read.json written, no attempt in its provenance yet: at its estimate, so a read prepared and
    not yet submitted counts against the cap as a proposed item does), and every family experiment's training runs
    (report.json's GPU-h once done, else exp.json's high estimate). exclude_job: the SLURM job id of the job asking,
    whose own running attempt is left out (it adds its own estimate)."""
    out = []
    pdir = C.INC_DIR / "_campaign" / "provenance"
    started = set()
    for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        name = p.stem
        if not family_job(name) or (names is not None and not any(name.startswith(n) for n in names)):
            continue
        rec = F._read_json(p) or {}
        for i, att in enumerate(rec.get("attempts") or []):
            started.add(name)
            if exclude_job and str(att.get("job_id") or "") == str(exclude_job) and not att.get("finished_utc"):
                continue
            h = _hours(att.get("started_utc"), att.get("finished_utc"))
            out.append({"item": "%s#%d" % (name, i), "state": "ended" if att.get("finished_utc") else "running",
                        "estimate": _estimate(name), "settled": h})
    tdir = F.root() / "test"
    for rj in sorted(tdir.glob("*/read.json")) if tdir.is_dir() else []:
        name = "v3famtest_%s" % rj.parent.name.lower()
        if name in started or (names is not None and not any(name.startswith(n) for n in names)):
            continue
        out.append({"item": "%s:prepared" % name, "state": "proposed", "estimate": READ_GPUH})
    if names is None:
        for exp in RC.V3FAM_EXPS.values():
            defn = F._read_json(C.INC_DIR / exp / "exp.json")
            if not isinstance(defn, dict):
                continue
            rep = F._read_json(C.INC_DIR / exp / "report.json") or {}
            est = float(((defn.get("cost_estimate") or {}).get("total_gpu_h") or [0, 0])[-1] or 0)
            if rep.get("done"):
                gh = rep.get("gpu_hours") or {}
                got = sum(float((v or {}).get("hours") or 0) for v in gh.values() if isinstance(v, dict))
                out.append({"item": "%s:runs" % exp, "state": "ended", "estimate": est, "settled": got})
            else:
                out.append({"item": "%s:runs" % exp, "state": "running", "estimate": est})
    return out


def spend():
    items = spend_items()
    return {"items": items, "gpuh": F.family_gpuh(items), "cap_gpuh": F.FAMILY_CAP_GPUH}


def person_spend():
    """The person steps' GPU-h (the sweep, the test reads, the re-analysis): what the ticker adds to its own."""
    items = spend_items(PERSON_JOBS)
    return {"items": items, "gpuh": F.family_gpuh(items)}


# ------------------------------------------------------------------ sealed-test reads (V3F.3.6)
def reads_done():
    """{arm: "read" | not run (family cap)} of the family's reads so far."""
    out = {}
    base = F.root() / "test"
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if (d / "read.json").exists():
                out[d.name] = "read"
    for p in sorted(F.cap_dir().glob("test_read_*.json")) if F.cap_dir().is_dir() else []:
        out.setdefault(p.stem[len("test_read_"):], F.NOT_RUN_CAP)
    return out


def read_sources(arm):
    """What a read of `arm` scores, pinned by sha256: its runs or exports, and its classifier."""
    if arm in F.REFERENCE_READS:
        return SYS.reference_sources(arm)
    if arm in ("D4F", "D4N", "D4R15", "BCLEAN"):
        from . import v3fam_trained as VT
        build = VT.TWO_STAGE_BUILD[arm]
        srcs = VT.sources(build)
        return {"mode": "two_stage", "build": build, "seeds": [SYS.source_record(srcs[s]) for s in F.SEEDS],
                "classifier": TS._clf_cfg()[2]}
    if arm == "D3":
        from . import v3fam_wbf as WB
        return {"mode": "inject_boxes", "stage1": [SYS.source_record(SYS.e3_source("B", s)) for s in F.SEEDS],
                "library": WB.wbf_impl()[1], "classifier": TS._clf_cfg()[2]}
    if arm in F.CHALLENGERS and arm.startswith("D2"):
        from . import v3fam_embed as EMB
        cl = F._read_json(EMB.pin_path(EMB.ARM_KEY[arm]))
        return {"mode": "view640" if EMB.ARM_CUT[arm] == "view640" else "two_stage",
                "stage1": [SYS.source_record(SYS.e3_source("B", s)) for s in F.SEEDS],
                "classifier_pin_sha256": TS._sha(EMB.pin_path(EMB.ARM_KEY[arm])), "pin": cl}
    raise F.V3famRefused("%s has no sealed-test read (V3F.3.6)" % arm)


def test_read(arm, read_by):
    """Prepare one sealed-test read (module docstring): checked before anything is written."""
    who = F.read_by_value(read_by)
    v = read_verdict()
    order = list((v.get("reads") or {}).get("order") or [])
    done = reads_done()
    pos = F.check_read(arm, order, done)
    td = test_dir(arm)
    if (td / "read.json").exists() or sorted(td.glob("s*/test.json")) or sorted(td.glob("*/test.json")):
        raise F.V3famRefused("%s's read was prepared or taken already: a read is taken once" % arm)
    try:
        before = F.check_cap(spend_items(), "test_read_%s" % arm, READ_GPUH)
    except F.V3famCapRefused:
        log("%s: %s" % (arm, F.NOT_RUN_CAP))
        raise
    srcs = read_sources(arm)
    logs = C.INC_DIR / "logs"
    # --export=INCAP_DECIDED_BY (deviation 3, 2026-10-12): Bridges-2's SBATCH_EXPORT=NONE gives a job none of the
    # submitter's variables unless sbatch names them, and check_submitter refuses a read whose job lacks it
    argv = ["sbatch", "--parsable", "--export=%s" % SUBMITTER_ENV, "--job-name=inc_build_v3famtest_%s" % arm.lower(),
            "--output=%s" % (logs / "%x_%j.out"), str(TS._build_script()), "inc2.v3fam", "score-test", "--arm", arm]
    env = DELEGATE_ACCOUNT if read_by == "delegate" else "the owner's account (%s...)" % DECIDED_BY_OWNER_PREFIX
    rec = {"format": FMT_READ, "arm": arm, "kind": pos["kind"], "position": pos["position"], "read_by": who,
           "read_by_key": read_by, "incap_decided_by": env, "verdict": family_verdict_path().name,
           "verdict_sha256": F._sha(family_verdict_path()), "sources": srcs, "argv": argv, "cap": before,
           "decided_by": F.DECIDED_BY, "written_utc": F._utc(),
           "note": "submitted by the person named in read_by with INCAP_DECIDED_BY naming them; the scores stay off "
                   "the platform's evidence (V3F.3.6)"}
    logs.mkdir(parents=True, exist_ok=True)
    F.write_once(td / "read.json", json.loads(json.dumps(rec, default=str)))
    return rec


def check_submitter(read_by_key, env=None):
    """The read-path guard (decision 20): INCAP_DECIDED_BY must name the person read_by names (the owner's
    account, human:..., or the delegate's assistant:claude-delegate). Returns the value."""
    got = (os.environ if env is None else env).get(SUBMITTER_ENV) or ""
    if read_by_key == "delegate":
        ok = got == DELEGATE_ACCOUNT
    elif read_by_key == "owner":
        ok = got.startswith(DECIDED_BY_OWNER_PREFIX) and len(got) > len(DECIDED_BY_OWNER_PREFIX)
    else:
        ok = False
    if not ok:
        raise F.V3famRefused("INCAP_DECIDED_BY is %r, not the person read_by names (%s): a read submitted by anyone "
                             "else is not a read under the amendment (V3F.3.6)%s" % (
                                 got, read_by_key, "" if got else "; on Bridges-2 sbatch passes it to the job only "
                                 "when named (sbatch --export=INCAP_DECIDED_BY ..., as read.json's argv does)"))
    return got


def _ref_test(token, kw, wk):
    """b_v2_m640's three recorded test scores reproduced through the family identity path (V3F.3.6 item 3);
    written once under v3fam/v1/test/REF/s<k>, kept for a later read."""
    out = []
    names = F.equivalence_names("test")
    for s in F.SEEDS:
        d = test_dir(REF_TEST_DIR) / ("s%d" % s)
        if (d / "test.json").exists():
            doc = F._read_json(d / "test.json") or {}
            if doc.get("repro_status") not in ("passed", "deviation"):
                raise F.V3famRefused("%s does not record a reproduced reference test score" % (d / "test.json"))
            out.append({"seed": s, "status": "kept", "repro": doc.get("repro_status")})
            continue
        src = SYS.e3_source("M", s)
        recorded = F._read_json(C.INC_DIR / TS.REFERENCE / "runs" / src["final_run_id"] / "scores" / "test.json")
        if not isinstance(recorded, dict):
            raise F.V3famRefused("b_v2_m640 seed %d has no recorded test score" % s)
        cfg = {"mode": "identity", "nms": "locked", "workers": wk}
        ps = F._family_pass(src["weights"], "test", cfg, "E3RS", test_token=token, **kw)
        cmp_ = F.compare_with_recorded(ps["result"], recorded, "b_v2_m640 seed %d test" % s)
        st = F.repro_status(cmp_["max_abs_diff"])
        if st == "refused":
            raise F.V3famRefused("b_v2_m640 seed %d's test score is not reproduced (%.4g > %g): no family test "
                                 "score is taken" % (s, cmp_["max_abs_diff"], F.DEVIATION_TOL))
        doc = F.family_doc("REF", s, "test", ps, cfg, equivalence=F.require_equivalence(names),
                           extra={"source": SYS.source_record(src), "vs_recorded": cmp_, "repro_status": st})
        F.commit_pass(d, "test", doc, ps)
        out.append({"seed": s, "status": "written", "repro": st})
    return out


def _test_passes(arm, token, kw, wk, embedder=None):
    """[(seed or None, ps, cfg, extra, classifier)] of arm's test passes (the read's inference)."""
    out = []
    if arm == "D8M":
        for srec in SYS.reference_sources(arm)["seeds"]:
            src = SYS.run_source(SYS.D8_ARMS[arm]["exp"], srec["seed"], testing_ok=TS._testing_ok())
            cfg = {"mode": "identity", "nms": "locked"}
            out.append((srec["seed"], F._family_pass(src["weights"], "test", cfg, "D8", test_token=token, **kw), cfg,
                        {"source": SYS.source_record(src)}, None))
        return out
    if arm == "D6":
        from . import v3fam_detr as DT
        backend = DT.default_backend(None)
        scaffold = SYS.e3_source("M", 0)
        for s in F.SEEDS:
            fr = DT.final_record(s)
            model = backend.load_model(fr["path"])
            rows, st = DT.predict_exam(backend, model, "test")
            cfg = {"mode": "inject_rows", "nms": "locked", "inject": rows}
            out.append((s, F._family_pass(scaffold["weights"], "test", cfg, "D6", test_token=token, **kw), cfg,
                        {"detr": {"final": {k: v for k, v in fr.items() if k != "path"}, "predictions": st},
                         "scaffold": SYS.source_record(scaffold)}, None))
        return out
    ccfg, _crec, shas = TS._clf_cfg()
    if arm in ("D4F", "D4N", "D4R15", "BCLEAN"):
        from . import v3fam_trained as VT
        srcs = VT.sources(VT.TWO_STAGE_BUILD[arm])
        embedder = embedder or V.BioclipEmbedder()
        for s in F.SEEDS:
            cfg = SYS._two_stage_cfg(ccfg, "B", embedder, wk)
            out.append((s, F._family_pass(srcs[s]["weights"], "test", cfg, arm, test_token=token, **kw), cfg,
                        {"source": SYS.source_record(srcs[s])}, shas))
        return out
    if arm == "D3":
        from . import v3fam_wbf as WB
        fn, impl = WB.wbf_impl()
        embedder = embedder or V.BioclipEmbedder()
        stage1s = []
        for s in F.SEEDS:
            cfg = SYS._two_stage_cfg(ccfg, "B", embedder, wk)
            stage1s.append(F._family_pass(SYS.e3_source("B", s)["weights"], "test", cfg, "D3", test_token=token,
                                          **kw)["stage1"])
        fused, counts = WB.fuse_exam(stage1s, SYS.image_sizes(SYS.exam_rows("test")), fn)
        scaffold = SYS.e3_source(*WB.SCAFFOLD)
        cfg = dict(ccfg, mode="inject_boxes", nms="locked", embedder=embedder, workers=wk, top_k=TS.TOP_K,
                   inject=fused)
        out.append((None, F._family_pass(scaffold["weights"], "test", cfg, "D3", test_token=token, **kw), cfg,
                    {"fusion": {"library": impl, "counts": counts}, "one_system": True}, shas))
        return out
    if arm.startswith("D2"):
        from . import v3fam_embed as EMB
        bb = EMB.ARM_BACKBONE[arm]
        for s in F.SEEDS:
            if arm == EMB.FT_ARM:
                ck = EMB.checkpoint(bb)
                emb, finfo = EMB._ft_embedder(s, ck)
                rows, _r = EMB.training_crops(images=False)
                import numpy as np
                counts = np.bincount([r["label"] for r in rows], minlength=C.NC).astype(np.float64)
                cfg = dict(EMB.ft_classifier_cfg(counts / counts.sum()), mode="two_stage", nms=TS.NMS["B"],
                           embedder=emb, workers=wk, top_k=TS.TOP_K)
                clf = {"key": "FT", "pin_sha256": finfo["pin_sha256"], "weights_sha256": finfo["sha256"]}
            else:
                emb, _ck = EMB.load_embedder(bb)
                dcfg, _rec, clf = EMB.load_d2_classifier(EMB.ARM_KEY[arm])
                mode = "view640" if EMB.ARM_CUT[arm] == "view640" else "two_stage"
                cfg = dict(dcfg, mode=mode, nms=TS.NMS["B"], embedder=emb, workers=wk, top_k=TS.TOP_K)
                if mode == "view640":
                    cfg["cut"] = EMB.e3_cut640
            src = SYS.e3_source("B", s)
            out.append((s, F._family_pass(src["weights"], "test", cfg, arm, test_token=token, **kw), cfg,
                        {"stage1_source": SYS.source_record(src)}, clf))
        return out
    raise F.V3famRefused("%s has no sealed-test read" % arm)


def score_test(arm, embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, env=None):
    """The read's job (module docstring): the submitter checked, the reference reproduced, the arm scored once."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    td = test_dir(arm)
    rr = F._read_json(td / "read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_READ or rr.get("arm") != arm:
        raise F.V3famRefused("%s has no prepared read (test-read --arm %s --read-by ... first)" % (arm, arm))
    submitter = check_submitter(rr.get("read_by_key"), env)
    read_verdict()
    if F._sha(family_verdict_path()) != rr.get("verdict_sha256"):
        raise F.V3famRefused("the verdict changed since the read was prepared")
    now = json.loads(json.dumps(read_sources(arm), default=str))
    if now != rr.get("sources"):
        raise F.V3famRefused("%s's weights or classifier changed since the read was prepared" % arm)
    done = sorted(str(p) for p in td.glob("*/test.json"))
    if done:
        raise F.V3famRefused("%s's test was scored already (%s): read once" % (arm, done))
    names = F.equivalence_names("test", e3_arms=("B",) if arm not in ("D6", "D8M") else ())
    t0 = time.time()
    out = {"reference": [], "arm": []}
    with F.e_trees_unchanged():
        with F.test_read_token() as token:
            with TS._workers(workers) as wk:
                out["reference"] = _ref_test(token, kw, wk)
                eqs = F.require_equivalence(names)
                for s, ps, cfg, extra, clf in _test_passes(arm, token, kw, wk, embedder=embedder):
                    x = dict(extra, read={"read_by": rr["read_by"], "incap_decided_by": submitter,
                                          "read_sha256": F._sha(td / "read.json"), "kind": rr["kind"]})
                    doc = F.family_doc(arm, s, "test", ps, cfg, equivalence=eqs, classifier=clf, extra=x)
                    F.commit_pass(td / ("s%d" % s if s is not None else "ens"), "test", doc, ps)
                    out["arm"].append({"seed": s, "species_map50_95": doc.get("species_map50_95")})
    log("%s's sealed test scored (%s), %.0fs; test-report --arm %s" % (arm, out["arm"], time.time() - t0, arm))
    return out


def _arrays_mean(runs, keys=None):
    vals = []
    for a in runs:
        b = F.subset_arrays(a, keys) if keys is not None else a
        vals.append(F.statistic_value(b, "species"))
    return vals


def test_report(arm):
    """The read's report, for people (capacity/v3fam_test_<x>.{json,md}): the arm's and the reference's test
    species mean +- sd, agnostic, the gap to 0.90, the stratum of the test images from dev's sessions."""
    rr = F._read_json(test_dir(arm) / "read.json")
    if not isinstance(rr, dict) or rr.get("format") != FMT_READ:
        raise F.V3famRefused("%s has no prepared read" % arm)
    base = test_dir(arm)
    got, ref = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        doc = F._read_json(d / "test.json")
        if isinstance(doc, dict):
            got.append((doc, SC.load_npz(d / doc["images"]["name"])))
    for s in F.SEEDS:
        d = test_dir(REF_TEST_DIR) / ("s%d" % s)
        doc = F._read_json(d / "test.json")
        if isinstance(doc, dict):
            ref.append((doc, SC.load_npz(d / doc["images"]["name"])))
    if not got:
        raise F.V3famRefused("%s's test has not been scored (score-test)" % arm)
    stratum = F.dev_session_stratum(SYS.exam_rows("test"), SYS.exam_rows("dev"))

    def side_of(pairs):
        if not pairs:
            return None
        sp = [float(d["species_map50_95"]) for d, _a in pairs]
        ag = [float(d.get("agnostic_map50_95")) for d, _a in pairs if d.get("agnostic_map50_95") is not None]
        st = _arrays_mean([a for _d, a in pairs], stratum)
        m, s = F._mean_sd(sp)
        return {"species": sp, "mean": m, "sd": s, "gap_to_target": TARGET - m, "agnostic": ag,
                "agnostic_mean": F._mean_sd(ag)[0] if ag else None,
                "stratum": {"n_images": len(stratum), "species": st, "mean": F._mean_sd(st)[0]}}
    rep = {"format": FMT_TEST_REPORT, "arm": arm, "label": LABELS.get(arm, arm), "kind": rr.get("kind"),
           "read_by": rr.get("read_by"), "target": TARGET, "arm_test": side_of(got), "reference_test": side_of(ref),
           "stratum_rule": "the test images whose session is one of dev's (V3F.3.6; 329 of 1,977 on 2026-10-11)",
           "note": "descriptive; a reference or challenger test number is a separate labelled row, never the "
                   "headline (V3F.3.6)", "written_utc": F._utc()}
    p = cap_path(TEST_RECORD % arm.lower())
    TS._write_json(p.with_suffix(".json"), rep)
    a = rep["arm_test"]
    p.with_suffix(".md").write_text(
        "# %s's sealed test (%s read, read by %s)\n\n12-class mAP50-95 %.4f +- %.4f (gap to 0.90: %.4f); the %d test "
        "images from dev's sessions: %.4f. Reference b_v2_m640: %s.\n" % (
            rep["label"], rep["kind"], rep["read_by"], a["mean"], a["sd"] or 0.0, a["gap_to_target"],
            a["stratum"]["n_images"], a["stratum"]["mean"] or 0.0,
            "%.4f" % rep["reference_test"]["mean"] if rep["reference_test"] else "not scored"))
    return rep


def reanalyse_test(read_by, env=None):
    """O4's re-analysis of stored test arrays (no inference; module docstring): capacity/v3fam_retest.json once. It is
    the last of the reads (V3F.3.6, decision 25): it refuses while a read of the verdict's order is neither prepared
    nor recorded not run, so it never takes the cap's room from a read before it. Its own job's running attempt
    is left out of the spend it checks (it adds its own estimate)."""
    who = F.read_by_value(read_by)
    env = os.environ if env is None else env
    submitter = check_submitter(read_by, env)
    v = read_verdict()
    out = cap_path(RETEST)
    if out.exists():
        raise F.V3famRefused("%s exists: the re-analysis runs once" % out)
    order = [o["arm"] for o in (v.get("reads") or {}).get("order") or []]
    done = reads_done()
    left = [a for a in order if a not in done]
    if left:
        raise F.V3famRefused("the re-analysis comes after the reads (V3F.3.6, decision 25): %s first (prepared with "
                             "test-read, or recorded %s)" % (left, F.NOT_RUN_CAP))
    t0 = time.time()
    with F.e_trees_unchanged():
        F.check_cap(spend_items(exclude_job=env.get("SLURM_JOB_ID")), "reanalyse_test", RETEST_GPUH)
        stratum = F.dev_session_stratum(SYS.exam_rows("test"), SYS.exam_rows("dev"))
        trig = d0s_trigger()
        h = [str(k)[len("test/"):] for k in trig.get("h_test") or []]
        minus = [r["key"] for r in SYS.exam_rows("test") if r["key"] not in set(h)] if h else None
        rows = {}
        for name, sub in (("A", "A"), ("M", "M"), ("B", "B"), ("reference", "reference")):
            pairs = []
            for s in F.SEEDS:
                d = TS.root() / "test" / sub / ("s%d" % s)
                jn = "test.identity.json" if sub == "reference" else "test.json"
                doc = F._read_json(d / jn)
                if not isinstance(doc, dict):
                    continue
                pairs.append(SC.load_npz(d / doc["images"]["name"]))
            if len(pairs) != len(F.SEEDS):
                rows[name] = {"status": "missing"}
                continue
            st = _arrays_mean(pairs, stratum)
            rows[name] = {"status": "ok", "stratum": {"species": st, "mean": F._mean_sd(st)[0]}}
            if minus is not None:
                tm = _arrays_mean(pairs, minus)
                rows[name]["test_minus_h"] = {"species": tm, "mean": F._mean_sd(tm)[0]}
        rec = {"format": FMT_RETEST, "read_by": who, "incap_decided_by": submitter, "stratum_images": len(stratum),
               "test_minus_h_images": None if minus is None else len(minus), "arms": rows, "reads_before": order,
               "decided_by": F.DECIDED_BY, "written_utc": F._utc(),
               "note": "re-analysis of E3's stored test arrays, no inference; descriptive (V3F.3.6)"}
        F.write_once(out, rec)
    log("re-analysis written (%.0fs)" % (time.time() - t0))
    return rec


# ------------------------------------------------------------------ the verdict's CLI (deviation 5)
def _items(text):
    return [x for x in (text or "").split(",") if x]


def main(argv=None):
    """python -m weed_optimizer_framework.tools.inc2.v3fam_verdict verdict [--cap-refused ITEM,...]
    [--ticker-failed ITEM,...] (L23L). Here, not in inc2/v3fam.py: every reproduction record pins that file's
    sha256 (inc2.v3fam.code_shas), so the ticker's plumbing never changes it (deviation 5). A refusal prints
    "[inc2.v3fam_verdict] ERROR: ..." and exits 2; any other error exits 1."""
    ap = argparse.ArgumentParser(description="V3F's verdict (L23L): the family's comparisons on its committed scores, "
                                             "written once (deviations 4 and 5: its continuations).")
    ap.add_argument("command", choices=("verdict",))
    ap.add_argument("--cap-refused", default=None,
                    help="the family items the ticker recorded 'not run (family cap)', comma-separated")
    ap.add_argument("--ticker-failed", default=None,
                    help="the score, build and D6 items the ticker recorded failed after their own second failure, "
                         "comma-separated (deviation 4)")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        verdict(cap_refused=_items(a.cap_refused), ticker_failed=_items(a.ticker_failed))
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, V.VerifyError, SUP.SupersedeRefused,
            ValueError) as e:
        print("[inc2.v3fam_verdict] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    # run the package's own instance of this module (python -m loads this file a second time as __main__; every
    # other module of the family imports the package's), so one module state and one set of classes serve the job
    import importlib
    sys.exit(importlib.import_module(__spec__.name if __spec__ else
                                     "weed_optimizer_framework.tools.inc2.v3fam_verdict").main())
