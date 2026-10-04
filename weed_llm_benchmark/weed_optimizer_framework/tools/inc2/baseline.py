"""Baselines on splits v2 (docs/CONTINUOUS_LOOP.md §4.4, §3.6, §2.6 L-4):
B_v2, its capacity arms, the canary, B0 u tsw and the stream's milestones.

    python -m weed_optimizer_framework.tools.inc2.baseline build --exp E (--manifest PATH | --union P1,P2,...)
        [--seeds 0,1,2] [--arm n640|s640|m640|m832|s1024 | --arch yolo11s --imgsz 640]
        [--final-exams dev,imageweeds,test]
        [--role b_v2|capacity|canary|union|milestone|baseline] [--testing | --testing-settings JSON] [--no-init]
    python -m weed_optimizer_framework.tools.inc2.baseline canary-verdict   --exp canary_v2 [--reference b0_v1]
    python -m weed_optimizer_framework.tools.inc2.baseline capacity-verdict --n b_v2 --arms b_v2_s640,b_v2_m640
        [--record b_v2_m832,b_v2_s1024] [--m M] [--out-dir DIR]
    python -m weed_optimizer_framework.tools.inc2.baseline secondary --exp <sid>_mNNN --weights PATH
        [--run-id secondary__incumbent] [--source TEXT]
    python -m weed_optimizer_framework.tools.inc2.baseline estimate --n-images N [--seeds 0,1,2] [--arm A]
    python -m weed_optimizer_framework.tools.inc2.baseline rescore-native --exp b_v2_m832 [--reference b_v2_m640]
        [--out-dir DIR]
    python -m weed_optimizer_framework.tools.inc2.baseline native-verdict [--arms b_v2_m832,b_v2_s1024]
        [--reference b_v2_m640] [--out-dir DIR]
    python -m weed_optimizer_framework.tools.inc2.baseline build --exp e2_w_m640_seed0
        --manifest INC_DIR/splits/v2/base_v2.jsonl --seeds 0 --arm m640 --role baseline --e2 W
    python -m weed_optimizer_framework.tools.inc2.baseline rescore-e2 [--out-dir DIR]
    python -m weed_optimizer_framework.tools.inc2.baseline e2-verdict [--out-dir DIR]
    python -m weed_optimizer_framework.tools.inc2.baseline e2-test-read --e2 W|S
    python -m weed_optimizer_framework.tools.inc2.baseline e2-test-report --e2 W|S [--out-dir DIR]

build writes a 'baseline' experiment for the pinned driver (inc/driver.py,
unchanged): one cold base run per seed (the arm's Protocol v3 cold recipe,
from the arm's COCO checkpoint), scored on dev, then one final run per seed on
the final exams (default splits v2's dev, imageweeds and test; test is read
only here, at a milestone, P10). The runs are executed by inc2.train through
run_inc2_job.sh: build sets INC_JOB_SCRIPT to REPO/weed_llm_benchmark/
run_inc2_job.sh before driver init (and refuses when it already names another
script), so the driver never submits the v1 executor for a v2 experiment.

Before anything is written (fail closed):
  * the arm resolves (inc2.recipes.resolve_arm): its checkpoint is in REPO and
    its sha256 is pinned into exp.json's "arm" (a testing build may lack it);
  * splits v2 are locked: LOCK v2 exists and the never-train index hashes as
    it records; a production build refuses a LOCK a testing build wrote;
  * the manifest passes inc2.train's own checks, the ones every base run
    repeats: check_manifest (keys, labels, image and label bytes, not an
    evaluation manifest of v1 or v2, by path or content) and guard_rows (the
    v2 never-train guard over the 8 flips and rotations, GuardV2 plus the
    index cross-check). --union checks every part, then that the parts are
    pairwise disjoint by key, image path and image sha256, then the guard
    over the union;
  * the definition passes the pinned driver's validate_definition and
    check_definition_data.
The manifest is copied into manifests/<sanitised stem>.jsonl (same bytes); a
union is written there as <exp>_union.jsonl. exp.json carries
inc2.recipes.stamp(arm) (protocol v3, protocol_package inc2, splits v2, the
arm record, init_weights = the arm's checkpoint), the role, the source
manifest(s) and whether LOCK v2 records it (source_locked), and
cost_estimate (inc2.recipes.baseline_cost: est., low and high GPU-h, the
projected longest run against the 8 h cold walltime) and research_only (the
§8 taint of the base's rows, from splits v2's base_v2_provenance.jsonl
when it hashes as LOCK v2 records: false only when every row is known and
none is research-only, else true). build_summary.json
records the manifest summary, the guard record, the LOCK status and the
sha256 of every input.

Roles and their contract rows (§4.4, §3.6, L-4):
  b_v2       LOCK v2's base_v2, arm n640, seeds 0..4 (milestone 0); finals
             dev, imageweeds, test
  capacity   LOCK v2's base_v2, an arm other than n640 (L-4's s640 or m640,
             or a measurement arm, m832 or s1024), seeds 0..2; finals dev,
             imageweeds, test (L-4: test read once per arm at R0); a
             measurement arm's finals dev, imageweeds (below)
  canary     LOCK v2's train_core (v1's train_core minus the L-8 drops),
             arm n640, seed 0; final exam dev only. exp.json's
             variant_drops records the dropped rows (key, image sha256, the
             evaluation image each matched) and the derivation: v1's
             train_core (b0_v1's base, by sha256) filtered by the L-8 list
             (by sha256, the one LOCK v2 records) is the canary's manifest
             byte for byte, or the build refuses
  union      --union (B0 u tsw), seeds 0..2; finals dev, imageweeds
  milestone  a stream pool P_s, seeds 0..4 (group E's inc2.stream milestone);
             finals dev, imageweeds, test
  baseline   any other manifest, seeds 0..2; finals dev, imageweeds
Test is read only at milestone reads (P10): b_v2, capacity and milestone. A
build of any other role that lists test is refused, and so is a build of a
measurement arm (inc2.recipes.MEASURE_ARMS) whatever its role: it is built
after R0, at no milestone read, so its finals are dev and imageweeds. Without
--role the role is inferred from what is built (the autopilot's L23B argv
names none): --union is union; LOCK v2's base_v2 (by sha256) is b_v2 on n640
and capacity on another arm; LOCK v2's train_core is canary; anything else is
baseline. A role given explicitly must match: b_v2, capacity and canary
refuse another manifest or arm, union needs --union.

The arm is --arm (an id), or --arch and --imgsz together (the autopilot's
L23B argv: --arch yolo11s --imgsz 640 is s640), which must name one row of
inc2.recipes.ARMS; both forms at once must agree.

canary-verdict: the canary's base__s0 dev mAP50-95 against b0_v1's base
seeds (report.json's dev 12-class mean and sd, else the runs' own scores):
passes when |canary - mean| <= sd (§4.4), the canary's scorer sidecar was
written and still hashes as run.json records (a baseline records a failed
sidecar without failing the run, so the canary is where a sidecar that does
not work on the cluster's Ultralytics shows up first), the run and its score
are production ones, and the canary trained what the reference's seed 0
trained (the reference exp.json's base manifest_sha256 and cold recipe, seed
0 among its seeds). Decision L-8 drops train_core rows that b0_v1 trained
on, so the manifest is accepted either as the reference's (same sha256) or
as the reference's minus the L-8 drops: the canary's variant_drops names the
reference's base sha256 and the list's sha256, the list still hashes to it
and LOCK v2 records it, and the reference's manifest (its copy, its source
or v1's train_core, whichever hashes as the reference records) filtered by
the listed image sha256s hashes to the canary's manifest. canary.json
records how the manifest matched and which rows were dropped. Written to
INC_DIR/<exp>/canary.json.

capacity-verdict (L-4's rule, dev only): every arm's base runs on the seeds
all arms share, their dev mAP50-95 (production scores, one scorer, one dev
manifest and key order). An arm qualifies when mean(arm) - mean(n640) >
2 x pooled sd, pooled sd = sqrt((sd(arm)^2 + sd(n640)^2) / 2); the stream
takes the qualifying arm with the highest dev mean (a tie: the fewer FLOPs),
else stays on n640. For the chosen arm: the measured cold rate (median ms
per image-epoch over its base runs' run.json), the incremental rate est. as
cold x 7.4 / 6.5 (the n640 ratio of §5.6), the per-step cost of one r0 step
with its truth arm at N = the base's images and M = ceil(0.10 N), and
truth_every = 1 when that is <= 25 GPU-h, else ceil(cost / 25) (L-4). Writes
INC_DIR/capacity/capacity_v1.json (the decision: it opens no test file) and
capacity_v1_report.{json,md} (for people: every arm's final dev, imageweeds
and test mean +- sd over its seeds, 12-class and agnostic, and the gap to
0.90; the milestone read of test at R0). Only L-4's grid arms
(inc2.recipes.GRID_ARMS) are candidates: an --arms experiment on a
measurement arm (m832, s1024) is refused. --record lists measurement arms
record only: their dev mean, sd and difference from n640 (on the seeds the
grid shares) and their finals (dev and imageweeds: no test, P10) in the
report, never in qualifying or chosen, so the decision is the one the grid
alone gives. With --record, an existing decision file is kept byte for byte
(the stream adopted it by sha256) and only the report is written; a
recomputed decision that differs from it is refused.

secondary: the milestone's second number (§3.6), the chain incumbent scored
on the milestone's final exams: writes runs/<run id>/spec.json (kind final,
init = the incumbent's weights, checked by inc2.train.validate_spec), a
one-line submission list and secondary.json (the weights' sha256 and the
source), and returns the sbatch argv of run_inc2_job.sh for the platform to
submit. It submits nothing itself; the driver does not track that run.

rescore-native (docs/CONTINUOUS_LOOP.md, group B, "Amendment (2026-10-01):
the measurement arms read at their own resolution (pre-registered)"): a
measurement arm's final runs scored by inc2.scorer_native at the arm's
training imgsz on dev and imageweeds (never test), and the reference's
(b_v2_m640) final runs on dev at 640 on the seeds both share, each only when
its native score is missing (a native score is written once; the 640 px
scores are never touched). Every final run must be done first. Writes
INC_DIR/<exp>/native_rescore.json (status complete and the dev files' names
and sha256s, no path: the platform reads it, dev only; the returned record
lists every native file) and then the native verdict. The autopilot's L23N
runs it as one GPU job (run_inc2_build.sh). Like the scorers' own CLIs,
rescore-native and native-verdict set YOLO_AUTOINSTALL=false and
YOLO_OFFLINE=true before Ultralytics is imported.

native-verdict: the pre-registered rule, from the native dev files only.
For each arm, on the seeds it shares with the reference: D = mean(arm's dev
species_map50_95 at its imgsz) - mean(reference's at 640); pooled sd =
sqrt((sd_arm^2 + sd_ref^2) / 2); SE(D) from a paired image bootstrap
(NATIVE_RESAMPLES resamples of the dev images under
stable_int(NATIVE_SEED_TEXT), one draw for every run; per run and resample
each species' AP50-95 by the scorer sidecar's method on its tie-broken
per-image arrays, the 12-class mean over the species with a GT box, the mean
over seeds per arm, the arm minus the reference); each species' difference
and SE the same way. An arm qualifies for a stream fork proposal only when D
> 2 x pooled sd, D > SE(D), and one of NATIVE_TARGET_SPECIES has a higher
mean dev AP at the arm's imgsz than the reference's at 640. Native files of
one arm or of the reference that disagree on the exam manifest, key order,
locked scorer, a setting other than imgsz or Ultralytics' version refuse,
and so does a test-mode score unless testing is allowed (tests). An arm
without its native dev scores (or the reference without its) is pending.
Writes INC_DIR/capacity/native_v1.json (the decision: dev only, no score
path, the platform's card reads it) and native_v1_report.{json,md} (for
people: dev and imageweeds at the native size and at 640). Qualifying
switches nothing: the stream's arm stays the capacity decision's,
capacity_v1.json is not touched, test is not read; the autopilot files card
X18 for a person.

build --e2 W|S (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-04): E2, the
12-class detector on E1-B's backbone (pre-registered)"): one of E2's six
experiments, e2_<w|s>_m640_seed<k>, b_v2_m640's definition except the init
(and for S the recipe, x1b): LOCK v2's base_v2, m640, role baseline (finals
dev and imageweeds), one seed k, init_weights the absolute path of E1-B's
base__s<k>/weights/final.pt. Before anything is written (e2_record):
b_v2_m640's exp.json must match the build in every key that defines training
but the init (the manifest and its image count, the arm record, the protocol
stamp, LOCK v2 and its never-train index, the decision exam; for W the
recipe, key for key) and its base runs must share one recorded training
environment; capacity/e1_v1.json must be a decided verdict that qualifies
E1-B; E1-B must be E1's arm B on m640 with cold_budget; its base run of seed
k must be done, under the current LOCK v2 with a clean guard, and its own
final.pt (no symlink) must hash as its run.json records. exp.json carries the
e2 record (the arm, the seed, the init by path and sha256 with E1-B's run.json
and exp.json sha256s, E1's verdict by sha256 and decision keys, the
reference's definition, what differs from it and what is declared), and the
research_only flag of base_v2 OR E1-B's (fail closed). inc2.train's own
e2_problems must accept the definition before the manifest is copied.

rescore-e2 (lever L23C, one GPU job of run_inc2_build.sh): every E2 final run
scored on dev at 640 by inc2.scorer_native (the locked scorer as a library,
reproducing each run's protocol dev score within 0.002), and any reference
(b_v2_m640) file that is missing; every final run must be done first. Then
the verdict (e2-verdict), then capacity/e2_rescore.json (complete; the files'
names and sha256s and the verdict's sha256, no path: the platform reads it).

e2-verdict: E2_RULE on the native dev files, dev only, record only:
capacity/e2_v1.json (beside e1_v1.json) and e2_v1_report.{json,md} (for
people: ImageWeeds too). A decided verdict is never rewritten: a
recomputation whose decision agrees keeps the file byte for byte, one that
disagrees refuses.

e2-test-read --e2 X: the one read of the sealed test, by a person, for the
arm E2's verdict chose (it must have qualified): one kind-final spec on test
per seed from the arm's base weights (as the verdict read them, by sha256),
one submission list per experiment and the run_inc2_job.sh argvs, all
checked before anything is written; refused before the verdict, for an arm
that is not the choice, and a second time. e2-test-report --e2 X reports
12-class and agnostic test against b_v2_m640's (12-class 0.8786) with the gap
to 0.90 in capacity/e2_test_<X>.{json,md}, which the platform's evidence
does not admit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import driver as D
from ..inc import pilot as P
from ..inc.scorer import TEST_ENV
from ..inc.scorer import testing as scorer_testing
from ..inc.splits import sanitise
from . import common as C2
from . import recipes as RC
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import train as T

BUILDER = "inc2.baseline build"
ROLES = ("b_v2", "capacity", "canary", "union", "milestone", "baseline")
ROLE_SEEDS = {"b_v2": (0, 1, 2, 3, 4), "milestone": (0, 1, 2, 3, 4), "canary": (0,)}
DEFAULT_SEEDS = (0, 1, 2)
FINAL_EXAMS = ("dev", "imageweeds", "test")
CANARY_FINAL_EXAMS = ("dev",)
NO_TEST_FINAL_EXAMS = ("dev", "imageweeds")
# P10 (L-1): test is read only at milestone reads -- milestone 0 (b_v2), the
# capacity arms at R0 (L-4) and the stream's milestones. A measurement arm
# (inc2.recipes.MEASURE_ARMS) is built after R0, at no milestone read: its
# finals are dev and imageweeds whatever its role, and test is refused.
TEST_ROLES = ("b_v2", "capacity", "milestone")
ROLE_FINAL_EXAMS = {"b_v2": FINAL_EXAMS, "capacity": FINAL_EXAMS, "milestone": FINAL_EXAMS,
                    "canary": CANARY_FINAL_EXAMS, "union": NO_TEST_FINAL_EXAMS, "baseline": NO_TEST_FINAL_EXAMS}
# The LOCK v2 manifest a role trains (by sha256), and the arm it needs.
ROLE_MANIFEST = {"b_v2": "base_v2", "capacity": "base_v2", "canary": "train_core"}
JOB_SCRIPT_NAME = "run_inc2_job.sh"
BUILD_SUMMARY = "build_summary.json"
CAPACITY_FORMAT = "inc2-capacity/1"
CAPACITY_NAME = "capacity_v1"
CANARY_FORMAT = "inc2-canary/1"
SECONDARY_FORMAT = "inc2-secondary/1"
TARGET_TEST = 0.90                    # the success measure's target (docs/CONTINUOUS_LOOP.md 1.2)
M_SHARE = 0.10                        # M = ceil(0.10 x |base|) (5.3)
INC_OVER_COLD = RC.INC_MS[0] / (0.5 * (RC.COLD_MS[0] + RC.COLD_MS[1]))     # 7.4 / 6.5 (5.6), est.
# The measurement arms read at their own resolution (amendment 2026-10-01, pre-registered).
NATIVE_FORMAT = "inc2-native-rescore/1"
NATIVE_VERDICT_FORMAT = "inc2-native-verdict/1"
NATIVE_NAME = "native_v1"
NATIVE_RECORD = "native_rescore.json"
NATIVE_REFERENCE = "b_v2_m640"
NATIVE_ARMS = ("b_v2_m832", "b_v2_s1024")
NATIVE_EXAM = "dev"
NATIVE_TARGET_SPECIES = ("Carpetweed", "SpottedSpurge", "Purslane")
NATIVE_SEED_TEXT = "inc2/native/diff_se"
NATIVE_RESAMPLES = 1000
# what every native dev file of one comparison must share (imgsz aside)
NATIVE_STAMPS = ("manifest_sha256", "key_order_sha256", "n_images", "locked_scorer_sha256", "ultralytics_version",
                 "native_production")
NATIVE_SETTINGS = ("batch", "conf", "iou", "half", "rect", "max_det", "image_correct_conf", "image_correct_iou")
NATIVE_RULE = ("an arm qualifies for a stream fork proposal only when D = mean(arm's dev species_map50_95 at its own "
               "imgsz) - mean(b_v2_m640's at 640), on the seeds both share, exceeds 2 x pooled sd (sqrt((sd_arm^2 + "
               "sd_ref^2) / 2)) AND the paired image-bootstrap SE of D, AND one of Carpetweed, SpottedSpurge, Purslane "
               "has a higher mean dev AP50-95 at the arm's imgsz than b_v2_m640's at 640; qualifying files a card for "
               "a person (no switch, no test read)")


class BaselineError(RuntimeError):
    """A condition under which the baseline must not be built or judged."""


def log(msg):
    print("[inc2.baseline] %s" % msg, flush=True)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _sha(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def _mean_sd(xs):
    xs = [float(x) for x in xs]
    return (statistics.fmean(xs) if xs else None, statistics.stdev(xs) if len(xs) >= 2 else None)


# ---------------------------------------------------------------- inputs
def job_script_path():
    return C.REPO / "weed_llm_benchmark" / JOB_SCRIPT_NAME


def ensure_job_script(testing=False):
    """Point the driver at run_inc2_job.sh (INC_JOB_SCRIPT) before init; refuse
    a different script already set, and (production) a missing one."""
    want = job_script_path()
    cur = os.environ.get("INC_JOB_SCRIPT")
    if cur and Path(cur).resolve() != want.resolve():
        raise BaselineError("INC_JOB_SCRIPT is %s, not the v2 executor's job script %s: a v2 experiment is "
                            "never run by another executor" % (cur, want))
    if not testing and not want.is_file():
        raise BaselineError("the v2 job script %s is missing (deploy the checkout first)" % want)
    os.environ["INC_JOB_SCRIPT"] = str(want)
    return str(want)


def v2_lock_status(testing=False):
    """LOCK v2 and its never-train index: a production build needs both, the
    index the LOCK records, and a LOCK not written by a testing build; a
    testing build only warns."""
    lock_path, idx = T.v2_lock_path(), T.v2_nevertrain_path()
    lock = _read_json(lock_path)
    out = {"lock": str(lock_path), "lock_sha256": _sha(lock_path), "index": str(idx), "index_sha256": _sha(idx),
           "testing_lock": bool((lock or {}).get("testing")) if isinstance(lock, dict) else None,
           "manifests": dict((lock or {}).get("manifests") or {}) if isinstance(lock, dict) else {}}
    problems = []
    if not isinstance(lock, dict):
        problems.append("%s is missing or unreadable: v2 experiments are built only on locked splits "
                        "(inc2.splits lock)" % lock_path)
    else:
        if lock.get("splits_version") != RC.SPLITS_VERSION:
            problems.append("%s is not a v2 LOCK" % lock_path)
        if lock.get("nevertrain_sha256") != out["index_sha256"]:
            problems.append("the v2 never-train index %s is not the one LOCK v2 records" % idx)
        if lock.get("testing") and not testing:
            problems.append("LOCK v2 was written by a testing build")
    if problems:
        if not testing:
            raise BaselineError("; ".join(problems))
        for p in problems:
            log("WARNING: %s (testing build)" % p)
    out["problems"] = problems
    return out


def locked_name(sha, lock_status):
    for name, want in sorted((lock_status.get("manifests") or {}).items()):
        if want == sha:
            return name
    return None


def check_training_manifest(path, production=True, what="training manifest"):
    """(rows, {image: dHash}, info) of a manifest that passes inc2.train's
    check_manifest and guard_rows (fail closed)."""
    try:
        rows, dhashes, info = T.check_manifest(path)
    except T.RunError as e:
        raise BaselineError("%s %s refused: %s" % (what, path, e))
    try:
        info["guard"] = T.guard_rows(rows, dhashes, production=production)
    except T.RunError as e:
        raise BaselineError("%s %s refused by the splits v2 never-train guard: %s" % (what, path, e))
    return rows, dhashes, info


def check_union(paths, production=True):
    """(rows, info, parts) of the union of several manifests: each passes
    check_manifest, the parts are pairwise disjoint (key, image path, image
    sha256), and the guard runs over the union."""
    parts, all_rows, dh = [], [], {}
    for p in paths:
        try:
            rows, dhashes, info = T.check_manifest(p)
        except T.RunError as e:
            raise BaselineError("union part %s refused: %s" % (p, e))
        for q in parts:
            try:
                D.check_disjoint(q["rows"], rows, Path(p).stem, "part %s" % q["name"])
            except D.DriverError as e:
                raise BaselineError("the union's parts overlap: %s" % e)
        parts.append({"name": Path(p).stem, "path": str(Path(p).resolve()), "rows": rows,
                      "sha256": info["train_manifest_sha256"], "images": len(rows)})
        all_rows += rows
        dh.update(dhashes)
    try:
        guard = T.guard_rows(all_rows, dh, production=production)
    except T.RunError as e:
        raise BaselineError("the union refused by the splits v2 never-train guard: %s" % e)
    return all_rows, guard, parts


def research_only_record(rows):
    """The research-only taint of a base (docs/CONTINUOUS_LOOP.md §8): each
    row's research_only flag from splits v2's base_v2_provenance.jsonl
    (matched by key and image sha256), read only when it hashes as LOCK v2
    records it. flag is False only when every row is known and none is
    research-only; a research-only row or a row whose licence is not known
    here makes it True (fail closed: P6 treats an unresolved licence as
    research-only, and a model trained on any research-only row is
    research-only). basis says which. A caller that knows better, such as
    the stream's milestone, passes its own record through build's extra."""
    prov = T.v2_dir() / "base_v2_provenance.jsonl"
    lock = _read_json(T.v2_lock_path())
    want = (lock or {}).get("provenance_sha256") if isinstance(lock, dict) else None
    got = _sha(prov)
    known, problem = {}, None
    if got is None:
        problem = "no provenance file"
    elif want is not None and got != want:
        problem = "the provenance file does not hash as LOCK v2 records it"
    else:
        for r in C.read_manifest(prov):
            known[(r.get("key"), r.get("sha256"))] = bool(r.get("research_only"))
    flagged = sum(1 for r in rows if known.get((r["key"], r["sha256"])) is True)
    unknown = sum(1 for r in rows if (r["key"], r["sha256"]) not in known)
    flag = bool(flagged or unknown)
    basis = ("research_only rows" if flagged else "rows of unknown licence" if unknown
             else "every row known, none research-only")
    return {"flag": flag, "basis": basis, "rows": len(rows), "research_only_rows": flagged, "unknown_rows": unknown,
            "provenance": {"path": str(prov), "sha256": got, "lock_sha256": want, "problem": problem},
            "note": "a model trained on any research-only row, or on a row whose licence is unknown, is "
                    "research-only (8, P6)"}


L8_DECISION = "L-8, docs/CONTINUOUS_LOOP.md 2.6 (human-delegated, 2026-09-29)"
L8_RULE = ("the canary trains b0_v1's base (v1's train_core, by sha256) minus the rows whose image sha256 the L-8 "
           "list names (by the list's sha256, the one LOCK v2 records); every other line byte-identical")


def variant_drops_record(manifest_sha, production=True):
    """The canary's record of decision L-8 (module docstring): the listed
    rows and the derivation of its manifest from v1's train_core. Refuses
    (BaselineError) unless v1's train_core (by the v1 LOCK's sha256) minus
    the listed image sha256s hashes to manifest_sha; the list must hash as
    LOCK v2 records (a production LOCK must record one)."""
    lock_path = T.v2_lock_path()
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise BaselineError("%s is missing or unreadable: the L-8 drops cannot be read" % lock_path)
    try:
        rows, rec = C2.read_variant_drops(lock, lock_path, production=production)
    except C2.Inc2Error as e:
        raise BaselineError("the L-8 list: %s" % e)
    v1_path = C.manifest_path("train_core")
    try:
        v1_sha = C.read_lock()["manifests"]["train_core"]
        with open(v1_path, "rb") as fh:
            data = fh.read()
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise BaselineError("v1's train_core (b0_v1's base) cannot be read: %s" % e)
    if hashlib.sha256(data).hexdigest() != v1_sha:
        raise BaselineError("%s does not hash as the v1 LOCK records" % v1_path)
    reduced, dropped = C2.filter_manifest_bytes(data, {r["sha256"] for r in rows})
    got = hashlib.sha256(reduced).hexdigest()
    if got != manifest_sha or sorted(r["key"] for r in dropped) != sorted(r["key"] for r in rows):
        raise BaselineError("the canary's manifest (%s) is not v1's train_core minus the L-8 drops (%s, %d row(s)): "
                            "it would not reproduce b0_v1" % (str(manifest_sha)[:12], got[:12], len(rows)))
    return {"decided_by": L8_DECISION, "rule": L8_RULE, "file": rec["path"], "sha256": rec["sha256"],
            "lock_sha256": lock.get(C2.VARIANT_DROPS_LOCK_KEY), "count": len(rows),
            "rows": [{k: r.get(k) for k in ("key", "image", "sha256", "source", "session", "reason", "match")}
                     for r in sorted(rows, key=lambda r: r["key"])],
            "reference_manifest_sha256": v1_sha, "reference_manifest": str(v1_path), "manifest_sha256": manifest_sha,
            "note": ("b0_v1 trained on these rows; the effect is negligible and b0_v1 is not re-run (L-8)"
                     if rows else "nothing dropped: the canary's manifest is b0_v1's")}


def canary_manifest_match(defn, ref_defn):
    """(ok, record): whether the canary trained the reference's base
    manifest (same sha256), or the reference's minus the L-8 drops its
    exp.json records (module docstring, canary-verdict)."""
    cb, rb = defn.get("base") or {}, ref_defn.get("base") or {}
    cs, rs = cb.get("manifest_sha256"), rb.get("manifest_sha256")
    if cs is not None and cs == rs:
        return True, {"how": "identical"}
    vd = defn.get("variant_drops")
    if not isinstance(vd, dict) or not vd.get("rows"):
        return False, {"how": None, "why": "another manifest than the reference's, and no L-8 drops recorded"}
    probs = []
    if vd.get("reference_manifest_sha256") != rs:
        probs.append("the recorded drops were taken from another manifest than the reference's")
    if vd.get("manifest_sha256") != cs:
        probs.append("the recorded drops do not name the canary's manifest")
    lock = _read_json(T.v2_lock_path())
    if not isinstance(lock, dict) or lock.get(C2.VARIANT_DROPS_LOCK_KEY) != vd.get("sha256"):
        probs.append("LOCK v2 does not record the L-8 list the canary recorded")
    path = T.v2_dir() / C2.VARIANT_DROPS_NAME
    if _sha(path) != vd.get("sha256"):
        probs.append("%s does not hash as the canary recorded" % path)
    listed = {}
    if not probs:
        for r in C.read_manifest(path):
            listed[str(r["sha256"])] = r["key"]
        if sorted(listed.values()) != sorted(r.get("key") for r in vd["rows"]):
            probs.append("the L-8 list's rows are not the ones the canary recorded")
    data = None
    for cand in (rb.get("manifest"), rb.get("source_manifest"), str(C.manifest_path("train_core"))):
        if cand and _sha(cand) == rs:
            with open(cand, "rb") as fh:
                data = fh.read()
            break
    if data is None:
        probs.append("no copy of the reference's base manifest hashes as the reference records")
    elif not probs:
        reduced, dropped = C2.filter_manifest_bytes(data, set(listed))
        if hashlib.sha256(reduced).hexdigest() != cs or len(dropped) != len(listed):
            probs.append("the reference's base minus the listed rows is not the canary's manifest")
    rec = {"how": None if probs else "reference minus the recorded L-8 drops", "problems": probs,
           "dropped": [{k: r.get(k) for k in ("key", "sha256", "match")} for r in vd["rows"]],
           "variant_drops_sha256": vd.get("sha256"), "reference_manifest_sha256": rs, "manifest_sha256": cs}
    return not probs, rec


def _final_exams(final_exams, role, arm=None):
    measure = arm in RC.MEASURE_ARMS
    ex = list(final_exams) if final_exams else list(NO_TEST_FINAL_EXAMS if measure else ROLE_FINAL_EXAMS[role])
    bad = [e for e in ex if e not in T.EXAMS]
    if bad or D.DECISION_EXAM not in ex or len(set(ex)) != len(ex):
        raise BaselineError("final exams %s: each must be one of %s, dev included, none twice" % (ex, list(T.EXAMS)))
    if "test" in ex and role not in TEST_ROLES:
        raise BaselineError("a %s build may not read test: test is read only at milestone reads (P10), roles %s"
                            % (role, list(TEST_ROLES)))
    if "test" in ex and measure:
        raise BaselineError("measurement arm %s may not read test: it is built after R0, at no milestone read (P10)"
                            % arm)
    return ex


def arm_arg(arm=None, arch=None, imgsz=None):
    """The arm id from --arm, or from --arch and --imgsz together (the
    autopilot's L23B argv), which must agree when both are given."""
    if arch is None and imgsz is None:
        return RC.arm_id(arm if arm is not None else RC.DEFAULT_ARM)
    try:
        a = RC.arm_from_arch(arch, imgsz)
    except RC.RecipeError as e:
        raise BaselineError(str(e))
    if arm is not None and RC.arm_id(arm) != a:
        raise BaselineError("--arm %s and --arch %s --imgsz %s name different arms" % (arm, arch, imgsz))
    return a


def resolve_role(role, locked, arm, union=False):
    """The build's role: checked against what is built when given, inferred
    when not (module docstring). locked is the LOCK v2 name of the manifest's
    sha256 (None when LOCK v2 does not list it)."""
    if role is None:
        if union:
            return "union"
        if locked == "base_v2":
            return "b_v2" if arm == RC.REFERENCE_ARM else "capacity"
        if locked == "train_core":
            return "canary"
        return "baseline"
    if role not in ROLES:
        raise BaselineError("role %r not in %s" % (role, list(ROLES)))
    if (role == "union") != bool(union):
        raise BaselineError("role union is built from --union, and --union builds role union only (got role %s)"
                            % role)
    want = ROLE_MANIFEST.get(role)
    if want is not None and locked != want:
        raise BaselineError("role %s trains LOCK v2's %s; this manifest is %s" % (role, want,
                                                                                "LOCK v2's %s" % locked if locked
                                                                                else "not a LOCK v2 manifest"))
    if role in ("b_v2", "canary") and arm != RC.REFERENCE_ARM:
        raise BaselineError("role %s is the continuity arm %s, not %s" % (role, RC.REFERENCE_ARM, arm))
    if role == "capacity" and arm == RC.REFERENCE_ARM:
        raise BaselineError("role capacity is an arm other than %s (L-4); %s on base_v2 is b_v2"
                            % (RC.REFERENCE_ARM, RC.REFERENCE_ARM))
    return role


def e1_record(manifest_sha, role, arm):
    """E1 (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-03): when a 'baseline'
    build trains one of the two manifests a complete splits/v3/summary.json
    records (by sha256, the manifest still hashing so), the record exp.json
    carries (the arm A or B, the summary's sha256, both manifests' sha256s),
    and the build trains inc2.recipes.cold_budget; else None (the cold
    table). An E1 manifest on another arm than the pre-registered one
    refuses."""
    if role != "baseline":
        return None
    from . import base3 as B3
    k, summ = B3.e1_arm_of(manifest_sha)
    if k is None:
        return None
    if arm not in RC.BUDGET_ARMS:
        raise BaselineError("E1's arms train on %s only (pre-registered), not %s" % (list(RC.BUDGET_ARMS), arm))
    sp = B3.out_dir() / B3.SUMMARY
    return {"arm": k, "summary": str(sp), "summary_sha256": _sha(sp),
            "manifests": {a: (v or {}).get("sha256") for a, v in sorted((summ.get("arms") or {}).items())},
            "decided_by": "docs/CONTINUOUS_LOOP.md, Amendment (2026-10-03): E1, weed-box base v3 (pre-registered)"}


E2_GUARD_KEYS = ("lock_sha256", "index_sha256", "checked", "refused", "crosscheck_hits")


def e2_record(letter, exp, manifest_sha, n_images, locked, role, arm_rec, seeds, exams, lock, testing=False):
    """E2 (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-04): the e2 record
    exp.json carries for arm `letter` (W or S) of seed seeds[0], after every
    check that defines the experiment (module docstring, build --e2), none
    of which writes anything. The init is E1-B's base run of the same seed,
    by its resolved absolute path and its sha256; E1-B is the experiment the
    decided, qualifying capacity/e1_v1.json names. Refusals that only a
    production build must meet (a testing reference, verdict or E1-B run, a
    reference without recorded runs) are warnings in a testing build."""
    def soft(msg):
        if not testing:
            raise BaselineError(msg)
        log("WARNING: %s (testing build)" % msg)

    if letter not in RC.E2_ARMS:
        raise BaselineError("--e2 %r: E2's arms are W (b_v2_m640's cold recipe) and S (x1b), pre-registered "
                            "(docs/CONTINUOUS_LOOP.md, Amendment 2026-10-04)" % (letter,))
    if role != "baseline":
        raise BaselineError("an E2 experiment is built as role baseline (finals dev and imageweeds; its sealed test is "
                            "read once, after the verdict, by e2-test-read), not %s" % role)
    if arm_rec["id"] != RC.E2_ARM:
        raise BaselineError("E2 is pre-registered on %s only, not %s" % (RC.E2_ARM, arm_rec["id"]))
    seeds = list(seeds)
    if len(seeds) != 1:
        raise BaselineError("an E2 experiment trains one seed (got %s): the pinned driver gives every base run of an "
                            "experiment one init, and seed s starts from E1-B's seed s" % seeds)
    s = seeds[0]
    if s not in RC.E2_SEEDS:
        raise BaselineError("E2's seeds are %s (pre-registered), not %s" % (list(RC.E2_SEEDS), s))
    want = RC.e2_exp(letter, s)
    if exp != want:
        raise BaselineError("E2-%s seed %d is experiment %s (pre-registered), not %s" % (letter, s, want, exp))
    if locked != "base_v2":
        raise BaselineError("E2 trains LOCK v2's base_v2 (by sha256); this manifest is %s"
                            % ("LOCK v2's %s" % locked if locked else "not a LOCK v2 manifest"))
    if list(exams) != list(RC.E2_FINAL_EXAMS):
        raise BaselineError("E2's finals are %s (got %s)" % (" and ".join(RC.E2_FINAL_EXAMS), list(exams)))
    # the reference: b_v2_m640's definition, which E2 differs from in the init (and E2-S in the recipe) only
    rname = RC.E2_REFERENCE_EXP
    rpath = C.INC_DIR / rname / "exp.json"
    ref = _read_json(rpath)
    if not isinstance(ref, dict):
        raise BaselineError("the reference %s has no exp.json: E2 is defined against its definition" % rname)
    st = RC.stamp(arm_rec)
    rb, rs2 = ref.get("base") or {}, ref.get("splits_v2") or {}
    pairs = [("base.manifest_sha256", rb.get("manifest_sha256"), manifest_sha),
             ("base.n_images", rb.get("n_images"), int(n_images)),
             ("arm", ref.get("arm"), arm_rec),
             ("protocol", ref.get("protocol"), st["protocol"]),
             ("protocol_package", ref.get("protocol_package"), st["protocol_package"]),
             ("splits_version", ref.get("splits_version"), st["splits_version"]),
             ("splits_v2.lock_sha256", rs2.get("lock_sha256"), lock["lock_sha256"]),
             ("splits_v2.nevertrain_sha256", rs2.get("nevertrain_sha256"), lock["index_sha256"]),
             ("decision_exam", ref.get("decision_exam"), D.DECISION_EXAM),
             ("type", ref.get("type"), "baseline")]
    compared = [k for k, _a, _b in pairs] + ["seeds"]
    diff = [k for k, a, b in pairs if a != b]
    if s not in (ref.get("seeds") or []):
        diff.append("seeds (it has no seed %d)" % s)
    if diff:
        raise BaselineError("this build differs from %s in %s; E2 differs from it in the init (and E2-S in the recipe) "
                            "only" % (rname, ", ".join(diff)))
    if letter == "W":
        compared.append("base.recipe")
        if rb.get("recipe") != RC.cold(RC.E2_ARM) or RC.cold(RC.E2_ARM) != RC.e2_recipe("W"):
            raise BaselineError("%s's base recipe is not inc2.recipes.cold('%s') (the table changed since it was "
                                "built): E2-W cannot train it key for key" % (rname, RC.E2_ARM))
    if ref.get("testing"):
        soft("the reference %s is a testing experiment" % rname)
    # the environment its base runs trained in (Ultralytics, torch): an E2 run trains in the same one
    envs = {}
    for k in ref.get("seeds") or []:
        rj = _read_json(C.INC_DIR / rname / "runs" / ("base__s%d" % int(k)) / "run.json")
        envs[int(k)] = {x: rj.get(x) for x in T.E2_ENV_KEYS} if isinstance(rj, dict) and rj.get("status") == "done" \
            else None
    known = [e for e in envs.values() if e is not None and all(e.get(x) for x in T.E2_ENV_KEYS)]
    one = {json.dumps(e, sort_keys=True) for e in known}
    if len(known) != len(envs) or len(one) != 1:
        soft("%s's base runs do not share one recorded training environment (%s): E2 trains in the reference's"
             % (rname, envs))
    training_env = json.loads(one.pop()) if len(one) == 1 else None
    # E1's verdict: decided, and E1-B qualified
    vp = T.e1_verdict_path()
    v = _read_json(vp)
    if not isinstance(v, dict) or v.get("format") != RC.E2_E1_FORMAT or v.get("status") != "decided":
        raise BaselineError("%s is missing or is not a decided E1 verdict: E2 starts only from an E1-B that qualified"
                            % vp)
    if v.get("qualifies") is not True:
        raise BaselineError("E1's verdict does not qualify E1-B (D %s, 2 x pooled sd %s, SE %s): E2 is not built"
                            % (v.get("diff"), v.get("two_pooled_sd"), v.get("se_diff")))
    if v.get("testing_allowed"):
        soft("%s was decided with test-mode scores allowed" % vp)
    e1b = v.get("exp")
    bdefn = _read_json(C.INC_DIR / e1b / "exp.json") if isinstance(e1b, str) and T.NAME_RE.fullmatch(e1b) else None
    if not (isinstance(bdefn, dict) and bdefn.get("type") == "baseline"
            and (bdefn.get("e1") or {}).get("arm") == "B" and bdefn.get("recipe_name") == RC.BUDGET_NAME
            and (bdefn.get("arm") or {}).get("id") == RC.E2_ARM and s in (bdefn.get("seeds") or [])):
        raise BaselineError("%s (the verdict's E1-B) is not E1's arm B on %s with %s and seed %d"
                            % (e1b, RC.E2_ARM, RC.BUDGET_NAME, s))
    if bdefn.get("testing"):
        soft("%s (E1-B) is a testing experiment" % e1b)
    if (bdefn.get("e1") or {}).get("summary_sha256") != v.get("summary_sha256"):
        raise BaselineError("%s was built from splits v3 summary %s; E1's verdict was decided on %s"
                            % (e1b, str((bdefn.get("e1") or {}).get("summary_sha256"))[:12],
                               str(v.get("summary_sha256"))[:12]))
    # E1-B's base run of this seed: done, its own final.pt, hashing as its run.json records
    run_id = RC.E2_INIT_RUN % s
    rd = C.INC_DIR / e1b / "runs" / run_id
    rj = _read_json(rd / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise BaselineError("%s/runs/%s is not done: E2-%s seed %d has no init" % (e1b, run_id, letter, s))
    if rj.get("testing") is not False:
        soft("%s/runs/%s is not a production run" % (e1b, run_id))
    w = rd / "weights" / "final.pt"
    if w.is_symlink():
        raise BaselineError("%s is a symlink: the init must be the base run's own file" % w)
    if not w.is_file():
        raise BaselineError("%s is missing" % w)
    wsha = _sha(w)
    if wsha != rj.get("weights_sha256"):
        raise BaselineError("%s hashes to %s, its run.json records %s" % (w, str(wsha)[:12],
                                                                         str(rj.get("weights_sha256"))[:12]))
    # the leak guard E1-B's run trained under: the current LOCK v2, nothing refused (E2's dev verdict and test
    # read inherit that judgement, including the sources a person un-quarantined)
    g = rj.get("guard") if isinstance(rj.get("guard"), dict) else {}
    gbad = [k for k, want in (("lock_sha256", lock["lock_sha256"]), ("index_sha256", lock["index_sha256"]),
                              ("refused", 0), ("crosscheck_hits", 0)) if g.get(k) != want]
    if gbad:
        soft("%s/runs/%s's never-train guard record is not the current LOCK v2's with nothing refused (%s)"
             % (e1b, run_id, ", ".join(gbad)))
    ro = bdefn.get("research_only") if isinstance(bdefn.get("research_only"), dict) else {}
    decision = {k: v.get(k) for k in RC.E2_E1_DECISION_KEYS}
    return {"arm": letter, "seed": s, "recipe_name": RC.E2_ARMS[letter],
            "init": {"exp": e1b, "run_id": run_id, "path": str(w.resolve()), "sha256": wsha,
                     "run_json_sha256": _sha(rd / "run.json"), "exp_json_sha256": _sha(C.INC_DIR / e1b / "exp.json"),
                     "weights_epoch": rj.get("weights_epoch"),
                     "e1_summary_sha256": (bdefn.get("e1") or {}).get("summary_sha256"),
                     "guard": {k: g.get(k) for k in E2_GUARD_KEYS},
                     "research_only": {"flag": ro.get("flag", "unknown"), "basis": ro.get("basis")}},
            "e1_verdict": {"path": str(vp), "sha256": _sha(vp), "decision": decision},
            "reference": {"exp": rname, "exp_json_sha256": _sha(rpath), "role": ref.get("role"),
                          "seeds": list(ref.get("seeds") or []), "final_exams": list(ref.get("final_exams") or []),
                          "manifest_sha256": rb.get("manifest_sha256"), "n_images": rb.get("n_images"),
                          "recipe": rb.get("recipe"), "arm": ref.get("arm"), "training_env": training_env},
            "compared": compared,
            "differs_from_reference": ["init_weights"] + (["base.recipe"] if letter == "S" else []),
            "declared": {"role": [ref.get("role"), "baseline"],
                         "final_exams": [list(ref.get("final_exams") or []), list(RC.E2_FINAL_EXAMS)],
                         "seeds": [list(ref.get("seeds") or []), [s]]},
            "decided_by": RC.E2_DECIDED_BY}


def e2_research_only(base_rec, e2rec):
    """An E2 model's research-only flag (8, P6): base_v2's flag OR E1-B's, as
    the e2 record holds it. An E1-B flag that is missing or 'unknown' counts
    as true (fail closed)."""
    ro = (e2rec.get("init") or {}).get("research_only") or {}
    e1b_flag = ro.get("flag", "unknown")
    flag = bool(base_rec.get("flag")) or e1b_flag is not False
    basis = "; ".join(x for x in (
        "base_v2: %s" % base_rec.get("basis") if base_rec.get("flag") else None,
        "E1-B (%s): %s" % ((e2rec.get("init") or {}).get("exp"), ro.get("basis") or "flag %r" % (e1b_flag,))
        if e1b_flag is not False else None) if x) or "base_v2 and E1-B both known clean"
    return dict(base_rec, flag=flag, basis=basis,
                init_research_only={"exp": (e2rec.get("init") or {}).get("exp"), "flag": e1b_flag,
                                    "basis": ro.get("basis")},
                note="a model trained from E1-B's weights inherits E1-B's research-only status (8, P6)")


def _e2_view(exp, letter, arm_rec, seeds, exams, sha, e2rec):
    """The keys of an E2 definition inc2.train's e2_problems reads, as the
    build will write them (the check runs before the manifest is copied)."""
    return {"exp": exp, "type": "baseline", "arm": arm_rec, "seeds": list(seeds), "final_exams": list(exams),
            "recipe_name": RC.E2_ARMS[letter] if letter == "S" else None,
            "base": {"recipe": RC.e2_recipe(letter), "manifest_sha256": sha},
            "init_weights": e2rec["init"]["path"], "e2": e2rec}


def e1_claim(manifest, manifest_sha):
    """inc2.base3.v3_claim: why a manifest belongs to a base v3 build (it lies
    under splits/v3, or a base v3 summary records its sha256, whatever that
    summary's status), or None. Such a manifest trains only as an E1 arm."""
    from . import base3 as B3
    return B3.v3_claim(manifest, manifest_sha)


# ------------------------------------------------------------------- build
def build_definition(exp, manifest=None, union=None, seeds=None, arm=None, final_exams=None,
                     role=None, testing=False, extra=None, arch=None, imgsz=None, e2=None):
    """Check every input and write the manifest copy and the summary; returns
    (definition, summary). Nothing is written before every check passed.
    role None infers it (module docstring). e2 'W' or 'S' builds that arm of
    E2 (module docstring, build --e2)."""
    if role is not None and role not in ROLES:
        raise BaselineError("role %r not in %s" % (role, list(ROLES)))
    if (manifest is None) == (not union):
        raise BaselineError("give exactly one of --manifest and --union")
    if e2 is not None:
        if e2 not in RC.E2_ARMS:
            raise BaselineError("--e2 %r: E2's arms are W (b_v2_m640's cold recipe) and S (x1b), pre-registered "
                                "(docs/CONTINUOUS_LOOP.md, Amendment 2026-10-04)" % (e2,))
        if union:
            raise BaselineError("E2 trains LOCK v2's base_v2 (--manifest), never a union")
    try:
        testing = P._check_testing(testing)
    except P.PilotError as e:
        raise BaselineError(str(e))
    aid = arm_arg(arm, arch, imgsz)
    try:
        arm_rec = RC.resolve_arm(aid, repo=C.REPO, require_weights=not testing)
    except RC.RecipeError as e:
        raise BaselineError(str(e))
    paths = D.Paths(exp)
    try:
        P._check_new(paths)
    except P.PilotError as e:
        raise BaselineError(str(e))
    lock = v2_lock_status(testing=bool(testing))
    production = not testing
    locked = None if union else locked_name(_sha(Path(os.path.abspath(str(manifest)))), lock)
    role = resolve_role(role, locked, arm_rec["id"], union=bool(union))
    try:
        seeds = P.check_seeds(list(ROLE_SEEDS.get(role, DEFAULT_SEEDS)) if seeds is None else seeds)
    except P.PilotError as e:
        raise BaselineError(str(e))
    exams = _final_exams(final_exams, role, arm_rec["id"])
    drops = e1 = e2rec = None
    if union:
        srcs = [Path(os.path.abspath(str(p))) for p in union]
        for p in srcs:
            claim = e1_claim(p, _sha(p))
            if claim:
                raise BaselineError("%s is a base v3 manifest (%s): it trains only as an E1 arm, never in a union"
                                    % (p, claim))
        rows, guard, parts = check_union(srcs, production=production)
        name = sanitise("%s_union" % exp)
        dst = paths.manifests / ("%s.jsonl" % name)
        sha = C.write_manifest(dst, rows)
        rows2, _dh, info = T.check_manifest(dst)
        info["guard"] = guard
        source = {"union": [{k: v for k, v in q.items() if k != "rows"} for q in parts],
                  "source_locked": [locked_name(q["sha256"], lock) for q in parts]}
        rows = rows2
    else:
        src = Path(os.path.abspath(str(manifest)))
        rows, _dh, info = check_training_manifest(src, production=production)
        sha = info["train_manifest_sha256"]
        if locked_name(sha, lock) != locked:
            raise BaselineError("%s changed while it was checked (the role was decided on other bytes)" % src)
        drops = variant_drops_record(sha, production=production) if role == "canary" else None
        e1 = e1_record(sha, role, arm_rec["id"])
        claim = e1_claim(src, sha)
        if claim and e1 is None:
            raise BaselineError("%s is a base v3 manifest (%s): it trains only as an E1 arm (role baseline on one of "
                                "the two manifests a complete splits/v3/summary.json records, recipe cold_budget); "
                                "this build (role %s) is not one, and would train the cold table" % (src, claim, role))
        if e2 is not None:
            if e1 is not None:
                raise BaselineError("%s is an E1 manifest: E2 trains LOCK v2's base_v2" % src)
            e2rec = e2_record(e2, exp, sha, len(rows), locked, role, arm_rec, seeds, exams, lock, testing=bool(testing))
            # inc2.train's own check on the definition as it will be written, before the manifest is copied
            probs = T.e2_problems(json.loads(json.dumps(_e2_view(exp, e2, arm_rec, seeds, exams, sha, e2rec))))
            if probs:
                raise BaselineError("the E2 definition fails inc2.train's own check: %s" % "; ".join(probs))
        name = sanitise(src.stem)
        dst = paths.manifests / ("%s.jsonl" % name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if C.sha256_file(dst) != sha:
            raise BaselineError("copy of %s does not hash like the manifest that was checked" % src)
        source = {"source_manifest": str(src), "source_locked": locked_name(sha, lock)}
    if e1 is not None:
        recipe = RC.cold_budget(arm_rec["id"], len(rows))
        cost = RC.budget_cost(len(rows), seeds, exams, arm_rec["id"])
    elif e2rec is not None:
        recipe = RC.e2_recipe(e2, arm_rec["id"])
        cost = RC.e2_cost(len(rows), e2, seeds, exams, arm_rec["id"])
    else:
        recipe = RC.cold(arm_rec["id"])
        cost = RC.baseline_cost(len(rows), seeds, exams, arm_rec["id"])
    defn = {"exp": exp, "type": "baseline", "builder": BUILDER, "role": role, "testing": testing,
            "seeds": list(seeds), "decision_exam": D.DECISION_EXAM, "final_exams": exams,
            "base": dict(P._entry(name, dst, rows, sha, recipe=recipe), **source),
            "cost_estimate": cost, "research_only": research_only_record(rows),
            "splits_v2": {"lock": lock["lock"], "lock_sha256": lock["lock_sha256"],
                          "nevertrain_sha256": lock["index_sha256"]}}
    if drops is not None:
        defn["variant_drops"] = drops
    if e1 is not None:
        defn.update(recipe_name=RC.BUDGET_NAME, budget=RC.budget_record(arm_rec["id"], len(rows)), e1=e1)
    defn.update(RC.stamp(arm_rec))
    if e2rec is not None:
        # E2: seed s starts from E1-B's seed-s weights (the pinned driver passes init_weights into the base run's
        # spec unchanged); E2-S names x1b for its base runs; the model inherits E1-B's research-only status
        defn["init_weights"] = e2rec["init"]["path"]
        defn["e2"] = e2rec
        if e2 == "S":
            defn["recipe_name"] = RC.E2_ARMS["S"]
        defn["research_only"] = e2_research_only(defn["research_only"], e2rec)
    if extra:
        defn.update(extra)
    if e2rec is not None:
        probs = T.e2_problems(json.loads(json.dumps(defn)))
        if probs:
            shutil.rmtree(str(paths.manifests), ignore_errors=True)
            raise BaselineError("the E2 definition fails inc2.train's own check: %s" % "; ".join(probs))
    try:
        D.validate_definition(json.loads(json.dumps(defn)))
        D.check_definition_data(defn)
    except D.DriverError as e:
        raise BaselineError("the pinned driver refuses the definition: %s" % e)
    code = {m: _sha(T.package_dir() / m) for m in ("tools/inc2/baseline.py", "tools/inc2/recipes.py",
                                                  "tools/inc2/train.py", "tools/inc2/guard.py",
                                                  "tools/inc/driver.py", "tools/inc/pilot.py")}
    summary = {"exp": exp, "testing": bool(testing), "built_utc": D._utc(), "builder": BUILDER, "role": role,
               "seeds": list(seeds), "arm": arm_rec, "final_exams": exams,
               "manifest": dict(P.manifest_summary(rows, sha, info), name=name, copy=str(dst), **source),
               "splits_v2": lock, "cold_recipe": recipe,
               "warmup": {"base": P.effective_warmup(recipe, len(rows))},
               "cost_estimate": cost, "research_only": defn["research_only"], "code": code}
    if drops is not None:
        summary["variant_drops"] = drops
    if e1 is not None:
        summary.update(recipe_name=RC.BUDGET_NAME, budget=defn["budget"], e1=e1)
    if e2rec is not None:
        summary["e2"] = e2rec
        if defn.get("recipe_name"):
            summary["recipe_name"] = defn["recipe_name"]
    D._write_json(paths.root / BUILD_SUMMARY, summary)
    return defn, summary


def build(exp, manifest=None, union=None, seeds=None, arm=None, final_exams=None, role=None,
          testing=False, backend=None, init=True, quiet=False, extra=None, arch=None, imgsz=None, e2=None):
    """build_definition, then driver init through run_inc2_job.sh. Returns
    (summary, definition, init result or None)."""
    defn, summary = build_definition(exp, manifest=manifest, union=union, seeds=seeds, arm=arm,
                                     final_exams=final_exams, role=role, testing=testing, extra=extra,
                                     arch=arch, imgsz=imgsz, e2=e2)
    log("%s (%s): baseline on %s, %d images, arm %s, seeds %s, finals %s; est. %s-%s GPU-h"
        % (exp, role, defn["base"]["manifest"], defn["base"]["n_images"], defn["arm"]["id"], defn["seeds"],
           defn["final_exams"], defn["cost_estimate"]["total_gpu_h"][0], defn["cost_estimate"]["total_gpu_h"][1]))
    if defn.get("e2"):
        log("E2-%s seed %d from %s (%s)" % (defn["e2"]["arm"], defn["e2"]["seed"], defn["e2"]["init"]["path"],
                                          defn["e2"]["init"]["sha256"][:12]))
    result = None
    if init:
        ensure_job_script(testing=bool(defn["testing"]))
        result = D.Driver(exp, backend=backend, quiet=quiet).init(defn)
    return summary, defn, result


# ------------------------------------------------------------------ verdicts
def base_dev_scores(exp, seeds=None, testing_ok=False):
    """{seed: (map50_95, score record)} of an experiment's base runs on dev
    (runs/base__s<k>/scores/dev.json), production scores unless testing_ok."""
    root = C.INC_DIR / exp
    defn = _read_json(root / "exp.json")
    if not isinstance(defn, dict):
        raise BaselineError("%s has no exp.json" % root)
    out = {}
    for s in (defn["seeds"] if seeds is None else seeds):
        rid = "base__s%d" % s
        p = root / "runs" / rid / "scores" / "dev.json"
        rj = _read_json(root / "runs" / rid / "run.json")
        d = _read_json(p)
        if not isinstance(d, dict) or not isinstance(rj, dict) or rj.get("status") != "done":
            raise BaselineError("%s's %s has no finished dev score (%s)" % (exp, rid, p))
        if d.get("exam") != "dev" or (d.get("production") is not True and not testing_ok):
            raise BaselineError("%s is not a production dev score" % p)
        out[s] = (float(d["map50_95"]), {"run_id": rid, "path": str(p), "sha256": _sha(p),
                                         "stamps": {k: d.get(k) for k in ("scorer_sha256", "manifest_sha256",
                                                                          "key_order_sha256", "n_images")},
                                         "production": d.get("production") is True, "run_json": rj})
    return defn, out


def canary_verdict(exp="canary_v2", reference="b0_v1", write=True, allow_testing=False):
    """The canary against b0_v1's seeds (module docstring). allow_testing
    lets a test-mode canary pass (tests only; the CLI never sets it)."""
    defn, scores = base_dev_scores(exp, seeds=[0], testing_ok=True)
    value, rec = scores[0]
    ref_path = C.INC_DIR / reference / "exp.json"
    ref_defn = _read_json(ref_path)
    same, match = None, None
    if isinstance(ref_defn, dict) and isinstance(ref_defn.get("base"), dict):
        cb, rb = defn.get("base") or {}, ref_defn["base"]
        manifest_ok, match = canary_manifest_match(defn, ref_defn)
        same = {"manifest_sha256": manifest_ok,
                "cold_recipe": cb.get("recipe") == rb.get("recipe") and cb.get("recipe") is not None,
                "seed_0": 0 in (ref_defn.get("seeds") or []) and 0 in (defn.get("seeds") or [])}
    same_run = bool(same) and all(same.values())
    rep_path = C.INC_DIR / reference / "report.json"
    rep = _read_json(rep_path)
    ref = None
    if isinstance(rep, dict):
        for f in rep.get("final") or []:
            if str(f.get("model", "")).startswith("base"):
                tw = ((f.get("exams") or {}).get("dev") or {}).get("twelve") or {}
                if tw.get("mean") is not None and tw.get("sd") is not None:
                    ref = {"mean": tw["mean"], "sd": tw["sd"], "n": tw.get("n"), "source": str(rep_path),
                           "sha256": _sha(rep_path)}
                break
    if ref is None:
        _d, rs = base_dev_scores(reference)
        m, sd = _mean_sd([v for v, _ in rs.values()])
        ref = {"mean": m, "sd": sd, "n": len(rs), "source": "runs of %s" % reference,
               "inputs": [r for _v, r in rs.values()]}
    within = abs(value - ref["mean"]) <= ref["sd"]
    sc = (rec["run_json"].get("sidecars") or {}).get("dev")
    sidecar_ok = bool(isinstance(sc, dict) and sc.get("status") != "failed" and sc.get("sha256")
                      and _sha(sc.get("path") or "") == sc.get("sha256"))
    production = rec["run_json"].get("testing") is False and rec["production"]
    passed = within and sidecar_ok and same_run and (production or allow_testing)
    doc = {"format": CANARY_FORMAT, "exp": exp, "reference": reference, "canary_dev": value,
           "reference_dev": ref, "abs_diff": abs(value - ref["mean"]), "within_one_sd": within,
           "sidecar_ok": sidecar_ok, "sidecar": sc, "passed": passed, "production": production,
           "testing_allowed": bool(allow_testing), "same_run_as_reference": same, "manifest_match": match,
           "rule": "|canary base__s0 dev mAP50-95 - mean(reference seeds)| <= sd(reference seeds) (4.4), the "
                   "canary's scorer sidecar was written (the v2 executor's whole path, the sidecar included), a "
                   "production run and score, and the reference's base manifest (or it minus the L-8 drops the "
                   "canary recorded, by the reference's sha256 and the list's sha256), cold recipe and seed 0",
           "inputs": {"canary_score": {k: rec[k] for k in ("run_id", "path", "sha256")},
                      "exp_json_sha256": _sha(C.INC_DIR / exp / "exp.json"),
                      "reference_exp_json": {"path": str(ref_path), "sha256": _sha(ref_path)}},
           "generated_utc": D._utc()}
    if write:
        doc["out"] = str(C.INC_DIR / exp / "canary.json")
        _write_json(doc["out"], doc)
    log("canary %s: dev %.4f vs %s %.4f +- %.4f: %s" % (exp, value, reference, ref["mean"], ref["sd"],
                                                         "PASS" if passed else "FAIL"))
    return doc


def capacity_decision(n_exp, arm_exps, m=None, testing_ok=False, record_exps=()):
    """L-4's rule on dev only (module docstring). Opens exp.json, the base
    runs' run.json and scores/dev.json, nothing else. record_exps
    (measurement arms) are read on the same seeds, manifest and scorer and
    listed record only: they never enter qualifying or chosen."""
    record_exps = list(record_exps or ())
    exps = [n_exp] + list(arm_exps)
    loaded, arms = {}, {}
    for e in exps + record_exps:
        defn = _read_json(C.INC_DIR / e / "exp.json")
        if not isinstance(defn, dict) or defn.get("type") != "baseline":
            raise BaselineError("%s is not a baseline experiment" % e)
        if not isinstance(defn.get("arm"), dict):
            raise BaselineError("%s pins no arm (build it with inc2.baseline)" % e)
        aid = RC.arm_id(defn["arm"])
        if aid in arms:
            raise BaselineError("arm %s appears twice (%s, %s)" % (aid, arms[aid], e))
        if e in record_exps and aid not in RC.MEASURE_ARMS:
            raise BaselineError("%s is arm %s, one of L-4's grid %s: it is a candidate (--arms), not a record"
                                % (e, aid, list(RC.GRID_ARMS)))
        if e not in record_exps and aid not in RC.GRID_ARMS:
            raise BaselineError("%s is measurement arm %s (trained at %d px, scored at 640 px): never a candidate "
                                "of L-4's decision; list it with --record" % (e, aid, RC.ARMS[aid]["imgsz"]))
        arms[aid] = e
        loaded[e] = defn
    if RC.arm_id(loaded[n_exp]["arm"]) != RC.REFERENCE_ARM:
        raise BaselineError("%s is arm %s; the rule compares every arm with %s" % (n_exp, loaded[n_exp]["arm"]["id"],
                                                                                 RC.REFERENCE_ARM))
    common = sorted(set.intersection(*[set(loaded[e]["seeds"]) for e in exps]))
    if len(common) < 2:
        raise BaselineError("the arms share seeds %s; the rule needs at least 2" % common)
    for e in record_exps:
        if not set(common) <= set(loaded[e]["seeds"]):
            raise BaselineError("record arm %s lacks seeds %s of the grid's %s"
                                % (e, sorted(set(common) - set(loaded[e]["seeds"])), common))
    manifests = {loaded[e]["base"]["manifest_sha256"] for e in exps + record_exps}
    if len(manifests) != 1:
        raise BaselineError("the arms were trained on different manifests (%d distinct)" % len(manifests))
    per, stamps = {}, None
    for e in exps + record_exps:
        _d, sc = base_dev_scores(e, seeds=common, testing_ok=testing_ok)
        for s, (_v, r) in sc.items():
            if stamps is None:
                stamps = r["stamps"]
            elif r["stamps"] != stamps:
                raise BaselineError("%s %s was scored by another scorer, dev manifest or key order" % (e, r["run_id"]))
        vals = [sc[s][0] for s in common]
        mean, sd = _mean_sd(vals)
        per[e] = {"arm": loaded[e]["arm"]["id"], "seeds": common, "dev": vals, "mean": mean, "sd": sd,
                  "inputs": [{k: sc[s][1][k] for k in ("run_id", "path", "sha256")} for s in common],
                  "run_json": [sc[s][1]["run_json"] for s in common]}
    n = per[n_exp]
    for e in arm_exps:
        a = per[e]
        pooled = math.sqrt((a["sd"] ** 2 + n["sd"] ** 2) / 2.0)
        a.update(diff_vs_n=a["mean"] - n["mean"], pooled_sd=pooled,
                 qualifies=(a["mean"] - n["mean"]) > 2.0 * pooled)
    qual = [e for e in arm_exps if per[e]["qualifies"]]
    chosen = max(qual, key=lambda e: (per[e]["mean"], -RC.ARMS[per[e]["arm"]]["gflops"])) if qual else n_exp
    arm = per[chosen]["arm"]
    cold_ms, n_runs = RC.measured_rates(per[chosen]["run_json"])
    n_images = int(loaded[chosen]["base"]["n_images"])
    mm = int(m) if m is not None else int(math.ceil(M_SHARE * n_images))
    if cold_ms is not None:
        inc_ms = cold_ms * INC_OVER_COLD
        cost = RC.step_cost(n_images, mm, arm, "r0", truth=True, cold_ms=cold_ms, inc_ms=inc_ms)
        basis = ("measured cold rate %.2f ms per image-epoch (median of %d base run(s) of %s); incremental est. as "
                 "cold x %.4f (7.4 / 6.5, n640, 5.6)" % (cold_ms, n_runs, chosen, INC_OVER_COLD))
        step_h = cost["gpu_h"][1]
    else:
        cost = RC.step_cost(n_images, mm, arm, "r0", truth=True)
        basis = "est. rates (no measured run): the high end of inc2.recipes.rates(%s)" % arm
        step_h = cost["gpu_h"][1]
    record = {}
    for e in record_exps:
        a = per.pop(e)
        pooled = math.sqrt((a["sd"] ** 2 + n["sd"] ** 2) / 2.0)
        a.pop("run_json", None)
        record[e] = dict(a, diff_vs_n=a["mean"] - n["mean"], pooled_sd=pooled,
                         above_2_pooled_sd=(a["mean"] - n["mean"]) > 2.0 * pooled, record_only=True,
                         trained_imgsz=RC.ARMS[a["arm"]]["imgsz"], scored_imgsz=640)
    for e in exps:
        per[e].pop("run_json", None)
    out = {"format": CAPACITY_FORMAT, "rule": "an arm qualifies when mean(arm) - mean(n640) > 2 x pooled sd on "
                                              "dev mAP50-95 (seeds all arms share); the best qualifying arm by "
                                              "dev mean is chosen, else n640 (L-4)",
           "n_exp": n_exp, "arm_exps": list(arm_exps), "seeds": common, "stamps": stamps, "arms": per,
           "qualifying": qual, "chosen_exp": chosen, "chosen_arm": arm,
           "chosen_arm_record": loaded[chosen]["arm"],
           "step_cost": dict(cost, basis=basis), "m": mm, "n_images": n_images,
           "truth_every": RC.truth_every(step_h), "truth_step_cap_gpu_h": RC.TRUTH_STEP_CAP_GPU_H,
           "note": "decided on dev only; test is in the separate report, for people"}
    if record_exps:
        out.update(record_exps=list(record_exps), record=record,
                   record_note="measurement arms, record only: never in qualifying or chosen; trained at their "
                               "imgsz and scored by the locked scorer at 640 px")
    return out


def capacity_report(decision, testing_ok=False):
    """Every arm's final scores on its final exams (dev, imageweeds, test),
    mean +- sd over seeds, 12-class and agnostic, and the gap to 0.90 on test.
    The decision's record arms (measurement arms) are listed too, marked
    record_only."""
    rows = {}
    record = list(decision.get("record_exps") or [])
    for e in [decision["n_exp"]] + list(decision["arm_exps"]) + record:
        defn = _read_json(C.INC_DIR / e / "exp.json") or {}
        out = {"arm": (defn.get("arm") or {}).get("id"), "seeds": defn.get("seeds"), "exams": {}, "inputs": []}
        if e in record:
            out["record_only"] = True
        for exam in defn.get("final_exams") or []:
            tw, ag = [], []
            for s in defn.get("seeds") or []:
                p = C.INC_DIR / e / "runs" / ("final__base__s%d" % s) / "scores" / ("%s.json" % exam)
                d = _read_json(p)
                if isinstance(d, dict) and (d.get("production") is True or testing_ok):
                    tw.append(float(d["map50_95"]))
                    ag.append(float(d["agnostic_map50_95"]))
                    out["inputs"].append({"path": str(p), "sha256": _sha(p)})
            m1, s1 = _mean_sd(tw)
            m2, s2 = _mean_sd(ag)
            out["exams"][exam] = {"n": len(tw), "twelve": {"mean": m1, "sd": s1}, "agnostic": {"mean": m2, "sd": s2},
                                  "complete": len(tw) == len(defn.get("seeds") or [])}
        t = out["exams"].get("test", {}).get("twelve", {}).get("mean")
        out["gap_to_target"] = (TARGET_TEST - t) if t is not None else None
        rows[e] = out
    return {"format": CAPACITY_FORMAT + "-report", "target_test_map50_95": TARGET_TEST, "arms": rows,
            "chosen_exp": decision["chosen_exp"], "chosen_arm": decision["chosen_arm"],
            "note": "the test column is the milestone read of R0 (L-4), for people; it is never an input of the "
                    "capacity decision"}


def _report_md(rep):
    lines = ["# Capacity grid at R0 (L-4)", "", "Chosen arm: **%s** (%s). Target test mAP50-95 %.2f." %
             (rep["chosen_arm"], rep["chosen_exp"], rep["target_test_map50_95"]), "",
             "| Experiment | Arm | Exam | 12-class mean +- sd | agnostic mean +- sd | n |", "|---|---|---|---|---|---|"]

    def f(x):
        return "-" if x is None else "%.4f" % x
    for e, r in rep["arms"].items():
        arm = "%s (record only, scored at 640 px)" % r["arm"] if r.get("record_only") else r["arm"]
        for exam, v in r["exams"].items():
            lines.append("| %s | %s | %s | %s +- %s | %s +- %s | %d |" % (e, arm, exam, f(v["twelve"]["mean"]),
                                                                         f(v["twelve"]["sd"]), f(v["agnostic"]["mean"]),
                                                                         f(v["agnostic"]["sd"]), v["n"]))
        lines.append("| %s | %s | gap to %.2f on test | %s | | |" % (e, arm, rep["target_test_map50_95"],
                                                                     f(r["gap_to_target"])))
    return "\n".join(lines) + "\n"


# What a record-only run must find unchanged in an existing decision file.
DECISION_KEYS = ("n_exp", "arm_exps", "seeds", "qualifying", "chosen_exp", "chosen_arm")


def capacity_verdict(n_exp="b_v2", arm_exps=("b_v2_s640", "b_v2_m640"), m=None, out_dir=None, write=True,
                     testing_ok=False, record_exps=()):
    """The decision and its report (module docstring). With record_exps and
    an existing decision file, that file is kept byte for byte and only the
    report is written; a recomputed decision that differs from it refuses."""
    decision = capacity_decision(n_exp, arm_exps, m=m, testing_ok=testing_ok, record_exps=record_exps)
    decision["generated_utc"] = D._utc()
    report = capacity_report(decision, testing_ok=testing_ok)
    report["generated_utc"] = decision["generated_utc"]
    if write:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        decision["out"] = str(d / ("%s.json" % CAPACITY_NAME))
        if record_exps and Path(decision["out"]).exists():
            old = _read_json(decision["out"])
            diff = [k for k in DECISION_KEYS if not isinstance(old, dict) or old.get(k) != decision.get(k)]
            if diff:
                raise BaselineError("%s holds another decision (%s differ); a record-only run never replaces it"
                                    % (decision["out"], ", ".join(diff)))
            decision["decision_kept"] = True
        else:
            _write_json(decision["out"], decision)
        report["decision_sha256"] = _sha(decision["out"])
        report["out"] = str(d / ("%s_report.json" % CAPACITY_NAME))
        _write_json(report["out"], report)
        md = d / ("%s_report.md" % CAPACITY_NAME)
        tmp = md.with_name(".%s.tmp" % md.name)
        tmp.write_text(_report_md(report))
        os.replace(tmp, md)
    log("capacity: chosen arm %s (%s); qualifying %s; truth every %d step(s)%s"
        % (decision["chosen_arm"], decision["chosen_exp"], decision["qualifying"], decision["truth_every"],
           "; record only: %s" % ", ".join("%s dev %.4f" % (e, r["mean"]) for e, r in decision["record"].items())
           if record_exps else ""))
    return decision, report


# ------------------------------------------------------------------ secondary
def secondary(exp, weights, run_id="secondary__incumbent", source=None):
    """Write the milestone's incumbent-scoring spec (module docstring);
    returns {spec, list, argv, record}."""
    root = C.INC_DIR / exp
    defn = _read_json(root / "exp.json")
    if not isinstance(defn, dict) or defn.get("type") != "baseline":
        raise BaselineError("%s is not a built baseline experiment" % exp)
    D.check_name(run_id, "run_id")
    if run_id in {"base__s%d" % s for s in defn["seeds"]} | {"final__base__s%d" % s for s in defn["seeds"]}:
        raise BaselineError("run id %s is one of the driver's own" % run_id)
    w = Path(weights).resolve()
    if not w.is_file():
        raise BaselineError("no weights at %s" % w)
    out = root / "runs" / run_id
    spec_path = out / "spec.json"
    spec = {"exp": exp, "run_id": run_id, "kind": "final", "init": str(w), "exams": list(defn["final_exams"]),
            "out_dir": str(out)}
    if out.exists() and sorted(os.listdir(out)) not in ([], ["spec.json"]):
        raise BaselineError("%s already holds a run" % out)
    if spec_path.exists() and _read_json(spec_path) != spec:
        raise BaselineError("%s holds another spec" % spec_path)
    D._write_json(spec_path, spec)
    try:
        T.validate_spec(spec, spec_path)
    except T.RunError as e:
        spec_path.unlink()
        raise BaselineError("the v2 executor refuses the spec: %s" % e)
    lst = root / "submissions" / ("%s.txt" % run_id)
    lst.parent.mkdir(parents=True, exist_ok=True)
    tmp = lst.with_name(".%s.%d.tmp" % (lst.name, os.getpid()))
    tmp.write_text("%s\n" % spec_path)
    os.replace(tmp, lst)
    (root / "logs").mkdir(parents=True, exist_ok=True)      # Slurm opens --output before the job starts
    script = job_script_path()
    argv = ["sbatch", "--parsable", "--array=0-0", "--job-name=inc_%s_%s" % (exp, run_id),
            "--output=%s" % (root / "logs" / "%x_%A_%a.out"), str(script), str(lst), exp]
    rec = {"format": SECONDARY_FORMAT, "exp": exp, "run_id": run_id, "weights": str(w), "weights_sha256": _sha(w),
           "source": source, "exams": list(defn["final_exams"]), "spec": str(spec_path),
           "spec_sha256": _sha(spec_path), "list": str(lst), "argv": argv, "job_script": str(script),
           "written_utc": D._utc(),
           "note": "the chain incumbent's secondary number (3.6); the driver does not track this run"}
    _write_json(root / ("%s.json" % run_id), rec)
    return {"spec": str(spec_path), "list": str(lst), "argv": argv, "record": rec}


# ------------------------------------------------------------ native resolution
def _final_id(seed):
    return "final__base__s%d" % int(seed)


def _native_arm(exp, measure=True):
    """(exp.json, arm, imgsz) of a measurement arm (measure) or of the
    reference arm (not measure); BaselineError otherwise."""
    try:
        defn, aid, imgsz = SN.load_arm(exp)
    except SN.NativeRefused as e:
        raise BaselineError(str(e))
    if measure and aid not in RC.MEASURE_ARMS:
        raise BaselineError("%s is arm %s: only a measurement arm %s is read at its own imgsz"
                            % (exp, aid, list(RC.MEASURE_ARMS)))
    if not measure and aid != SN.REFERENCE_ARM:
        raise BaselineError("%s is arm %s, not the reference arm %s" % (exp, aid, SN.REFERENCE_ARM))
    return defn, aid, imgsz


def _native_one(e, s, exam, size, batch=None, device=None):
    """One final run's native score (seed s, exam, imgsz), taken only when
    missing (a native score is written once); its record: run id, file name,
    sha256, imgsz, kept or written, native_production."""
    scores = C.INC_DIR / e / "runs" / _final_id(s) / "scores"
    js, _npz = SN.paths_for(scores, exam, size)
    status = "kept"
    if not js.is_file():
        try:
            SN.score_run(e, _final_id(s), exam, batch=batch, device=device)
        except SN.NativeRefused as x:
            raise BaselineError("%s %s on %s at %d px refused: %s" % (e, _final_id(s), exam, size, x))
        status = "written"
    d = _read_json(js) or {}
    return {"run_id": _final_id(s), "score": js.name, "sha256": _sha(js), "imgsz": size, "status": status,
            "native_production": d.get("native_production")}


def rescore_native(exp, reference=NATIVE_REFERENCE, out_dir=None, verdict=True, batch=None, device=None,
                   resamples=NATIVE_RESAMPLES):
    """Score a measurement arm's final runs at its imgsz (dev, imageweeds)
    and the reference's on dev at 640, each only when missing, then record
    the native verdict (module docstring). Returns the rescore record."""
    defn, aid, imgsz = _native_arm(exp)
    rdefn, _raid, rimgsz = _native_arm(reference, measure=False)
    # an arm that trains at 640 (the box-quality arms) is read at 640 on dev only: its other 640 px scores are
    # the locked scorer's own (inc2.scorer_native reads nothing else at 640)
    exams = [e for e in SN.EXAMS if e in (defn.get("final_exams") or [])
             and (imgsz != rimgsz or e in SN.REFERENCE_EXAMS)]
    if NATIVE_EXAM not in exams:
        raise BaselineError("%s's final exams %s hold no %s" % (exp, defn.get("final_exams"), NATIVE_EXAM))
    seeds = [int(s) for s in defn.get("seeds") or []]
    common = sorted(set(seeds) & set(int(s) for s in rdefn.get("seeds") or []))
    if len(common) < 2:
        raise BaselineError("%s and %s share seeds %s; the comparison needs at least 2" % (exp, reference, common))
    try:                                         # every final run done before anything is scored
        for s in seeds:
            SN.final_run(exp, _final_id(s), defn)
        for s in common:
            SN.final_run(reference, _final_id(s), rdefn)
    except SN.NativeRefused as e:
        raise BaselineError(str(e))
    scores = {exam: [_native_one(exp, s, exam, imgsz, batch, device) for s in seeds] for exam in exams}
    ref = {NATIVE_EXAM: [_native_one(reference, s, NATIVE_EXAM, rimgsz, batch, device) for s in common]}
    # the record the platform reads (evidence.ALLOWED): dev only and no path; the returned record lists every
    # native file, and the verdict's report (for people) every file it read
    rec = {"format": NATIVE_FORMAT, "exp": exp, "arm": aid, "imgsz": imgsz, "seeds": seeds,
           "reference": {"exp": reference, "imgsz": rimgsz, "seeds": common, "scores": ref},
           "scores": {NATIVE_EXAM: scores[NATIVE_EXAM]}, "n_native_scores": sum(len(v) for v in scores.values()),
           "status": "complete", "written_utc": D._utc(),
           "note": "native-resolution scores (scores/<exam>@<imgsz>.json): never test, never a protocol score; "
                   "dev listed only"}
    out = C.INC_DIR / exp / NATIVE_RECORD
    _write_json(out, rec)
    rec = dict(rec, scores=scores, out=str(out))
    log("%s: native scores at %d px complete (%s); reference %s at %d px on seeds %s"
        % (exp, imgsz, ", ".join("%s %d written" % (x, sum(1 for r in v if r["status"] == "written"))
                                 for x, v in scores.items()), reference, rimgsz, common))
    if verdict:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        old = _read_json(d / ("%s.json" % NATIVE_NAME))
        prev = list((old.get("arms") or {}) if isinstance(old, dict) else [])
        testing_ok = bool(defn.get("testing")) and scorer_testing()
        native_verdict(list(dict.fromkeys(prev + [exp])), reference=reference, out_dir=out_dir,
                       testing_ok=testing_ok, resamples=resamples)
    return rec


def _native_files(exp, seeds, imgsz, testing_ok):
    """({seed: (native dev score, per-image arrays, input record)}, [missing
    run ids]) of an experiment's final runs at imgsz."""
    out, missing = {}, []
    for s in seeds:
        rid = _final_id(s)
        js, npz = SN.paths_for(C.INC_DIR / exp / "runs" / rid / "scores", NATIVE_EXAM, imgsz)
        d = _read_json(js)
        if d is None:
            missing.append(rid)
            continue
        if (not isinstance(d, dict) or d.get("format") != SN.FORMAT or d.get("exam") != NATIVE_EXAM
                or d.get("imgsz") != imgsz or d.get("exp") != exp or d.get("run_id") != rid):
            raise BaselineError("%s is not %s %s's native %s score at %d px" % (js, exp, rid, NATIVE_EXAM, imgsz))
        if d.get("native_production") is not True and not testing_ok:
            raise BaselineError("%s is a test-mode score (%s): the verdict reads native-protocol scores only"
                                % (js, "; ".join(d.get("other_deviations") or []) or "not native_production"))
        img = d.get("images") or {}
        if _sha(npz) is None or _sha(npz) != img.get("sha256"):
            raise BaselineError("%s does not hash as %s records" % (npz, js))
        arrays = SC.load_npz(npz)
        if C.sha256_text("\n".join(str(k) for k in arrays["keys"])) != d.get("key_order_sha256"):
            raise BaselineError("%s is not in the exam key order %s records" % (npz, js))
        out[s] = (d, arrays, {"run_id": rid, "score": js.name, "sha256": _sha(js), "images_sha256": img["sha256"]})
    return out, missing


def native_bootstrap(arm_arrays, ref_arrays, resamples=NATIVE_RESAMPLES, seed_text=NATIVE_SEED_TEXT, ap_fn=None,
                     species=SC.SPECIES):
    """The paired image bootstrap of D = mean over arm runs - mean over
    reference runs of the 12-class mean dev AP50-95, and of each species'
    difference (module docstring). Every run's arrays must be in one key
    order with the same GT boxes (one exam). Returns {"se", "n_valid",
    "per_species": {s: {"se", "n_valid"}}}."""
    import numpy as np
    ap_fn = ap_fn or SC._ap_per_class()
    runs = list(arm_arrays) + list(ref_arrays)
    if not arm_arrays or not ref_arrays:
        raise BaselineError("the bootstrap needs runs on both sides")
    keys = [str(k) for k in runs[0]["keys"]]
    for a in runs[1:]:
        if [str(k) for k in a["keys"]] != keys:
            raise BaselineError("the runs' per-image arrays are not in one key order")
        if not (np.array_equal(a["target_img"], runs[0]["target_img"])
                and np.array_equal(a["target_cls"], runs[0]["target_cls"])):
            raise BaselineError("the runs' GT boxes differ: they were not scored on one exam")
    n = len(keys)
    if n < 2:
        raise BaselineError("an image bootstrap needs at least 2 exam images, got %d" % n)
    counts = np.stack([np.bincount(row, minlength=n) for row in SC.resample_indices(n, resamples, seed_text)])
    tb = [SC.tie_break(a) for a in runs]
    per = []
    for name in species:
        s_id = C.CLASS_NAMES.index(name)
        arrs = [SC.species_arrays(t, s_id) for t in tb]
        per.append((name, s_id, arrs, arrs[0][3]))
    k = len(arm_arrays)
    diffs, sdiffs = [], {name: [] for name in species}
    for b in range(resamples):
        m = counts[b]
        aps = {}
        for name, s_id, arrs, gt in per:
            n_gt = int((m * gt).sum())
            if n_gt <= 0:
                continue
            aps[name] = [SC.class_ap(np.repeat(tp, m[pimg], axis=0), np.repeat(conf, m[pimg]), n_gt, s_id, ap_fn)
                         for tp, conf, pimg, _gt in arrs]
        if not aps:
            continue
        twelve = [statistics.fmean(aps[nm][r] for nm in aps) for r in range(len(runs))]
        diffs.append(statistics.fmean(twelve[:k]) - statistics.fmean(twelve[k:]))
        for nm, vals in aps.items():
            sdiffs[nm].append(statistics.fmean(vals[:k]) - statistics.fmean(vals[k:]))

    def sd(v):
        return float(np.std(np.asarray(v, dtype=np.float64), ddof=1)) if len(v) >= 2 else None
    return {"se": sd(diffs), "n_valid": len(diffs),
            "per_species": {nm: {"se": sd(v), "n_valid": len(v)} for nm, v in sdiffs.items()}}


def _native_stamps(d):
    out = {k: d.get(k) for k in NATIVE_STAMPS}
    out.update(("settings.%s" % k, (d.get("settings") or {}).get(k)) for k in NATIVE_SETTINGS)
    return out


def native_decision(arm_exps, reference=NATIVE_REFERENCE, testing_ok=False, resamples=NATIVE_RESAMPLES,
                    seed_text=NATIVE_SEED_TEXT):
    """The pre-registered rule on the native dev files (module docstring).
    Opens the experiments' exp.json and their final runs' native dev scores
    and arrays, nothing else (no protocol score, no other exam)."""
    rdefn, raid, rimgsz = _native_arm(reference, measure=False)
    arms, qualifying, pending = {}, [], []
    for e in arm_exps:
        if not (C.INC_DIR / e / "exp.json").is_file():
            arms[e] = {"status": "pending", "why": "not built"}
            pending.append(e)
            continue
        defn, aid, imgsz = _native_arm(e)
        seeds = sorted(set(int(s) for s in defn.get("seeds") or []) & set(int(s) for s in rdefn.get("seeds") or []))
        if len(seeds) < 2:
            raise BaselineError("%s and %s share seeds %s; the rule needs at least 2" % (e, reference, seeds))
        a_in, a_miss = _native_files(e, seeds, imgsz, testing_ok)
        r_in, r_miss = _native_files(reference, seeds, rimgsz, testing_ok)
        if a_miss or r_miss:
            arms[e] = {"status": "pending", "arm": aid, "imgsz": imgsz, "seeds": seeds,
                       "missing": ["%s/%s" % (e, r) for r in a_miss] + ["%s/%s" % (reference, r) for r in r_miss]}
            pending.append(e)
            continue
        stamps = None
        for who, files in ((e, a_in), (reference, r_in)):
            for s in seeds:
                st = _native_stamps(files[s][0])
                if stamps is None:
                    stamps = st
                elif st != stamps:
                    diff = sorted(k for k in st if st[k] != stamps[k])
                    raise BaselineError("%s %s was scored on another exam, scorer or settings than the comparison's "
                                        "first file (%s differ)" % (who, _final_id(s), ", ".join(diff)))
        av = [float(a_in[s][0]["species_map50_95"]) for s in seeds]
        rv = [float(r_in[s][0]["species_map50_95"]) for s in seeds]
        ma, sa = _mean_sd(av)
        mr, sr = _mean_sd(rv)
        pooled = math.sqrt((sa ** 2 + sr ** 2) / 2.0)
        diff = ma - mr
        boot = native_bootstrap([a_in[s][1] for s in seeds], [r_in[s][1] for s in seeds], resamples=resamples,
                                seed_text=seed_text)
        per = {}
        for name in SC.SPECIES:
            pa = [a_in[s][0].get("per_class", {}).get(name) for s in seeds]
            pr = [r_in[s][0].get("per_class", {}).get(name) for s in seeds]
            if any(x is None for x in pa + pr):
                continue
            da = statistics.fmean(float(x) for x in pa) - statistics.fmean(float(x) for x in pr)
            per[name] = {"arm_mean": statistics.fmean(float(x) for x in pa),
                         "ref_mean": statistics.fmean(float(x) for x in pr), "diff": da,
                         "se": boot["per_species"].get(name, {}).get("se"),
                         "n_valid": boot["per_species"].get(name, {}).get("n_valid")}
        improved = [s for s in NATIVE_TARGET_SPECIES if s in per and per[s]["diff"] > 0]
        se = boot["se"]
        conds = {"above_2_pooled_sd": diff > 2.0 * pooled, "above_se": se is not None and diff > se,
                 "target_species_improved": bool(improved)}
        q = all(conds.values())
        arms[e] = {"status": "decided", "arm": aid, "imgsz": imgsz, "reference_imgsz": rimgsz, "seeds": seeds,
                   "dev": av, "mean": ma, "sd": sa, "reference_dev": rv, "reference_mean": mr, "reference_sd": sr,
                   "diff": diff, "pooled_sd": pooled, "two_pooled_sd": 2.0 * pooled, "se_diff": se,
                   "n_valid": boot["n_valid"], "per_species": per, "target_species": list(NATIVE_TARGET_SPECIES),
                   "improved_targets": improved, "conditions": conds, "qualifies": q, "stamps": stamps,
                   "inputs": [a_in[s][2] for s in seeds],
                   "reference_inputs": [r_in[s][2] for s in seeds]}
        if q:
            qualifying.append(e)
    return {"format": NATIVE_VERDICT_FORMAT, "rule": NATIVE_RULE,
            "pre_registered": "docs/CONTINUOUS_LOOP.md, group B, Amendment (2026-10-01): the measurement arms read "
                              "at their own resolution",
            "reference": {"exp": reference, "arm": raid, "imgsz": rimgsz},
            "bootstrap": {"seed_text": seed_text, "seed": C.stable_int(seed_text), "resamples": int(resamples),
                          "paired": "one draw of the dev images for every run", "ddof": 1,
                          "statistic": "the 12-class mean AP50-95 over the species with a GT box in the resample, "
                                       "averaged over seeds per arm; the arm minus the reference"},
            "arms": arms, "qualifying": qualifying, "pending": pending, "testing_allowed": bool(testing_ok),
            "on_qualifying": "the autopilot files card X18 for a person: a stream fork proposal; nothing switches, "
                             "capacity_v1.json is not touched, test is not read",
            "note": "dev only: the native dev files of the final runs; ImageWeeds is in the report, for people"}


def native_report(decision, testing_ok=False):
    """For people: each arm's dev and imageweeds at its imgsz and at 640
    (the protocol's scores of the same final runs), and the reference's dev
    at 640, mean +- sd over the seeds. Never test."""
    rows = {}
    ref = decision["reference"]["exp"]
    for e, a in list(decision["arms"].items()) + [(ref, {"imgsz": decision["reference"]["imgsz"]})]:
        defn = _read_json(C.INC_DIR / e / "exp.json") or {}
        size = a.get("imgsz")
        out = {"arm": (defn.get("arm") or {}).get("id"), "imgsz": size, "status": a.get("status"), "exams": {},
               "inputs": []}
        for exam in [x for x in SN.EXAMS if x in (defn.get("final_exams") or [])]:
            for label, name in (("native", SN.score_name(exam, size) if size else None), ("at_640", "%s.json" % exam)):
                if name is None:
                    continue
                vals = []
                for s in defn.get("seeds") or []:
                    p = C.INC_DIR / e / "runs" / _final_id(s) / "scores" / name
                    d = _read_json(p)
                    if isinstance(d, dict) and d.get("species_map50_95") is not None and (
                            label == "native" or d.get("production") is True or testing_ok):
                        vals.append(float(d["species_map50_95"]))
                        out["inputs"].append({"path": str(p), "sha256": _sha(p)})
                m, sd = _mean_sd(vals)
                out["exams"].setdefault(exam, {})[label] = {"n": len(vals), "mean": m, "sd": sd}
        rows[e] = out
    return {"format": NATIVE_VERDICT_FORMAT + "-report", "arms": rows, "qualifying": decision["qualifying"],
            "note": "for people: dev and imageweeds at each arm's imgsz and at 640; test is never scored at another "
                    "size and is not read here"}


def _native_md(rep, decision):
    def f(x):
        return "-" if x is None else "%.4f" % x
    lines = ["# Measurement arms at their own resolution", "",
             "Qualifying for a stream fork proposal: %s." % (", ".join(decision["qualifying"]) or "none"), "",
             "| Experiment | Arm | Exam | at own imgsz mean +- sd | at 640 mean +- sd | n |", "|---|---|---|---|---|---|"]
    for e, r in rep["arms"].items():
        for exam, v in r["exams"].items():
            nat, p = v.get("native") or {}, v.get("at_640") or {}
            lines.append("| %s | %s @ %s | %s | %s +- %s | %s +- %s | %s |" % (
                e, r["arm"], r["imgsz"], exam, f(nat.get("mean")), f(nat.get("sd")), f(p.get("mean")), f(p.get("sd")),
                nat.get("n", p.get("n"))))
    for e, a in decision["arms"].items():
        if a.get("status") == "decided":
            lines.append("")
            lines.append("%s: D = %.4f, 2 x pooled sd = %.4f, SE(D) = %s, improved targets %s -> %s"
                         % (e, a["diff"], a["two_pooled_sd"], f(a["se_diff"]), a["improved_targets"] or "none",
                            "qualifies" if a["qualifies"] else "does not qualify"))
    return "\n".join(lines) + "\n"


def native_verdict(arm_exps=NATIVE_ARMS, reference=NATIVE_REFERENCE, out_dir=None, write=True, testing_ok=False,
                   resamples=NATIVE_RESAMPLES):
    """The decision and its report (module docstring); returns (decision, report)."""
    decision = native_decision(list(arm_exps), reference=reference, testing_ok=testing_ok, resamples=resamples)
    decision["generated_utc"] = D._utc()
    report = native_report(decision, testing_ok=testing_ok)
    report["generated_utc"] = decision["generated_utc"]
    if write:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        decision["out"] = str(d / ("%s.json" % NATIVE_NAME))
        _write_json(decision["out"], decision)
        report["decision_sha256"] = _sha(decision["out"])
        report["out"] = str(d / ("%s_report.json" % NATIVE_NAME))
        _write_json(report["out"], report)
        md = d / ("%s_report.md" % NATIVE_NAME)
        tmp = md.with_name(".%s.tmp" % md.name)
        tmp.write_text(_native_md(report, decision))
        os.replace(tmp, md)
    log("native verdict: qualifying %s; pending %s; %s" % (
        decision["qualifying"], decision["pending"], "; ".join(
            "%s D %.4f vs 2 pooled sd %.4f, SE %s" % (e, a["diff"], a["two_pooled_sd"],
                                                     "-" if a["se_diff"] is None else "%.4f" % a["se_diff"])
            for e, a in decision["arms"].items() if a.get("status") == "decided") or "none decided"))
    return decision, report


# ---------------------------------------------------------------- E1 verdict
E1_FORMAT = "inc2-e1-verdict/1"
E1_NAME = "e1_v1"
AGNOSTIC_FORMAT = "inc2-agnostic-rescore/1"
AGNOSTIC_RECORD = "agnostic_rescore.json"
E1_EXAM = "dev"
E1_SEED_TEXT = "inc2/e1/agnostic_se"
E1_RESAMPLES = 1000
E1_TEST_RUN = "e1test__s%d"
E1_TEST_FORMAT = "inc2-e1-test-read/1"
E1_RULE = ("E1-B qualifies when D = mean(E1-B's class-agnostic dev mAP50-95) - mean(E1-A's), on the seeds both share, "
           "exceeds 2 x pooled sd (sqrt((sd_B^2 + sd_A^2) / 2)) AND the paired image-bootstrap SE of D (1,000 resamples "
           "of the dev images under stable_int('inc2/e1/agnostic_se'), the locked scorer's collapsed AP on each run's "
           "tie-broken per-image arrays, averaged over seeds per arm); record only: never the stream's arm, nothing "
           "switches, test is read once per arm only after this verdict (inc2.baseline e1-test-read)")


def _e1_defn(exp, want_arm):
    """exp.json of an E1 arm experiment (a baseline whose e1 record names want_arm)."""
    defn = _read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(defn, dict) or defn.get("type") != "baseline":
        raise BaselineError("%s is not a built baseline experiment" % exp)
    e1 = defn.get("e1") if isinstance(defn.get("e1"), dict) else None
    if e1 is None or defn.get("recipe_name") != RC.BUDGET_NAME:
        raise BaselineError("%s is not an E1 arm (no e1 record or not cold_budget)" % exp)
    if e1.get("arm") != want_arm:
        raise BaselineError("%s is E1's arm %s, not %s" % (exp, e1.get("arm"), want_arm))
    return defn


def _e1_pair(exp, reference):
    """(B's exp.json, A's exp.json, the seeds both share) of E1's two arms."""
    db, da = _e1_defn(exp, "B"), _e1_defn(reference, "A")
    if db["e1"].get("summary_sha256") != da["e1"].get("summary_sha256") or \
            db["e1"].get("manifests") != da["e1"].get("manifests"):
        raise BaselineError("%s and %s were built from different splits v3 summaries" % (exp, reference))
    if RC.arm_id(db["arm"]) != RC.arm_id(da["arm"]):
        raise BaselineError("%s and %s train different arms" % (exp, reference))
    seeds = sorted(set(int(s) for s in db["seeds"]) & set(int(s) for s in da["seeds"]))
    if len(seeds) < 2:
        raise BaselineError("%s and %s share seeds %s; the rule needs at least 2" % (exp, reference, seeds))
    return db, da, seeds


def rescore_agnostic(exp, reference, out_dir=None, verdict=True, batch=None, device=None, resamples=E1_RESAMPLES):
    """E1's rescore (lever L23E): every final run of both arms scored for its
    per-image class-agnostic arrays on dev (inc2.scorer_agnostic, each only
    when missing; every final run must be done first), then the verdict.
    Writes INC_DIR/<exp>/agnostic_rescore.json (status complete, the dev
    files' names and sha256s, no path: the platform reads it)."""
    from . import scorer_agnostic as SA
    db, da, seeds = _e1_pair(exp, reference)
    try:
        for e, d in ((exp, db), (reference, da)):
            for s in d["seeds"]:
                SA.final_run(e, _final_id(s))
    except SA.AgnosticRefused as e:
        raise BaselineError(str(e))
    out = {}
    for e, d in ((exp, db), (reference, da)):
        rows = []
        for s in [int(x) for x in d["seeds"]]:
            try:
                r = SA.score_run(e, _final_id(s), E1_EXAM, batch=batch, device=device)
            except SA.AgnosticRefused as x:
                raise BaselineError("%s %s: %s" % (e, _final_id(s), x))
            js, _npz = SA.paths_for(C.INC_DIR / e / "runs" / _final_id(s) / "scores", E1_EXAM)
            rows.append({"run_id": _final_id(s), "score": js.name, "sha256": _sha(js), "status": r.get("status")})
        out[e] = rows
    rec = {"format": AGNOSTIC_FORMAT, "exp": exp, "reference": reference, "seeds": seeds, "exam": E1_EXAM,
           "scores": out, "status": "complete", "written_utc": D._utc(),
           "note": "class-agnostic per-image arrays of the final runs on dev (scores/dev.agnostic.json): never test, "
                   "never a protocol score"}
    _write_json(C.INC_DIR / exp / AGNOSTIC_RECORD, rec)
    log("%s: agnostic rescore complete (%s)" % (exp, ", ".join("%s %d written" % (e, sum(1 for r in v if r["status"]
                                                                                           == "written"))
                                                                for e, v in out.items())))
    if verdict:
        e1_verdict(exp, reference, out_dir=out_dir, testing_ok=bool(db.get("testing")) and scorer_testing(),
                   resamples=resamples)
    return rec


def _agnostic_files(exp, seeds, testing_ok):
    """({seed: (record, arrays)}, [missing run ids]) of an experiment's
    agnostic dev files."""
    from . import scorer_agnostic as SA
    out, missing = {}, []
    for s in seeds:
        rid = _final_id(s)
        js, npz = SA.paths_for(C.INC_DIR / exp / "runs" / rid / "scores", E1_EXAM)
        d = _read_json(js)
        if d is None:
            missing.append(rid)
            continue
        if d.get("format") != SA.FORMAT or d.get("exp") != exp or d.get("run_id") != rid or d.get("exam") != E1_EXAM:
            raise BaselineError("%s is not %s %s's agnostic %s score" % (js, exp, rid, E1_EXAM))
        if d.get("agnostic_production") is not True and not testing_ok:
            raise BaselineError("%s is a test-mode score: the verdict reads production scores only" % js)
        if _sha(npz) is None or _sha(npz) != (d.get("images") or {}).get("sha256"):
            raise BaselineError("%s does not hash as %s records" % (npz, js))
        arrays = SA.load_npz(npz)
        if C.sha256_text("\n".join(str(k) for k in arrays["keys"])) != d.get("key_order_sha256"):
            raise BaselineError("%s is not in the exam key order %s records" % (npz, js))
        out[s] = (d, arrays)
    return out, missing


def agnostic_bootstrap(arm_arrays, ref_arrays, resamples=E1_RESAMPLES, seed_text=E1_SEED_TEXT):
    """The paired image bootstrap of D = mean over arm runs - mean over
    reference runs of the collapsed AP50-95 (module docstring, E1_RULE).
    Every run's arrays must share the key order and the GT counts (one
    exam). Returns {"se", "n_valid"}."""
    import numpy as np
    runs = list(arm_arrays) + list(ref_arrays)
    if not arm_arrays or not ref_arrays:
        raise BaselineError("the bootstrap needs runs on both sides")
    keys = [str(k) for k in runs[0]["keys"]]
    for a in runs[1:]:
        if [str(k) for k in a["keys"]] != keys or not np.array_equal(a["n_gt"], runs[0]["n_gt"]):
            raise BaselineError("the runs' per-image arrays are not of one exam (key order or GT counts differ)")
    n = len(keys)
    if n < 2:
        raise BaselineError("an image bootstrap needs at least 2 exam images, got %d" % n)
    gt = np.asarray(runs[0]["n_gt"], dtype=np.int64)
    counts = np.stack([np.bincount(row, minlength=n) for row in SC.resample_indices(n, resamples, seed_text)])
    tb = [SC.tie_break(a) for a in runs]
    k = len(arm_arrays)
    diffs = []
    for b in range(resamples):
        m = counts[b]
        n_gt = int((m * gt).sum())
        if n_gt <= 0:
            continue
        vals = []
        for a in tb:
            rep = m[a["pred_img"]]
            vals.append(_collapsed_ap50_95(np.repeat(a["tp"], rep, axis=0), np.repeat(a["conf"], rep), n_gt))
        diffs.append(statistics.fmean(vals[:k]) - statistics.fmean(vals[k:]))
    se = float(np.std(np.asarray(diffs, dtype=np.float64), ddof=1)) if len(diffs) >= 2 else None
    return {"se": se, "n_valid": len(diffs)}


def _collapsed_ap50_95(tp, conf, n_gt):
    """The locked scorer's class-collapsed AP50-95 (inc.scorer.collapsed_ap)."""
    from ..inc import scorer as S
    return S.collapsed_ap(tp, conf, n_gt)[0]


def e1_decision(exp, reference, testing_ok=False, resamples=E1_RESAMPLES, seed_text=E1_SEED_TEXT):
    """E1_RULE on the agnostic dev files (dev only)."""
    db, da, seeds = _e1_pair(exp, reference)
    b_in, b_miss = _agnostic_files(exp, seeds, testing_ok)
    a_in, a_miss = _agnostic_files(reference, seeds, testing_ok)
    base = {"format": E1_FORMAT, "rule": E1_RULE, "exp": exp, "reference": reference, "seeds": seeds,
            "exam": E1_EXAM, "arm": RC.arm_id(db["arm"]),
            "pre_registered": "docs/CONTINUOUS_LOOP.md, Amendment (2026-10-03): E1, weed-box base v3",
            "images": {"B": db["base"]["n_images"], "A": da["base"]["n_images"]},
            "summary_sha256": db["e1"].get("summary_sha256"), "testing_allowed": bool(testing_ok)}
    if b_miss or a_miss:
        return dict(base, status="pending", missing=["%s/%s" % (exp, r) for r in b_miss]
                    + ["%s/%s" % (reference, r) for r in a_miss])
    stamps = None
    for who, files in ((exp, b_in), (reference, a_in)):
        for s in seeds:
            st = {k: (files[s][0].get("recorded") or {}).get("stamps", {}).get(k)
                  for k in ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "n_images")}
            st["n_gt"] = files[s][0].get("n_gt")
            if stamps is None:
                stamps = st
            elif st != stamps:
                raise BaselineError("%s %s was scored on another exam or scorer than the comparison's first file (%s)"
                                    % (who, _final_id(s), sorted(k for k in st if st[k] != stamps[k])))
    bv = [float(b_in[s][0]["recorded"]["agnostic_map50_95"]) for s in seeds]
    av = [float(a_in[s][0]["recorded"]["agnostic_map50_95"]) for s in seeds]
    mb, sb = _mean_sd(bv)
    ma, sa = _mean_sd(av)
    pooled = math.sqrt((sb ** 2 + sa ** 2) / 2.0)
    diff = mb - ma
    boot = agnostic_bootstrap([b_in[s][1] for s in seeds], [a_in[s][1] for s in seeds], resamples=resamples,
                              seed_text=seed_text)
    se = boot["se"]
    conds = {"above_2_pooled_sd": diff > 2.0 * pooled, "above_se": se is not None and diff > se}
    return dict(base, status="decided", dev={"B": bv, "A": av}, mean={"B": mb, "A": ma}, sd={"B": sb, "A": sa},
                diff=diff, pooled_sd=pooled, two_pooled_sd=2.0 * pooled, se_diff=se, n_valid=boot["n_valid"],
                bootstrap={"seed_text": seed_text, "seed": C.stable_int(seed_text), "resamples": int(resamples),
                           "paired": "one draw of the dev images for every run", "ddof": 1},
                conditions=conds, qualifies=all(conds.values()), stamps=stamps,
                inputs={exp: [{"run_id": _final_id(s), "sha256": (b_in[s][0].get("images") or {}).get("sha256")}
                              for s in seeds],
                        reference: [{"run_id": _final_id(s), "sha256": (a_in[s][0].get("images") or {}).get("sha256")}
                                    for s in seeds]},
                note="dev only; record only (never the stream's arm); test is read once per arm after this verdict "
                     "(e1-test-read), by a person's submission")


def e1_report(decision):
    """For people: both arms' final scores on their final exams (dev and
    ImageWeeds), 12-class and agnostic, mean +- sd. Never test."""
    rows = {}
    for e in (decision["exp"], decision["reference"]):
        defn = _read_json(C.INC_DIR / e / "exp.json") or {}
        out = {"e1_arm": (defn.get("e1") or {}).get("arm"), "images": (defn.get("base") or {}).get("n_images"),
               "exams": {}}
        for exam in [x for x in defn.get("final_exams") or [] if x != "test"]:
            tw, ag = [], []
            for s in defn.get("seeds") or []:
                d = _read_json(C.INC_DIR / e / "runs" / _final_id(s) / "scores" / ("%s.json" % exam))
                if isinstance(d, dict) and d.get("agnostic_map50_95") is not None:
                    tw.append(float(d["map50_95"]))
                    ag.append(float(d["agnostic_map50_95"]))
            m1, s1 = _mean_sd(tw)
            m2, s2 = _mean_sd(ag)
            out["exams"][exam] = {"n": len(ag), "twelve": {"mean": m1, "sd": s1}, "agnostic": {"mean": m2, "sd": s2}}
        rows[e] = out
    return {"format": E1_FORMAT + "-report", "arms": rows, "qualifies": decision.get("qualifies"),
            "note": "for people: dev and ImageWeeds of both arms; the decision reads dev only; test is read once per "
                    "arm after the verdict"}


def _e1_md(rep, decision):
    def f(x):
        return "-" if x is None else "%.4f" % x
    lines = ["# E1: weed-box base v3 (class-agnostic, dev decides)", "",
             "| Experiment | Arm | Images | Exam | agnostic mean +- sd | 12-class mean +- sd | n |",
             "|---|---|---|---|---|---|---|"]
    for e, r in rep["arms"].items():
        for exam, v in r["exams"].items():
            lines.append("| %s | %s | %s | %s | %s +- %s | %s +- %s | %s |" % (
                e, r["e1_arm"], r["images"], exam, f(v["agnostic"]["mean"]), f(v["agnostic"]["sd"]),
                f(v["twelve"]["mean"]), f(v["twelve"]["sd"]), v["n"]))
    if decision.get("status") == "decided":
        lines += ["", "D = %s, 2 x pooled sd = %s, SE(D) = %s -> %s" % (
            f(decision["diff"]), f(decision["two_pooled_sd"]), f(decision["se_diff"]),
            "E1-B qualifies" if decision["qualifies"] else "E1-B does not qualify")]
    else:
        lines += ["", "pending: %s" % ", ".join(decision.get("missing") or [])]
    return "\n".join(lines) + "\n"


def e1_verdict(exp, reference, out_dir=None, write=True, testing_ok=False, resamples=E1_RESAMPLES):
    """The decision (capacity/e1_v1.json, dev only) and its report (people)."""
    decision = e1_decision(exp, reference, testing_ok=testing_ok, resamples=resamples)
    decision["generated_utc"] = D._utc()
    report = e1_report(decision)
    report["generated_utc"] = decision["generated_utc"]
    if write:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        decision["out"] = str(d / ("%s.json" % E1_NAME))
        _write_json(decision["out"], decision)
        report["decision_sha256"] = _sha(decision["out"])
        report["out"] = str(d / ("%s_report.json" % E1_NAME))
        _write_json(report["out"], report)
        md = d / ("%s_report.md" % E1_NAME)
        tmp = md.with_name(".%s.tmp" % md.name)
        tmp.write_text(_e1_md(report, decision))
        os.replace(tmp, md)
    if decision.get("status") == "decided":
        log("E1 verdict: D %.4f vs 2 pooled sd %.4f, SE %s -> %s" % (
            decision["diff"], decision["two_pooled_sd"], "-" if decision["se_diff"] is None else
            "%.4f" % decision["se_diff"], "qualifies" if decision["qualifies"] else "does not qualify"))
    else:
        log("E1 verdict pending: %s" % decision.get("missing"))
    return decision, report


def e1_test_read(exps, verdict_path=None, run_fmt=E1_TEST_RUN):
    """The milestone read of test for E1 (amendment 2026-10-03, P10): once
    per arm, only after capacity/e1_v1.json is decided. For each arm's seed,
    a kind-final spec on exam test from its base run's weights
    (runs/e1test__s<k>), a one-line submission list, and the sbatch argv of
    run_inc2_job.sh; a person submits it, the driver does not track it, and
    the scores stay off the platform's evidence. Refuses when the verdict is
    missing, pending, or not the one a previous read of the arm recorded,
    and when a test score of the arm already exists."""
    vp = Path(verdict_path or (C.INC_DIR / "capacity" / ("%s.json" % E1_NAME)))
    v = _read_json(vp)
    if not isinstance(v, dict) or v.get("format") != E1_FORMAT or v.get("status") != "decided":
        raise BaselineError("%s is not a decided E1 verdict: test is read only after it" % vp)
    vsha = _sha(vp)
    out = {}
    for exp in exps:
        if exp not in (v.get("exp"), v.get("reference")):
            raise BaselineError("%s is not one of the verdict's arms (%s, %s)" % (exp, v.get("exp"), v.get("reference")))
        root = C.INC_DIR / exp
        defn = _read_json(root / "exp.json")
        if not isinstance(defn, dict) or not isinstance(defn.get("e1"), dict):
            raise BaselineError("%s is not an E1 arm" % exp)
        prev = _read_json(root / "e1_test_read.json")
        if isinstance(prev, dict) and prev.get("verdict_sha256") != vsha:
            raise BaselineError("%s's test read was prepared under another verdict (%s): the verdict changed"
                                % (exp, str(prev.get("verdict_sha256"))[:12]))
        specs = []
        for s in [int(x) for x in defn["seeds"]]:
            rid = run_fmt % s
            out_d = root / "runs" / rid
            if (out_d / "scores" / "test.json").exists() or (out_d / "run.json").exists():
                raise BaselineError("%s/%s already holds a run or a test score: test is read once per arm" % (exp, rid))
            w = root / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
            if not w.is_file():
                raise BaselineError("%s has no base weights for seed %d (%s)" % (exp, s, w))
            spec = {"exp": exp, "run_id": rid, "kind": "final", "init": str(w.resolve()), "exams": ["test"],
                    "out_dir": str(out_d)}
            D._write_json(out_d / "spec.json", spec)
            try:
                T.validate_spec(spec, out_d / "spec.json")
            except T.RunError as e:
                (out_d / "spec.json").unlink()
                raise BaselineError("the v2 executor refuses the spec: %s" % e)
            specs.append(str(out_d / "spec.json"))
        lst = root / "submissions" / "e1test.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        tmp = lst.with_name(".%s.%d.tmp" % (lst.name, os.getpid()))
        tmp.write_text("".join("%s\n" % p for p in specs))
        os.replace(tmp, lst)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        argv = ["sbatch", "--parsable", "--array=0-%d" % (len(specs) - 1), "--job-name=inc_%s_e1test" % exp,
                "--output=%s" % (root / "logs" / "%x_%A_%a.out"), str(job_script_path()), str(lst), exp]
        rec = {"format": E1_TEST_FORMAT, "exp": exp, "verdict": str(vp), "verdict_sha256": vsha,
               "verdict_qualifies": v.get("qualifies"), "specs": specs, "list": str(lst), "argv": argv,
               "written_utc": D._utc(),
               "note": "the milestone read of test for E1 (once per arm, after the dev verdict); a person submits the "
                       "argv; the scores stay off the platform's evidence"}
        _write_json(root / "e1_test_read.json", rec)
        out[exp] = rec
    return out


# ---------------------------------------------------------------- E2 verdict
E2_FORMAT = "inc2-e2-verdict/1"
E2_NAME = "e2_v1"
E2_RESCORE_FORMAT = "inc2-e2-rescore/1"
E2_RESCORE_NAME = "e2_rescore.json"                 # in capacity/, beside the verdict
E2_REFERENCE = RC.E2_REFERENCE_EXP
E2_EXAM = "dev"
E2_IMGSZ = 640
E2_SEED_TEXT = "inc2/e2/species_se"
E2_RESAMPLES = 1000
E2_REPORTED_SPECIES = NATIVE_TARGET_SPECIES
E2_TEST_RUN = "e2test__s%d"
E2_TEST_FORMAT = "inc2-e2-test-read/1"
E2_TEST_RECORD = "e2_test_read.json"
E2_TEST_LIST = "submissions/e2test.txt"
E2_TEST_REPORT_FORMAT = "inc2-e2-test-report/1"
E2_RULE = ("for each arm (W: b_v2_m640's cold recipe; S: x1b), on the seeds it shares with b_v2_m640 (0, 1, 2), D = "
           "mean(arm's dev species_map50_95) - mean(b_v2_m640's), each final run's dev score at 640 by "
           "inc2.scorer_native (the locked scorer's code as a library); the arm qualifies when D > 2 x pooled sd "
           "(sqrt((sd_arm^2 + sd_ref^2) / 2), sample sd) AND D > the paired image-bootstrap SE of D (1,000 resamples "
           "of the dev images under stable_int('inc2/e2/species_se'), one draw for every run; per run and resample "
           "each species' AP50-95 on the run's tie-broken per-image arrays, the 12-class mean over the species with a "
           "GT box in the resample, the mean over seeds per arm, the arm minus the reference); when both qualify the "
           "larger D is E2's choice, a tie goes to S; record only: nothing switches; the sealed test is read once, "
           "for the chosen arm only, after this verdict, by a person (inc2.baseline e2-test-read)")
# What a recomputation must reproduce of a decided e2_v1.json (e2_verdict): the decision, never what is reported
# beside it (the protocol dev means, which a re-score attempt of a base run rewrites, or the native_v1 cross-check)
E2_DECISION_KEYS = ("status", "qualifying", "chosen")
E2_ARM_DECISION_KEYS = ("status", "seeds", "dev", "reference_dev", "diff", "pooled_sd", "se_diff", "conditions",
                        "qualifies")


def e2_exps():
    """{'W': [three experiments], 'S': [...]}: E2's pre-registered experiments."""
    return {k: [RC.e2_exp(k, s) for s in RC.E2_SEEDS] for k in ("W", "S")}


def _e2_defn(exp, letter):
    """exp.json of one of E2-letter's experiments (a baseline whose e2 record
    names the arm, one seed, arm m640, no test among its finals)."""
    defn = _read_json(C.INC_DIR / exp / "exp.json")
    e2 = defn.get("e2") if isinstance(defn, dict) else None
    if not (isinstance(defn, dict) and defn.get("type") == "baseline" and isinstance(e2, dict)
            and e2.get("arm") == letter and defn.get("seeds") == [e2.get("seed")]
            and (defn.get("arm") or {}).get("id") == RC.E2_ARM and "test" not in (defn.get("final_exams") or [])):
        raise BaselineError("%s is not E2-%s's experiment (a baseline whose e2 record names arm %s, one seed, arm %s, "
                            "no test among its finals)" % (exp, letter, letter, RC.E2_ARM))
    return defn


def _e2_base_run(exp, defn, testing_ok):
    """What E2's verdict reads of an experiment's base run: done, started
    from the recorded init, the arm's recipe, and (production) a production
    run whose recipe, init, environment and whole load passed inc2.train's
    checks; its final run carries its weights. Returns {run_id,
    init_sha256, weights_sha256, training_env}."""
    e2 = defn["e2"]
    s = int(e2["seed"])
    rid = "base__s%d" % s
    rj = _read_json(C.INC_DIR / exp / "runs" / rid / "run.json")
    fj = _read_json(C.INC_DIR / exp / "runs" / _final_id(s) / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise BaselineError("%s/%s is not done" % (exp, rid))
    probs = []
    if rj.get("init_sha256") != (e2.get("init") or {}).get("sha256"):
        probs.append("it did not start from the recorded init (%s, the record %s)"
                     % (str(rj.get("init_sha256"))[:12], str((e2.get("init") or {}).get("sha256"))[:12]))
    if rj.get("recipe_name") != RC.E2_ARMS[e2["arm"]]:
        probs.append("it trained %r, not E2-%s's %s" % (rj.get("recipe_name"), e2["arm"], RC.E2_ARMS[e2["arm"]]))
    if not testing_ok:
        if rj.get("testing") is not False:
            probs.append("it is not a production run")
        if rj.get("protocol_recipe") is not True:
            probs.append("its recipe departs from the table")
        if (rj.get("init_check") or {}).get("passed") is not True:
            probs.append("its init check did not pass")
        if (rj.get("training_env_check") or {}).get("passed") is not True:
            probs.append("it did not train in the reference's environment")
        it = rj.get("init_transfer") or {}
        if it.get("whole") is not True or it.get("equal") != it.get("tensors"):
            probs.append("it does not show the init loaded whole (init_transfer %s)"
                         % {k: it.get(k) for k in ("tensors", "equal", "first_differing")})
    if not isinstance(fj, dict) or fj.get("status") != "done" or fj.get("weights_sha256") != rj.get("weights_sha256"):
        probs.append("its final run %s does not carry its weights" % _final_id(s))
    if probs:
        raise BaselineError("%s/%s: %s" % (exp, rid, "; ".join(probs)))
    return {"run_id": rid, "init_sha256": rj.get("init_sha256"), "weights_sha256": rj.get("weights_sha256"),
            "training_env": {k: rj.get(k) for k in T.E2_ENV_KEYS}}


def _protocol_dev(exp, seed, testing_ok):
    """A base run's protocol dev score (scores/dev.json, species_map50_95;
    production unless testing_ok), or None: reported beside, never deciding."""
    d = _read_json(C.INC_DIR / exp / "runs" / ("base__s%d" % int(seed)) / "scores" / "dev.json")
    if not isinstance(d, dict) or d.get("exam") != "dev" or (d.get("production") is not True and not testing_ok):
        return None
    v = d.get("species_map50_95", d.get("map50_95"))
    return float(v) if v is not None else None


def _e2_mean(vals):
    vals = [v for v in vals if v is not None]
    m, sd = _mean_sd(vals)
    return {"mean": m, "sd": sd, "n": len(vals)}


def e2_decision(exps=None, reference=E2_REFERENCE, testing_ok=False, resamples=E2_RESAMPLES, seed_text=E2_SEED_TEXT):
    """E2_RULE on the native dev files of E2's final runs and the reference's
    (dev only). Reads the experiments' exp.json, their base and final runs'
    run.json, the native dev@640 files and arrays, the base runs' protocol
    dev scores (reported beside) and, recorded only, capacity/native_v1.json.
    It opens no other exam."""
    exps = exps or e2_exps()
    rdefn, raid, rimgsz = _native_arm(reference, measure=False)
    if rimgsz != E2_IMGSZ:
        raise BaselineError("the reference %s is read at %d px, not %d" % (reference, rimgsz, E2_IMGSZ))
    rseeds = sorted(int(x) for x in rdefn.get("seeds") or [])
    loaded, defns, starts = {}, {}, set()
    for letter in ("W", "S"):
        for e in exps.get(letter) or []:
            if not (C.INC_DIR / e / "exp.json").is_file():
                loaded.setdefault(letter, []).append((e, None))
                continue
            d = _e2_defn(e, letter)
            defns[e] = d
            loaded.setdefault(letter, []).append((e, d))
            starts.add(json.dumps([(d["e2"].get("init") or {}).get("exp"),
                                   (d["e2"].get("e1_verdict") or {}).get("decision")], sort_keys=True))
            if (d["e2"].get("reference") or {}).get("manifest_sha256") != (rdefn.get("base") or {}).get("manifest_sha256"):
                raise BaselineError("%s was built against another manifest than the reference %s's" % (e, reference))
    if len(starts) > 1:
        raise BaselineError("E2's experiments start from different E1-B records (the init's experiment or E1's "
                            "decision differ)")
    arms, pending, qualifying = {}, [], []
    for letter in ("W", "S"):
        items = loaded.get(letter) or []
        unbuilt = [e for e, d in items if d is None]
        if not items or unbuilt:
            arms[letter] = {"status": "pending", "why": "not built: %s" % (", ".join(unbuilt) or "no experiment")}
            pending.append(letter)
            continue
        seeds = [int(d["e2"]["seed"]) for _e, d in items]
        if len(set(seeds)) != len(seeds):
            raise BaselineError("E2-%s holds two experiments of one seed (%s)" % (letter, seeds))
        by_seed = {int(d["e2"]["seed"]): e for e, d in items}
        shared = sorted(set(seeds) & set(rseeds))
        if len(shared) < 2:
            raise BaselineError("E2-%s and %s share seeds %s; the rule needs at least 2" % (letter, reference, shared))
        a_in, missing = {}, []
        for s in shared:
            f, miss = _native_files(by_seed[s], [s], E2_IMGSZ, testing_ok)
            missing += ["%s/%s" % (by_seed[s], r) for r in miss]
            if s in f:
                a_in[s] = f[s]
        r_in, r_miss = _native_files(reference, shared, rimgsz, testing_ok)
        missing += ["%s/%s" % (reference, r) for r in r_miss]
        if missing:
            arms[letter] = {"status": "pending", "seeds": shared, "missing": missing}
            pending.append(letter)
            continue
        base, rbase, inputs, rinputs = {}, {}, [], []
        for s in shared:
            e = by_seed[s]
            br = _e2_base_run(e, defns[e], testing_ok)
            base[s] = br
            dn = a_in[s][0]
            if dn.get("weights_sha256") != br["weights_sha256"]:
                raise BaselineError("%s %s's native score names weights %s, its base run trained %s"
                                    % (e, _final_id(s), str(dn.get("weights_sha256"))[:12], br["weights_sha256"][:12]))
            inputs.append(dict(a_in[s][2], exp=e, weights_sha256=br["weights_sha256"], init_sha256=br["init_sha256"]))
            rj = _read_json(C.INC_DIR / reference / "runs" / _final_id(s) / "run.json") or {}
            if r_in[s][0].get("weights_sha256") != rj.get("weights_sha256"):
                raise BaselineError("%s %s's native score names weights %s, its run.json %s"
                                    % (reference, _final_id(s), str(r_in[s][0].get("weights_sha256"))[:12],
                                       str(rj.get("weights_sha256"))[:12]))
            rinputs.append(dict(r_in[s][2], exp=reference, weights_sha256=rj.get("weights_sha256")))
            brj = _read_json(C.INC_DIR / reference / "runs" / ("base__s%d" % s) / "run.json") or {}
            rbase[s] = {k: brj.get(k) for k in T.E2_ENV_KEYS}
        if not testing_ok:
            for who, files in ((letter, a_in), (reference, r_in)):
                for s in shared:
                    dn = files[s][0]
                    if (dn.get("vs_protocol_score") or {}).get("compared") is not True \
                            or (dn.get("protocol_score") or {}).get("production") is not True:
                        raise BaselineError("%s %s's native dev score was not checked against its production protocol "
                                            "score" % (who, _final_id(s)))
        stamps = None
        for who, files in ((letter, a_in), (reference, r_in)):
            for s in shared:
                st = _native_stamps(files[s][0])
                if stamps is None:
                    stamps = st
                elif st != stamps:
                    diff = sorted(k for k in st if st[k] != stamps[k])
                    raise BaselineError("%s %s was scored on another exam, scorer or settings than the comparison's "
                                        "first file (%s differ)" % (who, _final_id(s), ", ".join(diff)))
        av = [float(a_in[s][0]["species_map50_95"]) for s in shared]
        rv = [float(r_in[s][0]["species_map50_95"]) for s in shared]
        ma, sa = _mean_sd(av)
        mr, sr = _mean_sd(rv)
        pooled = math.sqrt((sa ** 2 + sr ** 2) / 2.0)
        diff = ma - mr
        boot = native_bootstrap([a_in[s][1] for s in shared], [r_in[s][1] for s in shared], resamples=resamples,
                                seed_text=seed_text)
        se = boot["se"]
        conds = {"above_2_pooled_sd": diff > 2.0 * pooled, "above_se": se is not None and diff > se}
        q = all(conds.values())
        ag_a = [a_in[s][0].get("agnostic_map50_95") for s in shared]
        ag_r = [r_in[s][0].get("agnostic_map50_95") for s in shared]
        species = {}
        for name in E2_REPORTED_SPECIES:
            pa = [(a_in[s][0].get("per_class") or {}).get(name) for s in shared]
            pr = [(r_in[s][0].get("per_class") or {}).get(name) for s in shared]
            if any(x is None for x in pa + pr):
                species[name] = {"arm_mean": None, "ref_mean": None, "diff": None, "se": None}
                continue
            am, rm = statistics.fmean(float(x) for x in pa), statistics.fmean(float(x) for x in pr)
            species[name] = {"arm_mean": am, "ref_mean": rm, "diff": am - rm,
                             "se": boot["per_species"].get(name, {}).get("se"),
                             "n_valid": boot["per_species"].get(name, {}).get("n_valid")}
        ga, gr = _e2_mean(ag_a), _e2_mean(ag_r)
        pa_, pr_ = (_e2_mean([_protocol_dev(by_seed[s], s, testing_ok) for s in shared]),
                    _e2_mean([_protocol_dev(reference, s, testing_ok) for s in shared]))
        envs = [base[s]["training_env"] for s in shared] + [rbase[s] for s in shared]
        arms[letter] = {
            "status": "decided", "recipe_name": RC.E2_ARMS[letter], "seeds": shared,
            "experiments": [by_seed[s] for s in shared],
            "dev": av, "mean": ma, "sd": sa, "reference_dev": rv, "reference_mean": mr, "reference_sd": sr,
            "diff": diff, "pooled_sd": pooled, "two_pooled_sd": 2.0 * pooled, "se_diff": se,
            "n_valid": boot["n_valid"], "conditions": conds, "qualifies": q, "stamps": stamps,
            "inputs": inputs, "reference_inputs": rinputs,
            "reported": {"agnostic": {"arm": ga, "reference": gr,
                                      "diff": (ga["mean"] - gr["mean"]) if None not in (ga["mean"], gr["mean"])
                                      else None},
                         "species": species,
                         "protocol_dev": {"arm": pa_, "reference": pr_},
                         "training_env": {"arm": [base[s]["training_env"] for s in shared],
                                          "reference": [rbase[s] for s in shared],
                                          "same": len({json.dumps(e, sort_keys=True) for e in envs}) == 1}}}
        if q:
            qualifying.append(letter)
    status = "decided" if all((arms.get(k) or {}).get("status") == "decided" for k in ("W", "S")) else "pending"
    chosen = None
    if status == "decided" and qualifying:
        best = max(arms[k]["diff"] for k in qualifying)
        tied = [k for k in qualifying if abs(arms[k]["diff"] - best) <= 1e-12]
        chosen = "S" if "S" in tied else tied[0]
    nv = _read_json(C.INC_DIR / "capacity" / ("%s.json" % NATIVE_NAME))
    match_nv = None
    if isinstance(nv, dict):
        nv_ref = {(r.get("run_id"), r.get("sha256")) for a in (nv.get("arms") or {}).values() if isinstance(a, dict)
                  for r in a.get("reference_inputs") or [] if isinstance(r, dict)}
        ours = {(r.get("run_id"), r.get("sha256")) for a in arms.values() for r in a.get("reference_inputs") or []}
        match_nv = bool(ours) and ours <= nv_ref
    first = next((d for d in defns.values()), None)
    return {"format": E2_FORMAT, "name": E2_NAME, "rule": E2_RULE, "pre_registered": RC.E2_DECIDED_BY,
            "status": status, "exam": E2_EXAM, "imgsz": E2_IMGSZ,
            "reference": {"exp": reference, "arm": raid, "seeds": rseeds},
            "init": {"exp": (first["e2"].get("init") or {}).get("exp") if first else None,
                     "e1_decision": (first["e2"].get("e1_verdict") or {}).get("decision") if first else None},
            "arms": arms, "qualifying": [k for k in ("W", "S") if k in qualifying], "chosen": chosen,
            "pending": pending,
            "bootstrap": {"seed_text": seed_text, "seed": C.stable_int(seed_text), "resamples": int(resamples),
                          "paired": "one draw of the dev images for every run", "ddof": 1,
                          "statistic": "the 12-class mean AP50-95 over the species with a GT box in the resample, "
                                       "averaged over seeds per arm; the arm minus the reference"},
            "reference_matches_native_v1": match_nv, "testing_allowed": bool(testing_ok),
            "on_decision": "record only: nothing switches (the stream's arm, pool and incumbent stay); the autopilot "
                           "raises one card; a person reads the chosen arm's sealed test once (e2-test-read)",
            "note": "dev only: the native dev files of the final runs at 640; ImageWeeds is in the report, for people"}


def _e2_canonical(d):
    """The decision of an E2 verdict document (E2_DECISION_KEYS, each arm's
    E2_ARM_DECISION_KEYS and its inputs' sha256s): what a recomputation must
    reproduce for a decided file to be kept."""
    out = {k: (d or {}).get(k) for k in E2_DECISION_KEYS}
    arms = {}
    for k, a in sorted(((d or {}).get("arms") or {}).items()):
        a = a if isinstance(a, dict) else {}
        x = {f: a.get(f) for f in E2_ARM_DECISION_KEYS}
        for f in ("inputs", "reference_inputs"):
            x[f] = [[r.get(g) for g in ("exp", "run_id", "sha256", "images_sha256", "weights_sha256")]
                    for r in a.get(f) or [] if isinstance(r, dict)]
        arms[k] = x
    out["arms"] = arms
    return json.loads(json.dumps(out, sort_keys=True))


def e2_report(decision, testing_ok=False):
    """For people: each arm's and the reference's finals on dev and
    ImageWeeds (scores/<exam>.json, production unless testing_ok), 12-class
    and agnostic, mean +- sd over the seeds. Never test (the reference's
    finals include it: filtered out)."""
    rows = {}
    ref = decision["reference"]["exp"]
    groups = [(k, [r.get("exp") for r in (a.get("inputs") or [])] or list(e2_exps()[k]))
              for k, a in sorted((decision.get("arms") or {}).items())]
    rseeds = sorted({s for a in (decision.get("arms") or {}).values() for s in (a.get("seeds") or [])}) \
        or decision["reference"].get("seeds") or []
    for label, exps in groups + [("reference", None)]:
        out = {"exams": {}, "experiments": exps if exps else [ref]}
        for exam in RC.E2_FINAL_EXAMS:
            tw, ag = [], []
            pairs = [(e, int(((_read_json(C.INC_DIR / e / "exp.json") or {}).get("seeds") or [0])[0])) for e in exps] \
                if exps else [(ref, s) for s in rseeds]
            for e, s in pairs:
                d = _read_json(C.INC_DIR / e / "runs" / _final_id(s) / "scores" / ("%s.json" % exam))
                if isinstance(d, dict) and d.get("exam", exam) == exam and (d.get("production") is True or testing_ok):
                    v = d.get("species_map50_95", d.get("map50_95"))
                    if v is not None:
                        tw.append(float(v))
                    if d.get("agnostic_map50_95") is not None:
                        ag.append(float(d["agnostic_map50_95"]))
            m1, s1 = _mean_sd(tw)
            m2, s2 = _mean_sd(ag)
            out["exams"][exam] = {"n": len(tw), "twelve": {"mean": m1, "sd": s1}, "agnostic": {"mean": m2, "sd": s2}}
        rows[label] = out
    return {"format": E2_FORMAT + "-report", "arms": rows, "qualifying": decision.get("qualifying"),
            "chosen": decision.get("chosen"), "status": decision.get("status"),
            "note": "for people: dev and ImageWeeds of each arm and of b_v2_m640; the decision reads dev only; the "
                    "sealed test is read once, for the chosen arm, after the verdict"}


def _e2_md(rep, decision):
    def f(x):
        return "-" if x is None else "%.4f" % x
    lines = ["# E2: the 12-class detector on E1-B's backbone (dev decides)", "",
             "| Arm | Exam | 12-class mean +- sd | agnostic mean +- sd | n |", "|---|---|---|---|---|"]
    for k, r in rep["arms"].items():
        for exam, v in r["exams"].items():
            lines.append("| %s | %s | %s +- %s | %s +- %s | %s |" % (
                "E2-%s" % k if k in RC.E2_ARMS else "b_v2_m640", exam, f(v["twelve"]["mean"]), f(v["twelve"]["sd"]),
                f(v["agnostic"]["mean"]), f(v["agnostic"]["sd"]), v["n"]))
    for k, a in sorted((decision.get("arms") or {}).items()):
        lines.append("")
        if a.get("status") == "decided":
            lines.append("E2-%s: D = %s, 2 x pooled sd = %s, SE(D) = %s -> %s" % (
                k, f(a["diff"]), f(a["two_pooled_sd"]), f(a["se_diff"]),
                "qualifies" if a["qualifies"] else "does not qualify"))
        else:
            lines.append("E2-%s: pending (%s)" % (k, a.get("why") or ", ".join(a.get("missing") or [])))
    lines += ["", "Chosen: %s" % ("E2-%s" % decision["chosen"] if decision.get("chosen") else "none")]
    return "\n".join(lines) + "\n"


def e2_verdict(exps=None, reference=E2_REFERENCE, out_dir=None, write=True, testing_ok=False,
               resamples=E2_RESAMPLES):
    """The decision (capacity/e2_v1.json, dev only) and its report (people).
    A decided e2_v1.json is never rewritten: a recomputation whose decision
    (_e2_canonical) agrees keeps the file byte for byte (decision["kept"]),
    one that disagrees refuses; a pending file is overwritten."""
    decision = e2_decision(exps, reference, testing_ok=testing_ok, resamples=resamples)
    decision["generated_utc"] = D._utc()
    report = e2_report(decision, testing_ok=testing_ok)
    report["generated_utc"] = decision["generated_utc"]
    if write:
        d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
        out = d / ("%s.json" % E2_NAME)
        decision["out"] = str(out)
        old = _read_json(out)
        if isinstance(old, dict) and old.get("status") == "decided":
            a, b = _e2_canonical(old), _e2_canonical(decision)
            if a != b:
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                raise BaselineError("%s holds a decided verdict this recomputation does not reproduce (%s differ): a "
                                    "decided E2 verdict is never rewritten; a person moves it aside" % (out, diff))
            decision["kept"] = True
        else:
            _write_json(out, {k: v for k, v in decision.items() if k != "kept"})
        report["decision_sha256"] = _sha(out)
        report["out"] = str(d / ("%s_report.json" % E2_NAME))
        _write_json(report["out"], report)
        md = d / ("%s_report.md" % E2_NAME)
        tmp = md.with_name(".%s.tmp" % md.name)
        tmp.write_text(_e2_md(report, decision))
        os.replace(tmp, md)
    log("E2 verdict: %s; %s" % (decision["status"], "; ".join(
        "E2-%s D %.4f vs 2 pooled sd %.4f, SE %s -> %s" % (k, a["diff"], a["two_pooled_sd"],
                                                           "-" if a["se_diff"] is None else "%.4f" % a["se_diff"],
                                                           "qualifies" if a["qualifies"] else "does not qualify")
        if a.get("status") == "decided" else "E2-%s pending" % k for k, a in sorted(decision["arms"].items()))))
    return decision, report


def rescore_e2(exps=None, reference=E2_REFERENCE, out_dir=None, batch=None, device=None, resamples=E2_RESAMPLES):
    """E2's rescore (lever L23C): every E2 final run scored on dev at 640 by
    inc2.scorer_native, and each reference final run of the seeds they share
    whose file is missing (each written once; every final run must be done
    first), then the verdict, then capacity/e2_rescore.json (complete: the
    files' names and sha256s and the verdict's sha256, no path). Returns that
    record. Refuses when the verdict is still pending afterwards."""
    exps = exps or e2_exps()
    defns = {}
    for letter in ("W", "S"):
        for e in exps.get(letter) or []:
            defns[e] = (letter, _e2_defn(e, letter))
    rdefn, _raid, rimgsz = _native_arm(reference, measure=False)
    rseeds = set(int(x) for x in rdefn.get("seeds") or [])
    shared = sorted({int(d["e2"]["seed"]) for _l, d in defns.values()} & rseeds)
    try:                                         # every final run done before anything is scored
        for e, (_l, d) in defns.items():
            SN.final_run(e, _final_id(d["e2"]["seed"]), d)
        for s in shared:
            SN.final_run(reference, _final_id(s), rdefn)
    except SN.NativeRefused as x:
        raise BaselineError(str(x))
    arms = {}
    for e, (letter, d) in defns.items():
        r = _native_one(e, int(d["e2"]["seed"]), E2_EXAM, E2_IMGSZ, batch, device)
        arms.setdefault(letter, []).append(dict(r, exp=e))
    ref = [_native_one(reference, s, E2_EXAM, rimgsz, batch, device) for s in shared]
    testing_ok = all(bool(d.get("testing")) for _l, d in defns.values()) and scorer_testing()
    decision, _rep = e2_verdict(exps, reference, out_dir=out_dir, testing_ok=testing_ok, resamples=resamples)
    if decision.get("status") != "decided":
        raise BaselineError("E2's verdict is pending after the rescore: %s"
                            % {k: a.get("why") or a.get("missing") for k, a in decision["arms"].items()
                               if a.get("status") != "decided"})
    vpath = Path(decision["out"])
    rec = {"format": E2_RESCORE_FORMAT, "status": "complete", "exam": E2_EXAM, "imgsz": E2_IMGSZ,
           "arms": {k: [{f: r.get(f) for f in ("exp", "run_id", "score", "sha256", "status")} for r in v]
                    for k, v in sorted(arms.items())},
           "reference": {"exp": reference,
                         "scores": [{f: r.get(f) for f in ("run_id", "score", "sha256", "status")} for r in ref]},
           "verdict": {"name": vpath.name, "sha256": _sha(vpath), "status": decision["status"],
                       "qualifying": decision["qualifying"], "chosen": decision["chosen"]},
           "written_utc": D._utc(), "note": "dev only; names and sha256s, no path"}
    _write_json(vpath.parent / E2_RESCORE_NAME, rec)
    log("E2 rescore complete (%s written, reference %d written); verdict %s, chosen %s" % (
        sum(1 for v in arms.values() for r in v if r["status"] == "written"),
        sum(1 for r in ref if r["status"] == "written"), decision["qualifying"] or "none qualifies",
        decision["chosen"]))
    return rec


def _e2_verdict_doc(verdict_path=None):
    vp = Path(verdict_path or (C.INC_DIR / "capacity" / ("%s.json" % E2_NAME)))
    v = _read_json(vp)
    if not isinstance(v, dict) or v.get("format") != E2_FORMAT or v.get("status") != "decided":
        raise BaselineError("%s is not a decided E2 verdict: test is read only after it" % vp)
    return vp, v


def e2_test_read(letter, verdict_path=None, run_fmt=E2_TEST_RUN):
    """The one read of E2's sealed test (amendment 2026-10-04, P10), by a
    person: for the arm the decided verdict chose (it qualified, and had the
    larger D), one kind-final spec on exam test per seed from its base run's
    weights (runs/e2test__s<k>), one submission list per experiment and the
    run_inc2_job.sh argv; the driver does not track the runs, and the scores
    stay off the platform's evidence. Everything is checked before anything
    is written. Refuses before the verdict, for an arm that is not the
    verdict's choice, once a read was prepared (or a run, attempt or test
    score exists), and when the base weights no longer hash as the verdict
    read them. Returns {exp: record}."""
    if letter not in RC.E2_ARMS:
        raise BaselineError("--e2 %r is not one of E2's arms %s" % (letter, ", ".join(sorted(RC.E2_ARMS, reverse=True))))
    vp, v = _e2_verdict_doc(verdict_path)
    vsha = _sha(vp)
    if letter not in (v.get("qualifying") or []):
        raise BaselineError("E2-%s did not qualify (qualifying: %s): its test is not read (pre-registered)"
                            % (letter, v.get("qualifying") or "none"))
    if v.get("chosen") != letter:
        raise BaselineError("E2-%s qualified but is not the verdict's choice (E2-%s, the larger D): only the chosen "
                            "arm's sealed test is read (pre-registered)" % (letter, v.get("chosen")))
    inputs = ((v.get("arms") or {}).get(letter) or {}).get("inputs") or []
    if not inputs:
        raise BaselineError("%s names no input of E2-%s" % (vp, letter))
    plan = []
    for inp in inputs:
        exp = inp.get("exp")
        defn = _e2_defn(exp, letter)
        s = int(defn["e2"]["seed"])
        root = C.INC_DIR / exp
        prev = _read_json(root / E2_TEST_RECORD)
        if prev is not None or (root / E2_TEST_RECORD).exists():
            prev = prev if isinstance(prev, dict) else {}
            raise BaselineError("%s's test read was prepared at %s under verdict %s%s; it is prepared once per arm; its "
                                "argv: %s" % (exp, prev.get("written_utc"), str(prev.get("verdict_sha256"))[:12],
                                              " (the verdict changed since)" if prev.get("verdict_sha256") != vsha
                                              else "", prev.get("argv")))
        rid = run_fmt % s
        out_d = root / "runs" / rid
        held = [n for n in ("run.json", "attempt.json", "scores/test.json", "spec.json") if (out_d / n).exists()]
        if held:
            raise BaselineError("%s/runs/%s already holds %s: test is read once per arm" % (exp, rid, ", ".join(held)))
        w = root / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
        wsha = _sha(w)
        if wsha is None or wsha != inp.get("weights_sha256"):
            raise BaselineError("%s's base weights hash to %s; the verdict was decided on %s"
                                % (exp, str(wsha)[:12], str(inp.get("weights_sha256"))[:12]))
        spec = {"exp": exp, "run_id": rid, "kind": "final", "init": str(w.resolve()), "exams": ["test"],
                "out_dir": str(out_d)}
        try:
            T.validate_spec(spec, out_d / "spec.json")
        except T.RunError as e:
            raise BaselineError("the v2 executor refuses the spec: %s" % e)
        plan.append((exp, s, root, out_d, spec, wsha))
    out = {}
    for exp, s, root, out_d, spec, wsha in plan:
        D._write_json(out_d / "spec.json", spec)
        lst = root / E2_TEST_LIST
        lst.parent.mkdir(parents=True, exist_ok=True)
        tmp = lst.with_name(".%s.%d.tmp" % (lst.name, os.getpid()))
        tmp.write_text("%s\n" % (out_d / "spec.json"))
        os.replace(tmp, lst)
        (root / "logs").mkdir(parents=True, exist_ok=True)      # Slurm opens --output before the job starts
        argv = ["sbatch", "--parsable", "--array=0-0", "--job-name=inc_%s_e2test" % exp,
                "--output=%s" % (root / "logs" / "%x_%A_%a.out"), str(job_script_path()), str(lst), exp]
        rec = {"format": E2_TEST_FORMAT, "exp": exp, "arm": letter, "seed": s, "verdict": str(vp),
               "verdict_sha256": vsha, "verdict_diff": ((v.get("arms") or {}).get(letter) or {}).get("diff"),
               "weights_sha256": wsha, "specs": [str(out_d / "spec.json")], "list": str(lst), "argv": argv,
               "written_utc": D._utc(),
               "note": "the one read of E2's sealed test, for the chosen arm, after the dev verdict; a person submits "
                       "the argv; the scores stay off the platform's evidence"}
        _write_json(root / E2_TEST_RECORD, rec)
        out[exp] = rec
    return out


def e2_test_report(letter, verdict_path=None, out_dir=None, testing_ok=False, reference=E2_REFERENCE):
    """The chosen arm's test read, for people: 12-class (species_map50_95)
    and agnostic test, mean +- sd over its seeds, against the reference's
    final test files of the same seeds, with the gap to 0.90; pending while
    a score is missing. Writes capacity/e2_test_<letter>.{json,md} (never on
    the platform's evidence)."""
    if letter not in RC.E2_ARMS:
        raise BaselineError("--e2 %r is not one of E2's arms" % (letter,))
    vp, v = _e2_verdict_doc(verdict_path)
    if v.get("chosen") != letter:
        raise BaselineError("E2-%s is not the verdict's choice (%s): its test was not read" % (letter, v.get("chosen")))
    inputs = ((v.get("arms") or {}).get(letter) or {}).get("inputs") or []
    missing, arm_rows, ref_rows = [], [], []

    def read(path, what):
        d = _read_json(path)
        if d is None:
            missing.append(what)
            return None
        if not isinstance(d, dict) or d.get("exam") != "test":
            raise BaselineError("%s is not a test score" % path)
        if d.get("production") is not True and not testing_ok:
            raise BaselineError("%s is a test-mode score: the report reads production scores only" % path)
        return d
    for inp in inputs:
        exp = inp.get("exp")
        s = int(_e2_defn(exp, letter)["e2"]["seed"])
        d = read(C.INC_DIR / exp / "runs" / (E2_TEST_RUN % s) / "scores" / "test.json", "%s/%s" % (exp, E2_TEST_RUN % s))
        r = read(C.INC_DIR / reference / "runs" / _final_id(s) / "scores" / "test.json",
                 "%s/%s" % (reference, _final_id(s)))
        if d is not None:
            arm_rows.append(d)
        if r is not None:
            ref_rows.append(r)

    def agg(rows):
        tw = _e2_mean([x.get("species_map50_95", x.get("map50_95")) for x in rows])
        ag = _e2_mean([x.get("agnostic_map50_95") for x in rows])
        return {"twelve": tw, "agnostic": ag}
    a, r = agg(arm_rows), agg(ref_rows)
    status = "pending" if missing else "complete"
    am, rm = a["twelve"]["mean"], r["twelve"]["mean"]
    rep = {"format": E2_TEST_REPORT_FORMAT, "status": status, "arm": letter, "verdict_sha256": _sha(vp),
           "missing": missing, "arm_scores": a, "reference": {"exp": reference, "scores": r},
           "d_test": {"twelve": (am - rm) if None not in (am, rm) else None,
                      "agnostic": (a["agnostic"]["mean"] - r["agnostic"]["mean"])
                      if None not in (a["agnostic"]["mean"], r["agnostic"]["mean"]) else None},
           "target_test_map50_95": TARGET_TEST,
           "gap_to_target": {"arm": (TARGET_TEST - am) if am is not None else None,
                             "reference": (TARGET_TEST - rm) if rm is not None else None},
           "testing_allowed": bool(testing_ok), "written_utc": D._utc(),
           "note": "the one read of E2's sealed test, for people; never the platform's evidence"}
    d = Path(out_dir) if out_dir else C.INC_DIR / "capacity"
    _write_json(d / ("e2_test_%s.json" % letter), rep)

    def f(x):
        return "-" if x is None else "%.4f" % x
    md = ["# E2-%s: the sealed test (one read, after the dev verdict)" % letter, "",
          "| | 12-class mean +- sd | agnostic mean +- sd | n |", "|---|---|---|---|",
          "| E2-%s | %s +- %s | %s +- %s | %d |" % (letter, f(am), f(a["twelve"]["sd"]), f(a["agnostic"]["mean"]),
                                                 f(a["agnostic"]["sd"]), a["twelve"]["n"]),
          "| %s | %s +- %s | %s +- %s | %d |" % (reference, f(rm), f(r["twelve"]["sd"]), f(r["agnostic"]["mean"]),
                                               f(r["agnostic"]["sd"]), r["twelve"]["n"]), "",
          "Gap to %.2f: E2-%s %s, %s %s. Status: %s%s." % (TARGET_TEST, letter, f(rep["gap_to_target"]["arm"]),
                                                         reference, f(rep["gap_to_target"]["reference"]), status,
                                                         " (missing %s)" % ", ".join(missing) if missing else "")]
    p = d / ("e2_test_%s.md" % letter)
    tmp = p.with_name(".%s.tmp" % p.name)
    tmp.write_text("\n".join(md) + "\n")
    os.replace(tmp, p)
    log("E2-%s test: %s (12-class %s vs %s; gap to %.2f %s)" % (letter, status, f(am), f(rm), TARGET_TEST,
                                                                f(rep["gap_to_target"]["arm"])))
    return rep


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Splits v2 baselines: B_v2, capacity arms, canary, B0 u tsw, "
                                             "milestones.")
    ap.add_argument("command", choices=("build", "canary-verdict", "capacity-verdict", "secondary", "estimate",
                                        "rescore-native", "native-verdict", "rescore-agnostic", "agnostic-verdict",
                                        "e1-test-read", "rescore-e2", "e2-verdict", "e2-test-read", "e2-test-report"))
    ap.add_argument("--exp", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--union", default=None, help="build: comma-separated manifests to merge (B0 u tsw)")
    ap.add_argument("--seeds", default=None, help="comma-separated (default by role: b_v2 and milestone 0-4, "
                                                  "canary 0, else 0-2)")
    ap.add_argument("--arm", default=None, choices=RC.ARM_IDS, help="default %s" % RC.DEFAULT_ARM)
    ap.add_argument("--arch", default=None, help="build: the arm's detector (yolo11n|yolo11s|yolo11m), with --imgsz")
    ap.add_argument("--imgsz", type=int, default=None, help="build: the arm's training imgsz, with --arch")
    ap.add_argument("--final-exams", default=None, help="comma-separated, from %s" % ",".join(T.EXAMS))
    ap.add_argument("--role", default=None, choices=ROLES, help="default: inferred from what is built")
    ap.add_argument("--testing", action="store_true", help="needs %s=1" % TEST_ENV)
    ap.add_argument("--testing-settings", default=None)
    ap.add_argument("--no-init", action="store_true", help="build: write the definition, do not driver-init")
    ap.add_argument("--reference", default=None, help="canary-verdict: the B0 experiment (default b0_v1); "
                                                       "rescore-native, native-verdict: the reference arm's "
                                                       "experiment (default %s)" % NATIVE_REFERENCE)
    ap.add_argument("--n", default="b_v2", help="capacity-verdict: the n640 experiment")
    ap.add_argument("--arms", default=None, help="capacity-verdict: the other arms' experiments (default "
                                                  "b_v2_s640,b_v2_m640); native-verdict: the measurement arms' "
                                                  "(default %s)" % ",".join(NATIVE_ARMS))
    ap.add_argument("--record", default=None, help="capacity-verdict: measurement arms' experiments, listed record "
                                                   "only (e.g. b_v2_m832,b_v2_s1024)")
    ap.add_argument("--m", type=int, default=None, help="capacity-verdict: M (default ceil(0.10 x |base|))")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--weights", default=None, help="secondary: the chain incumbent's final.pt")
    ap.add_argument("--run-id", default="secondary__incumbent")
    ap.add_argument("--source", default=None, help="secondary: where the weights come from (recorded)")
    ap.add_argument("--n-images", type=int, default=None, help="estimate: images in the base")
    ap.add_argument("--e2", default=None, choices=sorted(RC.E2_ARMS),
                    help="build: this experiment's E2 arm (W: b_v2_m640's cold recipe, S: x1b; docs/CONTINUOUS_LOOP.md, "
                         "Amendment 2026-10-04); e2-test-read, e2-test-report: the arm read")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.command in ("rescore-native", "native-verdict", "rescore-agnostic", "agnostic-verdict", "rescore-e2",
                     "e2-verdict"):
        # as the locked scorer's and the native scorer's CLIs: nothing is installed or fetched (before Ultralytics
        # is first imported, which this module's imports do not do)
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
    try:
        if a.command == "build":
            if not a.exp:
                raise BaselineError("build needs --exp")
            testing = P.testing_arg(a.testing, a.testing_settings)
            build(a.exp, manifest=a.manifest, union=a.union.split(",") if a.union else None,
                  seeds=a.seeds, arm=a.arm, arch=a.arch, imgsz=a.imgsz, role=a.role, testing=testing,
                  init=not a.no_init, quiet=a.quiet, final_exams=a.final_exams.split(",") if a.final_exams else None,
                  e2=a.e2)
        elif a.command in ("rescore-e2", "e2-verdict"):
            if a.exp:
                raise BaselineError("E2's six experiments are pre-registered (inc2.recipes.E2_EXPS); --exp is not "
                                    "taken")
            if a.command == "rescore-e2":
                rescore_e2(out_dir=a.out_dir)
            else:
                e2_verdict(out_dir=a.out_dir)
        elif a.command in ("e2-test-read", "e2-test-report"):
            if not a.e2:
                raise BaselineError("%s needs --e2 (the arm E2's verdict chose: W or S)" % a.command)
            if a.command == "e2-test-read":
                res = e2_test_read(a.e2)
                print(json.dumps({e: r["argv"] for e, r in res.items()}))
            else:
                e2_test_report(a.e2, out_dir=a.out_dir)
        elif a.command == "canary-verdict":
            canary_verdict(a.exp or "canary_v2", a.reference or "b0_v1")
        elif a.command == "capacity-verdict":
            capacity_verdict(a.n, [x for x in (a.arms or "b_v2_s640,b_v2_m640").split(",") if x], m=a.m,
                             out_dir=a.out_dir, record_exps=[x for x in (a.record or "").split(",") if x])
        elif a.command == "rescore-native":
            if not a.exp:
                raise BaselineError("rescore-native needs --exp (a measurement arm's experiment)")
            rescore_native(a.exp, reference=a.reference or NATIVE_REFERENCE, out_dir=a.out_dir)
        elif a.command == "native-verdict":
            native_verdict([x for x in (a.arms or ",".join(NATIVE_ARMS)).split(",") if x],
                           reference=a.reference or NATIVE_REFERENCE, out_dir=a.out_dir)
        elif a.command in ("rescore-agnostic", "agnostic-verdict"):
            if not a.exp or not a.reference:
                raise BaselineError("%s needs --exp (E1-B's experiment) and --reference (E1-A's)" % a.command)
            if a.command == "rescore-agnostic":
                rescore_agnostic(a.exp, a.reference, out_dir=a.out_dir)
            else:
                e1_verdict(a.exp, a.reference, out_dir=a.out_dir)
        elif a.command == "e1-test-read":
            if not a.exp:
                raise BaselineError("e1-test-read needs --exp (an E1 arm's experiment; comma-separated for both)")
            res = e1_test_read([x for x in a.exp.split(",") if x])
            print(json.dumps({e: r["argv"] for e, r in res.items()}))
        elif a.command == "secondary":
            if not a.exp or not a.weights:
                raise BaselineError("secondary needs --exp and --weights")
            res = secondary(a.exp, a.weights, run_id=a.run_id, source=a.source)
            print(json.dumps(res["argv"]))
        else:
            if a.n_images is None:
                raise BaselineError("estimate needs --n-images")
            seeds = P.check_seeds(a.seeds or ",".join(str(s) for s in DEFAULT_SEEDS))
            aid = arm_arg(a.arm, a.arch, a.imgsz)
            exams = NO_TEST_FINAL_EXAMS if aid in RC.MEASURE_ARMS else FINAL_EXAMS
            print(json.dumps(RC.baseline_cost(a.n_images, seeds, list(exams), aid), indent=1, sort_keys=True))
    except (BaselineError, P.PilotError, D.DriverError, RC.RecipeError) as e:
        print("[inc2.baseline] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
