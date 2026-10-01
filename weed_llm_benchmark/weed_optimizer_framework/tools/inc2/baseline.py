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


# ------------------------------------------------------------------- build
def build_definition(exp, manifest=None, union=None, seeds=None, arm=None, final_exams=None,
                     role=None, testing=False, extra=None, arch=None, imgsz=None):
    """Check every input and write the manifest copy and the summary; returns
    (definition, summary). Nothing is written before every check passed.
    role None infers it (module docstring)."""
    if role is not None and role not in ROLES:
        raise BaselineError("role %r not in %s" % (role, list(ROLES)))
    if (manifest is None) == (not union):
        raise BaselineError("give exactly one of --manifest and --union")
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
    drops = None
    if union:
        srcs = [Path(os.path.abspath(str(p))) for p in union]
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
        name = sanitise(src.stem)
        dst = paths.manifests / ("%s.jsonl" % name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if C.sha256_file(dst) != sha:
            raise BaselineError("copy of %s does not hash like the manifest that was checked" % src)
        source = {"source_manifest": str(src), "source_locked": locked_name(sha, lock)}
    cost = RC.baseline_cost(len(rows), seeds, exams, arm_rec["id"])
    defn = {"exp": exp, "type": "baseline", "builder": BUILDER, "role": role, "testing": testing,
            "seeds": list(seeds), "decision_exam": D.DECISION_EXAM, "final_exams": exams,
            "base": dict(P._entry(name, dst, rows, sha, recipe=RC.cold(arm_rec["id"])), **source),
            "cost_estimate": cost, "research_only": research_only_record(rows),
            "splits_v2": {"lock": lock["lock"], "lock_sha256": lock["lock_sha256"],
                          "nevertrain_sha256": lock["index_sha256"]}}
    if drops is not None:
        defn["variant_drops"] = drops
    defn.update(RC.stamp(arm_rec))
    if extra:
        defn.update(extra)
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
               "splits_v2": lock, "cold_recipe": RC.cold(arm_rec["id"]),
               "warmup": {"base": P.effective_warmup(RC.cold(arm_rec["id"]), len(rows))},
               "cost_estimate": cost, "research_only": defn["research_only"], "code": code}
    if drops is not None:
        summary["variant_drops"] = drops
    D._write_json(paths.root / BUILD_SUMMARY, summary)
    return defn, summary


def build(exp, manifest=None, union=None, seeds=None, arm=None, final_exams=None, role=None,
          testing=False, backend=None, init=True, quiet=False, extra=None, arch=None, imgsz=None):
    """build_definition, then driver init through run_inc2_job.sh. Returns
    (summary, definition, init result or None)."""
    defn, summary = build_definition(exp, manifest=manifest, union=union, seeds=seeds, arm=arm,
                                     final_exams=final_exams, role=role, testing=testing, extra=extra,
                                     arch=arch, imgsz=imgsz)
    log("%s (%s): baseline on %s, %d images, arm %s, seeds %s, finals %s; est. %s-%s GPU-h"
        % (exp, role, defn["base"]["manifest"], defn["base"]["n_images"], defn["arm"]["id"], defn["seeds"],
           defn["final_exams"], defn["cost_estimate"]["total_gpu_h"][0], defn["cost_estimate"]["total_gpu_h"][1]))
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


def rescore_native(exp, reference=NATIVE_REFERENCE, out_dir=None, verdict=True, batch=None, device=None,
                   resamples=NATIVE_RESAMPLES):
    """Score a measurement arm's final runs at its imgsz (dev, imageweeds)
    and the reference's on dev at 640, each only when missing, then record
    the native verdict (module docstring). Returns the rescore record."""
    defn, aid, imgsz = _native_arm(exp)
    rdefn, _raid, rimgsz = _native_arm(reference, measure=False)
    exams = [e for e in SN.EXAMS if e in (defn.get("final_exams") or [])]
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

    def one(e, s, exam, size):
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

    scores = {exam: [one(exp, s, exam, imgsz) for s in seeds] for exam in exams}
    ref = {NATIVE_EXAM: [one(reference, s, NATIVE_EXAM, rimgsz) for s in common]}
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


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Splits v2 baselines: B_v2, capacity arms, canary, B0 u tsw, "
                                             "milestones.")
    ap.add_argument("command", choices=("build", "canary-verdict", "capacity-verdict", "secondary", "estimate",
                                        "rescore-native", "native-verdict"))
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
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.command in ("rescore-native", "native-verdict"):
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
                  init=not a.no_init, quiet=a.quiet, final_exams=a.final_exams.split(",") if a.final_exams else None)
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
