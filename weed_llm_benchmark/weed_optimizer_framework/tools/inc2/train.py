"""The INC executor, Protocol v3 / splits v2: one run spec in, weights/final.pt
+ scores + run.json out (docs/CONTINUOUS_LOOP.md §4.3, §5.1; docs/INCREMENTAL_PROTOCOL_RUNNER.md,
"Protocol v3").

    python -m weed_optimizer_framework.tools.inc2.train --spec INC_DIR/<exp>/runs/<run_id>/spec.json [--force]
    python -m weed_optimizer_framework.tools.inc2.train --flock-check PATH

This file is a copy of inc/train.py (pinned; never edited). Everything the v1
executor does it does the same way: the run-dir lock and its heartbeat, the
spec checks, the code-drift check, CUDA and the AMP-check checkpoint, the
manifest's label and image hashes, materialising, the Ultralytics call, the
final-epoch check, the soup, the scorer SUBPROCESS (tools.inc.scorer, the
locked v1 scorer, reading the v1 LOCK and exams/v1, unchanged), re-runs and
re-scores, and the atomic run.json. What differs from inc/train.py:

  * exams: a spec may list only dev, imageweeds and test (splits v2's
    evaluation splits); ood22 / ood23, which v2 trains on as tsw22 / tsw23,
    are refused. test stays final-only;
  * the never-train guard (guard_rows) is splits v2's, fail-closed: every
    image's dHash and its 8 flips and rotations (inc2.guard.dhash_variants)
    go to inc2.guard.GuardV2.load(<v2 LOCK>).check; a reason that is not
    known to be harmless for a training row (NEVER_TRAIN_REASONS and any
    unknown one) stops the run. base_copy (a base_v2 row is a base copy by
    definition) and the Step 1 duplicate reasons are counted, not refused.
    Independently, every variant is looked up in the v2 never-train index
    itself (inc.common.NeverTrainGuard.load(<path>), its sha256 checked
    against the v2 LOCK) at HOLDOUT_NEAR_DUP_BITS: a hit stops the run even if
    GuardV2 passed it. An image whose bytes are one of the L-5 exclusions
    (splits/v2/l5_excluded.jsonl, its sha256 checked against LOCK v2; a
    production LOCK without that record is refused) stops the run too, and
    so does one whose bytes are an L-8 drop (the train_core rows within 6
    bits of an evaluation image under a flip or rotation,
    splits/v2/train_core_variant_drops.jsonl, likewise), under any key. A
    guard that cannot be imported or loaded stops the run;
  * check_manifest refuses the v1 and the v2 evaluation manifests, by path and
    by content (a manifest whose sha256 is a LOCK's dev / test / ood22 / ood23
    / imageweeds sha256, wherever it lies);
  * the recipe table is Protocol v3 (inc2.recipes): cold (base, union) and
    r0 / x1a / x1b (cand, null) at the arm's imgsz; freeze and lora are out of
    stream v1 (L-6). A production run with any departure is refused; a
    testing run records them (recipe_deviations);
  * the arm (L-4): exp.json's "arm" record (inc2.recipes.resolve_arm, pinned
    at build) gives the imgsz every recipe must have, and a cold run's init
    must be the arm's checkpoint (its file name, and its sha256 when the
    record pins one). An exp.json without "arm" is the continuity arm n640,
    and then only with init_weights yolo11n.pt (or none). A production cold
    run whose init is not the arm's is refused; a testing run records it
    (init_check);
  * CODE_MODULES adds every tools/inc2/*.py module present and
    tools/funnel/leak.py (the dHash variants) with the two funnel modules it
    imports (__init__.py, embed.py);
  * the scorer sidecar: after the scorer, every base, union, cand and soup
    run scored on dev runs python -m ...inc2.scorer_sidecar (a second
    subprocess that calls the pinned scorer as a library) for dev; its
    scores/dev.sidecar.json and scores/dev.images.npz carry the per-image
    AP inputs and the image-bootstrap SE of every species' dev AP that
    Protocol v3's species guard reads (inc2.gate3). The sidecar must name
    this run's weights and its recorded dev score (sha256). A failed sidecar
    fails the run at stage 'sidecar' (a re-run re-scores; nothing is
    retrained), except in a 'baseline' experiment, where no gate decision
    reads it: there it is recorded as sidecars.dev.status 'failed' with a
    warning and the run is done. run.json records it under
    'sidecars'; a re-run is a no-op only while those files still hash as
    recorded;
  * run.json says protocol 'v3', protocol_package 'inc2', and records the
    arm and the v2 guard record (reasons counted, the cross-check).

run_inc2_job.sh runs this executor (it exports INC_JOB_SCRIPT as itself, so
the pinned driver's in-job advance keeps submitting the v2 executor).

Testing is as in inc/train.py: exp.json "testing" AND INC_SCORER_TESTING=1;
the scorer and the sidecar then run in test mode on testing.device (else the
CPU) and every score is stamped TEST-.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import errno
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from . import recipes as RC

TRAIN_KINDS = ("base", "union", "cand", "null")
COLD_KINDS = ("base", "union")
RESUME_KINDS = TRAIN_KINDS + ("soup",)
KINDS = TRAIN_KINDS + ("soup", "final")
SPEC_KEYS = ("exp", "run_id", "kind", "init", "train_manifest", "soup_of", "recipe", "exams", "out_dir")
RECIPE_KEYS = ("trainer", "epochs", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
               "warmup_epochs", "warmup_bias_lr", "cos_lr", "freeze", "lora", "imgsz", "batch",
               "seed", "cache", "workers", "close_mosaic", "deterministic")
TRAINERS = ("full", "freeze", "lora")
LORA_KEYS = ("rank", "alpha")
TESTING_KEYS = ("imgsz", "batch", "device", "lock_check")
FINAL_ONLY_EXAMS = ("test",)
NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")

# Splits v2 (docs/CONTINUOUS_LOOP.md §4.1): the exams a spec may list, and
# the evaluation manifests no run trains on (v1's five and v2's three).
EXAMS = ("dev", "imageweeds", "test")
V1_EVAL_SPLITS = tuple(C.EVAL_SPLITS)              # dev, test, ood22, ood23, imageweeds
V2_EVAL_SPLITS = ("dev", "test", "imageweeds")
PROTOCOL = RC.PROTOCOL
PROTOCOL_PACKAGE = RC.PROTOCOL_PACKAGE

# The v2 guard (module docstring). A GuardV2 reason in NEVER_TRAIN_REASONS, or
# any reason not in HARMLESS_REASONS, refuses the run.
NEVER_TRAIN_REASONS = ("unhashable", "near_eval_v2", "near_eval_variant", "near_eval_embed")
HARMLESS_REASONS = ("base_copy", "exact_dup", "near_dup_intake")
GUARD_MODULE = "guard"                              # inc2.guard (splits v2 group)
# L-5 (docs/CONTINUOUS_LOOP.md §2.6): base_v2 drops the cwp10 and vanpe images
# of base B outright; splits v2 lists them in l5_excluded.jsonl (sha256 in
# LOCK v2). The executor refuses those image bytes in any v2 training run, so
# a re-listing under another key cannot bring them back.
L5_NAME = "l5_excluded.jsonl"
L5_REASON = "l5_excluded"
# L-8 (docs/CONTINUOUS_LOOP.md §2.6): train_core rows within 6 bits of an
# evaluation image under a flip or rotation leave base_v2 and train_core;
# splits v2 lists them (sha256 in LOCK v2). Refused by their bytes, as L-5.
VARIANT_DROPS_NAME = "train_core_variant_drops.jsonl"
VARIANT_DROPS_LOCK_KEY = "train_core_variant_drops_sha256"
VARIANT_DROP_REASON = "train_core_variant_drop"

# The scorer sidecar (module docstring): which runs get one, on which exam.
SIDECAR_KINDS = ("base", "union", "cand", "soup")
SIDECAR_EXAM = "dev"

TRAIN_NAME = "train"
FINAL_NAME = "final.pt"
RUN_JSON = "run.json"
ATTEMPT_JSON = "attempt.json"
LOCK_NAME = ".executor.lock"
OWNER_NAME = ".executor.owner"
TRASH_PREFIX = ".trash-"
VAL_SUBSET = "val_subset"
VAL_SUBSET_SIZE = 4
HASH_WORKERS = 8
ALLOW_DRIFT_ENV = "INC_ALLOW_DRIFT"
EXIT_DONE, EXIT_FAILED, EXIT_BUSY = 0, 1, 3

# The run-dir lock (module docstring); read at call time, so tests can shorten them.
LOCK_HEARTBEAT_SECONDS = 30
LOCK_STALE_SECONDS = 300
LOCK_WAIT_SECONDS = LOCK_STALE_SECONDS + 60
LOCK_POLL_SECONDS = 10
FLOCK_BUSY = frozenset((errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES))
FLOCK_UNSUPPORTED = frozenset((errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL,
                               getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)))

# RAM cache: Ultralytics' own estimate (check_cache_ram: 30 sampled images at
# imgsz, +50%) must fit in this share of the job's memory limit.
CACHE_RAM_SHARE = 0.6
CACHE_SAMPLE = 30
CACHE_SAFETY = 0.5

# Ultralytics' AMP check weights: safe_download's own minimum size.
AMP_MIN_BYTES = 100000
AMP_WEIGHTS_RE = re.compile(r"""YOLO\(\s*["']([^"']+\.pt)["']\s*\)""")

# Filesystems whose every locker runs on one node (flock_coherence).
LOCAL_FS = frozenset(("ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "ramfs", "zfs", "f2fs", "overlay",
                      "apfs", "hfs", "devtmpfs"))

# Ultralytics' tolerance in verify_image_label: a label it would drop the
# image for is refused here, before training, instead.
LABEL_MAX, LABEL_MIN = 1.01, -0.01

# Modules a run imports whose drift from the git-tracked copy matters; every
# tools/inc2/*.py present is added at run time (code_modules()).
CODE_MODULES = ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/lora.py", "tools/inc/scorer.py",
                "tools/cwd12_species.py", "tools/near_dup.py", "tools/mega_trainer.py", "tools/funnel/__init__.py",
                "tools/funnel/embed.py", "tools/funnel/leak.py")

# Fields carried into a resumed attempt's run.json from the attempt that trained.
CARRY_OVER = ("init", "init_sha256", "train_manifest", "train_manifest_sha256", "n_train_images",
              "n_train_boxes", "train_class_counts", "train_sources", "duplicate_label_rows",
              "duplicate_images", "dhash0_collisions", "guard", "materialised", "dataset_check",
              "train_dir", "train_kwargs", "trainable_params", "total_params", "weights_epoch", "lora",
              "cache", "amp_check_weights", "protocol_recipe", "recipe_deviations", "soup", "soup_of",
              "soup_sha256", "train_seconds", "weights_source", "arm", "init_check", "recipe_name")


class RunError(RuntimeError):
    """The run cannot go on; stage says where it stopped."""

    def __init__(self, stage, msg):
        super().__init__(msg)
        self.stage = stage


class LockError(RuntimeError):
    """The run dir's lock could not be taken for a reason other than another executor."""


class LockLost(RunError):
    """Another executor took the run dir over (this attempt's owner file was judged stale)."""

    def __init__(self, msg):
        super().__init__("lock", msg)


class Terminated(BaseException):
    """SIGTERM (Slurm time limit, scancel): record the failure, then exit."""


# SIGTERM while the attempt is being recorded (cleanup, run.json) is noted, not raised.
_SIG = {"closing": False, "late": None}


def log(msg):
    print("[inc2.train] %s" % msg, flush=True)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------ the spec
def validate_spec(spec, spec_path):
    """Raise RunError('spec', ...) unless spec follows the runner doc exactly."""
    def bad(msg):
        raise RunError("spec", "%s: %s" % (spec_path, msg))

    if not isinstance(spec, dict):
        bad("a spec is a JSON object")
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    if unknown:
        bad("unknown key(s) %s; a spec holds only %s" % (unknown, list(SPEC_KEYS)))
    for k in ("exp", "run_id", "kind", "exams", "out_dir"):
        if spec.get(k) in (None, "", []):
            bad("%r is required" % k)
    for k in ("exp", "run_id"):
        if not isinstance(spec[k], str) or not NAME_RE.fullmatch(spec[k]):
            bad("%s %r is not a path-safe name" % (k, spec[k]))
    kind = spec["kind"]
    if kind not in KINDS:
        bad("kind %r is not one of %s" % (kind, list(KINDS)))
    exams = spec["exams"]
    if not isinstance(exams, list) or not all(isinstance(e, str) and NAME_RE.fullmatch(e) for e in exams):
        bad("exams must be a list of split names, got %r" % (exams,))
    if len(set(exams)) != len(exams):
        bad("exams lists a split twice: %s" % exams)
    foreign = [e for e in exams if e not in EXAMS]
    if foreign:
        bad("exams %s are not splits v2 evaluation splits %s (ood22 and ood23 are training data in v2, "
            "as tsw22 and tsw23)" % (foreign, list(EXAMS)))
    final_only = sorted(set(exams) & set(FINAL_ONLY_EXAMS))
    if final_only and kind != "final":
        bad("a %s run may not be scored on %s: the sealed test is read only by kind 'final', "
            "at anchor points" % (kind, final_only))

    spec_dir = Path(spec_path).resolve().parent
    if not isinstance(spec["out_dir"], str) or Path(spec["out_dir"]).resolve() != spec_dir:
        bad("out_dir %r is not the spec's own directory %s" % (spec["out_dir"], spec_dir))
    want = (C.INC_DIR / spec["exp"] / "runs" / spec["run_id"]).resolve()
    if spec_dir != want:
        bad("the run dir %s is not INC_DIR/<exp>/runs/<run_id> = %s (INC_DIR %s)"
            % (spec_dir, want, C.INC_DIR))

    present = {k for k in ("init", "train_manifest", "soup_of") if spec.get(k) not in (None, "", [])}
    if kind in TRAIN_KINDS:
        need, forbid = {"init", "train_manifest"}, {"soup_of"}
    elif kind == "soup":
        # init is allowed and ignored: the runner doc's spec lists it for every kind,
        # and the driver writes cand[0]'s weights there. soup_of is what is averaged.
        need, forbid = {"soup_of"}, {"train_manifest"}
    else:
        need, forbid = {"init"}, {"train_manifest", "soup_of"}
    if need - present:
        bad("kind %s needs %s" % (kind, sorted(need - present)))
    if forbid & present:
        bad("kind %s takes no %s" % (kind, sorted(forbid & present)))
    for k in ("init", "train_manifest"):
        if k in present and not isinstance(spec[k], str):
            bad("%s must be a path string" % k)
    if kind == "soup":
        members = spec["soup_of"]
        if not isinstance(members, list) or len(members) < 2 or not all(isinstance(m, str) and m for m in members):
            bad("soup_of must list at least two checkpoint paths")
    if kind in TRAIN_KINDS:
        validate_recipe(spec.get("recipe"), bad)
    elif spec.get("recipe") is not None and not isinstance(spec["recipe"], dict):
        bad("recipe must be an object or null")
    return spec


def validate_recipe(r, bad):
    if not isinstance(r, dict):
        bad("a training run needs a recipe object")
    unknown = sorted(set(r) - set(RECIPE_KEYS))
    if unknown:
        bad("unknown recipe key(s) %s; a recipe holds only %s" % (unknown, list(RECIPE_KEYS)))
    missing = [k for k in RECIPE_KEYS if k not in r]
    if missing:
        bad("recipe is missing %s: every setting is explicit, none falls back to an Ultralytics "
            "default" % missing)
    t = r["trainer"]
    if t not in TRAINERS:
        bad("recipe.trainer %r is not one of %s" % (t, list(TRAINERS)))
    if (t == "lora") != (r["lora"] is not None):
        bad("recipe.trainer is 'lora' iff recipe.lora is set (trainer %r, lora %r)" % (t, r["lora"]))
    if (t == "freeze") != (r["freeze"] is not None):
        bad("recipe.trainer is 'freeze' iff recipe.freeze is set (trainer %r, freeze %r)" % (t, r["freeze"]))
    if t == "freeze" and (not _is_int(r["freeze"]) or r["freeze"] < 1):
        bad("recipe.freeze must be an int >= 1 (layers 0..n-1 frozen), got %r" % (r["freeze"],))
    if t == "lora":
        lo = r["lora"]
        if not isinstance(lo, dict) or sorted(lo) != sorted(LORA_KEYS):
            bad("recipe.lora must be {'rank': int, 'alpha': number}, got %r" % (lo,))
        if not _is_int(lo["rank"]) or lo["rank"] < 1 or not _is_num(lo["alpha"]) or lo["alpha"] <= 0:
            bad("recipe.lora rank must be an int >= 1 and alpha > 0, got %r" % (lo,))
    if not isinstance(r["optimizer"], str) or not r["optimizer"] or r["optimizer"].lower() == "auto":
        bad("recipe.optimizer must name an optimizer explicitly ('auto' ignores lr0), got %r"
            % (r["optimizer"],))
    for k, lo in (("epochs", 1), ("imgsz", 32), ("batch", 1), ("seed", 0), ("workers", 0),
                  ("close_mosaic", 0)):
        if not _is_int(r[k]) or r[k] < lo:
            bad("recipe.%s must be an int >= %d, got %r" % (k, lo, r[k]))
    for k in ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_bias_lr"):
        if not _is_num(r[k]) or r[k] < 0:
            bad("recipe.%s must be a finite number >= 0, got %r" % (k, r[k]))
    for k in ("cos_lr", "deterministic"):
        if not isinstance(r[k], bool):
            bad("recipe.%s must be true or false, got %r" % (k, r[k]))
    if r["cache"] not in (False, None, "ram", "disk"):
        bad("recipe.cache must be 'ram', 'disk', false or null, got %r" % (r["cache"],))


def protocol_deviations(kind, r, arm=RC.DEFAULT_ARM, recipe_name=None, n_images=None):
    """How recipe r departs from Protocol v3's table (inc2.recipes) for a run
    of this kind on this arm ([] when it does not); with recipe_name
    'cold_budget' (E1), how a base run departs from cold_budget(arm,
    n_images)."""
    return RC.deviations(kind, r, arm, recipe_name=recipe_name, n_images=n_images)


def experiment_budget(exp, arm):
    """(recipe_name, n_images, problems) of the equal-compute recipe exp.json
    names (E1, inc2.recipes.cold_budget): (None, None, []) when it names
    none. Only a 'baseline' experiment may name it; its base's n_images
    (the driver pins it against the manifest's sha256) gives the recipe, and
    its budget record must be the pre-registered one (inc2.recipes
    constants), so a definition cannot carry a budget of its own."""
    data = _read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(data, dict):
        raise RunError("recipe", "cannot read %s" % (C.INC_DIR / exp / "exp.json"))
    name = data.get("recipe_name")
    if name in (None, RC.COLD_NAME):
        return None, None, []
    if name != RC.BUDGET_NAME:
        return name, None, ["exp.json's recipe_name %r is not one inc2.recipes knows" % (name,)]
    probs = []
    if data.get("type") != "baseline":
        probs.append("%s is pre-registered for baseline experiments, not a %r one" % (name, data.get("type")))
    n = (data.get("base") or {}).get("n_images")
    if not _is_int(n) or n < 1:
        return name, None, probs + ["exp.json's base records no image count for %s" % name]
    probs += RC.check_budget_record(data.get("budget"), arm["id"], n)
    probs += e1_problems(data)
    return name, n, probs


def e1_problems(data):
    """How an exp.json naming cold_budget fails to be an E1 arm: its e1
    record names arm A or B, its base manifest (by sha256) is that arm of
    the complete splits/v3/summary.json (inc2.base3.e1_arm_of), and that
    summary still hashes as the e1 record says. [] when it is one."""
    from . import base3 as B3
    e1 = data.get("e1")
    if not isinstance(e1, dict) or e1.get("arm") not in ("A", "B"):
        return ["exp.json names %s without an E1 record (e1.arm A or B)" % RC.BUDGET_NAME]
    probs = []
    msha = (data.get("base") or {}).get("manifest_sha256")
    k, _summ = B3.e1_arm_of(msha) if msha else (None, None)
    if k is None:
        probs.append("its base manifest %s is not one of the two manifests a complete splits/v3/summary.json "
                     "records" % str(msha)[:12])
    elif k != e1["arm"]:
        probs.append("its base manifest is E1's arm %s, exp.json says %s" % (k, e1["arm"]))
    sp = B3.out_dir() / B3.SUMMARY
    cur = C.sha256_file(sp) if sp.is_file() else None
    if e1.get("summary_sha256") != cur:
        probs.append("exp.json was built from splits v3 summary %s; %s is now %s"
                     % (str(e1.get("summary_sha256"))[:12], sp, str(cur)[:12]))
    return probs


def testing_settings(exp):
    """(settings dict, or None for a production run; warnings). exp.json's
    "testing" flag AND INC_SCORER_TESTING=1 are both needed for test mode."""
    exp_json = C.INC_DIR / exp / "exp.json"
    warnings = []
    flag = None
    if exp_json.exists():
        data = _read_json(exp_json)
        if not isinstance(data, dict):
            raise RunError("testing", "cannot read %s" % exp_json)
        flag = data.get("testing")
    env = os.environ.get(S.TEST_ENV) == "1"
    if not flag:
        if env:
            warnings.append("%s=1 is set but %s is not a testing experiment: the scorer runs "
                            "without it" % (S.TEST_ENV, exp))
        return None, warnings
    if not env:
        raise RunError("testing", "experiment %s is marked testing in %s but %s=1 is not set; a "
                       "testing experiment only runs inside a test" % (exp, exp_json, S.TEST_ENV))
    settings = {} if flag is True else flag
    if not isinstance(settings, dict) or set(settings) - set(TESTING_KEYS):
        raise RunError("testing", "exp.json 'testing' must be true or an object with keys %s, got %r"
                       % (list(TESTING_KEYS), flag))
    return dict(settings), warnings


def experiment_arm(exp):
    """(arm record, warnings) of the experiment: exp.json's "arm"
    (inc2.recipes.check_arm_record), or the continuity arm n640 when there is
    none, provided exp.json's init_weights is absent or n640's checkpoint."""
    data = _read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(data, dict):
        raise RunError("recipe", "cannot read %s" % (C.INC_DIR / exp / "exp.json"))
    rec = data.get("arm")
    if rec is None:
        default = RC.ARMS[RC.DEFAULT_ARM]["model"]
        iw = data.get("init_weights")
        if iw not in (None, default):
            raise RunError("recipe", "exp.json has no 'arm' but init_weights %r: an experiment on another "
                           "detector than %s must pin its arm (inc2.recipes.stamp)" % (iw, default))
        return RC.resolve_arm(RC.DEFAULT_ARM, require_weights=False), [
            "exp.json pins no arm: the continuity arm %s is assumed" % RC.DEFAULT_ARM]
    try:
        RC.check_arm_record(rec)
    except RC.RecipeError as e:
        raise RunError("recipe", "exp.json's arm: %s" % e)
    if data.get("init_weights") not in (None, rec["model"]):
        raise RunError("recipe", "exp.json's init_weights %r is not its arm's checkpoint %s"
                       % (data.get("init_weights"), rec["model"]))
    return dict(rec), []


def experiment_type(exp):
    """exp.json's type ('baseline' or 'chain'), None when unreadable."""
    data = _read_json(C.INC_DIR / exp / "exp.json")
    return data.get("type") if isinstance(data, dict) else None


def experiment_research_only(exp):
    """The research-only flag of the models an experiment trains (§8, P6),
    as its exp.json records it: a stream segment's stream.research_only.models,
    a baseline's or milestone's research_only.flag. True when either is
    true; else the recorded value ("unknown" or false); "unknown" when a
    stream built the experiment (a stream block) but recorded neither, as
    its rows may be research-only (fail closed); None when exp.json records
    neither and no stream built it (nothing is invented)."""
    data = _read_json(C.INC_DIR / exp / "exp.json")
    if not isinstance(data, dict):
        return None
    vals = []
    ro = (data.get("stream") or {}).get("research_only") if isinstance(data.get("stream"), dict) else None
    if isinstance(ro, dict) and "models" in ro:
        vals.append(ro["models"])
    ro = data.get("research_only")
    if isinstance(ro, dict) and "flag" in ro:
        vals.append(ro["flag"])
    if not vals:
        return "unknown" if isinstance(data.get("stream"), dict) else None
    if any(v is True for v in vals):
        return True
    return next((v for v in vals if v is not False), False)


def init_check(kind, init_path, arm):
    """[] when a cold run's init is the arm's checkpoint (file name, and its
    sha256 when the arm pins one); else how it departs. Other kinds start from
    an incumbent and are not checked here."""
    if kind not in COLD_KINDS:
        return []
    out = []
    p = Path(init_path)
    if p.name != arm["model"]:
        out.append("init %s is not the arm's checkpoint %s" % (p.name, arm["model"]))
    want = arm.get("weights_sha256")
    if want is not None:
        got = _sha_or_none(resolve_local(init_path))
        if got != want:
            out.append("init %s hashes to %s, the arm pins %s" % (init_path, (got or "none")[:12], want[:12]))
    return out


def resolve_local(path):
    p = Path(path)
    return (p if p.is_absolute() else C.REPO / p).resolve()


# ------------------------------------------------------------------ the code
def package_dir():
    """The weed_optimizer_framework dir this module was imported from."""
    return Path(__file__).resolve().parents[2]


def code_modules(running=None):
    """CODE_MODULES plus every tools/inc2/*.py of the running copy, sorted."""
    running = Path(running or package_dir())
    inc2 = sorted("tools/inc2/%s" % p.name for p in (running / "tools" / "inc2").glob("*.py"))
    return tuple(CODE_MODULES) + tuple(inc2)


def code_provenance():
    """({module: sha256} of the running copy, [drifted modules]). The nested
    git-tracked copy is REPO/weed_llm_benchmark/weed_optimizer_framework; when
    that is not what runs, every code_modules() file must match it."""
    running = package_dir()
    nested = C.REPO / "weed_llm_benchmark" / "weed_optimizer_framework"
    code, drift = {}, []
    compare = nested.is_dir() and nested.resolve() != running
    for m in code_modules(running):
        p = running / m
        code[m] = C.sha256_file(p) if p.is_file() else None
        if compare and code[m] is not None:
            q = nested / m
            if not q.is_file() or C.sha256_file(q) != code[m]:
                drift.append(m)
    return {"package_dir": str(running), "nested_dir": str(nested) if compare else None,
            "modules": code}, drift


# ------------------------------------------------------------- the device
def amp_check_weights(cwd=None, weights_dir=None):
    """(name, path or None, places looked) for the checkpoint Ultralytics'
    check_amp loads on a CUDA device, the name read from the installed
    release's own source. Looked up where attempt_download_asset looks: the
    working directory, then SETTINGS['weights_dir']; path None means
    check_amp would download it. name None when the source does not name
    exactly one checkpoint."""
    import inspect
    from ultralytics.utils import SETTINGS, checks
    try:
        names = sorted(set(AMP_WEIGHTS_RE.findall(inspect.getsource(checks.check_amp))))
    except (OSError, TypeError, AttributeError):
        names = []
    if len(names) != 1:
        return None, None, names
    name = names[0]
    places = [Path(cwd or os.getcwd()) / name, Path(weights_dir or SETTINGS["weights_dir"]) / name]
    for p in places:
        if p.exists():
            return name, p, places
    return name, None, places


def require_amp_weights(cwd=None, weights_dir=None):
    """The record of the AMP-check checkpoint a CUDA training run will load;
    RunError('device') when it is absent (Ultralytics would download it in
    place while other array tasks load it) or incomplete."""
    name, path, places = amp_check_weights(cwd, weights_dir)
    if name is None:
        return {"name": None, "note": "could not tell which checkpoint Ultralytics' check_amp loads "
                                      "(found %s); if it downloads one, parallel tasks may race" % (places,)}
    if path is None:
        raise RunError("device", "Ultralytics' AMP check on a CUDA device loads %s, which is in neither "
                       "%s: it would be downloaded, written in place while the other array tasks load "
                       "it. Put a complete copy in the working directory (the job script runs from "
                       "REPO) first; the executor downloads nothing" % (name, [str(p) for p in places]))
    size = path.stat().st_size
    if size < AMP_MIN_BYTES or not zipfile.is_zipfile(str(path)):
        raise RunError("device", "%s (the checkpoint Ultralytics' AMP check loads) is incomplete: %d "
                       "bytes, not a whole torch checkpoint; replace it with a complete copy"
                       % (path, size))
    return {"name": name, "path": str(path.resolve()), "sha256": C.sha256_file(path), "bytes": size}


# -------------------------------------------------------------- the manifest
def resolve_weights(path, what, stage):
    """An absolute path, or one relative to REPO (e.g. 'yolo11n.pt')."""
    p = Path(path)
    if not p.is_absolute():
        p = C.REPO / p
    if not p.is_file():
        raise RunError(stage, "%s not found: %s (nothing is downloaded)" % (what, p))
    return p.resolve()


def read_label_strict(path):
    """The boxes of a label file, refusing anything Ultralytics would drop the
    image for (verify_image_label) or read as a segment."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    boxes = []
    for i, ln in enumerate(text.strip().splitlines(), 1):
        if not len(ln):
            continue
        t = ln.split()
        if len(t) != 5:
            raise ValueError("%s line %d has %d fields; INC labels are boxes 'cls cx cy w h'" % (path, i, len(t)))
        try:
            v = [float(x) for x in t]
        except ValueError:
            raise ValueError("%s line %d is not numeric: %r" % (path, i, ln))
        if not all(math.isfinite(x) for x in v):
            raise ValueError("%s line %d holds a non-finite number" % (path, i))
        if v[0] != int(v[0]) or not 0 <= v[0] < C.NC:
            raise ValueError("%s line %d: class %r outside 0..%d" % (path, i, t[0], C.NC - 1))
        if max(v[1:]) > LABEL_MAX or min(v) < LABEL_MIN:
            raise ValueError("%s line %d: coordinates outside [0, 1]: %r" % (path, i, ln))
        boxes.append((int(v[0]),) + tuple(v[1:]))
    return boxes


def _hash_image(path):
    try:
        sha = C.sha256_file(path)
    except OSError:
        sha = None
    return sha, C.dhash(path)


def hash_images(paths):
    """[(sha256, dHash)] per path, on HASH_WORKERS threads. On any exception
    (SIGTERM's Terminated included) the queued hashes are cancelled rather
    than awaited, so a killed run records its failure at once."""
    if not paths:
        return []
    first = _hash_image(paths[0])     # imports common.dhash's module once, before the threads start
    ex = ThreadPoolExecutor(max_workers=max(1, min(HASH_WORKERS, len(paths))))
    try:
        rest = list(ex.map(_hash_image, paths[1:]))
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return [first] + rest


def duplicate_images(rows, hashed):
    """(exact, near): groups of rows holding the same image bytes (sha256), and
    groups of distinct images with the same dHash (distance 0), each as
    {groups, extra_rows, across_source_session, first}."""
    by_sha = {}
    for r, (sha, dh) in zip(rows, hashed):
        by_sha.setdefault(sha, []).append((r, dh))
    by_dh = {}
    for group in by_sha.values():
        r, dh = group[0]
        if dh is not None:
            by_dh.setdefault(dh, []).append(r)

    def summary(groups):
        return {"groups": len(groups), "extra_rows": sum(len(g) - 1 for g in groups),
                "across_source_session": sum(len({(r["source"], r["session"]) for r in g}) > 1 for g in groups),
                "first": [[r["key"] for r in g] for g in groups[:5]]}

    exact = [[r for r, _ in g] for g in by_sha.values() if len(g) > 1]
    near = [g for g in by_dh.values() if len(g) > 1]
    return summary(exact), summary(near)


def check_manifest(manifest):
    """Rows of the train manifest after every check but the guard; plus the
    per-image dHashes (computed alongside the sha256 check) and counts."""
    manifest = Path(manifest).resolve()
    for version, split, path in eval_manifest_paths():
        if manifest == path.resolve():
            raise RunError("manifest", "%s is the %s %s split's manifest; it is never trained on"
                           % (manifest, version, split))
    if not manifest.is_file():
        raise RunError("manifest", "train manifest not found: %s" % manifest)
    sha = C.sha256_file(manifest)
    hit = eval_manifest_shas().get(sha)
    if hit:
        raise RunError("manifest", "%s holds the same bytes as the %s manifest (sha256 %s); an evaluation "
                       "split is never trained on, wherever its copy lies" % (manifest, hit, sha[:12]))
    try:
        rows = C.read_manifest(manifest)
    except ValueError as e:
        raise RunError("manifest", "cannot parse %s: %s" % (manifest, e))
    if not rows:
        raise RunError("manifest", "%s lists no images" % manifest)
    bad = [i for i, r in enumerate(rows) if not isinstance(r, dict) or any(k not in r for k in C.MANIFEST_KEYS)]
    if bad:
        raise RunError("manifest", "%s: %d row(s) lack %s, first line %d"
                       % (manifest, len(bad), list(C.MANIFEST_KEYS), bad[0] + 1))
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys) or not all(isinstance(k, str) and NAME_RE.fullmatch(k) for k in keys):
        raise RunError("manifest", "%s: keys must be unique path-safe names" % manifest)

    counts, sources = [0] * C.NC, {}
    n_boxes = dup_rows = 0
    problems = []
    for r in rows:
        try:
            with open(r["label"], "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != r["label_sha256"]:
                    problems.append("%s: label bytes differ from the manifest" % r["key"])
                    continue
            boxes = read_label_strict(r["label"])
        except (OSError, ValueError) as e:
            problems.append("%s: %s" % (r["key"], e))
            continue
        n_boxes += len(boxes)
        dup_rows += len(boxes) - len(set(boxes))
        for b in boxes:
            counts[b[0]] += 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    if problems:
        raise RunError("manifest", "%d label problem(s) in %s: %s" % (len(problems), manifest, problems[:3]))

    paths = [r["image"] for r in rows]
    hashed = hash_images(paths)
    mism = [r["key"] for r, (sha, _) in zip(rows, hashed) if sha != r["sha256"]]
    if mism:
        raise RunError("manifest", "%d image(s) missing or differing from the manifest's sha256: %s"
                       % (len(mism), mism[:3]))
    exact, near = duplicate_images(rows, hashed)
    return rows, {p: dh for p, (_, dh) in zip(paths, hashed)}, {
        "train_manifest": str(manifest),
        "train_manifest_sha256": C.sha256_file(manifest),
        "n_train_images": len(rows),
        "n_train_boxes": n_boxes,
        "duplicate_label_rows": dup_rows,
        "duplicate_images": exact,
        "dhash0_collisions": near,
        "train_class_counts": {C.CLASS_NAMES[i]: counts[i] for i in range(C.NC)},
        "train_sources": dict(sorted(sources.items())),
    }


def v2_dir():
    """INC_DIR/splits/v2 (docs/CONTINUOUS_LOOP.md §4.2), or inc2.common's own
    SPLITS_DIR when that module defines it."""
    try:
        cm = importlib.import_module(__package__ + ".common")
        d = getattr(cm, "SPLITS_DIR", None)
        if d is not None and Path(d).name == RC.SPLITS_VERSION:
            return Path(d)
    except ImportError:
        pass
    return C.INC_DIR / "splits" / RC.SPLITS_VERSION


def v2_lock_path():
    return v2_dir() / "LOCK.json"


def v2_nevertrain_path():
    return v2_dir() / "nevertrain_dhash.json"


def eval_manifest_paths():
    """[(version, split, path)] of every evaluation manifest (v1 and v2)."""
    out = [("v1", s, C.manifest_path(s)) for s in V1_EVAL_SPLITS]
    out += [("v2", s, v2_dir() / ("%s.jsonl" % s)) for s in V2_EVAL_SPLITS]
    return out


def eval_manifest_shas():
    """{sha256: 'v1 dev' ...} of the evaluation manifests: the LOCKs' records
    (v1, and v2 when it exists) and the files themselves where they exist."""
    out = {}
    for version, lock, splits in (("v1", C.LOCK_PATH, V1_EVAL_SPLITS), ("v2", v2_lock_path(), V2_EVAL_SPLITS)):
        data = _read_json(lock)
        mans = (data or {}).get("manifests") if isinstance(data, dict) else None
        if isinstance(mans, dict):
            for s in splits:
                if isinstance(mans.get(s), str):
                    out[mans[s]] = "%s %s" % (version, s)
    for version, split, path in eval_manifest_paths():
        sha = _sha_or_none(path)
        if sha:
            out.setdefault(sha, "%s %s" % (version, split))
    return out


def guard_module():
    """inc2.guard (splits v2): GuardV2 and dhash_variants. RunError('guard')
    when it cannot be imported: a guard that is not there clears nothing."""
    try:
        return importlib.import_module("%s.%s" % (__package__, GUARD_MODULE))
    except Exception as e:      # noqa: BLE001 -- any import failure is a missing guard
        raise RunError("guard", "cannot import the splits v2 guard %s.%s (%s: %s); the never-train guard "
                       "fails closed" % (__package__, GUARD_MODULE, type(e).__name__, e))


def _variant_values(variants):
    vals = variants.values() if isinstance(variants, dict) else (variants or ())
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return None
    return out


def _variants_of(g2, path):
    try:
        return g2.dhash_variants(path)
    except Exception:           # noqa: BLE001 -- an image whose variants cannot be computed is unhashable
        return None


def variant_hashes(g2, paths):
    """{path: variants or None} on HASH_WORKERS threads (cancelled on any
    exception, as hash_images)."""
    if not paths:
        return {}
    first = _variants_of(g2, paths[0])
    ex = ThreadPoolExecutor(max_workers=max(1, min(HASH_WORKERS, len(paths))))
    try:
        rest = list(ex.map(lambda p: _variants_of(g2, p), paths[1:]))
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return dict(zip(paths, [first] + rest))


def load_l5(lock, production=True):
    """({image sha256} of the L-5 excluded images, record). LOCK v2 records
    l5_excluded.jsonl's sha256; the file must hash to it. A production LOCK
    without that record is refused (every locked v2 split set has one); a
    testing LOCK without it gives an empty set, recorded."""
    path = v2_dir() / L5_NAME
    want = lock.get("l5_excluded_sha256")
    if want is None:
        if production:
            raise RunError("guard", "LOCK v2 records no l5_excluded_sha256: the L-5 exclusions cannot be checked")
        return set(), {"path": str(path), "sha256": None, "images": 0, "note": "LOCK v2 records none (testing)"}
    got = _sha_or_none(path)
    if got != want:
        raise RunError("guard", "%s hashes to %s, LOCK v2 records %s" % (path, (got or "none")[:12], str(want)[:12]))
    shas = set()
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    shas.add(str(json.loads(ln)["sha256"]))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RunError("guard", "cannot read %s: %s" % (path, e))
    return shas, {"path": str(path), "sha256": got, "images": len(shas)}


def load_variant_drops(lock, production=True):
    """({image sha256: key} of the L-8 drops, record): splits v2's
    train_core_variant_drops.jsonl, which must hash to the sha256 LOCK v2
    records. A production LOCK without that record is refused (every LOCK
    the v2 build writes has one, empty or not); a testing LOCK without it
    gives no drops, recorded."""
    path = v2_dir() / VARIANT_DROPS_NAME
    want = lock.get(VARIANT_DROPS_LOCK_KEY)
    if want is None:
        if production:
            raise RunError("guard", "LOCK v2 records no %s: the L-8 train_core drops cannot be checked"
                           % VARIANT_DROPS_LOCK_KEY)
        return {}, {"path": str(path), "sha256": None, "images": 0, "note": "LOCK v2 records none (testing)"}
    got = _sha_or_none(path)
    if got != want:
        raise RunError("guard", "%s hashes to %s, LOCK v2 records %s" % (path, (got or "none")[:12], str(want)[:12]))
    shas = {}
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    shas[str(r["sha256"])] = str(r["key"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RunError("guard", "cannot read %s: %s" % (path, e))
    return shas, {"path": str(path), "sha256": got, "images": len(shas), "keys": sorted(shas.values())}


def load_v2_guards(production=True):
    """(inc2.guard module, GuardV2 loaded from the v2 LOCK, the v2 never-train
    index as a pinned NeverTrainGuard, record, the L-5 image sha256s, the L-8
    drops {image sha256: key}). Fails closed (RunError 'guard') when the
    LOCK, the index, the L-5 or L-8 list or the guard cannot be read, or the
    index is not the one the LOCK records; a production run also refuses a
    LOCK a testing build wrote (a synthetic world)."""
    g2 = guard_module()
    lock_path, idx_path = v2_lock_path(), v2_nevertrain_path()
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise RunError("guard", "cannot read the splits v2 LOCK %s: training on v2 needs locked splits" % lock_path)
    if production and lock.get("testing"):
        raise RunError("guard", "the splits v2 LOCK %s was written by a testing build; a production run trains "
                                "on real locked splits only" % lock_path)
    want = lock.get("nevertrain_sha256")
    got = _sha_or_none(idx_path)
    if not isinstance(want, str) or got != want:
        raise RunError("guard", "the v2 never-train index %s hashes to %s, the v2 LOCK records %s"
                       % (idx_path, (got or "none")[:12], str(want)[:12]))
    try:
        guard = g2.GuardV2.load(lock_path)
    except Exception as e:      # noqa: BLE001 -- any load failure fails closed
        raise RunError("guard", "GuardV2.load(%s) failed: %s: %s" % (lock_path, type(e).__name__, e))
    try:
        index = C.NeverTrainGuard.load(idx_path)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise RunError("guard", "cannot load the v2 never-train index %s: %s" % (idx_path, e))
    l5, l5_rec = load_l5(lock, production=production)
    drops, drops_rec = load_variant_drops(lock, production=production)
    rec = {"splits_version": RC.SPLITS_VERSION, "lock": str(lock_path), "lock_sha256": _sha_or_none(lock_path),
           "index": str(idx_path), "index_sha256": got, "index_images": index.n,
           "bits": C.HOLDOUT_NEAR_DUP_BITS, "guard_module": getattr(g2, "__file__", None),
           "guard_module_sha256": _sha_or_none(getattr(g2, "__file__", "") or ""), "l5_excluded": l5_rec,
           "train_core_variant_drops": drops_rec}
    return g2, guard, index, rec, l5, drops


def guard_verdicts(rows, dhashes, production=True, guards=None, variants=None):
    """The splits v2 never-train guard's verdict on every row, refusing
    nothing (guard_rows refuses; inc2.base3 drops the rows instead): the L-5
    and L-8 lists by image sha256, GuardV2 over the dHash and its 8 flips and
    rotations, and the index cross-check. Returns (per, record, refused,
    crosscheck): per {key: {"reasons": [...], "refused": [[path, reason,
    match]], "crosscheck": [path, split, image, bits] or None, "variants":
    [8 ints] or None}}; a row is refused by any reason in
    NEVER_TRAIN_REASONS, or any reason not in HARMLESS_REASONS, and by a
    cross-check hit. guards: load_v2_guards' tuple (loaded when None);
    variants: {path: variants} when the caller already holds them."""
    g2, guard, index, rec, l5, drops = guards or load_v2_guards(production=production)
    rec = dict(rec)
    paths = [r["image"] for r in rows]
    if variants is None:
        variants = variant_hashes(g2, paths)
    reasons, refused, crosscheck = {}, [], []
    per = {}
    for r in rows:
        v = per.setdefault(r["key"], {"reasons": [], "refused": [], "crosscheck": None, "variants": None})
        if str(r.get("sha256")) in l5:
            reasons[L5_REASON] = reasons.get(L5_REASON, 0) + 1
            item = [str(r["image"]), L5_REASON, "an L-5 excluded image (%s)" % r["key"]]
            refused.append(item)
            v["reasons"].append(L5_REASON)
            v["refused"].append(item)
        if str(r.get("sha256")) in drops:
            reasons[VARIANT_DROP_REASON] = reasons.get(VARIANT_DROP_REASON, 0) + 1
            item = [str(r["image"]), VARIANT_DROP_REASON, "the L-8 train_core drop %s, listed as %s"
                    % (drops[str(r["sha256"])], r["key"])]
            refused.append(item)
            v["reasons"].append(VARIANT_DROP_REASON)
            v["refused"].append(item)
    by_path = {}
    for r in rows:
        by_path.setdefault(r["image"], []).append(r["key"])
    for p in paths:
        dh = dhashes.get(p)
        var = variants.get(p)
        vals = _variant_values(var) if var is not None else None
        if dh is None or vals is None:
            reason, match = "unhashable", None
        else:
            try:
                reason, match = guard.check(dh, var)
            except Exception as e:  # noqa: BLE001 -- a check that cannot run clears nothing
                reason, match = "guard_error", "%s: %s" % (type(e).__name__, e)
        keys = by_path.get(p) or []
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            for k in keys:
                per[k]["reasons"].append(reason)
                per[k]["match"] = match
            if reason in NEVER_TRAIN_REASONS or reason not in HARMLESS_REASONS:
                item = [str(p), reason, str(match)[:200]]
                refused.append(item)
                for k in keys:
                    per[k]["refused"].append(item)
        if vals is not None:
            for k in keys:
                per[k]["variants"] = list(vals)
            for v in [int(dh)] + vals:
                m = index.index.find(v)
                if m is not None:
                    (split, image), bits = m
                    hit = [str(p), split, image, bits]
                    crosscheck.append(hit)
                    for k in keys:
                        per[k]["crosscheck"] = hit
                    break
    rec.update(checked=len(paths), reasons=dict(sorted(reasons.items())), refused=len(refused),
               crosscheck_hits=len(crosscheck), first_refused=refused[:10], first_crosscheck=crosscheck[:10],
               never_train_reasons=list(NEVER_TRAIN_REASONS), harmless_reasons=list(HARMLESS_REASONS))
    return per, rec, refused, crosscheck


def guard_rows(rows, dhashes, production=True):
    """The splits v2 never-train guard over every image (module docstring),
    fail closed. Returns the guard record; raises RunError('guard') with it
    attached."""
    _per, rec, refused, crosscheck = guard_verdicts(rows, dhashes, production=production)
    reasons = rec["reasons"]
    if refused or crosscheck:
        err = RunError("guard", "splits v2 never-train guard: %d image(s) refused by GuardV2 or the L-5 / L-8 "
                                "lists (%s) and %d within %d bits of a dev / test / imageweeds image under a flip or rotation "
                                "(index cross-check); first: %s %s"
                       % (len(refused), dict(sorted(reasons.items())), len(crosscheck), C.HOLDOUT_NEAR_DUP_BITS,
                          refused[:2], crosscheck[:2]))
        err.guard = rec
        raise err
    return rec


def _jpeg_without_eoi(path):
    with open(path, "rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return False
        fh.seek(-2, os.SEEK_END)
        return fh.read(2) != b"\xff\xd9"


def materialise_run(rows, data_dir):
    """common.materialise, then: no-EOI JPEGs copied instead of linked, a
    VAL_SUBSET_SIZE-image val subset of the training images (see the module
    docstring), data.yaml pointing 'val' at it. Returns (data.yaml, record)."""
    C.materialise(rows, data_dir)
    img_dir = data_dir / "images"
    copied = []
    for r in rows:
        if _jpeg_without_eoi(r["image"]):
            link = img_dir / (r["key"] + (os.path.splitext(r["image"])[1].lower() or ".jpg"))
            link.unlink()
            shutil.copyfile(r["image"], link)
            copied.append(r["key"])
    val_rows = sorted(rows, key=lambda r: r["key"])[:VAL_SUBSET_SIZE]
    vdir = data_dir / VAL_SUBSET
    (vdir / "images").mkdir(parents=True)
    (vdir / "labels").mkdir()
    for r in val_rows:
        name = r["key"] + (os.path.splitext(r["image"])[1].lower() or ".jpg")
        shutil.copyfile(img_dir / name, vdir / "images" / name)
        shutil.copyfile(data_dir / "labels" / (r["key"] + ".txt"), vdir / "labels" / (r["key"] + ".txt"))
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as fh:
        fh.write("# val: %d training images, only for the final-epoch validation Ultralytics runs even\n"
                 "# with val=False; nothing is selected on it and no evaluation image is read.\n"
                 % len(val_rows))
        fh.write("path: %s\ntrain: images\nval: %s/images\nnc: %d\nnames:\n" % (data_dir, VAL_SUBSET, C.NC))
        for n in C.CLASS_NAMES:
            fh.write("  - %s\n" % n)
    return yaml_path, {"data_dir": str(data_dir), "jpegs_copied_not_linked": len(copied),
                       "copied": copied[:20], "val_subset": [r["key"] for r in val_rows],
                       "val_subset_note": "training images; Ultralytics' final-epoch validation only"}


def check_dataset_cache(data_dir, n_rows):
    """Ultralytics' own count of what it trained on (labels.cache), which must
    be every manifest image: a corrupt image is dropped with only a warning."""
    p = data_dir / "labels.cache"
    if not p.is_file():
        return {"checked": False, "note": "no labels.cache written"}
    import numpy as np
    cache = np.load(str(p), allow_pickle=True).item()
    nf, nm, ne, nc, n = cache["results"]
    used = len(cache.get("labels") or [])
    if nc or used != n_rows:
        raise RunError("train", "Ultralytics trained on %d of the manifest's %d images (%d corrupt): %s"
                       % (used, n_rows, nc, (cache.get("msgs") or [])[:3]))
    return {"checked": True, "images": used, "found": int(nf), "missing": int(nm), "empty": int(ne),
            "corrupt": int(nc)}


# ------------------------------------------------------------------ memory
def _cgroup_memory_limit(proc_cgroup="/proc/self/cgroup", root="/sys/fs/cgroup"):
    """The smallest memory limit on this process's cgroup path (v2 memory.max,
    v1 memory.limit_in_bytes, the cgroup and each of its parents), in bytes;
    None when there is none or it cannot be read."""
    try:
        with open(proc_cgroup) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    limits = []
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrls, rel = parts
        if hid == "0" and ctrls == "":
            base, fname = Path(root), "memory.max"
        elif "memory" in ctrls.split(","):
            base, fname = Path(root) / "memory", "memory.limit_in_bytes"
        else:
            continue
        p = base / rel.strip().lstrip("/")
        while True:
            try:
                v = (p / fname).read_text().strip()
            except OSError:
                v = ""
            if v.isdigit() and int(v) < (1 << 60):
                limits.append(int(v))
            if p == base or p == p.parent:
                break
            p = p.parent
    return min(limits) if limits else None


def job_memory_limit():
    """(bytes, source): the smallest of Slurm's --mem (SLURM_MEM_PER_NODE, MB),
    --mem-per-cpu times the CPUs, and the cgroup's limit; (None, None) when
    none is known."""
    found = []
    per_node = os.environ.get("SLURM_MEM_PER_NODE", "")
    if per_node.isdigit() and int(per_node) > 0:
        found.append((int(per_node) << 20, "SLURM_MEM_PER_NODE"))
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK") or ""
    if per_cpu.isdigit() and cpus.isdigit() and int(per_cpu) > 0 and int(cpus) > 0:
        found.append((int(per_cpu) * int(cpus) << 20, "SLURM_MEM_PER_CPU x %s CPUs" % cpus))
    cg = _cgroup_memory_limit()
    if cg:
        found.append((cg, "cgroup"))
    return min(found) if found else (None, None)


def ram_cache_estimate(rows, imgsz):
    """Ultralytics' check_cache_ram estimate, from up to CACHE_SAMPLE evenly
    spaced images (their header sizes only): mean(h * w * 3 * (imgsz /
    max(h, w))^2) * n * (1 + CACHE_SAFETY). None if no image can be read."""
    from PIL import Image
    n = len(rows)
    k = min(n, CACHE_SAMPLE)
    total = seen = 0
    for i in sorted({int(j * n / k) for j in range(k)}):
        try:
            with Image.open(rows[i]["image"]) as im:
                w, h = im.size
        except Exception:       # noqa: BLE001 -- an unreadable sample is skipped, as Ultralytics does
            continue
        if w and h:
            r = imgsz / max(w, h)
            total += h * w * 3 * r * r
            seen += 1
    if not seen:
        return None
    return int(total * n / seen * (1 + CACHE_SAFETY))


def choose_cache(recipe, rows, rec):
    """The cache setting to train with: the recipe's, except that 'ram' falls
    back to no cache when its estimate exceeds CACHE_RAM_SHARE of the job's
    memory limit. Recorded in rec['cache']."""
    want = recipe["cache"] or False
    info = {"requested": want, "used": want}
    if want == "ram":
        est = ram_cache_estimate(rows, recipe["imgsz"])
        limit, source = job_memory_limit()
        info.update(estimate_bytes=est, limit_bytes=limit, limit_source=source, share=CACHE_RAM_SHARE)
        if est and limit and est > CACHE_RAM_SHARE * limit:
            info["used"] = False
            rec["warnings"].append(
                "cache 'ram' needs about %.1f GB for %d images at imgsz %d, more than %d%% of this job's "
                "%.1f GB (%s): training without a cache instead (the same pixels, read each epoch)"
                % (est / 2 ** 30, len(rows), recipe["imgsz"], round(100 * CACHE_RAM_SHARE),
                   limit / 2 ** 30, source))
    rec["cache"] = info
    return info["used"]


# ------------------------------------------------------------------ training
def ultralytics_kwargs(recipe, data_yaml, out_dir, device, cache):
    """Every Ultralytics train argument of a full / freeze run. A YOLO(ckpt)
    keeps only imgsz, data, task and single_cls from the checkpoint's own
    args, and all four are set here."""
    r = recipe
    return {
        "data": str(data_yaml), "epochs": r["epochs"], "imgsz": r["imgsz"], "batch": r["batch"],
        "optimizer": r["optimizer"], "lr0": r["lr0"], "lrf": r["lrf"], "momentum": r["momentum"],
        "weight_decay": r["weight_decay"], "warmup_epochs": r["warmup_epochs"],
        "warmup_bias_lr": r["warmup_bias_lr"], "cos_lr": r["cos_lr"], "seed": r["seed"],
        "deterministic": r["deterministic"], "cache": cache, "workers": r["workers"],
        "close_mosaic": r["close_mosaic"], "freeze": r["freeze"] if r["trainer"] == "freeze" else None,
        "device": device, "project": str(out_dir), "name": TRAIN_NAME, "exist_ok": True,
        "val": False, "plots": False, "resume": False, "single_cls": False, "task": "detect",
    }


def lora_kwargs(recipe, data_yaml, out_dir, device, cache):
    r = recipe
    return {
        "data_yaml": str(data_yaml), "project": str(out_dir), "name": TRAIN_NAME,
        "epochs": r["epochs"], "lr0": r["lr0"], "seed": r["seed"], "imgsz": r["imgsz"],
        "batch": r["batch"], "device": device, "workers": r["workers"],
        "rank": r["lora"]["rank"], "alpha": r["lora"]["alpha"],
        "optimizer": r["optimizer"], "lrf": r["lrf"], "momentum": r["momentum"],
        "weight_decay": r["weight_decay"], "warmup_epochs": r["warmup_epochs"],
        "warmup_bias_lr": r["warmup_bias_lr"], "cos_lr": r["cos_lr"],
        "deterministic": r["deterministic"], "cache": cache,
        "close_mosaic": r["close_mosaic"], "val": False, "plots": False, "exist_ok": True,
        "single_cls": False,
    }


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def _nonfinite(net):
    import torch
    return [k for k, v in net.state_dict().items()
            if v.is_floating_point() and not bool(torch.isfinite(v).all())]


def _checkpoint_epoch(ck):
    """0-based epoch an Ultralytics checkpoint holds (a stripped one says -1
    and keeps train_results, whose 'epoch' column is 1-based)."""
    e = ck.get("epoch")
    if _is_int(e) and e >= 0:
        return e
    col = (ck.get("train_results") or {}).get("epoch") or []
    return int(round(float(max(col)))) - 1 if col else None


def _require_final_epoch(epoch, recipe, what, rec, extra=None):
    """The scored weights must be the final epoch's (module docstring, 8)."""
    last = recipe["epochs"] - 1
    if epoch == last:
        return
    rec["diverged"] = dict({"weights_epoch": epoch, "final_epoch": last, "epochs": recipe["epochs"]},
                           **(extra or {}))
    raise RunError("train", "diverged: %s holds (0-based) epoch %s, not the final epoch %d of %d. "
                   "Ultralytics skipped a save, which it does when the EMA holds NaN/Inf, so these "
                   "are not the final annealed weights the protocol scores" % (what, epoch, last,
                                                                               recipe["epochs"]))


def train_full(init, data_yaml, recipe, out_dir, device, n_rows, rec, cache):
    """YOLO(init).train(...) for 'full' / 'freeze'. Returns (the weights to
    score, the trainer's save_dir)."""
    import torch
    from ultralytics import YOLO
    kw = ultralytics_kwargs(recipe, data_yaml, out_dir, device, cache)
    rec["train_kwargs"] = kw
    model = YOLO(str(init))
    seen = {}

    def on_setup(trainer):
        ps = list(trainer.model.parameters())
        seen["total_params"] = sum(p.numel() for p in ps)
        seen["trainable_params"] = sum(p.numel() for p in ps if p.requires_grad)
        # fail before any epoch rather than after them; labels.cache is re-checked at the end
        ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        used = len(ds.im_files) if hasattr(ds, "im_files") else n_rows
        if used != n_rows:
            raise RunError("train", "Ultralytics loaded %d of the manifest's %d training images"
                           % (used, n_rows))

    model.add_callback("on_pretrain_routine_end", on_setup)
    model.train(**kw)
    trainer = model.trainer
    save_dir = Path(trainer.save_dir).resolve()
    if not _inside(save_dir, out_dir):
        raise RunError("train", "the trainer saved to %s, outside the run dir %s" % (save_dir, out_dir))
    last = save_dir / "weights" / "last.pt"
    if Path(trainer.last).resolve() != last or not last.is_file():
        raise RunError("train", "no last.pt in the trainer's save_dir %s" % save_dir)
    rec.update(train_dir=str(save_dir), **seen)
    ck = torch.load(str(last), map_location="cpu", weights_only=False)
    net = ck["ema"] if ck.get("ema") is not None else ck["model"]
    bad = _nonfinite(net)
    if bad:
        raise RunError("train", "%s holds NaN/Inf in %d tensor(s), first %s" % (last, len(bad), bad[:3]))
    rec["weights_epoch"] = _checkpoint_epoch(ck)
    _require_final_epoch(rec["weights_epoch"], recipe, "last.pt", rec)
    return last, save_dir


def train_lora_run(init, data_yaml, recipe, out_dir, device, rec, cache):
    from . import lora as L
    kw = lora_kwargs(recipe, data_yaml, out_dir, device, cache)
    rec["train_kwargs"] = kw
    args = dict(kw)
    merged = Path(L.train_lora(str(init), args.pop("data_yaml"), **args)).resolve()
    save_dir = merged.parent.parent
    if merged.name != L.MERGED_NAME or not merged.is_file() or not _inside(save_dir, out_dir):
        raise RunError("train", "train_lora returned %s, not a %s inside %s" % (merged, L.MERGED_NAME, out_dir))
    info = _read_json(save_dir / L.LORA_JSON) or {}
    rec.update(train_dir=str(save_dir), trainable_params=info.get("trainable_params"),
               total_params=info.get("total_params"), weights_epoch=info.get("merged_from_epoch"),
               lora={k: info.get(k) for k in ("rank", "alpha", "targets", "n_wrapped", "unwrapped_3x3",
                                               "adapter_params", "head_params", "transferred",
                                               "state_entries", "skipped_saves", "merged_from_epoch")})
    _require_final_epoch(info.get("merged_from_epoch"), recipe, L.MERGED_NAME, rec,
                         {"skipped_saves": info.get("skipped_saves")})
    return merged, save_dir


def install_weights(src, dst):
    """Copy src to dst atomically and check the copy; returns its sha256."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".%s.tmp" % dst.name)
    shutil.copyfile(src, tmp)
    sha = C.sha256_file(tmp)
    if sha != C.sha256_file(src):
        tmp.unlink()
        raise RunError("train", "copying %s to %s changed its bytes" % (src, dst))
    os.replace(tmp, dst)
    return sha


# ------------------------------------------------------------------- soup
def _names(net):
    names = getattr(net, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names) if names is not None else None


def make_soup(members, dst):
    """Uniform soup: every float tensor averaged in fp32, every other buffer
    (num_batches_tracked) from the first member. Refuses members whose modules,
    state keys, shapes or class names differ, or that hold modules other than
    torch / Ultralytics ones (a LoRA last.pt's adapters cannot be averaged).
    Saves a plain Ultralytics checkpoint (fp32) at dst; returns the record."""
    import torch
    from ultralytics import __version__ as ultralytics_version
    cks, nets, shas = [], [], []
    for p in members:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        net = ck.get("ema") if isinstance(ck, dict) and ck.get("ema") is not None else (
            ck.get("model") if isinstance(ck, dict) else None)
        if net is None or not hasattr(net, "state_dict"):
            raise RunError("soup", "%s is not an Ultralytics checkpoint" % p)
        foreign = sorted({type(m).__module__ for m in net.modules()
                          if not type(m).__module__.startswith(("torch.", "ultralytics."))})
        if foreign:
            raise RunError("soup", "%s holds modules from %s; soup only plain (merged) checkpoints" % (p, foreign))
        cks.append(ck)
        nets.append(net)
        shas.append(C.sha256_file(p))
    if len(set(shas)) != len(shas):
        raise RunError("soup", "soup_of lists the same weights twice (sha256 %s)" % shas)
    ref, ref_sd = nets[0], nets[0].state_dict()
    ref_arch = [(n, type(m).__name__) for n, m in ref.named_modules()]
    sds = [ref_sd]
    for p, net in zip(members[1:], nets[1:]):
        if _names(net) != _names(ref):
            raise RunError("soup", "%s and %s have different class names (%s vs %s)"
                           % (p, members[0], _names(net), _names(ref)))
        if [(n, type(m).__name__) for n, m in net.named_modules()] != ref_arch:
            raise RunError("soup", "%s and %s are different architectures" % (p, members[0]))
        sd = net.state_dict()
        if list(sd) != list(ref_sd) or any(sd[k].shape != ref_sd[k].shape
                                           or sd[k].is_floating_point() != ref_sd[k].is_floating_point()
                                           for k in sd):
            raise RunError("soup", "%s and %s have different state tensors" % (p, members[0]))
        sds.append(sd)
    if _names(ref) != C.CLASS_NAMES:
        raise RunError("soup", "the members' classes %s are not the INC class space" % _names(ref))
    avg = {}
    for k, v in ref_sd.items():
        if v.is_floating_point():
            acc = v.detach().float().clone()
            for sd in sds[1:]:
                acc += sd[k].detach().float()
            avg[k] = acc / len(sds)
        else:
            avg[k] = v.detach().clone()
    soup = copy.deepcopy(ref).float()
    soup.load_state_dict(avg, strict=True)
    bad = _nonfinite(soup)
    if bad:
        raise RunError("soup", "the soup holds NaN/Inf in %s" % bad[:3])
    if hasattr(soup, "args"):
        soup.args = dict(soup.args)
    if hasattr(soup, "criterion"):
        soup.criterion = None
    for p in soup.parameters():
        p.requires_grad_(False)
    record = {"of": [str(p) for p in members], "sha256": shas, "n": len(members),
              "dtype": "float32", "method": "uniform"}
    ckpt = {"date": datetime.datetime.now().isoformat(), "version": ultralytics_version,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com", "epoch": -1, "best_fitness": None,
            "model": soup, "ema": None, "updates": None, "optimizer": None, "scaler": None,
            "train_args": dict(cks[0].get("train_args") or {}), "train_metrics": {}, "soup": record}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".%s.tmp" % dst.name)
    torch.save(ckpt, str(tmp))
    os.replace(tmp, dst)
    return record, sum(p.numel() for p in soup.parameters())


# ------------------------------------------------------------------ scoring
def scorer_command(weights, exam, out, testing):
    """The scorer's argv. A production run passes no setting (the scorer's
    defaults are the protocol). A testing run passes testing's settings and
    scores on the CPU unless testing.device says otherwise."""
    cmd = [sys.executable, "-u", "-m", "weed_optimizer_framework.tools.inc.scorer",
           "--weights", str(weights), "--exam", exam, "--out", str(out)]
    if testing is not None:
        for k in ("imgsz", "batch"):
            if testing.get(k) is not None:
                cmd += ["--%s" % k, str(testing[k])]
        cmd += ["--device", str(testing.get("device") or "cpu")]
        if testing.get("lock_check") is False:
            cmd.append("--no-lock-check-for-tests")
    return cmd


def scorer_env(testing):
    root = str(package_dir().parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["INC_DIR"] = str(C.INC_DIR)
    env["REPO"] = str(C.REPO)
    if testing is None:
        env.pop(S.TEST_ENV, None)
    return env, root


def score_exams(weights, weights_sha, exams, scores_dir, testing):
    """Run the scorer once per exam, as a subprocess, and check that each
    score.json is about these weights and this exam. Returns {exam: record}."""
    scores_dir.mkdir(parents=True, exist_ok=True)
    env, root = scorer_env(testing)
    out = {}
    for exam in exams:
        dst = scores_dir / ("%s.json" % exam)
        cmd = scorer_command(weights, exam, dst, testing)
        log("scoring on %s: %s" % (exam, " ".join(cmd[2:])))
        r = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        sys.stdout.write(r.stdout)
        sys.stdout.flush()
        if r.returncode != 0:
            tail = "\n".join(r.stdout.strip().splitlines()[-15:])
            raise RunError("score", "the scorer exited %d on %s%s:\n%s"
                           % (r.returncode, exam, " (refused)" if r.returncode == 2 else "", tail))
        s = _read_json(dst)
        if not isinstance(s, dict) or s.get("exam") != exam or s.get("weights_sha256") != weights_sha:
            raise RunError("score", "%s is not a score of these weights on %s" % (dst, exam))
        if testing is None and s.get("production") is not True:
            raise RunError("score", "%s is not a production score (%s)" % (dst, s.get("deviations")))
        if testing is not None and s.get("production") is not False:
            raise RunError("score", "%s is a production score, but this is a testing experiment; "
                           "its scores must carry the TEST- stamp" % dst)
        out[exam] = {"path": str(dst), "sha256": C.sha256_file(dst), "production": s.get("production"),
                     "scorer_sha256": s.get("scorer_sha256")}
    return out


def needs_sidecar(spec):
    """True for a base, union, cand or soup run scored on dev."""
    return (isinstance(spec, dict) and spec.get("kind") in SIDECAR_KINDS
            and SIDECAR_EXAM in (spec.get("exams") or []))


def sidecar_command(weights, exam, score_path, out, testing):
    """The sidecar's argv: the scorer's settings exactly (scorer_command)."""
    cmd = [sys.executable, "-u", "-m", "weed_optimizer_framework.tools.inc2.scorer_sidecar",
           "--weights", str(weights), "--exam", exam, "--score", str(score_path), "--out", str(out)]
    if testing is not None:
        for k in ("imgsz", "batch"):
            if testing.get(k) is not None:
                cmd += ["--%s" % k, str(testing[k])]
        cmd += ["--device", str(testing.get("device") or "cpu")]
        if testing.get("lock_check") is False:
            cmd.append("--no-lock-check-for-tests")
    return cmd


def run_sidecar(weights, weights_sha, scores_dir, score_rec, testing, exam=SIDECAR_EXAM):
    """Run the scorer sidecar on exam as a subprocess and check that what it
    wrote is about these weights and this run's recorded score. Returns its
    record for run.json."""
    from . import scorer_sidecar as SC
    out, npz = SC.paths_for(scores_dir / ("%s%s" % (exam, SC.JSON_SUFFIX)))
    cmd = sidecar_command(weights, exam, score_rec["path"], out, testing)
    env, root = scorer_env(testing)
    log("sidecar on %s: %s" % (exam, " ".join(cmd[2:])))
    r = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(r.stdout)
    sys.stdout.flush()
    if r.returncode != 0:
        tail = "\n".join(r.stdout.strip().splitlines()[-15:])
        raise RunError("sidecar", "the scorer sidecar exited %d on %s%s:\n%s"
                       % (r.returncode, exam, " (refused)" if r.returncode == 2 else "", tail))
    d = _read_json(out)
    if (not isinstance(d, dict) or d.get("format") != SC.FORMAT or d.get("exam") != exam
            or d.get("weights_sha256") != weights_sha):
        raise RunError("sidecar", "%s is not a sidecar of these weights on %s" % (out, exam))
    if (d.get("score") or {}).get("sha256") != score_rec["sha256"]:
        raise RunError("sidecar", "%s was made from another score of %s than scores/%s.json" % (out, exam, exam))
    img = d.get("images") or {}
    if Path(img.get("path") or "").resolve() != npz.resolve() or _sha_or_none(npz) != img.get("sha256"):
        raise RunError("sidecar", "%s does not name the per-image arrays %s it wrote" % (out, npz))
    per = ((d.get("species_se") or {}).get("per_species") or {})
    return {"path": str(out), "sha256": C.sha256_file(out), "images": {"path": str(npz), "sha256": img["sha256"]},
            "resamples": (d.get("species_se") or {}).get("resamples"),
            "species_se": {s: (v or {}).get("se") for s, v in sorted(per.items())}}


# ------------------------------------------------------------------ cleanup
def _rm(path, removed):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        removed.append(str(path))
        path.unlink()
    elif path.is_dir():
        removed.append(str(path) + "/")
        shutil.rmtree(path)


def move_aside(path, removed):
    """Rename a dir to .trash-<name>-<unique> beside it (instant, where rmtree
    of a big materialised dir on Lustre is not); returns the new path or None."""
    path = Path(path)
    if not (path.is_dir() and not path.is_symlink()):
        _rm(path, removed)
        return None
    dst = path.with_name("%s%s-%d-%s" % (TRASH_PREFIX, path.name, os.getpid(), os.urandom(4).hex()))
    os.rename(str(path), str(dst))
    removed.append(str(path) + "/")
    return dst


def purge_dir(path):
    """Delete a dir moved aside; errors are left for the next attempt."""
    shutil.rmtree(str(path), ignore_errors=True)


def purge_trash(out_dir):
    for p in sorted(Path(out_dir).glob(TRASH_PREFIX + "*")):
        if p.is_dir() and not p.is_symlink():
            purge_dir(p)


def train_dirs(out_dir, *records):
    dirs = {out_dir / TRAIN_NAME}
    for rec in records:
        td = (rec or {}).get("train_dir")
        if td and _inside(td, out_dir) and Path(td).resolve() != out_dir.resolve():
            dirs.add(Path(td))
    return sorted(dirs)


def drop_train_weights(train_dir, removed):
    wdir = Path(train_dir) / "weights"
    if wdir.is_dir():
        for p in sorted(wdir.iterdir()):
            if p.suffix in (".pt", ".tmp") or p.name.endswith(".pt.tmp"):
                _rm(p, removed)
        if not any(wdir.iterdir()):
            wdir.rmdir()


# ------------------------------------------------------------------ the lock
def _flock(fh, op):
    """fcntl.flock, under a name the tests can replace."""
    fcntl.flock(fh.fileno(), op)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:             # EPERM: it exists, under another user
        return True
    return True


class RunDirLock:
    """One executor per run dir, across nodes (module docstring).

    acquire() returns True (this process owns the dir) or False (another
    executor does), and raises LockError when that cannot be told. check()
    raises LockLost once the owner file is no longer this process's;
    still_owned() looks now instead of waiting for the heartbeat."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.flock_path = self.out_dir / LOCK_NAME
        self.owner_path = self.out_dir / OWNER_NAME
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.token = os.urandom(8).hex()
        self.info = {"host": self.host, "pid": self.pid, "token": self.token,
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_array_task": ("%s_%s" % (os.environ.get("SLURM_ARRAY_JOB_ID"),
                                                     os.environ.get("SLURM_ARRAY_TASK_ID"))
                                          if os.environ.get("SLURM_ARRAY_TASK_ID") else None),
                     "started_utc": _utc()}
        self.fh = None
        self.flock = None
        self.ino = None
        self.lost = None
        self.took_over = []
        self.busy_owner = None
        self._stop = threading.Event()
        self._thread = None

    def record(self):
        return {"flock": self.flock, "owner_file": str(self.owner_path), "owner": self.info,
                "took_over": self.took_over}

    def acquire(self):
        try:
            self.fh = open(self.flock_path, "a")
        except OSError as e:
            raise LockError("cannot open %s: %s" % (self.flock_path, e))
        try:
            try:
                _flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.flock = "held"
            except OSError as e:
                if e.errno in FLOCK_BUSY:
                    self.busy_owner = _read_json(self.owner_path)
                    self._close()
                    return False
                if e.errno not in FLOCK_UNSUPPORTED:
                    raise LockError("fcntl.flock on %s failed: %s" % (self.flock_path, e))
                self.flock = ("unavailable on this mount (%s): the owner file alone guards the run dir"
                              % errno.errorcode.get(e.errno, e.errno))
            if not self._take_owner():
                self._close()
                return False
        except BaseException:
            self._close()
            raise
        self._thread = threading.Thread(target=self._beat, name="inc-run-dir-heartbeat", daemon=True)
        self._thread.start()
        return True

    def _take_owner(self):
        deadline = time.time() + LOCK_WAIT_SECONDS
        takeovers = 0
        while True:
            try:
                fd = os.open(str(self.owner_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                fd = None
            except OSError as e:
                raise LockError("cannot create %s: %s" % (self.owner_path, e))
            if fd is not None:
                try:
                    with os.fdopen(fd, "w") as fh:
                        json.dump(self.info, fh, sort_keys=True)
                        fh.write("\n")
                    self.ino = os.stat(self.owner_path).st_ino
                except BaseException:
                    try:
                        os.unlink(str(self.owner_path))
                    except OSError:
                        pass
                    raise
                return True
            try:
                st = os.stat(self.owner_path)
            except FileNotFoundError:
                continue                          # released meanwhile: try again
            except OSError as e:
                raise LockError("cannot stat %s: %s" % (self.owner_path, e))
            owner = _read_json(self.owner_path)
            why = self._stale(owner, st)
            if why and takeovers < 5:             # a sixth stale file in a row: something else is at work
                takeovers += 1
                self._take_over(owner, st, why)
                continue
            if time.time() >= deadline:
                self.busy_owner = owner
                return False
            time.sleep(LOCK_POLL_SECONDS)

    def _stale(self, owner, st):
        """Why the owner file is stale; None while its owner may be alive."""
        o = owner if isinstance(owner, dict) else {}
        pid = o.get("pid")
        if o.get("host") == self.host and _is_int(pid):
            if pid == self.pid and o.get("token") != self.token:
                return "left by an earlier attempt in this process (pid %d)" % pid
            if pid != self.pid and not _pid_alive(pid):
                return "its process %d on this host (%s) has exited" % (pid, self.host)
        age = time.time() - st.st_mtime
        if age > LOCK_STALE_SECONDS:
            return "its heartbeat is %.0f s old (stale after %d s)" % (age, LOCK_STALE_SECONDS)
        return None

    def _take_over(self, owner, st, why):
        """Rename the stale owner file aside (one taker wins the rename), and put
        it back if what was moved is not the file judged stale."""
        aside = self.owner_path.with_name("%s.stale.%d.%s" % (OWNER_NAME, self.pid, self.token))
        try:
            os.rename(str(self.owner_path), str(aside))
        except FileNotFoundError:
            return
        try:
            got = os.stat(aside)
        except OSError:
            got = None
        moved = _read_json(aside)
        same_token = not isinstance(owner, dict) or (isinstance(moved, dict)
                                                     and moved.get("token") == owner.get("token"))
        if got is None or got.st_ino != st.st_ino or got.st_mtime != st.st_mtime or not same_token:
            try:
                os.link(str(aside), str(self.owner_path))
            except OSError:
                pass
        else:
            self.took_over.append({"owner": owner, "reason": why, "utc": _utc()})
            log("took over the run dir from a stale owner (%s): %s" % (why, owner))
        try:
            os.unlink(str(aside))
        except OSError:
            pass

    def _mine(self):
        """True / False whether the owner file is still this process's (same
        inode and token: an inode number alone can be reused), None when it
        cannot be told now (a transient error, or a new owner mid-write)."""
        try:
            st = os.stat(self.owner_path)
            with open(self.owner_path) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return None
        return st.st_ino == self.ino and isinstance(data, dict) and data.get("token") == self.token

    def _beat(self):
        while not self._stop.wait(LOCK_HEARTBEAT_SECONDS):
            mine = self._mine()
            if mine is False:
                self.lost = "%s now belongs to another executor (this one was taken over as stale)" % self.owner_path
                return
            if mine:
                try:
                    os.utime(str(self.owner_path), None)
                except OSError:
                    pass                          # transient; the next beat tries again

    def check(self):
        if self.lost:
            raise LockLost(self.lost)

    def still_owned(self):
        if self.lost:
            return False
        if self._mine() is False:
            self.lost = "%s now belongs to another executor (this one was taken over as stale)" % self.owner_path
            return False
        return True                               # yes, or cannot tell now: do not orphan the record

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.ino is not None and not self.lost and self._mine():
            try:
                os.unlink(str(self.owner_path))
            except OSError:
                pass
        self._close()

    def _close(self):
        if self.fh is not None:
            if self.flock == "held":
                try:
                    _flock(self.fh, fcntl.LOCK_UN)
                except OSError:
                    pass
            self.fh.close()
            self.fh = None


def _mount_unescape(s):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def mount_of(path, mountinfo):
    """(mount point, fstype, options) of the mount holding the absolute path,
    from /proc/self/mountinfo text (the last of equal mount points wins, as it
    shadows the others); None when no line covers it."""
    best = None
    for ln in mountinfo.splitlines():
        f = ln.split()
        if "-" not in f[6:]:
            continue
        sep = f.index("-", 6)
        if len(f) < sep + 3:
            continue
        mnt = _mount_unescape(f[4])
        if not (path == mnt or mnt == "/" or path.startswith(mnt.rstrip("/") + "/")):
            continue
        if best is None or len(mnt) >= len(best[0]):
            opts = set(f[5].split(","))
            if len(f) > sep + 3:
                opts |= set(f[sep + 3].split(","))
            best = (mnt, f[sep + 1], opts)
    return best


def flock_verdict(mount):
    """(True / False / None = unknown, reason): whether fcntl.flock on the
    mount_of() result excludes processes on other nodes. Lustre: only when
    mounted with 'flock' ('localflock' is per node; neither = flock fails);
    NFS: unless local_lock=flock|all; GPFS: yes; a node-local filesystem: yes
    (every process that reaches it is on this node)."""
    if mount is None:
        return None, "no mount covers the path"
    fs, opts = mount[1], mount[2]
    if fs == "lustre":
        if "flock" in opts:
            return True, "Lustre mounted with 'flock': flock is coherent across every client node"
        if "localflock" in opts:
            return False, "Lustre mounted with 'localflock': flock is local to each node"
        return False, "Lustre mounted without 'flock' (noflock): flock fails"
    if fs in ("nfs", "nfs4"):
        local = {o.split("=", 1)[1] for o in opts if o.startswith("local_lock=")}
        if local & {"flock", "all"}:
            return False, "NFS with local_lock=%s: flock is local to this client" % "/".join(sorted(local))
        return True, "NFS: flock is taken as a whole-file lock on the server"
    if fs == "gpfs":
        return True, "GPFS: flock is coherent across nodes"
    if fs in LOCAL_FS:
        return True, "node-local %s: every process that can reach it runs on this node" % fs
    return None, "whether flock on %s is coherent across nodes is not known here" % fs


def flock_coherence(path, mountinfo=None):
    """{path, mount, fstype, options, cross_node, reason} for path's
    filesystem (flock_verdict), from /proc/self/mountinfo."""
    p = os.path.realpath(str(path))
    out = {"path": p, "mount": None, "fstype": None, "options": None, "cross_node": None, "reason": None}
    if mountinfo is None:
        try:
            with open("/proc/self/mountinfo") as fh:
                mountinfo = fh.read()
        except OSError:
            out["reason"] = "no /proc/self/mountinfo (not Linux): cannot tell"
            return out
    m = mount_of(p, mountinfo)
    if m is not None:
        out.update(mount=m[0], fstype=m[1], options=sorted(m[2]))
    out["cross_node"], out["reason"] = flock_verdict(m)
    return out


# ------------------------------------------------------------------ the run
def _env_record():
    rec = {"hostname": socket.gethostname(), "python_version": platform.python_version(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
           "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
           "ultralytics_version": None, "torch_version": None, "cuda_device": None}
    try:
        import torch
        import ultralytics
        rec.update(ultralytics_version=ultralytics.__version__, torch_version=torch.__version__)
        if torch.cuda.is_available():
            rec["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:     # recorded; the run itself fails later where it needs them
        rec["import_error"] = "%s: %s" % (type(e).__name__, e)
    return rec


def _previous(out_dir):
    prev = _read_json(out_dir / RUN_JSON)
    att = _read_json(out_dir / ATTEMPT_JSON)
    n = 0
    for d in (prev, att):
        if isinstance(d, dict) and _is_int(d.get("attempt")):
            n = max(n, d["attempt"])
    return (prev if isinstance(prev, dict) else None), (att if isinstance(att, dict) else None), n


def spec_digest(spec, spec_path):
    """sha256 of the spec's canonical JSON (sorted keys), so that rewriting the
    same spec with other whitespace or key order is not a new spec; the file's
    own sha256 when it does not parse."""
    if spec is None:
        return C.sha256_file(spec_path)
    return C.sha256_text(json.dumps(spec, sort_keys=True, separators=(",", ":")))


def _sha_or_none(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def _local_weights(path):
    p = Path(path)
    return (p if p.is_absolute() else C.REPO / p).resolve()


def prior_weights(spec, spec_sha, prev, out_dir):
    """(True, None) when the previous attempt's weights/final.pt can stand for
    this spec: same spec, trained, the file hashes to its weights_sha256, and
    what it was made from still hashes as recorded; else (False, why)."""
    if not isinstance(prev, dict):
        return False, "there is no earlier run.json"
    if not isinstance(spec, dict):
        return False, "the spec does not parse"
    if prev.get("spec_sha256") != spec_sha:
        return False, "the spec changed"
    if prev.get("trained") is not True or not prev.get("weights_sha256"):
        return False, "the earlier attempt left no weights"
    kind = spec.get("kind")
    final = out_dir / "weights" / FINAL_NAME
    if not final.exists():
        return False, "weights/final.pt is missing"
    if kind != "final" and final.is_symlink():
        return False, "weights/final.pt is a symlink"
    if _sha_or_none(final) != prev["weights_sha256"]:
        return False, "weights/final.pt no longer hashes to run.json's weights_sha256"
    if kind in TRAIN_KINDS or kind == "final":
        if _sha_or_none(_local_weights(spec.get("init") or "")) != prev.get("init_sha256"):
            return False, "the init weights changed"
    if kind in TRAIN_KINDS:
        if _sha_or_none(Path(spec.get("train_manifest") or "").resolve()) != prev.get("train_manifest_sha256"):
            return False, "the train manifest changed"
    if kind == "soup":
        shas = [_sha_or_none(_local_weights(p)) for p in spec.get("soup_of") or []]
        if shas != prev.get("soup_sha256"):
            return False, "a soup member changed"
    return True, None


def scores_intact(out_dir, spec, prev):
    """(True, None) when every exam's score file is there and hashes as the
    previous run.json recorded; else (False, why)."""
    recorded = prev.get("scores") or {}
    for e in spec.get("exams") or []:
        p = out_dir / "scores" / ("%s.json" % e)
        if not p.is_file():
            return False, "scores/%s.json is missing" % e
        if _sha_or_none(p) != (recorded.get(e) or {}).get("sha256"):
            return False, "scores/%s.json is not the one run.json recorded" % e
    if needs_sidecar(spec):
        sc = (prev.get("sidecars") or {}).get(SIDECAR_EXAM)
        if not isinstance(sc, dict) or sc.get("status") == "failed":
            return False, "no scorer sidecar is recorded for %s" % SIDECAR_EXAM
        for path, sha in ((sc.get("path"), sc.get("sha256")), ((sc.get("images") or {}).get("path"),
                                                               (sc.get("images") or {}).get("sha256"))):
            if not path or _sha_or_none(path) != sha:
                return False, "the scorer sidecar file %s is missing or not the one run.json recorded" % path
    return True, None


def _new_record(spec, spec_path, spec_sha, out_dir, attempt, force):
    rec = {"exp": spec.get("exp") if isinstance(spec, dict) else None,
           "run_id": spec.get("run_id") if isinstance(spec, dict) else out_dir.name,
           "kind": spec.get("kind") if isinstance(spec, dict) else None,
           "status": "failed", "error": None, "stage": None, "attempt": attempt,
           "resumed_from_attempt": None, "forced": bool(force), "trained": False,
           "spec_path": str(spec_path), "spec_sha256": spec_sha, "spec_file_sha256": _sha_or_none(spec_path),
           "spec": spec, "testing": False, "testing_settings": None, "warnings": [],
           "started_utc": _utc(), "out_dir": str(out_dir), "weights": None, "weights_sha256": None,
           "scores": {}, "n_train_images": None, "n_train_boxes": None, "trainable_params": None,
           "total_params": None, "guard": None, "seconds": None, "protocol": PROTOCOL,
           "protocol_package": PROTOCOL_PACKAGE, "splits_version": RC.SPLITS_VERSION, "sidecars": {}}
    rec.update(_env_record())
    return rec


def record_lock_failure(spec_path, out_dir, err):
    """A failed run.json at stage 'lock', best effort and never over a done
    record: the dir is not this process's to clean."""
    spec = _read_json(spec_path)
    prev, _, n = _previous(out_dir)
    if prev and prev.get("status") == "done":
        log("left %s's done run.json in place" % out_dir.name)
        return
    rec = _new_record(spec, spec_path, spec_digest(spec, spec_path), out_dir, n + 1, False)
    rec.update(stage="lock", error="%s: %s" % (type(err).__name__, err), seconds=0.0, finished_utc=_utc())
    try:
        write_json(out_dir / RUN_JSON, rec)
    except OSError as e:
        log("could not write %s: %s" % (out_dir / RUN_JSON, e))


def execute(spec_path, force=False):
    """Run one spec; returns the process exit code (0 done or no-op, 1 failed,
    3 another executor holds the run dir)."""
    spec_path = Path(spec_path).resolve()
    out_dir = spec_path.parent
    if not spec_path.is_file():
        log("no spec at %s" % spec_path)
        return EXIT_FAILED
    lock = RunDirLock(out_dir)
    try:
        held = lock.acquire()
    except LockError as e:
        log("cannot lock %s: %s" % (out_dir, e))
        record_lock_failure(spec_path, out_dir, e)
        return EXIT_FAILED
    if not held:
        log("another executor is working on %s (%s); leaving it alone"
            % (out_dir, lock.busy_owner or "flock held"))
        return EXIT_BUSY
    try:
        return _execute_locked(spec_path, out_dir, force, lock)
    finally:
        lock.release()


def _execute_locked(spec_path, out_dir, force, lock):
    t0 = time.time()
    spec = _read_json(spec_path)
    spec_sha = spec_digest(spec, spec_path)
    prev, prev_att, prev_n = _previous(out_dir)
    name = spec_path.parent.name

    reusable, why = (False, "--force") if force else prior_weights(spec, spec_sha, prev, out_dir)
    if prev and prev.get("status") == "done" and not force:
        if reusable:
            ok, why = scores_intact(out_dir, spec, prev)
            if ok:
                purge_trash(out_dir)
                log("%s is already done (attempt %s); nothing to do (--force re-runs it)"
                    % (name, prev.get("attempt")))
                return EXIT_DONE
        log("%s was done (attempt %s), but %s: running it again" % (name, prev.get("attempt"), why))
    resume_from = prev if reusable and isinstance(spec, dict) and spec.get("kind") in RESUME_KINDS else None

    attempt = prev_n + 1
    history = list((prev_att or {}).get("history") or [])
    if prev:
        entry = {k: prev.get(k) for k in ("attempt", "status", "stage", "slurm_job_id", "finished_utc")}
        entry["error_head"] = (prev.get("error") or "")[-300:] or None
        history.append(entry)
    write_json(out_dir / ATTEMPT_JSON, {"attempt": attempt, "started_utc": _utc(),
                                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                        "hostname": socket.gethostname(), "history": history[-20:]})
    if (out_dir / RUN_JSON).exists():
        (out_dir / RUN_JSON).unlink()

    rec = _new_record(spec, spec_path, spec_sha, out_dir, attempt, force)
    rec["lock"] = lock.record()
    ctx = {"removed": [], "created_final": False}
    if resume_from:
        # From the first line on, this attempt keeps the verified final.pt it re-scores,
        # and a failure before scoring still names it for the next attempt.
        for k in CARRY_OVER:
            if k in resume_from:
                rec[k] = resume_from[k]
        rec.update(resumed_from_attempt=resume_from.get("attempt"), trained=True,
                   weights=str(out_dir / "weights" / FINAL_NAME), weights_sha256=resume_from["weights_sha256"])
    _SIG.update(closing=False, late=None)
    try:
        try:
            _run(spec, spec_path, out_dir, rec, prev, resume_from, ctx, lock)
            rec["status"] = "done"
        except BaseException as e:      # noqa: B902 -- every failure, SIGTERM included, gets a run.json
            _SIG["closing"] = True
            rec["status"] = "failed"
            rec["stage"] = getattr(e, "stage", None) or rec["stage"] or "unexpected"
            rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            if getattr(e, "guard", None) is not None:
                rec["guard"] = e.guard
            if isinstance(e, KeyboardInterrupt):
                rec["warnings"].append("interrupted")
    finally:
        _SIG["closing"] = True          # until this attempt's process ends (reset by the next attempt)
        recorded = _finish(out_dir, rec, ctx, lock, t0, resume_from)
    if not recorded:
        log("%s: %s; the other executor keeps the run dir, this attempt wrote and removed nothing"
            % (rec["run_id"], lock.lost))
        return EXIT_BUSY
    if rec["status"] == "done":
        log("%s done in %.0fs (attempt %d%s): %s sha256 %s, scored on %s%s"
            % (rec["run_id"], rec["seconds"], attempt,
               ", re-scored attempt %s's weights" % rec["resumed_from_attempt"] if resume_from else "",
               FINAL_NAME, (rec["weights_sha256"] or "")[:12], ", ".join(rec["scores"]),
               " [TESTING]" if rec["testing"] else ""))
        return EXIT_DONE
    log("%s FAILED at %s (attempt %d): %s" % (rec["run_id"], rec["stage"], attempt,
                                             (rec["error"] or "").strip().splitlines()[-1:]))
    return EXIT_FAILED


def _finish(out_dir, rec, ctx, lock, t0, resume_from):
    """Cleanup and run.json, in the order that keeps the record (module
    docstring, 10). False when the run dir now belongs to another executor:
    then nothing is written or removed."""
    if not lock.still_owned():
        return False
    removed = ctx["removed"]
    trash = None
    try:
        trash = move_aside(out_dir / "data", removed)
        for td in train_dirs(out_dir, rec):
            drop_train_weights(td, removed)
        final = out_dir / "weights" / FINAL_NAME
        if ctx["created_final"] and not rec["trained"] and (final.exists() or final.is_symlink()):
            _rm(final, removed)
    except OSError as e:
        rec["warnings"].append("cleanup: %s" % e)
    if _SIG["late"] is not None:
        rec["warnings"].append("signal %d arrived while the attempt was being recorded" % _SIG["late"])
    rec["cleanup_removed"] = removed
    rec["lock"] = lock.record()
    rec["seconds"] = round(time.time() - t0, 1)
    prev_total = (resume_from or {}).get("seconds_all_attempts") or (resume_from or {}).get("seconds") or 0
    rec["seconds_all_attempts"] = round(prev_total + rec["seconds"], 1)
    rec["finished_utc"] = _utc()
    write_json(out_dir / RUN_JSON, rec)
    if trash is not None:
        purge_dir(trash)
    return True


def _run(spec, spec_path, out_dir, rec, prev, resume_from, ctx, lock):
    """The body of one attempt; fills rec, raises on any failure."""
    def stage(name):
        lock.check()
        rec["stage"] = name

    validate_spec(spec, spec_path)
    kind = spec["kind"]
    removed = ctx["removed"]

    stage("testing")
    testing, warns = testing_settings(spec["exp"])
    rec["warnings"] += warns
    rec["testing"], rec["testing_settings"] = testing is not None, testing
    stage("code")
    rec["code"], drift = code_provenance()
    rec["code_drift"] = drift
    if drift:
        msg = ("the running package %s differs from the git-tracked copy %s in %s; sync the outer copy "
               "(or set %s=1)" % (rec["code"]["package_dir"], rec["code"]["nested_dir"], drift, ALLOW_DRIFT_ENV))
        if os.environ.get(ALLOW_DRIFT_ENV) == "1":
            rec["warnings"].append(msg)
        else:
            raise RunError("code", msg)

    stage("recipe")
    ro = experiment_research_only(spec["exp"])
    if ro is not None:
        rec["research_only"] = ro           # §8: a model trained on any research-only row is research-only
    arm, warns = experiment_arm(spec["exp"])
    rec["warnings"] += warns
    rec["arm"] = arm
    if kind in TRAIN_KINDS:
        rname, n_base, bprobs = experiment_budget(spec["exp"], arm)
        devs = protocol_deviations(kind, spec["recipe"], arm, recipe_name=rname, n_images=n_base) + bprobs
        rec["recipe_name"] = RC.match(kind, spec["recipe"], arm, recipe_name=rname, n_images=n_base)
        rec["protocol_recipe"], rec["recipe_deviations"] = not devs, devs
        if devs and testing is None:
            raise RunError("recipe", "a production %s run trains a Protocol v3 recipe of its arm (%s) only; this "
                           "one departs from it in %s" % (kind, arm["id"], devs))
        departs = init_check(kind, spec["init"], arm)
        rec["init_check"] = {"passed": not departs, "departures": departs}
        if departs and testing is None:
            raise RunError("recipe", "a production %s run starts from its arm's checkpoint (%s): %s"
                           % (kind, arm["id"], departs))

    stage("device")
    import torch
    cuda = torch.cuda.is_available()
    if testing is None and not cuda:
        raise RunError("device", "no CUDA device: production runs train and score on a GPU")
    if testing is not None and testing.get("device") is not None:
        device = str(testing["device"])
    else:
        device = "0" if cuda else "cpu"
    rec["device"] = device
    if kind in TRAIN_KINDS and not resume_from and device.lower() not in ("cpu", "mps"):
        rec["amp_check_weights"] = require_amp_weights()
        if rec["amp_check_weights"].get("name") is None:
            rec["warnings"].append(rec["amp_check_weights"]["note"])

    stage("prepare")
    purge_trash(out_dir)
    if not resume_from:
        for name in ("data", "weights", "scores"):
            _rm(out_dir / name, removed)
        for td in train_dirs(out_dir, prev):
            _rm(td, removed)
    else:
        _rm(out_dir / "data", removed)
        _rm(out_dir / "scores", removed)
    final = out_dir / "weights" / FINAL_NAME

    if resume_from:
        stage("resume")
        if _sha_or_none(final) != resume_from["weights_sha256"]:
            raise RunError("resume", "%s changed since it was verified" % final)
    elif kind in TRAIN_KINDS:
        stage("init")
        init = resolve_weights(spec["init"], "init weights", "init")
        rec["init"], rec["init_sha256"] = str(init), C.sha256_file(init)
        stage("manifest")
        rows, dhashes, info = check_manifest(spec["train_manifest"])
        rec.update(info)
        for k, what in (("duplicate_images", "the same image bytes"), ("dhash0_collisions", "the same dHash")):
            d = info[k]
            if d["extra_rows"]:
                rec["warnings"].append("%s: %d group(s) of rows hold %s (%d extra row(s), %d group(s) across "
                                       "sources or sessions), first %s"
                                       % (k, d["groups"], what, d["extra_rows"], d["across_source_session"],
                                          d["first"][:2]))
        stage("guard")
        rec["guard"] = guard_rows(rows, dhashes, production=testing is None)
        stage("materialise")
        data_yaml, rec["materialised"] = materialise_run(rows, out_dir / "data")
        stage("train")
        t_train = time.time()
        recipe = spec["recipe"]
        cache = choose_cache(recipe, rows, rec)
        if recipe["trainer"] == "lora":
            src, save_dir = train_lora_run(init, data_yaml, recipe, out_dir, device, rec, cache)
        else:
            src, save_dir = train_full(init, data_yaml, recipe, out_dir, device, len(rows), rec, cache)
        rec["dataset_check"] = check_dataset_cache(out_dir / "data", len(rows))
        rec["train_seconds"] = round(time.time() - t_train, 1)
        stage("install")
        ctx["created_final"] = True
        rec["weights_sha256"] = install_weights(src, final)
        rec["weights"], rec["trained"] = str(final), True
        rec["weights_source"] = str(src)
        drop_train_weights(save_dir, removed)
    elif kind == "soup":
        stage("soup")
        members = [resolve_weights(p, "soup member", "soup") for p in spec["soup_of"]]
        ctx["created_final"] = True
        rec["soup"], rec["total_params"] = make_soup(members, final)
        rec["soup_of"], rec["soup_sha256"] = rec["soup"]["of"], rec["soup"]["sha256"]
        rec["weights"], rec["weights_sha256"], rec["trained"] = str(final), C.sha256_file(final), True
    else:   # final
        stage("link")
        init = resolve_weights(spec["init"], "init weights", "link")
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(".%s.tmp" % final.name)
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        ctx["created_final"] = True
        os.symlink(str(init), str(tmp))
        os.replace(tmp, final)
        rec["init"], rec["init_sha256"] = str(init), C.sha256_file(init)
        rec["weights"], rec["weights_sha256"], rec["trained"] = str(final), rec["init_sha256"], True

    stage("score")
    rec["scores"] = score_exams(final, rec["weights_sha256"], spec["exams"], out_dir / "scores", testing)
    if needs_sidecar(spec):
        stage("sidecar")
        try:
            rec["sidecars"] = {SIDECAR_EXAM: run_sidecar(final, rec["weights_sha256"], out_dir / "scores",
                                                         rec["scores"][SIDECAR_EXAM], testing)}
        except RunError as e:
            if experiment_type(spec["exp"]) != "baseline":
                raise
            rec["sidecars"] = {SIDECAR_EXAM: {"status": "failed", "error": str(e)[-2000:]}}
            rec["warnings"].append("the scorer sidecar failed (%s); a baseline records it and goes on, since no "
                                   "gate decision of this experiment reads it" % str(e).splitlines()[0][:300])
    stage(None)


# ------------------------------------------------------------------ the CLI
def _on_sigterm(signum, frame):
    if _SIG["closing"]:
        _SIG["late"] = signum
        return
    raise Terminated("received signal %d (Slurm time limit or scancel)" % signum)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The INC executor: train / soup / score one run spec.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--spec", help="INC_DIR/<exp>/runs/<run_id>/spec.json")
    what.add_argument("--flock-check", metavar="PATH",
                      help="print whether fcntl.flock on PATH's filesystem is coherent across nodes; "
                           "exit 0 yes, 1 no or unknown")
    ap.add_argument("--force", action="store_true", help="re-run a run that is already done, from scratch")
    a = ap.parse_args(argv)
    if a.flock_check is not None:
        v = flock_coherence(a.flock_check)
        log("flock on %s: %s (%s)" % (v["path"], {True: "coherent across nodes", False: "NOT coherent "
                                                  "across nodes", None: "unknown"}[v["cross_node"]], v["reason"]))
        print(json.dumps(v, sort_keys=True), flush=True)
        return 0 if v["cross_node"] is True else 1
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # a run never pip-installs anything
    old = None
    try:
        old = signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:          # not the main thread (an in-process test)
        pass
    try:
        return execute(a.spec, force=a.force)
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


if __name__ == "__main__":
    sys.exit(main())
