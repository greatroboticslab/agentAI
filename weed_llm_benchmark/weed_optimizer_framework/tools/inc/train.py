"""The INC executor: one run spec in, weights/final.pt + scores + run.json out
(docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Executor").

    python -m weed_optimizer_framework.tools.inc.train --spec INC_DIR/<exp>/runs/<run_id>/spec.json [--force]
    python -m weed_optimizer_framework.tools.inc.train --flock-check PATH

Kinds:
  base, union   cold runs; cand, null   incremental runs from an incumbent.
      recipe.trainer 'full' (every layer), 'freeze' (Ultralytics freeze=n) or
      'lora' (inc.lora.train_lora, adapters merged into plain convs).
  soup    uniform average of soup_of's checkpoints; no training.
  final   scoring only: weights/final.pt is a symlink to init. The only kind
          that may list the 'test' exam.

What a training run does, in order, failing closed at each step:
  1. the spec: only the documented keys (an unknown key, or a recipe key
     outside RECIPE_KEYS, is refused, not ignored), 'test' only for kind
     final, out_dir = the spec's own dir = INC_DIR/<exp>/runs/<run_id>;
  2. the code: the modules this run imports are hashed into run.json; when
     the running package is not the git-tracked nested copy under
     REPO/weed_llm_benchmark (the cluster runs the outer copy) and a module
     differs from its nested twin, the run fails (INC_ALLOW_DRIFT=1 runs the
     outer copy anyway, recorded as a warning);
  3. the recipe: a production run (not a testing experiment) trains only the
     protocol's recipe for its kind (PROTOCOL_*: imgsz 640, batch 32, SGD,
     deterministic, cosine to lrf 0.01; base / union: trainer full, 100
     epochs, lr0 0.01, warmup 3; cand / null: 30 epochs, warmup 1, lr0 =
     warmup_bias_lr = 0.002, or 0.01 for lora; freeze 11; lora rank 16 alpha
     32), so a wrong builder cannot produce 'production' scores; a testing run
     records its departures (recipe_deviations) instead;
  4. the device: production needs CUDA. On CUDA, Ultralytics' AMP check loads
     a small checkpoint (yolo26n.pt in 8.4.x; the name is read from the
     installed release's check_amp) from the working directory or its
     weights_dir, and otherwise downloads it into the working directory, in
     place, where the other array tasks would load each other's partial file.
     A CUDA training run refuses unless a complete copy is already there;
  5. the manifest: every row's label bytes and image bytes match the
     manifest's sha256, every label line is a box Ultralytics accepts (it would
     otherwise drop the image silently); exact duplicate images (one sha256
     under several keys) and dHash-0 pairs are counted in run.json
     (duplicate_images, dhash0_collisions) with a warning, since an image
     that is both in an increment and in its replay sample makes cand less
     new than it looks. Then NeverTrainGuard.load().assert_trainable() on
     every image: an image within HOLDOUT_NEAR_DUP_BITS of a dev / test /
     exam image, or one that cannot be hashed, stops the run;
  6. common.materialise into out_dir/data. JPEGs without an EOI marker at the
     end (bytes after it, typically) are copied, not linked: Ultralytics
     re-saves such a file in place, which through a symlink would rewrite the
     source photograph. (A JPEG cut short before its EOI cannot be decoded,
     so the guard has already refused it as unhashable.);
  7. training with every recipe setting passed explicitly (optimizer above
     all: 'auto' ignores lr0), project=out_dir, name='train', exist_ok=True,
     val=False, plots=False. Ultralytics still validates at the final epoch
     (and final_eval validates best.pt) on data.yaml's 'val', so 'val' points
     at VAL_SUBSET_SIZE images of the training set itself, in
     data/val_subset: those numbers only reach Ultralytics' own logs, no
     checkpoint is chosen on them (last.pt is scored, best.pt deleted), and no
     evaluation image is read. cache 'ram' is kept only while Ultralytics'
     own estimate of the cache (sampled images, +50%) fits in CACHE_RAM_SHARE
     of the job's memory limit (Slurm's, or the cgroup's): Ultralytics
     compares it with the node's free memory, which on a shared GPU node can
     exceed the job's --mem, and the cgroup then kills the run without a
     run.json. A downgrade to no cache (the same pixels, read each epoch) is
     recorded in run.json 'cache';
  8. the scored weights are copied from the trainer's own save_dir (never a
     hard-coded path) to weights/final.pt: last.pt for full / freeze,
     last_merged.pt for lora; they must hold the final epoch. Ultralytics
     skips a save when the EMA holds NaN/Inf (8.4.37), leaving an earlier
     epoch there, which is not the final annealed weights the protocol
     scores: such a run fails at stage train as diverged (run.json
     'diverged'). Then every *.pt under train/weights is deleted (final.pt is
     the scored copy; /ocean is tight on space);
  9. every exam is scored by the scorer as a SUBPROCESS; the executor never
     computes or copies a metric (run.json records each score file's path,
     sha256 and production stamp only);
 10. run.json is written atomically, done or failed (error = traceback,
     stage = where it failed). On the way out out_dir/data is first renamed
     aside (.trash-*), then run.json is written, and only then is the renamed
     dir deleted, so a SIGKILL during a long delete on Lustre cannot lose the
     record (a leftover .trash-* goes with the next attempt). A run dir then
     holds spec.json, run.json, attempt.json, the lock files,
     weights/final.pt, scores/ and the trainer's logs (train/: args.yaml,
     results.csv, lora.json).

Re-runs. A done run is a no-op unless --force when its spec is unchanged
(compared as canonical JSON, so a reformatted file is the same spec),
weights/final.pt still hashes to run.json's weights_sha256, what it was made
from still hashes as recorded (init and train manifest; the soup members; a
final run's init), and every score file is present and hashes as recorded.
Any other re-run is attempt n+1 (attempt.json keeps the count even when a run
was killed before writing run.json); the old run.json is removed when the
attempt starts, so a failed record never stands for a run in progress. A
training or soup run whose final.pt is verified that way, but whose scoring
failed or whose score files are missing or changed, re-scores that final.pt
instead of training again (--force retrains); the attempt keeps the file from
its first line, so an attempt that fails before scoring does not delete it,
and its failed run.json still names it for the next.

One executor per run dir, across nodes (RunDirLock): fcntl.flock on
.executor.lock, and .executor.owner created with O_EXCL, which the metadata
server serialises across nodes whatever the filesystem's flock mode (on Lustre
flock is coherent across nodes only when mounted with 'flock'). A busy flock,
or a live owner file after waiting up to LOCK_WAIT_SECONDS, exits 3 without
touching anything. An owner file is stale when its process on this host has
exited, or its heartbeat (touched every LOCK_HEARTBEAT_SECONDS) is older than
LOCK_STALE_SECONDS; a stale one is taken over (recorded). flock not supported
on the mount (ENOLCK, ENOSYS, ...) leaves the owner file alone in charge (a
warning); any other flock error is a failed run.json at stage lock (exit 1).
An attempt that finds its owner file taken over leaves the dir to the new
owner: it writes and removes nothing, and exits 3.

--flock-check PATH prints whether fcntl.flock on PATH's filesystem is
coherent across nodes, from /proc/self/mountinfo (exit 0 yes, 1 no or
unknown); run_inc_job.sh uses it to decide whether the driver may be advanced
from a compute node.

Testing. Production runs need CUDA, and the scorer subprocess never sees
INC_SCORER_TESTING. An experiment whose exp.json says "testing": true (the
driver's form), or "testing": {"imgsz", "batch", "device", "lock_check"},
runs in test mode only when INC_SCORER_TESTING=1 is also set (else refused).
Then training uses testing.device, else the GPU if there is one, else the CPU;
the scorer gets testing's imgsz / batch / lock_check (default: the protocol's)
and runs on testing.device, else the CPU, so that every score of a testing
experiment departs from the protocol and is stamped TEST-, even on a GPU node;
a testing run whose score comes back production=true fails. run.json says
testing=true. Nothing here ever sets the variable.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import errno
import fcntl
import hashlib
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

from . import common as C
from . import scorer as S

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

# The protocol's recipe (INCREMENTAL_PROTOCOL.md "Fixed conventions"; the
# runner doc's "Recipes" table for warmup_bias_lr, freeze and lora). Keys it
# does not fix (momentum, weight_decay, close_mosaic, cache, workers, seed and
# the cold warmup_bias_lr) are not checked.
PROTOCOL_ALL = {"optimizer": "SGD", "imgsz": 640, "batch": 32, "deterministic": True, "cos_lr": True,
                "lrf": 0.01}
PROTOCOL_COLD = {"trainer": "full", "epochs": 100, "lr0": 0.01, "warmup_epochs": 3}
PROTOCOL_INC = {"epochs": 30, "warmup_epochs": 1}
PROTOCOL_INC_LR0 = {"full": 0.002, "freeze": 0.002, "lora": 0.01}
PROTOCOL_FREEZE = 11
PROTOCOL_LORA = {"rank": 16, "alpha": 32}

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

# Modules a run imports whose drift from the git-tracked copy matters.
CODE_MODULES = ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/train.py",
                "tools/inc/lora.py", "tools/inc/scorer.py", "tools/cwd12_species.py",
                "tools/near_dup.py", "tools/mega_trainer.py")

# Fields carried into a resumed attempt's run.json from the attempt that trained.
CARRY_OVER = ("init", "init_sha256", "train_manifest", "train_manifest_sha256", "n_train_images",
              "n_train_boxes", "train_class_counts", "train_sources", "duplicate_label_rows",
              "duplicate_images", "dhash0_collisions", "guard", "materialised", "dataset_check",
              "train_dir", "train_kwargs", "trainable_params", "total_params", "weights_epoch", "lora",
              "cache", "amp_check_weights", "protocol_recipe", "recipe_deviations", "soup", "soup_of",
              "soup_sha256", "train_seconds", "weights_source")


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
    print("[inc.train] %s" % msg, flush=True)


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


def protocol_deviations(kind, r):
    """How recipe r departs from the protocol's recipe for a run of this kind
    ([] when it does not)."""
    want = dict(PROTOCOL_ALL)
    if kind in COLD_KINDS:
        want.update(PROTOCOL_COLD)
    else:
        lr0 = PROTOCOL_INC_LR0.get(r.get("trainer"))
        want.update(PROTOCOL_INC, lr0=lr0, warmup_bias_lr=lr0)
        if r.get("trainer") == "freeze":
            want["freeze"] = PROTOCOL_FREEZE
        if r.get("trainer") == "lora":
            want["lora"] = PROTOCOL_LORA
    out = []
    for k in sorted(want):
        v, got = want[k], r.get(k)
        same = (got is v) if isinstance(v, bool) or isinstance(got, bool) else got == v
        if not same:
            out.append("%s %r (protocol %r)" % (k, got, v))
    return out


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


# ------------------------------------------------------------------ the code
def package_dir():
    """The weed_optimizer_framework dir this module was imported from."""
    return Path(__file__).resolve().parents[2]


def code_provenance():
    """({module: sha256} of the running copy, [drifted modules]). The nested
    git-tracked copy is REPO/weed_llm_benchmark/weed_optimizer_framework; when
    that is not what runs, every CODE_MODULES file must match it."""
    running = package_dir()
    nested = C.REPO / "weed_llm_benchmark" / "weed_optimizer_framework"
    code, drift = {}, []
    compare = nested.is_dir() and nested.resolve() != running
    for m in CODE_MODULES:
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
    for split in C.EVAL_SPLITS:
        if manifest == C.manifest_path(split).resolve():
            raise RunError("manifest", "%s is the %s split's manifest; it is never trained on" % (manifest, split))
    if not manifest.is_file():
        raise RunError("manifest", "train manifest not found: %s" % manifest)
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


def guard_rows(rows, dhashes):
    """NeverTrainGuard.load().assert_trainable over every image (fail closed).
    Returns the guard record; raises RunError('guard') with it attached."""
    paths = [r["image"] for r in rows]
    try:
        guard = C.NeverTrainGuard.load()
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise RunError("guard", "cannot load the never-train index %s: %s" % (C.NEVER_TRAIN_INDEX, e))
    rec = {"index": str(C.NEVER_TRAIN_INDEX), "index_images": guard.n, "checked": len(paths),
           "bits": C.HOLDOUT_NEAR_DUP_BITS}
    lookup = dhashes.__getitem__
    try:
        guard.assert_trainable(paths, hash_fn=lookup)
    except RuntimeError as e:
        hits, unhashable = guard.check(paths, hash_fn=lookup)
        rec.update(hits=len(hits), unhashable=len(unhashable),
                   first_hits=[list(h) for h in hits[:10]], first_unhashable=unhashable[:10])
        err = RunError("guard", str(e))
        err.guard = rec
        raise err
    rec.update(hits=0, unhashable=0)
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
           "total_params": None, "guard": None, "seconds": None}
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
    if kind in TRAIN_KINDS:
        devs = protocol_deviations(kind, spec["recipe"])
        rec["protocol_recipe"], rec["recipe_deviations"] = not devs, devs
        if devs and testing is None:
            raise RunError("recipe", "a production %s run trains the protocol's recipe only; this one "
                           "departs from it in %s" % (kind, devs))

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
        rec["guard"] = guard_rows(rows, dhashes)
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
