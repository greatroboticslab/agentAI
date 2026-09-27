"""The INC driver: runs an experiment's definition forward, one idempotent call
at a time (docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Driver").

    python -m weed_optimizer_framework.tools.inc.driver init    --exp EXP [--definition FILE]
    python -m weed_optimizer_framework.tools.inc.driver advance --exp EXP [--quiet]
    python -m weed_optimizer_framework.tools.inc.driver status  --exp EXP
    python -m weed_optimizer_framework.tools.inc.driver unblock --exp EXP (--unit U ... | --all) --reason TEXT
    python -m weed_optimizer_framework.tools.inc.driver repin   --exp EXP --reason TEXT
    python -m weed_optimizer_framework.tools.inc.driver watch   --exp EXP [--interval S] [--max-hours H]

Everything lives in INC_DIR/<exp>/:
    exp.json         the definition (written once, by init, from a builder such
                     as inc/pilot.py); an experiment is never redefined
    state.json       progress, written atomically; carries a generation counter
    ledger.jsonl     append-only: every gate, soup and truth decision with the
                     score files it read (path and sha256) and the sha256 of the
                     driver's code; unblocks and code re-pins
    runs/<run_id>/   spec.json, and what the executor (inc/train.py) writes:
                     run.json, attempt.json, scores/<exam>.json, weights/final.pt
    manifests/       the replay samples and the training manifests of each run
    submissions/     one list file per sbatch array (line i+1 = task i's spec)
    logs/            Slurm logs

Exclusion. Every array task calls advance when it ends, on whatever node it
ran, so concurrent passes are the normal case. A pass holds two things:
  * a non-blocking fcntl.flock on state.json.lock (it excludes passes on one
    node; a mount without flock support is tolerated);
  * the lease advance.lease, created with O_CREAT|O_EXCL (atomic on Lustre and
    NFS, independent of fcntl): a random token, host, pid and an expiry
    (LEASE_SECONDS, renewed as the pass goes). A lease past its expiry, or one
    whose holder pid on this host is dead, is broken.
Before every state save, ledger append and sbatch, the pass checks that the
lease still holds its token and that state.json is still at the generation
it loaded; otherwise it stops without saving (ConcurrentPass). A caller that
finds the lock or lease held returns at once, leaving a request file; the
holder runs another pass after releasing, so no finished run is left
uncollected.

A pass (advance):
  0. repairs a ledger whose last line is partial (an advance killed
     mid-append): it is cut back to the last complete entry and the fragment is
     kept beside it; checks the driver's code against the hashes pinned in
     state.json at init and against the git-tracked nested copy (below);
  1. resolves submissions left 'submitting' (a pass died after sbatch, or
     sbatch timed out or answered without a job id): the job is looked up by
     name among jobs submitted after the submission was created (squeue, then
     sacct). Found -> tracked. Slurm unreachable -> left for the next pass.
     Definitely unknown to Slurm and older than SUBMIT_GRACE_SECONDS -> 'lost',
     and its runs are queued again. Only an sbatch that never started
     (SubmitNotStarted) requeues at once. A run whose submissions came to
     nothing MAX_SUBMIT_FAILURES times (a rejected account, a missing job
     script) fails with the last sbatch error, which blocks its unit;
  2. collects: a submitted run is complete when run.json says done, every exam
     in its spec has scores/<exam>.json and weights/final.pt exists. A failed
     run (run.json failed or incomplete; or its Slurm task ended without a
     run.json; or it was submitted more than 8 h ago, is absent from squeue,
     unknown to sacct or not pending/running there, and has no run.json) is
     resubmitted once, as the same spec in the same run dir: the executor
     counts the attempt (attempt.json), removes the stale run.json when the
     retry starts, and re-scores instead of retraining when only scoring
     failed. The driver skips a run.json whose sha256 it has already counted
     as a failure, whatever the branch; the failed attempt's seconds are kept
     in state.json. Before a task that ended without run.json is counted as a
     failure, the run dir is asked who holds it: the executor's owner file
     (.executor.owner, heartbeat younger than OWNER_FRESH_SECONDS), then
     attempt.json. When that names another Slurm job still pending or running
     (a duplicate task exits 3 when it finds the run dir held by that job),
     the run is re-tracked to that job; a fresh owner outside Slurm is waited
     for; nothing is counted either way. After a second failure the run's
     unit (a chain, the truth arm, the base runs or the final runs) is blocked
     with the error. A run marked failed whose run.json later says done (with
     its scores and weights) is recovered, and a block caused by it is lifted
     (ledger 'unblock', auto);
  3. advances every unit that is ready (experiment types below). A score file
     that cannot be read (OSError) is retried on the next pass; only after
     TRANSIENT_LIMIT passes in a row is the unit blocked. A gate refusal
     (ValueError / KeyError / TypeError) blocks at once;
  4. submits every newly created spec as ONE sbatch array (%40): the
     submission is saved as 'submitting' before sbatch runs, then the job id
     and every array index -> run_id are recorded. An array holding a cold
     (base / union, 100-epoch) run is given --time COLD_TIME_LIMIT; others keep
     the job script's 3 h.

Experiment types:
  baseline  base runs (cold, one per seed, scored on dev); when all are
            complete, final runs of each on every exam, test included.
  chain     base runs on P0 (cold, one per seed, on dev). Each recipe is an
            independent chain whose incumbent starts as base seed 0. For step
            k (1-based) of the sequence, from the chain's accepted pool
            (P0 + accepted increments), by exp.json's replay_mode:
              'sample' (the default; a definition without the key): replay
              R1 and R2, |D_k| images each, disjoint, drawn with
              numpy.random.default_rng(stable_int("<exp>/<recipe>/<k>"));
              cand s0..s2 on D_k + R1 and null s0..s2 on R1 + R2;
              'full' (full rehearsal): no sampling; cand s0..s2 on the whole
              accepted pool + D_k and null s0..s2 on the whole accepted pool;
              the step and its gate entry record replay_mode 'full', the
              pool's size and the accepted increments it holds (a
              sample-mode step records exactly what it did before);
            all from the incumbent with the chain's recipe, scored on
            dev; then gate.decide(incumbent dev, cands, nulls). ACCEPT runs a
            soup of the three cand weights, gate.choose_soup picks soup or cand
            s0 as the new incumbent and D_k joins the accepted pool. HOLD and
            REJECT keep the incumbent. The truth arm (exp.truth) is created on
            the first pass (init), all of it or none: per step, cold union runs
            on T_{k-1} + D_k, 3 seeds; T_0 = P0 and T_k grows only by clean
            increments; "without" is the "with" runs of the last clean step
            before k, or the base runs. Its decision is gate.truth_detail. When
            every chain has finished every step and every truth decision is
            made, final runs score each chain's incumbent, the base seeds and
            the T_final runs on every exam, test included; then
            state.done = true.

init refuses a definition whose manifests do not hash as recorded, or whose
base and steps share an image (by key, path or image sha256), before anything
is written; a chain needs at least min_seeds seeds (its gate config's), and
its truth_recipe must be the base recipe (the base runs are step 1's
"without"); replay_mode, when present, is 'sample' or 'full' and only in a
chain; gate, when present, is a valid gate block and only in a chain.

Gate config. exp.json's optional "gate" block holds GateConfig fields
(GATE_KEYS: every field but require_production, which "testing" sets); init
refuses an unknown key, a value of the wrong type or range, a block that is
not an object and a block on a baseline. Every gate.decide, choose_soup and
truth_detail call of the experiment is made with gate_config(defn):
GateConfig(**block), require_production off for a testing experiment. A
definition without the block (pilot_v1, pilot_v2, b0_v1, base_b_v1), with {}
or with {"flips_mode": "negative"} is decided exactly as before the block
existed: gate.GateConfig's defaults, and a v1 decision records no flips_mode.
{"flips_mode": "net"} is protocol v2 (docs/INCREMENTAL_PROTOCOL.md, Gate,
"Protocol v2 (net flips)"). The config is fixed at init: a definition with a
block pins it in state.json (GATE_PIN: the block and the whole resolved
config, flips_mode written out) and the ledger ('gate_pin/0'); a definition
without one pins nothing and writes exactly the state and ledger it always
did, its pin being GateConfig's defaults (pinned_gate_config). A pass whose
exp.json resolves to another config (an edited gate block or testing flag)
refuses to run; there is no operation that changes the gate of a running
experiment (build a new one). status marks a chain whose config is not the
defaults, e.g. "(gate flips net)", and says when exp.json no longer matches
the pin.

Code pinning. init records in state.json the sha256 of PINNED_MODULES (the
gate, this driver, common, splits, cwd12_species). A pass refuses to run when
the running files differ from the pins, or (when the running package is not
the git-tracked nested copy, as on the cluster's compute nodes) from their
nested twins; INC_ALLOW_DRIFT=1 lets the second through, recorded in every
ledger entry. Every ledger entry carries the module hashes. 'repin' accepts
new code explicitly, with a reason, as a ledger entry. The gate block (above)
changed gate.py and this driver: new experiments pin the new hashes at init;
pilot_v1, pilot_v2, b0_v1 and base_b_v1 are finished and are not re-advanced
or re-pinned (an advance of a done experiment returns before the code check).

Operations. 'unblock' lifts the named blocks: failed runs of the unit (and
the run that caused the block) go back to the queue with MAX_ATTEMPTS more
tries, recorded in the ledger. 'watch' calls advance every --interval seconds
until the experiment is done (run it under nohup or tmux on the login node, or
as a scrontab entry calling 'advance --quiet'): job-end advances alone stall
when the last running task's advance cannot submit, and run_inc_job.sh skips
them altogether when it cannot confirm cross-node flock (INC_JOB_ADVANCE=auto).
The driver itself does not need flock across nodes (the lease above), so an
in-job advance is safe on any mount.

Attribution. The driver records gate.decide's attribution (protocol steps
1-3). Protocol steps 4 (BioCLIP-2 label audit) and 5 (leave-one-source-out
runs for a multi-source increment) are not run; every gate entry lists them
under attribution_not_run, with the increment's sources.

Only a 'final' spec may list the sealed test exam; the driver refuses to write
any other, and final runs exist only once everything they report on is done.

Testing: an experiment is gated on test-mode scores (GateConfig(
require_production=False)) only when exp.json's "testing" is set (true, or the
executor's settings object {imgsz, batch, device, lock_check}) AND
INC_SCORER_TESTING=1; a testing experiment without the variable refuses to
advance. state.json, every ledger entry and every status line of such an
experiment say TESTING.

Where the built modules refine the runner doc, the module is followed:
  * gate.decide / choose_soup / truth_detail with gate_config(defn): their schema
    checks refuse test-stamped scores in production, and a refusal
    (ValueError / KeyError / TypeError) blocks the unit instead of deciding;
    the truth verdicts are gate's helps / hurts / neutral;
  * inc/train.py's spec rules: a soup spec carries soup_of and no init, a
    final spec init and no train_manifest, and only training specs a recipe
    (every RECIPE_KEYS key, seed included);
  * inc/train.py's re-runs (above) instead of moving a failed attempt aside;
  * inc/train.py's attempt.json and .executor.owner (slurm_job_id = the
    task's raw SLURM_JOB_ID; the owner file's heartbeat) and its exit 3 for a
    run dir another executor holds.
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import dataclasses
import datetime
import errno
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import common as C
from . import gate as G
from .scorer import TEST_ENV

SEEDS = (0, 1, 2)
MAX_ATTEMPTS = 2                      # a failed run is resubmitted once
STALE_SECONDS = 8 * 3600              # submitted, not queued, no run.json -> failed
ARRAY_CONCURRENCY = 40
MAX_PASSES = 20                       # passes per call when other callers leave requests
COLD_TIME_LIMIT = "08:00:00"          # sbatch --time for an array with a 100-epoch cold run
COLD_INIT = "yolo11n.pt"              # resolved against REPO by the executor
LEASE_SECONDS = 1800                  # an unrenewed advance lease is broken after this
SUBMIT_GRACE_SECONDS = 900            # an sbatch Slurm has not heard of is 'lost' only after this
LOOKUP_SLACK_SECONDS = 300            # clock skew allowed when matching a job's submit time
TRANSIENT_LIMIT = 12                  # passes in a row a unit may fail to read its inputs
MAX_SUBMIT_FAILURES = 3               # sbatch never ran / Slurm never heard of it -> the run fails
WATCH_INTERVAL = 600
DECISION_EXAM = G.DECISION_EXAM       # "dev"
SEALED_EXAM = "test"
FINAL_EXAMS = ("dev", "ood22", "ood23", "imageweeds", "test")
EXP_TYPES = ("baseline", "chain")
REPLAY_MODES = ("sample", "full")     # exp.json "replay_mode" of a chain; absent = "sample"
DEFAULT_REPLAY_MODE = "sample"
FLIPS_MODES = G.FLIPS_MODES           # exp.json "gate": {"flips_mode"}; absent = "negative" (protocol v1)
DEFAULT_FLIPS_MODE = G.DEFAULT_FLIPS_MODE
# exp.json "gate" may set any GateConfig field but require_production, which "testing" decides
GATE_KEYS = tuple(f.name for f in dataclasses.fields(G.GateConfig) if f.name != "require_production")
GATE_METRICS = ("map50_95", "map50")  # gate.METRIC_KEYS that have an "agnostic_" twin
GATE_PROBABILITIES = ("p_accept", "p_reject", "p_recipe_flag")
GATE_PIN = "gate_pin"                 # state.json key: the gate config fixed at init (definitions with a block)
TRAIN_KINDS = ("base", "union", "cand", "null")
KINDS = TRAIN_KINDS + ("soup", "final")
SPEC_KEYS = ("exp", "run_id", "kind", "init", "train_manifest", "soup_of", "recipe", "exams", "out_dir")
RECIPE_KEYS = ("trainer", "epochs", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
               "warmup_epochs", "warmup_bias_lr", "cos_lr", "freeze", "lora", "imgsz", "batch",
               "seed", "cache", "workers", "close_mosaic", "deterministic")
TRAINERS = ("full", "freeze", "lora")
TESTING_KEYS = ("imgsz", "batch", "device", "lock_check")   # inc/train.py's test-mode settings
COLD_KINDS = ("base", "union")
LIVE = ("queued", "submitting", "submitted")
SLURM_ENDED = frozenset({"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
                         "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"})
VERDICT_TO_TRUTH = {G.ACCEPT: G.HELPS, G.HOLD: G.NEUTRAL, G.REJECT: G.HURTS}
# Relative to the weed_optimizer_framework package: the code whose version decides.
PINNED_MODULES = ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/driver.py",
                  "tools/inc/gate.py", "tools/inc/splits.py", "tools/cwd12_species.py")
ALLOW_DRIFT_ENV = "INC_ALLOW_DRIFT"
OWNER_FILE = ".executor.owner"        # inc/train.py's OWNER_NAME: the live executor of a run dir
OWNER_FRESH_SECONDS = 300             # inc/train.py's LOCK_STALE_SECONDS (heartbeat every 30 s)
ATTRIBUTION_NOT_RUN = {
    "label_audit": "protocol attribution step 4 (BioCLIP-2 re-reads D_k's boxes) is not run by the driver",
    "leave_one_source_out": "protocol attribution step 5 (leave-one-source-out cand runs, 3 seeds each, "
                            "for an increment that mixes sources) is not run by the driver",
}
FOUND, ABSENT, UNKNOWN = "found", "absent", "unknown"      # Backend.lookup outcomes
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_PARSABLE = re.compile(r"(\d+)(?:;\S*)?")


class DriverError(RuntimeError):
    """A condition under which the driver must not go on."""


class SubmitNotStarted(DriverError):
    """sbatch never ran (no job script, no sbatch binary): nothing can be queued."""


class SubmitUncertain(DriverError):
    """sbatch ran but gave no job id: the array may or may not be queued."""


class ConcurrentPass(DriverError):
    """Another pass took the lease or saved state while this one ran."""


def log(msg):
    print("[inc.driver] %s" % msg, flush=True)


def _utc(ts=None):
    t = datetime.datetime.fromtimestamp(time.time() if ts is None else ts, datetime.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(utc):
    """Epoch seconds of a _utc() string, or None."""
    try:
        return float(calendar.timegm(time.strptime(str(utc), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _short(text, n=400):
    text = str(text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def job_script():
    """The sbatch array script: $INC_JOB_SCRIPT, else REPO/weed_llm_benchmark/run_inc_job.sh
    (the git-tracked copy on the cluster checkout)."""
    return Path(os.environ.get("INC_JOB_SCRIPT", str(C.REPO / "weed_llm_benchmark" / "run_inc_job.sh")))


def check_name(name, what="name"):
    if not isinstance(name, str) or not _SAFE.match(name):
        raise DriverError("%s %r is not path-safe ([A-Za-z0-9][A-Za-z0-9_-]*)" % (what, name))
    return name


# ------------------------------------------------------------------- code
def package_dir():
    """The weed_optimizer_framework dir this module was imported from."""
    return Path(__file__).resolve().parents[2]


def driver_code():
    """({package_dir, nested_dir, modules: {module: sha256}}, [drifted modules]).
    The git-tracked copy is REPO/weed_llm_benchmark/weed_optimizer_framework;
    when another copy runs (the cluster's job script imports the outer one),
    every PINNED_MODULES file must match its nested twin."""
    running = package_dir()
    nested = C.REPO / "weed_llm_benchmark" / "weed_optimizer_framework"
    compare = nested.is_dir() and nested.resolve() != running
    mods, drift = {}, []
    for m in PINNED_MODULES:
        p = running / m
        mods[m] = C.sha256_file(p) if p.is_file() else None
        if compare:
            q = nested / m
            if mods[m] is None or not q.is_file() or C.sha256_file(q) != mods[m]:
                drift.append(m)
    return {"package_dir": str(running), "nested_dir": str(nested) if compare else None,
            "modules": mods}, drift


def _drift_error(info, drift):
    return DriverError("the running package %s differs from the git-tracked copy %s in %s; sync the outer "
                       "copy (or export %s=1, which every ledger entry then records)"
                       % (info["package_dir"], info["nested_dir"], drift, ALLOW_DRIFT_ENV))


# ------------------------------------------------------------------ paths
class Paths:
    def __init__(self, exp):
        self.exp = check_name(exp, "experiment")
        self.root = Path(C.INC_DIR) / exp
        self.exp_json = self.root / "exp.json"
        self.state = self.root / "state.json"
        self.lock = self.root / "state.json.lock"
        self.lease = self.root / "advance.lease"
        self.request = self.root / "advance.request"
        self.ledger = self.root / "ledger.jsonl"
        self.runs = self.root / "runs"
        self.manifests = self.root / "manifests"
        self.submissions = self.root / "submissions"
        self.logs = self.root / "logs"

    def run_dir(self, rid):
        return self.runs / rid

    def spec(self, rid):
        return self.runs / rid / "spec.json"

    def run_json(self, rid):
        return self.runs / rid / "run.json"

    def attempt_json(self, rid):
        return self.runs / rid / "attempt.json"

    def owner(self, rid):
        return self.runs / rid / OWNER_FILE

    def score(self, rid, exam):
        return self.runs / rid / "scores" / ("%s.json" % exam)

    def weights(self, rid):
        return self.runs / rid / "weights" / "final.pt"


# ----------------------------------------------------------- definitions
def validate_recipe(recipe, where):
    """A recipe as exp.json holds it: every RECIPE_KEYS key but seed (the driver
    sets it per run), an explicit optimizer, and trainer consistent with
    freeze / lora."""
    if not isinstance(recipe, dict):
        raise DriverError("%s: recipe must be a dict" % where)
    want = set(RECIPE_KEYS) - {"seed"}
    if set(recipe) != want:
        raise DriverError("%s: recipe keys differ from the run spec's: missing %s, unknown %s"
                          % (where, sorted(want - set(recipe)), sorted(set(recipe) - want)))
    if recipe["trainer"] not in TRAINERS:
        raise DriverError("%s: trainer %r not in %s" % (where, recipe["trainer"], TRAINERS))
    if not isinstance(recipe["optimizer"], str) or recipe["optimizer"].lower() == "auto":
        raise DriverError("%s: the optimizer must be explicit, never 'auto'" % where)
    if (recipe["trainer"] == "lora") != (recipe["lora"] is not None):
        raise DriverError("%s: trainer is 'lora' iff recipe.lora is set" % where)
    if recipe["lora"] is not None and set(recipe["lora"]) != {"rank", "alpha"}:
        raise DriverError("%s: lora must be {rank, alpha}" % where)
    fr = recipe["freeze"]
    if (recipe["trainer"] == "freeze") != (fr is not None):
        raise DriverError("%s: trainer is 'freeze' iff recipe.freeze is set" % where)
    if fr is not None and (not isinstance(fr, int) or isinstance(fr, bool) or fr < 1):
        raise DriverError("%s: freeze must be a positive int (layers 0..n-1 frozen)" % where)
    return recipe


def _check_manifest_entry(entry, where):
    for k in ("name", "manifest", "manifest_sha256", "n_images"):
        if k not in entry:
            raise DriverError("%s: missing %r" % (where, k))
    check_name(entry["name"], "%s name" % where)
    if not Path(entry["manifest"]).is_absolute():
        raise DriverError("%s: manifest must be an absolute path" % where)


def validate_definition(defn):
    """Raise unless defn is a complete baseline or chain definition."""
    if not isinstance(defn, dict):
        raise DriverError("definition must be a dict")
    check_name(defn.get("exp"), "experiment")
    typ = defn.get("type")
    if typ not in EXP_TYPES:
        raise DriverError("type %r not in %s" % (typ, EXP_TYPES))
    seeds = defn.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) < 1 or len(set(seeds)) != len(seeds)
            or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)):
        raise DriverError("seeds must be distinct non-negative ints")
    t = defn.get("testing", False)
    if not (isinstance(t, bool) or (isinstance(t, dict) and not set(t) - set(TESTING_KEYS))):
        raise DriverError("testing must be true / false or an object with keys %s" % (TESTING_KEYS,))
    iw = defn.get("init_weights", COLD_INIT)
    if not isinstance(iw, str) or not iw:
        raise DriverError("init_weights must be a weights path or name")
    base = defn.get("base")
    if not isinstance(base, dict):
        raise DriverError("base must be a dict")
    _check_manifest_entry(base, "base")
    validate_recipe(base.get("recipe"), "base")
    exams = defn.get("final_exams")
    if not isinstance(exams, list) or not exams or DECISION_EXAM not in exams:
        raise DriverError("final_exams must list the exams, dev included")
    for e in exams:
        if e not in C.EVAL_SPLITS:
            raise DriverError("final exam %r is not an evaluation split %s" % (e, C.EVAL_SPLITS))
    if defn.get("decision_exam", DECISION_EXAM) != DECISION_EXAM:
        raise DriverError("decisions are taken on %r only" % DECISION_EXAM)
    if "replay_mode" in defn:
        if typ != "chain":
            raise DriverError("replay_mode belongs to a chain experiment; a %s has no replay" % typ)
        if defn["replay_mode"] not in REPLAY_MODES:
            raise DriverError("replay_mode %r not in %s" % (defn["replay_mode"], REPLAY_MODES))
    if "gate" in defn:
        if typ != "chain":
            raise DriverError("gate belongs to a chain experiment; a %s makes no gate decision" % typ)
        validate_gate(defn["gate"])
    if typ == "chain":
        min_seeds = gate_config(defn).min_seeds
        if len(seeds) < min_seeds:
            raise DriverError("a chain needs at least %d seeds (gate.decide refuses fewer); got %s"
                              % (min_seeds, seeds))
        steps = defn.get("steps")
        if not isinstance(steps, list) or not steps:
            raise DriverError("a chain needs steps")
        names = []
        for i, s in enumerate(steps):
            _check_manifest_entry(s, "step %d" % (i + 1))
            if not isinstance(s.get("clean"), bool):
                raise DriverError("step %s: clean must be a bool" % s["name"])
            names.append(s["name"])
        if len(set(names)) != len(names) or base["name"] in names:
            raise DriverError("step names must be unique and differ from the base's")
        recipes = defn.get("recipes")
        if not isinstance(recipes, dict) or not recipes:
            raise DriverError("a chain needs recipes")
        for r, rec in recipes.items():
            check_name(r, "recipe")
            if r in ("base", "truth", "final", "Tfinal"):
                raise DriverError("recipe name %r is reserved" % r)
            validate_recipe(rec, "recipe %s" % r)
        if not isinstance(defn.get("truth"), bool):
            raise DriverError("truth must be a bool")
        if defn["truth"]:
            validate_recipe(defn.get("truth_recipe"), "truth_recipe")
            if defn["truth_recipe"] != base["recipe"]:
                raise DriverError("truth_recipe must equal base.recipe: the base runs are step 1's "
                                  "'without' arm of the truth arm")
    return defn


def validate_gate(block):
    """Raise unless block is a valid exp.json "gate" block: an object whose keys
    are GATE_KEYS, each value of its GateConfig field's type and in range.
    Returns the GateConfig keyword arguments (floats as floats)."""
    if not isinstance(block, dict):
        raise DriverError("gate must be an object of GateConfig fields, got %s" % type(block).__name__)
    if "require_production" in block:
        raise DriverError("gate.require_production is not settable: exp.json's testing decides it "
                          "(a production experiment is never decided on test-mode scores)")
    unknown = sorted(str(k) for k in block if k not in GATE_KEYS)
    if unknown:
        raise DriverError("gate has unknown key(s) %s; allowed: %s" % (unknown, list(GATE_KEYS)))
    default = G.GateConfig()
    out = {}
    for k, v in block.items():
        want = getattr(default, k)
        if isinstance(want, str):
            if not isinstance(v, str):
                raise DriverError("gate.%s must be a string, got %r" % (k, v))
            allowed = FLIPS_MODES if k == "flips_mode" else GATE_METRICS if k == "metric" else None
            if allowed is not None and v not in allowed:
                raise DriverError("gate.%s %r not in %s" % (k, v, allowed))
            out[k] = v
        elif isinstance(want, int) and not isinstance(want, bool):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise DriverError("gate.%s must be a positive integer, got %r" % (k, v))
            out[k] = v
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DriverError("gate.%s must be a finite non-negative number, got %r" % (k, v))
            try:
                f = float(v)
            except OverflowError:
                raise DriverError("gate.%s must be a finite non-negative number, got an integer too large "
                                  "for a float (%d digits)" % (k, len(str(abs(v))))) from None
            if not math.isfinite(f) or f < 0:
                raise DriverError("gate.%s must be a finite non-negative number, got %r" % (k, v))
            if k in GATE_PROBABILITIES and f > 1:
                raise DriverError("gate.%s is a probability, got %r" % (k, v))
            out[k] = f
    p_accept, p_reject = out.get("p_accept", default.p_accept), out.get("p_reject", default.p_reject)
    if not p_reject < p_accept:
        raise DriverError("gate: p_reject (%r) must be below p_accept (%r)" % (p_reject, p_accept))
    try:
        G.GateConfig(**out)
    except (TypeError, ValueError) as e:
        raise DriverError("gate: %s" % e)
    return out


def gate_config(defn):
    """The GateConfig every gate.decide / choose_soup / truth_detail call of
    the experiment is made with: exp.json's gate block (absent = GateConfig()
    exactly), and require_production off only for a testing experiment."""
    kw = validate_gate(defn["gate"]) if "gate" in defn else {}
    if defn.get("testing"):
        kw["require_production"] = False
    return G.GateConfig(**kw)


def gate_pin_record(defn):
    """state.json's GATE_PIN for a definition with a gate block (None without
    one): the block as defined and the whole resolved config, flips_mode
    included, so the rule is recorded explicitly and not by absence."""
    if "gate" not in defn:
        return None
    return {"block": defn["gate"], "config": dataclasses.asdict(gate_config(defn))}


def pinned_gate_config(state):
    """The GateConfig an experiment was initialised with: state.json's GATE_PIN
    (written at init when exp.json has a gate block), else GateConfig's
    defaults with require_production off only for a testing experiment, which
    is what every experiment initialised without a block (pilot_v1, pilot_v2,
    b0_v1, base_b_v1 among them) has been decided with."""
    pin = state.get(GATE_PIN)
    if pin is None:
        return G.GateConfig(require_production=not state.get("testing"))
    try:
        return G.GateConfig(**pin["config"])
    except (KeyError, TypeError, ValueError) as e:
        raise DriverError("state.json's %s is not a GateConfig record: %s: %s" % (GATE_PIN, type(e).__name__, e))


def gate_changes(pinned, now):
    """{field: [pinned value, current value]} of every field that differs."""
    return {f.name: [getattr(pinned, f.name), getattr(now, f.name)] for f in dataclasses.fields(G.GateConfig)
            if getattr(pinned, f.name) != getattr(now, f.name)}


def gate_tag(cfg):
    """' (gate ...)' naming every field of cfg that is not the protocol's
    default ('flips net' for protocol v2, first), '' for the defaults."""
    base = G.GateConfig(require_production=cfg.require_production)
    parts = ["flips net"] if cfg.flips_mode == G.FLIPS_NET else []
    parts += ["%s=%s" % (f.name, getattr(cfg, f.name)) for f in dataclasses.fields(G.GateConfig)
              if f.name != "flips_mode" and getattr(cfg, f.name) != getattr(base, f.name)]
    return " (gate %s)" % ", ".join(parts) if parts else ""


def check_disjoint(pool_rows, d_rows, name, pool_name):
    """Raise when d_rows shares an image with pool_rows: the same key, image
    path or image sha256."""
    keys = {r["key"] for r in pool_rows}
    images = {r["image"] for r in pool_rows}
    shas = {r.get("sha256") for r in pool_rows} - {None, ""}
    clash = [r["key"] for r in d_rows
             if r["key"] in keys or r["image"] in images or r.get("sha256") in shas]
    if clash:
        raise DriverError("increment %s shares %d image(s) with %s, e.g. %s"
                          % (name, len(clash), pool_name, clash[:3]))


def check_definition_data(defn):
    """What init checks before writing anything: every manifest exists and
    hashes as recorded; for a chain, the base and every step are pairwise
    disjoint (so no T_k, accepted pool or replay sample can overlap an
    increment)."""
    entries = [defn["base"]] + list(defn.get("steps", []))
    rows = {}
    for e in entries:
        if not Path(e["manifest"]).is_file():
            raise DriverError("manifest %s does not exist" % e["manifest"])
        got = C.sha256_file(e["manifest"])
        if got != e["manifest_sha256"]:
            raise DriverError("manifest %s hashes to %s, the definition says %s"
                              % (e["manifest"], got[:12], e["manifest_sha256"][:12]))
        rows[e["name"]] = C.read_manifest(e["manifest"])
    if defn["type"] == "chain":
        base = defn["base"]["name"]
        steps = [s["name"] for s in defn["steps"]]
        for i, a in enumerate(steps):
            check_disjoint(rows[base], rows[a], a, "the base %s" % base)
            for b in steps[:i]:
                check_disjoint(rows[b], rows[a], a, "step %s" % b)
    return rows


def validate_spec(spec):
    """The run spec contract (docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Run spec")."""
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    if unknown:
        raise DriverError("spec has unknown keys %s" % unknown)
    for k in ("exp", "run_id", "kind", "exams", "out_dir"):
        if k not in spec:
            raise DriverError("spec lacks %r" % k)
    check_name(spec["run_id"], "run_id")
    kind = spec["kind"]
    if kind not in KINDS:
        raise DriverError("kind %r not in %s" % (kind, KINDS))
    if SEALED_EXAM in spec["exams"] and kind != "final":
        raise DriverError("%s: only a 'final' spec may list %r" % (spec["run_id"], SEALED_EXAM))
    for e in spec["exams"]:
        if e not in C.EVAL_SPLITS:
            raise DriverError("%s: exam %r is not an evaluation split" % (spec["run_id"], e))
    if (kind in TRAIN_KINDS) != ("train_manifest" in spec) or (kind in TRAIN_KINDS) != ("recipe" in spec):
        raise DriverError("%s: train_manifest and recipe go with the training kinds only" % spec["run_id"])
    if (kind == "soup") != ("soup_of" in spec):
        raise DriverError("%s: soup_of goes with kind soup only" % spec["run_id"])
    if (kind == "soup") == ("init" in spec):
        raise DriverError("%s: every kind but soup needs init; a soup takes none" % spec["run_id"])
    if kind == "soup" and len(spec["soup_of"]) < 2:
        raise DriverError("%s: a soup averages at least two checkpoints" % spec["run_id"])
    if "recipe" in spec:
        rec = dict(spec["recipe"])
        if not isinstance(rec.get("seed"), int):
            raise DriverError("%s: recipe.seed must be an int" % spec["run_id"])
        rec.pop("seed")
        validate_recipe(rec, spec["run_id"])
    return spec


def _comparable(defn):
    return {k: v for k, v in defn.items() if k not in ("created_utc", "initialised_utc")}


# -------------------------------------------------------------- replay
def replay_samples(pool_rows, n, seed_text):
    """(R1, R2): two disjoint samples of n rows from pool_rows, drawn with
    numpy.random.default_rng(stable_int(seed_text)) over the rows sorted by key,
    so the draw depends only on the pool's content. Raises if the pool holds
    fewer than 2n rows."""
    import numpy as np
    rows = sorted(pool_rows, key=lambda r: r["key"])
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise DriverError("the accepted pool holds duplicate keys")
    n = int(n)
    if n < 1:
        raise DriverError("an increment of %d images cannot be replayed against" % n)
    if 2 * n > len(rows):
        raise DriverError("the accepted pool (%d images) is too small for two disjoint replay "
                          "samples of %d" % (len(rows), n))
    perm = np.random.default_rng(C.stable_int(seed_text)).permutation(len(rows))
    return ([rows[i] for i in sorted(int(x) for x in perm[:n])],
            [rows[i] for i in sorted(int(x) for x in perm[n:2 * n])])


def full_rehearsal(pool_rows, d_rows):
    """(cand rows, null rows) of replay mode 'full': cand = the whole accepted
    pool + D_k, null = the whole accepted pool. No sampling; a manifest is
    written sorted by key, so the pool's order does not matter. Raises on a
    pool with duplicate keys or an empty increment."""
    keys = [r["key"] for r in pool_rows]
    if len(set(keys)) != len(keys):
        raise DriverError("the accepted pool holds duplicate keys")
    if not pool_rows:
        raise DriverError("the accepted pool is empty; there is nothing to rehearse")
    if not d_rows:
        raise DriverError("an increment of 0 images cannot be rehearsed against")
    return list(pool_rows) + list(d_rows), list(pool_rows)


# ------------------------------------------------------------------ lease
class Lease:
    """Mutual exclusion between advance passes on any nodes, independent of
    fcntl: a file created with O_CREAT|O_EXCL holding a random token, the
    holder's host and pid, and an expiry in wall-clock seconds (time.time()).

    acquire() takes a free lease, or breaks a stale one (expired; or held by a
    dead pid on this host; or unreadable and older than the lease period) by
    renaming it aside and checking that what was renamed is the stale lease
    it read. check() raises ConcurrentPass unless the file still holds this
    token, and renews the expiry when half of it is used."""

    def __init__(self, path, seconds=LEASE_SECONDS):
        self.path = Path(path)
        self.seconds = seconds
        self.token = None

    def _body(self):
        now = time.time()
        return {"token": self.token, "host": socket.gethostname(), "pid": os.getpid(),
                "acquired_utc": _utc(now), "expires_ts": now + self.seconds,
                "expires_utc": _utc(now + self.seconds)}

    @staticmethod
    def read(path):
        """{"body": dict or None, "mtime": float}, or None when there is no file."""
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
            with open(path) as fh:
                text = fh.read()
        except FileNotFoundError:
            return None
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        return {"body": body if isinstance(body, dict) else None, "mtime": mtime}

    def _stale(self, held):
        if held is None:
            return True
        body, now = held["body"], time.time()
        if body is None:                     # being written right now, or its writer died mid-write
            return now - held["mtime"] > self.seconds
        exp = body.get("expires_ts")
        if not isinstance(exp, (int, float)) or exp < now:
            return True
        if body.get("host") == socket.gethostname() and isinstance(body.get("pid"), int):
            try:
                os.kill(body["pid"], 0)
            except ProcessLookupError:
                return True
            except OSError:
                pass
        return False

    def _break(self, held):
        """True when the stale lease is gone (retry the create)."""
        if held is None:
            return True
        aside = self.path.with_name("%s.broken.%s" % (self.path.name, uuid.uuid4().hex[:12]))
        try:
            os.rename(self.path, aside)
        except FileNotFoundError:
            return True
        got = self.read(aside)
        same = (got is not None and got["body"] == held["body"]
                and (held["body"] is not None or got["mtime"] == held["mtime"]))
        if not same:                         # a fresh lease was renamed: put it back
            try:
                os.link(str(aside), str(self.path))
            except OSError:
                pass
            with contextlib.suppress(OSError):
                os.unlink(aside)
            return False
        with contextlib.suppress(OSError):
            os.unlink(aside)
        log("broke a stale advance lease %s (holder %s)" % (self.path, (held["body"] or {}).get("host")))
        return True

    def acquire(self):
        """True when this object now holds the lease, False when another does."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                held = self.read(self.path)
                if not self._stale(held) or not self._break(held):
                    return False
                continue
            self.token = uuid.uuid4().hex
            with os.fdopen(fd, "w") as fh:
                json.dump(self._body(), fh)
                fh.flush()
                os.fsync(fh.fileno())
            return True
        return False

    def check(self):
        held = self.read(self.path)
        body = (held or {}).get("body") or {}
        if self.token is None or body.get("token") != self.token:
            raise ConcurrentPass("this pass lost the advance lease %s (it now names %s on %s); another "
                                 "advance took over, so this one stops without saving"
                                 % (self.path, body.get("token"), body.get("host")))
        if body.get("expires_ts", 0) - time.time() < self.seconds / 2.0:
            tmp = self.path.with_name(".%s.%s.tmp" % (self.path.name, self.token[:12]))
            with open(tmp, "w") as fh:
                json.dump(self._body(), fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)

    def release(self):
        held = self.read(self.path)
        if held and held["body"] and self.token and held["body"].get("token") == self.token:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        self.token = None


# ------------------------------------------------------------- backends
class Backend:
    """How runs reach a GPU. submit() returns the array job id and raises
    SubmitNotStarted when sbatch never ran (anything else it raises leaves the
    outcome open); queued() the ids still in the queue, both 'jobid_index'
    and raw job ids (None when that cannot be known); task_states() {id:
    Slurm state} for the given array or raw job ids that the accounting knows
    (None when unknown); lookup(job_name, since_ts) (FOUND, job id) /
    (ABSENT, None) / (UNKNOWN, why) for a submission by name among jobs
    submitted after since_ts."""

    def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
        raise NotImplementedError

    def queued(self):
        raise NotImplementedError

    def task_states(self, job_ids):
        raise NotImplementedError

    def lookup(self, job_name, since_ts=None):
        raise NotImplementedError


def parse_parsable(out):
    """Job id from `sbatch --parsable` output ('123' or '123;cluster'), on any
    stdout line (warnings may come before or after it); the last one wins."""
    ids = [m.group(1) for m in (_PARSABLE.fullmatch(ln.strip()) for ln in str(out).splitlines()) if m]
    if not ids:
        raise SubmitUncertain("sbatch --parsable printed %r, not a job id" % _short(out, 200))
    return ids[-1]


def expand_task_ids(token):
    """'123_4' -> {'123_4'}; '123_[4-6,9%40]' -> {'123_4', '123_5', '123_6', '123_9'};
    a plain job id stays as it is."""
    token = token.strip()
    m = re.fullmatch(r"(\d+)_\[([0-9,\-]+)(?:%\d+)?\]", token)
    if not m:
        return {token} if token else set()
    out = set()
    for part in m.group(2).split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update("%s_%d" % (m.group(1), i) for i in range(int(a), int(b) + 1))
        elif part:
            out.add("%s_%d" % (m.group(1), int(part)))
    return out


def parse_squeue(text):
    """Every id on every line (`squeue -o '%i %A'`: the array task id and the
    raw job id)."""
    out = set()
    for ln in str(text).splitlines():
        for tok in ln.split():
            out |= expand_task_ids(tok)
    return out


def parse_sacct(text):
    """{id: state} from `sacct -n -X -P --format=JobID,JobIDRaw,State` (or
    JobID,State): every single array task under its 'jobid_index' id and its
    raw id, and plain jobs under their id ('CANCELLED by 42' -> 'CANCELLED');
    rows for still-unsplit pending ranges ('123_[4-9]') are skipped."""
    out = {}
    for ln in str(text).splitlines():
        parts = [p.strip() for p in ln.strip().split("|")]
        if len(parts) < 2 or not parts[-1]:
            continue
        jid, state = parts[0], parts[-1].split()[0]
        raw = parts[1] if len(parts) >= 3 else None
        if re.fullmatch(r"\d+(_\d+)?", jid):
            out[jid] = state
            if raw and raw.isdigit():
                out[raw] = state
    return out


def _local_ts(text):
    """Epoch seconds of a Slurm local time 'YYYY-MM-DDTHH:MM:SS', or None."""
    try:
        return time.mktime(time.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return None


def match_submitted(text, since_ts=None):
    """The base job id of the first 'id|submit time' row (squeue '%i|%V' or
    sacct JobID,Submit) submitted at or after since_ts (a row whose time does
    not parse is kept), else None."""
    for ln in str(text).splitlines():
        parts = ln.strip().split("|")
        m = re.match(r"(\d+)", parts[0].strip()) if parts and parts[0].strip() else None
        if not m:
            continue
        t = _local_ts(parts[1]) if len(parts) > 1 else None
        if since_ts is None or t is None or t >= since_ts:
            return m.group(1)
    return None


def submission_env(testing):
    """The environment sbatch runs with. SLURM_* / SBATCH_* of an enclosing job
    (the job script calls advance after its run) are dropped, so the new array
    takes only its own settings. INC_SCORER_TESTING reaches the jobs only for a
    testing experiment."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith(("SLURM_", "SBATCH_")) and k != "SLURM_CONF")}
    if not testing:
        env.pop(TEST_ENV, None)
    return env


class SlurmBackend(Backend):
    def __init__(self, script=None, user=None, timeout=180):
        self.script = Path(script) if script else job_script()
        self.user = user or os.environ.get("USER") or getpass.getuser()
        self.timeout = timeout

    def _run(self, argv, env=None):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, "", "%s: %s" % (type(e).__name__, e)
        return p.returncode, p.stdout, p.stderr

    def sbatch_argv(self, list_file, n, exp, job_name, log_dir, time_limit=None):
        return (["sbatch", "--parsable", "--array=0-%d%%%d" % (n - 1, ARRAY_CONCURRENCY),
                 "--job-name=%s" % job_name, "--output=%s" % (Path(log_dir) / "%x_%A_%a.out")]
                + (["--time=%s" % time_limit] if time_limit else [])
                + [str(self.script), str(list_file), exp])

    def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
        if not self.script.is_file():
            raise SubmitNotStarted("job script %s not found (set INC_JOB_SCRIPT)" % self.script)
        argv = self.sbatch_argv(list_file, n, exp, job_name, log_dir, time_limit)
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout, env=env)
        except OSError as e:
            raise SubmitNotStarted("sbatch could not be started: %s: %s" % (type(e).__name__, e))
        except subprocess.TimeoutExpired:
            raise SubmitUncertain("sbatch gave no answer within %d s" % self.timeout)
        if p.returncode != 0:          # e.g. 'Socket timed out on send/recv' after the job was queued
            raise SubmitUncertain("sbatch exited %d: %s" % (p.returncode, _short(p.stderr or p.stdout)))
        return parse_parsable(p.stdout)

    def queued(self):
        rc, out, _ = self._run(["squeue", "-h", "-r", "-u", self.user, "-o", "%i %A"])
        return parse_squeue(out) if rc == 0 else None

    def task_states(self, job_ids):
        job_ids = sorted(set(job_ids))
        if not job_ids:
            return {}
        rc, out, _ = self._run(["sacct", "-n", "-X", "-P", "-j", ",".join(job_ids),
                                "--format=JobID,JobIDRaw,State"])
        return parse_sacct(out) if rc == 0 else None

    def lookup(self, job_name, since_ts=None):
        since = None if since_ts is None else float(since_ts) - LOOKUP_SLACK_SECONDS
        rc, out, err = self._run(["squeue", "-h", "-u", self.user, "--name=%s" % job_name, "-o", "%i|%V"])
        if rc != 0:
            return UNKNOWN, "squeue exited %d: %s" % (rc, _short(err or out, 200))
        jid = match_submitted(out, since)
        if jid:
            return FOUND, jid
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since if since is not None
                                                                   else time.time() - 30 * 86400))
        rc, out, err = self._run(["sacct", "-n", "-X", "-P", "-u", self.user, "--name=%s" % job_name,
                                  "--format=JobID,Submit", "-S", start])
        if rc != 0:
            return UNKNOWN, "sacct exited %d: %s" % (rc, _short(err or out, 200))
        jid = match_submitted(out, since)
        return (FOUND, jid) if jid else (ABSENT, None)


class FakeBackend(Backend):
    """For tests: submit() records the array; run_pending() 'runs' every pending
    task by calling runner(spec_dict), which writes what the executor would
    (run.json, scores/<exam>.json, weights/final.pt) and returns the task's
    end state: 'COMPLETED' (default), 'FAILED', or 'LOST' (the task vanishes:
    absent from the queue and from sacct). Every task also has a raw job id
    (tasks[t]['raw'])."""

    def __init__(self, runner=None, first_job_id=1000, first_raw_id=50000):
        self.runner = runner
        self.submissions = []
        self.tasks = {}
        self._next = first_job_id
        self._raw = first_raw_id

    def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
        with open(list_file) as fh:
            specs = [ln.strip() for ln in fh if ln.strip()]
        if len(specs) != n:
            raise SubmitNotStarted("list file holds %d specs, array of %d" % (len(specs), n))
        jid = str(self._next)
        self._next += 1
        self.submissions.append({"job_id": jid, "job_name": job_name, "list_file": str(list_file),
                                 "n": n, "exp": exp, "specs": specs, "time_limit": time_limit,
                                 "testing_env": (env or {}).get(TEST_ENV)})
        for i, s in enumerate(specs):
            self.tasks["%s_%d" % (jid, i)] = {"spec": s, "state": "PENDING", "job_name": job_name,
                                              "raw": str(self._raw)}
            self._raw += 1
        return jid

    def queued(self):
        out = set()
        for t, v in self.tasks.items():
            if v["state"] in ("PENDING", "RUNNING"):
                out.update((t, v["raw"]))
        return out

    def task_states(self, job_ids):
        ids, out = set(job_ids), {}
        for t, v in self.tasks.items():
            if v["state"] != "LOST" and (t.split("_")[0] in ids or t in ids or v["raw"] in ids):
                out[t] = out[v["raw"]] = v["state"]
        return out

    def lookup(self, job_name, since_ts=None):
        for s in self.submissions:
            if s["job_name"] == job_name:
                return FOUND, s["job_id"]
        return ABSENT, None

    def pending(self):
        return sorted((t for t, v in self.tasks.items() if v["state"] == "PENDING"),
                      key=lambda t: tuple(int(x) for x in t.split("_")))

    def run_pending(self, runner=None, hold=None):
        """Run every pending task, except those whose spec hold(spec) is true
        for (they stay pending). Returns the number run."""
        runner = runner or self.runner
        done = 0
        for t in self.pending():
            spec = _read_json(self.tasks[t]["spec"])
            if hold is not None and hold(spec):
                continue
            self.tasks[t]["state"] = runner(spec) or "COMPLETED"
            done += 1
        return done


# ---------------------------------------------------------------- driver
class _Hold:
    def __init__(self, fh, lease):
        self.fh, self.lease = fh, lease


class Driver:
    def __init__(self, exp, backend=None, clock=time.time, quiet=False):
        self.paths = Paths(exp)
        self.exp = exp
        self.backend = backend
        self.clock = clock
        self.quiet = quiet
        self.defn = None
        self.state = None
        self.cfg = None
        self._ledger_ids = set()
        self._rows_cache = {}
        self._lease = None
        self._gen = None
        self._snapshot = None
        self._code = None

    # ------------------------------------------------------------ plumbing
    def _log(self, msg):
        if not self.quiet:
            log("%s: %s" % (self.exp, msg))

    def _backend(self):
        if self.backend is None:
            self.backend = SlurmBackend()
        return self.backend

    @property
    def testing(self):
        return bool(self.defn.get("testing"))

    @property
    def replay_mode(self):
        """exp.json's replay_mode; a definition without one (pilot_v1) is 'sample'."""
        return self.defn.get("replay_mode", DEFAULT_REPLAY_MODE)

    @property
    def runs(self):
        return self.state["runs"]

    def _load(self, for_advance=True):
        if not self.paths.exp_json.is_file():
            raise DriverError("%s does not exist; build the experiment and run init first"
                              % self.paths.exp_json)
        self.defn = validate_definition(_read_json(self.paths.exp_json))
        if self.defn["exp"] != self.exp:
            raise DriverError("%s defines experiment %r, not %r"
                              % (self.paths.exp_json, self.defn["exp"], self.exp))
        if not self.paths.state.is_file():
            raise DriverError("%s does not exist; run init" % self.paths.state)
        self.state = _read_json(self.paths.state)
        if for_advance:
            if self.testing and os.environ.get(TEST_ENV) != "1":
                raise DriverError("experiment %s is marked testing: it is gated on test-mode "
                                  "scores, which only happens with %s=1 set" % (self.exp, TEST_ENV))
            self.cfg = gate_config(self.defn)
            pinned = pinned_gate_config(self.state)
            if self.cfg != pinned:
                raise DriverError(
                    "exp.json's gate config differs from the one experiment %s was initialised with "
                    "(%s): %s as [pinned, now]. The gate config is fixed at init, so one experiment "
                    "is decided by one gate rule; a pass does not run on an edited gate block or testing flag. "
                    "Restore exp.json to the definition it was initialised with, or build a new experiment "
                    "under a new --exp name" % (self.exp, "state.json %s" % GATE_PIN if GATE_PIN in self.state
                                                  else "initialised without a gate block: the protocol's defaults",
                                                  gate_changes(pinned, self.cfg)))
            self._gen = int(self.state.get("generation", 0))
            self._snapshot = self._canonical()
            self._repair_ledger()
            self._ledger_ids = set()
            if self.paths.ledger.is_file():
                with open(self.paths.ledger) as fh:
                    for i, ln in enumerate(fh, 1):
                        if not ln.strip():
                            continue
                        try:
                            self._ledger_ids.add(json.loads(ln).get("id"))
                        except ValueError as e:
                            raise DriverError("%s line %d is not JSON (%s); only the last line can be cut "
                                              "by an interrupted append, and that is repaired "
                                              "automatically: inspect the file" % (self.paths.ledger, i, e))
            self._rows_cache = {}

    def _repair_ledger(self):
        """Cut a partial last line (an advance killed mid-append) back to the
        last complete entry; the fragment is kept beside the ledger."""
        p = self.paths.ledger
        if not p.is_file() or p.stat().st_size == 0:
            return
        with open(p, "rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
            self._fence()
            fh.seek(0)
            data = fh.read()
            cut = data.rfind(b"\n") + 1
            keep = self.paths.root / ("ledger.partial.%s.%s.txt"
                                      % (_utc().replace(":", "").replace("-", ""), uuid.uuid4().hex[:6]))
            with open(keep, "wb") as out:
                out.write(data[cut:])
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        log("%s: WARNING: %s ended in a partial line (%d bytes, an advance killed mid-append); cut back "
            "to the last complete entry, fragment kept in %s" % (self.exp, p, len(data) - cut, keep))

    def _fence(self):
        """Raise ConcurrentPass unless this pass still holds the lease and
        state.json is still at the generation it loaded."""
        if self._lease is None:
            raise DriverError("internal: state is written only under the advance lease")
        self._lease.check()
        try:
            disk = int(_read_json(self.paths.state).get("generation", 0))
        except (OSError, ValueError, TypeError) as e:
            raise ConcurrentPass("state.json cannot be re-read before writing: %s" % e)
        if disk != self._gen:
            raise ConcurrentPass("state.json is at generation %d but this pass loaded %d: another advance "
                                 "saved state meanwhile; this pass stops without saving" % (disk, self._gen))

    def _canonical(self):
        return json.dumps({k: v for k, v in self.state.items() if k not in ("generation", "updated_utc")},
                          sort_keys=True)

    def _save_state(self):
        """Write state.json (generation + 1) under the fence; a pass that
        changed nothing writes nothing."""
        self._fence()
        snap = self._canonical()
        if snap == self._snapshot:
            return
        self.state["generation"] = self._gen + 1
        self.state["updated_utc"] = _utc(self.clock())
        _write_json(self.paths.state, self.state)
        self._gen += 1
        self._snapshot = snap

    def _ledger(self, entry):
        if entry["id"] in self._ledger_ids:
            return False
        self._fence()
        entry = dict(entry, exp=self.exp, testing=self.testing, utc=_utc(self.clock()), code=self._code)
        self.paths.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.ledger, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._ledger_ids.add(entry["id"])
        return True

    @staticmethod
    def _score_inputs(pairs):
        """[(run_id, path)] -> ([score dicts], [{run_id, path, sha256}]); the
        sha256 is of the very bytes parsed. OSError / ValueError propagate."""
        scores, inputs = [], []
        for rid, path in pairs:
            with open(path, "rb") as fh:
                data = fh.read()
            scores.append(json.loads(data.decode("utf-8")))
            inputs.append({"run_id": rid, "path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        return scores, inputs

    def _dev(self, ids):
        return [(x, self.paths.score(x, DECISION_EXAM)) for x in ids]

    def _rows(self, entry):
        """A definition manifest's rows, after checking its sha256."""
        path = entry["manifest"]
        if path not in self._rows_cache:
            got = C.sha256_file(path)
            if got != entry["manifest_sha256"]:
                raise DriverError("manifest %s changed since the experiment was defined (%s != %s)"
                                  % (path, got[:12], entry["manifest_sha256"][:12]))
            self._rows_cache[path] = C.read_manifest(path)
        return self._rows_cache[path]

    def _write_manifest(self, path, rows):
        sha = C.write_manifest(path, rows)
        return {"path": str(path), "sha256": sha, "n_images": len(rows)}

    # ------------------------------------------------------------------ init
    def init(self, defn):
        """Write exp.json (once) and a fresh state.json, then advance: the base
        runs and the truth arm are created and submitted on that first pass."""
        defn = validate_definition(json.loads(json.dumps(defn)))
        if defn["exp"] != self.exp:
            raise DriverError("definition is for %r, not %r" % (defn["exp"], self.exp))
        info, drift = driver_code()
        if drift and os.environ.get(ALLOW_DRIFT_ENV) != "1":
            raise _drift_error(info, drift)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if self.paths.exp_json.exists():
            old = _read_json(self.paths.exp_json)
            if _comparable(old) != _comparable(defn):
                raise DriverError("%s exists with a different definition; an experiment is defined "
                                  "once (build it under a new --exp name)" % self.paths.exp_json)
        else:
            check_definition_data(defn)
            defn.setdefault("initialised_utc", _utc(self.clock()))
            _write_json(self.paths.exp_json, defn)
        if not self.paths.state.exists():
            _write_json(self.paths.state, self._new_state(defn, info))
        return self.advance()

    def _new_state(self, defn, code_info):
        st = {"exp": defn["exp"], "type": defn["type"], "testing": bool(defn.get("testing")),
              "created_utc": _utc(self.clock()), "updated_utc": _utc(self.clock()), "generation": 0,
              "code": {"modules": code_info["modules"], "package_dir": code_info["package_dir"],
                       "pinned_utc": _utc(self.clock()), "reason": "init"},
              "runs": {}, "submissions": [], "blocked": {}, "transient": {}, "unblocks": [],
              "repins": [], "done": False, "done_utc": None,
              "final": {"created": False, "runs": [], "sources": {}}}
        pin = gate_pin_record(defn)
        if pin is not None:
            # only with a gate block: a definition without one writes exactly the state it always did
            st[GATE_PIN] = pin
        if defn["type"] == "chain":
            st["chains"] = {r: {"phase": "start", "k": 0, "incumbent": None, "accepted": [],
                                "neutral": [], "quarantined": [], "steps": {}}
                            for r in defn["recipes"]}
            st["truth"] = {"enabled": bool(defn["truth"]), "steps": {}}
        return st

    # ------------------------------------------------------------- locking
    def _acquire(self):
        """A _Hold (flock handle or None, lease) or None when another caller
        holds either."""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        fh = open(self.paths.lock, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return None
            if e.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK, errno.EINVAL):
                raise DriverError("cannot lock %s: %s" % (self.paths.lock, e))
            self._log("fcntl.flock is not supported on %s (%s); the lease alone excludes other advances"
                      % (self.paths.lock.parent, e))
            fh = None
        lease = Lease(self.paths.lease)
        try:
            ok = lease.acquire()
        except OSError as e:
            self._unlock(fh)
            raise DriverError("cannot take the advance lease %s: %s" % (self.paths.lease, e))
        if not ok:
            self._unlock(fh)
            return None
        self._lease = lease
        return _Hold(fh, lease)

    @staticmethod
    def _unlock(fh):
        if fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _release(self, hold):
        try:
            hold.lease.release()
        except OSError as e:
            self._log("could not release the lease: %s" % e)
        self._lease = None
        self._unlock(hold.fh)

    @contextlib.contextmanager
    def _exclusive(self):
        hold = self._acquire()
        if hold is None:
            raise DriverError("another advance holds %s; try again in a moment" % self.paths.lease)
        try:
            yield
        finally:
            self._release(hold)

    # --------------------------------------------------------------- advance
    def advance(self):
        """One idempotent call: see the module docstring. Returns a summary dict;
        'locked' is True when another caller held the lock or lease."""
        if not self.paths.exp_json.is_file():
            raise DriverError("%s does not exist; build the experiment and run init first"
                              % self.paths.exp_json)
        result = {"exp": self.exp, "locked": False, "passes": 0, "submitted": 0, "job_ids": [],
                  "done": False, "lines": []}
        self.paths.request.touch()
        for _ in range(MAX_PASSES):
            hold = self._acquire()
            if hold is None:
                if result["passes"] == 0:
                    result["locked"] = True
                    self._log("another advance holds the lock; it will run a pass for this call")
                break
            try:
                with contextlib.suppress(FileNotFoundError):
                    self.paths.request.unlink()
                self._pass(result)
                result["passes"] += 1
            finally:
                self._release(hold)
            if not self.paths.request.exists():
                break
        if result["passes"]:
            result["lines"] = self.status_lines()
            result["done"] = bool(self.state.get("done"))
            if not self.quiet:
                for ln in result["lines"]:
                    log(ln)
        return result

    def _pass(self, result):
        self._load()
        if self.state.get("done"):
            return
        self._check_code()
        if GATE_PIN in self.state:
            self._ledger({"id": "gate_pin/0", "type": "gate_pin", "block": self.state[GATE_PIN]["block"],
                          "config": self.state[GATE_PIN]["config"]})
        self._recover_submissions()
        self._collect()
        self._ensure_base()
        if self.defn["type"] == "baseline":
            self._advance_baseline()
        else:
            self._advance_truth()
            for r in self.defn["recipes"]:
                self._advance_chain(r)
            self._advance_final_chain()
        self._save_state()
        self._submit_queued(result)
        self._save_state()

    def _check_code(self):
        info, drift = driver_code()
        if drift and os.environ.get(ALLOW_DRIFT_ENV) != "1":
            raise _drift_error(info, drift)
        self._code = {"modules": info["modules"], "drift": drift}
        pinned = self.state.get("code")
        if pinned is None:
            self.state["code"] = {"modules": info["modules"], "package_dir": info["package_dir"],
                                  "pinned_utc": _utc(self.clock()), "reason": "first pass"}
        elif pinned["modules"] != info["modules"]:
            changed = sorted(m for m in set(pinned["modules"]) | set(info["modules"])
                             if pinned["modules"].get(m) != info["modules"].get(m))
            raise DriverError("the driver's code changed since experiment %s was pinned (%s, %s): %s. One "
                              "experiment is decided by one version of the gate and the driver; to go on "
                              "with this code, run 'python -m weed_optimizer_framework.tools.inc.driver "
                              "repin --exp %s --reason TEXT' (recorded in the ledger)"
                              % (self.exp, pinned.get("pinned_utc"), pinned.get("reason"), changed, self.exp))
        self._ledger({"id": "code_pin/0", "type": "code_pin", "modules": self.state["code"]["modules"],
                      "package_dir": self.state["code"].get("package_dir")})

    # --------------------------------------------------------------- collect
    def _collect(self):
        waiting = []
        for rid, r in self.runs.items():
            if r["status"] == "submitted" and not self._collect_one(rid):
                waiting.append(rid)
            elif r["status"] == "failed":
                self._collect_late(rid)
        if not waiting:
            return
        queued = self._backend().queued()
        if queued is None:
            self._log("squeue unavailable; runs without run.json are left as they are")
            return
        gone = [rid for rid in waiting if self.runs[rid]["job"] not in queued]
        if not gone:
            return
        holders = {rid: self._holder_ids(rid) for rid in gone}
        ids = {self.runs[rid]["job_id"] for rid in gone}
        ids |= {j for hs in holders.values() for _, j in hs if j and j not in queued}
        states = self._backend().task_states(ids) or {}
        now = self.clock()
        for rid in gone:
            r = self.runs[rid]
            if self._collect_one(rid):        # finished between the first look and squeue
                continue
            state = states.get(r["job"])
            if state in SLURM_ENDED:
                why = "Slurm task %s ended %s without writing run.json" % (r["job"], state)
            elif state:
                continue                      # accounting says pending or running: squeue missed it
            elif r.get("submitted_ts") is not None and now - r["submitted_ts"] > STALE_SECONDS:
                why = ("submitted %.1f h ago as %s; not in squeue and no run.json"
                       % ((now - r["submitted_ts"]) / 3600.0, r["job"]))
            else:
                continue
            live = self._live_holder(holders[rid], queued, states)
            if live is None:
                self._fail(rid, why)
            elif live[1] and live[1] != r["job"]:
                self._retrack(rid, live[1], "%s; %s names job %s, still pending or running, which holds the "
                                            "run dir" % (why, live[0], live[1]))
            else:
                self._note(rid, "held", "%s; a live executor holds the run dir (%s, job %s): waiting for it"
                           % (why, live[0], live[1]))

    def _holder_ids(self, rid):
        """[(source, Slurm job id or None)] of who holds or last started on the
        run dir: the executor's owner file while its heartbeat is fresh, then
        attempt.json."""
        out = []
        try:
            mtime = self.paths.owner(rid).stat().st_mtime
            if time.time() - mtime <= OWNER_FRESH_SECONDS:
                o = _read_json(self.paths.owner(rid))
                j = o.get("slurm_job_id") if isinstance(o, dict) else None
                out.append((OWNER_FILE, str(j).strip() if j not in (None, "") else None))
        except (OSError, ValueError):
            pass
        try:
            a = _read_json(self.paths.attempt_json(rid))
            j = a.get("slurm_job_id") if isinstance(a, dict) else None
            if j not in (None, ""):
                out.append(("attempt.json", str(j).strip()))
        except (OSError, ValueError):
            pass
        return out

    @staticmethod
    def _live_holder(holders, queued, states):
        """(source, job id or None) of a holder that is still alive, else None:
        its Slurm job is queued or pending/running in sacct; or, for a fresh
        owner file, Slurm has not said that job ended (or it runs outside
        Slurm)."""
        for src, j in holders:
            if j and (j in queued or (states.get(j) and states[j] not in SLURM_ENDED)):
                return src, j
            if src == OWNER_FILE and not (j and states.get(j) in SLURM_ENDED):
                return src, j
        return None

    def _note(self, rid, event, why):
        r = self.runs[rid]
        ev = r.setdefault("events", [])
        if not ev or ev[-1].get("event") != event or ev[-1].get("why") != why:
            ev.append({"utc": _utc(self.clock()), "event": event, "why": why})
            self._log("%s: %s" % (rid, why))

    def _retrack(self, rid, holder, why):
        r = self.runs[rid]
        now = self.clock()
        r.setdefault("events", []).append({"utc": _utc(now), "event": "retracked", "from": r["job"],
                                           "to": holder, "why": why + " (not counted as a failure)"})
        r.update(job=holder, job_id=holder, array_index=None, submitted_ts=now)
        self._log("%s: %s; tracking job %s instead" % (rid, why, holder))

    def _collect_one(self, rid):
        """True when rid has a run.json of this attempt to read (the run is now
        complete, or failed). A run.json already counted as a failure is the
        previous attempt's, left until the retry starts: not this attempt's."""
        path = self.paths.run_json(rid)
        if not path.is_file():
            return False
        r = self.runs[rid]
        try:
            sha = C.sha256_file(path)
        except OSError:
            return False
        if sha in {h.get("run_json_sha256") for h in r["history"]}:
            return False
        try:
            rj = _read_json(path)
        except (OSError, ValueError) as e:
            self._fail(rid, "run.json unreadable: %s" % e, run_json_sha=sha)
            return True
        secs = rj.get("seconds") if isinstance(rj, dict) else None
        if not isinstance(rj, dict) or rj.get("status") != "done":
            rj = rj if isinstance(rj, dict) else {}
            self._fail(rid, "run %s at stage %s (executor attempt %s): %s"
                       % (rj.get("status"), rj.get("stage"), rj.get("attempt"), _short(rj.get("error"))),
                       run_json_sha=sha, seconds=secs)
            return True
        problem = self._outputs_missing(rid)
        if problem:
            self._fail(rid, "run.json says done but %s" % problem, run_json_sha=sha, seconds=secs)
        else:
            self._complete_run(rid, rj)
        return True

    def _outputs_missing(self, rid):
        try:
            spec = _read_json(self.paths.spec(rid))
        except (OSError, ValueError) as e:
            return "spec.json is unreadable (%s)" % e
        missing = [e for e in spec["exams"] if not self.paths.score(rid, e).is_file()]
        if missing:
            return "scores are missing for %s" % missing
        if not self.paths.weights(rid).exists():
            return "weights/final.pt is missing"
        return None

    def _complete_run(self, rid, rj):
        r = self.runs[rid]
        r["status"] = "complete"
        r["completed_utc"] = _utc(self.clock())
        r["seconds"] = rj.get("seconds")
        r["weights_sha256"] = rj.get("weights_sha256")

    def _collect_late(self, rid):
        """A run marked failed whose run.json now says done (a duplicate task,
        or a retry that outlived the failure count) is recovered, and the
        blocks it caused are lifted."""
        path = self.paths.run_json(rid)
        r = self.runs[rid]
        try:
            if not path.is_file():
                return
            sha = C.sha256_file(path)
            if sha in {h.get("run_json_sha256") for h in r["history"]}:
                return
            rj = _read_json(path)
        except (OSError, ValueError):
            return
        if not isinstance(rj, dict) or rj.get("status") != "done" or self._outputs_missing(rid):
            return
        self._complete_run(rid, rj)
        r["error"] = None
        r.setdefault("events", []).append({"utc": _utc(self.clock()), "event": "recovered",
                                           "why": "a done run.json (sha256 %s) appeared after the run was "
                                                  "marked failed" % sha[:12]})
        self._log("%s: marked failed, but a done run.json with its scores appeared; recovered" % rid)
        for unit, b in sorted(self.state["blocked"].items()):
            if (b.get("cause") or {}).get("run_id") == rid:
                self._unblock(unit, "run %s, which caused the block, completed after it was marked failed"
                              % rid, auto=True, reset=[])

    def _fail(self, rid, error, run_json_sha=None, seconds=None):
        r = self.runs[rid]
        r["history"].append({"attempt": r["attempt"], "job": r.get("job"), "error": error,
                             "utc": _utc(self.clock()), "run_json_sha256": run_json_sha,
                             "seconds": seconds if isinstance(seconds, (int, float)) else None})
        if r["attempt"] < r.get("attempt_limit", MAX_ATTEMPTS):
            r["attempt"] += 1
            r.update(status="queued", job=None, job_id=None, array_index=None, submission=None,
                     submitted_utc=None, submitted_ts=None)
            self._log("%s failed (%s); resubmitting as attempt %d" % (rid, _short(error, 160),
                                                                      r["attempt"]))
        else:
            r["status"] = "failed"
            r["error"] = error
            self._log("%s failed again: %s" % (rid, _short(error, 200)))

    # ------------------------------------------------------------ submitting
    def _recover_submissions(self):
        """Resolve submissions left 'submitting' (a pass died after sbatch, or
        sbatch's answer was not a job id) by their job name."""
        now = self.clock()
        for sub in self.state["submissions"]:
            if sub.get("status") != "submitting":
                continue
            since = sub.get("created_ts")
            if since is None:
                since = _ts(sub.get("created_utc"))
            kind, val = self._backend().lookup(sub["job_name"], since)
            sub["last_lookup"] = {"utc": _utc(now), "result": kind, "detail": val}
            if kind == FOUND:
                self._mark_submitted(sub, val)
                self._log("submission %s resolved: Slurm has it as job %s" % (sub["job_name"], val))
            elif kind == ABSENT and (since is None or now - since > SUBMIT_GRACE_SECONDS):
                sub.update(status="lost", lost_utc=_utc(now))
                self._log("submission %s is unknown to Slurm %.0f s after sbatch; its %d run(s) are queued "
                          "again" % (sub["job_name"], now - since if since else -1, len(sub["tasks"])))
                self._unsubmitted(sub, "submission %s never reached Slurm (last sbatch error: %s)"
                                  % (sub["job_name"], sub.get("error") or "none"))
            else:
                self._log("submission %s unresolved (%s%s); left for the next advance"
                          % (sub["job_name"], kind, ": %s" % val if val else ""))

    def _unsubmitted(self, sub, error):
        """sub's runs never reached Slurm: back to the queue, or failed after
        MAX_SUBMIT_FAILURES such submissions."""
        for rid in sub["tasks"]:
            r = self.runs[rid]
            if r["status"] != "submitting" or r.get("submission") != sub["n"]:
                continue
            r["submit_failures"] = r.get("submit_failures", 0) + 1
            r.update(status="queued", submission=None)
            if r["submit_failures"] >= MAX_SUBMIT_FAILURES:
                r.update(status="failed", error="%d submissions came to nothing; %s"
                         % (r["submit_failures"], error))
                r["submit_failures"] = 0

    def _mark_submitted(self, sub, jid):
        now = self.clock()
        sub.update(status="submitted", job_id=jid, submitted_utc=_utc(now))
        for i, rid in enumerate(sub["tasks"]):
            r = self.runs[rid]
            if r["status"] == "submitting" and r.get("submission") == sub["n"]:
                r.update(status="submitted", job="%s_%d" % (jid, i), job_id=jid, array_index=i,
                         submitted_utc=_utc(now), submitted_ts=now)

    def _submit_queued(self, result):
        ids = [rid for rid, r in self.runs.items() if r["status"] == "queued"]
        if not ids:
            return
        n = len(self.state["submissions"])
        list_file = self.paths.submissions / ("%04d.txt" % n)
        list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(list_file, "w") as fh:
            for rid in ids:
                fh.write("%s\n" % self.paths.spec(rid))
        now = self.clock()
        cold = any(self.runs[rid]["kind"] in COLD_KINDS for rid in ids)
        sub = {"n": n, "job_name": "inc_%s_%04d" % (self.exp, n), "list_file": str(list_file),
               "tasks": ids, "status": "submitting", "job_id": None, "created_utc": _utc(now),
               "created_ts": now, "time_limit": COLD_TIME_LIMIT if cold else None}
        self.state["submissions"].append(sub)
        for rid in ids:
            self.runs[rid]["status"] = "submitting"
            self.runs[rid]["submission"] = n
        self._save_state()                     # 'submitting' is on disk before sbatch runs
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        try:
            jid = self._backend().submit(list_file, len(ids), self.exp, sub["job_name"],
                                         self.paths.logs, env=submission_env(self.testing),
                                         time_limit=sub["time_limit"])
        except SubmitNotStarted as e:
            sub.update(status="error", error=_short("%s: %s" % (type(e).__name__, e)))
            self._unsubmitted(sub, "sbatch did not run: %s" % _short(e, 300))
            self._save_state()
            raise DriverError("sbatch did not run; %d run(s) stay queued for the next advance: %s"
                              % (len(ids), e))
        except Exception as e:  # noqa: BLE001 -- the array may be queued: never assume it is not
            sub.update(error=_short("%s: %s" % (type(e).__name__, e)), uncertain_utc=_utc(self.clock()))
            self._save_state()
            raise DriverError("sbatch for %s (%d run(s)) gave no job id (%s: %s); Slurm may have queued it, "
                              "so it stays 'submitting' and the next advance looks it up by name"
                              % (sub["job_name"], len(ids), type(e).__name__, e))
        self._mark_submitted(sub, jid)
        result["submitted"] += len(ids)
        result["job_ids"].append(jid)
        self._log("submitted %d run(s) as array job %s (%s)" % (len(ids), jid, list_file.name))

    # ------------------------------------------------------------ run specs
    def _new_run(self, rid, kind, owner, init, exams, manifest=None, recipe=None, seed=None,
                 soup_of=None):
        check_name(rid, "run_id")
        if rid in self.runs:
            raise DriverError("run %s already exists" % rid)
        spec = {"exp": self.exp, "run_id": rid, "kind": kind,
                "exams": list(exams), "out_dir": str(self.paths.run_dir(rid))}
        if init is not None:
            spec["init"] = str(init)
        if kind in TRAIN_KINDS:
            spec["train_manifest"] = str(manifest)
            spec["recipe"] = dict(recipe, seed=int(seed))
        if kind == "soup":
            spec["soup_of"] = [str(p) for p in soup_of]
        validate_spec(spec)
        run_dir = self.paths.run_dir(rid)
        spec_path = self.paths.spec(rid)
        if run_dir.exists():
            entries = sorted(os.listdir(run_dir))
            if entries == ["spec.json"] and _read_json(spec_path) == spec:
                pass                    # written by a pass that died (or was rolled back) before saving
            elif entries:
                raise DriverError("%s already holds %s, which this experiment's state does not "
                                  "know; refusing to reuse it" % (run_dir, entries[:5]))
            else:
                _write_json(spec_path, spec)
        else:
            _write_json(spec_path, spec)
        self.runs[rid] = {"run_id": rid, "kind": kind, "owner": owner, "spec": str(spec_path),
                          "status": "queued", "attempt": 1, "job": None, "job_id": None,
                          "array_index": None, "submission": None, "submitted_utc": None,
                          "submitted_ts": None, "error": None, "history": [], "seconds": None,
                          "created_utc": _utc(self.clock())}
        return rid

    @contextlib.contextmanager
    def _all_or_none(self):
        """Runs created inside the block (and final runs listed) are dropped
        from state if it raises."""
        before = set(self.runs)
        fin = self.state["final"]
        fin_runs, fin_sources = list(fin["runs"]), dict(fin["sources"])
        try:
            yield
        except BaseException:
            for rid in [x for x in self.runs if x not in before]:
                del self.runs[rid]
            fin["runs"], fin["sources"] = fin_runs, fin_sources
            raise

    def _failed(self, ids):
        """(run_id, error) of the first run among ids that failed for good, else None."""
        for rid in ids:
            r = self.runs.get(rid)
            if r is not None and r["status"] == "failed":
                return rid, r["error"]
        return None

    def _complete(self, ids):
        return all(self.runs.get(rid, {}).get("status") == "complete" for rid in ids)

    def _block(self, unit, error, cause=None):
        if unit not in self.state["blocked"]:
            entry = {"error": error, "utc": _utc(self.clock()), "cause": cause}
            if unit.startswith("chain:"):
                ph = self.state["chains"][unit[len("chain:"):]]["phase"]
                entry["phase_before"] = ph if ph != "blocked" else None
            self.state["blocked"][unit] = entry
            self._log("BLOCKED %s: %s" % (unit, _short(error, 300)))
        if unit.startswith("chain:"):
            self.state["chains"][unit[len("chain:"):]]["phase"] = "blocked"

    def _block_on_failed(self, unit, ids):
        f = self._failed(ids)
        if f:
            self._block(unit, "run %s failed for good (attempt %d): %s" % (f[0], self.runs[f[0]]["attempt"], f[1]),
                        cause={"kind": "failed_run", "run_id": f[0]})
            return True
        return False

    def _transient(self, unit, what, e):
        """An input that could not be read: retried on the next pass; the unit
        is blocked only after TRANSIENT_LIMIT passes in a row."""
        tr = self.state.setdefault("transient", {})
        now = _utc(self.clock())
        t = tr.get(unit) or {"count": 0, "first_utc": now}
        t.update(count=t["count"] + 1, last_utc=now, error="%s: %s: %s" % (what, type(e).__name__, e))
        tr[unit] = t
        if t["count"] >= TRANSIENT_LIMIT:
            self._block(unit, "%s failed on %d passes in a row; last: %s" % (what, t["count"], t["error"]),
                        cause={"kind": "transient"})
        else:
            self._log("%s: %s (pass %d of %d before blocking); retried on the next advance"
                      % (unit, _short(t["error"], 200), t["count"], TRANSIENT_LIMIT))

    def _clear_transient(self, unit):
        self.state.setdefault("transient", {}).pop(unit, None)

    def _unblock(self, unit, reason, auto, reset):
        b = self.state["blocked"].pop(unit)
        if unit.startswith("chain:"):
            r = unit[len("chain:"):]
            ch = self.state["chains"][r]
            if ch["phase"] == "blocked":
                ch["phase"] = b.get("phase_before") or self._infer_phase(ch)
        self._clear_transient(unit)
        rec = {"unit": unit, "auto": bool(auto), "reason": reason, "block": b, "reset_runs": list(reset),
               "utc": _utc(self.clock())}
        self.state.setdefault("unblocks", []).append(rec)
        self._ledger(dict(rec, id="unblock/%s/%d" % (unit, len(self.state["unblocks"])), type="unblock"))
        self._log("unblocked %s (%s): %s" % (unit, "auto" if auto else "by hand", _short(reason, 200)))

    @staticmethod
    def _infer_phase(ch):
        if ch.get("incumbent") is None:
            return "start"
        st = ch["steps"].get(str(ch["k"] + 1))
        if st and st.get("decision") and st["decision"]["verdict"] == G.ACCEPT and st.get("soup"):
            return "soup"
        return "step"

    def _reset_run(self, rid):
        r = self.runs[rid]
        r["attempt"] += 1
        r["attempt_limit"] = r["attempt"] + MAX_ATTEMPTS - 1
        r.update(status="queued", error=None, job=None, job_id=None, array_index=None, submission=None,
                 submitted_utc=None, submitted_ts=None)

    def _incumbent_of(self, rid, since):
        return {"run_id": rid, "weights": str(self.paths.weights(rid)),
                "dev_score": str(self.paths.score(rid, DECISION_EXAM)), "since_step": since}

    # ------------------------------------------------------------ base runs
    def cold_init(self):
        """What base and union runs start from: exp.init_weights (yolo11n.pt)."""
        return self.defn.get("init_weights") or COLD_INIT

    def base_ids(self):
        return ["base__s%d" % s for s in self.defn["seeds"]]

    def _ensure_base(self):
        base = self.defn["base"]
        for s, rid in zip(self.defn["seeds"], self.base_ids()):
            if rid not in self.runs:
                self._new_run(rid, "base", "base", self.cold_init(), [DECISION_EXAM],
                              manifest=base["manifest"], recipe=base["recipe"], seed=s)
        self._block_on_failed("base", self.base_ids())

    # ------------------------------------------------------------- baseline
    def _advance_baseline(self):
        ids = self.base_ids()
        if "base" in self.state["blocked"] or not self._complete(ids):
            return
        fin = self.state["final"]
        if not fin["created"]:
            with self._all_or_none():
                for s, rid in zip(self.defn["seeds"], ids):
                    self._new_final("final__base__s%d" % s, rid, self._incumbent_of(rid, 0)["weights"])
            fin.update(created=True, created_utc=_utc(self.clock()))
        self._finish_final()

    def _new_final(self, rid, source, weights):
        self._new_run(rid, "final", "final", weights, self.defn["final_exams"])
        self.state["final"]["runs"].append(rid)
        self.state["final"]["sources"][rid] = {"source": source, "weights": str(weights)}

    def _finish_final(self):
        fin = self.state["final"]
        if "final" in self.state["blocked"] or self._block_on_failed("final", fin["runs"]):
            return
        if fin["runs"] and self._complete(fin["runs"]):
            self.state["done"] = True
            self.state["done_utc"] = _utc(self.clock())
            self._log("experiment done")

    # ---------------------------------------------------------------- truth
    def _steps(self):
        return self.defn["steps"]

    def _step_tag(self, n):
        return "s%02d_%s" % (n, self._steps()[n - 1]["name"])

    def truth_with_ids(self, n):
        return ["truth__%s__union__s%d" % (self._step_tag(n), s) for s in self.defn["seeds"]]

    def last_clean_before(self, n):
        """1-based number of the last clean step before step n, or 0."""
        for j in range(n - 1, 0, -1):
            if self._steps()[j - 1]["clean"]:
                return j
        return 0

    def truth_without_ids(self, n):
        j = self.last_clean_before(n)
        return self.truth_with_ids(j) if j else self.base_ids()

    def _advance_truth(self):
        tr = self.state["truth"]
        if not tr["enabled"] or "truth" in self.state["blocked"]:
            return
        try:
            self._truth_transitions(tr)
        except DriverError as e:
            self._block("truth", str(e))
        except OSError as e:
            self._transient("truth", "the truth arm", e)

    def _truth_transitions(self, tr):
        steps = self._steps()
        if not tr.get("created"):
            # all of the arm or none of it: every step is checked before any run exists
            plan, t_rows = [], list(self._rows(self.defn["base"]))
            for n, step in enumerate(steps, 1):
                d_rows = self._rows(step)
                check_disjoint(t_rows, d_rows, step["name"], "T_%d" % (n - 1))
                plan.append((n, step, t_rows + d_rows))
                if step["clean"]:
                    t_rows = t_rows + d_rows
            mdir = self.paths.manifests / "truth"
            new_steps = {}
            with self._all_or_none():
                for n, step, rows in plan:
                    m = self._write_manifest(mdir / ("%s_with.jsonl" % self._step_tag(n)), rows)
                    for s, rid in zip(self.defn["seeds"], self.truth_with_ids(n)):
                        if rid not in self.runs:
                            self._new_run(rid, "union", "truth", self.cold_init(), [DECISION_EXAM],
                                          manifest=m["path"], recipe=self.defn["truth_recipe"], seed=s)
                    new_steps[str(n)] = {"n": n, "name": step["name"], "tag": self._step_tag(n),
                                         "clean": step["clean"], "manifest": m,
                                         "with": self.truth_with_ids(n),
                                         "without": self.truth_without_ids(n), "decision": None}
            tr["steps"] = new_steps
            tr["created"] = True
        for n in range(1, len(steps) + 1):
            st = tr["steps"][str(n)]
            if st["decision"] is not None:
                continue
            if self._block_on_failed("truth", st["with"] + st["without"]):
                return
            if not self._complete(st["with"] + st["without"]):
                continue
            try:
                w, w_in = self._score_inputs(self._dev(st["with"]))
                wo, wo_in = self._score_inputs(self._dev(st["without"]))
            except OSError as e:
                self._transient("truth", "reading the scores of truth step %s" % st["tag"], e)
                return
            try:
                d = G.truth_detail(w, wo, self.cfg)
            except (ValueError, KeyError, TypeError) as e:
                self._block("truth", "truth_detail refused %s: %s: %s" % (st["tag"], type(e).__name__, e))
                return
            self._clear_transient("truth")
            self._ledger({"id": "truth/%d" % n, "type": "truth", "k": n, "step": st["name"],
                          "tag": st["tag"], "clean": st["clean"], "detail": d,
                          "inputs": {"with": w_in, "without": wo_in}, "manifest": st["manifest"]})
            st["decision"] = {"verdict": d["verdict"], "p": d["p"], "with_mean": d["with_mean"],
                              "with_sd": d["with_sd"], "without_mean": d["without_mean"],
                              "without_sd": d["without_sd"],
                              "species_failed": d["species"]["failed"], "warnings": d["warnings"]}
            self._log("truth %s: %s (P=%.3f)" % (st["tag"], d["verdict"], d["p"]))

    def truth_complete(self):
        tr = self.state.get("truth") or {}
        if not tr.get("enabled"):
            return True
        return (bool(tr.get("created")) and len(tr["steps"]) == len(self._steps())
                and all(s["decision"] is not None for s in tr["steps"].values()))

    # --------------------------------------------------------------- chains
    def _advance_chain(self, r):
        unit = "chain:" + r
        ch = self.state["chains"][r]
        for _ in range(4 * len(self._steps()) + 4):
            if ch["phase"] in ("done", "blocked") or unit in self.state["blocked"]:
                return
            try:
                moved = self._chain_transition(r, ch)
            except DriverError as e:
                self._block(unit, str(e))
                return
            except OSError as e:
                self._transient(unit, "chain %s" % r, e)
                return
            if not moved:
                return

    def _chain_transition(self, r, ch):
        """One state change of chain r; False when it has to wait."""
        unit = "chain:" + r
        steps = self._steps()
        if ch["phase"] == "start":
            rid = self.base_ids()[0]
            if self._block_on_failed(unit, [rid]):
                return False
            if not self._complete([rid]):
                return False
            ch["incumbent"] = self._incumbent_of(rid, 0)
            ch["phase"] = "step"
            return True
        if ch["phase"] == "step":
            if ch["k"] >= len(steps):
                ch["phase"] = "done"
                self._log("chain %s: every step decided; incumbent %s" % (r, ch["incumbent"]["run_id"]))
                return False
            n = ch["k"] + 1
            st = ch["steps"].get(str(n))
            if st is None:
                ch["steps"][str(n)] = self._create_step(r, ch, n)
                return False
            ids = st["cand"] + st["null"]
            if self._block_on_failed(unit, ids) or not self._complete(ids):
                return False
            return self._decide(r, ch, st)
        if ch["phase"] == "soup":
            st = ch["steps"][str(ch["k"] + 1)]
            if self._block_on_failed(unit, [st["soup"]]) or not self._complete([st["soup"]]):
                return False
            return self._soup_done(r, ch, st)
        raise DriverError("chain %s in unknown phase %r" % (r, ch["phase"]))

    def _create_step(self, r, ch, n):
        step = self._steps()[n - 1]
        tag = self._step_tag(n)
        d_rows = self._rows(step)
        pool = list(self._rows(self.defn["base"]))
        by_name = {s["name"]: s for s in self._steps()}
        for a in ch["accepted"]:
            pool += self._rows(by_name[a])
        check_disjoint(pool, d_rows, step["name"], "chain %s's accepted pool" % r)
        mdir = self.paths.manifests / r
        full = self.replay_mode == "full"
        if full:
            seed_text = None                     # nothing is sampled
            cand_rows, null_rows = full_rehearsal(pool, d_rows)
            manifests = {"cand": self._write_manifest(mdir / ("%s_cand.jsonl" % tag), cand_rows),
                         "null": self._write_manifest(mdir / ("%s_null.jsonl" % tag), null_rows)}
        else:
            seed_text = "%s/%s/%d" % (self.exp, r, n)
            r1, r2 = replay_samples(pool, len(d_rows), seed_text)
            manifests = {"R1": self._write_manifest(mdir / ("%s_R1.jsonl" % tag), r1),
                         "R2": self._write_manifest(mdir / ("%s_R2.jsonl" % tag), r2),
                         "cand": self._write_manifest(mdir / ("%s_cand.jsonl" % tag), d_rows + r1),
                         "null": self._write_manifest(mdir / ("%s_null.jsonl" % tag), r1 + r2)}
        recipe = self.defn["recipes"][r]
        init = ch["incumbent"]["weights"]
        cand, null = [], []
        with self._all_or_none():
            for s in self.defn["seeds"]:
                cand.append(self._new_run("%s__%s__cand__s%d" % (r, tag, s), "cand", "chain:" + r, init,
                                          [DECISION_EXAM], manifest=manifests["cand"]["path"],
                                          recipe=recipe, seed=s))
                null.append(self._new_run("%s__%s__null__s%d" % (r, tag, s), "null", "chain:" + r, init,
                                          [DECISION_EXAM], manifest=manifests["null"]["path"],
                                          recipe=recipe, seed=s))
        self._log("chain %s: step %s created (|D_k|=%d, pool %d, %sincumbent %s)"
                  % (r, tag, len(d_rows), len(pool),
                     "replay full: cand %d, null %d images, " % (len(cand_rows), len(null_rows)) if full else "",
                     ch["incumbent"]["run_id"]))
        out = {"n": n, "name": step["name"], "tag": tag, "clean": step["clean"],
               "sources": sorted({str(x.get("source")) for x in d_rows}),
               "replay_seed_text": seed_text, "pool_images": len(pool), "d_images": len(d_rows),
               "manifests": manifests, "cand": cand, "null": null, "soup": None,
               "incumbent_before": ch["incumbent"], "incumbent_after": None, "decision": None,
               "soup_choice": None, "created_utc": _utc(self.clock())}
        if full:
            # a sample-mode step records nothing new (absent = 'sample', as in exp.json)
            out.update(replay_mode="full", pool_accepted=list(ch["accepted"]))
        return out

    @staticmethod
    def attribution_not_run(sources):
        """The protocol's attribution steps the driver does not run for a step
        with these sources."""
        out = {"label_audit": ATTRIBUTION_NOT_RUN["label_audit"]}
        if len(sources or []) > 1:
            out["leave_one_source_out"] = ATTRIBUTION_NOT_RUN["leave_one_source_out"]
        return out

    def _decide(self, r, ch, st):
        unit = "chain:" + r
        inc_rid, inc_path = ch["incumbent"]["run_id"], Path(ch["incumbent"]["dev_score"])
        try:
            (inc,), inc_in = self._score_inputs([(inc_rid, inc_path)])
            cands, cand_in = self._score_inputs(self._dev(st["cand"]))
            nulls, null_in = self._score_inputs(self._dev(st["null"]))
        except OSError as e:
            self._transient(unit, "reading the scores of step %s" % st["tag"], e)
            return False
        except ValueError as e:
            self._block(unit, "a score file of step %s is not JSON: %s" % (st["tag"], e))
            return False
        try:
            d = G.decide(inc, cands, nulls, self.cfg)
        except (ValueError, KeyError, TypeError) as e:
            self._block(unit, "the gate refused step %s: %s: %s" % (st["tag"], type(e).__name__, e))
            return False
        self._clear_transient(unit)
        n = st["n"]
        not_run = self.attribution_not_run(st.get("sources"))
        entry = {"id": "gate/%s/%d" % (r, n), "type": "gate", "chain": r, "k": n,
                 "step": st["name"], "tag": st["tag"], "clean": st["clean"],
                 "decision": d.to_dict(),
                 "inputs": {"inc": inc_in[0], "cand": cand_in, "null": null_in},
                 "incumbent_before": ch["incumbent"], "manifests": st["manifests"],
                 "replay_seed_text": st["replay_seed_text"], "sources": st.get("sources"),
                 "attribution_not_run": not_run}
        if st.get("replay_mode") == "full":
            # a sample-mode entry is written as before (no replay_mode key = 'sample')
            entry.update(replay_mode="full", pool_images=st["pool_images"], d_images=st["d_images"],
                         pool_accepted=st.get("pool_accepted"))
        self._ledger(entry)
        attr = d.attribution
        st["decision"] = {"verdict": d.verdict, "p_data": d.p_data, "p_recipe": d.p_recipe,
                          "reason": d.reason, "blame": attr.get("blame"),
                          "class_vs_loc": attr.get("class_vs_loc"),
                          "species_failed": attr.get("species_failed"),
                          "guards": {k: bool(v["passed"]) for k, v in d.guards.items()},
                          "warnings": d.warnings, "attribution_not_run": sorted(not_run)}
        self._log("chain %s %s: %s" % (r, st["tag"], _short(d.reason, 300)))
        if d.verdict == G.ACCEPT:
            with self._all_or_none():
                st["soup"] = self._new_run("%s__%s__soup" % (r, st["tag"]), "soup", "chain:" + r,
                                           None, [DECISION_EXAM],
                                           soup_of=[self.paths.weights(x) for x in st["cand"]])
            ch["phase"] = "soup"
            return True
        (ch["neutral"] if d.verdict == G.HOLD else ch["quarantined"]).append(st["name"])
        st["incumbent_after"] = ch["incumbent"]
        ch["k"] += 1
        return True

    def _soup_done(self, r, ch, st):
        unit = "chain:" + r
        try:
            (soup,), soup_in = self._score_inputs(self._dev([st["soup"]]))
            cands, cand_in = self._score_inputs(self._dev(st["cand"]))
        except OSError as e:
            self._transient(unit, "reading the soup scores of step %s" % st["tag"], e)
            return False
        except ValueError as e:
            self._block(unit, "a score file of step %s is not JSON: %s" % (st["tag"], e))
            return False
        try:
            choice = G.choose_soup(soup, cands, self.cfg)
            m = self.cfg.metric
            soup_value, cand_values = soup[m], [c[m] for c in cands]
        except (ValueError, KeyError, TypeError) as e:
            self._block(unit, "choose_soup refused step %s: %s: %s" % (st["tag"], type(e).__name__, e))
            return False
        self._clear_transient(unit)
        new_rid = st["soup"] if choice == "soup" else st["cand"][0]
        new_inc = self._incumbent_of(new_rid, st["n"])
        self._ledger({"id": "soup/%s/%d" % (r, st["n"]), "type": "soup", "chain": r, "k": st["n"],
                      "step": st["name"], "tag": st["tag"], "choice": choice,
                      "soup_value": soup_value, "cand_values": cand_values, "metric": m,
                      "inputs": {"soup": soup_in[0], "cand": cand_in},
                      "incumbent_before": ch["incumbent"], "incumbent_after": new_inc})
        ch["incumbent"] = new_inc
        ch["accepted"].append(st["name"])
        st["incumbent_after"] = new_inc
        st["soup_choice"] = choice
        ch["k"] += 1
        ch["phase"] = "step"
        self._log("chain %s %s: new incumbent %s (%s)" % (r, st["tag"], new_rid, choice))
        return True

    # ---------------------------------------------------------------- final
    def tfinal_ids(self):
        j = self.last_clean_before(len(self._steps()) + 1)
        return self.truth_with_ids(j) if j else []

    def _advance_final_chain(self):
        fin = self.state["final"]
        if not fin["created"]:
            if any(ch["phase"] != "done" for ch in self.state["chains"].values()):
                return
            if not self.truth_complete():
                return
            with self._all_or_none():
                for r, ch in self.state["chains"].items():
                    self._new_final("final__%s__incumbent" % r, ch["incumbent"]["run_id"],
                                    ch["incumbent"]["weights"])
                for s, rid in zip(self.defn["seeds"], self.base_ids()):
                    self._new_final("final__base__s%d" % s, rid, self.paths.weights(rid))
                if self.state["truth"]["enabled"]:
                    for s, rid in zip(self.defn["seeds"], self.tfinal_ids()):
                        self._new_final("final__Tfinal__s%d" % s, rid, self.paths.weights(rid))
            fin.update(created=True, created_utc=_utc(self.clock()))
            self._log("every chain and the truth arm are complete; %d final run(s) created"
                      % len(fin["runs"]))
        self._finish_final()

    # ------------------------------------------------------------ operations
    def unblock(self, units=(), reason="", all_units=False):
        """Lift the named blocks (or every block): failed runs of each unit,
        and the run that caused its block, go back to the queue with
        MAX_ATTEMPTS more tries; recorded in the ledger. Then advance."""
        if not str(reason).strip():
            raise DriverError("unblock needs a --reason (it goes into the ledger)")
        with self._exclusive():
            self._load()
            self._check_code()
            blocked = self.state["blocked"]
            targets = sorted(blocked) if all_units else list(units)
            if not targets:
                raise DriverError("nothing to unblock (blocked: %s)" % (sorted(blocked) or "none"))
            unknown = [u for u in targets if u not in blocked]
            if unknown:
                raise DriverError("not blocked: %s (blocked: %s)" % (unknown, sorted(blocked) or "none"))
            for u in targets:
                cause = (blocked[u].get("cause") or {}).get("run_id")
                reset = sorted(rid for rid, r in self.runs.items()
                               if r["status"] == "failed" and (r["owner"] == u or rid == cause))
                for rid in reset:
                    self._reset_run(rid)
                self._unblock(u, str(reason).strip(), auto=False, reset=reset)
            self._save_state()
        return self.advance()

    def repin(self, reason):
        """Accept the running code as this experiment's: its hashes replace the
        pins, and the change goes into the ledger."""
        if not str(reason).strip():
            raise DriverError("repin needs a --reason (it goes into the ledger)")
        with self._exclusive():
            self._load()
            info, drift = driver_code()
            if drift and os.environ.get(ALLOW_DRIFT_ENV) != "1":
                raise _drift_error(info, drift)
            old = (self.state.get("code") or {}).get("modules") or {}
            changed = sorted(m for m in set(old) | set(info["modules"]) if old.get(m) != info["modules"].get(m))
            if not changed:
                self._log("the running code matches the pins; nothing to repin")
                return {"changed": []}
            self._code = {"modules": info["modules"], "drift": drift}
            rec = {"utc": _utc(self.clock()), "reason": str(reason).strip(), "changed": changed,
                   "old": old, "new": info["modules"], "package_dir": info["package_dir"]}
            self.state.setdefault("repins", []).append(rec)
            self._ledger(dict(rec, id="code_repin/%d" % len(self.state["repins"]), type="code_repin"))
            self.state["code"] = {"modules": info["modules"], "package_dir": info["package_dir"],
                                  "pinned_utc": rec["utc"], "reason": rec["reason"]}
            self._save_state()
            self._log("re-pinned %s: %s" % (changed, rec["reason"]))
            return {"changed": changed}

    # --------------------------------------------------------------- status
    def status_lines(self):
        st, runs = self.state, self.state["runs"]
        tag = " [TESTING]" if st.get("testing") else ""

        def count(owner):
            own = [r for r in runs.values() if r["owner"] == owner]
            return (sum(1 for r in own if r["status"] == "complete"),
                    sum(1 for r in own if r["status"] in LIVE), len(own))

        def blocked(unit):
            b = st["blocked"].get(unit)
            return "; BLOCKED: %s" % _short(b["error"], 200) if b else ""

        lines = []
        c, p, t = count("base")
        lines.append("%s%s base: %d/%d complete, %d pending%s" % (self.exp, tag, c, t, p, blocked("base")))
        pinned = pinned_gate_config(st) if st.get("chains") is not None else None
        if pinned is not None:
            changed = gate_changes(pinned, gate_config(self.defn))
            if changed:
                lines.append("%s%s GATE CONFIG CHANGED: exp.json no longer resolves to the gate config pinned at "
                             "init (%s as [pinned, now]); advance refuses until it is restored"
                             % (self.exp, tag, changed))
        for r, ch in (st.get("chains") or {}).items():
            n_steps = len(self._steps())
            k = ch["k"]
            cur = ("-" if ch["phase"] in ("start", "done") or k >= n_steps else self._step_tag(k + 1))
            last = None
            for key in sorted(ch["steps"], key=int):
                if ch["steps"][key]["decision"]:
                    last = ch["steps"][key]
            last_s = ("%s %s P_data=%.2f" % (last["tag"], last["decision"]["verdict"], last["decision"]["p_data"])
                      if last else "none")
            _, p, _ = count("chain:" + r)
            lines.append("%s%s chain %s%s: %s, %d/%d steps decided, current %s, %d pending run(s); last %s; "
                         "incumbent %s%s" % (self.exp, tag, r,
                                             (" (replay full)" if self.replay_mode == "full" else "")
                                             + gate_tag(pinned),
                                             ch["phase"], k, n_steps, cur, p, last_s,
                                             (ch["incumbent"] or {}).get("run_id", "-"),
                                             blocked("chain:" + r)))
        tr = st.get("truth")
        if tr and tr.get("enabled"):
            c, p, t = count("truth")
            dec = [s for s in tr["steps"].values() if s["decision"]]
            last = max(dec, key=lambda s: s["n"]) if dec else None
            lines.append("%s%s truth: %d/%d runs complete, %d pending, %d/%d decisions; last %s%s"
                         % (self.exp, tag, c, t, p, len(dec), len(self._steps()),
                            ("%s %s P=%.2f" % (last["tag"], last["decision"]["verdict"], last["decision"]["p"])
                             if last else "none"), blocked("truth")))
        c, p, t = count("final")
        fin = st["final"]
        lines.append("%s%s final: %s%s" % (self.exp, tag, ("%d/%d complete, %d pending" % (c, t, p))
                                           if fin["created"] else "not started", blocked("final")))
        for sub in st.get("submissions", []):
            if sub.get("status") == "submitting":
                lines.append("%s%s submission %s: sbatch outcome not known yet (%d run(s)); resolved by job "
                             "name on the next advance%s" % (self.exp, tag, sub["job_name"], len(sub["tasks"]),
                                                             "; last error: %s" % _short(sub["error"], 160)
                                                             if sub.get("error") else ""))
        for unit, tt in sorted((st.get("transient") or {}).items()):
            lines.append("%s%s %s: inputs unreadable on %d pass(es) in a row, retrying: %s"
                         % (self.exp, tag, unit, tt["count"], _short(tt.get("error"), 160)))
        if not st.get("done") and not any(r["status"] in LIVE for r in runs.values()):
            lines.append("%s%s WAITING: no run is queued or running; %s"
                         % (self.exp, tag, "every open unit is blocked (see 'unblock')" if st["blocked"]
                            else "run advance"))
        lines.append("%s%s done: %s" % (self.exp, tag, "yes (%s)" % st["done_utc"] if st.get("done") else "no"))
        return lines

    def status(self):
        self._load(for_advance=False)
        return self.status_lines()


# ---------------------------------------------------------------- module API
def init(defn, backend=None, quiet=False, clock=time.time):
    return Driver(defn["exp"], backend=backend, quiet=quiet, clock=clock).init(defn)


def advance(exp, backend=None, quiet=False, clock=time.time):
    return Driver(exp, backend=backend, quiet=quiet, clock=clock).advance()


def status(exp):
    return Driver(exp).status()


def watch(exp, interval=WATCH_INTERVAL, max_hours=None, backend=None, sleep=time.sleep, clock=time.time,
          out=print):
    """advance every `interval` seconds until done (0) or max_hours passed (1).
    Errors are printed and the loop goes on."""
    t0, last = clock(), None
    while True:
        try:
            res = Driver(exp, backend=backend, quiet=True, clock=clock).advance()
            if res["lines"] and res["lines"] != last:
                for ln in res["lines"]:
                    out("[inc.driver] %s %s" % (_utc(), ln))
                last = res["lines"]
            if res["done"]:
                return 0
        except (DriverError, OSError, ValueError) as e:
            out("[inc.driver] %s ERROR: %s: %s" % (_utc(), type(e).__name__, e))
        if max_hours is not None and clock() - t0 > max_hours * 3600.0:
            return 1
        sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The INC driver (docs/INCREMENTAL_PROTOCOL_RUNNER.md).")
    ap.add_argument("command", choices=("init", "advance", "status", "unblock", "repin", "watch"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--definition", default=None,
                    help="init: a definition file; default INC_DIR/<exp>/exp.json (written by a builder)")
    ap.add_argument("--quiet", action="store_true", help="advance: no per-chain summary")
    ap.add_argument("--unit", action="append", default=[],
                    help="unblock: a blocked unit (base, truth, final, chain:<recipe>); repeatable")
    ap.add_argument("--all", action="store_true", help="unblock: every blocked unit")
    ap.add_argument("--reason", default="", help="unblock / repin: why (goes into the ledger)")
    ap.add_argument("--interval", type=float, default=WATCH_INTERVAL, help="watch: seconds between advances")
    ap.add_argument("--max-hours", type=float, default=None, help="watch: stop after this long")
    a = ap.parse_args(argv)
    try:
        if a.command == "status":
            for ln in status(a.exp):
                print(ln)
            return 0
        if a.command == "init":
            path = Path(a.definition) if a.definition else Paths(a.exp).exp_json
            if not path.is_file():
                raise DriverError("no definition at %s; build the experiment first "
                                  "(e.g. python -m weed_optimizer_framework.tools.inc.pilot build)" % path)
            Driver(a.exp, quiet=a.quiet).init(_read_json(path))
            return 0
        if a.command == "unblock":
            Driver(a.exp, quiet=a.quiet).unblock(a.unit, a.reason, all_units=a.all)
            return 0
        if a.command == "repin":
            Driver(a.exp, quiet=a.quiet).repin(a.reason)
            return 0
        if a.command == "watch":
            return watch(a.exp, interval=a.interval, max_hours=a.max_hours)
        advance(a.exp, quiet=a.quiet)
        return 0
    except DriverError as e:
        print("[inc.driver] ERROR: %s" % e, file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print("[inc.driver] ERROR (%s): %s" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
