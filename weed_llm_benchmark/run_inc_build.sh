#!/bin/bash
#SBATCH --job-name=inc_build
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# INC experiment build as a batch job: docs/INC_AUTOPILOT.md, component 5. The
# builders hash every image of their manifests, which outlives what the login
# node lets a process run; this job runs one of them, then one driver advance:
#
#   python -m weed_optimizer_framework.tools.inc.pilot build          --exp E [--replay-mode {sample,full}]
#   python -m weed_optimizer_framework.tools.inc.pilot build-baseline --exp E --manifest M [--seeds 0,1,2]
#   python -m weed_optimizer_framework.tools.inc.realloop build       --exp E --replay-mode R --recipes L [...]
#   python -m weed_optimizer_framework.tools.inc.driver advance       --exp E      (after a build that exited 0)
#
# The first argument names the builder module: pilot / realloop, or the
# dotted forms the policy table and the levers write (inc.pilot,
# weed_optimizer_framework.tools.inc.pilot).
#
# The builder writes the experiment and calls driver init, which submits the
# first runs; the advance after it is idempotent. The GPU stays idle: the
# allocation cis240145p is a GPU allocation, and RM-shared submissions fail
# with "Invalid qos" (run_inc_audit.sh, CHANGELOG v3.0.99.19). The job is
# still charged for the V100 it holds (up to the 4 h walltime), on top of the
# experiment's runs.
#
# Submit (the autopilot does it through inc_autopilot/remote.py submit build,
# which checks the arguments against levers.json and names the job
# inc_build_<exp>; Slurm opens the log file before this script runs, so the
# log dir must exist):
#   mkdir -p /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs
#   sbatch --job-name=inc_build_pilot_v2 run_inc_build.sh pilot build --exp pilot_v2 --replay-mode full
#
# Provenance: INC_DIR/_campaign/provenance/<exp>.json, outside the experiment
# directory (pilot.py refuses to build over an existing exp.json or
# state.json and deletes the outputs of a build that never reached driver
# init). Each run of this script appends one attempt: the job id, the builder
# argv, INCAP_PARENT_EXP / INCAP_TRIGGER / INCAP_APPROVAL_ID /
# INCAP_DECIDED_BY / INCAP_REQUESTED_UTC as remote.py passed them, the sha256
# of every module below in the outer and the nested copy, the build's exit
# status and its ERROR lines (a builder refusal names the missing
# prerequisite), the advance's exit status. A job killed by scancel, its time
# limit or a node's shutdown (SIGTERM) records status "killed" and the phase it
# was in; only SIGKILL leaves "running" behind.
#
# One builder per experiment: before anything else the job takes the lock
# INC_DIR/_campaign/locks/<exp>.build (O_EXCL: "<job id> <host> <utc>"), and
# releases it on exit. A second builder of the same experiment, however it was
# submitted, would have pilot.py delete the first one's manifests/ and labels/
# mid-build; it exits 1 instead, writing nothing (not even provenance). A lock
# whose job squeue no longer knows ("Invalid job id") or reports ended is
# stale and is taken over; when squeue cannot tell, the job refuses and a
# person removes the lock. remote.py also refuses to queue a build while an
# inc_build_<exp> or a hand-submitted inc_build is queued or running.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH; deploy there first. Nothing is synced here. Every module
# the builders import is hashed into the log, and the job stops when an outer
# module differs from the git-tracked nested copy. Set INC_BUILD_ALLOW_DRIFT=1
# to run the outer copies anyway (the log and the provenance record name
# every difference; remote.py never sets it).
#
# INC_BUILD_REPO and INC_BUILD_CONDA_SH replace the cluster paths below (tests
# only).
set -uo pipefail

REPO="${INC_BUILD_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC_BUILD_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
INC_ROOT="${INC_DIR:-$REPO/results/framework/inc}"

usage() {
    echo "usage: sbatch run_inc_build.sh {pilot build | pilot build-baseline | realloop build} --exp NAME [builder flags ...]" >&2
    exit 2
}

ARG_MOD="${1:-}"                        # pilot, inc.pilot or weed_optimizer_framework.tools.inc.pilot
CMD="${2:-}"
MOD="${ARG_MOD#weed_optimizer_framework.}"
MOD="${MOD#tools.}"
MOD="${MOD#inc.}"
case "$MOD $CMD" in
    "pilot build"|"pilot build-baseline"|"realloop build") shift 2 ;;
    *) usage ;;
esac
EXP=""
prev=""
for a in "$@"; do
    case "$a" in --exp=*) EXP="${a#--exp=}" ;; esac
    [ "$prev" = "--exp" ] && EXP="$a"
    prev="$a"
done
if ! [[ "$EXP" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
    echo "FATAL: --exp '$EXP' is missing or not an experiment name" >&2
    usage
fi
PROV="$INC_ROOT/_campaign/provenance/$EXP.json"
mkdir -p "$(dirname "$PROV")" || exit 1

# The per-experiment build lock (see the header).
LOCK="$INC_ROOT/_campaign/locks/$EXP.build"
mkdir -p "$(dirname "$LOCK")" || exit 1
HAVE_LOCK=0
take_lock() {
    ( set -o noclobber
      printf '%s %s %s\n' "${SLURM_JOB_ID:-none}" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK" ) 2>/dev/null
}
holder_state() {                        # alive | dead | unknown, for the job id that holds the lock
    local id="$1" out rc
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then echo unknown; return; fi
    if [ "$id" = "${SLURM_JOB_ID:-}" ]; then echo dead; return; fi      # this job, requeued
    out="$(squeue -h -j "$id" -o %T 2>&1)"; rc=$?
    if [ "$rc" != 0 ]; then
        case "$out" in *"Invalid job id"*) echo dead ;; *) echo unknown ;; esac
        return
    fi
    case "$out" in
        *PENDING*|*RUNNING*|*CONFIGURING*|*COMPLETING*|*SUSPENDED*|*REQUEUE*|*RESIZING*|*SIGNALING*|*STAGE_OUT*)
            echo alive ;;
        *) echo dead ;;
    esac
}
if take_lock; then
    HAVE_LOCK=1
else
    held="$(cat "$LOCK" 2>/dev/null)"
    holder="${held%% *}"
    hs="$(holder_state "$holder")"
    if [ "$hs" = dead ] && rm -f "$LOCK" && take_lock; then
        HAVE_LOCK=1
        echo "WARNING: took over the stale build lock of $EXP ($held): job $holder is no longer running"
    else
        echo "FATAL: another build of $EXP holds $LOCK ($held; its job is $hs): refused, nothing written." \
             "Remove the lock by hand only when no build of $EXP runs." >&2
        exit 1
    fi
fi
PHASE=start
PROV_STARTED=0
on_exit() {
    [ -n "${OUT_TMP:-}" ] && rm -f "$OUT_TMP"
    [ "$HAVE_LOCK" = 1 ] && rm -f "$LOCK"
}
on_term() {
    trap - TERM INT
    echo "=== killed (SIGTERM/SIGINT) during $PHASE  $(date) ===" >&2
    [ "$PROV_STARTED" = 1 ] && prov update status=killed "killed_during=$PHASE" finish
    exit 143
}
trap on_exit EXIT
trap on_term TERM INT

source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc.$MOD $CMD $*  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Every module the three builders import (lazily included), outer copy (what
# runs) vs nested (git-tracked).
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/driver.py tools/inc/gate.py tools/inc/splits.py
         tools/inc/scorer.py tools/inc/pilot.py tools/inc/train.py tools/inc/select.py tools/inc/verify.py
         tools/inc/relevance.py tools/inc/realloop.py tools/cwd12_species.py tools/near_dup.py
         tools/mega_trainer.py)
export INCB_EXP="$EXP" INCB_MODULES="${MODULES[*]}"

# prov start -- ARGV... : append an attempt; prov update KEY=VALUE ... : set
# fields of the last attempt (a VALUE that parses as JSON is stored as JSON);
# the key 'finish' stamps finished_utc. Written atomically (tmp + rename).
prov() {
    python - "$PROV" "$@" <<'PY' || echo "WARNING: could not write the provenance record $PROV" >&2
import datetime, hashlib, json, os, socket, sys

path, event, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
env = os.environ
rec = None
if os.path.exists(path):
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except ValueError:
        os.replace(path, "%s.unreadable.%s" % (path, now))
if not isinstance(rec, dict) or not isinstance(rec.get("attempts"), list):
    rec = {"format": "inc_autopilot.provenance/1", "exp": env["INCB_EXP"], "attempts": []}


def sha(p):
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if event == "start":
    argv = rest[1:] if rest[:1] == ["--"] else rest
    repo = env["REPO"]
    mods = {}
    for m in env.get("INCB_MODULES", "").split():
        mods[m] = {"outer": sha(os.path.join(repo, "weed_optimizer_framework", m)),
                   "nested": sha(os.path.join(repo, "weed_llm_benchmark", "weed_optimizer_framework", m))}
    rec["attempts"].append({
        "job_id": env.get("SLURM_JOB_ID"), "host": socket.gethostname(), "started_utc": now,
        "argv": argv, "module": argv[0] if argv else None, "command": argv[1] if len(argv) > 1 else None,
        "parent_exp": env.get("INCAP_PARENT_EXP") or None,
        "trigger": [t for t in env.get("INCAP_TRIGGER", "").split(",") if t],
        "approval_id": env.get("INCAP_APPROVAL_ID") or None,
        "decided_by": env.get("INCAP_DECIDED_BY") or None,
        "requested_utc": env.get("INCAP_REQUESTED_UTC") or None,
        "modules": mods, "drift": sorted(m for m, v in mods.items() if v["outer"] != v["nested"]),
        "status": "running"})
else:
    if not rec["attempts"]:
        rec["attempts"].append({"job_id": env.get("SLURM_JOB_ID"), "note": "no start record"})
    att = rec["attempts"][-1]
    for kv in rest:
        k, _, v = kv.partition("=")
        if k == "finish":
            att["finished_utc"] = now
            continue
        if k in ("refusal", "missing"):         # text, whatever it looks like
            att[k] = v
            continue
        try:
            att[k] = json.loads(v)
        except ValueError:
            att[k] = v
    att["updated_utc"] = now
tmp = "%s.%d.tmp" % (path, os.getpid())
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=1, sort_keys=True)
    fh.write("\n")
os.replace(tmp, path)
PY
}

prov start -- "$ARG_MOD" "$CMD" "$@"
PROV_STARTED=1
PHASE=drift_check

drift=0
for m in "${MODULES[@]}"; do
    OUTER=$REPO/weed_optimizer_framework/$m
    NESTED=$REPO/weed_llm_benchmark/weed_optimizer_framework/$m
    if [ ! -f "$OUTER" ]; then
        echo "FATAL: $OUTER missing: deploy the outer package copy first" >&2
        prov update status=refused_missing_module "missing=$m" finish
        exit 1
    fi
    if [ ! -f "$NESTED" ]; then
        state="no nested copy"
    elif cmp -s "$OUTER" "$NESTED"; then
        state="= nested"
    else
        state="DIFFERS from nested $(sha256sum "$NESTED" | cut -c1-16)"
        drift=1
    fi
    echo "module $m: $(sha256sum "$OUTER" | cut -c1-16)  $state"
done
if [ "$drift" = 1 ]; then
    if [ "${INC_BUILD_ALLOW_DRIFT:-0}" = 1 ]; then
        echo "WARNING: outer modules differ from the nested copy (above); running the outer ones (INC_BUILD_ALLOW_DRIFT=1)"
        prov update drift_allowed=true
    else
        echo "FATAL: outer modules differ from the nested copy (above): sync them, or set INC_BUILD_ALLOW_DRIFT=1" >&2
        prov update status=refused_drift finish
        exit 1
    fi
fi

python -u -c "import numpy; print('numpy', numpy.__version__)" \
    || { prov update status=env_failed finish; exit 1; }

OUT_TMP="$(mktemp "${TMPDIR:-/tmp}/inc_build_${EXP}.XXXXXX")" || OUT_TMP=""
PHASE=build
t0=$(date +%s)
if [ -n "$OUT_TMP" ]; then
    python -u -m "weed_optimizer_framework.tools.inc.$MOD" "$CMD" "$@" 2>&1 | tee "$OUT_TMP"
    rc=${PIPESTATUS[0]}
else
    python -u -m "weed_optimizer_framework.tools.inc.$MOD" "$CMD" "$@"
    rc=$?
fi
echo "=== build exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
if [ "$rc" != 0 ]; then
    refusal=""
    if [ -n "$OUT_TMP" ]; then
        refusal="$(grep -E '^\[inc\.[a-z]+\] ERROR' "$OUT_TMP" | tail -n 5)"
        [ -n "$refusal" ] || refusal="$(tail -n 5 "$OUT_TMP")"
        rm -f "$OUT_TMP"
    fi
    prov update status=build_failed "build_rc=$rc" "refusal=$refusal" finish
    exit "$rc"
fi
[ -n "$OUT_TMP" ] && rm -f "$OUT_TMP"
prov update status=built build_rc=0

PHASE=advance
python -u -m weed_optimizer_framework.tools.inc.driver advance --exp "$EXP"
arc=$?
if [ "$arc" = 0 ]; then
    prov update status=advanced advance_rc=0 finish
else
    echo "WARNING: driver advance --exp $EXP exited $arc; the next advance (the ticker's) retries" >&2
    prov update status=built_advance_failed "advance_rc=$arc" finish
fi
echo "=== done $(date) ==="
exit 0
