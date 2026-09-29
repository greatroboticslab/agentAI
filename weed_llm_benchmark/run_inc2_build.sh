#!/bin/bash
#SBATCH --job-name=inc2_build
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# The v2 (continuous loop) experiment builds as a batch job: docs/CONTINUOUS_LOOP.md
# §6.3 (L18, L20, L22, L23, L25, L27, L28) and §9 ("Files the plan leaves
# unowned": run_inc_build.sh stays the v1 builders' script, unchanged). The
# builders hash every image they check, which outlives what the login node
# lets a process run; this job runs one of them, then one driver advance of
# every experiment it built:
#
#   python -m weed_optimizer_framework.tools.inc2.splits   build | lock [...]
#   python -m weed_optimizer_framework.tools.inc2.baseline build --exp E --manifest M [--seeds ...] [...]
#   python -m weed_optimizer_framework.tools.inc2.pilot4   build --exp E [...]
#   python -m weed_optimizer_framework.tools.inc2.stream   init | build | milestone | fork | feasibility | bisect
#                                                          --stream SID [...]
#   python -m weed_optimizer_framework.tools.inc.driver    advance --exp E      (after a build that exited 0)
#
# The first argument names the module: inc2.<module>, tools.inc2.<module> or
# weed_optimizer_framework.tools.inc2.<module> (the forms the policy table and
# the levers write). Anything else is a usage error (exit 2), and so is a
# stream verb that builds nothing (commit, compare, rollback, summary ...:
# they read JSON and run where the ticker runs).
#
# The experiment an advance is run for: --exp when the builder takes one;
# for an inc2.stream verb, every "[inc2.stream] built experiment NAME" line it
# printed (a build, a milestone, Stage C, the bisect arms). The segments' and
# milestones' runs execute inc2.train through run_inc2_job.sh: this script
# exports INC_JOB_SCRIPT=$REPO/weed_llm_benchmark/run_inc2_job.sh before the
# builder runs, so the driver init inside the builder and the advance after it
# submit the v2 executor (the v1 script refuses every tsw image).
#
# Provenance: INC_DIR/_campaign/provenance/<name>.json, outside the
# experiment directory, where <name> is --exp, else stream_<sid> for an
# inc2.stream verb, else the module (splits). One attempt per run: the job id,
# the argv, INCAP_PARENT_EXP / INCAP_TRIGGER / INCAP_APPROVAL_ID /
# INCAP_DECIDED_BY / INCAP_REQUESTED_UTC as the autopilot passed them, the
# sha256 of every module below in the outer and the nested copy, the build's
# exit status and its ERROR lines, the experiments it built, the advances'
# exit statuses. A SIGTERM (scancel, the time limit) records "killed".
#
# One builder per name: the lock INC_DIR/_campaign/locks/<name>.build (O_EXCL:
# "<job id> <host> <utc>"), released on exit; a lock whose job squeue no longer
# knows is taken over. The stream module holds its own stream.lease besides.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework)
# through PYTHONPATH; deploy there first. Nothing is synced or reset here.
# Every module the v2 builders import is hashed into the log, and the job
# stops when an outer module differs from the git-tracked nested copy
# (INC_BUILD_ALLOW_DRIFT=1 runs the outer copies anyway, recorded).
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
export INC_JOB_SCRIPT="$REPO/weed_llm_benchmark/run_inc2_job.sh"

usage() {
    echo "usage: sbatch run_inc2_build.sh {inc2.splits build|lock | inc2.baseline build | inc2.pilot4 build |" \
         "inc2.stream init|build|milestone|fork|feasibility|bisect} [flags ...]" >&2
    exit 2
}

ARG_MOD="${1:-}"
CMD="${2:-}"
MOD="${ARG_MOD#weed_optimizer_framework.}"
MOD="${MOD#tools.}"
case "$MOD" in
    inc2.*) MOD="${MOD#inc2.}" ;;
    *) usage ;;
esac
case "$MOD $CMD" in
    "splits build"|"splits lock"|"baseline build"|"pilot4 build") shift 2 ;;
    "stream init"|"stream build"|"stream milestone"|"stream fork"|"stream feasibility"|"stream bisect") shift 2 ;;
    *) usage ;;
esac
EXP=""
SID=""
prev=""
for a in "$@"; do
    case "$a" in --exp=*) EXP="${a#--exp=}" ;; --stream=*) SID="${a#--stream=}" ;; esac
    [ "$prev" = "--exp" ] && EXP="$a"
    [ "$prev" = "--stream" ] && SID="$a"
    prev="$a"
done
NAME_RE='^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'
if [ "$MOD" = stream ]; then
    if ! [[ "$SID" =~ $NAME_RE ]]; then
        echo "FATAL: --stream '$SID' is missing or not a stream name" >&2
        usage
    fi
    NAME="stream_$SID"
elif [ -n "$EXP" ]; then
    if ! [[ "$EXP" =~ $NAME_RE ]]; then
        echo "FATAL: --exp '$EXP' is not an experiment name" >&2
        usage
    fi
    NAME="$EXP"
elif [ "$MOD" = splits ]; then
    NAME="splits"
else
    echo "FATAL: inc2.$MOD $CMD needs --exp" >&2
    usage
fi
PROV="$INC_ROOT/_campaign/provenance/$NAME.json"
mkdir -p "$(dirname "$PROV")" || exit 1

LOCK="$INC_ROOT/_campaign/locks/$NAME.build"
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
        echo "WARNING: took over the stale build lock of $NAME ($held): job $holder is no longer running"
    else
        echo "FATAL: another build of $NAME holds $LOCK ($held; its job is $hs): refused, nothing written." >&2
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
echo "=== inc2.$MOD $CMD $*  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "INC_JOB_SCRIPT=$INC_JOB_SCRIPT"

# Every module the v2 builders import (lazily included), outer copy (what runs)
# vs nested (git-tracked). inc2.splits build runs the embedding copy scan
# (funnel.embed, funnel.leak, funnel.estimate, semisup_labeler) and reads the
# funnel domain config (licences, lab groups, the embedder, the augmentation
# families): the same set run_inc2_splits.sh checks (funnel.estimate imports
# funnel.domain, which imports funnel.ledger). inc2.stream reads the queue
# through inc2.step1_stream, whose library modules are listed too.
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/driver.py tools/inc/gate.py tools/inc/splits.py
         tools/inc/scorer.py tools/inc/pilot.py tools/inc/train.py tools/inc/select.py tools/inc/verify.py
         tools/inc/report.py tools/cwd12_species.py tools/near_dup.py tools/mega_trainer.py
         tools/semisup_labeler.py
         tools/funnel/__init__.py tools/funnel/leak.py tools/funnel/embed.py tools/funnel/estimate.py
         tools/funnel/domain.py tools/funnel/ledger.py tools/funnel/domains/weed.json
         tools/inc2/__init__.py tools/inc2/common.py tools/inc2/guard.py tools/inc2/splits.py tools/inc2/recipes.py
         tools/inc2/embed_calibration.py
         tools/inc2/train.py tools/inc2/baseline.py tools/inc2/pilot4.py tools/inc2/gate3.py
         tools/inc2/scorer_sidecar.py tools/inc2/step1_stream.py tools/inc2/mask.py tools/inc2/stream.py
         tools/inc2/stream_report.py)
export INCB_NAME="$NAME" INCB_MODULES="${MODULES[*]}"

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
    rec = {"format": "inc_autopilot.provenance/1", "exp": env["INCB_NAME"], "script": "run_inc2_build.sh",
           "attempts": []}


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
        "job_script": env.get("INC_JOB_SCRIPT"),
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
        if k in ("refusal", "missing", "built_exp"):     # text, whatever it looks like
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

OUT_TMP="$(mktemp "${TMPDIR:-/tmp}/inc2_build_${NAME}.XXXXXX")" || OUT_TMP=""
PHASE=build
t0=$(date +%s)
if [ -n "$OUT_TMP" ]; then
    python -u -m "weed_optimizer_framework.tools.inc2.$MOD" "$CMD" "$@" 2>&1 | tee "$OUT_TMP"
    rc=${PIPESTATUS[0]}
else
    python -u -m "weed_optimizer_framework.tools.inc2.$MOD" "$CMD" "$@"
    rc=$?
fi
echo "=== build exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
BUILT=""
if [ -n "$OUT_TMP" ]; then
    BUILT="$(sed -n 's/^\[inc2\.stream\] built experiment \([A-Za-z0-9][A-Za-z0-9_-]*\)$/\1/p' "$OUT_TMP" | tr '\n' ' ')"
fi
if [ "$rc" != 0 ]; then
    refusal=""
    if [ -n "$OUT_TMP" ]; then
        refusal="$(grep -E '^\[inc2\.[a-z0-9_]+\] ERROR' "$OUT_TMP" | tail -n 5)"
        [ -n "$refusal" ] || refusal="$(tail -n 5 "$OUT_TMP")"
    fi
    prov update status=build_failed "build_rc=$rc" "refusal=$refusal" "built_exp=$BUILT" finish
    exit "$rc"
fi
if [ "$MOD" != stream ] && [ -n "$EXP" ]; then
    BUILT="$EXP"
fi
BUILT="$(echo "$BUILT" | xargs)"
prov update status=built build_rc=0 "built_exp=$BUILT"

PHASE=advance
arc_all=0
for e in $BUILT; do
    python -u -m weed_optimizer_framework.tools.inc.driver advance --exp "$e"
    arc=$?
    if [ "$arc" != 0 ]; then
        echo "WARNING: driver advance --exp $e exited $arc; the next advance (the ticker's) retries" >&2
        arc_all=$arc
    fi
done
if [ "$arc_all" = 0 ]; then
    prov update status=advanced advance_rc=0 finish
else
    prov update status=built_advance_failed "advance_rc=$arc_all" finish
fi
echo "=== done $(date) ==="
exit 0
