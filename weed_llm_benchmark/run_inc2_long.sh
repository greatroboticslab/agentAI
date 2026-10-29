#!/bin/bash
#SBATCH --job-name=inc_long_ds1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%j.out
#
# DS1's long path (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-29): DS1, scaling
# verified training data at the weed level", DS.6, DS.9.1; lever L32X on the
# TRAIN lane):
#
#   sbatch run_inc2_long.sh inc2.ds1_long smoke --arm A             -> provenance and lock ds1_long_<a>_smoke
#   sbatch run_inc2_long.sh inc2.ds1_long train --arm A --seed K     -> ds1_long_<a>_s<K>, K = 0, 1, 2
#   sbatch --time=36:00:00 run_inc2_long.sh inc2.ds1_long train --arm S2X|S1UX --seed K --walltime-h 36
#                                                                    (DS.6's one re-submission of S2x)
#
# A = S0X, S2X, S1UX (the scaled pair) or a 1.2 M arm moved here by its loader
# projection (S1C, S1U, S2, S1CR, S0, KD, KT, KF). The first argument names the
# module (inc2.ds1_long, tools.inc2.ds1_long or
# weed_optimizer_framework.tools.inc2.ds1_long); any other module or verb is a
# usage error (exit 2). Job names follow V3F's D6: submitted as
# inc_long_ds1_<a>_<phase> (sbatch -J); the #SBATCH line is the default.
#
# Time limit 24 h on one V100-32 (the smoke projects one seed from its epoch and
# the seeds are proposed only within 0.8 x 24 h); 36 h only for S2x's (S1ux's)
# one re-submission, whose smoke projects within 28.8 h: the submitter passes
# --time=36:00:00 and --walltime-h 36, which inc2.ds1_long checks against the
# smoke. A seed that reaches its limit is killed before its export (SIGTERM: the
# provenance records "killed"); the platform records it failed.
#
# What runs: the bench env (Ultralytics 8.4.37, torch), offline (HF_HUB_OFFLINE=1,
# YOLO_OFFLINE=true, YOLO_AUTOINSTALL=false), after an import check. Nothing is
# advanced: the long path builds no inc experiment.
#
# Provenance: INC_DIR/_campaign/provenance/<name>.json (one attempt per run: job
# id, argv, INCAP_* as the autopilot passed them, the sha256 of every module below
# in the outer and the nested copy, the exit status and its ERROR lines). One job
# per name: the lock INC_DIR/_campaign/locks/<name>.build (O_EXCL), released on
# exit; a lock whose job squeue no longer knows is taken over. The job stops when
# an outer module differs from the git-tracked nested copy (INC_BUILD_ALLOW_DRIFT=1
# runs the outer copies anyway, recorded).
#
# INC_BUILD_REPO and INC_BUILD_CONDA_SH replace the cluster paths below (tests only).
set -uo pipefail

REPO="${INC_BUILD_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC_BUILD_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
export REPO

usage() {
    echo "usage: sbatch run_inc2_long.sh inc2.ds1_long {smoke --arm A | train --arm A --seed 0|1|2 [--walltime-h 24|36]}" >&2
    exit 2
}

ARG_MOD="${1:-}"
CMD="${2:-}"
MOD="${ARG_MOD#weed_optimizer_framework.}"
MOD="${MOD#tools.}"
[ "$MOD" = inc2.ds1_long ] || usage
shift 2 2>/dev/null || usage
ARM=""
SEED=""
WALL=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --arm) ARM="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --walltime-h) WALL="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
case "$ARM" in S0X|S2X|S1UX|S1C|S1U|S2|S1CR|S0|KD|KT|KF) ;; *) echo "FATAL: --arm is not a long-path arm" >&2; usage ;; esac
ARM_LC="$(echo "$ARM" | tr '[:upper:]' '[:lower:]')"
case "$CMD" in
    smoke) [ -z "$SEED" ] && [ -z "$WALL" ] || usage; NAME="ds1_long_${ARM_LC}_smoke"; ARGS=(--arm "$ARM") ;;
    train)
        case "$SEED" in 0|1|2) ;; *) echo "FATAL: train needs --seed 0, 1 or 2" >&2; usage ;; esac
        case "$WALL" in ""|24|36) ;; *) echo "FATAL: --walltime-h 24 or 36" >&2; usage ;; esac
        NAME="ds1_long_${ARM_LC}_s$SEED"
        ARGS=(--arm "$ARM" --seed "$SEED")
        [ -n "$WALL" ] && ARGS+=(--walltime-h "$WALL") ;;
    *) usage ;;
esac

cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
INC_ROOT="${INC_DIR:-$REPO/results/framework/inc}"
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
        echo "WARNING: took over the stale lock of $NAME ($held): job $holder is no longer running"
    else
        echo "FATAL: another job of $NAME holds $LOCK ($held; its job is $hs): refused, nothing written." >&2
        exit 1
    fi
fi
PHASE=start
PROV_STARTED=0
PYPID=""
on_exit() {
    [ -n "${OUT_TMP:-}" ] && rm -f "$OUT_TMP"
    [ "$HAVE_LOCK" = 1 ] && rm -f "$LOCK"
}
on_term() {
    trap - TERM INT
    echo "=== killed (SIGTERM/SIGINT) during $PHASE  $(date) ===" >&2
    # the training process cleans its attempt on SIGTERM; give it the signal and a moment
    [ -n "$PYPID" ] && kill -TERM "$PYPID" 2>/dev/null && sleep 20
    [ "$PROV_STARTED" = 1 ] && prov update status=killed "killed_during=$PHASE" finish
    exit 143
}
trap on_exit EXIT
trap on_term TERM INT

source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc2.ds1_long $CMD ${ARGS[*]}  $(date)  job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Every module inc2.ds1_long imports (directly or through inc2.ds1, inc2.ds1_data, inc2.train), outer copy (what
# runs) vs nested (git-tracked).
MODULES=(tools/inc/__init__.py tools/inc/common.py tools/inc/driver.py tools/inc/gate.py tools/inc/splits.py
         tools/inc/scorer.py tools/inc/pilot.py tools/inc/train.py tools/inc/select.py tools/inc/verify.py
         tools/inc/audit.py tools/cwd12_species.py tools/near_dup.py tools/mega_trainer.py tools/semisup_labeler.py
         tools/inc2/__init__.py tools/inc2/common.py tools/inc2/guard.py tools/inc2/recipes.py tools/inc2/train.py
         tools/inc2/base3.py tools/inc2/base3_v2.json tools/inc2/scorer_sidecar.py tools/inc2/scorer_native.py
         tools/inc2/twostage.py tools/inc2/v3fam.py tools/inc2/ds1.py tools/inc2/ds1_data.py tools/inc2/ds1_long.py
         tools/inc2/ds1_sheets.py)
export INCB_NAME="$NAME" INCB_MODULES="${MODULES[*]}"

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
    rec = {"format": "inc_autopilot.provenance/1", "exp": env["INCB_NAME"], "script": "run_inc2_long.sh",
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
        if k == "refusal":
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

prov start -- "$ARG_MOD" "$CMD" "${ARGS[@]}"
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

export HF_HUB_OFFLINE=1 YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
PHASE=env
python -u -c "import torch, ultralytics; print('torch', torch.__version__, 'ultralytics', ultralytics.__version__)" \
    || { prov update status=env_failed finish; exit 1; }

OUT_TMP="$(mktemp "${TMPDIR:-/tmp}/inc2_long_${NAME}.XXXXXX")" || OUT_TMP=""
PHASE=run
t0=$(date +%s)
if [ -n "$OUT_TMP" ]; then
    python -u -m weed_optimizer_framework.tools.inc2.ds1_long "$CMD" "${ARGS[@]}" > >(tee "$OUT_TMP") 2>&1 &
else
    python -u -m weed_optimizer_framework.tools.inc2.ds1_long "$CMD" "${ARGS[@]}" &
fi
PYPID=$!
wait "$PYPID"
rc=$?
PYPID=""
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
if [ "$rc" != 0 ]; then
    refusal=""
    if [ -n "$OUT_TMP" ]; then
        sleep 1
        refusal="$(grep -E '^\[inc2\.ds1_long\] ERROR' "$OUT_TMP" | tail -n 5)"
        [ -n "$refusal" ] || refusal="$(tail -n 5 "$OUT_TMP")"
    fi
    prov update status=failed "rc=$rc" "refusal=$refusal" finish
    exit "$rc"
fi
prov update status=done rc=0 finish
echo "=== done $(date) ==="
exit 0
