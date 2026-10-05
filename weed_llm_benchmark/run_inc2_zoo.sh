#!/bin/bash
#SBATCH --job-name=inc_zoo
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/logs/%x_%A_%a.out
#
# The model-zoo audit (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-04): Z1, the
# model-zoo audit (pre-registered usage)"): every detector the project trained,
# scored under one protocol by the locked scorer. One chain of five GPU-shared
# jobs, submitted from the login node:
#
#   bash run_inc2_zoo.sh preflight --version v1          (login: read-only checks)
#   bash run_inc2_zoo.sh submit --version v1 --shards-a 32 --shards-c 16 --concurrency 4 --max-gpu-hours 40
#                                                       (login: the five jobs, the record)
#   sbatch run_inc2_zoo.sh inventory --version v1 --shards-a N --shards-c M --max-gpu-hours H
#   sbatch --array=0-(N-1) run_inc2_zoo.sh score --version v1 --stage a|c
#   sbatch run_inc2_zoo.sh select --version v1 --shards-c M
#   sbatch run_inc2_zoo.sh report --version v1
#
# submit and preflight refuse inside a Slurm job; inventory, score, select and
# report refuse outside one. submit runs inc2.zoo submit, which submits the
# chain without a hold (inventory; score array A afterok:inventory; select
# afterany:A; score array C afterok:select; report afterany:C), writes
# INC_DIR/capacity/zoo_<version>.json as each job id is obtained, cancels what
# it submitted when an sbatch fails, and prints `INCZOO {"job_ids": [...],
# "names": [...]}` last (the platform's stream_remote parses it).
#
# score: the task's exam root is prepared first (inc2.zoo exam-root prints only
# its path: a node-local copy under $LOCAL when there is room, else the Lustre
# zoo root), then the shard is scored with INC_DIR at that root and
# INC_ZOO_SOURCE at the INC tree. The node-local copy is removed on exit (the
# trap is set before the copy is made; its path is fixed by the job and task).
#
# Provenance: INC_DIR/_zoo/<version>/provenance/<name>.json (one attempt per
# run: job id, argv, INCAP_* as the platform passed them, every module's sha256
# in the outer and the nested copy, the exit status and the refusal lines);
# locks: INC_DIR/_zoo/<version>/locks/<name>.build (O_EXCL, stale holders taken
# over through squeue). Neither lives under _campaign/, so remote.status never
# reads them as builds.
#
# The job imports the OUTER package copy ($REPO/weed_optimizer_framework); an
# outer module that differs from the nested (git-tracked) copy stops the job
# (INC_BUILD_ALLOW_DRIFT=1 runs the outer ones anyway, recorded). Nothing is
# synced, reset or copied here.
#
# INC_BUILD_REPO and INC_BUILD_CONDA_SH replace the cluster paths below (tests only).
set -uo pipefail

# this script's own absolute path, taken before the cd below (a relative invocation, `bash run_inc2_zoo.sh
# submit ...` from the nested directory, resolves against the caller's directory, not $REPO)
ZOO_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="${INC_BUILD_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC_BUILD_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
export REPO
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
INC_ROOT="${INC_DIR:-$REPO/results/framework/inc}"
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false HF_HUB_OFFLINE=1

usage() {
    echo "usage: bash run_inc2_zoo.sh {submit|preflight} --version V [...]" \
         " | sbatch run_inc2_zoo.sh {inventory|score|select|report} --version V [...]" >&2
    exit 2
}

MODE="${1:-}"
[ -n "$MODE" ] || usage
shift
case "$MODE" in
    submit|preflight)
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            echo "FATAL: $MODE runs on the login node, not inside a Slurm job ($SLURM_JOB_ID)" >&2
            exit 2
        fi ;;
    inventory|score|select|report)
        if [ -z "${SLURM_JOB_ID:-}" ]; then
            echo "FATAL: $MODE runs inside a Slurm job (sbatch run_inc2_zoo.sh $MODE ...)" >&2
            exit 2
        fi ;;
    *) usage ;;
esac
VERSION=""
STAGE=""
SHARD=""
prev=""
for a in "$@"; do
    [ "$prev" = "--version" ] && VERSION="$a"
    [ "$prev" = "--stage" ] && STAGE="$a"
    [ "$prev" = "--shard" ] && SHARD="$a"
    prev="$a"
done
if ! [[ "$VERSION" =~ ^v[0-9]{1,3}$ ]]; then
    echo "FATAL: --version '$VERSION' is missing or not a zoo version" >&2
    usage
fi

# Every module inc2.zoo imports (lazily included) and every config it opens: the
# test computes the closure from a full test-world run and fails when one is missing.
# The package's own __init__ imports the agent framework (config, memory, monitor, brain, orchestrator and the
# tools they import), so those are part of every run's closure too.
MODULES=(__init__.py config.py memory.py monitor.py brain.py orchestrator.py
         tools/__init__.py tools/evaluator.py tools/label_gen.py tools/model_discovery.py tools/vlm_pool.py
         tools/web_identifier.py tools/yolo_trainer.py
         tools/inc/__init__.py tools/inc/common.py tools/inc/scorer.py tools/inc/splits.py
         tools/cwd12_species.py tools/near_dup.py tools/mega_trainer.py tools/registry_lock.py
         tools/dataset_discovery.py
         tools/funnel/__init__.py tools/funnel/leak.py tools/funnel/domain.py tools/funnel/domains/weed.json
         tools/inc2/__init__.py tools/inc2/common.py tools/inc2/guard.py tools/inc2/base3.py tools/inc2/base3_v2.json
         tools/inc2/scorer_sidecar.py tools/inc2/scorer_agnostic.py tools/inc2/zoo.py tools/inc2/zoo_v1.json)
export ZOO_MODULES="${MODULES[*]}" ZOO_SCRIPT

drift_check() {                         # 0 when the outer copy equals the nested one in every module
    local drift=0 m OUTER NESTED
    for m in "${MODULES[@]}"; do
        OUTER=$REPO/weed_optimizer_framework/$m
        NESTED=$REPO/weed_llm_benchmark/weed_optimizer_framework/$m
        if [ ! -f "$OUTER" ]; then
            echo "FATAL: $OUTER missing: deploy the outer package copy first" >&2
            return 2
        fi
        if [ -f "$NESTED" ] && ! cmp -s "$OUTER" "$NESTED"; then
            echo "module $m: outer DIFFERS from nested" >&2
            drift=1
        fi
    done
    if [ "$drift" = 1 ]; then
        if [ "${INC_BUILD_ALLOW_DRIFT:-0}" = 1 ]; then
            echo "WARNING: outer modules differ from the nested copy; running the outer ones (INC_BUILD_ALLOW_DRIFT=1)"
            return 0
        fi
        echo "FATAL: outer modules differ from the nested copy (above): sync them, or set INC_BUILD_ALLOW_DRIFT=1" >&2
        return 1
    fi
    return 0
}

source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }

if [ "$MODE" = submit ] || [ "$MODE" = preflight ]; then
    drift_check || exit 1
    exec python -m weed_optimizer_framework.tools.inc2.zoo "$MODE" "$@"
fi

# ---------------------------------------------------------------- job modes
if [ "$MODE" = score ]; then
    case "$STAGE" in a|c) ;; *) echo "FATAL: score needs --stage a|c" >&2; usage ;; esac
    I="${SHARD:-${SLURM_ARRAY_TASK_ID:-}}"
    if ! [[ "$I" =~ ^[0-9]{1,3}$ ]]; then
        echo "FATAL: score needs --shard or an array task index" >&2
        usage
    fi
    NAME="zoo_${VERSION}_score_${STAGE}_t$(printf '%03d' "$I")"
else
    NAME="zoo_${VERSION}_${MODE}"
fi
ZOO_D="$INC_ROOT/_zoo/$VERSION"
PROV="$ZOO_D/provenance/$NAME.json"
LOCK="$ZOO_D/locks/$NAME.build"
mkdir -p "$(dirname "$PROV")" "$(dirname "$LOCK")" || exit 1
HAVE_LOCK=0
LOCAL_ROOT=""
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
        echo "FATAL: another run of $NAME holds $LOCK ($held; its job is $hs): refused, nothing written." >&2
        exit 1
    fi
fi
PHASE=start
PROV_STARTED=0
on_exit() {
    [ -n "${OUT_TMP:-}" ] && rm -f "$OUT_TMP"
    # the node-local exam copy only: its path lies under $LOCAL, never the Lustre zoo root
    if [ -n "$LOCAL_ROOT" ] && [ -n "${LOCAL:-}" ] && [ "${LOCAL_ROOT#"$LOCAL"/}" != "$LOCAL_ROOT" ]; then
        rm -rf "$LOCAL_ROOT"
    fi
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

echo "=== inc2.zoo $MODE $*  $(date)  job ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_JOB_ID:-}-${SLURM_ARRAY_TASK_ID:-}) on $(hostname) ==="
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
export ZOO_NAME="$NAME" ZOO_MODULES

# prov start -- ARGV... : append an attempt; prov update KEY=VALUE ... : set fields
# of the last attempt (a VALUE that parses as JSON is stored as JSON; 'finish'
# stamps finished_utc). Written atomically (tmp + rename).
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
    rec = {"format": "inc2-zoo-job-provenance/1", "name": env["ZOO_NAME"], "script": "run_inc2_zoo.sh",
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
    for m in env.get("ZOO_MODULES", "").split():
        mods[m] = {"outer": sha(os.path.join(repo, "weed_optimizer_framework", m)),
                   "nested": sha(os.path.join(repo, "weed_llm_benchmark", "weed_optimizer_framework", m))}
    rec["attempts"].append({
        "job_id": env.get("SLURM_JOB_ID"), "array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env.get("SLURM_ARRAY_TASK_ID"), "host": socket.gethostname(), "started_utc": now,
        "argv": argv, "parent_exp": env.get("INCAP_PARENT_EXP") or None,
        "trigger": [t for t in env.get("INCAP_TRIGGER", "").split(",") if t],
        "approval_id": env.get("INCAP_APPROVAL_ID") or None, "decided_by": env.get("INCAP_DECIDED_BY") or None,
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
        if k in ("refusal",):
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

prov start -- "$MODE" "$@"
PROV_STARTED=1
PHASE=drift_check
drift_check
dc=$?
if [ "$dc" != 0 ]; then
    prov update status=refused_drift finish
    exit 1
fi

OUT_TMP="$(mktemp "${TMPDIR:-/tmp}/inc2_zoo_${NAME}.XXXXXX")" || OUT_TMP=""
run_py() {                              # run_py ARGS...: python -u -m inc2.zoo ARGS, output also to OUT_TMP
    if [ -n "$OUT_TMP" ]; then
        python -u -m weed_optimizer_framework.tools.inc2.zoo "$@" 2>&1 | tee "$OUT_TMP"
        return "${PIPESTATUS[0]}"
    fi
    python -u -m weed_optimizer_framework.tools.inc2.zoo "$@"
}

t0=$(date +%s)
if [ "$MODE" = score ]; then
    PHASE=exam_root
    if [ -n "${LOCAL:-}" ]; then
        # the path exam-root uses for a node-local copy (inc2.zoo exam_root_cmd), fixed before it is made
        LOCAL_ROOT="$LOCAL/inc_zoo_${VERSION}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${STAGE}${I}"
    fi
    ROOT="$(INC_DIR="$INC_ROOT" python -m weed_optimizer_framework.tools.inc2.zoo exam-root --version "$VERSION" \
            --stage "$STAGE" --shard "$I")"
    rrc=$?
    if [ "$rrc" != 0 ] || [ -z "$ROOT" ]; then
        prov update status=exam_root_failed "rc=$rrc" finish
        exit 1
    fi
    echo "exam root: $ROOT"
    PHASE=score
    INC_ZOO_SOURCE="$INC_ROOT" INC_DIR="$ROOT" run_py score --version "$VERSION" --stage "$STAGE" --shard "$I"
    rc=$?
else
    PHASE="$MODE"
    INC_DIR="$INC_ROOT" run_py "$MODE" "$@"
    rc=$?
fi
echo "=== inc2.zoo $MODE exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
if [ "$rc" != 0 ]; then
    refusal=""
    if [ -n "$OUT_TMP" ]; then
        refusal="$(grep -E '^\[inc2\.zoo\] ERROR' "$OUT_TMP" | tail -n 5)"
        [ -n "$refusal" ] || refusal="$(tail -n 5 "$OUT_TMP")"
    fi
    prov update status=failed "rc=$rc" "refusal=$refusal" finish
    exit "$rc"
fi
prov update status=done rc=0 finish
echo "=== done $(date) ==="
exit 0
