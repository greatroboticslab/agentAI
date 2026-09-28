#!/bin/bash
#SBATCH --job-name=inc_funnel
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/funnel/logs/%x_%j.out
#
# The funnel audit's cluster verbs as batch jobs: docs/FUNNEL_AUDIT.md (the
# contract, section 10 F3-F10) and docs/FUNNEL_AUDIT_RUNNER.md (5.6.2, 6.1).
# One job runs one verb of the funnel CLI:
#
#   python -u -m weed_optimizer_framework.tools.funnel VERB --prereg PATH [--out DIR] [options]
#
# VERB is one of census, leak, embed-judges, qualify, draw, sheets, rl-b,
# ingest, estimate, map, recover. fetch runs on the lab (it needs the
# network) and sbatch-args is a local helper, so neither is a job.
#
# Partition. Every verb runs on GPU-shared with one V100: the allocation
# cis240145p is a GPU allocation and RM-shared submissions fail with "Invalid
# qos" (run_inc_audit.sh, run_inc_build.sh, CHANGELOG v3.0.99.19). The CPU
# verbs leave the V100 idle, as run_inc_audit.sh and run_inc_build.sh do. 45G
# is run_inc_audit.sh's size for loading every crop embedding, which census
# does too. rl-b needs an 80 GB GPU for the reference labeller's model; its
# extra flags come from the CLI's one table (SBATCH_RESOURCES):
#
#   sbatch $(python -m weed_optimizer_framework.tools.funnel sbatch-args rl-b) run_inc_funnel.sh rl-b --prereg ...
#
# Submit from $REPO (Slurm opens the log file before this script runs, so the
# log dir must exist):
#   mkdir -p $REPO/results/framework/inc/funnel/logs
#   sbatch run_inc_funnel.sh census --prereg $INC/funnel/prereg_v1.json --out $INC/funnel/
#   sbatch --array=0-3 run_inc_funnel.sh embed-judges --stage embed --nshards 4 --prereg ...
#   sbatch run_inc_funnel.sh embed-judges --stage judges --prereg ...
#
# What the job checks before the verb runs (a refusal exits 2, an environment
# failure exits 1, and the verb's own exit code is the job's otherwise):
#   * the verb is one of the list above; in an array job only
#     embed-judges --stage embed runs, and each task i gets --shard i unless
#     --shard was given;
#   * a GPU verb (class gpu: leak, embed-judges) sees a GPU; rl-b (class
#     gpu_large) sees one with at least 80 GB. Otherwise the job prints the
#     sbatch line to use and exits 2;
#   * $REPO/docs/FUNNEL_AUDIT.md exists (the prereg loader checks its sha256);
#   * the code: this job imports the git-tracked NESTED copy
#     ($REPO/weed_llm_benchmark) and logs the sha256 of every file of the
#     funnel package there, and of the INC and tools modules it calls
#     (inc/{__init__,common,verify,select,relevance,driver,gate}.py,
#     cwd12_species.py, near_dup.py, semisup_labeler.py, and mega_trainer.py,
#     whose dHash every never-train and near-duplicate check uses). It refuses when the
#     script that runs, or $REPO/run_inc_funnel.sh, differs from the nested
#     copy of this script;
#   * the imports each verb needs (numpy, sklearn, joblib; torch,
#     transformers and open_clip for the GPU verbs; torch and transformers
#     for recover, whose H6 detector computes DINOv2 descriptors; PIL for
#     leak, sheets, recover and rl-b).
#
# rl-b. ollama is started inside this allocation on a per-job port, exactly as
# run_inc_plan.sh does (the same binary and model store, one request slot,
# flash attention, the 30 min load timeout, two bounded warm-ups). The model
# tag is the domain config's reference_labeller.backends.RL-B.model, read
# through the funnel's own loader from the prereg's domain;
# FUNNEL_OLLAMA_ENDPOINT points the CLI at the server, and --model is added
# unless given (the CLI refuses any other tag).
#
# INC_FUNNEL_REPO and INC_FUNNEL_CONDA_SH replace the cluster paths below, and
# INC_FUNNEL_DRY_RUN=1 stops after every check with the command that would
# run, starting no server (tests only).
set -uo pipefail

REPO="${INC_FUNNEL_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CONDA_SH="${INC_FUNNEL_CONDA_SH:-/jet/home/byler/miniconda3/etc/profile.d/conda.sh}"
DRY_RUN="${INC_FUNNEL_DRY_RUN:-0}"
export REPO
INC_DIR="$REPO/results/framework/inc"
export INC_DIR
CODE="$REPO/weed_llm_benchmark"
FUN="$CODE/weed_optimizer_framework/tools/funnel"
LOGS="$INC_DIR/funnel/logs"
MOD=weed_optimizer_framework.tools.funnel
cd "$CODE" || exit 1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
# Compute nodes have no internet: DINOv2, BioCLIP-2 and the text towers come from the Hugging Face cache.
export HF_HUB_OFFLINE=1
RLB_NUM_CTX=8192                          # rl.OllamaClient's num_ctx (runner 5.4.2); the warm-up loads at the same
GPU_LARGE_MIB=76294                       # 80 GB (80e9 bytes) in MiB, rounded up

usage() {
    echo "usage: sbatch [\$(python -m $MOD sbatch-args VERB)] run_inc_funnel.sh VERB --prereg PATH [--out DIR] [options]" >&2
    echo "  VERB: census leak embed-judges qualify draw sheets rl-b ingest estimate map recover (fetch runs on the lab)" >&2
    exit 2
}

VERB="${1:-}"
case "$VERB" in
    census|leak|embed-judges|qualify|draw|sheets|rl-b|ingest|estimate|map|recover) shift ;;
    *) usage ;;
esac
ARGS=("$@")

# The option values the job itself needs: --stage, --shard, --prereg, --model.
arg_value() {                             # arg_value NAME: the value of --NAME or --NAME=VALUE, else empty
    local want="--$1" prev="" a
    for a in ${ARGS[@]+"${ARGS[@]}"}; do
        case "$a" in "$want="*) echo "${a#*=}"; return ;; esac
        [ "$prev" = "$want" ] && { echo "$a"; return; }
        prev="$a"
    done
}
has_arg() {                               # has_arg NAME: --NAME or --NAME=VALUE was given
    local a
    for a in ${ARGS[@]+"${ARGS[@]}"}; do
        case "$a" in "--$1"|"--$1="*) return 0 ;; esac
    done
    return 1
}

# Array jobs: only embed-judges --stage embed is sharded; task i does shard i.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$VERB" != embed-judges ] || [ "$(arg_value stage)" != embed ]; then
        echo "FATAL: only 'embed-judges --stage embed' runs as an array job (got $VERB ${ARGS[*]-})" >&2
        exit 2
    fi
    has_arg shard || ARGS+=(--shard "$SLURM_ARRAY_TASK_ID")
fi

mkdir -p "$LOGS" || exit 1
source "$CONDA_SH" || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "=== inc_funnel $VERB ${ARGS[*]-}  $(date)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "python: $(command -v python)  $(python -V 2>&1)"

# The verb's resource class, from the CLI's table (VERB_CLASS).
VCLASS="$(python -c 'import sys; from weed_optimizer_framework.tools.funnel.__main__ import VERB_CLASS; print(VERB_CLASS[sys.argv[1]])' "$VERB")"
[ -n "$VCLASS" ] || { echo "FATAL: cannot read the verb class of $VERB from $MOD" >&2; exit 1; }
echo "verb class: $VCLASS"

refuse_gpu() {
    echo "FATAL: $1" >&2
    echo "submit it as:  sbatch \$(python -m $MOD sbatch-args $VERB) run_inc_funnel.sh $VERB ${ARGS[*]-}" >&2
    exit 2
}
GPUS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)"
echo "GPU: ${GPUS:-none}"
case "$VCLASS" in
    gpu)
        [ -n "$GPUS" ] || refuse_gpu "$VERB needs a GPU and nvidia-smi lists none" ;;
    gpu_large)
        MAX_MIB=0
        while IFS=, read -r _name mib; do
            mib="${mib//[^0-9]/}"
            [ -n "$mib" ] && [ "$mib" -gt "$MAX_MIB" ] && MAX_MIB="$mib"
        done <<< "$GPUS"
        [ "$MAX_MIB" -ge "$GPU_LARGE_MIB" ] || \
            refuse_gpu "$VERB needs a GPU of at least 80 GB ($GPU_LARGE_MIB MiB); the largest listed has $MAX_MIB MiB" ;;
esac

[ -f "$REPO/docs/FUNNEL_AUDIT.md" ] || { echo "FATAL: the contract $REPO/docs/FUNNEL_AUDIT.md is missing" >&2; exit 2; }

sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi
}
# This script: the one that runs must be the nested (git-tracked) copy.
SELF="$CODE/run_inc_funnel.sh"
[ -f "$SELF" ] || { echo "FATAL: $SELF missing: the nested copy is what this job runs" >&2; exit 1; }
echo "script $SELF: $(sha "$SELF")"
if [ -r "$0" ] && ! cmp -s "$0" "$SELF"; then
    echo "FATAL: the script that runs ($0) differs from $SELF: submit the git-tracked copy" >&2
    exit 2
fi
if [ -f "$REPO/run_inc_funnel.sh" ] && ! cmp -s "$REPO/run_inc_funnel.sh" "$SELF"; then
    echo "FATAL: $REPO/run_inc_funnel.sh differs from $SELF: sync it from the nested copy" >&2
    exit 2
fi
# The code: every file of the funnel package, and the modules it calls, from the nested copy.
[ -d "$FUN" ] || { echo "FATAL: $FUN missing in the nested copy" >&2; exit 1; }
FUN_FILES="$(cd "$CODE/weed_optimizer_framework" && find tools/funnel -type f \( -name '*.py' -o -name '*.json' \) \
             ! -path '*/__pycache__/*' | LC_ALL=C sort)"
for m in $FUN_FILES tools/inc/__init__.py tools/inc/common.py tools/inc/verify.py tools/inc/select.py \
         tools/inc/relevance.py tools/inc/driver.py tools/inc/gate.py tools/cwd12_species.py tools/near_dup.py \
         tools/semisup_labeler.py tools/mega_trainer.py; do
    f="$CODE/weed_optimizer_framework/$m"
    [ -f "$f" ] || { echo "FATAL: module missing in the nested copy: $m" >&2; exit 1; }
    echo "module $m: $(sha "$f")"
done

IMPORTS="numpy, sklearn, joblib"
[ "$VCLASS" = gpu ] && IMPORTS="$IMPORTS, torch, transformers, open_clip"
# recover runs the H6 copy detector (DINOv2 descriptors) on every recovered image, unmasked and masked
[ "$VERB" = recover ] && IMPORTS="$IMPORTS, torch, transformers"
case "$VERB" in leak|sheets|recover|rl-b) IMPORTS="$IMPORTS, PIL" ;; esac
python -u -c "import $IMPORTS; print('imports: $IMPORTS')" || { echo "FATAL: the bench env lacks one of: $IMPORTS" >&2; exit 1; }

SERVE_PID=""
stop_server() {
    if [ -n "$SERVE_PID" ]; then
        kill "$SERVE_PID" 2>/dev/null
        wait "$SERVE_PID" 2>/dev/null
        SERVE_PID=""
    fi
}
trap stop_server EXIT

if [ "$VERB" = rl-b ]; then
    PREREG="$(arg_value prereg)"
    [ -n "$PREREG" ] || { echo "FATAL: rl-b needs --prereg (the model tag comes from its domain)" >&2; exit 2; }
    RLB_MODEL="$(python -c '
import sys
from weed_optimizer_framework.tools.funnel import domain as D
_pre, dom = D.load_pair(sys.argv[1])
print(((dom.raw.get("reference_labeller") or {}).get("backends") or {}).get("RL-B", {}).get("model") or "")
' "$PREREG")"
    [ -n "$RLB_MODEL" ] || { echo "FATAL: the domain of $PREREG names no reference_labeller.backends.RL-B.model" >&2; exit 2; }
    # GPU-shared nodes host several jobs at once; a fixed port attaches one job to another job's server.
    PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))
    export OLLAMA_HOST="127.0.0.1:$PORT"
    export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
    export OLLAMA_KEEP_ALIVE=30m
    # One request per job: one KV cache, not one per parallel slot.
    export OLLAMA_NUM_PARALLEL=1
    # No f32 KQ buffer on top of the KV cache.
    export OLLAMA_FLASH_ATTENTION=1
    # A large load off Lustre outruns the default start timeout (run_inc_plan.sh, job 45765065).
    export OLLAMA_LOAD_TIMEOUT=30m
    export FUNNEL_OLLAMA_ENDPOINT="http://127.0.0.1:$PORT"
    has_arg model || ARGS+=(--model "$RLB_MODEL")
    echo "[cfg] model=$RLB_MODEL endpoint=$FUNNEL_OLLAMA_ENDPOINT num_ctx=$RLB_NUM_CTX"
    if [ "$DRY_RUN" != 1 ]; then
        /ocean/projects/cis240145p/byler/ollama/bin/ollama serve > "$LOGS/inc_funnel_serve_${SLURM_JOB_ID:-0}.log" 2>&1 &
        SERVE_PID=$!
        for i in $(seq 1 60); do
            curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
            sleep 2
        done
        TAGS="$LOGS/inc_funnel_tags_${SLURM_JOB_ID:-0}.json"
        curl -sf "http://127.0.0.1:$PORT/api/tags" -o "$TAGS" 2>/dev/null || {
            echo "FATAL: ollama did not come up on port $PORT" >&2; exit 1; }
        echo "[ollama] serving on $PORT"
        TAGS="$TAGS" MODEL="$RLB_MODEL" python -c '
import json, os, sys
have = {m.get("name") for m in json.load(open(os.environ["TAGS"])).get("models", [])}
tag = os.environ["MODEL"]
sys.exit(0 if tag in have or (":" not in tag and tag + ":latest" in have) else 1)
' || { echo "FATAL: $RLB_MODEL is not in the ollama store $OLLAMA_MODELS" >&2; exit 1; }
        # Load the weights before the first real request (run_inc_plan.sh): two bounded attempts.
        echo "[ollama] warming $RLB_MODEL $(date)"
        WARM_RC=1
        for attempt in 1 2; do
            if curl -sf -m 1200 -X POST "http://127.0.0.1:$PORT/api/generate" \
                 -H 'Content-Type: application/json' \
                 -d "{\"model\":\"$RLB_MODEL\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_ctx\":$RLB_NUM_CTX,\"num_predict\":1}}" \
                 -o "$LOGS/inc_funnel_warm_${SLURM_JOB_ID:-0}.json" 2>/dev/null; then
                WARM_RC=0; break
            fi
            echo "[ollama] warm attempt $attempt did not answer; retrying $(date)"
        done
        [ "$WARM_RC" = 0 ] || { echo "FATAL: $RLB_MODEL never loaded on this node after 2 attempts" >&2; exit 1; }
        echo "[ollama] $RLB_MODEL loaded $(date)"
    fi
fi

t0=$(date +%s)
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN: python -u -m $MOD $VERB ${ARGS[*]-}"
    rc=0
else
    python -u -m "$MOD" "$VERB" ${ARGS[@]+"${ARGS[@]}"}
    rc=$?
fi
stop_server
echo "=== exit $rc after $(( $(date +%s) - t0 ))s  $(date) ==="
exit $rc
