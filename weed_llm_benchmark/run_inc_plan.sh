#!/bin/bash
#SBATCH --job-name=inc_plan
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/_campaign/plans/logs/%x_%j.out
#
# The INC research brain: one staged, dev-only digest in, one plan reply out
# (docs/INC_AUTOPILOT.md (c)). Cloned from run_llm_review.sh: ollama inside this
# allocation on a per-job port, the weights warmed before the real request, the
# model from model_router's "planner" role. Brains that analyse or decide run on
# a cluster model as a job; the lab never calls a model for this.
#
# The plan is advisory and never on the critical path. The lab stages the digest,
# submits this job, and goes on with its deterministic proposal; it pulls the
# reply back, validates it item by item (inc_autopilot/validate.py) and gives up
# on it after 2 h (brain_plan.PLAN_TIMEOUT_S, the walltime above). Every exit path
# after the input check writes PLAN_OUTPUT, so a failure is seen at once.
#
# Differences from run_llm_review.sh, on purpose:
#   * No rsync of the nested package over the outer copy. Running INC experiments
#     import pinned modules from the outer copy; rewriting it under them is the
#     drift the INC scripts refuse. This job imports the git-tracked NESTED copy
#     ($REPO/weed_llm_benchmark) instead, and logs the sha256 of what it imports.
#   * The model is the first of the planner role's model and fallbacks that is in
#     the ollama store, so a planner tag that was never pulled falls back to the
#     next one instead of failing the load.
#   * Warm-up retries are bounded so warm-up (<= 2 x 1200 s) plus the completion
#     (PLAN_TIMEOUT, default 3000 s) fit inside the 2 h walltime.
#
# Context and GPU. The digest chooses its own num_ctx from its size (brain_plan.
# choose_num_ctx): at least 49152, at most brain_plan.MAX_NUM_CTX = 98304. The
# job must hold the largest, and one H100 80 GB on GPU-shared does. The planner
# (qwen3.8:27b, manifest context_length 262144) loads about 19 GB of Q4 weights.
# Its attention layout is not recorded in the repo, so the KV cache is sized on
# an ASSUMED (unverified) dense-32B layout, 256 KiB per token at f16: 25.8 GB at
# 98304 tokens, 50.8 GB in all with about 6 GB of runtime. The cap also fits at
# twice that per token (a dense 27B with 16 KV heads or head_dim 256): 76.5 GB
# of 80. 131072 would need 93.7 GB at twice the KV, and the model's full 262144
# would need 93.7 GB even under the assumption.
# OLLAMA_NUM_PARALLEL=1 keeps ollama to one KV cache of num_ctx, not one per
# parallel slot. OLLAMA_FLASH_ATTENTION=1 keeps llama.cpp from allocating the
# f32 KQ buffer (num_ctx x batch x query heads, about 13 GB at 98304 tokens with
# a 512 batch and 64 heads) that the 6 GB of runtime does not cover. A num_ctx
# over the cap (a PLAN_NUM_CTX override) is refused before the server starts.
# Before warm-up the job logs the model's own layout (/api/show model_info,
# brain_plan.layout_line): its KV bytes per token against the assumed ones and
# what num_ctx needs on this GPU. After warm-up it logs how much of the model is
# resident on the GPU (/api/ps size_vram against size). The first run settles
# the assumption.
#
#   env: PLAN_INPUT    staged digest json, under $REPO/results/framework/inc/_campaign/plans/  (required)
#        PLAN_OUTPUT   reply json (default: PLAN_INPUT with .input.json -> .json)
#        PLAN_MODEL    ollama tag (default: model_router.resolve("planner") and its fallbacks)
#        PLAN_NUM_CTX  context the server is asked to hold  (default: the digest's num_ctx;
#                      at most brain_plan.MAX_NUM_CTX)
#        PLAN_TIMEOUT  seconds to wait for the completion   (default 3000)
#
# Submit (from $REPO; Slurm opens the log file before this script runs, so the
# log dir must exist):
#   mkdir -p $REPO/results/framework/inc/_campaign/plans/logs
#   sbatch --export=ALL,PLAN_INPUT=$REPO/results/framework/inc/_campaign/plans/<campaign>/<n>.input.json \
#          weed_llm_benchmark/run_inc_plan.sh
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
# Resolved like the paths it is compared with below: a symlink anywhere in the
# /ocean path must not make every input look outside the plans dir.
PLANS="$(realpath -m "$REPO/results/framework/inc/_campaign/plans")"
CODE="$REPO/weed_llm_benchmark"
cd "$CODE" || exit 1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

INPUT="${PLAN_INPUT:?PLAN_INPUT is required}"
case "$(realpath -m "$INPUT")" in
    "$PLANS"/*) ;;
    *) echo "FATAL: PLAN_INPUT must lie under $PLANS: $INPUT" >&2; exit 2 ;;
esac
[ -f "$INPUT" ] || { echo "FATAL: digest not found: $INPUT" >&2; exit 1; }
OUTPUT="${PLAN_OUTPUT:-${INPUT%.input.json}.json}"
case "$(realpath -m "$OUTPUT")" in
    "$PLANS"/*.json) ;;
    *) echo "FATAL: PLAN_OUTPUT must be a .json under $PLANS: $OUTPUT" >&2; exit 2 ;;
esac
[ "$OUTPUT" != "$INPUT" ] || { echo "FATAL: PLAN_OUTPUT equals PLAN_INPUT" >&2; exit 2; }
TIMEOUT="${PLAN_TIMEOUT:-3000}"

# GPU-shared nodes host several jobs at once; a fixed port attaches one job to
# another job's server.
PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }

# The digest was sized for its own num_ctx; ask the server for exactly that.
NUM_CTX="${PLAN_NUM_CTX:-$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("num_ctx") or 49152))' "$INPUT" 2>/dev/null || echo 49152)}"

# A failure before the model answers still leaves a reply the lab can read.
fail() {
    PLAN_FAIL_REASON="$1" INPUT="$INPUT" OUTPUT="$OUTPUT" MODEL="${PLAN_MODEL:-}" python3 - <<'PY'
import json, os, time
out = {"schema": "inc-plan-reply/1", "ok": False, "reason": os.environ["PLAN_FAIL_REASON"],
       "model": os.environ.get("MODEL") or None, "plan": None, "parse_problems": [],
       "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "place": "cluster",
       "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    d = json.load(open(os.environ["INPUT"]))
    out.update({"campaign": d.get("campaign"), "n": d.get("n"), "exp": d.get("exp"),
                "digest_sha256": d.get("sha256")})
except Exception as exc:
    out["reason"] += "; digest unreadable: %s" % exc
tmp = os.environ["OUTPUT"] + ".tmp"
with open(tmp, "w") as fh:
    json.dump(out, fh, sort_keys=True, indent=1)
os.replace(tmp, os.environ["OUTPUT"])
PY
    echo "FATAL: $1" >&2
}

# The GPU requested above is sized for brain_plan.MAX_NUM_CTX and no more.
MAX_CTX="$(python3 -c 'from weed_optimizer_framework.tools.inc_autopilot import brain_plan as B; print(B.MAX_NUM_CTX)' 2>/dev/null || echo 98304)"
case "$NUM_CTX" in
    ''|*[!0-9]*) fail "num_ctx is not a positive integer: $NUM_CTX"; exit 2 ;;
esac
if [ "$NUM_CTX" -gt "$MAX_CTX" ]; then
    fail "num_ctx $NUM_CTX is over MAX_NUM_CTX $MAX_CTX, the most this job's GPU is sized for"
    exit 2
fi

echo "=== inc_plan job=${SLURM_JOB_ID:-none} port=$PORT ctx=$NUM_CTX (max $MAX_CTX) $(date) on $(hostname) ==="
echo "[cfg] input=$INPUT"
echo "[cfg] output=$OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in tools/inc_autopilot/__init__.py tools/inc_autopilot/model.py tools/inc_autopilot/brain_plan.py \
         tools/inc_autopilot/corpus.py tools/brain/supervisor.py tools/model_router.py; do
    f="$CODE/weed_optimizer_framework/$m"
    [ -f "$f" ] || { fail "module missing in the nested copy: $m"; exit 1; }
    echo "module $m: $(sha256sum "$f" | cut -c1-16)"
done

export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
export OLLAMA_KEEP_ALIVE=30m
# One request per job: one KV cache of NUM_CTX, not one per parallel slot.
export OLLAMA_NUM_PARALLEL=1
# No f32 KQ buffer of NUM_CTX x batch x heads on top of the KV cache (the GPU
# sizing in brain_plan.MAX_NUM_CTX assumes flash attention).
export OLLAMA_FLASH_ATTENTION=1
# A 19 GB load off Lustre outruns the default start timeout (job 45765065).
export OLLAMA_LOAD_TIMEOUT=30m
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve > "$PLANS/logs/inc_plan_serve_${SLURM_JOB_ID:-0}.log" 2>&1 &
SERVE_PID=$!
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/api/tags" -o "$PLANS/logs/inc_plan_tags_${SLURM_JOB_ID:-0}.json" 2>/dev/null || {
    fail "ollama did not come up on port $PORT"; kill $SERVE_PID 2>/dev/null; exit 1; }
echo "[ollama] serving on $PORT"

# The model comes from model_router, which owns placement: the planner role and
# then its fallbacks, first one present in the store.
if [ -z "${PLAN_MODEL:-}" ]; then
    PLAN_MODEL="$(TAGS="$PLANS/logs/inc_plan_tags_${SLURM_JOB_ID:-0}.json" python3 - <<'PY'
import json, os
from weed_optimizer_framework.tools import model_router
try:
    have = {m.get("name") for m in json.load(open(os.environ["TAGS"])).get("models", [])}
except Exception:
    have = set()
r = model_router.resolve("planner")
spec = model_router.ROLES.get("planner", {})
for cand in [r.get("model")] + list(spec.get("fallbacks", [])):
    cand = str(cand or "")
    tag = cand.partition(":")[2] if cand.split(":")[0] in ("vllm", "ollama") else cand
    if not tag:
        continue
    full = tag if ":" in tag else tag + ":latest"
    if tag in have or full in have:
        print(tag)
        break
PY
)"
fi
if [ -z "$PLAN_MODEL" ]; then
    fail "no planner-role model (or fallback) is in the ollama store"
    kill $SERVE_PID 2>/dev/null; exit 1
fi
echo "[cfg] model=$PLAN_MODEL"

# The model's own attention layout (/api/show model_info) against the layout
# brain_plan.MAX_NUM_CTX was sized on. Logged, never enforced: the resident size
# after warm-up is the real figure.
curl -sf -m 120 -X POST "http://127.0.0.1:$PORT/api/show" -H 'Content-Type: application/json' \
     -d "{\"model\":\"$PLAN_MODEL\",\"name\":\"$PLAN_MODEL\"}" 2>/dev/null \
  | NUM_CTX="$NUM_CTX" MODEL="$PLAN_MODEL" TAGS="$PLANS/logs/inc_plan_tags_${SLURM_JOB_ID:-0}.json" \
    GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)" \
    python3 -c '
import json, os, sys
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as B
try:
    info = json.load(sys.stdin).get("model_info") or {}
    weights, tag = None, os.environ["MODEL"]
    try:
        for m in json.load(open(os.environ["TAGS"])).get("models", []):
            if m.get("name") in (tag, tag + ":latest"):
                weights = (int(m.get("size") or 0) / 1e9) or None
    except Exception:
        pass
    try:
        gpu = float(os.environ.get("GPU_MIB") or 0) * 1048576 / 1e9 or B.JOB_GPU_MEM_GB
    except ValueError:
        gpu = B.JOB_GPU_MEM_GB
    print(B.layout_line(info, int(os.environ["NUM_CTX"]), weights, gpu))
except Exception as exc:
    print("[layout] /api/show unreadable: %s" % exc)
' || echo "[layout] /api/show did not answer"

# Load the weights before the real request: ollama loads on the first request and
# its internal start timeout is shorter than a large load from Lustre (job
# 45765065 came back HTTP 500 with tokens_in=0).
echo "[ollama] warming $PLAN_MODEL $(date)"
WARM_RC=1
for attempt in 1 2; do
    if curl -sf -m 1200 -X POST "http://127.0.0.1:$PORT/api/generate" \
         -H 'Content-Type: application/json' \
         -d "{\"model\":\"$PLAN_MODEL\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_ctx\":$NUM_CTX,\"num_predict\":1}}" \
         -o "$PLANS/logs/inc_plan_warm_${SLURM_JOB_ID:-0}.json" 2>/dev/null; then
        WARM_RC=0; break
    fi
    echo "[ollama] warm attempt $attempt did not answer; retrying $(date)"
done
if [ "$WARM_RC" != "0" ]; then
    fail "$PLAN_MODEL never loaded on this node after 2 attempts"
    kill $SERVE_PID 2>/dev/null; exit 1
fi
echo "[ollama] $PLAN_MODEL loaded $(date)"
# How much of the model (weights and the num_ctx KV cache) is on the GPU. Below
# size, ollama put layers on the CPU: the context is more than this GPU holds.
curl -sf "http://127.0.0.1:$PORT/api/ps" 2>/dev/null | python3 -c '
import json, sys
try:
    for m in json.load(sys.stdin).get("models", []):
        size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
        print("[ollama] resident %s: %.1f GB of %.1f GB on the GPU%s" % (
            m.get("name"), vram / 1e9, size / 1e9,
            "" if vram >= size else " -- PARTLY ON THE CPU, the context is larger than the GPU holds"))
except Exception as exc:
    print("[ollama] /api/ps unreadable: %s" % exc)
' || echo "[ollama] /api/ps did not answer"

python3 -u -m weed_optimizer_framework.tools.inc_autopilot.brain_plan run \
    --input "$INPUT" --output "$OUTPUT" --endpoint "http://127.0.0.1:$PORT/v1" \
    --model "$PLAN_MODEL" --num-ctx "$NUM_CTX" --timeout "$TIMEOUT"
RC=$?

kill $SERVE_PID 2>/dev/null
wait $SERVE_PID 2>/dev/null
echo "=== inc_plan done rc=$RC $(date) ==="
exit $RC
