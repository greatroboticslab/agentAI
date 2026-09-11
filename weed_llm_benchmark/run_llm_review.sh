#!/bin/bash
#SBATCH --job-name=llmreview
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/llm_review_%j.out
#
# Review one evidence bundle with a CLUSTER model.
#
# Why this file exists. docs/TIERED_SUPERVISION_PLAN.md named it in two places and
# it was never written, so there was no path by which a cluster model could review
# anything. The reviewer was therefore wired to the only endpoint that could answer
# synchronously -- ollama on the lab's RTX 3060 -- and all nine campaign reviews
# were produced by a 7.6B code model while 458 GB of verified weights sat unused on
# this cluster. The standing rule (2026-07-04, restated 2026-09-11) is that any
# brain that analyses, reviews or decides runs on a cluster model, async, with the
# wait shown; the lab box hosts only the small guide.
#
# The lab cannot call a cluster model over HTTP -- compute nodes hold no persistent
# endpoint and login-to-compute TCP is unverified here -- so a cluster brain is a
# job, and this is that job. It starts ollama inside its own allocation on a
# per-job port, reviews the bundle, writes the verdict artifact in the same shape
# the in-process reviewer writes, and exits.
#
#   env: REVIEW_BUNDLE   path to the evidence bundle json            (required)
#        REVIEW_MODEL    ollama tag; default from model_router's deep_review role
#        REVIEW_TIER     tier name recorded in the artifact          (default deep)
#        REVIEW_DOMAIN   domain the artifact is filed under          (default weed)
#        REVIEW_NUM_CTX  context the server is asked to hold         (default 32768)
#        REVIEW_TIMEOUT  seconds to wait for one completion          (default 900)
#   writes: results/framework/_brain/<domain>/reviews/<ts>_r<round>_<step>.json
#           and .../cluster_latest_review.json
#
# Submit:
#   sbatch --export=ALL,REVIEW_BUNDLE=$PWD/results/framework/_brain/weed/latest_bundle.json \
#          run_llm_review.sh
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

BUNDLE="${REVIEW_BUNDLE:?REVIEW_BUNDLE is required}"
[ -f "$BUNDLE" ] || { echo "FATAL: bundle not found: $BUNDLE" >&2; exit 1; }
TIER="${REVIEW_TIER:-deep}"
DOMAIN="${REVIEW_DOMAIN:-weed}"
NUM_CTX="${REVIEW_NUM_CTX:-32768}"
TIMEOUT="${REVIEW_TIMEOUT:-900}"

# GPU-shared nodes host several jobs at once, so a fixed port silently attaches
# one job to another job's server. Same defect class this layer exists to catch.
PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))

if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rsync -a --delete "$REPO/weed_llm_benchmark/weed_optimizer_framework/" \
        "$REPO/weed_optimizer_framework/" && echo "[sync] outer package refreshed"
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }

# The model comes from model_router, which is the one component that owns
# placement. Reading a tier id straight out of a config block is how the lab 7B
# ended up doing the work; the router knows deep_review lives on the cluster.
if [ -z "${REVIEW_MODEL:-}" ]; then
    REVIEW_MODEL="$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
try:
    from weed_optimizer_framework.tools import model_router
    r = model_router.resolve("deep_review")
    m = str(r.get("model") or "")
    print(m.partition(":")[2] if m.split(":")[0] in ("vllm", "ollama") else m)
except Exception:
    print("glm-4.7-flash")
PY
)"
fi
echo "=== llm_review job=${SLURM_JOB_ID:-none} tier=$TIER model=$REVIEW_MODEL ctx=$NUM_CTX port=$PORT $(date) ==="
echo "[cfg] bundle=$BUNDLE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
export OLLAMA_KEEP_ALIVE=30m
# A 19 GB load off Lustre outruns the default start timeout; job 45765065 died on it.
export OLLAMA_LOAD_TIMEOUT=30m
/ocean/projects/cis240145p/byler/ollama/bin/ollama serve > "results/framework/llm_review_serve_${SLURM_JOB_ID:-0}.log" 2>&1 &
SERVE_PID=$!
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 || {
    echo "FATAL: ollama did not come up on port $PORT" >&2; kill $SERVE_PID 2>/dev/null; exit 1; }
echo "[ollama] serving on $PORT"

# Load the weights BEFORE the review asks anything. ollama loads a model on its
# first request, and the internal llama-server start timeout is shorter than a
# 19 GB load from Lustre: job 45765065 spent 304 s and came back
# `HTTP 500 {"error":"timed out waiting for llama-server to start"}` with
# tokens_in=0 -- a failure that looks like the model refusing rather than the
# model never arriving. Warm it with a generous client timeout and a trivial
# prompt, and refuse to run the review if it never answers.
echo "[ollama] warming $REVIEW_MODEL (this is the slow part) $(date)"
WARM_RC=1
for attempt in 1 2 3; do
    if curl -sf -m 1800 -X POST "http://127.0.0.1:$PORT/api/generate" \
         -H 'Content-Type: application/json' \
         -d "{\"model\":\"$REVIEW_MODEL\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_ctx\":$NUM_CTX,\"num_predict\":1}}" \
         -o "results/framework/llm_review_warm_${SLURM_JOB_ID:-0}.json" 2>/dev/null; then
        WARM_RC=0; break
    fi
    echo "[ollama] warm attempt $attempt did not answer; retrying $(date)"
done
if [ "$WARM_RC" != "0" ]; then
    echo "FATAL: $REVIEW_MODEL never loaded on this node after 3 attempts" >&2
    kill $SERVE_PID 2>/dev/null
    exit 1
fi
echo "[ollama] $REVIEW_MODEL loaded $(date)"

BUNDLE="$BUNDLE" TIER="$TIER" DOMAIN="$DOMAIN" PORT="$PORT" REPO="$REPO" \
REVIEW_MODEL="$REVIEW_MODEL" NUM_CTX="$NUM_CTX" TIMEOUT="$TIMEOUT" \
python3 -u - <<'PYEOF'
import json, os, sys, time, traceback
sys.path.insert(0, ".")
from weed_optimizer_framework.tools.brain import supervisor as sup

B = os.environ["BUNDLE"]; TIER = os.environ["TIER"]; DOMAIN = os.environ["DOMAIN"]
PORT = os.environ["PORT"]; REPO = os.environ["REPO"]
MODEL = os.environ["REVIEW_MODEL"]; NCTX = int(os.environ["NUM_CTX"])
TMO = int(os.environ["TIMEOUT"])
ENDPOINT = "http://127.0.0.1:%s/v1" % PORT

bundle = json.load(open(B))
step = bundle.get("step") or "unknown"
rnd = bundle.get("round")
started = time.time()
rec = {}
try:
    client = sup.OpenAICompatClient(endpoint=ENDPOINT, model=MODEL,
                                    timeout_s=TMO, api="ollama")
    rec = sup.review(bundle, client=client, num_ctx=NCTX,
                     case_id="%s_r%s_%s" % (DOMAIN, rnd, step))
except Exception as exc:
    rec = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc),
           "traceback": traceback.format_exc()[-2000:]}

rec.update({"domain": DOMAIN, "step": step, "round": rnd, "model": MODEL,
            "endpoint": ENDPOINT, "api": "ollama", "ts": started,
            "elapsed_s": round(time.time() - started, 2),
            # Stated on the record: where this ran, which tier it is, and that
            # nothing in the loop read it back.
            "place": "cluster", "tier": TIER,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "bundle_path": B, "mode": "shadow", "applied": False})

base = os.path.join(REPO, "results", "framework", "_brain", DOMAIN)
d = os.path.join(base, "reviews")
os.makedirs(d, exist_ok=True)
name = "%d_r%s_%s_%s.json" % (int(started), rnd, step, TIER)
for path in (os.path.join(d, name), os.path.join(base, "cluster_latest_review.json")):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, sort_keys=True)
    os.replace(tmp, path)

v = rec.get("verdict") or {}
print("[review] tier=%s model=%s ok=%s verdict=%s findings=%d resolved=%d "
      "tokens_in=%s %.1fs" % (TIER, MODEL, rec.get("ok"), v.get("verdict"),
                              len(v.get("findings") or []),
                              len(rec.get("accepted_findings") or []),
                              rec.get("tokens_in"), rec.get("elapsed_s", 0)))
if not rec.get("ok"):
    print("[review] reason:", str(rec.get("reason"))[:400])
print("[review] wrote", os.path.join(d, name))
PYEOF
RC=$?

kill $SERVE_PID 2>/dev/null
wait $SERVE_PID 2>/dev/null
echo "=== llm_review done rc=$RC $(date) ==="
