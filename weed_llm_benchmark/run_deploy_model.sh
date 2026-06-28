#!/bin/bash
#SBATCH --job-name=deploy_model
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/model_deploy_%j.out
#
# v3.0.153 — DEPLOY a model onto our cluster so it becomes selectable in the
# agent/training model dropdowns. This is the gate the user requested: big LLMs
# (DeepSeek-V4, latest GLM, ...) only appear in the catalog AFTER this job has
# successfully PULLED the model and verified it can actually generate on our GPU.
# Same on-demand pattern as run_llm_infer.sh — no persistent server.
#   env: DEPLOY_MODEL  (ollama tag, e.g. deepseek-v4, glm4 — must exist in ollama lib)
#        DEPLOY_JOBTAG (names the result file)
#   writes: results/framework/model_deploy/<JOBTAG>.json  {ok,model,sample,...}
set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
git fetch origin 2>&1 | tail -1
git reset --hard origin/main 2>&1 | tail -1
[ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ] && \
  cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/" 2>/dev/null

export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models
OLLAMA_BIN=/ocean/projects/cis240145p/byler/ollama/bin/ollama
MODEL="${DEPLOY_MODEL:?DEPLOY_MODEL required}"
JOBTAG="${DEPLOY_JOBTAG:-job${SLURM_JOB_ID:-0}}"
DIR="results/framework/model_deploy"; mkdir -p "$DIR"
OUT="$DIR/${JOBTAG}.json"

echo "=== deploy model=$MODEL jobtag=$JOBTAG ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && { echo "ollama ready (${i}s)"; break; }
  sleep 1
done

# Pull the model (may be large — that's why this is a batch job).
PULL_LOG=$("$OLLAMA_BIN" pull "$MODEL" 2>&1 | tail -3)
echo "$PULL_LOG"

DEPLOY_MODEL="$MODEL" JOBTAG="$JOBTAG" OUT="$OUT" PULL_LOG="$PULL_LOG" python - <<'PYEOF'
import os, json, time, urllib.request
res = {"jobtag": os.environ["JOBTAG"], "model": os.environ["DEPLOY_MODEL"],
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "pull_log": os.environ.get("PULL_LOG","")}
try:
    data = json.dumps({"model": os.environ["DEPLOY_MODEL"],
                       "prompt": "Reply with exactly: DEPLOY_OK",
                       "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    sample = (out.get("response") or "").strip()
    res["ok"] = bool(sample)          # generated something → it loads + runs on our GPU
    res["sample"] = sample[:200]
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {e}"
json.dump(res, open(os.environ["OUT"], "w"), indent=2, default=str)
print("wrote", os.environ["OUT"], "ok=", res.get("ok"))
PYEOF
kill $OLLAMA_PID 2>/dev/null
echo "=== DONE $(date) ==="
