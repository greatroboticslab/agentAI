#!/bin/bash
#SBATCH --job-name=llm_infer
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/framework/llm_infer_%j.out
#
# v3.0.143 — ON-DEMAND model inference job (the cluster side of our self-hosted
# model gateway). The always-on lab server submits this when an authenticated
# caller hits /api/llm/infer; it spins up OUR model on a GPU, answers ONE prompt,
# writes the result to a file, and exits. No paid API, no persistent server —
# same on-demand pattern as the harvest/Gemma agent.
#   env: LLM_MODEL (ollama tag, default gemma4 — already cached)
#        LLM_JOBTAG (names the prompt+result files)
#   reads:  results/framework/llm_infer/<JOBTAG>.prompt
#   writes: results/framework/llm_infer/<JOBTAG>.json  {ok,text,model,...}
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
MODEL="${LLM_MODEL:-gemma4}"
JOBTAG="${LLM_JOBTAG:-job${SLURM_JOB_ID:-0}}"
DIR="results/framework/llm_infer"; mkdir -p "$DIR"
OUT="$DIR/${JOBTAG}.json"
PROMPT_FILE="$DIR/${JOBTAG}.prompt"

echo "=== llm_infer model=$MODEL jobtag=$JOBTAG ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && { echo "ollama ready (${i}s)"; break; }
  sleep 1
done
"$OLLAMA_BIN" pull "$MODEL" 2>&1 | tail -2

LLM_MODEL="$MODEL" JOBTAG="$JOBTAG" OUT="$OUT" PROMPT_FILE="$PROMPT_FILE" python - <<'PYEOF'
import os, json, time, urllib.request
pf = os.environ["PROMPT_FILE"]
prompt = open(pf, encoding="utf-8", errors="replace").read() if os.path.isfile(pf) else "Reply with: OK"
res = {"jobtag": os.environ["JOBTAG"], "model": os.environ["LLM_MODEL"],
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
try:
    data = json.dumps({"model": os.environ["LLM_MODEL"], "prompt": prompt,
                       "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    res["ok"] = True
    res["text"] = out.get("response", "")
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {e}"
json.dump(res, open(os.environ["OUT"], "w"), indent=2, default=str)
print("wrote", os.environ["OUT"], "ok=", res.get("ok"))
PYEOF
kill $OLLAMA_PID 2>/dev/null
echo "=== DONE $(date) ==="
