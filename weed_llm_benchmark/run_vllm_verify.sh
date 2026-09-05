#!/bin/bash
#SBATCH --job-name=vllm_verify
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=results/framework/vllm_verify_%j.out
#
# v3.25.0 — verify that a locally staged open-weight model actually serves on this
# cluster, and MEASURE what it costs, before any part of the platform is allowed to
# depend on it. The model catalog must only ever contain models that have generated
# on our own GPUs (same rule as run_deploy_model.sh), and a supervisor tier that is
# specified in tokens per second needs those numbers measured, not assumed.
#
# Serving runs inside the apptainer image so no cluster Python environment has to
# carry vLLM. The server binds 127.0.0.1 on a per-job port: GPU-shared nodes host
# several jobs at once and a fixed port silently attaches one job to another job's
# server (the same class of defect this campaign exists to catch).
#
#   env: VLLM_MODEL_PATH   (default the staged DeepSeek-V4-Flash checkpoint)
#        VLLM_SERVED_NAME  (name the OpenAI-compatible API answers to)
#        VLLM_TP           (tensor-parallel size; must equal the GPU count)
#        VLLM_MAXLEN       (max model len)
#        VLLM_GPU_UTIL     (fraction of each GPU vLLM may use)
#        VLLM_JOBTAG       (names the result file)
#        VLLM_KV_DTYPE     (--kv-cache-dtype; some models require fp8)
#        VLLM_EXTRA        (extra flags appended verbatim to vllm serve)
#   writes: results/framework/model_deploy/<JOBTAG>.json
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

SIF=/ocean/projects/cis240145p/byler/containers/vllm-openai.sif
MODEL="${VLLM_MODEL_PATH:-/ocean/projects/cis240145p/byler/models/hf/DeepSeek-V4-Flash}"
NAME="${VLLM_SERVED_NAME:-deepseek-v4-flash}"
TP="${VLLM_TP:-4}"
MAXLEN="${VLLM_MAXLEN:-65536}"
UTIL="${VLLM_GPU_UTIL:-0.90}"
EXTRA="${VLLM_EXTRA:-}"
# DeepSeek-V4's attention in vLLM 0.28 asserts an fp8 KV cache
# (_resolve_dsv4_kv_cache_dtype), so the default "auto" aborts every worker at
# model-build time. Kept as a variable because it is a per-model property.
KVDTYPE="${VLLM_KV_DTYPE:-}"
JOBTAG="${VLLM_JOBTAG:-job${SLURM_JOB_ID:-0}}"
PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))
OUTDIR="$REPO/results/framework/model_deploy"
OUT="$OUTDIR/${JOBTAG}.json"
LOG="$REPO/results/framework/vllm_serve_${JOBTAG}.log"
mkdir -p "$OUTDIR"

echo "=== vllm_verify model=$MODEL name=$NAME tp=$TP maxlen=$MAXLEN kv=${KVDTYPE:-auto} port=$PORT $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
[ -f "$SIF" ] || { echo "FATAL: container image missing: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "FATAL: model directory missing: $MODEL" >&2; exit 1; }

export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_VLLM_LOGGING_LEVEL=INFO
export APPTAINERENV_HF_HOME=/ocean/projects/cis240145p/byler/hf_cache
export APPTAINERENV_OMP_NUM_THREADS=8

# Serving in the SBATCH main body, never under `srun --overlap`: a step's cgroup
# takes its children with it when the step ends.
apptainer exec --nv -B /ocean "$SIF" \
    vllm serve "$MODEL" \
      --served-model-name "$NAME" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization "$UTIL" \
      --host 127.0.0.1 --port "$PORT" \
      --trust-remote-code \
      ${KVDTYPE:+--kv-cache-dtype "$KVDTYPE"} \
      $EXTRA > "$LOG" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null' EXIT

T0=$(date +%s)
READY=0
for i in $(seq 1 240); do          # up to 40 min: a 150 GB checkpoint loads from Lustre
    if ! kill -0 $SERVE_PID 2>/dev/null; then
        # An unrecognised flag or a bad checkpoint kills the server in seconds and
        # the poll would otherwise report only "never became healthy". Print what
        # it actually said, and stop waiting.
        echo "server process exited early after ${i}0s; last log lines:"; tail -40 "$LOG"; break
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 10
done
LOAD_S=$(( $(date +%s) - T0 ))
echo "ready=$READY after ${LOAD_S}s"

MODEL="$MODEL" NAME="$NAME" PORT="$PORT" OUT="$OUT" LOG="$LOG" READY="$READY" \
LOAD_S="$LOAD_S" TP="$TP" MAXLEN="$MAXLEN" JOBTAG="$JOBTAG" python3 - <<'PYEOF'
import json, os, time, urllib.request, urllib.error

port = os.environ["PORT"]; base = "http://127.0.0.1:%s/v1" % port
res = {"jobtag": os.environ["JOBTAG"], "kind": "vllm", "model_path": os.environ["MODEL"],
       "served_name": os.environ["NAME"], "tp": int(os.environ["TP"]),
       "max_model_len": int(os.environ["MAXLEN"]), "load_s": int(os.environ["LOAD_S"]),
       "kv_cache_dtype": os.environ.get("KVDTYPE") or "auto",
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False}

def post(path, payload, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t

if os.environ["READY"] != "1":
    res["error"] = "server never became healthy"
    try:
        res["log_tail"] = open(os.environ["LOG"], errors="replace").read()[-4000:]
    except Exception:
        pass
else:
    try:
        # 1. smallest possible generation: proves the weights load and run here.
        out, dt = post("/chat/completions", {
            "model": os.environ["NAME"], "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with exactly: SERVE_OK"}]})
        res["sample"] = out["choices"][0]["message"]["content"].strip()[:200]
        res["smoke_s"] = round(dt, 2)
        res["ok"] = bool(res["sample"])

        # 2. prefill cost at review scale. A supervisor review carries an evidence
        # bundle of roughly this size, so this is the number the cadence is built on.
        filler = ("Cluster job log line: epoch %d/60 map50-95 0.5%03d elapsed 421s "
                  "walltime 43200s pool 8583 iterations per epoch.\n")
        big = "".join(filler % (i % 60 + 1, i % 1000) for i in range(1400))
        for label, prompt in (("prefill_12k", big[:48000]),):
            out, dt = post("/chat/completions", {
                "model": os.environ["NAME"], "max_tokens": 8, "temperature": 0,
                "messages": [{"role": "user", "content":
                              prompt + "\n\nAnswer with one word: OK"}]})
            res[label + "_s"] = round(dt, 2)
            res[label + "_tokens_in"] = out.get("usage", {}).get("prompt_tokens")

        # 3. decode rate at batch 1 — the other half of the latency budget.
        out, dt = post("/chat/completions", {
            "model": os.environ["NAME"], "max_tokens": 256, "temperature": 0,
            "messages": [{"role": "user", "content":
                          "Write 250 words about crop field imagery."}]})
        gen = out.get("usage", {}).get("completion_tokens") or 0
        res["decode_tokens"] = gen
        res["decode_s"] = round(dt, 2)
        res["tok_s"] = round(gen / dt, 2) if dt > 0 else None

        # 4. strict JSON output — every supervisor verdict is schema-validated, so a
        # model that cannot be held to a schema cannot hold the role.
        schema = {"type": "object",
                  "properties": {"verdict": {"type": "string"},
                                 "confidence": {"type": "number"}},
                  "required": ["verdict", "confidence"]}
        try:
            out, dt = post("/chat/completions", {
                "model": os.environ["NAME"], "max_tokens": 128, "temperature": 0,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "schema": schema}},
                "messages": [{"role": "user", "content":
                              "A training job hit its walltime at epoch 24 of 60. "
                              "Reply with verdict and confidence."}]})
            txt = out["choices"][0]["message"]["content"]
            res["json_mode"] = json.loads(txt)
            res["json_mode_ok"] = True
            res["json_mode_s"] = round(dt, 2)
        except Exception as e:
            res["json_mode_ok"] = False
            res["json_mode_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    except Exception as e:
        res["ok"] = False
        res["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        try:
            res["log_tail"] = open(os.environ["LOG"], errors="replace").read()[-4000:]
        except Exception:
            pass

json.dump(res, open(os.environ["OUT"], "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1)[:2000])
PYEOF

kill $SERVE_PID 2>/dev/null
wait $SERVE_PID 2>/dev/null
echo "=== vllm_verify done $(date) ==="
