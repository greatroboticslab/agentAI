#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/framework/eval_generic_%j.out
#
# v3.0.175 (per-domain evaluator) — GENERIC evaluation job. Runs Ultralytics
# model.val() of a model on a dataset's val/test split and reports metrics.
# Env (set by the dashboard /api/eval/submit):
#   EVAL_TASK   = detect | classify | segment   (default detect)
#   EVAL_MODEL  = auto | weights path            (auto → this domain's latest
#                 trained best.pt, else the base yolo11n weight for the task)
#   EVAL_DATA   = cluster path to the staged dataset (yaml for det/seg, image
#                 folder root for classify)
#   EVAL_DOMAIN = the project/domain id (for the result file + model lookup)
#   EVAL_JOBTAG = label for outputs
# Writes results/framework/eval_results/<domain>_<jobtag>.json with metrics.
set -e
eval "$(conda shell.bash hook)"
conda activate bench
command -v python >/dev/null 2>&1 || { echo "FATAL: conda activate failed" >&2; exit 2; }
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

git fetch origin 2>&1 | tail -1
git reset --hard origin/main 2>&1 | tail -1
[ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ] && \
  cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/" 2>/dev/null

TASK="${EVAL_TASK:-detect}"
MODEL="${EVAL_MODEL:-auto}"
DATA="${EVAL_DATA:?EVAL_DATA required}"
DOMAIN="${EVAL_DOMAIN:-generic}"
JOBTAG="${EVAL_JOBTAG:-job${SLURM_JOB_ID:-0}}"

echo "=== generic eval: task=$TASK model=$MODEL data=$DATA domain=$DOMAIN ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# classify: descend to the real root that holds train/ + val/
if [ "$TASK" = "classify" ] && [ ! -d "$DATA/val" ] && [ ! -d "$DATA/train" ]; then
  found=$(find "$DATA" -maxdepth 4 -type d \( -name val -o -name train \) 2>/dev/null | head -1)
  [ -n "$found" ] && DATA="$(dirname "$found")" && echo "classify: resolved data root -> $DATA"
fi

# auto model: this domain's most-recent trained best.pt, else the base weight.
# Search broadly: Ultralytics saves classification runs under runs/classify/<project>/,
# so the best.pt can live under runs/*/ too — find it wherever it is for this domain.
if [ "$MODEL" = "auto" ] || [ -z "$MODEL" ]; then
  BEST=$(find . -path "*/${DOMAIN}_*/run*/weights/best.pt" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  if [ -n "$BEST" ]; then
    MODEL="$BEST"; echo "auto model -> latest trained best.pt: $MODEL"
  else
    case "$TASK" in
      classify) MODEL="yolo11n-cls.pt" ;;
      segment)  MODEL="yolo11n-seg.pt" ;;
      *)        MODEL="yolo11n.pt" ;;
    esac
    echo "auto model -> no trained model for '$DOMAIN'; using base weight $MODEL (baseline only)"
  fi
fi

OUTDIR="results/framework/eval_generic/${DOMAIN}_${JOBTAG}"
RESDIR="results/framework/eval_results"
mkdir -p "$RESDIR"

EVAL_TASK="$TASK" EVAL_MODEL="$MODEL" EVAL_DATA="$DATA" \
EVAL_DOMAIN="$DOMAIN" EVAL_JOBTAG="$JOBTAG" OUTDIR="$OUTDIR" RESDIR="$RESDIR" python - <<'PYEOF'
import os, json, time
res = {"domain": os.environ["EVAL_DOMAIN"], "task": os.environ["EVAL_TASK"],
       "model": os.environ["EVAL_MODEL"], "jobtag": os.environ["EVAL_JOBTAG"],
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "baseline": os.environ["EVAL_MODEL"].startswith("yolo11")}
try:
    from ultralytics import YOLO
    task = os.environ["EVAL_TASK"]
    model = YOLO(os.environ["EVAL_MODEL"])
    r = model.val(task=task, data=os.environ["EVAL_DATA"], imgsz=640,
                  project=os.environ["OUTDIR"], name="run", exist_ok=True, verbose=True)
    rd = getattr(r, "results_dict", {}) or {}
    metric = None
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)",
              "metrics/accuracy_top1", "metrics/mAP50(B)"):
        if k in rd:
            metric = round(float(rd[k]), 4); res["metric_key"] = k; break
    res["metric"] = metric
    res["metrics"] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                      for k, v in rd.items()}
    res["ok"] = True
    print("EVAL metric:", metric)
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {e}"
    print("EVAL FAILED:", res["error"])
out = os.path.join(os.environ["RESDIR"], f"{os.environ['EVAL_DOMAIN']}_{os.environ['EVAL_JOBTAG']}.json")
json.dump(res, open(out, "w"), indent=2, default=str)
print("wrote", out)
PYEOF
echo "=== ALL DONE $(date) ==="
