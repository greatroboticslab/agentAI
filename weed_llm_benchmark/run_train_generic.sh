#!/bin/bash
#SBATCH --job-name=train_gen
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/framework/train_generic_%j.out
#
# v3.0.140 (roadmap #3) — GENERIC training job. One whitelisted template that
# trains any task with Ultralytics (already installed): detection / classification
# / segmentation. Driven entirely by env vars set by the dashboard:
#   TRAIN_TASK   = detect | classify | segment      (default detect)
#   TRAIN_MODEL  = auto | a model id / weights        (default auto per task)
#   TRAIN_EPOCHS = int                                (default 20)
#   TRAIN_DATA   = cluster path to the staged dataset (yaml for det/seg, image
#                  folder root for classify)
#   TRAIN_DOMAIN = the agent/domain id (for the result file)
#   TRAIN_JOBTAG = label for outputs
# Writes results/framework/train_results/<domain>_<jobtag>.json with the metric.
# Cluster is on-demand: this runs only while the job is queued+running, then exits.
set -e
eval "$(conda shell.bash hook)"
conda activate bench
command -v python >/dev/null 2>&1 || { echo "FATAL: conda activate failed" >&2; exit 2; }
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

# sync latest code (mirror nested->outer; results/ is gitignored so data is safe)
git fetch origin 2>&1 | tail -1
git reset --hard origin/main 2>&1 | tail -1
[ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ] && \
  cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/" 2>/dev/null

TASK="${TRAIN_TASK:-detect}"
MODEL="${TRAIN_MODEL:-auto}"
EPOCHS="${TRAIN_EPOCHS:-20}"
DATA="${TRAIN_DATA:?TRAIN_DATA required}"
DOMAIN="${TRAIN_DOMAIN:-generic}"
JOBTAG="${TRAIN_JOBTAG:-job${SLURM_JOB_ID:-0}}"

echo "=== generic train: task=$TASK model=$MODEL epochs=$EPOCHS data=$DATA domain=$DOMAIN ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# v3.0.163: robust classification data root. Ultralytics classify needs the dir
# that DIRECTLY contains train/ (+ val/). Uploaded zips that carry a wrapper
# folder (e.g. images/train/...) get extracted one level too deep, so DATA=.../images
# ends up holding another dir instead of train/. Descend to find the real root.
if [ "$TASK" = "classify" ] && [ ! -d "$DATA/train" ]; then
  found=$(find "$DATA" -maxdepth 4 -type d -name train 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    DATA="$(dirname "$found")"
    echo "classify: resolved data root -> $DATA"
  fi
fi

# default model per task if 'auto'
if [ "$MODEL" = "auto" ] || [ -z "$MODEL" ]; then
  case "$TASK" in
    classify) MODEL="yolo11n-cls.pt" ;;
    segment)  MODEL="yolo11n-seg.pt" ;;
    *)        MODEL="yolo11n.pt" ;;
  esac
fi

OUTDIR="results/framework/train_generic/${DOMAIN}_${JOBTAG}"
RESDIR="results/framework/train_results"
mkdir -p "$RESDIR"

TRAIN_TASK="$TASK" TRAIN_MODEL="$MODEL" TRAIN_EPOCHS="$EPOCHS" TRAIN_DATA="$DATA" \
TRAIN_DOMAIN="$DOMAIN" TRAIN_JOBTAG="$JOBTAG" OUTDIR="$OUTDIR" RESDIR="$RESDIR" python - <<'PYEOF'
import os, json, time, glob
res = {"domain": os.environ["TRAIN_DOMAIN"], "task": os.environ["TRAIN_TASK"],
       "model": os.environ["TRAIN_MODEL"], "epochs": int(os.environ["TRAIN_EPOCHS"]),
       "jobtag": os.environ["TRAIN_JOBTAG"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
try:
    from ultralytics import YOLO
    task = os.environ["TRAIN_TASK"]
    model = YOLO(os.environ["TRAIN_MODEL"])
    r = model.train(task=task, data=os.environ["TRAIN_DATA"],
                    epochs=int(os.environ["TRAIN_EPOCHS"]), imgsz=640,
                    project=os.environ["OUTDIR"], name="run", exist_ok=True, verbose=True)
    # pull the headline metric per task from results_dict
    rd = getattr(r, "results_dict", {}) or {}
    metric = None
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)",
              "metrics/accuracy_top1", "metrics/mAP50(B)"):
        if k in rd:
            metric = round(float(rd[k]), 4); res["metric_key"] = k; break
    res["metric"] = metric
    res["metrics"] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                      for k, v in rd.items()}
    # Ultralytics may nest classification runs under runs/<task>/<OUTDIR>/ — search
    # recursively for this run's best.pt wherever it landed.
    _tag = os.path.basename(os.environ["OUTDIR"])
    best = sorted(glob.glob(f"**/{_tag}/run*/weights/best.pt", recursive=True)
                  + glob.glob(os.environ["OUTDIR"] + "/run*/weights/best.pt"),
                  key=lambda p: os.path.getmtime(p))
    res["best_pt"] = best[-1] if best else None
    res["ok"] = True
    print("RESULT metric:", metric)
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {e}"
    print("TRAIN FAILED:", res["error"])
out = os.path.join(os.environ["RESDIR"], f"{os.environ['TRAIN_DOMAIN']}_{os.environ['TRAIN_JOBTAG']}.json")
json.dump(res, open(out, "w"), indent=2, default=str)
print("wrote", out)
PYEOF
echo "=== ALL DONE $(date) ==="
