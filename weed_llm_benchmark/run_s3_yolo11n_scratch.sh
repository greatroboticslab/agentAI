#!/bin/bash
#SBATCH --job-name=s3_y11nsc
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_y11nsc_s%a_%j.out
#
# S3 fairness control — YOLO11n trained FROM SCRATCH on the sealed cwd12 protocol.
#
# Mamba-YOLO-T reached 0.8266 +/- 0.0064 while COCO-pretrained YOLO11n reached
# 0.8755 +/- 0.0029, but that is not an architecture comparison: the fork ships no
# Mamba weights, so it trains from a YAML with random init, while YOLO11n started
# from yolo11n.pt. This run removes the asymmetry by initialising YOLO11n from
# yolo11n.yaml too. Submit: sbatch --array=1-3 run_s3_yolo11n_scratch.sh
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
SEEDS=(0 101 102 103); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-1}]}"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
echo "seed=$SEED python=$(which python)"; nvidia-smi --query-gpu=name --format=csv,noheader
python -u - <<PY
import csv, json, os, time
from ultralytics import YOLO
REPO="$REPO"; SEED=int("$SEED")
m=YOLO("yolo11n.yaml")   # v3.22.23: FROM SCRATCH — fair vs Mamba-YOLO-T, which has no released weights
m.train(data=REPO+"/cwd12_sealed.yaml", epochs=100, imgsz=640, batch=-1,
        seed=SEED, deterministic=True, patience=30, workers=4, device=0,
        project=REPO+"/results/framework/s3_yolo11n", name="scratch_s%d"%SEED,
        verbose=False, cos_lr=True)
d=REPO+"/results/framework/s3_yolo11n/s%d"%SEED
csvp=[os.path.join(r,"results.csv") for r in sorted(
      [d]+[d+str(i) for i in range(2,6)], key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
      if os.path.exists(os.path.join(r,"results.csv"))]
rows=list(csv.DictReader(open(csvp[-1]))) if csvp else []
col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
out={"family":"yolo11","model":"YOLO11n (from scratch)","seed":SEED,"epochs_cap":100,
     "data":"cwd12_sealed.yaml","epochs_ran":len(rows),
     "best_map50_95": round(max(vals),4) if vals else None,
     "ok":bool(vals),"job":os.environ.get("SLURM_JOB_ID"),
     "ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out, open(REPO+"/results/framework/s3_yolo11n_scratch_seed%d.json"%SEED,"w"), indent=1)
print("[s3]", out)
PY
