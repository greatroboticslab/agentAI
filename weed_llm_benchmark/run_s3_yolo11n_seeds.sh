#!/bin/bash
#SBATCH --job-name=s3_y11n
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_y11n_s%a_%j.out
#
# S3 method family #2 — YOLO11n on the sealed cwd12 protocol, 3 seeds.
# The historical 0.865 baseline was a single run on the pre-split full set; this
# gives the family a sealed mean±std. Submit: sbatch --array=1-3 run_s3_yolo11n_seeds.sh
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
m=YOLO(REPO+"/yolo11n.pt")
m.train(data=REPO+"/cwd12_sealed.yaml", epochs=100, imgsz=640, batch=-1,
        seed=SEED, deterministic=True, patience=30, workers=4, device=0,
        project=REPO+"/results/framework/s3_yolo11n", name="s%d"%SEED,
        verbose=False, cos_lr=True)
d=REPO+"/results/framework/s3_yolo11n/s%d"%SEED
csvp=[os.path.join(r,"results.csv") for r in sorted(
      [d]+[d+str(i) for i in range(2,6)], key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
      if os.path.exists(os.path.join(r,"results.csv"))]
rows=list(csv.DictReader(open(csvp[-1]))) if csvp else []
col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
out={"family":"yolo11","model":"YOLO11n","seed":SEED,"epochs_cap":100,
     "data":"cwd12_sealed.yaml","epochs_ran":len(rows),
     "best_map50_95": round(max(vals),4) if vals else None,
     "ok":bool(vals),"job":os.environ.get("SLURM_JOB_ID"),
     "ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out, open(REPO+"/results/framework/s3_yolo11n_seed%d.json"%SEED,"w"), indent=1)
print("[s3]", out)
PY
