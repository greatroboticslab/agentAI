#!/bin/bash
#SBATCH --job-name=s3_mamba
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_mamba_s%a_%j.out
#
# S3 method family #3 — Mamba-YOLO-T on the sealed cwd12 protocol.
#
# The fork's own mbyolo_train.py is NOT used. Two reasons, both measured:
#   1. it does `ROOT + opt.data` on every path argument, so an absolute --data
#      becomes /Mamba-YOLO//ocean/... (this killed job 44323304 in 113 s);
#   2. its `task_type = {"train": YOLO(c).train(...), "val": ..., "test": ...}`
#      is a dict literal, so every branch is evaluated eagerly — and YOLO has no
#      .test(), so the process dies after training regardless of the task asked
#      for. It also exposes no --seed.
# Calling the fork's ultralytics API directly fixes all three and lets this
# family use the same seeded protocol as the others.
#
# Submit: sbatch --array=1-1 run_s3_mamba_t.sh   (add 2-3 once seed 101 is healthy)
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
MY=/ocean/projects/cis240145p/byler/harry/Mamba-YOLO
SEEDS=(0 101 102 103); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-1}]}"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate mambayolo
echo "seed=$SEED python=$(which python)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd "$MY" || exit 1                      # the fork's ultralytics must shadow any other
python -u - <<PY
import csv, json, os, time
REPO="$REPO"; SEED=int("$SEED")
import torch, selective_scan_cuda_oflex          # torch first: the ext needs libc10
import ultralytics
from ultralytics import YOLO
print("[s3] fork ultralytics:", ultralytics.__file__, ultralytics.__version__)
cfg="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml"
m=YOLO(cfg)
m.train(data=REPO+"/cwd12_sealed.yaml", epochs=100, imgsz=640, batch=16,
        seed=SEED, deterministic=True, patience=30, workers=4, device=0,
        optimizer="SGD", amp=True, cos_lr=True, verbose=False,
        project=REPO+"/results/framework/s3_mamba", name="t_s%d"%SEED)
d=REPO+"/results/framework/s3_mamba/t_s%d"%SEED
cands=[p for p in [d]+[d+str(i) for i in range(2,6)] if os.path.exists(os.path.join(p,"results.csv"))]
rows=list(csv.DictReader(open(os.path.join(cands[-1],"results.csv")))) if cands else []
col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
out={"family":"mamba-yolo","model":"Mamba-YOLO-T","seed":SEED,"epochs_cap":100,
     "data":"cwd12_sealed.yaml (train 3,671 / holdout-val 1,977)",
     "epochs_ran":len(rows),"best_map50_95":round(max(vals),4) if vals else None,
     "ok":bool(vals),"results_csv":cands[-1] if cands else None,
     "job":os.environ.get("SLURM_JOB_ID"),"ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out, open(REPO+"/results/framework/s3_mamba_t_seed%d.json"%SEED,"w"), indent=1)
print("[s3]", out)
PY
