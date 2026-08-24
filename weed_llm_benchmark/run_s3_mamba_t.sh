#!/bin/bash
#SBATCH --job-name=s3_mamba
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_mamba_t_%j.out
#
# S3 method family #3 — Mamba-YOLO-T on the sealed cwd12 protocol.
# First run of this family: pipeline validation on the fork's default seed (the
# training script exposes no --seed flag; recorded as seed=default in the
# artifact). Defaults in mbyolo_train.py assume 8 GPUs / batch 512 / 128 workers,
# so every resource knob is pinned explicitly for one V100.
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
MY=/ocean/projects/cis240145p/byler/harry/Mamba-YOLO
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate mambayolo
echo "python: $(which python)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd "$MY"
python -u mbyolo_train.py --task train \
  --data "$REPO/cwd12_sealed.yaml" \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --amp --batch_size 16 --epochs 100 --device 0 --workers 4 --imgsz 640 \
  --project "$REPO/results/framework/s3_mamba" --name t_cwd12_default
python -u - <<'PY'
import csv, glob, json, os, time
REPO="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
runs=sorted(glob.glob(REPO+"/results/framework/s3_mamba/t_cwd12_default*/results.csv"), key=os.path.getmtime)
out={"family":"mamba-yolo","model":"Mamba-YOLO-T","seed":"default (no --seed flag in mbyolo_train.py)",
     "data":"cwd12_sealed.yaml (train 3,671 / holdout val 1,977)","epochs_cap":100,
     "job":os.environ.get("SLURM_JOB_ID"),"ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
if runs:
    rows=list(csv.DictReader(open(runs[-1])))
    col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
    vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
    out.update(results_csv=runs[-1], epochs_ran=len(rows),
               best_map50_95=(round(max(vals),4) if vals else None),
               final_map50_95=(round(vals[-1],4) if vals else None), ok=bool(vals))
else:
    out.update(ok=False, error="no results.csv produced")
dest=REPO+"/results/framework/s3_mamba_t_cwd12.json"
json.dump(out, open(dest,"w"), indent=1)
print("[s3] wrote", dest, "->", {k:out.get(k) for k in ("ok","epochs_ran","best_map50_95")})
PY
