#!/bin/bash
#SBATCH --job-name=s3_ttaE
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_ttaE_%j.out
#
# The ensemble half of the TTA ceiling, split out of run_s3_tta_ceiling.sh (job
# 44463762) after that job's own logs gave the real inference rate: 250 images in
# 73 s per view-pass, i.e. 0.292 s. Arm 6 fuses 3 checkpoints x 3 scales x hflip =
# 18 passes over 1,977 images, which is ~2.9 h on its own and does not fit inside
# the 3 h that job asked for. Arms 1-5 do fit and are left running there rather
# than restarted, so nothing already computed is thrown away.
#
# Running this concurrently costs a second V100-shared allocation (1 SU/GPU-hour
# against a 4,000 SU envelope with ~215 spent) and returns the answer hours
# sooner than a serial resubmit would.
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
OUT=$REPO/results/framework/s3_tta_ceiling
YAML=$REPO/cwd12_sealed.yaml
cd "$REPO"
mkdir -p "$OUT"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$REPO
nvidia-smi --query-gpu=name --format=csv,noheader

W101=$REPO/results/framework/s3_yolo11n/s101/weights/best.pt
W102=$REPO/results/framework/s3_yolo11n/s102/weights/best.pt
W103=$REPO/results/framework/s3_yolo11n/s103/weights/best.pt

echo "=== arm 6: ens3_tta (3 seeds x 3 scales x hflip = 18 views) ==="
python -u -m weed_optimizer_framework.tools.wbf_tta_eval --data-yaml "$YAML" \
  --weights "$W101" "$W102" "$W103" --imgszs 512 640 768 --hflip \
  --label ens3_tta --out "$OUT/ens3_tta.json"

echo "=== anchor: Ultralytics validator, same checkpoint + yaml ==="
# The card's 0.8759 came from this validator, not from wbf_tta_eval's matcher.
# Measuring it here turns the difference between the two into a reported number
# instead of an assumption that they agree.
python -u - <<'PY'
import json
from ultralytics import YOLO
REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
OUT = f"{REPO}/results/framework/s3_tta_ceiling"
m = YOLO(f"{REPO}/results/framework/s3_yolo11n/s102/weights/best.pt")
r = m.val(data=f"{REPO}/cwd12_sealed.yaml", imgsz=640, split="val", verbose=False)
json.dump({"label": "ultralytics_val_s102", "map50_95": float(r.box.map),
           "map50": float(r.box.map50), "imgsz": 640},
          open(f"{OUT}/ultralytics_val_s102.json", "w"), indent=2)
print("ultralytics mAP50-95:", round(float(r.box.map), 4))
PY

echo "=== summary over every arm present ==="
python -u - <<'PY'
import glob, json, os
OUT = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_tta_ceiling"
rows = {}
for f in sorted(glob.glob(f"{OUT}/*.json")):
    if os.path.basename(f) == "summary.json":
        continue
    d = json.load(open(f))
    lab = d.get("label") or os.path.basename(f)[:-5]
    rows[lab] = d.get("mAP50_95", d.get("map50_95"))
base = rows.get("plain_s102")
order = ["plain_s102", "wbf_s102_640", "tta_scales", "tta_scales_flip",
         "ens3_640", "ens3_tta", "ultralytics_val_s102"]
print(f"\n{'arm':24s} {'mAP50-95':>9s}  {'vs plain':>9s}")
for k in order:
    if rows.get(k) is None:
        print(f"{k:24s} {'(missing)':>9s}")
        continue
    d = "" if (base is None or k == "plain_s102") else f"{rows[k]-base:+.4f}"
    print(f"{k:24s} {rows[k]:9.4f}  {d:>9s}")
json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=2)
print("\nSeed noise on this holdout is +/-0.0030 (n=3): an effect under about")
print("0.006 is not distinguishable from run-to-run variation.")
PY
echo "[ttaE] DONE"
