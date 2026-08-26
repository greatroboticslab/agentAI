#!/bin/bash
#SBATCH --job-name=s3_tta
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_tta_%j.out
#
# S3's last unexecuted recipe: the test-time-augmentation / weighted-box-fusion
# ceiling. BEST_MODEL_CARD.md limit #3 says the headline 0.8759 is a plain
# single-model forward pass and that this was never run. This runs it.
#
# The question is not "does TTA help" in general — it is how much headroom is
# left on THIS holdout for free, without retraining anything, and whether the
# deployable model can be pushed toward the RF-DETR number (0.8974) by inference
# tricks alone.
#
# Design note that decides whether the answer means anything: every arm is scored
# by the SAME matcher in wbf_tta_eval.py, including a plain no-WBF single-scale
# baseline. Comparing a WBF number from this script against the 0.8759 from
# Ultralytics' validator would confound the augmentation with the metric
# implementation, and the two do not have to agree. The Ultralytics number is
# also re-measured here, so the offset between the two metrics is reported rather
# than assumed to be zero.
#
# Arms isolate each lever instead of only running the maximal recipe:
#   plain_s102        1 model, 640, no flip, no fusion   <- matcher baseline
#   wbf_s102_640      1 model, 640, no flip, WBF          <- fusion alone
#   tta_scales        1 model, 3 scales, WBF              <- multi-scale alone
#   tta_scales_flip   1 model, 3 scales + hflip, WBF      <- full single-model TTA
#   ens3_640          3 seeds, 640, WBF                   <- ensembling alone
#   ens3_tta          3 seeds, 3 scales + hflip, WBF      <- the ceiling
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
OUT=$REPO/results/framework/s3_tta_ceiling
YAML=$REPO/cwd12_sealed.yaml
cd "$REPO"
mkdir -p "$OUT"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=$REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

W101=$REPO/results/framework/s3_yolo11n/s101/weights/best.pt
W102=$REPO/results/framework/s3_yolo11n/s102/weights/best.pt
W103=$REPO/results/framework/s3_yolo11n/s103/weights/best.pt
for w in "$W101" "$W102" "$W103"; do
  [ -f "$w" ] || { echo "[tta] MISSING $w — refusing to report a partial matrix"; exit 1; }
done

RUN="python -u -m weed_optimizer_framework.tools.wbf_tta_eval --data-yaml $YAML"

echo "=== arm 1/6: plain_s102 (matcher baseline) ==="
$RUN --weights "$W102" --imgszs 640 --no-wbf \
     --label plain_s102 --out "$OUT/plain_s102.json"

echo "=== arm 2/6: wbf_s102_640 (fusion alone) ==="
$RUN --weights "$W102" --imgszs 640 \
     --label wbf_s102_640 --out "$OUT/wbf_s102_640.json"

echo "=== arm 3/6: tta_scales (multi-scale alone) ==="
$RUN --weights "$W102" --imgszs 512 640 768 \
     --label tta_scales --out "$OUT/tta_scales.json"

echo "=== arm 4/6: tta_scales_flip (full single-model TTA) ==="
$RUN --weights "$W102" --imgszs 512 640 768 --hflip \
     --label tta_scales_flip --out "$OUT/tta_scales_flip.json"

echo "=== arm 5/6: ens3_640 (ensembling alone) ==="
$RUN --weights "$W101" "$W102" "$W103" --imgszs 640 \
     --label ens3_640 --out "$OUT/ens3_640.json"

echo "=== arm 6/6: ens3_tta (the ceiling) ==="
$RUN --weights "$W101" "$W102" "$W103" --imgszs 512 640 768 --hflip \
     --label ens3_tta --out "$OUT/ens3_tta.json"

echo "=== anchor: Ultralytics validator on the same checkpoint + yaml ==="
python -u - <<'PY'
import json, os
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

echo "=== summary ==="
python -u - <<'PY'
import glob, json, os
OUT = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_tta_ceiling"
rows = {}
for f in glob.glob(f"{OUT}/*.json"):
    d = json.load(open(f))
    lab = d.get("label") or os.path.basename(f)[:-5]
    rows[lab] = d.get("mAP50_95", d.get("map50_95"))
base = rows.get("plain_s102")
order = ["plain_s102", "wbf_s102_640", "tta_scales", "tta_scales_flip",
         "ens3_640", "ens3_tta", "ultralytics_val_s102"]
print(f"\n{'arm':24s} {'mAP50-95':>9s}  {'vs plain':>9s}")
for k in order:
    if k not in rows or rows[k] is None:
        continue
    d = "" if (base is None or k == "plain_s102") else f"{rows[k]-base:+.4f}"
    print(f"{k:24s} {rows[k]:9.4f}  {d:>9s}")
json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=2)
print("\nSeed noise on this holdout is +/-0.0030 (n=3), so an effect under")
print("about 0.006 is not distinguishable from run-to-run variation.")
PY
echo "[tta] DONE"
