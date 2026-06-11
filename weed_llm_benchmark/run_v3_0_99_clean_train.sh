#!/bin/bash
#SBATCH --job-name=v3099_clean
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/v3_0_99_clean_train_%j.out

# v3.0.99.28 (D) — CLEAN-SUBSET training: test the "quality > quantity" hypothesis.
#
# The 175K-noisy baseline plateaus at mAP50-95≈0.67 because the bulk of its boxes
# are AI-auto-labeled (loose). mAP50-95 measures box TIGHTNESS, so loose boxes cap
# it no matter how much data we add. This run trains on a QUALITY CORE instead:
#   - include_autolabel = False  → NO 175K auto-labeled (loose) data
#   - min_dino_score = ${CLEAN_DINO_MIN:-0.45} → keep only high-similarity weed
#       data (DINOv2 trusted-pool score ≥ thr). At 0.45 this keeps cottonweed_sp8
#       (0.76, 8-species human cotton labels) + the few high-scoring rf weed sets
#       (zig-zag 0.60, leopard 0.61, karthikeya 0.64, school 0.64) and drops the
#       ~0.30 looser rf datasets + off-topic garbage (coconut/beehive ~0.12).
#   - val_dataset_root = cottonweeddet12 → HONEST clean gold val (hand-labeled
#       holdout test+valid, 1977 imgs). mAP50-95 here is paper-grade, not a slice
#       of the noisy training corpus.
#   - NEVER_TRAIN (cottonweeddet12/weedsense/francesco) + per-image holdout-stem
#       filter stay active → zero eval leakage.
#
# KNOWN CAVEAT (P1): the quality core covers ~8/12 CWD12 species; Eclipta/
# Goosegrass/Morningglory/Nutsedge have no clean train data yet → their per-class
# mAP will be ~0 and drag the 12-species mean. Report per-species + the 8-covered
# mean alongside the 12-mean so the quality signal isn't masked by coverage.
#
# This is a SMALL probe (epochs ${CLEAN_EPOCHS:-80}, patience 25) — not the 175K
# big run. Goal: does clean+tight beat 0.67 on the covered species?

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

# --- sync latest code: fetch+reset --hard origin/main + mirror nested → outer ---
# reset --hard (not pull) so local cluster changes never block the sync; the
# min_dino_score gate (v3.0.99.28) MUST be present or the experiment is wrong.
echo "[sync] git fetch + reset --hard origin/main ..."
git fetch origin 2>&1 | tail -2
git reset --hard origin/main 2>&1 | tail -2
echo "[sync] HEAD now: $(git log --oneline -1 2>/dev/null)"
NESTED_PKG="$REPO/weed_llm_benchmark/weed_optimizer_framework"
OUTER_PKG="$REPO/weed_optimizer_framework"
if [ -d "$NESTED_PKG" ]; then
    if cp -ar "$NESTED_PKG" "$REPO/" 2>/dev/null; then
        echo "[sync] nested → outer ok ($(find $OUTER_PKG -type f | wc -l) files)"
    fi
fi

echo "=== v3.0.99.28 (D) CLEAN-SUBSET train — quality>quantity probe ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CLEAN_DINO_MIN=${CLEAN_DINO_MIN:-0.45}  CLEAN_EPOCHS=${CLEAN_EPOCHS:-80}"

CLEAN_DINO_MIN="${CLEAN_DINO_MIN:-0.45}" CLEAN_EPOCHS="${CLEAN_EPOCHS:-80}" python - <<'PYEOF'
import os, json, logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
strategy = {
    "include_autolabel": False,                       # NO 175K loose auto-labels
    "min_dino_score": float(os.environ["CLEAN_DINO_MIN"]),  # quality core gate (D)
    "val_dataset_root": f"{REPO}/downloads/cottonweeddet12",  # clean gold val
    "fresh_start": True,
    "epochs": int(os.environ["CLEAN_EPOCHS"]),
    "imgsz": 1024,
    "batch_size": 8,
    "lr": 0.001,
    "patience": 25,
    "workers": 4,
}
best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_99_clean")
print("\n=== CLEAN-SUBSET TRAIN COMPLETE ===")
print("best_pt:", best_pt)
print(json.dumps(summary, indent=2, default=str))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

# Honest separate eval on cwd12 gold (test+valid) — the paper-grade mAP50-95.
if [ $EXIT_CODE -eq 0 ]; then
    BEST=$(ls -t results/framework/mega_iterv3_0_99_clean/train*/weights/best.pt 2>/dev/null | head -1)
    echo "[eval] best.pt = $BEST"
    if [ -n "$BEST" ]; then
        BEST_PT="$BEST" python eval_v3_0_23.py 2>&1 | tail -40 \
          || echo "[eval] eval_v3_0_23.py failed — inspect manually"
    fi
fi
echo "=== ALL DONE $(date) ==="
