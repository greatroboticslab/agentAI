#!/bin/bash
#SBATCH --job-name=s2_dino
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/framework/s2_dino_scores_%j.out
#
# S2 prerequisite — DINOv2 per-slug quality scoring.
#
# `mega_trainer._merge_datasets(min_dino_score=...)` reads
# results/framework/dinov2_curator/slug_scores.json. That file was absent, so any
# "curated" run would have silently trained on the full raw pool under a curated
# label. This job produces the scores, so the curated tier of M1 (and the S2/S3
# quality gates) rest on measured values rather than a disabled gate.
#
# Submit:  sbatch run_s2_dino_scores.sh
# After it completes, read the printed distribution and pick MIN_DINO_SCORE, then:
#   sbatch --array=1-3 --job-name=m1cur --export=ALL,TIER=curated,MIN_DINO_SCORE=<x> \
#          run_m1_merged_seeds.sh
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rsync -a --delete \
        "$REPO/weed_llm_benchmark/weed_optimizer_framework/" \
        "$REPO/weed_optimizer_framework/" \
        && echo "[sync] outer package refreshed from the tracked nested copy"
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo "=== 1/3 build trusted reference pool ==="
python -u -m weed_optimizer_framework.tools.dinov2_curator build-reference 2>&1 | tail -20

echo "=== 2/3 score every registry slug against the pool ==="
python -u -m weed_optimizer_framework.tools.dinov2_curator score-all 2>&1 | tail -30

echo "=== 3/3 distribution of the resulting scores ==="
python -u - <<'PYEOF'
import json, statistics
from pathlib import Path
p = Path("results/framework/dinov2_curator/slug_scores.json")
if not p.is_file():
    raise SystemExit("FAILED: %s was not produced — the curated tier stays blocked" % p)
raw = json.loads(p.read_text()) or {}
vals = sorted(float(r["score"]) for r in raw.values()
              if isinstance(r, dict) and r.get("score") is not None)
print("scored slugs: %d" % len(vals))
if vals:
    q = lambda f: vals[min(len(vals) - 1, int(len(vals) * f))]
    print("min=%.3f p10=%.3f p25=%.3f median=%.3f p75=%.3f p90=%.3f max=%.3f"
          % (vals[0], q(.10), q(.25), statistics.median(vals), q(.75), q(.90), vals[-1]))
    print("kept-at-threshold:")
    for t in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
        n = sum(1 for v in vals if v >= t)
        print("  >= %.2f -> %3d / %3d slugs (%.0f%%)" % (t, n, len(vals), 100.0 * n / len(vals)))
    print("\nPick MIN_DINO_SCORE so the curated tier keeps a clear minority of the "
          "pool (the point of the tier is contrast with raw, not a near-copy).")
PYEOF

echo "=== S2 dino scoring finished rc=$? $(date) ==="
