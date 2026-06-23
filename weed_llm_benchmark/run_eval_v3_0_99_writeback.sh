#!/bin/bash
#SBATCH --job-name=v3099_wb
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/framework/v3_0_99_writeback_%j.out
#
# v3.0.124 — salvage + complete the train->round write-back verification.
# Job 41612776 (clean_train, 15 epochs) reached epoch 15/15 but hit its 12h
# TIMEOUT before the in-job eval + round write-back could run. best.pt WAS
# saved, so this short job runs the SAME production eval (eval_v3_0_23.py on the
# cwd12 holdout) on that checkpoint, then stamps the round via rounds
# record-train — closing the loop end-to-end on the real model.
set -e
eval "$(conda shell.bash hook)"
conda activate bench
command -v python >/dev/null 2>&1 || { echo "FATAL: conda activate failed" >&2; exit 2; }
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

# sync latest code (record_train_result) + nested -> outer mirror (results/ is
# gitignored, so best.pt is untouched by reset --hard).
echo "[sync] git fetch + reset --hard origin/main ..."
git fetch origin 2>&1 | tail -1
git reset --hard origin/main 2>&1 | tail -1
echo "[sync] HEAD now: $(git log --oneline -1 2>/dev/null)"
if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    cp -ar "$REPO/weed_llm_benchmark/weed_optimizer_framework" "$REPO/" 2>/dev/null \
      && echo "[sync] nested -> outer ok"
fi

BEST="$REPO/results/framework/mega_iterv3_0_99_clean/train2/weights/best.pt"
echo "[eval] best.pt = $BEST"
if [ ! -f "$BEST" ]; then
    echo "FATAL: best.pt not found — nothing to eval" >&2
    exit 3
fi

echo "=== eval on cwd12 gold holdout (test+valid) ==="
BEST_PT="$BEST" python eval_v3_0_23.py 2>&1 | tail -40

EVAL_JSON="results/v3_0_23_eval/v3_0_23_eval.json"
if [ -f "$EVAL_JSON" ]; then
    echo "=== record train result into round meta ==="
    python -m weed_optimizer_framework.tools.rounds record-train \
        --eval "$EVAL_JSON" --model-label "yolo-clean-15ep" \
      && echo "[round] train result recorded into registry (round meta)" \
      || echo "[round] record-train FAILED"
else
    echo "[round] no eval JSON at $EVAL_JSON — cannot record"
fi
echo "=== ALL DONE $(date) ==="
