#!/bin/bash
#SBATCH --job-name=owl_pre
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_50_owl_preannotate_%j.out

# v3.0.50 — OWLv2 image-conditioned pre-annotation for active learning.
# Reads per-species exemplar crops (green) and proposes red bboxes on a
# target image dir. See weed_optimizer_framework/tools/owl_preannotate.py
# and memory/project_roboflow_pipeline_plan.md.
#
# Triggered from /control via cluster_action 'owl_preannotate_one'.
# Parameterized via env vars (whitelisted action passes nothing — uses
# defaults below; to run a different species/target, sbatch this script
# manually with the env exported).

set -e
eval "$(conda shell.bash hook)" && conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

# --- Defaults (override via env) ---
OWL_SPECIES="${OWL_SPECIES:-Goosegrass}"
OWL_TARGET_DIR="${OWL_TARGET_DIR:-$REPO/downloads/cottonweeddet12/valid/images}"
OWL_EXEMPLAR_CONFIG="${OWL_EXEMPLAR_CONFIG:-$REPO/results/framework/owl_exemplars/${OWL_SPECIES}.json}"
OWL_OUT_DIR="${OWL_OUT_DIR:-$REPO/results/framework/owl_red_proposals/${OWL_SPECIES}}"
OWL_CONF="${OWL_CONF:-0.30}"
OWL_MAX="${OWL_MAX:-50}"
OWL_DRY_RUN="${OWL_DRY_RUN:-0}"

echo "=== v3.0.50 OWL pre-annotate ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "  species:    $OWL_SPECIES"
echo "  target:     $OWL_TARGET_DIR"
echo "  exemplars:  $OWL_EXEMPLAR_CONFIG"
echo "  out:        $OWL_OUT_DIR"
echo "  conf:       $OWL_CONF"
echo "  max-images: $OWL_MAX"

# Sanity: exemplar config must exist (gate against accidental runs)
if [ ! -f "$OWL_EXEMPLAR_CONFIG" ]; then
    echo "FATAL: exemplar config not found: $OWL_EXEMPLAR_CONFIG"
    echo "       Create one with the structure:"
    echo "       {\"species\":\"$OWL_SPECIES\",\"exemplars\":[{\"image\":\"/abs/path.jpg\",\"bbox_yolo\":[cx,cy,w,h]}, ...]}"
    exit 2
fi

DRY_FLAG=""
[ "$OWL_DRY_RUN" = "1" ] && DRY_FLAG="--dry-run"

python -u -m weed_optimizer_framework.tools.owl_preannotate \
    --species "$OWL_SPECIES" \
    --target-dir "$OWL_TARGET_DIR" \
    --exemplar-config "$OWL_EXEMPLAR_CONFIG" \
    --out-dir "$OWL_OUT_DIR" \
    --conf-threshold "$OWL_CONF" \
    --max-images "$OWL_MAX" \
    $DRY_FLAG 2>&1 | tee -a $REPO/results/framework/v3_0_50_owl_preannotate_${SLURM_JOB_ID}.log

echo "=== exit: $? ==="
