#!/bin/bash
#SBATCH --job-name=weed_roboflow
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_roboflow_%j.out
#SBATCH --error=slurm_roboflow_%j.err

# ==============================================================
# Full Pipeline: Roboflow download -> LLM detect -> Upload back
# ==============================================================
# Usage:
#   sbatch run_roboflow_bridge.sh PROJECT_ID VERSION MODEL_KEY
#   sbatch run_roboflow_bridge.sh weed2okok 1 qwen7b
#   sbatch run_roboflow_bridge.sh weed2okok 1 qwen3b
#   sbatch run_roboflow_bridge.sh weed2okok 1 minicpm
#   sbatch run_roboflow_bridge.sh weed2okok 1 internvl2
#
# Models: qwen7b, qwen3b, minicpm, internvl2
# Upload project auto-named: PROJECT-MODELNAME (e.g. weed2okok-qwen25-vl-7b)
# ==============================================================

set -e

PROJECT="${1:?Usage: sbatch run_roboflow_bridge.sh PROJECT_ID [VERSION] [MODEL_KEY]}"
VERSION="${2:-1}"
MODEL_KEY="${3:-qwen7b}"

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark

echo "============================================"
echo "Roboflow <-> LLM Weed Detection Pipeline"
echo "============================================"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "Project: $PROJECT (v$VERSION)"
echo "Model:   $MODEL_KEY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Properly initialize conda then activate
eval "$(conda shell.bash hook)"
conda activate qwen
export HF_HOME="/ocean/projects/cis240145p/byler/hf_cache"
export TRANSFORMERS_CACHE="/ocean/projects/cis240145p/byler/hf_cache/hub"

# Verify critical packages
python -c "import accelerate; print(f'accelerate {accelerate.__version__}')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run full pipeline (auto-names upload project with model name)
python roboflow_bridge.py --all \
    --project "$PROJECT" \
    --version "$VERSION" \
    --model-key "$MODEL_KEY"

echo ""
echo "[+] Pipeline complete at $(date)"
