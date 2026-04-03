#!/bin/bash
#SBATCH --job-name=clone_train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/clone_and_train/slurm_%j.out

eval "$(conda shell.bash hook)"
conda activate bench

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

echo "=== Clone and Train — External Weed Detection Models ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python run_clone_and_train.py

echo "=== Done ==="
echo "Date: $(date)"
