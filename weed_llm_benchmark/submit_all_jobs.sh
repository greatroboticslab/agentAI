#!/bin/bash
# ==============================================================
# Submit all benchmark jobs to SLURM
# ==============================================================
# Usage:
#   bash submit_all_jobs.sh              # Submit all jobs
#   bash submit_all_jobs.sh --hf-only    # Only HuggingFace models
#   bash submit_all_jobs.sh --ollama-only # Only Ollama models
#   bash submit_all_jobs.sh --dry-run    # Print commands without submitting
#   bash submit_all_jobs.sh --dataset cottonweeddet12  # Single dataset
# ==============================================================

set -e

# Parse arguments
DRY_RUN=false
HF_ONLY=false
OLLAMA_ONLY=false
FILTER_DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --hf-only)   HF_ONLY=true; shift ;;
        --ollama-only) OLLAMA_ONLY=true; shift ;;
        --dataset)   FILTER_DATASET="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Experiment matrix
DATASETS=("cottonweeddet12" "weed2okok" "deepweeds")
HF_MODELS=("qwen7b" "qwen3b" "minicpm" "internvl2" "florence2")
OLLAMA_MODELS=("moondream" "llava:13b" "llama3.2-vision:11b")

# Filter dataset if specified
if [ -n "$FILTER_DATASET" ]; then
    DATASETS=("$FILTER_DATASET")
fi

echo "============================================"
echo "Weed Detection Benchmark - Job Submission"
echo "============================================"
echo "Date: $(date)"
echo "Datasets: ${DATASETS[*]}"
if [ "$OLLAMA_ONLY" = false ]; then
    echo "HF Models: ${HF_MODELS[*]}"
fi
if [ "$HF_ONLY" = false ]; then
    echo "Ollama Models: ${OLLAMA_MODELS[*]}"
fi
if [ "$DRY_RUN" = true ]; then
    echo "MODE: DRY RUN (no jobs will be submitted)"
fi
echo "============================================"
echo ""

JOB_COUNT=0

# Submit HuggingFace model jobs
if [ "$OLLAMA_ONLY" = false ]; then
    echo "--- HuggingFace Models ---"
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${HF_MODELS[@]}"; do
            CMD="sbatch --job-name=wb_${MODEL}_${DATASET} run_benchmark_hf.sh $MODEL $DATASET"
            if [ "$DRY_RUN" = true ]; then
                echo "  [dry-run] $CMD"
            else
                echo -n "  Submitting ${MODEL} x ${DATASET}... "
                JOB_ID=$($CMD | awk '{print $NF}')
                echo "Job $JOB_ID"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
    echo ""
fi

# Submit Ollama model jobs
if [ "$HF_ONLY" = false ]; then
    echo "--- Ollama Models ---"
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${OLLAMA_MODELS[@]}"; do
            MODEL_CLEAN=$(echo "$MODEL" | tr ':/' '-')
            CMD="sbatch --job-name=wb_${MODEL_CLEAN}_${DATASET} run_benchmark_ollama.sh $MODEL $DATASET"
            if [ "$DRY_RUN" = true ]; then
                echo "  [dry-run] $CMD"
            else
                echo -n "  Submitting ${MODEL} x ${DATASET}... "
                JOB_ID=$($CMD | awk '{print $NF}')
                echo "Job $JOB_ID"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
    echo ""
fi

# Also submit YOLO baselines
echo "--- YOLO Baselines ---"
for DATASET in "${DATASETS[@]}"; do
    # Zero-shot
    CMD="sbatch --job-name=wb_yolo_zs_${DATASET} run_yolo_baseline.sh $DATASET zero-shot"
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] $CMD"
    else
        echo -n "  Submitting YOLO zero-shot x ${DATASET}... "
        JOB_ID=$($CMD | awk '{print $NF}')
        echo "Job $JOB_ID"
    fi
    JOB_COUNT=$((JOB_COUNT + 1))

    # Fine-tune (50 epochs)
    CMD="sbatch --job-name=wb_yolo_ft_${DATASET} run_yolo_baseline.sh $DATASET fine-tune 50"
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] $CMD"
    else
        echo -n "  Submitting YOLO fine-tune x ${DATASET}... "
        JOB_ID=$($CMD | awk '{print $NF}')
        echo "Job $JOB_ID"
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "============================================"
echo "Total jobs submitted: $JOB_COUNT"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
