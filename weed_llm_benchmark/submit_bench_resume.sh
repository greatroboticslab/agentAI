#!/bin/bash
# Finish the two supervision-benchmark arms that a 4 h walltime cut short.
#
#   L2 glm-4.7-flash   78 of 149 committed  -> 71 model calls left
#   L3 qwen3.8-27b     92 of 149 committed  -> 57 model calls left
#
# Both re-runs used to need a full 12 h job because bench re-asked every case.
# BENCH_RESUME=1 reads the committed verdicts back -- only where the case, arm,
# repeat, model, bundle hash and rubric hash all match -- and calls the model for
# the remainder, so these fit in walltimes the backfill scheduler can place. The
# GPU-shared queue was 2,900 jobs deep when these were sized; a 12 h request sat
# with no start estimate while a 4 h one was given one.
#
# Sizing, from the jobs that timed out:
#   glm-4.7-flash  27.6 tok/s, 13 min ollama load, ~2.9 min/case -> 71 cases ~3.7 h
#   qwen3.8-27b    L2+L3 reached 241 cases in 4 h                -> 57 cases ~1.1 h
#
# Run from the repo root on a Bridges-2 login node.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

COMMON="--partition=GPU-shared --gres=gpu:h100-80:1 --cpus-per-task=12 --mem=80G --export=ALL"

export VERIFY_BACKEND=ollama VERIFY_BENCH=1 BENCH_RESUME=1 BENCH_REPEATS=1 BENCH_SPLIT=dev

export OLLAMA_MODEL=glm-4.7-flash BENCH_ARMS=A0,A0p,L2 VLLM_JOBTAG=brain_bglm3
sbatch $COMMON --job-name=brain_bglm3 --time=06:00:00 run_model_verify.sh

export OLLAMA_MODEL=qwen3.8:27b BENCH_ARMS=L2,L3 VLLM_JOBTAG=brain_b27c
sbatch $COMMON --job-name=brain_b27c --time=05:00:00 run_model_verify.sh
