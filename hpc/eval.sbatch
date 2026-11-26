#!/bin/bash
# MATH-500 (or other benchmarks) on a base model or a trained LoRA adapter, on one GPU of SJSU CoE HPC3.
#   sbatch hpc/eval.sbatch <run_id> <step> <estimator> <opponent> [benchmarks...]
#   run_id "base" evaluates $MODEL itself (default Qwen/Qwen3-1.7B-Base): eval id eval-base-<model>-hpc, e.g.
#   MODEL=Qwen/Qwen3-4B-Base gives eval-base-qwen3-4b-hpc. EVAL_ID=... overrides the id (e.g. SPIRAL's checkpoint).
#   For a LoRA run on another base model, set MODEL to that base model too.
#   SAMPLES=2 TEMPERATURE=0.6 TOP_P=0.95 sbatch ... draws several answers per problem (default: one greedy answer).
#   EVAL_SUFFIX=-suite sbatch ... appends to the eval id, so a second benchmark set gets its own outputs/ and results/
#   files instead of extending an existing one.
# Resumable: rerunning skips problems already in outputs/<eval id>/per_problem.jsonl.
#SBATCH --job-name=sp-eval
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
RUN_ID=$1
STEP=$2
ESTIMATOR=$3
OPPONENT=$4
shift 4
BENCHMARKS=${*:-math500}
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L

if [ "$RUN_ID" = base ]; then
  SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
  EVAL_ID=${EVAL_ID:-eval-base-${SLUG%-base}-hpc${EVAL_SUFFIX:-}}
  LORA=()
else
  EVAL_ID=eval-${RUN_ID}-step${STEP}${EVAL_SUFFIX:-}
  SEED=${RUN_ID##*_s}
  LORA=(--lora "checkpoints/$RUN_ID/step_$STEP" --seed "$SEED" --game SpiralKuhnPoker-v1 --estimator "$ESTIMATOR"
        --opponent "$OPPONENT")
fi
# shellcheck disable=SC2086
exec python -u -m selfplay.evaluate --model "$MODEL" "${LORA[@]}" --benchmarks $BENCHMARKS \
  --samples "${SAMPLES:-1}" --temperature "${TEMPERATURE:-0}" --top-p "${TOP_P:-1.0}" --max-tokens 4096 \
  --run-id "$EVAL_ID" --step "$STEP" --out-dir "outputs/$EVAL_ID" --print-summary
