#!/bin/bash
# MATH-500 (or other benchmarks) on a base model or a trained LoRA adapter, on one GPU of SJSU CoE HPC3.
#   sbatch hpc/eval.sbatch <run_id> <step> <estimator> <opponent> [benchmarks...]
#   run_id "base" evaluates Qwen3-1.7B-Base itself.
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
  EVAL_ID=eval-base-qwen3-1.7b-hpc
  LORA=()
else
  EVAL_ID=eval-${RUN_ID}-step${STEP}
  LORA=(--lora "checkpoints/$RUN_ID/step_$STEP" --game SpiralKuhnPoker-v1 --estimator "$ESTIMATOR"
        --opponent "$OPPONENT")
fi
# shellcheck disable=SC2086
exec python -u -m selfplay.evaluate --model "$MODEL" "${LORA[@]}" --benchmarks $BENCHMARKS \
  --temperature 0 --max-tokens 4096 --run-id "$EVAL_ID" --step "$STEP" --out-dir "outputs/$EVAL_ID" --print-summary
