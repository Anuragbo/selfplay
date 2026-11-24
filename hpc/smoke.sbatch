#!/bin/bash
# Quick end-to-end check on one GPU: tests, one self-play training step, a few MATH-500 problems through vLLM with the
# resulting adapter. Writes only under smoke/ (deleted first).
#   sbatch hpc/smoke.sbatch
#SBATCH --job-name=sp-smoke
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L
rm -rf smoke
python -m pytest -q -p no:cacheprovider 2>&1 | tail -5
python -u -m selfplay.train_single --config configs/selfplay.json --run-id selfplay_qwen3-1.7b_shared_s0 --steps 1 \
  --checkpoints smoke/checkpoints
python -u -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --lora smoke/checkpoints/selfplay_qwen3-1.7b_shared_s0/step_1 \
  --benchmarks math500 --limit 8 --temperature 0 --max-tokens 4096 --run-id smoke --out-dir smoke/eval
echo SMOKE_OK
