#!/bin/bash
# An untrained model playing SPIRAL's Kuhn Poker against never_fold, with the settings of the Check-in 2 baseline
# (results/kuhn-base-qwen3-1.7b.json): 100 matches, temperature 0.6, top-p 1.0, 4,096-token replies.
#   MODEL=Qwen/Qwen3-4B-Base sbatch hpc/kuhn_base.sbatch      -> outputs/kuhn-base-qwen3-4b-hpc/
#SBATCH --job-name=sp-kuhn
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}
SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
RUN_ID=kuhn-base-${SLUG%-base}-hpc
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L
exec python -u -m selfplay.play_kuhn --model "$MODEL" --opponents never_fold --matches 100 --transcripts 5 \
  --temperature 0.6 --top-p 1.0 --max-tokens 4096 --run-id "$RUN_ID" --out-dir "outputs/$RUN_ID" --print-summary
