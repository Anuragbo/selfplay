#!/bin/bash
# Train one run of a config with selfplay.train_single on one GPU of SJSU CoE HPC3.
#   sbatch hpc/train.sbatch configs/selfplay.json selfplay_qwen3-1.7b_shared_s0
# Resubmitting the same command resumes from the newest checkpoint. Slurm sends SIGTERM 10 minutes before the time
# limit; train_single then finishes the current step, saves it and exits.
#SBATCH --job-name=sp-train
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
CONFIG=$1
RUN_ID=$2
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L
exec python -u -m selfplay.train_single --config "$CONFIG" --run-id "$RUN_ID" --checkpoints checkpoints
