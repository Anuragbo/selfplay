#!/bin/bash
# Elo games: each model plays 100 matches against each scripted opponent (never_fold, equilibrium, calling_station,
# random) with the training sampling settings (temperature 0.6, top-p 0.95, 512-token replies).
#   sbatch hpc/elo.sbatch <player> [<player> ...]
#   A player is "base" (the untrained $MODEL) or "<run_id>:<step>" (a LoRA checkpoint of a run trained from $MODEL).
#   MODEL=Qwen/Qwen3-4B-Base sbatch hpc/elo.sbatch base selfplay_qwen3-4b_rae_s0:30
# Writes outputs/elo/<player id>/ (resumable: finished players are skipped); rated by explore/elo.py.
#SBATCH --job-name=sp-elo
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}
MATCHES=${MATCHES:-100}
SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L

for player in "$@"; do
  if [ "$player" = base ]; then
    id=base-${SLUG%-base} run_id=$id step=0 lora=()
  else
    run_id=${player%:*} step=${player##*:} id=$run_id-step$step lora=(--lora "checkpoints/$run_id/step_$step")
    [ -f "checkpoints/$run_id/step_$step/complete" ] || { echo "no complete checkpoint for $player"; continue; }
  fi
  out=outputs/elo/$id
  [ -f "$out/results.jsonl" ] && { echo "skip $id"; continue; }
  python -u -m selfplay.play_kuhn --model "$MODEL" "${lora[@]}" \
    --opponents never_fold equilibrium calling_station random --matches "$MATCHES" --transcripts 0 \
    --temperature 0.6 --top-p 0.95 --max-tokens 512 --seed 54321 --run-id "$run_id" --step "$step" --out-dir "$out"
done
echo ELO_OK
