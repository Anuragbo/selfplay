#!/bin/bash
# Transcript audit: replay the base model and every checkpoint of the given runs against never_fold with the
# training sampling settings (temperature 0.6, top-p 0.95, 512-token replies), keeping every transcript.
# Training itself saved only metrics, so this is how the replies behind the reply-length collapse are inspected.
#   sbatch hpc/audit.sbatch <run_id> [<run_id> ...]
# Writes outputs/audit/<run_id>-step<N>/ (and outputs/audit/base/); analysed by explore/transcript_audit.py.
#SBATCH --job-name=sp-audit
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
MATCHES=${MATCHES:-64}
STEPS=${STEPS:-10 20 30}
MODEL=${MODEL:-Qwen/Qwen3-1.7B-Base}
cd "${SLURM_SUBMIT_DIR:-$HOME/selfplay}"
source hpc/env.sh
nvidia-smi -L

replay() {  # <out dir> <run id> <step> [--lora dir]
  local out=$1 run_id=$2 step=$3
  shift 3
  [ -f "$out/transcripts.jsonl" ] && { echo "skip $out"; return; }
  python -u -m selfplay.play_kuhn --model "$MODEL" "$@" --opponents never_fold --matches "$MATCHES" \
    --transcripts "$MATCHES" --temperature 0.6 --top-p 0.95 --max-tokens 512 --seed 12345 \
    --run-id "$run_id" --step "$step" --out-dir "$out"
}

replay outputs/audit/base base 0
for RUN_ID in "$@"; do
  for step in $STEPS; do
    dir=checkpoints/$RUN_ID/step_$step
    [ -f "$dir/complete" ] || { echo "no complete checkpoint $dir"; continue; }
    replay "outputs/audit/$RUN_ID-step$step" "$RUN_ID" "$step" --lora "$dir"
  done
done
echo AUDIT_OK
