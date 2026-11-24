#!/bin/bash
# One-time setup, run on coe-hpc1 (the only login server with internet; CentOS 7). Home is shared with coe-hpc3.
#   ssh sjsu-hpc 'bash ~/selfplay/hpc/setup.sh'
# Builds a portable Python 3.12 venv with uv (python-build-standalone runs on both CentOS 7 and Rocky 9), choosing
# wheels for HPC3's glibc 2.34, then downloads the models and benchmarks. Safe to re-run.
set -euo pipefail
SP_ROOT=${SP_ROOT:-$HOME/selfplay}
cd "$SP_ROOT"

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
export PATH=$HOME/.local/bin:$PATH

uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python --python-platform x86_64-manylinux_2_34 \
  -r requirements-gpu.txt -r requirements-dev.txt

export HF_HOME=$SP_ROOT/hf_cache HF_HUB_DISABLE_XET=1
for model in ${MODELS:-Qwen/Qwen3-1.7B-Base}; do
  .venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" \
    "$model"
done

bash data/fetch_benchmarks.sh
echo "setup done"
