# Sourced by the sbatch scripts. Compute nodes have no internet: everything comes from the shared home, where
# hpc/setup.sh (run on coe-hpc1, which does have internet) put the environment, the model and the benchmarks.
export SP_ROOT=${SP_ROOT:-$HOME/selfplay}
export HF_HOME=$SP_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTHONPATH=$SP_ROOT${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$SP_ROOT/logs"
# shellcheck disable=SC1091
source "$SP_ROOT/.venv/bin/activate"
