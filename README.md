# Does Self-Play on Text Games Teach Reasoning at Small Scale?

CMPE 260 (Reinforcement Learning), San José State University.

**Team:** Anurag Bodapally (019094218) · Shubham Baid (018221333) · Syed Hassan Muzammil (018768243)

SPIRAL (Liu et al., ICLR 2026) reports that a language model trained only to win zero-sum text games
improves on math benchmarks it never trained on. Published tests of this effect use models of 4B parameters
or larger. We test whether it persists at 1.7B, using Qwen3-4B-Base (SPIRAL's model) as a positive control.

## Status

Check-in 2: the pipeline is implemented and tested (103 tests): SPIRAL's Kuhn Poker ported to TextArena 0.7.4,
scripted opponents, advantage estimators (RAE and a shared baseline), answer grading, paired statistics, a
resumable vLLM evaluation runner, a script for models playing Kuhn Poker, and the training-grid launcher.
First GPU results, on a single NVIDIA T4 (Google Colab):

- MATH-500, greedy decoding: Qwen3-1.7B-Base 59.2%; Qwen3-4B-Base 70.8% (SPIRAL reports 73.4); paired difference
  +11.6 points [+7.0, +16.2].
- Qwen3-1.7B-Base vs. "never fold" in Kuhn Poker: 0/100 wins, all lost by an invalid move.

The whole study is designed to run on one 15 GB GPU; a single-GPU fp16 training loop is the next step. Every
number in the check-in reports comes from `results/`.

| Data source | What it is | Used for |
|---|---|---|
| SPIRAL `KuhnPoker-v1` | Game environment (5 hands per match) | Training signal |
| Qwen3-1.7B-Base, Qwen3-4B-Base | Base models (Hugging Face) | Policies we train |
| `Spiral-RL/Spiral-Qwen3-4B` | SPIRAL's released checkpoint | Validating the evaluation code |
| MATH-500, OlympiadBench, Minerva Math, AIME 2024, AIME 2025, AMC 2023 | SPIRAL's math evaluation suite | Measuring transfer |

## Repository layout

```
selfplay/      the pipeline (importable package)
  envs/          SPIRAL KuhnPoker-v1 rules on TextArena 0.7.4, registered as SpiralKuhnPoker-v1
  opponents.py   scripted Kuhn Poker policies, playable from text observations
  advantages.py  RAE and shared-baseline estimators for UnstableBaselines
  grading.py     \boxed{} extraction and Math-Verify grading; benchmarks.py: benchmark loaders
  stats.py       paired bootstrap, McNemar, minimum detectable effect
  evaluate.py    benchmark evaluation (vLLM, resumable, results contract)
  play_kuhn.py   a model plays Kuhn Poker against scripted opponents
  train.py       training-grid launcher for UnstableBaselines
configs/       grid.json: the 12 self-play runs and scripted-opponent controls
notebooks/     Colab (T4) notebooks for the GPU runs
explore/       data exploration scripts (CPU only)
tests/         pytest tests (tests/data: golden transcripts from SPIRAL's environment)
results/       JSON outputs; every number in the reports comes from here
figures/       figures generated from results/
data/          fetch_benchmarks.sh; downloaded data goes to data/raw/ (not committed)
third_party/   SPIRAL's Kuhn Poker environment (Apache-2.0, unmodified)
```

## Setup

Python 3.10 or newer. SPIRAL's environment needs an older TextArena, so it runs in a separate environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m venv .venv-spiral
.venv-spiral/bin/pip install textarena==0.6.4
```

## Reproducing the exploration

```bash
bash data/fetch_benchmarks.sh
python explore/benchmark_stats.py data/raw results/benchmark_stats.json

python explore/kuhn_env_explore.py textarena results/kuhn_textarena.json
python explore/kuhn_env_explore.py textarena_fixed results/kuhn_textarena_fixed.json
.venv-spiral/bin/python explore/kuhn_env_explore.py spiral results/kuhn_spiral.json \
    --spiral-env third_party/spiral_kuhn_env.py

python explore/checkin2_stats.py results results/checkin2_stats.json   # paired MATH-500 comparison
python explore/make_figures.py results figures
```

Each Kuhn Poker match-up plays 10,000 matches (`--games` to change), alternating seats, with a fixed seed
per match, so results are reproducible exactly.

## Running the pipeline

```bash
# Benchmarks and base-model game play (GPU; on Colab use notebooks/ 00 -> 08 -> 09)
pip install -r requirements-gpu.txt
python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --benchmarks math500     --run-id eval-base-qwen3-1.7b --out-dir outputs/eval-base-qwen3-1.7b
python -m selfplay.play_kuhn --model Qwen/Qwen3-1.7B-Base --opponents never_fold --matches 200     --temperature 0.6 --run-id kuhn-base-qwen3-1.7b --out-dir outputs/kuhn-base-qwen3-1.7b

# Training with the UnstableBaselines launcher (needs unstable-rl 0.2.2 and two bfloat16-capable GPUs)
python -m selfplay.train --list
python -m selfplay.train --run-id selfplay_qwen3-1.7b_rae_s0 --dry-run
python -m selfplay.train --run-id selfplay_qwen3-1.7b_rae_s0
```

Evaluation writes `results.jsonl` rows `{run_id, step, model, game, estimator, opponent, seed, benchmark, metric,
value, n_samples, timestamp}` plus per-problem correctness; training exports LoRA adapters to
`checkpoints/{run_id}/step_{N}/` with a `run_config.json`.

To re-record the golden transcripts that test the port against SPIRAL's environment:

```bash
.venv-spiral/bin/python explore/spiral_golden_transcripts.py third_party/spiral_kuhn_env.py     tests/data/spiral_kuhn_golden.json
```

## Tests and linting

```bash
pytest
ruff check .
```

## References

1. B. Liu et al., "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning," ICLR 2026. arXiv:2506.24119
2. L. Guertler et al., "TextArena," arXiv:2504.11442, 2025
3. S. Wang et al., "Tina: Tiny Reasoning Models via LoRA," arXiv:2504.15777, 2025
4. L. Guertler, UnstableBaselines (unstable-rl). https://github.com/LeonGuertler/UnstableBaselines
