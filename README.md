# Does Self-Play on Text Games Teach Reasoning at Small Scale?

CMPE 260 (Reinforcement Learning), San José State University.

**Team:** Anurag Bodapally (019094218) · Shubham Baid (018221333) · Syed Hassan Muzammil (018768243)

SPIRAL (Liu et al., ICLR 2026) reports that a language model trained only to win zero-sum text games
improves on math benchmarks it never trained on. Published tests of this effect use models of 4B parameters
or larger. We test whether it persists at 1.7B, using Qwen3-4B-Base (SPIRAL's model) as a positive control.

## Status

Check-in 3: the pipeline and a single-GPU fp16 REINFORCE training loop are implemented and tested (119 tests).
All GPU runs use one NVIDIA T4 (Google Colab).

Base models (Check-in 2):

- MATH-500, greedy decoding: Qwen3-1.7B-Base 59.2%; Qwen3-4B-Base 70.8% (SPIRAL reports 73.4); paired difference
  +11.6 points [+7.0, +16.2].
- Qwen3-1.7B-Base vs. "never fold" in Kuhn Poker: 0/100 wins, all lost by an invalid move.

Pilot training (Check-in 3): Qwen3-1.7B-Base, RAE, seed 0, 30 steps of 8 matches against "never fold":

- Invalid-move losses 100% → 27.5% (first vs. last 5 steps); format errors 37.7% → 0.7%; 3 wins in 240 matches.
- MATH-500 59.2% → 62.2%: +3.0 points [−1.2, +7.0], exact McNemar p = 0.18 (not significant).
- Shared-baseline run (same settings, partial: 20 of 30 steps): invalid-move losses 70.0% vs. 52.5% for RAE over
  steps 16–20; one seed each, intervals overlap.

The pilot is a scaled-down version of the design in `configs/grid.json` (see `configs/pilot.json` for the
differences). Every number in the check-in reports comes from `results/`.

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
  generation.py  shared vLLM / Hugging Face generation, with optional LoRA adapters
  train_single.py  single-GPU fp16 REINFORCE + LoRA training against a scripted opponent, resumable
  train.py       training-grid launcher for UnstableBaselines
configs/       grid.json: the full design (12 self-play runs, 6 controls); pilot.json: the single-GPU pilot
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

python explore/checkin2_stats.py results results/checkin2_stats.json   # base models, paired MATH-500
python explore/checkin3_stats.py results results/checkin3_stats.json   # pilot training and transfer
python explore/make_figures.py results figures
```

Each Kuhn Poker match-up plays 10,000 matches (`--games` to change), alternating seats, with a fixed seed
per match, so results are reproducible exactly.

## Running the pipeline

```bash
# Benchmarks and base-model game play (GPU; on Colab use notebooks/ 00 -> 08 -> 09)
pip install -r requirements-gpu.txt
python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --benchmarks math500 \
    --run-id eval-base-qwen3-1.7b --out-dir outputs/eval-base-qwen3-1.7b
python -m selfplay.play_kuhn --model Qwen/Qwen3-1.7B-Base --opponents never_fold --matches 200 \
    --temperature 0.6 --run-id kuhn-base-qwen3-1.7b --out-dir outputs/kuhn-base-qwen3-1.7b

# Pilot training on one GPU (resumes from the newest checkpoint), then MATH-500 on the trained adapter
python -m selfplay.train_single --config configs/pilot.json --list
python -m selfplay.train_single --config configs/pilot.json --run-id pilot_qwen3-1.7b_rae_s0 --steps 30
python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --lora checkpoints/pilot_qwen3-1.7b_rae_s0/step_30 \
    --benchmarks math500 --run-id eval-pilot_qwen3-1.7b_rae_s0-step30 --step 30 --game SpiralKuhnPoker-v1 \
    --estimator rae --opponent never_fold --out-dir outputs/eval-pilot_qwen3-1.7b_rae_s0-step30

# Full grid with the UnstableBaselines launcher (needs unstable-rl 0.2.2 and two bfloat16-capable GPUs)
python -m selfplay.train --list
python -m selfplay.train --run-id selfplay_qwen3-1.7b_rae_s0 --dry-run
python -m selfplay.train --run-id selfplay_qwen3-1.7b_rae_s0
```

Evaluation writes `results.jsonl` rows `{run_id, step, model, game, estimator, opponent, seed, benchmark, metric,
value, n_samples, timestamp}` plus per-problem correctness; training exports LoRA adapters to
`checkpoints/{run_id}/step_{N}/` with a `run_config.json`, and `train_single` also writes one row per step to
`checkpoints/{run_id}/metrics.jsonl`.

To re-record the golden transcripts that test the port against SPIRAL's environment:

```bash
.venv-spiral/bin/python explore/spiral_golden_transcripts.py third_party/spiral_kuhn_env.py \
    tests/data/spiral_kuhn_golden.json
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
