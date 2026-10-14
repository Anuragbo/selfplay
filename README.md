# Does Self-Play on Text Games Teach Reasoning at Small Scale?

CMPE 260 (Reinforcement Learning), San José State University.

**Team:** Anurag Bodapally (019094218) · Shubham Baid (018221333) · Syed Hassan Muzammil (018768243)

SPIRAL (Liu et al., ICLR 2026) reports that a language model trained only to win zero-sum text games
improves on math benchmarks it never trained on. Published tests of this effect use models of 4B parameters
or larger. We test whether it persists at 1.7B, using Qwen3-4B-Base (SPIRAL's model) as a positive control.

## Status

Check-in 1: data sources identified and explored. The report is in `docs/`; every number in it is produced
by the scripts in `explore/` and stored in `results/`.

| Data source | What it is | Used for |
|---|---|---|
| SPIRAL `KuhnPoker-v1` | Game environment (5 hands per match) | Training signal |
| Qwen3-1.7B-Base, Qwen3-4B-Base | Base models (Hugging Face) | Policies we train |
| `Spiral-RL/Spiral-Qwen3-4B` | SPIRAL's released checkpoint | Validating the evaluation code |
| MATH-500, OlympiadBench, Minerva Math, AIME 2024, AIME 2025, AMC 2023 | SPIRAL's math evaluation suite | Measuring transfer |

## Repository layout

```
explore/       data exploration scripts (CPU only)
tests/         unit tests for the exploration code
results/       JSON outputs of the exploration scripts
figures/       figures generated from results/
data/          fetch_benchmarks.sh; downloaded data goes to data/raw/ (not committed)
third_party/   SPIRAL's Kuhn Poker environment (Apache-2.0, unmodified)
docs/          proposal and check-in reports
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

python explore/make_figures.py results figures
```

Each Kuhn Poker match-up plays 10,000 matches (`--games` to change), alternating seats, with a fixed seed
per match, so results are reproducible exactly.

## Tests and linting

```bash
pytest
ruff check .
```

## References

1. B. Liu et al., "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning," ICLR 2026. arXiv:2506.24119
2. L. Guertler et al., "TextArena," arXiv:2504.11442, 2025
3. S. Wang et al., "Tina: Tiny Reasoning Models via LoRA," arXiv:2504.15777, 2025
