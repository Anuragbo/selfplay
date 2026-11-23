# Progress: Does Self-Play on Text Games Teach Reasoning at Small Scale?

## Introduction

Reinforcement learning improves reasoning in language models, but it usually needs curated problem sets with
verifiable answers. SPIRAL [1] showed that a model trained only to win zero-sum text games, such as Kuhn Poker,
also improves on math benchmarks it never saw in training. That transfer has only been tested at 4B parameters and
above.

We ask whether it holds at small scale, on one consumer GPU:

1. Can a 1.7B model learn the game at all?
2. Is training stable under self-play?
3. Does any game improvement carry over to math, measured with paired tests against the untrained model?

We also test SPIRAL's key ingredient, role-conditioned advantage estimation (RAE), against a simpler shared baseline.

## Background

- **SPIRAL [1]:** multi-turn, multi-agent self-play with REINFORCE [6] on zero-sum games from TextArena [2].
  Training Qwen3-4B-Base [3] on Kuhn Poker alone raises MATH-500 from 73.4% to 76.4%. Its training library is
  UnstableBaselines [5], which targets multiple GPUs.
- **Kuhn Poker [8]:** a three-card poker game (J, Q, K) with one betting round per hand. A match is 5 hands. Each
  turn the model reads the game text and must reply with its reasoning and an action in `\boxed{}`. The first
  invalid move loses the match.
- **RAE vs. shared baseline:** both subtract a moving-average baseline b from the match reward R,
  b ← (1 − τ)·b + τ·R, with τ = 0.001. RAE keeps one b per player seat, so an advantage that comes from the seat
  itself is not credited to the policy. The shared baseline keeps one b for both seats. SPIRAL reports that without
  RAE, self-play becomes unstable and reasoning length collapses.
- **Our setup:**
  - Training: Qwen3-1.7B-Base with LoRA adapters [7] (rank 32), so it
    fits one GPU, in our own single-GPU REINFORCE loop.
  - Evaluation: MATH-500 [4], graded with Math-Verify and served with vLLM.
  - Statistics: an exact McNemar test and a paired bootstrap on the same 500 problems.
  - Planned: the rest of SPIRAL's math suite (OlympiadBench, Minerva Math, AIME 2024 and 2025,
    AMC 2023).

## Setup of the runs so far

Model: Qwen3-1.7B-Base (4B is evaluated only, not trained). Game: SPIRAL's Kuhn Poker
(5 hands per match). Math: MATH-500, greedy decoding, 4,096 tokens. Training runs: one RTX 5080, LoRA rank 32,
REINFORCE, 30 steps × 8 matches, seed 0.

## Experiments

| # | Experiment | Result |
|---|---|---|
| 1 | Scripted-policy profiling (10,000 matches per pairing) and power analysis | "Never fold" beats the equilibrium strategy (59% wins). MATH-500 alone detects only ±2.5–3.5 points. |
| 2 | Base models (Qwen3-1.7B-Base, Qwen3-4B-Base) on MATH-500 | 1.7B: 59.2%. 4B: 70.8% (SPIRAL reports 73.4%, inside our 95% CI), so the evaluation reproduces SPIRAL. |
| 3 | Base model (Qwen3-1.7B-Base) vs. never_fold | 0/100 wins, all lost by an invalid move. |
| 4 | Qwen3-1.7B-Base + RAE, trained vs. never_fold | Learned legal play, never learned to win. MATH-500 −11.4 points. |
| 5 | Qwen3-1.7B-Base + shared baseline, trained vs. never_fold | Learned to win (40% in the last 5 steps). MATH-500 +4.4 points. |
| 6 | Qwen3-1.7B-Base + RAE, self-play | Learned to play itself, not never_fold. MATH-500 −4.8 points. |

## Results (all on the 5080)

### Game

| Run | Invalid-move losses, first → last 5 steps | Wins, last 5 steps | Tokens per turn, last 5 steps |
|---|---|---|---|
| Pilot RAE | 100% → 17.5% | 0% | 69 |
| Pilot shared | 95% → 7.5% | 40% | 261 |
| Self-play RAE | 85% → 7.5% (vs. itself) | 3% vs. never_fold at step 30 (66% lost by invalid move) | 81 |

### MATH-500

| Model | Accuracy | Accuracy on answered problems | vs. base (95% CI) | McNemar p |
|---|---|---|---|---|
| Base (Qwen3-1.7B-Base, untrained) | 59.8% | 66.6% | — | — |
| Pilot RAE | 48.4% | 51.8% | −11.4 (−16.0, −7.0) | 1.0×10⁻⁶ |
| Pilot shared | **64.2%** | 67.4% | **+4.4 (+0.6, +8.4)** | 0.033 |
| Self-play RAE | 55.0% | 58.9% | −4.8 (−9.0, −0.6) | 0.031 |

## Key findings

1. **The shared baseline beat RAE on both the game and math.** This is the opposite of SPIRAL's ordering. But with
   one seed per arm and almost identical advantages from the two estimators at this scale (finding 7), the gap is
   more likely run-to-run variance than an effect of the estimator.
2. **The shared baseline's math gain is mostly answer formatting.** It answers more often (no-answer rate 10.2% → 4.8%),
   but its accuracy on answered problems barely moves (66.6% → 67.4%).
3. **RAE hurts reasoning, with and without self-play.** Accuracy on answered problems falls in both RAE runs
   (to 51.8% and 58.9%), so the losses are not a formatting effect.
4. **Under RAE, game replies collapse to about 70–80 tokens**, against about 260 with the shared baseline. This is a
   candidate cause of the math drop.
5. **Self-play taught the model to play itself, not the game.** It almost stopped making invalid moves against
   itself (7.5%), but still lost 66% of matches against never_fold by an invalid move.
6. **Results vary a lot between seeds.** The same RAE configuration scored 62.2% on a T4 (Check-in 3) and 48.4% on the
   5080.
7. **30 steps is undertraining.** It is 240 matches, about 0.5% of SPIRAL-scale training (51,200). With τ = 0.001,
   each baseline moves only about 3% in 30 steps, so RAE and the shared baseline barely differ in how they act.

## Next steps

1. **Self-play shared baseline, seed 0**, then MATH-500. This completes the self-play RAE vs. shared comparison.
2. **Seeds 1 and 2** for all four arms (pilot and self-play, RAE and shared), to separate a real effect from seed noise.
3. **Longer runs (60–100 steps)**, possibly with a larger τ, so the two baselines actually differ.
4. **Transcript audit**: why RAE replies get shorter, and which never_fold situations self-play never trains on.
5. **Full math suite** (six benchmarks) with the pooled paired test, on the base model and every trained checkpoint.
6. **Move results into `results/`** and regenerate the statistics and figures.

## References

[1] B. Liu et al., "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn
Reinforcement Learning," ICLR 2026. arXiv:2506.24119.<br>
[2] L. Guertler et al., "TextArena," arXiv:2504.11442, 2025.<br>
[3] Qwen Team, "Qwen3 Technical Report," arXiv:2505.09388, 2025.<br>
[4] D. Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset," NeurIPS 2021.
arXiv:2103.03874.<br>
[5] UnstableBaselines (unstable-rl 0.2.2), github.com/LeonGuertler/UnstableBaselines.<br>
[6] R. J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,"
Machine Learning 8, 1992.<br>
[7] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685, 2021.<br>
[8] H. W. Kuhn, "A Simplified Two-Person Poker," Contributions to the Theory of Games I, 1950.
