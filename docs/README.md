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

## Check-in 5 update: seeds 1–2 on the SJSU HPC (H100)

Nine more training runs, all on one NVIDIA H100 each, with the same settings as above: self-play shared seed 0, and
seeds 1 and 2 of all four arms. Every number below is compared with the base model evaluated on the same H100
(58.8%; the T4 run gave 59.2%, and the two agree on 98% of problems, McNemar p = 0.75). The 5080 runs above stay
separate, because their files are not yet in `results/`. Sources: `results/INDEX.md` lists every file;
`results/seed_stats.json` (from `explore/seed_stats.py`) holds the pooled statistics.

### MATH-500 per run (H100)

| Arm | Seed | Accuracy | No answer | vs. base (95% CI) | McNemar p | Final tokens per move |
|---|---|---|---|---|---|---|
| Base (untrained) | — | 58.8% | 10.8% | — | — | — |
| Self-play, RAE | 1 | **64.0%** | 3.4% | **+5.2 (+1.2, +9.2)** | 0.015 | 120 |
| Self-play, RAE | 2 | 55.8% | 7.4% | −3.0 (−7.0, +1.0) | 0.17 | 28 ⚠ |
| Self-play, shared | 0 | 62.4% | 3.4% | +3.6 (−0.2, +7.4) | 0.082 | 98 |
| Self-play, shared | 1 | **11.0%** | 36.4% | **−47.8 (−52.4, −43.0)** | <10⁻⁵⁰ | 6 ⚠ |
| Self-play, shared | 2 | 55.2% | 4.8% | −3.6 (−7.6, +0.4) | 0.095 | 26 ⚠ |
| Pilot, RAE | 1 | **52.8%** | 7.2% | **−6.0 (−10.4, −1.6)** | 0.008 | 233 |
| Pilot, RAE | 2 | 57.4% | 3.6% | −1.4 (−5.6, +2.8) | 0.58 | 249 |
| Pilot, shared | 1 | 61.2% | 4.6% | +2.4 (−1.6, +6.4) | 0.28 | 207 |
| Pilot, shared | 2 | **50.6%** | 7.2% | **−8.2 (−12.4, −4.0)** | 0.0002 | 5 ⚠ |

⚠ = reply-length collapse: below 50 tokens per move at the end of training.

### Pooled over seeds

The interval resamples both seeds and problems (hierarchical bootstrap). With two or three seeds it still understates
seed-to-seed variation. A t interval over the seed means is far wider: about ±50 points for every arm.

| Arm | Seeds | Mean change | 95% CI | Seeds above base |
|---|---|---|---|---|
| Self-play, RAE | 2 | +1.1 | (−5.6, +7.8) | 1 of 2 |
| Self-play, shared | 3 | −15.9 | (−46.6, +3.6) | 1 of 3 |
| Pilot, RAE | 2 | −3.7 | (−9.0, +1.4) | 0 of 2 |
| Pilot, shared | 2 | −2.9 | (−11.0, +5.0) | 1 of 2 |
| All nine runs | 9 | −6.5 | (−18.3, +1.8) | 3 of 9 |

### Findings

1. **No arm transfers reliably to MATH-500 at this scale.** Every arm with two or more seeds has seeds on both sides
   of zero, or both below. All pooled intervals include zero, and the spread between seeds of one arm (5–11 points,
   leaving out the 11% run) is larger than any effect.
2. **Reply-length collapse is the main failure, and it is not specific to self-play or to RAE.** In 4 of 9 runs the
   game replies shrank to 5–30 tokens per move, often a bare `\boxed{[call]}` with no reasoning (Fig. 7):
   3 of the 5 shared-baseline runs (one of them against the scripted opponent) and 1 of the 4 RAE runs. The worst case
   (self-play shared s1) lost its math ability: on MATH-500 it loops on fragments such as
   "Step-By-Path-Reasoning-…" until the 4,096-token limit (46% of problems truncated).
3. **This revises Check-in 4 findings 1, 3 and 4.** With more seeds, the shared baseline no longer beats RAE, and RAE
   no longer consistently hurts: self-play RAE s1 is the largest significant gain (+5.2). Reply shortening happens
   under both estimators and is, if anything, more frequent with the shared baseline. The one-seed differences in
   Check-in 4 were seed variance.
4. **Where accuracy rises, most of the gain is formatting.** The runs above the base cut the no-answer rate from 10.8%
   to 3.4–4.6%. Accuracy on answered problems barely moves (65.9% base, 66.3% for self-play RAE s1).
5. **The collapse is what the game's reward asks for (transcript audit).** We replayed checkpoints 10, 20 and 30 of
   every run, plus the base model, for 64 matches each against never_fold with the training sampling settings
   (`results/transcript_audit.json`).
   - At step 10, every run still loses 67–100% of matches to invalid moves. These are mostly actions that are
     illegal in the current position, such as `[check]` when facing a bet: the reply reasons about an earlier hand
     in the history. By steps 20 and 30, at most 5% of replies are truncated.
   - The fully collapsed checkpoints answer with nothing but the boxed action. Self-play shared s1 at step 30 is
     exactly the never_fold policy: `\boxed{[bet]}` when it can bet and `\boxed{[call]}` when facing a bet, whatever
     its card, with no invalid moves and 44% wins. Across the three checkpoints with only bare replies, invalid moves
     are 0–34% and wins 30–50%.
   - The cause is SPIRAL's KuhnPoker-v1 rules, which we kept for fidelity (`results/kuhn_spiral.json`): bets do not
     change the stakes, and the match goes to the winner of the final hand. Against never_fold, every hand reaches
     showdown unless the model folds, so no policy can win more than about half of the matches, and any policy that
     never folds and never makes an invalid move gets there. Reasoning only adds the risk of an invalid move.
   - No trained checkpoint plays its card consistently: the gap between how often it bets or calls with a K and with
     a J ranges from −0.44 to +0.52 and changes sign between steps and seeds. The base model, when opening, bets
     with a K far more often than with a J (+0.72), but loses every match to invalid moves.
   - Early truncation adds to the first drop in length. In steps 1–5, 21–62% of training replies hit the 512-token
     cap, and a truncated reply has no `\boxed{}`, so it loses the format bonus and the match. This falls to 7–18%
     by steps 6–15 in every run.
6. **Exploratory only (the subset was chosen after seeing the results):** among the five runs that did not collapse,
   self-play scores +6.1 points (+1.1, +11.3) above the pilot runs. A run designed to prevent the collapse would be
   needed to test this.

### Self-play with a KL penalty (H100)

To prevent the collapse, we trained six more self-play runs with a KL penalty to the base model (k3 estimator,
`kl_coef` 0.05, `configs/selfplay_kl.json`), for 60 steps instead of 30: RAE and shared, seeds 0–2, all other settings
unchanged. Each run was evaluated on MATH-500 at steps 30 and 60. Sources: `results/eval-selfplay-kl_*.json`;
`results/seed_stats.json` (step 30) and `results/seed_stats_step60.json` (step 60).

| Estimator | Seed | Step 30 vs. base (95% CI) | Step 60 accuracy | Step 60 vs. base (95% CI) | McNemar p | No answer | Final tokens per move |
|---|---|---|---|---|---|---|---|
| RAE | 0 | +1.2 (−2.8, +5.0) | 59.2% | +0.4 (−3.6, +4.4) | 0.92 | 4.6% | 335 |
| RAE | 1 | +0.8 (−3.0, +4.6) | 61.6% | +2.8 (−1.2, +7.0) | 0.21 | 3.4% | 187 |
| RAE | 2 | +0.2 (−3.8, +4.2) | 60.4% | +1.6 (−2.4, +5.4) | 0.48 | 3.4% | 267 |
| Shared | 0 | +0.4 (−3.6, +4.4) | 62.6% | +3.8 (−0.2, +8.0) | 0.087 | 5.6% | 381 |
| Shared | 1 | +0.6 (−3.2, +4.4) | 59.8% | +1.0 (−3.0, +5.0) | 0.70 | 3.6% | 298 |
| Shared | 2 | +0.4 (−3.4, +4.2) | **64.4%** | **+5.6 (+1.8, +9.4)** | 0.006 | 3.4% | 234 |

Pooled over the six runs (hierarchical bootstrap; t interval over seeds in brackets):

| Checkpoint | Mean change | 95% CI | t interval | Runs above base |
|---|---|---|---|---|
| Step 30 | +0.6 | (−2.6, +3.8) | [+0.2, +1.0] | 6 of 6 |
| Step 60 | +2.5 | (−0.9, +6.1) | [+0.5, +4.6] | 6 of 6 |

7. **The KL penalty prevents the collapse.** No KL run fell below 187 tokens per move; without KL, 4 of 9 runs fell
   to 5–30 by step 30.
8. **With KL, every run is at or slightly above the base model, and the seeds agree.** The seed spread at step 30 is
   0.4 points (5–11 without KL). The step-60 gain of +2.5 is consistent across seeds (t interval excludes zero), but
   the hierarchical interval, which also counts problem sampling, includes zero. Only shared s2 at step 60 is
   significant on its own (p = 0.006), which does not survive a correction for the 12 comparisons (Bonferroni
   threshold 0.004). Estimators again do not differ: RAE +1.6 and shared +3.5 at step 60, intervals overlapping.
9. **As before, the gain is formatting, not reasoning.** The no-answer rate falls from 10.8% (base) to 6.5% at
   step 30 and 4.0% at step 60 (means over runs), while accuracy on answered problems is slightly *below* the base
   (63.6% and 63.9% vs. 65.9%). KL keeps the replies long but, as the audit predicts (finding 5), the game reward still
   does not pay for reasoning.

## Next steps

1. ~~Self-play shared baseline, seed 0~~ — done (H100, 62.4%).
2. ~~Seeds 1 and 2 for all four arms~~ — done (H100). The three 5080 seed-0 runs are still to be added to `results/`.
3. ~~Prevent the length collapse~~ — done (findings 7–9): the KL penalty keeps replies long and every run at or
   slightly above the base, with the gain coming from fewer unanswered problems.
4. ~~Transcript audit~~ — done (finding 5). Because the game itself rewards a reasoning-free policy, a fair test of
   transfer needs a game where the decisions matter: for example Kuhn Poker with standard stakes (a called bet
   doubles the pot, and the match goes to the chip leader), the variant profiled in Check-in 1
   (`results/kuhn_textarena_fixed.json`).
5. **Full math suite** (six benchmarks) on the base model and the nine H100 checkpoints: queued
   (`*-suite` results).
6. ~~Move results into `results/`~~ — H100 results are in `results/` and labelled (`results/INDEX.md`); pooled
   statistics in `results/seed_stats.json`, figures in `figures/fig6_seed_effects.png` and
   `figures/fig7_reply_length.png`.

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
