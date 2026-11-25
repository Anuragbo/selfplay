"""Generate the report figures from the JSON files in results/ (Check-in 1: Figs. 1-2; 2: 3-4; 3: 5; 5: 6-7).

Fig. 6 reads results/seed_stats.json, written by explore/seed_stats.py.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SPIRAL_MATH500_GAIN = 3.0  # Qwen3-4B-Base, Kuhn Poker only: 73.4 -> 76.4 (SPIRAL, Table 1)
SPIRAL_MATH500_BASE_4B = 73.4  # Qwen3-4B-Base before training (SPIRAL, Table 1)
SUITE = {"AIME24": "AIME 2024", "AMC23": "AMC 2023", "Minerva": "Minerva Math",
         "MATH-500": "MATH-500", "OlympiadBench": "OlympiadBench"}
CHANGE_RATES = (0.16, 0.12, 0.08)
CHECKIN3_RUNS = ("train-pilot_qwen3-1.7b_rae_s0.json", "train-pilot_qwen3-1.7b_shared_s0.json")  # Colab T4
H100 = "NVIDIA H100 PCIe"

KUHN_VARIANTS = [
    ("TextArena 0.7.4 (PyPI)", "kuhn_textarena.json", BLUE),
    ("SPIRAL KuhnPoker-v1", "kuhn_spiral.json", ORANGE),
    ("Corrected stakes (control)", "kuhn_textarena_fixed.json", AQUA),
]
KUHN_ROWS = [
    ("Never fold", "never_fold vs equilibrium"),
    ("Uniform random", "random vs equilibrium"),
    ("Calling station", "calling_station vs equilibrium"),
    ("Equilibrium (mirror)", "equilibrium vs equilibrium"),
]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    })


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decisive_share(m: dict) -> float:
    """Percentage of non-drawn matches won."""
    return 100 * m["a_win"] / (m["a_win"] + m["a_loss"])


def detectable_effect_figure(results: Path, out: Path) -> None:
    stats = _load(results / "benchmark_stats.json")
    sizes = {label: stats[key]["n"] for label, key in SUITE.items()}
    pooled = sum(sizes.values())
    ns = [10 ** (1.2 + k * 0.02) for k in range(111)]

    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    for rate, color in zip(CHANGE_RATES, (AQUA, ORANGE, BLUE), strict=True):
        ax.plot(ns, [196 * math.sqrt(rate / n) for n in ns], color=color, lw=2, label=f"{int(rate * 100)}%")
    for k, (label, n) in enumerate(list(sizes.items()) + [("All five pooled", pooled)]):
        ax.axvline(n, color=GRID, lw=1, zorder=0)
        ax.text(n, 32 if k % 2 == 0 else 27, f"{label}\n(n={n:,})", ha="center", va="top", fontsize=6.6,
                color=INK_SECONDARY, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=True, facecolor="white", edgecolor="none",
              framealpha=1, fontsize=7.5, title="Answers that differ before vs. after", title_fontsize=7.5)
    ax.axhline(SPIRAL_MATH500_GAIN, color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(17, SPIRAL_MATH500_GAIN + 0.25, f"SPIRAL's MATH-500 gain at 4B: +{SPIRAL_MATH500_GAIN:.1f} points",
            fontsize=7.5, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(16, 3000)
    ax.set_ylim(0, 33)
    ticks = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xlabel("Number of evaluation problems (log scale)")
    ax.set_ylabel("Paired 95% CI half-width\n(accuracy points)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def kuhn_matchups_figure(results: Path, out: Path) -> None:
    data = {name: _load(results / file)["matchups"] for name, file, _ in KUHN_VARIANTS}
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    height = 0.25
    for i, (_, key) in enumerate(KUHN_ROWS):
        y = len(KUHN_ROWS) - 1 - i
        for j, (name, _, color) in enumerate(KUHN_VARIANTS):
            yy = y + (1 - j) * (height + 0.02)
            value = _decisive_share(data[name][key])
            ax.barh(yy, value, height=height, color=color, label=name if i == 0 else None)
            ax.text(value + 0.8, yy, f"{value:.1f}%", va="center", fontsize=7.5, color=INK)
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(KUHN_ROWS)))
    ax.set_yticklabels([label for label, _ in reversed(KUHN_ROWS)])
    ax.set_xlim(0, 78)
    ax.set_xlabel("Share of decisive matches won against the textbook equilibrium (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def math500_base_figure(results: Path, out: Path) -> None:
    stats = _load(results / "checkin2_stats.json")
    names = list(stats["accuracy"])
    fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=200)
    for y, (name, color) in enumerate(zip(names, (BLUE, ORANGE), strict=True)):
        acc = stats["accuracy"][name]
        value, low, high = (100 * acc[k] for k in ("accuracy", "ci_low", "ci_high"))
        ax.barh(y, value, height=0.5, color=color)
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=INK, capsize=3, lw=1)
        ax.text(high + 1.0, y, f"{value:.1f}%  [{low:.1f}, {high:.1f}]", va="center", fontsize=7.5, color=INK)
    ax.plot([SPIRAL_MATH500_BASE_4B], [1], marker="D", color=INK, ms=5, zorder=5)
    ax.annotate(f"SPIRAL reports {SPIRAL_MATH500_BASE_4B:.1f}", (SPIRAL_MATH500_BASE_4B, 1), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    boot = stats["difference_4b_minus_1_7b"]["bootstrap"]
    ax.text(2, -0.62, f"Paired difference 4B - 1.7B: {100 * boot['mean_diff']:+.1f} points, 95% CI "
            f"[{100 * boot['ci_low']:+.1f}, {100 * boot['ci_high']:+.1f}], McNemar "
            f"p = {stats['difference_4b_minus_1_7b']['mcnemar']['p_value']:.1e}", fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.85, 1.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("MATH-500 accuracy, greedy decoding, 500 problems (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def kuhn_base_figure(results: Path, out: Path) -> None:
    data = _load(results / "kuhn-base-qwen3-1.7b.json")
    counts = {int(k): v for k, v in data["match_summary"]["model_turns_per_match_counts"].items()}
    metrics = {r["metric"]: r["value"] for r in data["results"]}
    turns = range(1, max(counts) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    ax.bar(list(turns), [counts.get(t, 0) for t in turns], width=0.6, color=ORANGE)
    for t in turns:
        ax.text(t, counts.get(t, 0) + 0.8, str(counts.get(t, 0)), ha="center", fontsize=7.5, color=INK)
    n = sum(counts.values())
    ax.text(0.98, 0.95, f"{n} matches vs. never fold: {round(metrics['win_rate'] * n)} won, "
            f"{round(metrics['loss_by_invalid_move_rate'] * n)} lost by an invalid move\n"
            f"Replies without \\boxed{{}}: {100 * metrics['format_error_rate']:.0f}%   "
            f"Cut off at 4,096 tokens: {100 * metrics['truncated_rate']:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(list(turns))
    ax.set_ylim(0, max(counts.values()) * 1.45)
    ax.set_xlabel("Model turns per match (the last one was the invalid move that lost it)")
    ax.set_ylabel("Matches")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def training_curves_figure(results: Path, out: Path, smooth: int = 5) -> None:
    runs = [_load(results / name) for name in CHECKIN3_RUNS if (results / name).exists()]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200, sharex=True)
    panels = (("invalid_move_loss_rate", "Matches lost by an invalid move (%)"),
              ("format_error_rate", "Replies without \\boxed{} (%)"))
    for ax, (key, title) in zip(axes, panels, strict=True):
        for run, color in zip(runs, (BLUE, ORANGE, AQUA, MUTED), strict=False):
            rows = run["metrics"]
            steps = [r["step"] for r in rows]
            values = [100 * r[key] for r in rows]
            smoothed = [sum(values[max(0, i - smooth + 1):i + 1]) / len(values[max(0, i - smooth + 1):i + 1])
                        for i in range(len(values))]
            ax.plot(steps, values, color=color, lw=0.6, alpha=0.35)
            ax.plot(steps, smoothed, color=color, lw=1.8, label=run["label"])
        ax.set_title(title, fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Training step")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    if runs:
        axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def seed_effects_figure(results: Path, out: Path, floor: float = -15.0) -> None:
    """Per-seed MATH-500 change vs. the same-GPU base, one row per arm, with the seed-pooled mean and interval."""
    stats = _load(results / "seed_stats.json")
    arms = stats["primary"]["arms"]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(arms) + 1.0), dpi=200)
    for y, arm in enumerate(arms):
        color = BLUE if arm["estimator"] == "rae" else ORANGE
        runs = [r for r in stats["runs"] if r["arm"] == arm["arm"] and r["estimator"] == arm["estimator"]]
        for r in runs:
            x = 100 * r["change"]
            ax.scatter(max(x, floor), y + 0.12, s=28, zorder=3, color=color if not r["collapsed"] else "white",
                       edgecolors=color, linewidths=1.2)
            if x < floor:
                ax.annotate(f"{x:+.0f} (s{r['seed']})", (floor, y + 0.12), xytext=(4, -9),
                            textcoords="offset points", fontsize=6.5, color=MUTED)
        lo, hi = (100 * v for v in arm["hierarchical_ci"])
        ax.plot([max(lo, floor), hi], [y - 0.15, y - 0.15], color=INK, lw=1.4)
        ax.plot(max(100 * arm["mean_change"], floor), y - 0.15, marker="|", markersize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    names = {"rae": "RAE", "shared": "shared"}
    ax.set_yticks(range(len(arms)),
                  [f"{a['arm'].replace(' vs never_fold', '')}, {names[a['estimator']]}" for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(floor - 1, None)
    ax.set_xlabel(f"MATH-500 change vs. base ({100 * stats['base_accuracy']:.1f}%), percentage points")
    ax.set_title("Dots: seeds (hollow = reply-length collapse). Bar: mean with seed + problem bootstrap 95% CI",
                 fontsize=7.5, color=INK_SECONDARY)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def reply_length_figure(results: Path, out: Path) -> None:
    """Reply length per move over training for every H100 run: self-play and pilot side by side."""
    runs = [d for d in (_load(p) for p in sorted(results.glob("train-*.json")))
            if (d.get("versions") or {}).get("gpu") == H100]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200, sharey=True)
    for ax, prefix, title in ((axes[0], "selfplay", "Self-play"), (axes[1], "pilot", "Pilot vs. never_fold")):
        for run in (r for r in runs if r["run_id"].startswith(prefix)):
            cfg = run["config"]
            rows = run["metrics"]
            ax.plot([r["step"] for r in rows], [r["completion_tokens_per_turn"] for r in rows], lw=1.4,
                    color=BLUE if cfg["estimator"] == "rae" else ORANGE, ls=("-", "--", ":")[cfg["seed"] % 3],
                    label=f"{cfg['estimator']} s{cfg['seed']}")
        ax.set_title(title, fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("Training step")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    axes[0].set_ylabel("Tokens per move")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory with the result JSON files")
    parser.add_argument("figures", type=Path, help="output directory for PNG figures")
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    _style()
    detectable_effect_figure(args.results, args.figures / "fig1_detectable_effect.png")
    kuhn_matchups_figure(args.results, args.figures / "fig2_kuhn_matchups.png")
    math500_base_figure(args.results, args.figures / "fig3_math500_base.png")
    kuhn_base_figure(args.results, args.figures / "fig4_kuhn_base_turns.png")
    training_curves_figure(args.results, args.figures / "fig5_training_curves.png")
    if (args.results / "seed_stats.json").exists():
        seed_effects_figure(args.results, args.figures / "fig6_seed_effects.png")
    reply_length_figure(args.results, args.figures / "fig7_reply_length.png")
    print(f"figures written to {args.figures}")


if __name__ == "__main__":
    main()
