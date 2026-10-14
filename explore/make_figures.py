"""Generate the Check-in 1 figures from the JSON files in results/."""

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
SUITE = {"AIME24": "AIME 2024", "AMC23": "AMC 2023", "Minerva": "Minerva Math",
         "MATH-500": "MATH-500", "OlympiadBench": "OlympiadBench"}
CHANGE_RATES = (0.16, 0.12, 0.08)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory with the result JSON files")
    parser.add_argument("figures", type=Path, help="output directory for PNG figures")
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    _style()
    detectable_effect_figure(args.results, args.figures / "fig1_detectable_effect.png")
    kuhn_matchups_figure(args.results, args.figures / "fig2_kuhn_matchups.png")
    print(f"figures written to {args.figures}")


if __name__ == "__main__":
    main()
