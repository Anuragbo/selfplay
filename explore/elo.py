"""Elo ratings of models against the fixed scripted opponents of SPIRAL's Kuhn Poker.

Players are the four scripted policies and every model replayed by hpc/elo.sbatch. The scripted policies are
anchored by their 10,000-game matchups from Check-in 1 (results/kuhn_spiral.json); each model adds its results
against the four of them. Ratings maximise the Bradley-Terry likelihood on the Elo scale,
P(i beats j) = 1 / (1 + 10^((R_j - R_i) / 400)), with equilibrium fixed at 1000. One virtual win and one virtual loss
against equilibrium per model keep a model that never wins at a finite rating. 95% intervals come from a parametric
bootstrap over all match counts. SPIRAL's matches have no draws.

    python explore/elo.py outputs/elo results/elo.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

ANCHOR, ANCHOR_RATING = "equilibrium", 1000.0
SCALE = 400 / np.log(10)
PRIOR_GAMES = 1  # virtual wins and losses against the anchor, per model


def anchor_games(kuhn: dict) -> list[tuple[str, str, int, int]]:
    """(player a, player b, wins of a, games) from the scripted matchups, skipping self-matches."""
    n = kuhn["n_games_per_matchup"]
    games = []
    for name, m in kuhn["matchups"].items():
        a, b = name.split(" vs ")
        if a != b:
            games.append((a, b, round(m["a_win"] * n), n))
    return games


def model_games(directory: Path) -> tuple[str, dict, list[tuple[str, str, int, int]]]:
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
    wins = [r for r in rows if r["metric"] == "win_rate"]
    info = {"run_id": wins[0]["run_id"], "step": wins[0]["step"], "model": wins[0]["model"]}
    player = directory.name
    games = [(player, r["opponent"], round(r["value"] * r["n_samples"]), r["n_samples"]) for r in wins]
    return player, info, games


def fit(games: list[tuple[str, str, float, float]], players: list[str], iters: int = 2000) -> dict:
    """Maximum-likelihood ratings by Newton's method, anchor held fixed."""
    index = {p: i for i, p in enumerate(players)}
    a = np.array([index[g[0]] for g in games])
    b = np.array([index[g[1]] for g in games])
    w = np.array([g[2] for g in games], dtype=float)
    n = np.array([g[3] for g in games], dtype=float)
    free = np.array([p != ANCHOR for p in players])
    theta = np.zeros(len(players))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
        grad = np.zeros(len(players))
        np.add.at(grad, a, w - n * p)
        np.add.at(grad, b, -(w - n * p))
        h = n * p * (1 - p)
        hess = np.zeros((len(players), len(players)))
        np.add.at(hess, (a, a), -h)
        np.add.at(hess, (b, b), -h)
        np.add.at(hess, (a, b), h)
        np.add.at(hess, (b, a), h)
        step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        theta[free] -= step
        if np.abs(step).max() < 1e-10:
            break
    return {p: float(ANCHOR_RATING + SCALE * theta[index[p]]) for p in players}


def rate(anchors: list, models: dict, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Point ratings and bootstrap intervals; observed games are resampled, the virtual prior games are not."""
    observed = anchors + [g for m in models.values() for g in m["games"]]
    prior = [(p, ANCHOR, PRIOR_GAMES, 2 * PRIOR_GAMES) for p in models]
    players = sorted({g[0] for g in observed + prior} | {g[1] for g in observed + prior})
    point = fit(observed + prior, players)
    rng = np.random.default_rng(seed)
    boots = [fit([(x, y, rng.binomial(n, w / n), n) for x, y, w, n in observed] + prior, players, iters=50)
             for _ in range(n_boot)]
    return {p: {"rating": point[p], "ci": [float(np.quantile([r[p] for r in boots], q)) for q in (0.025, 0.975)]}
            for p in players}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("elo_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--anchors", type=Path, default=Path("results/kuhn_spiral.json"))
    args = parser.parse_args()
    anchors = anchor_games(json.loads(args.anchors.read_text()))
    models = {}
    for d in sorted(args.elo_dir.iterdir()):
        if (d / "results.jsonl").exists():
            player, info, games = model_games(d)
            models[player] = {**info, "games": games}
    ratings = rate(anchors, models)
    out = {"anchor": {ANCHOR: ANCHOR_RATING}, "prior_games_vs_anchor": PRIOR_GAMES,
           "scripted": {p: ratings[p] for p in ratings if p not in models},
           "models": [{"player": p, **{k: v for k, v in m.items() if k != "games"}, **ratings[p],
                       "results": {g[1]: {"wins": g[2], "games": g[3]} for g in m["games"]}}
                      for p, m in models.items()]}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for p, r in sorted(ratings.items(), key=lambda kv: -kv[1]["rating"]):
        print(f"{p:42} {r['rating']:7.0f}  [{r['ci'][0]:6.0f}, {r['ci'][1]:6.0f}]")


if __name__ == "__main__":
    main()
