import json

import pytest

from elo import ANCHOR_RATING, anchor_games, fit, model_games, rate


def test_fit_recovers_known_rating_gap():
    # 76% wins is a 200-point Elo gap: 1 / (1 + 10^(-200/400)) = 0.7597
    games = [("strong", "equilibrium", 7597, 10000)]
    ratings = fit(games, ["equilibrium", "strong"])
    assert ratings["equilibrium"] == ANCHOR_RATING
    assert ratings["strong"] == pytest.approx(1200, abs=1)


def test_anchor_games_skip_mirror_matchups():
    kuhn = {"n_games_per_matchup": 100, "matchups": {
        "never_fold vs equilibrium": {"a_win": 0.6}, "equilibrium vs equilibrium": {"a_win": 0.5}}}
    assert anchor_games(kuhn) == [("never_fold", "equilibrium", 60, 100)]


def test_model_games_read_play_kuhn_rows(tmp_path):
    d = tmp_path / "run-step30"
    d.mkdir()
    rows = [{"run_id": "run", "step": 30, "model": "m", "opponent": o, "metric": "win_rate", "value": v,
             "n_samples": 100} for o, v in (("never_fold", 0.4), ("equilibrium", 0.5))]
    rows.append({**rows[0], "metric": "loss_by_invalid_move_rate", "value": 0.1})
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    player, info, games = model_games(d)
    assert player == "run-step30" and info["step"] == 30
    assert games == [("run-step30", "never_fold", 40, 100), ("run-step30", "equilibrium", 50, 100)]


def test_rate_keeps_a_model_that_never_wins_finite_and_ordered():
    anchors = [("never_fold", "equilibrium", 5934, 10000)]
    models = {"loser": {"games": [("loser", "equilibrium", 0, 100), ("loser", "never_fold", 0, 100)]},
              "even": {"games": [("even", "equilibrium", 50, 100), ("even", "never_fold", 40, 100)]}}
    r = rate(anchors, models, n_boot=50)
    assert r["loser"]["rating"] < r["even"]["rating"] < r["never_fold"]["rating"]
    assert r["loser"]["rating"] > 0 and r["loser"]["ci"][0] <= r["loser"]["rating"] <= r["loser"]["ci"][1]
