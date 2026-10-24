import json
from pathlib import Path

import pytest

from selfplay.train import expand_grid, export_checkpoints

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "grid.json").read_text(encoding="utf-8"))


def test_grid_has_twelve_self_play_runs_and_controls():
    runs = expand_grid(CONFIG)
    self_play = [r for r in runs.values() if r["opponent"] == "self"]
    controls = [r for r in runs.values() if r["opponent"] != "self"]
    assert len(self_play) == 12  # 2 sizes x 2 estimators x 3 seeds
    assert {(r["model"], r["estimator"], r["seed"]) for r in self_play} == {
        (m, e, s) for m in CONFIG["models"].values() for e in ("rae", "shared") for s in (0, 1, 2)}
    assert len(controls) == 6 and {r["opponent"] for r in controls} == {"never_fold"}
    assert all(r["learning_steps"] == 400 and r["game"] == "SpiralKuhnPoker-v1" for r in runs.values())


def test_estimator_is_the_only_difference_within_a_pair():
    runs = expand_grid(CONFIG)
    a, b = runs["selfplay_qwen3-1.7b_rae_s0"], runs["selfplay_qwen3-1.7b_shared_s0"]
    assert {k for k in a if a[k] != b[k]} == {"run_id", "estimator"}


def test_unknown_control_opponent_is_rejected():
    bad = {**CONFIG, "controls": {**CONFIG["controls"], "opponent": "bluffer"}}
    with pytest.raises(ValueError):
        expand_grid(bad)


def test_export_follows_checkpoint_contract(tmp_path):
    ub = tmp_path / "ub" / "checkpoints"
    for i in range(60):  # 60 updates -> iteration-0 .. iteration-59
        (ub / f"iteration-{i}").mkdir(parents=True)
        (ub / f"iteration-{i}" / "adapter_model.safetensors").write_text(str(i))
    run = {**expand_grid(CONFIG)["selfplay_qwen3-4b_rae_s2"], "save_every": 25}
    exported = export_checkpoints(ub, run, tmp_path / "checkpoints")
    assert [p.name for p in exported] == ["step_25", "step_50", "step_60"]
    step = tmp_path / "checkpoints" / "selfplay_qwen3-4b_rae_s2" / "step_25"
    assert (step / "adapter_model.safetensors").read_text() == "24"
    assert json.loads((step / "run_config.json").read_text())["step"] == 25


def test_worker_hook_resolves_scripted_opponents():
    textarena = pytest.importorskip("textarena")
    from selfplay.opponents import ScriptedAgent
    from selfplay.train import install_worker_hooks

    original = textarena.agents.OpenRouterAgent
    try:
        install_worker_hooks()
        install_worker_hooks()  # idempotent
        agent = textarena.agents.OpenRouterAgent("scripted:never_fold")
        assert isinstance(agent, ScriptedAgent) and agent.name == "never_fold"
        assert "SpiralKuhnPoker-v1" in textarena.envs.registration.ENV_REGISTRY
    finally:
        textarena.agents.OpenRouterAgent = original
        textarena.agents._selfplay_patched = False


def test_cli_dry_run(capsys, monkeypatch):
    import sys

    from selfplay import train

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(Path(__file__).resolve().parents[1] / "configs" /
                                                                   "grid.json"), "--list"])
    train.main()
    assert len(capsys.readouterr().out.split()) == 18
    monkeypatch.setattr(sys, "argv", ["train", "--run-id", "selfplay_qwen3-1.7b_shared_s1", "--dry-run", "--config",
                                      str(Path(__file__).resolve().parents[1] / "configs" / "grid.json")])
    train.main()
    assert json.loads(capsys.readouterr().out)["estimator"] == "shared"
