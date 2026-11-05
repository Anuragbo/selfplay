import json

import pytest

from checkin3_stats import summarize_run, transfer_summary, window_rate


def _row(step, invalid, win=0.0):
    return {"step": step, "matches": 8, "invalid_move_loss_rate": invalid, "win_rate": win, "format_error_rate": 0.5,
            "truncated_rate": 0.1, "completion_tokens_per_turn": 300.0, "rollout_seconds": 100.0,
            "update_seconds": 20.0}


def test_window_rate_weights_by_matches():
    rate = window_rate([_row(1, 1.0), {**_row(2, 0.5), "matches": 24}], "invalid_move_loss_rate")
    assert rate["rate"] == pytest.approx(0.625) and rate["matches"] == 32
    assert 0 <= rate["ci_low"] < rate["rate"] < rate["ci_high"] <= 1


def test_summarize_run_compares_first_and_last_steps():
    rows = [_row(s, 1.0) for s in range(1, 6)] + [_row(s, 0.5, 0.25) for s in range(6, 11)]
    summary = summarize_run({"run_id": "r", "metrics": rows})
    assert summary["steps_completed"] == 10 and summary["seconds_per_step"] == pytest.approx(120)
    assert summary["invalid_move_loss_rate"]["first"]["rate"] == 1.0
    assert summary["invalid_move_loss_rate"]["last"]["rate"] == 0.5
    assert summary["win_rate"]["last"]["rate"] == 0.25


def _eval_file(path, run_id, model, value, bits, step=0, no_answer=0.0):
    rows = [{"metric": "accuracy", "value": value, "model": model, "step": step},
            {"metric": "no_answer_rate", "value": no_answer, "model": model, "step": step}]
    path.write_text(json.dumps({"run_id": run_id, "results": rows, "correct": {"MATH-500": bits}}), encoding="utf-8")


def test_transfer_summary_is_paired_against_the_base_model(tmp_path):
    _eval_file(tmp_path / "eval-base-qwen3-1.7b.json", "base", "Qwen/Qwen3-1.7B-Base", 0.5, "1,0,1,0")
    trained = tmp_path / "eval-pilot_x.json"
    _eval_file(trained, "pilot", "Qwen/Qwen3-1.7B-Base", 0.75, "1,1,1,0", step=30, no_answer=0.25)
    t = transfer_summary(tmp_path, trained)
    assert t["accuracy_when_answered"]["base"] == pytest.approx(0.5)
    assert t["accuracy_when_answered"]["trained"] == pytest.approx(1.0)
    assert t["detectable_effect"] == pytest.approx(1.96 * (0.25 / 4) ** 0.5)
    assert t["bootstrap"]["mean_diff"] == pytest.approx(0.25) and t["mcnemar"]["only_b_correct"] == 1
    assert t["step"] == 30 and t["trained"]["accuracy"] == 0.75 and t["base"]["accuracy"] == 0.5
