import json
from pathlib import Path

import pytest

from selfplay.advantages import NormalizeRewardsByEnv, make_estimator
from selfplay.train_single import (
    Example,
    examples_from_matches,
    latest_checkpoint,
    read_metrics,
    resolve_runs,
    rewind_metrics,
    rollout_metrics,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = json.loads((CONFIGS / "pilot.json").read_text(encoding="utf-8"))
SELF_CONFIG = json.loads((CONFIGS / "selfplay.json").read_text(encoding="utf-8"))
RUN = resolve_runs(CONFIG)["pilot_qwen3-1.7b_rae_s0"]


def _match(reward, lost, completions, pid=0):
    transcript = [{"pid": 1, "action": "[bet]"}]
    transcript += [{"pid": pid, "observation": f"obs{k}", "completion": c, "action": "x", "finish_reason": f}
                   for k, (c, f) in enumerate(completions)]
    return {"reward": reward, "model_pid": pid, "lost_by_invalid_move": lost, "transcript": transcript,
            "model_turns": len(completions), "format_errors": 0, "truncated": 0, "completion_tokens": 10}


def test_shaping_follows_unstable_baselines():
    estimator = make_estimator("rae", tau=0.0)
    match = _match(-1, True, [(r"\boxed{bet}", "stop"), ("no answer", "length")])
    first, last = examples_from_matches([match], estimator, RUN)
    assert first.reward == pytest.approx(-1 + 1.5 + 1)   # boxed, valid
    assert last.reward == pytest.approx(-1 + 0 - 1)      # no box, the losing invalid move
    assert first.finished and not last.finished
    assert "obs0" in first.prompt


def test_estimator_sees_each_match_once():
    estimator = make_estimator("rae", tau=0.5)
    examples_from_matches([_match(1, False, [(r"\boxed{bet}", "stop")] * 3)], estimator, RUN)
    assert estimator.baselines["SpiralKuhnPoker-v1"][0] == pytest.approx(0.5)


def test_constant_rewards_normalize_to_zero():
    examples = [Example("p", "r", True, -2.0) for _ in range(4)]
    assert [e.reward for e in NormalizeRewardsByEnv()(examples)] == [0.0] * 4


def test_rollout_metrics():
    row = rollout_metrics([_match(-1, True, [("a", "stop")]), _match(1, False, [("b", "stop"), ("c", "stop")])])
    assert row["win_rate"] == 0.5 and row["invalid_move_loss_rate"] == 0.5 and row["model_turns_per_match"] == 1.5


def test_latest_checkpoint_ignores_interrupted_saves(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for step, complete in ((5, True), (10, True), (15, False)):
        (tmp_path / f"step_{step}").mkdir()
        if complete:
            (tmp_path / f"step_{step}" / "complete").touch()
    assert latest_checkpoint(tmp_path)[0] == 10


def test_rewind_drops_rows_after_the_checkpoint(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in range(1, 8)))
    rewind_metrics(tmp_path, 5)
    assert [r["step"] for r in read_metrics(tmp_path, 100)] == [1, 2, 3, 4, 5]


def test_pilot_config_runs_are_valid():
    runs = resolve_runs(CONFIG)
    assert {"rae", "shared"} == {r["estimator"] for r in runs.values()}
    assert all(r["opponent"] == "never_fold" and r["steps"] > 0 for r in runs.values())


def _self_match(i, reward0, lost0=False):
    transcript = [{"pid": 0, "observation": "a", "completion": r"\boxed{bet}", "action": "[bet]",
                   "finish_reason": "stop"},
                  {"pid": 1, "observation": "b", "completion": "no box", "action": "x", "finish_reason": "stop"}]
    base = {"match": i, "opponent": "self", "transcript": transcript, "model_turns": 1, "format_errors": 0,
            "truncated": 0, "completion_tokens": 5}
    return [{**base, "model_pid": 0, "reward": reward0, "lost_by_invalid_move": lost0},
            {**base, "model_pid": 1, "reward": 0 if lost0 else -reward0, "lost_by_invalid_move": False}]


def test_self_play_trains_each_seat_on_its_own_replies():
    estimator = make_estimator("rae", tau=0.5)
    seat0, seat1 = examples_from_matches(_self_match(0, 1), estimator, RUN)
    assert "Observation: a" in seat0.prompt and "Observation: b" in seat1.prompt
    assert seat0.reward == pytest.approx(0.5 + 1.5 + 1)       # R - b with b = 0.5, boxed, valid
    assert seat1.reward == pytest.approx(-0.5 + 0 + 1)        # lost the match, no box, valid move
    assert estimator.baselines["SpiralKuhnPoker-v1"] == {0: 0.5, 1: -0.5}


def test_self_play_metrics_count_matches_not_seats():
    row = rollout_metrics(_self_match(0, 1) + _self_match(1, 0) + _self_match(2, -1, lost0=True))
    assert row["matches"] == 3 and row["seat0_win_rate"] == pytest.approx(1 / 3)
    assert row["draw_rate"] == pytest.approx(1 / 3) and row["invalid_move_match_rate"] == pytest.approx(1 / 3)


def test_selfplay_config_runs_are_valid():
    runs = resolve_runs(SELF_CONFIG)
    assert len(runs) == 6 and {r["estimator"] for r in runs.values()} == {"rae", "shared"}
    assert all(r["opponent"] == "self" and r["eval_opponent"] == "never_fold" for r in runs.values())
    pilot = resolve_runs(CONFIG)["pilot_qwen3-1.7b_rae_s0"]
    same = {k for k in pilot if k not in ("run_id", "opponent", "save_every")}
    assert all(runs["selfplay_qwen3-1.7b_rae_s0"][k] == pilot[k] for k in same)


torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")


class _Tok:
    """Byte-level tokenizer, enough to exercise the training step on a tiny random model."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + b % 60 for b in text.encode()][:80]}


def _tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from selfplay.train_single import wrap_with_lora

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                         use_sliding_window=False, layer_types=["full_attention"] * 2)
    return wrap_with_lora(Qwen3ForCausalLM(config), {**RUN, "lora": {**RUN["lora"], "lora_rank": 4, "lora_alpha": 4}})


def _optim(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=1e-2), torch.amp.GradScaler("cpu", enabled=False)


def test_encode_batch_marks_reply_tokens_and_truncates_prompt():
    from selfplay.train_single import encode_batch

    ids, attention, reply = encode_batch([Example("abcd", "xy", True, 1.0), Example("a", "x", False, 1.0)], _Tok(), 100)
    assert ids.shape == (2, 7) and attention[1].sum() == 2
    assert reply[0].tolist() == [False] * 4 + [True] * 3     # reply plus end-of-text
    assert reply[1].tolist() == [False, True] + [False] * 5
    short, _, mask = encode_batch([Example("abcdef", "xy", False, 1.0)], _Tok(), 4)
    assert short.shape == (1, 4) and mask.sum() == 2


def test_update_changes_only_the_adapter():
    from selfplay.train_single import reinforce_update

    model = _tiny_model()
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer, scaler = _optim(model)
    examples = [Example("hello", "world", True, 1.0), Example("foo", "bar baz", False, -1.0)]
    stats = reinforce_update(model, _Tok(), examples, optimizer, scaler, {**RUN, "mini_batch_size": 1})
    assert stats["examples"] == 2 and stats["grad_norm"] > 0 and stats["loss"] == stats["loss"]
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora_" in n for n in changed)


def test_training_moves_probability_toward_rewarded_reply():
    from selfplay.train_single import encode_batch, reinforce_update

    model = _tiny_model()
    optimizer, scaler = _optim(model)
    good, bad = Example("prompt", "aaaa", False, 1.0), Example("prompt", "zzzz", False, -1.0)
    run = {**RUN, "max_new_tokens": 4}

    def good_log_prob():
        ids, attention, reply = encode_batch([good], _Tok(), 100)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        token = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return (token * reply[:, 1:]).sum().item()

    start = good_log_prob()
    for _ in range(15):
        reinforce_update(model, _Tok(), [good, bad], optimizer, scaler, run)
    assert good_log_prob() > start


def test_train_loop_saves_resumes_and_continues(tmp_path):
    pytest.importorskip("textarena")
    from selfplay.train_single import train

    calls = []

    def generate(prompts, n):
        calls.append(len(prompts))
        return [[(r"\boxed{bet}", 3, "stop")] for _ in prompts]

    run = {**RUN, "steps": 2, "save_every": 1, "matches_per_step": 2, "mini_batch_size": 2, "max_new_tokens": 4}
    model = _tiny_model()
    assert train(run, model, _Tok(), tmp_path, generate=generate, device="cpu") == 2
    assert latest_checkpoint(tmp_path)[0] == 2
    assert (tmp_path / "step_2" / "run_config.json").exists()
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    resumed = _tiny_model()
    assert train({**run, "steps": 3}, resumed, _Tok(), tmp_path, generate=generate, device="cpu") == 3
    assert [r["step"] for r in read_metrics(tmp_path, 100)] == [1, 2, 3]
    fresh = _tiny_model()
    assert train(run, fresh, _Tok(), tmp_path, generate=generate, device="cpu") == 3
    assert any(not torch.equal(p, weights[n]) for n, p in fresh.named_parameters() if n in weights)


def test_self_play_train_loop_logs_eval_metrics(tmp_path):
    pytest.importorskip("textarena")
    from selfplay.train_single import train

    def generate(prompts, n):
        return [[(r"\boxed{call}" if "[call]" in p else r"\boxed{check}", 3, "stop")] for p in prompts]

    run = {**resolve_runs(SELF_CONFIG)["selfplay_qwen3-1.7b_rae_s0"], "steps": 2, "save_every": 2,
           "matches_per_step": 2, "mini_batch_size": 4, "max_new_tokens": 4, "eval_matches": 4, "eval_every": 2}
    assert train(run, _tiny_model(), _Tok(), tmp_path, generate=generate, device="cpu") == 2
    first, second = read_metrics(tmp_path, 100)
    assert first["matches"] == 2 and "seat0_win_rate" in first and "eval_win_rate" not in first
    assert second["eval_matches"] == 4 and "eval_seat0_win_rate" not in second


def test_first_ctrl_c_finishes_the_step_saves_and_stops(tmp_path):
    pytest.importorskip("textarena")
    import os
    import signal

    from selfplay.train_single import train

    def generate(prompts, n):
        if not getattr(generate, "sent", False):
            generate.sent = True
            os.kill(os.getpid(), signal.SIGINT)  # as if Ctrl-c were pressed during step 1's rollout
        return [[(r"\boxed{bet}", 3, "stop")] for _ in prompts]

    run = {**RUN, "steps": 30, "save_every": 10, "matches_per_step": 2, "mini_batch_size": 2, "max_new_tokens": 4}
    assert train(run, _tiny_model(), _Tok(), tmp_path, generate=generate, device="cpu") == 1
    assert latest_checkpoint(tmp_path)[0] == 1
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # handlers restored
    assert train({**run, "steps": 2}, _tiny_model(), _Tok(), tmp_path, generate=generate, device="cpu") == 2
    assert [r["step"] for r in read_metrics(tmp_path, 100)] == [1, 2]


def test_time_budget_saves_the_last_step(tmp_path):
    pytest.importorskip("textarena")
    from selfplay.train_single import train

    def generate(prompts, n):
        return [[(r"\boxed{bet}", 3, "stop")] for _ in prompts]

    run = {**RUN, "steps": 30, "save_every": 10, "matches_per_step": 2, "mini_batch_size": 2, "max_new_tokens": 4}
    assert train(run, _tiny_model(), _Tok(), tmp_path, generate=generate, device="cpu", max_seconds=0.001) == 1
    assert latest_checkpoint(tmp_path)[0] == 1
