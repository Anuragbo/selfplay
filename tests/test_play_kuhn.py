import pytest

from selfplay.play_kuhn import QWEN3_ZS, extract_action, summarize

pytest.importorskip("textarena")

from selfplay.opponents import ScriptedAgent  # noqa: E402
from selfplay.play_kuhn import play  # noqa: E402

META = {"run_id": "t", "step": 0, "model": "m", "game": "g", "estimator": "none", "seed": 0}


def test_extract_action_follows_unstable_baselines():
    assert extract_action(r"I think \boxed{bet}") == ("[bet]", True)
    assert extract_action(r"\boxed{[call]}") == ("[call]", True)
    assert extract_action(r"\boxed{fold} no wait \boxed{check}") == ("[check]", True)
    assert extract_action("I will [check]") == ("I will [check]", False)
    assert extract_action(r"\boxed{}") == (r"\boxed{}", False)


def test_prompt_wraps_observation():
    p = QWEN3_ZS.format(obs="OBS")
    assert "Observation: OBS\n" in p and r"\boxed{}" in p and p.endswith("<|im_start|>assistant\n")


def _scripted_model(policy):
    """A fake LLM that plays a scripted policy and answers in \\boxed{}."""
    agent = ScriptedAgent(policy, seed=0)

    def generate(prompts, n):
        out = []
        for p in prompts:
            obs = p.split("Observation: ", 1)[1].rsplit("\nPlease reason", 1)[0]
            out.append([(rf"\boxed{{{agent(obs).strip('[]')}}}", 5, "stop")] * n)
        return out

    return generate


def test_never_fold_model_beats_random():
    matches = play(_scripted_model("never_fold"), "random", 300, keep_transcripts=2)
    rows = {r["metric"]: r["value"] for r in summarize(matches, META)}
    assert rows["win_rate"] > 0.6 and rows["format_error_rate"] == 0 and rows["loss_by_invalid_move_rate"] == 0
    assert 4 <= rows["model_turns_per_match"] <= 8  # about half of 11.3 decisions per match
    assert "transcript" in matches[0] and "transcript" not in matches[2]
    assert {m["model_pid"] for m in matches} == {0, 1}


def test_reply_echoing_game_text_does_not_break_the_opponent():
    """An unboxed reply containing a valid action stays in the history, including any game text it repeats."""
    reply = "Starting round 9 and more text. Your available actions are: [fold]. I will [check] or [call]"
    matches = play(lambda prompts, n: [[(reply, 7, "stop")] for _ in prompts], "never_fold", 20)
    assert len(matches) == 20 and all(m["format_errors"] == m["model_turns"] for m in matches)


def test_unformatted_reply_loses_by_invalid_move():
    matches = play(lambda prompts, n: [[("I am not sure", 7, "length")]] * len(prompts), "never_fold", 10)
    rows = {r["metric"]: r["value"] for r in summarize(matches, META)}
    assert rows["win_rate"] == 0 and rows["loss_by_invalid_move_rate"] == 1 and rows["format_error_rate"] == 1
    assert rows["model_turns_per_match"] == 1 and rows["truncated_rate"] == 1
