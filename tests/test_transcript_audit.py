import json

import pytest

from transcript_audit import audit_checkpoint, parse_turn, policy_table

OPEN = "Starting round 1 out of 5 rounds.\nYour card is: {card}\nYour available actions are: [check], [bet]"
FACING = "Player 1, submitted move: '[bet]'.\nYour card is: {card}\nYour available actions are: [fold], [call]"


def _turn(obs, text, action, finish="stop"):
    return {"pid": 0, "observation": obs, "completion": text, "action": action, "finish_reason": finish}


def test_parse_turn_reads_card_situation_and_reply_shape():
    bare = parse_turn(_turn(OPEN.format(card="K"), "\\boxed{[bet]}", "[bet]"))
    assert (bare["card"], bare["situation"], bare["bare"], bare["mentions_card"]) == ("K", "no bet", True, False)
    text = "I hold the J, the lowest card, so folding is best. \\boxed{[fold]}"
    long = parse_turn(_turn(FACING.format(card="J"), text, "[fold]", finish="length"))
    assert (long["situation"], long["bare"], long["mentions_card"], long["truncated"]) == ("facing bet", False, True,
                                                                                          True)


def test_policy_table_measures_card_sensitivity():
    replies = [{"situation": "no bet", "card": "K", "action": "[bet]"},
               {"situation": "no bet", "card": "J", "action": "[check]"},
               {"situation": "facing bet", "card": "K", "action": "[call]"},
               {"situation": "facing bet", "card": "J", "action": "[call]"}]
    table = policy_table(replies)
    assert table["no bet"]["k_minus_j"] == 1.0 and table["facing bet"]["k_minus_j"] == 0.0
    assert table["no bet"]["Q"] == {"n": 0, "aggressive": None}


def test_audit_checkpoint_only_counts_the_model_seat(tmp_path):
    turns = [_turn(OPEN.format(card="K"), "\\boxed{[bet]}", "[bet]"), {"pid": 1, "action": "[call]"},
             {**_turn(OPEN.format(card="Q"), "\\boxed{[bet]}", "[bet]"), "pid": 1}]
    (tmp_path / "transcripts.jsonl").write_text(json.dumps({"match": 0, "opponent": "never_fold", "turns": turns}))
    (tmp_path / "matches.jsonl").write_text(json.dumps(
        {"match": 0, "opponent": "never_fold", "model_pid": 0, "reward": 1, "lost_by_invalid_move": False,
         "model_turns": 1, "completion_tokens": 6}))
    (tmp_path / "run_config.json").write_text(json.dumps({"run_id": "r", "step": 10}))
    row = audit_checkpoint(tmp_path)
    assert row["replies"] == 1 and row["bare_reply_rate"] == 1.0 and row["win_rate"] == 1.0
    assert row["tokens_per_reply"] == pytest.approx(6.0)
    assert row["most_common_replies"][0] == {"text": "\\boxed{[bet]}", "share": 1.0}
