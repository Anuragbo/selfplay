"""Audit replayed Kuhn Poker transcripts (hpc/audit.sbatch): what the replies look like as training goes on.

For each replayed checkpoint (outputs/audit/<run_id>-step<N>/ and outputs/audit/base/), against never_fold:
reply length, the share of bare replies (nothing but the boxed action), whether the reply mentions the player's card,
the most common replies, and the action policy per card: whether the model still plays K differently from J.

    python explore/transcript_audit.py outputs/audit results/transcript_audit.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BOXED = re.compile(r"\\boxed\{(.*?)\}")
CARD = re.compile(r"Your card is: ([JQK])")
LEGAL = re.compile(r"Your available actions are: (.*)")
MENTIONS_CARD = re.compile(r"\b(J|Q|K|Jack|Queen|King)\b")
AGGRESSIVE = {"[bet]", "[call]"}
BARE_CHARS = 20  # a reply is "bare" when, without its \boxed{...}, fewer than this many non-space characters remain
EXAMPLE_CHARS = 400


def parse_turn(turn: dict) -> dict:
    """Card held, situation (opening or facing a bet), chosen action and reply shape for one model reply."""
    obs, text = turn["observation"], turn["completion"]
    legal = LEGAL.findall(obs)
    situation = "facing bet" if legal and "[call]" in legal[-1] else "no bet"
    rest = re.sub(r"\s", "", BOXED.sub("", text))
    cards = CARD.findall(obs)
    return {"card": cards[-1] if cards else None, "situation": situation, "action": turn["action"],
            "bare": len(rest) < BARE_CHARS, "mentions_card": bool(MENTIONS_CARD.search(BOXED.sub("", text))),
            "chars": len(text), "truncated": turn["finish_reason"] == "length", "text": text}


def _rate(rows: list[dict], key: str) -> float | None:
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def policy_table(replies: list[dict]) -> dict:
    """P(bet or call) per situation and card, and the K-minus-J gap: near 0 means the action ignores the card."""
    table = {}
    for situation in ("no bet", "facing bet"):
        row = {}
        for card in "JQK":
            rs = [r for r in replies if r["situation"] == situation and r["card"] == card]
            row[card] = {"n": len(rs), "aggressive": sum(r["action"] in AGGRESSIVE for r in rs) / len(rs) if rs
                         else None}
        k, j = row["K"]["aggressive"], row["J"]["aggressive"]
        row["k_minus_j"] = k - j if k is not None and j is not None else None
        table[situation] = row
    return table


def audit_checkpoint(directory: Path) -> dict:
    transcripts = [json.loads(line) for line in (directory / "transcripts.jsonl").read_text().splitlines() if line]
    matches = [json.loads(line) for line in (directory / "matches.jsonl").read_text().splitlines() if line]
    config = json.loads((directory / "run_config.json").read_text())
    pids = {(m["match"], m["opponent"]): m["model_pid"] for m in matches}
    replies = [parse_turn(t) for tr in transcripts for t in tr["turns"]
               if "completion" in t and t["pid"] == pids[(tr["match"], tr["opponent"])]]
    common = Counter(re.sub(r"\s+", " ", r["text"]).strip() for r in replies).most_common(3)
    turns = sum(m["model_turns"] for m in matches)
    return {"run_id": config["run_id"], "step": config["step"], "matches": len(matches), "replies": len(replies),
            "win_rate": sum(m["reward"] > 0 for m in matches) / len(matches),
            "invalid_move_loss_rate": sum(m["lost_by_invalid_move"] for m in matches) / len(matches),
            "tokens_per_reply": sum(m["completion_tokens"] for m in matches) / max(turns, 1),
            "truncated_rate": _rate(replies, "truncated"), "bare_reply_rate": _rate(replies, "bare"),
            "mentions_card_rate": _rate(replies, "mentions_card"),
            "most_common_replies": [{"text": t[:EXAMPLE_CHARS], "share": n / len(replies)} for t, n in common],
            "policy": policy_table(replies),
            "examples": [r["text"][:EXAMPLE_CHARS] for r in replies[:2]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = [audit_checkpoint(d) for d in sorted(args.audit_dir.iterdir()) if (d / "transcripts.jsonl").exists()]
    rows.sort(key=lambda r: (r["run_id"] != "base", r["run_id"], r["step"]))
    args.out.write_text(json.dumps({"opponent": "never_fold", "sampling": "temperature 0.6, top-p 0.95, 512 tokens",
                                    "bare_reply_chars": BARE_CHARS, "checkpoints": rows}, indent=2) + "\n")
    print(f"{'run':32} step  tok/reply  bare  card  trunc  invalid  win  K-J(open)  K-J(facing)")
    for r in rows:
        gaps = [r["policy"][s]["k_minus_j"] for s in ("no bet", "facing bet")]
        print(f"{r['run_id']:32} {r['step']:4} {r['tokens_per_reply']:9.0f} {r['bare_reply_rate']:5.2f} "
              f"{r['mentions_card_rate']:5.2f} {r['truncated_rate']:6.2f} {r['invalid_move_loss_rate']:8.2f} "
              f"{r['win_rate']:4.2f} " + "  ".join("    n/a  " if g is None else f"{g:+9.2f}" for g in gaps))


if __name__ == "__main__":
    main()
