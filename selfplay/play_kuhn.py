"""A language model plays SPIRAL's Kuhn Poker against scripted opponents, before any training.

Games run exactly as UnstableBaselines runs them: TextArena's default observation wrapper, the "qwen3-zs" prompt
template, the action taken from the last \\boxed{} (or the raw text if there is none), and error_allowance = 0,
so the first invalid move loses the match. Seats alternate; all matches advance in lockstep so the model's turns
are generated in one batch.

Reports, per opponent: win rate, how often a match was lost by an invalid move, how often a reply had no \\boxed{}
answer (format error), decisions and completion tokens per model turn. Writes matches.jsonl (one row per match),
transcripts.jsonl (the first few matches in full) and results.jsonl (results-contract rows) to --out-dir.

Example (Colab, T4):
  python -m selfplay.play_kuhn --model Qwen/Qwen3-1.7B-Base --opponents never_fold --matches 200 \\
      --run-id kuhn-base-1.7b --out-dir outputs/kuhn-base-1.7b --print-summary
"""

import argparse
import json
import re
import time
from pathlib import Path

from selfplay.envs import SPIRAL_KUHN_ID, register_envs
from selfplay.generation import Generate, add_generation_args, make_generator, versions
from selfplay.opponents import POLICIES, ScriptedAgent

# UnstableBaselines 0.2.2, unstable/utils/templates.py: "qwen3-zs" and extract_action_and_format_feedback.
QWEN3_ZS = ("<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\n"
            "Observation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            "<|im_end|>\n<|im_start|>assistant\n")


def extract_action(raw: str) -> tuple[str, bool]:
    """Action string for the environment and whether the reply had a non-empty \\boxed{} answer."""
    matches = re.findall(r"\\boxed\{(.*?)\}", raw)
    if matches and matches[-1].strip():
        last = matches[-1].strip()
        return (f"[{last}]" if "[" not in last else last), True
    return raw, False


def play(generate: Generate, opponent: str, n_matches: int, *, seed: int = 0, keep_transcripts: int = 0,
         make_env=None) -> list[dict]:
    """Play n_matches between the model and one scripted opponent; returns one summary dict per match."""
    import textarena as ta

    register_envs()
    make_env = make_env or (lambda: ta.make(SPIRAL_KUHN_ID))
    games = []
    for i in range(n_matches):
        env = make_env()
        env.reset(num_players=2, seed=seed + i)
        env.state.error_allowance = 0
        games.append({"env": env, "model_pid": i % 2, "opponent": ScriptedAgent(opponent, seed=seed + i),
                      "done": False, "turns": [], "transcript": [] if i < keep_transcripts else None})

    while True:
        waiting = []
        for g in games:
            while not g["done"]:
                pid, obs = g["env"].get_observation()
                if pid == g["model_pid"]:
                    waiting.append((g, obs))
                    break
                action = g["opponent"](obs)
                g["done"], _ = g["env"].step(action)
                if g["transcript"] is not None:
                    g["transcript"].append({"pid": pid, "action": action})
        if not waiting:
            break
        outputs = generate([QWEN3_ZS.format(obs=obs) for _, obs in waiting], 1)
        for (g, obs), [(text, n_tokens, finish)] in zip(waiting, outputs, strict=True):
            action, formatted = extract_action(text)
            g["done"], _ = g["env"].step(action)
            g["turns"].append({"formatted": formatted, "tokens": n_tokens, "truncated": finish == "length"})
            if g["transcript"] is not None:
                g["transcript"].append({"pid": g["model_pid"], "observation": obs, "completion": text,
                                        "action": action, "finish_reason": finish})

    matches = []
    for i, g in enumerate(games):
        rewards, game_info = g["env"].close()
        me = g["model_pid"]
        matches.append({
            "match": i, "seed": seed + i, "opponent": opponent, "model_pid": me, "reward": rewards[me],
            "lost_by_invalid_move": bool(game_info[me]["invalid_move"]), "model_turns": len(g["turns"]),
            "format_errors": sum(not t["formatted"] for t in g["turns"]),
            "truncated": sum(t["truncated"] for t in g["turns"]),
            "completion_tokens": sum(t["tokens"] for t in g["turns"]),
            **({"transcript": g["transcript"]} if g["transcript"] is not None else {}),
        })
    return matches


def summarize(matches: list[dict], meta: dict) -> list[dict]:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = []
    for opponent in sorted({m["opponent"] for m in matches}):
        ms = [m for m in matches if m["opponent"] == opponent]
        turns = sum(m["model_turns"] for m in ms)
        metrics = {
            "win_rate": sum(m["reward"] > 0 for m in ms) / len(ms),
            "loss_by_invalid_move_rate": sum(m["lost_by_invalid_move"] for m in ms) / len(ms),
            "format_error_rate": sum(m["format_errors"] for m in ms) / max(turns, 1),
            "truncated_rate": sum(m["truncated"] for m in ms) / max(turns, 1),
            "model_turns_per_match": turns / len(ms),
            "completion_tokens_per_turn": sum(m["completion_tokens"] for m in ms) / max(turns, 1),
        }
        for metric, value in metrics.items():
            rows.append({**meta, "opponent": opponent, "benchmark": f"{SPIRAL_KUHN_ID} vs {opponent}",
                         "metric": metric, "value": value, "n_samples": len(ms), "timestamp": stamp})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser, max_tokens=2048)
    parser.add_argument("--opponents", nargs="+", default=["never_fold"], choices=sorted(POLICIES))
    parser.add_argument("--matches", type=int, default=200, help="matches per opponent")
    parser.add_argument("--transcripts", type=int, default=5, help="matches per opponent kept in full")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--estimator", default="none")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(prompt_template=QWEN3_ZS, versions=versions())
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    generate = make_generator(args)
    matches = []
    for opponent in args.opponents:
        t0 = time.time()
        matches += play(generate, opponent, args.matches, seed=args.seed, keep_transcripts=args.transcripts)
        print(f"{opponent}: {args.matches} matches in {time.time() - t0:.0f}s", flush=True)

    meta = {"run_id": args.run_id, "step": args.step, "model": args.model, "game": SPIRAL_KUHN_ID,
            "estimator": args.estimator, "seed": args.seed}
    rows = summarize(matches, meta)
    with (args.out_dir / "matches.jsonl").open("w", encoding="utf-8") as f, \
            (args.out_dir / "transcripts.jsonl").open("w", encoding="utf-8") as t:
        for m in matches:
            transcript = m.pop("transcript", None)
            f.write(json.dumps(m) + "\n")
            if transcript is not None:
                t.write(json.dumps({"match": m["match"], "opponent": m["opponent"], "turns": transcript}) + "\n")
    (args.out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    for r in rows:
        print(f"{r['benchmark']:36s} {r['metric']:28s} {r['value']:.4f}", flush=True)
    if args.print_summary:
        summary = json.dumps({"results": rows, "versions": config["versions"]})
        print(f"SUMMARY_BEGIN\n{summary}\nSUMMARY_END", flush=True)


if __name__ == "__main__":
    main()
