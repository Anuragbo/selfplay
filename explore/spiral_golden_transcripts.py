"""Record full Kuhn Poker transcripts from SPIRAL's own environment, for testing the port.

Plays matches with random (often invalid) actions and records, for every decision, the acting player and the
(sender, message) pairs it observed, plus the final rewards. Run it in the venv with textarena==0.6.4:

    python explore/spiral_golden_transcripts.py third_party/spiral_kuhn_env.py tests/data/spiral_kuhn_golden.json

tests/test_kuhn_spiral.py replays the same matches on selfplay.envs.kuhn_spiral and requires identical output.
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

ACTIONS = ["[check]", "[bet]", "[call]", "[fold]", "I bet", "[Bet] then [fold]"]


def record(make_env, n_matches: int) -> list[dict]:
    """Play n_matches with a seeded random action script; used for both environments."""
    matches = []
    for seed in range(n_matches):
        env = make_env()
        env.reset(num_players=2, seed=seed)
        rng = random.Random(seed)
        log = []
        while True:
            pid, obs = env.get_observation()
            log.append([pid, [[m[0], m[1]] for m in obs]])
            done, _ = env.step(rng.choice(ACTIONS))
            if done:
                break
        rewards = {str(k): v for k, v in (env.state.rewards or {}).items()}
        matches.append({"seed": seed, "log": log, "rewards": rewards})
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spiral_env", type=Path, help="path to third_party/spiral_kuhn_env.py")
    parser.add_argument("output", type=Path, help="output JSON path")
    parser.add_argument("--matches", type=int, default=100)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("spiral_kuhn_env", args.spiral_env)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matches = record(lambda: module.KuhnPokerEnv(ante=1, max_rounds=5), args.matches)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matches, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(matches)} matches to {args.output}")


if __name__ == "__main__":
    main()
