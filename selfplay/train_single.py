"""Train LoRA adapters with REINFORCE on one GPU, against a scripted Kuhn Poker opponent.

One model copy alternates between playing matches and updating the adapter, so the loop fits a single 15 GB T4 in
fp16. Rewards follow UnstableBaselines: the match reward goes through the advantage estimator, each reply then gets
+1.5 for a \\boxed{} answer and +1 for a valid move or -1 for an invalid one, and rewards are z-scored per batch.

Checkpoints go to {checkpoints}/{run_id}/step_{N}/ with a run_config.json. A run resumes from its newest complete
checkpoint, and metrics.jsonl gets one row per training step.

    python -m selfplay.train_single --config configs/pilot.json --list
    python -m selfplay.train_single --config configs/pilot.json --run-id pilot_qwen3-1.7b_rae_s0
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from selfplay.advantages import NormalizeRewardsByEnv, make_estimator
from selfplay.envs import SPIRAL_KUHN_ID
from selfplay.generation import Generate, hf_generate_fn, versions
from selfplay.play_kuhn import QWEN3_ZS, extract_action, play

COMPLETE_MARKER = "complete"
METRICS_FILE = "metrics.jsonl"


@dataclass
class Example:
    prompt: str
    reply: str
    finished: bool
    reward: float
    env_id: str = SPIRAL_KUHN_ID


def examples_from_matches(matches: list[dict], estimator, run: dict) -> list[Example]:
    """One shaped training example per model reply; the estimator sees each match reward once."""
    examples = []
    for match in matches:
        advantage = estimator(match["reward"], match["model_pid"], SPIRAL_KUHN_ID)
        turns = [t for t in match["transcript"] if "completion" in t]
        for k, turn in enumerate(turns):
            invalid = match["lost_by_invalid_move"] and k == len(turns) - 1
            _, formatted = extract_action(turn["completion"])
            reward = advantage + (run["format_reward"] if formatted else 0.0)
            reward += run["invalid_move_penalty"] if invalid else run["valid_move_reward"]
            examples.append(Example(QWEN3_ZS.format(obs=turn["observation"]), turn["completion"],
                                    turn["finish_reason"] != "length", reward))
    return examples


def rollout_metrics(matches: list[dict]) -> dict:
    n, turns = len(matches), sum(m["model_turns"] for m in matches)
    return {
        "matches": n,
        "win_rate": sum(m["reward"] > 0 for m in matches) / n,
        "invalid_move_loss_rate": sum(m["lost_by_invalid_move"] for m in matches) / n,
        "format_error_rate": sum(m["format_errors"] for m in matches) / max(turns, 1),
        "truncated_rate": sum(m["truncated"] for m in matches) / max(turns, 1),
        "model_turns_per_match": turns / n,
        "completion_tokens_per_turn": sum(m["completion_tokens"] for m in matches) / max(turns, 1),
    }


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    """Newest step directory whose save finished, or None."""
    steps = [(int(p.name.split("_")[1]), p) for p in run_dir.glob("step_*") if (p / COMPLETE_MARKER).exists()]
    return max(steps, default=None)


def read_metrics(run_dir: Path, up_to_step: int) -> list[dict]:
    path = run_dir / METRICS_FILE
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["step"] <= up_to_step]


def rewind_metrics(run_dir: Path, up_to_step: int) -> None:
    """Drop rows written after the checkpoint being resumed, so a step is never counted twice."""
    rows = read_metrics(run_dir, up_to_step)
    (run_dir / METRICS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def encode_batch(examples: list[Example], tok, max_train_len: int):
    """Right-padded input ids, attention mask and a mask selecting reply tokens, keeping the end of long prompts."""
    import torch

    rows = []
    for example in examples:
        prompt = tok(example.prompt, add_special_tokens=False)["input_ids"]
        reply = tok(example.reply, add_special_tokens=False)["input_ids"]
        if example.finished:
            reply = [*reply, tok.eos_token_id]
        reply = reply[:max_train_len]
        prompt = prompt[max(0, len(prompt) + len(reply) - max_train_len):]
        rows.append((prompt + reply, len(prompt)))
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    reply_mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, (ids, prompt_len) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention[i, :len(ids)] = 1
        reply_mask[i, prompt_len:len(ids)] = True
    return input_ids, attention, reply_mask


def reinforce_update(model, tok, examples: list[Example], optimizer, scaler, run: dict) -> dict:
    """One optimizer step over all examples, in mini-batches of similar length."""
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ordered = sorted(examples, key=lambda e: len(e.prompt) + len(e.reply))
    device_type = model.device.type
    total_loss = 0.0
    for start in range(0, len(ordered), run["mini_batch_size"]):
        chunk = ordered[start:start + run["mini_batch_size"]]
        input_ids, attention, reply_mask = (t.to(model.device) for t in encode_batch(chunk, tok, run["max_train_len"]))
        rewards = torch.tensor([e.reward for e in chunk], dtype=torch.float32, device=model.device)
        targets = reply_mask[:, 1:]
        base = model.get_base_model()
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=scaler.is_enabled()):
            hidden = base.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
            logits = base.lm_head(hidden[:, :-1][targets])  # reply positions only: a full-vocab tensor is huge
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids[:, 1:][targets].unsqueeze(-1)).squeeze(-1)
        rows = targets.nonzero()[:, 0]
        reply_log_prob = torch.zeros(len(chunk), device=model.device).index_add(0, rows, token_log_probs)
        loss = -(rewards * reply_log_prob / run["max_new_tokens"]).sum() / len(ordered)
        scaler.scale(loss).backward()
        total_loss += loss.item()
    scaler.unscale_(optimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, run["gradient_clipping"])
    scaler.step(optimizer)
    scaler.update()
    model.eval()
    return {"loss": total_loss, "grad_norm": float(grad_norm), "examples": len(ordered)}


def build_model(run: dict, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run["model"], padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(run["model"], dtype=dtype, device_map={"": device})
    return wrap_with_lora(base, run), tok


def wrap_with_lora(base, run: dict):
    from peft import LoraConfig, get_peft_model

    lora = run["lora"]
    config = LoraConfig(r=lora["lora_rank"], lora_alpha=lora["lora_alpha"], lora_dropout=lora["lora_dropout"],
                        target_modules=lora["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(base, config, autocast_adapter_dtype=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()
    return model


def _rng_state() -> dict:
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(run_dir: Path, step: int, model, optimizer, scaler, estimator, run: dict) -> Path:
    import torch

    path = run_dir / f"step_{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "baselines": estimator.baselines, "rng": _rng_state()}, path / "trainer_state.pt")
    (path / "run_config.json").write_text(json.dumps({**run, "step": step}, indent=2) + "\n", encoding="utf-8")
    (path / COMPLETE_MARKER).touch()
    return path


def load_checkpoint(path: Path, model, optimizer, scaler, estimator) -> int:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(path / "adapter_model.safetensors"))
    state = torch.load(path / "trainer_state.pt", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    estimator.baselines = state["baselines"]
    _restore_rng(state["rng"])
    return state["step"]


def train(run: dict, model, tok, run_dir: Path, *, generate: Generate | None = None, device: str = "cuda",
          max_seconds: float | None = None) -> int:
    """Train until run["steps"] or max_seconds; returns the last completed step."""
    import numpy as np
    import torch

    random.seed(run["seed"])
    np.random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    generate = generate or hf_generate_fn(model, tok, max_tokens=run["max_new_tokens"],
                                          temperature=run["temperature"], top_p=run["top_p"],
                                          batch_size=run["generation_batch_size"])
    estimator = make_estimator(run["estimator"], tau=run["rae_tau"])
    normalize = NormalizeRewardsByEnv()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=run["learning_rate"])
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    step = 0
    found = latest_checkpoint(run_dir)
    if found:
        step = load_checkpoint(found[1], model, optimizer, scaler, estimator)
        print(f"resumed from step {step}", flush=True)
    rewind_metrics(run_dir, step)

    started = time.time()
    while step < run["steps"] and (max_seconds is None or time.time() - started < max_seconds):
        t0 = time.time()
        matches = play(generate, run["opponent"], run["matches_per_step"], seed=run["seed"] * 1_000_000 + step * 1000,
                       keep_transcripts=run["matches_per_step"])
        t1 = time.time()
        examples = normalize(examples_from_matches(matches, estimator, run))
        update = reinforce_update(model, tok, examples, optimizer, scaler, run)
        step += 1
        row = {"step": step, **rollout_metrics(matches), **update, "rollout_seconds": t1 - t0,
               "update_seconds": time.time() - t1}
        with (run_dir / METRICS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {step}: invalid {row['invalid_move_loss_rate']:.2f} win {row['win_rate']:.2f} "
              f"format-err {row['format_error_rate']:.2f} loss {row['loss']:.4f} "
              f"{row['rollout_seconds']:.0f}s+{row['update_seconds']:.0f}s", flush=True)
        if step % run["save_every"] == 0 or step == run["steps"]:
            save_checkpoint(run_dir, step, model, optimizer, scaler, estimator, run)
    return step


def resolve_runs(config: dict) -> dict[str, dict]:
    return {r["run_id"]: {**config["defaults"], **r} for r in config["runs"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--run-id", help="run to train (see --list)")
    parser.add_argument("--list", action="store_true", help="print the run ids in the config")
    parser.add_argument("--steps", type=int, help="override the configured number of steps")
    parser.add_argument("--max-seconds", type=float, help="stop after the step that passes this time budget")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    runs = resolve_runs(json.loads(args.config.read_text(encoding="utf-8")))
    if args.list:
        print("\n".join(runs))
        return
    if args.run_id not in runs:
        raise SystemExit(f"--run-id must be one of: {', '.join(runs)}")
    run = {**runs[args.run_id], **({"steps": args.steps} if args.steps else {})}
    run_dir = args.checkpoints / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "versions.json").write_text(json.dumps(versions(), indent=2) + "\n", encoding="utf-8")
    model, tok = build_model(run)
    step = train(run, model, tok, run_dir, max_seconds=args.max_seconds)
    print(f"{run['run_id']}: stopped at step {step} of {run['steps']}", flush=True)


if __name__ == "__main__":
    main()
