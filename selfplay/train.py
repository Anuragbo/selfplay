"""Launch one cell of the training grid with UnstableBaselines (unstable-rl 0.2.2).

  python -m selfplay.train --config configs/grid.json --list
  python -m selfplay.train --config configs/grid.json --run-id selfplay_qwen3-1.7b_rae_s0 --dry-run
  python -m selfplay.train --config configs/grid.json --run-id selfplay_qwen3-1.7b_rae_s0

The grid is models x estimators x seeds for mirror self-play (12 runs), plus scripted-opponent controls.
This mirrors unstable.runtime.build, with four changes that build() does not allow:
  1. the final-reward transform is our estimator (selfplay.advantages: RAE or a shared baseline);
  2. Ray workers register SpiralKuhnPoker-v1 and resolve "scripted:<policy>" opponents to selfplay.opponents
     (build() only knows OpenRouter models as fixed opponents), via a worker setup hook;
  3. Python, NumPy and PyTorch are seeded from the run's seed;
  4. checkpoints are exported to checkpoints/{run_id}/step_{N}/ with a run_config.json (the checkpoint contract).
This launcher needs two GPUs with bfloat16 support (UnstableBaselines' learner autocasts to bfloat16); it does not
run on a single T4.
"""

import argparse
import json
import shutil
from itertools import product
from pathlib import Path

from selfplay.envs import SPIRAL_KUHN_ID
from selfplay.opponents import POLICIES, SCRIPTED_PREFIX, is_scripted, make_scripted


def expand_grid(config: dict) -> dict[str, dict]:
    """All runs in the grid, keyed by run_id."""
    runs = {}
    defaults = config["defaults"]
    for (size, model), estimator, seed in product(config["models"].items(), config["estimators"], config["seeds"]):
        run_id = f"selfplay_{size}_{estimator}_s{seed}"
        runs[run_id] = {**defaults, "run_id": run_id, "model": model, "estimator": estimator, "seed": seed,
                        "opponent": "self"}
    controls = config.get("controls")
    if controls:
        if controls["opponent"] not in POLICIES:
            raise ValueError(f"unknown control opponent {controls['opponent']!r}")
        for (size, model), estimator, seed in product(config["models"].items(), controls["estimators"],
                                                      controls["seeds"]):
            run_id = f"scripted-{controls['opponent']}_{size}_{estimator}_s{seed}"
            runs[run_id] = {**defaults, "run_id": run_id, "model": model, "estimator": estimator, "seed": seed,
                            "opponent": controls["opponent"]}
    return runs


def install_worker_hooks() -> None:
    """Runs in every Ray worker: register our environment and let fixed-opponent names be scripted policies."""
    import textarena as ta

    from selfplay.envs import register_envs

    register_envs()
    if getattr(ta.agents, "_selfplay_patched", False):
        return
    openrouter = ta.agents.OpenRouterAgent

    def agent_factory(model_name: str, *args, **kwargs):
        return make_scripted(model_name) if is_scripted(model_name) else openrouter(model_name, *args, **kwargs)

    ta.agents.OpenRouterAgent = agent_factory
    ta.agents._selfplay_patched = True


def reward_pipeline(run: dict):
    """Final-reward estimator, step shaping and batch normalization; the same for both arms except the estimator."""
    import unstable.reward_transformations as retra

    from selfplay.advantages import NormalizeRewardsByEnv, make_estimator

    final = retra.ComposeFinalRewardTransforms([make_estimator(run["estimator"], tau=run["rae_tau"])])
    step = retra.ComposeStepRewardTransforms([retra.RewardForFormat(run["format_reward"]),
                                              retra.PenaltyForInvalidMove(run["valid_move_reward"],
                                                                          run["invalid_move_penalty"])])
    sampling = retra.ComposeSamplingRewardTransforms([NormalizeRewardsByEnv()])  # NaN-safe replacement
    return final, step, sampling


def launch(run: dict, repo: Path) -> Path:
    """Build the UnstableBaselines actors for one run and train. Returns UnstableBaselines' output directory."""
    import random

    import numpy as np
    import ray
    import torch
    import unstable

    random.seed(run["seed"])
    np.random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    ray.init(namespace="unstable", runtime_env={"worker_process_setup_hook": "selfplay.train.install_worker_hooks",
                                                "env_vars": {"PYTHONPATH": str(repo)}})
    install_worker_hooks()

    self_play = run["opponent"] == "self"
    train_env = unstable.TrainEnvSpec(env_id=run["game"], num_players=2, num_actors=2 if self_play else 1,
                                      prompt_template=run["prompt_template"])
    eval_env = unstable.EvalEnvSpec(env_id=run["game"], num_players=2, prompt_template=run["prompt_template"],
                                    fixed_opponent=SCRIPTED_PREFIX + run["eval_opponent"])
    env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(train_env_specs=[train_env],
                                                                         eval_env_specs=[eval_env])
    tracker = unstable.Tracker.options(name="Tracker").remote(run_name=run["run_id"], wandb_project=None)
    registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    ray.get(registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
    samplers = unstable.samplers.model_samplers
    if self_play:
        model_sampler = samplers.BaseModelSampler(model_registry=registry)
    else:
        ray.get(registry.add_fixed.remote(name=SCRIPTED_PREFIX + run["opponent"]))
        model_sampler = samplers.FixedOpponentModelSampler(model_registry=registry)
    scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
        model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

    final, step, sampling = reward_pipeline(run)
    buffer = unstable.StepBuffer.options(name="Buffer").remote(
        max_buffer_size=run["batch_size"] * 2, tracker=tracker, final_reward_transformation=final,
        step_reward_transformation=step, sampling_reward_transformation=sampling)
    vllm_cfg = {"model_name": run["model"], "temperature": run["temperature"], "top_p": run["top_p"],
                "max_tokens": run["max_generation_len"], "max_parallel_seq": 128, "max_loras": 8,
                "lora_config": run["lora"], "max_model_len": 8192}
    collector = unstable.Collector.options(name="Collector").remote(vllm_config=vllm_cfg, tracker=tracker,
                                                                    buffer=buffer, game_scheduler=scheduler)
    learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
        model_name=run["model"], lora_cfg=run["lora"], batch_size=run["batch_size"],
        mini_batch_size=run["mini_batch_size"], learning_rate=run["learning_rate"],
        grad_clip=run["gradient_clipping"], buffer=buffer, tracker=tracker, model_registry=registry,
        activation_checkpointing=True, gradient_checkpointing=True, use_trainer_cache=False)
    ray.get(learner.initialize_algorithm.remote(max_train_len=run["max_train_len"],
                                                max_generation_len=run["max_generation_len"]))
    out_dir = Path(ray.get(tracker.get_checkpoints_dir.remote())).parent
    try:
        collector.collect.remote(num_train_workers=run["num_collection_workers"],
                                 num_eval_workers=run["num_eval_workers"])
        ray.get(learner.train.remote(run["learning_steps"]))
    finally:
        ray.kill(collector, no_restart=True)
        ray.shutdown()
    return out_dir


def export_checkpoints(ub_checkpoints: Path, run: dict, dest_root: Path) -> list[Path]:
    """Copy UnstableBaselines' iteration-{i} adapters to {dest_root}/{run_id}/step_{N}/ every save_every steps.

    UnstableBaselines saves iteration-i after its i-th update, so the adapter after N updates is iteration-(N-1).
    The final step is always exported.
    """
    steps = sorted(int(p.name.split("-")[1]) + 1 for p in ub_checkpoints.glob("iteration-*"))
    keep = [n for n in steps if n % run["save_every"] == 0 or n == steps[-1]] if steps else []
    exported = []
    for n in keep:
        dest = dest_root / run["run_id"] / f"step_{n}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ub_checkpoints / f"iteration-{n - 1}", dest)
        (dest / "run_config.json").write_text(json.dumps({**run, "step": n}, indent=2) + "\n", encoding="utf-8")
        exported.append(dest)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/grid.json"))
    parser.add_argument("--run-id", help="grid cell to run (see --list)")
    parser.add_argument("--list", action="store_true", help="print the run ids in the grid")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved run config and exit")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    runs = expand_grid(json.loads(args.config.read_text(encoding="utf-8")))
    if args.list:
        print("\n".join(runs))
        return
    if args.run_id not in runs:
        raise SystemExit(f"--run-id must be one of the {len(runs)} grid cells (see --list)")
    run = runs[args.run_id]
    if run["game"] != SPIRAL_KUHN_ID:
        raise SystemExit(f"only {SPIRAL_KUHN_ID} is registered")
    if args.dry_run:
        print(json.dumps(run, indent=2))
        return
    out_dir = launch(run, repo=Path(__file__).resolve().parents[1])
    for path in export_checkpoints(out_dir / "checkpoints", run, args.checkpoints):
        print("exported", path)


if __name__ == "__main__":
    main()
