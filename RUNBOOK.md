# Runbook: running the next steps on Colab

How to pick up the project and run the next experiments on a free Colab T4. Read `README.md` first for what the
project is. Every number in the reports must come from a file in `results/`, so each task below ends by saving a
result file.

## 0. One-time setup (each person)

| Step | What to do |
|---|---|
| Repository | Make sure you can open https://github.com/Anuragbo/selfplay (it is private). |
| GitHub token | GitHub → Settings → Developer settings → Fine-grained tokens → read-only access to `Anuragbo/selfplay`. In Colab, open the 🔑 Secrets panel, add it as `GH_TOKEN`, and turn on notebook access. |
| Shared Drive folder | Open the shared `CMPE260_selfplay` folder in Google Drive → right-click → **Organize → Add shortcut → My Drive**. This makes the path `/content/drive/MyDrive/CMPE260_selfplay` work in Colab. |
| Local copy (for committing results) | `git clone`, then `python -m venv .venv` and `pip install -r requirements-dev.txt` (see README). |

**Rule: one person per run.** Two people training the same `run_id` at the same time will overwrite each other's
checkpoints. Say in the team chat which run you are starting.

## 1. Where things are on Drive

```
CMPE260_selfplay/checkin3/pilot/
  checkpoints/{run_id}/        LoRA adapter per saved step (step_2, step_4, ...), metrics.jsonl, versions.json
  outputs/                     raw evaluation and game-play outputs
  results/                     packed result files, ready to copy into the repo's results/
  logs/                        training and evaluation logs
```

Existing checkpoints: `pilot_qwen3-1.7b_rae_s0` (complete, step 30) and `pilot_qwen3-1.7b_shared_s0` (stopped at
step 20; resume it, do not restart it).

## 2. Colab cells

Open a new notebook, set **Runtime → Change runtime type → T4 GPU**, and paste these cells.

**Cell A: setup (run at the start of every session; safe to re-run)**

```python
import os, json, pathlib, subprocess, sys
from google.colab import drive, userdata
drive.mount("/content/drive")
P = pathlib.Path("/content/drive/MyDrive/CMPE260_selfplay/checkin3/pilot")
for d in ("checkpoints", "outputs", "results", "logs"):
    (P / d).mkdir(parents=True, exist_ok=True)
REPO = pathlib.Path("/content/cmpe260-selfplay")
if not REPO.exists():
    url = f"https://{userdata.get('GH_TOKEN')}@github.com/Anuragbo/selfplay.git"
    subprocess.run(["git", "clone", "-q", url, str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "pull", "-q"], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "NO GPU")
if not pathlib.Path(".deps_ok").exists():
    subprocess.run("pip install -q -r requirements-gpu.txt", shell=True, check=True)
    pathlib.Path(".deps_ok").touch()
if not pathlib.Path("data/raw/aime25.jsonl").exists():
    subprocess.run("bash data/fetch_benchmarks.sh", shell=True, check=True)

def start(name, cmd):
    """Run a long job in the background; its log goes to Drive. The cell returns at once."""
    log = P / "logs" / f"{name}.log"
    subprocess.Popen(f"nohup {cmd} >> {log} 2>&1 &", shell=True)
    print("started; log:", log)
```

**Cell B: status (run any time)**

```python
print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader; ps aux | grep -c '[s]elfplay'",
                     shell=True, capture_output=True, text=True).stdout)
for log in sorted((P / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)[-2:]:
    print("==", log.name); print("".join(open(log).readlines()[-8:]))
```

If the session dies, reconnect, run Cell A, and start the same command again. Training resumes from the newest
checkpoint; evaluation resumes from the last finished chunk.

## 3. Tasks

These match Section 2 ("Next steps") of the planned-vs-executed document. Times are for one T4.

| # | Task | Command (paste into a cell after Cell A) | Time |
|---|---|---|---|
| 2 | 4B base model vs. never fold | `start("kuhn-base-qwen3-4b", f"python -m selfplay.play_kuhn --model Qwen/Qwen3-4B-Base --opponents never_fold --matches 100 --temperature 0.6 --max-tokens 4096 --run-id kuhn-base-qwen3-4b --out-dir {P}/outputs/kuhn-base-qwen3-4b")` | ≈30 min |
| 3a | Resume the shared-baseline run (20 → 30) | `start("pilot_qwen3-1.7b_shared_s0", f"python -m selfplay.train_single --config configs/pilot.json --run-id pilot_qwen3-1.7b_shared_s0 --checkpoints {P}/checkpoints")` | ≈45 min |
| 3b, 4, 10–12 | MATH-500 evaluation | See "Evaluation commands" below | ≈20 min each |
| 5, 7 | New scripted runs (seeds, 4B) | Same as 3a with another `--run-id` from `python -m selfplay.train_single --list` | 2 h (1.7B), ≈4.7 h (4B, untested) |
| 6 | Longer scripted runs | Add a run to `configs/pilot.json` with a new `run_id` (e.g. `pilot60_qwen3-1.7b_rae_s0`) and `"steps": 60`, commit, then as 3a | ≈4 min per step |
| 8–9 | Self-play | **Not runnable yet.** `selfplay/train_single.py` only plays a scripted opponent; self-play needs code first. | — |
| 13 | Elo, trace length, transcript audit | Not built yet | — |

To stop a long run before the daily GPU quota runs out, add `--max-seconds 7200`: it stops after the step that
passes 2 hours. Checkpoints are saved every 2 steps, so at most one step is redone when the run resumes.

**Evaluation commands** (one at a time; vLLM uses most of the GPU)

```python
# Trained adapter (task 3b: replace the run id and step)
RUN, STEP = "pilot_qwen3-1.7b_shared_s0", 30
start(f"eval-{RUN}-step{STEP}", f"python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --lora {P}/checkpoints/{RUN}/step_{STEP} "
      f"--benchmarks math500 --temperature 0 --max-tokens 4096 --run-id eval-{RUN}-step{STEP} --step {STEP} "
      f"--game SpiralKuhnPoker-v1 --estimator shared --opponent never_fold --out-dir {P}/outputs/eval-{RUN}-step{STEP}")

# Task 4: base 1.7B again on today's software (new run id, so the old file is kept)
start("eval-base-qwen3-1.7b-v2", f"python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --benchmarks math500 "
      f"--temperature 0 --max-tokens 4096 --run-id eval-base-qwen3-1.7b-v2 --out-dir {P}/outputs/eval-base-qwen3-1.7b-v2")

# Task 10: two sampled answers per problem, for the detectable effect (then see notebooks/09, last cell)
start("eval-base-qwen3-1.7b-t0.6x2", f"python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --benchmarks math500 "
      f"--samples 2 --temperature 0.6 --top-p 0.95 --max-tokens 4096 --run-id eval-base-qwen3-1.7b-t0.6x2 "
      f"--out-dir {P}/outputs/eval-base-qwen3-1.7b-t0.6x2")

# Task 12: SPIRAL's released 4B checkpoint (should score near its published numbers)
start("eval-spiral-qwen3-4b", f"python -m selfplay.evaluate --model Spiral-RL/Spiral-Qwen3-4B --benchmarks math500 "
      f"--temperature 0 --max-tokens 4096 --run-id eval-spiral-qwen3-4b --out-dir {P}/outputs/eval-spiral-qwen3-4b")
```

For task 11 (full suite), use `--benchmarks math500 olympiadbench minerva aime24 aime25 amc23`. If vLLM fails to
install or start, add `--backend hf` (slower).

## 4. Pack a finished job into a result file

Run in Colab after a job finishes. It writes `{P}/results/<name>.json`, in the same format as the files in the repo.

```python
from selfplay.evaluate import _read_jsonl, summarize

def save(name, obj):
    path = P / "results" / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print("wrote", path)

def pack_train(run_id, label):            # e.g. pack_train("pilot_qwen3-1.7b_shared_s0", "Shared baseline, seed 0")
    d = P / "checkpoints" / run_id
    steps = sorted(int(s.name.split("_")[1]) for s in d.glob("step_*") if (s / "complete").exists())
    rows = [json.loads(line) for line in open(d / "metrics.jsonl") if line.strip()]
    config = json.load(open(d / f"step_{steps[-1]}" / "run_config.json"))
    save(f"train-{run_id}", {"run_id": run_id, "label": label, "config": config,
                             "metrics": [r for r in rows if r["step"] <= steps[-1]]})

def pack_eval(run_id):                    # e.g. pack_eval("eval-pilot_qwen3-1.7b_shared_s0-step30")
    d = P / "outputs" / run_id
    config = json.load(open(d / "run_config.json"))
    _, bits = summarize(_read_jsonl(d / "per_problem.jsonl"), {})
    results = [json.loads(line) for line in open(d / "results.jsonl")]
    save(run_id, {"run_id": run_id, "results": results, "correct": bits, "versions": config["versions"]})

def pack_kuhn(run_id):                    # e.g. pack_kuhn("kuhn-base-qwen3-4b")
    d = P / "outputs" / run_id
    config = json.load(open(d / "run_config.json"))
    results = [json.loads(line) for line in open(d / "results.jsonl")]
    save(run_id, {"run_id": run_id, "results": results, "versions": config["versions"]})
```

Name rules (the statistics script depends on them): training → `train-<run_id>.json`; evaluation of a trained
adapter → `eval-pilot_<run>-step<N>.json`; base models → `eval-base-*.json`; game play → `kuhn-base-*.json`.

## 5. Bring results into the repository

1. Download the new file from `CMPE260_selfplay/checkin3/pilot/results/` into the repo's `results/` folder.
2. Recompute statistics and figures:
   ```bash
   python explore/checkin3_stats.py results results/checkin3_stats.json
   python explore/make_figures.py results figures
   git checkout -- figures/fig1_detectable_effect.png figures/fig2_kuhn_matchups.png figures/fig3_math500_base.png figures/fig4_kuhn_base_turns.png
   ```
   The `git checkout` line undoes byte-level changes to figures whose data did not change.
3. Check: `pytest` and `ruff check .` must pass.
4. Commit only the result file, `results/checkin3_stats.json` and any changed figure, with a message that names
   the run.

## 6. If something goes wrong

| Problem | What to do |
|---|---|
| "Cannot connect to GPU backend" / "continue without GPU" | The daily free GPU quota is used up. Wait (usually about a day) or use another account; runs resume from Drive. |
| Session disconnected mid-run | Reconnect, run Cell A, start the same command. Training resumes from the newest complete checkpoint. |
| Out of memory on 4B | Training: check that the run uses `mini_batch_size` 1 (already set for 4B runs). Evaluation: add `--gpu-memory-utilization 0.85`. |
| vLLM import or start errors | Add `--backend hf`. |
| Drive path not found | The shortcut from step 0 is missing, or Drive is not mounted (run Cell A and allow access). |
