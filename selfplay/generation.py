"""Text generation backends shared by the evaluation and game-play scripts.

generate(prompts, n) returns, for each prompt, n tuples of (completion text, completion tokens, finish reason).
vLLM is the default; the transformers backend is a slower fallback for GPUs vLLM does not support.
"""

import argparse
import platform
from collections.abc import Callable

Generate = Callable[[list[str], int], list[list[tuple[str, int, str]]]]


def add_generation_args(parser: argparse.ArgumentParser, max_tokens: int = 4096) -> None:
    g = parser.add_argument_group("generation")
    g.add_argument("--model", required=True, help="Hugging Face id or local path of the base model")
    g.add_argument("--lora", help="LoRA adapter directory (checkpoints/{run_id}/step_{N}/)")
    g.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=max_tokens)
    g.add_argument("--max-model-len", type=int, default=None,
                   help="context length (default: --max-tokens + 8192, room for long game histories)")
    g.add_argument("--dtype", default="float16", help="float16 on T4 (no bfloat16 support)")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    g.add_argument("--max-lora-rank", type=int, default=32)
    g.add_argument("--hf-batch-size", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)


def make_generator(args) -> Generate:
    if args.max_model_len is None:
        args.max_model_len = args.max_tokens + 8192  # game histories grow during a match
    return vllm_generator(args) if args.backend == "vllm" else hf_generator(args)


def vllm_generator(args) -> Generate:
    from vllm import LLM, SamplingParams

    kwargs = {"enable_lora": True, "max_lora_rank": args.max_lora_rank} if args.lora else {}
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, **kwargs)
    lora = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("adapter", 1, args.lora)

    def generate(prompts: list[str], n: int):
        params = SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                seed=args.seed)
        outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
        return [[(o.text, len(o.token_ids), o.finish_reason or "") for o in out.outputs] for out in outputs]

    return generate


def hf_generator(args) -> Generate:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto")
    if args.lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    torch.manual_seed(args.seed)
    stop_ids = {tok.pad_token_id, tok.eos_token_id}

    def generate(prompts: list[str], n: int):
        results = []
        for i in range(0, len(prompts), args.hf_batch_size):
            batch = tok(prompts[i:i + args.hf_batch_size], return_tensors="pt", padding=True).to(model.device)
            sample = args.temperature > 0
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=args.max_tokens, do_sample=sample, num_return_sequences=n,
                                     temperature=args.temperature if sample else None,
                                     top_p=args.top_p if sample else None, pad_token_id=tok.pad_token_id)
            new = out[:, batch["input_ids"].shape[1]:].tolist()
            for j in range(0, len(new), n):
                group = []
                for seq in new[j:j + n]:
                    ids = next((seq[:k] for k, t in enumerate(seq) if t in stop_ids), seq)
                    finish = "length" if len(ids) >= args.max_tokens else "stop"
                    group.append((tok.decode(ids, skip_special_tokens=True), len(ids), finish))
                results.append(group)
        return results

    return generate


def versions() -> dict:
    found = {"python": platform.python_version()}
    for pkg in ("torch", "vllm", "transformers", "math_verify", "peft", "textarena"):
        try:
            found[pkg] = __import__(pkg).__version__
        except Exception:
            found[pkg] = None
    try:
        import torch

        found["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        found["gpu"] = None
    return found
