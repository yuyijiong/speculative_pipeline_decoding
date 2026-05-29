"""
Utilities and demo for pipelined speculative decoding vs standard ``generate``.

Checkpoints must have ``config['version'] == 10`` (saved by ``train.py``).
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import time
from typing import Any

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

from pipeline_model import Qwen3SpeculativePipelineModel

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pipeline vs standard generation on Qwen3"
    )
    p.add_argument(
        "--spec_head_ckpt",
        type=_non_empty_ckpt_path,
        default="",
        help="Path to speculation_head.pt (state_dict + config); must not be empty.",
    )
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; greedy when <= 0 (default), stochastic when > 0.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed so sampling (temperature > 0) is reproducible across runs.",
    )
    p.add_argument(
        "--pipeline_first",
        action="store_true",
        help="Run pipeline before standard generate (default: standard first)",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify pipeline draft tokens against the base model (use --no-verify to disable).",
    )
    p.add_argument(
        "--use_deepest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deepest available per-position snapshots for speculation rows.",
    )
    p.add_argument(
        "--draft_top_k",
        type=int,
        default=4,
        help="Draft beam width for pipeline models with draft-tree ``generate`` (EAGLE3-style top-k chains); 1 matches single-chain decoding.",
    )
    return p.parse_args()

def _non_empty_ckpt_path(s: str) -> str:
    t = (s or "").strip()
    if not t:
        raise argparse.ArgumentTypeError(
            "--spec_head_ckpt must be a non-empty path to a speculation head checkpoint."
        )
    return t


def _read_spec_config(path: str) -> dict[str, Any]:
    """Load and validate the ``config`` dict inside speculation_head.pt."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected a dict checkpoint at {path!r}, got {type(ckpt).__name__}")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Checkpoint at {path!r} has no 'config' dict; cannot infer pipeline shape."
        )
    if "num_stages" not in cfg:
        raise ValueError(
            f"Checkpoint 'config' at {path!r} must include 'num_stages' (re-save with current training script)."
        )
    return cfg


def _resolve_base_model_path(cfg: dict[str, Any], ckpt_path: str) -> str:
    model_path = str(cfg.get("base_model_path", "")).strip()
    if not model_path:
        raise ValueError(
            f"Checkpoint 'config' at {ckpt_path!r} must include non-empty 'base_model_path'."
        )
    return model_path


def _infer_pipeline_kind(cfg: dict[str, Any]) -> int:
    """Require checkpoint version 10 (this release)."""
    ver = int(cfg.get("version", 0) or 0)
    if ver != 10:
        raise ValueError(
            f"Unsupported checkpoint version {ver}; this release expects config['version'] == 10."
        )
    return ver


def _draft_and_spec_init(cfg: dict[str, Any]) -> tuple[list[int] | None, list[int] | None]:
    raw = cfg.get("draft_token_ids")
    draft_token_ids: list[int] | None
    if raw is None:
        draft_token_ids = None
    else:
        draft_token_ids = [int(x) for x in raw]
    raw_spec_init = cfg.get("spec_init_from_base_layers")
    if raw_spec_init is not None:
        spec_init_from_base_layers = [int(x) for x in raw_spec_init]
    else:
        spec_init_from_base_layers = None
    return draft_token_ids, spec_init_from_base_layers


def _pipeline_init_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    draft_token_ids, spec_init_from_base_layers = _draft_and_spec_init(cfg)
    raw_shallow = cfg.get("shallow_hidden_layer_indices")
    shallow_hidden_layer_indices = (
        [[int(y) for y in x] for x in raw_shallow] if raw_shallow is not None else None
    )
    kw: dict[str, Any] = {
        "num_stages": int(cfg["num_stages"]),
        "num_spec_layers": int(cfg.get("num_spec_layers", 1)),
        "draft_token_ids": draft_token_ids,
        "spec_init_from_base_layers": spec_init_from_base_layers,
        "shallow_hidden_layer_indices": shallow_hidden_layer_indices,
    }
    if "trained_with_use_deepest" in cfg:
        kw["trained_with_use_deepest"] = bool(cfg["trained_with_use_deepest"])
    return kw


def build_pipeline_from_spec_ckpt(
    base_model: Qwen3ForCausalLM,
    ckpt_path: str,
    cfg: dict[str, Any],
    map_location: str,
) -> Qwen3SpeculativePipelineModel:
    """Construct ``Qwen3SpeculativePipelineModel`` and load speculation-head weights."""
    _infer_pipeline_kind(cfg)
    pipeline = Qwen3SpeculativePipelineModel(base_model=base_model, **_pipeline_init_kwargs(cfg))
    pipeline.load_speculation_head(ckpt_path, map_location=map_location)
    return pipeline





def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def theoretical_speedup_vs_standard(
    *,
    num_stages: int,
    num_new_tokens: int,
    pipeline_decode_steps: int,
) -> tuple[float, float, float, float]:
    """Toy timing model for decode only (prefill excluded).

    - Standard decoding: 1 second per new token -> throughput = 1 token/s.
    - Pipeline: each decode-loop iteration takes ``1 / num_stages`` seconds;
      rollbacks increase ``pipeline_decode_steps`` and thus wall time.

    Returns ``(t_std_sec, t_pipe_sec, v_std_tok_s, v_pipe_tok_s)`` and callers
    can derive improvement as ``(v_pipe / v_std - 1) * 100``.
    """
    n = max(int(num_stages), 1)
    nt = int(num_new_tokens)
    if nt <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    t_std = float(nt) * 1.0
    v_std = nt / t_std
    steps = max(int(pipeline_decode_steps), 1)
    t_pipe = steps / n
    v_pipe = nt / t_pipe
    return (t_std, t_pipe, v_std, v_pipe)


def run_standard_generate(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
) -> tuple[torch.LongTensor, float]:
    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": attention_mask.to(input_ids.device),
    }
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature

    _sync_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids, **kwargs)
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    return out[0], elapsed


def run_pipeline_generate(
    pipeline_model: Any,
    input_ids: torch.LongTensor,
    device: torch.device,
    *,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    verify: bool,
    use_deepest: bool = False,
    draft_top_k: int = 1,
) -> tuple[torch.LongTensor, float, list[bool], int, dict[str, float]]:
    _sync_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
            "temperature": temperature,
            "verify": verify,
        }
        if "use_deepest" in inspect.signature(pipeline_model.generate).parameters:
            gen_kw["use_deepest"] = use_deepest
        if "draft_top_k" in inspect.signature(pipeline_model.generate).parameters:
            gen_kw["draft_top_k"] = int(draft_top_k)
        elif int(draft_top_k) != 1:
            raise ValueError(
                f"{type(pipeline_model).__name__}.generate has no draft_top_k; "
                "use a pipeline model whose generate() supports draft_top_k."
            )
        new_ids, token_acceptance, decode_loop_steps = pipeline_model.generate(input_ids, **gen_kw)
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    full = torch.cat(
        [input_ids[0], torch.tensor(new_ids, device=device, dtype=torch.long)],
        dim=0,
    )
    raw = getattr(pipeline_model, "_last_generate_timing", None)
    timing: dict[str, float]
    if isinstance(raw, dict):
        timing = {k: float(v) for k, v in raw.items()}
    else:
        timing = {}
    return full, elapsed, token_acceptance, decode_loop_steps, timing


if __name__ == "__main__":
    args = parse_args()
    _set_global_seed(int(args.seed))
    dtype = torch.bfloat16
    ckpt_path = args.spec_head_ckpt
    spec_cfg = _read_spec_config(ckpt_path)
    base_model_path = _resolve_base_model_path(spec_cfg, ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map={"":0},
        trust_remote_code=True,
    )

    _infer_pipeline_kind(spec_cfg)

    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = build_pipeline_from_spec_ckpt(
        base_model,
        ckpt_path,
        spec_cfg,
        map_location=map_loc,
    )
    num_stages = int(pipeline.num_stages)

    device = next(pipeline.base_model.parameters()).device

    prompts=["What is 24*36? Answer briefly.",
             "What is the capital of China?",
            "Introduce LLM."]
    for prompt in prompts:
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        if attn is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attn.to(device)

        greedy = float(args.temperature) <= 0.0

        if args.pipeline_first:
            seq_pipe, t_pipe, acceptance, pipe_steps, _pipe_timing = run_pipeline_generate(
                pipeline,
                input_ids,
                device,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                temperature=args.temperature,
                verify=args.verify,
                use_deepest=args.use_deepest,
                draft_top_k=args.draft_top_k,
            )
            seq_std, t_std = run_standard_generate(
                base_model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                temperature=args.temperature,
            )
        else:
            seq_std, t_std = run_standard_generate(
                base_model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                temperature=args.temperature,
            )
            seq_pipe, t_pipe, acceptance, pipe_steps, _pipe_timing = run_pipeline_generate(
                pipeline,
                input_ids,
                device,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                temperature=args.temperature,
                verify=args.verify,
                use_deepest=args.use_deepest,
                draft_top_k=args.draft_top_k,
            )

        text_std = tokenizer.decode(seq_std, skip_special_tokens=False)
        text_pipe = tokenizer.decode(seq_pipe, skip_special_tokens=False)

        match = torch.equal(seq_std.cpu(), seq_pipe.cpu())
        n_std = seq_std.numel() - input_ids.shape[1]
        n_pipe = seq_pipe.numel() - input_ids.shape[1]

        print("\n\n\n--- Prompt ---")
        print(prompt.rstrip())
        print("--- Timing ---")
        print(f"  Standard HuggingFace generate: {t_std * 1000:.2f} ms")
        print(f"  Pipeline generate:             {t_pipe * 1000:.2f} ms")
        n_accepted = sum(acceptance)
        accept_rate = (n_pipe / pipe_steps) if pipe_steps > 0 else 0.0

        # print("--- Sequence comparison ---")
        # print(f"  New tokens (standard): {n_std}")
        # print(f"  New tokens (pipeline): {n_pipe}")
        # if greedy:
        #     print(f"  Full token ids match: {match}")
        # else:
        #     print("  Full token ids match: (not expected under sampling)")
        # print()
        print("--- Speculation acceptance ---",f"draft_topk={args.draft_top_k}")
        print(f"  Per-flag accepted: {n_accepted} / {len(acceptance)}")
        print(f"  Generated tokens / decode steps: {n_pipe} / {pipe_steps}")
        print(f"  Acceptance rate (tokens/step): {accept_rate:.3f}")
        t_th_std, t_th_pipe, v_th_std, v_th_pipe = theoretical_speedup_vs_standard(
            num_stages=num_stages,
            num_new_tokens=n_pipe,
            pipeline_decode_steps=pipe_steps,
        )
        th_pct = (v_th_pipe / v_th_std - 1.0) * 100.0 if v_th_std > 0 else 0.0
        print("--- Theoretical decode speed (toy model, prefill excluded) ---")
        print(
            f"  Standard: {v_th_std:.3f} tok/s "
            f"(T = {t_th_std:.1f}s for {n_pipe} tokens at 1s/token)"
        )
        print(
            f"  Pipeline: {v_th_pipe:.3f} tok/s "
            f"(T = {t_th_pipe:.4f}s = {pipe_steps} steps × 1/{num_stages}s/step)"
        )
        print(f"  Throughput gain vs standard: {th_pct:+.1f}%")
        print()
        print("--- Standard generate ---")
        print(text_std)
        print()
        print("--- Pipeline generate ---")
        print(text_pipe)

