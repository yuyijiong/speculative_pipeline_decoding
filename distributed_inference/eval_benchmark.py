"""
Evaluate multi-GPU pipeline-parallel decoding on benchmark prompts (real wall-clock speed).

Uses the same datasets and generation settings as ``eval.py``:
all rows from ``eval_data/{mt_bench,humaneval,gsm8k}/question.jsonl`` (first turn only).

Launch (``world_size`` must equal ``num_stages + 1``, or ``num_stages`` with ``--merge_last_stage``)::

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \\
    distributed_inference/eval_benchmark.py \\
    --spec_head_ckpt /path/to/speculation_head_final.pt \\
    --rank_gpus 0,1,2,3,4 --async_comm

Outputs under ``--output_dir``::

    raw/mp_pipeline_eval__<checkpoint_tag>__nt<total>__per_sample.jsonl
    summary/mp_pipeline_eval__<checkpoint_tag>__nt<total>__summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import (  # noqa: E402
    DATASET_CONFIG,
    UnifiedItem,
    _encode_prompt,
    _normalize_draft_top_k_list,
    _normalize_spec_head_ckpt_list,
    _normalize_temperature_list,
    _per_sample_file_stem,
    _set_torch_rng_for_eval_sample,
    ckpt_to_filename_tag,
    unify_from_jsonl,
)
from distributed_inference.comm import CtrlOpcode, PipelineP2P, make_ctrl_tensor
from distributed_inference.device import (
    PhaseTimeout,
    init_dist_rank_device,
    parse_rank_gpu_ids,
)
from distributed_inference.run_generate import _resolve_base_model_path, _run_one_session
from distributed_inference.loader import (
    load_prefill_rank0_bundle,
    load_spec_checkpoint_config,
    load_stage_rank_bundle,
)
from distributed_inference.prefill import broadcast_input_ids
from distributed_inference.stage_worker import StageWorker
from distributed_inference.topology import expected_world_size


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("float32", "fp32"):
        return torch.float32
    raise ValueError(
        f"Unsupported dtype {name!r}; use bfloat16, float16, or float32."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU distributed pipeline eval: wall-clock speed and acceptance"
    )
    p.add_argument(
        "--spec_head_ckpt",
        nargs="+",
        type=str,
        required=True,
        help="One or more speculation_head checkpoints (num_stages must match torchrun world_size).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output_distributed",
    )
    p.add_argument("--data_dir", type=str, default="eval_data")
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[0.0, 1.0],
    )
    p.add_argument(
        "--draft_top_k",
        nargs="+",
        type=int,
        default=[1],
        help="Recorded for parity with eval.py (not used by distributed decode)",
    )
    p.add_argument(
        "--use_deepest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top_k", type=int, default=50, help="Sampling top-k when temperature > 0")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--rank_gpus",
        type=str,
        default="",
        help="Comma-separated physical GPU ids, length = world_size (num_stages+1, or num_stages with --merge_last_stage)",
    )
    p.add_argument("--prefill_timeout_sec", type=float, default=120.0)
    p.add_argument("--decode_timeout_sec", type=float, default=60.0)
    p.add_argument("--warmup_iters", type=int, default=1)
    p.add_argument("--warmup_new_tokens", type=int, default=4)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model weights dtype for base LM and speculation module (bfloat16, float16, float32)",
    )
    p.add_argument("--async_comm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--merge_last_stage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Co-locate the last pipeline stage on rank 0 with speculation (CUDA streams). "
            "world_size must equal num_stages."
        ),
    )
    p.add_argument(
        "--sync_mode",
        type=str,
        default="barrier",
        choices=("barrier", "comm_only"),
    )
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def _load_all_items(data_dir: Path) -> list[UnifiedItem]:
    all_items: list[UnifiedItem] = []
    for _ds, rel in DATASET_CONFIG:
        for it in unify_from_jsonl(rel, data_dir):
            all_items.append(it)
    return all_items


def _decode_timing_from_session(session: Any) -> dict[str, float]:
    bd: dict[str, float] = dict(getattr(session, "decode_breakdown", {}) or {})
    wall_sec = float(session.decode_sec)
    pure_comm_sec = float(bd.get("pure_comm_sec", bd.get("comm_sec", 0.0)))
    pipeline_wait_sec = float(bd.get("pipeline_wait_sec", 0.0))
    cycle_max_stage_sec = float(bd.get("decode_cycle_max_stage_sec", 0.0))
    verify_sec = float(bd.get("verify_sec", 0.0))
    if "decode_compute_sec" in bd:
        compute_sec = float(bd["decode_compute_sec"])
    elif cycle_max_stage_sec > 0.0:
        compute_sec = float(cycle_max_stage_sec + verify_sec)
    else:
        compute_sec = float(
            max(wall_sec - pure_comm_sec - pipeline_wait_sec, 0.0)
        )
    return {
        "wall_sec": wall_sec,
        "pure_comm_sec": pure_comm_sec,
        "pipeline_wait_sec": pipeline_wait_sec,
        "compute_sec": compute_sec,
        "cycle_max_stage_sec": cycle_max_stage_sec,
        "rank0_gpu_sec": float(bd.get("rank0_gpu_sec", 0.0)),
        "spec_forward_sec": float(bd.get("spec_forward_sec", 0.0)),
        "verify_sec": verify_sec,
        "driver_update_sec": float(bd.get("driver_update_sec", 0.0)),
        "unaccounted_sec": float(bd.get("unaccounted_sec", 0.0)),
    }


def _sample_row(
    *,
    idx: int,
    it: UnifiedItem,
    session: Any,
    out_ids: list[int],
    accept: list[bool],
    prompt_len: int,
    num_stages: int,
    gen_text: str,
    temperature: float,
    draft_top_k: int,
    use_deepest: bool,
) -> dict[str, Any]:
    new_tokens = int(len(out_ids))
    steps = int(session.decode_steps)
    n_flags = len(accept)
    n_acc = int(sum(1 for x in accept if x))
    acc_rate = (float(new_tokens) / float(steps)) if steps else 0.0
    equiv = float(num_stages) * acc_rate
    prefill_sec = float(session.prefill_sec)
    decode_timing = _decode_timing_from_session(session)
    decode_sec = decode_timing["wall_sec"]
    decode_pure_comm_sec = decode_timing["pure_comm_sec"]
    decode_pipeline_wait_sec = decode_timing["pipeline_wait_sec"]
    decode_compute_sec = decode_timing["compute_sec"]
    prefill_tok_s = float(prompt_len) / prefill_sec if prefill_sec > 0 else 0.0
    decode_tok_s = float(new_tokens) / decode_sec if decode_sec > 0 else 0.0
    decode_compute_tok_s = (
        float(new_tokens) / decode_compute_sec if decode_compute_sec > 0 else 0.0
    )
    return {
        "index": int(idx),
        "dataset": it.dataset,
        "question_id": it.question_id,
        "question": it.question,
        "generated": gen_text,
        "prompt_len": int(prompt_len),
        "new_tokens": new_tokens,
        "decode_loop_steps": steps,
        "acceptance_rate": acc_rate,
        "equivalent_accept_length": equiv,
        "n_accepted": n_acc,
        "n_acceptance_flags": n_flags,
        "pipeline_prefill_wall_sec": prefill_sec,
        "pipeline_post_prefill_setup_sec": float(session.post_prefill_setup_sec),
        "pipeline_decode_wall_sec": decode_sec,
        "pipeline_decode_pure_comm_sec": decode_pure_comm_sec,
        "pipeline_decode_pipeline_wait_sec": decode_pipeline_wait_sec,
        "pipeline_decode_compute_sec": decode_compute_sec,
        "pipeline_decode_cycle_max_stage_sec": decode_timing["cycle_max_stage_sec"],
        "pipeline_decode_rank0_gpu_sec": decode_timing["rank0_gpu_sec"],
        "pipeline_decode_spec_forward_sec": decode_timing["spec_forward_sec"],
        "pipeline_decode_verify_sec": decode_timing["verify_sec"],
        "pipeline_decode_driver_update_sec": decode_timing["driver_update_sec"],
        "pipeline_decode_unaccounted_sec": decode_timing["unaccounted_sec"],
        "pipeline_prefill_tok_s": prefill_tok_s,
        "pipeline_decode_tok_s": decode_tok_s,
        "pipeline_decode_compute_tok_s": decode_compute_tok_s,
        "num_stages": int(num_stages),
        "temperature": float(temperature),
        "greedy": bool(float(temperature) == 0.0),
        "use_deepest": bool(use_deepest),
        "draft_top_k": int(draft_top_k),
    }


def _aggregate_mp_pipeline(
    per_sample: list[dict[str, Any]],
    ds: Optional[str],
) -> dict[str, float]:
    sub = [r for r in per_sample if ds is None or r.get("dataset") == ds]
    if not sub:
        return {
            "mean_prefill_wall_sec": 0.0,
            "mean_decode_wall_sec": 0.0,
            "mean_decode_pure_comm_sec": 0.0,
            "mean_decode_pipeline_wait_sec": 0.0,
            "mean_decode_compute_sec": 0.0,
            "mean_acceptance_rate": 0.0,
            "mean_equivalent_accept_length": 0.0,
            "pooled_acceptance_rate": 0.0,
            "pooled_equivalent_accept_length": 0.0,
            "aggregate_prefill_tok_s": 0.0,
            "aggregate_decode_tok_s": 0.0,
            "aggregate_decode_compute_tok_s": 0.0,
            "total_new_tokens": 0.0,
            "total_decode_loop_steps": 0.0,
            "total_prompt_tokens": 0.0,
            "total_decode_wall_sec": 0.0,
            "total_decode_pure_comm_sec": 0.0,
            "total_decode_pipeline_wait_sec": 0.0,
            "total_decode_compute_sec": 0.0,
            "total_n_accepted": 0.0,
            "total_n_acceptance_flags": 0.0,
            "num_samples": 0.0,
        }
    def _row_pure_comm(r: dict[str, Any]) -> float:
        if "pipeline_decode_pure_comm_sec" in r:
            return float(r["pipeline_decode_pure_comm_sec"])
        return float(r.get("pipeline_decode_comm_sec", 0.0))

    def _row_pipeline_wait(r: dict[str, Any]) -> float:
        return float(r.get("pipeline_decode_pipeline_wait_sec", 0.0))

    n = float(len(sub))
    mean_prefill = sum(float(r["pipeline_prefill_wall_sec"]) for r in sub) / n
    mean_decode = sum(float(r["pipeline_decode_wall_sec"]) for r in sub) / n
    mean_decode_pure_comm = sum(_row_pure_comm(r) for r in sub) / n
    mean_decode_pipeline_wait = sum(_row_pipeline_wait(r) for r in sub) / n
    mean_decode_compute = sum(float(r["pipeline_decode_compute_sec"]) for r in sub) / n
    mean_acc = sum(float(r["acceptance_rate"]) for r in sub) / n
    mean_equiv = sum(float(r["equivalent_accept_length"]) for r in sub) / n
    tot_new = sum(int(r["new_tokens"]) for r in sub)
    tot_steps = sum(int(r["decode_loop_steps"]) for r in sub)
    tot_prompt = sum(int(r["prompt_len"]) for r in sub)
    tot_prefill = sum(float(r["pipeline_prefill_wall_sec"]) for r in sub)
    tot_decode = sum(float(r["pipeline_decode_wall_sec"]) for r in sub)
    tot_decode_pure_comm = sum(_row_pure_comm(r) for r in sub)
    tot_decode_pipeline_wait = sum(_row_pipeline_wait(r) for r in sub)
    tot_decode_compute = sum(float(r["pipeline_decode_compute_sec"]) for r in sub)
    tot_flags = sum(int(r.get("n_acceptance_flags", 0) or 0) for r in sub)
    tot_acc = sum(int(r.get("n_accepted", 0) or 0) for r in sub)
    n_stages = int(sub[0].get("num_stages", 0) or 0)
    pooled_acc = (float(tot_new) / float(tot_steps)) if tot_steps else 0.0
    pooled_equiv = float(n_stages) * pooled_acc
    agg_prefill_tok_s = float(tot_prompt) / tot_prefill if tot_prefill > 0 else 0.0
    agg_decode_tok_s = float(tot_new) / tot_decode if tot_decode > 0 else 0.0
    agg_decode_compute_tok_s = (
        float(tot_new) / tot_decode_compute if tot_decode_compute > 0 else 0.0
    )
    return {
        "mean_prefill_wall_sec": mean_prefill,
        "mean_decode_wall_sec": mean_decode,
        "mean_decode_pure_comm_sec": mean_decode_pure_comm,
        "mean_decode_pipeline_wait_sec": mean_decode_pipeline_wait,
        "mean_decode_compute_sec": mean_decode_compute,
        "mean_acceptance_rate": mean_acc,
        "mean_equivalent_accept_length": mean_equiv,
        "pooled_acceptance_rate": pooled_acc,
        "pooled_equivalent_accept_length": pooled_equiv,
        "aggregate_prefill_tok_s": agg_prefill_tok_s,
        "aggregate_decode_tok_s": agg_decode_tok_s,
        "aggregate_decode_compute_tok_s": agg_decode_compute_tok_s,
        "total_new_tokens": float(tot_new),
        "total_decode_loop_steps": float(tot_steps),
        "total_prompt_tokens": float(tot_prompt),
        "total_decode_wall_sec": float(tot_decode),
        "total_decode_pure_comm_sec": float(tot_decode_pure_comm),
        "total_decode_pipeline_wait_sec": float(tot_decode_pipeline_wait),
        "total_decode_compute_sec": float(tot_decode_compute),
        "total_n_accepted": float(tot_acc),
        "total_n_acceptance_flags": float(tot_flags),
        "num_samples": n,
    }


def _print_aggregates(per_sample: list[dict[str, Any]], wall_sec: float) -> None:
    o = _aggregate_mp_pipeline(per_sample, None)
    print()
    print(
        f"Overall — mean prefill {o['mean_prefill_wall_sec']:.4f}s, "
        f"mean decode wall {o['mean_decode_wall_sec']:.4f}s "
        f"(pure comm {o['mean_decode_pure_comm_sec']:.4f}s, "
        f"pipeline wait {o['mean_decode_pipeline_wait_sec']:.4f}s, "
        f"compute Σ(max stage + verify) {o['mean_decode_compute_sec']:.4f}s); "
        f"mean acceptance rate {o['mean_acceptance_rate']:.4f}, "
        f"mean equiv. accept length {o['mean_equivalent_accept_length']:.4f}"
    )
    print(
        f"Overall — pooled acceptance {o['pooled_acceptance_rate']:.4f}, "
        f"pooled equiv. accept length {o['pooled_equivalent_accept_length']:.4f} "
        f"[{int(o['total_new_tokens'])} tok / {int(o['total_decode_loop_steps'])} steps]"
    )
    print(
        f"Overall — aggregate prefill {o['aggregate_prefill_tok_s']:.2f} tok/s, "
        f"decode wall {o['aggregate_decode_tok_s']:.2f} tok/s, "
        f"decode compute {o['aggregate_decode_compute_tok_s']:.2f} tok/s"
    )
    for ds, _ in DATASET_CONFIG:
        m = _aggregate_mp_pipeline(per_sample, ds)
        print(
            f"  [{ds}] mean_prefill={m['mean_prefill_wall_sec']:.4f}s, "
            f"mean_decode_wall={m['mean_decode_wall_sec']:.4f}s, "
            f"mean_decode_compute={m['mean_decode_compute_sec']:.4f}s, "
            f"mean_acc={m['mean_acceptance_rate']:.4f}, "
            f"mean_equiv={m['mean_equivalent_accept_length']:.4f}, "
            f"pooled_acc={m['pooled_acceptance_rate']:.4f}, "
            f"agg_decode_wall={m['aggregate_decode_tok_s']:.2f} tok/s, "
            f"agg_decode_compute={m['aggregate_decode_compute_tok_s']:.2f} tok/s"
        )
    print(f"\nWall time (this run config): {wall_sec:.2f}s")


def _load_models_for_ckpt(
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    spec_ckpt_path: str,
    base_path: str,
    attn_implementation: str,
    p2p: PipelineP2P,
    merge_last_stage: bool,
) -> tuple[Any, Any, Any, int | None]:
    if rank == 0:
        from transformers import AutoTokenizer

        prefill_bundle = load_prefill_rank0_bundle(
            base_model_path=base_path,
            spec_ckpt_path=spec_ckpt_path,
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos = getattr(prefill_bundle.pipe.config, "eos_token_id", None)
        worker = None
        if merge_last_stage:
            worker_bundle = load_stage_rank_bundle(
                rank=0,
                base_model_path=base_path,
                spec_ckpt_path=spec_ckpt_path,
                dtype=dtype,
                device=device,
                attn_implementation=attn_implementation,
                merge_last_stage=True,
            )
            worker = StageWorker(worker_bundle, p2p)
        return prefill_bundle, tokenizer, eos, worker
    worker_bundle = load_stage_rank_bundle(
        rank=rank,
        base_model_path=base_path,
        spec_ckpt_path=spec_ckpt_path,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
        merge_last_stage=merge_last_stage,
    )
    worker = StageWorker(worker_bundle, p2p)
    return None, None, None, worker


def _run_eval_for_ckpt(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    args: argparse.Namespace,
    spec_ckpt_path: str,
    indexed: list[tuple[int, UnifiedItem]],
    out_dir: Path,
    rank_gpus: list[int] | None,
) -> None:
    dtype = _resolve_dtype(str(args.dtype))
    spec_cfg = load_spec_checkpoint_config(spec_ckpt_path)
    num_stages = int(spec_cfg["num_stages"])
    merge_last_stage = bool(args.merge_last_stage)
    expected_ws = expected_world_size(num_stages, merge_last_stage=merge_last_stage)
    if world_size != expected_ws:
        if rank == 0:
            raise ValueError(
                f"Checkpoint num_stages={num_stages} merge_last_stage={merge_last_stage} "
                f"requires world_size={expected_ws}, got {world_size}. "
                f"Relaunch torchrun with --nproc_per_node={expected_ws}."
            )
        dist.barrier()
        raise ValueError("world_size mismatch")

    base_path = _resolve_base_model_path(spec_cfg, args.base_model_path, spec_ckpt_path)
    timeout = PhaseTimeout(
        prefill_sec=float(args.prefill_timeout_sec),
        decode_sec=float(args.decode_timeout_sec),
    )
    p2p = PipelineP2P(
        rank,
        world_size,
        device,
        async_comm=bool(args.async_comm),
        merge_last_stage=merge_last_stage,
    )
    ctrl_buf = make_ctrl_tensor(opcode=CtrlOpcode.GO, cycle_id=0, device=device)
    done_flag = torch.zeros(1, dtype=torch.int64, device=device)

    prefill_bundle, tokenizer, eos, worker = _load_models_for_ckpt(
        rank=rank,
        device=device,
        dtype=dtype,
        spec_ckpt_path=spec_ckpt_path,
        base_path=base_path,
        attn_implementation=str(args.attn_implementation),
        p2p=p2p,
        merge_last_stage=merge_last_stage,
    )
    dist.barrier()

    trained_deepest = bool(spec_cfg.get("trained_with_use_deepest", False))
    use_deepest = trained_deepest or bool(args.use_deepest)
    use_chat = not bool(args.no_chat_template)
    enable_thinking = bool(args.enable_thinking)
    n_total = len(indexed)
    temperatures = _normalize_temperature_list(list(args.temperature))
    draft_top_ks = _normalize_draft_top_k_list(list(args.draft_top_k))

    ckpt_tag = ckpt_to_filename_tag(spec_ckpt_path)
    base_name = f"mp_pipeline_eval__{ckpt_tag}__nt{n_total}"
    raw_dir = out_dir / "raw"
    summary_dir = out_dir / "summary"
    if rank == 0:
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

    results_meta: list[dict[str, Any]] = []

    def _broadcast_prompt_for_index(sample_index: int) -> torch.LongTensor:
        if rank == 0:
            _idx, item = indexed[sample_index]
            _set_torch_rng_for_eval_sample(int(args.seed), int(_idx))
            ids, _attn = _encode_prompt(
                tokenizer,
                item.question,
                use_chat,
                device,
                enable_thinking=enable_thinking,
            )
        else:
            ids = None
        return broadcast_input_ids(rank, device, ids)

    for temp in temperatures:
        for dtk in draft_top_ks:
            greedy = bool(float(temp) == 0.0)
            n_warmup = max(int(args.warmup_iters), 0)
            warmup_tok = max(int(args.warmup_new_tokens), 1)
            if n_warmup > 0 and n_total > 0:
                warmup_ids = _broadcast_prompt_for_index(0)
                for _ in range(n_warmup):
                    _run_one_session(
                        rank=rank,
                        device=device,
                        dtype=dtype,
                        input_ids=warmup_ids,
                        prefill_bundle=prefill_bundle,
                        worker=worker,
                        p2p=p2p,
                        ctrl_buf=ctrl_buf,
                        done_flag=done_flag,
                        timeout=timeout,
                        greedy=greedy,
                        temperature=float(temp),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        verify=bool(args.verify),
                        use_deepest=use_deepest,
                        eos_token_id=eos,
                        max_new_tokens=warmup_tok,
                        release_base_layers=False,
                        sync_mode=str(args.sync_mode),
                        merge_last_stage=merge_last_stage,
                    )

            if rank == 0:
                t0 = time.perf_counter()
                rows: list[dict[str, Any]] = []
                it = tqdm(
                    range(n_total),
                    total=n_total,
                    desc=f"mp pipeline T={temp} dtk={dtk}",
                    unit="sample",
                )
            else:
                it = range(n_total)

            for si in it:
                if rank == 0:
                    idx, item = indexed[si]
                    input_ids, _attn = _encode_prompt(
                        tokenizer,
                        item.question,
                        use_chat,
                        device,
                        enable_thinking=enable_thinking,
                    )
                    _set_torch_rng_for_eval_sample(int(args.seed), int(idx))
                    prompt_len = int(input_ids.shape[1])
                else:
                    input_ids = None
                    item = None
                    idx = si
                    prompt_len = 0

                input_ids = broadcast_input_ids(rank, device, input_ids)

                result = _run_one_session(
                    rank=rank,
                    device=device,
                    dtype=dtype,
                    input_ids=input_ids,
                    prefill_bundle=prefill_bundle,
                    worker=worker,
                    p2p=p2p,
                    ctrl_buf=ctrl_buf,
                    done_flag=done_flag,
                    timeout=timeout,
                    greedy=greedy,
                    temperature=float(temp),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    verify=bool(args.verify),
                    use_deepest=use_deepest,
                    eos_token_id=eos,
                    max_new_tokens=int(args.max_new_tokens),
                    release_base_layers=False,
                    sync_mode=str(args.sync_mode),
                    merge_last_stage=merge_last_stage,
                )

                if rank == 0:
                    session, out_ids, accept = result
                    gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    rows.append(
                        _sample_row(
                            idx=int(idx),
                            it=item,
                            session=session,
                            out_ids=out_ids,
                            accept=accept,
                            prompt_len=prompt_len,
                            num_stages=num_stages,
                            gen_text=gen_text,
                            temperature=float(temp),
                            draft_top_k=int(dtk),
                            use_deepest=use_deepest,
                        )
                    )

            if rank == 0:
                elapsed = time.perf_counter() - t0
                stem = _per_sample_file_stem(
                    base_name,
                    float(temp),
                    int(dtk),
                    temperatures=temperatures,
                    draft_top_ks=draft_top_ks,
                )
                per_sample_path = raw_dir / f"{stem}.jsonl"
                with per_sample_path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                overall = _aggregate_mp_pipeline(rows, None)
                per_dataset = {
                    ds: _aggregate_mp_pipeline(rows, ds) for ds, _ in DATASET_CONFIG
                }
                results_meta.append(
                    {
                        "temperature": float(temp),
                        "greedy": greedy,
                        "draft_top_k": int(dtk),
                        "use_deepest": bool(use_deepest),
                        "enable_thinking": enable_thinking,
                        "total_wall_sec": float(elapsed),
                        "overall": overall,
                        "per_dataset": per_dataset,
                        "per_sample_path": str(per_sample_path),
                    }
                )
                print(f"\n--- temperature={temp} draft_top_k={dtk} ---")
                _print_aggregates(rows, elapsed)
                print(f"Wrote: {per_sample_path}")

    if rank == 0:
        summary_path = summary_dir / f"{base_name}__summary.json"
        summary: dict[str, Any] = {
            "checkpoint_path": spec_ckpt_path,
            "model": base_path,
            "dtype": str(args.dtype),
            "data_dir": str(Path(args.data_dir).resolve()),
            "num_prompts": n_total,
            "seed": int(args.seed),
            "max_new_tokens": int(args.max_new_tokens),
            "verify": bool(args.verify),
            "use_deepest": bool(use_deepest),
            "enable_thinking": enable_thinking,
            "num_stages": num_stages,
            "merge_last_stage": merge_last_stage,
            "world_size": world_size,
            "rank_gpus": rank_gpus,
            "pipeline_implementation": "distributed_inference",
            "temperatures_evaluated": temperatures,
            "draft_top_ks_evaluated": draft_top_ks,
            "note_draft_top_k": "distributed inference does not use draft_top_k at decode time",
            "results": results_meta,
            "total_wall_sec": float(sum(float(r["total_wall_sec"]) for r in results_meta)),
        }
        if len(draft_top_ks) == 1:
            summary["draft_top_k"] = int(draft_top_ks[0])
        if len(results_meta) == 1:
            r0 = results_meta[0]
            summary["temperature"] = r0["temperature"]
            summary["greedy"] = r0["greedy"]
            summary["overall"] = r0["overall"]
            summary["per_dataset"] = r0["per_dataset"]
            summary["per_sample_path"] = r0["per_sample_path"]
            summary["total_wall_sec"] = float(r0["total_wall_sec"])
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {summary_path}")

    dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    ckpt_list = _normalize_spec_head_ckpt_list(args.spec_head_ckpt)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    rank_gpus = parse_rank_gpu_ids(args.rank_gpus) if str(args.rank_gpus).strip() else None
    rank, world_size, device = init_dist_rank_device(
        rank_gpus,
        init_timeout_minutes=max(float(args.prefill_timeout_sec) / 60.0, 1.0),
    )

    indexed: list[tuple[int, UnifiedItem]] = []
    if rank == 0:
        all_items = _load_all_items(data_dir)
        indexed = list(enumerate(all_items))
        n_total_t = torch.tensor([len(indexed)], dtype=torch.int64, device=device)
    else:
        n_total_t = torch.zeros(1, dtype=torch.int64, device=device)
    dist.broadcast(n_total_t, src=0)
    n_total = int(n_total_t.item())

    if rank != 0:
        indexed = [(i, UnifiedItem("", i, "")) for i in range(n_total)]

    for ckpt_idx, ckpt_path in enumerate(ckpt_list):
        if rank == 0 and len(ckpt_list) > 1:
            print(f"\n=== Checkpoint {ckpt_idx + 1}/{len(ckpt_list)}: {ckpt_path} ===\n")
        _run_eval_for_ckpt(
            rank=rank,
            world_size=world_size,
            device=device,
            args=args,
            spec_ckpt_path=ckpt_path,
            indexed=indexed,
            out_dir=out_dir,
            rank_gpus=rank_gpus,
        )

    dist.destroy_process_group()
