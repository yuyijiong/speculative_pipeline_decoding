"""
Multi-process pipeline parallel speculative decoding entry point.

Launch::

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \\
    distributed_inference/run_generate.py \\
    --spec_head_ckpt /path/to/speculation_head_final.pt \\
    --rank_gpus 0,1,2,3,4

``world_size`` must equal ``num_stages + 1`` (default), or ``num_stages`` when
``--merge_last_stage`` is set (last stage co-located on rank 0 with speculation).
Rank 0 holds speculation module, embedding, final norm, and lm head; ranks 1..n hold
pipeline stages (or ranks 1..n-1 when ``merge_last_stage``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from distributed_inference import _paths  # noqa: F401
from distributed_inference.comm import CtrlOpcode, PipelineP2P, make_ctrl_tensor
from distributed_inference.decode import run_rank0_decode, run_stage_worker_loop
from distributed_inference.device import (
    PhaseTimeout,
    init_dist_rank_device,
    parse_rank_gpu_ids,
    sync_device,
)
from distributed_inference.loader import (
    PrefillRank0Bundle,
    load_prefill_rank0_bundle,
    load_rank0_decode_bundle,
    load_spec_checkpoint_config,
    load_stage_rank_bundle,
)
from distributed_inference.prefill import PrefillResult, broadcast_input_ids, run_prefill
from distributed_inference.rank0_controller import Rank0Controller
from distributed_inference.stage_worker import StageWorker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-process pipeline parallel speculative decoding"
    )
    p.add_argument(
        "--spec_head_ckpt",
        type=str,
        required=True,
        help="Path to speculation_head_final.pt (state_dict + config).",
    )
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--use_deepest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--rank_gpus",
        type=str,
        default="",
        help=(
            "Comma-separated physical GPU ids, length = world_size "
            "(num_stages+1, or num_stages with --merge_last_stage). "
            "rank i uses rank_gpus[i]. If empty, use LOCAL_RANK from torchrun."
        ),
    )
    p.add_argument(
        "--prefill_timeout_sec",
        type=float,
        default=60.0,
        help="Prefill phase wall-clock timeout in seconds.",
    )
    p.add_argument(
        "--decode_timeout_sec",
        type=float,
        default=30.0,
        help="Decoding phase wall-clock timeout per phase check.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Introduce LLM.",
        help="User prompt for chat-template tokenization.",
    )
    p.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup prefill+decode sessions before the timed run.",
    )
    p.add_argument(
        "--warmup_new_tokens",
        type=int,
        default=4,
        help="max_new_tokens for each warmup decode session.",
    )
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")#"eager")#
    p.add_argument("--async_comm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--sync_mode",
        type=str,
        default="barrier",
        choices=("barrier", "comm_only"),
        help="Cycle sync: dist.barrier (default) or comm wait only.",
    )
    p.add_argument(
        "--merge_last_stage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Co-locate the last pipeline stage on rank 0 with the speculation module "
            "(CUDA streams). world_size must equal num_stages."
        ),
    )
    p.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=("nccl", "gloo"),
        help="torch.distributed process group backend (default: nccl). "
        "gloo stages CUDA tensors via CPU for send/recv.",
    )
    return p.parse_args()


def _resolve_base_model_path(spec_cfg: dict, arg_path: str, ckpt_path: str) -> str:
    model_path = str(arg_path or spec_cfg.get("base_model_path", "")).strip()
    if not model_path:
        raise ValueError(
            f"base_model_path required (arg or checkpoint config at {ckpt_path!r})."
        )
    return model_path


@dataclass
class SessionTiming:
    prefill_sec: float = 0.0
    post_prefill_setup_sec: float = 0.0
    decode_sec: float = 0.0
    decode_steps: int = 0
    n_new_tokens: int = 0
    n_accepted: int = 0
    decode_breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTiming:
    dist_init_sec: float = 0.0
    load_sec: float = 0.0
    tokenize_sec: float = 0.0
    warmup_sessions: list[SessionTiming] = field(default_factory=list)
    timed: SessionTiming = field(default_factory=SessionTiming)

    @property
    def inference_sec(self) -> float:
        t = self.timed
        return t.prefill_sec + t.post_prefill_setup_sec + t.decode_sec

    @property
    def script_total_sec(self) -> float:
        return (
            self.dist_init_sec
            + self.load_sec
            + self.tokenize_sec
            + self.inference_sec
        )


def _init_rank0_controller(
    prefill_bundle: PrefillRank0Bundle,
    prefill_result: PrefillResult,
    device: torch.device,
    *,
    use_deepest: bool,
    release_base_layers: bool,
) -> tuple[Rank0Controller, int, int]:
    r0_bundle = load_rank0_decode_bundle(prefill_bundle, device)
    if release_base_layers:
        prefill_bundle.pipe.base_model.model.layers = torch.nn.ModuleList()
        torch.cuda.empty_cache()

    trained_deepest = bool(getattr(prefill_bundle.pipe, "trained_with_use_deepest", False))
    rank0 = Rank0Controller(r0_bundle, use_deepest=trained_deepest or use_deepest)
    rank0.build_position_snapshots_from_prefill(
        prefill_result.tensors_by_idx, prefill_result.seq_len
    )
    rank0.init_spec_kv_from_prefill(prefill_result.seq_len)

    s0 = int(prefill_result.seq_len)
    first_id = int(prefill_result.first_token_id)
    rank0.verified_up_to = s0 + 1
    rank0.next_position = s0 + 1
    rank0.generated_ids = [first_id]
    rank0.token_acceptance = [True]
    emb = rank0.b.embed_tokens(torch.tensor([[first_id]], device=device))
    rank0.pipeline = [
        {"hs": emb, "pos": s0, "snap": rank0._initial_snap(emb)},
    ]
    return rank0, s0, first_id


def _run_one_session(
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    input_ids: torch.LongTensor,
    prefill_bundle: PrefillRank0Bundle | None,
    worker: StageWorker | None,
    p2p: PipelineP2P,
    ctrl_buf: torch.Tensor,
    done_flag: torch.Tensor,
    timeout: PhaseTimeout,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    verify: bool,
    use_deepest: bool,
    eos_token_id: int | None,
    max_new_tokens: int,
    release_base_layers: bool,
    sync_mode: str,
    merge_last_stage: bool = False,
) -> tuple[SessionTiming, list[int], list[bool]] | tuple[None, list[int], list[bool]]:
    timeout.set_phase("prefill")
    sync_device(device)
    prefill_start = time.perf_counter()
    prefill_result = run_prefill(
        rank=rank,
        device=device,
        input_ids=input_ids,
        prefill_bundle=prefill_bundle,
        worker_bundle=worker.b if worker is not None else None,
        dtype=dtype,
        timeout=timeout,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        merge_last_stage=merge_last_stage,
    )
    dist.barrier()
    prefill_sec = time.perf_counter() - prefill_start
    sync_device(device)

    session = SessionTiming(prefill_sec=prefill_sec)

    if rank == 0:
        assert prefill_bundle is not None and prefill_result is not None
        setup_start = time.perf_counter()
        rank0, s0, first_id = _init_rank0_controller(
            prefill_bundle,
            prefill_result,
            device,
            use_deepest=use_deepest,
            release_base_layers=release_base_layers,
        )
        session.post_prefill_setup_sec = time.perf_counter() - setup_start

        timeout.set_phase("decode")
        out_ids, accept, steps, decode_timing = run_rank0_decode(
            rank0,
            p2p,
            s0=s0,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            verify=verify,
            eos_token_id=eos_token_id,
            device=device,
            done_flag=done_flag,
            initial_go_token_id=first_id,
            timeout=timeout,
            sync_mode=sync_mode,
            last_stage_worker=worker if merge_last_stage else None,
        )
        session.decode_sec = float(decode_timing["decode_wall_sec"])
        session.decode_steps = int(steps)
        session.n_new_tokens = len(out_ids)
        session.n_accepted = sum(accept)
        session.decode_breakdown = {
            k: v
            for k, v in decode_timing.items()
            if k not in ("decode_wall_sec", "decode_steps")
        }
    else:
        assert worker is not None
        timeout.set_phase("decode")
        run_stage_worker_loop(
            worker,
            p2p,
            ctrl_buf,
            hidden_size=worker.b.hidden_size,
            done_flag=done_flag,
            timeout=timeout,
            sync_mode=sync_mode,
        )
        out_ids = []
        accept = []

    done_flag.zero_()
    dist.barrier()

    if rank == 0:
        return session, out_ids, accept
    return None, [], []


def _forward_stage_timing_rows(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec_sec = float(breakdown.get("spec_forward_sec", 0.0))
    spec_steps = int(breakdown.get("spec_forward_steps", 0))
    if spec_steps > 0:
        rows.append(
            {
                "label": "spec",
                "total_sec": spec_sec,
                "active_steps": spec_steps,
                "avg_sec": spec_sec / float(spec_steps),
            }
        )
    stage_totals = breakdown.get("stage_forward_sec", [])
    stage_counts = breakdown.get("stage_forward_active_steps", [])
    for si, tot in enumerate(stage_totals):
        cnt = int(stage_counts[si]) if si < len(stage_counts) else 0
        if cnt <= 0:
            continue
        tot_sec = float(tot)
        rows.append(
            {
                "label": f"stage_{si}",
                "total_sec": tot_sec,
                "active_steps": cnt,
                "avg_sec": tot_sec / float(cnt),
            }
        )
    return rows


def _print_forward_stage_timing(breakdown: dict[str, Any]) -> None:
    rows = _forward_stage_timing_rows(breakdown)
    if not rows:
        return
    print("\n  --- forward compute per active step ---")
    avg_ms_list = [f"{row['label']}={row['avg_sec'] * 1000.0:.3f}ms" for row in rows]
    print(f"  per-step avg (active only): [{', '.join(avg_ms_list)}]")
    for row in rows:
        print(
            f"  [{row['label']:>7}]  "
            f"total={row['total_sec'] * 1000.0:9.2f} ms  "
            f"active={int(row['active_steps']):4d}  "
            f"avg={row['avg_sec'] * 1000.0:8.3f} ms"
        )


def _print_decode_breakdown(
    *,
    label: str,
    decode_sec: float,
    decode_steps: int,
    breakdown: dict[str, Any],
) -> None:
    if not breakdown:
        return

    wall_ms = decode_sec * 1000.0
    spec_ms = breakdown.get("spec_forward_sec", 0.0) * 1000.0
    verify_ms = breakdown.get("verify_sec", 0.0) * 1000.0
    driver_ms = breakdown.get("driver_update_sec", 0.0) * 1000.0
    rank0_gpu_ms = breakdown.get("rank0_gpu_sec", 0.0) * 1000.0
    if rank0_gpu_ms <= 0.0:
        rank0_gpu_ms = spec_ms + verify_ms + driver_ms
    pipeline_wait_ms = breakdown.get("pipeline_wait_sec", 0.0) * 1000.0
    pure_comm_ms = breakdown.get(
        "pure_comm_sec", breakdown.get("comm_sec", 0.0)
    ) * 1000.0
    compute_ms = float(breakdown.get("decode_compute_sec", 0.0)) * 1000.0
    if compute_ms <= 0.0:
        compute_ms = float(
            max(decode_sec - pure_comm_ms / 1000.0 - pipeline_wait_ms / 1000.0, 0.0)
        ) * 1000.0
    cycle_max_ms = breakdown.get("decode_cycle_max_stage_sec", 0.0) * 1000.0
    pure_comm_pct = (pure_comm_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
    pipeline_wait_pct = (pipeline_wait_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
    rank0_gpu_pct = (rank0_gpu_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
    ms_per_step = (wall_ms / decode_steps) if decode_steps > 0 else 0.0

    print(f"\n  --- decode breakdown ({label}, rank 0) ---")
    print(f"  rank0 GPU:           {rank0_gpu_ms:8.2f} ms  ({rank0_gpu_pct:5.1f}%)")
    spec_steps = int(breakdown.get("spec_forward_steps", 0))
    if spec_ms > 0.0 and spec_steps > 0:
        print(f"    spec_forward:      {spec_ms:8.2f} ms")
        print(f"    verify:            {verify_ms:8.2f} ms")
        print(f"    driver_update:     {driver_ms:8.2f} ms")
        print(f"    per-call spec:     {spec_ms / spec_steps:8.3f} ms  ({spec_steps} calls)")
    print(f"  pipeline wait:       {pipeline_wait_ms:8.2f} ms  ({pipeline_wait_pct:5.1f}%)")
    print(f"    recv_snap:         {breakdown.get('recv_snap_sec', 0.0) * 1000:8.2f} ms")
    print(f"    recv_verify:       {breakdown.get('recv_verify_sec', 0.0) * 1000:8.2f} ms")
    print(f"  pure communication:  {pure_comm_ms:8.2f} ms  ({pure_comm_pct:5.1f}%)")
    print(f"    wait_comm:         {breakdown.get('wait_comm_sec', 0.0) * 1000:8.2f} ms")
    print(f"    ctrl_prepare:      {breakdown.get('ctrl_prepare_sec', 0.0) * 1000:8.2f} ms")
    print(f"    ctrl_broadcast:    {breakdown.get('ctrl_broadcast_sec', 0.0) * 1000:8.2f} ms")
    print(f"    cycle_sync:        {breakdown.get('cycle_sync_sec', 0.0) * 1000:8.2f} ms")
    print(f"    discard_comm:      {breakdown.get('discard_comm_sec', 0.0) * 1000:8.2f} ms")
    print(f"    shutdown:          {breakdown.get('shutdown_sec', 0.0) * 1000:8.2f} ms")
    unaccounted_ms = breakdown.get("unaccounted_sec", 0.0) * 1000.0
    if unaccounted_ms > 0.01:
        print(f"  unaccounted:         {unaccounted_ms:8.2f} ms")
    print(
        f"  compute Σ(max stage + verify): {compute_ms:8.2f} ms  "
        f"({(compute_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0:5.1f}% of wall)"
    )
    if cycle_max_ms > 0.0:
        print(f"    cycle max stage:   {cycle_max_ms:8.2f} ms")
        print(f"    verify:            {verify_ms:8.2f} ms")
    _print_forward_stage_timing(breakdown)
    if decode_steps > 0:
        print(f"  per-step (wall):     {ms_per_step:8.3f} ms")
        print(
            f"  per-step pure comm:  {pure_comm_ms / decode_steps:8.3f} ms  "
            f"pipeline wait: {pipeline_wait_ms / decode_steps:8.3f} ms"
        )

    rb_count = int(breakdown.get("rollback_count", 0))
    if rb_count > 0:
        rb_wall_ms = breakdown.get("rollback_wall_sec", 0.0) * 1000.0
        rb_avg_ms = breakdown.get("rollback_avg_sec", 0.0) * 1000.0
        rb_pct = (rb_wall_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0
        print(f"\n  --- rollback / reject ({label}, rank 0) ---")
        print(f"  rollback count:      {rb_count:8d}")
        print(f"  rollback total:      {rb_wall_ms:8.2f} ms  ({rb_pct:5.1f}% of decode wall)")
        print(f"  rollback avg:        {rb_avg_ms:8.3f} ms")
        print(
            f"    verify (reject):   {breakdown.get('rollback_verify_sec', 0.0) * 1000:8.2f} ms  "
            f"({breakdown.get('rollback_verify_sec', 0.0) / rb_count * 1000:.3f} ms/call)"
        )
        print(
            f"    discard comm:      {breakdown.get('rollback_discard_comm_sec', 0.0) * 1000:8.2f} ms  "
            f"({breakdown.get('rollback_discard_comm_sec', 0.0) / rb_count * 1000:.3f} ms/call)"
        )
        print(
            f"    apply_reject:      {breakdown.get('rollback_apply_reject_sec', 0.0) * 1000:8.2f} ms  "
            f"({breakdown.get('rollback_apply_reject_sec', 0.0) / rb_count * 1000:.3f} ms/call)"
        )
    elif breakdown.get("verify_sec", 0.0) > 0.0:
        print(f"\n  --- rollback / reject ({label}, rank 0) ---")
        print("  rollback count:      0")


def _print_timing_report(
    *,
    prompt: str,
    verify: bool,
    timing: RunTiming,
    out_ids: list[int],
    accept: list[bool],
    attn_implementation: str,
    async_comm: bool,
    sync_mode: str,
    dist_backend: str,
) -> None:
    steps = timing.timed.decode_steps
    n_new = timing.timed.n_new_tokens
    n_accepted = timing.timed.n_accepted
    decode_sec = timing.timed.decode_sec
    accept_rate = (n_new / steps) if steps > 0 else 0.0
    tokens_per_step = (n_new / steps) if steps > 0 else 0.0
    ms_per_step = (decode_sec * 1000.0 / steps) if steps > 0 else 0.0
    ms_per_token = (decode_sec * 1000.0 / n_new) if n_new > 0 else 0.0

    print("\n--- Multi-process pipeline generate ---")
    print(f"Prompt: {prompt!r}")
    print(f"verify: {verify}")
    print(f"attn_implementation: {attn_implementation}")
    print(f"async_comm: {async_comm}")
    print(f"sync_mode: {sync_mode}")
    print(f"dist_backend: {dist_backend}")
    print("\n--- Segmented timing (rank 0) ---")
    print(f"  dist init:           {timing.dist_init_sec * 1000:.2f} ms")
    print(f"  model load:          {timing.load_sec * 1000:.2f} ms")
    print(f"  tokenize:            {timing.tokenize_sec * 1000:.2f} ms")
    for i, ws in enumerate(timing.warmup_sessions):
        w_total = ws.prefill_sec + ws.post_prefill_setup_sec + ws.decode_sec
        print(
            f"  warmup[{i}] (excluded): "
            f"prefill={ws.prefill_sec * 1000:.2f} ms, "
            f"setup={ws.post_prefill_setup_sec * 1000:.2f} ms, "
            f"decode={ws.decode_sec * 1000:.2f} ms "
            f"(total {w_total * 1000:.2f} ms, {ws.n_new_tokens} tok / {ws.decode_steps} steps)"
        )
    t = timing.timed
    print("  --- timed run ---")
    print(f"  prefill:             {t.prefill_sec * 1000:.2f} ms")
    print(f"  post-prefill setup:  {t.post_prefill_setup_sec * 1000:.2f} ms")
    print(f"  decode:              {t.decode_sec * 1000:.2f} ms")
    _print_decode_breakdown(
        label="timed run",
        decode_sec=t.decode_sec,
        decode_steps=steps,
        breakdown=t.decode_breakdown,
    )
    print(f"  inference subtotal:  {timing.inference_sec * 1000:.2f} ms")
    print(f"  script total:        {timing.script_total_sec * 1000:.2f} ms")
    print("\n--- Generation ---")
    print(f"Decode steps:       {steps}")
    print(f"New tokens:         {n_new}")
    print(f"ms/step:            {ms_per_step:.3f}")
    print(f"ms/token:           {ms_per_token:.3f}")
    print(f"tokens/step:        {tokens_per_step:.3f}")
    n_rejected = len(accept) - n_accepted
    print(f"Acceptance:         {n_accepted} / {len(accept)} ({accept_rate:.3f} tokens/step)")
    rb_count = int(t.decode_breakdown.get("rollback_count", 0))
    if rb_count > 0 or n_rejected > 0:
        rb_avg_ms = t.decode_breakdown.get("rollback_avg_sec", 0.0) * 1000.0
        print(
            f"Rollbacks:          {rb_count} "
            f"(rejected tokens: {n_rejected}, avg {rb_avg_ms:.3f} ms/rollback)"
        )
    print(f"Generated ids:      {list(out_ids)}")


if __name__ == "__main__":
    args = parse_args()

    if not str(args.spec_head_ckpt).strip():
        raise ValueError("--spec_head_ckpt is required.")

    dist_init_start = time.perf_counter()
    rank_gpus = parse_rank_gpu_ids(args.rank_gpus) if args.rank_gpus.strip() else None
    rank, world_size, device = init_dist_rank_device(
        rank_gpus,
        init_timeout_minutes=max(args.prefill_timeout_sec / 60.0, 1.0),
        backend=str(args.dist_backend),
    )
    dist_init_sec = time.perf_counter() - dist_init_start

    dtype = torch.float16
    merge_last_stage = bool(args.merge_last_stage)

    spec_cfg = load_spec_checkpoint_config(args.spec_head_ckpt)
    num_stages = int(spec_cfg["num_stages"])
    expected_ws = num_stages + (0 if merge_last_stage else 1)
    if world_size != expected_ws:
        if rank == 0:
            raise ValueError(
                f"spec ckpt num_stages={num_stages} merge_last_stage={merge_last_stage} "
                f"requires world_size={expected_ws}, got {world_size}. "
                f"Launch with --nproc_per_node={expected_ws}."
            )
        dist.barrier()
        raise ValueError("world_size mismatch")

    base_path = _resolve_base_model_path(spec_cfg, args.base_model_path, args.spec_head_ckpt)
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

    prefill_bundle = None
    worker: StageWorker | None = None
    run_timing = RunTiming(dist_init_sec=dist_init_sec)

    load_start = time.perf_counter()
    if rank == 0:
        from transformers import AutoTokenizer

        prefill_bundle = load_prefill_rank0_bundle(
            base_model_path=base_path,
            spec_ckpt_path=args.spec_head_ckpt,
            dtype=dtype,
            device=device,
            attn_implementation=args.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        eos = getattr(prefill_bundle.pipe.config, "eos_token_id", None)
        if merge_last_stage:
            worker_bundle = load_stage_rank_bundle(
                rank=0,
                base_model_path=base_path,
                spec_ckpt_path=args.spec_head_ckpt,
                dtype=dtype,
                device=device,
                attn_implementation=args.attn_implementation,
                merge_last_stage=True,
            )
            worker = StageWorker(worker_bundle, p2p)
    else:
        worker_bundle = load_stage_rank_bundle(
            rank=rank,
            base_model_path=base_path,
            spec_ckpt_path=args.spec_head_ckpt,
            dtype=dtype,
            device=device,
            attn_implementation=args.attn_implementation,
            merge_last_stage=merge_last_stage,
        )
        worker = StageWorker(worker_bundle, p2p)
        eos = None
        tokenizer = None
    dist.barrier()
    run_timing.load_sec = time.perf_counter() - load_start

    tokenize_sec = 0.0
    if rank == 0:
        tokenize_start = time.perf_counter()
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = batch["input_ids"].to(device)
        tokenize_sec = time.perf_counter() - tokenize_start
    else:
        input_ids = None
    run_timing.tokenize_sec = tokenize_sec

    input_ids = broadcast_input_ids(rank, device, input_ids)
    greedy = float(args.temperature) <= 0.0

    trained_deepest = bool(spec_cfg.get("trained_with_use_deepest", False))
    use_deepest = trained_deepest or bool(args.use_deepest)

    n_warmup = max(int(args.warmup_iters), 0)
    warmup_new_tokens = max(int(args.warmup_new_tokens), 1)
    n_sessions = n_warmup + 1

    final_out_ids: list[int] = []
    final_accept: list[bool] = []

    for sess_idx in range(n_sessions):
        is_warmup = sess_idx < n_warmup
        max_tok = warmup_new_tokens if is_warmup else int(args.max_new_tokens)
        release_layers = not is_warmup

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
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            verify=bool(args.verify),
            use_deepest=use_deepest,
            eos_token_id=eos,
            max_new_tokens=max_tok,
            release_base_layers=release_layers,
            sync_mode=str(args.sync_mode),
            merge_last_stage=merge_last_stage,
        )

        if rank == 0:
            session_timing, out_ids, accept = result
            if is_warmup:
                run_timing.warmup_sessions.append(session_timing)
            else:
                run_timing.timed = session_timing
                final_out_ids = list(out_ids)
                final_accept = list(accept)

    if rank == 0:
        text = tokenizer.decode(final_out_ids, skip_special_tokens=False)
        _print_timing_report(
            prompt=args.prompt,
            verify=bool(args.verify),
            timing=run_timing,
            out_ids=final_out_ids,
            accept=final_accept,
            attn_implementation=str(args.attn_implementation),
            async_comm=bool(args.async_comm),
            sync_mode=str(args.sync_mode),
            dist_backend=str(args.dist_backend),
        )
        print(f"Generated text:\n{text}")

    dist.barrier()
    dist.destroy_process_group()
