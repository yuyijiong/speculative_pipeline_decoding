"""
Multi-process pipeline parallel speculative decoding entry point (v11).

Launch::

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
    distributed_inference/example_mp_pipeline_generate.py \
    --spec_head_ckpt Qwen3.5-4B_s4_l4.pt \
    --rank_gpus 0,1,2,3,4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --standalone --nproc_per_node=9 \
    distributed_inference/example_mp_pipeline_generate.py \
    --spec_head_ckpt Qwen3.5-4B_s8_l4.pt \
    --rank_gpus 0,1,2,3,4,5,6,7,8 --dist_backend gloo

``world_size`` must equal ``num_stages + 1``.
Rank 0 holds speculation module, embedding, final norm, and lm head; ranks 1..n hold
pipeline stages.

Multi-node (8 stages, 9 GPUs across 2 nodes)::

export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   # set to your NIC
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=Your_Host_IP \
  --master_port=29500 \
  distributed_inference/example_mp_pipeline_generate.py \
  --spec_head_ckpt Qwen3.5-4B_s8_l4.pt --dist_backend nccl

export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   # same as head node
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr=Your_Host_IP \
  --master_port=29500 \
  distributed_inference/example_mp_pipeline_generate.py \
  --spec_head_ckpt Qwen3.5-4B_s8_l4.pt --dist_backend nccl
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TextIO

import torch
import torch.distributed as dist

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distributed_inference.comm import (  # noqa: E402
    CtrlOpcode,
    PipelineP2P,
    make_ctrl_tensor,
)
from distributed_inference.decode import (  # noqa: E402
    run_rank0_decode,
    run_stage_worker_loop,
)
from distributed_inference.device import (  # noqa: E402
    PhaseTimeout,
    init_dist_rank_device,
    parse_rank_gpu_ids,
    sync_device,
)
from distributed_inference.loader import (  # noqa: E402
    PrefillRank0Bundle,
    format_ckpt_key_info_lines,
    load_prefill_rank0_bundle,
    load_rank0_decode_bundle,
    load_spec_checkpoint_config,
    load_stage_rank_bundle,
)
from distributed_inference.prefill import (  # noqa: E402
    PrefillResult,
    broadcast_input_ids,
    run_prefill,
)
from distributed_inference.rank0_controller import Rank0Controller  # noqa: E402
from distributed_inference.stage_worker import StageWorker  # noqa: E402

_EXAMPLE_LOG_PATH = _ROOT / "v11_mp_example.log"


class _TeeStdout:
    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self._console = console
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._console.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._console.isatty()


@contextmanager
def _tee_stdout_to_file(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_f:
        sys.stdout = _TeeStdout(original_stdout, log_f)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-process pipeline parallel speculative decoding (v11)"
    )
    p.add_argument(
        "--spec_head_ckpt",
        type=str,
        default="Qwen3.5-4B_s4_l4.pt",
        help="Path to v11 speculation head checkpoint (state_dict + config), e.g. Qwen3.5-4B_s8_l4.pt.",
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
            "Comma-separated physical GPU ids, length = num_stages+1 "
            "(world_size). rank i uses rank_gpus[i]. If empty, use LOCAL_RANK "
            "from torchrun."
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
    p.add_argument(
        "--profile_timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable timing diagnostics gathered once after generation.",
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
        prefill_bundle.pipe._decoder_backbone.layers = torch.nn.ModuleList()
        torch.cuda.empty_cache()

    trained_deepest = bool(getattr(prefill_bundle.pipe, "trained_with_use_deepest", False))
    rank0 = Rank0Controller(r0_bundle, use_deepest=trained_deepest or use_deepest)
    rank0.build_position_snapshots_from_prefill(
        prefill_result.tensors_by_idx, prefill_result.seq_len
    )
    rank0.init_spec_kv_from_prefill(prefill_result.seq_len)
    rank0.capture_cuda_graphs()

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
    profile_timing: bool = False,
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
            profile_timing=profile_timing,
        )
        session.decode_sec = float(decode_timing["decode_wall_sec"])
        session.decode_steps = int(steps)
        session.n_new_tokens = len(out_ids)
        session.n_accepted = sum(accept)
        if profile_timing:
            session.decode_breakdown = {
                k: v
                for k, v in decode_timing.items()
                if k not in ("decode_wall_sec", "decode_steps")
            }
        else:
            session.decode_breakdown = {}
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
            profile_timing=profile_timing,
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


def _print_forward_layer_timing(breakdown: dict[str, Any]) -> None:
    rows = breakdown.get("forward_layer_avg_sec", [])
    if not rows:
        return
    print("\n  --- forward compute per decoder layer ---")
    for row in rows:
        print(
            f"  [layer {int(row['layer_idx']):>3}]  "
            f"avg={float(row['avg_sec']) * 1000.0:8.3f} ms  "
            f"total={float(row.get('total_sec', 0.0)) * 1000.0:9.2f} ms  "
            f"calls={int(row.get('count', 0)):4d}"
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

    def _ms(key: str, default: float = 0.0) -> float:
        return float(breakdown.get(key, default)) * 1000.0

    def _avg_ms(avg_key: str, total_key: str, steps_key: str, fallback_steps: float) -> float:
        if avg_key in breakdown:
            return _ms(avg_key)
        steps = float(breakdown.get(steps_key, 0.0))
        den = steps if steps > 0 else fallback_steps
        if den <= 0:
            return 0.0
        return _ms(total_key) / den

    wall_ms = decode_sec * 1000.0
    spec_steps = int(breakdown.get("spec_forward_steps", 0))
    full_steps = int(breakdown.get("full_pipeline_steps", 0))
    step_den = float(spec_steps) if spec_steps > 0 else float(max(decode_steps, 1))
    full_den = float(full_steps) if full_steps > 0 else step_den

    recv_ms = _ms("rank0_recv_wait_sec", breakdown.get("pipeline_wait_sec", 0.0))
    pure_comm_ms = _ms("pure_comm_sec", breakdown.get("comm_sec", 0.0))
    local_ms = _ms(
        "rank0_local_compute_sec",
        breakdown.get(
            "rank0_gpu_sec",
            float(breakdown.get("spec_forward_sec", 0.0))
            + float(breakdown.get("verify_sec", 0.0))
            + float(breakdown.get("driver_update_sec", 0.0)),
        ),
    )
    seq_ms = _ms("rank0_sequential_sec")
    if seq_ms <= 0.0:
        seq_ms = recv_ms + pure_comm_ms + _ms("verify_sec") + _ms("driver_update_sec")
    cycle_wall_ms = _ms("cycle_wall_sec")
    max_stage_ms = _ms(
        "max_stage_forward_sec",
        breakdown.get("decode_cycle_max_stage_sec", 0.0),
    )
    max_stage_avg_ms = _ms("max_stage_forward_avg_sec")
    crit_ms = _ms(
        "critical_path_compute_sec",
        breakdown.get("decode_compute_sec", 0.0),
    )
    if crit_ms <= 0.0 and max_stage_ms > 0.0:
        crit_ms = max_stage_ms + _ms("verify_sec")
    unaccounted_ms = _ms(
        "rank0_unaccounted_sec",
        breakdown.get("unaccounted_sec", 0.0),
    )

    # Per-active-step means (empty / fill steps excluded).
    recv_verify_avg = _avg_ms(
        "recv_verify_avg_sec", "recv_verify_sec", "recv_verify_steps", full_den
    )
    recv_snap_avg = _avg_ms(
        "recv_snap_avg_sec", "recv_snap_sec", "recv_snap_steps", step_den
    )
    recv_wait_avg = _avg_ms(
        "rank0_recv_wait_avg_sec",
        "full_pipeline_recv_wait_sec",
        "full_pipeline_steps",
        full_den,
    )
    spec_avg = _avg_ms(
        "spec_forward_avg_sec", "spec_forward_sec", "spec_forward_steps", step_den
    )
    verify_avg = _avg_ms("verify_avg_sec", "verify_sec", "verify_steps", full_den)
    driver_avg = _avg_ms(
        "driver_update_avg_sec", "driver_update_sec", "driver_update_steps", step_den
    )
    cycle_wall_avg = _avg_ms(
        "cycle_wall_avg_sec", "cycle_wall_sec", "cycle_wall_steps", step_den
    )
    pure_comm_avg = pure_comm_ms / step_den if step_den > 0 else 0.0

    def _pct(part_ms: float) -> float:
        return (part_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0

    print(f"\n  --- decode breakdown ({label}) ---")
    print(f"  decode wall:         {wall_ms:8.2f} ms")
    if cycle_wall_ms > 0.0:
        print(
            f"  cycle wall sum:      {cycle_wall_ms:8.2f} ms  "
            f"({_pct(cycle_wall_ms):5.1f}% of decode wall)"
        )
    phases_ms = _ms("cycle_phases_sec")
    if phases_ms > 0.0:
        gap_ms = _ms("cycle_phases_gap_sec")
        print(
            f"  cycle phases sum:    {phases_ms:8.2f} ms  "
            f"(additive; gap vs cycle_wall={gap_ms:6.3f} ms)"
        )
    print(
        f"  rank0 sequential:    {seq_ms:8.2f} ms  ({_pct(seq_ms):5.1f}%)  "
        f"[additive cycle phases + shutdown; excludes GPU spec]"
    )
    if unaccounted_ms > 0.01:
        print(
            f"  rank0 unaccounted:   {unaccounted_ms:8.2f} ms  "
            f"({_pct(unaccounted_ms):5.1f}%)"
        )

    # Additive partition of cycle_wall (sum of these ≈ cycle_wall).
    post_recv_avg = _avg_ms(
        "post_recv_avg_sec", "post_recv_sec", "post_recv_steps", full_den
    )
    cycle_other_avg = _avg_ms(
        "cycle_other_avg_sec", "cycle_other_sec", "cycle_other_steps", step_den
    )
    stage0_hs_avg = _avg_ms(
        "stage0_hs_send_avg_sec", "stage0_hs_send_sec", "stage0_hs_send_steps", step_den
    )
    last_hs_avg = _avg_ms(
        "last_stage_hs_send_avg_sec",
        "last_stage_hs_send_sec",
        "last_stage_hs_send_steps",
        full_den,
    )
    ctrl_prep_avg = _avg_ms(
        "ctrl_prepare_avg_sec", "ctrl_prepare_sec", "ctrl_prepare_steps", step_den
    )
    ctrl_bcast_avg = _avg_ms(
        "ctrl_broadcast_avg_sec", "ctrl_broadcast_sec", "ctrl_broadcast_steps", step_den
    )
    cycle_sync_avg = _avg_ms(
        "cycle_sync_avg_sec", "cycle_sync_sec", "cycle_sync_steps", step_den
    )
    discard_avg = _avg_ms(
        "discard_comm_avg_sec", "discard_comm_sec", "discard_comm_steps", step_den
    )

    print(f"\n  --- A. additive cycle phases (sum == cycle_wall) ---")
    additive_rows = [
        ("ctrl_prepare", "ctrl_prepare_sec", ctrl_prep_avg),
        ("ctrl_broadcast", "ctrl_broadcast_sec", ctrl_bcast_avg),
        ("post_recv", "post_recv_sec", post_recv_avg),
        ("spec_forward", "spec_forward_sec", spec_avg),
        ("recv_verify", "recv_verify_sec", recv_verify_avg),
        ("snap_progress", "snap_progress_sec", _avg_ms(
            "snap_progress_avg_sec", "snap_progress_sec", "snap_progress_steps", full_den
        )),
        ("verify", "verify_sec", verify_avg),
        ("recv_snap", "recv_snap_sec", recv_snap_avg),
        ("driver_update", "driver_update_sec", driver_avg),
        ("cycle_sync", "cycle_sync_sec", cycle_sync_avg),
        ("discard_comm", "discard_comm_sec", discard_avg),
        ("cycle_other", "cycle_other_sec", cycle_other_avg),
    ]
    for name, key, avg in additive_rows:
        tot = _ms(key)
        if tot < 0.005 and avg < 0.0005:
            continue
        pct_c = (tot / cycle_wall_ms * 100.0) if cycle_wall_ms > 0 else 0.0
        print(
            f"  {name:16s} {tot:8.2f} ms  ({pct_c:5.1f}% cycle)  "
            f"avg {avg:8.3f} ms"
        )

    print(f"\n  --- B. rank0 recv wait (subset of A; blocked on stages / P2P) ---")
    print(f"  recv wait total:     {recv_ms:8.2f} ms  ({_pct(recv_ms):5.1f}%)")
    print(
        f"    recv_verify:       {_ms('recv_verify_sec'):8.2f} ms  "
        f"(avg {recv_verify_avg:8.3f} ms × {int(breakdown.get('recv_verify_steps', 0))} active)"
    )
    print(
        f"    recv_snap:         {_ms('recv_snap_sec'):8.2f} ms  "
        f"(avg {recv_snap_avg:8.3f} ms × {int(breakdown.get('recv_snap_steps', 0))} active)"
    )

    print(f"\n  --- C. rank0 local compute (serial spec is already in A) ---")
    print(f"  local compute sum:   {local_ms:8.2f} ms  (spec+verify+driver; not wall %)")
    print(f"    spec_forward:      {_ms('spec_forward_sec'):8.2f} ms  (serial host=GPU)")
    print(f"    verify:            {_ms('verify_sec'):8.2f} ms  (host wall)")
    v_kernel_avg = _avg_ms(
        "verify_kernel_avg_sec", "verify_kernel_sec", "verify_kernel_steps", full_den
    )
    v_copy_avg = _avg_ms(
        "verify_copy_avg_sec", "verify_copy_sec", "verify_copy_steps", full_den
    )
    v_decide_avg = _avg_ms(
        "verify_decide_avg_sec", "verify_decide_sec", "verify_decide_steps", full_den
    )
    v_non_kernel_avg = float(breakdown.get("verify_non_kernel_avg_sec", 0.0) or 0.0) * 1000.0
    if _ms("verify_kernel_sec") > 0.0 or _ms("verify_decide_sec") > 0.0:
        print(
            f"      verify_kernel:   {_ms('verify_kernel_sec'):8.2f} ms  "
            f"(avg {v_kernel_avg:8.3f} ms; host wall + sync)"
        )
        print(
            f"      verify_copy:     {_ms('verify_copy_sec'):8.2f} ms  "
            f"(avg {v_copy_avg:8.3f} ms; hs→graph buffer)"
        )
        print(
            f"      verify_decide:   {_ms('verify_decide_sec'):8.2f} ms  "
            f"(avg {v_decide_avg:8.3f} ms; argmax/sample + D2H sync)"
        )
        print(
            f"      verify_non_kernel avg: {v_non_kernel_avg:8.3f} ms  "
            f"(wall − kernel; copy+decide+host gap)"
        )
        gsteps = int(breakdown.get("verify_graph_steps", 0))
        vsteps = int(breakdown.get("verify_steps", 0))
        if vsteps > 0:
            print(
                f"      cuda_graph hits: {gsteps:8d} / {vsteps} verify calls"
            )
    print(f"    driver_update:     {_ms('driver_update_sec'):8.2f} ms")
    if spec_steps > 0:
        print(f"    per-call spec:     {spec_avg:8.3f} ms  ({spec_steps} calls)")

    print(f"\n  --- D. control / sync overhead (subset of A) ---")
    print(f"  pure_comm total:     {pure_comm_ms:8.2f} ms  ({_pct(pure_comm_ms):5.1f}%)")
    print(f"    ctrl_prepare:      {_ms('ctrl_prepare_sec'):8.2f} ms")
    print(f"    ctrl_broadcast:    {_ms('ctrl_broadcast_sec'):8.2f} ms")
    print(f"    cycle_sync:        {_ms('cycle_sync_sec'):8.2f} ms")
    print(f"    discard_comm:      {_ms('discard_comm_sec'):8.2f} ms")
    print(f"    shutdown:          {_ms('shutdown_sec'):8.2f} ms")

    print(f"\n  --- E. hs send (worker gather; stage0 vs last-stage payload) ---")
    print(
        f"  stage0_hs_send:     {_ms('stage0_hs_send_sec'):8.2f} ms  "
        f"(avg {stage0_hs_avg:8.3f} ms; intra-node typically)"
    )
    print(
        f"  last_stage_hs_send: {_ms('last_stage_hs_send_sec'):8.2f} ms  "
        f"(avg {last_hs_avg:8.3f} ms; may be inter-node)"
    )

    print(f"\n  --- F. stage compute (worker gather; not additive with A) ---")
    print(
        f"  max stage total:     {max_stage_ms:8.2f} ms  "
        f"(slowest stage cumulative forward)"
    )
    if max_stage_avg_ms > 0.0:
        print(f"  max stage avg/step:  {max_stage_avg_ms:8.3f} ms  (per active step)")
    print(
        f"  critical estimate:   {crit_ms:8.2f} ms  "
        f"(max_stage_total + verify; capacity view, not wall partition)"
    )
    _print_forward_stage_timing(breakdown)
    _print_forward_layer_timing(breakdown)

    if decode_steps > 0 or spec_steps > 0:
        print(
            f"\n  --- per-active-step means "
            f"(phase dens; full-pipe={full_steps}, spec={spec_steps}) ---"
        )
        print(f"  wall / spec-step:    {wall_ms / step_den:8.3f} ms")
        print(f"  cycle_wall avg:      {cycle_wall_avg:8.3f} ms")
        print(f"  cycle_other avg:     {cycle_other_avg:8.3f} ms")
        print(f"  stage0_hs_send avg:  {stage0_hs_avg:8.3f} ms")
        print(f"  last_hs_send avg:    {last_hs_avg:8.3f} ms")
        print(f"  recv_wait avg:       {recv_wait_avg:8.3f} ms  (full-pipe only)")
        print(f"  recv_verify avg:     {recv_verify_avg:8.3f} ms")
        print(f"  recv_snap avg:       {recv_snap_avg:8.3f} ms")
        print(f"  pure_comm / step:    {pure_comm_avg:8.3f} ms")
        print(f"  spec avg:            {spec_avg:8.3f} ms  (GPU events)")
        print(f"  verify avg:          {verify_avg:8.3f} ms  (host wall)")
        if v_kernel_avg > 0.0 or v_decide_avg > 0.0:
            print(f"  verify_kernel avg:   {v_kernel_avg:8.3f} ms  (host wall + sync)")
            print(f"  verify_copy avg:     {v_copy_avg:8.3f} ms")
            print(f"  verify_decide avg:   {v_decide_avg:8.3f} ms")
            print(f"  verify_non_kernel:   {v_non_kernel_avg:8.3f} ms")
        print(f"  driver avg:          {driver_avg:8.3f} ms")
        if max_stage_avg_ms > 0.0:
            print(f"  max_stage avg:       {max_stage_avg_ms:8.3f} ms")

    by_depth_steps = [int(x) for x in breakdown.get("by_depth_steps", [])]
    if by_depth_steps:
        print(f"\n  --- by pipeline fill (depth=1..n) ---")

        def _depth_avg_ms(avg_key: str, total_key: str, i: int) -> float:
            avgs = breakdown.get(avg_key)
            if isinstance(avgs, list) and i < len(avgs):
                return float(avgs[i]) * 1000.0
            totals = breakdown.get(total_key, []) or []
            dens = by_depth_steps
            if i >= len(totals) or dens[i] <= 0:
                return 0.0
            return float(totals[i]) * 1000.0 / float(dens[i])

        for i, steps_d in enumerate(by_depth_steps):
            depth = i + 1
            print(
                f"  depth={depth}: steps={steps_d:5d}  "
                f"cycle_wall={_depth_avg_ms('by_depth_cycle_wall_avg_sec', 'by_depth_cycle_wall_sec', i):7.3f} ms  "
                f"spec={_depth_avg_ms('by_depth_spec_forward_avg_sec', 'by_depth_spec_forward_sec', i):7.3f} ms  "
                f"recv_wait={_depth_avg_ms('by_depth_recv_wait_avg_sec', 'by_depth_recv_wait_sec', i):7.3f} ms"
            )

    rb_count = int(breakdown.get("rollback_count", 0))
    if rb_count > 0:
        rb_wall_ms = _ms("rollback_wall_sec")
        rb_avg_ms = _ms("rollback_avg_sec")
        print(f"\n  --- rollback / reject ({label}) ---")
        print(f"  rollback count:      {rb_count:8d}")
        print(
            f"  rollback total:      {rb_wall_ms:8.2f} ms  "
            f"({_pct(rb_wall_ms):5.1f}% of decode wall)"
        )
        print(f"  rollback avg:        {rb_avg_ms:8.3f} ms")
        print(
            f"    verify (reject):   {_ms('rollback_verify_sec'):8.2f} ms  "
            f"({_ms('rollback_verify_sec') / rb_count:.3f} ms/call)"
        )
        print(
            f"    discard comm:      {_ms('rollback_discard_comm_sec'):8.2f} ms  "
            f"({_ms('rollback_discard_comm_sec') / rb_count:.3f} ms/call)"
        )
        print(
            f"    apply_reject:      {_ms('rollback_apply_reject_sec'):8.2f} ms  "
            f"({_ms('rollback_apply_reject_sec') / rb_count:.3f} ms/call)"
        )
    elif float(breakdown.get("verify_sec", 0.0)) > 0.0:
        print(f"\n  --- rollback / reject ({label}) ---")
        print("  rollback count:      0")


def _print_timing_report(
    *,
    prompt: str,
    verify: bool,
    profile_timing: bool,
    timing: RunTiming,
    accept: list[bool],
    attn_implementation: str,
    dist_backend: str,
    spec_cfg: dict[str, Any],
    base_model_path: str,
    spec_ckpt_path: str,
) -> None:
    steps = timing.timed.decode_steps
    n_new = timing.timed.n_new_tokens
    n_accepted = timing.timed.n_accepted
    decode_sec = timing.timed.decode_sec
    accept_rate = (n_new / steps) if steps > 0 else 0.0
    tokens_per_step = (n_new / steps) if steps > 0 else 0.0
    ms_per_step = (decode_sec * 1000.0 / steps) if steps > 0 else 0.0
    ms_per_token = (decode_sec * 1000.0 / n_new) if n_new > 0 else 0.0

    print("\n--- Multi-process pipeline generate (v11) ---")
    for line in format_ckpt_key_info_lines(
        spec_cfg,
        base_model_path=base_model_path,
        spec_ckpt_path=spec_ckpt_path,
    ):
        print(line)
    print(f"Prompt: {prompt!r}")
    print(f"verify: {verify}")
    print(f"profile_timing: {profile_timing}")
    print(f"attn_implementation: {attn_implementation}")
    print("p2p_comm: async")
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
    print(f"prefill_sec: {t.prefill_sec:.4f}")
    print(f"decode_sec: {decode_sec:.4f}")
    print(f"decode_steps: {steps}")
    print(f"new_tokens: {n_new}")
    print(f"acceptance: {n_accepted} / {len(accept)}")
    print(f"profile_timing: {profile_timing}")


if __name__ == "__main__":
    args = parse_args()

    if not str(args.spec_head_ckpt).strip():
        raise ValueError("--spec_head_ckpt is required for v11 multi-process decoding.")

    dist_init_start = time.perf_counter()
    rank_gpus = parse_rank_gpu_ids(args.rank_gpus) if args.rank_gpus.strip() else None
    rank, world_size, device = init_dist_rank_device(
        rank_gpus,
        init_timeout_minutes=max(args.prefill_timeout_sec / 60.0, 1.0),
        backend=str(args.dist_backend),
    )
    dist_init_sec = time.perf_counter() - dist_init_start

    dtype = torch.float16

    spec_cfg = load_spec_checkpoint_config(args.spec_head_ckpt)
    num_stages = int(spec_cfg["num_stages"])
    expected_ws = num_stages + 1
    if world_size != expected_ws:
        if rank == 0:
            raise ValueError(
                f"spec ckpt num_stages={num_stages} requires world_size={expected_ws}, "
                f"got {world_size}. Launch with --nproc_per_node={expected_ws}."
            )
        dist.barrier()
        raise ValueError("world_size mismatch")

    base_path = _resolve_base_model_path(spec_cfg, args.base_model_path, args.spec_head_ckpt)
    timeout = PhaseTimeout(
        prefill_sec=float(args.prefill_timeout_sec),
        decode_sec=float(args.decode_timeout_sec),
    )
    p2p = PipelineP2P(rank, world_size, device)
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
    else:
        worker_bundle = load_stage_rank_bundle(
            rank=rank,
            base_model_path=base_path,
            spec_ckpt_path=args.spec_head_ckpt,
            dtype=dtype,
            device=device,
            attn_implementation=args.attn_implementation,
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
            profile_timing=bool(args.profile_timing),
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
        with _tee_stdout_to_file(_EXAMPLE_LOG_PATH):
            _print_timing_report(
                prompt=args.prompt,
                verify=bool(args.verify),
                profile_timing=bool(args.profile_timing),
                timing=run_timing,
                accept=final_accept,
                attn_implementation=str(args.attn_implementation),
                dist_backend=str(args.dist_backend),
                spec_cfg=spec_cfg,
                base_model_path=base_path,
                spec_ckpt_path=str(args.spec_head_ckpt),
            )
            print(f"Generated text:\n{text}")

    dist.barrier()
    dist.destroy_process_group()
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 distributed_inference/example_mp_pipeline_generate.py --spec_head_ckpt Qwen3.5-4B_s4_l4.pt --no-verify
