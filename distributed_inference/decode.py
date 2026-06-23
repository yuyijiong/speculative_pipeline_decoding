"""Multi-process parallel decode loop (rank0 verify/spec || stage workers)."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch
import torch.distributed as dist

from .comm import (
    CtrlOpcode,
    PipelineP2P,
    assert_p2p_meta,
    broadcast_ctrl,
    make_ctrl_tensor,
    parse_ctrl,
)
from .device import PhaseTimeout, sync_device
from .rank0_controller import Rank0Controller
from .stage_worker import StageWorker
from .topology import rank_for_stage


@dataclass
class DecodeTimingBreakdown:
    """Per-phase decode wall time accumulated on rank 0 only (seconds)."""

    wait_comm_sec: float = 0.0
    ctrl_prepare_sec: float = 0.0
    ctrl_broadcast_sec: float = 0.0
    spec_forward_sec: float = 0.0
    spec_forward_steps: int = 0
    recv_snap_sec: float = 0.0
    recv_verify_sec: float = 0.0
    cycle_sync_sec: float = 0.0
    verify_sec: float = 0.0
    discard_comm_sec: float = 0.0
    driver_update_sec: float = 0.0
    shutdown_sec: float = 0.0
    rollback_count: int = 0
    rollback_wall_sec: float = 0.0
    rollback_verify_sec: float = 0.0
    rollback_discard_comm_sec: float = 0.0
    rollback_apply_reject_sec: float = 0.0
    decode_cycle_max_stage_sec: float = 0.0
    decode_compute_sec: float = 0.0
    stage_forward_sec: List[float] = field(default_factory=list)
    stage_forward_active_steps: List[int] = field(default_factory=list)

    @property
    def pure_comm_sec(self) -> float:
        """Small control/P2P/barrier overhead (analogous to baseline ``comm_sec``)."""
        return (
            self.wait_comm_sec
            + self.ctrl_prepare_sec
            + self.ctrl_broadcast_sec
            + self.cycle_sync_sec
            + self.discard_comm_sec
            + self.shutdown_sec
        )

    @property
    def pipeline_wait_sec(self) -> float:
        """Rank0 blocked waiting for stage-worker forward + P2P (critical path, not pure comm)."""
        return float(self.recv_snap_sec) + float(self.recv_verify_sec)

    @property
    def comm_sec(self) -> float:
        return self.pure_comm_sec

    @property
    def accounted_sec(self) -> float:
        return (
            self.pure_comm_sec
            + self.pipeline_wait_sec
            + self.spec_forward_sec
            + self.verify_sec
            + self.driver_update_sec
        )

    @property
    def rank0_gpu_sec(self) -> float:
        return (
            float(self.spec_forward_sec)
            + float(self.verify_sec)
            + float(self.driver_update_sec)
        )

    def to_dict(self) -> dict[str, Any]:
        rb_count = int(self.rollback_count)
        rb_wall = float(self.rollback_wall_sec)
        return {
            "wait_comm_sec": float(self.wait_comm_sec),
            "ctrl_prepare_sec": float(self.ctrl_prepare_sec),
            "ctrl_broadcast_sec": float(self.ctrl_broadcast_sec),
            "spec_forward_sec": float(self.spec_forward_sec),
            "spec_forward_steps": float(self.spec_forward_steps),
            "recv_snap_sec": float(self.recv_snap_sec),
            "recv_verify_sec": float(self.recv_verify_sec),
            "cycle_sync_sec": float(self.cycle_sync_sec),
            "verify_sec": float(self.verify_sec),
            "discard_comm_sec": float(self.discard_comm_sec),
            "driver_update_sec": float(self.driver_update_sec),
            "shutdown_sec": float(self.shutdown_sec),
            "pure_comm_sec": float(self.pure_comm_sec),
            "pipeline_wait_sec": float(self.pipeline_wait_sec),
            "comm_sec": float(self.pure_comm_sec),
            "rank0_gpu_sec": float(self.rank0_gpu_sec),
            "accounted_sec": float(self.accounted_sec),
            "rollback_count": float(rb_count),
            "rollback_wall_sec": rb_wall,
            "rollback_verify_sec": float(self.rollback_verify_sec),
            "rollback_discard_comm_sec": float(self.rollback_discard_comm_sec),
            "rollback_apply_reject_sec": float(self.rollback_apply_reject_sec),
            "rollback_avg_sec": float(rb_wall / rb_count) if rb_count > 0 else 0.0,
            "decode_cycle_max_stage_sec": float(self.decode_cycle_max_stage_sec),
            "decode_compute_sec": float(self.decode_compute_sec),
            "stage_forward_sec": [float(x) for x in self.stage_forward_sec],
            "stage_forward_active_steps": [int(x) for x in self.stage_forward_active_steps],
        }


def _acc_time(breakdown: DecodeTimingBreakdown, field: str, dt: float) -> None:
    setattr(breakdown, field, float(getattr(breakdown, field)) + float(dt))


def _acc_rollback(
    breakdown: DecodeTimingBreakdown,
    *,
    verify_sec: float,
    discard_comm_sec: float,
    apply_reject_sec: float,
) -> None:
    breakdown.rollback_count += 1
    _acc_time(breakdown, "rollback_verify_sec", verify_sec)
    _acc_time(breakdown, "rollback_discard_comm_sec", discard_comm_sec)
    _acc_time(breakdown, "rollback_apply_reject_sec", apply_reject_sec)
    _acc_time(
        breakdown,
        "rollback_wall_sec",
        verify_sec + discard_comm_sec + apply_reject_sec,
    )


def _reduce_max_cycle_forward(
    cycle_forward_buf: torch.Tensor,
    local_forward_sec: float,
) -> float:
    cycle_forward_buf.fill_(float(local_forward_sec))
    dist.all_reduce(cycle_forward_buf, op=dist.ReduceOp.MAX)
    return float(cycle_forward_buf.item())


def _sync_cycle_forward_timing(
    *,
    local_forward_sec: float,
    cycle_forward_buf: torch.Tensor,
    gather_buf: torch.Tensor,
) -> tuple[float, list[float]]:
    cycle_max_sec = _reduce_max_cycle_forward(cycle_forward_buf, local_forward_sec)
    local_t = torch.tensor(
        [float(local_forward_sec)], dtype=torch.float64, device=cycle_forward_buf.device
    )
    dist.all_gather_into_tensor(gather_buf, local_t)
    return cycle_max_sec, gather_buf.tolist()


def _acc_stage_forward_round(
    breakdown: DecodeTimingBreakdown,
    *,
    depth: int,
    gathered_forward_sec: list[float],
    merge_last_stage: bool = False,
    num_stages: int = 0,
) -> None:
    n = len(breakdown.stage_forward_sec)
    upper = depth
    if merge_last_stage and depth >= num_stages:
        upper = num_stages - 1
    for si in range(min(upper, n)):
        breakdown.stage_forward_sec[si] += float(gathered_forward_sec[si + 1])
        breakdown.stage_forward_active_steps[si] += 1


def _end_cycle_sync(
    worker: StageWorker,
    p2p: PipelineP2P,
    *,
    sync_mode: str,
    cycle_forward_buf: torch.Tensor | None = None,
    gather_buf: torch.Tensor | None = None,
    cycle_forward_sec: float = 0.0,
) -> float:
    cycle_max_sec = 0.0
    if cycle_forward_buf is not None and gather_buf is not None:
        cycle_max_sec, _ = _sync_cycle_forward_timing(
            local_forward_sec=cycle_forward_sec,
            cycle_forward_buf=cycle_forward_buf,
            gather_buf=gather_buf,
        )
    sync_device(worker.device)
    if sync_mode == "barrier":
        dist.barrier()
    else:
        p2p.wait_all()
    return cycle_max_sec


def run_stage_worker_loop(
    worker: StageWorker,
    p2p: PipelineP2P,
    ctrl_buf: torch.Tensor,
    *,
    hidden_size: int,
    done_flag: torch.Tensor,
    timeout: PhaseTimeout,
    sync_mode: str = "barrier",
) -> None:
    dtype = worker.b.compute_dtype
    cycle_forward_buf = torch.zeros(1, dtype=torch.float64, device=worker.device)
    gather_buf = torch.zeros(
        dist.get_world_size(), dtype=torch.float64, device=worker.device
    )
    with torch.inference_mode():
        while True:
            worker.wait_comm()
            timeout.check()
            broadcast_ctrl(ctrl_buf, src=0)
            parsed = parse_ctrl(ctrl_buf)
            op = int(parsed["opcode"])
            if op == int(CtrlOpcode.SHUTDOWN):
                dist.barrier()
                break
            cycle_id = int(parsed["cycle_id"])
            if op == int(CtrlOpcode.DISCARD):
                crop_length = int(parsed["crop_length"])
                worker.on_discard(crop_length, int(parsed["token_id"]))
                _end_cycle_sync(worker, p2p, sync_mode=sync_mode)
                worker._cycle_forward_sec = 0.0
                continue
            if op != int(CtrlOpcode.GO):
                continue

            depth = int(parsed["pipeline_depth"])
            positions = list(parsed["positions"])
            if worker.b.stage_idx == 0:
                worker.set_pending_token(int(parsed["token_id"]))

            worker.begin_cycle_forward()

            if worker.b.stage_idx < depth:
                send_verify, verify_hs, verify_pos, slot_pos, valid, snaps = worker.on_go(
                    cycle_id=cycle_id,
                    pipeline_depth=depth,
                    positions=positions,
                )
                hs_out = worker.slot_hs
                worker.post_forward_send(
                    hs_out,
                    cycle_id=cycle_id,
                    pos=slot_pos,
                    snaps=snaps,
                    send_verify=send_verify,
                    verify_pos=verify_pos,
                    pipeline_depth=depth,
                    valid=valid,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
            elif worker.b.stage_idx < worker.b.num_stages - 1:
                tail_pos = int(positions[-1]) if positions else 0
                worker.relay_invalid_hs_downstream(
                    cycle_id=cycle_id,
                    pos=tail_pos,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )

            worker.end_cycle_recv_upstream(
                hidden_size,
                dtype,
                cycle_id=cycle_id,
                pipeline_depth=depth,
                positions=positions,
            )
            _end_cycle_sync(
                worker,
                p2p,
                sync_mode=sync_mode,
                cycle_forward_buf=cycle_forward_buf,
                gather_buf=gather_buf,
                cycle_forward_sec=float(worker._cycle_forward_sec),
            )
            worker._cycle_forward_sec = 0.0


def run_rank0_decode(
    rank0: Rank0Controller,
    p2p: PipelineP2P,
    *,
    s0: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    verify: bool,
    eos_token_id: int | None,
    device: torch.device,
    done_flag: torch.Tensor,
    initial_go_token_id: int,
    timeout: PhaseTimeout,
    sync_mode: str = "barrier",
    last_stage_worker: StageWorker | None = None,
) -> Tuple[List[int], List[bool], int, dict[str, Any]]:
    decode_steps = 0
    dtype = rank0.b.lm_head.weight.dtype
    num_stages = int(rank0.n)
    next_go_token_id = int(initial_go_token_id)
    decode_wall_start = time.perf_counter()
    breakdown = DecodeTimingBreakdown(
        stage_forward_sec=[0.0] * num_stages,
        stage_forward_active_steps=[0] * num_stages,
    )
    cycle_forward_buf = torch.zeros(1, dtype=torch.float64, device=device)
    gather_buf = torch.zeros(
        dist.get_world_size(), dtype=torch.float64, device=device
    )
    use_barrier = sync_mode == "barrier"
    merge_last_stage = bool(p2p.merge_last_stage)
    if merge_last_stage:
        if last_stage_worker is None:
            raise ValueError("merge_last_stage requires last_stage_worker on rank 0")
    elif last_stage_worker is not None:
        raise ValueError("last_stage_worker is only valid when merge_last_stage=True")
    last_stage_stream = (
        torch.cuda.Stream(device=device) if merge_last_stage else None
    )

    try:
        with torch.inference_mode():
            while rank0.verified_up_to - s0 < max_new_tokens:
                t0 = time.perf_counter()
                p2p.wait_all()
                _acc_time(breakdown, "wait_comm_sec", time.perf_counter() - t0)

                timeout.check()
                decode_steps += 1
                t0 = time.perf_counter()
                rank0.cycle_id += 1
                cycle_id = rank0.cycle_id
                depth = len(rank0.pipeline)
                positions = rank0.pipeline_positions()
                newest_pos0 = int(rank0.pipeline[0]["pos"]) if rank0.pipeline else 0
                oldest_needed0 = newest_pos0 - rank0.n + 1

                ctrl = make_ctrl_tensor(
                    opcode=CtrlOpcode.GO,
                    cycle_id=cycle_id,
                    pipeline_depth=depth,
                    positions=positions,
                    token_id=next_go_token_id,
                    inject_pos=rank0.next_position,
                    device=device,
                )
                _acc_time(breakdown, "ctrl_prepare_sec", time.perf_counter() - t0)

                t0 = time.perf_counter()
                broadcast_ctrl(ctrl, src=0)
                _acc_time(breakdown, "ctrl_broadcast_sec", time.perf_counter() - t0)

                # Workers start stage forward on other GPUs as soon as they receive GO.
                # Rank0 speculation runs here in parallel with stage compute.
                # Reuse pending_spec_logits after commit/pop (v11): inputs are valid at cycle start.
                pending_spec_logits = None
                cycle_spec_sec = 0.0
                cycle_last_stage_sec = 0.0
                local_last_snaps: dict[int, torch.Tensor] = {}
                local_verify_hs = None
                local_verify_pos = -1
                if merge_last_stage and last_stage_worker is not None:
                    last_stage_worker.begin_cycle_forward()

                if oldest_needed0 >= 0:
                    t0 = time.perf_counter()
                    pending_spec_logits = rank0.run_spec_forward(cycle_id)
                    cycle_spec_sec = time.perf_counter() - t0
                    _acc_time(breakdown, "spec_forward_sec", cycle_spec_sec)
                    breakdown.spec_forward_steps += 1

                if merge_last_stage and depth == rank0.n:
                    assert last_stage_worker is not None
                    t0 = time.perf_counter()
                    ctx = (
                        torch.cuda.stream(last_stage_stream)
                        if last_stage_stream is not None
                        else nullcontext()
                    )
                    with ctx:
                        _sv, local_verify_hs, local_verify_pos, _pos, _valid, local_last_snaps = (
                            last_stage_worker.on_go(
                                cycle_id=cycle_id,
                                pipeline_depth=depth,
                                positions=positions,
                            )
                        )
                    cycle_last_stage_sec = time.perf_counter() - t0
                    breakdown.stage_forward_sec[num_stages - 1] += cycle_last_stage_sec
                    breakdown.stage_forward_active_steps[num_stages - 1] += 1

                t0 = time.perf_counter()
                for stage_idx in range(depth):
                    if merge_last_stage and stage_idx == num_stages - 1:
                        continue
                    timeout.check()
                    src = rank_for_stage(
                        stage_idx, num_stages, merge_last_stage=merge_last_stage
                    )
                    expected_spos = int(positions[stage_idx])
                    recv_c, spos, valid, shards = p2p.recv_snap_batch(
                        rank0.b.hidden_size, dtype, src=src, max_indices=16
                    )
                    assert_p2p_meta(
                        "snap_batch",
                        cycle_id=recv_c,
                        expected_cycle_id=cycle_id,
                        token_pos=spos,
                        expected_token_pos=expected_spos,
                        peer_rank=src,
                        local_rank=0,
                    )
                    if valid:
                        rank0.merge_snap_batch(recv_c, spos, shards)
                        for e in rank0.pipeline:
                            if int(e["pos"]) == int(spos):
                                e["snap"].update(shards)
                _acc_time(breakdown, "recv_snap_sec", time.perf_counter() - t0)

                if merge_last_stage and last_stage_worker is not None:
                    if depth == rank0.n and local_last_snaps:
                        spos = int(positions[-1])
                        rank0.merge_snap_batch(cycle_id, spos, local_last_snaps)
                        for e in rank0.pipeline:
                            if int(e["pos"]) == int(spos):
                                e["snap"].update(local_last_snaps)
                    if last_stage_stream is not None:
                        torch.cuda.current_stream(device=device).wait_stream(
                            last_stage_stream
                        )
                    last_stage_worker.end_cycle_recv_upstream(
                        rank0.b.hidden_size,
                        dtype,
                        cycle_id=cycle_id,
                        pipeline_depth=depth,
                        positions=positions,
                    )

                verify_hs = None
                verify_pos = -1
                if depth == rank0.n:
                    expected_verify_pos = int(positions[-1])
                    if merge_last_stage:
                        verify_hs = local_verify_hs
                        verify_pos = local_verify_pos
                    else:
                        t0 = time.perf_counter()
                        verify_hs, recv_c, verify_pos = p2p.recv_verify_hs(
                            rank0.b.hidden_size, dtype
                        )
                        assert_p2p_meta(
                            "verify_hs",
                            cycle_id=recv_c,
                            expected_cycle_id=cycle_id,
                            token_pos=verify_pos,
                            expected_token_pos=expected_verify_pos,
                            peer_rank=p2p.num_stages,
                            local_rank=0,
                        )
                        _acc_time(breakdown, "recv_verify_sec", time.perf_counter() - t0)

                t0 = time.perf_counter()
                rank0_local_fwd = float(cycle_spec_sec)
                if merge_last_stage and cycle_last_stage_sec > 0.0:
                    rank0_local_fwd = max(rank0_local_fwd, float(cycle_last_stage_sec))
                cycle_max_stage_sec, gathered_forward_sec = _sync_cycle_forward_timing(
                    local_forward_sec=rank0_local_fwd,
                    cycle_forward_buf=cycle_forward_buf,
                    gather_buf=gather_buf,
                )
                _acc_stage_forward_round(
                    breakdown,
                    depth=depth,
                    gathered_forward_sec=gathered_forward_sec,
                    merge_last_stage=merge_last_stage,
                    num_stages=num_stages,
                )
                sync_device(device)
                if use_barrier:
                    dist.barrier()
                else:
                    p2p.wait_all()
                _acc_time(breakdown, "cycle_sync_sec", time.perf_counter() - t0)
                _acc_time(
                    breakdown, "decode_cycle_max_stage_sec", cycle_max_stage_sec
                )
                _acc_time(breakdown, "decode_compute_sec", cycle_max_stage_sec)

                t0 = time.perf_counter()
                cycle_verify_sec = 0.0
                if depth >= rank0.n:
                    completed_pos = int(positions[-1])
                    rank0.commit_completed_snap(completed_pos, cycle_id)
                    target_pos = completed_pos + 1
                    crop_length = target_pos
                    target_gen_idx = target_pos - s0

                    if target_gen_idx < len(rank0.generated_ids):
                        speculated_id = int(rank0.generated_ids[target_gen_idx])
                        if verify and verify_hs is not None:
                            t_verify = time.perf_counter()
                            accepted, verified_next_id = rank0.verify_with_hs(
                                verify_hs,
                                target_pos,
                                speculated_id,
                                greedy=greedy,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                            cycle_verify_sec = time.perf_counter() - t_verify
                            _acc_time(breakdown, "verify_sec", cycle_verify_sec)
                            if not accepted:
                                t_rollback = time.perf_counter()
                                verify_dt = t_rollback - t_verify
                                rank0.cycle_id += 1
                                next_go_token_id = int(verified_next_id)
                                discard = make_ctrl_tensor(
                                    opcode=CtrlOpcode.DISCARD,
                                    cycle_id=rank0.cycle_id,
                                    crop_length=crop_length,
                                    token_id=next_go_token_id,
                                    verify_pos=target_pos,
                                    inject_pos=target_pos,
                                    device=device,
                                )
                                t_discard = time.perf_counter()
                                broadcast_ctrl(discard, src=0)
                                t_apply = time.perf_counter()
                                rank0.apply_reject(crop_length, verified_next_id, s0)
                                if last_stage_worker is not None:
                                    last_stage_worker.crop_kv_after_reject(crop_length)
                                t_after_apply = time.perf_counter()
                                if use_barrier:
                                    dist.barrier()
                                else:
                                    p2p.wait_all()
                                discard_dt = time.perf_counter() - t_discard
                                apply_dt = t_after_apply - t_apply
                                _acc_time(
                                    breakdown, "discard_comm_sec", discard_dt
                                )
                                _acc_rollback(
                                    breakdown,
                                    verify_sec=verify_dt,
                                    discard_comm_sec=discard_dt - apply_dt,
                                    apply_reject_sec=apply_dt,
                                )
                                _acc_time(
                                    breakdown, "driver_update_sec", time.perf_counter() - t0
                                )
                                _acc_time(
                                    breakdown, "decode_compute_sec", cycle_verify_sec
                                )
                                if verified_next_id == eos_token_id:
                                    break
                                continue

                        rank0.verified_up_to = target_pos + 1
                        if speculated_id == eos_token_id:
                            _acc_time(breakdown, "driver_update_sec", time.perf_counter() - t0)
                            _acc_time(
                                breakdown, "decode_compute_sec", cycle_verify_sec
                            )
                            break

                    ev_snap = dict(rank0.completed_snaps[completed_pos])
                    rank0.prev_evicted_snap = ev_snap
                    rank0.prev_evicted_pos = completed_pos
                    rank0.pipeline.pop()

                if pending_spec_logits is None:
                    sync_device(device)
                    t_spec = time.perf_counter()
                    pending_spec_logits = rank0.run_spec_forward(cycle_id)
                    sync_device(device)
                    _acc_time(breakdown, "spec_forward_sec", time.perf_counter() - t_spec)
                    breakdown.spec_forward_steps += 1

                next_id = rank0.sample_spec_token(
                    pending_spec_logits,
                    rank0.next_position,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                rank0.generated_ids.append(next_id)
                rank0.token_acceptance.append(True)
                emb = rank0.b.embed_tokens(torch.tensor([[next_id]], device=device))
                rank0.pipeline.insert(
                    0,
                    {"hs": emb, "pos": rank0.next_position, "snap": rank0._initial_snap(emb)},
                )
                rank0.ingest_g0_snap(rank0.next_position, next_id, cycle_id)
                rank0.next_position += 1
                next_go_token_id = int(next_id)
                _acc_time(breakdown, "driver_update_sec", time.perf_counter() - t0)
                if cycle_verify_sec > 0.0:
                    _acc_time(breakdown, "decode_compute_sec", cycle_verify_sec)
    finally:
        t0 = time.perf_counter()
        shutdown = make_ctrl_tensor(
            opcode=CtrlOpcode.SHUTDOWN, cycle_id=rank0.cycle_id, device=device
        )
        broadcast_ctrl(shutdown, src=0)
        dist.barrier()
        done_flag.fill_(1)
        _acc_time(breakdown, "shutdown_sec", time.perf_counter() - t0)

    decode_wall_sec = time.perf_counter() - decode_wall_start
    timing: dict[str, Any] = {
        "decode_wall_sec": float(decode_wall_sec),
        "decode_steps": float(decode_steps),
        "num_stages": float(num_stages),
    }
    timing.update(breakdown.to_dict())
    timing["unaccounted_sec"] = float(max(decode_wall_sec - breakdown.accounted_sec, 0.0))
    timing["decode_compute_sec"] = float(breakdown.decode_compute_sec)
    rank0.last_timing = timing
    return (
        rank0.generated_ids[:max_new_tokens],
        rank0.token_acceptance[:max_new_tokens],
        decode_steps,
        timing,
    )
