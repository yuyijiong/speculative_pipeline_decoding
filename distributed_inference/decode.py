"""Multi-process parallel decode loop (rank0 verify/spec || stage workers)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

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
from .loader import snap_indices_produced_on_stage
from .rank0_controller import Rank0Controller
from .stage_worker import StageWorker
from .topology import rank_for_stage


# Contiguous rank-0 wall phases inside one decode cycle. Their sum equals
# ``cycle_wall_sec`` (plus ``cycle_other_sec`` for any residual gap).
# Spec runs serially on the default stream after posting recv (no second CUDA
# stream): post_recv → spec_forward → recv_verify → …
# Fill may place ``recv_snap_sec`` right after ``ctrl_broadcast`` when draining
# snaps deferred from the previous fill cycle (overlap with stage forward).
CYCLE_ADDITIVE_PHASES: tuple[str, ...] = (
    "ctrl_prepare_sec",
    "ctrl_broadcast_sec",
    "post_recv_sec",
    "spec_forward_sec",
    "recv_verify_sec",
    "snap_progress_sec",
    "verify_sec",
    "recv_snap_sec",
    "driver_update_sec",
    "cycle_sync_sec",
    "discard_comm_sec",
    "cycle_other_sec",
)


@dataclass
class _DeferredSnaps:
    """Snap irecvs deferred until after the next GO broadcast (overlap stage forward)."""

    origin_depth: int
    items: List[Tuple[int, Any, int, int]]  # stage_idx, recv, cycle_id, token_pos


def _merge_snap_into_pipeline(
    rank0: Rank0Controller,
    recv_c: int,
    spos: int,
    shards: dict[int, torch.Tensor],
) -> None:
    rank0.merge_snap_batch(recv_c, spos, shards)
    for e in rank0.pipeline:
        if int(e["pos"]) == int(spos):
            e["snap"].update(shards)


def _stage_snap_indices(rank0: Rank0Controller, stage_idx: int) -> list[int]:
    return sorted(
        snap_indices_produced_on_stage(
            rank0.b.snap_want,
            int(stage_idx),
            rank0.b.stage_layer_ranges,
        )
    )


def _wait_merge_snap_recv(
    rank0: Rank0Controller,
    *,
    stage_idx: int,
    snap_recv: Any,
    expected_cycle_id: int,
    expected_pos: int,
    num_stages: int,
    timeout: PhaseTimeout,
) -> None:
    del stage_idx, num_stages  # peer identity is baked into the posted recv slot
    timeout.check()
    recv_c, spos, valid, shards = snap_recv.wait()
    if int(recv_c) != int(expected_cycle_id) or int(spos) != int(expected_pos):
        raise RuntimeError(
            f"P2P fixed-snap slot mismatch: cycle/pos=({recv_c},{spos}) != "
            f"expected ({expected_cycle_id},{expected_pos})"
        )
    if valid:
        _merge_snap_into_pipeline(rank0, recv_c, spos, shards)


@dataclass
class DecodeTimingBreakdown:
    """Per-phase decode timers accumulated on rank 0 (seconds).

    Two views are exported (do not mix them):

    1. **Additive cycle wall** (contiguous host timeline, no overlap):
       phases in ``CYCLE_ADDITIVE_PHASES``. Per cycle,
       ``sum(phases) == cycle_wall`` (``cycle_other`` absorbs residual).
       ``rank0_sequential_sec`` is the sum of these phases over all cycles
       (plus ``shutdown_sec`` outside the loop). Spec is **serial** on the
       default stream: ``post_recv → spec_forward → recv_verify``.

    2. **Remote stage compute** (worker gather): ``stage_forward_sec[]``.
       Spec wall overlaps remote stage compute while recv is posted; after
       spec finishes, ``recv_verify`` only waits for the remaining stage work.

    Per-step averages must use each phase's ``*_steps`` / ``*_active_steps``
    denominator (empty / skipped cycles are excluded), not a single global
    ``spec_forward_steps`` for every key.

    Fill-sensitive timers (``recv_wait``, ``cycle_wall``, spec vs remaining recv)
    are also accumulated **per pipeline depth** (fill level 1..n) in the
    ``by_depth_*`` lists (index ``depth - 1``).

    Host diagnostics:
    - ``spec_forward_sec``: serial speculation on the default stream, timed with
      CUDA events + ``event.synchronize()`` (stream-scoped; does **not** drain
      NCCL recv streams posted earlier in the cycle).
    - ``post_recv_sec``: posting fixed early-stage snap ``irecv``s and last-stage
      ``verify_hs`` + snap ``irecv``s (no meta; only active stages; pooled buffers).
    - ``verify_kernel_sec`` / ``verify_copy_sec`` / ``verify_decide_sec``:
      contiguous host-wall split of ``verify_sec`` (``perf_counter``;
      kernel uses **stream-scoped** sync only — never device-wide sync, which
      would incorrectly absorb in-flight snap ``irecv`` wait). ``copy +
      kernel + decide ≈ verify``.
    - ``recv_verify_sec``: blocking wait until ``verify_hs`` is consumable on
      rank-0 (NCCL ``work.wait()`` plus, when profiling, a sync copy into the
      verify graph input buffer). Does **not** include snap ``irecv`` wait.
    - ``snap_progress_sec``: lightweight verify prep after ``recv_verify``
      (draft-id lookup). All snap/verify payload ``irecv``s are already posted
      in ``post_recv``.
    - ``recv_snap_sec`` (fill / accept): blocking wait for non-last-stage snaps is
      **deferred** until after the next GO broadcast. Stages do not need rank0 to
      finish those snap recvs before the next forward. The wait overlaps stage
      forward and should be near-zero; by_depth attributes it to ``origin_depth``.
      Fill ``cycle_wall`` should approach ``max_stage``. On full-pipe **accept**,
      only the last-stage snap is waited in-cycle (needed for
      ``commit_completed_snap``); early-stage snaps defer so rank0 can sync/GO
      immediately after verify+driver. **Reject** still waits all snaps in-cycle.
    - ``stage0_hs_send_sec`` / ``last_stage_hs_send_sec``: worker gather of
      stage0→stage1 hs send vs last-stage verify_hs send (intra- vs inter-node).
    """

    wait_comm_sec: float = 0.0
    wait_comm_steps: int = 0
    ctrl_prepare_sec: float = 0.0
    ctrl_prepare_steps: int = 0
    ctrl_broadcast_sec: float = 0.0
    ctrl_broadcast_steps: int = 0
    post_recv_sec: float = 0.0
    post_recv_steps: int = 0
    cycle_other_sec: float = 0.0
    cycle_other_steps: int = 0
    spec_forward_sec: float = 0.0
    spec_forward_steps: int = 0
    # Worker-gathered hs send walls (filled at shutdown).
    stage0_hs_send_sec: float = 0.0
    stage0_hs_send_steps: int = 0
    last_stage_hs_send_sec: float = 0.0
    last_stage_hs_send_steps: int = 0
    recv_snap_sec: float = 0.0
    recv_snap_steps: int = 0
    recv_verify_sec: float = 0.0
    recv_verify_steps: int = 0
    # Lightweight verify prep between recv_verify and verify (draft-id lookup).
    snap_progress_sec: float = 0.0
    snap_progress_steps: int = 0
    cycle_sync_sec: float = 0.0
    cycle_sync_steps: int = 0
    verify_sec: float = 0.0
    verify_steps: int = 0
    # Verify sub-breakdown: host-wall split of verify_sec (additive).
    verify_copy_sec: float = 0.0
    verify_copy_steps: int = 0
    verify_kernel_sec: float = 0.0
    verify_kernel_steps: int = 0
    verify_decide_sec: float = 0.0
    verify_decide_steps: int = 0
    verify_graph_steps: int = 0
    discard_comm_sec: float = 0.0
    discard_comm_steps: int = 0
    driver_update_sec: float = 0.0
    driver_update_steps: int = 0
    shutdown_sec: float = 0.0
    shutdown_steps: int = 0
    cycle_wall_sec: float = 0.0
    cycle_wall_steps: int = 0
    # Steady-state (full pipeline) recv-wait diagnostics.
    full_pipeline_steps: int = 0
    full_pipeline_recv_wait_sec: float = 0.0
    # Per fill level (depth=1..n); index = depth - 1.
    by_depth_steps: List[int] = field(default_factory=list)
    by_depth_recv_wait_sec: List[float] = field(default_factory=list)
    by_depth_recv_snap_sec: List[float] = field(default_factory=list)
    by_depth_recv_verify_sec: List[float] = field(default_factory=list)
    by_depth_cycle_wall_sec: List[float] = field(default_factory=list)
    by_depth_spec_forward_sec: List[float] = field(default_factory=list)
    by_depth_cycle_other_sec: List[float] = field(default_factory=list)
    rollback_count: int = 0
    rollback_wall_sec: float = 0.0
    rollback_verify_sec: float = 0.0
    rollback_discard_comm_sec: float = 0.0
    rollback_apply_reject_sec: float = 0.0
    # Filled at shutdown from worker gather (stage forwards only; excludes spec).
    max_stage_forward_sec: float = 0.0
    max_stage_forward_avg_sec: float = 0.0
    # Alias kept for older summaries: same as max_stage_forward_sec.
    decode_cycle_max_stage_sec: float = 0.0
    # Critical-path compute estimate: max_stage_forward + verify (not wall partition).
    critical_path_compute_sec: float = 0.0
    stage_forward_sec: List[float] = field(default_factory=list)
    stage_forward_active_steps: List[int] = field(default_factory=list)

    @property
    def ctrl_overhead_sec(self) -> float:
        """GO ctrl tensor build + broadcast."""
        return float(self.ctrl_prepare_sec) + float(self.ctrl_broadcast_sec)

    @property
    def sync_overhead_sec(self) -> float:
        """End-of-cycle P2P ``wait_all`` plus shutdown barrier."""
        return float(self.cycle_sync_sec) + float(self.shutdown_sec)

    @property
    def pure_comm_sec(self) -> float:
        """Control / sync / discard overhead (excludes stage-recv wait / post_recv)."""
        return (
            self.ctrl_overhead_sec
            + self.sync_overhead_sec
            + float(self.discard_comm_sec)
        )

    @property
    def rank0_recv_wait_sec(self) -> float:
        """Rank0 blocked on snap/verify P2P (mostly remote stage compute + transfer)."""
        return float(self.recv_snap_sec) + float(self.recv_verify_sec)

    @property
    def pipeline_wait_sec(self) -> float:
        """Alias of ``rank0_recv_wait_sec`` (historical name)."""
        return self.rank0_recv_wait_sec

    @property
    def rank0_local_compute_sec(self) -> float:
        """Rank0 local work: spec (GPU events) + verify + driver (wall). Not a GPU sum."""
        return (
            float(self.spec_forward_sec)
            + float(self.verify_sec)
            + float(self.driver_update_sec)
        )

    @property
    def rank0_gpu_sec(self) -> float:
        """Deprecated alias of ``rank0_local_compute_sec``."""
        return self.rank0_local_compute_sec

    @property
    def cycle_phases_sec(self) -> float:
        """Sum of contiguous in-cycle phases (should equal ``cycle_wall_sec``)."""
        return float(sum(float(getattr(self, k)) for k in CYCLE_ADDITIVE_PHASES))

    @property
    def rank0_sequential_sec(self) -> float:
        """Additive rank-0 wall: all cycle phases + shutdown (excludes GPU ``spec_forward``)."""
        return self.cycle_phases_sec + float(self.shutdown_sec)

    @property
    def decode_compute_sec(self) -> float:
        """Alias of ``critical_path_compute_sec`` for older eval columns."""
        return float(self.critical_path_compute_sec)

    def to_dict(self) -> dict[str, Any]:
        rb_count = int(self.rollback_count)
        rb_wall = float(self.rollback_wall_sec)
        max_stage = float(self.max_stage_forward_sec)
        full_steps = int(self.full_pipeline_steps)

        def _avg(total: float, steps: int) -> float:
            return float(total) / float(steps) if steps > 0 else 0.0

        def _avg_list(totals: List[float], steps: List[int]) -> List[float]:
            n = max(len(totals), len(steps))
            out: List[float] = []
            for i in range(n):
                tot = float(totals[i]) if i < len(totals) else 0.0
                cnt = int(steps[i]) if i < len(steps) else 0
                out.append(_avg(tot, cnt))
            return out

        by_depth_steps = [int(x) for x in self.by_depth_steps]
        by_depth = {
            "by_depth_steps": by_depth_steps,
            "by_depth_recv_wait_sec": [float(x) for x in self.by_depth_recv_wait_sec],
            "by_depth_recv_snap_sec": [float(x) for x in self.by_depth_recv_snap_sec],
            "by_depth_recv_verify_sec": [
                float(x) for x in self.by_depth_recv_verify_sec
            ],
            "by_depth_cycle_wall_sec": [float(x) for x in self.by_depth_cycle_wall_sec],
            "by_depth_spec_forward_sec": [
                float(x) for x in self.by_depth_spec_forward_sec
            ],
            "by_depth_cycle_other_sec": [
                float(x) for x in self.by_depth_cycle_other_sec
            ],
            "by_depth_recv_wait_avg_sec": _avg_list(
                self.by_depth_recv_wait_sec, by_depth_steps
            ),
            "by_depth_recv_snap_avg_sec": _avg_list(
                self.by_depth_recv_snap_sec, by_depth_steps
            ),
            "by_depth_recv_verify_avg_sec": _avg_list(
                self.by_depth_recv_verify_sec, by_depth_steps
            ),
            "by_depth_cycle_wall_avg_sec": _avg_list(
                self.by_depth_cycle_wall_sec, by_depth_steps
            ),
            "by_depth_spec_forward_avg_sec": _avg_list(
                self.by_depth_spec_forward_sec, by_depth_steps
            ),
            "by_depth_cycle_other_avg_sec": _avg_list(
                self.by_depth_cycle_other_sec, by_depth_steps
            ),
        }

        cycle_phases = float(self.cycle_phases_sec)
        cycle_wall = float(self.cycle_wall_sec)
        return {
            # --- atomic rank-0 timers (additive cycle phases) ---
            "wait_comm_sec": float(self.wait_comm_sec),
            "wait_comm_steps": float(self.wait_comm_steps),
            "ctrl_prepare_sec": float(self.ctrl_prepare_sec),
            "ctrl_prepare_steps": float(self.ctrl_prepare_steps),
            "ctrl_broadcast_sec": float(self.ctrl_broadcast_sec),
            "ctrl_broadcast_steps": float(self.ctrl_broadcast_steps),
            "post_recv_sec": float(self.post_recv_sec),
            "post_recv_steps": float(self.post_recv_steps),
            "cycle_other_sec": float(self.cycle_other_sec),
            "cycle_other_steps": float(self.cycle_other_steps),
            "spec_forward_sec": float(self.spec_forward_sec),
            "spec_forward_steps": float(self.spec_forward_steps),
            "stage0_hs_send_sec": float(self.stage0_hs_send_sec),
            "stage0_hs_send_steps": float(self.stage0_hs_send_steps),
            "last_stage_hs_send_sec": float(self.last_stage_hs_send_sec),
            "last_stage_hs_send_steps": float(self.last_stage_hs_send_steps),
            "recv_snap_sec": float(self.recv_snap_sec),
            "recv_snap_steps": float(self.recv_snap_steps),
            "recv_verify_sec": float(self.recv_verify_sec),
            "recv_verify_steps": float(self.recv_verify_steps),
            "snap_progress_sec": float(self.snap_progress_sec),
            "snap_progress_steps": float(self.snap_progress_steps),
            "cycle_sync_sec": float(self.cycle_sync_sec),
            "cycle_sync_steps": float(self.cycle_sync_steps),
            "verify_sec": float(self.verify_sec),
            "verify_steps": float(self.verify_steps),
            "verify_copy_sec": float(self.verify_copy_sec),
            "verify_copy_steps": float(self.verify_copy_steps),
            "verify_kernel_sec": float(self.verify_kernel_sec),
            "verify_kernel_steps": float(self.verify_kernel_steps),
            "verify_decide_sec": float(self.verify_decide_sec),
            "verify_decide_steps": float(self.verify_decide_steps),
            "verify_graph_steps": float(self.verify_graph_steps),
            "discard_comm_sec": float(self.discard_comm_sec),
            "discard_comm_steps": float(self.discard_comm_steps),
            "driver_update_sec": float(self.driver_update_sec),
            "driver_update_steps": float(self.driver_update_steps),
            "shutdown_sec": float(self.shutdown_sec),
            "shutdown_steps": float(self.shutdown_steps),
            "cycle_wall_sec": cycle_wall,
            "cycle_wall_steps": float(self.cycle_wall_steps),
            "cycle_phases_sec": cycle_phases,
            "cycle_phases_gap_sec": float(max(cycle_wall - cycle_phases, 0.0)),
            "full_pipeline_steps": float(full_steps),
            # --- derived buckets (totals) ---
            "ctrl_overhead_sec": float(self.ctrl_overhead_sec),
            "sync_overhead_sec": float(self.sync_overhead_sec),
            "pure_comm_sec": float(self.pure_comm_sec),
            "rank0_recv_wait_sec": float(self.rank0_recv_wait_sec),
            "pipeline_wait_sec": float(self.pipeline_wait_sec),
            "rank0_local_compute_sec": float(self.rank0_local_compute_sec),
            "rank0_gpu_sec": float(self.rank0_local_compute_sec),
            "rank0_sequential_sec": float(self.rank0_sequential_sec),
            # --- per-active-step averages (empty steps excluded) ---
            "wait_comm_avg_sec": _avg(self.wait_comm_sec, self.wait_comm_steps),
            "ctrl_prepare_avg_sec": _avg(self.ctrl_prepare_sec, self.ctrl_prepare_steps),
            "ctrl_broadcast_avg_sec": _avg(
                self.ctrl_broadcast_sec, self.ctrl_broadcast_steps
            ),
            "post_recv_avg_sec": _avg(self.post_recv_sec, self.post_recv_steps),
            "cycle_other_avg_sec": _avg(self.cycle_other_sec, self.cycle_other_steps),
            "spec_forward_avg_sec": _avg(self.spec_forward_sec, self.spec_forward_steps),
            "stage0_hs_send_avg_sec": _avg(
                self.stage0_hs_send_sec, self.stage0_hs_send_steps
            ),
            "last_stage_hs_send_avg_sec": _avg(
                self.last_stage_hs_send_sec, self.last_stage_hs_send_steps
            ),
            "recv_snap_avg_sec": _avg(self.recv_snap_sec, self.recv_snap_steps),
            "recv_verify_avg_sec": _avg(self.recv_verify_sec, self.recv_verify_steps),
            "snap_progress_avg_sec": _avg(
                self.snap_progress_sec, self.snap_progress_steps
            ),
            "cycle_sync_avg_sec": _avg(self.cycle_sync_sec, self.cycle_sync_steps),
            "verify_avg_sec": _avg(self.verify_sec, self.verify_steps),
            "verify_copy_avg_sec": _avg(self.verify_copy_sec, self.verify_copy_steps),
            "verify_kernel_avg_sec": _avg(
                self.verify_kernel_sec, self.verify_kernel_steps
            ),
            "verify_decide_avg_sec": _avg(
                self.verify_decide_sec, self.verify_decide_steps
            ),
            # Wall beyond pure GPU kernel (copy + decide + host/sync gap).
            "verify_non_kernel_avg_sec": (
                _avg(self.verify_sec, self.verify_steps)
                - _avg(self.verify_kernel_sec, self.verify_kernel_steps)
            ),
            "discard_comm_avg_sec": _avg(self.discard_comm_sec, self.discard_comm_steps),
            "driver_update_avg_sec": _avg(
                self.driver_update_sec, self.driver_update_steps
            ),
            "cycle_wall_avg_sec": _avg(self.cycle_wall_sec, self.cycle_wall_steps),
            "cycle_phases_avg_sec": _avg(cycle_phases, int(self.cycle_wall_steps)),
            # Steady-state mean of (snap+verify) wait on full-pipeline cycles only.
            "full_pipeline_recv_wait_sec": float(self.full_pipeline_recv_wait_sec),
            "rank0_recv_wait_avg_sec": _avg(
                self.full_pipeline_recv_wait_sec, full_steps
            ),
            "pipeline_wait_avg_sec": _avg(
                self.full_pipeline_recv_wait_sec, full_steps
            ),
            # --- stage gather (shutdown) ---
            "max_stage_forward_sec": max_stage,
            "max_stage_forward_avg_sec": float(self.max_stage_forward_avg_sec),
            "decode_cycle_max_stage_sec": max_stage,
            "critical_path_compute_sec": float(self.critical_path_compute_sec),
            "decode_compute_sec": float(self.critical_path_compute_sec),
            # critical_path avg ≈ max_stage_avg + verify_avg (active dens).
            "critical_path_compute_avg_sec": float(self.max_stage_forward_avg_sec)
            + _avg(self.verify_sec, self.verify_steps),
            # --- rollback ---
            "rollback_count": float(rb_count),
            "rollback_wall_sec": rb_wall,
            "rollback_verify_sec": float(self.rollback_verify_sec),
            "rollback_discard_comm_sec": float(self.rollback_discard_comm_sec),
            "rollback_apply_reject_sec": float(self.rollback_apply_reject_sec),
            "rollback_avg_sec": float(rb_wall / rb_count) if rb_count > 0 else 0.0,
            "stage_forward_sec": [float(x) for x in self.stage_forward_sec],
            "stage_forward_active_steps": [int(x) for x in self.stage_forward_active_steps],
            **by_depth,
        }


def _acc_time(breakdown: DecodeTimingBreakdown, field: str, dt: float) -> None:
    setattr(breakdown, field, float(getattr(breakdown, field)) + float(dt))


def _acc_phase(
    breakdown: DecodeTimingBreakdown,
    field: str,
    dt: float,
    *,
    steps_field: str | None = None,
) -> None:
    """Accumulate a phase timer and bump its active-step counter (skip empty)."""
    _acc_time(breakdown, field, dt)
    count_field = steps_field or field.replace("_sec", "_steps")
    if hasattr(breakdown, count_field):
        setattr(breakdown, count_field, int(getattr(breakdown, count_field)) + 1)


def _acc_by_depth(
    breakdown: DecodeTimingBreakdown,
    field: str,
    depth: int,
    dt: float,
) -> None:
    """Accumulate ``dt`` into ``by_depth_*[depth-1]`` (depth is fill level 1..n)."""
    if depth <= 0:
        return
    lst = getattr(breakdown, field)
    idx = int(depth) - 1
    while len(lst) <= idx:
        lst.append(0 if field == "by_depth_steps" else 0.0)
    if field == "by_depth_steps":
        lst[idx] = int(lst[idx]) + 1
    else:
        lst[idx] = float(lst[idx]) + float(dt)


class _CycleClock:
    """Contiguous host-wall markers so phase deltas tile a single cycle.

    Usage::

        clk = _CycleClock.start(enabled=profile_timing)
        ... work A ...
        clk.mark("ctrl_prepare_sec")   # accumulates A
        ... work B ...
        clk.mark("ctrl_broadcast_sec")
        wall = clk.finish(breakdown, depth)  # residual → cycle_other + cycle_wall
    """

    __slots__ = ("t0", "mark_t", "phases", "enabled")

    def __init__(self, *, enabled: bool) -> None:
        now = time.perf_counter()
        self.t0 = now
        self.mark_t = now
        self.phases: dict[str, float] = {k: 0.0 for k in CYCLE_ADDITIVE_PHASES}
        self.enabled = enabled

    @classmethod
    def start(cls, *, enabled: bool) -> "_CycleClock":
        return cls(enabled=enabled)

    def mark(self, field: str) -> float:
        """Close the open interval into ``field``; return its delta."""
        now = time.perf_counter()
        dt = now - self.mark_t
        self.mark_t = now
        if self.enabled:
            if field not in self.phases:
                raise KeyError(f"unknown cycle phase: {field}")
            self.phases[field] = float(self.phases[field]) + float(dt)
        return dt

    def finish(
        self,
        breakdown: DecodeTimingBreakdown,
        depth: int,
    ) -> float:
        """Flush residual to ``cycle_other``, accumulate all phases + cycle_wall."""
        if not self.enabled:
            return 0.0
        # Anything still open (Python between last mark and finish) → other.
        self.mark("cycle_other_sec")
        wall = time.perf_counter() - self.t0
        # Tiny float drift: fold wall − sum(phases) into cycle_other.
        phases_sum = float(sum(self.phases.values()))
        drift = wall - phases_sum
        if abs(drift) > 1e-12:
            self.phases["cycle_other_sec"] = float(
                self.phases["cycle_other_sec"]
            ) + float(drift)
        for field, dt in self.phases.items():
            if dt > 1e-12:
                _acc_phase(breakdown, field, dt)
        _acc_phase(breakdown, "cycle_wall_sec", wall)
        if depth > 0:
            _acc_by_depth(breakdown, "by_depth_cycle_wall_sec", depth, wall)
            _acc_by_depth(
                breakdown,
                "by_depth_spec_forward_sec",
                depth,
                self.phases.get("spec_forward_sec", 0.0),
            )
            _acc_by_depth(
                breakdown,
                "by_depth_cycle_other_sec",
                depth,
                self.phases.get("cycle_other_sec", 0.0),
            )
        return wall


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


def _profile_layer_rows(
    *,
    layer_sec: list[float],
    layer_count: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, total_sec in enumerate(layer_sec):
        count = int(layer_count[layer_idx]) if layer_idx < len(layer_count) else 0
        if count <= 0:
            continue
        rows.append(
            {
                "layer_idx": int(layer_idx),
                "avg_sec": float(total_sec) / float(count),
                "total_sec": float(total_sec),
                "count": count,
            }
        )
    return rows


def _collect_profile_timing(
    *,
    worker: StageWorker | None,
    device: torch.device,
    num_stages: int,
    num_layers: int,
    rank0_extra_forward_sec: float = 0.0,
) -> dict[str, Any]:
    sync_device(device)
    stage_sec = torch.zeros(int(num_stages), dtype=torch.float64, device=device)
    stage_count = torch.zeros(int(num_stages), dtype=torch.float64, device=device)
    layer_sec = torch.zeros(int(num_layers), dtype=torch.float64, device=device)
    layer_count = torch.zeros(int(num_layers), dtype=torch.float64, device=device)
    # [stage0_hs_send_sec, stage0_hs_send_steps, last_stage_hs_send_sec, last_stage_hs_send_steps]
    hs_send = torch.zeros(4, dtype=torch.float64, device=device)

    local_stage_idx = -1
    local_stage_sec = 0.0
    local_stage_count = 0.0
    if worker is not None:
        local_stage_idx = int(worker.b.stage_idx)
        local_stage_sec = float(worker._profile_stage_forward_sec)
        local_stage_count = float(worker._profile_stage_forward_active_steps)
        if 0 <= local_stage_idx < int(num_stages):
            stage_sec[local_stage_idx] = local_stage_sec
            stage_count[local_stage_idx] = local_stage_count
        for idx, total_sec in enumerate(worker._profile_layer_forward_sec):
            if idx >= int(num_layers):
                break
            layer_sec[idx] = float(total_sec)
        for idx, count in enumerate(worker._profile_layer_forward_count):
            if idx >= int(num_layers):
                break
            layer_count[idx] = float(count)
        hs_send[0] = float(getattr(worker, "_profile_stage0_hs_send_sec", 0.0))
        hs_send[1] = float(getattr(worker, "_profile_stage0_hs_send_steps", 0))
        hs_send[2] = float(getattr(worker, "_profile_last_stage_hs_send_sec", 0.0))
        hs_send[3] = float(getattr(worker, "_profile_last_stage_hs_send_steps", 0))

    dist.all_reduce(stage_sec, op=dist.ReduceOp.SUM)
    dist.all_reduce(stage_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(layer_sec, op=dist.ReduceOp.SUM)
    dist.all_reduce(layer_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(hs_send, op=dist.ReduceOp.SUM)

    extra_forward_sec = float(rank0_extra_forward_sec)
    local_rank_forward_sec = float(local_stage_sec) + extra_forward_sec
    rank_summary = torch.tensor(
        [
            float(local_stage_idx),
            float(local_stage_sec),
            float(local_stage_count),
            extra_forward_sec,
            local_rank_forward_sec,
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_summaries = torch.zeros(
        dist.get_world_size(), 5, dtype=torch.float64, device=device
    )
    dist.all_gather_into_tensor(rank_summaries, rank_summary)

    stage_sec_list = [float(x) for x in stage_sec.tolist()]
    stage_count_list = [int(x) for x in stage_count.tolist()]
    layer_sec_list = [float(x) for x in layer_sec.tolist()]
    layer_count_list = [int(x) for x in layer_count.tolist()]
    rank_rows = []
    for rank_idx, row in enumerate(rank_summaries.tolist()):
        rank_rows.append(
            {
                "rank": int(rank_idx),
                "stage_idx": int(row[0]),
                "stage_forward_sec": float(row[1]),
                "stage_forward_active_steps": int(row[2]),
                "extra_forward_sec": float(row[3]),
                "rank_forward_sec": float(row[4]),
            }
        )
    # Stage-only max (excludes rank0 spec "extra"). Prefer this for diagnosing
    # which pipeline stage is slow; do not mix with spec_forward_sec.
    max_stage_forward_sec = max(stage_sec_list, default=0.0)
    max_stage_forward_avg_sec = 0.0
    for tot, cnt in zip(stage_sec_list, stage_count_list):
        if cnt <= 0:
            continue
        avg = float(tot) / float(cnt)
        if avg > max_stage_forward_avg_sec:
            max_stage_forward_avg_sec = avg
    hs_send_list = [float(x) for x in hs_send.tolist()]
    return {
        "stage_forward_sec": stage_sec_list,
        "stage_forward_active_steps": stage_count_list,
        "forward_layer_avg_sec": _profile_layer_rows(
            layer_sec=layer_sec_list,
            layer_count=layer_count_list,
        ),
        "rank_forward_summary": rank_rows,
        "max_stage_forward_sec": float(max_stage_forward_sec),
        "max_stage_forward_avg_sec": float(max_stage_forward_avg_sec),
        "stage0_hs_send_sec": float(hs_send_list[0]),
        "stage0_hs_send_steps": int(hs_send_list[1]),
        "last_stage_hs_send_sec": float(hs_send_list[2]),
        "last_stage_hs_send_steps": int(hs_send_list[3]),
        # Legacy: max over ranks of (stage_forward + extra). Prefer max_stage_*.
        "rank_forward_max_sec": max(
            (row["rank_forward_sec"] for row in rank_rows), default=0.0
        ),
    }


def _end_cycle_sync(
    worker: StageWorker,
    p2p: PipelineP2P,
    *,
    cycle_forward_buf: torch.Tensor | None = None,
    gather_buf: torch.Tensor | None = None,
    cycle_forward_sec: float = 0.0,
    profile_timing: bool = False,
) -> float:
    del cycle_forward_buf, gather_buf, cycle_forward_sec
    if profile_timing:
        sync_device(worker.device)
    p2p.wait_all()
    return 0.0


def run_stage_worker_loop(
    worker: StageWorker,
    p2p: PipelineP2P,
    ctrl_buf: torch.Tensor,
    *,
    hidden_size: int,
    done_flag: torch.Tensor,
    timeout: PhaseTimeout,
    profile_timing: bool = False,
) -> None:
    dtype = worker.b.compute_dtype
    worker.reset_profile_timing()
    with torch.inference_mode():
        while True:
            timeout.check()
            broadcast_ctrl(ctrl_buf, src=0)
            parsed = parse_ctrl(ctrl_buf)
            op = int(parsed["opcode"])
            if op == int(CtrlOpcode.SHUTDOWN):
                dist.barrier()
                if profile_timing:
                    _collect_profile_timing(
                        worker=worker,
                        device=worker.device,
                        num_stages=int(worker.b.num_stages),
                        num_layers=int(worker.b.num_layers),
                    )
                break
            cycle_id = int(parsed["cycle_id"])
            if op == int(CtrlOpcode.DISCARD):
                crop_length = int(parsed["crop_length"])
                worker.on_discard(crop_length, int(parsed["token_id"]))
                _end_cycle_sync(worker, p2p, profile_timing=profile_timing)
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
                    profile_timing=profile_timing,
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
                    profile_timing=profile_timing,
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
                profile_timing=profile_timing,
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
    profile_timing: bool = False,
) -> Tuple[List[int], List[bool], int, dict[str, Any]]:
    decode_steps = 0
    dtype = rank0.b.lm_head.weight.dtype
    num_stages = int(rank0.n)
    next_go_token_id = int(initial_go_token_id)
    decode_wall_start = time.perf_counter()
    breakdown = DecodeTimingBreakdown(
        stage_forward_sec=[0.0] * num_stages,
        stage_forward_active_steps=[0] * num_stages,
        by_depth_steps=[0] * num_stages,
        by_depth_recv_wait_sec=[0.0] * num_stages,
        by_depth_recv_snap_sec=[0.0] * num_stages,
        by_depth_recv_verify_sec=[0.0] * num_stages,
        by_depth_cycle_wall_sec=[0.0] * num_stages,
        by_depth_spec_forward_sec=[0.0] * num_stages,
        by_depth_cycle_other_sec=[0.0] * num_stages,
    )
    profile_summary: dict[str, Any] | None = None
    ctrl_buf = make_ctrl_tensor(opcode=CtrlOpcode.GO, cycle_id=0, device=device)
    # Snap irecvs deferred until after the next GO (fill + accept early stages).
    deferred_snaps: Optional[_DeferredSnaps] = None

    def _drain_deferred_snaps(clk: _CycleClock | None = None) -> float:
        """Wait/merge deferred snaps before post_recv / spec_forward.

        Returns host-wall seconds spent. When ``clk`` is set, closes into
        ``recv_snap_sec``. by_depth snap/wait are attributed to ``origin_depth``.
        """
        nonlocal deferred_snaps
        if deferred_snaps is None:
            return 0.0
        origin_depth = int(deferred_snaps.origin_depth)
        for stage_idx, snap_recv, exp_c, exp_pos in deferred_snaps.items:
            _wait_merge_snap_recv(
                rank0,
                stage_idx=int(stage_idx),
                snap_recv=snap_recv,
                expected_cycle_id=int(exp_c),
                expected_pos=int(exp_pos),
                num_stages=num_stages,
                timeout=timeout,
            )
        deferred_snaps = None
        if clk is not None:
            dt = clk.mark("recv_snap_sec")
        else:
            dt = 0.0
        if profile_timing and origin_depth > 0 and dt > 0.0:
            _acc_by_depth(breakdown, "by_depth_recv_snap_sec", origin_depth, dt)
            _acc_by_depth(breakdown, "by_depth_recv_wait_sec", origin_depth, dt)
        return float(dt)

    def _sync_cycle_end(clk: _CycleClock | None = None) -> None:
        """Drain in-flight P2P sends (single wait_all per cycle; no barrier)."""
        if profile_timing and clk is not None:
            if profile_timing:
                sync_device(device)
            p2p.wait_all()
            clk.mark("cycle_sync_sec")
            return
        if profile_timing:
            sync_device(device)
        p2p.wait_all()

    try:
        with torch.inference_mode():
            while rank0.verified_up_to - s0 < max_new_tokens:
                clk = _CycleClock.start(enabled=profile_timing)

                timeout.check()
                decode_steps += 1
                rank0.cycle_id += 1
                cycle_id = rank0.cycle_id
                depth = len(rank0.pipeline)
                positions = rank0.pipeline_positions()

                ctrl = make_ctrl_tensor(
                    opcode=CtrlOpcode.GO,
                    cycle_id=cycle_id,
                    pipeline_depth=depth,
                    positions=positions,
                    token_id=next_go_token_id,
                    inject_pos=rank0.next_position,
                    device=device,
                    out=ctrl_buf,
                )
                clk.mark("ctrl_prepare_sec")

                broadcast_ctrl(ctrl, src=0)
                clk.mark("ctrl_broadcast_sec")

                # Drain fill snaps deferred from the previous cycle. Must finish
                # before post_recv (pooled buffer reuse) and before spec_forward
                # (staircase needs merged snaps). Overlaps with stage forward
                # started by the GO above.
                _drain_deferred_snaps(clk)

                # Workers start stage forward on GO. Rank0 posts all fixed
                # payload irecvs immediately (early-stage snaps + last-stage
                # verify_hs→snaps) so transfer overlaps stage compute + spec.
                last_stage_recv = None
                snap_recvs: list[tuple[int, Any]] = []
                if depth == rank0.n:
                    last_stage_recv = p2p.post_last_stage_snap_verify_recv(
                        rank0.b.hidden_size,
                        dtype,
                        indices=_stage_snap_indices(rank0, num_stages - 1),
                        cycle_id=cycle_id,
                        token_pos=int(positions[-1]),
                    )
                for stage_idx in range(depth):
                    if depth == rank0.n and stage_idx == num_stages - 1:
                        continue
                    indices = _stage_snap_indices(rank0, stage_idx)
                    if not indices:
                        continue
                    src = rank_for_stage(stage_idx, num_stages)
                    snap_recvs.append(
                        (
                            stage_idx,
                            p2p.post_snap_fixed_recv(
                                rank0.b.hidden_size,
                                dtype,
                                src=src,
                                indices=indices,
                                cycle_id=cycle_id,
                                token_pos=int(positions[stage_idx]),
                            ),
                        )
                    )
                if depth > 0:
                    clk.mark("post_recv_sec")

                cycle_recv_sec = 0.0
                cycle_spec_sec = 0.0
                pending_spec_logits: torch.Tensor | None = None
                if rank0.pipeline:
                    # Serial on the default stream. Profile with stream-scoped wait
                    # (CUDA event), never torch.cuda.synchronize(): a device-wide sync
                    # would also drain NCCL recv streams posted above and inflate
                    # spec_forward into ~stage time on full-pipeline cycles.
                    if profile_timing and device.type == "cuda":
                        stream = torch.cuda.current_stream(device=device)
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end = torch.cuda.Event(enable_timing=True)
                        ev_start.record(stream)
                        pending_spec_logits = rank0.run_spec_forward(cycle_id)
                        ev_end.record(stream)
                        ev_end.synchronize()  # this stream only; NCCL stays concurrent
                        cycle_spec_sec = ev_start.elapsed_time(ev_end) / 1000.0
                        # Fold host interval into the additive clock, then overwrite
                        # with GPU-event elapsed so the phase matches real spec work.
                        clk.mark("spec_forward_sec")
                        clk.phases["spec_forward_sec"] = float(cycle_spec_sec)
                    else:
                        pending_spec_logits = rank0.run_spec_forward(cycle_id)
                        cycle_spec_sec = clk.mark("spec_forward_sec")
                else:
                    raise RuntimeError("Spec forward did not run despite non-empty pipeline.")

                verify_hs = None
                verify_pos = -1
                cycle_verify_sec = 0.0
                has_completed = depth >= rank0.n
                completed_pos = int(positions[-1]) if has_completed else -1
                target_pos = completed_pos + 1
                crop_length = target_pos
                target_gen_idx = target_pos - s0
                speculated_id: int | None = None
                verified_next_id = -1
                rejected = False

                if has_completed:
                    expected_verify_pos = int(positions[-1])
                    if last_stage_recv is None:
                        raise RuntimeError("Missing posted last-stage snap+verify recv.")
                    recv_verify_dt = 0.0
                    if profile_timing:
                        t_recv_verify = time.perf_counter()
                    verify_hs, recv_c, verify_pos = last_stage_recv.wait_verify()
                    if profile_timing:
                        recv_verify_dt += time.perf_counter() - t_recv_verify
                    else:
                        recv_verify_dt = clk.mark("recv_verify_sec")
                    assert_p2p_meta(
                        "verify_hs",
                        cycle_id=recv_c,
                        expected_cycle_id=cycle_id,
                        token_pos=verify_pos,
                        expected_token_pos=expected_verify_pos,
                        peer_rank=p2p.num_stages,
                        local_rank=0,
                    )

                    # Lightweight verify prep (draft id); snap payloads already posted.
                    speculated_id = 0
                    verified_next_id = 0
                    do_verify = False
                    if target_gen_idx < len(rank0.generated_ids):
                        speculated_id = int(rank0.generated_ids[target_gen_idx])
                        verified_next_id = int(speculated_id)
                        do_verify = bool(verify and verify_hs is not None)
                    clk.mark("snap_progress_sec")

                    verify_input_staged = False
                    if do_verify and profile_timing:
                        t_stage = time.perf_counter()
                        if rank0.verify_graph is not None:
                            rank0.verify_graph.stage_input_sync(verify_hs)
                            verify_input_staged = True
                        elif device.type == "cuda":
                            verify_hs = verify_hs.clone()
                        recv_verify_dt += time.perf_counter() - t_stage

                    if profile_timing:
                        clk.phases["recv_verify_sec"] = float(
                            clk.phases["recv_verify_sec"]
                        ) + float(recv_verify_dt)
                        clk.mark_t = time.perf_counter()
                    cycle_recv_sec += float(recv_verify_dt)

                    if do_verify:
                        if profile_timing:
                            accepted, verified_next_id, vparts = rank0.verify_with_hs(
                                verify_hs,
                                target_pos,
                                speculated_id,
                                greedy=greedy,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                profile_timing=True,
                                input_staged=verify_input_staged,
                            )
                            copy_dt = float(vparts["copy_sec"])
                            kernel_dt = float(vparts["kernel_sec"])
                            decide_dt = float(vparts["decide_sec"])
                            parts_sum = copy_dt + kernel_dt + decide_dt
                            wall_dt = clk.mark("verify_sec")
                            # Keep verify_sec == copy+kernel+decide; call glue → other.
                            if abs(wall_dt - parts_sum) > 1e-12:
                                clk.phases["verify_sec"] = (
                                    float(clk.phases["verify_sec"])
                                    - float(wall_dt)
                                    + float(parts_sum)
                                )
                                clk.phases["cycle_other_sec"] = float(
                                    clk.phases["cycle_other_sec"]
                                ) + float(wall_dt - parts_sum)
                            cycle_verify_sec = parts_sum
                            _acc_phase(breakdown, "verify_copy_sec", copy_dt)
                            _acc_phase(breakdown, "verify_kernel_sec", kernel_dt)
                            _acc_phase(breakdown, "verify_decide_sec", decide_dt)
                            if float(vparts.get("used_graph", 0.0)) > 0.0:
                                breakdown.verify_graph_steps += 1
                        else:
                            accepted, verified_next_id = rank0.verify_with_hs(
                                verify_hs,
                                target_pos,
                                speculated_id,
                                greedy=greedy,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                            cycle_verify_sec = clk.mark("verify_sec")
                        rejected = not bool(accepted)

                # Snap wait: reject waits all in-cycle. Accept defers early-stage
                # snaps until after the next GO (last-stage snap stays in-cycle for
                # commit_completed_snap). Fill defers all early-stage snaps.
                cycle_snap_sec = 0.0
                will_reject = bool(
                    has_completed
                    and speculated_id is not None
                    and rejected
                )
                if has_completed:
                    if last_stage_recv is None:
                        raise RuntimeError("Missing posted last-stage snap+verify recv.")
                    recv_c, spos, valid, shards = last_stage_recv.wait_snaps()
                    assert_p2p_meta(
                        "last_stage_snap_verify",
                        cycle_id=recv_c,
                        expected_cycle_id=cycle_id,
                        token_pos=spos,
                        expected_token_pos=int(positions[-1]),
                        peer_rank=p2p.num_stages,
                        local_rank=0,
                    )
                    if valid:
                        _merge_snap_into_pipeline(rank0, recv_c, spos, shards)

                    if will_reject:
                        for stage_idx, snap_recv in snap_recvs:
                            expected_spos = int(positions[stage_idx])
                            _wait_merge_snap_recv(
                                rank0,
                                stage_idx=stage_idx,
                                snap_recv=snap_recv,
                                expected_cycle_id=cycle_id,
                                expected_pos=expected_spos,
                                num_stages=num_stages,
                                timeout=timeout,
                            )
                        cycle_snap_sec = clk.mark("recv_snap_sec")
                        cycle_recv_sec += cycle_snap_sec
                    elif snap_recvs:
                        if deferred_snaps is not None:
                            raise RuntimeError(
                                "Previous deferred snap recv was not drained."
                            )
                        deferred_snaps = _DeferredSnaps(
                            origin_depth=int(depth),
                            items=[
                                (
                                    int(stage_idx),
                                    snap_recv,
                                    int(cycle_id),
                                    int(positions[stage_idx]),
                                )
                                for stage_idx, snap_recv in snap_recvs
                            ],
                        )
                        # Last-stage snap only; early stages deferred to next GO.
                        cycle_snap_sec = clk.mark("recv_snap_sec")
                        cycle_recv_sec += cycle_snap_sec
                elif depth > 0:
                    if deferred_snaps is not None:
                        raise RuntimeError(
                            "Previous deferred snap recv was not drained."
                        )
                    deferred_snaps = _DeferredSnaps(
                        origin_depth=int(depth),
                        items=[
                            (
                                int(stage_idx),
                                snap_recv,
                                int(cycle_id),
                                int(positions[stage_idx]),
                            )
                            for stage_idx, snap_recv in snap_recvs
                        ],
                    )

                if has_completed:
                    rank0.commit_completed_snap(completed_pos, cycle_id)

                if profile_timing and has_completed:
                    breakdown.full_pipeline_steps += 1
                    _acc_time(
                        breakdown, "full_pipeline_recv_wait_sec", cycle_recv_sec
                    )

                if profile_timing and depth > 0:
                    cycle_verify_recv_sec = float(clk.phases.get("recv_verify_sec", 0.0))
                    _acc_by_depth(breakdown, "by_depth_steps", depth, 0.0)
                    _acc_by_depth(
                        breakdown, "by_depth_recv_wait_sec", depth, cycle_recv_sec
                    )
                    _acc_by_depth(
                        breakdown, "by_depth_recv_snap_sec", depth, cycle_snap_sec
                    )
                    _acc_by_depth(
                        breakdown,
                        "by_depth_recv_verify_sec",
                        depth,
                        cycle_verify_recv_sec,
                    )
                    # by_depth_spec_forward filled in CycleClock.finish

                if has_completed and speculated_id is not None:
                    if rejected:
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
                            out=ctrl_buf,
                        )
                        # Build discard ctrl counted as driver_update prelude.
                        clk.mark("driver_update_sec")
                        _sync_cycle_end(clk)
                        broadcast_ctrl(discard, src=0)
                        rank0.apply_reject(
                            crop_length,
                            verified_next_id,
                            s0,
                            completed_pos=completed_pos,
                        )
                        p2p.wait_all()
                        discard_dt = clk.mark("discard_comm_sec")
                        # apply_reject is inside discard_comm mark; split for rollback only.
                        _acc_rollback(
                            breakdown,
                            verify_sec=cycle_verify_sec,
                            discard_comm_sec=discard_dt,
                            apply_reject_sec=0.0,
                        )
                        clk.finish(breakdown, depth)
                        if verified_next_id == eos_token_id:
                            break
                        continue

                    rank0.verified_up_to = target_pos + 1
                    if speculated_id == eos_token_id:
                        clk.mark("driver_update_sec")
                        _sync_cycle_end(clk)
                        clk.finish(breakdown, depth)
                        break

                if has_completed:
                    rank0.pending_deepest_snap = dict(rank0.completed_snaps[completed_pos])
                    rank0.pending_deepest_pos = completed_pos
                    rank0.pipeline.pop()

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
                clk.mark("driver_update_sec")
                _sync_cycle_end(clk)
                clk.finish(breakdown, depth)
    finally:
        t0 = time.perf_counter()
        # Finish any in-flight fill snap irecvs before SHUTDOWN.
        if deferred_snaps is not None:
            _drain_deferred_snaps(None)
        shutdown = make_ctrl_tensor(
            opcode=CtrlOpcode.SHUTDOWN,
            cycle_id=rank0.cycle_id,
            device=device,
            out=ctrl_buf,
        )
        broadcast_ctrl(shutdown, src=0)
        dist.barrier()
        if profile_timing:
            profile_summary = _collect_profile_timing(
                worker=None,
                device=device,
                num_stages=num_stages,
                num_layers=int(rank0.b.num_layers),
                rank0_extra_forward_sec=float(breakdown.spec_forward_sec),
            )
            breakdown.stage_forward_sec = list(profile_summary["stage_forward_sec"])
            breakdown.stage_forward_active_steps = list(
                profile_summary["stage_forward_active_steps"]
            )
            breakdown.max_stage_forward_sec = float(
                profile_summary["max_stage_forward_sec"]
            )
            breakdown.max_stage_forward_avg_sec = float(
                profile_summary["max_stage_forward_avg_sec"]
            )
            breakdown.stage0_hs_send_sec = float(
                profile_summary.get("stage0_hs_send_sec", 0.0)
            )
            breakdown.stage0_hs_send_steps = int(
                profile_summary.get("stage0_hs_send_steps", 0)
            )
            breakdown.last_stage_hs_send_sec = float(
                profile_summary.get("last_stage_hs_send_sec", 0.0)
            )
            breakdown.last_stage_hs_send_steps = int(
                profile_summary.get("last_stage_hs_send_steps", 0)
            )
            breakdown.decode_cycle_max_stage_sec = float(
                breakdown.max_stage_forward_sec
            )
            breakdown.critical_path_compute_sec = (
                float(breakdown.max_stage_forward_sec)
                + float(breakdown.verify_sec)
            )
        done_flag.fill_(1)
        if profile_timing:
            _acc_phase(breakdown, "shutdown_sec", time.perf_counter() - t0)

    decode_wall_sec = time.perf_counter() - decode_wall_start
    timing: dict[str, Any] = {
        "decode_wall_sec": float(decode_wall_sec),
        "decode_steps": float(decode_steps),
        "num_stages": float(num_stages),
        "timing_profile_enabled": bool(profile_timing),
    }
    if profile_timing:
        timing.update(breakdown.to_dict())
        if profile_summary is not None:
            timing.update(
                {
                    "forward_layer_avg_sec": profile_summary["forward_layer_avg_sec"],
                    "rank_forward_summary": profile_summary["rank_forward_summary"],
                }
            )
        # Gap between decode wall and additive rank-0 phases (should be small).
        timing["rank0_unaccounted_sec"] = float(
            max(decode_wall_sec - breakdown.rank0_sequential_sec, 0.0)
        )
        timing["cycle_phases_gap_sec"] = float(
            max(breakdown.cycle_wall_sec - breakdown.cycle_phases_sec, 0.0)
        )
    rank0.last_timing = timing
    return (
        rank0.generated_ids[:max_new_tokens],
        rank0.token_acceptance[:max_new_tokens],
        decode_steps,
        timing,
    )
