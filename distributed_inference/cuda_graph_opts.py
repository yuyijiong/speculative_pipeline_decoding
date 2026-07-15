"""Lightweight CUDA Graph helpers for v11 rank-0 aggregation and verify.

CUDA requires capture on a **non-default** stream; after capture, graphs are
replayed on the default/current stream (serial with decode, no side stream):
- speculation snap aggregation (``aggr_projs`` / g-row fusion), one graph per
  reachable ``(num_active, has_pending)`` pattern (fill without pending;
  pending only at full pipeline or post-reject);
- verify ``final_norm + lm_head``.

Spec decoder layers, sampling, NCCL, and accept/reject stay outside the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stream_sync(device: torch.device) -> None:
    """Sync only the current stream (does not drain NCCL recv streams)."""
    if device.type == "cuda":
        torch.cuda.current_stream(device=device).synchronize()


def _capture_stream(device: torch.device) -> torch.cuda.Stream:
    """Non-default stream required by ``torch.cuda.graph`` capture."""
    return torch.cuda.Stream(device=device)


@dataclass
class _AggrPatternBuffers:
    """Static inputs/outputs for one ``(num_active, has_pending)`` aggregation graph."""

    depths: Tuple[int, ...]
    # snap_bufs[row_i][feat_j] -> (1, 1, H)
    snap_bufs: List[List[torch.Tensor]]
    out: torch.Tensor  # (1, num_rows, H)
    graph: torch.cuda.CUDAGraph


class SpecAggrGraphRunner:
    """CUDA graphs for fusing snapshot hidden states into speculation g-rows."""

    def __init__(
        self,
        *,
        num_stages: int,
        hidden_size: int,
        aggr_projs: nn.ModuleList,
        aggr_feature_indices: Sequence[Sequence[int]],
        stage_depth_to_aggr_idx: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("SpecAggrGraphRunner requires a CUDA device")
        self.n = int(num_stages)
        self.hidden_size = int(hidden_size)
        self.aggr_projs = aggr_projs
        self.aggr_feature_indices = [tuple(int(x) for x in row) for row in aggr_feature_indices]
        self.stage_depth_to_aggr_idx = [int(x) for x in stage_depth_to_aggr_idx]
        self.device = device
        self.dtype = dtype
        self._patterns: Dict[Tuple[int, bool], _AggrPatternBuffers] = {}
        self._captured = False

    @staticmethod
    def row_depths(num_active: int, has_pending: bool, num_stages: int) -> Tuple[int, ...]:
        """Nominal depths oldest→newest matching ``_build_spec_rows_at_sync`` staircase."""
        n = int(num_stages)
        d = int(num_active)
        if d < 1:
            raise ValueError(f"num_active must be >= 1, got {d}")
        depths: List[int] = []
        if has_pending:
            depths.append(n)
            cap = n - 1
        else:
            cap = n
        for lag in range(d - 1, -1, -1):
            depths.append(min(int(lag), cap))
        return tuple(depths)

    def _fuse_rows_into(
        self,
        snap_bufs: List[List[torch.Tensor]],
        depths: Sequence[int],
        out: torch.Tensor,
    ) -> None:
        rows: List[torch.Tensor] = []
        for row_i, depth in enumerate(depths):
            aggr_i = self.stage_depth_to_aggr_idx[int(depth)]
            n_feat = len(self.aggr_feature_indices[aggr_i])
            vecs = snap_bufs[row_i][:n_feat]
            rows.append(self.aggr_projs[aggr_i](torch.cat(vecs, dim=-1)))
        out.copy_(torch.cat(rows, dim=1))

    def _alloc_snap_bufs(self, depths: Sequence[int]) -> List[List[torch.Tensor]]:
        snap_bufs: List[List[torch.Tensor]] = []
        for depth in depths:
            aggr_i = self.stage_depth_to_aggr_idx[int(depth)]
            n_feat = len(self.aggr_feature_indices[aggr_i])
            snap_bufs.append(
                [
                    torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
                    for _ in range(n_feat)
                ]
            )
        return snap_bufs

    def capture_all(self) -> None:
        """Capture only reachable ``(num_active, has_pending)`` patterns.

        ``pending_deepest`` is set only after a completed token leaves the pipeline
        (steady accept) or on reject, so fill never has pending:

        - ``(d, False)`` for ``d = 1..n`` — pipeline fill / first full cycle
        - ``(n, True)`` — steady state after accept (pending + full window)
        - ``(1, True)`` — post-reject (pipeline reset to one token + pending)
        """
        if self._captured:
            return
        keys: List[Tuple[int, bool]] = [(d, False) for d in range(1, self.n + 1)]
        keys.append((self.n, True))
        keys.append((1, True))
        # Dedup when n == 1: (1, False) and (1, True) are both needed; (n, True) == (1, True).
        seen: set[Tuple[int, bool]] = set()
        uniq: List[Tuple[int, bool]] = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        with torch.cuda.device(self.device):
            for num_active, has_pending in uniq:
                self._capture_one(num_active, has_pending)
        self._captured = True

    def _capture_one(self, num_active: int, has_pending: bool) -> None:
        key = (int(num_active), bool(has_pending))
        if key in self._patterns:
            return
        depths = self.row_depths(num_active, has_pending, self.n)
        num_rows = len(depths)
        snap_bufs = self._alloc_snap_bufs(depths)
        out = torch.zeros(1, num_rows, self.hidden_size, device=self.device, dtype=self.dtype)
        # Capture must use a non-default stream; replay stays on default.
        capture_stream = _capture_stream(self.device)

        # Warm dummy inputs so capture sees real kernel paths.
        for row_bufs in snap_bufs:
            for buf in row_bufs:
                buf.normal_()

        _sync(self.device)
        with torch.inference_mode():
            for _ in range(2):
                self._fuse_rows_into(snap_bufs, depths, out)
        _sync(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph, stream=capture_stream):
                self._fuse_rows_into(snap_bufs, depths, out)
        _sync(self.device)

        self._patterns[key] = _AggrPatternBuffers(
            depths=depths,
            snap_bufs=snap_bufs,
            out=out,
            graph=graph,
        )

    def try_replay(
        self,
        *,
        num_active: int,
        has_pending: bool,
        row_snaps: Sequence[Dict[int, torch.Tensor]],
        row_depths: Sequence[int],
    ) -> Optional[torch.Tensor]:
        """
        Replay aggregation if the pattern is captured and depths match the nominal staircase.

        ``row_snaps`` / ``row_depths`` are oldest→newest (pending row first when present).
        Returns fused ``(1, num_rows, H)`` or ``None`` to signal eager fallback.
        Captured on a temporary non-default stream; replayed on the
        default/current stream (serial with decode).
        """
        key = (int(num_active), bool(has_pending))
        pat = self._patterns.get(key)
        if pat is None:
            return None
        if tuple(int(d) for d in row_depths) != pat.depths:
            return None
        if len(row_snaps) != len(pat.depths):
            return None

        for row_i, depth in enumerate(pat.depths):
            aggr_i = self.stage_depth_to_aggr_idx[int(depth)]
            hf_indices = self.aggr_feature_indices[aggr_i]
            snap = row_snaps[row_i]
            for feat_j, hf_idx in enumerate(hf_indices):
                src = snap[int(hf_idx)]
                if src.shape != pat.snap_bufs[row_i][feat_j].shape:
                    return None
                pat.snap_bufs[row_i][feat_j].copy_(src)

        pat.graph.replay()
        return pat.out


class VerifyGraphRunner:
    """CUDA graph for ``final_norm(hs) -> lm_head``.

    Capture uses a temporary non-default stream (CUDA requirement); replay
    runs on the default/current stream so serial decode needs no side stream.
    """

    def __init__(
        self,
        *,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("VerifyGraphRunner requires a CUDA device")
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.device = device
        self.dtype = dtype
        self.hs = torch.zeros(1, 1, int(hidden_size), device=device, dtype=dtype)
        self.logits = torch.zeros(1, 1, int(vocab_size), device=device, dtype=dtype)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

    def _forward_into_buffer(self) -> None:
        self.logits.copy_(self.lm_head(self.final_norm(self.hs)))

    def capture(self) -> None:
        if self._captured:
            return
        # Capture must use a non-default stream; replay stays on default.
        capture_stream = _capture_stream(self.device)
        self.hs.normal_()
        _sync(self.device)
        with torch.inference_mode():
            for _ in range(2):
                self._forward_into_buffer()
        _sync(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph, stream=capture_stream):
                self._forward_into_buffer()
        _sync(self.device)
        self.graph = graph
        self._captured = True

    def run(self, hs: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("VerifyGraphRunner.capture() must be called first")
        if hs.shape != self.hs.shape:
            return self.lm_head(self.final_norm(hs))
        if hs.dtype != self.hs.dtype:
            hs = hs.to(dtype=self.hs.dtype)
        self.hs.copy_(hs)
        self.graph.replay()
        return self.logits

    def stage_input_sync(self, hs: torch.Tensor) -> None:
        """Copy ``verify_hs`` into the graph input buffer and block until ready.

        Must run during ``recv_verify`` (before ``verify``). ``work.wait()`` on
        the NCCL irecv often returns before the buffer is consumable on the
        default stream; this sync copy captures that remaining wait.
        """
        if self.graph is None:
            raise RuntimeError("VerifyGraphRunner.capture() must be called first")
        if hs.shape != self.hs.shape:
            raise ValueError(
                f"verify_hs shape {tuple(hs.shape)} != graph input {tuple(self.hs.shape)}"
            )
        if hs.dtype != self.hs.dtype:
            hs = hs.to(dtype=self.hs.dtype)
        self.hs.copy_(hs)
        _stream_sync(self.device)

    def run_profiled(
        self, hs: torch.Tensor, *, input_staged: bool = False
    ) -> tuple[torch.Tensor, float, float, bool]:
        """Return ``(logits, copy_sec, kernel_sec, used_graph)``.

        Both ``copy_sec`` and ``kernel_sec`` are contiguous host-wall times
        (``perf_counter``); kernel includes **stream-scoped** sync so only
        verify graph work on the default stream is attributed (never
        ``torch.cuda.synchronize``, which would also drain in-flight snap
        ``irecv``s posted in the same cycle).
        """
        import time

        if self.graph is None:
            raise RuntimeError("VerifyGraphRunner.capture() must be called first")
        if hs.shape != self.hs.shape:
            t0 = time.perf_counter()
            logits = self.lm_head(self.final_norm(hs))
            _sync(self.device)
            kernel_sec = time.perf_counter() - t0
            return logits, 0.0, float(kernel_sec), False
        if input_staged:
            copy_sec = 0.0
        else:
            t0 = time.perf_counter()
            if hs.dtype != self.hs.dtype:
                hs = hs.to(dtype=self.hs.dtype)
            self.hs.copy_(hs)
            copy_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.graph.replay()
        _stream_sync(self.device)
        kernel_sec = time.perf_counter() - t1
        return self.logits, float(copy_sec), float(kernel_sec), True


def module_param_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype
