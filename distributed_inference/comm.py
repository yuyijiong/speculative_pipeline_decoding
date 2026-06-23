"""Control-plane broadcast and P2P hidden-state messages with fixed metadata."""

from __future__ import annotations

from enum import IntEnum
from typing import List

import torch
from torch.distributed import Work

from .dist_io import (
    clear_send_staging,
    dist_broadcast,
    dist_irecv,
    dist_isend,
    dist_recv,
    dist_send,
)

CTRL_OPCODE = 0
CTRL_CYCLE = 1
CTRL_PIPELINE_DEPTH = 2
CTRL_TOKEN_ID = 3
CTRL_VERIFY_POS = 4
CTRL_INJECT_POS = 5
CTRL_CROP_LENGTH = 6
CTRL_POSITIONS_START = 7
CTRL_LEN = 7 + 16


class CtrlOpcode(IntEnum):
    GO = 0
    DISCARD = 1
    SHUTDOWN = 2


def make_ctrl_tensor(
    *,
    opcode: CtrlOpcode,
    cycle_id: int,
    pipeline_depth: int = 0,
    token_id: int = 0,
    verify_pos: int = -1,
    inject_pos: int = -1,
    crop_length: int = -1,
    positions: list[int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    buf = torch.zeros(CTRL_LEN, dtype=torch.int64, device=dev)
    buf[CTRL_OPCODE] = int(opcode)
    buf[CTRL_CYCLE] = int(cycle_id)
    buf[CTRL_PIPELINE_DEPTH] = int(pipeline_depth)
    buf[CTRL_TOKEN_ID] = int(token_id)
    buf[CTRL_VERIFY_POS] = int(verify_pos)
    buf[CTRL_INJECT_POS] = int(inject_pos)
    buf[CTRL_CROP_LENGTH] = int(crop_length)
    if positions is not None:
        n = min(len(positions), CTRL_LEN - CTRL_POSITIONS_START)
        for i in range(n):
            buf[CTRL_POSITIONS_START + i] = int(positions[i])
    return buf


def parse_ctrl(buf: torch.Tensor) -> dict[str, int | list[int]]:
    b = buf.view(-1)
    depth = int(b[CTRL_PIPELINE_DEPTH].item())
    positions: list[int] = []
    for i in range(depth):
        positions.append(int(b[CTRL_POSITIONS_START + i].item()))
    return {
        "opcode": int(b[CTRL_OPCODE].item()),
        "cycle_id": int(b[CTRL_CYCLE].item()),
        "pipeline_depth": depth,
        "token_id": int(b[CTRL_TOKEN_ID].item()),
        "verify_pos": int(b[CTRL_VERIFY_POS].item()),
        "inject_pos": int(b[CTRL_INJECT_POS].item()),
        "crop_length": int(b[CTRL_CROP_LENGTH].item()),
        "positions": positions,
    }


def broadcast_ctrl(buf: torch.Tensor, *, src: int = 0) -> None:
    dist_broadcast(buf, src=src)


def assert_p2p_meta(
    kind: str,
    *,
    cycle_id: int,
    expected_cycle_id: int,
    token_pos: int | None = None,
    expected_token_pos: int | None = None,
    peer_rank: int | None = None,
    local_rank: int | None = None,
) -> None:
    """Raise if recv-side P2P metadata does not match the active decode cycle."""
    if int(cycle_id) != int(expected_cycle_id):
        peer_s = f", peer rank={peer_rank}" if peer_rank is not None else ""
        local_s = f", local rank={local_rank}" if local_rank is not None else ""
        raise RuntimeError(
            f"P2P metadata mismatch [{kind}]: cycle_id={cycle_id} != expected "
            f"{expected_cycle_id}{peer_s}{local_s}"
        )
    if expected_token_pos is not None and token_pos is not None:
        if int(token_pos) != int(expected_token_pos):
            peer_s = f", peer rank={peer_rank}" if peer_rank is not None else ""
            local_s = f", local rank={local_rank}" if local_rank is not None else ""
            raise RuntimeError(
                f"P2P metadata mismatch [{kind}]: token_pos={token_pos} != expected "
                f"{expected_token_pos}{peer_s}{local_s}"
            )


class PipelineP2P:
    """
    Fixed topology: rank0 -> rank1 (token_id via ctrl), rank i -> rank i+1 (hs),
    each stage -> rank0 (snap batches), last stage -> rank0 (verify_hs).
    """

    EMPTY_HS_SHAPE = (1, 1, 0)

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        async_comm: bool = False,
        merge_last_stage: bool = False,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.merge_last_stage = bool(merge_last_stage)
        self.num_stages = (
            self.world_size if self.merge_last_stage else self.world_size - 1
        )
        self.device = device
        self.async_comm = bool(async_comm)
        self._verify_buf: torch.Tensor | None = None
        self._hs_recv_buf: torch.Tensor | None = None
        self._hs_send_buf: torch.Tensor | None = None
        self._pending: List[Work] = []

    def _hs_shape(self, hidden_size: int, seq_len: int) -> tuple[int, ...]:
        return (1, int(seq_len), int(hidden_size))

    def wait_all(self) -> None:
        for work in self._pending:
            work.wait()
        self._pending.clear()
        clear_send_staging()

    def _track(self, work: Work) -> None:
        if self.async_comm:
            self._pending.append(work)

    def send_hs(
        self,
        hs: torch.Tensor,
        *,
        cycle_id: int,
        token_pos: int,
        valid: bool,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        from .topology import hs_send_dst_rank

        dst = hs_send_dst_rank(
            self.rank, self.world_size, merge_last_stage=self.merge_last_stage
        )
        if valid:
            seq_len = int(hs.shape[1])
            meta = torch.tensor(
                [int(cycle_id), int(token_pos), 1, seq_len],
                dtype=torch.int64,
                device=self.device,
            )
            payload = hs.contiguous()
            if payload.dtype != dtype:
                payload = payload.to(dtype=dtype)
            if self.async_comm:
                if (
                    self._hs_send_buf is None
                    or self._hs_send_buf.shape != payload.shape
                    or self._hs_send_buf.dtype != payload.dtype
                ):
                    self._hs_send_buf = torch.empty_like(payload)
                self._hs_send_buf.copy_(payload)
                payload = self._hs_send_buf
                self._track(dist_isend(meta, dst=dst))
                self._track(dist_isend(payload, dst=dst))
            else:
                dist_send(meta, dst=dst)
                dist_send(payload, dst=dst)
        else:
            meta = torch.tensor(
                [int(cycle_id), int(token_pos), 0, 1],
                dtype=torch.int64,
                device=self.device,
            )
            dummy = torch.zeros(
                self._hs_shape(hidden_size, 1), dtype=dtype, device=self.device
            )
            if self.async_comm:
                self._track(dist_isend(meta, dst=dst))
                self._track(dist_isend(dummy, dst=dst))
            else:
                dist_send(meta, dst=dst)
                dist_send(dummy, dst=dst)

    def recv_hs(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        recv_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, int, int, int, bool]:
        from .topology import hs_recv_src_rank

        src = hs_recv_src_rank(
            self.rank, self.world_size, merge_last_stage=self.merge_last_stage
        )
        meta = torch.empty(4, dtype=torch.int64, device=self.device)
        if self.async_comm:
            dist_irecv(meta, src=src).wait()
        else:
            dist_recv(meta, src=src)
        cycle_id = int(meta[0].item())
        token_pos = int(meta[1].item())
        valid = int(meta[2].item()) != 0
        seq_len = int(meta[3].item())
        shape = self._hs_shape(hidden_size, seq_len)
        if (
            recv_buf is None
            or tuple(recv_buf.shape) != shape
            or recv_buf.dtype != dtype
            or recv_buf.device != self.device
        ):
            if recv_buf is None:
                if self._hs_recv_buf is None or self._hs_recv_buf.shape != shape:
                    self._hs_recv_buf = torch.empty(shape, dtype=dtype, device=self.device)
                buf = self._hs_recv_buf
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
        else:
            buf = recv_buf
        if self.async_comm:
            dist_irecv(buf, src=src).wait()
        else:
            dist_recv(buf, src=src)
        if not valid:
            return None, cycle_id, token_pos, seq_len, False
        return buf, cycle_id, token_pos, seq_len, True

    def send_verify_hs(
        self, hs: torch.Tensor, *, cycle_id: int, token_pos: int
    ) -> None:
        meta = torch.tensor(
            [int(cycle_id), int(token_pos), 1],
            dtype=torch.int64,
            device=self.device,
        )
        payload = hs.contiguous()
        if self.async_comm:
            if (
                self._hs_send_buf is None
                or self._hs_send_buf.shape != payload.shape
                or self._hs_send_buf.dtype != payload.dtype
            ):
                self._hs_send_buf = torch.empty_like(payload)
            self._hs_send_buf.copy_(payload)
            payload = self._hs_send_buf
            self._track(dist_isend(meta, dst=0))
            self._track(dist_isend(payload, dst=0))
        else:
            dist_send(meta, dst=0)
            dist_send(payload, dst=0)

    def recv_verify_hs(
        self, hidden_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, int, int]:
        if self.merge_last_stage:
            raise RuntimeError("recv_verify_hs is unused when merge_last_stage=True")
        meta = torch.empty(3, dtype=torch.int64, device=self.device)
        shape = self._hs_shape(hidden_size, 1)
        if self._verify_buf is None or self._verify_buf.shape != shape:
            self._verify_buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist_recv(meta, src=self.num_stages)
        cycle_id = int(meta[0].item())
        token_pos = int(meta[1].item())
        valid = int(meta[2].item())
        dist_recv(self._verify_buf, src=self.num_stages)
        if valid == 0:
            raise RuntimeError("Received invalid verify_hs from last stage.")
        return self._verify_buf, cycle_id, token_pos

    def send_snap_batch(
        self,
        *,
        cycle_id: int,
        token_pos: int,
        valid: bool,
        indices: list[int],
        tensors: list[torch.Tensor],
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        k = len(indices)
        if k != len(tensors):
            raise ValueError(f"snap batch indices/tensors length mismatch: {k} vs {len(tensors)}")
        seq_len = int(tensors[0].shape[1]) if k > 0 and valid else 1
        meta = torch.tensor(
            [int(cycle_id), int(token_pos), int(valid), int(k), seq_len],
            dtype=torch.int64,
            device=self.device,
        )
        dist_send(meta, dst=0)
        if not valid or k == 0:
            return
        idx_t = torch.tensor(indices, dtype=torch.int64, device=self.device)
        dist_send(idx_t, dst=0)
        for t in tensors:
            payload = t.contiguous()
            if payload.dtype != dtype:
                payload = payload.to(dtype=dtype)
            dist_send(payload, dst=0)

    def recv_snap_batch(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        src: int,
        max_indices: int = 16,
    ) -> tuple[int, int, bool, dict[int, torch.Tensor]]:
        meta = torch.empty(5, dtype=torch.int64, device=self.device)
        dist_recv(meta, src=src)
        cycle_id = int(meta[0].item())
        token_pos = int(meta[1].item())
        valid = int(meta[2].item()) != 0
        k = int(meta[3].item())
        seq_len = int(meta[4].item())
        if k > max_indices:
            raise ValueError(f"snap batch has {k} indices > max_indices={max_indices}")
        if not valid or k == 0:
            return cycle_id, token_pos, False, {}
        idx_buf = torch.empty(k, dtype=torch.int64, device=self.device)
        dist_recv(idx_buf, src=src)
        out: dict[int, torch.Tensor] = {}
        shape = self._hs_shape(hidden_size, seq_len)
        for i in range(k):
            idx = int(idx_buf[i].item())
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            dist_recv(buf, src=src)
            out[idx] = buf
        return cycle_id, token_pos, True, out
