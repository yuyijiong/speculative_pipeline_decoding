"""Control-plane broadcast and P2P hidden-state messages with fixed metadata."""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Sequence

import torch
from torch.distributed import Work

from .dist_io import (
    clear_send_staging,
    dist_broadcast,
    dist_irecv,
    dist_isend,
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

# Decode snap batches are small; keep a fixed upper bound for pooled idx buffers.
DEFAULT_MAX_SNAP_INDICES = 16


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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build (or in-place fill) a control-plane tensor."""
    if out is None:
        dev = device if device is not None else torch.device("cpu")
        buf = torch.zeros(CTRL_LEN, dtype=torch.int64, device=dev)
    else:
        buf = out
        if buf.numel() < CTRL_LEN or buf.dtype != torch.int64:
            raise ValueError("ctrl out buffer must be int64 with length >= CTRL_LEN")
        buf.zero_()
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
    each stage -> rank0 (snap batches), last stage -> rank0 (snap + verify_hs).

    Meta / dummy / stage-to-stage hs buffers are pooled and reused across cycles.
    Stage→rank0 snaps (early + last-stage verify) use a fixed wire with no meta;
    ``wait()`` returns clones so callers may retain shards.
    """

    EMPTY_HS_SHAPE = (1, 1, 0)

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        max_snap_indices: int = DEFAULT_MAX_SNAP_INDICES,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_stages = self.world_size - 1
        self.device = device
        self.max_snap_indices = int(max_snap_indices)
        self._hs_recv_buf: torch.Tensor | None = None
        self._pending: List[Work] = []
        self._send_keepalive: List[torch.Tensor] = []

        # Pooled send/recv scratch (allocated lazily on first use).
        self._send_hs_meta: torch.Tensor | None = None
        self._send_dummy_hs: torch.Tensor | None = None
        self._recv_hs_meta: torch.Tensor | None = None

        # Persistent async recv slots (buffers live for process lifetime).
        self._snap_recv_by_src: Dict[int, PostedSnapFixedRecv] = {}
        self._last_stage_recv: LastStageSnapVerifyRecv | None = None

    def _hs_shape(self, hidden_size: int, seq_len: int) -> tuple[int, ...]:
        return (1, int(seq_len), int(hidden_size))

    def _ensure_buf(
        self,
        cur: torch.Tensor | None,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            cur is None
            or tuple(cur.shape) != tuple(shape)
            or cur.dtype != dtype
            or cur.device != self.device
        ):
            return torch.empty(shape, dtype=dtype, device=self.device)
        return cur

    def _hs_meta_send(self) -> torch.Tensor:
        self._send_hs_meta = self._ensure_buf(self._send_hs_meta, (4,), torch.int64)
        return self._send_hs_meta

    def _dummy_hs(self, hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
        shape = self._hs_shape(hidden_size, 1)
        self._send_dummy_hs = self._ensure_buf(self._send_dummy_hs, shape, dtype)
        self._send_dummy_hs.zero_()
        return self._send_dummy_hs

    def _hs_meta_recv(self) -> torch.Tensor:
        self._recv_hs_meta = self._ensure_buf(self._recv_hs_meta, (4,), torch.int64)
        return self._recv_hs_meta

    def wait_all(self) -> None:
        for work in self._pending:
            work.wait()
        self._pending.clear()
        self._send_keepalive.clear()
        clear_send_staging()

    def _track(self, work: Work) -> None:
        self._pending.append(work)

    def _send_ordered(self, tensor: torch.Tensor, *, dst: int) -> None:
        payload = tensor.contiguous()
        self._send_keepalive.append(payload)
        self._track(dist_isend(payload, dst=int(dst)))

    def _recv_ordered(self, tensor: torch.Tensor, *, src: int) -> None:
        dist_irecv(tensor, src=int(src)).wait()

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

        dst = hs_send_dst_rank(self.rank, self.world_size)
        meta = self._hs_meta_send()
        if valid:
            seq_len = int(hs.shape[1])
            meta[0] = int(cycle_id)
            meta[1] = int(token_pos)
            meta[2] = 1
            meta[3] = seq_len
            payload = hs.contiguous()
            if payload.dtype != dtype:
                payload = payload.to(dtype=dtype)
            self._send_ordered(meta, dst=dst)
            self._send_ordered(payload, dst=dst)
        else:
            meta[0] = int(cycle_id)
            meta[1] = int(token_pos)
            meta[2] = 0
            meta[3] = 1
            self._send_ordered(meta, dst=dst)
            self._send_ordered(self._dummy_hs(hidden_size, dtype), dst=dst)

    def recv_hs(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        recv_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, int, int, int, bool]:
        from .topology import hs_recv_src_rank

        src = hs_recv_src_rank(self.rank, self.world_size)
        meta = self._hs_meta_recv()
        self._recv_ordered(meta, src=src)
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
                self._hs_recv_buf = self._ensure_buf(self._hs_recv_buf, shape, dtype)
                buf = self._hs_recv_buf
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
        else:
            buf = recv_buf
        self._recv_ordered(buf, src=src)
        if not valid:
            return None, cycle_id, token_pos, seq_len, False
        return buf, cycle_id, token_pos, seq_len, True

    def post_last_stage_snap_verify_recv(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        indices: Sequence[int],
        cycle_id: int,
        token_pos: int,
    ) -> "LastStageSnapVerifyRecv":
        """Post fixed last-stage verify_hs + snap irecvs (no meta / indices)."""
        idx = [int(x) for x in indices]
        slot = self._last_stage_recv
        if (
            slot is None
            or slot.hidden_size != int(hidden_size)
            or slot.dtype != dtype
            or list(slot.indices) != idx
        ):
            slot = LastStageSnapVerifyRecv(
                self, hidden_size, dtype, indices=idx
            )
            self._last_stage_recv = slot
        slot.reset_and_post(cycle_id=int(cycle_id), token_pos=int(token_pos))
        return slot

    def post_snap_fixed_recv(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        src: int,
        indices: Sequence[int],
        cycle_id: int,
        token_pos: int,
    ) -> "PostedSnapFixedRecv":
        """Post fixed early-stage snap payload irecvs (no meta / indices on wire)."""
        idx = [int(x) for x in indices]
        if not idx:
            raise ValueError("post_snap_fixed_recv requires a non-empty indices list")
        src_i = int(src)
        slot = self._snap_recv_by_src.get(src_i)
        if (
            slot is None
            or slot.hidden_size != int(hidden_size)
            or slot.dtype != dtype
            or list(slot.indices) != idx
        ):
            slot = PostedSnapFixedRecv(
                self, hidden_size, dtype, src=src_i, indices=idx
            )
            self._snap_recv_by_src[src_i] = slot
        slot.reset_and_post(cycle_id=int(cycle_id), token_pos=int(token_pos))
        return slot

    def send_last_stage_fixed(
        self,
        *,
        indices: Sequence[int],
        tensors: Sequence[torch.Tensor],
        verify_hs: torch.Tensor,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Send last-stage fixed wire: verify_hs → hs × k (no meta / indices)."""
        idx = [int(x) for x in indices]
        if len(idx) != len(tensors):
            raise ValueError(
                f"last-stage snap indices/tensors length mismatch: "
                f"{len(idx)} vs {len(tensors)}"
            )
        expected = self._hs_shape(hidden_size, 1)
        verify_payload = verify_hs.contiguous()
        if verify_payload.dtype != dtype:
            verify_payload = verify_payload.to(dtype=dtype)
        if tuple(verify_payload.shape) != expected:
            raise ValueError(
                f"verify_hs shape {tuple(verify_payload.shape)} != expected {expected}"
            )
        self._send_ordered(verify_payload, dst=0)
        for t in tensors:
            payload = t.contiguous()
            if payload.dtype != dtype:
                payload = payload.to(dtype=dtype)
            if tuple(payload.shape) != expected:
                raise ValueError(
                    f"snap hs shape {tuple(payload.shape)} != expected {expected}"
                )
            self._send_ordered(payload, dst=0)

    def send_snap_fixed(
        self,
        *,
        indices: Sequence[int],
        tensors: Sequence[torch.Tensor],
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Send early-stage snaps as fixed decode payloads: hs × k only.

        Caller must skip inactive stages / empty ``indices``. Wire order matches
        sorted ``indices``; each hs must be ``(1, 1, hidden_size)``.
        """
        idx = [int(x) for x in indices]
        k = len(idx)
        if k == 0:
            return
        if k != len(tensors):
            raise ValueError(f"snap fixed indices/tensors length mismatch: {k} vs {len(tensors)}")
        expected = self._hs_shape(hidden_size, 1)
        for t in tensors:
            payload = t.contiguous()
            if payload.dtype != dtype:
                payload = payload.to(dtype=dtype)
            if tuple(payload.shape) != expected:
                raise ValueError(
                    f"snap hs shape {tuple(payload.shape)} != expected {expected}"
                )
            self._send_ordered(payload, dst=0)


class PostedSnapFixedRecv:
    """Reusable early-posted fixed snap recv from one stage → rank0.

    Wire: hs × k only (no meta / indices). ``indices`` and decode shape are
    known at init; ``irecv``s are posted immediately on ``reset_and_post``.
    """

    def __init__(
        self,
        p2p: PipelineP2P,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        src: int,
        indices: Sequence[int],
    ) -> None:
        self.p2p = p2p
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.src = int(src)
        self.indices = [int(x) for x in indices]
        if not self.indices:
            raise ValueError("PostedSnapFixedRecv requires non-empty indices")
        shape = p2p._hs_shape(self.hidden_size, 1)
        self._payload_storage: list[torch.Tensor] = [
            torch.empty(shape, dtype=self.dtype, device=p2p.device)
            for _ in self.indices
        ]
        self._works: List[Work] = []
        self._done = False
        self.cycle_id = -1
        self.token_pos = -1

    def reset_and_post(self, *, cycle_id: int, token_pos: int) -> None:
        self.cycle_id = int(cycle_id)
        self.token_pos = int(token_pos)
        self._done = False
        self._works = [
            dist_irecv(buf, src=self.src) for buf in self._payload_storage
        ]

    def wait(self) -> tuple[int, int, bool, dict[int, torch.Tensor]]:
        if self._done:
            return self.cycle_id, self.token_pos, False, {}
        for work in self._works:
            work.wait()
        # Clone so pooled payload buffers can be reused next cycle.
        out = {
            int(idx): buf.clone()
            for idx, buf in zip(self.indices, self._payload_storage)
        }
        self._done = True
        return self.cycle_id, self.token_pos, True, out


class LastStageSnapVerifyRecv:
    """Fixed last-stage recv: verify_hs → hs × k (no meta / indices).

    All payload ``irecv``s are posted in wire order on ``reset_and_post`` so
    transfer overlaps stage forward + rank0 spec. ``wait_verify`` only blocks
    on ``verify_hs``; ``wait_snaps`` drains the already-posted snap works.
    """

    def __init__(
        self,
        p2p: PipelineP2P,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        indices: Sequence[int],
    ) -> None:
        self.p2p = p2p
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.src = int(p2p.num_stages)
        self.indices = [int(x) for x in indices]
        shape = p2p._hs_shape(self.hidden_size, 1)
        self._verify_hs = torch.empty(shape, dtype=self.dtype, device=p2p.device)
        self._payload_storage: list[torch.Tensor] = [
            torch.empty(shape, dtype=self.dtype, device=p2p.device)
            for _ in self.indices
        ]
        self._verify_work: Work | None = None
        self._snap_works: List[Work] = []
        self._verify_done = False
        self._snaps_done = False
        self.cycle_id = -1
        self.token_pos = -1

    def reset_and_post(self, *, cycle_id: int, token_pos: int) -> None:
        self.cycle_id = int(cycle_id)
        self.token_pos = int(token_pos)
        self._verify_done = False
        self._snaps_done = False
        # Wire order: verify_hs then snap hs × k.
        self._verify_work = dist_irecv(self._verify_hs, src=self.src)
        self._snap_works = [
            dist_irecv(buf, src=self.src) for buf in self._payload_storage
        ]

    def wait_verify(self) -> tuple[torch.Tensor, int, int]:
        """Block on the verify_hs irecv only (snap irecvs stay in flight)."""
        if self._verify_done:
            return self._verify_hs, self.cycle_id, self.token_pos
        assert self._verify_work is not None
        self._verify_work.wait()
        self._verify_done = True
        return self._verify_hs, self.cycle_id, self.token_pos

    def wait_snaps(self) -> tuple[int, int, bool, dict[int, torch.Tensor]]:
        if not self._verify_done:
            self.wait_verify()
        if self._snaps_done:
            return self.cycle_id, self.token_pos, False, {}
        for work in self._snap_works:
            work.wait()
        out = {
            int(idx): buf.clone()
            for idx, buf in zip(self.indices, self._payload_storage)
        }
        self._snaps_done = True
        return self.cycle_id, self.token_pos, bool(self.indices), out

    def wait(
        self,
    ) -> tuple[int, int, bool, dict[int, torch.Tensor], torch.Tensor, int]:
        verify_hs, cycle_id, verify_pos = self.wait_verify()
        snap_c, token_pos, valid, out = self.wait_snaps()
        return snap_c, token_pos, valid, out, verify_hs, verify_pos
