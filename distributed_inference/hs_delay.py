"""Ping-pong buffer so stage workers keep last-cycle inbound hs across recv reuse."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .comm import PipelineP2P

from .comm import assert_p2p_meta


class HsInboundPingPong:
    def __init__(self) -> None:
        self._bufs: list[Optional[torch.Tensor]] = [None, None]
        self._valid = [False, False]
        self._read_idx = 0
        self._write_idx = 1
        self.inbound_hs: Optional[torch.Tensor] = None
        self.inbound_valid = False

    def clear(self) -> None:
        self._bufs = [None, None]
        self._valid = [False, False]
        self.inbound_hs = None
        self.inbound_valid = False

    def end_cycle_recv(
        self,
        p2p: PipelineP2P,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        expected_cycle_id: int,
        expected_token_pos: int,
    ) -> None:
        write_idx = self._write_idx
        hs, cycle_id, token_pos, _seq_len, valid = p2p.recv_hs(
            hidden_size,
            dtype,
            recv_buf=self._bufs[write_idx],
        )
        assert_p2p_meta(
            "pipeline_hs",
            cycle_id=cycle_id,
            expected_cycle_id=expected_cycle_id,
            token_pos=token_pos,
            expected_token_pos=expected_token_pos,
            peer_rank=p2p.rank - 1,
            local_rank=p2p.rank,
        )
        if valid and hs is not None:
            self._bufs[write_idx] = hs
            self._valid[write_idx] = True
        else:
            self._valid[write_idx] = False
        self._read_idx, self._write_idx = self._write_idx, self._read_idx

    def begin_cycle(self) -> None:
        self.inbound_valid = bool(self._valid[self._read_idx])
        self.inbound_hs = self._bufs[self._read_idx] if self.inbound_valid else None
