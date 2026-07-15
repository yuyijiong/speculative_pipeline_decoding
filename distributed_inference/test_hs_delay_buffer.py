"""Unit test: pipeline hs delay must survive recv buffer reuse (ping-pong, no clone)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distributed_inference.hs_delay import HsInboundPingPong


class _FakeP2P:
    def __init__(self, payloads: list[torch.Tensor]) -> None:
        self.device = payloads[0].device
        self.rank = 1
        self._payloads = payloads
        self._i = 0

    def recv_hs(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        recv_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, int, int, int, bool]:
        payload = self._payloads[self._i]
        self._i += 1
        if recv_buf is None:
            recv_buf = torch.empty_like(payload)
        recv_buf.copy_(payload)
        return recv_buf, 0, 0, 1, True


if __name__ == "__main__":
    p2p = _FakeP2P(
        [
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
            torch.tensor([[[9.0, 9.0, 9.0, 9.0]]]),
        ]
    )
    slot = HsInboundPingPong()

    slot.end_cycle_recv(
        p2p,
        hidden_size=4,
        dtype=torch.float32,
        expected_cycle_id=0,
        expected_token_pos=0,
    )
    slot.begin_cycle()
    assert slot.inbound_hs is not None
    assert torch.allclose(slot.inbound_hs, torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    assert slot.inbound_valid

    slot.end_cycle_recv(
        p2p,
        hidden_size=4,
        dtype=torch.float32,
        expected_cycle_id=0,
        expected_token_pos=0,
    )
    slot.begin_cycle()
    assert slot.inbound_hs is not None
    assert torch.allclose(slot.inbound_hs, torch.tensor([[[9.0, 9.0, 9.0, 9.0]]]))
    print("hs delay buffer test passed")
