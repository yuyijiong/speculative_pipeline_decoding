"""Rank >= 1 pipeline stage worker."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from .comm import PipelineP2P
from .device import sync_device
from .hs_delay import HsInboundPingPong
from .kv_ops import crop_stage_shard_after_rejection
from .loader import StageRankBundle


class StageWorker:
    def __init__(self, bundle: StageRankBundle, p2p: PipelineP2P) -> None:
        self.b = bundle
        self.p2p = p2p
        self.device = bundle.device
        self.slot_hs: Optional[torch.Tensor] = None
        self.pending_token_id: Optional[int] = None
        self.inbound_hs: Optional[torch.Tensor] = None
        self.inbound_valid = False
        self._hs_ping_pong = HsInboundPingPong()
        self._cycle_forward_sec = 0.0
        self.clear_buffers()

    def clear_buffers(self) -> None:
        self.p2p.wait_all()
        self._hs_ping_pong.clear()
        self.inbound_hs = None
        self.inbound_valid = False

    def set_pending_token(self, token_id: int) -> None:
        self.pending_token_id = int(token_id)

    def wait_comm(self) -> None:
        self.p2p.wait_all()

    def begin_cycle_forward(self) -> None:
        self._cycle_forward_sec = 0.0
        if self.b.stage_idx == 0:
            return
        self._hs_ping_pong.begin_cycle()
        self.inbound_hs = self._hs_ping_pong.inbound_hs
        self.inbound_valid = self._hs_ping_pong.inbound_valid

    def end_cycle_recv_upstream(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        cycle_id: int,
        pipeline_depth: int,
        positions: List[int],
    ) -> None:
        if self.b.stage_idx == 0:
            return
        si = int(self.b.stage_idx)
        upstream = si - 1
        depth = int(pipeline_depth)
        if upstream < depth:
            expected_token_pos = int(positions[upstream])
        else:
            expected_token_pos = int(positions[-1]) if positions else -1
        self._hs_ping_pong.end_cycle_recv(
            self.p2p,
            hidden_size,
            dtype,
            expected_cycle_id=int(cycle_id),
            expected_token_pos=expected_token_pos,
        )

    def _forward_stage(
        self,
        hidden_states: torch.Tensor,
        position: int,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        compute_dtype = next(self.b.layers[0].parameters()).dtype
        if hidden_states.dtype != compute_dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)
        snaps: Dict[int, torch.Tensor] = {}
        pos_t = torch.tensor([[int(position)]], device=self.device, dtype=torch.long)
        with torch.cuda.device(self.device):
            position_embeddings = self.b.rotary_emb(hidden_states, pos_t)
            lo = self.b.stage_layer_start
            hi = self.b.stage_layer_end
            for local_i, layer_idx in enumerate(range(lo, hi)):
                hidden_states = self.b.layers[local_i](
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=pos_t,
                    past_key_values=self.b.stage_cache_view,
                    use_cache=True,
                )
                out_idx = layer_idx + 1
                if out_idx in self.b.snap_want:
                    snaps[out_idx] = hidden_states
        return hidden_states, snaps

    def on_discard(self, crop_length: int, token_id: int) -> None:
        self.p2p.wait_all()
        crop_stage_shard_after_rejection(
            self.b.kv_shard,
            stage_idx=self.b.stage_idx,
            num_stages=self.b.num_stages,
            stage_layer_start=self.b.stage_layer_start,
            num_layers=self.b.num_layers,
            crop_length=int(crop_length),
            linear_layer_indices=self.b.linear_cache_layer_indices,
        )
        self.clear_buffers()
        self.slot_hs = None
        if self.b.stage_idx > 0:
            return
        self.set_pending_token(int(token_id))

    def on_go(
        self,
        *,
        cycle_id: int,
        pipeline_depth: int,
        positions: List[int],
    ) -> tuple[bool, Optional[torch.Tensor], int, int, bool, Dict[int, torch.Tensor]]:
        si = self.b.stage_idx
        if si >= int(pipeline_depth):
            return False, None, -1, -1, False, {}

        pos = int(positions[si])
        if si == 0:
            if self.pending_token_id is None:
                raise RuntimeError("stage0 missing pending token_id before GO forward")
            tid = int(self.pending_token_id)
            self.pending_token_id = None
            hs = self.b.embed_tokens(torch.tensor([[tid]], device=self.device))
            snap0 = {0: hs} if 0 in self.b.local_snap_indices else {}
        else:
            if not self.inbound_valid or self.inbound_hs is None:
                raise RuntimeError(f"stage {si} missing inbound hs for GO(cycle={cycle_id})")
            hs = self.inbound_hs
            snap0 = {}

        sync_device(self.device)
        t_fwd = time.perf_counter()
        hs_out, collected = self._forward_stage(hs, pos)
        sync_device(self.device)
        self._cycle_forward_sec = time.perf_counter() - t_fwd
        self.slot_hs = hs_out
        snaps = {**snap0, **collected}

        send_verify = si == self.b.num_stages - 1 and int(pipeline_depth) == self.b.num_stages
        verify_pos = int(positions[self.b.num_stages - 1]) if send_verify else -1
        return send_verify, hs_out if send_verify else None, verify_pos, pos, True, snaps

    def post_forward_send(
        self,
        hs_out: Optional[torch.Tensor],
        *,
        cycle_id: int,
        pos: int,
        snaps: Dict[int, torch.Tensor],
        send_verify: bool,
        verify_pos: int,
        pipeline_depth: int,
        valid: bool,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        sync_device(self.device)
        if self.p2p.merge_last_stage and self.b.stage_idx == self.b.num_stages - 1:
            return
        idx = sorted(snaps.keys())
        tensors = [snaps[i] for i in idx]
        self.p2p.send_snap_batch(
            cycle_id=cycle_id,
            token_pos=int(pos),
            valid=valid and len(idx) > 0,
            indices=idx,
            tensors=tensors,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        if self.b.stage_idx < self.b.num_stages - 1:
            if valid and hs_out is not None:
                self.p2p.send_hs(
                    hs_out,
                    cycle_id=cycle_id,
                    token_pos=int(pos),
                    valid=True,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
            else:
                self.p2p.send_hs(
                    hs_out if hs_out is not None else torch.empty(1, 1, hidden_size, dtype=dtype, device=self.device),
                    cycle_id=cycle_id,
                    token_pos=int(pos),
                    valid=False,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
        if send_verify and hs_out is not None and not self.p2p.merge_last_stage:
            self.p2p.send_verify_hs(hs_out, cycle_id=cycle_id, token_pos=verify_pos)

    def crop_kv_after_reject(self, crop_length: int) -> None:
        crop_stage_shard_after_rejection(
            self.b.kv_shard,
            stage_idx=self.b.stage_idx,
            num_stages=self.b.num_stages,
            stage_layer_start=self.b.stage_layer_start,
            num_layers=self.b.num_layers,
            crop_length=int(crop_length),
            linear_layer_indices=self.b.linear_cache_layer_indices,
        )

    def relay_invalid_hs_downstream(
        self,
        *,
        cycle_id: int,
        pos: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        if self.b.stage_idx >= self.b.num_stages - 1:
            return
        dummy = torch.empty(1, 1, hidden_size, dtype=dtype, device=self.device)
        self.p2p.send_hs(
            dummy,
            cycle_id=cycle_id,
            token_pos=int(pos),
            valid=False,
            hidden_size=hidden_size,
            dtype=dtype,
        )
