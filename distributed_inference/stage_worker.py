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
        self._forward_done_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if self.device.type == "cuda" else None
        )
        self.reset_profile_timing()
        self.clear_buffers()

    def clear_buffers(self) -> None:
        self.p2p.wait_all()
        self._hs_ping_pong.clear()
        self.inbound_hs = None
        self.inbound_valid = False

    def set_pending_token(self, token_id: int) -> None:
        self.pending_token_id = int(token_id)

    def reset_profile_timing(self) -> None:
        n_layers = int(self.b.num_layers)
        self._profile_stage_forward_sec = 0.0
        self._profile_stage_forward_active_steps = 0
        self._profile_layer_forward_sec = [0.0] * n_layers
        self._profile_layer_forward_count = [0] * n_layers
        self._profile_stage0_hs_send_sec = 0.0
        self._profile_stage0_hs_send_steps = 0
        self._profile_last_stage_hs_send_sec = 0.0
        self._profile_last_stage_hs_send_steps = 0

    def _record_layer_forward(self, layer_idx: int, dt: float) -> None:
        idx = int(layer_idx)
        if 0 <= idx < len(self._profile_layer_forward_sec):
            self._profile_layer_forward_sec[idx] += float(dt)
            self._profile_layer_forward_count[idx] += 1

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
        *,
        profile_timing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        compute_dtype = next(self.b.layers[0].parameters()).dtype
        if hidden_states.dtype != compute_dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)
        snaps: Dict[int, torch.Tensor] = {}
        pos_t = torch.tensor([[int(position)]], device=self.device, dtype=torch.long)
        layer_events = []
        use_cuda_events = bool(profile_timing and self.device.type == "cuda")
        with torch.cuda.device(self.device):
            position_embeddings = self.b.rotary_emb(hidden_states, pos_t)
            lo = self.b.stage_layer_start
            hi = self.b.stage_layer_end
            for local_i, layer_idx in enumerate(range(lo, hi)):
                if use_cuda_events:
                    stream = torch.cuda.current_stream(device=self.device)
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                    ev_start.record(stream)
                elif profile_timing:
                    sync_device(self.device)
                    t_layer = time.perf_counter()
                hidden_states = self.b.layers[local_i](
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=pos_t,
                    past_key_values=self.b.stage_cache_view,
                    use_cache=True,
                )
                if use_cuda_events:
                    ev_end.record(stream)
                    layer_events.append((int(layer_idx), ev_start, ev_end))
                elif profile_timing:
                    sync_device(self.device)
                    self._record_layer_forward(layer_idx, time.perf_counter() - t_layer)
                out_idx = layer_idx + 1
                if out_idx in self.b.snap_want:
                    snaps[out_idx] = hidden_states
        if layer_events:
            layer_events[-1][2].synchronize()
            for layer_idx, ev_start, ev_end in layer_events:
                self._record_layer_forward(layer_idx, ev_start.elapsed_time(ev_end) / 1000.0)
        if self._forward_done_event is not None:
            self._forward_done_event.record(torch.cuda.current_stream(device=self.device))
        return hidden_states, snaps

    def _sync_for_profile(self, profile_timing: bool) -> None:
        if profile_timing:
            sync_device(self.device)

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
        profile_timing: bool = False,
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

        if profile_timing:
            self._sync_for_profile(profile_timing)
            t_fwd = time.perf_counter()
            hs_out, collected = self._forward_stage(hs, pos, profile_timing=True)
            self._sync_for_profile(profile_timing)
            self._cycle_forward_sec = time.perf_counter() - t_fwd
            self._profile_stage_forward_sec += float(self._cycle_forward_sec)
            self._profile_stage_forward_active_steps += 1
        else:
            hs_out, collected = self._forward_stage(hs, pos, profile_timing=False)
            self._cycle_forward_sec = 0.0
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
        profile_timing: bool = False,
    ) -> None:
        # Ensure forward kernels finished writing hs/snaps before NCCL reads them.
        # Prefer a stream event over torch.cuda.synchronize (full-device drain).
        if self._forward_done_event is not None:
            torch.cuda.current_stream(device=self.device).wait_event(self._forward_done_event)
        idx = sorted(self.b.local_snap_indices)
        if send_verify and hs_out is not None:
            # Fixed wire: verify_hs → hs × k. Time only verify_hs send for fair
            # comparison with stage0 send_hs.
            if not valid:
                raise RuntimeError(
                    f"last stage must be valid when sending verify_hs "
                    f"(cycle={cycle_id}, pos={pos})"
                )
            missing = [i for i in idx if i not in snaps]
            if missing:
                raise KeyError(
                    f"stage {self.b.stage_idx} missing snap indices {missing}; "
                    f"have {sorted(snaps)}"
                )
            tensors = [snaps[i] for i in idx]
            verify_payload = hs_out.contiguous()
            if verify_payload.dtype != dtype:
                verify_payload = verify_payload.to(dtype=dtype)
            if profile_timing and self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            t0 = time.perf_counter() if profile_timing else 0.0
            # send_last_stage_fixed sends verify then snaps; time verify alone by
            # splitting: send verify first via ordered path timing, then snaps.
            expected = self.p2p._hs_shape(hidden_size, 1)
            if tuple(verify_payload.shape) != expected:
                raise ValueError(
                    f"verify_hs shape {tuple(verify_payload.shape)} != expected {expected}"
                )
            self.p2p._send_ordered(verify_payload, dst=0)
            if profile_timing:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(device=self.device)
                self._profile_last_stage_hs_send_sec += time.perf_counter() - t0
                self._profile_last_stage_hs_send_steps += 1
            for t in tensors:
                payload = t.contiguous()
                if payload.dtype != dtype:
                    payload = payload.to(dtype=dtype)
                if tuple(payload.shape) != expected:
                    raise ValueError(
                        f"snap hs shape {tuple(payload.shape)} != expected {expected}"
                    )
                self.p2p._send_ordered(payload, dst=0)
            return
        # Early-stage: fixed wire (hs × k only). Inactive / empty local snaps skip.
        if valid and idx:
            missing = [i for i in idx if i not in snaps]
            if missing:
                raise KeyError(
                    f"stage {self.b.stage_idx} missing snap indices {missing}; "
                    f"have {sorted(snaps)}"
                )
            self.p2p.send_snap_fixed(
                indices=idx,
                tensors=[snaps[i] for i in idx],
                hidden_size=hidden_size,
                dtype=dtype,
            )
        if self.b.stage_idx < self.b.num_stages - 1:
            from .topology import hs_send_dst_rank

            dst = hs_send_dst_rank(self.p2p.rank, self.p2p.world_size)
            time_hs = bool(profile_timing and self.b.stage_idx == 0)
            meta = self.p2p._hs_meta_send()
            if valid and hs_out is not None:
                seq_len = int(hs_out.shape[1])
                meta[0] = int(cycle_id)
                meta[1] = int(pos)
                meta[2] = 1
                meta[3] = seq_len
                payload = hs_out.contiguous()
                if payload.dtype != dtype:
                    payload = payload.to(dtype=dtype)
            else:
                meta[0] = int(cycle_id)
                meta[1] = int(pos)
                meta[2] = 0
                meta[3] = 1
                payload = self.p2p._dummy_hs(int(hidden_size), dtype)
            self.p2p._send_ordered(meta, dst=dst)
            if time_hs and self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            t0 = time.perf_counter() if time_hs else 0.0
            self.p2p._send_ordered(payload, dst=dst)
            if time_hs:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(device=self.device)
                self._profile_stage0_hs_send_sec += time.perf_counter() - t0
                self._profile_stage0_hs_send_steps += 1
        del verify_pos
        del pipeline_depth

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
        self.p2p.send_hs(
            self.p2p._dummy_hs(int(hidden_size), dtype),
            cycle_id=cycle_id,
            token_pos=int(pos),
            valid=False,
            hidden_size=hidden_size,
            dtype=dtype,
        )
