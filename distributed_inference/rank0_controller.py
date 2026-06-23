"""Rank-0 decode controller: verify, speculation, snap fusion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .pipeline_model import (
    _sampling_probs_hf_style,
    _verify_pipeline_draft_token,
)

from .loader import Rank0Bundle


class Rank0Controller:
    def __init__(self, bundle: Rank0Bundle, *, use_deepest: bool) -> None:
        self.b = bundle
        self.use_deepest = bool(use_deepest)
        self.device = bundle.device
        self.n = int(bundle.num_stages)
        self.spec_past_kv = DynamicCache()
        self.completed_snaps: Dict[int, Dict[int, torch.Tensor]] = {}
        self.generated_ids: List[int] = []
        self.token_acceptance: List[bool] = []
        self.draft_full_q: Dict[int, torch.Tensor] = {}
        self.pipeline: List[Dict[str, Any]] = []
        self.prev_evicted_snap: Optional[Dict[int, torch.Tensor]] = None
        self.prev_evicted_pos: Optional[int] = None
        self.verified_up_to = 0
        self.next_position = 0
        self.cycle_id = 0
        self.snap_buf: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
        self.last_timing: dict[str, float] = {}

    def _rollback_spec_kv(self, crop_length: int) -> None:
        if self.spec_past_kv.get_seq_length() > crop_length:
            self.spec_past_kv.crop(crop_length)

    def merge_snap_batch(self, cycle_id: int, pos: int, shards: Dict[int, torch.Tensor]) -> None:
        gbuf = self.snap_buf.setdefault(int(cycle_id), {})
        pbuf = gbuf.setdefault(int(pos), {})
        pbuf.update(shards)

    def purge_snaps_from_pos(self, crop_length: int) -> None:
        for pos in [p for p in self.completed_snaps if p >= crop_length]:
            del self.completed_snaps[pos]
        for cid in list(self.snap_buf.keys()):
            for pos in [p for p in self.snap_buf[cid] if p >= crop_length]:
                del self.snap_buf[cid][pos]

    def clear_bufs_from_cycle(self, from_cycle: int) -> None:
        for g in [k for k in self.snap_buf if k >= from_cycle]:
            del self.snap_buf[g]

    def _initial_snap(self, hs: torch.Tensor) -> Dict[int, torch.Tensor]:
        snap: Dict[int, torch.Tensor] = {}
        if 0 in self.b.snap_want:
            snap[0] = hs
        return snap

    def _ensure_snap_embedding(
        self,
        snap: Dict[int, torch.Tensor],
        pos: int,
        active_by_pos: Dict[int, Dict[str, Any]],
    ) -> Dict[int, torch.Tensor]:
        if 0 in snap:
            return snap
        if int(pos) in active_by_pos:
            out = dict(snap)
            out[0] = active_by_pos[int(pos)]["hs"]
            return out
        raise KeyError(
            f"Missing HF index 0 in snapshot for position {pos}; "
            f"have indices {sorted(int(k) for k in snap.keys())}."
        )

    def _fuse_row(self, snap: Dict[int, torch.Tensor], depth: int) -> torch.Tensor:
        aggr_i = self.b.stage_depth_to_aggr_idx[int(depth)]
        hf_indices = self.b.aggr_feature_indices[aggr_i]
        proj = self.b.speculation_module.aggr_projs[aggr_i]
        vecs = [snap[int(idx)] for idx in hf_indices]
        return proj(torch.cat(vecs, dim=-1))

    def _g0_row(self, hs: torch.Tensor) -> torch.Tensor:
        return self.b.speculation_module.aggr_projs[0](hs)

    def _choose_depth(
        self,
        snap: Dict[int, torch.Tensor],
        nominal_depth: int,
        *,
        search_hi: Optional[int] = None,
    ) -> int:
        nd = int(nominal_depth)
        if nd <= 0:
            return nd
        if not self.use_deepest:
            hi = nd
        else:
            hi = int(search_hi) if search_hi is not None else self.n
            if hi > self.n:
                hi = self.n
        for d in range(hi, -1, -1):
            aggr_i = self.b.stage_depth_to_aggr_idx[d]
            hf_indices = self.b.aggr_feature_indices[aggr_i]
            if all(int(idx) in snap for idx in hf_indices):
                return d
        return nd

    def _snap_for_pos(self, pos: int, cycle_id: int) -> Dict[int, torch.Tensor]:
        if pos in {int(e["pos"]) for e in self.pipeline}:
            entry = next(e for e in self.pipeline if int(e["pos"]) == int(pos))
            snap = dict(entry["snap"])
        elif pos in self.completed_snaps:
            snap = dict(self.completed_snaps[pos])
        else:
            snap = {}
        gbuf = self.snap_buf.get(int(cycle_id), {})
        if pos in gbuf:
            snap.update(gbuf[pos])
        return snap

    def run_spec_forward(self, cycle_id: int) -> torch.Tensor:
        if not self.pipeline:
            raise RuntimeError("Pipeline empty before speculation.")
        newest_pos = int(self.pipeline[0]["pos"])
        pipeline_depth = len(self.pipeline)
        oldest_needed = newest_pos - self.n + 1
        pos_start = 0 if oldest_needed < 0 else oldest_needed
        warmup = oldest_needed < 0

        active_by_pos = {int(e["pos"]): e for e in self.pipeline}
        rows: List[torch.Tensor] = []
        pos_list: List[int] = []
        has_evicted = self.prev_evicted_snap is not None and self.prev_evicted_pos is not None
        fused_search_hi = (self.n - 1) if (self.use_deepest and has_evicted) else None

        if has_evicted:
            ev_snap = self._ensure_snap_embedding(
                dict(self.prev_evicted_snap), int(self.prev_evicted_pos), active_by_pos
            )
            ev_depth = self._choose_depth(ev_snap, self.n, search_hi=self.n)
            rows.append(self._fuse_row(ev_snap, ev_depth))
            pos_list.append(int(self.prev_evicted_pos))

        for pos in range(pos_start, newest_pos + 1):
            depth_in_window = newest_pos - pos
            snap_src = self._snap_for_pos(int(pos), cycle_id)
            if int(pos) in active_by_pos:
                snap_src = {**snap_src, **active_by_pos[int(pos)]["snap"]}
            if depth_in_window == 0:
                rows.append(self._g0_row(active_by_pos[newest_pos]["hs"]))
            else:
                if not snap_src:
                    raise KeyError(f"Missing snapshot for position {pos}.")
                snap_src = self._ensure_snap_embedding(snap_src, int(pos), active_by_pos)
                if warmup and depth_in_window >= pipeline_depth:
                    depth = self._choose_depth(snap_src, self.n, search_hi=self.n)
                else:
                    depth = self._choose_depth(
                        snap_src, depth_in_window, search_hi=fused_search_hi
                    )
                rows.append(self._fuse_row(snap_src, depth))
            pos_list.append(pos)

        cur_in = torch.cat(rows, dim=1)
        min_p = min(pos_list)
        self._rollback_spec_kv(min_p)
        pos_ids = torch.tensor([pos_list], device=self.device, dtype=torch.long)
        proc = self.b.speculation_module.forward_inference_with_rotary(
            cur_in,
            pos_ids,
            attention_mask=None,
            past_key_values=self.spec_past_kv,
            use_cache=True,
        )
        proc = self.b.final_norm(proc)
        return self.b.speculation_module.lm_head(proc[:, -1:, :])

    def verify_with_hs(
        self,
        hs: torch.Tensor,
        target_pos: int,
        speculated_id: int,
        *,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[bool, int]:
        hs_normed = self.b.final_norm(hs)
        logits = self.b.lm_head(hs_normed)
        vlog1 = logits[0, 0]
        if greedy:
            return _verify_pipeline_draft_token(
                vlog1, speculated_id, True, temperature, top_k, top_p
            )
        q_full = self.draft_full_q[target_pos]
        return _verify_pipeline_draft_token(
            vlog1, speculated_id, False, temperature, top_k, top_p, q_full
        )

    def sample_spec_token(
        self,
        spec_logits: torch.Tensor,
        next_position: int,
        *,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        logits_1d = spec_logits[0, 0]
        meta = self.b.draft_vocab_meta
        if greedy:
            pick = logits_1d.argmax()
            if meta["use_draft_vocab"]:
                tid = meta["_draft_token_ids"].to(logits_1d.device)
                return int(tid[pick].item())
            return int(pick.item())
        if meta["use_draft_vocab"]:
            probs_d = _sampling_probs_hf_style(
                logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
            )
            v_full = int(self.b.vocab_size)
            q_full = torch.zeros(v_full, device=logits_1d.device, dtype=probs_d.dtype)
            tid = meta["_draft_token_ids"].to(logits_1d.device).long()
            q_full.index_add_(0, tid, probs_d)
            d_idx = int(torch.multinomial(probs_d, 1).item())
            self.draft_full_q[next_position] = q_full.detach()
            return int(tid[d_idx].item())
        q_full = _sampling_probs_hf_style(
            logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
        )
        self.draft_full_q[next_position] = q_full.detach()
        return int(torch.multinomial(q_full, 1).item())

    def ingest_g0_snap(self, pos: int, token_id: int, cycle_id: int) -> None:
        emb = self.b.embed_tokens(torch.tensor([[token_id]], device=self.device))
        gbuf = self.snap_buf.setdefault(int(cycle_id), {})
        pbuf = gbuf.setdefault(int(pos), {})
        pbuf[0] = emb

    def pipeline_positions(self) -> List[int]:
        return [int(e["pos"]) for e in self.pipeline]

    def apply_reject(self, crop_length: int, verified_next_id: int, s0: int) -> None:
        target_gen_idx = int(crop_length) - int(s0)
        self.generated_ids = self.generated_ids[:target_gen_idx]
        self.generated_ids.append(int(verified_next_id))
        self.token_acceptance = self.token_acceptance[:target_gen_idx]
        self.token_acceptance.append(False)
        self._rollback_spec_kv(int(crop_length))
        self.purge_snaps_from_pos(int(crop_length))
        for pk in [k for k in list(self.draft_full_q.keys()) if k >= crop_length]:
            del self.draft_full_q[pk]
        emb = self.b.embed_tokens(
            torch.tensor([[int(verified_next_id)]], device=self.device)
        )
        self.pipeline = [
            {
                "hs": emb,
                "pos": int(crop_length),
                "snap": self._initial_snap(emb),
            }
        ]
        self.next_position = int(crop_length) + 1
        self.verified_up_to = int(crop_length) + 1
        self.prev_evicted_snap = None
        self.prev_evicted_pos = None
        self.clear_bufs_from_cycle(self.cycle_id)

    def commit_completed_snap(self, completed_pos: int, cycle_id: int) -> None:
        entry = next(
            (e for e in self.pipeline if int(e["pos"]) == int(completed_pos)), None
        )
        if entry is not None:
            snap = dict(entry["snap"])
            gbuf = self.snap_buf.get(int(cycle_id), {})
            if int(completed_pos) in gbuf:
                snap.update(gbuf[int(completed_pos)])
            if 0 not in snap and 0 in self.b.snap_want:
                snap[0] = entry["hs"]
        else:
            snap = self._snap_for_pos(int(completed_pos), cycle_id)
        self.completed_snaps[int(completed_pos)] = snap

    def build_position_snapshots_from_prefill(
        self, tensors_by_idx: Dict[int, torch.Tensor], seq_len: int
    ) -> None:
        if 0 not in tensors_by_idx:
            raise KeyError("Prefill snapshots missing HF index 0 (embedding).")
        for pos in range(seq_len):
            snap: Dict[int, torch.Tensor] = {}
            for idx, ten in tensors_by_idx.items():
                snap[int(idx)] = ten[:, pos : pos + 1, :]
            self.completed_snaps[pos] = snap

    def init_spec_kv_from_prefill(self, s0: int) -> None:
        prefill_cache_len = max(0, s0 - self.n + 1)
        if prefill_cache_len <= 0:
            return
        prefill_rows = []
        for pos in range(prefill_cache_len):
            depth = self._choose_depth(self.completed_snaps[pos], self.n)
            prefill_rows.append(self._fuse_row(self.completed_snaps[pos], depth))
        prefill_gn = torch.cat(prefill_rows, dim=1)
        prefill_pos = torch.arange(prefill_cache_len, device=self.device, dtype=torch.long).unsqueeze(
            0
        )
        with torch.inference_mode():
            self.b.speculation_module.forward_inference_with_rotary(
                prefill_gn,
                prefill_pos,
                past_key_values=self.spec_past_kv,
                use_cache=True,
            )
