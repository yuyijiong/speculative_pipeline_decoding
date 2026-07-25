"""Rank-0 decode controller: verify, speculation, snap fusion (v11)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline_model import (  # noqa: E402
    _sampling_probs_hf_style,
    _verify_pipeline_draft_token,
)

from .cuda_graph_opts import (  # noqa: E402
    SpecAggrGraphRunner,
    VerifyGraphRunner,
    module_param_dtype,
)
from .loader import Rank0Bundle


class Rank0Controller:
    def __init__(
        self,
        bundle: Rank0Bundle,
        *,
        use_deepest: bool = True,
        enable_cuda_graphs: bool = True,
    ) -> None:
        if use_deepest is False:
            warnings.warn(
                "Rank0Controller inference now always uses deepest completed-token rows; "
                "`use_deepest=False` is ignored.",
                stacklevel=2,
            )
        self.b = bundle
        self.device = bundle.device
        self.n = int(bundle.num_stages)
        self.spec_past_kv = DynamicCache()
        self.spec_cache_len = 0
        self.completed_snaps: Dict[int, Dict[int, torch.Tensor]] = {}
        self.generated_ids: List[int] = []
        self.token_acceptance: List[bool] = []
        self.draft_full_q: Dict[int, torch.Tensor] = {}
        self.pipeline: List[Dict[str, Any]] = []
        self.pending_deepest_snap: Optional[Dict[int, torch.Tensor]] = None
        self.pending_deepest_pos: Optional[int] = None
        self.verified_up_to = 0
        self.next_position = 0
        self.cycle_id = 0
        self.snap_buf: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
        self.last_timing: dict[str, float] = {}
        self.enable_cuda_graphs = bool(enable_cuda_graphs) and self.device.type == "cuda"
        self.spec_aggr_graph: Optional[SpecAggrGraphRunner] = None
        self.verify_graph: Optional[VerifyGraphRunner] = None
        self._cuda_graphs_ready = False

    def _rollback_spec_kv(self, target_length: int) -> None:
        if self.spec_past_kv.get_seq_length() > target_length:
            self.spec_past_kv.crop(target_length)

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

    def capture_cuda_graphs(self) -> None:
        """Capture aggregation + verify graphs once after prefill (no-op if disabled)."""
        if not self.enable_cuda_graphs or self._cuda_graphs_ready:
            return
        dtype = module_param_dtype(self.b.speculation_module.aggr_projs[0])
        self.spec_aggr_graph = SpecAggrGraphRunner(
            num_stages=self.n,
            hidden_size=int(self.b.hidden_size),
            aggr_projs=self.b.speculation_module.aggr_projs,
            aggr_feature_indices=self.b.aggr_feature_indices,
            stage_depth_to_aggr_idx=self.b.stage_depth_to_aggr_idx,
            device=self.device,
            dtype=dtype,
        )
        self.spec_aggr_graph.capture_all()
        self.verify_graph = VerifyGraphRunner(
            final_norm=self.b.final_norm,
            lm_head=self.b.lm_head,
            hidden_size=int(self.b.hidden_size),
            vocab_size=int(self.b.vocab_size),
            device=self.device,
            dtype=module_param_dtype(self.b.lm_head),
        )
        self.verify_graph.capture()
        self._cuda_graphs_ready = True

    def _fuse_row(self, snap: Dict[int, torch.Tensor], depth: int) -> torch.Tensor:
        aggr_i = self.b.stage_depth_to_aggr_idx[int(depth)]
        hf_indices = self.b.aggr_feature_indices[aggr_i]
        proj = self.b.speculation_module.aggr_projs[aggr_i]
        vecs = [snap[int(idx)] for idx in hf_indices]
        return proj(torch.cat(vecs, dim=-1))

    def _g0_row(self, hs: torch.Tensor) -> torch.Tensor:
        return self.b.speculation_module.aggr_projs[0](hs)

    def _completed_token_depth(self) -> int:
        return self.n

    def _staircase_depth_for_pos(self, pos: int, newest_pos: int) -> int:
        delta = int(newest_pos) - int(pos)
        return min(self.n, max(0, delta))

    def _choose_depth(
        self,
        snap: Dict[int, torch.Tensor],
        nominal_depth: int,
        *,
        cap_at_n_minus_1: bool = False,
    ) -> int:
        nd = int(nominal_depth)
        if nd <= 0:
            return nd
        hi = (self.n - 1) if cap_at_n_minus_1 else self.n
        for d in range(hi, -1, -1):
            aggr_i = self.b.stage_depth_to_aggr_idx[d]
            hf_indices = self.b.aggr_feature_indices[aggr_i]
            if all(int(idx) in snap for idx in hf_indices):
                return d
        return nd

    def _build_spec_rows_at_sync(
        self,
    ) -> Tuple[List[int], int, List[Dict[int, torch.Tensor]], List[int]]:
        """
        Build row metadata for one speculation step under the v11 deepest-cache invariant.

        The speculation KV cache holds a contiguous prefix of completed tokens as ``g_n``.
        Active pipeline tokens are transient rows discarded after each forward.

        Returns
        -------
        pos_list
            Contiguous positions for the fused rows (oldest→newest).
        keep_len_after
            Spec KV length to keep after the forward (deepest prefix only).
        row_snaps
            Snapshots used for each row (oldest→newest); for graph / eager fuse.
        row_depths
            Chosen aggregation depths for each row (oldest→newest).
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline empty before speculation.")
        newest_pos = int(self.pipeline[0]["pos"])
        active_by_pos = {int(e["pos"]): e for e in self.pipeline}
        pos_list: List[int] = []
        row_snaps: List[Dict[int, torch.Tensor]] = []
        row_depths: List[int] = []
        keep_len_after = int(self.spec_cache_len)
        cap_active = False

        if (self.pending_deepest_snap is None) != (self.pending_deepest_pos is None):
            raise RuntimeError("pending_deepest_snap and pending_deepest_pos must be set together.")
        if self.pending_deepest_snap is not None and self.pending_deepest_pos is not None:
            p = int(self.pending_deepest_pos)
            if p != int(self.spec_cache_len):
                raise RuntimeError(
                    f"Pending deepest row must extend the cached prefix contiguously: "
                    f"pending_pos={p}, spec_cache_len={self.spec_cache_len}."
                )
            ev_snap = self._ensure_snap_embedding(
                dict(self.pending_deepest_snap), p, active_by_pos
            )
            ev_depth = self._choose_depth(ev_snap, self._completed_token_depth())
            pos_list.append(p)
            row_snaps.append(ev_snap)
            row_depths.append(int(ev_depth))
            keep_len_after += 1
            cap_active = True

        active_entries = sorted(self.pipeline, key=lambda e: int(e["pos"]))
        expected_first_active = keep_len_after
        first_active = int(active_entries[0]["pos"])
        if first_active != expected_first_active:
            raise RuntimeError(
                f"Active pipeline rows must start after the deepest prefix: "
                f"first_active={first_active}, expected={expected_first_active}."
            )
        for entry in active_entries:
            pos = int(entry["pos"])
            pipe_depth = self._staircase_depth_for_pos(pos, newest_pos)
            if pipe_depth == 0:
                row_snaps.append({0: entry["hs"]})
                row_depths.append(0)
            else:
                snap_src = self._ensure_snap_embedding(dict(entry["snap"]), pos, active_by_pos)
                depth = self._choose_depth(
                    snap_src, pipe_depth, cap_at_n_minus_1=cap_active
                )
                row_snaps.append(snap_src)
                row_depths.append(int(depth))
            pos_list.append(pos)
        return pos_list, keep_len_after, row_snaps, row_depths

    def _fuse_rows_eager(
        self,
        row_snaps: List[Dict[int, torch.Tensor]],
        row_depths: List[int],
    ) -> torch.Tensor:
        rows = [
            self._fuse_row(snap, depth) for snap, depth in zip(row_snaps, row_depths)
        ]
        return torch.cat(rows, dim=1)

    def run_spec_forward(self, cycle_id: int) -> torch.Tensor:
        del cycle_id  # snaps for active rows live on pipeline entries from prior cycles
        pos_list, keep_len_after, row_snaps, row_depths = self._build_spec_rows_at_sync()
        num_active = len(self.pipeline)
        has_pending = self.pending_deepest_snap is not None

        cur_in: Optional[torch.Tensor] = None
        if self.spec_aggr_graph is not None:
            cur_in = self.spec_aggr_graph.try_replay(
                num_active=num_active,
                has_pending=has_pending,
                row_snaps=row_snaps,
                row_depths=row_depths,
            )
        if cur_in is None:
            cur_in = self._fuse_rows_eager(row_snaps, row_depths)

        cur_cache_len = int(self.spec_past_kv.get_seq_length())
        first_pos = int(pos_list[0])
        if cur_cache_len != first_pos:
            raise RuntimeError(
                f"Spec cache invariant violated: cache_len={cur_cache_len}, first_new_pos={first_pos}."
            )
        expected_pos = list(range(first_pos, first_pos + len(pos_list)))
        if [int(p) for p in pos_list] != expected_pos:
            raise RuntimeError(
                f"Spec rows must be contiguous so cache_position matches RoPE positions: "
                f"pos_list={pos_list}, expected={expected_pos}."
            )
        pos_ids = torch.tensor([pos_list], device=self.device, dtype=torch.long)
        proc = self.b.speculation_module.forward_inference_with_rotary(
            cur_in,
            pos_ids,
            attention_mask=None,
            past_key_values=self.spec_past_kv,
            use_cache=True,
        )
        self._rollback_spec_kv(int(keep_len_after))
        self.spec_cache_len = int(keep_len_after)
        if self.pending_deepest_snap is not None:
            self.pending_deepest_snap = None
            self.pending_deepest_pos = None
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
        profile_timing: bool = False,
        input_staged: bool = False,
    ) -> Tuple[bool, int] | Tuple[bool, int, Dict[str, float]]:
        """Verify draft token against target logits from ``hs``.

        When ``profile_timing`` is True, also returns a dict with contiguous
        host-wall splits (``perf_counter``). If ``input_staged`` is True, the
        graph input was already copied during ``recv_verify`` and ``copy_sec``
        is 0. ``copy_sec + kernel_sec + decide_sec`` equals the profiled call wall:
        - ``copy_sec``: dtype convert + static-buffer copy (0 if staged / eager)
        - ``kernel_sec``: ``final_norm + lm_head`` (graph replay or eager)
        - ``decide_sec``: argmax/sampling (includes D2H sync)
        - ``used_graph``: 1.0 if CUDA graph path, else 0.0
        """
        import time

        if not profile_timing:
            if self.verify_graph is not None:
                logits = self.verify_graph.run(hs)
            else:
                logits = self.b.lm_head(self.b.final_norm(hs))
            vlog1 = logits[0, 0]
            if greedy:
                accepted, next_id = _verify_pipeline_draft_token(
                    vlog1, speculated_id, True, temperature, top_k, top_p
                )
            else:
                q_full = self.draft_full_q[target_pos]
                accepted, next_id = _verify_pipeline_draft_token(
                    vlog1, speculated_id, False, temperature, top_k, top_p, q_full
                )
            return bool(accepted), int(next_id)

        copy_sec = 0.0
        kernel_sec = 0.0
        used_graph = 0.0
        t_all = time.perf_counter()
        if self.verify_graph is not None:
            if self.device.type == "cuda":
                logits, copy_sec, kernel_sec, graphed = self.verify_graph.run_profiled(
                    hs, input_staged=input_staged
                )
                used_graph = 1.0 if graphed else 0.0
            else:
                t0 = time.perf_counter()
                logits = self.verify_graph.run(hs)
                kernel_sec = time.perf_counter() - t0
                used_graph = 1.0
        else:
            t0 = time.perf_counter()
            logits = self.b.lm_head(self.b.final_norm(hs))
            if self.device.type == "cuda":
                torch.cuda.current_stream(device=self.device).synchronize()
            kernel_sec = time.perf_counter() - t0

        # Contiguous with copy/kernel: decide starts immediately after.
        t_decide = time.perf_counter()
        vlog1 = logits[0, 0]
        if greedy:
            accepted, next_id = _verify_pipeline_draft_token(
                vlog1, speculated_id, True, temperature, top_k, top_p
            )
        else:
            q_full = self.draft_full_q[target_pos]
            accepted, next_id = _verify_pipeline_draft_token(
                vlog1, speculated_id, False, temperature, top_k, top_p, q_full
            )
        decide_sec = time.perf_counter() - t_decide
        # Fold any residual host glue so copy+kernel+decide == call wall.
        residual = (time.perf_counter() - t_all) - (
            float(copy_sec) + float(kernel_sec) + float(decide_sec)
        )
        if residual > 0.0:
            decide_sec = float(decide_sec) + float(residual)
        return (
            bool(accepted),
            int(next_id),
            {
                "copy_sec": float(copy_sec),
                "kernel_sec": float(kernel_sec),
                "decide_sec": float(decide_sec),
                "used_graph": float(used_graph),
            },
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

    def apply_reject(
        self,
        crop_length: int,
        verified_next_id: int,
        s0: int,
        *,
        completed_pos: int,
    ) -> None:
        target_gen_idx = int(crop_length) - int(s0)
        self.generated_ids = self.generated_ids[:target_gen_idx]
        self.generated_ids.append(int(verified_next_id))
        self.token_acceptance = self.token_acceptance[:target_gen_idx]
        self.token_acceptance.append(False)
        self._rollback_spec_kv(int(self.spec_cache_len))
        if int(completed_pos) in self.completed_snaps:
            self.pending_deepest_snap = dict(self.completed_snaps[int(completed_pos)])
            self.pending_deepest_pos = int(completed_pos)
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
            snap = {}
            gbuf = self.snap_buf.get(int(cycle_id), {})
            if int(completed_pos) in gbuf:
                snap.update(gbuf[int(completed_pos)])
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
        """Cache all prompt positions ``0..s0-1`` as deepest ``g_n`` rows at once."""
        if s0 <= 0:
            self.spec_cache_len = 0
            return
        prefill_rows = []
        for pos in range(s0):
            depth = self._choose_depth(self.completed_snaps[pos], self._completed_token_depth())
            prefill_rows.append(self._fuse_row(self.completed_snaps[pos], depth))
        prefill_gn = torch.cat(prefill_rows, dim=1)
        prefill_pos = torch.arange(s0, device=self.device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            self.b.speculation_module.forward_inference_with_rotary(
                prefill_gn,
                prefill_pos,
                past_key_values=self.spec_past_kv,
                use_cache=True,
            )
        self.spec_cache_len = int(s0)
