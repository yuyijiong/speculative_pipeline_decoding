"""Rank0-only prefill and KV shard distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from . import _paths  # noqa: F401
from pipeline_linear_cache import make_pipeline_dynamic_cache

from .cache import split_unified_cache_into_shards
from .cache_meta import cache_layer_types_from_config
from .dist_io import dist_broadcast
from .dist_log import dist_log
from .device import PhaseTimeout, sync_device
from .kv_transfer import recv_cache_shard, send_cache_shard
from .loader import PrefillRank0Bundle, StageRankBundle
from .pipeline_model import _sampling_probs_hf_style
from .topology import rank_for_stage


@dataclass
class PrefillResult:
    first_token_id: int
    seq_len: int
    tensors_by_idx: Dict[int, torch.Tensor]


def _extract_prefill_snapshots(
    pipe,
    all_hs: tuple[torch.Tensor, ...],
) -> Dict[int, torch.Tensor]:
    snap_want = pipe._snap_indices_needed()
    out: Dict[int, torch.Tensor] = {}
    for idx in sorted(snap_want):
        if idx < len(all_hs):
            out[int(idx)] = all_hs[int(idx)]
    return out


def run_prefill(
    *,
    rank: int,
    device: torch.device,
    input_ids: torch.LongTensor,
    prefill_bundle: PrefillRank0Bundle | None,
    worker_bundle: StageRankBundle | None,
    dtype: torch.dtype,
    timeout: PhaseTimeout,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    merge_last_stage: bool = False,
) -> PrefillResult | None:
    timeout.check()
    num_stages = (
        int(prefill_bundle.pipe.num_stages)
        if prefill_bundle is not None
        else int(worker_bundle.num_stages)
    )

    if rank == 0:
        assert prefill_bundle is not None
        dist_log("prefill: rank0 forward start")
        pipe = prefill_bundle.pipe
        n = int(pipe.num_stages)
        _, seq_len = input_ids.shape
        if seq_len <= n:
            raise ValueError(f"prefill length {seq_len} must be > num_stages {n}")

        layer_types = cache_layer_types_from_config(pipe.config)
        past_kv = make_pipeline_dynamic_cache(pipe.config, n)
        with torch.inference_mode():
            outputs = pipe.base_model(
                input_ids=input_ids.to(device),
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hs = outputs.hidden_states
            tensors_by_idx = _extract_prefill_snapshots(pipe, all_hs)

            hs_normed = pipe.final_norm(outputs.hidden_states[-1][:, -1:, :])
            logits = pipe.lm_head(hs_normed)[0, 0]
            if greedy:
                first_token_id = int(logits.argmax().item())
            else:
                probs = _sampling_probs_hf_style(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                first_token_id = int(torch.multinomial(probs, 1).item())

        dist_log("prefill: rank0 forward done, sending KV shards")
        sharded = split_unified_cache_into_shards(
            past_kv, n, stage_layer_ranges=pipe.stage_layer_ranges
        )
        for stage_idx in range(n):
            timeout.check()
            dest = rank_for_stage(stage_idx, n, merge_last_stage=merge_last_stage)
            if dest == 0:
                assert worker_bundle is not None
                assert int(worker_bundle.stage_idx) == n - 1
                worker_bundle.kv_shard.layers = sharded.shards[stage_idx].layers
                dist_log(f"prefill: rank0 assigned local KV shard for stage {stage_idx}")
                continue
            dist_log(f"prefill: rank0 sending KV shard to rank {dest} (stage {stage_idx})")
            send_cache_shard(
                sharded.shards[stage_idx],
                dest,
                device,
                max_snapshots=n,
                global_layer_offset=pipe.stage_layer_ranges[stage_idx][0],
                layer_types=layer_types,
            )

        sync_device(device)
        dist_log("prefill: rank0 KV send done")
        return PrefillResult(
            first_token_id=int(first_token_id),
            seq_len=int(seq_len),
            tensors_by_idx=tensors_by_idx,
        )

    assert worker_bundle is not None
    stage_idx = int(worker_bundle.stage_idx)
    dist_log(f"prefill: stage {stage_idx} waiting for KV shard from rank0")
    shard = recv_cache_shard(
        0,
        device,
        worker_bundle.compute_dtype,
        max_snapshots=num_stages,
        global_layer_offset=worker_bundle.stage_layer_start,
        layer_types=list(worker_bundle.layer_types),
    )
    worker_bundle.kv_shard.layers = shard.layers
    sync_device(device)
    dist_log(f"prefill: stage {stage_idx} KV recv done")
    return None


def broadcast_input_ids(
    rank: int, device: torch.device, input_ids: torch.LongTensor | None
) -> torch.LongTensor:
    if rank == 0:
        assert input_ids is not None
        flat = input_ids.view(-1).to(device)
        n = flat.numel()
        n_t = torch.tensor([n], dtype=torch.int64, device=device)
        dist_log(f"broadcast_input_ids: rank0 broadcasting n={n}")
        dist_broadcast(n_t, src=0)
        dist_broadcast(flat, src=0)
        return input_ids.to(device)
    dist_log("broadcast_input_ids: worker waiting for n")
    n_t = torch.zeros(1, dtype=torch.int64, device=device)
    dist_broadcast(n_t, src=0)
    n = int(n_t.item())
    dist_log(f"broadcast_input_ids: worker received n={n}, waiting for tokens")
    flat = torch.empty(n, dtype=torch.int64, device=device)
    dist_broadcast(flat, src=0)
    dist_log("broadcast_input_ids: worker received tokens")
    return flat.view(1, -1)
