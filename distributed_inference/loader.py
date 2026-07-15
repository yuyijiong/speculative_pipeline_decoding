"""Load per-rank model shards for multi-process pipeline decoding (v11)."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache

from .cache import StageCacheView, make_stage_sharded_caches
from .topology import stage_idx_for_rank
from .cache_meta import cache_layer_types_from_config
from .device import cast_module, module_compute_dtype

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modeling_qwen3_pipeline_v11 import (  # noqa: E402
    Qwen3PipelineModelV11,
    _decoder_relevant_config,
    _linear_and_hybrid_attention_layer_indices_for_cache,
    num_hidden_layers_from_hf_config,
    resolve_stage_layer_ranges,
    stage_layers_from_spec_cfg,
)


def hf_index_owner_stage(hf_idx: int, stage_layer_ranges: Sequence[Tuple[int, int]]) -> int:
    if int(hf_idx) == 0:
        return 0
    layer_idx = int(hf_idx) - 1
    for si, (start, end) in enumerate(stage_layer_ranges):
        if start <= layer_idx < end:
            return si
    raise ValueError(
        f"HF hidden-state index {hf_idx} (decoder layer {layer_idx}) is not owned by any stage"
    )


def snap_indices_produced_on_stage(
    snap_want: Set[int],
    stage_idx: int,
    stage_layer_ranges: Sequence[Tuple[int, int]],
) -> Set[int]:
    out: Set[int] = set()
    for idx in snap_want:
        if hf_index_owner_stage(idx, stage_layer_ranges) == int(stage_idx):
            out.add(int(idx))
    return out


@dataclass
class Rank0Bundle:
    embed_tokens: nn.Embedding
    final_norm: nn.Module
    lm_head: nn.Linear
    speculation_module: Any
    config: Any
    hidden_size: int
    vocab_size: int
    num_stages: int
    num_layers: int
    layers_per_stage: int
    stage_layer_ranges: List[Tuple[int, int]]
    aggr_feature_bound: List[int]
    aggr_feature_indices: List[Tuple[int, ...]]
    stage_depth_to_aggr_idx: List[int]
    snap_want: Set[int]
    linear_cache_layer_indices: List[int]
    draft_vocab_meta: dict[str, Any]
    device: torch.device


@dataclass
class StageRankBundle:
    rank: int
    stage_idx: int
    layers: nn.ModuleList
    rotary_emb: nn.Module
    embed_tokens: nn.Embedding | None
    kv_shard: DynamicCache
    stage_cache_view: StageCacheView
    config: Any
    hidden_size: int
    num_stages: int
    layers_per_stage: int
    stage_layer_start: int
    stage_layer_end: int
    stage_layer_ranges: List[Tuple[int, int]]
    num_layers: int
    snap_want: Set[int]
    local_snap_indices: Set[int]
    linear_cache_layer_indices: List[int]
    layer_types: List[str]
    device: torch.device
    compute_dtype: torch.dtype


@dataclass
class PrefillRank0Bundle:
    """Full base model on rank0 used only during prefill, then released."""

    base_model: PreTrainedModel
    pipe: Qwen3PipelineModelV11
    linear_cache_layer_indices: List[int]
    snap_want: Set[int]


def v11_init_from_spec_cfg(spec_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = spec_cfg.get("draft_token_ids")
    draft_token_ids = None if raw is None else [int(x) for x in raw]
    raw_spec = spec_cfg.get("spec_init_from_base_layers")
    spec_init = None if raw_spec is None else [int(x) for x in raw_spec]
    raw_bound = spec_cfg.get("aggr_feature_bound")
    aggr_feature_bound = None if raw_bound is None else [int(x) for x in raw_bound]
    kw: dict[str, Any] = {
        "num_stages": int(spec_cfg["num_stages"]),
        "num_spec_layers": int(spec_cfg.get("num_spec_layers", 1)),
        "draft_token_ids": draft_token_ids,
        "spec_init_from_base_layers": spec_init,
        "aggr_feature_bound": aggr_feature_bound,
    }
    if "trained_with_use_deepest" in spec_cfg:
        kw["trained_with_use_deepest"] = bool(spec_cfg["trained_with_use_deepest"])
    stage_layers = stage_layers_from_spec_cfg(spec_cfg)
    if stage_layers is not None:
        kw["stage_layers"] = stage_layers
    return kw


def format_ckpt_key_info_lines(
    spec_cfg: dict[str, Any],
    *,
    base_model_path: str,
    spec_ckpt_path: str = "",
) -> List[str]:
    """Human-readable checkpoint summary for example/eval logs."""
    raw_bound = spec_cfg.get("aggr_feature_bound")
    if raw_bound is None:
        aggr_str = "default"
    elif isinstance(raw_bound, str):
        aggr_str = raw_bound.strip()
    else:
        aggr_str = ",".join(str(int(x)) for x in raw_bound)

    raw_sl = spec_cfg.get("stage_layers")
    if isinstance(raw_sl, str) and raw_sl.strip():
        stage_layers_str = raw_sl.strip()
    elif isinstance(raw_sl, list) and raw_sl:
        if all(isinstance(row, list) for row in raw_sl):
            if all(len(row) == 1 for row in raw_sl):
                stage_layers_str = ";".join(str(int(row[0])) for row in raw_sl)
            else:
                stage_layers_str = ";".join(str(len(row)) for row in raw_sl)
        else:
            stage_layers_str = str(raw_sl)
    else:
        raw_ranges = spec_cfg.get("stage_layer_ranges")
        if raw_ranges:
            stage_layers_str = ";".join(
                str(int(r[1]) - int(r[0])) for r in raw_ranges
            )
        else:
            stage_layers_str = "uniform"

    lines = [
        f"base_model: {base_model_path}",
        f"num_stages: {int(spec_cfg['num_stages'])}",
        f"num_spec_layers: {int(spec_cfg.get('num_spec_layers', 1))}",
        f"aggr_feature_bound: {aggr_str}",
        f"stage_layers: {stage_layers_str}",
    ]
    if spec_ckpt_path:
        lines.insert(0, f"spec_head_ckpt: {spec_ckpt_path}")
    return lines


def load_spec_checkpoint_config(spec_ckpt_path: str) -> dict[str, Any]:
    try:
        ckpt = torch.load(spec_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(spec_ckpt_path, map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint at {spec_ckpt_path!r} has no config dict.")
    version = int(cfg.get("version", 0) or 0)
    if version not in (0, 11):
        raise ValueError(
            f"Checkpoint version {cfg.get('version')} is not compatible with v11 multi-process decoding."
        )
    return cfg


def load_prefill_rank0_bundle(
    *,
    base_model_path: str,
    spec_ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
) -> PrefillRank0Bundle:
    spec_cfg = load_spec_checkpoint_config(spec_ckpt_path)
    init_kw = v11_init_from_spec_cfg(spec_cfg)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    base.to(dtype=dtype)
    base.eval()
    pipe = Qwen3PipelineModelV11(base_model=base, **init_kw)
    pipe.load_speculation_head(spec_ckpt_path, map_location=str(device))
    cast_module(pipe.speculation_module, device=device, dtype=dtype)
    pipe.eval()
    dec_cfg = _decoder_relevant_config(pipe.config)
    linear_cache_layer_indices = _linear_and_hybrid_attention_layer_indices_for_cache(dec_cfg)
    snap_want = pipe._snap_indices_needed()
    return PrefillRank0Bundle(
        base_model=base,
        pipe=pipe,
        linear_cache_layer_indices=linear_cache_layer_indices,
        snap_want=snap_want,
    )


def load_rank0_decode_bundle(
    prefill_bundle: PrefillRank0Bundle,
    device: torch.device,
) -> Rank0Bundle:
    pipe = prefill_bundle.pipe
    draft_vocab_meta: dict[str, Any] = {
        "use_draft_vocab": bool(pipe._use_draft_vocab),
        "draft_vocab_size": int(pipe.draft_vocab_size),
    }
    if pipe._use_draft_vocab:
        draft_vocab_meta["_draft_token_ids"] = pipe._draft_token_ids
        draft_vocab_meta["_t2d_bool"] = pipe._t2d_bool
        draft_vocab_meta["_token_id_to_draft_idx"] = pipe._token_id_to_draft_idx

    return Rank0Bundle(
        embed_tokens=pipe.embed_tokens,
        final_norm=pipe.final_norm,
        lm_head=pipe.lm_head,
        speculation_module=pipe.speculation_module,
        config=pipe.config,
        hidden_size=int(pipe.hidden_size),
        vocab_size=int(pipe.vocab_size),
        num_stages=int(pipe.num_stages),
        num_layers=int(pipe.num_layers),
        layers_per_stage=int(pipe.layers_per_stage),
        stage_layer_ranges=[tuple(r) for r in pipe.stage_layer_ranges],
        aggr_feature_bound=list(pipe.aggr_feature_bound),
        aggr_feature_indices=[tuple(row) for row in pipe.aggr_feature_indices],
        stage_depth_to_aggr_idx=list(pipe.stage_depth_to_aggr_idx),
        snap_want=set(prefill_bundle.snap_want),
        linear_cache_layer_indices=list(prefill_bundle.linear_cache_layer_indices),
        draft_vocab_meta=draft_vocab_meta,
        device=device,
    )


def load_stage_rank_bundle(
    *,
    rank: int,
    base_model_path: str,
    spec_ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str = "flash_attention_2",
) -> StageRankBundle:
    spec_cfg = load_spec_checkpoint_config(spec_ckpt_path)
    num_stages = int(spec_cfg["num_stages"])
    stage_idx = stage_idx_for_rank(rank, num_stages)
    if stage_idx is None:
        raise ValueError(f"load_stage_rank_bundle expects a stage rank, got rank={rank}")
    if rank < 1:
        raise ValueError(f"load_stage_rank_bundle expects rank>=1, got {rank}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    base.to(dtype=dtype)
    base.eval()
    n_layers = num_hidden_layers_from_hf_config(base.config)
    stage_layers = stage_layers_from_spec_cfg(spec_cfg)
    stage_layer_ranges = resolve_stage_layer_ranges(
        stage_layers,
        num_stages=num_stages,
        num_layers=n_layers,
    )
    lo, hi = stage_layer_ranges[stage_idx]
    local_layers = nn.ModuleList([base.model.layers[i] for i in range(lo, hi)])
    cast_module(local_layers, device=device, dtype=dtype)
    rotary = cast_module(copy.deepcopy(base.model.rotary_emb), device=device, dtype=dtype)

    embed: nn.Embedding | None = None
    if stage_idx == 0:
        embed = cast_module(copy.deepcopy(base.model.embed_tokens), device=device, dtype=dtype)

    dec_cfg = _decoder_relevant_config(base.config)
    linear_cache_layer_indices = _linear_and_hybrid_attention_layer_indices_for_cache(dec_cfg)
    layer_types = cache_layer_types_from_config(base.config)
    sharded = make_stage_sharded_caches(base.config, num_stages, stage_layer_ranges=stage_layer_ranges)
    shard = sharded.shards[stage_idx]
    view = sharded.view(stage_idx)

    init_kw = v11_init_from_spec_cfg(spec_cfg)
    pipe = Qwen3PipelineModelV11(base_model=base, **init_kw)
    snap_want = pipe._snap_indices_needed()
    local_snap = snap_indices_produced_on_stage(snap_want, stage_idx, stage_layer_ranges)
    hidden_size = int(dec_cfg.hidden_size)
    del base, pipe
    lps = hi - lo

    compute_dtype = (
        module_compute_dtype(embed) if embed is not None else module_compute_dtype(local_layers[0])
    )

    return StageRankBundle(
        rank=int(rank),
        stage_idx=stage_idx,
        layers=local_layers,
        rotary_emb=rotary,
        embed_tokens=embed,
        kv_shard=shard,
        stage_cache_view=view,
        config=spec_cfg,
        hidden_size=hidden_size,
        num_stages=num_stages,
        layers_per_stage=lps,
        stage_layer_start=int(lo),
        stage_layer_end=int(hi),
        stage_layer_ranges=[tuple(r) for r in stage_layer_ranges],
        num_layers=n_layers,
        snap_want=snap_want,
        local_snap_indices=local_snap,
        linear_cache_layer_indices=linear_cache_layer_indices,
        layer_types=layer_types,
        device=device,
        compute_dtype=compute_dtype,
    )
