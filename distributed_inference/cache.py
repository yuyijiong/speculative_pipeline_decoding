"""Per-stage KV cache shards and views for multi-GPU pipeline decoding."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

_SPEC_ROOT = Path(__file__).resolve().parent.parent
if str(_SPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPEC_ROOT))

from pipeline_linear_cache import (  # noqa: E402
    PipelineLinearAttentionAndFullAttentionLayer,
    PipelineLinearAttentionLayer,
    make_pipeline_dynamic_cache,
)

__all__ = [
    "PipelineLinearAttentionAndFullAttentionLayer",
    "PipelineLinearAttentionLayer",
    "StageShardedCaches",
    "StageCacheView",
    "make_stage_sharded_caches",
    "split_unified_cache_into_shards",
]


@dataclass
class StageShardedCaches:
    shards: Tuple[DynamicCache, ...]
    stage_layer_ranges: Tuple[Tuple[int, int], ...]
    num_stages: int

    @property
    def layers_per_stage(self) -> int:
        sizes = {end - start for start, end in self.stage_layer_ranges}
        if len(sizes) != 1:
            raise ValueError(
                f"layers_per_stage is undefined for non-uniform stage_layer_ranges={self.stage_layer_ranges}"
            )
        return next(iter(sizes))

    @property
    def num_layers(self) -> int:
        return int(self.stage_layer_ranges[-1][1])

    def view(self, stage_idx: int) -> "StageCacheView":
        offset = int(self.stage_layer_ranges[int(stage_idx)][0])
        return StageCacheView(
            self.shards[int(stage_idx)],
            offset,
            self.num_layers,
        )


def _default_stage_layer_ranges(num_layers: int, num_stages: int) -> Tuple[Tuple[int, int], ...]:
    if num_layers % int(num_stages) != 0:
        raise ValueError(f"num_layers ({num_layers}) must be divisible by num_stages ({num_stages})")
    lps = num_layers // int(num_stages)
    return tuple((s * lps, (s + 1) * lps) for s in range(int(num_stages)))


def make_stage_sharded_caches(
    config: Any,
    num_stages: int,
    *,
    stage_layer_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> StageShardedCaches:
    unified = make_pipeline_dynamic_cache(config, num_stages)
    n_layers = len(unified.layers)
    ranges = (
        tuple((int(start), int(end)) for start, end in stage_layer_ranges)
        if stage_layer_ranges is not None
        else _default_stage_layer_ranges(n_layers, num_stages)
    )
    if ranges[-1][1] != n_layers:
        raise ValueError(
            f"stage_layer_ranges must end at num_layers={n_layers}, got {list(ranges)}"
        )
    all_layers = unified.layers
    unified.layers = []
    shards: List[DynamicCache] = []
    for lo, hi in ranges:
        shard = DynamicCache()
        shard.layers = list(all_layers[lo:hi])
        shards.append(shard)
    return StageShardedCaches(tuple(shards), ranges, int(num_stages))


def split_unified_cache_into_shards(
    unified: DynamicCache,
    num_stages: int,
    *,
    stage_layer_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> StageShardedCaches:
    n_layers = len(unified.layers)
    ranges = (
        tuple((int(start), int(end)) for start, end in stage_layer_ranges)
        if stage_layer_ranges is not None
        else _default_stage_layer_ranges(n_layers, num_stages)
    )
    shards: List[DynamicCache] = []
    for lo, hi in ranges:
        shard = DynamicCache()
        shard.layers = list(unified.layers[lo:hi])
        shards.append(shard)
    unified.layers = []
    return StageShardedCaches(tuple(shards), ranges, int(num_stages))


class _GlobalIndexedLayers(Sequence[Any]):
    def __init__(
        self,
        shard: DynamicCache,
        offset: int,
        num_layers: int,
    ) -> None:
        self._shard = shard
        self._offset = int(offset)
        self._num_layers = int(num_layers)

    def __getitem__(self, layer_idx: Union[int, slice]) -> Any:
        if isinstance(layer_idx, slice):
            start, stop, step = layer_idx.indices(self._num_layers)
            return [self[i] for i in range(start, stop, step or 1)]
        li = int(layer_idx) - self._offset
        shard_layers = self._shard.layers
        if li < 0 or li >= len(shard_layers):
            raise IndexError(
                f"global layer_idx {layer_idx} not in stage cache "
                f"[{self._offset}, {self._offset + len(shard_layers)})"
            )
        return shard_layers[li]

    def __len__(self) -> int:
        return self._num_layers


class StageCacheView(Cache):
    def __init__(
        self,
        shard: DynamicCache,
        global_layer_offset: int,
        num_layers: int,
    ) -> None:
        self._shard = shard
        self._offset = int(global_layer_offset)
        self._num_layers = int(num_layers)
        super().__init__(
            layers=_GlobalIndexedLayers(shard, self._offset, self._num_layers),
        )

    def _local(self, layer_idx: int) -> int:
        li = int(layer_idx) - self._offset
        if li < 0 or li >= len(self._shard.layers):
            raise IndexError(
                f"global layer_idx {layer_idx} not in stage cache "
                f"[{self._offset}, {self._offset + len(self._shard.layers)})"
            )
        return li

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._shard.update(key_states, value_states, self._local(layer_idx), cache_kwargs)

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._shard.update_conv_state(conv_states, self._local(layer_idx), **kwargs)

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._shard.update_recurrent_state(recurrent_states, self._local(layer_idx), **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._shard.get_seq_length(self._local(layer_idx))

    def crop(self, max_length: int) -> None:
        self._shard.crop(max_length)

    def get_max_cache_shape(self) -> Optional[int]:
        return self._shard.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        return self._shard.get_mask_sizes(cache_position, self._local(layer_idx))

    def has_previous_state(self, layer_idx: int) -> bool:
        return self._shard.has_previous_state(self._local(layer_idx))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._shard, name)
