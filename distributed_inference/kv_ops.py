"""Per-rank KV cache crop after speculative rejection."""

from __future__ import annotations

from typing import Sequence, Set

from transformers.cache_utils import Cache, DynamicLayer

from .cache import (
    PipelineLinearAttentionAndFullAttentionLayer,
    PipelineLinearAttentionLayer,
)


def crop_stage_shard_after_rejection(
    shard: Cache,
    *,
    stage_idx: int,
    num_stages: int,
    stage_layer_start: int,
    num_layers: int,
    crop_length: int,
    linear_layer_indices: Sequence[int],
) -> None:
    n = int(num_stages)
    linear_set: Set[int] = {int(i) for i in linear_layer_indices}
    global_offset = int(stage_layer_start)
    for local_idx in range(len(shard.layers)):
        global_idx = global_offset + local_idx
        layer = shard.layers[local_idx]
        if global_idx in linear_set and isinstance(
            layer, (PipelineLinearAttentionLayer, PipelineLinearAttentionAndFullAttentionLayer)
        ):
            rewind_steps = n - 1 - int(stage_idx)
            layer.crop_rewind(rewind_steps)
            if isinstance(layer, PipelineLinearAttentionAndFullAttentionLayer):
                DynamicLayer.crop(layer, crop_length)
        elif global_idx not in linear_set:
            layer.crop(crop_length)
