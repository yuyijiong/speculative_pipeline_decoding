"""
Rolling-snapshot linear attention cache layers for pipelined speculative decoding.

Qwen3.5 ``LinearAttentionLayer.crop`` is a no-op in HuggingFace. These subclasses keep a
FIFO of post-update snapshots (up to ``max_snapshots`` entries, typically ``num_stages``) so
each layer can be rewound by a stage-dependent number of decode steps on draft rejection,
without manual deep-copy history in the pipeline driver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.cache_utils import (
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)


@dataclass
class _LinearAttentionSnapshot:
    has_previous_state: bool
    conv_states: Optional[torch.Tensor] = None
    is_conv_states_initialized: bool = False
    recurrent_states: Optional[torch.Tensor] = None
    is_recurrent_states_initialized: bool = False


def _capture_linear_snapshot(layer: LinearAttentionCacheLayerMixin) -> _LinearAttentionSnapshot:
    snap = _LinearAttentionSnapshot(has_previous_state=bool(layer.has_previous_state))
    if layer.is_conv_states_initialized:
        snap.is_conv_states_initialized = True
        snap.conv_states = layer.conv_states.clone()
    if layer.is_recurrent_states_initialized:
        snap.is_recurrent_states_initialized = True
        snap.recurrent_states = layer.recurrent_states.clone()
    return snap


def _restore_linear_snapshot(layer: LinearAttentionCacheLayerMixin, snap: _LinearAttentionSnapshot) -> None:
    layer.has_previous_state = snap.has_previous_state
    if snap.is_conv_states_initialized:
        if not layer.is_conv_states_initialized:
            layer.lazy_initialization(conv_states=snap.conv_states)
        layer.conv_states.copy_(snap.conv_states)
        layer.is_conv_states_initialized = True
    if snap.is_recurrent_states_initialized:
        if not layer.is_recurrent_states_initialized:
            layer.lazy_initialization(recurrent_states=snap.recurrent_states)
        layer.recurrent_states.copy_(snap.recurrent_states)
        layer.is_recurrent_states_initialized = True


class PipelineLinearAttentionLayer(LinearAttentionLayer):
    """
    Linear attention cache with a rolling post-update snapshot buffer.

    Snapshots are taken **after** each layer update so conv (including in-place
    ``causal_conv1d_update`` during decode) and recurrent states stay consistent.
    The latest entry in ``_snapshot_buffer`` always matches the live cache tensors.

    Args:
        max_snapshots: Rolling window size; use ``num_stages`` for pipeline decoding.
    """

    def __init__(self, max_snapshots: int = 1):
        super().__init__()
        if max_snapshots < 1:
            raise ValueError(f"max_snapshots must be >= 1, got {max_snapshots}.")
        self.max_snapshots = int(max_snapshots)
        self._snapshot_buffer: List[_LinearAttentionSnapshot] = []

    def _push_post_update_snapshot(self) -> None:
        if not (self.is_conv_states_initialized or self.is_recurrent_states_initialized):
            return
        if not self.has_previous_state:
            return
        self._snapshot_buffer.append(_capture_linear_snapshot(self))
        overflow = len(self._snapshot_buffer) - self.max_snapshots
        if overflow > 0:
            del self._snapshot_buffer[:overflow]

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = super().update_conv_state(conv_states, **kwargs)
        return out

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = super().update_recurrent_state(recurrent_states, **kwargs)
        self._push_post_update_snapshot()
        return out

    def crop_rewind(self, rewind_steps: int) -> None:
        """Restore this layer's state to ``rewind_steps`` layer updates ago (0 = keep current)."""
        rewind_steps = int(rewind_steps)
        if rewind_steps < 0:
            raise ValueError(f"rewind_steps must be >= 0, got {rewind_steps}.")
        if rewind_steps == 0:
            return
        # Buffer[-1] is the current post-update state; go back k steps via buffer[-(k+1)].
        if len(self._snapshot_buffer) < rewind_steps + 1:
            raise RuntimeError(
                f"Cannot rewind linear attention cache by {rewind_steps} step(s): need "
                f"{rewind_steps + 1} snapshot(s) in buffer, got {len(self._snapshot_buffer)} "
                f"(max_snapshots={self.max_snapshots})."
            )
        snap = self._snapshot_buffer[-(rewind_steps + 1)]
        _restore_linear_snapshot(self, snap)
        del self._snapshot_buffer[-(rewind_steps + 1) :]

    def crop(self, max_length: int) -> None:
        """HF ``DynamicCache.crop`` entry point; linear layers ignore sequence length."""
        del max_length

    def clear_snapshot_history(self) -> None:
        self._snapshot_buffer.clear()

    def __deepcopy__(self, memo: dict) -> "PipelineLinearAttentionLayer":
        cls = self.__class__
        copied = cls(max_snapshots=self.max_snapshots)
        if self.is_conv_states_initialized:
            copied.lazy_initialization(conv_states=self.conv_states)
            copied.conv_states.copy_(self.conv_states)
            copied.has_previous_state = self.has_previous_state
        if self.is_recurrent_states_initialized:
            copied.lazy_initialization(recurrent_states=self.recurrent_states)
            copied.recurrent_states.copy_(self.recurrent_states)
        copied._snapshot_buffer = copy.deepcopy(self._snapshot_buffer, memo)
        return copied


class PipelineLinearAttentionAndFullAttentionLayer(LinearAttentionAndFullAttentionLayer):
    """Hybrid layer: KV uses ``DynamicLayer.crop``; linear state uses ``crop_rewind``."""

    def __init__(self, max_snapshots: int = 1):
        LinearAttentionAndFullAttentionLayer.__init__(self)
        if max_snapshots < 1:
            raise ValueError(f"max_snapshots must be >= 1, got {max_snapshots}.")
        self.max_snapshots = int(max_snapshots)
        self._snapshot_buffer: List[_LinearAttentionSnapshot] = []

    def _push_post_update_snapshot(self) -> None:
        if not (self.is_conv_states_initialized or self.is_recurrent_states_initialized):
            return
        if not self.has_previous_state:
            return
        self._snapshot_buffer.append(_capture_linear_snapshot(self))
        overflow = len(self._snapshot_buffer) - self.max_snapshots
        if overflow > 0:
            del self._snapshot_buffer[:overflow]

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return LinearAttentionLayer.update_conv_state(self, conv_states, **kwargs)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = LinearAttentionLayer.update_recurrent_state(self, recurrent_states, **kwargs)
        self._push_post_update_snapshot()
        return out

    def crop_rewind(self, rewind_steps: int) -> None:
        rewind_steps = int(rewind_steps)
        if rewind_steps < 0:
            raise ValueError(f"rewind_steps must be >= 0, got {rewind_steps}.")
        if rewind_steps == 0:
            return
        if len(self._snapshot_buffer) < rewind_steps + 1:
            raise RuntimeError(
                f"Cannot rewind hybrid linear attention cache by {rewind_steps} step(s): need "
                f"{rewind_steps + 1} snapshot(s) in buffer, got {len(self._snapshot_buffer)} "
                f"(max_snapshots={self.max_snapshots})."
            )
        snap = self._snapshot_buffer[-(rewind_steps + 1)]
        _restore_linear_snapshot(self, snap)
        del self._snapshot_buffer[-(rewind_steps + 1) :]

    def crop(self, max_length: int) -> None:
        DynamicLayer.crop(self, max_length)

    def clear_snapshot_history(self) -> None:
        self._snapshot_buffer.clear()

    def __deepcopy__(self, memo: dict) -> "PipelineLinearAttentionAndFullAttentionLayer":
        cls = self.__class__
        copied = cls(max_snapshots=self.max_snapshots)
        if self.is_conv_states_initialized:
            copied.lazy_initialization(conv_states=self.conv_states)
            copied.conv_states.copy_(self.conv_states)
            copied.has_previous_state = self.has_previous_state
        if self.is_recurrent_states_initialized:
            copied.lazy_initialization(recurrent_states=self.recurrent_states)
            copied.recurrent_states.copy_(self.recurrent_states)
        if self.is_initialized:
            copied.lazy_initialization(self.keys, self.values)
            copied.keys.copy_(self.keys)
            copied.values.copy_(self.values)
        copied._snapshot_buffer = copy.deepcopy(self._snapshot_buffer, memo)
        return copied


def _adopt_pipeline_linear_layer(
    layer: LinearAttentionCacheLayerMixin,
    max_snapshots: int,
) -> LinearAttentionCacheLayerMixin:
    if isinstance(layer, PipelineLinearAttentionAndFullAttentionLayer):
        layer.max_snapshots = max_snapshots
        return layer
    if isinstance(layer, PipelineLinearAttentionLayer):
        layer.max_snapshots = max_snapshots
        return layer
    if isinstance(layer, LinearAttentionAndFullAttentionLayer):
        adopted = PipelineLinearAttentionAndFullAttentionLayer(max_snapshots=max_snapshots)
    elif isinstance(layer, LinearAttentionLayer):
        adopted = PipelineLinearAttentionLayer(max_snapshots=max_snapshots)
    else:
        raise TypeError(f"Unsupported linear cache layer type: {type(layer)!r}")

    if layer.is_conv_states_initialized:
        adopted.lazy_initialization(conv_states=layer.conv_states)
        adopted.conv_states.copy_(layer.conv_states)
        adopted.has_previous_state = layer.has_previous_state
    if layer.is_recurrent_states_initialized:
        adopted.lazy_initialization(recurrent_states=layer.recurrent_states)
        adopted.recurrent_states.copy_(layer.recurrent_states)
    if isinstance(layer, LinearAttentionAndFullAttentionLayer) and layer.is_initialized:
        adopted.lazy_initialization(layer.keys, layer.values)
        adopted.keys.copy_(layer.keys)
        adopted.values.copy_(layer.values)
    return adopted


def install_pipeline_linear_cache_layers(
    past_kv: Cache,
    linear_layer_indices: Sequence[int],
    num_stages: int,
) -> None:
    """Replace linear/hybrid cache slots in ``past_kv`` with pipeline snapshot-aware layers."""
    if not linear_layer_indices:
        return
    if not hasattr(past_kv, "layers"):
        raise TypeError(f"Cannot install pipeline linear cache on type {type(past_kv)}.")
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}.")
    for i in linear_layer_indices:
        li = int(i)
        if li < 0 or li >= len(past_kv.layers):
            raise RuntimeError(f"layer index {li} out of range for past_kv ({len(past_kv.layers)} layers).")
        past_kv.layers[li] = _adopt_pipeline_linear_layer(past_kv.layers[li], num_stages)


def crop_pipeline_cache_after_rejection(
    past_kv: Cache,
    target_length: int,
    *,
    linear_layer_indices: Sequence[int],
    layers_per_stage: int,
    num_stages: int,
) -> None:
    """
    Roll back ``past_kv`` after draft rejection: standard KV ``crop`` plus stage-aware
    linear/hybrid ``crop_rewind``.
    """
    if layers_per_stage <= 0:
        raise ValueError(f"layers_per_stage must be > 0, got {layers_per_stage}.")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be > 0, got {num_stages}.")
    linear_set: Set[int] = {int(i) for i in linear_layer_indices}
    if not hasattr(past_kv, "layers"):
        raise TypeError(f"Cannot crop pipeline cache for type {type(past_kv)}.")
    for layer_idx, layer in enumerate(past_kv.layers):
        if layer_idx in linear_set and isinstance(
            layer, (PipelineLinearAttentionLayer, PipelineLinearAttentionAndFullAttentionLayer)
        ):
            stage_idx = layer_idx // int(layers_per_stage)
            if stage_idx < 0 or stage_idx >= int(num_stages):
                raise RuntimeError(
                    f"layer index {layer_idx} maps to invalid stage {stage_idx} "
                    f"(layers_per_stage={layers_per_stage}, num_stages={num_stages})."
                )
            rewind_steps = int(num_stages) - 1 - stage_idx
            layer.crop_rewind(rewind_steps)
            if isinstance(layer, PipelineLinearAttentionAndFullAttentionLayer):
                DynamicLayer.crop(layer, target_length)
        elif layer_idx not in linear_set:
            layer.crop(target_length)
    for layer_idx in linear_set:
        layer = past_kv.layers[layer_idx]
        if isinstance(
            layer, (PipelineLinearAttentionLayer, PipelineLinearAttentionAndFullAttentionLayer)
        ):
            layer.clear_snapshot_history()


def make_pipeline_dynamic_cache(config: Any, num_stages: int) -> DynamicCache:
    """Build a ``DynamicCache`` whose linear/hybrid slots use pipeline snapshot layers."""
    dec = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    past = DynamicCache(config=dec)
    layer_types = getattr(dec, "layer_types", None)
    if layer_types is None:
        return past
    layer_types = list(layer_types)
    n_skip = int(getattr(dec, "num_kv_shared_layers", 0) or 0)
    if n_skip > 0:
        layer_types = layer_types[:-n_skip]
    linear_indices = [i for i, t in enumerate(layer_types) if t in ("linear_attention", "hybrid")]
    install_pipeline_linear_cache_layers(past, linear_indices, num_stages)
    return past
