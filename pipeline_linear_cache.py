"""
Rolling-snapshot linear attention cache layers for pipelined speculative decoding.

Qwen3.5 ``LinearAttentionLayer.crop`` is a no-op in HuggingFace. These subclasses keep a
FIFO of post-update snapshots (up to ``max_snapshots`` entries, typically ``num_stages``) so
each layer can be rewound by a stage-dependent number of decode steps on draft rejection,
without manual deep-copy history in the pipeline driver.

Snapshot storage is a **preallocated ring buffer**: each push does in-place ``copy_`` into a
fixed slot and advances a head pointer (no per-step ``clone()``). Metadata flags live on
the CPU so the decode hot path only issues the necessary GPU memcpys.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.cache_utils import (
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)


def _tensor_is_inference(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    is_inf = getattr(t, "is_inference", None)
    if callable(is_inf):
        return bool(is_inf())
    return bool(getattr(t, "_is_inference", False))


class _PipelineLinearSnapshotRingMixin:
    """
    Preallocated ring of post-update linear (conv / recurrent) states.

    Layout:
      - ``_conv_ring`` / ``_recurrent_ring``: ``[max_snapshots, *state_shape]`` (GPU)
      - ``_head`` / ``_count``: CPU ints
      - flag lists: CPU bools (never touch the GPU on the push hot path)

    After the first ``seed_snapshot_from_current_state`` / ``_ensure_ring_storage``,
    ``_push_post_update_snapshot`` only does ``copy_`` + pointer math.
    """

    max_snapshots: int
    _head: int
    _count: int
    _ring_ready: bool
    _conv_ring: Optional[torch.Tensor]
    _recurrent_ring: Optional[torch.Tensor]
    _flags_has_previous: List[bool]
    _flags_conv_init: List[bool]
    _flags_recurrent_init: List[bool]

    def _init_snapshot_ring(self, max_snapshots: int) -> None:
        if max_snapshots < 1:
            raise ValueError(f"max_snapshots must be >= 1, got {max_snapshots}.")
        self.max_snapshots = int(max_snapshots)
        self._conv_ring = None
        self._recurrent_ring = None
        self._flags_has_previous = [False] * self.max_snapshots
        self._flags_conv_init = [False] * self.max_snapshots
        self._flags_recurrent_init = [False] * self.max_snapshots
        self._reset_ring_metadata()

    def _reset_ring_metadata(self) -> None:
        self._head = 0
        self._count = 0
        self._ring_ready = False

    def _ensure_ring_storage(self) -> None:
        """Allocate fixed ring tensors once live state shapes/devices are known.

        Allocation always exits ``InferenceMode`` so ring buffers are normal tensors.
        Prefill often runs under ``torch.inference_mode()``; tensors created there become
        inference tensors and cannot be inplace-updated after leaving that context.
        """
        need_conv = bool(getattr(self, "is_conv_states_initialized", False))
        need_rec = bool(getattr(self, "is_recurrent_states_initialized", False))
        if not need_conv and not need_rec:
            return

        n = int(self.max_snapshots)
        with torch.inference_mode(False):
            if need_conv:
                live = self.conv_states
                if (
                    self._conv_ring is None
                    or self._conv_ring.shape[1:] != live.shape
                    or self._conv_ring.device != live.device
                    or self._conv_ring.dtype != live.dtype
                    or _tensor_is_inference(self._conv_ring)
                ):
                    self._conv_ring = torch.empty(
                        (n, *live.shape), device=live.device, dtype=live.dtype
                    )
                    self._ring_ready = False
            if need_rec:
                live = self.recurrent_states
                if (
                    self._recurrent_ring is None
                    or self._recurrent_ring.shape[1:] != live.shape
                    or self._recurrent_ring.device != live.device
                    or self._recurrent_ring.dtype != live.dtype
                    or _tensor_is_inference(self._recurrent_ring)
                ):
                    self._recurrent_ring = torch.empty(
                        (n, *live.shape), device=live.device, dtype=live.dtype
                    )
                    self._ring_ready = False

        if len(self._flags_has_previous) != n:
            self._flags_has_previous = [False] * n
            self._flags_conv_init = [False] * n
            self._flags_recurrent_init = [False] * n
            self._ring_ready = False

        self._ring_ready = True

    def _ring_slot_index(self, steps_back: int) -> int:
        """Index of the snapshot ``steps_back`` updates before the latest (0 = latest)."""
        if steps_back < 0:
            raise ValueError(f"steps_back must be >= 0, got {steps_back}.")
        if self._count <= steps_back:
            raise RuntimeError(
                f"Cannot access snapshot {steps_back} step(s) back: need "
                f"{steps_back + 1} snapshot(s), got {self._count} "
                f"(max_snapshots={self.max_snapshots})."
            )
        return (self._head - steps_back) % int(self.max_snapshots)

    def _write_live_into_slot(self, slot: int) -> None:
        """In-place copy of live conv/recurrent state into ``ring[slot]`` (GPU memcpy only)."""
        self._flags_has_previous[slot] = bool(self.has_previous_state)
        if self.is_conv_states_initialized:
            assert self._conv_ring is not None
            self._conv_ring[slot].copy_(self.conv_states)
            self._flags_conv_init[slot] = True
        else:
            self._flags_conv_init[slot] = False
        if self.is_recurrent_states_initialized:
            assert self._recurrent_ring is not None
            self._recurrent_ring[slot].copy_(self.recurrent_states)
            self._flags_recurrent_init[slot] = True
        else:
            self._flags_recurrent_init[slot] = False

    def _restore_live_from_slot(self, slot: int) -> None:
        # Live cache tensors may have been created under ``inference_mode`` (prefill).
        with torch.inference_mode():
            self.has_previous_state = bool(self._flags_has_previous[slot])
            if self._flags_conv_init[slot]:
                assert self._conv_ring is not None
                if not self.is_conv_states_initialized:
                    self.lazy_initialization(conv_states=self._conv_ring[slot])
                self.conv_states.copy_(self._conv_ring[slot])
                self.is_conv_states_initialized = True
            if self._flags_recurrent_init[slot]:
                assert self._recurrent_ring is not None
                if not self.is_recurrent_states_initialized:
                    self.lazy_initialization(recurrent_states=self._recurrent_ring[slot])
                self.recurrent_states.copy_(self._recurrent_ring[slot])
                self.is_recurrent_states_initialized = True

    def _push_post_update_snapshot(self) -> None:
        # Hot path: assume ring was seeded after prefill. Only GPU work is copy_.
        if not self._ring_ready:
            if not (self.is_conv_states_initialized or self.is_recurrent_states_initialized):
                return
            if not self.has_previous_state:
                return
            self._ensure_ring_storage()
        if not self.has_previous_state:
            return

        if self._count == 0:
            slot = 0
        else:
            slot = (self._head + 1) % int(self.max_snapshots)
        self._write_live_into_slot(slot)
        self._head = slot
        self._count = min(self._count + 1, int(self.max_snapshots))

    def seed_snapshot_from_current_state(self) -> None:
        """Record the live conv/recurrent tensors as the sole snapshot (post-prefill / post-rewind)."""
        if not (self.is_conv_states_initialized or self.is_recurrent_states_initialized):
            self._reset_ring_metadata()
            return
        if not self.has_previous_state:
            self._reset_ring_metadata()
            return
        self._ensure_ring_storage()
        self._write_live_into_slot(0)
        self._head = 0
        self._count = 1

    def crop_rewind(self, rewind_steps: int) -> None:
        """Restore this layer's state to ``rewind_steps`` layer updates ago (0 = keep current)."""
        rewind_steps = int(rewind_steps)
        if rewind_steps < 0:
            raise ValueError(f"rewind_steps must be >= 0, got {rewind_steps}.")
        if rewind_steps == 0:
            self.seed_snapshot_from_current_state()
            return
        slot = self._ring_slot_index(rewind_steps)
        self._restore_live_from_slot(slot)
        # Match prior semantics: after rewind, history collapses to the restored state only.
        self.seed_snapshot_from_current_state()

    def clear_snapshot_history(self) -> None:
        self._reset_ring_metadata()

    def capture_snapshot_cuda_graph(self, *, stream: torch.cuda.Stream | None = None) -> bool:
        """
        No-op retained for API compatibility.

        Capturing a CUDA graph around a 1–2 tiny ``copy_`` per linear layer is a net loss
        (graph replay / stream sync overhead dominates the memcpy). Prefer the eager
        ring ``copy_`` path.
        """
        del stream
        return False

    def _copy_ring_into(self, other: "_PipelineLinearSnapshotRingMixin") -> None:
        other.max_snapshots = int(self.max_snapshots)
        other._head = int(self._head)
        other._count = int(self._count)
        other._ring_ready = bool(self._ring_ready)
        other._conv_ring = None if self._conv_ring is None else self._conv_ring.clone()
        other._recurrent_ring = (
            None if self._recurrent_ring is None else self._recurrent_ring.clone()
        )
        other._flags_has_previous = list(self._flags_has_previous)
        other._flags_conv_init = list(self._flags_conv_init)
        other._flags_recurrent_init = list(self._flags_recurrent_init)


def capture_pipeline_linear_snapshot_cuda_graphs(
    past_kv: Cache,
    linear_layer_indices: Sequence[int],
) -> int:
    """
    Deprecated no-op: per-layer snapshot CUDA graphs hurt decode latency for tiny memcpys.

    Kept so callers (profile script / older code) do not break. Always returns 0.
    """
    del past_kv, linear_layer_indices
    return 0


class PipelineLinearAttentionLayer(_PipelineLinearSnapshotRingMixin, LinearAttentionLayer):
    """
    Linear attention cache with a rolling post-update snapshot ring buffer.

    Snapshots are taken **after** each layer update so conv (including in-place
    ``causal_conv1d_update`` during decode) and recurrent states stay consistent.
    The ring head always points at the latest post-update state.

    Args:
        max_snapshots: Rolling window size; use ``num_stages`` for pipeline decoding.
    """

    def __init__(self, max_snapshots: int = 1):
        LinearAttentionLayer.__init__(self)
        self._init_snapshot_ring(max_snapshots)

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = super().update_conv_state(conv_states, **kwargs)
        return out

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = super().update_recurrent_state(recurrent_states, **kwargs)
        self._push_post_update_snapshot()
        return out

    def crop(self, max_length: int) -> None:
        """HF ``DynamicCache.crop`` entry point; linear layers ignore sequence length."""
        del max_length

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
        self._copy_ring_into(copied)
        return copied


class PipelineLinearAttentionAndFullAttentionLayer(
    _PipelineLinearSnapshotRingMixin, LinearAttentionAndFullAttentionLayer
):
    """Hybrid layer: KV uses ``DynamicLayer.crop``; linear state uses ``crop_rewind``."""

    def __init__(self, max_snapshots: int = 1):
        LinearAttentionAndFullAttentionLayer.__init__(self)
        self._init_snapshot_ring(max_snapshots)

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return LinearAttentionLayer.update_conv_state(self, conv_states, **kwargs)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = LinearAttentionLayer.update_recurrent_state(self, recurrent_states, **kwargs)
        self._push_post_update_snapshot()
        return out

    def crop(self, max_length: int) -> None:
        DynamicLayer.crop(self, max_length)

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
        self._copy_ring_into(copied)
        return copied


def _adopt_pipeline_linear_layer(
    layer: LinearAttentionCacheLayerMixin,
    max_snapshots: int,
) -> LinearAttentionCacheLayerMixin:
    if isinstance(layer, PipelineLinearAttentionAndFullAttentionLayer):
        if int(layer.max_snapshots) != int(max_snapshots):
            layer._init_snapshot_ring(max_snapshots)
            if layer.is_conv_states_initialized or layer.is_recurrent_states_initialized:
                if layer.has_previous_state:
                    layer.seed_snapshot_from_current_state()
        return layer
    if isinstance(layer, PipelineLinearAttentionLayer):
        if int(layer.max_snapshots) != int(max_snapshots):
            layer._init_snapshot_ring(max_snapshots)
            if layer.is_conv_states_initialized or layer.is_recurrent_states_initialized:
                if layer.has_previous_state:
                    layer.seed_snapshot_from_current_state()
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


def seed_pipeline_linear_snapshots_after_prefill(
    past_kv: Cache,
    linear_layer_indices: Sequence[int],
) -> None:
    """
    After prefill + ``install_pipeline_linear_cache_layers``, adopted linear layers have
    correct live conv/recurrent tensors but an empty ring. Seed one snapshot per layer so
    the first ``crop_rewind(k)`` rewinds relative to the end of prefill.
    """
    if not linear_layer_indices:
        return
    if not hasattr(past_kv, "layers"):
        return
    for i in linear_layer_indices:
        li = int(i)
        if li < 0 or li >= len(past_kv.layers):
            continue
        layer = past_kv.layers[li]
        if isinstance(layer, (PipelineLinearAttentionLayer, PipelineLinearAttentionAndFullAttentionLayer)):
            layer.seed_snapshot_from_current_state()


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
