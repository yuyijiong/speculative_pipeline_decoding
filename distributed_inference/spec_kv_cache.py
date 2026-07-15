"""Preallocated speculation KV cache (avoids DynamicCache cat/crop host sync).

``transformers.DynamicCache`` grows with ``torch.cat`` and rolls back with slice
reassignment (``keys = keys[..., :max_length, :]``). Those reallocations commonly
force a host-side CUDA synchronize, so ``spec_launch`` wall time collapses to the
full GPU ``spec_forward`` time and speculation cannot overlap remote stage compute.

This cache keeps fixed ``[B, H, max_len, D]`` buffers and only bumps an integer
length. ``update`` writes in-place; ``crop`` only resets the length counter.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


class _SpecStaticLayer:
    __slots__ = ("keys", "values", "seq_len", "max_len", "is_initialized", "is_sliding")

    def __init__(self) -> None:
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.seq_len: int = 0
        self.max_len: int = 0
        self.is_initialized: bool = False
        self.is_sliding: bool = False

    def lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor, max_len: int) -> None:
        bsz, n_heads, _, k_dim = key_states.shape
        v_dim = value_states.shape[-1]
        self.max_len = int(max_len)
        self.keys = torch.zeros(
            (bsz, n_heads, self.max_len, k_dim),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.zeros(
            (bsz, n_heads, self.max_len, v_dim),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.seq_len = 0
        self.is_initialized = True

    def ensure_capacity(self, need: int) -> None:
        """Refuse to grow: realloc would host-sync and defeat async speculation.

        Call ``SpecStaticKVCache.preallocate`` with enough ``max_cache_len`` up front.
        """
        if not self.is_initialized or self.keys is None:
            return
        if need <= self.max_len:
            return
        raise RuntimeError(
            f"SpecStaticKVCache capacity exceeded: need={need}, max_len={self.max_len}. "
            "Increase max_new_tokens_hint / preallocate max_cache_len."
        )

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_len = int(key_states.shape[-2])
        if not self.is_initialized:
            # Caller should have pre-sized; fall back to a generous default.
            self.lazy_init(key_states, value_states, max_len=max(kv_len * 4, 256))
        assert self.keys is not None and self.values is not None
        self.ensure_capacity(self.seq_len + kv_len)
        start = self.seq_len
        end = start + kv_len
        # In-place write (no torch.cat / no new storage).
        self.keys[:, :, start:end].copy_(key_states)
        self.values[:, :, start:end].copy_(value_states)
        self.seq_len = end
        # Return only the valid prefix so attention cost tracks real length
        # (matches DynamicCache semantics; unlike HF StaticCache which returns full buf).
        return self.keys[:, :, : self.seq_len], self.values[:, :, : self.seq_len]

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.seq_len - abs(max_length)
        if max_length < 0:
            max_length = 0
        if self.seq_len > max_length:
            self.seq_len = int(max_length)

    def get_seq_length(self) -> int:
        return int(self.seq_len)

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return int(self.seq_len) + int(query_length), 0


class SpecStaticKVCache:
    """Static KV cache compatible with ``PipelineDecoderLayer`` / HF Cache.update API."""

    is_compileable = False

    def __init__(self, num_layers: int, *, max_cache_len: int = 0) -> None:
        n = int(num_layers)
        if n < 1:
            raise ValueError(f"num_layers must be >= 1, got {n}")
        self.layers: List[_SpecStaticLayer] = [_SpecStaticLayer() for _ in range(n)]
        self._max_cache_len = int(max_cache_len)
        # Instance attribute (not property): create_causal_mask does
        # ``False in past_key_values.is_sliding`` / ``.index(False)``.
        self.is_sliding: List[bool] = [False] * n

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def is_initialized(self) -> bool:
        return bool(self.layers) and all(layer.is_initialized for layer in self.layers)

    @property
    def max_cache_len(self) -> int:
        return int(self._max_cache_len)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        # Match DynamicCache semantics: mask covers seen tokens + current query.
        if layer_idx >= len(self.layers):
            return int(query_length), 0
        return self.layers[layer_idx].get_mask_sizes(int(query_length))

    def get_max_cache_shape(self) -> int:
        return int(self._max_cache_len) if self._max_cache_len > 0 else -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        idx = int(layer_idx)
        while len(self.layers) <= idx:
            self.layers.append(_SpecStaticLayer())
            self.is_sliding.append(False)
        layer = self.layers[idx]
        if not layer.is_initialized and self._max_cache_len > 0:
            layer.lazy_init(key_states, value_states, self._max_cache_len)
        return layer.update(key_states, value_states)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(int(max_length))

    def reset(self) -> None:
        for layer in self.layers:
            layer.crop(0)

    def preallocate(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        max_cache_len: int,
    ) -> None:
        """Eagerly allocate all layers (call once after knowing prompt length)."""
        self._max_cache_len = int(max_cache_len)
        fake = torch.zeros(
            (int(batch_size), int(num_heads), 0, int(head_dim)),
            dtype=dtype,
            device=device,
        )
        for layer in self.layers:
            layer.lazy_init(fake, fake, self._max_cache_len)
