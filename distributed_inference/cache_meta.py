"""Cache layer typing for Qwen3.5-style DynamicCache (by global layer index)."""

from __future__ import annotations

from typing import Any, List


def cache_layer_types_from_config(config: Any) -> List[str]:
    dec = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    lt = getattr(dec, "layer_types", None)
    if lt is None:
        return []
    out = list(lt)
    n_skip = int(getattr(dec, "num_kv_shared_layers", 0) or 0)
    if n_skip > 0:
        out = out[:-n_skip]
    return out


def cache_kind_for_global_layer(global_idx: int, layer_types: List[str]) -> int:
    """0 = full-attention KV, 1 = linear attention, 2 = hybrid."""
    if global_idx < 0 or global_idx >= len(layer_types):
        return 0
    t = layer_types[global_idx]
    if t == "hybrid":
        return 2
    if t == "linear_attention":
        return 1
    return 0
