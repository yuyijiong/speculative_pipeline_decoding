"""Transfer DynamicCache shard tensors from rank0 to stage ranks after prefill."""

from __future__ import annotations

from typing import Any, List

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .dist_io import dist_recv, dist_send
from .cache import (
    PipelineLinearAttentionAndFullAttentionLayer,
    PipelineLinearAttentionLayer,
)
from .cache_meta import cache_kind_for_global_layer


def _send_tensor(t: torch.Tensor | None, dst: int, device: torch.device) -> None:
    if t is None:
        flag = torch.zeros(1, dtype=torch.int64, device=device)
        dist_send(flag, dst=dst)
        return
    flag = torch.ones(1, dtype=torch.int64, device=device)
    dist_send(flag, dst=dst)
    shape_list = list(t.shape)
    ndim = torch.tensor([len(shape_list)], dtype=torch.int64, device=device)
    dist_send(ndim, dst=dst)
    shape = torch.tensor(shape_list, dtype=torch.int64, device=device)
    dist_send(shape, dst=dst)
    dist_send(t.contiguous(), dst=dst)


def _recv_tensor(device: torch.device, src: int, dtype: torch.dtype) -> torch.Tensor | None:
    flag = torch.empty(1, dtype=torch.int64, device=device)
    dist_recv(flag, src=src)
    if int(flag.item()) == 0:
        return None
    ndim_t = torch.empty(1, dtype=torch.int64, device=device)
    dist_recv(ndim_t, src=src)
    ndim = int(ndim_t.item())
    shape = torch.empty(ndim, dtype=torch.int64, device=device)
    dist_recv(shape, src=src)
    dims = [int(x) for x in shape.tolist()]
    buf = torch.empty(dims, dtype=dtype, device=device)
    dist_recv(buf, src=src)
    return buf


def _send_dynamic_body(layer: Any, dst: int, device: torch.device) -> None:
    is_init = bool(getattr(layer, "is_initialized", False))
    init = torch.tensor([1 if is_init else 0], dtype=torch.int64, device=device)
    dist_send(init, dst=dst)
    if is_init:
        _send_tensor(layer.keys, dst, device)
        _send_tensor(layer.values, dst, device)


def _recv_dynamic_body(device: torch.device, src: int, dtype: torch.dtype) -> DynamicLayer:
    init = torch.empty(1, dtype=torch.int64, device=device)
    dist_recv(init, src=src)
    layer = DynamicLayer()
    if int(init.item()) == 0:
        return layer
    keys = _recv_tensor(device, src, dtype)
    values = _recv_tensor(device, src, dtype)
    layer.dtype = keys.dtype
    layer.device = keys.device
    layer.keys = keys
    layer.values = values
    layer.is_initialized = True
    return layer


def _send_linear_body(layer: Any, dst: int, device: torch.device) -> None:
    flags = torch.tensor(
        [
            int(getattr(layer, "has_previous_state", False)),
            int(getattr(layer, "is_conv_states_initialized", False)),
            int(getattr(layer, "is_recurrent_states_initialized", False)),
        ],
        dtype=torch.int64,
        device=device,
    )
    dist_send(flags, dst=dst)
    conv = layer.conv_states if getattr(layer, "is_conv_states_initialized", False) else None
    rec = layer.recurrent_states if getattr(layer, "is_recurrent_states_initialized", False) else None
    _send_tensor(conv, dst, device)
    _send_tensor(rec, dst, device)


def _recv_linear_body(
    device: torch.device, src: int, dtype: torch.dtype, *, max_snapshots: int
) -> PipelineLinearAttentionLayer:
    flags = torch.empty(3, dtype=torch.int64, device=device)
    dist_recv(flags, src=src)
    layer = PipelineLinearAttentionLayer(max_snapshots=max_snapshots)
    layer.has_previous_state = bool(int(flags[0].item()))
    conv = _recv_tensor(device, src, dtype)
    rec = _recv_tensor(device, src, dtype)
    if conv is not None:
        if not layer.is_conv_states_initialized:
            layer.lazy_initialization(conv_states=conv)
        layer.conv_states.copy_(conv)
        layer.is_conv_states_initialized = True
    if rec is not None:
        if not layer.is_recurrent_states_initialized:
            layer.lazy_initialization(recurrent_states=rec)
        layer.recurrent_states.copy_(rec)
        layer.is_recurrent_states_initialized = True
    layer.seed_snapshot_from_current_state()
    return layer


def _send_hybrid_body(layer: Any, dst: int, device: torch.device) -> None:
    flags = torch.tensor(
        [
            int(getattr(layer, "has_previous_state", False)),
            int(getattr(layer, "is_conv_states_initialized", False)),
            int(getattr(layer, "is_recurrent_states_initialized", False)),
            int(getattr(layer, "is_initialized", False)),
        ],
        dtype=torch.int64,
        device=device,
    )
    dist_send(flags, dst=dst)
    conv = layer.conv_states if getattr(layer, "is_conv_states_initialized", False) else None
    rec = layer.recurrent_states if getattr(layer, "is_recurrent_states_initialized", False) else None
    _send_tensor(conv, dst, device)
    _send_tensor(rec, dst, device)
    if getattr(layer, "is_initialized", False):
        _send_tensor(layer.keys, dst, device)
        _send_tensor(layer.values, dst, device)
    else:
        _send_tensor(None, dst, device)
        _send_tensor(None, dst, device)


def _recv_hybrid_body(
    device: torch.device, src: int, dtype: torch.dtype, *, max_snapshots: int
) -> PipelineLinearAttentionAndFullAttentionLayer:
    flags = torch.empty(4, dtype=torch.int64, device=device)
    dist_recv(flags, src=src)
    layer = PipelineLinearAttentionAndFullAttentionLayer(max_snapshots=max_snapshots)
    layer.has_previous_state = bool(int(flags[0].item()))
    conv = _recv_tensor(device, src, dtype)
    rec = _recv_tensor(device, src, dtype)
    if conv is not None:
        if not layer.is_conv_states_initialized:
            layer.lazy_initialization(conv_states=conv)
        layer.conv_states.copy_(conv)
        layer.is_conv_states_initialized = True
    if rec is not None:
        if not layer.is_recurrent_states_initialized:
            layer.lazy_initialization(recurrent_states=rec)
        layer.recurrent_states.copy_(rec)
        layer.is_recurrent_states_initialized = True
    if int(flags[3].item()):
        keys = _recv_tensor(device, src, dtype)
        values = _recv_tensor(device, src, dtype)
        layer.dtype = keys.dtype
        layer.device = keys.device
        layer.keys = keys
        layer.values = values
        layer.is_initialized = True
    layer.seed_snapshot_from_current_state()
    return layer


def _send_layer_with_kind(layer: Any, kind: int, dst: int, device: torch.device) -> None:
    kind_t = torch.tensor([int(kind)], dtype=torch.int64, device=device)
    dist_send(kind_t, dst=dst)
    if kind == 0:
        _send_dynamic_body(layer, dst, device)
    elif kind == 1:
        _send_linear_body(layer, dst, device)
    else:
        _send_hybrid_body(layer, dst, device)


def _recv_layer_with_kind(
    kind: int,
    device: torch.device,
    src: int,
    dtype: torch.dtype,
    *,
    max_snapshots: int,
) -> Any:
    if kind == 0:
        return _recv_dynamic_body(device, src, dtype)
    if kind == 1:
        return _recv_linear_body(device, src, dtype, max_snapshots=max_snapshots)
    return _recv_hybrid_body(device, src, dtype, max_snapshots=max_snapshots)


def send_cache_shard(
    shard: DynamicCache,
    dst: int,
    device: torch.device,
    *,
    max_snapshots: int,
    global_layer_offset: int,
    layer_types: List[str],
) -> None:
    n = len(shard.layers)
    meta = torch.tensor([n], dtype=torch.int64, device=device)
    dist_send(meta, dst=dst)
    for local_idx, layer in enumerate(shard.layers):
        global_idx = int(global_layer_offset) + int(local_idx)
        kind = cache_kind_for_global_layer(global_idx, layer_types)
        _send_layer_with_kind(layer, kind, dst, device)


def recv_cache_shard(
    src: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_snapshots: int,
    global_layer_offset: int,
    layer_types: List[str],
) -> DynamicCache:
    meta = torch.empty(1, dtype=torch.int64, device=device)
    dist_recv(meta, src=src)
    n = int(meta.item())
    shard = DynamicCache()
    layers = []
    for local_idx in range(n):
        global_idx = int(global_layer_offset) + int(local_idx)
        kind = cache_kind_for_global_layer(global_idx, layer_types)
        kind_t = torch.empty(1, dtype=torch.int64, device=device)
        dist_recv(kind_t, src=src)
        recv_kind = int(kind_t.item())
        if recv_kind != kind:
            raise RuntimeError(
                f"KV shard kind mismatch at global layer {global_idx}: "
                f"expected {kind}, got {recv_kind}."
            )
        layers.append(_recv_layer_with_kind(kind, device, src, dtype, max_snapshots=max_snapshots))
    shard.layers = layers
    return shard
