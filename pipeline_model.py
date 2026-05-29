"""
Pipelined speculative decoding for Qwen3-family models.
========================================================

Implementation of the parallel target/speculation schedule:

- **Inference**: target pipeline step and speculation run in parallel each round. Speculation consumes
  ``[g_n^{evicted}, g_{n-1}, ..., g_1, g_0]`` (``n+1`` rows when an evicted token exists), so the
  reusable KV prefix rolls back one more slot than v9 (drop last ``n`` speculative keys when full).
- **Training**: spec tower sees ``[g_n, g_{n-1}, ..., g_1, g_0]`` — **(n+1)·S** tokens; ``g_0`` 仅来自
  token embedding，经 ``g0_proj`` **FC** 后再输入塔内；**仅** ``g_0^t`` 为 query。Attention 与 v7 一致：
  ``g_0^t`` may attend ``g_k^T`` iff ``T<=t`` and ``k=min(n, t-T)``（远处为 ``g_n``，与推理 + KV 一致）。
  当 ``simulated_pipeline_fill = n-a`` 时，``g_{n-1}..g_{n-a}`` 复用融合后的 ``g_n`` 行。

- Per-stage projections; **``g_0``** 仅取自 token embedding，经独立 **FC** ``g0_proj`` 后再与其它 ``g_k`` 一并进入 speculation tower。
- Fixed memory ``g_n..g_1`` 上对 ``g_n..g_2`` 使用 per-layer FC（与 v9 相同）。
- Speculation tower uses ``Qwen3DecoderLayer`` plus rotary cloned from the base model.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3DecoderLayer,
    eager_attention_forward,
)

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:  # older transformers
    from transformers.models.qwen3.modeling_qwen3 import create_causal_mask


@contextmanager
def _null_ctx():
    yield


def _is_valid_multinomial_probs(p: torch.Tensor) -> bool:
    if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)) or torch.any(p < 0):
        return False
    tot = p.sum()
    return bool(float(tot) > torch.finfo(p.dtype).eps * max(1.0, float(p.numel())))


def _sanitize_probs_for_multinomial(p: torch.Tensor) -> torch.Tensor:
    """Finite nonnegative weights summing to 1; avoids CUDA multinomial assert on NaN/inf/<0."""
    out = p.detach().float().clamp(min=0.0)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    tot = out.sum()
    if float(tot) <= torch.finfo(out.dtype).eps * max(1.0, float(out.numel())):
        out = torch.full_like(out, 1.0 / float(out.numel()))
    else:
        out = out / tot
    return out.to(dtype=p.dtype, device=p.device)


def _sampling_probs_hf_style(
    logits_1d: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Match HuggingFace-style sampling distribution: temperature / top-k / top-p on logits, then softmax.
    `logits_1d` is shape [V].
    """
    vocab = int(logits_1d.shape[-1])
    scores = logits_1d.unsqueeze(0).float()
    processors = LogitsProcessorList()
    if temperature > 0 and abs(float(temperature) - 1.0) > 1e-15:
        processors.append(TemperatureLogitsWarper(float(temperature)))
    tk = int(top_k)
    if tk > 0 and tk < vocab:
        processors.append(TopKLogitsWarper(top_k=tk))
    tp = float(top_p)
    if tp < 1.0 - 1e-15:
        processors.append(TopPLogitsWarper(top_p=tp))
    dummy_input_ids = torch.zeros((scores.shape[0], 1), device=scores.device, dtype=torch.long)
    warped = processors(dummy_input_ids, scores)
    probs = F.softmax(warped[0], dim=-1)
    if _is_valid_multinomial_probs(probs):
        return probs
    return _sanitize_probs_for_multinomial(probs)


def _verify_pipeline_draft_token(
    verified_logits_1d: torch.Tensor,
    speculated_id: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    q_full: Optional[torch.Tensor] = None,
) -> Tuple[bool, int]:
    """
    Speculative-decoding verify step.

    Returns (accepted, next_token_id). If accepted is True, next_token_id equals speculated_id.
    If accepted is False, next_token_id is the replacement (greedy: target argmax; sampling:
    rejection sample with residual max(0, p - q) when the draft proposal is rejected).

    For sampling, `q_full` is the draft proposal distribution over full vocabulary (same length as
    target logits), built with the same temperature / top_k / top_p as `p`.
    """
    if greedy:
        tid = int(verified_logits_1d.argmax(dim=-1).item())
        if speculated_id == tid:
            return True, speculated_id
        return False, tid

    if q_full is None:
        raise ValueError("q_full is required for sampling verify")

    p = _sampling_probs_hf_style(
        verified_logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
    )
    q = q_full.to(device=p.device, dtype=p.dtype)
    v = int(p.shape[-1])
    if int(q.shape[-1]) != v:
        raise ValueError(f"q_full length {q.shape[-1]} does not match target vocab {v}")

    px = p[speculated_id]
    qx = q[speculated_id]
    u = torch.rand((), device=p.device, dtype=p.dtype)
    if qx > 0 and u * qx <= px:
        return True, speculated_id

    resid = torch.clamp(p - q, min=0.0)
    rs = resid.sum()
    if rs <= torch.finfo(resid.dtype).eps * max(1.0, float(v)):
        src = p if _is_valid_multinomial_probs(p) else _sanitize_probs_for_multinomial(p)
        y = int(torch.multinomial(src, 1).item())
    else:
        resid_probs = resid / rs
        src = resid_probs if _is_valid_multinomial_probs(resid_probs) else _sanitize_probs_for_multinomial(resid_probs)
        y = int(torch.multinomial(src, 1).item())
    return False, y


def _try_get_dtensor_type():
    try:
        from torch.distributed.tensor import DTensor

        return DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor

            return DTensor
        except ImportError:
            return None


def _materialize_fsdp_tensor(t: torch.Tensor) -> torch.Tensor:
    dtensor_type = _try_get_dtensor_type()
    if dtensor_type is None or not isinstance(t, dtensor_type):
        return t
    import torch.distributed as dist

    def _from_local() -> torch.Tensor:
        to_local = getattr(t, "to_local", None)
        if not callable(to_local):
            raise RuntimeError("DTensor has no to_local(); cannot materialize without distributed.")
        return to_local().detach()

    if not (dist.is_available() and dist.is_initialized()):
        return _from_local()
    try:
        return t.full_tensor().detach()
    except RuntimeError:
        return _from_local()


def _materialize_state_dict_for_load(
    state: Dict[str, torch.Tensor],
    map_location: str | torch.device | None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v = _materialize_fsdp_tensor(v)
            if map_location is not None:
                v = v.to(map_location)
        out[k] = v
    return out


def _assert_supported_main_model_type(config) -> None:
    mt = getattr(config, "model_type", None)
    supported = {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama"}
    if mt not in supported:
        raise ValueError(f"Unsupported base model_type={mt!r}. Expected one of {sorted(supported)}.")


def _decoder_relevant_config(config: Any) -> Any:
    """Configs like ``Qwen3_5Config`` keep decoder fields on ``text_config``."""
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config


def num_hidden_layers_from_hf_config(config: Any) -> int:
    """Decoder layer count for standard configs and composite Qwen3.5-style configs."""
    c = _decoder_relevant_config(config)
    n = getattr(c, "num_hidden_layers", None)
    if n is not None:
        return int(n)
    n = getattr(c, "num_layers", None)
    if n is not None:
        return int(n)
    raise AttributeError(
        f"Cannot resolve num_hidden_layers from config type {type(config).__name__!r} "
        f"(decoder subconfig type {type(c).__name__!r})."
    )


def _linear_and_hybrid_attention_layer_indices_for_cache(dec_cfg: Any) -> List[int]:
    """
    Layer indices whose cache slots must be rebuilt from a prefix forward after rejection.

    Must match ``DynamicCache(config=...)`` indexing: ``layer_types`` with optional
    ``num_kv_shared_layers`` trim. Do **not** use ``isinstance(..., LinearAttentionLayer)``:
    HF uses the same class for empty placeholders (e.g. ``layer_types[i] == \"moe\"``), and
    overwriting those breaks the cache.
    """
    lt = getattr(dec_cfg, "layer_types", None)
    if lt is None:
        return []
    lt = list(lt)
    n_skip = int(getattr(dec_cfg, "num_kv_shared_layers", 0) or 0)
    if n_skip > 0:
        lt = lt[:-n_skip]
    return [i for i, t in enumerate(lt) if t in ("linear_attention", "hybrid")]


def _speculation_attn_config(config) -> Any:
    c = copy.deepcopy(config)
    return c


def _get_apply_rotary_pos_emb(config):
    mt = getattr(config, "model_type", None)
    if mt in ("qwen3", "qwen3_moe"):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    if mt in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text", "qwen3_5_moe"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    if mt == "llama":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    return apply_rotary_pos_emb


def _clone_rotary_embedding_from_base(base_rotary_emb: nn.Module, *, device: torch.device) -> nn.Module:
    rotary = copy.deepcopy(base_rotary_emb)
    rotary.to(device=device)
    return rotary


class PipelineDecoderLayer(nn.Module):
    """
    Decoder layer wrapper with explicit attention forward so RoPE application can follow
    the base model's rotary style (e.g., Qwen3.5 partial RoPE) instead of Qwen3 hardcoded path.
    """

    def __init__(
        self,
        config,
        *,
        layer_idx: int,
        apply_rotary_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self._apply_rotary_fn = apply_rotary_fn
        base_layer = Qwen3DecoderLayer(config, layer_idx=self.layer_idx)
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, use_cache
        if position_embeddings is None:
            raise ValueError("position_embeddings is required for PipelineDecoderLayer.forward")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        sa = self.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, sa.head_dim)
        query_states = sa.q_proj(hidden_states).view(hidden_shape)
        key_states = sa.k_proj(hidden_states).view(hidden_shape)
        value_states = sa.v_proj(hidden_states).view(hidden_shape)
        if hasattr(sa, "q_norm"):
            query_states = sa.q_norm(query_states)
        if hasattr(sa, "k_norm"):
            key_states = sa.k_norm(key_states)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_fn(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            sa,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else sa.attention_dropout,
            scaling=sa.scaling,
            sliding_window=sa.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = sa.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _force_spec_layer_full_attention(layer: nn.Module) -> None:
    layer.self_attn.layer_type = "full_attention"


def _copy_matching_decoder_layer_weights_from_base(
    dst_layer: nn.Module,
    src_layer: nn.Module,
    *,
    spec_layer_idx: int,
    base_layer_idx: int,
) -> None:
    dst_state = dst_layer.state_dict()
    src_state = src_layer.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    unmatched_dst: List[str] = []
    shape_mismatch: List[str] = []
    for k, v in dst_state.items():
        if k not in src_state:
            unmatched_dst.append(k)
            continue
        src_v = src_state[k]
        if src_v.shape != v.shape:
            shape_mismatch.append(k)
            continue
        matched[k] = src_v.detach().to(device=v.device, dtype=v.dtype)
    dst_state.update(matched)
    dst_layer.load_state_dict(dst_state, strict=False)
    unmatched_src = [k for k in src_state.keys() if k not in dst_state]
    if unmatched_dst or unmatched_src or shape_mismatch:
        warnings.warn(
            (
                f"Spec layer {spec_layer_idx} init from base layer {base_layer_idx}: "
                f"copied {len(matched)} params, "
                f"dst_only={len(unmatched_dst)}, src_only={len(unmatched_src)}, shape_mismatch={len(shape_mismatch)}. "
                "Unmatched params keep random initialization."
            ),
            stacklevel=2,
        )

def _build_pipeline_training_mask(
    attention_mask_ns: torch.Tensor,
    *,
    n: int,
    s: int,
    mask_dtype: torch.dtype,
) -> torch.Tensor:
    """
    ``[B, (n+1)*S]`` padding mask -> ``[B, 1, S, (n+1)*S]`` additive mask.

    Layout (stage-major): blocks ``b=0..n`` are ``g_n, g_{n-1}, ..., g_0``; each block has time
    ``0..S-1``. Only ``g_0`` queries (last block). Structural rule (v7 with ``i=0``): ``g_0^t`` attends
    ``g_k^T`` iff ``T<=t`` and ``k = min(n, t-T)``.
    """
    b, lseq = attention_mask_ns.shape
    n1 = int(n) + 1
    if lseq != n1 * s:
        raise ValueError(f"expected length (n+1)*S={n1 * s}, got {lseq}")
    device = attention_mask_ns.device
    neg = torch.finfo(mask_dtype).min

    pos_list: List[int] = []
    i_list: List[int] = []
    for block in range(n1):
        i_st = int(n) - int(block)
        for t in range(s):
            pos_list.append(t)
            i_list.append(i_st)
    pos = torch.tensor(pos_list, device=device, dtype=torch.long)  # [(n+1)S]
    i_t = torch.tensor(i_list, device=device, dtype=torch.long)  # [(n+1)S]

    pq = torch.arange(s, device=device, dtype=torch.long).view(1, 1, s, 1)
    iq = torch.zeros((1, 1, s, 1), device=device, dtype=torch.long)
    pk = pos.view(1, 1, 1, lseq)
    ik = i_t.view(1, 1, 1, lseq)
    n_t = torch.tensor(int(n), device=device, dtype=torch.long)

    k_required = torch.minimum(n_t, iq + (pq - pk))
    structural = (pk <= pq) & (ik == k_required)

    v = attention_mask_ns.to(dtype=torch.bool)
    q_valid = v[:, int(n) * s : n1 * s].view(b, 1, s, 1)
    k_valid = v.view(b, 1, 1, lseq)
    return torch.where(structural & q_valid & k_valid, torch.zeros((), device=device, dtype=mask_dtype), neg)


class SpeculationHeadTransformer(nn.Module):
    """
    Speculation transformer (v10): per-stage ``g_1..g_n`` projections plus ``g_0``.
    ``g_0`` 仅由 token embedding 经可学习 FC ``g0_proj`` 得到；训练为 **(n+1)·S** 布局 ``[g_n,...,g_0]``，
    仅 ``g_0`` 为 query，结构 mask 同 v7；推理为 decoder + KV（``n`` 或 ``n+1`` 个 query 位置）。

    Fixed 段 ``g_n..g_1`` 上对 stage ``>1`` 使用与 v9 相同的 per-layer FC。
    Uses ``Qwen3DecoderLayer`` plus rotary cloned from the base model.
    """

    def __init__(
        self,
        config,
        dtype: torch.dtype,
        device: torch.device,
        base_rotary_emb: nn.Module,
        apply_rotary_fn,
        stage_feature_hf_indices: Sequence[Sequence[int]],
        num_spec_layers: int = 1,
        init_weights_from_base_layer_indices: Optional[Sequence[int]] = None,
        base_decoder_layers: Optional[nn.ModuleList] = None,
        draft_vocab_size: Optional[int] = None,
        decoder_layer_cls: Optional[type] = None,
    ):
        super().__init__()
        self.config = config
        self._decoder_layer_cls = decoder_layer_cls or PipelineDecoderLayer
        self.num_spec_layers = int(num_spec_layers)
        if self.num_spec_layers < 1:
            raise ValueError(f"num_spec_layers must be >= 1, got {num_spec_layers}")
        if (init_weights_from_base_layer_indices is None) ^ (base_decoder_layers is None):
            raise ValueError(
                "init_weights_from_base_layer_indices and base_decoder_layers must be both set or both omitted."
            )
        if init_weights_from_base_layer_indices is not None:
            mapping = list(init_weights_from_base_layer_indices)
            if len(mapping) != self.num_spec_layers:
                raise ValueError(
                    f"init_weights_from_base_layer_indices must have length {self.num_spec_layers}, got {len(mapping)}"
                )
            n_base = num_hidden_layers_from_hf_config(config)
            for i, j in enumerate(mapping):
                if not isinstance(j, int) or j < 0 or j >= n_base:
                    raise ValueError(f"init_weights_from_base_layer_indices[{i}] out of range: {j}")

        self.stage_feature_hf_indices = [tuple(int(x) for x in row) for row in stage_feature_hf_indices]
        if not self.stage_feature_hf_indices:
            raise ValueError("stage_feature_hf_indices must be non-empty.")
        h = int(config.hidden_size)
        self.stage_projs = nn.ModuleList(
            [
                nn.Linear(len(row) * h, h, bias=False, device=device, dtype=dtype)
                for row in self.stage_feature_hf_indices
            ]
        )
        # g_0 只能来自 token embedding (H_0)，进入 speculation 塔前必须经过可学习 FC（与其它 g 的 stage_proj 一致）
        self.g0_proj = nn.Linear(h, h, bias=False, device=device, dtype=dtype)
        nn.init.normal_(self.g0_proj.weight, std=0.02)

        self.draft_vocab_size = int(draft_vocab_size) if draft_vocab_size is not None else int(config.vocab_size)
        self.lm_head = nn.Linear(h, self.draft_vocab_size, bias=False, device=device, dtype=dtype)
        self.rotary_emb = _clone_rotary_embedding_from_base(base_rotary_emb, device=device)
        self._apply_rotary_fn = apply_rotary_fn
        self.spec_layers = nn.ModuleList(
            [
                self._decoder_layer_cls(config, layer_idx=i, apply_rotary_fn=self._apply_rotary_fn)
                for i in range(self.num_spec_layers)
            ]
        )
        self.spec_layers.to(device=device, dtype=dtype)
        for layer in self.spec_layers:
            _force_spec_layer_full_attention(layer)
        if init_weights_from_base_layer_indices is not None:
            for spec_i, base_i in enumerate(init_weights_from_base_layer_indices):
                _copy_matching_decoder_layer_weights_from_base(
                    self.spec_layers[spec_i],
                    base_decoder_layers[base_i],  # type: ignore[index]
                    spec_layer_idx=spec_i,
                    base_layer_idx=base_i,
                )
                _force_spec_layer_full_attention(self.spec_layers[spec_i])

        self.num_stages = len(self.stage_feature_hf_indices)
        self.fixed_stage_per_layer_projs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(
                            int(config.hidden_size),
                            int(config.hidden_size),
                            bias=False,
                            device=device,
                            dtype=dtype,
                        )
                        for _ in range(max(self.num_stages, 0))
                    ]
                )
                for _ in range(self.num_spec_layers)
            ]
        )
        for layer_projs in self.fixed_stage_per_layer_projs:
            for p in layer_projs:
                nn.init.eye_(p.weight)

    def _infer_stage_ids(self, q_len: int, device: torch.device) -> torch.LongTensor:
        if q_len == self.num_stages + 1:
            return torch.arange(self.num_stages, -1, -1, device=device, dtype=torch.long)
        if q_len == self.num_stages:
            return torch.arange(self.num_stages - 1, -1, -1, device=device, dtype=torch.long)
        return torch.full((q_len,), self.num_stages, device=device, dtype=torch.long)

    def _apply_fixed_stage_fc(
        self,
        h: torch.Tensor,
        stage_ids: torch.LongTensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.num_stages < 1:
            return h
        out = h
        for i in range(self.num_stages):
            stage_value = self.num_stages - i
            idx = torch.nonzero(stage_ids == stage_value, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            out[:, idx, :] = self.fixed_stage_per_layer_projs[layer_idx][i](out[:, idx, :])
        return out

    def _apply_inference_fixed_transform(
        self,
        fixed_hidden_states: torch.Tensor,
        fixed_stage_ids: torch.LongTensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self._apply_fixed_stage_fc(fixed_hidden_states, fixed_stage_ids, layer_idx)

    def forward_inference_g1_only_with_rotary(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        stage_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != int(self.config.hidden_size):
            raise ValueError(
                f"hidden_states last dim must be hidden_size={self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        _, q_len, _ = hidden_states.shape
        device = hidden_states.device
        layer_past = past_key_values if use_cache else None
        past_seen = past_key_values.get_seq_length() if layer_past is not None else 0
        cache_position = torch.arange(past_seen, past_seen + q_len, device=device, dtype=torch.long)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": layer_past,
            "position_ids": position_ids,
        }
        attn_mask = create_causal_mask(**mask_kwargs)

        if stage_ids is None:
            stage_ids = self._infer_stage_ids(q_len, device=device)
        if stage_ids.shape != (q_len,):
            raise ValueError(f"stage_ids must have shape [{q_len}], got {tuple(stage_ids.shape)}")

        fixed_idx = torch.nonzero(stage_ids > 0, as_tuple=False).squeeze(-1)
        query_idx = torch.nonzero(stage_ids == 0, as_tuple=False).squeeze(-1)
        base_fixed = hidden_states[:, fixed_idx, :] if fixed_idx.numel() > 0 else None
        g_query_cur = hidden_states[:, query_idx, :] if query_idx.numel() > 0 else None

        out = hidden_states
        for li, layer in enumerate(self.spec_layers):
            full_in = hidden_states.clone()
            if fixed_idx.numel() > 0 and base_fixed is not None:
                fixed_cur = self._apply_inference_fixed_transform(base_fixed, stage_ids[fixed_idx], li)
                full_in[:, fixed_idx, :] = fixed_cur
            if query_idx.numel() > 0 and g_query_cur is not None:
                full_in[:, query_idx, :] = g_query_cur
            position_embeddings = self.rotary_emb(full_in, position_ids)
            out = layer(
                full_in,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=layer_past,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if query_idx.numel() > 0:
                g_query_cur = out[:, query_idx, :]
        if query_idx.numel() > 0:
            return out[:, query_idx, :]
        return out

    def forward_inference_with_rotary(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        stage_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Backward-compatible alias.
        return self.forward_inference_g1_only_with_rotary(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stage_ids=stage_ids,
        )

    def forward_training_g1_only_with_rotary(
        self,
        hidden_states_ns: torch.Tensor,
        position_ids_ns: torch.LongTensor,
        attention_mask_g1_to_ns: torch.Tensor,
        *,
        num_stages: int,
    ) -> torch.Tensor:
        bsz, n_s, hidden = hidden_states_ns.shape
        if hidden != int(self.config.hidden_size):
            raise ValueError(
                f"hidden_states last dim must be hidden_size={self.config.hidden_size}, got {hidden_states_ns.shape[-1]}"
            )
        n_bl = int(num_stages) + 1
        if n_s % n_bl != 0:
            raise ValueError(f"(n+1)*S must divide sequence length; got nS={n_s}, n={num_stages}")
        s = n_s // n_bl
        n_pipe = int(num_stages)
        fixed_len = n_pipe * s
        if attention_mask_g1_to_ns.shape != (bsz, 1, s, n_s):
            raise ValueError(
                f"attention_mask_g1_to_ns must be [B,1,S,(n+1)S], got {tuple(attention_mask_g1_to_ns.shape)}"
            )

        fixed_memory_0 = hidden_states_ns[:, :fixed_len, :]
        g1_cur = hidden_states_ns[:, fixed_len:, :]

        stage_ids = torch.repeat_interleave(
            torch.arange(n_pipe, 0, -1, device=hidden_states_ns.device, dtype=torch.long),
            s,
            dim=0,
        )

        for li, layer in enumerate(self.spec_layers):
            fixed_memory = self._apply_fixed_stage_fc(fixed_memory_0, stage_ids, li)
            full_in = torch.cat([fixed_memory, g1_cur], dim=1)
            full_ln = layer.input_layernorm(full_in)
            sa = layer.self_attn

            input_shape = full_ln.shape[:-1]
            hidden_shape = (*input_shape, -1, sa.head_dim)
            q_full = sa.q_proj(full_ln).view(hidden_shape)
            k_full = sa.k_proj(full_ln).view(hidden_shape)
            v_full = sa.v_proj(full_ln).view(hidden_shape)
            if hasattr(sa, "q_norm"):
                q_full = sa.q_norm(q_full)
            if hasattr(sa, "k_norm"):
                k_full = sa.k_norm(k_full)

            q_full = q_full.transpose(1, 2).contiguous()
            k_full = k_full.transpose(1, 2).contiguous()
            v_full = v_full.transpose(1, 2).contiguous()

            full_pos_emb = self.rotary_emb(full_in, position_ids_ns)
            cos, sin = full_pos_emb
            q_full, k_full = self._apply_rotary_fn(q_full, k_full, cos, sin, unsqueeze_dim=1)
            q_g1 = q_full[:, :, fixed_len:, :]

            n_heads = int(q_g1.shape[1])
            n_kv_heads = int(k_full.shape[1])
            if n_heads % n_kv_heads != 0:
                raise ValueError(f"GQA requires num_heads % num_kv_heads == 0; got {n_heads} % {n_kv_heads}")
            n_rep = n_heads // n_kv_heads
            k_attn = (
                k_full
                if n_rep == 1
                else k_full[:, :, None, :, :]
                .expand(k_full.shape[0], k_full.shape[1], n_rep, k_full.shape[2], k_full.shape[3])
                .reshape(k_full.shape[0], k_full.shape[1] * n_rep, k_full.shape[2], k_full.shape[3])
            )
            v_attn = (
                v_full
                if n_rep == 1
                else v_full[:, :, None, :, :]
                .expand(v_full.shape[0], v_full.shape[1], n_rep, v_full.shape[2], v_full.shape[3])
                .reshape(v_full.shape[0], v_full.shape[1] * n_rep, v_full.shape[2], v_full.shape[3])
            )

            dropout_p = float(getattr(sa, "attention_dropout", 0.0)) if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q_g1,
                k_attn,
                v_attn,
                attn_mask=attention_mask_g1_to_ns,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, s, -1)
            attn_out = sa.o_proj(attn_out)

            g1_after_attn = g1_cur + attn_out
            g1_cur = g1_after_attn + layer.mlp(layer.post_attention_layernorm(g1_after_attn))

        return g1_cur

    def forward_with_rotary(self, *args, **kwargs) -> torch.Tensor:
        # Backward-compatible alias used by existing inference code paths.
        return self.forward_inference_g1_only_with_rotary(*args, **kwargs)


class Qwen3SpeculativePipelineModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        num_stages: int = 2,
        num_spec_layers: int = 1,
        spec_init_from_base_layers: Optional[Sequence[int]] = None,
        draft_token_ids: Optional[Sequence[int]] = None,
        shallow_hidden_layer_indices: Optional[Sequence[Sequence[int]]] = None,
        trained_with_use_deepest: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.base_model_path = str(
            getattr(base_model, "name_or_path", None) or getattr(self.config, "_name_or_path", None) or ""
        )
        _assert_supported_main_model_type(self.config)
        self.num_stages = int(num_stages)
        self.num_spec_layers = int(num_spec_layers)
        self.trained_with_use_deepest = bool(trained_with_use_deepest)
        self.spec_init_from_base_layers: Optional[List[int]] = (
            list(spec_init_from_base_layers) if spec_init_from_base_layers is not None else None
        )

        self.num_layers = num_hidden_layers_from_hf_config(self.config)
        if self.num_layers % self.num_stages != 0:
            raise ValueError(
                f"num_layers ({self.num_layers}) must be divisible by num_stages ({self.num_stages})"
            )
        self.layers_per_stage = self.num_layers // self.num_stages
        dec_cfg = _decoder_relevant_config(self.config)
        self.hidden_size = int(dec_cfg.hidden_size)
        self.vocab_size = int(dec_cfg.vocab_size)
        self.dtype = self.base_model.dtype
        self.device = self.base_model.device

        if self.spec_init_from_base_layers is not None:
            if len(self.spec_init_from_base_layers) != self.num_spec_layers:
                raise ValueError(
                    f"spec_init_from_base_layers must have length num_spec_layers ({self.num_spec_layers}), "
                    f"got {len(self.spec_init_from_base_layers)}"
                )
            for j in self.spec_init_from_base_layers:
                if j < 0 or j >= self.num_layers:
                    raise ValueError(f"spec_init_from_base_layers entry out of range: {j}")

        self.shallow_hidden_layer_indices = self._normalize_stage_feature_indices(shallow_hidden_layer_indices)

        v_full = int(dec_cfg.vocab_size)
        if draft_token_ids is None:
            self._use_draft_vocab = False
            self.draft_vocab_size = v_full
            self._draft_token_ids: Optional[torch.Tensor] = None
        else:
            d_ids = sorted({int(x) for x in draft_token_ids})
            for tid in d_ids:
                if tid < 0 or tid >= v_full:
                    raise ValueError(f"draft_token_ids contains out-of-range id {tid} (vocab_size={v_full})")
            self._use_draft_vocab = True
            self.draft_vocab_size = len(d_ids)
            self.register_buffer("_draft_token_ids", torch.tensor(d_ids, dtype=torch.long), persistent=True)
            t2d = torch.zeros(v_full, dtype=torch.bool)
            t2d[torch.tensor(d_ids, dtype=torch.long)] = True
            self.register_buffer("_t2d_bool", t2d, persistent=True)
            to_draft = torch.full((v_full,), -1, dtype=torch.long)
            for i, tid in enumerate(d_ids):
                to_draft[tid] = i
            self.register_buffer("_token_id_to_draft_idx", to_draft, persistent=True)

        spec_cfg = _speculation_attn_config(dec_cfg)
        self.speculation_module = SpeculationHeadTransformer(
            spec_cfg,
            self.dtype,
            self.device,
            base_rotary_emb=self.base_model.model.rotary_emb,
            apply_rotary_fn=_get_apply_rotary_pos_emb(self.config),
            stage_feature_hf_indices=self.shallow_hidden_layer_indices,
            num_spec_layers=self.num_spec_layers,
            init_weights_from_base_layer_indices=self.spec_init_from_base_layers,
            base_decoder_layers=self.base_model.model.layers if self.spec_init_from_base_layers is not None else None,
            draft_vocab_size=self.draft_vocab_size,
        )

        if self._use_draft_vocab:
            with torch.no_grad():
                base_w = self.base_model.lm_head.weight
                for i in range(self.draft_vocab_size):
                    tid = int(self._draft_token_ids[i].item())
                    self.speculation_module.lm_head.weight[i].copy_(base_w[tid])

        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        self._stage_streams: List[Optional[torch.cuda.Stream]] = [
            torch.cuda.Stream() if torch.cuda.is_available() else None for _ in range(self.num_stages)
        ]

    @property
    def layers(self) -> nn.ModuleList:
        return self.base_model.model.layers

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.base_model.model.embed_tokens

    @property
    def final_norm(self) -> nn.Module:
        return self.base_model.model.norm

    @property
    def lm_head(self) -> nn.Linear:
        return self.base_model.lm_head

    @property
    def rotary_emb(self) -> nn.Module:
        """RoPE module for the frozen base transformer (not used by the speculation tower)."""
        return self.base_model.model.rotary_emb

    @property
    def speculation_head(self) -> SpeculationHeadTransformer:
        return self.speculation_module

    def _default_stage_feature_indices(self) -> List[Tuple[int, ...]]:
        """
        HF ``hidden_states`` indices: ``0`` = embeddings, ``k>=1`` = output after layer ``k-1``.

        - ``g_1``: ``h[0]``, ``h[lps]``
        - ``g_2``: ``h[0]``, ``h[lps]``, ``h[lps*2]``
        - ``g_i`` (``i>=3``): ``h[0]``, ``h[lps*i//3]``, ``h[lps*i//3*2]``, ``h[lps*i]``,
          every listed ``h[·]`` index is clamped to ``[0, min(num_layers, n*lps-2)]``.

        Rows are ordered ``g_n, g_{n-1}, ..., g_1`` (same as ``shallow_hidden_layer_indices``).
        """
        n = self.num_stages
        l = self.num_layers
        lps = self.layers_per_stage

        cap_hf = min(int(l), int(n) * int(lps) - 2)

        def clamp_hf(x: int) -> int:
            return max(0, min(int(x), cap_hf, int(l)))

        def indices_for_stage_i(i: int) -> Tuple[int, ...]:
            if i == 1:
                raw = (0, int(lps))
            elif i == 2:
                raw = (0, int(lps), int(lps) * 2)
            else:
                ii = int(i)
                li = int(lps) * ii
                q = li // 3
                raw = (0, q, q * 2, li)
            clamped = tuple(sorted({clamp_hf(int(x)) for x in raw}))
            return clamped

        rows: List[Tuple[int, ...]] = []
        for i in range(n, 0, -1):
            rows.append(indices_for_stage_i(i))
        return rows

    def _normalize_stage_feature_indices(
        self,
        shallow_hidden_layer_indices: Optional[Sequence[Sequence[int]]],
    ) -> List[Tuple[int, ...]]:
        if shallow_hidden_layer_indices is None:
            rows = self._default_stage_feature_indices()
        else:
            rows = [tuple(int(x) for x in row) for row in shallow_hidden_layer_indices]

        if len(rows) != self.num_stages:
            raise ValueError(
                f"shallow_hidden_layer_indices must have length num_stages={self.num_stages}, got {len(rows)}"
            )
        max_hf_idx = self.num_layers
        out: List[Tuple[int, ...]] = []
        for i, row in enumerate(rows):
            if len(row) < 1:
                raise ValueError(f"shallow_hidden_layer_indices[{i}] must be non-empty.")
            checked: List[int] = []
            for j in row:
                if j < 0 or j > max_hf_idx:
                    raise ValueError(
                        f"shallow_hidden_layer_indices[{i}] has invalid hidden-state index {j}; "
                        f"expected range [0, {max_hf_idx}]"
                    )
                checked.append(int(j))
            out.append(tuple(checked))
        return out

    def _snap_indices_needed(self) -> Set[int]:
        want: Set[int] = set()
        for row in self.shallow_hidden_layer_indices:
            for idx in row:
                want.add(int(idx))
        return want

    def _initial_pipeline_snap(self, hs: torch.Tensor) -> Dict[int, torch.Tensor]:
        """HF index ``0`` is embedding output; capture it when the pipeline entry is created."""
        snap: Dict[int, torch.Tensor] = {}
        if 0 in self._snap_indices_needed():
            snap[0] = hs
        return snap

    def _fuse_from_hf_indices(
        self,
        all_hs: Tuple[torch.Tensor, ...],
        hf_indices: Sequence[int],
        proj: nn.Linear,
    ) -> torch.Tensor:
        vecs = [all_hs[int(idx)] for idx in hf_indices]
        return proj(torch.cat(vecs, dim=-1))

    def _build_training_expanded_inputs(
        self,
        all_hs: Tuple[torch.Tensor, ...],
        simulated_pipeline_fill: Optional[int] = None,
    ) -> torch.Tensor:
        """
        v10 training layout (``n+1`` blocks of length ``S``), stage-major:
        ``[g_n, g_{n-1}, ..., g_1, g_0]``; ``g_0`` 仅由 ``all_hs[0]``（embedding）经 ``g0_proj`` FC 得到。

        Let ``fill = simulated_pipeline_fill`` and ``a = n - fill``. For blocks ``b=1..n-1``
        (i.e. nominal ``g_{n-b}`` with ``b`` in ``1..n-1``), when ``b <= a`` reuse the fused ``g_n``
        row; otherwise use the staircase row ``fused_rows[b]``. Block ``0`` is always ``g_n``;
        block ``n`` is always ``g_0``.
        """
        n = self.num_stages
        if simulated_pipeline_fill is None:
            fill = n
        else:
            fill = int(simulated_pipeline_fill)
            if fill < 1 or fill > n:
                raise ValueError(
                    f"simulated_pipeline_fill must be in [1, {n}], got {fill}"
                )

        fused_rows: List[torch.Tensor] = []
        for block in range(self.num_stages):
            fused_rows.append(
                self._fuse_from_hf_indices(
                    all_hs,
                    self.shallow_hidden_layer_indices[block],
                    self.speculation_module.stage_projs[block],
                )
            )
        g0_row = self.speculation_module.g0_proj(all_hs[0])

        a = n - fill
        rows: List[torch.Tensor] = []
        rows.append(fused_rows[0])
        for b in range(1, n):
            if b <= a:
                rows.append(fused_rows[0])
            else:
                rows.append(fused_rows[b])
        rows.append(g0_row)
        return torch.cat(rows, dim=1)

    def _build_inference_row_from_snap(
        self,
        snap: Dict[int, torch.Tensor],
        i_stages: int,
    ) -> torch.Tensor:
        # i_stages: n..1. stage_feature_hf_indices is ordered [g_n..g_1].
        block = self.num_stages - i_stages
        hf_indices = self.shallow_hidden_layer_indices[block]
        proj = self.speculation_module.stage_projs[block]
        vecs = [snap[int(idx)] for idx in hf_indices]
        return proj(torch.cat(vecs, dim=-1))

    def _build_inference_g0_row_from_hs(self, hs: torch.Tensor) -> torch.Tensor:
        """``g_0``：当前 token 的 embedding（``[B,1,H]``）经 ``g0_proj`` FC 后再作为 speculation 输入。"""
        return self.speculation_module.g0_proj(hs)

    def _choose_inference_i_stages_for_snap(
        self,
        snap: Dict[int, torch.Tensor],
        i_nominal: int,
        use_deepest: bool,
        *,
        search_hi: Optional[int] = None,
    ) -> int:
        """
        Pick which g_{i_stages} block (and matching projection) to use for one speculation row.

        ``i_nominal`` is the layout role before ``use_deepest`` upgrade (larger = deeper in the
        pipelined window). When ``use_deepest`` is True, upgrade up to ``search_hi`` (default
        ``num_stages`` = full ``g_n``). Always pick the deepest ``i_stages`` in ``[1, hi]`` whose
        HF indices are all present in ``snap``; downgrade when snapshots are still partial.
        """
        n = self.num_stages
        inom = int(i_nominal)
        if inom <= 0:
            return inom
        if not use_deepest:
            hi = inom
        else:
            hi = int(search_hi) if search_hi is not None else n
            if hi > n:
                hi = n
        for i_stages in range(hi, 0, -1):
            block = n - i_stages
            hf_indices = self.shallow_hidden_layer_indices[block]
            if all(int(idx) in snap for idx in hf_indices):
                return i_stages
        return inom

    def _extract_position_snapshots_from_hidden_states(
        self,
        all_hs: Tuple[torch.Tensor, ...],
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Convert full-sequence hidden_states into per-position snapshot dicts keyed by HF hidden-state index.
        all_hs[k]: [B, S, H], batch must be 1 in generate().
        """
        if all_hs[0].shape[0] != 1:
            raise ValueError("Position snapshot extraction expects batch_size=1.")
        need = sorted(self._snap_indices_needed())
        seq_len = int(all_hs[0].shape[1])
        snaps_by_pos: Dict[int, Dict[int, torch.Tensor]] = {}
        for pos in range(seq_len):
            snap: Dict[int, torch.Tensor] = {}
            for idx in need:
                snap[idx] = all_hs[idx][:, pos : pos + 1, :]
            snaps_by_pos[pos] = snap
        return snaps_by_pos

    def training_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        simulated_pipeline_fill: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if labels is None:
            raise ValueError("labels must be provided for training_forward")
        if labels.shape[1] != input_ids.shape[1]:
            raise ValueError("labels and input_ids must have the same sequence length")

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_logits = outputs.logits
            all_hs = outputs.hidden_states

        b, s, _ = all_hs[0].shape
        n = self.num_stages
        if attention_mask is None:
            attn2d = torch.ones((b, s), device=input_ids.device, dtype=torch.long)
        else:
            attn2d = attention_mask.to(device=input_ids.device, dtype=torch.long)

        spec_hidden = self._build_training_expanded_inputs(
            all_hs,
            simulated_pipeline_fill=simulated_pipeline_fill,
        )
        pos_ns = torch.arange(s, device=spec_hidden.device, dtype=torch.long).repeat(n + 1).unsqueeze(0).expand(b, -1)
        attn_ns = attn2d.repeat(1, n + 1)
        mask_4d = _build_pipeline_training_mask(attn_ns, n=n, s=s, mask_dtype=self.dtype)

        g1_processed = self.speculation_module.forward_training_g1_only_with_rotary(
            spec_hidden,
            pos_ns,
            mask_4d,
            num_stages=n,
        )
        g1_processed = self.final_norm(g1_processed)
        spec_logits = self.speculation_module.lm_head(g1_processed)

        teacher_target = teacher_logits.detach()
        if self._use_draft_vocab:
            t2d = self._t2d_bool.to(device=teacher_target.device)
            teacher_target = teacher_target[..., t2d]
            k = int(self.draft_vocab_size)
        else:
            k = int(teacher_logits.shape[-1])

        spec_logits_flat = spec_logits.float().reshape(-1, k)
        teacher_target_flat = teacher_target.float().reshape(-1, k)

        next_labels = torch.full((b, s), -100, dtype=labels.dtype, device=labels.device)
        if s > 1:
            next_labels[:, : s - 1] = labels[:, 1:s]
        valid_mask_2d = next_labels != -100
        teacher_argmax_full = teacher_logits.argmax(dim=-1)
        if self._use_draft_vocab:
            teacher_argmax_draft = self._token_id_to_draft_idx.to(teacher_argmax_full.device)[teacher_argmax_full]
            teacher_in_draft = teacher_argmax_draft >= 0
            valid_mask_2d = valid_mask_2d & teacher_in_draft
            target_for_acc_2d = teacher_argmax_draft.to(spec_logits.device)
        else:
            target_for_acc_2d = teacher_argmax_full.to(spec_logits.device)

        pred_2d = spec_logits.argmax(dim=-1)
        correct_2d = (pred_2d == target_for_acc_2d) & valid_mask_2d
        valid_count = valid_mask_2d.sum()
        if valid_count > 0:
            acc = correct_2d.sum().float() / valid_count.float()
        else:
            acc = spec_logits.sum() * 0.0

        valid = valid_mask_2d.reshape(-1).to(spec_logits_flat.device)
        if not valid.any():
            z = spec_logits_flat.sum() * 0.0
            return {"loss": z, "kl_loss": z.detach(), "acc": z.detach()}

        spec_v = spec_logits_flat[valid]
        teacher_v = teacher_target_flat[valid]

        teacher_probs = F.softmax(teacher_v / temperature, dim=-1)
        student_log_probs = F.log_softmax(spec_v / temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
        return {"loss": kl_loss, "kl_loss": kl_loss.detach(), "acc": acc.detach()}

    def _forward_stage_with_snapshots(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        start_layer: int,
        end_layer: int,
        past_key_values: Cache,
        snapshot_hf_indices: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        snaps: Dict[int, torch.Tensor] = {}
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx in range(start_layer, end_layer):
            hidden_states = self.layers[layer_idx](
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            out_idx = layer_idx + 1
            if snapshot_hf_indices is not None and out_idx in snapshot_hf_indices:
                snaps[out_idx] = hidden_states
        return hidden_states, snaps

    def _rollback_kv_cache(self, past_kv: Cache, target_length: int) -> None:
        if hasattr(past_kv, "crop") and callable(past_kv.crop):
            past_kv.crop(target_length)
            return
        key_cache = getattr(past_kv, "key_cache", None)
        if key_cache is not None:
            for i in range(len(key_cache)):
                cur_len = past_kv.key_cache[i].shape[2]
                if cur_len > target_length:
                    past_kv.key_cache[i] = past_kv.key_cache[i][:, :, :target_length, :]
                    past_kv.value_cache[i] = past_kv.value_cache[i][:, :, :target_length, :]
            if hasattr(past_kv, "_seen_tokens"):
                past_kv._seen_tokens = target_length
            return
        raise TypeError(f"Cannot roll back KV cache: unsupported past_key_values type {type(past_kv)}")

    def _snapshot_linear_attention_cache_layers(
        self,
        past_kv: Cache,
        layer_indices: Sequence[int],
    ) -> Dict[int, Any]:
        """
        Clone cache slots for linear/hybrid attention layers only.
        """
        if not layer_indices:
            return {}
        if not hasattr(past_kv, "layers"):
            raise TypeError(f"Cannot snapshot linear cache layers for type {type(past_kv)}.")
        snaps: Dict[int, Any] = {}
        for i in layer_indices:
            li = int(i)
            if li < 0 or li >= len(past_kv.layers):
                raise RuntimeError(f"layer index {i} out of range for past_kv ({len(past_kv.layers)} layers).")
            snaps[li] = copy.deepcopy(past_kv.layers[li])
        return snaps

    def _restore_linear_attention_cache_layers_from_stage_snapshots(
        self,
        past_kv: Cache,
        snapshot_history: Sequence[Dict[int, Any]],
        layer_indices: Sequence[int],
        layers_per_stage: int,
        num_stages: int,
    ) -> None:
        """
        Restore linear/hybrid cache slots from rolling stage snapshots.

        Snapshot history stores one copy per decode iteration (oldest -> newest). For a rollback to
        the currently verified prefix, shallower stages must rewind more steps than deeper stages.
        """
        if not layer_indices:
            return
        if not hasattr(past_kv, "layers"):
            raise TypeError(f"Cannot restore linear cache layers for type {type(past_kv)}.")
        if layers_per_stage <= 0:
            raise ValueError(f"layers_per_stage must be > 0, got {layers_per_stage}.")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be > 0, got {num_stages}.")
        if len(snapshot_history) < num_stages:
            raise RuntimeError(
                f"Need at least {num_stages} linear-cache snapshots for rollback, got {len(snapshot_history)}."
            )
        latest_idx = len(snapshot_history) - 1
        for i in layer_indices:
            li = int(i)
            stage_idx = li // int(layers_per_stage)
            if stage_idx < 0 or stage_idx >= int(num_stages):
                raise RuntimeError(
                    f"layer index {li} maps to invalid stage {stage_idx} "
                    f"(layers_per_stage={layers_per_stage}, num_stages={num_stages})."
                )
            rewind_steps = int(num_stages) - 1 - stage_idx
            hist_idx = latest_idx - rewind_steps
            if hist_idx < 0:
                raise RuntimeError(
                    f"Cannot restore layer {li}: history index {hist_idx} < 0 (rewind={rewind_steps})."
                )
            src = snapshot_history[hist_idx]
            if li not in src:
                raise RuntimeError(f"Linear cache snapshot at history[{hist_idx}] misses layer {li}.")
            past_kv.layers[li] = copy.deepcopy(src[li])

    def _rollback_kv_cache_after_rejection(
        self,
        past_kv: Cache,
        target_length: int,
        linear_cache_snapshots: Sequence[Dict[int, Any]],
        linear_layer_indices: Sequence[int],
        layers_per_stage: int,
        num_stages: int,
    ) -> None:
        """
        Roll back target-model cache on draft rejection:
        - ``crop`` all layers for standard KV slots;
        - restore linear/hybrid slots from rolling snapshots (no full-prefix rebuild).
        """
        self._rollback_kv_cache(past_kv, target_length)
        self._restore_linear_attention_cache_layers_from_stage_snapshots(
            past_kv,
            linear_cache_snapshots,
            linear_layer_indices,
            layers_per_stage,
            num_stages,
        )

    def _rollback_spec_kv_cache(self, spec_past: Cache, target_length: int) -> None:
        if hasattr(spec_past, "crop") and callable(spec_past.crop):
            if spec_past.get_seq_length() > target_length:
                spec_past.crop(target_length)
            return
        raise TypeError(f"Cannot roll back speculation KV cache: unsupported type {type(spec_past)}")

    def _materialize_unique_past_and_spec_kv(self, chains: List[Dict[str, Any]]) -> None:
        """
        Draft-tree siblings may share ``past_kv`` / ``spec_past_kv`` (lazy fork). This loop runs
        chains sequentially; the first forward on a shared cache would corrupt later siblings.
        For each duplicate ``id(past_kv)``, keep the first chain's reference and deep-copy both
        caches for later chains so each survivor gets an isolated writable cache before forwards.
        """
        seen: Set[int] = set()
        for chain in chains:
            pid = id(chain["past_kv"])
            if pid in seen:
                chain["past_kv"] = copy.deepcopy(chain["past_kv"])
                chain["spec_past_kv"] = copy.deepcopy(chain["spec_past_kv"])
            else:
                seen.add(pid)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        verify: bool = True,
        use_deepest: Optional[bool] = None,
        draft_top_k: int = 1,
    ) -> Tuple[List[int], List[bool], int]:
        # If the head was trained with use_deepest, inference always uses deepest; otherwise honor
        # ``use_deepest`` (None -> False).
        trained_deepest = bool(getattr(self, "trained_with_use_deepest", False))
        if trained_deepest:
            ud = True
        elif use_deepest is None:
            ud = False
        else:
            ud = bool(use_deepest)
        dk = int(draft_top_k)
        if dk <= 0:
            raise ValueError(f"draft_top_k must be > 0, got {draft_top_k}")
        if dk == 1:
            return self._generate_single_chain(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                verify=verify,
                use_deepest=ud,
            )
        return self._generate_with_draft_tree_topk(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            draft_top_k=dk,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            verify=verify,
            use_deepest=ud,
        )

    @torch.no_grad()
    def _generate_single_chain(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        verify: bool = True,
        use_deepest: bool = False,
    ) -> Tuple[List[int], List[bool], int]:
        device = input_ids.device
        b, s0 = input_ids.shape
        if b != 1:
            raise ValueError("Pipeline generation only supports batch_size=1")
        n = self.num_stages
        lps = self.layers_per_stage
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        def _sync_dev() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        _sync_dev()
        prefill_wall_start = time.perf_counter()

        def _record_ready_event() -> Optional[torch.cuda.Event]:
            if device.type != "cuda":
                return None
            ev = torch.cuda.Event(enable_timing=False)
            ev.record(torch.cuda.current_stream())
            return ev

        snap_want = self._snap_indices_needed()
        dec_cfg = _decoder_relevant_config(self.config)
        linear_cache_layer_indices = _linear_and_hybrid_attention_layer_indices_for_cache(dec_cfg)
        linear_cache_snapshots: List[Dict[int, Any]] = []

        outputs = self.base_model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv: Cache = outputs.past_key_values
        all_hs = outputs.hidden_states

        completed_snaps: Dict[int, Dict[int, torch.Tensor]] = self._extract_position_snapshots_from_hidden_states(
            all_hs
        )

        prefill_cache_len = max(0, s0 - n + 1)
        spec_past_kv = DynamicCache()
        if prefill_cache_len > 0:
            prefill_rows: List[torch.Tensor] = []
            for pos in range(prefill_cache_len):
                prefill_rows.append(self._build_inference_row_from_snap(completed_snaps[pos], n))
            prefill_gn = torch.cat(prefill_rows, dim=1)
            prefill_pos = torch.arange(prefill_cache_len, device=device, dtype=torch.long).unsqueeze(0)
            self.speculation_module.forward_inference_g1_only_with_rotary(
                prefill_gn,
                prefill_pos,
                attention_mask=None,
                past_key_values=spec_past_kv,
                use_cache=True,
            )

        first_logits = outputs.logits[:, -1, :]
        if greedy:
            first_token_id = int(first_logits.argmax(dim=-1).item())
        else:
            probs = _sampling_probs_hf_style(
                first_logits[0], temperature=temperature, top_k=top_k, top_p=top_p
            )
            first_token_id = int(torch.multinomial(probs, 1).item())

        generated_ids: List[int] = [first_token_id]
        token_acceptance: List[bool] = [True]

        del all_hs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        first_emb = self.embed_tokens(torch.tensor([[first_token_id]], device=device))
        first_ready = _record_ready_event()
        pipeline: List[Dict[str, Any]] = [
            {
                "hs": first_emb,
                "pos": s0,
                "snap": self._initial_pipeline_snap(first_emb),
                "ready_event": first_ready,
            }
        ]

        draft_full_q: Dict[int, torch.Tensor] = {}
        next_position = s0 + 1
        verified_up_to = s0 + 1
        decode_loop_steps = 0
        prev_evicted_snap: Optional[Dict[int, torch.Tensor]] = None
        prev_evicted_pos: Optional[int] = None

        def run_spec_parallel() -> torch.Tensor:
            """
            v10 speculation rows (oldest -> newest positions):
            optional ``[g_n^{evicted}, g_{n-1}, ..., g_0]`` (``n+1`` rows) else ``[g_{n-1},...,g_0]`` (``n``).

            When ``use_deepest`` and there is **no** evicted prefix (e.g. pipeline not yet full),
            in-window fused rows may upgrade to ``g_n`` if that token's snapshot already contains
            all HF indices for the ``g_n`` block. With an evicted ``g_n`` row present, fused rows
            cap at ``g_{n-1}`` so ``g_n`` stays on the evicted slot only.
            """
            if not pipeline:
                raise RuntimeError("Pipeline is unexpectedly empty before speculation.")
            newest_pos = int(pipeline[0]["pos"])
            oldest_needed = newest_pos - n + 1
            if oldest_needed < 0:
                raise ValueError(
                    f"Need {n} speculation rows but only positions >=0 exist: newest_pos={newest_pos}."
                )

            active_by_pos: Dict[int, Dict[str, Any]] = {int(e["pos"]): e for e in pipeline}
            rows: List[torch.Tensor] = []
            pos_list: List[int] = []
            has_evicted_prefix = prev_evicted_snap is not None and prev_evicted_pos is not None
            fused_search_hi = (n - 1) if (use_deepest and has_evicted_prefix) else None
            if has_evicted_prefix:
                ev_i = self._choose_inference_i_stages_for_snap(
                    prev_evicted_snap, n, use_deepest, search_hi=n
                )
                rows.append(self._build_inference_row_from_snap(prev_evicted_snap, ev_i))
                pos_list.append(int(prev_evicted_pos))
            for pos in range(oldest_needed, newest_pos + 1):
                i_nominal_pipe = newest_pos - pos
                if pos in active_by_pos:
                    snap_src = active_by_pos[pos]["snap"]
                else:
                    if pos not in completed_snaps:
                        raise KeyError(f"Missing completed snapshot for position {pos}.")
                    snap_src = completed_snaps[pos]
                if i_nominal_pipe == 0:
                    rows.append(self._build_inference_g0_row_from_hs(active_by_pos[newest_pos]["hs"]))
                else:
                    i_stages = self._choose_inference_i_stages_for_snap(
                        snap_src, i_nominal_pipe, use_deepest, search_hi=fused_search_hi
                    )
                    rows.append(self._build_inference_row_from_snap(snap_src, i_stages))
                pos_list.append(pos)
            cur_in = torch.cat(rows, dim=1)
            min_p = min(pos_list)
            self._rollback_spec_kv_cache(spec_past_kv, min_p)
            pos_ids = torch.tensor([pos_list], device=device, dtype=torch.long)
            proc = self.speculation_module.forward_inference_g1_only_with_rotary(
                cur_in,
                pos_ids,
                attention_mask=None,
                past_key_values=spec_past_kv,
                use_cache=True,
            )
            proc = self.final_norm(proc)
            return self.speculation_module.lm_head(proc[:, -1:, :])

        acc_timings = [0.0, 0.0]

        _sync_dev()
        decode_wall_start = time.perf_counter()

        def _finalize_generate_timing(
            out_ids: List[int], out_accept: List[bool], out_steps: int
        ) -> Tuple[List[int], List[bool], int]:
            _sync_dev()
            decode_wall_end = time.perf_counter()
            decode_wall_sec = decode_wall_end - decode_wall_start
            prefill_wall_sec = decode_wall_start - prefill_wall_start
            ideal_saved = acc_timings[1]
            ideal_decode_wall_sec = max(decode_wall_sec - ideal_saved, 1e-12)
            self._last_generate_timing = {
                "prefill_wall_sec": float(prefill_wall_sec),
                "decode_wall_sec": float(decode_wall_sec),
                "pipeline_stage_stream_gpu_sec": float(acc_timings[0]),
                "pipeline_ideal_parallel_saved_sec": float(ideal_saved),
                "ideal_decode_wall_sec": float(ideal_decode_wall_sec),
            }
            return out_ids, out_accept, out_steps

        while verified_up_to - s0 < max_new_tokens:
            decode_loop_steps += 1
            newest_pos0 = int(pipeline[0]["pos"])
            oldest_needed0 = newest_pos0 - n + 1
            pending_spec_logits: Optional[torch.Tensor] = None
            if oldest_needed0 >= 0:
                pending_spec_logits = run_spec_parallel()

            use_streams = torch.cuda.is_available() and len(pipeline) > 1
            L_pl = len(pipeline)
            stage_ev_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
            iter_stage_sec_cpu = 0.0
            for stage_idx in range(L_pl):
                entry = pipeline[stage_idx]
                start_layer = stage_idx * lps
                end_layer = (stage_idx + 1) * lps
                stream = self._stage_streams[stage_idx] if use_streams else None
                if use_streams and stream is not None:
                    ready_event = entry.get("ready_event")
                    if ready_event is not None:
                        stream.wait_event(ready_event)
                ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
                if device.type == "cuda":
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    strm = stream if stream is not None else torch.cuda.current_stream(device=device)
                    ev_s.record(strm)
                    with ctx:
                        entry["hs"], collected = self._forward_stage_with_snapshots(
                            entry["hs"],
                            torch.tensor([[entry["pos"]]], device=device),
                            start_layer,
                            end_layer,
                            past_kv,
                            snapshot_hf_indices=snap_want,
                        )
                        entry["snap"].update(collected)
                        entry["ready_event"] = _record_ready_event() if use_streams else None
                    ev_e.record(strm)
                    stage_ev_pairs.append((ev_s, ev_e))
                else:
                    t0_s = time.perf_counter()
                    with ctx:
                        entry["hs"], collected = self._forward_stage_with_snapshots(
                            entry["hs"],
                            torch.tensor([[entry["pos"]]], device=device),
                            start_layer,
                            end_layer,
                            past_kv,
                            snapshot_hf_indices=snap_want,
                        )
                        entry["snap"].update(collected)
                        entry["ready_event"] = _record_ready_event() if use_streams else None
                    iter_stage_sec_cpu += time.perf_counter() - t0_s

            if device.type == "cuda":
                _sync_dev()
                iter_stage_ms = 0.0
                for ev_s, ev_e in stage_ev_pairs:
                    iter_stage_ms += ev_s.elapsed_time(ev_e)
                iter_stage_sec = iter_stage_ms / 1000.0
            else:
                iter_stage_sec = iter_stage_sec_cpu

            acc_timings[0] += iter_stage_sec
            if L_pl > 1:
                acc_timings[1] += iter_stage_sec * float(L_pl - 1) / float(L_pl)
            if linear_cache_layer_indices:
                linear_cache_snapshots.append(
                    self._snapshot_linear_attention_cache_layers(past_kv, linear_cache_layer_indices)
                )

            if use_streams and pipeline:
                newest_event = pipeline[0].get("ready_event")
                if newest_event is not None:
                    torch.cuda.current_stream().wait_event(newest_event)
                if len(pipeline) >= n:
                    completed_event = pipeline[-1].get("ready_event")
                    if completed_event is not None:
                        torch.cuda.current_stream().wait_event(completed_event)

            if len(pipeline) >= n:
                completed = pipeline[-1]
                completed_pos = int(completed["pos"])
                completed_snaps[completed_pos] = completed["snap"]
                target_pos = int(completed["pos"]) + 1
                target_gen_idx = target_pos - s0

                if target_gen_idx < len(generated_ids):
                    speculated_id = int(generated_ids[target_gen_idx])
                    if verify:
                        verified_hs_normed = self.final_norm(completed["hs"])
                        verified_logits = self.lm_head(verified_hs_normed)
                        vlog1 = verified_logits[0, 0]
                        if greedy:
                            accepted, verified_next_id = _verify_pipeline_draft_token(
                                vlog1, speculated_id, True, temperature, top_k, top_p
                            )
                        else:
                            accepted, verified_next_id = _verify_pipeline_draft_token(
                                vlog1,
                                speculated_id,
                                False,
                                temperature,
                                top_k,
                                top_p,
                                draft_full_q[target_pos],
                            )

                        if not accepted:
                            if use_streams:
                                for ent in pipeline:
                                    ev2 = ent.get("ready_event")
                                    if ev2 is not None:
                                        torch.cuda.current_stream().wait_event(ev2)
                            generated_ids = generated_ids[:target_gen_idx]
                            generated_ids.append(verified_next_id)
                            token_acceptance = token_acceptance[:target_gen_idx]
                            token_acceptance.append(False)

                            self._rollback_kv_cache_after_rejection(
                                past_kv,
                                target_pos,
                                linear_cache_snapshots,
                                linear_cache_layer_indices,
                                lps,
                                n,
                            )
                            self._rollback_spec_kv_cache(spec_past_kv, target_pos)
                            for old_pos in [p for p in completed_snaps if p >= target_pos]:
                                del completed_snaps[old_pos]

                            correct_emb = self.embed_tokens(torch.tensor([[verified_next_id]], device=device))
                            pipeline = [
                                {
                                    "hs": correct_emb,
                                    "pos": target_pos,
                                    "snap": self._initial_pipeline_snap(correct_emb),
                                    "ready_event": _record_ready_event(),
                                }
                            ]
                            next_position = target_pos + 1
                            verified_up_to = target_pos + 1
                            linear_cache_snapshots = []
                            for pk in [k for k in list(draft_full_q.keys()) if k >= target_pos]:
                                del draft_full_q[pk]
                            prev_evicted_snap = None
                            prev_evicted_pos = None
                            if verified_next_id == eos_token_id:
                                break
                            continue

                    verified_up_to = target_pos + 1
                    if speculated_id == eos_token_id:
                        end = target_gen_idx + 1
                        return _finalize_generate_timing(
                            generated_ids[:end],
                            token_acceptance[:end],
                            decode_loop_steps,
                        )

                prev_evicted_snap = dict(completed["snap"])
                prev_evicted_pos = int(completed_pos)
                pipeline.pop()
                if linear_cache_layer_indices and len(linear_cache_snapshots) >= n:
                    linear_cache_snapshots.pop(0)

            if pending_spec_logits is not None:
                spec_logits = pending_spec_logits
            else:
                spec_logits = run_spec_parallel()
            logits_1d = spec_logits[0, 0]
            if greedy:
                pick = logits_1d.argmax()
                if self._use_draft_vocab:
                    tid = self._draft_token_ids.to(device)
                    next_id = int(tid[pick].item())
                else:
                    next_id = int(pick.item())
            else:
                if self._use_draft_vocab:
                    probs_d = _sampling_probs_hf_style(
                        logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    v_full = int(self.vocab_size)
                    q_full = torch.zeros(v_full, device=device, dtype=probs_d.dtype)
                    tid = self._draft_token_ids.to(device).long()
                    q_full.index_add_(0, tid, probs_d)
                    if not _is_valid_multinomial_probs(q_full):
                        q_full = _sanitize_probs_for_multinomial(q_full)
                    d_src = probs_d if _is_valid_multinomial_probs(probs_d) else _sanitize_probs_for_multinomial(probs_d)
                    d_idx = int(torch.multinomial(d_src, 1).item())
                    next_id = int(tid[d_idx].item())
                else:
                    q_full = _sampling_probs_hf_style(
                        logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    q_src = q_full if _is_valid_multinomial_probs(q_full) else _sanitize_probs_for_multinomial(q_full)
                    next_id = int(torch.multinomial(q_src, 1).item())
                draft_full_q[next_position] = q_full.detach()

            generated_ids.append(next_id)
            token_acceptance.append(True)

            new_emb = self.embed_tokens(torch.tensor([[next_id]], device=device))
            pipeline.insert(
                0,
                {
                    "hs": new_emb,
                    "pos": next_position,
                    "snap": self._initial_pipeline_snap(new_emb),
                    "ready_event": _record_ready_event(),
                },
            )
            next_position += 1

        return _finalize_generate_timing(
            generated_ids[:max_new_tokens],
            token_acceptance[:max_new_tokens],
            decode_loop_steps,
        )

    @torch.no_grad()
    def _generate_with_draft_tree_topk(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        draft_top_k: int = 4,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        verify: bool = True,
        use_deepest: bool = False,
    ) -> Tuple[List[int], List[bool], int]:
        device = input_ids.device
        b, s0 = input_ids.shape
        if b != 1:
            raise ValueError("Pipeline generation only supports batch_size=1")
        n = self.num_stages
        lps = self.layers_per_stage
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        def _sync_dev() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        _sync_dev()
        prefill_wall_start = time.perf_counter()

        snap_want = self._snap_indices_needed()
        dec_cfg = _decoder_relevant_config(self.config)
        linear_cache_layer_indices = _linear_and_hybrid_attention_layer_indices_for_cache(dec_cfg)

        outputs = self.base_model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        base_past_kv: Cache = outputs.past_key_values
        all_hs = outputs.hidden_states
        completed_snaps = self._extract_position_snapshots_from_hidden_states(all_hs)

        spec_past_kv = DynamicCache()
        prefill_cache_len = max(0, s0 - n + 1)
        if prefill_cache_len > 0:
            prefill_rows: List[torch.Tensor] = []
            for pos in range(prefill_cache_len):
                prefill_rows.append(self._build_inference_row_from_snap(completed_snaps[pos], n))
            prefill_gn = torch.cat(prefill_rows, dim=1)
            prefill_pos = torch.arange(prefill_cache_len, device=device, dtype=torch.long).unsqueeze(0)
            self.speculation_module.forward_inference_g1_only_with_rotary(
                prefill_gn,
                prefill_pos,
                attention_mask=None,
                past_key_values=spec_past_kv,
                use_cache=True,
            )

        first_logits = outputs.logits[:, -1, :]
        if greedy:
            first_token_id = int(first_logits.argmax(dim=-1).item())
        else:
            probs = _sampling_probs_hf_style(
                first_logits[0], temperature=temperature, top_k=top_k, top_p=top_p
            )
            first_token_id = int(torch.multinomial(probs, 1).item())

        verified_ids: List[int] = [first_token_id]
        verified_acceptance: List[bool] = [True]

        del all_hs, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        first_emb = self.embed_tokens(torch.tensor([[first_token_id]], device=device))
        chain0: Dict[str, Any] = {
            "past_kv": base_past_kv,
            "spec_past_kv": spec_past_kv,
            "pipeline": [
                {
                    "hs": first_emb,
                    "pos": s0,
                    "snap": self._initial_pipeline_snap(first_emb),
                    "ready_event": None,
                }
            ],
            "completed_snaps": completed_snaps,
            "draft_full_q": {},
            "linear_cache_snapshots": [],
            "next_position": s0 + 1,
            "generated_ids": [first_token_id],
            "token_acceptance": [True],
            "score": 0.0,
            "prev_evicted_snap": None,
            "prev_evicted_pos": None,
        }
        chains: List[Dict[str, Any]] = [chain0]
        verified_up_to = s0 + 1
        decode_loop_steps = 0
        acc_timings = [0.0, 0.0]

        _sync_dev()
        decode_wall_start = time.perf_counter()

        def _finalize_generate_timing(
            out_ids: List[int], out_accept: List[bool], out_steps: int
        ) -> Tuple[List[int], List[bool], int]:
            _sync_dev()
            decode_wall_end = time.perf_counter()
            decode_wall_sec = decode_wall_end - decode_wall_start
            prefill_wall_sec = decode_wall_start - prefill_wall_start
            ideal_saved = acc_timings[1]
            ideal_decode_wall_sec = max(decode_wall_sec - ideal_saved, 1e-12)
            self._last_generate_timing = {
                "prefill_wall_sec": float(prefill_wall_sec),
                "decode_wall_sec": float(decode_wall_sec),
                "pipeline_stage_stream_gpu_sec": float(acc_timings[0]),
                "pipeline_ideal_parallel_saved_sec": float(ideal_saved),
                "ideal_decode_wall_sec": float(ideal_decode_wall_sec),
            }
            return out_ids, out_accept, out_steps

        def _lazy_fork_chain_for_branch(chain: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "past_kv": chain["past_kv"],
                "spec_past_kv": chain["spec_past_kv"],
                "pipeline": [
                    {
                        "hs": e["hs"].clone(),
                        "pos": int(e["pos"]),
                        "snap": dict(e["snap"]),
                        "ready_event": None,
                    }
                    for e in chain["pipeline"]
                ],
                "completed_snaps": dict(chain["completed_snaps"]),
                "draft_full_q": dict(chain["draft_full_q"]),
                "linear_cache_snapshots": list(chain["linear_cache_snapshots"]),
                "next_position": int(chain["next_position"]),
                "generated_ids": list(chain["generated_ids"]),
                "token_acceptance": list(chain["token_acceptance"]),
                "score": float(chain["score"]),
                "prev_evicted_snap": None if chain.get("prev_evicted_snap") is None else dict(chain["prev_evicted_snap"]),
                "prev_evicted_pos": chain.get("prev_evicted_pos"),
            }

        def _run_spec_parallel_for_chain(chain: Dict[str, Any]) -> torch.Tensor:
            pipeline = chain["pipeline"]
            if not pipeline:
                raise RuntimeError("Pipeline is unexpectedly empty before speculation.")
            newest_pos = int(pipeline[0]["pos"])
            oldest_needed = newest_pos - n + 1
            if oldest_needed < 0:
                raise ValueError(
                    f"Need {n} speculation rows but only positions >=0 exist: newest_pos={newest_pos}."
                )

            active_by_pos: Dict[int, Dict[str, Any]] = {int(e["pos"]): e for e in pipeline}
            rows: List[torch.Tensor] = []
            pos_list: List[int] = []
            prev_evicted_snap = chain.get("prev_evicted_snap")
            prev_evicted_pos = chain.get("prev_evicted_pos")
            has_evicted_prefix = prev_evicted_snap is not None and prev_evicted_pos is not None
            fused_search_hi = (n - 1) if (use_deepest and has_evicted_prefix) else None
            if has_evicted_prefix:
                ev_i = self._choose_inference_i_stages_for_snap(
                    prev_evicted_snap, n, use_deepest, search_hi=n
                )
                rows.append(self._build_inference_row_from_snap(prev_evicted_snap, ev_i))
                pos_list.append(int(prev_evicted_pos))
            for pos in range(oldest_needed, newest_pos + 1):
                i_nominal_pipe = newest_pos - pos
                if pos in active_by_pos:
                    snap_src = active_by_pos[pos]["snap"]
                else:
                    if pos not in chain["completed_snaps"]:
                        raise KeyError(f"Missing completed snapshot for position {pos}.")
                    snap_src = chain["completed_snaps"][pos]
                if i_nominal_pipe == 0:
                    rows.append(self._build_inference_g0_row_from_hs(active_by_pos[newest_pos]["hs"]))
                else:
                    i_stages = self._choose_inference_i_stages_for_snap(
                        snap_src, i_nominal_pipe, use_deepest, search_hi=fused_search_hi
                    )
                    rows.append(self._build_inference_row_from_snap(snap_src, i_stages))
                pos_list.append(pos)

            cur_in = torch.cat(rows, dim=1)
            min_p = min(pos_list)
            self._rollback_spec_kv_cache(chain["spec_past_kv"], min_p)
            pos_ids = torch.tensor([pos_list], device=device, dtype=torch.long)
            proc = self.speculation_module.forward_inference_g1_only_with_rotary(
                cur_in,
                pos_ids,
                attention_mask=None,
                past_key_values=chain["spec_past_kv"],
                use_cache=True,
            )
            proc = self.final_norm(proc)
            return self.speculation_module.lm_head(proc[:, -1:, :])

        def _expand_chain_once(chain: Dict[str, Any]) -> List[Dict[str, Any]]:
            spec_logits = chain.pop("_pending_spec_logits", None)
            if spec_logits is None:
                spec_logits = _run_spec_parallel_for_chain(chain)
            logits_1d = spec_logits[0, 0]
            pos = int(chain["next_position"])
            q_full: Optional[torch.Tensor] = None

            if greedy:
                probs_1d = F.softmax(logits_1d.float(), dim=-1)
                k_take = min(int(draft_top_k), int(probs_1d.shape[-1]))
                conf_vals, picked_idx = torch.topk(probs_1d, k_take, dim=-1)
                if self._use_draft_vocab:
                    tid = self._draft_token_ids.to(device).long()
                    token_ids = tid[picked_idx.long()]
                else:
                    token_ids = picked_idx.long()
            else:
                if self._use_draft_vocab:
                    probs_d = _sampling_probs_hf_style(
                        logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    v_full = int(self.vocab_size)
                    q_full = torch.zeros(v_full, device=device, dtype=probs_d.dtype)
                    tid = self._draft_token_ids.to(device).long()
                    q_full.index_add_(0, tid, probs_d)
                    if not _is_valid_multinomial_probs(q_full):
                        q_full = _sanitize_probs_for_multinomial(q_full)
                    k_take = min(int(draft_top_k), int(probs_d.shape[-1]))
                    conf_vals, picked_idx = torch.topk(probs_d, k_take, dim=-1)
                    token_ids = tid[picked_idx.long()]
                else:
                    q_full = _sampling_probs_hf_style(
                        logits_1d, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    if not _is_valid_multinomial_probs(q_full):
                        q_full = _sanitize_probs_for_multinomial(q_full)
                    k_take = min(int(draft_top_k), int(q_full.shape[-1]))
                    conf_vals, token_ids = torch.topk(q_full, k_take, dim=-1)
                    token_ids = token_ids.long()

            out_children: List[Dict[str, Any]] = []
            for i in range(int(token_ids.shape[0])):
                nid = int(token_ids[i].item())
                conf = float(conf_vals[i].item())
                child = _lazy_fork_chain_for_branch(chain)
                child["generated_ids"].append(nid)
                child["token_acceptance"].append(True)
                if not greedy:
                    child["draft_full_q"][pos] = q_full.detach()
                new_emb = self.embed_tokens(torch.tensor([[nid]], device=device))
                child["pipeline"].insert(
                    0,
                    {
                        "hs": new_emb,
                        "pos": pos,
                        "snap": self._initial_pipeline_snap(new_emb),
                        "ready_event": None,
                    },
                )
                child["next_position"] = pos + 1
                child["score"] = float(child["score"] + torch.log(torch.tensor(max(conf, 1e-20))).item())
                out_children.append(child)
            return out_children

        while len(verified_ids) < max_new_tokens:
            decode_loop_steps += 1
            self._materialize_unique_past_and_spec_kv(chains)
            for chain in chains:
                pl = chain["pipeline"]
                newest_pos0 = int(pl[0]["pos"])
                oldest_needed0 = newest_pos0 - n + 1
                chain["_pending_spec_logits"] = (
                    _run_spec_parallel_for_chain(chain) if oldest_needed0 >= 0 else None
                )
            iter_stage_sec_cpu = 0.0
            verify_candidates: List[Dict[str, Any]] = []
            chains_needing_tail_pop: List[Dict[str, Any]] = []

            for chain_idx, chain in enumerate(chains):
                pipeline = chain["pipeline"]
                for stage_idx in range(len(pipeline)):
                    entry = pipeline[stage_idx]
                    start_layer = stage_idx * lps
                    end_layer = (stage_idx + 1) * lps
                    t0_s = time.perf_counter()
                    entry["hs"], collected = self._forward_stage_with_snapshots(
                        entry["hs"],
                        torch.tensor([[entry["pos"]]], device=device),
                        start_layer,
                        end_layer,
                        chain["past_kv"],
                        snapshot_hf_indices=snap_want,
                    )
                    entry["snap"].update(collected)
                    iter_stage_sec_cpu += time.perf_counter() - t0_s

                if linear_cache_layer_indices:
                    chain["linear_cache_snapshots"].append(
                        self._snapshot_linear_attention_cache_layers(
                            chain["past_kv"], linear_cache_layer_indices
                        )
                    )

                if len(pipeline) >= n:
                    completed = pipeline[-1]
                    completed_pos = int(completed["pos"])
                    chain["completed_snaps"][completed_pos] = completed["snap"]
                    target_pos = completed_pos + 1
                    target_gen_idx = target_pos - s0
                    if target_pos == verified_up_to and target_gen_idx < len(chain["generated_ids"]):
                        speculated_id = int(chain["generated_ids"][target_gen_idx])
                        verified_hs_normed = self.final_norm(completed["hs"])
                        verified_logits = self.lm_head(verified_hs_normed)
                        vlog1 = verified_logits[0, 0]
                        if verify:
                            if greedy:
                                accepted, verified_next_id = _verify_pipeline_draft_token(
                                    vlog1, speculated_id, True, temperature, top_k, top_p
                                )
                            else:
                                if target_pos not in chain["draft_full_q"]:
                                    raise KeyError(
                                        f"Missing draft proposal distribution q_full for position {target_pos}."
                                    )
                                accepted, verified_next_id = _verify_pipeline_draft_token(
                                    vlog1,
                                    speculated_id,
                                    False,
                                    temperature,
                                    top_k,
                                    top_p,
                                    chain["draft_full_q"][target_pos],
                                )
                        else:
                            accepted, verified_next_id = True, speculated_id
                        verify_candidates.append(
                            {
                                "chain_idx": chain_idx,
                                "target_pos": target_pos,
                                "target_gen_idx": target_gen_idx,
                                "speculated_id": speculated_id,
                                "accepted": bool(accepted),
                                "verified_next_id": int(verified_next_id),
                                "score": float(chain["score"]),
                            }
                        )
                    chains_needing_tail_pop.append(chain)

            acc_timings[0] += iter_stage_sec_cpu

            if verify_candidates:
                accepted_candidates = [c for c in verify_candidates if c["accepted"]]
                if accepted_candidates:
                    chosen = max(accepted_candidates, key=lambda x: x["score"])
                    keep_idx = int(chosen["chain_idx"])
                    chosen_chain = chains[keep_idx]
                    accepted_id = int(chosen["speculated_id"])
                    verified_ids.append(accepted_id)
                    verified_acceptance.append(True)
                    verified_up_to += 1
                    for pk in [k for k in list(chosen_chain["draft_full_q"].keys()) if k < verified_up_to]:
                        del chosen_chain["draft_full_q"][pk]
                    if len(chosen_chain["pipeline"]) >= n:
                        completed_tail = chosen_chain["pipeline"][-1]
                        chosen_chain["prev_evicted_snap"] = dict(completed_tail["snap"])
                        chosen_chain["prev_evicted_pos"] = int(completed_tail["pos"])
                        chosen_chain["pipeline"].pop()
                        if linear_cache_layer_indices and len(chosen_chain["linear_cache_snapshots"]) >= n:
                            chosen_chain["linear_cache_snapshots"].pop(0)
                    chains = [chosen_chain]
                    if accepted_id == eos_token_id:
                        return _finalize_generate_timing(
                            verified_ids[:max_new_tokens],
                            verified_acceptance[:max_new_tokens],
                            decode_loop_steps,
                        )
                else:
                    chosen = max(verify_candidates, key=lambda x: x["score"])
                    chosen_chain = chains[int(chosen["chain_idx"])]
                    target_pos = int(chosen["target_pos"])
                    target_gen_idx = int(chosen["target_gen_idx"])
                    verified_next_id = int(chosen["verified_next_id"])

                    self._rollback_kv_cache_after_rejection(
                        chosen_chain["past_kv"],
                        target_pos,
                        chosen_chain["linear_cache_snapshots"],
                        linear_cache_layer_indices,
                        lps,
                        n,
                    )
                    self._rollback_spec_kv_cache(chosen_chain["spec_past_kv"], target_pos)
                    for old_pos in [p for p in chosen_chain["completed_snaps"] if p >= target_pos]:
                        del chosen_chain["completed_snaps"][old_pos]
                    for pk in [k for k in list(chosen_chain["draft_full_q"].keys()) if k >= target_pos]:
                        del chosen_chain["draft_full_q"][pk]

                    chosen_chain["generated_ids"] = chosen_chain["generated_ids"][:target_gen_idx]
                    chosen_chain["generated_ids"].append(verified_next_id)
                    chosen_chain["token_acceptance"] = chosen_chain["token_acceptance"][:target_gen_idx]
                    chosen_chain["token_acceptance"].append(False)

                    correct_emb = self.embed_tokens(torch.tensor([[verified_next_id]], device=device))
                    chosen_chain["pipeline"] = [
                        {
                            "hs": correct_emb,
                            "pos": target_pos,
                            "snap": self._initial_pipeline_snap(correct_emb),
                            "ready_event": None,
                        }
                    ]
                    chosen_chain["next_position"] = target_pos + 1
                    chosen_chain["linear_cache_snapshots"] = []
                    chosen_chain["score"] = 0.0
                    chosen_chain["prev_evicted_snap"] = None
                    chosen_chain["prev_evicted_pos"] = None
                    chains = [chosen_chain]

                    verified_ids.append(verified_next_id)
                    verified_acceptance.append(False)
                    verified_up_to = target_pos + 1
                    if verified_next_id == eos_token_id:
                        return _finalize_generate_timing(
                            verified_ids[:max_new_tokens],
                            verified_acceptance[:max_new_tokens],
                            decode_loop_steps,
                        )
                    continue
            else:
                for c in chains_needing_tail_pop:
                    if len(c["pipeline"]) >= n:
                        ct = c["pipeline"][-1]
                        c["prev_evicted_snap"] = dict(ct["snap"])
                        c["prev_evicted_pos"] = int(ct["pos"])
                        c["pipeline"].pop()
                        if linear_cache_layer_indices and len(c["linear_cache_snapshots"]) >= n:
                            c["linear_cache_snapshots"].pop(0)

            if len(verified_ids) >= max_new_tokens:
                break

            expanded_chains: List[Dict[str, Any]] = []
            for chain in chains:
                expanded_chains.extend(_expand_chain_once(chain))
            if not expanded_chains:
                break
            expanded_chains.sort(key=lambda x: x["score"], reverse=True)
            chains = expanded_chains[: int(draft_top_k)]

        return _finalize_generate_timing(
            verified_ids[:max_new_tokens],
            verified_acceptance[:max_new_tokens],
            decode_loop_steps,
        )

    def save_speculation_head(self, path: str) -> None:
        cfg: Dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "draft_vocab_size": self.draft_vocab_size,
            "num_stages": self.num_stages,
            "num_spec_layers": self.num_spec_layers,
            "version": 10,
            "trained_with_use_deepest": bool(self.trained_with_use_deepest),
            "shallow_hidden_layer_indices": [list(x) for x in self.shallow_hidden_layer_indices],
            "base_model_path": self.base_model_path,
        }
        if self.spec_init_from_base_layers is not None:
            cfg["spec_init_from_base_layers"] = self.spec_init_from_base_layers
        if self._use_draft_vocab:
            cfg["draft_token_ids"] = self._draft_token_ids.detach().cpu().tolist()
        torch.save({"state_dict": self.speculation_module.state_dict(), "config": cfg}, path)

    def load_speculation_head(self, path: str, map_location: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict):
            cfg = ckpt.get("config")
            if isinstance(cfg, dict) and "trained_with_use_deepest" in cfg:
                self.trained_with_use_deepest = bool(cfg["trained_with_use_deepest"])
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = _materialize_state_dict_for_load(state, map_location)
        self.speculation_module.load_state_dict(state)
