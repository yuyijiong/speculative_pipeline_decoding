"""
Pipelined speculative decoding for Qwen3-family models (v11).
===========================================================

See ``结构v11.md`` for ``aggr_feature_bound``, training mask, and inference schedule.

- **Inference**: same parallel target/speculation schedule as v10; rows use ``g_(f(d), d)`` via
  ``aggr_feature_bound`` depth mapping.
- **Training**: ``m`` aggregation blocks ``[g_{m-1}, ..., g_0]`` (not ``n+1``); all blocks are
  Q/K/V; only ``g_0`` feeds ``lm_head``.
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

from pipeline_linear_cache import (
    crop_pipeline_cache_after_rejection,
    install_pipeline_linear_cache_layers,
)
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
    Layer indices whose cache slots use pipeline snapshot rewind after rejection.

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
    """Deep copy base decoder config; inference keeps the same attn as the target LLM."""
    c = copy.deepcopy(config)
    return c


@contextmanager
def _force_sdpa_attn_context(config):
    """Training-only: 4D custom mask requires SDPA (FA2 cannot consume it)."""
    saved: Dict[str, Any] = {}
    for name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        if hasattr(config, name):
            saved[name] = getattr(config, name)
            setattr(config, name, "sdpa")
    if "_attn_implementation" not in saved:
        raise AttributeError("config is missing required attribute _attn_implementation")
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


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

def _build_pipeline_training_mask_v11(
    attention_mask_ms: torch.Tensor,
    *,
    n: int,
    m: int,
    s: int,
    aggr_to_min_depth: Sequence[int],
    stage_depth_to_aggr_idx: Sequence[int],
    mask_dtype: torch.dtype,
    simulated_pipeline_fill: int,
) -> torch.Tensor:
    """
    ``[B, m*S]`` padding mask -> ``[B, 1, m*S, m*S]`` additive mask.

    Layout (stage-major): blocks ``b=0..m-1`` are ``g_{m-1}, ..., g_0``; each block has time ``0..S-1``.
    Query at block ``bq``, position ``t`` uses representative depth ``d_q = aggr_to_min_depth[m-1-bq]``.
    Key at block ``bk``, position ``T`` is visible iff ``T<=t`` and the required key block index
    ``stage_depth_to_aggr_idx[d']`` equals ``m-1-bk``, where ``d' = min(n, d_q + t - T)``.

    When ``simulated_pipeline_fill < n`` (partial pipeline), keys at distance ``t-T >= fill`` must
  attend the deepest block ``g_m`` (aggr index ``m-1``) instead of ``g(d')`` — unlike v10 we only
    change the mask (v11 may map multiple depths to one block).
    """
    b, lseq = attention_mask_ms.shape
    m_i = int(m)
    if lseq != m_i * s:
        raise ValueError(f"expected length m*S={m_i * s}, got {lseq}")
    device = attention_mask_ms.device
    neg = torch.finfo(mask_dtype).min

    pos_list: List[int] = []
    block_list: List[int] = []
    for block in range(m_i):
        for t in range(s):
            pos_list.append(t)
            block_list.append(block)
    pos = torch.tensor(pos_list, device=device, dtype=torch.long)
    blocks = torch.tensor(block_list, device=device, dtype=torch.long)

    aggr_to_min = torch.tensor(list(aggr_to_min_depth), device=device, dtype=torch.long)
    depth_to_aggr = torch.tensor(list(stage_depth_to_aggr_idx), device=device, dtype=torch.long)
    n_t = torch.tensor(int(n), device=device, dtype=torch.long)

    aggr_i_flat = m_i - 1 - blocks
    dq_flat = aggr_to_min[aggr_i_flat]
    dq = dq_flat.view(1, 1, lseq, 1)
    pq = pos.view(1, 1, lseq, 1)
    pk = pos.view(1, 1, 1, lseq)
    aggr_k = aggr_i_flat.view(1, 1, 1, lseq)

    delta = pq - pk
    d_prime = torch.minimum(n_t, dq + delta)
    d_prime = d_prime.clamp(min=0)
    k_aggr_required = depth_to_aggr[d_prime]
    fill = int(simulated_pipeline_fill)
    if fill < int(n):
        fill_t = torch.tensor(fill, device=device, dtype=torch.long)
        deepest_aggr = torch.tensor(m_i - 1, device=device, dtype=torch.long)
        k_aggr_required = torch.where(delta >= fill_t, deepest_aggr, k_aggr_required)
    structural = (pk <= pq) & (aggr_k == k_aggr_required)

    v = attention_mask_ms.to(dtype=torch.bool)
    q_valid = v.view(b, 1, lseq, 1)
    k_valid = v.view(b, 1, 1, lseq)
    return torch.where(structural & q_valid & k_valid, torch.zeros((), device=device, dtype=mask_dtype), neg)


class SpeculationTransformerModuleV11(nn.Module):
    """
    Speculation transformer (v11): ``m`` aggregation types via one input FC each; training layout
    ``[g_{m-1}, ..., g_0]`` with all rows as Q/K/V; inference uses standard decoder + KV cache.
    """

    def __init__(
        self,
        config,
        dtype: torch.dtype,
        device: torch.device,
        base_rotary_emb: nn.Module,
        apply_rotary_fn,
        aggr_feature_indices: Sequence[Sequence[int]],
        num_aggr_types: int,
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
        self.num_aggr_types = int(num_aggr_types)
        if self.num_spec_layers < 1:
            raise ValueError(f"num_spec_layers must be >= 1, got {num_spec_layers}")
        if self.num_aggr_types < 1:
            raise ValueError(f"num_aggr_types must be >= 1, got {num_aggr_types}")
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

        self.aggr_feature_indices = [tuple(int(x) for x in row) for row in aggr_feature_indices]
        if len(self.aggr_feature_indices) != self.num_aggr_types:
            raise ValueError(
                f"aggr_feature_indices length {len(self.aggr_feature_indices)} != num_aggr_types {self.num_aggr_types}"
            )
        h = int(config.hidden_size)
        self.aggr_projs = nn.ModuleList(
            [
                nn.Linear(len(row) * h, h, bias=False, device=device, dtype=dtype)
                for row in self.aggr_feature_indices
            ]
        )
        for proj in self.aggr_projs:
            nn.init.normal_(proj.weight, std=0.02)

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

    def _decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        *,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        _, q_len, _ = hidden_states.shape
        device = hidden_states.device
        layer_past = past_key_values if use_cache else None
        past_seen = past_key_values.get_seq_length() if layer_past is not None else 0
        cache_position = torch.arange(past_seen, past_seen + q_len, device=device, dtype=torch.long)
        if attention_mask is None and use_cache:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": layer_past,
                "position_ids": position_ids,
            }
            attn_mask = create_causal_mask(**mask_kwargs)
        else:
            attn_mask = attention_mask

        out = hidden_states
        for layer in self.spec_layers:
            position_embeddings = self.rotary_emb(out, position_ids)
            out = layer(
                out,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=layer_past,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
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
        del stage_ids
        if hidden_states.shape[-1] != int(self.config.hidden_size):
            raise ValueError(
                f"hidden_states last dim must be hidden_size={self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        out = self._decoder_forward(
            hidden_states,
            position_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return out[:, -1:, :]

    def forward_inference_g1_only_with_rotary(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_inference_with_rotary(*args, **kwargs)

    def forward_training_with_rotary(
        self,
        hidden_states_ms: torch.Tensor,
        position_ids_ms: torch.LongTensor,
        attention_mask_ms: torch.Tensor,
        *,
        num_aggr_types: int,
    ) -> torch.Tensor:
        bsz, ms_len, hidden = hidden_states_ms.shape
        if hidden != int(self.config.hidden_size):
            raise ValueError(
                f"hidden_states last dim must be hidden_size={self.config.hidden_size}, got {hidden_states_ms.shape[-1]}"
            )
        m = int(num_aggr_types)
        if ms_len % m != 0:
            raise ValueError(f"m*S must divide sequence length; got mS={ms_len}, m={m}")
        s = ms_len // m
        if attention_mask_ms.shape != (bsz, 1, ms_len, ms_len):
            raise ValueError(
                f"attention_mask_ms must be [B,1,m*S,m*S], got {tuple(attention_mask_ms.shape)}"
            )

        with _force_sdpa_attn_context(self.config):
            out = self._decoder_forward(
                hidden_states_ms,
                position_ids_ms,
                attention_mask_ms,
                past_key_values=None,
                use_cache=False,
            )
        return out[:, (m - 1) * s :, :]

    def forward_with_rotary(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_inference_with_rotary(*args, **kwargs)


def stage_layers_from_spec_cfg(spec_cfg: dict[str, Any]) -> Optional[List[int]]:
    """Parse optional per-stage layer counts from a speculation-head checkpoint config."""
    raw = spec_cfg.get("stage_layers")
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        if not parts:
            return None
        return [int(p) for p in parts]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise ValueError(f"stage_layers must be a semicolon-separated string or list, got {type(raw)!r}")


def resolve_stage_layer_ranges(
    stage_layers: Optional[Sequence[int]],
    *,
    num_stages: int,
    num_layers: int,
) -> List[Tuple[int, int]]:
    """Convert per-stage layer counts (or ``None`` for uniform split) to ``[(lo, hi), ...]``."""
    n_stages = int(num_stages)
    n_layers = int(num_layers)
    if stage_layers is None:
        if n_layers % n_stages != 0:
            raise ValueError(
                f"num_layers ({n_layers}) must be divisible by num_stages ({n_stages}) "
                "when stage_layers is not set"
            )
        lps = n_layers // n_stages
        return [(s * lps, (s + 1) * lps) for s in range(n_stages)]
    counts = [int(x) for x in stage_layers]
    if len(counts) != n_stages:
        raise ValueError(
            f"stage_layers length {len(counts)} must equal num_stages ({n_stages})"
        )
    if sum(counts) != n_layers:
        raise ValueError(
            f"stage_layers sum {sum(counts)} must equal num_layers ({n_layers})"
        )
    ranges: List[Tuple[int, int]] = []
    start = 0
    for count in counts:
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def default_aggr_feature_bound(num_layers: int, num_stages: int) -> List[int]:
    """Default ``aggr_feature_bound`` for ``num_layers`` and ``num_stages`` (see ``结构v11.md``)."""
    n = int(num_stages)
    l = int(num_layers)
    if n < 1:
        raise ValueError(f"num_stages must be >= 1, got {n}")
    if l % n != 0:
        raise ValueError(f"num_layers ({l}) must be divisible by num_stages ({n})")
    lps = l // n
    raw = [0, lps, 3 * lps, 6 * lps, l - 1]
    return sorted({max(0, min(l, int(x))) for x in raw})


class Qwen3PipelineModelV11(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        num_stages: int = 2,
        num_spec_layers: int = 1,
        spec_init_from_base_layers: Optional[Sequence[int]] = None,
        draft_token_ids: Optional[Sequence[int]] = None,
        aggr_feature_bound: Optional[Sequence[int]] = None,
        trained_with_use_deepest: bool = False,
        stage_layers: Optional[Sequence[int]] = None,
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
        self.stage_layer_ranges = resolve_stage_layer_ranges(
            list(stage_layers) if stage_layers is not None else None,
            num_stages=self.num_stages,
            num_layers=self.num_layers,
        )
        stage_sizes = [hi - lo for lo, hi in self.stage_layer_ranges]
        if len(set(stage_sizes)) == 1:
            self.layers_per_stage = stage_sizes[0]
        else:
            self.layers_per_stage = max(stage_sizes)
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

        self.aggr_feature_bound = self._normalize_aggr_feature_bound(aggr_feature_bound)
        self.num_aggr_types = len(self.aggr_feature_bound)
        self.aggr_feature_indices = [
            tuple(self.aggr_feature_bound[: i + 1]) for i in range(self.num_aggr_types)
        ]
        self.stage_depth_to_aggr_idx = [self._depth_to_aggr_idx(d) for d in range(self.num_stages + 1)]
        self.aggr_to_min_depth = self._compute_aggr_to_min_depth()

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
        self.speculation_module = SpeculationTransformerModuleV11(
            spec_cfg,
            self.dtype,
            self.device,
            base_rotary_emb=self.base_model.model.rotary_emb,
            apply_rotary_fn=_get_apply_rotary_pos_emb(self.config),
            aggr_feature_indices=self.aggr_feature_indices,
            num_aggr_types=self.num_aggr_types,
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
    def speculation_head(self) -> SpeculationTransformerModuleV11:
        return self.speculation_module

    def _normalize_aggr_feature_bound(
        self,
        aggr_feature_bound: Optional[Sequence[int]],
    ) -> List[int]:
        if aggr_feature_bound is None:
            bounds = default_aggr_feature_bound(self.num_layers, self.num_stages)
        else:
            bounds = [int(x) for x in aggr_feature_bound]
        if not bounds:
            raise ValueError("aggr_feature_bound must be non-empty.")
        if bounds[0] != 0:
            raise ValueError(f"aggr_feature_bound must start with 0, got {bounds[0]}")
        max_hf = self.num_layers
        out: List[int] = []
        prev = -1
        for j in bounds:
            if j < 0 or j > max_hf:
                raise ValueError(
                    f"aggr_feature_bound entry {j} out of range [0, {max_hf}]"
                )
            if j <= prev:
                raise ValueError(f"aggr_feature_bound must be strictly increasing, got {bounds}")
            out.append(int(j))
            prev = j
        if len(out) > self.num_stages + 1:
            raise ValueError(
                f"aggr_feature_bound length {len(out)} exceeds num_stages+1={self.num_stages + 1}"
            )
        return out

    def _depth_to_available_hs_index(self, depth: int) -> int:
        return min(self.num_layers, int(depth) * self.layers_per_stage)

    def _depth_to_aggr_idx(self, depth: int) -> int:
        avail = self._depth_to_available_hs_index(depth)
        idx = 0
        for i, bound in enumerate(self.aggr_feature_bound):
            if int(bound) <= avail:
                idx = i
        return idx

    def _compute_aggr_to_min_depth(self) -> List[int]:
        m = self.num_aggr_types
        out = [self.num_stages + 1] * m
        for d in range(self.num_stages + 1):
            ai = self.stage_depth_to_aggr_idx[d]
            if d < out[ai]:
                out[ai] = d
        return out

    def _snap_indices_needed(self) -> Set[int]:
        return set(int(x) for x in self.aggr_feature_bound)

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

    def _normalize_simulated_pipeline_fill(self, simulated_pipeline_fill: Optional[int]) -> int:
        n = self.num_stages
        if simulated_pipeline_fill is None:
            return n
        fill = int(simulated_pipeline_fill)
        if fill < 1 or fill > n:
            raise ValueError(f"simulated_pipeline_fill must be in [1, {n}], got {fill}")
        return fill

    def _build_training_expanded_inputs(
        self,
        all_hs: Tuple[torch.Tensor, ...],
        simulated_pipeline_fill: Optional[int] = None,
    ) -> torch.Tensor:
        """
        v11 training layout (``m`` blocks of length ``S``), stage-major: ``[g_{m-1}, ..., g_0]``.
        Each block uses ``aggr_projs[i]`` over ``aggr_feature_indices[i]`` from teacher hidden states.

        ``simulated_pipeline_fill`` only affects the training attention mask (see
        ``_build_pipeline_training_mask_v11``); tensor layout is unchanged for all fill values.
        """
        m = self.num_aggr_types
        rows: List[torch.Tensor] = []
        for aggr_i in range(m - 1, -1, -1):
            rows.append(
                self._fuse_from_hf_indices(
                    all_hs,
                    self.aggr_feature_indices[aggr_i],
                    self.speculation_module.aggr_projs[aggr_i],
                )
            )
        return torch.cat(rows, dim=1)

    def _build_inference_row_from_snap(
        self,
        snap: Dict[int, torch.Tensor],
        depth: int,
    ) -> torch.Tensor:
        aggr_i = self.stage_depth_to_aggr_idx[int(depth)]
        hf_indices = self.aggr_feature_indices[aggr_i]
        proj = self.speculation_module.aggr_projs[aggr_i]
        vecs = [snap[int(idx)] for idx in hf_indices]
        return proj(torch.cat(vecs, dim=-1))

    def _build_inference_g0_row_from_hs(self, hs: torch.Tensor) -> torch.Tensor:
        return self._build_inference_row_from_snap({0: hs}, depth=0)

    def _choose_inference_depth_for_snap(
        self,
        snap: Dict[int, torch.Tensor],
        nominal_depth: int,
        use_deepest: bool,
        *,
        search_hi: Optional[int] = None,
    ) -> int:
        n = self.num_stages
        nd = int(nominal_depth)
        if nd <= 0:
            return nd
        if not use_deepest:
            hi = nd
        else:
            hi = int(search_hi) if search_hi is not None else n
            if hi > n:
                hi = n
        for d in range(hi, -1, -1):
            aggr_i = self.stage_depth_to_aggr_idx[d]
            hf_indices = self.aggr_feature_indices[aggr_i]
            if all(int(idx) in snap for idx in hf_indices):
                return d
        return nd

    def _choose_inference_i_stages_for_snap(
        self,
        snap: Dict[int, torch.Tensor],
        i_nominal: int,
        use_deepest: bool,
        *,
        search_hi: Optional[int] = None,
    ) -> int:
        return self._choose_inference_depth_for_snap(
            snap, i_nominal, use_deepest, search_hi=search_hi
        )

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
        m = self.num_aggr_types
        fill = self._normalize_simulated_pipeline_fill(simulated_pipeline_fill)
        if attention_mask is None:
            attn2d = torch.ones((b, s), device=input_ids.device, dtype=torch.long)
        else:
            attn2d = attention_mask.to(device=input_ids.device, dtype=torch.long)

        spec_hidden = self._build_training_expanded_inputs(
            all_hs,
            simulated_pipeline_fill=fill,
        )
        pos_ms = torch.arange(s, device=spec_hidden.device, dtype=torch.long).repeat(m).unsqueeze(0).expand(b, -1)
        attn_ms = attn2d.repeat(1, m)
        mask_4d = _build_pipeline_training_mask_v11(
            attn_ms,
            n=n,
            m=m,
            s=s,
            aggr_to_min_depth=self.aggr_to_min_depth,
            stage_depth_to_aggr_idx=self.stage_depth_to_aggr_idx,
            mask_dtype=self.dtype,
            simulated_pipeline_fill=fill,
        )

        g0_processed = self.speculation_module.forward_training_with_rotary(
            spec_hidden,
            pos_ms,
            mask_4d,
            num_aggr_types=m,
        )
        g0_processed = self.final_norm(g0_processed)
        spec_logits = self.speculation_module.lm_head(g0_processed)

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

    def _rollback_kv_cache_after_rejection(
        self,
        past_kv: Cache,
        target_length: int,
        linear_layer_indices: Sequence[int],
        layers_per_stage: int,
        num_stages: int,
    ) -> None:
        """Roll back target-model cache on draft rejection (KV crop + linear snapshot rewind)."""
        crop_pipeline_cache_after_rejection(
            past_kv,
            target_length,
            linear_layer_indices=linear_layer_indices,
            layers_per_stage=layers_per_stage,
            num_stages=num_stages,
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

        outputs = self.base_model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv: Cache = outputs.past_key_values
        install_pipeline_linear_cache_layers(past_kv, linear_cache_layer_indices, n)
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
        install_pipeline_linear_cache_layers(base_past_kv, linear_cache_layer_indices, n)
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
            "num_aggr_types": self.num_aggr_types,
            "version": 11,
            "trained_with_use_deepest": bool(self.trained_with_use_deepest),
            "aggr_feature_bound": list(self.aggr_feature_bound),
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
            if isinstance(cfg, dict):
                if int(cfg.get("version", 0) or 0) not in (0, 11):
                    raise ValueError(
                        f"Checkpoint version {cfg.get('version')} is not compatible with Qwen3PipelineModelV11."
                    )
                if "trained_with_use_deepest" in cfg:
                    self.trained_with_use_deepest = bool(cfg["trained_with_use_deepest"])
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = _materialize_state_dict_for_load(state, map_location)
        self.speculation_module.load_state_dict(state)


# Backward-compatible names used by train / inference scripts in this package.
Qwen3SpeculativePipelineModel = Qwen3PipelineModelV11
SpeculationHeadTransformer = SpeculationTransformerModuleV11
