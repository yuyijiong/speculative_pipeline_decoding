"""Re-export single-process pipeline helpers used by distributed rank 0."""

from __future__ import annotations

from ._paths import REPO_ROOT  # noqa: F401

from pipeline_model import (  # noqa: E402,F401
    Qwen3PipelineModelV11,
    Qwen3PipelineModelV11 as Qwen3SpeculativePipelineModel,
    SpeculationTransformerModuleV11 as SpeculationHeadTransformer,
    _decoder_relevant_config,
    _linear_and_hybrid_attention_layer_indices_for_cache,
    _sampling_probs_hf_style,
    _verify_pipeline_draft_token,
    default_aggr_feature_bound,
    num_hidden_layers_from_hf_config,
    resolve_stage_layer_ranges,
    stage_layers_from_spec_cfg,
)
