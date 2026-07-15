"""Re-export v11 single-process pipeline model helpers for multi-process rank0."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modeling_qwen3_pipeline_v11 import (  # noqa: E402,F401
    Qwen3PipelineModelV11 as Qwen3SpeculativePipelineModel,
    SpeculationTransformerModuleV11 as SpeculationHeadTransformer,
    _decoder_relevant_config,
    _linear_and_hybrid_attention_layer_indices_for_cache,
    _sampling_probs_hf_style,
    _verify_pipeline_draft_token,
    default_aggr_feature_bound,
    num_hidden_layers_from_hf_config,
)
