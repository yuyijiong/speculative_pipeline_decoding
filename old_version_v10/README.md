# v10 (archived)

This folder keeps the **v10** pipeline speculation implementation used in the [paper](https://arxiv.org/abs/2605.30852). The top-level scripts (`train.py`, `pipeline_model.py`, …) implement **v11**, which is the recommended path for new training and checkpoints.

## v11 vs. v10 (key differences)

| Topic | v10 (this folder) | v11 (top-level scripts) |
|-------|-------------------|-------------------------|
| Aggregation config | `shallow_hidden_layer_indices`: `n` semicolon-separated groups for `g_n … g_1` (per pipeline-stage depth) | `aggr_feature_bound`: `m` HF hidden-state anchor indices for `g_0 … g_{m-1}` |
| Training layout | `(n+1)·S` tokens: `[g_n, g_{n-1}, …, g_1, g_0]` | `m·S` tokens: `[g_{m-1}, …, g_0]` (`num_aggr_types` = `m`) |
| Attention roles | Only `g_0` rows act as **query**; other rows are key/value | **All** `m` aggregation rows are query, key, and value |
| `g_0` source | Token embedding only, via a dedicated `g0_proj` FC | One of `m` aggregation types; each `g_k` is an FC over selected hidden states (embedding for `g_0`) |
| Output head | `lm_head` on `g_0` query output (same idea) | `lm_head` on the `g_0` block output only |
| Inference schedule | Speculation runs in parallel with one pipeline step; features from pipeline **input** depths | Same parallel schedule; row `g_k` at pipeline depth `d` uses anchor `g_{f(d), d}` from `aggr_feature_bound` |

Checkpoints store `config['version'] == 10` or `11` respectively. Weights are **not** interchangeable between versions.

## Checkpoints

- **v10:** [`v10` branch on Hugging Face](https://huggingface.co/yuyijiong/speculative_pipeline_decoding/tree/v10)
- **v11:** [main Hugging Face repo](https://huggingface.co/yuyijiong/speculative_pipeline_decoding)

## Usage

Run scripts from the parent `speculative_pipeline_decoding/` directory so `pipeline_linear_cache` resolves correctly:

```bash
cd speculative_pipeline_decoding
python old_version_v10/train.py ...
python old_version_v10/pipeline_inference.py ...
```

v10 checkpoints can still be loaded by the main `pipeline_inference.py` and `eval.py` via dynamic import of this module.
