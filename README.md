# Speculative Pipeline Decoding

Original implementation of [**Speculative Pipeline Decoding: Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism**](https://arxiv.org/abs/2605.30852). This is a novel speculative decoding paradigm, expected to address the issues of increasing difficulty and latency bubbles in traditional SD. Compatible with Qwen3, Qwen3.5, Llama3.1, etc. The target model runs in a multi-stage pipeline while a lightweight speculation head drafts tokens in parallel; drafts are verified against the base model for lossless generation. This paradigm is totally different from the traditional speculative decoding, and achieves higher acceptance rate and zero latency bubble.

> **Important**
>
> The **single-process** scripts (`pipeline_inference.py`, `eval.py`) are intended to **demonstrate algorithmic correctness** and report theoretical speedup from per-stage GPU timers. They are not tuned for production and can be **slower than standard autoregressive decoding** on one GPU because of Python-side sequential overhead.
>
> For **real multi-GPU wall-clock benchmarks** (prefill/decode tok/s, acceptance on bundled datasets), use [`distributed_inference/`](distributed_inference/) with `torchrun` — see [Distributed inference](#distributed-inference-real-speed-benchmarks).

## Method overview
![Method](method.png)
The architecture of Speculative Pipeline Decoding when the number of stages is 3. The target LLM is partitioned into 3 stages. At the start point of this round, tokens (e.g., $x_5$ to $x_7$) reside in the pipeline at varying depths while others (e.g., $x_1$ to $x_4$) are fully processed tokens. For each token, hidden states from passed stages are projected via FC layers to form an aggregated feature, serving as the input to the Pipeline Speculation Module. The Speculation Module speculates the next token ($x_8$) simultaneously with the target LLM's pipeline forward step. Then $x_8$'s token embedding is added to the pipeline for the next round, while the target LLM verifies the oldest token in the pipeline ($x_6$) based on ground-truth output logits of the token $x_5$ that is just popped out of the pipeline.

## Repository layout

```
speculative_pipeline_decoding/
├── pipeline_model.py           # Qwen3SpeculativePipelineModel + speculation head
├── train.py                    # Train the speculation head
├── pipeline_inference.py       # Load checkpoint, run pipeline / HF generate (single GPU)
├── eval.py                     # Benchmark on bundled eval sets (theoretical speedup)
├── distributed_inference/      # Multi-GPU torch.distributed inference (real wall-clock)
├── old_version_v10/            # Archived earlier implementation
├── eval_data/                  # MT-Bench, HumanEval, GSM8K prompts (EAGLE jsonl format)
├── draft_vocab/                # Pre-built draft vocabularies (token id subsets)
├── requirements.txt
└── README.md
```

## v11 vs. v10 (key differences)

| Topic | v10 (paper / `old_version_v10/`) | v11 (`main` / top-level scripts) |
|-------|----------------------------------|----------------------------------|
| Aggregation config | `shallow_hidden_layer_indices`: `n` semicolon-separated groups for `g_n … g_1` (per pipeline-stage depth) | `aggr_feature_bound`: `m` HF hidden-state anchor indices for `g_0 … g_{m-1}` |
| Training layout | `(n+1)·S` tokens: `[g_n, g_{n-1}, …, g_1, g_0]` | `m·S` tokens: `[g_{m-1}, …, g_0]` (`num_aggr_types` = `m`) |
| Attention roles | Only `g_0` rows act as **query**; other rows are key/value | **All** `m` aggregation rows are query, key, and value |
| `g_0` source | Token embedding only, via a dedicated `g0_proj` FC | One of `m` aggregation types; each `g_k` is an FC over selected hidden states (embedding for `g_0`) |
| Output head | `lm_head` on `g_0` query output (same idea) | `lm_head` on the `g_0` block output only |
| Inference schedule | Speculation runs in parallel with one pipeline step; features from pipeline **input** depths | Same parallel schedule; row `g_k` at pipeline depth `d` uses anchor `g_{f(d), d}` from `aggr_feature_bound` |

Checkpoints store `config['version'] == 11` and are compatible with weights trained by this release's `train.py`. v10 checkpoints (`config['version'] == 10`) are **not** interchangeable with v11 weights — load them via `old_version_v10/` or the [`v10`](https://huggingface.co/yuyijiong/speculative_pipeline_decoding/tree/v10) branch on Hugging Face.

## Requirements

- Linux with NVIDIA GPU (CUDA)
- Python 3.10+
- PyTorch 2.8+, `transformers` (Qwen3 / Qwen3.5 support), FlashAttention 2

## Draft vocabulary

The `draft_vocab/` directory ships JSON files that list a **draft token subset** (`token_ids`) and `draft_vocab_size`. When passed via `--draft_vocab_json`, the speculation module’s `lm_head` output dimension is restricted to that subset (for smaller, faster draft logits).

| File | Base tokenizer | Draft size |
|------|----------------|------------|
| `draft_vocab/ultrachat_qwen3.5_4b_top_50k.json` | Qwen3.5 series | 50k |
| `draft_vocab/ultrachat_qwen3.5_4b_top_32k.json` | Qwen3.5 series | 32k |
| `draft_vocab/ultrachat_qwen3_0.6b_top_32k.json` | Qwen3 series | 32k |

These vocabularies were built from the training corpora listed in each file’s metadata (Ultrachat-200k, ShareGPT, SmolTalk, etc.). Pick the file that matches your `--model_name`. Omit `--draft_vocab_json` (or pass an empty string) to use the full base-model vocabulary.

## Training data

Training datasets are:

- [Ultrachat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [SmolTalk-Chinese](https://huggingface.co/datasets/opencsg/smoltalk-chinese)

## Training
Example of training the **Pipeline Speculation Module**:

```bash
accelerate launch --num_processes 8 train.py \
  --model_name Qwen/Qwen3.5-4B \
  --data_path /path/to/train.parquet \
  --draft_vocab_json draft_vocab/ultrachat_qwen3.5_4b_top_50k.json \
  --num_stages 8 \
  --num_spec_layers 4 \
  --output_dir ./training_output/my_run
```

Important flags:

| Flag | Description |
|------|-------------|
| `--num_stages` | Pipeline depth `n` (target stages) |
| `--num_spec_layers` | Transformer layers in the speculation module |
| `--aggr_feature_bound` | Comma-separated HF hidden-state anchors for `g_0..g_{m-1}` (`auto` for default) |
| `--draft_vocab_json` | Path to a draft vocabulary JSON under `draft_vocab/` (see above). Empty string = full base vocabulary. |


Output: `speculation_head_final.pt` under `--output_dir` (includes `state_dict` and `config` with `base_model_path`, `num_stages`, etc.).


## Our trained checkpoints
See [HF](https://huggingface.co/yuyijiong/speculative_pipeline_decoding)


## Quick demo (pipeline vs standard generate)

```bash
python pipeline_inference.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --max_new_tokens 100 \
  --temperature 0.0
```

If the checkpoint’s `config["base_model_path"]` already points to a valid local path or Hugging Face id on your machine, you can omit `--base_model_path`.

Use `--temperature 1.0` for stochastic decoding, `--draft_top_k 4` for draft-tree top-k, and `--no-verify` only for debugging (not lossless).

## Evaluation

Bundled prompts live under `eval_data/`:

- `eval_data/mt_bench/question.jsonl`
- `eval_data/humaneval/question.jsonl`
- `eval_data/gsm8k/question.jsonl`

Each line is EAGLE-style: `{"question_id": ..., "turns": ["user prompt", ...]}`; only the **first** turn is used.

```bash
python eval.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --data_dir eval_data \
  --output_dir ./eval_output \
  --gpus 0 \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --draft_top_k 4
```

Results:

- `eval_output/raw/pipeline_eval__*__per_sample.jsonl` — per-sample metrics
- `eval_output/summary/pipeline_eval__*__summary.json` — aggregates (acceptance rate, theoretical speedup)

Multi-GPU: `--gpus 0,1,2,3`. Optional baseline cache: `--baseline --baseline_cache_dir ./eval_output/baseline`.

## Distributed inference (real speed benchmarks)

Multi-GPU pipeline-parallel decoding via `torch.distributed` lives under [`distributed_inference/`](distributed_inference/). It shards the target model across ranks, overlaps stage forwards with speculation, and reports **wall-clock** prefill/decode time — use this path to measure end-to-end throughput, not the theoretical metrics from `eval.py`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/run_generate.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --rank_gpus 0,1,2,3,4 \
  --max_new_tokens 512 \
  --temperature 0.0
```

`--nproc_per_node` must equal `num_stages + 1` from the checkpoint (or `num_stages` with `--merge_last_stage`). Dataset eval with the same prompts as `eval.py`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/eval_benchmark.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --data_dir eval_data \
  --output_dir ./eval_output_distributed \
  --rank_gpus 0,1,2,3,4
```

See [`distributed_inference/README.md`](distributed_inference/README.md) for rank layout, uneven `stage_layers` checkpoints, and timing breakdown fields.

## Citation

If you use this repo, please cite our paper:

```bibtex
@misc{yu2026speculativepipelinedecodinghigheraccruacy,
      title={Speculative Pipeline Decoding: Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism}, 
      author={Yijiong Yu and Huazheng Wang and Shuai Yuan and Ruilong Ren and Ji Pei},
      year={2026},
      eprint={2605.30852},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.30852}, 
}
```
