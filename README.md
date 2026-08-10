# Speculative Pipeline Decoding

Implementation of [**Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism**](https://arxiv.org/abs/2605.30852). SPD partitions the target LLM into $n$ pipeline stages so that $n$ tokens of a single sequence advance at different depths in parallel. A **Pipeline Draft Module (PDM)** aggregates multi-depth target hidden states to predict one next token per cycle and runs concurrently with each pipeline step, yielding bounded draft difficulty, higher acceptance, and hidden draft latency. Compatible with Qwen3, Qwen3.5, Llama3.1, etc.; drafts are verified against the base model for lossless generation.

> **Important**
>
> The **single-process** scripts (`pipeline_inference.py`, `eval.py`) are intended to **demonstrate algorithmic correctness** and report theoretical speedup from per-stage GPU timers. They are not tuned for production and can be **slower than standard autoregressive decoding** on one GPU because of Python-side sequential overhead.
>
> For **real multi-GPU wall-clock benchmarks** (prefill/decode tok/s, acceptance on bundled datasets), use [`distributed_inference/`](distributed_inference/) with `torchrun` — see [Distributed inference](#distributed-inference).

## Method overview

![Speculative Pipeline Decoding (n=4 stages)](method.png)

**Speculative Pipeline Decoding with $n{=}4$ stages** (figure above). The target LLM is partitioned into four stages. At the beginning of a saturated cycle, in-flight tokens $x_{t-3},\ldots,x_t$ occupy the pipeline at varying depths, while $x_1,\ldots,x_{t-4}$ are already fully processed. For each token, hidden states at a fixed set of layer checkpoints (selected by how far that token has progressed) are concatenated and projected into one aggregated feature for the **Pipeline Draft Module (PDM)**. The PDM drafts $\hat{x}_{t+1}$ concurrently with the target pipeline forward. After the forward, the oldest in-flight token $x_{t-3}$ exits and yields logits that verify $x_{t-2}$; if accepted, $\hat{x}_{t+1}$ enters stage 1 for the next cycle.

Two design choices distinguish SPD from multi-token drafting (e.g., EAGLE-3) and prior pipeline drafting (PPSD):

1. **Multi-depth feature aggregation.** The PDM conditions only on target-LLM hidden states aggregated across pipeline depths—including partial states of in-flight tokens—so prediction stays in the target's feature space and draft difficulty is bounded by the fixed width $n$.
2. **Pre-step, overlapped drafting.** The PDM uses features available *before* the current pipeline forward and runs in parallel with stage compute on a dedicated rank, hiding draft latency behind each pipeline step.

## Repository layout

```
speculative_pipeline_decoding/
├── pipeline_model.py           # Qwen3SpeculativePipelineModel + speculation head (v11)
├── train.py                    # Train the speculation head (v11)
├── pipeline_inference.py       # Load checkpoint, run pipeline / HF generate (single GPU)
├── eval.py                     # Benchmark on bundled eval sets (theoretical speedup)
├── distributed_inference/      # Multi-GPU torch.distributed inference (real wall-clock)
├── old_version_v10/            # Archived v10 implementation (see its README)
├── eval_data/                  # MT-Bench, HumanEval, GSM8K prompts (EAGLE jsonl format)
├── draft_vocab/                # Pre-built draft vocabularies (token id subsets)
├── BENCHMARK_RESULTS.md        # Wall-clock speedup tables (default + Open-PerfectBlend)
├── requirements.txt
└── README.md
```

**Further reading**

| Topic | Location |
| --- | --- |
| Benchmark tables (Eagle3, PPSD, ours; default & Open-PerfectBlend training) | [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) |
| v10 archived code, v10 vs v11 differences, v10 checkpoints | [old_version_v10/README.md](old_version_v10/README.md) |
| Multi-GPU distributed inference | [distributed_inference/README.md](distributed_inference/README.md) |

Current release is **v11** (`config['version'] == 11`). The paper implementation is **v10** in [`old_version_v10/`](old_version_v10/).

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

Alternative training on [openeurollm/open-perfectblend-decontaminated](https://huggingface.co/datasets/openeurollm/open-perfectblend-decontaminated) is documented in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#open-perfectblend-decontaminated-training).

## Training

Example of training the **Pipeline Draft Module (PDM)**:

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

## Checkpoints

Pre-trained v11 checkpoints: [Hugging Face — speculative_pipeline_decoding](https://huggingface.co/yuyijiong/speculative_pipeline_decoding)

Wall-clock numbers and additional checkpoint links (Open-PerfectBlend training, Eagle3 baselines): [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)

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

For published speedup tables, see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## Distributed inference

Multi-GPU pipeline-parallel decoding via `torch.distributed` lives under [`distributed_inference/`](distributed_inference/). It shards the target model across ranks, overlaps stage forwards with speculation, and reports **wall-clock** prefill/decode time — use this path to measure end-to-end throughput, not the theoretical metrics from `eval.py`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/example_mp_pipeline_generate.py \
  --spec_head_ckpt Qwen3.5-4B_s4_l4.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --rank_gpus 0,1,2,3,4 \
  --max_new_tokens 512 \
  --temperature 0.0
```

`--nproc_per_node` must equal `num_stages + 1` from the checkpoint. Dataset eval with the same prompts as `eval.py`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt Qwen3.5-4B_s4_l4.pt \
  --data_dir eval_data \
  --output_dir ./eval_output_distributed \
  --rank_gpus 0,1,2,3,4
```

See [`distributed_inference/README.md`](distributed_inference/README.md) for rank layout, timing breakdown fields, and multi-node launch.

## Citation

If you use this repo, please cite our paper:

```bibtex
@misc{yu2026speculativepipelinedecodinghigheraccruacy,
      title={Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism}, 
      author={Yijiong Yu and Huazheng Wang and Shuai Yuan and Ruilong Ren and Ji Pei},
      year={2026},
      eprint={2605.30852},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.30852}, 
}
```
