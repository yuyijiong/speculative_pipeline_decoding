# Speculative Pipeline Decoding

Official implementation of **pipelined speculative decoding**. Compatible with Qwen3, Qwen3.5, Llama3.1, etc. The target model runs in a multi-stage pipeline while a lightweight speculation head drafts tokens in parallel; drafts are verified against the base model for lossless generation. This paradigm is totally different from the traditional speculative decoding, and achieves higher acceptance rate and zero latency bubble.

> **Important**
>
> This repository is intended to **demonstrate algorithmic correctness** only. It has **not** been tuned for production performance: there is no system-level optimization, no integration with dedicated inference engines (e.g. vLLM, sglang), and the reference implementation still contains many **sequential** steps on the Python side. As a result, **wall-clock latency can be higher than standard autoregressive decoding** in this codebase, even when acceptance rates look favorable. Reported speedup metrics in `eval.py` is theoretical; treat measured end-to-end time here as a correctness baseline, not a deployment benchmark.

## Repository layout

```
speculative_pipeline_decoding/
├── pipeline_model.py       # Qwen3SpeculativePipelineModel + speculation head
├── train.py                # Train the speculation head
├── pipeline_inference.py   # Load checkpoint, run pipeline / HF generate
├── eval.py                 # Benchmark on bundled eval sets
├── eval_data/              # MT-Bench, HumanEval, GSM8K prompts (EAGLE jsonl format)
├── draft_vocab/            # Pre-built draft vocabularies (token id subsets)
├── requirements.txt
└── README.md
```

Checkpoints store `config['version'] == 10` and are compatible with weights trained by this release’s `train.py`.

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
| `--num_spec_layers` | Transformer Layers in the speculation module |
| `--draft_vocab_json` | Path to a draft vocabulary JSON under `draft_vocab/` (see above). Empty string = full base vocabulary. |


Output: `speculation_head_final.pt` under `--output_dir` (includes `state_dict` and `config` with `base_model_path`, `num_stages`, etc.).

## Quick demo (pipeline vs standard generate)

```bash
python pipeline_inference.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --max_new_tokens 100 \
  --temperature 0.0
```

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

## Citation

If you use this code, please cite our paper:

```bibtex
% TODO: add BibTeX when available
```

## License

Please refer to the license file in the repository root (to be added).
