# Distributed Pipeline Inference

Multi-GPU **layer-wise pipeline parallel** inference for speculative pipeline decoding. Uses `torch.distributed` (NCCL) with one rank per pipeline stage plus a dedicated speculation rank on rank 0. This path measures **real wall-clock** prefill/decode time across devices (unlike single-process `eval.py`, which reports theoretical speedup from per-stage GPU timers).

## Requirements

- Linux, multiple NVIDIA GPUs, NCCL
- Same dependencies as the parent repo (`torch` 2.8+, `transformers`, FlashAttention 2)
- v11 speculation-head checkpoint (`config['version'] == 11`)

## Rank layout

- **Default:** `world_size = num_stages + 1`. Rank 0 runs speculation + embedding + final norm + lm head; ranks `1..num_stages` each hold one pipeline stage shard.
- **`--merge_last_stage`:** `world_size = num_stages`. Rank 0 also runs the last pipeline stage (CUDA streams); ranks `1..num_stages-1` hold earlier stages.

`--rank_gpus` is a comma-separated list of physical GPU ids, length = `world_size`. Rank `i` uses `rank_gpus[i]`. If omitted, each rank uses `LOCAL_RANK` from `torchrun`.

## Quick demo

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/run_generate.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --rank_gpus 0,1,2,3,4 \
  --prompt "Introduce LLM." \
  --max_new_tokens 128 \
  --temperature 0.0
```

With `--merge_last_stage` on 4 stages (4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  distributed_inference/run_generate.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --rank_gpus 0,1,2,3 \
  --merge_last_stage
```

## Benchmark eval (real speed)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/eval_benchmark.py \
  --spec_head_ckpt /path/to/speculation_head_final.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --data_dir eval_data \
  --output_dir ./eval_output_distributed \
  --rank_gpus 0,1,2,3,4 \
  --async_comm
```

Outputs mirror `eval.py` layout under `--output_dir` (`raw/` per-sample jsonl, `summary/` aggregates) with wall-clock decode/prefill timings and acceptance metrics.

## Module layout

| File | Role |
|------|------|
| `run_generate.py` | CLI: single-prompt generate + timing breakdown |
| `eval_benchmark.py` | CLI: MT-Bench / HumanEval / GSM8K eval |
| `loader.py` | Per-rank model shards + spec head |
| `prefill.py` | Prompt prefill + KV shard distribution |
| `decode.py` | Decode loop (rank 0 driver + stage workers) |
| `rank0_controller.py` | Verify, speculation, snap fusion on rank 0 |
| `stage_worker.py` | Stage forward + P2P on worker ranks |
| `comm.py` | Pipeline P2P and control messages |
| `topology.py` | Rank ↔ stage mapping |
| `cache.py` | Per-stage KV cache shards |

## Notes

- All ranks must `barrier` after model load before decoding; otherwise P2P can hang.
- `--async_comm` overlaps communication with compute; `--sync_mode comm_only` skips `dist.barrier` per cycle (debug).
- Checkpoints with uneven `stage_layers` in config are supported (non-uniform layer splits per stage).
