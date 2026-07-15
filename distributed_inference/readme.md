# Distributed Pipeline Inference

Multi-GPU **layer-wise pipeline parallel** inference for speculative pipeline decoding. Uses `torch.distributed` (NCCL) with one rank per pipeline stage plus a dedicated speculation rank on rank 0. This path measures **real wall-clock** prefill/decode time across devices (unlike single-process `eval.py`, which reports theoretical speedup from per-stage GPU timers).

## Requirements

- Linux, multiple NVIDIA GPUs, NCCL
- Same dependencies as the parent repo (`torch` 2.8+, `transformers`, FlashAttention 2)
- v11 speculation-head checkpoint (`config['version'] == 11`)

Published checkpoints use names like `Qwen3.5-4B_s{num_stages}_l{num_spec_layers}.pt` (for example `Qwen3.5-4B_s4_l4.pt`, `Qwen3.5-4B_s8_l4.pt`). See the [Hugging Face collection](https://huggingface.co/yuyijiong/speculative_pipeline_decoding) for downloads.

## Rank layout

`world_size = num_stages + 1`. Rank 0 runs speculation + embedding + final norm + lm head; ranks `1..num_stages` each hold one pipeline stage shard.

`--rank_gpus` is a comma-separated list of physical GPU ids, length = `world_size`. Rank `i` uses `rank_gpus[i]`. If omitted, each rank uses `LOCAL_RANK` from `torchrun`.

## Quick demo

Single node, 4 stages (5 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/example_mp_pipeline_generate.py \
  --spec_head_ckpt Qwen3.5-4B_s4_l4.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --rank_gpus 0,1,2,3,4 \
  --prompt "Introduce LLM." \
  --max_new_tokens 128 \
  --temperature 0.0
```

Single node, 8 stages (9 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --standalone --nproc_per_node=9 \
  distributed_inference/example_mp_pipeline_generate.py \
  --spec_head_ckpt Qwen3.5-4B_s8_l4.pt \
  --rank_gpus 0,1,2,3,4,5,6,7,8
```

If the checkpoint `config["base_model_path"]` already points to a valid Hugging Face id or local path, omit `--base_model_path`.

## Benchmark eval (real speed)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt Qwen3.5-4B_s4_l4.pt \
  --base_model_path Qwen/Qwen3.5-4B \
  --data_dir eval_data \
  --output_dir ./eval_output_distributed \
  --rank_gpus 0,1,2,3,4
```

Multiple checkpoints (different `num_stages`) can be passed to `--spec_head_ckpt`; they are evaluated sequentially. On a single node, mismatched `world_size` triggers an automatic `torchrun` relaunch per checkpoint.

Outputs under `--output_dir`:

- `raw/mp_pipeline_eval__<checkpoint_tag>__nt<total>__per_sample.jsonl`
- `summary/mp_pipeline_eval__<checkpoint_tag>__nt<total>__summary.json`

Reports wall-clock prefill/decode timings, acceptance rate, equivalent accept length, and throughput (same datasets as `eval.py`: MT-Bench, HumanEval, GSM8K under `eval_data/`).

## Module layout

| File | Role |
|------|------|
| `example_mp_pipeline_generate.py` | CLI: single-prompt generate + timing breakdown |
| `eval_mp_pipeline_dataset.py` | CLI: MT-Bench / HumanEval / GSM8K eval |
| `loader.py` | Per-rank model shards + spec head |
| `prefill.py` | Prompt prefill + KV shard distribution |
| `decode.py` | Decode loop (rank 0 driver + stage workers) |
| `rank0_controller.py` | Verify, speculation, snap fusion on rank 0 |
| `stage_worker.py` | Stage forward + P2P on worker ranks |
| `comm.py` | Pipeline P2P and control messages |
| `topology.py` | Rank ↔ stage mapping |
| `cache.py` | Per-stage KV cache shards |
| `cuda_graph_opts.py` | CUDA-graph verify path on rank 0 |

## Implementation notes (v11)

- **Fixed-wire snap protocol**: stage→rank0 snap batches use a fixed on-wire layout (no per-cycle meta tensors for snap indices), with pooled recv buffers reused across decode cycles.
- **Verify CUDA graph**: rank-0 verification can run inside a captured CUDA graph; `stage_input_sync` copies `verify_hs` on the default stream before replay, using stream-scoped sync so in-flight snap `irecv`s are not drained.
- **Cycle sync**: end-of-cycle synchronization is a single `wait_all()` on outstanding P2P ops (no per-cycle `dist.barrier`), keeping NCCL recv streams concurrent with verify work.
- **Timing**: `--profile_timing` in `example_mp_pipeline_generate.py` (on by default) and `eval_mp_pipeline_dataset.py` (off by default) reports per-phase decode breakdown (`recv_snap_sec`, `recv_verify_sec`, `verify_kernel_sec`, `hs_path_sec`, etc.).

## Notes

- All ranks must `barrier` after model load before decoding; otherwise P2P can hang.
- `--profile_timing` enables detailed per-stage timing in `example_mp_pipeline_generate.py` (on by default) and optional breakdown in eval.
- Checkpoints with uneven `stage_layers` in config are supported when `config['version'] == 12`; v11 checkpoints use uniform stage splits.
