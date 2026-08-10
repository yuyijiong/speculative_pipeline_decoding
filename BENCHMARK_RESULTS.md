# Benchmark results

Wall-clock speedup on MT-Bench, GSM8K, and HumanEval (`max_new_tokens=512`, 80 prompts per dataset). All results use `draft_top_k=1`. Baseline is single-GPU autoregressive decoding on the same hardware. Table cells are **theoretical speedup / wall-clock speedup** (higher is better).

- **Theoretical speedup (ours):** `mean_equivalent_accept_length` (draft latency overlapped with the pipeline).
- **Wall-clock speedup:** `mean_decode_tok_s` relative to baseline.

Reproduce our wall-clock numbers with [`distributed_inference/eval_mp_pipeline_dataset.py`](distributed_inference/eval_mp_pipeline_dataset.py). Checkpoint naming: `Qwen3.5-{4B,9B}_s{stages}_l{spec_layers}.pt`.

---

## Default training mix

Checkpoints trained on the corpora listed in the [main README](README.md#training-data) (Ultrachat-200k, ShareGPT, SmolTalk, etc.). See [Hugging Face — main branch](https://huggingface.co/yuyijiong/speculative_pipeline_decoding).

### Qwen3.5-4B (L=32, baseline ≈ 35.26 tok/s)

#### Temp=0

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.47 / 1.88 | 2.11 / 1.66 | 2.71 / 2.04 | 2.58 / 1.95 |
| eagle3 num_steps=7 | 2.72 / 1.97 | 2.14 / 1.65 | 3.10 / 2.17 | 2.92 / 2.10 |
| eagle3 num_steps=15 | 2.39 / 1.62 | 1.81 / 1.30 | 2.73 / 1.72 | 2.63 / 1.83 |
| PPSD stages=4 layers=1 | 1.54 / 1.28 | 1.41 / 1.17 | 1.52 / 1.26 | 1.70 / 1.41 |
| PPSD stages=8 layers=1 | 1.68 / 1.25 | 1.38 / 1.04 | 1.60 / 1.20 | 2.05 / 1.52 |
| PPSD stages=16 layers=1 | 1.40 / 0.89 | 1.18 / 0.75 | 1.36 / 0.86 | 1.65 / 1.04 |
| **ours** stages=4 layers=4 | **2.64 / 2.27** | **2.19 / 1.90** | **2.61 / 2.24** | **3.11 / 2.65** |
| **ours** stages=8 layers=4 | **3.51 / 2.41** | **2.63 / 1.84** | **3.32 / 2.30** | **4.59 / 3.10** |
| **ours** stages=8 layers=3 | **3.39 / 2.53** | **2.55 / 1.93** | **3.22 / 2.43** | **4.40 / 3.23** |
| **ours** stages=8 layers=2 | **3.32 / 2.47** | **2.48 / 1.88** | **3.15 / 2.36** | **4.34 / 3.17** |
| **ours** stages=16 layers=2 | **3.70 / 1.43** | **2.59 / 1.04** | **3.35 / 1.32** | **5.17 / 1.94** |

#### Temp=1

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.03 / 1.47 | 1.80 / 1.31 | 2.10 / 1.53 | 2.19 / 1.58 |
| eagle3 num_steps=7 | 2.06 / 1.46 | 1.73 / 1.26 | 2.15 / 1.55 | 2.29 / 1.56 |
| eagle3 num_steps=15 | 1.80 / 1.17 | 1.45 / 0.98 | 1.87 / 1.21 | 2.08 / 1.33 |
| PPSD stages=4 layers=1 | 1.39 / 1.06 | 1.30 / 0.99 | 1.37 / 1.05 | 1.50 / 1.14 |
| PPSD stages=8 layers=1 | 1.44 / 0.95 | 1.27 / 0.84 | 1.40 / 0.92 | 1.66 / 1.09 |
| PPSD stages=16 layers=1 | 1.20 / 0.65 | 1.06 / 0.58 | 1.19 / 0.64 | 1.33 / 0.72 |
| **ours** stages=4 layers=4 | **2.48 / 1.95** | **2.09 / 1.67** | **2.49 / 1.96** | **2.85 / 2.21** |
| **ours** stages=8 layers=4 | **3.13 / 1.84** | **2.41 / 1.45** | **3.02 / 1.80** | **3.96 / 2.28** |
| **ours** stages=8 layers=3 | **3.05 / 2.05** | **2.40 / 1.66** | **2.91 / 1.99** | **3.84 / 2.50** |
| **ours** stages=8 layers=2 | **2.96 / 2.03** | **2.29 / 1.62** | **2.86 / 1.99** | **3.73 / 2.47** |
| **ours** stages=16 layers=2 | **3.19 / 1.06** | **2.37 / 0.81** | **2.96 / 1.00** | **4.24 / 1.37** |

### Qwen3.5-9B (L=32, baseline ≈ 34.18 tok/s)

#### Temp=0

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.69 / 2.24 | 2.28 / 1.95 | 2.88 / 2.36 | 2.92 / 2.40 |
| eagle3 num_steps=7 | 3.17 / 2.54 | 2.35 / 1.99 | 3.44 / 2.69 | 3.71 / 2.94 |
| eagle3 num_steps=15 | 2.88 / 2.18 | 2.01 / 1.62 | 3.13 / 2.22 | 3.49 / 2.69 |
| PPSD stages=4 layers=1 | 1.90 / 1.53 | 1.57 / 1.28 | 1.85 / 1.49 | 2.27 / 1.82 |
| PPSD stages=8 layers=1 | 1.96 / 1.44 | 1.51 / 1.13 | 1.82 / 1.34 | 2.55 / 1.85 |
| PPSD stages=16 layers=1 | 1.56 / 0.98 | 1.23 / 0.78 | 1.48 / 0.92 | 1.98 / 1.23 |
| **ours** stages=4 layers=4 | **2.70 / 2.32** | **2.24 / 1.94** | **2.68 / 2.30** | **3.19 / 2.73** |
| **ours** stages=8 layers=4 | **3.62 / 2.44** | **2.69 / 1.85** | **3.48 / 2.35** | **4.70 / 3.10** |
| **ours** stages=8 layers=3 | **3.61 / 2.67** | **2.65 / 2.01** | **3.43 / 2.55** | **4.74 / 3.44** |
| **ours** stages=8 layers=2 | **3.46 / 2.56** | **2.59 / 1.96** | **3.26 / 2.42** | **4.54 / 3.31** |
| **ours** stages=16 layers=2 | **3.90 / 1.49** | **2.71 / 1.08** | **3.52 / 1.37** | **5.48 / 2.02** |

#### Temp=1

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.13 / 1.81 | 1.86 / 1.60 | 2.23 / 1.88 | 2.29 / 1.96 |
| eagle3 num_steps=7 | 2.26 / 1.78 | 1.89 / 1.53 | 2.35 / 1.91 | 2.53 / 1.89 |
| eagle3 num_steps=15 | 1.95 / 1.40 | 1.55 / 1.22 | 2.00 / 1.47 | 2.30 / 1.51 |
| PPSD stages=4 layers=1 | 1.71 / 1.27 | 1.44 / 1.08 | 1.67 / 1.24 | 2.03 / 1.49 |
| PPSD stages=8 layers=1 | 1.68 / 1.10 | 1.36 / 0.90 | 1.57 / 1.03 | 2.12 / 1.37 |
| PPSD stages=16 layers=1 | 1.32 / 0.72 | 1.10 / 0.61 | 1.27 / 0.69 | 1.60 / 0.86 |
| **ours** stages=4 layers=4 | **2.57 / 2.01** | **2.13 / 1.72** | **2.54 / 1.99** | **3.03 / 2.33** |
| **ours** stages=8 layers=4 | **3.34 / 1.93** | **2.52 / 1.51** | **3.18 / 1.86** | **4.31 / 2.43** |
| **ours** stages=8 layers=3 | **3.33 / 2.21** | **2.50 / 1.73** | **3.16 / 2.13** | **4.32 / 2.78** |
| **ours** stages=8 layers=2 | **3.16 / 2.15** | **2.36 / 1.68** | **3.01 / 2.07** | **4.11 / 2.69** |
| **ours** stages=16 layers=2 | **3.46 / 1.13** | **2.45 / 0.83** | **3.11 / 1.03** | **4.83 / 1.53** |

---

## Open-PerfectBlend decontaminated training

Checkpoints trained on **[openeurollm/open-perfectblend-decontaminated](https://huggingface.co/datasets/openeurollm/open-perfectblend-decontaminated)** — a decontaminated variant of [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) with benchmark overlap removed — instead of the default training mix above. Acceptance rates are **much higher** on the same eval prompts, but performance may be worse on Chinese prompts because Open-PerfectBlend is English-only.

**Checkpoints:**

| Method | Hugging Face |
| --- | --- |
| **ours (v11)** | [`v11_open-perfectblend`](https://huggingface.co/yuyijiong/speculative_pipeline_decoding/tree/v11_open-perfectblend) — `Qwen3.5-4B_s4_l4.pt`, `Qwen3.5-9B_s4_l4.pt`, `Qwen3.5-9B_s8_l3.pt` |
| **Eagle3 (Qwen3.5-4B)** | [yuyijiong/Qwen3.5-4B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-4B-Eagle3) |
| **Eagle3 (Qwen3.5-9B)** | [yuyijiong/Qwen3.5-9B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3) |

Our v11 wall-clock numbers come from `distributed_inference/eval_mp_pipeline_dataset.py`. Eagle3 numbers are from SGLang evaluation of the corresponding EAGLE3 draft models.

### Qwen3.5-4B (L=32, baseline ≈ 35.26 tok/s)

#### Temp=0

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.76 / 1.91 | 2.32 / 1.66 | 3.08 / 2.10 | 2.88 / 1.98 |
| eagle3 num_steps=7 | 3.50 / 2.10 | 2.61 / 1.68 | 4.19 / 2.42 | 3.71 / 2.21 |
| eagle3 num_steps=15 | 3.66 / 1.70 | 2.66 / 1.32 | 4.41 / 1.95 | 3.92 / 1.82 |
| **ours** stages=4 layers=4 | **2.77 / 2.42** | **2.24 / 1.97** | **2.84 / 2.48** | **3.23 / 2.80** |
| **ours** stages=8 layers=3 | **3.80 / 2.78** | **2.66 / 1.99** | **3.77 / 2.78** | **4.97 / 3.56** |

#### Temp=1

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.29 / 1.53 | 1.99 / 1.38 | 2.43 / 1.62 | 2.45 / 1.60 |
| eagle3 num_steps=7 | 2.61 / 1.51 | 2.11 / 1.26 | 2.91 / 1.69 | 2.82 / 1.56 |
| eagle3 num_steps=15 | 2.72 / 1.23 | 2.16 / 0.98 | 2.82 / 1.26 | 3.19 / 1.44 |
| **ours** stages=4 layers=4 | **2.65 / 2.11** | **2.17 / 1.77** | **2.75 / 2.19** | **3.02 / 2.37** |
| **ours** stages=8 layers=3 | **3.43 / 2.13** | **2.46 / 1.59** | **3.43 / 2.14** | **4.38 / 2.64** |

### Qwen3.5-9B (L=32, baseline ≈ 34.18 tok/s)

#### Temp=0

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.97 / 2.25 | 2.45 / 1.92 | 3.26 / 2.43 | 3.18 / 2.39 |
| eagle3 num_steps=7 | 3.97 / 2.59 | 2.80 / 2.00 | 4.62 / 2.90 | 4.48 / 2.88 |
| eagle3 num_steps=15 | 4.34 / 2.19 | 2.84 / 1.59 | 5.08 / 2.41 | 5.09 / 2.56 |
| **ours** stages=4 layers=4 | **2.79 / 2.43** | **2.25 / 1.97** | **2.88 / 2.50** | **3.25 / 2.83** |
| **ours** stages=8 layers=3 | **3.86 / 2.72** | **2.68 / 1.97** | **3.86 / 2.70** | **5.04 / 3.50** |

#### Temp=1

| method | overall | mt bench | gsm8k | humaneval |
| --- | --- | --- | --- | --- |
| eagle3 num_steps=3 | 2.34 / 1.74 | 2.04 / 1.54 | 2.50 / 1.85 | 2.48 / 1.83 |
| eagle3 num_steps=7 | 2.75 / 1.80 | 2.20 / 1.48 | 2.98 / 1.97 | 3.07 / 1.94 |
| eagle3 num_steps=15 | 2.80 / 1.43 | 2.20 / 1.19 | 2.89 / 1.48 | 3.32 / 1.63 |
| **ours** stages=4 layers=4 | **2.68 / 2.13** | **2.16 / 1.77** | **2.74 / 2.17** | **3.13 / 2.45** |
| **ours** stages=8 layers=3 | **3.55 / 2.17** | **2.53 / 1.62** | **3.56 / 2.19** | **4.56 / 2.71** |
