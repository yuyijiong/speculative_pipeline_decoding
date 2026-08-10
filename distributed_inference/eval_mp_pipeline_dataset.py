"""
Evaluate multi-GPU pipeline-parallel decoding (v11) on real benchmark prompts.

Uses the same datasets and generation settings as ``eval.py``:
rows from ``eval_data/{mt_bench,humaneval,gsm8k}/question.jsonl`` (first turn only;
use ``--prompts_per_dataset N`` to take only the first N per dataset),
``max_new_tokens=512``, temperatures ``[0.0, 1.0]``, ``draft_top_k=[1]`` (recorded in
outputs; multi-process v11 does not branch on draft top-k).

Per sample (rank 0): prefill wall time, decode wall time, decode compute time
(per-round ``max(stage forwards incl. speculation) + verify``, summed over rounds;
excludes pure communication and rank-0 pipeline wait), acceptance rate
(``new_tokens / decode_loop_steps``), equivalent accept length
(``theoretical_pipeline_parallel_factor * rate``; uniform splits use ``num_stages``),
draft-flag counts, and decode throughput (wall-clock and compute-only tok/s).
Summary JSON reports **arithmetic means** across samples plus **pooled** rates
(token/step totals), split by dataset and overall.

Launch (per checkpoint, ``world_size`` must equal ``num_stages + 1``). Multiple checkpoints
with different ``num_stages`` are evaluated sequentially (in-process when ``world_size``
matches; otherwise rank 0 relaunches ``torchrun`` on single-node jobs, or run ``python``
without ``torchrun`` to auto-launch all)::

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 \
    distributed_inference/eval_mp_pipeline_dataset.py \
    --rank_gpus 0,1,2,3,4 --spec_head_ckpt Qwen3.5-4B_s4_l4.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=9 \
    distributed_inference/eval_mp_pipeline_dataset.py \
    --rank_gpus 0,1,2,3,4,5,6,7,8 --spec_head_ckpt Qwen3.5-4B_s8_l4.pt \
    --profile_timing --prompts_per_dataset 3 --temperature 0.0

Multi-node (8 stages, 9 GPUs across 2 nodes)::

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=10.132.16.106 --master_port=29500 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt \
  "/share/yyj/pipeline_decoding/train_eval_runs-v11-4b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-4B_s8_l3_c4b280357f/speculation_head_final.pt" \
   "/share/yyj/pipeline_decoding/train_eval_runs-v11-9b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-9B_s8_l3_6756e5d551/speculation_head_final.pt" \

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=10.132.16.106 --master_port=29500 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt \
  "/share/yyj/pipeline_decoding/train_eval_runs-v11-4b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-4B_s8_l3_c4b280357f/speculation_head_final.pt" \
   "/share/yyj/pipeline_decoding/train_eval_runs-v11-9b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-9B_s8_l3_6756e5d551/speculation_head_final.pt" \

Outputs under ``--output_dir``::

    raw/mp_pipeline_eval__<checkpoint_tag>__nt<total>__per_sample.jsonl
    raw/...__t<temp>__per_sample.jsonl   (multiple temperatures)
    summary/mp_pipeline_eval__<checkpoint_tag>__nt<total>__summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle("eval_v11_mp")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import (  # noqa: E402
    DATASET_CONFIG,
    UnifiedItem,
    _encode_prompt,
    _normalize_draft_top_k_list,
    _normalize_spec_head_ckpt_list,
    _normalize_temperature_list,
    _per_sample_file_stem,
    _set_torch_rng_for_eval_sample,
    ckpt_to_filename_tag,
    unify_from_jsonl,
)
from distributed_inference.comm import (  # noqa: E402
    CtrlOpcode,
    PipelineP2P,
    make_ctrl_tensor,
)
from distributed_inference.device import (  # noqa: E402
    PhaseTimeout,
    init_dist_rank_device,
    parse_rank_gpu_ids,
)
from distributed_inference.example_mp_pipeline_generate import (  # noqa: E402
    _forward_stage_timing_rows,
    _resolve_base_model_path,
    _run_one_session,
)
from distributed_inference.loader import (  # noqa: E402
    load_prefill_rank0_bundle,
    load_spec_checkpoint_config,
    load_stage_rank_bundle,
)
from distributed_inference.prefill import broadcast_input_ids  # noqa: E402
from distributed_inference.stage_worker import StageWorker  # noqa: E402
from distributed_inference.topology import expected_world_size  # noqa: E402

_DEFAULT_SPEC_HEAD: list[str] = ["Qwen3.5-4B_s4_l4.pt"]


def pipeline_parallel_factor_from_spec_cfg(spec_cfg: dict[str, Any]) -> float:
    """Ideal pipeline parallelism from a speculation-head config (uniform v11: ``num_stages``)."""
    version = int(spec_cfg.get("version", 0) or 0)
    num_stages = int(spec_cfg["num_stages"])
    if version != 12:
        return float(max(num_stages, 1))
    num_layers = 0
    for key in ("num_layers", "num_hidden_layers"):
        raw = spec_cfg.get(key)
        if raw is not None and int(raw) > 0:
            num_layers = int(raw)
            break
    raw_ranges = spec_cfg.get("stage_layer_ranges")
    if raw_ranges is None:
        return float(max(num_stages, 1))
    counts = [int(row[1]) - int(row[0]) for row in raw_ranges]
    max_stage_layers = max(counts) if counts else 1
    if num_layers <= 0:
        num_layers = max(int(row[1]) for row in raw_ranges)
    return float(num_layers) / float(max(max_stage_layers, 1))

# Cumulative timers in the decode breakdown (seconds). Prefer matching
# ``*_avg_sec`` (empty steps excluded) when building per-step ms profiles.
_PROFILE_CUMULATIVE_SEC_KEYS: tuple[str, ...] = (
    # Atomic rank-0 phases (additive cycle wall partition)
    "wait_comm_sec",
    "ctrl_prepare_sec",
    "ctrl_broadcast_sec",
    "post_recv_sec",
    "cycle_other_sec",
    "cycle_phases_sec",
    "cycle_phases_gap_sec",
    "spec_forward_sec",
    "recv_snap_sec",
    "recv_verify_sec",
    "snap_progress_sec",
    "cycle_sync_sec",
    "verify_sec",
    "verify_copy_sec",
    "verify_kernel_sec",
    "verify_decide_sec",
    "discard_comm_sec",
    "driver_update_sec",
    "shutdown_sec",
    "cycle_wall_sec",
    "stage0_hs_send_sec",
    "last_stage_hs_send_sec",
    # Derived buckets (totals; avg uses dedicated keys below when present)
    "ctrl_overhead_sec",
    "sync_overhead_sec",
    "pure_comm_sec",
    "rank0_recv_wait_sec",
    "pipeline_wait_sec",
    "rank0_local_compute_sec",
    "rank0_sequential_sec",
    "rank0_unaccounted_sec",
    # Stage gather
    "max_stage_forward_sec",
    "decode_cycle_max_stage_sec",
    "critical_path_compute_sec",
    "decode_compute_sec",
    # Rollback
    "rollback_wall_sec",
    "rollback_verify_sec",
    "rollback_discard_comm_sec",
    "rollback_apply_reject_sec",
)
# Already an average over the phase's own active steps (empty steps excluded).
_PROFILE_AVG_SEC_KEYS: tuple[str, ...] = (
    "wait_comm_avg_sec",
    "ctrl_prepare_avg_sec",
    "ctrl_broadcast_avg_sec",
    "post_recv_avg_sec",
    "cycle_other_avg_sec",
    "cycle_phases_avg_sec",
    "spec_forward_avg_sec",
    "recv_snap_avg_sec",
    "recv_verify_avg_sec",
    "snap_progress_avg_sec",
    "cycle_sync_avg_sec",
    "verify_avg_sec",
    "verify_copy_avg_sec",
    "verify_kernel_avg_sec",
    "verify_decide_avg_sec",
    "verify_non_kernel_avg_sec",
    "discard_comm_avg_sec",
    "driver_update_avg_sec",
    "cycle_wall_avg_sec",
    "stage0_hs_send_avg_sec",
    "last_stage_hs_send_avg_sec",
    "rank0_recv_wait_avg_sec",
    "pipeline_wait_avg_sec",
    "critical_path_compute_avg_sec",
    "rollback_avg_sec",
    "max_stage_forward_avg_sec",
)
# Map cumulative key -> active-step count field (fallback when *_avg_sec absent).
_PROFILE_ACTIVE_STEPS_FOR_SEC: dict[str, str] = {
    "wait_comm_sec": "wait_comm_steps",
    "ctrl_prepare_sec": "ctrl_prepare_steps",
    "ctrl_broadcast_sec": "ctrl_broadcast_steps",
    "post_recv_sec": "post_recv_steps",
    "cycle_other_sec": "cycle_other_steps",
    "cycle_phases_sec": "cycle_wall_steps",
    "spec_forward_sec": "spec_forward_steps",
    "recv_snap_sec": "recv_snap_steps",
    "recv_verify_sec": "recv_verify_steps",
    "snap_progress_sec": "snap_progress_steps",
    "cycle_sync_sec": "cycle_sync_steps",
    "verify_sec": "verify_steps",
    "verify_copy_sec": "verify_copy_steps",
    "verify_kernel_sec": "verify_kernel_steps",
    "verify_decide_sec": "verify_decide_steps",
    "discard_comm_sec": "discard_comm_steps",
    "driver_update_sec": "driver_update_steps",
    "shutdown_sec": "shutdown_steps",
    "cycle_wall_sec": "cycle_wall_steps",
    "stage0_hs_send_sec": "stage0_hs_send_steps",
    "last_stage_hs_send_sec": "last_stage_hs_send_steps",
    "rank0_recv_wait_sec": "full_pipeline_steps",
    "pipeline_wait_sec": "full_pipeline_steps",
    "rollback_wall_sec": "rollback_count",
    "rollback_verify_sec": "rollback_count",
    "rollback_discard_comm_sec": "rollback_count",
    "rollback_apply_reject_sec": "rollback_count",
}
_TIMING_DECIMALS = 4

_TORCHRUN_ENV_VARS: tuple[str, ...] = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU pipeline v11 eval: speed (prefill/decode) and acceptance on real datasets"
    )
    p.add_argument(
        "--spec_head_ckpt",
        nargs="+",
        type=str,
        default=_normalize_spec_head_ckpt_list(_DEFAULT_SPEC_HEAD),
        help=(
            "One or more speculation_head checkpoints. Checkpoints with different num_stages "
            "are evaluated sequentially (auto torchrun relaunch on single-node)."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output_distributed",
    )
    p.add_argument("--data_dir", type=str, default="eval_data")
    p.add_argument(
        "--prompts_per_dataset",
        type=int,
        default=None,
        help=(
            "Take the first N prompts from each dataset (mt_bench/humaneval/gsm8k). "
            "Omit or pass nothing to use all prompts."
        ),
    )
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[0.0,1.0],
    )
    p.add_argument(
        "--draft_top_k",
        nargs="+",
        type=int,
        default=[1],
        help="Recorded for parity with eval.py (not used by mp v11 decode)",
    )
    p.add_argument(
        "--use_deepest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top_k", type=int, default=50, help="Sampling top-k when temperature > 0")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--rank_gpus",
        type=str,
        default="",
        help="Comma-separated physical GPU ids, length = world_size (num_stages+1)",
    )
    p.add_argument("--prefill_timeout_sec", type=float, default=120.0)
    p.add_argument("--decode_timeout_sec", type=float, default=60.0)
    p.add_argument("--warmup_iters", type=int, default=1)
    p.add_argument("--warmup_new_tokens", type=int, default=4)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model weights dtype for base LM and speculation module (bfloat16, float16, float32)",
    )
    p.add_argument(
        "--profile_timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-stage compute timing diagnostics (off by default for eval throughput).",
    )
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--_no_auto_launch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()

def _is_distributed_launched() -> bool:
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


def _is_single_node_job() -> bool:
    return int(os.environ.get("NNODES", "1")) <= 1


def _torchrun_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _fresh_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _TORCHRUN_ENV_VARS:
        env.pop(key, None)
    return env


def _needs_parent_sequential_launch(
    args: argparse.Namespace,
    ckpt_list: list[str],
) -> bool:
    if bool(args._no_auto_launch) or not _is_distributed_launched():
        return False
    if not _is_single_node_job() or not ckpt_list:
        return False
    groups = _group_consecutive_ckpts_by_world_size(ckpt_list)
    world_size = _torchrun_world_size()
    return not all(ws == world_size for ws, _ in groups)


def _expected_world_size_for_ckpt(ckpt_path: str) -> int:
    spec_cfg = load_spec_checkpoint_config(ckpt_path)
    return expected_world_size(int(spec_cfg["num_stages"]))


def _split_ckpts_by_world_size(
    ckpt_list: list[str],
    world_size: int,
) -> tuple[list[str], list[str]]:
    matching: list[str] = []
    deferred: list[str] = []
    for ckpt in ckpt_list:
        if _expected_world_size_for_ckpt(ckpt) == world_size:
            matching.append(ckpt)
        else:
            deferred.append(ckpt)
    return matching, deferred


def _group_consecutive_ckpts_by_world_size(
    ckpt_list: list[str],
) -> list[tuple[int, list[str]]]:
    groups: list[tuple[int, list[str]]] = []
    for ckpt in ckpt_list:
        ws = _expected_world_size_for_ckpt(ckpt)
        if groups and groups[-1][0] == ws:
            groups[-1][1].append(ckpt)
        else:
            groups.append((ws, [ckpt]))
    return groups


def _append_nargs_plus_flag(argv: list[str], name: str, values: list[str]) -> None:
    if not values:
        return
    argv.append(f"--{name}")
    argv.extend(values)


def _append_bool_flag(argv: list[str], name: str, value: bool) -> None:
    argv.append(f"--{name}" if value else f"--no-{name}")


def _worker_cli_args_from_namespace(
    args: argparse.Namespace,
    ckpt_paths: list[str],
    *,
    rank_gpus_arg: str | None = None,
) -> list[str]:
    argv: list[str] = ["--_no_auto_launch"]
    argv.extend(["--spec_head_ckpt", *ckpt_paths])
    argv.extend(["--output_dir", str(args.output_dir)])
    argv.extend(["--data_dir", str(args.data_dir)])
    if args.prompts_per_dataset is not None:
        argv.extend(["--prompts_per_dataset", str(int(args.prompts_per_dataset))])
    if str(args.base_model_path).strip():
        argv.extend(["--base_model_path", str(args.base_model_path)])
    argv.extend(["--seed", str(int(args.seed))])
    argv.extend(["--max_new_tokens", str(int(args.max_new_tokens))])
    _append_nargs_plus_flag(
        argv,
        "temperature",
        [str(float(t)) for t in _normalize_temperature_list(list(args.temperature))],
    )
    _append_nargs_plus_flag(
        argv,
        "draft_top_k",
        [str(int(d)) for d in _normalize_draft_top_k_list(list(args.draft_top_k))],
    )
    _append_bool_flag(argv, "use_deepest", bool(args.use_deepest))
    _append_bool_flag(argv, "verify", bool(args.verify))
    argv.extend(["--top_k", str(int(args.top_k))])
    argv.extend(["--top_p", str(float(args.top_p))])
    if rank_gpus_arg is not None:
        if rank_gpus_arg.strip():
            argv.extend(["--rank_gpus", rank_gpus_arg])
    elif str(args.rank_gpus).strip():
        argv.extend(["--rank_gpus", str(args.rank_gpus)])
    argv.extend(["--prefill_timeout_sec", str(float(args.prefill_timeout_sec))])
    argv.extend(["--decode_timeout_sec", str(float(args.decode_timeout_sec))])
    argv.extend(["--warmup_iters", str(int(args.warmup_iters))])
    argv.extend(["--warmup_new_tokens", str(int(args.warmup_new_tokens))])
    argv.extend(["--attn_implementation", str(args.attn_implementation)])
    argv.extend(["--dtype", str(args.dtype)])
    _append_bool_flag(argv, "profile_timing", bool(args.profile_timing))
    if bool(args.no_chat_template):
        argv.append("--no_chat_template")
    _append_bool_flag(argv, "enable_thinking", bool(args.enable_thinking))
    return argv


def _cuda_env_and_rank_gpus_for_ws(
    rank_gpus: list[int] | None,
    world_size: int,
) -> tuple[dict[str, str], str]:
    env = _fresh_subprocess_env()
    if rank_gpus is None:
        return env, ""
    if len(rank_gpus) < world_size:
        raise ValueError(
            f"--rank_gpus has {len(rank_gpus)} entries but checkpoint needs "
            f"world_size={world_size}."
        )
    physical = rank_gpus[:world_size]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical)
    return env, ",".join(str(i) for i in range(world_size))


def _launch_eval_subprocess(
    args: argparse.Namespace,
    ckpt_paths: list[str],
    rank_gpus: list[int] | None,
) -> None:
    if not ckpt_paths:
        return
    ws = _expected_world_size_for_ckpt(ckpt_paths[0])
    for ckpt in ckpt_paths[1:]:
        other_ws = _expected_world_size_for_ckpt(ckpt)
        if other_ws != ws:
            raise ValueError(
                "Cannot launch one subprocess for checkpoints with different world_size: "
                f"{ckpt_paths[0]!r} needs {ws}, {ckpt!r} needs {other_ws}."
            )
    env, rank_gpus_arg = _cuda_env_and_rank_gpus_for_ws(rank_gpus, ws)
    script = str(Path(__file__).resolve())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={ws}",
        script,
        *_worker_cli_args_from_namespace(
            args,
            ckpt_paths,
            rank_gpus_arg=rank_gpus_arg,
        ),
    ]
    label = ckpt_paths[0] if len(ckpt_paths) == 1 else f"{len(ckpt_paths)} checkpoints"
    print(f"\nLaunching eval subprocess (world_size={ws}): {label}")
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Eval subprocess exited with code {proc.returncode} for {label!r}."
        )


def _launcher_main(args: argparse.Namespace) -> None:
    ckpt_list = _normalize_spec_head_ckpt_list(args.spec_head_ckpt)
    rank_gpus = parse_rank_gpu_ids(args.rank_gpus) if str(args.rank_gpus).strip() else None
    groups = _group_consecutive_ckpts_by_world_size(ckpt_list)
    for group_idx, (ws, ckpts) in enumerate(groups):
        if len(ckpt_list) > 1:
            print(
                f"\n=== Launcher: group {group_idx + 1}/{len(groups)} "
                f"world_size={ws} ({len(ckpts)} checkpoint"
                f"{'s' if len(ckpts) != 1 else ''}) ===\n"
            )
        _launch_eval_subprocess(args, ckpts, rank_gpus)


def _round_timing_sec(value: float) -> float:
    return round(float(value), _TIMING_DECIMALS)


def _round_timing_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _round_timing_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_timing_dict(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, _TIMING_DECIMALS)
    return obj


def _round_sample_row_timing(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key, value in out.items():
        if not isinstance(value, float):
            continue
        if key.endswith("_sec") or key.endswith("_ms") or key.endswith("_tok_s"):
            out[key] = _round_timing_sec(value)
    if "pipeline_decode_timing_profile" in out:
        out["pipeline_decode_timing_profile"] = _round_timing_dict(
            out["pipeline_decode_timing_profile"]
        )
    return out


def _sec_key_to_ms_key(key: str) -> str:
    if key.endswith("_sec"):
        return key[: -len("_sec")] + "_ms"
    return key + "_ms"


def _sec_to_ms(sec: float) -> float:
    return float(sec) * 1000.0


def _round_aggregate_timing(agg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _round_timing_sec(value) if isinstance(value, float) else value
        for key, value in agg.items()
    }


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("float32", "fp32"):
        return torch.float32
    raise ValueError(
        f"Unsupported dtype {name!r}; use bfloat16, float16, or float32."
    )





def _load_all_items(
    data_dir: Path,
    *,
    prompts_per_dataset: int | None = None,
) -> list[UnifiedItem]:
    if prompts_per_dataset is not None and int(prompts_per_dataset) < 1:
        raise ValueError(
            f"prompts_per_dataset must be >= 1 or omitted, got {prompts_per_dataset}"
        )
    all_items: list[UnifiedItem] = []
    for ds_name, rel in DATASET_CONFIG:
        rows = unify_from_jsonl(rel, data_dir)
        if prompts_per_dataset is not None:
            n = int(prompts_per_dataset)
            if len(rows) < n:
                raise ValueError(
                    f"dataset {ds_name} has only {len(rows)} prompt(s) in "
                    f"{data_dir / rel}, but prompts_per_dataset={n}"
                )
            rows = rows[:n]
        all_items.extend(rows)
    return all_items


def _acceptance_sample_fields(
    *,
    out_ids: list[int],
    accept: list[bool],
    decode_steps: int,
    pipeline_parallel_factor: float,
) -> dict[str, Any]:
    new_tokens = int(len(out_ids))
    steps = int(decode_steps)
    n_flags = len(accept)
    n_acc = int(sum(1 for x in accept if x))
    acc_rate = (float(new_tokens) / float(steps)) if steps else 0.0
    equiv = float(pipeline_parallel_factor) * acc_rate
    return {
        "new_tokens": new_tokens,
        "decode_loop_steps": steps,
        "acceptance_rate": acc_rate,
        "equivalent_accept_length": equiv,
        "n_accepted": n_acc,
        "n_acceptance_flags": n_flags,
        "theoretical_pipeline_parallel_factor": float(pipeline_parallel_factor),
    }


def _wall_timing_sample_fields(
    *,
    session: Any,
    prompt_len: int,
    new_tokens: int,
) -> dict[str, Any]:
    prefill_sec = float(session.prefill_sec)
    decode_sec = float(session.decode_sec)
    prefill_tok_s = float(prompt_len) / prefill_sec if prefill_sec > 0 else 0.0
    decode_tok_s = float(new_tokens) / decode_sec if decode_sec > 0 else 0.0
    return {
        "pipeline_prefill_wall_sec": prefill_sec,
        "pipeline_post_prefill_setup_sec": float(session.post_prefill_setup_sec),
        "pipeline_decode_wall_sec": decode_sec,
        "pipeline_prefill_tok_s": prefill_tok_s,
        "pipeline_decode_tok_s": decode_tok_s,
    }


def _decode_timing_from_session(
    session: Any,
    *,
    profile_timing: bool = False,
) -> dict[str, float]:
    wall_sec = float(session.decode_sec)
    if not profile_timing:
        return {"wall_sec": wall_sec}

    bd: dict[str, float] = dict(getattr(session, "decode_breakdown", {}) or {})
    pure_comm_sec = float(bd.get("pure_comm_sec", bd.get("comm_sec", 0.0)))
    recv_wait_sec = float(
        bd.get("rank0_recv_wait_sec", bd.get("pipeline_wait_sec", 0.0))
    )
    max_stage_sec = float(
        bd.get("max_stage_forward_sec", bd.get("decode_cycle_max_stage_sec", 0.0))
    )
    verify_sec = float(bd.get("verify_sec", 0.0))
    if "critical_path_compute_sec" in bd:
        compute_sec = float(bd["critical_path_compute_sec"])
    elif "decode_compute_sec" in bd:
        compute_sec = float(bd["decode_compute_sec"])
    elif max_stage_sec > 0.0:
        compute_sec = float(max_stage_sec + verify_sec)
    else:
        compute_sec = float(max(wall_sec - pure_comm_sec - recv_wait_sec, 0.0))
    local_compute_sec = float(
        bd.get(
            "rank0_local_compute_sec",
            bd.get("rank0_gpu_sec", 0.0),
        )
    )
    return {
        "wall_sec": wall_sec,
        "pure_comm_sec": pure_comm_sec,
        "pipeline_wait_sec": recv_wait_sec,
        "rank0_recv_wait_sec": recv_wait_sec,
        "compute_sec": compute_sec,
        "critical_path_compute_sec": compute_sec,
        "cycle_max_stage_sec": max_stage_sec,
        "max_stage_forward_sec": max_stage_sec,
        "max_stage_forward_avg_sec": float(bd.get("max_stage_forward_avg_sec", 0.0)),
        "rank0_local_compute_sec": local_compute_sec,
        "rank0_gpu_sec": local_compute_sec,
        "spec_forward_sec": float(bd.get("spec_forward_sec", 0.0)),
        "verify_sec": verify_sec,
        "driver_update_sec": float(bd.get("driver_update_sec", 0.0)),
        "cycle_wall_sec": float(bd.get("cycle_wall_sec", 0.0)),
        "rank0_sequential_sec": float(bd.get("rank0_sequential_sec", 0.0)),
        "rank0_unaccounted_sec": float(
            bd.get("rank0_unaccounted_sec", bd.get("unaccounted_sec", 0.0))
        ),
        "unaccounted_sec": float(
            bd.get("rank0_unaccounted_sec", bd.get("unaccounted_sec", 0.0))
        ),
    }


def _timing_profile_from_breakdown(bd: dict[str, Any]) -> dict[str, Any]:
    """Per-sample timing profile: per-active-step averages in ms (empty steps excluded)."""
    if not bd:
        return {}
    steps = int(bd.get("spec_forward_steps", 0))
    full_steps = int(bd.get("full_pipeline_steps", 0))
    fallback_den = float(steps) if steps > 0 else 0.0

    avg_by_base: dict[str, float] = {}
    for key in _PROFILE_AVG_SEC_KEYS:
        if key not in bd:
            continue
        # wait_comm_avg_sec -> wait_comm_ms
        base = key[: -len("_avg_sec")] if key.endswith("_avg_sec") else key
        avg_by_base[base] = float(bd[key])
    # Alias cumulative keys onto the correct active-step averages.
    if "max_stage_forward" in avg_by_base:
        avg_by_base.setdefault("decode_cycle_max_stage", avg_by_base["max_stage_forward"])
    if "critical_path_compute" in avg_by_base:
        avg_by_base.setdefault("decode_compute", avg_by_base["critical_path_compute"])
    if "rank0_recv_wait" in avg_by_base:
        avg_by_base.setdefault("pipeline_wait", avg_by_base["rank0_recv_wait"])

    per_step_ms: dict[str, Any] = {}
    for key in _PROFILE_CUMULATIVE_SEC_KEYS:
        if key not in bd and key.replace("_sec", "") not in avg_by_base:
            # Still allow avg-only keys that map to this cumulative name.
            base = key[: -len("_sec")] if key.endswith("_sec") else key
            if base not in avg_by_base:
                continue
        ms_key = _sec_key_to_ms_key(key)
        base = key[: -len("_sec")] if key.endswith("_sec") else key
        if base in avg_by_base:
            per_step_ms[ms_key] = _sec_to_ms(avg_by_base[base])
            continue
        sec = float(bd.get(key, 0.0))
        steps_key = _PROFILE_ACTIVE_STEPS_FOR_SEC.get(key)
        den = float(bd[steps_key]) if steps_key and float(bd.get(steps_key, 0)) > 0 else fallback_den
        per_step_ms[ms_key] = (_sec_to_ms(sec) / den) if den > 0 else 0.0

    # Also emit avg-only keys that have no cumulative twin in the loop above.
    for key in _PROFILE_AVG_SEC_KEYS:
        if key not in bd:
            continue
        base = key[: -len("_avg_sec")] if key.endswith("_avg_sec") else key
        ms_key = f"{base}_ms"
        if ms_key not in per_step_ms:
            per_step_ms[ms_key] = _sec_to_ms(float(bd[key]))

    stage_totals = [float(x) for x in bd.get("stage_forward_sec", [])]
    stage_counts = [int(x) for x in bd.get("stage_forward_active_steps", [])]
    stage_forward_ms = [
        (_sec_to_ms(tot) / float(cnt)) if cnt > 0 else 0.0
        for tot, cnt in zip(stage_totals, stage_counts)
    ]
    if len(stage_forward_ms) < len(stage_totals):
        stage_forward_ms.extend(
            [0.0] * (len(stage_totals) - len(stage_forward_ms))
        )
    per_step_ms["stage_forward_ms"] = stage_forward_ms

    counts: dict[str, Any] = {
        "spec_forward_steps": steps,
        "decode_steps": steps,
        "full_pipeline_steps": full_steps,
        "recv_verify_steps": int(bd.get("recv_verify_steps", 0)),
        "recv_snap_steps": int(bd.get("recv_snap_steps", 0)),
        "verify_steps": int(bd.get("verify_steps", 0)),
        "verify_copy_steps": int(bd.get("verify_copy_steps", 0)),
        "verify_kernel_steps": int(bd.get("verify_kernel_steps", 0)),
        "verify_decide_steps": int(bd.get("verify_decide_steps", 0)),
        "verify_graph_steps": int(bd.get("verify_graph_steps", 0)),
        "rollback_count": int(bd.get("rollback_count", 0)),
        "by_depth_steps": [int(x) for x in bd.get("by_depth_steps", [])],
    }

    def _by_depth_ms(avg_key: str, total_key: str) -> list[float]:
        avgs = bd.get(avg_key)
        if isinstance(avgs, list) and avgs:
            return [_sec_to_ms(float(x)) for x in avgs]
        totals = [float(x) for x in bd.get(total_key, [])]
        dens = [int(x) for x in bd.get("by_depth_steps", [])]
        out: list[float] = []
        for i, tot in enumerate(totals):
            den = dens[i] if i < len(dens) else 0
            out.append((_sec_to_ms(tot) / float(den)) if den > 0 else 0.0)
        return out

    by_depth_ms = {
        "by_depth_steps": [int(x) for x in bd.get("by_depth_steps", [])],
        "recv_wait_ms": _by_depth_ms(
            "by_depth_recv_wait_avg_sec", "by_depth_recv_wait_sec"
        ),
        "recv_snap_ms": _by_depth_ms(
            "by_depth_recv_snap_avg_sec", "by_depth_recv_snap_sec"
        ),
        "recv_verify_ms": _by_depth_ms(
            "by_depth_recv_verify_avg_sec", "by_depth_recv_verify_sec"
        ),
        "cycle_wall_ms": _by_depth_ms(
            "by_depth_cycle_wall_avg_sec", "by_depth_cycle_wall_sec"
        ),
        "spec_forward_ms": _by_depth_ms(
            "by_depth_spec_forward_avg_sec", "by_depth_spec_forward_sec"
        ),
        "cycle_other_ms": _by_depth_ms(
            "by_depth_cycle_other_avg_sec", "by_depth_cycle_other_sec"
        ),
    }

    forward_rows: list[dict[str, Any]] = []
    for row in _forward_stage_timing_rows(bd):
        forward_rows.append(
            {
                "label": str(row["label"]),
                "avg_ms": _sec_to_ms(float(row["avg_sec"])),
                "active_steps": int(row["active_steps"]),
            }
        )
    return {
        "per_step_ms": per_step_ms,
        "counts": counts,
        "forward_per_active_step": forward_rows,
        "by_depth_ms": by_depth_ms,
    }


def _mean_dicts(dicts: list[dict[str, Any]], keys: tuple[str, ...] | list[str]) -> dict[str, float]:
    if not dicts:
        return {}
    n = float(len(dicts))
    out: dict[str, float] = {}
    for key in keys:
        out[key] = sum(float(d.get(key, 0.0)) for d in dicts) / n
    return out


def _mean_list_field(dicts: list[dict[str, Any]], key: str) -> list[float]:
    if not dicts:
        return []
    max_len = max(len(d.get(key, []) or []) for d in dicts)
    n = float(len(dicts))
    out: list[float] = []
    for i in range(max_len):
        s = 0.0
        for d in dicts:
            lst = d.get(key, []) or []
            if i < len(lst):
                s += float(lst[i])
        out.append(s / n)
    return out


def _aggregate_by_depth_ms(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample-mean of per-depth (fill level) averages."""
    by_depth_dicts = [p.get("by_depth_ms", {}) or {} for p in profiles]
    if not any(by_depth_dicts):
        return {}
    keys = [
        "by_depth_steps",
        "recv_wait_ms",
        "recv_snap_ms",
        "recv_verify_ms",
        "cycle_wall_ms",
        "spec_forward_ms",
        "cycle_other_ms",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        if any(key in d for d in by_depth_dicts):
            out[key] = _mean_list_field(by_depth_dicts, key)
    return out


def _aggregate_forward_per_active_step(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sample-mean of each label's per-active-step avg_ms (not pooled over steps)."""
    by_label: dict[str, list[float]] = {}
    mean_active: dict[str, list[float]] = {}
    for profile in profiles:
        for row in profile.get("forward_per_active_step", []):
            label = str(row["label"])
            by_label.setdefault(label, []).append(float(row.get("avg_ms", 0.0)))
            mean_active.setdefault(label, []).append(float(row.get("active_steps", 0)))
    out: list[dict[str, Any]] = []
    for label in sorted(by_label.keys()):
        vals = by_label[label]
        acts = mean_active[label]
        n = float(len(vals))
        out.append(
            {
                "label": label,
                "avg_ms": sum(vals) / n if n else 0.0,
                "mean_active_steps": sum(acts) / n if n else 0.0,
            }
        )
    return out


def _aggregate_timing_profile(
    per_sample: list[dict[str, Any]],
    ds: Optional[str],
) -> dict[str, Any]:
    """Aggregate timing profiles by arithmetic mean over samples (no pool / totals)."""
    sub = [
        r
        for r in per_sample
        if (ds is None or r.get("dataset") == ds)
        and "pipeline_decode_timing_profile" in r
    ]
    if not sub:
        return {}
    profiles = [r["pipeline_decode_timing_profile"] for r in sub]
    per_step_dicts = [p.get("per_step_ms", {}) for p in profiles]
    count_dicts = [p.get("counts", {}) for p in profiles]

    # Discover ms keys present in any sample (stable order from canonical list first).
    ms_keys: list[str] = []
    seen: set[str] = set()
    for key in _PROFILE_CUMULATIVE_SEC_KEYS + _PROFILE_AVG_SEC_KEYS:
        ms_key = _sec_key_to_ms_key(key)
        if any(ms_key in d for d in per_step_dicts) and ms_key not in seen:
            ms_keys.append(ms_key)
            seen.add(ms_key)
    for d in per_step_dicts:
        for k in d:
            if k == "stage_forward_ms" or k in seen:
                continue
            if k.endswith("_ms"):
                ms_keys.append(k)
                seen.add(k)

    mean_per_step = _mean_dicts(per_step_dicts, ms_keys)
    mean_per_step["stage_forward_ms"] = _mean_list_field(per_step_dicts, "stage_forward_ms")
    mean_counts = _mean_dicts(
        count_dicts,
        [
            "spec_forward_steps",
            "decode_steps",
            "full_pipeline_steps",
            "recv_verify_steps",
            "recv_snap_steps",
            "verify_steps",
            "verify_copy_steps",
            "verify_kernel_steps",
            "verify_decide_steps",
            "verify_graph_steps",
            "rollback_count",
        ],
    )
    return {
        "num_samples": len(sub),
        "mean_per_step_ms": mean_per_step,
        "mean_counts": mean_counts,
        "forward_per_active_step": _aggregate_forward_per_active_step(profiles),
        "by_depth_ms": _aggregate_by_depth_ms(profiles),
    }


def _sample_row(
    *,
    idx: int,
    it: UnifiedItem,
    session: Any,
    out_ids: list[int],
    accept: list[bool],
    prompt_len: int,
    num_stages: int,
    gen_text: str,
    temperature: float,
    draft_top_k: int,
    use_deepest: bool,
    pipeline_parallel_factor: float,
    profile_timing: bool = False,
) -> dict[str, Any]:
    acceptance = _acceptance_sample_fields(
        out_ids=out_ids,
        accept=accept,
        decode_steps=int(session.decode_steps),
        pipeline_parallel_factor=pipeline_parallel_factor,
    )
    row: dict[str, Any] = {
        "index": int(idx),
        "dataset": it.dataset,
        "question_id": it.question_id,
        "question": it.question,
        "generated": gen_text,
        "prompt_len": int(prompt_len),
        **acceptance,
        **_wall_timing_sample_fields(
            session=session,
            prompt_len=prompt_len,
            new_tokens=int(acceptance["new_tokens"]),
        ),
        "num_stages": int(num_stages),
        "temperature": float(temperature),
        "greedy": bool(float(temperature) == 0.0),
        "use_deepest": bool(use_deepest),
        "draft_top_k": int(draft_top_k),
    }
    if profile_timing:
        decode_timing = _decode_timing_from_session(
            session,
            profile_timing=True,
        )
        decode_compute_sec = decode_timing["compute_sec"]
        new_tokens = int(acceptance["new_tokens"])
        row.update(
            {
                "pipeline_decode_pure_comm_sec": decode_timing["pure_comm_sec"],
                "pipeline_decode_pipeline_wait_sec": decode_timing[
                    "pipeline_wait_sec"
                ],
                "pipeline_decode_recv_wait_sec": decode_timing["rank0_recv_wait_sec"],
                "pipeline_decode_compute_sec": decode_compute_sec,
                "pipeline_decode_critical_path_compute_sec": decode_timing[
                    "critical_path_compute_sec"
                ],
                "pipeline_decode_cycle_max_stage_sec": decode_timing[
                    "cycle_max_stage_sec"
                ],
                "pipeline_decode_max_stage_forward_sec": decode_timing[
                    "max_stage_forward_sec"
                ],
                "pipeline_decode_max_stage_forward_avg_sec": decode_timing[
                    "max_stage_forward_avg_sec"
                ],
                "pipeline_decode_rank0_local_compute_sec": decode_timing[
                    "rank0_local_compute_sec"
                ],
                "pipeline_decode_rank0_gpu_sec": decode_timing["rank0_gpu_sec"],
                "pipeline_decode_spec_forward_sec": decode_timing["spec_forward_sec"],
                "pipeline_decode_verify_sec": decode_timing["verify_sec"],
                "pipeline_decode_driver_update_sec": decode_timing["driver_update_sec"],
                "pipeline_decode_cycle_wall_sec": decode_timing["cycle_wall_sec"],
                "pipeline_decode_rank0_sequential_sec": decode_timing[
                    "rank0_sequential_sec"
                ],
                "pipeline_decode_rank0_unaccounted_sec": decode_timing[
                    "rank0_unaccounted_sec"
                ],
                "pipeline_decode_unaccounted_sec": decode_timing["unaccounted_sec"],
                "pipeline_decode_compute_tok_s": (
                    float(new_tokens) / decode_compute_sec
                    if decode_compute_sec > 0
                    else 0.0
                ),
            }
        )
        row["pipeline_decode_timing_profile"] = _timing_profile_from_breakdown(
            dict(getattr(session, "decode_breakdown", {}) or {})
        )
    return row


def _aggregate_mp_pipeline(
    per_sample: list[dict[str, Any]],
    ds: Optional[str],
    *,
    profile_timing: bool = False,
) -> dict[str, float]:
    sub = [r for r in per_sample if ds is None or r.get("dataset") == ds]
    if not sub:
        empty: dict[str, float] = {
            "mean_prefill_wall_sec": 0.0,
            "mean_decode_wall_sec": 0.0,
            "mean_decode_loop_steps": 0.0,
            "mean_acceptance_rate": 0.0,
            "mean_equivalent_accept_length": 0.0,
            "pooled_acceptance_rate": 0.0,
            "pooled_equivalent_accept_length": 0.0,
            "aggregate_prefill_tok_s": 0.0,
            "aggregate_decode_tok_s": 0.0,
            "mean_decode_tok_s": 0.0,
            "total_new_tokens": 0.0,
            "total_decode_loop_steps": 0.0,
            "total_prompt_tokens": 0.0,
            "total_decode_wall_sec": 0.0,
            "total_n_accepted": 0.0,
            "total_n_acceptance_flags": 0.0,
            "num_samples": 0.0,
        }
        if profile_timing:
            empty.update(
                {
                    "mean_decode_pure_comm_sec": 0.0,
                    "mean_decode_pipeline_wait_sec": 0.0,
                    "mean_decode_compute_sec": 0.0,
                    "aggregate_decode_compute_tok_s": 0.0,
                    "total_decode_pure_comm_sec": 0.0,
                    "total_decode_pipeline_wait_sec": 0.0,
                    "total_decode_compute_sec": 0.0,
                }
            )
        return empty

    def _row_pure_comm(r: dict[str, Any]) -> float:
        if "pipeline_decode_pure_comm_sec" in r:
            return float(r["pipeline_decode_pure_comm_sec"])
        return float(r.get("pipeline_decode_comm_sec", 0.0))

    def _row_pipeline_wait(r: dict[str, Any]) -> float:
        return float(r.get("pipeline_decode_pipeline_wait_sec", 0.0))

    n = float(len(sub))
    mean_prefill = sum(float(r["pipeline_prefill_wall_sec"]) for r in sub) / n
    mean_decode = sum(float(r["pipeline_decode_wall_sec"]) for r in sub) / n
    mean_decode_steps = sum(int(r["decode_loop_steps"]) for r in sub) / n
    mean_acc = sum(float(r["acceptance_rate"]) for r in sub) / n
    mean_equiv = sum(float(r["equivalent_accept_length"]) for r in sub) / n
    tot_new = sum(int(r["new_tokens"]) for r in sub)
    tot_steps = sum(int(r["decode_loop_steps"]) for r in sub)
    tot_prompt = sum(int(r["prompt_len"]) for r in sub)
    tot_prefill = sum(float(r["pipeline_prefill_wall_sec"]) for r in sub)
    tot_decode = sum(float(r["pipeline_decode_wall_sec"]) for r in sub)
    tot_flags = sum(int(r.get("n_acceptance_flags", 0) or 0) for r in sub)
    tot_acc = sum(int(r.get("n_accepted", 0) or 0) for r in sub)
    n_stages = int(sub[0].get("num_stages", 0) or 0)
    parallel_factor = float(
        sub[0].get("theoretical_pipeline_parallel_factor", 0) or 0
    )
    if parallel_factor <= 0.0:
        parallel_factor = float(n_stages)
    pooled_acc = (float(tot_new) / float(tot_steps)) if tot_steps else 0.0
    pooled_equiv = parallel_factor * pooled_acc
    agg_prefill_tok_s = float(tot_prompt) / tot_prefill if tot_prefill > 0 else 0.0
    agg_decode_tok_s = float(tot_new) / tot_decode if tot_decode > 0 else 0.0
    mean_decode_tok_s = (
        sum(float(r["pipeline_decode_tok_s"]) for r in sub) / n
    )
    out: dict[str, float] = {
        "mean_prefill_wall_sec": mean_prefill,
        "mean_decode_wall_sec": mean_decode,
        "mean_decode_loop_steps": mean_decode_steps,
        "mean_acceptance_rate": mean_acc,
        "mean_equivalent_accept_length": mean_equiv,
        "pooled_acceptance_rate": pooled_acc,
        "pooled_equivalent_accept_length": pooled_equiv,
        "aggregate_prefill_tok_s": agg_prefill_tok_s,
        "aggregate_decode_tok_s": agg_decode_tok_s,
        "mean_decode_tok_s": mean_decode_tok_s,
        "total_new_tokens": float(tot_new),
        "total_decode_loop_steps": float(tot_steps),
        "total_prompt_tokens": float(tot_prompt),
        "total_decode_wall_sec": float(tot_decode),
        "total_n_accepted": float(tot_acc),
        "total_n_acceptance_flags": float(tot_flags),
        "num_samples": n,
    }
    if profile_timing:
        mean_decode_pure_comm = sum(_row_pure_comm(r) for r in sub) / n
        mean_decode_pipeline_wait = sum(_row_pipeline_wait(r) for r in sub) / n
        mean_decode_compute = (
            sum(float(r["pipeline_decode_compute_sec"]) for r in sub) / n
        )
        tot_decode_pure_comm = sum(_row_pure_comm(r) for r in sub)
        tot_decode_pipeline_wait = sum(_row_pipeline_wait(r) for r in sub)
        tot_decode_compute = sum(float(r["pipeline_decode_compute_sec"]) for r in sub)
        agg_decode_compute_tok_s = (
            float(tot_new) / tot_decode_compute if tot_decode_compute > 0 else 0.0
        )
        out.update(
            {
                "mean_decode_pure_comm_sec": mean_decode_pure_comm,
                "mean_decode_pipeline_wait_sec": mean_decode_pipeline_wait,
                "mean_decode_compute_sec": mean_decode_compute,
                "aggregate_decode_compute_tok_s": agg_decode_compute_tok_s,
                "total_decode_pure_comm_sec": float(tot_decode_pure_comm),
                "total_decode_pipeline_wait_sec": float(tot_decode_pipeline_wait),
                "total_decode_compute_sec": float(tot_decode_compute),
            }
        )
    return out


def _print_aggregates(
    per_sample: list[dict[str, Any]],
    wall_sec: float,
    *,
    profile_timing: bool = False,
) -> None:
    o = _aggregate_mp_pipeline(per_sample, None, profile_timing=profile_timing)
    print()
    if profile_timing:
        decode_summary = (
            f"mean decode wall {o['mean_decode_wall_sec']:.4f}s "
            f"(pure_comm {o['mean_decode_pure_comm_sec']:.4f}s, "
            f"recv_wait {o['mean_decode_pipeline_wait_sec']:.4f}s, "
            f"crit_path {o['mean_decode_compute_sec']:.4f}s)"
        )
        tok_summary = (
            f"aggregate prefill {o['aggregate_prefill_tok_s']:.2f} tok/s, "
            f"decode wall {o['aggregate_decode_tok_s']:.2f} tok/s, "
            f"mean decode {o['mean_decode_tok_s']:.2f} tok/s, "
            f"decode compute {o['aggregate_decode_compute_tok_s']:.2f} tok/s"
        )
    else:
        decode_summary = f"mean decode wall {o['mean_decode_wall_sec']:.4f}s"
        tok_summary = (
            f"aggregate prefill {o['aggregate_prefill_tok_s']:.2f} tok/s, "
            f"decode wall {o['aggregate_decode_tok_s']:.2f} tok/s, "
            f"mean decode {o['mean_decode_tok_s']:.2f} tok/s"
        )
    print(
        f"Overall — mean prefill {o['mean_prefill_wall_sec']:.4f}s, "
        f"{decode_summary}; "
        f"mean acceptance rate {o['mean_acceptance_rate']:.4f}, "
        f"mean equiv. accept length {o['mean_equivalent_accept_length']:.4f}"
    )
    print(
        f"Overall — pooled acceptance {o['pooled_acceptance_rate']:.4f}, "
        f"pooled equiv. accept length {o['pooled_equivalent_accept_length']:.4f} "
        f"[{int(o['total_new_tokens'])} tok / {int(o['total_decode_loop_steps'])} steps]"
    )
    print(f"Overall — {tok_summary}")
    for ds, _ in DATASET_CONFIG:
        m = _aggregate_mp_pipeline(per_sample, ds, profile_timing=profile_timing)
        if profile_timing:
            print(
                f"  [{ds}] mean_prefill={m['mean_prefill_wall_sec']:.4f}s, "
                f"mean_decode_wall={m['mean_decode_wall_sec']:.4f}s, "
                f"mean_decode_compute={m['mean_decode_compute_sec']:.4f}s, "
                f"mean_acc={m['mean_acceptance_rate']:.4f}, "
                f"mean_equiv={m['mean_equivalent_accept_length']:.4f}, "
                f"pooled_acc={m['pooled_acceptance_rate']:.4f}, "
                f"agg_decode_wall={m['aggregate_decode_tok_s']:.2f} tok/s, "
                f"mean_decode={m['mean_decode_tok_s']:.2f} tok/s, "
                f"agg_decode_compute={m['aggregate_decode_compute_tok_s']:.2f} tok/s"
            )
        else:
            print(
                f"  [{ds}] mean_prefill={m['mean_prefill_wall_sec']:.4f}s, "
                f"mean_decode_wall={m['mean_decode_wall_sec']:.4f}s, "
                f"mean_decode_steps={m['mean_decode_loop_steps']:.2f}, "
                f"mean_acc={m['mean_acceptance_rate']:.4f}, "
                f"mean_equiv={m['mean_equivalent_accept_length']:.4f}, "
                f"pooled_acc={m['pooled_acceptance_rate']:.4f}, "
                f"agg_decode_wall={m['aggregate_decode_tok_s']:.2f} tok/s, "
                f"mean_decode={m['mean_decode_tok_s']:.2f} tok/s "
                f"[{int(m['total_new_tokens'])} tok / {int(m['total_decode_loop_steps'])} steps]"
            )
    if profile_timing:
        tp = _aggregate_timing_profile(per_sample, None)
        if tp:
            print("\nOverall timing profile (sample-mean per step, ms):")
            mps = tp.get("mean_per_step_ms", {})
            # Ordered for diagnosis; values are per-active-step (empty steps excluded).
            highlight = [
                ("cycle_wall_ms", "cycle_wall"),
                ("post_recv_ms", "post_recv"),
                ("spec_forward_ms", "spec"),
                ("recv_verify_ms", "recv_verify"),
                ("snap_progress_ms", "snap_progress"),
                ("verify_ms", "verify"),
                ("verify_kernel_ms", "verify_kernel"),
                ("verify_copy_ms", "verify_copy"),
                ("verify_decide_ms", "verify_decide"),
                ("verify_non_kernel_avg_ms", "verify_non_kernel"),
                ("recv_snap_ms", "recv_snap"),
                ("driver_update_ms", "driver"),
                ("cycle_sync_ms", "cycle_sync"),
                ("cycle_other_ms", "cycle_other"),
                ("stage0_hs_send_ms", "stage0_hs_send"),
                ("last_stage_hs_send_ms", "last_hs_send"),
                ("max_stage_forward_avg_ms", "max_stage_avg"),
            ]
            parts = []
            seen_labels: set[str] = set()
            for key, label in highlight:
                if key not in mps:
                    continue
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                parts.append(f"{label}={float(mps.get(key, 0.0)):.4f}ms")
            if parts:
                print(f"  mean per-step: {', '.join(parts)}")
            by_depth = tp.get("by_depth_ms", {}) or {}
            depth_steps = by_depth.get("by_depth_steps", []) or []
            if depth_steps:
                print("  by pipeline fill (depth=1..n, sample-mean avg_ms):")
                n_depth = len(depth_steps)
                for d in range(n_depth):
                    steps_d = float(depth_steps[d]) if d < len(depth_steps) else 0.0

                    def _dval(key: str) -> float:
                        lst = by_depth.get(key, []) or []
                        return float(lst[d]) if d < len(lst) else 0.0

                    print(
                        f"    depth={d + 1}: steps={steps_d:.1f}, "
                        f"cycle_wall={_dval('cycle_wall_ms'):.3f}ms, "
                        f"spec={_dval('spec_forward_ms'):.3f}ms, "
                        f"recv_wait={_dval('recv_wait_ms'):.3f}ms, "
                        f"recv_verify={_dval('recv_verify_ms'):.3f}ms, "
                        f"recv_snap={_dval('recv_snap_ms'):.3f}ms"
                    )
            print("  forward per active step (sample-mean avg_ms):")
            for row in tp.get("forward_per_active_step", []):
                print(
                    f"  [{row['label']:>7}] avg={float(row['avg_ms']):8.4f} ms  "
                    f"mean_active_steps={float(row['mean_active_steps']):7.2f}"
                )
        for ds, _ in DATASET_CONFIG:
            ds_tp = _aggregate_timing_profile(per_sample, ds)
            if not ds_tp:
                continue
            parts = [
                f"{row['label']}={float(row['avg_ms']):.4f}ms"
                for row in ds_tp.get("forward_per_active_step", [])
            ]
            print(f"  [{ds}] timing profile: {', '.join(parts)}")
    print(f"\nWall time (this run config): {wall_sec:.4f}s")


def _load_models_for_ckpt(
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    spec_ckpt_path: str,
    base_path: str,
    attn_implementation: str,
    p2p: PipelineP2P,
) -> tuple[Any, Any, Any, int | None]:
    if rank == 0:
        from transformers import AutoTokenizer

        prefill_bundle = load_prefill_rank0_bundle(
            base_model_path=base_path,
            spec_ckpt_path=spec_ckpt_path,
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos = getattr(prefill_bundle.pipe.config, "eos_token_id", None)
        return prefill_bundle, tokenizer, eos, None
    worker_bundle = load_stage_rank_bundle(
        rank=rank,
        base_model_path=base_path,
        spec_ckpt_path=spec_ckpt_path,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )
    worker = StageWorker(worker_bundle, p2p)
    return None, None, None, worker


def _run_eval_for_ckpt(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    args: argparse.Namespace,
    spec_ckpt_path: str,
    indexed: list[tuple[int, UnifiedItem]],
    out_dir: Path,
    rank_gpus: list[int] | None,
) -> None:
    dtype = _resolve_dtype(str(args.dtype))
    spec_cfg = load_spec_checkpoint_config(spec_ckpt_path)
    num_stages = int(spec_cfg["num_stages"])
    pipeline_parallel_factor = pipeline_parallel_factor_from_spec_cfg(spec_cfg)
    expected_ws = expected_world_size(num_stages)
    if world_size != expected_ws:
        if rank == 0:
            raise ValueError(
                f"Checkpoint num_stages={num_stages} requires world_size={expected_ws}, "
                f"got {world_size}. Relaunch torchrun with --nproc_per_node={expected_ws}."
            )
        dist.barrier()
        raise ValueError("world_size mismatch")

    base_path = _resolve_base_model_path(spec_cfg, args.base_model_path, spec_ckpt_path)
    timeout = PhaseTimeout(
        prefill_sec=float(args.prefill_timeout_sec),
        decode_sec=float(args.decode_timeout_sec),
    )
    p2p = PipelineP2P(rank, world_size, device)
    ctrl_buf = make_ctrl_tensor(opcode=CtrlOpcode.GO, cycle_id=0, device=device)
    done_flag = torch.zeros(1, dtype=torch.int64, device=device)

    prefill_bundle, tokenizer, eos, worker = _load_models_for_ckpt(
        rank=rank,
        device=device,
        dtype=dtype,
        spec_ckpt_path=spec_ckpt_path,
        base_path=base_path,
        attn_implementation=str(args.attn_implementation),
        p2p=p2p,
    )
    dist.barrier()

    trained_deepest = bool(spec_cfg.get("trained_with_use_deepest", False))
    use_deepest = trained_deepest or bool(args.use_deepest)
    use_chat = not bool(args.no_chat_template)
    enable_thinking = bool(args.enable_thinking)
    n_total = len(indexed)
    temperatures = _normalize_temperature_list(list(args.temperature))
    draft_top_ks = _normalize_draft_top_k_list(list(args.draft_top_k))

    ckpt_tag = ckpt_to_filename_tag(spec_ckpt_path)
    base_name = f"mp_pipeline_eval__{ckpt_tag}__nt{n_total}"
    raw_dir = out_dir / "raw"
    summary_dir = out_dir / "summary"
    if rank == 0:
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

    results_meta: list[dict[str, Any]] = []

    def _broadcast_prompt_for_index(sample_index: int) -> torch.LongTensor:
        if rank == 0:
            _idx, item = indexed[sample_index]
            _set_torch_rng_for_eval_sample(int(args.seed), int(_idx))
            ids, _attn = _encode_prompt(
                tokenizer,
                item.question,
                use_chat,
                device,
                enable_thinking=enable_thinking,
            )
        else:
            ids = None
        return broadcast_input_ids(rank, device, ids)

    for temp in temperatures:
        for dtk in draft_top_ks:
            greedy = bool(float(temp) == 0.0)
            n_warmup = max(int(args.warmup_iters), 0)
            warmup_tok = max(int(args.warmup_new_tokens), 1)
            if n_warmup > 0 and n_total > 0:
                warmup_ids = _broadcast_prompt_for_index(0)
                for _ in range(n_warmup):
                    _run_one_session(
                        rank=rank,
                        device=device,
                        dtype=dtype,
                        input_ids=warmup_ids,
                        prefill_bundle=prefill_bundle,
                        worker=worker,
                        p2p=p2p,
                        ctrl_buf=ctrl_buf,
                        done_flag=done_flag,
                        timeout=timeout,
                        greedy=greedy,
                        temperature=float(temp),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        verify=bool(args.verify),
                        use_deepest=use_deepest,
                        eos_token_id=eos,
                        max_new_tokens=warmup_tok,
                        release_base_layers=False,
                        profile_timing=bool(args.profile_timing),
                    )

            if rank == 0:
                t0 = time.perf_counter()
                rows: list[dict[str, Any]] = []
                it = tqdm(
                    range(n_total),
                    total=n_total,
                    desc=f"mp pipeline T={temp} dtk={dtk}",
                    unit="sample",
                )
            else:
                it = range(n_total)

            for si in it:
                if rank == 0:
                    idx, item = indexed[si]
                    input_ids, _attn = _encode_prompt(
                        tokenizer,
                        item.question,
                        use_chat,
                        device,
                        enable_thinking=enable_thinking,
                    )
                    _set_torch_rng_for_eval_sample(int(args.seed), int(idx))
                    prompt_len = int(input_ids.shape[1])
                else:
                    input_ids = None
                    item = None
                    idx = si
                    prompt_len = 0

                input_ids = broadcast_input_ids(rank, device, input_ids)

                result = _run_one_session(
                    rank=rank,
                    device=device,
                    dtype=dtype,
                    input_ids=input_ids,
                    prefill_bundle=prefill_bundle,
                    worker=worker,
                    p2p=p2p,
                    ctrl_buf=ctrl_buf,
                    done_flag=done_flag,
                    timeout=timeout,
                    greedy=greedy,
                    temperature=float(temp),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    verify=bool(args.verify),
                    use_deepest=use_deepest,
                    eos_token_id=eos,
                    max_new_tokens=int(args.max_new_tokens),
                    release_base_layers=False,
                    profile_timing=bool(args.profile_timing),
                )

                if rank == 0:
                    session, out_ids, accept = result
                    gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    rows.append(
                        _sample_row(
                            idx=int(idx),
                            it=item,
                            session=session,
                            out_ids=out_ids,
                            accept=accept,
                            prompt_len=prompt_len,
                            num_stages=num_stages,
                            gen_text=gen_text,
                            temperature=float(temp),
                            draft_top_k=int(dtk),
                            use_deepest=use_deepest,
                            pipeline_parallel_factor=pipeline_parallel_factor,
                            profile_timing=bool(args.profile_timing),
                        )
                    )

            if rank == 0:
                elapsed = time.perf_counter() - t0
                stem = _per_sample_file_stem(
                    base_name,
                    float(temp),
                    int(dtk),
                    temperatures=temperatures,
                    draft_top_ks=draft_top_ks,
                )
                per_sample_path = raw_dir / f"{stem}.jsonl"
                with per_sample_path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(
                            json.dumps(
                                _round_sample_row_timing(row),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                overall = _round_aggregate_timing(
                    _aggregate_mp_pipeline(
                        rows,
                        None,
                        profile_timing=bool(args.profile_timing),
                    )
                )
                per_dataset = {
                    ds: _round_aggregate_timing(
                        _aggregate_mp_pipeline(
                            rows,
                            ds,
                            profile_timing=bool(args.profile_timing),
                        )
                    )
                    for ds, _ in DATASET_CONFIG
                }
                result_entry: dict[str, Any] = {
                    "temperature": float(temp),
                    "greedy": greedy,
                    "draft_top_k": int(dtk),
                    "use_deepest": bool(use_deepest),
                    "enable_thinking": enable_thinking,
                    "profile_timing": bool(args.profile_timing),
                    "total_wall_sec": _round_timing_sec(elapsed),
                    "overall": overall,
                    "per_dataset": per_dataset,
                    "per_sample_path": str(per_sample_path),
                }
                if args.profile_timing:
                    result_entry["overall_timing_profile"] = _round_timing_dict(
                        _aggregate_timing_profile(rows, None)
                    )
                    result_entry["per_dataset_timing_profile"] = {
                        ds: _round_timing_dict(_aggregate_timing_profile(rows, ds))
                        for ds, _ in DATASET_CONFIG
                    }
                results_meta.append(result_entry)
                print(f"\n--- temperature={temp} draft_top_k={dtk} ---")
                _print_aggregates(
                    rows,
                    elapsed,
                    profile_timing=bool(args.profile_timing),
                )
                print(f"Wrote: {per_sample_path}")

    if rank == 0:
        summary_path = summary_dir / f"{base_name}__summary.json"
        summary: dict[str, Any] = {
            "checkpoint_path": spec_ckpt_path,
            "model": base_path,
            "dtype": str(args.dtype),
            "data_dir": str(Path(args.data_dir).resolve()),
            "prompts_per_dataset": (
                int(args.prompts_per_dataset)
                if args.prompts_per_dataset is not None
                else None
            ),
            "num_prompts": n_total,
            "seed": int(args.seed),
            "max_new_tokens": int(args.max_new_tokens),
            "verify": bool(args.verify),
            "use_deepest": bool(use_deepest),
            "enable_thinking": enable_thinking,
            "num_stages": num_stages,
            "theoretical_pipeline_parallel_factor": float(pipeline_parallel_factor),
            "world_size": world_size,
            "rank_gpus": rank_gpus,
            "pipeline_implementation": "distributed_inference_v11",
            "profile_timing": bool(args.profile_timing),
            "temperatures_evaluated": temperatures,
            "draft_top_ks_evaluated": draft_top_ks,
            "note_draft_top_k": "multi-process v11 does not use draft_top_k at decode time",
            "results": results_meta,
            "total_wall_sec": _round_timing_sec(
                float(sum(float(r["total_wall_sec"]) for r in results_meta))
            ),
        }
        if len(draft_top_ks) == 1:
            summary["draft_top_k"] = int(draft_top_ks[0])
        if len(results_meta) == 1:
            r0 = results_meta[0]
            summary["temperature"] = r0["temperature"]
            summary["greedy"] = r0["greedy"]
            summary["overall"] = r0["overall"]
            summary["per_dataset"] = r0["per_dataset"]
            summary["per_sample_path"] = r0["per_sample_path"]
            summary["total_wall_sec"] = r0["total_wall_sec"]
            if "overall_timing_profile" in r0:
                summary["overall_timing_profile"] = r0["overall_timing_profile"]
            if "per_dataset_timing_profile" in r0:
                summary["per_dataset_timing_profile"] = r0[
                    "per_dataset_timing_profile"
                ]
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {summary_path}")

    dist.barrier()


def _distributed_eval_main(args: argparse.Namespace) -> None:
    ckpt_list = _normalize_spec_head_ckpt_list(args.spec_head_ckpt)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    rank_gpus = parse_rank_gpu_ids(args.rank_gpus) if str(args.rank_gpus).strip() else None
    rank, world_size, device = init_dist_rank_device(
        rank_gpus,
        init_timeout_minutes=max(float(args.prefill_timeout_sec) / 60.0, 1.0),
    )

    matching_ckpts, deferred_ckpts = _split_ckpts_by_world_size(
        ckpt_list,
        world_size,
    )
    if rank == 0 and deferred_ckpts:
        if _is_single_node_job():
            print(
                f"\nNote: {len(deferred_ckpts)} checkpoint(s) need a different world_size "
                f"than the current torchrun ({world_size}); "
                f"{len(matching_ckpts)} will run in-process, "
                f"{len(deferred_ckpts)} via subprocess relaunch after."
            )
            for ckpt in deferred_ckpts:
                need_ws = _expected_world_size_for_ckpt(ckpt)
                print(f"  deferred: {ckpt} (needs world_size={need_ws})")
        else:
            print(
                f"\nWarning: skipping {len(deferred_ckpts)} checkpoint(s) on multi-node "
                f"because world_size != {world_size}. Relaunch each separately:",
                flush=True,
            )
            for ckpt in deferred_ckpts:
                need_ws = _expected_world_size_for_ckpt(ckpt)
                print(f"  skipped: {ckpt} (needs world_size={need_ws})", flush=True)
            deferred_ckpts = []

    indexed: list[tuple[int, UnifiedItem]] = []
    if rank == 0:
        all_items = _load_all_items(
            data_dir,
            prompts_per_dataset=args.prompts_per_dataset,
        )
        indexed = list(enumerate(all_items))
        n_total_t = torch.tensor([len(indexed)], dtype=torch.int64, device=device)
        limit = args.prompts_per_dataset
        print(
            f"Loaded {n_total_t.item()} prompts "
            f"(prompts_per_dataset={'all' if limit is None else int(limit)})"
        )
    else:
        n_total_t = torch.zeros(1, dtype=torch.int64, device=device)
    dist.broadcast(n_total_t, src=0)
    n_total = int(n_total_t.item())

    if rank != 0:
        indexed = [(i, UnifiedItem("", i, "")) for i in range(n_total)]

    for ckpt_idx, ckpt_path in enumerate(matching_ckpts):
        if rank == 0 and len(matching_ckpts) > 1:
            print(f"\n=== Checkpoint {ckpt_idx + 1}/{len(matching_ckpts)}: {ckpt_path} ===\n")
        _run_eval_for_ckpt(
            rank=rank,
            world_size=world_size,
            device=device,
            args=args,
            spec_ckpt_path=ckpt_path,
            indexed=indexed,
            out_dir=out_dir,
            rank_gpus=rank_gpus,
        )

    dist.destroy_process_group()

    if rank == 0 and deferred_ckpts and _is_single_node_job():
        deferred_groups = _group_consecutive_ckpts_by_world_size(deferred_ckpts)
        for group_idx, (ws, ckpts) in enumerate(deferred_groups):
            print(
                f"\n=== Subprocess relaunch: group {group_idx + 1}/{len(deferred_groups)} "
                f"world_size={ws} ({len(ckpts)} checkpoint"
                f"{'s' if len(ckpts) != 1 else ''}) ===\n"
            )
            _launch_eval_subprocess(args, ckpts, rank_gpus)


if __name__ == "__main__":
    args = parse_args()
    ckpt_list = _normalize_spec_head_ckpt_list(args.spec_head_ckpt)

    if not bool(args._no_auto_launch) and not _is_distributed_launched():
        _launcher_main(args)
    elif _needs_parent_sequential_launch(args, ckpt_list):
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "\nNote: parent torchrun world_size does not match all checkpoints; "
                "rank 0 will launch fresh torchrun subprocesses in list order."
            )
            _launcher_main(args)
    else:
        _distributed_eval_main(args)
