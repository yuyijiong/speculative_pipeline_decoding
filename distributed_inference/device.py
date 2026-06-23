"""Per-rank CUDA device selection and decode/prefill timeout helpers."""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn


def module_compute_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def cast_module(module: nn.Module, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    return module.to(device=device, dtype=dtype)


def parse_rank_gpu_ids(s: str) -> list[int]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in parts]


def init_dist_rank_device(
    rank_gpu_ids: list[int] | None = None,
    *,
    init_timeout_minutes: float = 30.0,
    backend: str = "nccl",
) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-process pipeline decoding.")

    env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank_gpu_ids is not None:
        if env_rank >= len(rank_gpu_ids):
            raise ValueError(
                f"--rank_gpus length ({len(rank_gpu_ids)}) must equal world_size and "
                f"be indexed by global rank, but global rank={env_rank}. "
                f"Do not pass per-node lists (e.g. only 0,1 on the second node). "
                f"Either omit --rank_gpus and rely on CUDA_VISIBLE_DEVICES + LOCAL_RANK, "
                f"or pass the same full mapping on every node, e.g. "
                f"--rank_gpus 0,1,2,3,4,5,6,0,1 for 7 processes on node0 and 2 on node1."
            )
        gpu_id = int(rank_gpu_ids[env_rank])
    elif "LOCAL_RANK" in os.environ:
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = env_rank

    n_cuda = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= n_cuda:
        raise ValueError(
            f"rank {env_rank} mapped to cuda:{gpu_id}, but device_count={n_cuda}. "
            f"Set --rank_gpus or CUDA_VISIBLE_DEVICES so each process gets a unique GPU."
        )

    torch.cuda.set_device(gpu_id)

    if not dist.is_initialized():
        init_kwargs = {
            "backend": str(backend),
            "timeout": timedelta(minutes=float(init_timeout_minutes)),
        }
        if str(backend) == "nccl":
            init_kwargs["device_id"] = torch.device(f"cuda:{gpu_id}")
        dist.init_process_group(**init_kwargs)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank_gpu_ids is not None:
        if len(rank_gpu_ids) != world_size:
            raise ValueError(
                f"--rank_gpus length ({len(rank_gpu_ids)}) must equal world_size ({world_size})."
            )
        gpu_id = int(rank_gpu_ids[rank])
        torch.cuda.set_device(gpu_id)

    device = torch.device(f"cuda:{gpu_id}")
    return rank, world_size, device


class PhaseTimeout:
    """Wall-clock guard for prefill vs decode (distinct limits per architecture doc)."""

    def __init__(self, *, prefill_sec: float, decode_sec: float) -> None:
        self.prefill_sec = float(prefill_sec)
        self.decode_sec = float(decode_sec)
        self._phase = "prefill"
        self._t0 = time.perf_counter()

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)
        self._t0 = time.perf_counter()

    def check(self) -> None:
        elapsed = time.perf_counter() - self._t0
        limit = self.prefill_sec if self._phase == "prefill" else self.decode_sec
        if elapsed > limit:
            raise TimeoutError(
                f"{self._phase} phase exceeded {limit:.1f}s timeout (elapsed {elapsed:.1f}s)."
            )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
