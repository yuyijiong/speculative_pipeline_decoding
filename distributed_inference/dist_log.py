"""Rank-labeled diagnostic prints for multi-process pipeline debugging.

Enabled only when environment variable ``DEBUG=1``.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def dist_log(msg: str) -> None:
    if os.environ.get("DEBUG") != "1":
        return
    host = socket.gethostname()
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        rank_s = f"rank={rank}/{world_size}"
    else:
        rank_s = "rank=?"
    local_rank = os.environ.get("LOCAL_RANK", "?")
    node_rank = os.environ.get("NODE_RANK", "?")
    cuda_dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(
        f"[dist][{host}] {rank_s} local_rank={local_rank} "
        f"node_rank={node_rank} cuda={cuda_dev} {msg}",
        flush=True,
    )
