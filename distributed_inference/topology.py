"""Rank ↔ stage mapping for multi-process pipeline v11.

Fixed topology (world_size = num_stages + 1):
  - rank 0: speculation / controller only
  - ranks 1..num_stages: stages 0..num_stages-1
  - last stage (rank num_stages) sends snap + verify_hs to rank 0
"""

from __future__ import annotations


def expected_world_size(num_stages: int) -> int:
    return int(num_stages) + 1


def stage_idx_for_rank(rank: int, num_stages: int) -> int | None:
    r = int(rank)
    n = int(num_stages)
    if r == 0:
        return None
    if 1 <= r <= n:
        return r - 1
    raise ValueError(f"rank {r} invalid for num_stages={n}")


def rank_for_stage(stage_idx: int, num_stages: int) -> int:
    si = int(stage_idx)
    n = int(num_stages)
    if si < 0 or si >= n:
        raise ValueError(f"stage_idx {si} out of range [0, {n})")
    return si + 1


def hs_send_dst_rank(rank: int, world_size: int) -> int:
    return int(rank) + 1


def hs_recv_src_rank(rank: int, world_size: int) -> int:
    return int(rank) - 1
