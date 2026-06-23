"""Rank ↔ stage mapping for multi-process pipeline v11."""

from __future__ import annotations


def expected_world_size(num_stages: int, *, merge_last_stage: bool) -> int:
    n = int(num_stages)
    return n if merge_last_stage else n + 1


def stage_idx_for_rank(
    rank: int, num_stages: int, *, merge_last_stage: bool
) -> int | None:
    r = int(rank)
    n = int(num_stages)
    if merge_last_stage:
        if r == 0:
            return n - 1
        if 1 <= r < n:
            return r - 1
        raise ValueError(f"rank {r} invalid for num_stages={n} merge_last_stage=True")
    if r == 0:
        return None
    if 1 <= r <= n:
        return r - 1
    raise ValueError(f"rank {r} invalid for num_stages={n} merge_last_stage=False")


def rank_for_stage(
    stage_idx: int, num_stages: int, *, merge_last_stage: bool
) -> int:
    si = int(stage_idx)
    n = int(num_stages)
    if si < 0 or si >= n:
        raise ValueError(f"stage_idx {si} out of range [0, {n})")
    if merge_last_stage and si == n - 1:
        return 0
    return si + 1


def hs_send_dst_rank(
    rank: int, world_size: int, *, merge_last_stage: bool
) -> int:
    r = int(rank)
    ws = int(world_size)
    if merge_last_stage and r == ws - 1:
        return 0
    return r + 1


def hs_recv_src_rank(
    rank: int, world_size: int, *, merge_last_stage: bool
) -> int:
    r = int(rank)
    ws = int(world_size)
    if merge_last_stage and r == 0:
        return ws - 1
    return r - 1
