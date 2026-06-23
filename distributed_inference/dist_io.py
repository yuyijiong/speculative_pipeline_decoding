"""torch.distributed send/recv/broadcast with CPU staging for the gloo backend."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import Work


def uses_cpu_dist_staging() -> bool:
    return dist.is_initialized() and dist.get_backend() == "gloo"


def _needs_cpu_staging(tensor: torch.Tensor) -> bool:
    return uses_cpu_dist_staging() and tensor.device.type == "cuda"


class _GlooRecvWork:
    def __init__(self, work: Work, cpu_buf: torch.Tensor, dst: torch.Tensor) -> None:
        self._work = work
        self._cpu_buf = cpu_buf
        self._dst = dst

    def wait(self) -> bool:
        self._work.wait()
        self._dst.copy_(self._cpu_buf)
        return True


_send_staging: list[torch.Tensor] = []


def clear_send_staging() -> None:
    _send_staging.clear()


def dist_send(tensor: torch.Tensor, *, dst: int) -> None:
    if _needs_cpu_staging(tensor):
        payload = tensor.detach().cpu().contiguous()
        dist.send(payload, dst=dst)
        return
    dist.send(tensor.contiguous(), dst=dst)


def dist_recv(tensor: torch.Tensor, *, src: int) -> None:
    if _needs_cpu_staging(tensor):
        cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        dist.recv(cpu_buf, src=src)
        tensor.copy_(cpu_buf)
        return
    dist.recv(tensor, src=src)


def dist_broadcast(tensor: torch.Tensor, *, src: int) -> None:
    if not _needs_cpu_staging(tensor):
        dist.broadcast(tensor, src=src)
        return
    if dist.get_rank() == src:
        cpu_t = tensor.detach().cpu().contiguous()
    else:
        cpu_t = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    dist.broadcast(cpu_t, src=src)
    if dist.get_rank() != src:
        tensor.copy_(cpu_t)


def dist_isend(tensor: torch.Tensor, *, dst: int) -> Work:
    if _needs_cpu_staging(tensor):
        payload = tensor.detach().cpu().contiguous()
        _send_staging.append(payload)
        return dist.isend(payload, dst=dst)
    return dist.isend(tensor.contiguous(), dst=dst)


def dist_irecv(tensor: torch.Tensor, *, src: int) -> Work:
    if _needs_cpu_staging(tensor):
        cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        return _GlooRecvWork(dist.irecv(cpu_buf, src=src), cpu_buf, tensor)
    return dist.irecv(tensor, src=src)
