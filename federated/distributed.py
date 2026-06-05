from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_ddp(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: str | int = "29500",
    backend: str = "nccl",
    timeout_seconds: int | None = None,
) -> None:
    if world_size <= 1:
        return

    if is_dist_ready():
        return

    kwargs: dict[str, Any] = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "init_method": f"tcp://{master_addr}:{master_port}",
    }

    if timeout_seconds is not None:
        kwargs["timeout"] = timedelta(seconds=timeout_seconds)

    dist.init_process_group(**kwargs)


def cleanup_ddp() -> None:
    if is_dist_ready():
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist_ready():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_dist_ready():
        return obj

    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def broadcast_state_dict(
    state_dict: dict[str, torch.Tensor] | None,
    device: str | torch.device,
    src: int = 0,
) -> dict[str, torch.Tensor]:
    if not is_dist_ready():
        if state_dict is None:
            raise ValueError("state_dict cannot be None when distributed is disabled.")
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    rank = dist.get_rank()
    device = torch.device(device)

    metadata = None
    if rank == src:
        if state_dict is None:
            raise ValueError("Source rank must provide a state_dict.")

        metadata = [
            (key, tuple(value.shape), value.dtype)
            for key, value in state_dict.items()
        ]

    metadata = broadcast_object(metadata, src=src)

    received: dict[str, torch.Tensor] = {}
    for key, shape, dtype in metadata:
        if rank == src:
            assert state_dict is not None
            tensor = state_dict[key].detach().to(device=device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)

        dist.broadcast(tensor, src=src)
        received[key] = tensor.detach().cpu().clone()

    return received
