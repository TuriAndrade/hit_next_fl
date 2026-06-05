from __future__ import annotations
from typing import Optional

import os
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import (
    canonical_dataset_name,
    get_dataset_paths,
    save_config,
    set_seed,
    parse_args_json,
)

from datetime import timedelta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised ECG training.")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--task-name", type=str, default="clf")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--model-extra-args", type=json.loads, default={})
    parser.add_argument("--task-extra-args", type=json.loads, default={})
    parser.add_argument("--n-train-samples", type=int, default=None)
    parser.add_argument("--n-val-samples", type=int, default=None)
    parser.add_argument("--n-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-name", type=str, default=None)

    # DDP
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def _build_task(args, world_size: int):
    from tasks import tasks

    if args.model_name not in tasks:
        raise ValueError(
            f"Invalid model_name: {args.model_name}. "
            f"Available models: {list(tasks.keys())}"
        )

    model_tasks = tasks[args.model_name]

    if args.task_name not in model_tasks:
        raise ValueError(
            f"Invalid task_name: {args.task_name} for model "
            f"{args.model_name}. Available tasks: {list(model_tasks.keys())}"
        )

    h5_path, csv_path = get_dataset_paths(args.dataset_name)

    return model_tasks[args.task_name](
        h5_path=h5_path,
        csv_path=csv_path,
        model_extra_args=args.model_extra_args,
        task_extra_args=args.task_extra_args,
        n_train_samples=args.n_train_samples,
        n_val_samples=args.n_val_samples,
        n_test_samples=args.n_test_samples,
        world_size=world_size,
    )


def _close_loader(loader) -> None:
    if loader is not None and hasattr(loader.dataset, "close"):
        loader.dataset.close()


def _setup_ddp(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: str | int = "29500",
    backend: str = "nccl",
    timeout_seconds: Optional[int] = None,
) -> None:
    if world_size <= 1:
        return

    if dist.is_available() and dist.is_initialized():
        return

    kwargs = dict(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )

    if timeout_seconds is not None:
        kwargs["timeout"] = timedelta(seconds=timeout_seconds)

    dist.init_process_group(**kwargs)


def _cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _train_worker(rank: int, world_size: int, args) -> None:
    torch.cuda.set_device(rank)

    device = f"cuda:{rank}"
    dataset_name = canonical_dataset_name(args.dataset_name)
    save_dir = (
        Path(os.environ.get("SAVE_DIR", "experiments"))
        / f"{args.model_name}-{dataset_name}-{args.task_name}-seed_{args.seed}"
    )

    if args.save_name is not None:
        save_dir = save_dir / args.save_name

    task = None
    train_loader = None
    val_loader = None
    test_loader = None
    trainer = None
    optimizer = None
    lr_scheduler = None
    wd_scheduler = None

    if world_size > 1:
        _setup_ddp(
            rank=rank,
            world_size=world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )

    try:
        set_seed(args.seed)

        task = _build_task(args, world_size)
        accum_steps = task.compute_accum_steps()

        train_loader, val_loader, test_loader = task.make_loaders(
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )

        trainer = task.make_trainer(
            save_dir=save_dir,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        optimizer, lr_scheduler, wd_scheduler = task.make_optimizer(
            trainer=trainer,
            train_loader=train_loader,
        )

        if rank == 0:
            save_config(
                config={
                    "args": vars(args),
                    "task": vars(task),
                    "runtime": {
                        "world_size": world_size,
                        "accum_steps": accum_steps,
                    },
                },
                path=save_dir / "config.json",
            )

        train_result = trainer.fit(
            train_loader=train_loader,
            criterion=task.criterion,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            **task.fit_kwargs(),
        )

        eval_summary = task.evaluate(
            trainer=trainer,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        if rank == 0:
            save_config(eval_summary, save_dir / "eval_summary.json")
            print(f"Finished seed {args.seed}: {train_result['summary']}")
            print(f"Evaluation: {eval_summary}")

    finally:
        _close_loader(train_loader)
        _close_loader(val_loader)
        _close_loader(test_loader)

        del task, train_loader, val_loader, test_loader
        del trainer, optimizer, lr_scheduler, wd_scheduler

        gc.collect()
        torch.cuda.empty_cache()

        if world_size > 1:
            _cleanup_ddp()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs found.")

    print(f"Using {world_size} GPU(s)")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Dataset: {canonical_dataset_name(args.dataset_name)}")
    print(f"Seed: {args.seed}")

    if world_size == 1:
        _train_worker(rank=0, world_size=1, args=args)
    else:
        mp.spawn(
            _train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
