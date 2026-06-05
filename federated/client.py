from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils import save_config, set_seed

from .config import ClientConfig, build_task


@dataclass
class ClientResult:
    name: str
    task_name: str
    round_idx: int
    num_examples: float
    state_dict: dict[str, torch.Tensor] | None
    summary: dict[str, Any]
    history: list[dict[str, Any]]


def _close_loader(loader) -> None:
    if loader is not None and hasattr(loader.dataset, "close"):
        loader.dataset.close()


def _local_task_extra_args(
    *,
    base_task_extra_args: dict[str, Any],
    local_epochs: int,
    keep_client_ckpts: bool,
    local_keep_best: bool,
    local_early_stopping: bool,
) -> dict[str, Any]:
    task_extra_args = dict(base_task_extra_args)
    task_extra_args["epochs"] = local_epochs
    task_extra_args["save_ckpt"] = keep_client_ckpts

    task_extra_args.setdefault("keep_best", local_keep_best)
    task_extra_args.setdefault("early_stopping", local_early_stopping)

    return task_extra_args


def train_client(
    *,
    model_name: str,
    client: ClientConfig,
    model_extra_args: dict[str, Any],
    task_extra_args: dict[str, Any],
    global_state_dict: dict[str, torch.Tensor],
    round_idx: int,
    local_epochs: int,
    keep_client_ckpts: bool,
    local_keep_best: bool,
    local_early_stopping: bool,
    save_dir: Path,
    device: str,
    rank: int,
    world_size: int,
    seed: int,
) -> ClientResult:
    task = None
    train_loader = None
    val_loader = None
    test_loader = None
    trainer = None
    optimizer = None
    lr_scheduler = None
    wd_scheduler = None

    try:
        client_seed = seed + (round_idx * 1000)
        set_seed(client_seed)

        local_task_extra_args = _local_task_extra_args(
            base_task_extra_args=task_extra_args,
            local_epochs=local_epochs,
            keep_client_ckpts=keep_client_ckpts,
            local_keep_best=local_keep_best,
            local_early_stopping=local_early_stopping,
        )

        task = build_task(
            model_name=model_name,
            client=client,
            model_extra_args=model_extra_args,
            task_extra_args=local_task_extra_args,
            world_size=world_size,
        )

        train_loader, val_loader, test_loader = task.make_loaders(
            seed=client_seed,
            rank=rank,
            world_size=world_size,
        )

        trainer = task.make_trainer(
            save_dir=save_dir,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        trainer.load_state_dict(global_state_dict)

        optimizer, lr_scheduler, wd_scheduler = task.make_optimizer(
            trainer=trainer,
            train_loader=train_loader,
        )

        if rank == 0:
            save_config(
                {
                    "client": client,
                    "round_idx": round_idx,
                    "local_epochs": local_epochs,
                    "task": task,
                },
                save_dir / "client_config.json",
            )

        fit_kwargs = task.fit_kwargs()
        fit_kwargs["ckpt_name"] = "client_model.pt"

        train_result = trainer.fit(
            train_loader=train_loader,
            criterion=task.criterion,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            **fit_kwargs,
        )

        last_record = train_result["summary"].get("last_record") or {}
        num_examples = float(last_record.get("train_n", len(train_loader.dataset)))

        return ClientResult(
            name=client.name,
            task_name=client.task_name,
            round_idx=round_idx,
            num_examples=num_examples,
            state_dict=train_result["state_dict"] if rank == 0 else None,
            summary=train_result["summary"],
            history=train_result["history"],
        )

    finally:
        _close_loader(train_loader)
        _close_loader(val_loader)
        _close_loader(test_loader)

        del task, train_loader, val_loader, test_loader
        del trainer, optimizer, lr_scheduler, wd_scheduler

        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_client(
    *,
    model_name: str,
    client: ClientConfig,
    model_extra_args: dict[str, Any],
    task_extra_args: dict[str, Any],
    global_state_dict: dict[str, torch.Tensor],
    save_dir: Path,
    device: str,
    rank: int,
    world_size: int,
    seed: int,
) -> dict[str, Any] | None:
    task = None
    val_loader = None
    test_loader = None
    trainer = None

    try:
        set_seed(seed)

        task = build_task(
            model_name=model_name,
            client=client,
            model_extra_args=model_extra_args,
            task_extra_args=task_extra_args,
            world_size=world_size,
        )

        val_loader = task.make_loader(
            group="val",
            n_samples=client.n_val_samples,
            shuffle=False,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        test_loader = task.make_loader(
            group="test",
            n_samples=client.n_test_samples,
            shuffle=False,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

        trainer = task.make_trainer(
            save_dir=save_dir,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        trainer.load_state_dict(global_state_dict)

        summary = task.evaluate(
            trainer=trainer,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        if rank == 0:
            save_config(summary, save_dir / "eval_summary.json")
            return summary

        return None

    finally:
        _close_loader(val_loader)
        _close_loader(test_loader)

        del task, val_loader, test_loader, trainer

        gc.collect()
        torch.cuda.empty_cache()
