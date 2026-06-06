from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from tasks import TaskDefinition
from utils import save_config, set_seed

from .config import ClientConfig


@dataclass
class ClientUpdate:
    name: str
    round_idx: int
    num_examples: float
    state_dict: dict[str, torch.Tensor] | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class ClientValidation:
    name: str
    loss: float
    num_examples: float


def _close_loader(loader) -> None:
    if loader is not None and hasattr(loader.dataset, "close"):
        loader.dataset.close()


class FederatedClient:
    def __init__(
        self,
        *,
        config: ClientConfig,
        task_definition: TaskDefinition,
        model_extra_args: dict[str, Any],
        task_extra_args: dict[str, Any],
        device: str,
        rank: int,
        world_size: int,
    ):
        self.config = config
        self.task_definition = task_definition
        self.model_extra_args = dict(model_extra_args)
        self.task_extra_args = dict(task_extra_args)
        self.device = device
        self.rank = rank
        self.world_size = world_size

    @property
    def name(self) -> str:
        return self.config.name

    def _create_task(self, *, epochs: int | None = None):
        task_extra_args = dict(self.task_extra_args)
        if epochs is not None:
            task_extra_args["epochs"] = epochs

        model = self.task_definition.create_model(self.model_extra_args)
        return self.task_definition.create_task(
            model=model,
            h5_path=self.config.h5_path,
            csv_path=self.config.csv_path,
            task_extra_args=task_extra_args,
            n_train_samples=self.config.n_train_samples,
            n_val_samples=self.config.n_val_samples,
            n_test_samples=self.config.n_test_samples,
            world_size=self.world_size,
        )

    def train(
        self,
        *,
        global_state_dict: dict[str, torch.Tensor],
        round_idx: int,
        local_epochs: int,
        keep_checkpoint: bool,
        save_dir: Path,
        seed: int,
    ) -> ClientUpdate:
        task = None
        train_loader = None
        val_loader = None
        trainer = None
        optimizer = None
        lr_scheduler = None
        wd_scheduler = None

        try:
            client_seed = seed + round_idx * 1000
            set_seed(client_seed)

            task = self._create_task(epochs=local_epochs)
            train_loader = task.make_loader(
                group="train",
                n_samples=self.config.n_train_samples,
                shuffle=True,
                seed=client_seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            val_loader = task.make_loader(
                group="val",
                n_samples=self.config.n_val_samples,
                shuffle=False,
                seed=client_seed,
                rank=self.rank,
                world_size=self.world_size,
            )

            trainer = task.make_trainer(
                save_dir=save_dir,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
            )
            trainer.load_state_dict(global_state_dict)

            optimizer, lr_scheduler, wd_scheduler = task.make_optimizer(
                trainer=trainer,
                train_loader=train_loader,
            )

            if self.rank == 0:
                save_config(
                    {
                        "client": self.config,
                        "round": round_idx,
                        "local_epochs": local_epochs,
                        "task": task,
                    },
                    save_dir / "client_config.json",
                )

            fit_kwargs = task.fit_kwargs()
            fit_kwargs.update(
                epochs=local_epochs,
                save_ckpt=keep_checkpoint,
                ckpt_name="client_model.pt",
            )
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
            num_examples = float(
                last_record.get("train_n", len(train_loader.dataset))
            )

            return ClientUpdate(
                name=self.name,
                round_idx=round_idx,
                num_examples=num_examples,
                state_dict=(
                    train_result["state_dict"] if self.rank == 0 else None
                ),
                summary=train_result["summary"],
            )

        finally:
            _close_loader(train_loader)
            _close_loader(val_loader)

            del task, train_loader, val_loader, trainer
            del optimizer, lr_scheduler, wd_scheduler

            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(
        self,
        *,
        global_state_dict: dict[str, torch.Tensor],
        save_dir: Path,
        seed: int,
    ) -> ClientValidation:
        task = None
        val_loader = None
        trainer = None

        try:
            set_seed(seed)
            task = self._create_task()
            val_loader = task.make_loader(
                group="val",
                n_samples=self.config.n_val_samples,
                shuffle=False,
                seed=seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            trainer = task.make_trainer(
                save_dir=save_dir,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
            )
            trainer.load_state_dict(global_state_dict)
            trainer.model.eval()

            loss_sum = 0.0
            num_examples = 0

            for batch in val_loader:
                x = batch[0].to(trainer.device, non_blocking=True)
                y = batch[1].to(trainer.device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=task.use_amp):
                    loss = task.criterion(trainer.model(x), y)

                if loss.ndim > 0:
                    loss = loss.mean()

                batch_size = y.size(0)
                loss_sum += float(loss.item()) * batch_size
                num_examples += batch_size

            if trainer.ddp:
                totals = torch.tensor(
                    [loss_sum, float(num_examples)],
                    dtype=torch.float64,
                    device=trainer.device,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                loss_sum = float(totals[0].item())
                num_examples = int(totals[1].item())

            if num_examples == 0:
                raise RuntimeError(
                    f"Validation loader for client {self.name} is empty."
                )

            return ClientValidation(
                name=self.name,
                loss=loss_sum / num_examples,
                num_examples=float(num_examples),
            )

        finally:
            _close_loader(val_loader)
            del task, val_loader, trainer

            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(
        self,
        *,
        global_state_dict: dict[str, torch.Tensor],
        save_dir: Path,
        seed: int,
    ) -> dict[str, Any] | None:
        task = None
        val_loader = None
        test_loader = None
        trainer = None

        try:
            set_seed(seed)
            task = self._create_task()
            val_loader = task.make_loader(
                group="val",
                n_samples=self.config.n_val_samples,
                shuffle=False,
                seed=seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            test_loader = task.make_loader(
                group="test",
                n_samples=self.config.n_test_samples,
                shuffle=False,
                seed=seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            trainer = task.make_trainer(
                save_dir=save_dir,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
            )
            trainer.load_state_dict(global_state_dict)

            summary = task.evaluate(
                trainer=trainer,
                val_loader=val_loader,
                test_loader=test_loader,
            )

            if self.rank == 0:
                save_config(summary, save_dir / "eval_summary.json")
                return summary

            return None

        finally:
            _close_loader(val_loader)
            _close_loader(test_loader)
            del task, val_loader, test_loader, trainer

            gc.collect()
            torch.cuda.empty_cache()
