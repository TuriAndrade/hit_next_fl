from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Supervised trainer
# -----------------------------------------------------------------------------
class SupervisedTrainer:
    """
    Supervised trainer with optional GPU DDP.

    Assumptions:
    - training runs on CUDA only;
    - each batch have format (x, y, ...);
    - criterion is a callable receiving criterion(model(x), y);
    - the trainer monitors loss only;
    - if validation is provided, best checkpoint is selected by val loss;
    - otherwise, last checkpoint is used;
    - in DDP, only rank 0 writes files and prints progress.
    """

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def setup_ddp(
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

    @staticmethod
    def cleanup_ddp() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def is_dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        save_dir: str | Path,
        device: str | torch.device,
        rank: int = 0,
        world_size: int = 1,
        ddp: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.ddp = ddp and world_size > 1
        self.device = torch.device(device)
        self.criterion = criterion

        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError(
                "SupervisedTrainer requires an explicit CUDA device, e.g. 'cuda:0'."
            )

        torch.cuda.set_device(self.device)

        self.save_dir = Path(save_dir)
        if self.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.ddp:
            if not self.is_dist_ready():
                raise RuntimeError(
                    "ddp=True requires an initialized process group. "
                    "Call SupervisedTrainer.setup_ddp(...) before creating the trainer."
                )

            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(self.device)

        if self.ddp:
            model = DDP(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )

        self.model = model

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def raw_model(self) -> nn.Module:
        return self.unwrap_model(self.model)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.raw_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        self.raw_model.load_state_dict(state_dict, strict=strict)

    def _autocast_context(self, use_amp: bool):
        return torch.autocast(device_type="cuda", enabled=use_amp)

    def _snapshot_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone() for k, v in self.raw_model.state_dict().items()
        }

    def _save_json(self, obj: Dict[str, Any], filename: str) -> None:
        if not self.is_main_process:
            return

        with open(self.save_dir / filename, "w") as f:
            json.dump(obj, f, indent=2)

    def _save_checkpoint(
        self,
        filename: str,
        state_dict: Dict[str, torch.Tensor],
        history: list[dict[str, Any]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Any = None,
        wd_scheduler: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.is_main_process:
            return

        checkpoint: Dict[str, Any] = {
            "model": state_dict,
            "history": history,
        }

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
            checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        if wd_scheduler is not None and hasattr(wd_scheduler, "state_dict"):
            checkpoint["wd_scheduler"] = wd_scheduler.state_dict()
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, self.save_dir / filename)

    def _reduce_loss_sum(self, loss_sum: float, n_samples: int) -> tuple[float, int]:
        if not self.ddp:
            return loss_sum, n_samples

        tensor = torch.tensor(
            [loss_sum, float(n_samples)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return float(tensor[0].item()), int(tensor[1].item())

    def _run_epoch(
        self,
        loader,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Any,
        wd_scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        use_amp: bool,
        grad_clip_norm: float | None,
        epoch: int,
        train: bool,
        accum_steps: int,
    ) -> Dict[str, float]:
        phase = "train" if train else "val"
        self.model.train(mode=train)

        if train and optimizer is None:
            raise ValueError("optimizer cannot be None during training.")

        loss_sum = 0.0
        n_samples = 0
        latest_lr = None
        latest_wd = None

        if train:
            optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            loader,
            desc=f"[{phase}] epoch {epoch}",
            unit="batch",
            leave=False,
            disable=not self.is_main_process,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        n_batches = len(loader)

        with torch.set_grad_enabled(train):
            for step, batch in enumerate(pbar):
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)

                should_step = (step + 1) % accum_steps == 0 or (step + 1) == n_batches

                if self.ddp and train and not should_step:
                    sync_context = self.model.no_sync()
                else:
                    sync_context = nullcontext()

                with sync_context:
                    with self._autocast_context(use_amp):
                        logits = self.model(x)
                        loss = self.criterion(logits, y)

                    batch_size = y.size(0)
                    loss_sum += float(loss.detach().item()) * batch_size
                    n_samples += batch_size

                    if train:
                        loss = loss / accum_steps
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                if train and should_step:
                    if grad_clip_norm is not None:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            grad_clip_norm,
                        )

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    if lr_scheduler is not None:
                        latest_lr = lr_scheduler.step()
                    if wd_scheduler is not None:
                        latest_wd = wd_scheduler.step()

                if self.is_main_process:
                    running_loss = loss_sum / max(n_samples, 1)
                    pbar.set_postfix(loss=f"{running_loss:.4f}")

        pbar.close()

        loss_sum, n_samples = self._reduce_loss_sum(loss_sum, n_samples)
        avg_loss = loss_sum / max(n_samples, 1)

        metrics = {
            f"{phase}_loss": avg_loss,
            f"{phase}_n": float(n_samples),
        }

        if train:
            if latest_lr is not None:
                metrics["lr"] = float(latest_lr)
            if latest_wd is not None:
                metrics["wd"] = float(latest_wd)

        if self.is_main_process:
            tqdm.write(f"Epoch {epoch} • {phase} loss={avg_loss:.6f}")

        return metrics

    def _plot_history(self, history: list[dict[str, Any]]) -> None:
        if not self.is_main_process or plt is None or not history:
            return

        keys = sorted(
            {
                key
                for rec in history
                for key, value in rec.items()
                if key != "epoch" and isinstance(value, (int, float))
            }
        )

        if not keys:
            return

        n = len(keys)
        ncols = 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        for i, key in enumerate(keys):
            ax = axes[i // ncols, i % ncols]

            epochs = []
            values = []
            for rec in history:
                if key in rec:
                    epochs.append(rec["epoch"])
                    values.append(rec[key])

            ax.plot(epochs, values)
            ax.set_title(key.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        for j in range(len(keys), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_metrics.png")
        plt.close(fig)

    def fit(
        self,
        train_loader,
        val_loader=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Any = None,
        wd_scheduler: Any = None,
        use_amp: bool = False,
        grad_clip_norm: float | None = None,
        epochs: int = 1,
        min_epochs: int = 0,
        accum_steps: int = 1,
        early_stopping: bool = False,
        patience: int = 5,
        plot_interval: int = 5,
        save_ckpt: bool = False,
        ckpt_name: str = "model.pt",
    ) -> Dict[str, Any]:
        hist = []
        best_loss = math.inf
        best_epoch = -1
        best_state_dict = None
        wait = 0
        stopped_early = False

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = self._run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip_norm=grad_clip_norm,
                epoch=epoch,
                train=True,
                accum_steps=accum_steps,
            )

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    loader=val_loader,
                    optimizer=None,
                    lr_scheduler=None,
                    wd_scheduler=None,
                    scaler=None,
                    use_amp=use_amp,
                    grad_clip_norm=None,
                    epoch=epoch,
                    train=False,
                    accum_steps=1,
                )

            record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
            }
            hist.append(record)
            self._save_json({"history": hist}, "history.json")

            if epoch % plot_interval == 0:
                self._plot_history(hist)

            if val_loader is not None:
                current_loss = record["val_loss"]

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state_dict = self._snapshot_state_dict()
                    wait = 0
                else:
                    wait += 1

                if early_stopping and epoch >= min_epochs and wait >= patience:
                    stopped_early = True
                    if self.is_main_process:
                        tqdm.write(
                            f"Early stopping at epoch {epoch}. "
                            f"Best loss={best_loss:.6f} at epoch {best_epoch}."
                        )
                    break
            else:
                best_loss = record["train_loss"]
                best_epoch = epoch
                best_state_dict = self._snapshot_state_dict()

        last_epoch = hist[-1]["epoch"] if hist else 0

        if save_ckpt:
            self._save_checkpoint(
                ckpt_name,
                state_dict=best_state_dict or self._snapshot_state_dict(),
                history=hist,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                extra={
                    "epochs": last_epoch,
                    "use_validation": val_loader is not None,
                    "best_loss": best_loss,
                },
            )

        if best_state_dict is not None:
            self.raw_model.load_state_dict(best_state_dict)

        self._plot_history(hist)

        summary = {
            "epochs_ran": last_epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "stopped_early": stopped_early,
            "ddp": self.ddp,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        self._save_json(summary, "train_summary.json")

        return summary

    @torch.no_grad()
    def evaluate(self, loader, save_name: str = "eval.json") -> Dict[str, float]:
        metrics = self._run_epoch(
            loader=loader,
            optimizer=None,
            lr_scheduler=None,
            wd_scheduler=None,
            scaler=None,
            use_amp=False,
            grad_clip_norm=None,
            epoch=0,
            train=False,
            accum_steps=1,
        )
        self._save_json(metrics, save_name)
        return metrics
