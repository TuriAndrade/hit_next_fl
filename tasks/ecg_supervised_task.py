from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import torch

from tqdm import tqdm

from optimizers import AdamW
from trainers import SupervisedTrainer


@dataclass
class ECGSupervisedTask(ABC):
    h5_path: str
    csv_path: str
    ecg_size: tuple[int, int]
    target_cols: list[str]
    criterion: torch.nn.Module
    model: torch.nn.Module
    n_train_samples: int | None
    n_val_samples: int | None
    n_test_samples: int | None
    world_size: int

    # dataloader
    num_workers: int
    pin_memory: bool
    drop_last: bool

    # training
    batch_size: int
    gpu_batch_size: int
    epochs: int
    min_epochs: int
    patience: int
    early_stopping: bool
    keep_best: bool
    use_amp: bool
    grad_clip_norm: float | None
    plot_interval: int
    save_ckpt: bool

    # optimizer
    start_lr: float
    ref_lr: float
    final_lr: float
    ref_wd: float
    final_wd: float
    warmup_epochs: int

    def make_loader(
        self,
        group: str,
        n_samples: int | None,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
    ):
        from datasets import ECGDataset

        return ECGDataset.get_dataloader(
            h5_path=self.h5_path,
            csv_path=self.csv_path,
            csv_target_cols=self.target_cols,
            csv_metadata_cols=[],
            signal_crop_len=self.ecg_size[0],
            group=group,
            n_samples=n_samples,
            batch_size=self.gpu_batch_size,
            num_workers=self.num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=self.drop_last if shuffle else False,
            pin_memory=self.pin_memory,
        )

    def make_loaders(self, seed: int, rank: int, world_size: int):
        train_loader = self.make_loader(
            group="train",
            n_samples=self.n_train_samples,
            shuffle=True,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

        val_loader = self.make_loader(
            group="val",
            n_samples=self.n_val_samples,
            shuffle=False,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

        test_loader = self.make_loader(
            group="test",
            n_samples=self.n_test_samples,
            shuffle=False,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

        return train_loader, val_loader, test_loader

    def make_trainer(
        self,
        save_dir: Path,
        device: str,
        rank: int,
        world_size: int,
    ):
        return SupervisedTrainer(
            model=self.model,
            save_dir=save_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            ddp=world_size > 1,
        )

    def compute_accum_steps(self) -> int:
        per_step_batch_size = self.gpu_batch_size * self.world_size

        if self.batch_size < per_step_batch_size:
            raise ValueError("batch_size must be >= gpu_batch_size * world_size.")

        if self.batch_size % per_step_batch_size != 0:
            raise ValueError(
                "batch_size must be divisible by gpu_batch_size * world_size."
            )

        return self.batch_size // per_step_batch_size

    def make_optimizer(self, *, trainer: SupervisedTrainer, train_loader):
        accum_steps = self.compute_accum_steps()
        epoch_len = math.ceil(len(train_loader) / accum_steps)

        adamw = AdamW(
            models=trainer.raw_model,
            use_lr_scheduler=(
                self.start_lr != self.ref_lr or self.ref_lr != self.final_lr
            ),
            use_wd_scheduler=self.ref_wd != self.final_wd,
            warmup_steps=self.warmup_epochs * epoch_len,
            start_lr=self.start_lr,
            ref_lr=self.ref_lr,
            final_lr=self.final_lr,
            ref_wd=self.ref_wd,
            final_wd=self.final_wd,
            T_max=self.epochs * epoch_len,
        )

        return (
            adamw.get_optimizer(),
            adamw.get_lr_scheduler(),
            adamw.get_wd_scheduler(),
        )

    def fit_kwargs(self) -> dict:
        return {
            "use_amp": self.use_amp,
            "grad_clip_norm": self.grad_clip_norm,
            "epochs": self.epochs,
            "min_epochs": self.min_epochs,
            "accum_steps": self.compute_accum_steps(),
            "early_stopping": self.early_stopping,
            "keep_best": self.keep_best,
            "patience": self.patience,
            "plot_interval": self.plot_interval,
            "save_ckpt": self.save_ckpt,
            "ckpt_name": "model.pt",
        }


@dataclass
class ECGMultilabelClassification(ECGSupervisedTask):
    ECG_SIZE = (2560, 12)
    TARGET_CLF_COLUMNS = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

    def _class_preds_and_targets(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        class_idx: int,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits[:, class_idx])
        preds = probs >= threshold
        targets = y[:, class_idx].bool()

        return preds, targets

    def _evaluate_class_metric(
        self,
        trainer: SupervisedTrainer,
        loader,
        metric_name: str,
        class_idx: int,
        threshold: float,
        use_amp: bool | None = None,
    ) -> float:
        if use_amp is None:
            use_amp = self.use_amp

        def metric(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            preds, targets = self._class_preds_and_targets(
                logits=logits,
                y=y,
                class_idx=class_idx,
                threshold=threshold,
            )

            if metric_name == "acc":
                return (preds == targets).float()

            if metric_name == "tp":
                return (preds & targets).float()

            if metric_name == "pred_pos":
                return preds.float()

            if metric_name == "target_pos":
                return targets.float()

            raise ValueError(f"Invalid metric_name: {metric_name}")

        result = trainer.evaluate(
            loader=loader,
            metric=metric,
            metric_name=metric_name,
            use_amp=use_amp,
            verbose=False,
        )

        return float(result[metric_name])

    def _evaluate_class_metrics(
        self,
        trainer: SupervisedTrainer,
        loader,
        class_idx: int,
        threshold: float,
        use_amp: bool | None = None,
        eps: float = 1e-8,
    ) -> dict:
        acc = self._evaluate_class_metric(
            trainer=trainer,
            loader=loader,
            metric_name="acc",
            class_idx=class_idx,
            threshold=threshold,
            use_amp=use_amp,
        )

        tp = self._evaluate_class_metric(
            trainer=trainer,
            loader=loader,
            metric_name="tp",
            class_idx=class_idx,
            threshold=threshold,
            use_amp=use_amp,
        )

        pred_pos = self._evaluate_class_metric(
            trainer=trainer,
            loader=loader,
            metric_name="pred_pos",
            class_idx=class_idx,
            threshold=threshold,
            use_amp=use_amp,
        )

        target_pos = self._evaluate_class_metric(
            trainer=trainer,
            loader=loader,
            metric_name="target_pos",
            class_idx=class_idx,
            threshold=threshold,
            use_amp=use_amp,
        )

        precision = tp / (pred_pos + eps)
        recall = tp / (target_pos + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate(
        self,
        trainer: SupervisedTrainer,
        val_loader,
        test_loader,
        thresholds: list[float] | None = None,
        threshold_metric: str = "f1",
        use_amp: bool | None = None,
    ) -> dict:
        if thresholds is None:
            thresholds = [
                0.001,
                0.003,
                0.005,
                0.01,
                0.03,
                0.05,
                0.1,
                0.3,
                0.5,
            ]

        if len(thresholds) == 0:
            raise ValueError("thresholds cannot be empty.")

        if threshold_metric not in {"acc", "precision", "recall", "f1"}:
            raise ValueError(
                "threshold_metric must be one of: " "acc, precision, recall, f1."
            )

        n_classes = len(self.target_cols)

        best_thresholds = {}
        best_val_metrics = {}
        test_metrics = {}

        for class_idx, class_name in enumerate(self.target_cols):
            best_threshold = float(thresholds[0])
            best_val_metric = -1.0

            for threshold in tqdm(
                thresholds,
                desc=f"Tuning {class_name} threshold on {threshold_metric}",
                disable=not trainer.is_main_process,
            ):
                threshold = float(threshold)

                val_metrics = self._evaluate_class_metrics(
                    trainer=trainer,
                    loader=val_loader,
                    class_idx=class_idx,
                    threshold=threshold,
                    use_amp=use_amp,
                )

                val_metric = val_metrics[threshold_metric]

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_threshold = threshold

            best_thresholds[class_name] = best_threshold
            best_val_metrics[class_name] = best_val_metric

            test_metrics[class_name] = self._evaluate_class_metrics(
                trainer=trainer,
                loader=test_loader,
                class_idx=class_idx,
                threshold=best_threshold,
                use_amp=use_amp,
            )

        return {
            "classes": self.target_cols,
            "n_classes": n_classes,
            "threshold_metric": threshold_metric,
            "best_thresholds": best_thresholds,
            f"best_val_{threshold_metric}": best_val_metrics,
            "test_metrics": test_metrics,
        }
