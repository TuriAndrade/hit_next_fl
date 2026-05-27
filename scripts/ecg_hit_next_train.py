from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from datasets import ECGDataset
from models import ECGHiTNeXt
from optimizers import AdamW
from trainers import SupervisedTrainer
from utils import save_config, set_seed, parse_args_json

_ECG_SIZE = (2560, 12)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _get_dataset_paths(dataset_name: str) -> tuple[str, str]:
    env_names = {
        "code15": ("CODE15_H5_PATH", "CODE15_CSV_PATH"),
        "chapman": ("CHAPMAN_H5_PATH", "CHAPMAN_CSV_PATH"),
        "ptbxl": ("PTBXL_H5_PATH", "PTBXL_CSV_PATH"),
    }

    h5_env_name, csv_env_name = env_names[dataset_name]

    h5_path = os.getenv(h5_env_name)
    csv_path = os.getenv(csv_env_name)

    if h5_path is None:
        raise ValueError(f"Missing environment variable: {h5_env_name}")
    if csv_path is None:
        raise ValueError(f"Missing environment variable: {csv_env_name}")

    return h5_path, csv_path


def _build_model_kwargs(args) -> dict[str, Any]:
    kwargs = {
        "ecg_size": _ECG_SIZE,
        "num_stages": 4,
        "hidden_dim": [96, 192, 384, 768],
        "layers": [4, 4, 12, 4],
        "heads": [4, 8, 16, 32],
        "downsample_factor": 4,
        "window_size": _ECG_SIZE[0] // (4**4),
        "expansion_factor": 4,
        "merge_conv_config": {
            "kernel_size": 10,
            "stride": 4,
            "padding": 4,
        },
        "merge_p_drop": 0.1,
        "attn_p_drop": 0.1,
        "rpe_type": "combined",
        "use_ape": True,
        "cosine_sim": True,
        "shift": True,
        "shift_size": None,
        "apply_out_mlp": True,
        "out_mlp_hidden_dim": [512],
        "out_mlp_out_dim": len(args.target_cols),
        "out_mlp_dropout": 0.7,
        "out_mlp_norm": True,
    }

    kwargs.update(args.model_extra_args)

    return kwargs


def _build_criterion(args) -> torch.nn.Module:
    if args.loss == "bce":
        return torch.nn.BCEWithLogitsLoss()

    if args.loss == "mse":
        return torch.nn.MSELoss()

    raise ValueError(f"Invalid loss: {args.loss}")


def _get_save_name(args, run_seed: int) -> str:
    return (
        f"{args.save_name}_{run_seed}" if args.save_name is not None else str(run_seed)
    )


def _get_run_dir(args, run_seed: int) -> Path:
    return Path(args.save_dir) / _get_save_name(args, run_seed)


def _get_final_eval_path(args, run_seed: int) -> Path:
    eval_name = "test_loss_eval.json" if args.eval_on_test else "val_loss_eval.json"
    return _get_run_dir(args, run_seed) / eval_name


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(description="Supervised ECG model training.")

    # ----- Data -----
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["code15", "ptbxl", "chapman"],
        help="Dataset name. Options: code15, ptbxl, chapman.",
    )
    parser.add_argument(
        "--target-cols",
        type=json.loads,
        default=["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
        help='Target columns. Example: --target-cols \'["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]\'',
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--drop-last", action="store_true")

    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=None,
        help="Optional number of training samples to use.",
    )
    parser.add_argument(
        "--n-val-samples",
        type=int,
        default=None,
        help="Optional number of validation samples to use.",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=None,
        help="Optional number of test samples to use.",
    )

    # ----- Model -----
    parser.add_argument(
        "--model-extra-args",
        type=json.loads,
        default={},
        help="JSON dict with ECGHiTNeXt keyword arguments to override the defaults.",
    )
    parser.add_argument("--load-model-path", type=str, default=None)

    # ----- Training -----
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Global effective batch size across all GPUs after gradient accumulation.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=64,
        help="Mini-batch size processed by each GPU before gradient accumulation.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "mse"],
        default="bce",
        help="Loss function to use. Options: bce, mse.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--plot-interval", type=int, default=5)

    # ----- Optimizer -----
    parser.add_argument("--start-lr", type=float, default=1e-5)
    parser.add_argument("--ref-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--ref-wd", type=float, default=1e-3)
    parser.add_argument("--final-wd", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # ----- Seeds -----
    parser.add_argument(
        "--run-seeds",
        type=json.loads,
        default=[0],
        help="List of seeds to run. Example: --run-seeds '[0, 1, 2]'",
    )

    # ----- DDP -----
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")

    # ----- IO -----
    parser.add_argument("--save-dir", type=str, default="experiments")
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--save-ckpt", action="store_true")
    parser.add_argument("--save-ckpt-all-seeds", action="store_true")
    parser.add_argument("--eval-on-test", action="store_true")

    return parser


def _parse_args():
    parser = _build_parser()
    args = parse_args_json(parser)

    args.h5_path, args.csv_path = _get_dataset_paths(args.dataset_name)

    return args


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def _build_loader(
    args,
    group: str,
    n_samples: int | None,
    shuffle: bool,
    seed: int,
    rank: int,
    world_size: int,
):
    return ECGDataset.get_dataloader(
        h5_path=args.h5_path,
        csv_path=args.csv_path,
        csv_target_cols=args.target_cols,
        csv_metadata_cols=[],
        signal_crop_len=_ECG_SIZE[0],
        group=group,
        n_samples=n_samples,
        batch_size=args.gpu_batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=args.drop_last if shuffle else False,
        pin_memory=args.pin_memory,
    )


def _build_model(args) -> torch.nn.Module:
    model = ECGHiTNeXt(**_build_model_kwargs(args))

    if args.load_model_path is not None:
        checkpoint = torch.load(args.load_model_path, map_location="cpu")
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)

    return model


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def _train_seed(
    rank: int, world_size: int, args, run_seed: int, save_ckpt: bool
) -> None:
    run_dir = _get_run_dir(args, run_seed)
    device = f"cuda:{rank}"

    per_step_batch_size = args.gpu_batch_size * world_size
    if args.batch_size < per_step_batch_size:
        raise ValueError("batch_size should be >= gpu_batch_size * number_of_gpus.")
    if args.batch_size % per_step_batch_size != 0:
        raise ValueError(
            "batch_size should be divisible by gpu_batch_size * number_of_gpus."
        )

    accum_steps = args.batch_size // per_step_batch_size

    if world_size > 1:
        SupervisedTrainer.setup_ddp(
            rank=rank,
            world_size=world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )

    train_loader = None
    val_loader = None
    test_loader = None
    trainer = None
    adamw = None
    optimizer = None
    lr_scheduler = None
    wd_scheduler = None

    try:
        set_seed(run_seed)

        # ----- Dataloaders -----
        train_loader = _build_loader(
            args=args,
            group="train",
            n_samples=args.n_train_samples,
            shuffle=True,
            seed=run_seed,
            rank=rank,
            world_size=world_size,
        )
        val_loader = _build_loader(
            args=args,
            group="val",
            n_samples=args.n_val_samples,
            shuffle=False,
            seed=run_seed,
            rank=rank,
            world_size=world_size,
        )
        if args.eval_on_test:
            test_loader = _build_loader(
                args=args,
                group="test",
                n_samples=args.n_test_samples,
                shuffle=False,
                seed=run_seed,
                rank=rank,
                world_size=world_size,
            )

        # ----- Model, loss, trainer -----
        model = _build_model(args)
        criterion = _build_criterion(args)

        trainer = SupervisedTrainer(
            model=model,
            criterion=criterion,
            save_dir=run_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            ddp=world_size > 1,
        )

        # ----- Optimizer -----
        epoch_len = math.ceil(len(train_loader) / accum_steps)
        adamw = AdamW(
            models=trainer.raw_model,
            use_lr_scheduler=args.start_lr != args.ref_lr
            or args.ref_lr != args.final_lr,
            use_wd_scheduler=args.ref_wd != args.final_wd,
            warmup_steps=args.warmup_epochs * epoch_len,
            start_lr=args.start_lr,
            ref_lr=args.ref_lr,
            final_lr=args.final_lr,
            ref_wd=args.ref_wd,
            final_wd=args.final_wd,
            T_max=args.epochs * epoch_len,
        )
        optimizer = adamw.get_optimizer()
        lr_scheduler = adamw.get_lr_scheduler()
        wd_scheduler = adamw.get_wd_scheduler()

        # ----- Save config -----
        if rank == 0:
            save_config(
                config={
                    **vars(args),
                    "run_seed": run_seed,
                    "world_size": world_size,
                    "accum_steps": accum_steps,
                    "epoch_len": epoch_len,
                    "ecg_size": list(_ECG_SIZE),
                    "n_targets": len(args.target_cols),
                    "model_kwargs": _build_model_kwargs(args),
                },
                path=run_dir / "config.json",
            )

        # ----- Train -----
        summary = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            use_amp=args.use_amp,
            grad_clip_norm=args.grad_clip_norm,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            accum_steps=accum_steps,
            early_stopping=args.early_stopping,
            patience=args.patience,
            plot_interval=args.plot_interval,
            save_ckpt=save_ckpt,
            ckpt_name="model.pt",
        )

        # ----- Eval loss -----
        trainer.evaluate(val_loader, save_name="val_loss_eval.json")
        if args.eval_on_test and test_loader is not None:
            trainer.evaluate(test_loader, save_name="test_loss_eval.json")

        if rank == 0:
            print(f"Finished seed {run_seed}: {summary}")

    finally:
        for loader in [train_loader, val_loader, test_loader]:
            if loader is not None and hasattr(loader.dataset, "close"):
                loader.dataset.close()

        del train_loader, val_loader, test_loader
        del trainer, adamw, optimizer, lr_scheduler, wd_scheduler

        gc.collect()
        torch.cuda.empty_cache()

        if world_size > 1:
            SupervisedTrainer.cleanup_ddp()


def _train_worker(
    rank: int, world_size: int, args, run_seed: int, save_ckpt: bool
) -> None:
    _train_seed(
        rank=rank,
        world_size=world_size,
        args=args,
        run_seed=run_seed,
        save_ckpt=save_ckpt,
    )


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs found.")

    print(f"Using {world_size} GPU(s).")
    print(f"Dataset: {args.dataset_name}")
    print(f"HDF5: {args.h5_path}")
    print(f"CSV: {args.csv_path}")

    for i, run_seed in enumerate(args.run_seeds):
        final_eval_path = _get_final_eval_path(args, run_seed)
        if not args.override and final_eval_path.exists():
            print(f"Experiment already done: found {final_eval_path}. Skipping.")
            continue

        print(f"\n===== Training seed {run_seed} =====")
        save_ckpt = args.save_ckpt and (i == 0 or args.save_ckpt_all_seeds)

        if world_size == 1:
            _train_seed(
                rank=0,
                world_size=1,
                args=args,
                run_seed=run_seed,
                save_ckpt=save_ckpt,
            )
        else:
            mp.spawn(
                _train_worker,
                args=(world_size, args, run_seed, save_ckpt),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    main()
