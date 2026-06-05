from __future__ import annotations

from pathlib import Path

import torch

from models import ECGHiTNeXt
from .ecg_supervised_task import ECGMultilabelClassification


def clf_config(
    h5_path: Path | str,
    csv_path: Path | str,
    model_extra_args: dict | None = None,
    task_extra_args: dict | None = None,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
    n_test_samples: int | None = None,
    world_size: int = 1,
) -> ECGMultilabelClassification:
    model_extra_args = model_extra_args or {}
    task_extra_args = task_extra_args or {}

    ecg_size = (2560, 12)
    target_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    out_dim = len(target_cols)

    model_kwargs = {
        "ecg_size": ecg_size,
        "num_stages": 4,
        "hidden_dim": [96, 192, 384, 768],
        "layers": [4, 4, 12, 4],
        "heads": [4, 8, 16, 32],
        "downsample_factor": 4,
        "window_size": 10,
        "expansion_factor": 4,
        "merge_conv_config": {
            "kernel_size": 10,
            "stride": 4,
            "padding": 4,
        },
        "merge_p_drop": 0.1,
        "attn_p_drop": 0.1,
        "rpe_type": "combined",
        "use_ape": False,
        "cosine_sim": True,
        "shift": True,
        "shift_size": None,
        "apply_out_mlp": True,
        "out_mlp_hidden_dim": [512],
        "out_mlp_out_dim": out_dim,
        "out_mlp_dropout": 0.5,
        "out_mlp_norm": True,
    }

    model_kwargs.update(model_extra_args)

    task_kwargs = {
        "h5_path": h5_path,
        "csv_path": csv_path,
        "ecg_size": ecg_size,
        "target_cols": target_cols,
        "criterion": torch.nn.BCEWithLogitsLoss(),
        "model": ECGHiTNeXt(**model_kwargs),
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "n_test_samples": n_test_samples,
        "world_size": world_size,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
        "batch_size": 128,
        "gpu_batch_size": 64,
        "epochs": 100,
        "min_epochs": 0,
        "patience": 5,
        "early_stopping": True,
        "keep_best": True,
        "use_amp": True,
        "grad_clip_norm": None,
        "plot_interval": 5,
        "save_ckpt": True,
        "start_lr": 1e-5,
        "ref_lr": 1e-4,
        "final_lr": 1e-5,
        "ref_wd": 1e-3,
        "final_wd": 1e-3,
        "warmup_epochs": 5,
    }

    task_kwargs.update(task_extra_args)
    return ECGMultilabelClassification(**task_kwargs)


tasks = {
    "clf": clf_config,
}
