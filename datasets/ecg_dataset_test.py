from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from dotenv import load_dotenv
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datasets import ECGDataset

TARGET_COLS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
ECG_SIZE = (2560, 12)

DATASETS = {
    "code15": {
        "h5_env": "CODE15_H5_PATH",
        "csv_env": "CODE15_CSV_PATH",
    },
    "chapman": {
        "h5_env": "CHAPMAN_H5_PATH",
        "csv_env": "CHAPMAN_CSV_PATH",
    },
    "ptbxl": {
        "h5_env": "PTBXL_H5_PATH",
        "csv_env": "PTBXL_CSV_PATH",
    },
}


def get_dataset_paths(dataset_name: str) -> tuple[str, str]:
    load_dotenv()

    h5_path = os.environ[DATASETS[dataset_name]["h5_env"]]
    csv_path = os.environ[DATASETS[dataset_name]["csv_env"]]

    assert Path(h5_path).exists(), f"HDF5 path does not exist: {h5_path}"
    assert Path(csv_path).exists(), f"CSV path does not exist: {csv_path}"

    return h5_path, csv_path


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
@pytest.mark.parametrize("group", ["train", "val", "test"])
def test_real_dataset_getitem(dataset_name: str, group: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    dataset = ECGDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group=group,
        n_samples=4,
        seed=0,
    )

    assert len(dataset) == 4

    x, y, metadata = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(metadata, dict)

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    assert x.shape == ECG_SIZE
    assert y.shape == (len(TARGET_COLS),)
    assert metadata == {}

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()

    dataset.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataset_n_samples_is_deterministic(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    dataset_a = ECGDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        seed=123,
    )

    dataset_b = ECGDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        seed=123,
    )

    assert len(dataset_a) == 8
    assert len(dataset_b) == 8
    assert np.array_equal(dataset_a.hdf5_indices, dataset_b.hdf5_indices)

    dataset_a.close()
    dataset_b.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataset_n_samples_changes_with_seed(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    dataset_a = ECGDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        seed=1,
    )

    dataset_b = ECGDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        seed=2,
    )

    assert len(dataset_a) == 8
    assert len(dataset_b) == 8
    assert not np.array_equal(dataset_a.hdf5_indices, dataset_b.hdf5_indices)

    dataset_a.close()
    dataset_b.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataloader_single_gpu_shuffle(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    loader = ECGDataset.get_dataloader(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        batch_size=4,
        num_workers=0,
        rank=0,
        world_size=1,
        shuffle=True,
        seed=0,
        drop_last=False,
        pin_memory=False,
    )

    assert len(loader.dataset) == 8
    assert isinstance(loader.sampler, RandomSampler)

    x, y, metadata = next(iter(loader))

    assert x.shape == (4, ECG_SIZE[0], ECG_SIZE[1])
    assert y.shape == (4, len(TARGET_COLS))
    assert metadata == {}

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()

    loader.dataset.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataloader_single_gpu_no_shuffle(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    loader = ECGDataset.get_dataloader(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="val",
        n_samples=8,
        batch_size=4,
        num_workers=0,
        rank=0,
        world_size=1,
        shuffle=False,
        seed=0,
        drop_last=False,
        pin_memory=False,
    )

    assert len(loader.dataset) == 8
    assert isinstance(loader.sampler, SequentialSampler)

    x, y, metadata = next(iter(loader))

    assert x.shape == (4, ECG_SIZE[0], ECG_SIZE[1])
    assert y.shape == (4, len(TARGET_COLS))
    assert metadata == {}

    loader.dataset.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataloader_ddp_sampler(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    loader_rank0 = ECGDataset.get_dataloader(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        batch_size=2,
        num_workers=0,
        rank=0,
        world_size=2,
        shuffle=True,
        seed=0,
        drop_last=False,
        pin_memory=False,
    )

    loader_rank1 = ECGDataset.get_dataloader(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        batch_size=2,
        num_workers=0,
        rank=1,
        world_size=2,
        shuffle=True,
        seed=0,
        drop_last=False,
        pin_memory=False,
    )

    assert isinstance(loader_rank0.sampler, DistributedSampler)
    assert isinstance(loader_rank1.sampler, DistributedSampler)

    assert len(loader_rank0.dataset) == 8
    assert len(loader_rank1.dataset) == 8

    assert np.array_equal(
        loader_rank0.dataset.hdf5_indices,
        loader_rank1.dataset.hdf5_indices,
    )

    rank0_indices = list(iter(loader_rank0.sampler))
    rank1_indices = list(iter(loader_rank1.sampler))

    assert len(rank0_indices) == 4
    assert len(rank1_indices) == 4
    assert set(rank0_indices).isdisjoint(set(rank1_indices))

    x0, y0, metadata0 = next(iter(loader_rank0))
    x1, y1, metadata1 = next(iter(loader_rank1))

    assert x0.shape == (2, ECG_SIZE[0], ECG_SIZE[1])
    assert y0.shape == (2, len(TARGET_COLS))
    assert metadata0 == {}

    assert x1.shape == (2, ECG_SIZE[0], ECG_SIZE[1])
    assert y1.shape == (2, len(TARGET_COLS))
    assert metadata1 == {}

    loader_rank0.dataset.close()
    loader_rank1.dataset.close()


@pytest.mark.parametrize("dataset_name", ["code15", "chapman", "ptbxl"])
def test_real_dataloader_num_workers(dataset_name: str):
    h5_path, csv_path = get_dataset_paths(dataset_name)

    loader = ECGDataset.get_dataloader(
        h5_path=h5_path,
        csv_path=csv_path,
        csv_target_cols=TARGET_COLS,
        csv_metadata_cols=[],
        signal_crop_len=ECG_SIZE[0],
        group="train",
        n_samples=8,
        batch_size=4,
        num_workers=2,
        rank=0,
        world_size=1,
        shuffle=True,
        seed=0,
        drop_last=False,
        pin_memory=False,
    )

    x, y, metadata = next(iter(loader))

    assert x.shape == (4, ECG_SIZE[0], ECG_SIZE[1])
    assert y.shape == (4, len(TARGET_COLS))
    assert metadata == {}

    loader.dataset.close()
