#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class ECGDataset(Dataset):
    """
    Generic ECG HDF5 + CSV dataset.

    Expected HDF5 structure
    -----------------------

    The HDF5 file must contain:

    dataset.h5
    ├── signal   float32, shape (N, T, 12)
    ├── exam_id  int64,   shape (N,)
    ├── train
    │   └── hdf5_index
    ├── val
    │   └── hdf5_index
    └── test
        └── hdf5_index

    where:

    - signal[i] is the ECG signal of sample i
    - signal shape is (time, leads)
    - train/val/test groups are optional
    - each split group contains:
        hdf5_index : int64 array indexing rows in signal

    If group=None:
        iterate through the entire dataset.

    If n_samples is not None:
        use a deterministic random subset of n_samples examples,
        controlled by seed.

    ----------------------------------------------------------------------

    Expected CSV structure
    ----------------------

    The CSV must contain exactly one row per HDF5 sample
    in the SAME ORDER as the HDF5 datasets.

    Required columns:

    - target columns passed in csv_target_cols

    Optional metadata columns:

    - columns passed in csv_metadata_cols

    Typical CSV example:

    hdf5_index,exam_id,patient_id,source_dataset,1dAVb,RBBB,LBBB,SB,AF,ST
    0,590673,57158,code15,0,0,0,0,0,0
    1,214626,800764,code15,0,0,0,0,0,0
    ...

    Important:
    - row i in the CSV corresponds to signal[i]
    - CSV rows must match the HDF5 sample count
    - target columns are converted to float32
    - metadata columns are returned as strings

    ----------------------------------------------------------------------

    Returns
    -------

    __getitem__(idx) returns:

        signal   : torch.float32, shape (crop_len, 12)
        targets  : torch.float32, shape (n_targets,)
        metadata : dict[str, str]
    """

    def __init__(
        self,
        h5_path: str | Path,
        csv_path: str | Path,
        csv_target_cols: Sequence[str],
        csv_metadata_cols: Sequence[str] = [],
        signal_crop_len: int = 2560,
        group: str | None = None,
        n_samples: int | None = None,
        seed: int = 0,
    ):
        if group is not None and group not in {"train", "val", "test"}:
            raise ValueError("group must be None, 'train', 'val', or 'test'.")

        self.h5_path = Path(h5_path)
        self.csv_path = Path(csv_path)
        self.csv_target_cols = list(csv_target_cols)
        self.csv_metadata_cols = list(csv_metadata_cols)
        self.signal_crop_len = signal_crop_len
        self.group = group
        self.n_samples = n_samples
        self.seed = seed

        self.h5_file = None
        self.signal_ds = None

        df = pd.read_csv(self.csv_path, low_memory=False)

        required_cols = self.csv_target_cols + self.csv_metadata_cols
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing CSV columns: {missing_cols}")

        self.targets = (
            df[self.csv_target_cols]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float32)
        )

        self.metadata = df[self.csv_metadata_cols].astype(str)

        with h5py.File(self.h5_path, "r") as h5:
            if "signal" not in h5:
                raise KeyError(
                    f"HDF5 is missing dataset 'signal'. Keys: {list(h5.keys())}"
                )

            signal_shape = h5["signal"].shape

            if len(signal_shape) != 3:
                raise ValueError(
                    f"Expected signal shape (N, T, D), got {signal_shape}."
                )

            n_signals = signal_shape[0]

            if len(df) != n_signals:
                raise ValueError(f"CSV rows ({len(df)}) != HDF5 signals ({n_signals}).")

            if self.group is None:
                self.hdf5_indices = np.arange(n_signals, dtype=np.int64)
            else:
                if self.group not in h5:
                    raise KeyError(f"HDF5 is missing group '{self.group}'.")

                if "hdf5_index" not in h5[self.group]:
                    raise KeyError(
                        f"HDF5 group '{self.group}' is missing 'hdf5_index'."
                    )

                self.hdf5_indices = np.asarray(
                    h5[self.group]["hdf5_index"],
                    dtype=np.int64,
                )

        if self.n_samples is not None:
            if self.n_samples <= 0:
                raise ValueError("n_samples must be positive.")

            if self.n_samples > len(self.hdf5_indices):
                print(
                    f"n_samples ({self.n_samples}) larger than dataset size ({len(self.hdf5_indices)})."
                    f"using ({len(self.hdf5_indices)}) samples."
                )

            generator = torch.Generator().manual_seed(self.seed)
            selected_indices = torch.randperm(
                len(self.hdf5_indices),
                generator=generator,
            )[: self.n_samples]

            self.hdf5_indices = self.hdf5_indices[selected_indices.numpy()]

    def _center_crop(self, signal: np.ndarray) -> np.ndarray:
        length = signal.shape[0]

        if self.signal_crop_len > length:
            raise ValueError(
                f"Cannot crop signal of length {length} to {self.signal_crop_len}."
            )

        start = (length - self.signal_crop_len) // 2
        end = start + self.signal_crop_len

        return signal[start:end, :]

    def __len__(self) -> int:
        return len(self.hdf5_indices)

    def _open_h5_if_needed(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
            self.signal_ds = self.h5_file["signal"]

    def __getitem__(self, idx: int):
        self._open_h5_if_needed()

        h5_idx = int(self.hdf5_indices[idx])

        signal = np.asarray(self.signal_ds[h5_idx], dtype=np.float32)
        signal = self._center_crop(signal)

        x = torch.tensor(signal, dtype=torch.float32)
        y = torch.tensor(self.targets[h5_idx], dtype=torch.float32)

        metadata = self.metadata.iloc[h5_idx].to_dict()

        return x, y, metadata

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.signal_ds = None

    def __del__(self):
        self.close()

    @staticmethod
    def get_dataloader(
        h5_path: str | Path,
        csv_path: str | Path,
        csv_target_cols: Sequence[str],
        csv_metadata_cols: Sequence[str] = [],
        signal_crop_len: int = 2560,
        group: str | None = None,
        n_samples: int | None = None,
        batch_size: int = 256,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True,
    ) -> DataLoader:
        dataset = ECGDataset(
            h5_path=h5_path,
            csv_path=csv_path,
            csv_target_cols=csv_target_cols,
            csv_metadata_cols=csv_metadata_cols,
            signal_crop_len=signal_crop_len,
            group=group,
            n_samples=n_samples,
            seed=seed,
        )

        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            )
        else:
            if shuffle:
                generator = torch.Generator().manual_seed(seed)
                sampler = RandomSampler(dataset, generator=generator)
            else:
                sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
