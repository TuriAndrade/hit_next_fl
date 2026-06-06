from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tasks import get_task_definition
from utils import canonical_dataset_name, get_dataset_paths


@dataclass(frozen=True)
class ClientConfig:
    name: str
    h5_path: str
    csv_path: str
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_test_samples: int | None = None


@dataclass
class FederatedConfig:
    model_name: str
    task_name: str
    client_names: list[str]
    rounds: int
    local_epochs: int
    client_fraction: float
    clients_per_round: int | None
    aggregation: str
    keep_best: bool
    early_stopping: bool
    patience: int
    keep_client_ckpts: bool
    seed: int
    save_dir: Path
    model_extra_args: dict[str, Any] = field(default_factory=dict)
    task_extra_args: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        get_task_definition(self.model_name, self.task_name)

        if not self.client_names:
            raise ValueError("At least one client is required.")

        if len(set(self.client_names)) != len(self.client_names):
            raise ValueError("Client names must be unique.")

        if self.rounds <= 0:
            raise ValueError("rounds must be positive.")

        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1].")

        if self.clients_per_round is not None:
            if self.clients_per_round <= 0:
                raise ValueError("clients_per_round must be positive.")
            if self.clients_per_round > len(self.client_names):
                raise ValueError("clients_per_round cannot exceed number of clients.")

        if self.aggregation not in {"weighted", "uniform"}:
            raise ValueError("aggregation must be 'weighted' or 'uniform'.")

        if self.patience <= 0:
            raise ValueError("patience must be positive.")

    def selected_client_names(self, round_idx: int) -> list[str]:
        if self.clients_per_round is not None:
            n_selected = self.clients_per_round
        else:
            n_selected = max(
                1,
                math.ceil(len(self.client_names) * self.client_fraction),
            )

        if n_selected == len(self.client_names):
            return list(self.client_names)

        rng = random.Random(self.seed + round_idx)
        return rng.sample(self.client_names, n_selected)


def parse_client_names(clients: str) -> list[str]:
    names = [
        canonical_dataset_name(name)
        for name in clients.split(",")
        if name.strip()
    ]

    if not names:
        raise ValueError("No clients were provided.")

    if len(set(names)) != len(names):
        raise ValueError("Client names must be unique.")

    return names


def _parse_sample_counts(value: Any, name: str) -> list[int | None] | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    elif isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return None

        if value.startswith("["):
            raw_values = json.loads(value)
            if not isinstance(raw_values, list):
                raise ValueError(f"{name} JSON value must be a list.")
        else:
            raw_values = [item.strip() for item in value.split(",")]
    else:
        raise TypeError(f"{name} must be a list, comma-separated string, or None.")

    parsed: list[int | None] = []
    for item in raw_values:
        if item is None:
            parsed.append(None)
            continue

        if isinstance(item, str):
            item = item.strip()
            if item.lower() in {"", "none", "null"}:
                parsed.append(None)
                continue

        count = int(item)
        if count <= 0:
            raise ValueError(f"{name} sample counts must be positive or null.")

        parsed.append(count)

    return parsed


def _sample_counts_for_clients(
    value: Any,
    n_clients: int,
    name: str,
) -> list[int | None]:
    parsed = _parse_sample_counts(value, name)
    if parsed is None:
        return [None] * n_clients

    if len(parsed) != n_clients:
        raise ValueError(
            f"{name} must have one value per client. "
            f"Expected {n_clients}, got {len(parsed)}."
        )

    return parsed


def create_client_configs(
    *,
    clients: str,
    n_train_samples: Any,
    n_val_samples: Any,
    n_test_samples: Any,
) -> list[ClientConfig]:
    client_names = parse_client_names(clients)
    n_clients = len(client_names)
    train_counts = _sample_counts_for_clients(
        n_train_samples,
        n_clients,
        "n_train_samples",
    )
    val_counts = _sample_counts_for_clients(
        n_val_samples,
        n_clients,
        "n_val_samples",
    )
    test_counts = _sample_counts_for_clients(
        n_test_samples,
        n_clients,
        "n_test_samples",
    )

    configs = []
    for index, name in enumerate(client_names):
        h5_path, csv_path = get_dataset_paths(name)
        configs.append(
            ClientConfig(
                name=name,
                h5_path=h5_path,
                csv_path=csv_path,
                n_train_samples=train_counts[index],
                n_val_samples=val_counts[index],
                n_test_samples=test_counts[index],
            )
        )

    return configs
