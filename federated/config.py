from __future__ import annotations

import math
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import canonical_dataset_name, get_dataset_paths


def _task_registry():
    from tasks import tasks

    return tasks


@dataclass
class ClientConfig:
    name: str
    dataset_name: str
    task_name: str
    h5_path: str
    csv_path: str
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_test_samples: int | None = None
    task_extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedConfig:
    model_name: str
    task_name: str
    clients: list[ClientConfig]
    rounds: int
    local_epochs: int
    client_fraction: float
    clients_per_round: int | None
    aggregation: str
    eval_every: int
    save_every: int
    keep_client_ckpts: bool
    local_keep_best: bool
    local_early_stopping: bool
    evaluate_final: bool
    seed: int
    save_dir: Path
    model_extra_args: dict[str, Any] = field(default_factory=dict)
    task_extra_args: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        tasks = _task_registry()

        if self.model_name not in tasks:
            raise ValueError(
                f"Invalid model_name: {self.model_name}. "
                f"Available models: {list(tasks.keys())}"
            )

        if len(self.clients) == 0:
            raise ValueError("At least one client is required.")

        if self.rounds <= 0:
            raise ValueError("rounds must be positive.")

        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1].")

        if self.clients_per_round is not None:
            if self.clients_per_round <= 0:
                raise ValueError("clients_per_round must be positive.")
            if self.clients_per_round > len(self.clients):
                raise ValueError("clients_per_round cannot exceed number of clients.")

        if self.aggregation not in {"weighted", "uniform"}:
            raise ValueError("aggregation must be 'weighted' or 'uniform'.")

        if self.eval_every < 0:
            raise ValueError("eval_every must be >= 0.")

        if self.save_every <= 0:
            raise ValueError("save_every must be positive.")

        model_tasks = tasks[self.model_name]
        if self.task_name not in model_tasks:
            raise ValueError(
                f"Invalid task_name '{self.task_name}' for model "
                f"'{self.model_name}'. Available tasks: {list(model_tasks.keys())}"
            )

        for client in self.clients:
            if client.task_name not in model_tasks:
                raise ValueError(
                    f"Invalid task_name '{client.task_name}' for model "
                    f"'{self.model_name}'. Available tasks: {list(model_tasks.keys())}"
                )

    def selected_clients(self, round_idx: int) -> list[ClientConfig]:
        if self.clients_per_round is not None:
            n_selected = self.clients_per_round
        else:
            n_selected = max(1, math.ceil(len(self.clients) * self.client_fraction))

        if n_selected == len(self.clients):
            return list(self.clients)

        rng = random.Random(self.seed + round_idx)
        return rng.sample(self.clients, n_selected)


def parse_client_names(clients: str) -> list[str]:
    names = [name.strip() for name in clients.split(",")]
    names = [name for name in names if name]

    if len(names) == 0:
        raise ValueError("No clients were provided.")

    return names


def canonical_client_name(client_name: str) -> str:
    return canonical_dataset_name(client_name)


def get_client_extra_args(
    client_extra_args: dict[str, Any],
    client_name: str,
    task_name: str,
) -> dict[str, Any]:
    if client_name in client_extra_args:
        return dict(client_extra_args[client_name])
    if task_name in client_extra_args:
        return dict(client_extra_args[task_name])
    return {}


def _parse_sample_counts(value: Any, name: str) -> list[int | None] | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    elif isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, str):
        value = value.strip()

        if value == "":
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
        return [None for _ in range(n_clients)]

    if len(parsed) != n_clients:
        raise ValueError(
            f"{name} must have one value per client. "
            f"Expected {n_clients}, got {len(parsed)}."
        )

    return parsed


def make_client_configs(
    clients: str,
    task_name: str,
    n_train_samples: Any,
    n_val_samples: Any,
    n_test_samples: Any,
    client_task_extra_args: dict[str, Any] | None = None,
) -> list[ClientConfig]:
    client_task_extra_args = client_task_extra_args or {}
    configs: list[ClientConfig] = []
    client_names = parse_client_names(clients)
    train_counts = _sample_counts_for_clients(
        value=n_train_samples,
        n_clients=len(client_names),
        name="n_train_samples",
    )
    val_counts = _sample_counts_for_clients(
        value=n_val_samples,
        n_clients=len(client_names),
        name="n_val_samples",
    )
    test_counts = _sample_counts_for_clients(
        value=n_test_samples,
        n_clients=len(client_names),
        name="n_test_samples",
    )

    for idx, client in enumerate(client_names):
        name = canonical_client_name(client)
        h5_path, csv_path = get_dataset_paths(name)

        configs.append(
            ClientConfig(
                name=name,
                dataset_name=name,
                task_name=task_name,
                h5_path=h5_path,
                csv_path=csv_path,
                n_train_samples=train_counts[idx],
                n_val_samples=val_counts[idx],
                n_test_samples=test_counts[idx],
                task_extra_args=get_client_extra_args(
                    client_extra_args=client_task_extra_args,
                    client_name=name,
                    task_name=task_name,
                ),
            )
        )

    return configs


def build_task(
    *,
    model_name: str,
    client: ClientConfig,
    model_extra_args: dict[str, Any],
    task_extra_args: dict[str, Any],
    world_size: int,
):
    tasks = _task_registry()
    model_tasks = tasks[model_name]

    merged_task_extra_args = dict(task_extra_args)
    merged_task_extra_args.update(client.task_extra_args)

    return model_tasks[client.task_name](
        h5_path=client.h5_path,
        csv_path=client.csv_path,
        model_extra_args=model_extra_args,
        task_extra_args=merged_task_extra_args,
        n_train_samples=client.n_train_samples,
        n_val_samples=client.n_val_samples,
        n_test_samples=client.n_test_samples,
        world_size=world_size,
    )
