from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .ecg_hit_next_classification import (
    create_model as create_ecg_hit_next_classification_model,
    create_task as create_ecg_hit_next_classification_task,
)
from .ecg_supervised_task import ECGSupervisedTask


@dataclass(frozen=True)
class TaskDefinition:
    model_factory: Callable[..., nn.Module]
    task_factory: Callable[..., ECGSupervisedTask]

    def create_model(self, model_extra_args: dict | None = None) -> nn.Module:
        return self.model_factory(model_extra_args=model_extra_args)

    def create_task(self, **kwargs) -> ECGSupervisedTask:
        return self.task_factory(**kwargs)


tasks = {
    "ecg_hit_next": {
        "clf": TaskDefinition(
            model_factory=create_ecg_hit_next_classification_model,
            task_factory=create_ecg_hit_next_classification_task,
        ),
    },
}


def get_task_definition(model_name: str, task_name: str) -> TaskDefinition:
    if model_name not in tasks:
        raise ValueError(
            f"Invalid model_name: {model_name}. "
            f"Available models: {list(tasks.keys())}"
        )

    model_tasks = tasks[model_name]
    if task_name not in model_tasks:
        raise ValueError(
            f"Invalid task_name '{task_name}' for model '{model_name}'. "
            f"Available tasks: {list(model_tasks.keys())}"
        )

    return model_tasks[task_name]


__all__ = ["TaskDefinition", "get_task_definition", "tasks"]
