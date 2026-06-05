from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]


def _normalize_weights(weights: Sequence[float], n_items: int) -> list[float]:
    if len(weights) != n_items:
        raise ValueError(f"Expected {n_items} weights, got {len(weights)}.")

    weights = [float(w) for w in weights]

    if any(w < 0 for w in weights):
        raise ValueError("Aggregation weights must be non-negative.")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("At least one aggregation weight must be positive.")

    return [w / weight_sum for w in weights]


def _validate_state_dicts(state_dicts: Sequence[StateDict]) -> list[str]:
    if len(state_dicts) == 0:
        raise ValueError("At least one state_dict is required.")

    keys = list(state_dicts[0].keys())
    key_set = set(keys)

    for idx, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != key_set:
            missing = key_set - set(state_dict.keys())
            extra = set(state_dict.keys()) - key_set
            raise ValueError(
                f"state_dict {idx} has incompatible keys. "
                f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
            )

    for key in keys:
        shape = tuple(state_dicts[0][key].shape)
        dtype = state_dicts[0][key].dtype

        for idx, state_dict in enumerate(state_dicts[1:], start=1):
            value = state_dict[key]
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"state_dict {idx} tensor '{key}' has shape "
                    f"{tuple(value.shape)}, expected {shape}."
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"state_dict {idx} tensor '{key}' has dtype "
                    f"{value.dtype}, expected {dtype}."
                )

    return keys


def fedavg_state_dicts(
    state_dicts: Sequence[StateDict],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Aggregate model weights with FedAvg.

    Floating tensors are averaged. Non-floating tensors, such as integer
    counters, are copied from the first client because averaging them is not
    meaningful for model state.
    """
    keys = _validate_state_dicts(state_dicts)

    if weights is None:
        weights = [1.0 for _ in state_dicts]

    normalized_weights = _normalize_weights(weights, len(state_dicts))
    aggregated: dict[str, torch.Tensor] = {}

    for key in keys:
        first = state_dicts[0][key].detach().cpu()

        if not torch.is_floating_point(first):
            aggregated[key] = first.clone()
            continue

        accum_dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
        accum = torch.zeros_like(first, dtype=accum_dtype)

        for state_dict, weight in zip(state_dicts, normalized_weights):
            value = state_dict[key].detach().cpu().to(dtype=accum_dtype)
            accum.add_(value, alpha=weight)

        aggregated[key] = accum.to(dtype=first.dtype)

    return aggregated
