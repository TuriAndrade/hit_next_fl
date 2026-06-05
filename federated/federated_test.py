from __future__ import annotations

import pytest
import torch

from federated.aggregation import fedavg_state_dicts
from federated.config import (
    FederatedConfig,
    canonical_client_name,
    make_client_configs,
)


@pytest.fixture(autouse=True)
def dataset_env(monkeypatch):
    monkeypatch.setenv("CODE15_H5_PATH", "/tmp/code15.h5")
    monkeypatch.setenv("CODE15_CSV_PATH", "/tmp/code15.csv")
    monkeypatch.setenv("PTBXL_H5_PATH", "/tmp/ptbxl.h5")
    monkeypatch.setenv("PTBXL_CSV_PATH", "/tmp/ptbxl.csv")
    monkeypatch.setenv("CHAPMAN_H5_PATH", "/tmp/chapman.h5")
    monkeypatch.setenv("CHAPMAN_CSV_PATH", "/tmp/chapman.csv")


def test_fedavg_uniform_weights():
    states = [
        {"w": torch.tensor([1.0, 3.0])},
        {"w": torch.tensor([3.0, 5.0])},
    ]

    out = fedavg_state_dicts(states)

    assert torch.allclose(out["w"], torch.tensor([2.0, 4.0]))


def test_fedavg_weighted_by_examples():
    states = [
        {"w": torch.tensor([0.0])},
        {"w": torch.tensor([10.0])},
    ]

    out = fedavg_state_dicts(states, weights=[1, 3])

    assert torch.allclose(out["w"], torch.tensor([7.5]))


def test_fedavg_copies_non_float_tensors():
    states = [
        {"counter": torch.tensor([1], dtype=torch.long)},
        {"counter": torch.tensor([9], dtype=torch.long)},
    ]

    out = fedavg_state_dicts(states)

    assert out["counter"].item() == 1
    assert out["counter"].dtype == torch.long


def test_fedavg_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        fedavg_state_dicts(
            [
                {"a": torch.tensor([1.0])},
                {"b": torch.tensor([1.0])},
            ]
        )


def test_client_aliases():
    assert canonical_client_name("code15") == "code15"
    assert canonical_client_name("ptbxl") == "ptbxl"
    assert canonical_client_name("chapman-shaoxing") == "chapman"


def test_make_client_configs_uses_per_client_samples():
    clients = make_client_configs(
        clients="code15,ptbxl,chapman",
        task_name="clf",
        n_train_samples="10,20,30",
        n_val_samples="[2, null, 4]",
        n_test_samples=[3, 5, None],
    )

    assert [client.name for client in clients] == ["code15", "ptbxl", "chapman"]
    assert [client.dataset_name for client in clients] == ["code15", "ptbxl", "chapman"]
    assert [client.task_name for client in clients] == ["clf", "clf", "clf"]
    assert [client.h5_path for client in clients] == [
        "/tmp/code15.h5",
        "/tmp/ptbxl.h5",
        "/tmp/chapman.h5",
    ]
    assert [client.n_train_samples for client in clients] == [10, 20, 30]
    assert [client.n_val_samples for client in clients] == [2, None, 4]
    assert [client.n_test_samples for client in clients] == [3, 5, None]


def test_make_client_configs_rejects_sample_count_length_mismatch():
    with pytest.raises(ValueError, match="one value per client"):
        make_client_configs(
            clients="code15,ptbxl,chapman",
            task_name="clf",
            n_train_samples="10,20",
            n_val_samples=None,
            n_test_samples=None,
        )


def test_selected_clients_is_deterministic():
    clients = make_client_configs(
        clients="code15,ptbxl,chapman",
        task_name="clf",
        n_train_samples=None,
        n_val_samples=None,
        n_test_samples=None,
    )
    config = FederatedConfig(
        model_name="ecg_hit_next",
        task_name="clf",
        clients=clients,
        rounds=2,
        local_epochs=1,
        client_fraction=0.67,
        clients_per_round=None,
        aggregation="weighted",
        eval_every=0,
        save_every=1,
        keep_client_ckpts=False,
        local_keep_best=False,
        local_early_stopping=False,
        evaluate_final=False,
        seed=7,
        save_dir="unused",
    )

    first = [client.name for client in config.selected_clients(1)]
    second = [client.name for client in config.selected_clients(1)]

    assert first == second
    assert len(first) == 3
