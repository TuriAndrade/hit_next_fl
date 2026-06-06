from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from federated.aggregation import aggregate_scalars, fedavg_state_dicts
from federated.client import ClientUpdate, ClientValidation, FederatedClient
from federated.config import (
    ClientConfig,
    FederatedConfig,
    create_client_configs,
    parse_client_names,
)
from federated.run import _build_parser
from federated.server import FederatedServer
from tasks import TaskDefinition, get_task_definition


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


def test_aggregate_scalars_uniformly():
    value = aggregate_scalars([1.0, 3.0])

    assert value == pytest.approx(2.0)


def test_aggregate_scalars_with_weights():
    value = aggregate_scalars([1.0, 3.0], weights=[3.0, 1.0])

    assert value == pytest.approx(1.5)


def test_aggregate_scalars_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_scalars([1.0, math.nan])


def test_client_aliases():
    assert parse_client_names("code15,ptbxl,chapman-shaoxing") == [
        "code15",
        "ptbxl",
        "chapman",
    ]


def test_create_client_configs_uses_per_client_samples():
    clients = create_client_configs(
        clients="code15,ptbxl,chapman",
        n_train_samples="10,20,30",
        n_val_samples="[2, null, 4]",
        n_test_samples=[3, 5, None],
    )

    assert [client.name for client in clients] == ["code15", "ptbxl", "chapman"]
    assert [client.h5_path for client in clients] == [
        "/tmp/code15.h5",
        "/tmp/ptbxl.h5",
        "/tmp/chapman.h5",
    ]
    assert [client.n_train_samples for client in clients] == [10, 20, 30]
    assert [client.n_val_samples for client in clients] == [2, None, 4]
    assert [client.n_test_samples for client in clients] == [3, 5, None]


def test_create_client_configs_rejects_sample_count_length_mismatch():
    with pytest.raises(ValueError, match="one value per client"):
        create_client_configs(
            clients="code15,ptbxl,chapman",
            n_train_samples="10,20",
            n_val_samples=None,
            n_test_samples=None,
        )


def test_selected_clients_is_deterministic():
    config = FederatedConfig(
        model_name="ecg_hit_next",
        task_name="clf",
        client_names=["code15", "ptbxl", "chapman"],
        rounds=2,
        local_epochs=1,
        client_fraction=0.67,
        clients_per_round=None,
        aggregation="weighted",
        keep_best=True,
        early_stopping=True,
        patience=5,
        keep_client_ckpts=False,
        seed=7,
        save_dir=Path("unused"),
    )

    first = config.selected_client_names(1)
    second = config.selected_client_names(1)

    assert first == second
    assert len(first) == 3


def test_federated_cli_uses_final_only_evaluation_options():
    parser = _build_parser()
    args = parser.parse_args(
        ["--keep-best", "--early-stopping", "--patience", "3"]
    )

    assert args.keep_best is True
    assert args.early_stopping is True
    assert args.patience == 3
    assert not hasattr(args, "eval_every")
    assert not hasattr(args, "save_every")
    assert not hasattr(args, "no_final_eval")
    assert not hasattr(args, "local_keep_best")
    assert not hasattr(args, "local_early_stopping")
    assert not hasattr(args, "client_task_extra_args")


def test_server_tracks_minimum_validation_loss():
    server = object.__new__(FederatedServer)
    server.config = SimpleNamespace(keep_best=True)
    server.best_validation_loss = math.inf
    server.best_round = 0
    server.rounds_without_improvement = 0
    server.best_state = None
    server.last_validation_loss = None

    first_state = {"weight": torch.tensor([1.0])}
    second_state = {"weight": torch.tensor([2.0])}

    assert server._update_best(
        round_idx=1,
        validation_loss=0.5,
        global_state=first_state,
    )
    assert not server._update_best(
        round_idx=2,
        validation_loss=0.6,
        global_state=second_state,
    )

    assert server.best_round == 1
    assert server.best_validation_loss == pytest.approx(0.5)
    assert server.rounds_without_improvement == 1
    assert torch.equal(server.best_state["weight"], first_state["weight"])


def test_task_factory_uses_explicit_model():
    definition = get_task_definition("ecg_hit_next", "clf")
    model = nn.Identity()

    task = definition.create_task(
        model=model,
        h5_path="/tmp/data.h5",
        csv_path="/tmp/data.csv",
        task_extra_args={"epochs": 3},
        world_size=1,
    )

    assert task.model is model
    assert task.epochs == 3


def test_client_applies_local_epochs_without_mutating_shared_task_args():
    classification = get_task_definition("ecg_hit_next", "clf")
    definition = TaskDefinition(
        model_factory=lambda model_extra_args=None: nn.Identity(),
        task_factory=classification.task_factory,
    )
    client = FederatedClient(
        config=ClientConfig(
            name="code15",
            h5_path="/tmp/data.h5",
            csv_path="/tmp/data.csv",
        ),
        task_definition=definition,
        model_extra_args={},
        task_extra_args={"epochs": 100},
        device="cuda:0",
        rank=0,
        world_size=1,
    )

    task = client._create_task(epochs=3)

    assert task.epochs == 3
    assert client.task_extra_args["epochs"] == 100


def test_server_initializes_model_without_client_task():
    config = FederatedConfig(
        model_name="ecg_hit_next",
        task_name="clf",
        client_names=["code15"],
        rounds=1,
        local_epochs=1,
        client_fraction=1.0,
        clients_per_round=None,
        aggregation="weighted",
        keep_best=False,
        early_stopping=False,
        patience=5,
        keep_client_ckpts=False,
        seed=0,
        save_dir=Path("unused"),
    )
    server = FederatedServer(
        config=config,
        clients=[SimpleNamespace(name="code15")],
        model_factory=lambda: nn.Linear(2, 1),
        device="cuda:0",
    )

    state = server._create_initial_state()

    assert set(state) == {"weight", "bias"}


def test_server_selects_best_global_round(tmp_path):
    class FakeClient:
        name = "code15"

        def __init__(self):
            self.validation_losses = iter([0.5, 0.4, 0.6])
            self.evaluated_weight = None

        def train(self, *, round_idx, **kwargs):
            return ClientUpdate(
                name=self.name,
                round_idx=round_idx,
                num_examples=10,
                state_dict={"weight": torch.tensor([[float(round_idx)]])},
                summary={},
            )

        def validate(self, **kwargs):
            return ClientValidation(
                name=self.name,
                loss=next(self.validation_losses),
                num_examples=5,
            )

        def evaluate(self, *, global_state_dict, **kwargs):
            self.evaluated_weight = global_state_dict["weight"].item()
            return {"ok": True}

    config = FederatedConfig(
        model_name="ecg_hit_next",
        task_name="clf",
        client_names=["code15"],
        rounds=5,
        local_epochs=1,
        client_fraction=1.0,
        clients_per_round=None,
        aggregation="weighted",
        keep_best=True,
        early_stopping=True,
        patience=1,
        keep_client_ckpts=False,
        seed=0,
        save_dir=tmp_path,
    )
    client = FakeClient()
    server = FederatedServer(
        config=config,
        clients=[client],
        model_factory=lambda: nn.Linear(1, 1, bias=False),
        device="cpu",
    )

    summary = server.run()

    assert summary["stopped_early"] is True
    assert summary["last_round"] == 3
    assert summary["selected_round"] == 2
    assert summary["selected_validation_loss"] == pytest.approx(0.4)
    assert client.evaluated_weight == pytest.approx(2.0)
    assert (tmp_path / "global_best.pt").exists()
