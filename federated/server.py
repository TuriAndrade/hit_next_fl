from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import save_config, set_seed

from .aggregation import aggregate_scalars, fedavg_state_dicts
from .client import ClientUpdate, ClientValidation, FederatedClient
from .config import FederatedConfig
from .distributed import barrier, broadcast_object, broadcast_state_dict


class FederatedServer:
    def __init__(
        self,
        *,
        config: FederatedConfig,
        clients: Sequence[FederatedClient],
        model_factory: Callable[[], nn.Module],
        device: str,
        rank: int = 0,
        world_size: int = 1,
        resume_from: str | Path | None = None,
    ):
        self.config = config
        self.model_factory = model_factory
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.resume_from = Path(resume_from) if resume_from else None

        self.clients = {client.name: client for client in clients}
        self.history: list[dict[str, Any]] = []
        self.start_round = 1
        self.best_round = 0
        self.best_validation_loss = math.inf
        self.rounds_without_improvement = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.last_validation_loss: float | None = None

        self.config.validate()
        self._validate_clients()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _log(self, message: str) -> None:
        if self.is_main_process:
            tqdm.write(message)

    def _validate_clients(self) -> None:
        expected = set(self.config.client_names)
        actual = set(self.clients)

        if actual != expected:
            raise ValueError(
                "Configured clients do not match server clients. "
                f"Missing: {sorted(expected - actual)}. "
                f"Unexpected: {sorted(actual - expected)}."
            )

    @staticmethod
    def _snapshot_state(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        }

    def _create_initial_state(self) -> dict[str, torch.Tensor] | None:
        if not self.is_main_process:
            return None

        set_seed(self.config.seed)
        model = self.model_factory()
        state_dict = self._snapshot_state(model.state_dict())
        del model
        return state_dict

    def _load_training_state(
        self,
    ) -> tuple[int, dict[str, torch.Tensor] | None]:
        if self.resume_from is None:
            return 1, self._create_initial_state()

        checkpoint = None
        if self.is_main_process:
            checkpoint = torch.load(self.resume_from, map_location="cpu")
            self.history = list(checkpoint.get("history", []))
            self.best_round = int(checkpoint.get("best_round", 0))
            self.best_validation_loss = float(
                checkpoint.get("best_validation_loss", math.inf)
            )
            self.rounds_without_improvement = int(
                checkpoint.get("rounds_without_improvement", 0)
            )

            last_loss = checkpoint.get("last_validation_loss")
            if last_loss is not None:
                self.last_validation_loss = float(last_loss)

            if (
                self.config.keep_best
                and checkpoint.get("selection") == "best"
            ):
                self.best_state = self._snapshot_state(checkpoint["model"])

        last_round = broadcast_object(
            int(checkpoint.get("last_round", checkpoint["round"]))
            if self.is_main_process
            else None,
            src=0,
        )
        state_dict = None
        if self.is_main_process:
            state_dict = checkpoint.get("last_model", checkpoint["model"])

        return last_round + 1, state_dict

    def _aggregation_weights(
        self,
        values: Sequence[ClientUpdate | ClientValidation],
    ) -> list[float]:
        if self.config.aggregation == "uniform":
            return [1.0] * len(values)

        return [value.num_examples for value in values]

    def _aggregate_updates(
        self,
        updates: Sequence[ClientUpdate],
    ) -> dict[str, torch.Tensor] | None:
        if not self.is_main_process:
            return None

        state_dicts = []
        for update in updates:
            if update.state_dict is None:
                raise RuntimeError(
                    f"Missing model update from client {update.name}."
                )
            state_dicts.append(update.state_dict)

        return fedavg_state_dicts(
            state_dicts=state_dicts,
            weights=self._aggregation_weights(updates),
        )

    def _train_round(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        round_idx: int,
    ) -> list[ClientUpdate]:
        selected_names = self.config.selected_client_names(round_idx)
        self._log(
            f"Federated round {round_idx}/{self.config.rounds}: "
            f"{', '.join(selected_names)}"
        )

        updates = []
        for name in selected_names:
            update = self.clients[name].train(
                global_state_dict=global_state,
                round_idx=round_idx,
                local_epochs=self.config.local_epochs,
                keep_checkpoint=self.config.keep_client_ckpts,
                save_dir=(
                    self.config.save_dir
                    / "clients"
                    / f"round_{round_idx:03d}"
                    / name
                ),
                seed=self.config.seed,
            )
            updates.append(update)
            barrier()

        return updates

    def _validate_global_model(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        round_idx: int,
    ) -> dict[str, Any] | None:
        results = []
        for name in self.config.client_names:
            self._log(f"Validating global model on {name}")
            results.append(
                self.clients[name].validate(
                    global_state_dict=global_state,
                    save_dir=self.config.save_dir,
                    seed=self.config.seed,
                )
            )

        if not self.is_main_process:
            return None

        weights = self._aggregation_weights(results)
        weight_sum = sum(weights)
        validation_loss = aggregate_scalars(
            values=[result.loss for result in results],
            weights=weights,
        )
        summary = {
            "round": round_idx,
            "aggregation": self.config.aggregation,
            "loss": validation_loss,
            "clients": [
                {
                    "name": result.name,
                    "loss": result.loss,
                    "num_examples": result.num_examples,
                    "aggregation_weight": weight / weight_sum,
                }
                for result, weight in zip(results, weights)
            ],
        }
        self._log(
            f"Round {round_idx} validation loss={validation_loss:.6f} "
            f"({self.config.aggregation})"
        )
        return summary

    def _update_best(
        self,
        *,
        round_idx: int,
        validation_loss: float,
        global_state: dict[str, torch.Tensor],
    ) -> bool:
        self.last_validation_loss = validation_loss
        improved = validation_loss < self.best_validation_loss

        if improved:
            self.best_round = round_idx
            self.best_validation_loss = validation_loss
            self.rounds_without_improvement = 0

            if self.config.keep_best:
                self.best_state = self._snapshot_state(global_state)
        else:
            self.rounds_without_improvement += 1

        return improved

    def _record_round(
        self,
        *,
        round_idx: int,
        updates: Sequence[ClientUpdate],
        validation: dict[str, Any],
        improved: bool,
    ) -> None:
        self.history.append(
            {
                "round": round_idx,
                "clients": [
                    {
                        "name": update.name,
                        "num_examples": update.num_examples,
                        "summary": update.summary,
                    }
                    for update in updates
                ],
                "validation": validation,
                "improved": improved,
                "best_round": self.best_round,
                "best_validation_loss": self.best_validation_loss,
                "rounds_without_improvement": (
                    self.rounds_without_improvement
                ),
            }
        )
        save_config(
            {"history": self.history},
            self.config.save_dir / "history.json",
        )

    def _save_run_config(self) -> None:
        if not self.is_main_process:
            return

        save_config(
            {
                "federated": self.config,
                "runtime": {
                    "world_size": self.world_size,
                    "device": self.device,
                    "resume_from": self.resume_from,
                },
            },
            self.config.save_dir / "config.json",
        )

    def _select_final_state(
        self,
        *,
        last_round: int,
        last_state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self.config.keep_best:
            if self.best_state is None:
                raise RuntimeError("Best model state was not recorded.")

            return self.best_state, {
                "selection": "best",
                "round": self.best_round,
                "validation_loss": self.best_validation_loss,
            }

        return last_state, {
            "selection": "last",
            "round": last_round,
            "validation_loss": self.last_validation_loss,
        }

    def _save_final_checkpoint(
        self,
        *,
        last_round: int,
        last_state: dict[str, torch.Tensor],
        selected_state: dict[str, torch.Tensor],
        selected: dict[str, Any],
    ) -> None:
        if not self.is_main_process:
            return

        checkpoint = {
            "round": last_round,
            "last_round": last_round,
            "selected_round": selected["round"],
            "selection": selected["selection"],
            "validation_loss": selected["validation_loss"],
            "last_validation_loss": self.last_validation_loss,
            "best_round": self.best_round,
            "best_validation_loss": self.best_validation_loss,
            "rounds_without_improvement": self.rounds_without_improvement,
            "model": selected_state,
            "history": self.history,
            "config": self.config,
        }
        if selected["selection"] == "best" and selected["round"] != last_round:
            checkpoint["last_model"] = last_state

        filename = (
            "global_best.pt"
            if selected["selection"] == "best"
            else "global_last.pt"
        )
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.config.save_dir / filename)

    def _evaluate_final_model(
        self,
        *,
        selected_state: dict[str, torch.Tensor],
        selected: dict[str, Any],
    ) -> dict[str, Any] | None:
        summaries = {}
        eval_root = self.config.save_dir / "final_eval"

        for name in self.config.client_names:
            self._log(f"Evaluating final model on {name}")
            summary = self.clients[name].evaluate(
                global_state_dict=selected_state,
                save_dir=eval_root / name,
                seed=self.config.seed + selected["round"],
            )
            if self.is_main_process:
                summaries[name] = summary

        if not self.is_main_process:
            return None

        output = {
            "round": selected["round"],
            "selection": selected["selection"],
            "clients": summaries,
        }
        save_config(output, eval_root / "eval_summary.json")
        return output

    def run(self) -> dict[str, Any] | None:
        self._save_run_config()

        self.start_round, global_state = self._load_training_state()
        global_state = broadcast_state_dict(
            global_state,
            device=self.device,
            src=0,
        )

        last_round = self.start_round - 1
        stopped_early = False

        for round_idx in range(self.start_round, self.config.rounds + 1):
            last_round = round_idx
            updates = self._train_round(
                global_state=global_state,
                round_idx=round_idx,
            )
            global_state = self._aggregate_updates(updates)
            global_state = broadcast_state_dict(
                global_state,
                device=self.device,
                src=0,
            )

            validation = self._validate_global_model(
                global_state=global_state,
                round_idx=round_idx,
            )

            decision = None
            if self.is_main_process:
                assert validation is not None
                improved = self._update_best(
                    round_idx=round_idx,
                    validation_loss=float(validation["loss"]),
                    global_state=global_state,
                )
                self._record_round(
                    round_idx=round_idx,
                    updates=updates,
                    validation=validation,
                    improved=improved,
                )
                decision = {
                    "stop": (
                        self.config.early_stopping
                        and self.rounds_without_improvement
                        >= self.config.patience
                    ),
                    "best_round": self.best_round,
                    "best_validation_loss": self.best_validation_loss,
                }

            decision = broadcast_object(decision, src=0)
            if decision["stop"]:
                stopped_early = True
                self._log(
                    f"Early stopping at round {round_idx}. "
                    f"Best loss={decision['best_validation_loss']:.6f} "
                    f"at round {decision['best_round']}."
                )
                break

        selected_state = None
        selected = None
        if self.is_main_process:
            selected_state, selected = self._select_final_state(
                last_round=last_round,
                last_state=global_state,
            )
            self._save_final_checkpoint(
                last_round=last_round,
                last_state=global_state,
                selected_state=selected_state,
                selected=selected,
            )

        selected = broadcast_object(selected, src=0)
        selected_state = broadcast_state_dict(
            selected_state,
            device=self.device,
            src=0,
        )
        final_eval = self._evaluate_final_model(
            selected_state=selected_state,
            selected=selected,
        )

        if not self.is_main_process:
            return None

        summary = {
            "rounds_ran": max(0, last_round - self.start_round + 1),
            "start_round": self.start_round,
            "last_round": last_round,
            "stopped_early": stopped_early,
            "selection": selected["selection"],
            "selected_round": selected["round"],
            "selected_validation_loss": selected["validation_loss"],
            "best_round": self.best_round,
            "best_validation_loss": self.best_validation_loss,
            "clients": self.config.client_names,
            "history": self.history,
            "final_eval": final_eval,
        }
        save_config(
            summary,
            self.config.save_dir / "federated_summary.json",
        )
        return summary
