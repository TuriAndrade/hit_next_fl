from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils import save_config, set_seed

from .aggregation import fedavg_state_dicts
from .client import ClientResult, evaluate_client, train_client
from .config import FederatedConfig, build_task
from .distributed import barrier, broadcast_object, broadcast_state_dict


class FederatedServer:
    def __init__(
        self,
        config: FederatedConfig,
        device: str,
        rank: int = 0,
        world_size: int = 1,
        resume_from: str | Path | None = None,
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.history: list[dict[str, Any]] = []
        self.start_round = 1

        self.config.validate()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _log(self, message: str) -> None:
        if self.is_main_process:
            tqdm.write(message)

    def _save_checkpoint(
        self,
        *,
        round_idx: int,
        state_dict: dict[str, torch.Tensor],
        filename: str,
    ) -> None:
        if not self.is_main_process:
            return

        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "round": round_idx,
                "model": state_dict,
                "history": self.history,
                "config": self.config,
            },
            self.config.save_dir / filename,
        )

    def _save_history(self) -> None:
        if self.is_main_process:
            save_config({"history": self.history}, self.config.save_dir / "history.json")

    def _build_initial_state(self) -> dict[str, torch.Tensor] | None:
        if not self.is_main_process:
            return None

        set_seed(self.config.seed)

        first_client = self.config.clients[0]
        task = build_task(
            model_name=self.config.model_name,
            client=first_client,
            model_extra_args=self.config.model_extra_args,
            task_extra_args=self.config.task_extra_args,
            world_size=self.world_size,
        )

        return {
            key: value.detach().cpu().clone()
            for key, value in task.model.state_dict().items()
        }

    def _load_resume_state(self) -> tuple[int, dict[str, torch.Tensor] | None]:
        if self.resume_from is None:
            return 1, self._build_initial_state()

        checkpoint = None
        if self.is_main_process:
            checkpoint = torch.load(self.resume_from, map_location="cpu")
            self.history = list(checkpoint.get("history", []))

        round_idx = broadcast_object(
            int(checkpoint["round"]) + 1 if self.is_main_process else None,
            src=0,
        )

        state_dict = checkpoint["model"] if self.is_main_process else None
        return round_idx, state_dict

    def _aggregation_weights(self, results: list[ClientResult]) -> list[float]:
        if self.config.aggregation == "uniform":
            return [1.0 for _ in results]

        return [result.num_examples for result in results]

    def _aggregate_results(
        self,
        results: list[ClientResult],
    ) -> dict[str, torch.Tensor] | None:
        if not self.is_main_process:
            return None

        state_dicts = []
        for result in results:
            if result.state_dict is None:
                raise RuntimeError(f"Missing state_dict for client {result.name}.")
            state_dicts.append(result.state_dict)

        return fedavg_state_dicts(
            state_dicts=state_dicts,
            weights=self._aggregation_weights(results),
        )

    def _round_record(
        self,
        *,
        round_idx: int,
        selected_results: list[ClientResult],
    ) -> dict[str, Any]:
        return {
            "round": round_idx,
            "clients": [
                {
                    "name": result.name,
                    "task_name": result.task_name,
                    "num_examples": result.num_examples,
                    "summary": result.summary,
                }
                for result in selected_results
            ],
        }

    def _save_config(self) -> None:
        if self.is_main_process:
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

    def _evaluate_global(
        self,
        *,
        state_dict: dict[str, torch.Tensor],
        round_idx: int,
        final: bool = False,
    ) -> dict[str, Any] | None:
        eval_root = self.config.save_dir / ("final_eval" if final else f"round_{round_idx:03d}_eval")
        summaries: dict[str, Any] = {}

        for client in self.config.clients:
            self._log(f"Evaluating global model on {client.name}")

            summary = evaluate_client(
                model_name=self.config.model_name,
                client=client,
                model_extra_args=self.config.model_extra_args,
                task_extra_args=self.config.task_extra_args,
                global_state_dict=state_dict,
                save_dir=eval_root / client.name,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size,
                seed=self.config.seed + round_idx,
            )

            if self.is_main_process:
                summaries[client.name] = summary

        if self.is_main_process:
            output = {
                "round": round_idx,
                "final": final,
                "clients": summaries,
            }
            save_config(output, eval_root / "eval_summary.json")
            return output

        return None

    def run(self) -> dict[str, Any] | None:
        self._save_config()

        self.start_round, global_state = self._load_resume_state()
        global_state = broadcast_state_dict(global_state, device=self.device, src=0)

        if self.is_main_process and self.start_round == 1:
            self._save_checkpoint(
                round_idx=0,
                state_dict=global_state,
                filename="global_initial.pt",
            )

        last_round = self.start_round - 1

        for round_idx in range(self.start_round, self.config.rounds + 1):
            last_round = round_idx
            selected_clients = self.config.selected_clients(round_idx)
            selected_names = ", ".join(client.name for client in selected_clients)
            self._log(f"Federated round {round_idx}/{self.config.rounds}: {selected_names}")

            results: list[ClientResult] = []

            for client in selected_clients:
                client_save_dir = (
                    self.config.save_dir
                    / "clients"
                    / f"round_{round_idx:03d}"
                    / client.name
                )

                result = train_client(
                    model_name=self.config.model_name,
                    client=client,
                    model_extra_args=self.config.model_extra_args,
                    task_extra_args=self.config.task_extra_args,
                    global_state_dict=global_state,
                    round_idx=round_idx,
                    local_epochs=self.config.local_epochs,
                    keep_client_ckpts=self.config.keep_client_ckpts,
                    local_keep_best=self.config.local_keep_best,
                    local_early_stopping=self.config.local_early_stopping,
                    save_dir=client_save_dir,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    seed=self.config.seed,
                )

                results.append(result)
                barrier()

            global_state = self._aggregate_results(results)
            global_state = broadcast_state_dict(global_state, device=self.device, src=0)

            if self.is_main_process:
                self.history.append(
                    self._round_record(
                        round_idx=round_idx,
                        selected_results=results,
                    )
                )
                self._save_history()
                self._save_checkpoint(
                    round_idx=round_idx,
                    state_dict=global_state,
                    filename="global_latest.pt",
                )

                if round_idx % self.config.save_every == 0:
                    self._save_checkpoint(
                        round_idx=round_idx,
                        state_dict=global_state,
                        filename=f"global_round_{round_idx:03d}.pt",
                    )

            if self.config.eval_every > 0 and round_idx % self.config.eval_every == 0:
                self._evaluate_global(
                    state_dict=global_state,
                    round_idx=round_idx,
                    final=False,
                )

        final_eval = None
        if self.config.evaluate_final:
            final_eval = self._evaluate_global(
                state_dict=global_state,
                round_idx=last_round,
                final=True,
            )

        if self.is_main_process:
            summary = {
                "rounds_ran": max(0, last_round - self.start_round + 1),
                "start_round": self.start_round,
                "last_round": last_round,
                "clients": [client.name for client in self.config.clients],
                "history": self.history,
                "final_eval": final_eval,
            }
            save_config(summary, self.config.save_dir / "federated_summary.json")
            return summary

        return None
