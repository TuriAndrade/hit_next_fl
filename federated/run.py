from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp

from utils import parse_args_json, set_seed

from .config import FederatedConfig, make_client_configs
from .distributed import cleanup_ddp, setup_ddp
from .server import FederatedServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated ECG training.")

    parser.add_argument("--model-name", type=str, default="ecg_hit_next")
    parser.add_argument("--task-name", type=str, default="clf")
    parser.add_argument("--clients", type=str, default="code15,ptbxl,chapman")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--aggregation", type=str, choices=["weighted", "uniform"], default="weighted")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--keep-client-ckpts", action="store_true")
    parser.add_argument("--local-keep-best", action="store_true")
    parser.add_argument("--local-early-stopping", action="store_true")
    parser.add_argument("--no-final-eval", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--model-extra-args", type=json.loads, default={})
    parser.add_argument("--task-extra-args", type=json.loads, default={})
    parser.add_argument("--client-task-extra-args", type=json.loads, default={})
    parser.add_argument("--n-train-samples", type=str, default=None)
    parser.add_argument("--n-val-samples", type=str, default=None)
    parser.add_argument("--n-test-samples", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def _save_dir(args) -> Path:
    base = Path(os.environ.get("SAVE_DIR", "experiments"))
    save_dir = base / "federated" / f"{args.model_name}-fedavg-seed_{args.seed}"

    if args.save_name is not None:
        save_dir = save_dir / args.save_name

    return save_dir


def _build_config(args) -> FederatedConfig:
    return FederatedConfig(
        model_name=args.model_name,
        task_name=args.task_name,
        clients=make_client_configs(
            clients=args.clients,
            task_name=args.task_name,
            n_train_samples=args.n_train_samples,
            n_val_samples=args.n_val_samples,
            n_test_samples=args.n_test_samples,
            client_task_extra_args=args.client_task_extra_args,
        ),
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=args.client_fraction,
        clients_per_round=args.clients_per_round,
        aggregation=args.aggregation,
        eval_every=args.eval_every,
        save_every=args.save_every,
        keep_client_ckpts=args.keep_client_ckpts,
        local_keep_best=args.local_keep_best,
        local_early_stopping=args.local_early_stopping,
        evaluate_final=not args.no_final_eval,
        seed=args.seed,
        save_dir=_save_dir(args),
        model_extra_args=args.model_extra_args,
        task_extra_args=args.task_extra_args,
    )


def _train_worker(rank: int, world_size: int, args) -> None:
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    if world_size > 1:
        setup_ddp(
            rank=rank,
            world_size=world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )

    try:
        set_seed(args.seed)

        config = _build_config(args)
        server = FederatedServer(
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
            resume_from=args.resume_from,
        )

        summary = server.run()

        if rank == 0:
            print(f"Finished federated run: {summary}")

    finally:
        gc.collect()
        torch.cuda.empty_cache()

        if world_size > 1:
            cleanup_ddp()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs found.")

    print(f"Using {world_size} GPU(s)")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Clients: {args.clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Seed: {args.seed}")

    if world_size == 1:
        _train_worker(rank=0, world_size=1, args=args)
    else:
        mp.spawn(
            _train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
