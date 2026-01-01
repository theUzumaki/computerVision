"""Training orchestrator - creates reproducible experiment runs.

Implement training logic in your `Trainer` and `DataLoader`.

Expected minimal interfaces (convention, not enforced):
"""

class Trainer:
    def create_model(self, model_cfg: dict) -> Any:
        """Return a model (any object) configured using model_cfg.

        Config mapping:
          - `nn_training.model.name` or other `nn_training.model.*` keys -> used to select architecture / model args
        """
        raise NotImplementedError

    def train(self, model, train_data, val_data=None, *, epochs=10, batch_size=32, learning_rate=1e-3, checkpoint_dir=None, save_frequency=10, device: str = "cpu", resume: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run training and save checkpoints to `checkpoint_dir`.

        Default values and config mappings (where runner will pull defaults):
          - epochs -> `nn_training.training.epochs`
          - batch_size -> `nn_training.training.batch_size`
          - learning_rate -> `nn_training.training.learning_rate`
          - checkpoint_dir -> `nn_training.checkpoints.dir`
          - save_frequency -> `nn_training.checkpoints.save_frequency`
          - device -> `nn_training.training.device` (optional)
          - seed -> `nn_training.training.seed` (optional)
          - resume -> CLI `--resume` or explicit path

        The method should save periodic checkpoints and update a history dict with
        lists for time-series metrics and keys `best_checkpoint` and `best_metric`.

        Expected history schema (returned dict or saved to `results/history.json`):
          - `epochs` (optional): [1, 2, ...] (if missing, epoch indices are implied by list positions)
          - `train_loss`, `val_loss`: lists of floats per epoch
          - `train_acc`, `val_acc`: lists of floats per epoch (optional)
          - `lr`: list of learning-rate values per epoch (optional)
          - `best_checkpoint`: string path to best checkpoint file (optional, recommended)
          - `best_metric`: float value of the metric used to select the best checkpoint (optional)

        The runner expects epoch-aligned lists so `results/history.csv` will be a clean table.

        Return a history dict following the schema above (example: {
            "epochs": [1,2,3], "train_loss": [...], "val_loss": [...],
            "best_checkpoint": "experiments/.../checkpoints/best_checkpoint.pth",
            "best_metric": 0.87
        }).
        """
        raise NotImplementedError

class DataLoader:
    def load(self, path: str) -> Any:
        """Load data from the given path and return it."""
        raise NotImplementedError
    
    def split(self, data) -> (train, val, test):
        """Split data into train, validation, and test sets."""
        raise NotImplementedError

"""
Usage example (from command line):
    python -m nn_training.train --trainer mypkg.Trainer --dataloader mypkg.DataLoader --exp-name quicktest

"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import load_config, Config

import torch


# --- Helpers ---------------------------------------------------------------

def import_from_string(path: str):
    """Import class from dotted path (pkg.module.Class or pkg.module:Class)."""
    if ":" in path:
        module_path, symbol = path.split(":", 1)
    else:
        parts = path.split(".")
        module_path = ".".join(parts[:-1])
        symbol = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, symbol)

def now_ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def setup_experiment(cfg: Config, exp_name: Optional[str], seed: Optional[int] = None) -> Dict[str, Any]:
    """Create experiment directory structure and save metadata."""
    root = Path(cfg.get_experiments_dir() or "experiments")
    timestamp = now_ts()
    run_id = f"{timestamp}_{exp_name}" if exp_name else timestamp
    exp_root = root / run_id

    ckpt_dir = exp_root / "checkpoints"
    logs_dir = exp_root / "logs"
    results_dir = exp_root / "results"

    for d in (ckpt_dir, logs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save config copy
    cfg_src = Path("configs/default_config.yaml")
    if cfg_src.exists():
        (exp_root / "config_used.yaml").write_text(cfg_src.read_text())

    # Save metadata
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "exp_name": exp_name,
        "seed": seed,
        "python_version": sys.version.splitlines()[0],
    }

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        metadata["git_commit"] = commit
    except Exception:
        pass

    save_json(exp_root / "metadata.json", metadata)

    return {
        "root": exp_root,
        "checkpoints_dir": ckpt_dir,
        "logs_dir": logs_dir,
        "results_dir": results_dir,
        "metadata": metadata,
    }


# --- History writers ------------------------------------------------------


def write_history_json(results_dir: Path, history: Dict[str, Any]) -> None:
    save_json(results_dir / "history.json", history)

def write_history_csv(results_dir: Path, history: Dict[str, Any]) -> None:
    """Write epoch-aligned metrics to CSV."""
    if not history:
        return

    keys = [k for k, v in history.items() if isinstance(v, list)]
    if not keys:
        return

    length = len(history[keys[0]])
    rows = [{k: (history[k][i] if i < len(history[k]) else None) for k in keys} for i in range(length)]

    with open(results_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_training(args: argparse.Namespace) -> int:
    """Main training orchestrator."""
    cfg = Config(load_config(args.config) if args.config else load_config())
    exp = setup_experiment(cfg, args.exp_name, seed=args.seed)
    print(f"Experiment: {exp['root']}")

    # Instantiate components
    trainer = import_from_string(args.trainer)()
    dataloader = import_from_string(args.dataloader)()

    # Load and split data
    data_dir = cfg.get_data_dir("processed_dir") or cfg.get_data_dir("raw_dir")
    data = dataloader.load(data_dir)
    
    try:
        train_data, val_data, _ = dataloader.split(data)
    except Exception:
        train_data, val_data = data, None

    # Create model
    model = trainer.create_model(cfg.get_model_config())

    # Resolve training params (CLI > config > defaults)
    training_params = cfg.get_training_params()
    epochs = args.epochs if args.epochs is not None else int(training_params.get("epochs", 10))
    batch_size = args.batch_size if args.batch_size is not None else int(training_params.get("batch_size", 32))
    learning_rate = args.lr if args.lr is not None else float(training_params.get("learning_rate", 1e-3))
    save_frequency = args.save_every if args.save_every is not None else int(cfg._cfg.get("nn_training", {}).get("checkpoints", {}).get("save_frequency", 10))
    device = args.device or cfg._cfg.get("nn_training", {}).get("training", {}).get("device", "cpu")
    seed = args.seed if args.seed is not None else cfg._cfg.get("nn_training", {}).get("training", {}).get("seed")

    # Train
    history = trainer.train(
        model,
        train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=str(exp["checkpoints_dir"]),
        save_frequency=save_frequency,
        device=device,
        resume=args.resume,
        seed=seed,
    )

    # Save history
    write_history_json(exp["results_dir"], history or {})
    write_history_csv(exp["results_dir"], history or {})

    print(f"Training complete. Artifacts: {exp['root']}")
    return 0

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training runner")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--trainer", required=True, help="Dotted path to Trainer class")
    p.add_argument("--dataloader", required=True, help="Dotted path to DataLoader class")
    p.add_argument("--exp-name", default=None, help="Experiment name")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_training(args))
