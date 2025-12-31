"""
Minimal training orchestrator (nn_training/train.py)

Template for creating a reproducible run directory and collecting
essential artifacts (config copy, metadata, history, checkpoints).

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

        Return a history dict (e.g., {"train_loss": [...], "val_loss": [...], "best_checkpoint": "path"}).
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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import load_config, Config

import torch


# --- Helpers ---------------------------------------------------------------

def import_from_string(path: str):
    """Import a class or callable from a dotted string.

    Accepts 'pkg.module:Symbol' or 'pkg.module.Symbol'.
    """
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
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


# --- Experiment setup -----------------------------------------------------


def setup_experiment(cfg: Config, exp_name: Optional[str], seed: Optional[int] = None) -> Dict[str, Any]:
    root = Path(cfg.get_experiments_dir() or "experiments")
    timestamp = now_ts()
    run_id = f"{timestamp}_{exp_name}" if exp_name else timestamp
    exp_root = root / run_id

    ckpt_dir = exp_root / "checkpoints"
    logs_dir = exp_root / "logs"
    results_dir = exp_root / "results"

    for d in (ckpt_dir, logs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config YAML if present
    cfg_src = Path("configs/default_config.yaml")
    if cfg_src.exists():
        save_text(exp_root / "config_used.yaml", cfg_src.read_text())

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
        # not available - that's fine
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
    # Writes lists in history to CSV if possible
    if not history:
        return

    # If values are lists with equal length, write tabularly
    keys = [k for k, v in history.items() if isinstance(v, list)]
    if not keys:
        # fallback: write entire dict as json
        save_json(results_dir / "history_extra.json", history)
        return

    length = len(history[keys[0]])
    rows = []
    for i in range(length):
        row = {k: (history[k][i] if i < len(history[k]) else None) for k in keys}
        rows.append(row)

    csv_path = results_dir / "history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --- Main runner ----------------------------------------------------------


def run_training(args: argparse.Namespace) -> int:
    cfg_dict = load_config(args.config) if args.config else load_config()
    cfg = Config(cfg_dict)

    exp = setup_experiment(cfg, args.exp_name, seed=args.seed)
    print(f"Experiment root: {exp['root']}")

    # Instantiate user components
    trainer_cls = import_from_string(args.trainer)
    dataloader_cls = import_from_string(args.dataloader)

    trainer = trainer_cls()
    dataloader = dataloader_cls()

    # Load data
    data_dir = cfg.get_data_dir("processed_dir") or cfg.get_data_dir("raw_dir")
    train_data = dataloader.load(data_dir)
    val_data = None
    try:
        train_data, val_data, _ = dataloader.split(train_data)
    except Exception:
        # dataloader may not implement split
        pass

    # Create model and train
    model_cfg = cfg.get_model_config()
    model = trainer.create_model(model_cfg)

    training_params = cfg.get_training_params()

    # Resolve values: CLI args take precedence, otherwise use config defaults
    epochs = int(args.epochs) if getattr(args, "epochs", None) is not None else int(training_params.get("epochs", 10))
    batch_size = int(args.batch_size) if getattr(args, "batch_size", None) is not None else int(training_params.get("batch_size", 32))
    learning_rate = float(args.lr) if getattr(args, "lr", None) is not None else float(training_params.get("learning_rate", 1e-3))
    save_frequency = int(args.save_every) if getattr(args, "save_every", None) is not None else int(cfg._cfg.get("nn_training", {}).get("checkpoints", {}).get("save_frequency", 10))
    device = args.device if getattr(args, "device", None) else cfg._cfg.get("nn_training", {}).get("training", {}).get("device", "cpu")
    resume = getattr(args, "resume", None)
    seed = args.seed if getattr(args, "seed", None) is not None else cfg._cfg.get("nn_training", {}).get("training", {}).get("seed", None)

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
        resume=resume,
        seed=seed,
    )

    if history is None:
        history = {}

    write_history_json(exp["results_dir"], history)
    write_history_csv(exp["results_dir"], history)

    print("Training finished. Artifacts stored under:", exp["root"])
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal training runner")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--trainer", required=True, help="Dotted path to Trainer class (pkg.module.Class)")
    p.add_argument("--dataloader", required=True, help="Dotted path to DataLoader class (pkg.module.Class)")
    p.add_argument("--exp-name", default=None, help="Short experiment name")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (falls back to config)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size (falls back to config)")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate (falls back to config)")
    p.add_argument("--save-every", type=int, default=None, help="Save checkpoints every N epochs (falls back to config)")
    p.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu or cuda:0)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_training(args))
