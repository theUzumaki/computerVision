"""Pipeline runner - orchestrates data preparation, training, and validation.

Usage:
    python scripts/run_pipeline.py --exp-name quicktest

Configure component paths in configs/default_config.yaml under 'pipeline' section,
or override via CLI arguments.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.loader import load_config, Config
from nn_training.train import setup_experiment


def import_from_string(path: str):
    """Import class from dotted path."""
    if not path:
        raise ValueError("Empty import path")
    if ":" in path:
        module_path, symbol = path.split(":", 1)
    else:
        parts = path.split(".")
        module_path = ".".join(parts[:-1])
        symbol = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, symbol)

def run_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline: prepare -> train -> evaluate."""
    cfg = Config(load_config(args.config) if args.config else load_config())
    pipeline_cfg = cfg._cfg.get("pipeline", {})

    # Resolve component paths (CLI > config)
    dataloader_path = args.dataloader or pipeline_cfg.get("dataloader")
    cv_enhancer_path = args.cv_enhancer or pipeline_cfg.get("cv_enhancer")
    trainer_path = args.trainer or pipeline_cfg.get("trainer")
    validator_path = args.validator or pipeline_cfg.get("validator")

    steps = args.steps.split(",") if args.steps else pipeline_cfg.get("steps", ["prepare", "train", "evaluate"])

    # Setup experiment
    exp = setup_experiment(cfg, args.exp_name, seed=args.seed)
    print(f"Experiment: {exp['root']}")

    # Instantiate components
    dataloader = import_from_string(dataloader_path)() if dataloader_path else None
    cv_enhancer = import_from_string(cv_enhancer_path)() if cv_enhancer_path else None
    trainer = import_from_string(trainer_path)() if trainer_path else None
    validator = import_from_string(validator_path)() if validator_path else None

    # Prepare step
    data_obj = None
    if "prepare" in steps:
        if not dataloader:
            raise RuntimeError("No dataloader for 'prepare' step")
        
        print("Loading data...")
        data_obj = dataloader.load(cfg.get_data_dir("raw_dir"))

        if cv_enhancer:
            prep_cfg = cfg.get_cv_preprocessing()
            if prep_cfg:
                print("Preprocessing...")
                data_obj = cv_enhancer.preprocess(data_obj, **prep_cfg)

            if cfg.get_cv_augmentation_enabled():
                print("Augmenting...")
                data_obj = cv_enhancer.augment(data_obj)
        
        print("Prepare complete.")

    # Train step
    if "train" in steps:
        if not trainer:
            raise RuntimeError("No trainer for 'train' step")

        # Resolve training params (CLI > config)
        training_params = cfg.get_training_params()
        epochs = args.epochs if args.epochs is not None else int(training_params.get("epochs", 10))
        batch_size = args.batch_size if args.batch_size is not None else int(training_params.get("batch_size", 32))
        lr = args.lr if args.lr is not None else float(training_params.get("learning_rate", 1e-3))
        save_freq = args.save_every if args.save_every is not None else int(cfg._cfg.get("nn_training", {}).get("checkpoints", {}).get("save_frequency", 10))
        device = args.device or cfg._cfg.get("nn_training", {}).get("training", {}).get("device", "cpu")
        seed = args.seed if args.seed is not None else cfg._cfg.get("nn_training", {}).get("training", {}).get("seed")

        model = trainer.create_model(cfg.get_model_config())
        
        train_data, val_data = None, None
        if data_obj is not None:
            try:
                train_data, val_data, _ = dataloader.split(data_obj)
            except Exception:
                train_data = data_obj

        print(f"Training for {epochs} epochs...")
        history = trainer.train(
            model,
            train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            checkpoint_dir=str(exp["checkpoints_dir"]),
            save_frequency=save_freq,
            device=device,
            resume=args.resume,
            seed=seed,
        )
        print("Training complete.")

    # Evaluate step
    if "evaluate" in steps and validator:
        print("Evaluating...")
        try:
            eval_results = validator.evaluate(model, data_obj)
            (exp["results_dir"] / "final_evaluation.json").write_text(str(eval_results))
            
            try:
                validator.plot_results(eval_results, plot_type="final", save_path=str(exp["results_dir"]))
            except Exception:
                pass
            
            print("Evaluation complete.")
        except Exception as e:
            print(f"Evaluation failed: {e}")

    print(f"Pipeline complete. Artifacts: {exp['root']}")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline runner")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataloader", type=str, default=None)
    p.add_argument("--cv-enhancer", dest="cv_enhancer", type=str, default=None)
    p.add_argument("--trainer", type=str, default=None)
    p.add_argument("--validator", type=str, default=None)
    p.add_argument("--steps", type=str, default=None, help="Comma-separated: prepare,train,evaluate")
    p.add_argument("--exp-name", default=None)
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
    sys.exit(run_pipeline(args))
