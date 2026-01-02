"""Concrete DataLoader for `van_gogh_dataset`.

This module provides a single concrete `DataLoader` class that loads the
`metadata.json`, validates images, and performs stratified splits.

The loader prefers to return a pandas.DataFrame (columns: 'file_path','label','artist')
when `pandas` is available; otherwise it returns a plain `list[dict]` and
`split()` will return lists as well.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Tuple, Optional, Sequence, List, Dict, Any

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


class DataLoader:
    """Concrete loader for the `van_gogh_dataset`.

    Usage:
        loader = DataLoader()
        data = loader.load()              # uses defaults from configs/default_config.yaml
        or
        data = loader.load('data/raw')    # provide explicit path

    Returns either a pandas.DataFrame (if pandas is installed) or a list[dict].
    """

    def __init__(self, metadata_filename: Optional[str] = None):
        # Do not hardcode a metadata filename here; prefer the project config.
        self.metadata_filename = metadata_filename

    def _config_meta(self) -> Tuple[Path, str]:
        """Read required dataset config and return (base_dir, metadata_filename).

        Raises a ValueError listing missing keys if any required config is absent.
        """
        from configs.loader import load_config

        cfg = load_config() or {}
        raw = cfg.get("data", {}).get("raw_dir")
        dataset = cfg.get("pipeline", {}).get("dataloader_config", {}).get("dataset_name")
        metadata = cfg.get("pipeline", {}).get("dataloader_config", {}).get("metadata_file")

        missing = []
        if raw is None:
            missing.append("data.raw_dir")
        if dataset is None:
            missing.append("pipeline.dataloader_config.dataset_name")
        if metadata is None and self.metadata_filename is None:
            missing.append("pipeline.dataloader_config.metadata_file (or pass metadata_filename to DataLoader)")

        if missing:
            raise ValueError("Missing configuration keys: " + ", ".join(missing))

        return Path(raw) / dataset, (metadata if metadata is not None else self.metadata_filename)  # type: ignore

    def load(self, data_path: Optional[str] = None) -> Any:
        """Load metadata and validate files (very small, config-first).

        - Call with no args to use config values (strict: errors if keys are missing).
        - Or pass the full path to the metadata json file.
        """
        if data_path is None:
            base_dir, metadata_file = self._config_meta()
            meta_path = Path(base_dir) / metadata_file
        else:
            p = Path(data_path)
            if p.is_file():
                meta_path = p.resolve()
                base_dir = p.parent
            else:
                raise ValueError("data_path must be the path to the metadata json file when provided")

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        rows: List[Dict[str, Any]] = []
        missing = 0
        for e in meta:
            img = (base_dir / e.get("file_path", "")).resolve()
            if not img.exists():
                missing += 1
                continue
            rows.append({"file_path": str(img), "label": int(e.get("label", 0)), "artist": e.get("artist", "")})

        if missing:
            print(f"⚠️  Warning: {missing} missing files skipped")

        if pd is not None:
            return pd.DataFrame(rows)
        return rows

    def split(self, data: Any, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[Any, Any, Any]:
        """Simple stratified split preserving input type (DataFrame or list)."""
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        rng = np.random.RandomState(seed)

        if pd is not None and isinstance(data, pd.DataFrame):
            labels = data["label"].to_numpy()
            indices = np.arange(len(data))
            return_type = "df"
        elif isinstance(data, list):
            labels = np.array([int(d["label"]) for d in data])
            indices = np.arange(len(data))
            return_type = "list"
        else:
            raise TypeError("Unsupported data type; expected DataFrame or list")

        train_idx, val_idx, test_idx = [], [], []
        for lbl in np.unique(labels):
            idx = indices[labels == lbl]
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(np.floor(train_ratio * n))
            n_val = int(np.floor(val_ratio * n))

            train_idx.append(idx[:n_train])
            val_idx.append(idx[n_train:n_train + n_val])
            test_idx.append(idx[n_train + n_val:])

        train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
        val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
        test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

        if return_type == "df":
            return data.loc[train_idx].reset_index(drop=True), data.loc[val_idx].reset_index(drop=True), data.loc[test_idx].reset_index(drop=True)
        return [data[i] for i in train_idx], [data[i] for i in val_idx], [data[i] for i in test_idx]


if __name__ == "__main__":
    loader = DataLoader()

    try:
        # Use the configuration-driven path by default
        data = loader.load()
        if pd is not None and isinstance(data, pd.DataFrame):
            counts = data["label"].value_counts().to_dict()
            total = len(data)
        else:
            counts = {0: sum(1 for d in data if d["label"] == 0), 1: sum(1 for d in data if d["label"] == 1)}
            total = len(data)

        print(f"Loaded {total} items. Label counts:\n{counts}")

        train, val, test = loader.split(data, seed=123)
        print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    except Exception as e:
        print(f"Error running loader smoke test: {e}")

