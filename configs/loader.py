"""Simple config loader and accessor for the project.

Usage:
    from configs.loader import load_config, Config

    cfg_dict = load_config()  # loads configs/default_config.yaml by default
    cfg = Config(cfg_dict)
    print(cfg.get_training_params())

This module keeps dependencies minimal; it requires PyYAML. If PyYAML is not
installed it raises an ImportError with an install suggestion.
"""
from pathlib import Path
from typing import Any, Dict, Optional


default_config_path = Path(__file__).parent / "default_config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config file and return it as a dictionary.

    Args:
        path: Path to a YAML config file. If None, uses the package default.
    Returns:
        Dictionary with configuration.
    """
    try:
        import yaml
    except Exception as e:  # pragma: no cover - informative error
        raise ImportError(
            "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
        ) from e

    config_path = Path(path) if path is not None else default_config_path
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


class Config:
    """Convenience wrapper around a raw config dict.

    Provides small helper accessors for common keys used by each layer.
    Keep this thin and feel free to extend with typed dataclasses later.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg or {}

    # Data
    def get_data_dir(self, key: str = "raw_dir") -> Optional[str]:
        return self._cfg.get("data", {}).get(key)

    # CV enhancement
    def get_cv_preprocessing(self) -> Dict[str, Any]:
        return self._cfg.get("cv_enhancement", {}).get("preprocessing", {})

    def get_cv_augmentation_enabled(self) -> bool:
        return bool(self._cfg.get("cv_enhancement", {}).get("augmentation", {}).get("enabled", False))

    # Training
    def get_training_params(self) -> Dict[str, Any]:
        return self._cfg.get("nn_training", {}).get("training", {})

    def get_model_config(self) -> Dict[str, Any]:
        return self._cfg.get("nn_training", {}).get("model", {})

    def get_checkpoint_dir(self) -> Optional[str]:
        return self._cfg.get("nn_training", {}).get("checkpoints", {}).get("dir")

    # Validation
    def get_validation_metrics(self) -> list:
        return self._cfg.get("validation", {}).get("metrics", [])

    def get_visualization_output_dir(self) -> Optional[str]:
        return self._cfg.get("validation", {}).get("visualization", {}).get("output_dir")

    # Experiments
    def get_experiments_dir(self, key: str = "results_dir") -> Optional[str]:
        return self._cfg.get("experiments", {}).get(key)

    # Generic
    def as_dict(self) -> Dict[str, Any]:
        return self._cfg


if __name__ == "__main__":  # quick usage demo
    cfg_dict = load_config()
    cfg = Config(cfg_dict)
    print("Training params:", cfg.get_training_params())
    print("CV preprocessing:", cfg.get_cv_preprocessing())
