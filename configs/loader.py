"""Config loader for the project.

Usage:
    from configs.loader import load_config, Config
    
    cfg = Config(load_config())
    print(cfg.get_training_params())
"""
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config file.
    
    Args:
        path: Path to config file. Defaults to default_config.yaml.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Install PyYAML: pip install pyyaml") from e
    
    if path is None:
        path = Path(__file__).parent / "default_config.yaml"
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Config accessor with helper methods."""

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg or {}

    def get_data_dir(self, key: str = "raw_dir") -> Optional[str]:
        return self._cfg.get("data", {}).get(key)

    def get_cv_preprocessing(self) -> Dict[str, Any]:
        return self._cfg.get("cv_enhancement", {}).get("preprocessing", {})

    def get_cv_augmentation_enabled(self) -> bool:
        return self._cfg.get("cv_enhancement", {}).get("augmentation", {}).get("enabled", False)

    def get_training_params(self) -> Dict[str, Any]:
        return self._cfg.get("nn_training", {}).get("training", {})

    def get_model_config(self) -> Dict[str, Any]:
        return self._cfg.get("nn_training", {}).get("model", {})

    def get_checkpoint_dir(self) -> Optional[str]:
        return self._cfg.get("nn_training", {}).get("checkpoints", {}).get("dir")

    def get_validation_metrics(self) -> list:
        return self._cfg.get("validation", {}).get("metrics", [])

    def get_visualization_output_dir(self) -> Optional[str]:
        return self._cfg.get("validation", {}).get("visualization", {}).get("output_dir")

    def get_experiments_dir(self, key: str = "results_dir") -> Optional[str]:
        return self._cfg.get("experiments", {}).get(key)

    def as_dict(self) -> Dict[str, Any]:
        return self._cfg

if __name__ == "__main__":
    cfg = Config(load_config())
    print("Training params:", cfg.get_training_params())
    print("CV preprocessing:", cfg.get_cv_preprocessing())
