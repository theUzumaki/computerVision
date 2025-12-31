"""Validation API - Interface for evaluation and visualization."""


class ValidationAPI:
    """
    Interface for validation and visualization operations.

    Config mapping (keys in `configs/default_config.yaml`):
      - `validation.metrics` -> compute_metrics(..., metrics=[...])
      - `validation.visualization.save_plots` -> plot_results(..., save when True)
      - `validation.visualization.output_dir` -> default save_path for plot_results
      - `validation.reports.output_dir` -> used by reporting utilities (external or implemented here)

    Implementations should accept config via kwargs or a config dict.
    """
    
    def compute_metrics(self, predictions, targets, metrics: list) -> dict:
        """Compute specified metrics. Returns dict of metric_name -> value."""
        raise NotImplementedError
    
    def evaluate(self, model, test_data, **kwargs) -> dict:
        """Evaluate a model on test data."""
        raise NotImplementedError
    
    def plot_results(self, results: dict, plot_type: str, save_path: str = None) -> None:
        """Generate visualization plots."""
        raise NotImplementedError
