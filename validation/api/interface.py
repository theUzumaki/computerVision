"""Validation API - Interface for evaluation and visualization."""


class ValidationAPI:
    """Base interface for model validation and visualization.
    
    Implement these methods to provide metrics computation, evaluation, and plotting.
    """
    
    def compute_metrics(self, predictions, targets, metrics: list) -> dict:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            metrics: List of metric names to compute (e.g., ['accuracy', 'f1'])
            
        Returns:
            Dict mapping metric names to values
        """
        raise NotImplementedError
    
    def evaluate(self, model, test_data, **kwargs) -> dict:
        """Evaluate model on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results dict
        """
        raise NotImplementedError
    
    def plot_results(self, results: dict, plot_type: str, save_path: str = None) -> None:
        """Generate visualization from results.
        
        Args:
            results: Evaluation results dict
            plot_type: Type of plot (e.g., 'confusion_matrix', 'roc_curve')
            save_path: Path to save plot (optional)
        """
        raise NotImplementedError

    def plot_from_history(self, history, plot_types: list, save_path: str = None) -> dict:
        """Generate plots from training history.
        
        Args:
            history: History dict or path to history.json/csv
            plot_types: List of plot types (e.g., ['loss', 'accuracy', 'lr'])
            save_path: Directory to save plots (optional)
            
        Returns:
            Dict mapping plot_type to saved file path
        """
        raise NotImplementedError
