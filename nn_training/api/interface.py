"""Training API - Interface for model creation and training."""


class NNTrainingAPI:
    """Base interface for neural network training.
    
    Implement these methods to provide model creation, training, and inference.
    """
    
    def create_model(self, config: dict):
        """Create model from configuration.
        
        Args:
            config: Model configuration dict (e.g., architecture, hyperparameters)
            
        Returns:
            Model object
        """
        raise NotImplementedError
    
    def train(self, model, train_data, val_data=None, **kwargs) -> dict:
        """Train model and return history.
        
        Args:
            model: Model to train
            train_data: Training dataset
            val_data: Validation dataset (optional)
            **kwargs: Training params (epochs, batch_size, learning_rate, etc.)
            
        Returns:
            History dict with metrics per epoch
        """
        raise NotImplementedError
    
    def save_checkpoint(self, model, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            path: File path for checkpoint
        """
        raise NotImplementedError
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint.
        
        Args:
            path: Checkpoint file path
            
        Returns:
            Loaded model
        """
        raise NotImplementedError
    
    def predict(self, model, data):
        """Run inference with model.
        
        Args:
            model: Trained model
            data: Input data
            
        Returns:
            Predictions
        """
        raise NotImplementedError
