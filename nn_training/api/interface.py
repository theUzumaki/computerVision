"""NN Training API - Interface for model training."""


class NNTrainingAPI:
    """
    Interface for neural network training operations.

    Config mapping (keys in `configs/default_config.yaml`):
      - `nn_training.model.name` -> create_model(config) may use `name` to select architecture
      - `nn_training.training.epochs` -> train(..., epochs=...)
      - `nn_training.training.batch_size` -> train(..., batch_size=...)
      - `nn_training.training.learning_rate` -> train(..., learning_rate=...)
      - `nn_training.checkpoints.dir` -> default path for save_checkpoint
      - `nn_training.checkpoints.save_frequency` -> used by training loop to schedule checkpoints

    Implementations should accept config via the `config` dict or kwargs.
    """
    
    def create_model(self, config: dict):
        """Create a model based on configuration.

        Example config keys: `model.name`, other model hyperparameters.
        """
        raise NotImplementedError
    
    def train(self, model, train_data, val_data=None, **kwargs) -> dict:
        """Train a model. Returns training history.

        Expected kwargs: `epochs`, `batch_size`, `learning_rate`, etc.
        """
        raise NotImplementedError
    
    def save_checkpoint(self, model, path: str) -> None:
        """Save model checkpoint. `path` may default to `nn_training.checkpoints.dir`."""
        raise NotImplementedError
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        raise NotImplementedError
    
    def predict(self, model, data):
        """Run inference with a model."""
        raise NotImplementedError
