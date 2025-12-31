"""Data Loader - Interface for data loading."""


class DataLoader:
    """Interface for data loading and management."""
    
    def load(self, data_path: str):
        """Load data from a path."""
        raise NotImplementedError
    
    def split(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split data into train, validation, and test sets."""
        raise NotImplementedError
