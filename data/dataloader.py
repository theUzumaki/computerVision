"""DataLoader interface for loading and splitting datasets."""


class DataLoader:
    """Base interface for data loading.
    
    Implement these methods in your concrete DataLoader class.
    """
    
    def load(self, data_path: str):
        """Load dataset from path.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Dataset object (any format your trainer expects)
        """
        raise NotImplementedError
    
    def split(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split data into train, validation, and test sets.
        
        Args:
            data: Dataset to split
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            test_ratio: Fraction for testing (default 0.1)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        raise NotImplementedError
