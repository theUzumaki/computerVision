"""CV Enhancement API - Interface for image preprocessing and augmentation."""

import numpy as np


class CVEnhancementAPI:
    """Base interface for computer vision enhancement operations.
    
    Implement these methods to provide preprocessing, augmentation, and filtering.
    Config values are passed via kwargs or can be loaded from default_config.yaml.
    """
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply preprocessing to an image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Config options (e.g., resize=(224,224), normalize=True)
            
        Returns:
            Preprocessed image as numpy array
        """
        raise NotImplementedError
    
    def augment(self, image: np.ndarray, **kwargs) -> list:
        """Apply data augmentation to an image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Augmentation config (e.g., enabled=True, rotation=15)
            
        Returns:
            List of augmented images
        """
        raise NotImplementedError
    
    def apply_filter(self, image: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
        """Apply an image filter.
        
        Args:
            image: Input image as numpy array
            filter_name: Name of filter to apply (e.g., 'blur', 'sharpen')
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered image as numpy array
        """
        raise NotImplementedError
