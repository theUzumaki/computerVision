"""CV Enhancement API - Interface for preprocessing and augmentation."""

import numpy as np


class CVEnhancementAPI:
    """
    Interface for CV enhancement operations.

    Config mapping (keys in `configs/default_config.yaml`):
      - `cv_enhancement.preprocessing.resize` -> preprocess(..., resize=(W,H))
      - `cv_enhancement.preprocessing.normalize` -> preprocess(..., normalize=True)
      - `cv_enhancement.augmentation.enabled` -> augment(..., enabled=True)
      - `cv_enhancement.filters` -> apply_filter(..., filter_name)

    Implementations should accept config values via kwargs or a config dict.
    """
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply preprocessing to an image.

        Expected kwargs from config: `resize`, `normalize`, etc.
        """
        raise NotImplementedError
    
    def augment(self, image: np.ndarray, **kwargs) -> list:
        """Apply augmentation to an image.

        Expected kwargs from config: `enabled`, augmentation params.
        """
        raise NotImplementedError
    
    def apply_filter(self, image: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
        """Apply a filter to an image.

        `filter_name` should be one of entries in `cv_enhancement.filters`.
        """
        raise NotImplementedError
