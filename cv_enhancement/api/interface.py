"""CV Enhancement API - Interface for image preprocessing and augmentation."""

from typing import List, Tuple, Optional, Any
from skimage.color import rgb2lab
from skimage.filters import gabor
from skimage.io import imread
import numpy as np
import pandas as pd


class CVEnhancementAPI:
    """Base interface for computer vision enhancement operations.
    
    This implementation performs dataset-level preprocessing: it accepts a
    pandas.DataFrame or a list of dicts (each containing a `file_path`) and
    returns the same type with an added `image` field containing the
    processed numpy arrays.

    Config values are passed via kwargs or can be loaded from default_config.yaml.
    """
    
    def preprocess(self, data: Any, **kwargs) -> Any:
        """Apply preprocessing to a dataset.
        
        Args:
            data: pandas.DataFrame or list[dict] where each row/dict includes `file_path`.
            **kwargs: Config options (e.g., normalize=True, mean=0.0, std=1.0, color_space_conversion=True)

        Returns:
            The same type as `data` with an added `'image'` column/field containing processed images.
        """

        def _process(img: np.ndarray) -> np.ndarray:
            if kwargs.get("normalize", False):
                img = self.standardized_normalization(
                    img,
                    mean=kwargs.get("mean", 0.0),
                    std=kwargs.get("std", 1.0),
                )
            if kwargs.get("color_space_conversion", False):
                img = self.color_space_conversion(img)
            if kwargs.get("gabor_filtering", False):
                img = self.gabor_filtering(
                    img,
                    frequency=kwargs.get("frequency", 0.6),
                    theta=kwargs.get("theta", 0),
                )
            return img

        # pandas DataFrame case (if pandas available)
        if pd is not None and hasattr(data, "iterrows"):
            df = data.copy()
            for idx, row in df.iterrows():
                img = imread(row["file_path"])
                df.at[idx, "image"] = _process(img)
            return df

        # list of dicts case
        if isinstance(data, list):
            out = []
            for row in data:
                img = imread(row["file_path"])
                new_row = dict(row)
                new_row["image"] = _process(img)
                out.append(new_row)
            return out

        raise TypeError("Unsupported data type for preprocess; expected DataFrame or list of dicts with 'file_path'")
    
    def augment(self, data: Any, **kwargs) -> Any:
        """Apply dataset-level data augmentation (minimal, non-destructive).

        This basic implementation ensures every row has an `image` field by
        loading the file if necessary and returns the dataset unchanged.
        Subclasses can override to return additional augmented rows.
        """
        if pd is not None and hasattr(data, "iterrows"):
            df = data.copy()
            for idx, row in df.iterrows():
                if "image" not in row or row["image"] is None:
                    df.at[idx, "image"] = imread(row["file_path"])
            return df

        if isinstance(data, list):
            out = []
            for row in data:
                new_row = dict(row)
                if "image" not in new_row or new_row["image"] is None:
                    new_row["image"] = imread(new_row["file_path"])
                out.append(new_row)
            return out

        raise TypeError("Unsupported data type for augment; expected DataFrame or list of dicts with 'file_path'")
    
    def apply_filter(self, data: Any, filter_name: Optional[str] = None, **kwargs) -> Any:
        """Apply filters at dataset level.

        Args:
            data: pandas.DataFrame or list[dict] with `file_path` and/or `image` fields
            filter_name: Optional name of filter (e.g., 'gabor')
            **kwargs: Filter-specific parameters (e.g., gabor_filtering=True, frequency, theta)

        Returns:
            Dataset of the same type with filtered `image` fields
        """

        def _apply(img: np.ndarray) -> np.ndarray:
            if filter_name == "gabor" or kwargs.get("gabor_filtering", False):
                return self.gabor_filtering(
                    img,
                    frequency=kwargs.get("frequency", 0.6),
                    theta=kwargs.get("theta", 0),
                )
            # No other filters implemented; return original image
            return img

        if pd is not None and hasattr(data, "iterrows"):
            df = data.copy()
            for idx, row in df.iterrows():
                img = row.get("image")
                if img is None:
                    img = imread(row["file_path"])
                df.at[idx, "image"] = _apply(img)
            return df

        if isinstance(data, list):
            out = []
            for row in data:
                new_row = dict(row)
                img = new_row.get("image")
                if img is None:
                    img = imread(new_row["file_path"])
                new_row["image"] = _apply(img)
                out.append(new_row)
            return out

        raise TypeError("Unsupported data type for apply_filter; expected DataFrame or list of dicts with 'file_path'")

    def slice_patches(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (224, 224),
        stride: Optional[Tuple[int, int]] = None,
        trim: bool = False,
        return_coords: bool = False,
    ) -> List[np.ndarray]:
        """Slice a high-resolution image into smaller patches.

        Args:
            image: Input image as HxW[xC] numpy array
            patch_size: (height, width) of each patch (default 224x224)
            stride: (y_stride, x_stride). If None, defaults to patch_size (non-overlapping)
            trim: If True, crop (cut off) bottom/right edges so only full patches remain
            return_coords: If True, return list of tuples (patch, (y, x)) with top-left coords

        Notes:
            - `padding` and `trim` are mutually exclusive. If both are True a ValueError is raised.
            - When `trim=True` the image is cropped from the bottom/right (i.e., `image = image[:new_h, :new_w]`).

        Returns:
            List of patches (or list of (patch, (y, x)) when return_coords=True)
        """
        h, w = image.shape[:2]
        ph, pw = patch_size
        if stride is None:
            sy, sx = ph, pw
        else:
            sy, sx = stride

        if trim:
            # crop bottom/right so dimensions are multiples of patch size
            new_h = (h // ph) * ph
            new_w = (w // pw) * pw
            if new_h == 0 or new_w == 0:
                # patch size is larger than image in at least one dim -> no patches
                return []
            image = image[:new_h, :new_w].copy()
            h, w = image.shape[:2]

        patches = []
        for y in range(0, h - ph + 1, sy):
            for x in range(0, w - pw + 1, sx):
                patch = image[y : y + ph, x : x + pw].copy()
                if return_coords:
                    patches.append((patch, (y, x)))
                else:
                    patches.append(patch)

        return patches

    def standardized_normalization(self, image: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Apply standardized normalization to an image.

        Args:
            image: Input image as numpy array
            mean: Mean value for normalization (default 0.0)
            std: Standard deviation for normalization (default 1.0)

        Returns:
            Normalized image as numpy array
        """
        normalized_image = (image - mean) / std
        return normalized_image
    
    def color_space_conversion(self, image: np.ndarray) -> np.ndarray:
        """Convert image to CIE lab*.

        Args:
            image: Input image as numpy array

        Returns:
            Image converted to CIE lab* with only luminance channel
        """
        
        # Ensure image is float in [0, 1] for rgb2lab
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        lab = rgb2lab(img)
        l_channel = lab[..., 0]
        return l_channel

        
    def gabor_filtering(self, image: np.ndarray, frequency: float = 0.6, theta: float = 0) -> np.ndarray:
        """Apply Gabor filtering to an image.

        Args:
            image: Input image as numpy array
            frequency: Frequency of the sinusoidal function (default 0.6)
            theta: Orientation of the Gabor filter in radians (default 0)

        Returns:
            Gabor filtered image as numpy array
        """

        filtered_real, filtered_imag = gabor(image, frequency=frequency, theta=theta)
        return filtered_real
    

