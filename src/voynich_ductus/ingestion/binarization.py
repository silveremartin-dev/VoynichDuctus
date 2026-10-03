"""
Adaptive binarization and parchment background normalization for historical manuscripts.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola, threshold_niblack, threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter


class Binarizer:
    """
    Adaptive thresholding algorithms optimized for parchment degradation,
    bleed-through, uneven illumination, and iron-gall ink fading.
    """

    def __init__(self, method: str = "sauvola", window_size: int = 25, k: float = 0.2, r: float = 128.0):
        """
        Args:
            method: 'sauvola', 'niblack', 'wolf', or 'otsu'
            window_size: Window size for local threshold estimation
            k: Sauvola / Niblack dynamic parameter
            r: Dynamic range of standard deviation (default 128 for 8-bit image)
        """
        self.method = method
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.k = k
        self.r = r

    def to_grayscale(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Converts input image (PIL or numpy) into normalized float grayscale [0.0, 1.0]."""
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image

        if arr.ndim == 3:
            if arr.shape[2] == 4:  # RGBA
                arr = arr[:, :, :3]
            gray = rgb2gray(arr)
        else:
            gray = arr.astype(np.float64) / 255.0 if arr.dtype == np.uint8 else arr

        return gray

    def wolf_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Wolf-Jolion adaptive thresholding for degraded historical document images.
        T = m + k * (s / R - 1) * (m - M)
        where M is minimum gray value, R is max standard deviation.
        """
        mean = gaussian_filter(gray, sigma=self.window_size / 6)
        mean_sq = gaussian_filter(gray ** 2, sigma=self.window_size / 6)
        variance = np.maximum(mean_sq - mean ** 2, 0)
        std = np.sqrt(variance)

        min_val = np.min(gray)
        max_std = np.max(std)
        if max_std == 0:
            max_std = 1.0

        threshold = mean + self.k * (std / max_std - 1.0) * (mean - min_val)
        return threshold

    def binarize(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Executes binarization.
        Returns:
            Binary boolean numpy array where True represents Ink and False represents Parchement/Background.
        """
        gray = self.to_grayscale(image)

        if self.method == "sauvola":
            thresh = threshold_sauvola(gray, window_size=self.window_size, k=self.k, r=self.r / 255.0)
            binary = gray < thresh
        elif self.method == "niblack":
            thresh = threshold_niblack(gray, window_size=self.window_size, k=self.k)
            binary = gray < thresh
        elif self.method == "wolf":
            thresh = self.wolf_threshold(gray)
            binary = gray < thresh
        elif self.method == "otsu":
            thresh = threshold_otsu(gray)
            binary = gray < thresh
        else:
            raise ValueError(f"Unknown binarization method: {self.method}")

        return binary

    def remove_small_artifacts(self, binary: np.ndarray, min_size: int = 5) -> np.ndarray:
        """Removes tiny speckles and noise from parchment grain."""
        from skimage.morphology import remove_small_objects
        try:
            return remove_small_objects(binary, max_size=min_size)
        except TypeError:
            return remove_small_objects(binary, min_size=min_size)
