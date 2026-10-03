"""
Color normalization, illumination flattening, and chromatic pigment separation.
Filters out colored illustrations (green foliage, blue baths, ochre roots) to isolate pure iron-gall text ink.
"""

from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2gray
from scipy.ndimage import gaussian_filter, zoom


class ColorIlluminationNormalizer:
    """
    Normalizes historical parchment scans:
    1. Fast multi-scale flat-field illumination correction (removes parchment shading & vignetting).
    2. Adaptive local contrast stretching for crisp stroke definition.
    3. Chromatic pigment filtering (discriminates monochrome text ink from colored illustrations).
    """

    def __init__(self, bg_sigma: float = 20.0, saturation_thresh: float = 0.18, max_ink_brightness: float = 0.68):
        self.bg_sigma = bg_sigma
        self.saturation_thresh = saturation_thresh
        self.max_ink_brightness = max_ink_brightness

    def to_rgb_array(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Ensures float32 RGB array normalized in [0.0, 1.0]."""
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
        return arr

    def flatten_illumination_fast(self, rgb: np.ndarray, downscale: int = 8) -> np.ndarray:
        """
        Fast illumination estimation via pyramid downsampling and Gaussian filtering.
        """
        H, W, C = rgb.shape
        # Adapt downscale factor to avoid over-shrinking small test images
        actual_downscale = max(1, min(downscale, H // 16, W // 16))
        
        # Subsample for speed
        subsampled = rgb[::actual_downscale, ::actual_downscale, :]
        sigma = max(2.0, min(self.bg_sigma, subsampled.shape[0] / 4.0, subsampled.shape[1] / 4.0))
        bg_small = gaussian_filter(subsampled, sigma=(sigma, sigma, 0))
        bg_small = np.clip(bg_small, 0.1, 1.0)

        # Upscale background model to full resolution
        zoom_factors = (H / bg_small.shape[0], W / bg_small.shape[1], 1.0)
        bg_full = zoom(bg_small, zoom_factors, order=1)

        # Flat field division
        flat = np.clip(rgb / bg_full, 0.0, 1.0)
        return flat

    def extract_ink_mask_chromatic(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates text ink from colored illustrations using chromaticity analysis:
        - Text ink: Low chroma (neutral gray, dark brown sepia, carbon black), high contrast.
        - Illustrations: High chroma (green chlorophyll leaves, ochre roots, blue baths, red dyes).
        
        Returns:
            ink_mask: Boolean array (True where text ink is present, False for parchment and colored paint)
            cleaned_grayscale: Illumination-normalized float grayscale image
        """
        rgb = self.to_rgb_array(image)

        # 1. Fast illumination flattening
        flat_rgb = self.flatten_illumination_fast(rgb)

        # 2. Chroma calculation in RGB: max(R,G,B) - min(R,G,B)
        # Text ink is achromatic (low chroma < 0.16)
        # Colored pigments have vivid chroma (> 0.20)
        chroma = np.max(flat_rgb, axis=2) - np.min(flat_rgb, axis=2)

        # Also compute HSV for selective color hue discrimination
        hsv = rgb2hsv(flat_rgb)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 3. Detect colored illustration pigments
        # Green plant foliage: hue in [0.15, 0.48] and chroma > 0.15
        # Blue baths/stars: hue in [0.50, 0.75] and chroma > 0.15
        # High saturation / vivid paints: chroma > 0.22 or (sat > 0.35 and val > 0.3)
        is_green_foliage = (hue >= 0.15) & (hue <= 0.48) & (chroma > 0.12)
        is_blue_water = (hue >= 0.50) & (hue <= 0.75) & (chroma > 0.12)
        is_colored_pigment = is_green_foliage | is_blue_water | (chroma > 0.22) | ((sat > 0.35) & (val > 0.35))

        # 4. Text ink is dark and neutral
        gray = rgb2gray(flat_rgb)
        
        # Text ink must be darker than threshold and NOT part of colored illustrations
        is_dark_ink = (gray < self.max_ink_brightness) & (val < 0.80)
        pure_ink_mask = is_dark_ink & (~is_colored_pigment)

        return pure_ink_mask, gray

