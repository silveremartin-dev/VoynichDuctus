"""
Adaptive binarization and parchment background normalization for historical manuscripts.
All spatial windows and speckle filters are calibrated strictly in physical millimeters (mm / mm^2)
via MetricScaleCalibration (zero hardcoded pixel thresholds).
"""

from typing import Optional, Union
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_niblack, threshold_otsu, threshold_sauvola

from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration


class Binarizer:
    """
    Adaptive thresholding algorithms optimized for parchment degradation,
    bleed-through, uneven illumination, and iron-gall ink fading.
    """

    def __init__(
        self,
        method: str = "sauvola",
        window_mm: float = 4.1,
        k: float = 0.2,
        r: float = 128.0,
        scale_calib: Optional[MetricScaleCalibration] = None,
    ):
        self.method = method
        self.window_mm = float(window_mm)
        self.k = float(k)
        self.r = float(r)
        self.scale_calib = scale_calib

    def _window_for_image(self, shape: tuple) -> int:
        calib = self.scale_calib
        if calib is None:
            default_c = MetricScaleCalibration(manuscript_type="voynich")
            if shape[0] >= default_c.mm_to_px(75.0):
                calib = MetricScaleCalibration(
                    manuscript_type="voynich", image_height_px=int(shape[0]), image_width_px=int(shape[1])
                )
            else:
                calib = default_c
        w = max(3, int(round(calib.mm_to_px(self.window_mm))))
        return w if w % 2 == 1 else w + 1

    def to_grayscale(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Converts input image (PIL or numpy) into normalized float grayscale [0.0, 1.0]."""
        arr = np.array(image) if isinstance(image, Image.Image) else image
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return rgb2gray(arr)
        return arr.astype(np.float64) / 255.0 if arr.dtype == np.uint8 else arr

    def wolf_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Wolf-Jolion adaptive thresholding using metric-calibrated window."""
        win = self._window_for_image(gray.shape)
        mean = gaussian_filter(gray, sigma=win / 6.0)
        mean_sq = gaussian_filter(gray ** 2, sigma=win / 6.0)
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
        min_val = np.min(gray)
        max_std = float(np.max(std)) or 1.0
        return mean + self.k * (std / max_std - 1.0) * (mean - min_val)

    def binarize(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Executes metric-calibrated binarization (`True` = Ink, `False` = Parchment)."""
        gray = self.to_grayscale(image)
        win = self._window_for_image(gray.shape)

        if self.method == "sauvola":
            return gray < threshold_sauvola(gray, window_size=win, k=self.k, r=self.r / 255.0)
        if self.method == "niblack":
            return gray < threshold_niblack(gray, window_size=win, k=self.k)
        if self.method == "wolf":
            return gray < self.wolf_threshold(gray)
        if self.method == "otsu":
            return gray < threshold_otsu(gray)
        raise ValueError(f"Unknown binarization method: {self.method}")

    def binarize_manuscript(self, image: Union[np.ndarray, Image.Image], contrast_stretch: bool = True) -> np.ndarray:
        """Adaptive contrast-normalized binarization for historical manuscript vellum/parchment."""
        gray = self.to_grayscale(image)
        if gray.size == 0:
            return np.zeros((0, 0), dtype=bool)
        if contrast_stretch:
            p2, p98 = np.percentile(gray, (2, 98))
            if p98 > p2 + 1e-4:
                gray = np.clip((gray - p2) / (p98 - p2), 0.0, 1.0)
        try:
            win = self._window_for_image(gray.shape)
            t_otsu = threshold_otsu(gray)
            t_sauv = threshold_sauvola(gray, window_size=win, k=min(self.k, 0.08), r=0.5)
            return gray < np.maximum(t_otsu * 0.95, t_sauv)
        except Exception:
            return gray < 0.5

    def remove_small_artifacts(self, binary: np.ndarray, min_area_mm2: float = 0.06) -> np.ndarray:
        """Removes microscopic parchment grain speckles smaller than `min_area_mm2`."""
        from skimage.morphology import remove_small_objects
        calib = self.scale_calib or MetricScaleCalibration(manuscript_type="voynich")
        min_sz = max(1, int(round(calib.area_mm2_to_px2(min_area_mm2))))
        try:
            return remove_small_objects(binary, max_size=min_sz)
        except TypeError:
            return remove_small_objects(binary, min_size=min_sz)
