"""
Topological skeletonization and distance-transform-based stroke width estimation.
"""

from typing import Tuple, Dict, Any
import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin
from scipy.ndimage import distance_transform_edt


class Skeletonizer:
    """
    Computes 1D medial skeletons from 2D ink shapes and samples the local stroke thickness (plein/délié).
    """

    def __init__(self, method: str = "medial_axis"):
        """
        Args:
            method: 'medial_axis', 'skeletonize', or 'thin' (Zhang-Suen)
        """
        self.method = method

    def extract_skeleton(self, binary_ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 1-pixel wide skeleton and local stroke radius.
        
        Args:
            binary_ink: 2D boolean array (True = ink)
            
        Returns:
            skeleton: 2D boolean array of skeleton pixels
            stroke_width: 2D float array containing estimated stroke diameter (2 * radius) at each skeleton pixel
        """
        if not np.any(binary_ink):
            return np.zeros_like(binary_ink, dtype=bool), np.zeros_like(binary_ink, dtype=float)

        # Distance transform provides distance to nearest background pixel (radius)
        dist_map = distance_transform_edt(binary_ink)

        if self.method == "medial_axis":
            skel, dist = medial_axis(binary_ink, return_distance=True)
            stroke_diameter = dist * 2.0
        elif self.method == "skeletonize":
            skel = skeletonize(binary_ink)
            stroke_diameter = dist_map * 2.0 * skel
        elif self.method == "thin":
            skel = thin(binary_ink)
            stroke_diameter = dist_map * 2.0 * skel
        else:
            raise ValueError(f"Unknown skeletonization method: {self.method}")

        return skel, stroke_diameter

    def get_skeleton_coordinates(self, skeleton: np.ndarray, stroke_diameter: np.ndarray) -> np.ndarray:
        """
        Returns an array of (y, x, width) for all active skeleton pixels.
        """
        ys, xs = np.where(skeleton)
        widths = stroke_diameter[ys, xs]
        return np.column_stack([ys, xs, widths])
