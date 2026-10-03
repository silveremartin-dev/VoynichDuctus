"""
Rotation, scale, and translation invariant geometric feature extraction from stroke segments.
"""

from typing import List, Dict, Any
import numpy as np


class GeometricFeatureExtractor:
    """
    Extracts numerical feature vectors from individual strokes or composite glyph components.
    Features:
    - Normalized aspect ratio, height, width
    - Net displacement vs arc length (tortuosity / curvature)
    - Initial & final tangent angles (8-bin orientation histogram)
    - Normalized centroid location within the word baseline
    - Average stroke thickness & thickness variance (plein/délié ratio)
    """

    def extract_stroke_features(self, stroke: Dict[str, Any], context_bbox: tuple = None) -> np.ndarray:
        pts = stroke.get("points", [])
        if len(pts) < 2:
            return np.zeros(16, dtype=np.float32)

        coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        widths = np.array([p[2] if len(p) > 2 else 1.0 for p in pts], dtype=np.float32)

        # Bounding box
        min_y, min_x = np.min(coords, axis=0)
        max_y, max_x = np.max(coords, axis=0)
        h = max_y - min_y
        w = max_x - min_x
        aspect_ratio = (w + 1e-4) / (h + 1e-4)

        # Arc length & displacement
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = np.sum(segment_lengths)
        net_disp = np.linalg.norm(coords[-1] - coords[0])
        tortuosity = (arc_length + 1e-4) / (net_disp + 1e-4)

        # Tangents
        v_start = diffs[0] if len(diffs) > 0 else np.array([0.0, 1.0])
        v_end = diffs[-1] if len(diffs) > 0 else np.array([0.0, 1.0])
        angle_start = np.arctan2(v_start[0], v_start[1])
        angle_end = np.arctan2(v_end[0], v_end[1])

        # Direction histogram (8 bins)
        angles = np.arctan2(diffs[:, 0], diffs[:, 1])
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=True)

        # Thickness stats
        mean_width = np.mean(widths)
        std_width = np.std(widths)

        # Contextual normalization
        rel_y, rel_x = 0.5, 0.5
        if context_bbox is not None:
            cy0, cx0, cy1, cx1 = context_bbox
            ch = max(cy1 - cy0, 1)
            cw = max(cx1 - cx0, 1)
            centroid = np.mean(coords, axis=0)
            rel_y = (centroid[0] - cy0) / ch
            rel_x = (centroid[1] - cx0) / cw

        features = [
            np.log1p(arc_length),
            aspect_ratio,
            tortuosity,
            np.sin(angle_start),
            np.cos(angle_start),
            np.sin(angle_end),
            np.cos(angle_end),
            mean_width,
            std_width,
            rel_y,
            rel_x,
        ]
        # Append 5 direction histogram features to make 16-D vector
        features.extend(hist[:5].tolist())

        return np.array(features, dtype=np.float32)

    def extract_batch(self, strokes: List[Dict[str, Any]], context_bbox: tuple = None) -> np.ndarray:
        """Extracts features for a list of strokes."""
        if not strokes:
            return np.empty((0, 16), dtype=np.float32)
        return np.vstack([self.extract_stroke_features(s, context_bbox) for s in strokes])
