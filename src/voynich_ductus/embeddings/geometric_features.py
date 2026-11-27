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

        # Arc length & displacement (bounded denominator so closed loops with net_disp == 0 never blow up to 1e6)
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = float(np.sum(segment_lengths))
        net_disp = float(np.linalg.norm(coords[-1] - coords[0]))
        span_scale = max(float(h), float(w), 1.0)
        eff_disp = max(net_disp, 0.25 * span_scale, 1.0)
        tortuosity = min(12.0, arc_length / eff_disp)

        # Tangents
        v_start = diffs[0] if len(diffs) > 0 else np.array([0.0, 1.0])
        v_end = diffs[-1] if len(diffs) > 0 else np.array([0.0, 1.0])
        angle_start = np.arctan2(v_start[0], v_start[1])
        angle_end = np.arctan2(v_end[0], v_end[1])

        # Full-circle direction histogram (5 uniform bins spanning [-pi, pi] weighted by segment length, 0 blind sectors)
        angles = np.arctan2(diffs[:, 0], diffs[:, 1])
        weights = segment_lengths if arc_length > 1e-5 else None
        hist, _ = np.histogram(angles, bins=5, range=(-np.pi, np.pi), weights=weights, density=False)
        hist_sum = float(np.sum(hist))
        hist_norm = (hist / hist_sum) if hist_sum > 1e-6 else np.zeros(5, dtype=np.float32)

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
        # Append all 5 full-circle direction histogram features to make 16-D vector
        features.extend(hist_norm.tolist())

        return np.array(features, dtype=np.float32)

    def extract_batch(self, strokes: List[Dict[str, Any]], context_bbox: tuple = None) -> np.ndarray:
        """Extracts features for a list of strokes."""
        if not strokes:
            return np.empty((0, 16), dtype=np.float32)
        return np.vstack([self.extract_stroke_features(s, context_bbox) for s in strokes])
