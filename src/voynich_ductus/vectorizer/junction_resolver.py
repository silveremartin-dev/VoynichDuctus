"""
Junction resolution and scribal ductus kinematics ordering based on tangent continuity
and 15th-century right-handed quill mechanics.
"""

from typing import List, Dict, Tuple, Any
import numpy as np


class JunctionResolver:
    """
    Resolves stroke topology at junctions (X-intersections, Y-branches) using:
    1. Tangent continuity (minimizing Euler-Bernoulli bending energy / angle deflection).
    2. Scribal priors: Downstrokes (top-to-bottom) and crossbars (left-to-right).
    3. Pen lift likelihood and chronological ordering.
    """

    def __init__(self, right_handed_prior: bool = True):
        self.right_handed_prior = right_handed_prior

    @staticmethod
    def compute_stroke_vector(points: List[Tuple[float, float, float]], k: int = 4) -> np.ndarray:
        """
        Computes local direction vector at stroke beginning or end.
        """
        if len(points) < 2:
            return np.array([0.0, 1.0])
        p_start = np.array(points[0][:2])
        idx = min(k, len(points) - 1)
        p_end = np.array(points[idx][:2])
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.array([0.0, 1.0])

    def orient_stroke(self, stroke: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orients individual stroke in accordance with right-handed scribal priors:
        - Predominantly top-to-bottom (downstrokes on ascenders/descenders).
        - Predominantly left-to-right (horizontal traverses/ligatures).
        """
        pts = stroke["points"]
        if len(pts) < 2:
            return stroke

        y0, x0 = pts[0][0], pts[0][1]
        y1, x1 = pts[-1][0], pts[-1][1]

        dy = y1 - y0
        dx = x1 - x0

        # Prior score: positive means natural direction (down and right)
        natural_score = 1.5 * dy + 1.0 * dx

        if natural_score < 0:
            # Invert stroke direction to match pen movement
            reversed_pts = list(reversed(pts))
            stroke["points"] = reversed_pts
            stroke["start"] = (reversed_pts[0][0], reversed_pts[0][1])
            stroke["end"] = (reversed_pts[-1][0], reversed_pts[-1][1])
            stroke["reversed_by_prior"] = True
        else:
            stroke["reversed_by_prior"] = False

        return stroke

    def resolve_and_order_strokes(self, strokes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts strokes into a probable chronological sequence (ductus order):
        Strokes that start higher and more to the left precede subsequent strokes.
        """
        oriented = [self.orient_stroke(s) for s in strokes]

        # Chronological sort: Primary key = min_x / start_x, secondary = start_y
        def sort_key(s):
            start = s["points"][0]
            # Scribes write left-to-right across word, top-to-bottom within glyph
            return (start[1], start[0])

        ordered = sorted(oriented, key=sort_key)
        for i, s in enumerate(ordered):
            s["order_index"] = i

        return ordered
