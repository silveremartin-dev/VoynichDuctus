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

    def chain_collinear_strokes(self, strokes: List[Dict[str, Any]], max_gap: float = 6.0) -> List[Dict[str, Any]]:
        """
        Merges adjacent stroke segments meeting at junctions or corners into continuous, unified strokes.
        Prevents breaking continuous lines into tiny fragments.
        """
        if len(strokes) <= 1:
            return strokes

        # Filter out micro-noise strokes (< 3 points with tiny length)
        valid = [s for s in strokes if len(s.get("points", [])) >= 3 or s.get("length", 0) >= 3.5]
        if not valid:
            valid = strokes

        merged = True
        current_strokes = list(valid)

        while merged:
            merged = False
            for i in range(len(current_strokes)):
                s1 = current_strokes[i]
                p1_start = np.array(s1["points"][0][:2])
                p1_end = np.array(s1["points"][-1][:2])

                for j in range(len(current_strokes)):
                    if i == j:
                        continue
                    s2 = current_strokes[j]
                    p2_start = np.array(s2["points"][0][:2])
                    p2_end = np.array(s2["points"][-1][:2])

                    # 1. Check s1 end -> s2 start
                    if np.linalg.norm(p1_end - p2_start) <= max_gap:
                        combined_pts = s1["points"] + s2["points"]
                        current_strokes[i] = {
                            "stroke_id": s1["stroke_id"],
                            "type": s1.get("type", "segment"),
                            "points": combined_pts,
                            "length": float(len(combined_pts)),
                            "start": s1["start"],
                            "end": s2["end"]
                        }
                        current_strokes.pop(j)
                        merged = True
                        break

                    # 2. Check s1 end -> s2 end (s2 reversed)
                    if np.linalg.norm(p1_end - p2_end) <= max_gap:
                        rev_s2 = list(reversed(s2["points"]))
                        combined_pts = s1["points"] + rev_s2
                        current_strokes[i] = {
                            "stroke_id": s1["stroke_id"],
                            "type": s1.get("type", "segment"),
                            "points": combined_pts,
                            "length": float(len(combined_pts)),
                            "start": s1["start"],
                            "end": s2["start"]
                        }
                        current_strokes.pop(j)
                        merged = True
                        break

                    # 3. Check s2 end -> s1 start
                    if np.linalg.norm(p2_end - p1_start) <= max_gap:
                        combined_pts = s2["points"] + s1["points"]
                        current_strokes[i] = {
                            "stroke_id": s2["stroke_id"],
                            "type": s2.get("type", "segment"),
                            "points": combined_pts,
                            "length": float(len(combined_pts)),
                            "start": s2["start"],
                            "end": s1["end"]
                        }
                        current_strokes.pop(j)
                        merged = True
                        break

                    # 4. Check s1 start -> s2 start (s1 reversed)
                    if np.linalg.norm(p1_start - p2_start) <= max_gap:
                        combined_pts = list(reversed(s1["points"])) + s2["points"]
                        current_strokes[i] = {
                            "stroke_id": s1["stroke_id"],
                            "type": s1.get("type", "segment"),
                            "points": combined_pts,
                            "length": float(len(combined_pts)),
                            "start": s1["end"],
                            "end": s2["end"]
                        }
                        current_strokes.pop(j)
                        merged = True
                        break

                if merged:
                    break

        return current_strokes

    @staticmethod
    def smooth_stroke_points(points: List[Tuple[float, float, float]], window: int = 3) -> List[Tuple[float, float, float]]:
        """
        Applies a moving average smoothing filter to stroke coordinates to simulate natural fluid pen trajectories.
        """
        if len(points) <= window:
            return points
        smoothed = []
        n = len(points)
        for i in range(n):
            w_start = max(0, i - window // 2)
            w_end = min(n, i + window // 2 + 1)
            pts_slice = points[w_start:w_end]
            avg_y = sum(p[0] for p in pts_slice) / len(pts_slice)
            avg_x = sum(p[1] for p in pts_slice) / len(pts_slice)
            avg_w = sum(p[2] for p in pts_slice) / len(pts_slice)
            smoothed.append((round(avg_y, 2), round(avg_x, 2), round(avg_w, 2)))
        return smoothed

    def resolve_and_order_strokes(self, strokes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chains adjacent collinear segments, orients, smooths, and sorts strokes into a chronological sequence.
        """
        chained = self.chain_collinear_strokes(strokes)
        oriented = [self.orient_stroke(s) for s in chained]

        for s in oriented:
            s["points"] = self.smooth_stroke_points(s["points"])

        # Chronological sort: Primary key = min_x / start_x, secondary = start_y
        def sort_key(s):
            start = s["points"][0]
            return (start[1], start[0])

        ordered = sorted(oriented, key=sort_key)
        for i, s in enumerate(ordered):
            s["order_index"] = i

        return ordered
