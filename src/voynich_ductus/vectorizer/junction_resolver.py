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

    @staticmethod
    def _tangent_cosine(pts1: List[Tuple[float, float, float]], pts2: List[Tuple[float, float, float]], k: int = 4) -> float:
        """
        Computes the cosine of the angle between the exit tangent of stroke 1 and entry tangent of stroke 2.
        Values >= 0.0 indicate smooth forward continuity; values < 0.0 indicate sharp hairpins.
        """
        if len(pts1) < 2 or len(pts2) < 2:
            return 1.0
        # Exit vector of pts1
        idx1 = max(0, len(pts1) - 1 - k)
        v1 = np.array(pts1[-1][:2]) - np.array(pts1[idx1][:2])
        # Entry vector of pts2
        idx2 = min(len(pts2) - 1, k)
        v2 = np.array(pts2[idx2][:2]) - np.array(pts2[0][:2])

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-4 or n2 < 1e-4:
            return 1.0
        return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    def chain_collinear_strokes(self, strokes: List[Dict[str, Any]], max_gap: float = 8.0) -> List[Dict[str, Any]]:
        """
        Merges adjacent stroke segments meeting at junctions or corners into continuous, unified strokes.
        Enforces tangent continuity (cos theta >= -0.1) to prevent unnatural hairpin reversals.
        """
        if len(strokes) <= 1:
            return strokes

        # Filter out micro-spur artifacts (< 5 points or < 5px length when substantial strokes exist)
        total_len = sum(s.get("length", len(s.get("points", []))) for s in strokes)
        valid = []
        for s in strokes:
            s_len = s.get("length", len(s.get("points", [])))
            if s_len >= 5.0 or len(s.get("points", [])) >= 5 or len(strokes) <= 2:
                valid.append(s)

        if not valid:
            valid = strokes

        merged = True
        current_strokes = list(valid)

        while merged:
            merged = False
            best_merge = None
            best_cos = -2.0

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

                    # Option 1: s1 end -> s2 start
                    if np.linalg.norm(p1_end - p2_start) <= max_gap:
                        cos_sim = self._tangent_cosine(s1["points"], s2["points"])
                        if cos_sim >= -0.1 and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s1["points"] + s2["points"], s1["start"], s2["end"])

                    # Option 2: s1 end -> s2 end (s2 reversed)
                    if np.linalg.norm(p1_end - p2_end) <= max_gap:
                        rev_s2 = list(reversed(s2["points"]))
                        cos_sim = self._tangent_cosine(s1["points"], rev_s2)
                        if cos_sim >= -0.1 and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s1["points"] + rev_s2, s1["start"], s2["start"])

                    # Option 3: s2 end -> s1 start
                    if np.linalg.norm(p2_end - p1_start) <= max_gap:
                        cos_sim = self._tangent_cosine(s2["points"], s1["points"])
                        if cos_sim >= -0.1 and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s2["points"] + s1["points"], s2["start"], s1["end"])

                    # Option 4: s1 start -> s2 start (s1 reversed)
                    if np.linalg.norm(p1_start - p2_start) <= max_gap:
                        rev_s1 = list(reversed(s1["points"]))
                        cos_sim = self._tangent_cosine(rev_s1, s2["points"])
                        if cos_sim >= -0.1 and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, rev_s1 + s2["points"], s1["end"], s2["end"])

            if best_merge:
                i, j, combined_pts, start_pt, end_pt = best_merge
                s_base = current_strokes[i]
                current_strokes[i] = {
                    "stroke_id": s_base["stroke_id"],
                    "type": s_base.get("type", "segment"),
                    "points": combined_pts,
                    "length": float(len(combined_pts)),
                    "start": start_pt,
                    "end": end_pt
                }
                current_strokes.pop(j)
                merged = True

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
