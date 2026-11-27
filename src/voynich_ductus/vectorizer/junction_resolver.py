"""
Junction resolution and scribal ductus kinematics ordering based on tangent continuity
and 15th-century right-handed quill mechanics.
"""

import math
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
        - Closed loops preserve their sequential cycle closure intact.
        - Predominantly top-to-bottom (downstrokes on ascenders/descenders).
        - Predominantly left-to-right (horizontal traverses/ligatures).
        """
        pts = stroke["points"]
        if len(pts) < 2:
            return stroke

        y0, x0 = pts[0][0], pts[0][1]
        y1, x1 = pts[-1][0], pts[-1][1]

        if len(pts) >= 6 and np.hypot(y1 - y0, x1 - x0) <= 3.5:
            stroke["reversed_by_prior"] = False
            return stroke

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
        dy1 = float(pts1[-1][0]) - float(pts1[idx1][0])
        dx1 = float(pts1[-1][1]) - float(pts1[idx1][1])
        # Entry vector of pts2
        idx2 = min(len(pts2) - 1, k)
        dy2 = float(pts2[idx2][0]) - float(pts2[0][0])
        dx2 = float(pts2[idx2][1]) - float(pts2[0][1])

        n1, n2 = math.hypot(dy1, dx1), math.hypot(dy2, dx2)
        if n1 < 1e-4 or n2 < 1e-4:
            return 1.0
        val = (dy1 * dy2 + dx1 * dx2) / (n1 * n2)
        return max(-1.0, min(1.0, val))

    def chain_collinear_strokes(
        self,
        strokes: List[Dict[str, Any]],
        max_gap: float = 8.0,
        min_cos: float = -0.1,
        max_strokes: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Merges adjacent stroke segments meeting at junctions or corners into continuous, unified strokes.
        Enforces tangent continuity (cos theta >= min_cos) to prevent unnatural hairpin reversals,
        and if max_strokes > 0, iteratively coalesces remaining close segments or drops minor fragments
        until stroke count respects physical pen-lift bounds.
        """
        if len(strokes) <= 1:
            return strokes

        # Filter out micro-spur artifacts (< 5 points or < 5px length when substantial strokes exist)
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
                p1_s = s1["points"][0]
                p1_e = s1["points"][-1]
                p1sy, p1sx = float(p1_s[0]), float(p1_s[1])
                p1ey, p1ex = float(p1_e[0]), float(p1_e[1])

                for j in range(len(current_strokes)):
                    if i == j:
                        continue
                    s2 = current_strokes[j]
                    p2_s = s2["points"][0]
                    p2_e = s2["points"][-1]
                    p2sy, p2sx = float(p2_s[0]), float(p2_s[1])
                    p2ey, p2ex = float(p2_e[0]), float(p2_e[1])

                    # Option 1: s1 end -> s2 start
                    if math.hypot(p1ey - p2sy, p1ex - p2sx) <= max_gap:
                        cos_sim = self._tangent_cosine(s1["points"], s2["points"])
                        if cos_sim >= min_cos and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s1["points"] + s2["points"], s1["start"], s2["end"])

                    # Option 2: s1 end -> s2 end (s2 reversed)
                    if math.hypot(p1ey - p2ey, p1ex - p2ex) <= max_gap:
                        rev_s2 = list(reversed(s2["points"]))
                        cos_sim = self._tangent_cosine(s1["points"], rev_s2)
                        if cos_sim >= min_cos and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s1["points"] + rev_s2, s1["start"], s2["start"])

                    # Option 3: s2 end -> s1 start
                    if math.hypot(p2ey - p1sy, p2ex - p1sx) <= max_gap:
                        cos_sim = self._tangent_cosine(s2["points"], s1["points"])
                        if cos_sim >= min_cos and cos_sim > best_cos:
                            best_cos = cos_sim
                            best_merge = (i, j, s2["points"] + s1["points"], s2["start"], s1["end"])

                    # Option 4: s1 start -> s2 start (s1 reversed)
                    if math.hypot(p1sy - p2sy, p1sx - p2sx) <= max_gap:
                        rev_s1 = list(reversed(s1["points"]))
                        cos_sim = self._tangent_cosine(rev_s1, s2["points"])
                        if cos_sim >= min_cos and cos_sim > best_cos:
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

        # Second pass if count still exceeds physical ceiling: merge adjacent junction legs even across sharper bends
        if max_strokes > 0 and len(current_strokes) > max_strokes:
            gap_relaxed = max_gap * 1.6
            changed = True
            while changed and len(current_strokes) > max_strokes:
                changed = False
                best_pair = None
                best_dist = gap_relaxed + 1.0
                for i in range(len(current_strokes)):
                    s1 = current_strokes[i]
                    p1_s, p1_e = s1["points"][0], s1["points"][-1]
                    p1sy, p1sx = float(p1_s[0]), float(p1_s[1])
                    p1ey, p1ex = float(p1_e[0]), float(p1_e[1])
                    for j in range(i + 1, len(current_strokes)):
                        s2 = current_strokes[j]
                        p2_s, p2_e = s2["points"][0], s2["points"][-1]
                        p2sy, p2sx = float(p2_s[0]), float(p2_s[1])
                        p2ey, p2ex = float(p2_e[0]), float(p2_e[1])
                        candidates = [
                            (math.hypot(p1ey - p2sy, p1ex - p2sx), s1["points"] + s2["points"], s1["start"], s2["end"]),
                            (math.hypot(p1ey - p2ey, p1ex - p2ex), s1["points"] + list(reversed(s2["points"])), s1["start"], s2["start"]),
                            (math.hypot(p2ey - p1sy, p2ex - p1sx), s2["points"] + s1["points"], s2["start"], s1["end"]),
                            (math.hypot(p1sy - p2sy, p1sx - p2sx), list(reversed(s1["points"])) + s2["points"], s1["end"], s2["end"]),
                        ]
                        for d_val, comb, st, en in candidates:
                            if d_val < best_dist:
                                best_dist = d_val
                                best_pair = (i, j, comb, st, en)
                if best_pair is not None and best_dist <= gap_relaxed:
                    i, j, comb, st, en = best_pair
                    s_base = current_strokes[i]
                    current_strokes[i] = {
                        "stroke_id": s_base["stroke_id"],
                        "type": s_base.get("type", "segment"),
                        "points": comb,
                        "length": float(len(comb)),
                        "start": st,
                        "end": en
                    }
                    current_strokes.pop(j)
                    changed = True

            if len(current_strokes) > max_strokes:
                current_strokes.sort(key=lambda s: len(s.get("points", [])), reverse=True)
                current_strokes = current_strokes[:max_strokes]

        return current_strokes

    @staticmethod
    def smooth_stroke_points(points: List[Tuple[float, float, float]], window: int = 3) -> List[Tuple[float, float, float]]:
        """
        Applies a moving average smoothing filter to stroke coordinates (wrapping periodically for closed loops).
        """
        if len(points) <= window:
            return points
        n = len(points)
        is_closed = (n >= 6 and float(np.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1])) <= 3.5)
        half_w = window // 2
        smoothed = []
        if is_closed:
            ring = points[:-1]
            m = len(ring)
            for i in range(m):
                pts_slice = [ring[(i + k) % m] for k in range(-half_w, half_w + 1)]
                avg_y = sum(p[0] for p in pts_slice) / len(pts_slice)
                avg_x = sum(p[1] for p in pts_slice) / len(pts_slice)
                avg_w = sum(p[2] for p in pts_slice) / len(pts_slice)
                smoothed.append((round(avg_y, 2), round(avg_x, 2), round(avg_w, 2)))
            smoothed.append(smoothed[0])
            return smoothed

        for i in range(n):
            w_start = max(0, i - half_w)
            w_end = min(n, i + half_w + 1)
            pts_slice = points[w_start:w_end]
            avg_y = sum(p[0] for p in pts_slice) / len(pts_slice)
            avg_x = sum(p[1] for p in pts_slice) / len(pts_slice)
            avg_w = sum(p[2] for p in pts_slice) / len(pts_slice)
            smoothed.append((round(avg_y, 2), round(avg_x, 2), round(avg_w, 2)))
        return smoothed

    def resolve_and_order_strokes(
        self,
        strokes: List[Dict[str, Any]],
        max_gap: float = 8.0,
        min_cos: float = -0.1,
        max_strokes: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Chains adjacent collinear segments, orients, smooths, and sorts strokes into a chronological sequence.
        """
        chained = self.chain_collinear_strokes(strokes, max_gap=max_gap, min_cos=min_cos, max_strokes=max_strokes)
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
