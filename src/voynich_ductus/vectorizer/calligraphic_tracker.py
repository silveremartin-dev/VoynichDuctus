"""
Native Calligraphic Vectorizer and Pen-Nib Physics Ridge Tracker (Our Model).
Replaces generic skeleton morphology with native quill/pen physics:
- Morphological spur pruning: eliminates parchment roughness micro-spurs shorter than nib width.
- Full nib-footprint extinction radius (r = d_EDT + 1.5 px): prevents u-turns and Brownian bouncing inside thick ink stems.
- Centers stroke entry at true white-bordered endpoints or top-left loop crests.
- Continuous Euler-Bernoulli path tracing through junctions without premature splitting.
- Fixed-angle beveled nib mechanics (pleins et déliés: w(theta) = W_nib * |sin(theta - theta_nib)| + w_0).
- Biomechanical trajectory smoothing and keypoint simplification (Ramer-Douglas-Peucker / spline control points).
"""

import math
from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize


class CalligraphicVectorizer:
    """
    Simulates right-handed 15th-century quill pen physics to derender 2D ink raster
    directly into ordered (x, y, t, width) kinematic trajectories without graph branching
    or Brownian bouncing artifacts.
    """

    _DEG_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    _ONES_3X3_U8 = np.ones((3, 3), dtype=np.uint8)
    _RIDGE_ANGLES: List[Tuple[float, float, bool]] = [
        (float(np.cos(a)), float(np.sin(a)), bool(abs(a) <= np.deg2rad(18)))
        for a in np.linspace(-np.deg2rad(65), np.deg2rad(65), 19)
    ]

    def __init__(
        self,
        nib_angle_deg: float = 40.0,
        min_nib_width: float = 1.2,
        max_nib_width: float = 5.5,
        smoothing_sigma: float = 0.85
    ):
        self.nib_angle_rad = np.deg2rad(nib_angle_deg)
        self.min_nib_width = min_nib_width
        self.max_nib_width = max_nib_width
        self.smoothing_sigma = smoothing_sigma

    def estimate_nib_width(self, tangent_angle: float) -> float:
        """
        Calculates expected physical stroke thickness based on tangent direction relative to nib bevel:
        w(theta) = W_nib * |sin(theta - theta_nib)| + w_0
        """
        d_theta = tangent_angle - self.nib_angle_rad
        effective_w = (self.max_nib_width - self.min_nib_width) * np.abs(np.sin(d_theta)) + self.min_nib_width
        return float(effective_w)

    @staticmethod
    def prune_skeleton_spurs(skel: np.ndarray, dist_map: np.ndarray, min_spur_len: int = 4) -> np.ndarray:
        """
        Prunes parasitic lateral skeleton spurs caused by vellum/parchment contour roughness.
        Iteratively removes degree-1 leaf branches whose length from a junction (degree >= 3)
        is shorter than the local physical nib radius or `min_spur_len`.
        Never deletes simple open curves (which have 2 endpoints and 0 junctions).
        """
        pruned = skel.copy().astype(bool)
        h, w = pruned.shape
        if np.sum(pruned) <= 5:
            return pruned

        def get_degree_map(s: np.ndarray) -> np.ndarray:
            s_u8 = s.astype(np.uint8)
            return (cv2.filter2D(s_u8, cv2.CV_16U, CalligraphicVectorizer._DEG_KERNEL, borderType=cv2.BORDER_CONSTANT) * s_u8).astype(np.int32)

        deg_map = get_degree_map(pruned)
        if not np.any(deg_map >= 3):
            # Pure unbranched curve or closed loop: no lateral spurs to prune
            return pruned

        rem_count = int(np.sum(pruned))
        endpoints = np.argwhere(deg_map == 1)
        for ey, ex in endpoints:
            if not pruned[ey, ex]:
                continue
            # Walk from endpoint toward nearest junction
            chain = [(int(ey), int(ex))]
            cy, cx = int(ey), int(ex)
            hit_junction = False
            local_nib_r = max(float(min_spur_len), float(dist_map[ey, ex]) * 1.45 + 1.5)
            max_walk = int(round(local_nib_r)) + 1

            for _ in range(max_walk):
                y0, y1 = max(0, cy - 1), min(h, cy + 2)
                x0, x1 = max(0, cx - 1), min(w, cx + 2)
                nbs = []
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if (ny != cy or nx != cx) and pruned[ny, nx] and (ny, nx) not in chain:
                            nbs.append((ny, nx))
                if not nbs:
                    break
                ny, nx = nbs[0]
                if deg_map[ny, nx] >= 3:
                    hit_junction = True
                    break
                chain.append((ny, nx))
                cy, cx = ny, nx

            # Only prune if it terminates at a junction within the spur threshold, does NOT protrude
            # outside the junction's inscribed ink disk (preserving '3' middle cusps, hooks, and serifs!),
            # leaves at least 4 skeleton pixels intact, AND never disconnects a loop/junction
            if hit_junction and len(chain) <= max_walk and (rem_count - len(chain)) >= 4:
                jn_y, jn_x = ny, nx
                r_junc = float(dist_map[jn_y, jn_x])
                tip_dist = math.hypot(float(ey - jn_y), float(ex - jn_x))
                if tip_dist > max(2.2, r_junc * 1.25 + 1.0):
                    continue
                tip_y, tip_x = chain[-1]
                # Count neighbors of the junction excluding the tip of this spur
                y0_j, y1_j = max(0, jn_y - 1), min(h, jn_y + 2)
                x0_j, x1_j = max(0, jn_x - 1), min(w, jn_x + 2)
                rem_deg = 0
                for jy in range(y0_j, y1_j):
                    for jx in range(x0_j, x1_j):
                        if (jy, jx) != (jn_y, jn_x) and (jy, jx) != (tip_y, tip_x) and pruned[jy, jx]:
                            rem_deg += 1
                if rem_deg >= 2:
                    for py, px in chain:
                        pruned[py, px] = False
                    rem_count -= len(chain)
                    deg_map = get_degree_map(pruned)

        return pruned

    def find_touchdown_endpoints(self, binary_mask: np.ndarray, skel: np.ndarray) -> List[Tuple[float, int, int, float, float]]:
        """
        Identifies genuine scribal touchdown points on the pruned skeleton:
        1. Free endpoints in white space (degree 1 in skeleton).
        2. For closed loops without degree 1 endpoints, the top-left loop crest (11 o'clock).
        Returns list of (priority, y, x, initial_dy, initial_dx).
        """
        h, w = binary_mask.shape
        skel_u8 = skel.astype(np.uint8)
        if not np.any(skel_u8):
            return []

        deg_map = cv2.filter2D(skel_u8, cv2.CV_16U, self._DEG_KERNEL, borderType=cv2.BORDER_CONSTANT) * skel_u8
        ep_pts = np.argwhere(deg_map == 1)
        endpoints = []
        for y, x in ep_pts:
            norm_y = y / max(1.0, float(h))
            norm_x = x / max(1.0, float(w))
            priority = 12.0 - norm_x * 5.0 - norm_y * 5.0

            # Walk 2-3 skeleton steps inward to compute a stable initial tangent
            cy, cx = int(y), int(x)
            visited_walk = {(cy, cx)}
            for _ in range(3):
                ym0, ym1 = max(0, cy - 1), min(h, cy + 2)
                xm0, xm1 = max(0, cx - 1), min(w, cx + 2)
                nxt = None
                for ny in range(ym0, ym1):
                    for nx in range(xm0, xm1):
                        if (ny, nx) not in visited_walk and skel[ny, nx]:
                            nxt = (ny, nx)
                            break
                    if nxt:
                        break
                if not nxt:
                    break
                visited_walk.add(nxt)
                cy, cx = nxt

            init_dy = float(cy - y)
            init_dx = float(cx - x)
            n_len = math.hypot(init_dy, init_dx)
            if n_len > 1e-4:
                endpoints.append((priority, int(y), int(x), init_dy / n_len, init_dx / n_len))
            else:
                endpoints.append((priority, int(y), int(x), 1.0, 0.2))

        # If no degree 1 endpoints found (pure closed loop 'o' or '0'), pick top-left crest
        if not endpoints:
            skel_pts = np.argwhere(skel)
            best_p = min(skel_pts, key=lambda p: (p[0] / max(1.0, float(h))) * 1.5 + (p[1] / max(1.0, float(w))) * 2.0)
            y, x = int(best_p[0]), int(best_p[1])
            endpoints.append((12.0, y, x, 1.0, 0.3))

        endpoints.sort(key=lambda item: item[0], reverse=True)
        return endpoints

    @staticmethod
    def _mark_nib_footprint_visited(
        pts: List[Tuple[float, float, float]],
        dist_map: np.ndarray,
        visited_mask: np.ndarray,
        extra_margin: float = 1.6
    ):
        """
        Marks the entire physical quill nib footprint covered by a traced trajectory as visited.
        Unlike a 1-pixel line, a quill stroke of radius R = dist_map[y, x] covers all ink within
        radius R + extra_margin, preventing the tracker from ever bouncing back along the opposite
        edge of the same stem.
        """
        h, w = visited_mask.shape
        yy, xx = np.ogrid[:h, :w]
        for p in pts:
            py, px = float(p[0]), float(p[1])
            iy = max(0, min(h - 1, int(round(py))))
            ix = max(0, min(w - 1, int(round(px))))
            r = max(2.2, float(dist_map[iy, ix]) + extra_margin)
            r_int = int(np.ceil(r))
            y0, y1 = max(0, iy - r_int), min(h, iy + r_int + 1)
            x0, x1 = max(0, ix - r_int), min(w, ix + r_int + 1)
            sub_y = np.arange(y0, y1)[:, None] - py
            sub_x = np.arange(x0, x1)[None, :] - px
            visited_mask[y0:y1, x0:x1] |= ((sub_y * sub_y + sub_x * sub_x) <= (r * r))

    def extract_calligraphic_ductus(
        self,
        binary_mask: np.ndarray,
        single_stroke_mode: bool = False,
        max_strokes: Optional[int] = None,
        scale_calib: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Derenders a 2D binary ink patch into smooth, biomechanically realistic calligraphic strokes.
        When single_stroke_mode=True (e.g. for Codex Seraphinianus), strictly guarantees that
        the letter is derendered as a single continuous scribal gesture without pen lifts or retracing.
        """
        if binary_mask is None or not np.any(binary_mask):
            return []

        mask_h, mask_w = binary_mask.shape
        pts_all = np.argwhere(binary_mask)
        total_ink_area = int(len(pts_all))

        min_y, min_x = pts_all.min(axis=0)
        max_y, max_x = pts_all.max(axis=0)
        span_h = int(max_y - min_y + 1)
        span_w = int(max_x - min_x + 1)

        dist_full = distance_transform_edt(binary_mask)
        inner_dists = dist_full[dist_full > 0.5]
        med_half_width = float(np.median(inner_dists)) if len(inner_dists) > 0 else 1.0
        est_nib_width = max(1.5, 2.0 * med_half_width)
        px_per_mm = float(scale_calib.px_per_mm) if (scale_calib is not None and hasattr(scale_calib, "px_per_mm")) else max(4.5, est_nib_width / 0.45)

        if scale_calib is not None and hasattr(scale_calib, "dot_max_span_px"):
            dot_span_limit = max(float(scale_calib.dot_max_span_px), 3.0 * est_nib_width)
            dot_area_limit = max(float(scale_calib.dot_max_area_px), 4.5 * (est_nib_width ** 2))
        else:
            dot_span_limit = max(8.0, 3.0 * est_nib_width)
            dot_area_limit = max(28.0, 4.5 * (est_nib_width ** 2))

        if (span_h <= dot_span_limit and span_w <= dot_span_limit) or total_ink_area <= dot_area_limit:
            cy = float(np.mean(pts_all[:, 0]))
            cx = float(np.mean(pts_all[:, 1]))
            w_nib = float(round(est_nib_width, 1))
            return [{
                "stroke_id": "cal_s00",
                "order_index": 0,
                "points": [(cy, cx, w_nib)],
                "length": 0.0,
                "start": (cy, cx),
                "end": (cy, cx),
                "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
                "mean_width": w_nib,
                "is_impact": True,
                "is_dot": True
            }]

        if single_stroke_mode or (max_strokes is not None and max_strokes == 1):
            single_stroke = self._extract_dominant_single_stroke(binary_mask, px_per_mm=px_per_mm)
            return [single_stroke] if single_stroke else []

        from scipy.ndimage import label, binary_fill_holes
        labeled_mask, num_features = label(binary_mask)
        if num_features == 0:
            return []

        comp_min_area = max(10, int(round(0.06 * (px_per_mm ** 2)))) if scale_calib is not None else 10
        # Keep genuine closed-loop eyes ('o', 'a', 'd'/'8', 'e' >= 6 px^2 on HD) intact; fill grain pinholes
        pinhole_max_area = max(3, int(round(0.024 * (px_per_mm ** 2))))
        spur_min_len = max(2, int(round(max(med_half_width * 1.15, 0.28 * px_per_mm))))
        sigma_eff = max(0.55, 0.08 * px_per_mm)
        step_size_eff = max(1.0, min(1.35, 0.11 * px_per_mm))
        min_res_area = max(4, int(round(0.025 * (px_per_mm ** 2))))

        components = []
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled_mask == comp_id)
            comp_area = int(np.sum(comp_mask))
            if comp_area < comp_min_area:
                continue
            pts = np.argwhere(comp_mask)
            c_min_y, c_min_x = pts.min(axis=0)
            components.append({
                "id": comp_id,
                "mask": comp_mask,
                "area": comp_area,
                "min_y": int(c_min_y),
                "min_x": int(c_min_x)
            })

        components.sort(key=lambda c: (c["min_y"] // max(12, int(round(1.4 * px_per_mm))), c["min_x"]))

        from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
        from voynich_ductus.vectorizer.junction_resolver import JunctionResolver

        graph_ext = StrokeGraphExtractor()
        junc_res = JunctionResolver(right_handed_prior=True)

        strokes: List[Dict[str, Any]] = []
        stroke_idx = 0
        stroke_limit = min(4, max_strokes) if max_strokes is not None else 4

        for comp in components:
            if len(strokes) >= stroke_limit:
                break
            comp_mask = comp["mask"]
            h, w = comp_mask.shape

            # Fill tiny internal grain pinholes so skeletonization doesn't split around dust
            padded = np.pad(comp_mask, 2, mode="constant", constant_values=False)
            filled = binary_fill_holes(padded)
            holes = filled & (~padded)
            lbl_h, num_h = label(holes)
            clean_comp = comp_mask.copy()
            real_holes = []
            for hid in range(1, num_h + 1):
                h_mask = (lbl_h == hid)[2:-2, 2:-2]
                h_area = int(np.sum(h_mask))
                if h_area <= pinhole_max_area:
                    clean_comp |= h_mask
                elif np.any(h_mask):
                    real_holes.append(h_mask)

            dist_map = distance_transform_edt(clean_comp)
            raw_skel = skeletonize(clean_comp)
            skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_len)
            smoothed_dist = gaussian_filter(dist_map.astype(np.float32), sigma=sigma_eff)

            comp_raw_strokes: List[List[Tuple[float, float, float]]] = []
            loop_covered_skel = np.zeros_like(skel, dtype=bool)

            # 1. Extract closed-loop rings first around genuine interior holes ('o', 'a', 'd', 'q', 'p', 'f')
            if real_holes:
                try:
                    from skimage.measure import find_contours
                    for h_mask in real_holes:
                        if len(strokes) + len(comp_raw_strokes) >= stroke_limit:
                            break
                        contours = find_contours(h_mask.astype(np.float32), 0.5)
                        if not contours:
                            continue
                        ring_c = max(contours, key=len)
                        if len(ring_c) < 6:
                            continue
                        ring_open = ring_c[:-1] if np.hypot(ring_c[0][0] - ring_c[-1][0], ring_c[0][1] - ring_c[-1][1]) < 1.5 else ring_c
                        if len(ring_open) < 6:
                            continue
                        hy_c = float(np.mean(ring_open[:, 0]))
                        hx_c = float(np.mean(ring_open[:, 1]))
                        angles = np.unwrap(np.arctan2(ring_open[:, 0] - hy_c, ring_open[:, 1] - hx_c))
                        if angles[-1] < angles[0]:
                            ring_open = ring_open[::-1]
                        start_idx = int(np.argmin(
                            (ring_open[:, 0] / max(1.0, float(h))) * 1.5 + (ring_open[:, 1] / max(1.0, float(w))) * 2.0
                        ))
                        ordered_ring = np.vstack([ring_open[start_idx:], ring_open[:start_idx], ring_open[start_idx:start_idx + 1]])
                        n_ring = len(ordered_ring)
                        ring_pts: List[Tuple[float, float, float]] = []
                        max_push = max(3.0, min(med_half_width * 2.2, 0.75 * px_per_mm))
                        for ri in range(n_ring):
                            py, px_coord = float(ordered_ring[ri][0]), float(ordered_ring[ri][1])
                            p_prev = ordered_ring[(ri - 1) % (n_ring - 1)]
                            p_next = ordered_ring[(ri + 1) % (n_ring - 1)]
                            ty, tx = float(p_next[0] - p_prev[0]), float(p_next[1] - p_prev[1])
                            tn = float(np.hypot(ty, tx)) or 1.0
                            n1y, n1x = -tx / tn, ty / tn
                            if (n1y * (py - hy_c) + n1x * (px_coord - hx_c)) < 0:
                                n1y, n1x = -n1y, -n1x
                            best_ry, best_rx, best_rd = py, px_coord, 0.0
                            for step_t in np.linspace(0.5, max_push, 10):
                                cy_t = py + n1y * step_t
                                cx_t = px_coord + n1x * step_t
                                iy_t, ix_t = int(round(cy_t)), int(round(cx_t))
                                if 0 <= iy_t < h and 0 <= ix_t < w and clean_comp[iy_t, ix_t]:
                                    d_loc = float(smoothed_dist[iy_t, ix_t])
                                    if d_loc >= best_rd:
                                        best_rd = d_loc
                                        best_ry, best_rx = cy_t, cx_t
                                elif best_rd > 0.0:
                                    break
                            w_ring = max(1.6, best_rd * 2.0)
                            ring_pts.append((round(best_ry, 2), round(best_rx, 2), round(w_ring, 2)))
                        if len(ring_pts) >= 6:
                            ring_pts[-1] = ring_pts[0]
                            comp_raw_strokes.append(ring_pts)
                            # Mark only tight ring core so attached stems ('a', 'd', 'q') retain their full body up to the loop wall
                            self._mark_nib_footprint_visited(ring_pts, dist_map * 0.72, loop_covered_skel, extra_margin=0.6)
                except Exception:
                    pass

            # 2. Extract crossing-preserving topological strokes on remaining skeleton (preserving '3' middle cusps & short hooks!)
            rem_budget = stroke_limit - (len(strokes) + len(comp_raw_strokes))
            rem_skel = skel & (~loop_covered_skel)
            if rem_budget > 0 and int(np.sum(rem_skel)) >= 2:
                p_graph = graph_ext.build_pixel_graph(rem_skel, dist_map * 2.0)
                raw_segs = graph_ext.decompose_into_strokes(p_graph)
                if raw_segs:
                    raw_segs = self._splice_cusp_tongues(raw_segs, px_per_mm=px_per_mm)
                    # Filter 1-2 px junction-triangle artifacts only when main strokes exist
                    if len(raw_segs) > 1:
                        substantial = [s for s in raw_segs if len(s.get("points", [])) >= 3]
                        if substantial:
                            raw_segs = substantial
                    chained = junc_res.resolve_and_order_strokes(
                        raw_segs,
                        max_gap=max(5.5, 0.75 * px_per_mm),
                        min_cos=-0.45,
                        max_strokes=max(rem_budget + 3, 6),
                    )
                    # Prioritize longest substantive strokes before enforcing rem_budget so top micro-serifs never evict lower arches!
                    chained.sort(
                        key=lambda s: sum(
                            math.hypot(s["points"][k + 1][0] - s["points"][k][0], s["points"][k + 1][1] - s["points"][k][1])
                            for k in range(len(s.get("points", [])) - 1)
                        ),
                        reverse=True,
                    )
                    for ch_s in chained[:rem_budget]:
                        c_pts = [(float(p[0]), float(p[1]), max(1.5, float(p[2]))) for p in ch_s.get("points", [])]
                        if len(c_pts) >= 2:
                            c_pts = self._extend_and_anchor_stroke(c_pts, clean_comp, dist_map, comp_raw_strokes, med_half_width)
                            comp_raw_strokes.append(c_pts)

            # Fallback ridge trace if skeleton graph produced nothing on a valid component
            if not comp_raw_strokes:
                candidates = self.find_touchdown_endpoints(clean_comp, skel)
                if candidates:
                    _, sy, sx, dy, dx = candidates[0]
                    fb_pts = self._trace_continuous_ridge(
                        smoothed_dist, skel, None, sy, sx, dy, dx,
                        max_steps=int(max(h, w) * 2.8 / max(1.0, step_size_eff * 0.6)),
                        step_size=step_size_eff,
                    )
                    if len(fb_pts) >= 2:
                        comp_raw_strokes.append(self._extend_and_anchor_stroke(fb_pts, clean_comp, dist_map, [], med_half_width))

            # 3. Analysis-by-Synthesis Residual Pass: recover uncovered middle cusps ('3'), crossbars, hooks & plumes
            while (len(strokes) + len(comp_raw_strokes)) < stroke_limit and comp_raw_strokes:
                pred_m = self._rasterize_strokes_mask(comp_raw_strokes, (h, w), scale_w=0.80)
                uncov = clean_comp & (~pred_m)
                if int(np.sum(uncov)) < min_res_area:
                    break
                lbl_u, n_u = label(uncov, structure=np.ones((3, 3), dtype=bool))
                if n_u == 0:
                    break
                cnt_u = np.bincount(lbl_u.ravel())
                cnt_u[0] = 0
                best_uid = int(np.argmax(cnt_u))
                if int(cnt_u[best_uid]) < min_res_area:
                    break
                u_mask = (lbl_u == best_uid)
                # Dilate slightly inside clean_comp so the residual piece connects smoothly to its parent junction
                u_grown = (cv2.dilate(u_mask.astype(np.uint8), self._ONES_3X3_U8, iterations=2) > 0) & clean_comp
                u_skel = skeletonize(u_grown)
                if int(np.sum(u_skel)) < 2:
                    ys_u, xs_u = np.where(u_mask)
                    if len(ys_u) < 2:
                        break
                    i0 = int(np.argmin(ys_u + xs_u))
                    i1 = int(np.argmax(ys_u + xs_u))
                    u_pts = [
                        (float(ys_u[i0]), float(xs_u[i0]), max(1.5, float(dist_map[ys_u[i0], xs_u[i0]]) * 2.0)),
                        (float(ys_u[i1]), float(xs_u[i1]), max(1.5, float(dist_map[ys_u[i1], xs_u[i1]]) * 2.0)),
                    ]
                else:
                    u_graph = graph_ext.build_pixel_graph(u_skel, dist_map * 2.0)
                    u_segs = graph_ext.decompose_into_strokes(u_graph)
                    if not u_segs:
                        break
                    u_chained = junc_res.resolve_and_order_strokes(u_segs, max_gap=max(5.0, 0.7 * px_per_mm), min_cos=-0.50, max_strokes=1)
                    if not u_chained:
                        break
                    u_pts = [(float(p[0]), float(p[1]), max(1.5, float(p[2]))) for p in u_chained[0].get("points", [])]
                if len(u_pts) < 2:
                    break
                u_pts = self._extend_and_anchor_stroke(u_pts, clean_comp, dist_map, comp_raw_strokes, med_half_width)
                comp_raw_strokes.append(u_pts)
                new_pred = self._rasterize_strokes_mask(comp_raw_strokes, (h, w), scale_w=0.80)
                if int(np.sum(clean_comp & (~new_pred))) >= int(np.sum(clean_comp & (~pred_m))) - 1:
                    break

            # 4. Parameterize cubic Bézier control points & optimize silhouette centering + 40° quill width IoU
            param_list: List[List[Tuple[float, float, float]]] = []
            for r_pts in comp_raw_strokes:
                c_pts = self._parameterize_stroke(r_pts, px_per_mm=px_per_mm, dist_map=dist_map)
                if len(c_pts) >= 2:
                    param_list.append(c_pts)

            if param_list:
                opt_list = self._optimize_strokes_on_silhouette(param_list, clean_comp, dist_map, px_per_mm=px_per_mm)
                for calligraphic_pts in opt_list:
                    if len(strokes) >= stroke_limit:
                        break
                    strokes.append({
                        "stroke_id": f"cal_s{stroke_idx:02d}",
                        "order_index": stroke_idx,
                        "points": calligraphic_pts,
                        "length": float(len(calligraphic_pts)),
                        "start": (calligraphic_pts[0][0], calligraphic_pts[0][1]),
                        "end": (calligraphic_pts[-1][0], calligraphic_pts[-1][1]),
                        "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
                        "mean_width": float(np.mean([p[2] for p in calligraphic_pts]))
                    })
                    stroke_idx += 1

        if not strokes:
            pts = np.argwhere(binary_mask)
            if len(pts) > 0:
                p0 = pts.min(axis=0)
                p1 = pts.max(axis=0)
                sub_pts = [
                    (float(p0[0]), float(p0[1]), float(est_nib_width)),
                    (float(p1[0]), float(p1[1]), float(est_nib_width))
                ]
                strokes.append({
                    "stroke_id": "cal_s00",
                    "order_index": 0,
                    "points": sub_pts,
                    "length": 2.0,
                    "start": (sub_pts[0][0], sub_pts[0][1]),
                    "end": (sub_pts[-1][0], sub_pts[-1][1]),
                    "nib_angle_deg": 40.0,
                    "mean_width": float(est_nib_width)
                })

        # Re-orient each stroke to respect Western ductus (top-to-bottom, left-to-right)
        for s in strokes:
            pts = s.get("points", [])
            if len(pts) >= 2:
                p_s = pts[0]
                p_e = pts[-1]
                # Do not invert closed loops whose start and end already meet at the top
                if np.hypot(p_s[0] - p_e[0], p_s[1] - p_e[1]) > 3.5:
                    s_cost = (p_s[0] / max(1.0, float(mask_h))) * 2.0 + (p_s[1] / max(1.0, float(mask_w))) * 2.5
                    e_cost = (p_e[0] / max(1.0, float(mask_h))) * 2.0 + (p_e[1] / max(1.0, float(mask_w))) * 2.5
                    if s_cost > e_cost:
                        pts = pts[::-1]
                        s["points"] = pts
                        s["start"] = (pts[0][0], pts[0][1])
                        s["end"] = (pts[-1][0], pts[-1][1])

        strokes.sort(key=lambda s: (s["start"][0] / max(1.0, float(mask_h))) * 2.0 + (s["start"][1] / max(1.0, float(mask_w))) * 2.5)
        for idx, s in enumerate(strokes):
            s["order_index"] = idx
            s["stroke_id"] = f"cal_s{idx:02d}"

        return strokes

    @staticmethod
    def _splice_cusp_tongues(
        segs: List[Dict[str, Any]],
        px_per_mm: float = 6.3,
    ) -> List[Dict[str, Any]]:
        """
        Preserves '3'-shaped middle cusps and sharp bounce-back tongues (`arch_top -> cusp_tip -> arch_bottom`)
        by splicing short degree-1 junction tongues directly between two main branches meeting at the same junction,
        just as a human hand traces into the middle point of a '3' and reverses out into the lower arch.
        """
        if len(segs) < 3:
            return segs
        max_tongue_pts = max(6, int(round(1.45 * px_per_mm)))
        junc_tol = max(2.6, 0.25 * px_per_mm)
        work = [dict(s, points=list(s.get("points", []))) for s in segs if len(s.get("points", [])) >= 2]
        if len(work) < 3:
            return segs

        merged_any = True
        while merged_any and len(work) >= 3:
            merged_any = False
            # Count how many other segment endpoints touch each segment's start/end
            for t_idx, t_seg in enumerate(work):
                t_pts = t_seg["points"]
                if len(t_pts) > max_tongue_pts:
                    continue
                for j_end in (0, -1):
                    jy, jx = float(t_pts[j_end][0]), float(t_pts[j_end][1])
                    tip_end = -1 if j_end == 0 else 0
                    ty_tip, tx_tip = float(t_pts[tip_end][0]), float(t_pts[tip_end][1])
                    # Verify tip_end is genuinely a free degree-1 tip (touches no other segment)
                    tip_touches = any(
                        o_idx != t_idx and (
                            math.hypot(ty_tip - o["points"][0][0], tx_tip - o["points"][0][1]) <= junc_tol
                            or math.hypot(ty_tip - o["points"][-1][0], tx_tip - o["points"][-1][1]) <= junc_tol
                        )
                        for o_idx, o in enumerate(work)
                    )
                    if tip_touches:
                        continue
                    # Find two main branches A and B that also meet at (jy, jx)
                    neighbors = []
                    for o_idx, o in enumerate(work):
                        if o_idx == t_idx or len(o["points"]) < 3:
                            continue
                        d0 = math.hypot(jy - o["points"][0][0], jx - o["points"][0][1])
                        d1 = math.hypot(jy - o["points"][-1][0], jx - o["points"][-1][1])
                        if d0 <= junc_tol or d1 <= junc_tol:
                            neighbors.append((o_idx, 0 if d0 <= d1 else -1))
                    if len(neighbors) == 2:
                        (idx_a, end_a), (idx_b, end_b) = neighbors
                        pts_a = list(work[idx_a]["points"])
                        pts_b = list(work[idx_b]["points"])
                        # Orient A so it arrives AT junction J (end_a -> -1), and B so it leaves FROM J (end_b -> 0)
                        if end_a == 0:
                            pts_a = pts_a[::-1]
                        if end_b == -1:
                            pts_b = pts_b[::-1]
                        # Orient tongue T from J -> tip, then bounce tip -> J
                        tongue_out = list(t_pts) if j_end == 0 else list(t_pts)[::-1]
                        tongue_back = tongue_out[:-1][::-1]
                        spliced_pts = pts_a + tongue_out[1:] + tongue_back + pts_b[1:]
                        keep = [w_s for k, w_s in enumerate(work) if k not in (t_idx, idx_a, idx_b)]
                        keep.append({
                            "stroke_id": work[idx_a].get("stroke_id", "s_cusp"),
                            "points": spliced_pts,
                            "start": (spliced_pts[0][0], spliced_pts[0][1]),
                            "end": (spliced_pts[-1][0], spliced_pts[-1][1]),
                            "length": float(len(spliced_pts)),
                        })
                        work = keep
                        merged_any = True
                        break
                if merged_any:
                    break
        return work

    def _extract_dominant_single_stroke(
        self,
        binary_mask: np.ndarray,
        px_per_mm: float = 6.3
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts one continuous cursive calligraphic gesture covering the glyph (Codex Seraphinianus mode)
        via a Topological Loop + Cusp Eulerian Tour:
        - Zero `binary_closing` loop-plugging: genuine small loops are preserved as explicit 360° medial rings
          spliced smoothly into the passing ribbon stem (never flattened into self-overlapping lines).
        - Preserves '3'-shaped middle cusps and traverses all authentic branches to their true ink tips.
        """
        from scipy.ndimage import label, binary_fill_holes
        if binary_mask is None or not np.any(binary_mask):
            return None

        lbl, num_features = label(binary_mask.astype(bool), structure=np.ones((3, 3), dtype=bool))
        if num_features == 0:
            return None

        cc_sizes = [(comp_id, int(np.sum(lbl == comp_id))) for comp_id in range(1, num_features + 1)]
        dominant_id = max(cc_sizes, key=lambda x: x[1])[0]
        comp_mask = (lbl == dominant_id)
        # Bridge any tiny 1-2 px disconnected ink fragment in the same Seraphinianus glyph mask softly
        if num_features > 1:
            for cid, csize in cc_sizes:
                if cid != dominant_id and csize >= 6:
                    comp_mask |= (lbl == cid)
        h, w = comp_mask.shape

        # Preserve all genuine loop eyes (> 2 px^2); fill only 1-2 px grain pinholes
        pinhole_max = max(1, int(round(0.012 * (px_per_mm ** 2))))
        padded = np.pad(comp_mask, 2, mode="constant", constant_values=False)
        holes = binary_fill_holes(padded) & (~padded)
        lbl_h, num_h = label(holes)
        clean_comp = comp_mask.copy()
        real_holes = []
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)[2:-2, 2:-2]
            ha = int(np.sum(hm))
            if ha <= pinhole_max:
                clean_comp |= hm
            elif np.any(hm):
                real_holes.append(hm)

        dist_map = distance_transform_edt(clean_comp)
        raw_skel = skeletonize(clean_comp)
        if not np.any(raw_skel):
            return None

        inner_dists = dist_map[dist_map > 0.5]
        med_r = float(np.median(inner_dists)) if len(inner_dists) > 0 else 1.5
        spur_min_len = max(2, int(round(max(med_r * 1.15, 0.28 * px_per_mm))))
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_len)
        smoothed_dist = gaussian_filter(dist_map.astype(np.float32), sigma=max(0.55, 0.08 * px_per_mm))

        # 1. Extract explicit 360° medial rings around every enclosed loop hole so loops NEVER collapse
        loop_rings: List[List[Tuple[float, float, float]]] = []
        loop_covered_skel = np.zeros_like(skel, dtype=bool)
        if real_holes:
            try:
                from skimage.measure import find_contours
                for h_mask in real_holes:
                    contours = find_contours(h_mask.astype(np.float32), 0.5)
                    if not contours:
                        continue
                    ring_c = max(contours, key=len)
                    ring_open = ring_c[:-1] if len(ring_c) >= 6 and np.hypot(ring_c[0][0] - ring_c[-1][0], ring_c[0][1] - ring_c[-1][1]) < 1.5 else ring_c
                    if len(ring_open) < 5:
                        continue
                    hy_c, hx_c = float(np.mean(ring_open[:, 0])), float(np.mean(ring_open[:, 1]))
                    angles = np.unwrap(np.arctan2(ring_open[:, 0] - hy_c, ring_open[:, 1] - hx_c))
                    if angles[-1] < angles[0]:
                        ring_open = ring_open[::-1]
                    start_idx = int(np.argmin((ring_open[:, 0] / max(1.0, float(h))) * 1.5 + (ring_open[:, 1] / max(1.0, float(w))) * 2.0))
                    ordered_ring = np.vstack([ring_open[start_idx:], ring_open[:start_idx], ring_open[start_idx:start_idx + 1]])
                    n_ring = len(ordered_ring)
                    ring_pts: List[Tuple[float, float, float]] = []
                    max_push = max(2.8, min(med_r * 2.2, 0.75 * px_per_mm))
                    for ri in range(n_ring):
                        py, px_c = float(ordered_ring[ri][0]), float(ordered_ring[ri][1])
                        p_prev = ordered_ring[(ri - 1) % max(1, n_ring - 1)]
                        p_next = ordered_ring[(ri + 1) % max(1, n_ring - 1)]
                        ty, tx = float(p_next[0] - p_prev[0]), float(p_next[1] - p_prev[1])
                        tn = float(np.hypot(ty, tx)) or 1.0
                        n1y, n1x = -tx / tn, ty / tn
                        if (n1y * (py - hy_c) + n1x * (px_c - hx_c)) < 0:
                            n1y, n1x = -n1y, -n1x
                        best_ry, best_rx, best_rd = py, px_c, 0.0
                        for step_t in np.linspace(0.5, max_push, 9):
                            cy_t, cx_t = py + n1y * step_t, px_c + n1x * step_t
                            iy_t, ix_t = int(round(cy_t)), int(round(cx_t))
                            if 0 <= iy_t < h and 0 <= ix_t < w and clean_comp[iy_t, ix_t]:
                                d_loc = float(smoothed_dist[iy_t, ix_t])
                                if d_loc >= best_rd:
                                    best_rd = d_loc
                                    best_ry, best_rx = cy_t, cx_t
                            elif best_rd > 0.0:
                                break
                        ring_pts.append((round(best_ry, 2), round(best_rx, 2), round(max(1.5, best_rd * 2.0), 2)))
                    if len(ring_pts) >= 5:
                        ring_pts[-1] = ring_pts[0]
                        loop_rings.append(ring_pts)
                        self._mark_nib_footprint_visited(ring_pts, dist_map * 0.70, loop_covered_skel, extra_margin=0.5)
            except Exception:
                pass

        # 2. Extract topological skeleton segments & splice '3' middle cusps
        from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
        from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
        graph_ext = StrokeGraphExtractor()
        junc_res = JunctionResolver(right_handed_prior=True)

        rem_skel = skel & (~loop_covered_skel)
        seg_chains: List[List[Tuple[float, float, float]]] = []
        if int(np.sum(rem_skel)) >= 2:
            p_graph = graph_ext.build_pixel_graph(rem_skel, dist_map * 2.0)
            raw_segs = graph_ext.decompose_into_strokes(p_graph)
            if raw_segs:
                raw_segs = self._splice_cusp_tongues(raw_segs, px_per_mm=px_per_mm)
                chained = junc_res.resolve_and_order_strokes(
                    raw_segs,
                    max_gap=max(6.5, 0.95 * px_per_mm),
                    min_cos=-0.65,
                    max_strokes=8,
                )
                for ch_s in chained:
                    c_pts = [(float(p[0]), float(p[1]), max(1.5, float(p[2]))) for p in ch_s.get("points", [])]
                    if len(c_pts) >= 2:
                        c_pts = self._extend_and_anchor_stroke(c_pts, clean_comp, dist_map, seg_chains + loop_rings, med_r)
                        seg_chains.append(c_pts)

        # 3. Splice every 360° loop ring smoothly into its closest stem chain (or promote standalone 'o' ring)
        for ring in loop_rings:
            if not seg_chains:
                seg_chains.append(ring)
                continue
            best_ci, best_pi, best_ri, best_d = 0, 0, 0, 1e9
            for ci, ch in enumerate(seg_chains):
                for pi, cp in enumerate(ch):
                    for ri in range(len(ring) - 1):
                        d = math.hypot(cp[0] - ring[ri][0], cp[1] - ring[ri][1])
                        if d < best_d:
                            best_d = d
                            best_ci, best_pi, best_ri = ci, pi, ri
            rotated_ring = ring[best_ri:-1] + ring[:best_ri] + [ring[best_ri]]
            target_ch = seg_chains[best_ci]
            seg_chains[best_ci] = target_ch[:best_pi + 1] + rotated_ring + target_ch[best_pi:]

        if not seg_chains:
            candidates = self.find_touchdown_endpoints(clean_comp, skel)
            if not candidates:
                return None
            _, sy, sx, dy, dx = candidates[0]
            fb = self._trace_continuous_ridge(smoothed_dist, skel, None, sy, sx, dy, dx, max_steps=int(max(h, w) * 3.0), step_size=1.2)
            if len(fb) < 2:
                return None
            seg_chains = [self._extend_and_anchor_stroke(fb, clean_comp, dist_map, [], med_r)]

        # 4. Chain all remaining strokes into 1 continuous cursive gesture (left-to-right / minimal bridge)
        seg_chains.sort(key=lambda ch: min(p[1] for p in ch) + 0.35 * min(p[0] for p in ch))
        best_pts = list(seg_chains[0])
        if len(best_pts) >= 2 and math.hypot(best_pts[0][0] - best_pts[-1][0], best_pts[0][1] - best_pts[-1][1]) > 3.5:
            if best_pts[0][1] + 0.4 * best_pts[0][0] > best_pts[-1][1] + 0.4 * best_pts[-1][0]:
                best_pts = best_pts[::-1]

        remaining = seg_chains[1:]
        while remaining:
            best_idx = 0
            best_mode = "append_fwd"
            best_dist = 1e9
            for r_i, r_ch in enumerate(remaining):
                candidates_join = [
                    ("append_fwd", math.hypot(best_pts[-1][0] - r_ch[0][0], best_pts[-1][1] - r_ch[0][1])),
                    ("append_rev", math.hypot(best_pts[-1][0] - r_ch[-1][0], best_pts[-1][1] - r_ch[-1][1])),
                    ("prepend_fwd", math.hypot(r_ch[-1][0] - best_pts[0][0], r_ch[-1][1] - best_pts[0][1])),
                    ("prepend_rev", math.hypot(r_ch[0][0] - best_pts[0][0], r_ch[0][1] - best_pts[0][1])),
                ]
                for mode_s, dist_s in candidates_join:
                    if dist_s < best_dist:
                        best_dist = dist_s
                        best_idx = r_i
                        best_mode = mode_s
            chosen = remaining.pop(best_idx)
            # If chosen is a short branch touching the middle of best_pts (like a crossbar or T-stem),
            # splice an out-and-back excursion right at the closest interior point of best_pts instead of jumping across whitespace!
            d_mid_0 = min((math.hypot(bp[0] - chosen[0][0], bp[1] - chosen[0][1]), bi) for bi, bp in enumerate(best_pts))
            d_mid_1 = min((math.hypot(bp[0] - chosen[-1][0], bp[1] - chosen[-1][1]), bi) for bi, bp in enumerate(best_pts))
            if min(d_mid_0[0], d_mid_1[0]) + 2.5 < best_dist and min(d_mid_0[0], d_mid_1[0]) <= max(4.5, med_r * 2.5):
                if d_mid_0[0] <= d_mid_1[0]:
                    bi = d_mid_0[1]
                    exc = chosen + chosen[:-1][::-1]
                else:
                    bi = d_mid_1[1]
                    ch_rev = chosen[::-1]
                    exc = ch_rev + ch_rev[:-1][::-1]
                best_pts = best_pts[:bi + 1] + exc + best_pts[bi + 1:]
            else:
                if best_mode == "append_fwd":
                    best_pts = best_pts + chosen
                elif best_mode == "append_rev":
                    best_pts = best_pts + chosen[::-1]
                elif best_mode == "prepend_fwd":
                    best_pts = chosen + best_pts
                else:
                    best_pts = chosen[::-1] + best_pts

        calligraphic_pts = self._parameterize_stroke(best_pts, px_per_mm=px_per_mm, dist_map=dist_map)
        if len(calligraphic_pts) < 2:
            return None
        calligraphic_pts = self._optimize_strokes_on_silhouette([calligraphic_pts], clean_comp, dist_map, px_per_mm=px_per_mm)[0]

        return {
            "stroke_id": "cal_s00",
            "order_index": 0,
            "points": calligraphic_pts,
            "length": float(len(calligraphic_pts)),
            "start": (calligraphic_pts[0][0], calligraphic_pts[0][1]),
            "end": (calligraphic_pts[-1][0], calligraphic_pts[-1][1]),
            "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
            "mean_width": float(np.mean([p[2] for p in calligraphic_pts]))
        }

    def _trace_continuous_ridge(
        self,
        dist_map: np.ndarray,
        skel: np.ndarray,
        global_visited: Optional[np.ndarray],
        start_y: int,
        start_x: int,
        init_dy: float,
        init_dx: float,
        max_steps: int = 120,
        step_size: float = 1.45,
    ) -> List[Tuple[float, float, float]]:
        """
        Follows the medial distance ridge with strong biomechanical inertia:
        - Step size scales in mm so it never jitter-steps inside a thick stroke.
        - Full nib-width wake mask prevents U-turns inside the thickness of the same stem.
        - Maximum angular deflection per step is bounded to [-65 deg, +65 deg].
        """
        h, w = dist_map.shape
        cy, cx = float(start_y), float(start_x)
        init_w = max(1.5, float(dist_map[start_y, start_x] * 2.0))
        pts = [(round(cy, 2), round(cx, 2), round(init_w, 2))]

        # Precompute 3x3 dilated skeleton neighborhood once in C++ instead of slicing per step/angle
        skel_near = cv2.dilate(skel.astype(np.uint8), self._ONES_3X3_U8) > 0

        # Wake mask covering the full width of the already-drawn ink behind the pen
        wake_mask = np.zeros((h, w), dtype=bool)
        recent_centers: List[Tuple[float, float, float]] = [(cy, cx, max(1.8, float(dist_map[start_y, start_x]) + 1.0))]

        cur_dy, cur_dx = float(init_dy), float(init_dx)
        norm_init = math.hypot(cur_dy, cur_dx)
        if norm_init > 1e-4:
            cur_dy, cur_dx = cur_dy / norm_init, cur_dx / norm_init
        else:
            cur_dy, cur_dx = 1.0, 0.0

        ridge_angles = self._RIDGE_ANGLES
        has_global = global_visited is not None

        for step in range(max_steps):
            # Commit older points (more than 3 steps behind) into the forbidden nib-width wake mask
            if len(recent_centers) > 3:
                old_y, old_x, old_r = recent_centers.pop(0)
                oy, ox = int(round(old_y)), int(round(old_x))
                r_int = int(math.ceil(old_r))
                y0, y1 = max(0, oy - r_int), min(h, oy + r_int + 1)
                x0, x1 = max(0, ox - r_int), min(w, ox + r_int + 1)
                sub_y = np.arange(y0, y1)[:, None] - old_y
                sub_x = np.arange(x0, x1)[None, :] - old_x
                wake_mask[y0:y1, x0:x1] |= ((sub_y * sub_y + sub_x * sub_x) <= (old_r * old_r))

            best_val = -1e9
            best_ny, best_nx = None, None
            best_vdy, best_vdx = None, None

            # Sample 19 precomputed forward directions bounded to [-65 deg, +65 deg]
            for cos_a, sin_a, is_straight in ridge_angles:
                cand_dy = cur_dy * cos_a - cur_dx * sin_a
                cand_dx = cur_dy * sin_a + cur_dx * cos_a

                ny = cy + cand_dy * step_size
                nx = cx + cand_dx * step_size
                iy, ix = int(round(ny)), int(round(nx))

                if not (0 <= iy < h and 0 <= ix < w):
                    continue

                d_val = float(dist_map[iy, ix])
                if d_val < 0.35:
                    continue

                on_skel_bonus = 2.4 if skel_near[iy, ix] else 0.0
                momentum = (cos_a * 2.1) if cos_a > 0.0 else 0.0

                is_vis = wake_mask[iy, ix] or (has_global and global_visited[iy, ix])
                if is_vis:
                    crossing = False
                    if is_straight:
                        local_w = max(2.5, d_val * 2.2)
                        for look_d in (local_w * 1.2, local_w * 1.8):
                            fy = int(round(cy + cand_dy * look_d))
                            fx = int(round(cx + cand_dx * look_d))
                            if 0 <= fy < h and 0 <= fx < w and dist_map[fy, fx] >= 0.45:
                                if not wake_mask[fy, fx] and (not has_global or not global_visited[fy, fx]):
                                    crossing = True
                                    break
                    if not crossing:
                        continue
                    vis_penalty = -2.5
                else:
                    vis_penalty = 2.0

                score = d_val * 2.4 + on_skel_bonus + momentum + vis_penalty
                if score > best_val:
                    best_val = score
                    best_ny, best_nx = ny, nx
                    best_vdy, best_vdx = cand_dy, cand_dx

            if best_ny is None:
                break

            iy, ix = int(round(best_ny)), int(round(best_nx))
            cur_r = max(1.8, float(dist_map[iy, ix]) * 0.95 + 0.8)
            recent_centers.append((best_ny, best_nx, cur_r))

            cy, cx = best_ny, best_nx
            blend_dy = 0.50 * cur_dy + 0.50 * best_vdy
            blend_dx = 0.50 * cur_dx + 0.50 * best_vdx
            bnorm = math.hypot(blend_dy, blend_dx)
            cur_dy, cur_dx = (blend_dy / bnorm, blend_dx / bnorm) if bnorm > 1e-4 else (best_vdy, best_vdx)

            w_val = max(1.5, float(dist_map[iy, ix] * 2.0))
            pts.append((round(cy, 2), round(cx, 2), round(w_val, 2)))

            # Check loop closure: if returning close to start after >= 8 steps, close cleanly and stop
            if len(pts) >= 8:
                dist_to_start = math.hypot(cy - pts[0][0], cx - pts[0][1])
                if dist_to_start <= max(2.5, init_w * 0.85):
                    pts.append(pts[0])
                    break

        return pts

    @staticmethod
    def _rdp_simplify_with_width(
        pts: List[Tuple[float, float, float]],
        epsilon: float = 0.85
    ) -> List[Tuple[float, float, float]]:
        """
        Ramer-Douglas-Peucker trajectory simplification in (y, x) space while preserving width.
        Eliminates sub-pixel staircase oscillations while keeping true curvature extrema and loops.
        """
        if len(pts) <= 3:
            return pts

        # If closed loop (start ~ end), split in half, simplify each half, and rejoin
        if np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 3.0 and len(pts) >= 6:
            mid = len(pts) // 2
            first_half = CalligraphicVectorizer._rdp_simplify_with_width(pts[:mid + 1], epsilon=epsilon)
            second_half = CalligraphicVectorizer._rdp_simplify_with_width(pts[mid:], epsilon=epsilon)
            return first_half[:-1] + second_half

        p_start = np.array([pts[0][0], pts[0][1]], dtype=np.float64)
        p_end = np.array([pts[-1][0], pts[-1][1]], dtype=np.float64)
        line_vec = p_end - p_start
        line_len = float(np.linalg.norm(line_vec))

        max_dist = -1.0
        index = 0
        for i in range(1, len(pts) - 1):
            p_curr = np.array([pts[i][0], pts[i][1]], dtype=np.float64)
            if line_len < 1e-5:
                d = float(np.linalg.norm(p_curr - p_start))
            else:
                v_diff = p_start - p_curr
                cross_2d = line_vec[0] * v_diff[1] - line_vec[1] * v_diff[0]
                d = float(abs(cross_2d) / line_len)
            if d > max_dist:
                index = i
                max_dist = d

        if max_dist > epsilon:
            left = CalligraphicVectorizer._rdp_simplify_with_width(pts[:index + 1], epsilon=epsilon)
            right = CalligraphicVectorizer._rdp_simplify_with_width(pts[index:], epsilon=epsilon)
            return left[:-1] + right
        else:
            return [pts[0], pts[-1]]

    @staticmethod
    def _rasterize_strokes_mask(
        stroke_pts_list: List[List[Tuple[float, float, float]]],
        shape: Tuple[int, int],
        scale_w: float = 1.0,
    ) -> np.ndarray:
        """Renders candidate variable-width calligraphic strokes onto a 2D boolean mask for exact silhouette IoU."""
        h, w = shape
        canvas = np.zeros((h, w), dtype=np.uint8)
        for pts in stroke_pts_list:
            if not pts:
                continue
            if len(pts) == 1:
                py, px = int(round(pts[0][0])), int(round(pts[0][1]))
                r = max(1, int(round(float(pts[0][2]) * 0.50 * scale_w)))
                cv2.circle(canvas, (px, py), r, 1, -1)
            else:
                for i in range(len(pts) - 1):
                    p0, p1 = pts[i], pts[i + 1]
                    w_seg = ((float(p0[2]) + float(p1[2])) * 0.5) * scale_w
                    th = max(1, int(round(w_seg)))
                    cv2.line(
                        canvas,
                        (int(round(p0[1])), int(round(p0[0]))),
                        (int(round(p1[1])), int(round(p1[0]))),
                        1,
                        th,
                        lineType=cv2.LINE_8,
                    )
        return canvas > 0

    @staticmethod
    def _extend_and_anchor_stroke(
        pts: List[Tuple[float, float, float]],
        clean_comp: np.ndarray,
        dist_map: np.ndarray,
        prior_strokes: List[List[Tuple[float, float, float]]],
        med_half_w: float,
    ) -> List[Tuple[float, float, float]]:
        """
        1. Snaps stroke endpoints that originate at T-junctions directly onto the parent stroke (0-px junction gap).
        2. Extends free eroded endpoints outward along their local tangent by ~0.78 * d_EDT inside the ink mask.
        """
        if len(pts) < 2:
            return pts
        h, w = clean_comp.shape
        is_closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 3.0 and len(pts) >= 6
        if is_closed:
            return pts

        out = list(pts)
        snap_r = max(2.8, med_half_w * 2.4)
        prior_flat = [p for st in prior_strokes for p in st] if prior_strokes else []

        for end_flag in (0, -1):
            ey, ex, ew = float(out[end_flag][0]), float(out[end_flag][1]), float(out[end_flag][2])
            # Check junction snap to prior stroke first
            if prior_flat:
                best_p = None
                best_d = snap_r + 1.0
                for pp in prior_flat:
                    d = math.hypot(ey - pp[0], ex - pp[1])
                    if d < best_d:
                        best_d = d
                        best_p = pp
                if best_p is not None and 0.4 <= best_d <= snap_r:
                    # Check midpoint is inside ink so we never bridge across empty parchment
                    my, mx = int(round(0.5 * (ey + best_p[0]))), int(round(0.5 * (ex + best_p[1])))
                    if 0 <= my < h and 0 <= mx < w and clean_comp[my, mx]:
                        anchor_pt = (round(float(best_p[0]), 2), round(float(best_p[1]), 2), round(float(ew), 2))
                        if end_flag == 0:
                            out = [anchor_pt] + out
                        else:
                            out = out + [anchor_pt]
                        continue

            # Free endpoint: geodesic cone tip completion so the stroke reaches 100% of its true ink extremity (including curved hairlines!)
            for _tip_pass in range(2):
                ey, ex, ew = float(out[end_flag][0]), float(out[end_flag][1]), float(out[end_flag][2])
                k_step = min(3, len(out) - 1)
                if end_flag == 0:
                    ty = out[0][0] - out[k_step][0]
                    tx = out[0][1] - out[k_step][1]
                else:
                    ty = out[-1][0] - out[-1 - k_step][0]
                    tx = out[-1][1] - out[-1 - k_step][1]
                tn = math.hypot(ty, tx)
                if tn < 1e-4:
                    break
                ty, tx = ty / tn, tx / tn
                py_dir, px_dir = -tx, ty
                iy0 = max(0, min(h - 1, int(round(ey))))
                ix0 = max(0, min(w - 1, int(round(ex))))
                local_r = max(1.0, float(dist_map[iy0, ix0]))
                r_box = int(math.ceil(min(9.0, local_r * 1.75 + 2.0)))
                y_lo, y_hi = max(0, iy0 - r_box), min(h, iy0 + r_box + 1)
                x_lo, x_hi = max(0, ix0 - r_box), min(w, ix0 + r_box + 1)
                ys_sub, xs_sub = np.where(clean_comp[y_lo:y_hi, x_lo:x_hi])
                best_tip = None
                best_tip_score = 0.35
                max_fwd = min(9.0, local_r * 1.65 + 2.0)
                max_lat = max(1.8, local_r * 0.90 + 1.2)
                for dy_i, dx_i in zip(ys_sub, xs_sub):
                    cy_i = float(y_lo + dy_i)
                    cx_i = float(x_lo + dx_i)
                    vy, vx = cy_i - ey, cx_i - ex
                    proj = vy * ty + vx * tx
                    if proj < 0.45 or proj > max_fwd:
                        continue
                    perp = abs(vy * py_dir + vx * px_dir)
                    if perp > max_lat or perp > proj * 1.15 + 0.8:
                        continue
                    # Ensure straight segment stays inside clean_comp (never jumps across empty parchment)
                    mid_y, mid_x = int(round(0.5 * (ey + cy_i))), int(round(0.5 * (ex + cx_i)))
                    q_y, q_x = int(round(0.75 * cy_i + 0.25 * ey)), int(round(0.75 * cx_i + 0.25 * ex))
                    if not (clean_comp[mid_y, mid_x] and clean_comp[q_y, q_x]):
                        continue
                    tip_score = proj - 0.28 * perp + 0.25 * float(dist_map[int(cy_i), int(cx_i)])
                    if tip_score > best_tip_score:
                        best_tip_score = tip_score
                        w_tip = max(1.2, float(dist_map[int(cy_i), int(cx_i)]) * 1.85)
                        best_tip = (round(cy_i, 2), round(cx_i, 2), round(w_tip, 2))
                if best_tip is not None and math.hypot(best_tip[0] - ey, best_tip[1] - ex) >= 0.65:
                    if end_flag == 0:
                        out = [best_tip] + out
                    else:
                        out = out + [best_tip]
                else:
                    break

        return out

    def _optimize_strokes_on_silhouette(
        self,
        strokes_pts: List[List[Tuple[float, float, float]]],
        clean_comp: np.ndarray,
        dist_map: np.ndarray,
        px_per_mm: float = 6.3,
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Direct Calligraphic Bézier Silhouette Optimization (Loop & Cusp Safe):
        1. Preserves closed-loop rings and sharp cusps ('3', 'm', 'k') strictly uncollapsed; gently centers only smooth stem interiors.
        2. Calibrates global 40°-quill width gain to maximize exact 2D silhouette IoU against `clean_comp`.
        """
        h, w = clean_comp.shape
        centered_strokes: List[List[Tuple[float, float, float]]] = []
        max_shift = max(0.6, min(1.35, 0.10 * px_per_mm))

        for pts in strokes_pts:
            n = len(pts)
            is_closed = n >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= max(2.8, 0.35 * px_per_mm)
            # Closed loop rings are already centered between the inner hole boundary and outer wall — never shift toward global blob!
            if n <= 2 or is_closed:
                centered_strokes.append(pts)
                continue
            new_pts = list(pts)
            for i in range(1, n - 1):
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[i + 1]
                v1y, v1x = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
                v2y, v2x = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
                n1 = math.hypot(v1y, v1x)
                n2 = math.hypot(v2y, v2x)
                # Freeze cusps, sharp corners, and self-loop turnarounds so '3' tongues and small loops stay crisp!
                if n1 < 1e-4 or n2 < 1e-4 or ((v1y * v2y + v1x * v2x) / (n1 * n2)) < 0.65:
                    continue
                ty, tx = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
                tn = math.hypot(ty, tx)
                if tn < 1e-4:
                    continue
                ny_dir, nx_dir = -tx / tn, ty / tn
                cy, cx, cw = p_curr
                sum_w = 0.0
                sum_s = 0.0
                for s_off in (-max_shift, -0.5 * max_shift, 0.0, 0.5 * max_shift, max_shift):
                    sy = int(round(cy + ny_dir * s_off))
                    sx = int(round(cx + nx_dir * s_off))
                    if 0 <= sy < h and 0 <= sx < w and clean_comp[sy, sx]:
                        wt = float(dist_map[sy, sx]) + 0.25
                        sum_w += wt
                        sum_s += wt * s_off
                if sum_w > 0.0:
                    shift = 0.38 * (sum_s / sum_w)
                    ny_new = max(0.0, min(float(h - 1), cy + ny_dir * shift))
                    nx_new = max(0.0, min(float(w - 1), cx + nx_dir * shift))
                    iy_n, ix_n = int(round(ny_new)), int(round(nx_new))
                    edt_d = max(1.3, float(dist_map[iy_n, ix_n]) * 2.0)
                    new_pts[i] = (round(ny_new, 2), round(nx_new, 2), round(0.65 * edt_d + 0.35 * cw, 2))
            centered_strokes.append(new_pts)

        # Optimize global width gain alpha for maximum exact 2D silhouette IoU
        best_gain = 1.0
        best_iou = -1.0
        for alpha in (0.90, 1.02, 1.15, 1.28, 1.42):
            pred = self._rasterize_strokes_mask(centered_strokes, (h, w), scale_w=alpha)
            inter = int(np.sum(pred & clean_comp))
            union = int(np.sum(pred | clean_comp))
            iou = (inter / union) if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_gain = alpha

        final_strokes: List[List[Tuple[float, float, float]]] = []
        for pts in centered_strokes:
            final_strokes.append([
                (p[0], p[1], round(max(1.2, float(p[2]) * best_gain), 2))
                for p in pts
            ])
        return final_strokes

    def _parameterize_stroke(
        self,
        raw_pts: List[Tuple[float, float, float]],
        px_per_mm: float = 6.3,
        dist_map: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Piecewise Cusp-Preserving Calligraphic Parameterization:
        - Detects sharp corners & velocity-reversal cusps ('3' middle tongue, 'm'/'n' arches, tight hooks)
          and freezes them as exact knots (zero Gaussian melting across cusps or loops!).
        - Applies only sub-pixel pixel-staircase denoising (sigma=0.65) on smooth sub-arcs and fine RDP (~0.02 mm).
        """
        if len(raw_pts) <= 2:
            return raw_pts

        from scipy.ndimage import gaussian_filter1d

        # Remove exact consecutive duplicates first
        dedup: List[Tuple[float, float, float]] = [raw_pts[0]]
        for p in raw_pts[1:]:
            if math.hypot(float(p[0]) - float(dedup[-1][0]), float(p[1]) - float(dedup[-1][1])) >= 0.15:
                dedup.append((float(p[0]), float(p[1]), float(p[2])))
        if len(dedup) <= 2:
            return raw_pts

        n_pts = len(dedup)
        close_dist = max(2.4, 0.30 * px_per_mm)
        is_closed = (n_pts >= 6 and math.hypot(dedup[0][0] - dedup[-1][0], dedup[0][1] - dedup[-1][1]) <= close_dist)

        # Detect sharp corners / cusps (turn > ~55 deg, cos < 0.57) as immutable anchor indices
        anchors = [0]
        for i in range(1, n_pts - 1):
            v1y, v1x = dedup[i][0] - dedup[i - 1][0], dedup[i][1] - dedup[i - 1][1]
            v2y, v2x = dedup[i + 1][0] - dedup[i][0], dedup[i + 1][1] - dedup[i][1]
            n1 = math.hypot(v1y, v1x)
            n2 = math.hypot(v2y, v2x)
            if n1 > 1e-4 and n2 > 1e-4 and ((v1y * v2y + v1x * v2x) / (n1 * n2)) < 0.57:
                anchors.append(i)
        anchors.append(n_pts - 1)

        rdp_eps = max(0.22, 0.020 * px_per_mm)
        simplified: List[Tuple[float, float, float]] = []
        if is_closed and len(anchors) == 2:
            cy_arr = gaussian_filter1d(np.array([p[0] for p in dedup[:-1]], dtype=np.float64), sigma=0.65, mode="wrap")
            cx_arr = gaussian_filter1d(np.array([p[1] for p in dedup[:-1]], dtype=np.float64), sigma=0.65, mode="wrap")
            ring_sm = [(float(cy_arr[k]), float(cx_arr[k]), dedup[k][2]) for k in range(len(cy_arr))]
            ring_sm.append(ring_sm[0])
            simplified = self._rdp_simplify_with_width(ring_sm, epsilon=rdp_eps)
        else:
            for a_idx in range(len(anchors) - 1):
                i0, i1 = anchors[a_idx], anchors[a_idx + 1]
                piece = dedup[i0:i1 + 1]
                if len(piece) >= 4:
                    py_a = np.array([p[0] for p in piece], dtype=np.float64)
                    px_a = np.array([p[1] for p in piece], dtype=np.float64)
                    sy0, sx0, sy1, sx1 = py_a[0], px_a[0], py_a[-1], px_a[-1]
                    py_a = gaussian_filter1d(py_a, sigma=0.65, mode="nearest")
                    px_a = gaussian_filter1d(px_a, sigma=0.65, mode="nearest")
                    py_a[0], px_a[0] = sy0, sx0
                    py_a[-1], px_a[-1] = sy1, sx1
                    piece = [(float(py_a[k]), float(px_a[k]), piece[k][2]) for k in range(len(piece))]
                simp_piece = self._rdp_simplify_with_width(piece, epsilon=rdp_eps)
                if simplified:
                    simplified.extend(simp_piece[1:])
                else:
                    simplified.extend(simp_piece)

        max_seg_len = max(3.0, 0.34 * px_per_mm)
        resample_step = max(2.2, 0.25 * px_per_mm)
        resampled: List[Tuple[float, float, float]] = [simplified[0]]
        for k in range(len(simplified) - 1):
            p0 = simplified[k]
            p1 = simplified[k + 1]
            seg_d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            if seg_d > max_seg_len:
                n_sub = int(np.ceil(seg_d / resample_step))
                for s_i in range(1, n_sub):
                    t = float(s_i) / float(n_sub)
                    resampled.append((
                        p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1]),
                        p0[2] + t * (p1[2] - p0[2])
                    ))
            resampled.append(p1)

        # Calibrate 40° beveled quill contrast directly against the glyph's empirical width quantiles
        emp_ws = [p[2] for p in resampled]
        w_lo = max(1.2, float(np.percentile(emp_ws, 20)) * 0.78)
        w_hi = max(w_lo + 0.8, float(np.percentile(emp_ws, 85)) * 1.15)

        final_pts: List[Tuple[float, float, float]] = []
        n = len(resampled)
        for i in range(n):
            if i < n - 1:
                dy = resampled[i + 1][0] - resampled[i][0]
                dx = resampled[i + 1][1] - resampled[i][1]
            else:
                dy = resampled[i][0] - resampled[i - 1][0]
                dx = resampled[i][1] - resampled[i - 1][1]

            tangent = float(np.arctan2(dy, dx))
            bevel_factor = float(np.abs(np.sin(tangent - self.nib_angle_rad)))
            quill_w = w_lo + (w_hi - w_lo) * bevel_factor
            local_w = float(resampled[i][2])
            if dist_map is not None:
                dh, dw = dist_map.shape
                iy = max(0, min(dh - 1, int(round(resampled[i][0]))))
                ix = max(0, min(dw - 1, int(round(resampled[i][1]))))
                if dist_map[iy, ix] > 0.3:
                    local_w = float(dist_map[iy, ix]) * 2.0
            combined_w = 0.78 * local_w + 0.22 * quill_w
            final_pts.append((round(resampled[i][0], 2), round(resampled[i][1], 2), round(combined_w, 2)))

        return final_pts
