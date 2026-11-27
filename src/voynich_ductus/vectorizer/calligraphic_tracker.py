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

            # Only prune if it actually terminates at a junction within the spur threshold,
            # leaves at least 4 skeleton pixels intact, AND never disconnects a loop/junction
            if hit_junction and len(chain) <= max_walk and (rem_count - len(chain)) >= 4:
                jn_y, jn_x = ny, nx
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

        comp_min_area = max(12, int(round(0.08 * (px_per_mm ** 2)))) if scale_calib is not None else 12
        # Keep genuine closed-loop holes ('o', 'a', 'd', 'e' ~10-40 px^2) intact; fill only grain pinholes
        pinhole_max_area = max(2, int(round(0.025 * (px_per_mm ** 2))))
        spur_min_len = max(3, int(round(max(med_half_width * 1.4, 0.40 * px_per_mm))))
        sigma_eff = max(float(self.smoothing_sigma), 0.14 * px_per_mm)
        step_size_eff = max(1.15, min(1.55, 0.13 * px_per_mm))
        extra_margin_eff = max(1.3, 0.14 * px_per_mm)
        min_rem_pixels = max(7, int(round(max(est_nib_width * 1.8, 0.85 * px_per_mm))))

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
        min_res_area = max(8, int(round(0.065 * (px_per_mm ** 2))))

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

            # 2. Extract crossing-preserving topological strokes on remaining skeleton (0 junction gaps!)
            rem_budget = stroke_limit - (len(strokes) + len(comp_raw_strokes))
            rem_skel = skel & (~loop_covered_skel)
            min_branch_pts = max(3, int(round(0.32 * px_per_mm)))
            if rem_budget > 0 and int(np.sum(rem_skel)) >= min_branch_pts:
                p_graph = graph_ext.build_pixel_graph(rem_skel, dist_map * 2.0)
                raw_segs = graph_ext.decompose_into_strokes(p_graph)
                if raw_segs and len(raw_segs) > 1:
                    non_micro = [s for s in raw_segs if len(s.get("points", [])) >= min_branch_pts]
                    if non_micro:
                        raw_segs = non_micro
                if raw_segs:
                    chained = junc_res.resolve_and_order_strokes(
                        raw_segs,
                        max_gap=max(5.5, 0.75 * px_per_mm),
                        min_cos=-0.28,
                        max_strokes=rem_budget,
                    )
                    for ch_s in chained[:rem_budget]:
                        c_pts = [(float(p[0]), float(p[1]), max(1.6, float(p[2]))) for p in ch_s.get("points", [])]
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

            # 3. Analysis-by-Synthesis Residual Pass: recover any uncovered crossbars/shoulders/plumes
            while (len(strokes) + len(comp_raw_strokes)) < stroke_limit and comp_raw_strokes:
                pred_m = self._rasterize_strokes_mask(comp_raw_strokes, (h, w), scale_w=1.05)
                uncov = clean_comp & (~(cv2.dilate(pred_m.astype(np.uint8), self._ONES_3X3_U8) > 0))
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
                    break
                u_graph = graph_ext.build_pixel_graph(u_skel, dist_map * 2.0)
                u_segs = graph_ext.decompose_into_strokes(u_graph)
                if not u_segs:
                    break
                u_chained = junc_res.resolve_and_order_strokes(u_segs, max_gap=max(5.0, 0.7 * px_per_mm), min_cos=-0.35, max_strokes=1)
                if not u_chained:
                    break
                u_pts = [(float(p[0]), float(p[1]), max(1.6, float(p[2]))) for p in u_chained[0].get("points", [])]
                if len(u_pts) < 2:
                    break
                u_pts = self._extend_and_anchor_stroke(u_pts, clean_comp, dist_map, comp_raw_strokes, med_half_width)
                comp_raw_strokes.append(u_pts)
                # Guarantee termination if residual does not shrink
                new_pred = self._rasterize_strokes_mask(comp_raw_strokes, (h, w), scale_w=1.05)
                if int(np.sum(clean_comp & (~new_pred))) >= int(np.sum(clean_comp & (~pred_m))) - 2:
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

    def _extract_dominant_single_stroke(
        self,
        binary_mask: np.ndarray,
        px_per_mm: float = 6.3
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts exactly one continuous calligraphic gesture covering the glyph (Codex Seraphinianus mode).
        Prunes contour roughness spurs and uses full nib-footprint extinction so the single ribbon
        never bounces back and forth inside the thickness of its own stroke.
        """
        from scipy.ndimage import label, binary_closing
        if binary_mask is None or not np.any(binary_mask):
            return None

        closed_mask = binary_closing(binary_mask.astype(bool), structure=np.ones((3, 3), dtype=bool))
        lbl, num_features = label(closed_mask)
        if num_features == 0:
            return None

        cc_sizes = [(comp_id, int(np.sum(lbl == comp_id))) for comp_id in range(1, num_features + 1)]
        dominant_id = max(cc_sizes, key=lambda x: x[1])[0]
        comp_mask = (lbl == dominant_id)
        h, w = comp_mask.shape

        dist_map = distance_transform_edt(comp_mask)
        raw_skel = skeletonize(comp_mask)
        if not np.any(raw_skel):
            return None

        inner_dists = dist_map[dist_map > 0.5]
        med_r = float(np.median(inner_dists)) if len(inner_dists) > 0 else 1.5
        spur_min_len = max(3, int(round(max(med_r * 1.5, 0.45 * px_per_mm))))
        step_size_eff = max(1.45, 0.22 * px_per_mm)
        extra_margin_eff = max(1.6, 0.16 * px_per_mm)
        min_rem_branch = max(6, int(round(0.65 * px_per_mm)))
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_len)

        smoothed_dist = gaussian_filter(dist_map.astype(np.float32), sigma=max(self.smoothing_sigma, 0.14 * px_per_mm))
        candidates = self.find_touchdown_endpoints(comp_mask, skel)
        if not candidates:
            return None

        best_pts = []
        best_score = -1e9
        total_skel_pts = max(1, int(np.sum(skel)))

        for cand in candidates:
            prio, sy, sx, dy, dx = cand
            fwd_pts = self._trace_continuous_ridge(
                smoothed_dist, skel, None, sy, sx, dy, dx,
                max_steps=int(max(h, w) * 3.5 / max(1.0, step_size_eff * 0.6)),
                step_size=step_size_eff,
            )
            if len(fwd_pts) < 2:
                continue

            vis_mask = np.zeros_like(skel, dtype=bool)
            self._mark_nib_footprint_visited(fwd_pts, dist_map, vis_mask, extra_margin=extra_margin_eff)

            bwd_pts = self._trace_continuous_ridge(
                smoothed_dist, skel, vis_mask.copy(), sy, sx, -dy, -dx,
                max_steps=int(max(h, w) * 2.0 / max(1.0, step_size_eff * 0.6)),
                step_size=step_size_eff,
            )
            if len(bwd_pts) >= 3:
                sim_pts = bwd_pts[:0:-1] + fwd_pts
                self._mark_nib_footprint_visited(bwd_pts, dist_map, vis_mask, extra_margin=extra_margin_eff)
            else:
                sim_pts = fwd_pts

            skel_cov = float(np.sum(skel & vis_mask)) / float(total_skel_pts)
            ys = [p[0] for p in sim_pts]
            xs = [p[1] for p in sim_pts]
            y_span = (max(ys) - min(ys)) if ys else 0
            x_span = (max(xs) - min(xs)) if xs else 0
            cand_score = skel_cov * 150.0 + (y_span / float(max(1, h))) * 45.0 + (x_span / float(max(1, w))) * 35.0 + prio * 1.5
            if cand_score > best_score:
                best_score = cand_score
                best_pts = sim_pts

        if not best_pts or len(best_pts) < 2:
            return None

        # Controlled continuation pass ONLY for major unvisited branches
        vis_mask = np.zeros_like(skel, dtype=bool)
        self._mark_nib_footprint_visited(best_pts, dist_map, vis_mask, extra_margin=extra_margin_eff * 1.1)

        for _ext_pass in range(3):
            rem_skel = skel & (~vis_mask)
            if int(np.sum(rem_skel)) < min_rem_branch:
                break
            lbl_rem, n_rem = label(rem_skel, structure=np.ones((3, 3), dtype=bool))
            if n_rem == 0:
                break
            branch_sizes = [(bid, int(np.sum(lbl_rem == bid))) for bid in range(1, n_rem + 1)]
            best_bid, best_bsize = max(branch_sizes, key=lambda x: x[1])
            if best_bsize < min_rem_branch:
                break

            rem_coords = np.argwhere(lbl_rem == best_bid)
            traj_arr = np.array([[p[0], p[1]] for p in best_pts], dtype=np.float32)

            d_end = np.linalg.norm(rem_coords - traj_arr[-1], axis=1)
            d_start = np.linalg.norm(rem_coords - traj_arr[0], axis=1)
            idx_end = int(np.argmin(d_end))
            idx_start = int(np.argmin(d_start))

            if float(d_end[idx_end]) <= float(d_start[idx_start]):
                uy, ux = int(rem_coords[idx_end][0]), int(rem_coords[idx_end][1])
                attach_mode = "append"
                ref_y, ref_x = best_pts[-1][0], best_pts[-1][1]
            else:
                uy, ux = int(rem_coords[idx_start][0]), int(rem_coords[idx_start][1])
                attach_mode = "prepend"
                ref_y, ref_x = best_pts[0][0], best_pts[0][1]

            nb_rem = [p for p in rem_coords if 1.0 <= np.hypot(p[0] - uy, p[1] - ux) <= max(5.0, 0.7 * px_per_mm)]
            if nb_rem:
                mean_nb = np.mean(nb_rem, axis=0)
                udy, udx = float(mean_nb[0] - uy), float(mean_nb[1] - ux)
            else:
                udy, udx = float(uy - ref_y), float(ux - ref_x)
            unorm = float(np.hypot(udy, udx))
            udy, udx = (udy / unorm, udx / unorm) if unorm > 1e-4 else (1.0, 0.0)

            ext_pts = self._trace_continuous_ridge(
                smoothed_dist, skel, vis_mask.copy(), uy, ux, udy, udx,
                max_steps=int(max(h, w) * 1.5 / max(1.0, step_size_eff * 0.6)),
                step_size=step_size_eff,
            )
            self._mark_nib_footprint_visited(ext_pts, dist_map, vis_mask, extra_margin=extra_margin_eff * 1.1)
            vis_mask |= (lbl_rem == best_bid)
            if len(ext_pts) < 3:
                continue

            if attach_mode == "append":
                best_pts = best_pts + ext_pts
            else:
                best_pts = ext_pts[::-1] + best_pts

        best_pts = self._extend_and_anchor_stroke(best_pts, comp_mask, dist_map, [], med_r)
        p_start = best_pts[0]
        p_end = best_pts[-1]
        if math.hypot(p_start[0] - p_end[0], p_start[1] - p_end[1]) > max(3.5, 0.35 * px_per_mm) and p_start[0] > p_end[0] + 2.0:
            best_pts = best_pts[::-1]

        calligraphic_pts = self._parameterize_stroke(best_pts, px_per_mm=px_per_mm, dist_map=dist_map)
        if len(calligraphic_pts) < 2:
            return None
        calligraphic_pts = self._optimize_strokes_on_silhouette([calligraphic_pts], comp_mask, dist_map, px_per_mm=px_per_mm)[0]

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

            # Free endpoint: recover skeleton tip erosion along outward tangent inside clean_comp
            k_step = min(3, len(out) - 1)
            if end_flag == 0:
                ty = out[0][0] - out[k_step][0]
                tx = out[0][1] - out[k_step][1]
            else:
                ty = out[-1][0] - out[-1 - k_step][0]
                tx = out[-1][1] - out[-1 - k_step][1]
            tn = math.hypot(ty, tx)
            if tn < 1e-4:
                continue
            ty, tx = ty / tn, tx / tn
            iy0 = max(0, min(h - 1, int(round(ey))))
            ix0 = max(0, min(w - 1, int(round(ex))))
            local_r = max(1.0, float(dist_map[iy0, ix0]))
            max_ext = min(5.0, local_r * 0.78)
            best_ext = None
            for d_step in np.linspace(0.7, max_ext, 5):
                ny, nx = ey + ty * float(d_step), ex + tx * float(d_step)
                iny, inx = int(round(ny)), int(round(nx))
                if 0 <= iny < h and 0 <= inx < w and clean_comp[iny, inx] and dist_map[iny, inx] >= 0.45:
                    best_ext = (round(ny, 2), round(nx, 2), round(max(1.4, float(dist_map[iny, inx]) * 1.8), 2))
                else:
                    break
            if best_ext is not None:
                if end_flag == 0:
                    out = [best_ext] + out
                else:
                    out = out + [best_ext]

        return out

    def _optimize_strokes_on_silhouette(
        self,
        strokes_pts: List[List[Tuple[float, float, float]]],
        clean_comp: np.ndarray,
        dist_map: np.ndarray,
        px_per_mm: float = 6.3,
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Direct Calligraphic Bézier Silhouette Optimization:
        1. Nudges interior control knots along their local normal toward the perpendicular ink ridge inside `clean_comp`.
        2. Calibrates global 40°-quill width gain to maximize exact 2D silhouette IoU against `clean_comp`.
        """
        h, w = clean_comp.shape
        centered_strokes: List[List[Tuple[float, float, float]]] = []
        max_shift = max(1.2, min(3.0, 0.22 * px_per_mm))

        for pts in strokes_pts:
            n = len(pts)
            if n <= 2:
                centered_strokes.append(pts)
                continue
            is_closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= max(2.8, 0.35 * px_per_mm) and n >= 6
            new_pts = list(pts)
            i_start = 0 if is_closed else 1
            i_end = (n - 1) if is_closed else (n - 1)
            for i in range(i_start, i_end):
                p_prev = pts[(i - 1) % (n - 1)] if is_closed else pts[i - 1]
                p_next = pts[(i + 1) % (n - 1)] if is_closed else pts[i + 1]
                ty, tx = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
                tn = math.hypot(ty, tx)
                if tn < 1e-4:
                    continue
                ny_dir, nx_dir = -tx / tn, ty / tn
                cy, cx, cw = pts[i]
                # Weighted 1D center of EDT mass along normal within [-max_shift, +max_shift]
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
                    shift = 0.55 * (sum_s / sum_w)
                    ny_new = max(0.0, min(float(h - 1), cy + ny_dir * shift))
                    nx_new = max(0.0, min(float(w - 1), cx + nx_dir * shift))
                    iy_n, ix_n = int(round(ny_new)), int(round(nx_new))
                    edt_d = max(1.4, float(dist_map[iy_n, ix_n]) * 2.0)
                    new_pts[i] = (round(ny_new, 2), round(nx_new, 2), round(0.65 * edt_d + 0.35 * cw, 2))
            if is_closed:
                new_pts[-1] = new_pts[0]
            centered_strokes.append(new_pts)

        # Optimize global width gain alpha for maximum exact 2D silhouette IoU
        best_gain = 1.0
        best_iou = -1.0
        for alpha in (0.92, 1.05, 1.18, 1.32, 1.48):
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
                (p[0], p[1], round(max(1.3, float(p[2]) * best_gain), 2))
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
        Applies metric-calibrated biomechanical Gaussian trajectory smoothing,
        simplifies micro-oscillations with Douglas-Peucker (in mm), and computes
        40° beveled quill nib width anchored to the glyph's empirical EDT thickness.
        """
        if len(raw_pts) <= 2:
            return raw_pts

        from scipy.ndimage import gaussian_filter1d

        coords_y = np.array([float(p[0]) for p in raw_pts], dtype=np.float64)
        coords_x = np.array([float(p[1]) for p in raw_pts], dtype=np.float64)
        widths = np.array([float(p[2]) for p in raw_pts], dtype=np.float64)

        close_dist = max(2.8, 0.35 * px_per_mm)
        is_closed = (len(raw_pts) >= 6 and np.hypot(coords_y[0] - coords_y[-1], coords_x[0] - coords_x[-1]) <= close_dist)

        step_ds = float(np.median(np.hypot(np.diff(coords_y), np.diff(coords_x)))) if len(raw_pts) >= 3 else 1.5
        step_ds = max(0.8, step_ds)
        smooth_sigma_pts = max(1.0, min(3.8, (0.20 * px_per_mm) / step_ds))

        if len(coords_y) >= 4:
            if is_closed:
                coords_y = gaussian_filter1d(coords_y[:-1], sigma=smooth_sigma_pts, mode="wrap")
                coords_x = gaussian_filter1d(coords_x[:-1], sigma=smooth_sigma_pts, mode="wrap")
                coords_y = np.append(coords_y, coords_y[0])
                coords_x = np.append(coords_x, coords_x[0])
            else:
                sy0, sx0 = coords_y[0], coords_x[0]
                sy1, sx1 = coords_y[-1], coords_x[-1]
                coords_y = gaussian_filter1d(coords_y, sigma=smooth_sigma_pts, mode="nearest")
                coords_x = gaussian_filter1d(coords_x, sigma=smooth_sigma_pts, mode="nearest")
                coords_y[0], coords_x[0] = sy0, sx0
                coords_y[-1], coords_x[-1] = sy1, sx1

        pre_smoothed = [
            (float(coords_y[i]), float(coords_x[i]), float(widths[i]))
            for i in range(len(raw_pts))
        ]

        rdp_eps = max(0.42, 0.050 * px_per_mm)
        simplified = self._rdp_simplify_with_width(pre_smoothed, epsilon=rdp_eps)

        max_seg_len = max(3.6, 0.42 * px_per_mm)
        resample_step = max(2.8, 0.32 * px_per_mm)
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
