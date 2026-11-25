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

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import medial_axis


class CalligraphicVectorizer:
    """
    Simulates right-handed 15th-century quill pen physics to derender 2D ink raster
    directly into ordered (x, y, t, width) kinematic trajectories without graph branching
    or Brownian bouncing artifacts.
    """

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
            deg = np.zeros_like(s, dtype=np.int32)
            pts = np.argwhere(s)
            for y, x in pts:
                y0, y1 = max(0, y - 1), min(h, y + 2)
                x0, x1 = max(0, x - 1), min(w, x + 2)
                deg[y, x] = int(np.sum(s[y0:y1, x0:x1])) - 1
            return deg

        deg_map = get_degree_map(pruned)
        junctions = np.argwhere(deg_map >= 3)
        if len(junctions) == 0:
            # Pure unbranched curve or closed loop: no lateral spurs to prune
            return pruned

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

            # Only prune if it actually terminates at a junction within the spur threshold
            # and leaves at least 4 skeleton pixels intact
            if hit_junction and len(chain) <= max_walk and (np.sum(pruned) - len(chain)) >= 4:
                for py, px in chain:
                    pruned[py, px] = False
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
        skel_pts = np.argwhere(skel)
        if len(skel_pts) == 0:
            return []

        endpoints = []
        for y, x in skel_pts:
            y_min, y_max = max(0, y - 1), min(h, y + 2)
            x_min, x_max = max(0, x - 1), min(w, x + 2)
            deg = int(np.sum(skel[y_min:y_max, x_min:x_max])) - 1

            if deg == 1:
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
                n_len = float(np.hypot(init_dy, init_dx))
                if n_len > 1e-4:
                    endpoints.append((priority, int(y), int(x), init_dy / n_len, init_dx / n_len))
                else:
                    endpoints.append((priority, int(y), int(x), 1.0, 0.2))

        # If no degree 1 endpoints found (pure closed loop 'o' or '0'), pick top-left crest
        if not endpoints:
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
            single_stroke = self._extract_dominant_single_stroke(binary_mask)
            return [single_stroke] if single_stroke else []

        from scipy.ndimage import label, binary_fill_holes
        labeled_mask, num_features = label(binary_mask)
        if num_features == 0:
            return []

        components = []
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled_mask == comp_id)
            comp_area = int(np.sum(comp_mask))
            if comp_area < 12:
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

        components.sort(key=lambda c: (c["min_y"] // 15, c["min_x"]))

        strokes = []
        stroke_idx = 0
        stroke_limit = min(4, max_strokes) if max_strokes is not None else 4

        for comp in components:
            if len(strokes) >= stroke_limit:
                break
            comp_mask = comp["mask"]
            h, w = comp_mask.shape

            # Fill tiny 1-2px internal pinholes caused by parchment grain so medial axis doesn't split around dust
            padded = np.pad(comp_mask, 2, mode="constant", constant_values=False)
            filled = binary_fill_holes(padded)
            holes = filled & (~padded)
            lbl_h, num_h = label(holes)
            clean_comp = comp_mask.copy()
            for hid in range(1, num_h + 1):
                h_mask = (lbl_h == hid)[2:-2, 2:-2]
                if int(np.sum(h_mask)) <= 5:
                    clean_comp |= h_mask

            raw_skel, dist_map = medial_axis(clean_comp, return_distance=True)
            skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=max(3, int(round(med_half_width * 1.4))))
            smoothed_dist = gaussian_filter(dist_map.astype(np.float32), sigma=self.smoothing_sigma)

            candidates = self.find_touchdown_endpoints(clean_comp, skel)
            if not candidates:
                continue

            comp_visited = np.zeros_like(clean_comp, dtype=bool)
            total_skel = max(1, int(np.sum(skel)))

            best_raw_pts = []
            best_score = -1e9

            for cand in candidates:
                prio, sy, sx, dy, dx = cand
                sim_pts = self._trace_continuous_ridge(
                    smoothed_dist, skel, None, sy, sx, dy, dx,
                    max_steps=int(max(h, w) * 2.8)
                )
                if len(sim_pts) < 2:
                    continue
                sim_vis = np.zeros_like(clean_comp, dtype=bool)
                self._mark_nib_footprint_visited(sim_pts, dist_map, sim_vis, extra_margin=1.6)
                skel_cov = float(np.sum(skel & sim_vis)) / float(total_skel)
                ys = [p[0] for p in sim_pts]
                xs = [p[1] for p in sim_pts]
                span = np.hypot(max(ys) - min(ys), max(xs) - min(xs))
                # Reward high skeleton coverage with compact, non-redundant stroke length
                efficiency = span / max(1.0, float(len(sim_pts)))
                cand_score = skel_cov * 100.0 + (span / max(1.0, float(max(h, w)))) * 35.0 + efficiency * 15.0 + prio * 1.5
                if cand_score > best_score:
                    best_score = cand_score
                    best_raw_pts = sim_pts

            if not best_raw_pts or len(best_raw_pts) < 2:
                continue

            # Orient top-to-bottom / left-to-right
            if (best_raw_pts[0][0] * 1.5 + best_raw_pts[0][1]) > (best_raw_pts[-1][0] * 1.5 + best_raw_pts[-1][1]):
                best_raw_pts = best_raw_pts[::-1]

            self._mark_nib_footprint_visited(best_raw_pts, dist_map, comp_visited, extra_margin=1.8)

            calligraphic_pts = self._parameterize_stroke(best_raw_pts)
            if len(calligraphic_pts) >= 2:
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

            # Extract genuine secondary strokes (crossbars, second gallows stem, detached bowl)
            # Only if a substantial contiguous unvisited skeleton branch (>= max(7 px, 18% of skeleton)) remains
            remaining_skel = skel & (~comp_visited)
            min_rem_pixels = max(6, int(round(est_nib_width * 1.6)))
            while (
                int(np.sum(remaining_skel)) >= min_rem_pixels
                and (float(np.sum(remaining_skel)) / float(total_skel)) >= 0.16
                and len(strokes) < stroke_limit
            ):
                # Isolate largest connected unvisited skeleton branch
                lbl_rem, n_rem = label(remaining_skel, structure=np.ones((3, 3), dtype=bool))
                if n_rem == 0:
                    break
                branch_sizes = [(bid, int(np.sum(lbl_rem == bid))) for bid in range(1, n_rem + 1)]
                best_bid, best_bsize = max(branch_sizes, key=lambda x: x[1])
                if best_bsize < min_rem_pixels:
                    break

                branch_skel = (lbl_rem == best_bid)
                branch_cands = self.find_touchdown_endpoints(clean_comp, branch_skel)
                if not branch_cands:
                    break
                _, sy, sx, dy, dx = branch_cands[0]

                raw_pts = self._trace_continuous_ridge(
                    smoothed_dist, branch_skel, comp_visited, sy, sx, dy, dx,
                    max_steps=int(max(h, w) * 2.0)
                )
                self._mark_nib_footprint_visited(raw_pts, dist_map, comp_visited, extra_margin=1.8)
                comp_visited |= branch_skel

                if len(raw_pts) >= 3:
                    calligraphic_pts = self._parameterize_stroke(raw_pts)
                    if len(calligraphic_pts) >= 2:
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

                remaining_skel = skel & (~comp_visited)

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

    def _extract_dominant_single_stroke(self, binary_mask: np.ndarray) -> Optional[Dict[str, Any]]:
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

        raw_skel, dist_map = medial_axis(comp_mask, return_distance=True)
        if not np.any(raw_skel):
            return None

        inner_dists = dist_map[dist_map > 0.5]
        med_r = float(np.median(inner_dists)) if len(inner_dists) > 0 else 1.5
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=max(3, int(round(med_r * 1.4))))

        smoothed_dist = gaussian_filter(dist_map.astype(np.float32), sigma=self.smoothing_sigma)
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
                max_steps=int(max(h, w) * 3.5)
            )
            if len(fwd_pts) < 2:
                continue

            vis_mask = np.zeros_like(skel, dtype=bool)
            self._mark_nib_footprint_visited(fwd_pts, dist_map, vis_mask, extra_margin=1.6)

            bwd_pts = self._trace_continuous_ridge(
                smoothed_dist, skel, vis_mask.copy(), sy, sx, -dy, -dx,
                max_steps=int(max(h, w) * 2.0)
            )
            if len(bwd_pts) >= 3:
                sim_pts = bwd_pts[:0:-1] + fwd_pts
                self._mark_nib_footprint_visited(bwd_pts, dist_map, vis_mask, extra_margin=1.6)
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

        # Controlled continuation pass ONLY for major unvisited branches (>= 6 px)
        vis_mask = np.zeros_like(skel, dtype=bool)
        self._mark_nib_footprint_visited(best_pts, dist_map, vis_mask, extra_margin=1.8)

        for _ext_pass in range(3):
            rem_skel = skel & (~vis_mask)
            if int(np.sum(rem_skel)) < 6:
                break
            lbl_rem, n_rem = label(rem_skel, structure=np.ones((3, 3), dtype=bool))
            if n_rem == 0:
                break
            branch_sizes = [(bid, int(np.sum(lbl_rem == bid))) for bid in range(1, n_rem + 1)]
            best_bid, best_bsize = max(branch_sizes, key=lambda x: x[1])
            if best_bsize < 6:
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

            nb_rem = [p for p in rem_coords if 1.0 <= np.hypot(p[0] - uy, p[1] - ux) <= 5.0]
            if nb_rem:
                mean_nb = np.mean(nb_rem, axis=0)
                udy, udx = float(mean_nb[0] - uy), float(mean_nb[1] - ux)
            else:
                udy, udx = float(uy - ref_y), float(ux - ref_x)
            unorm = float(np.hypot(udy, udx))
            udy, udx = (udy / unorm, udx / unorm) if unorm > 1e-4 else (1.0, 0.0)

            ext_pts = self._trace_continuous_ridge(
                smoothed_dist, skel, vis_mask.copy(), uy, ux, udy, udx,
                max_steps=int(max(h, w) * 1.5)
            )
            self._mark_nib_footprint_visited(ext_pts, dist_map, vis_mask, extra_margin=1.8)
            vis_mask |= (lbl_rem == best_bid)
            if len(ext_pts) < 3:
                continue

            if attach_mode == "append":
                best_pts = best_pts + ext_pts
            else:
                best_pts = ext_pts[::-1] + best_pts

        p_start = best_pts[0]
        p_end = best_pts[-1]
        if p_start[0] > p_end[0] + 2.0:
            best_pts = best_pts[::-1]

        calligraphic_pts = self._parameterize_stroke(best_pts)
        if len(calligraphic_pts) < 2:
            return None

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
        max_steps: int = 120
    ) -> List[Tuple[float, float, float]]:
        """
        Follows the medial distance ridge with strong biomechanical inertia:
        - Step size adapts to local nib radius so it never jitter-steps inside a thick stroke.
        - Full nib-width wake mask prevents U-turns inside the thickness of the same stem.
        - Maximum angular deflection per step is bounded to [-50 deg, +50 deg].
        """
        h, w = dist_map.shape
        cy, cx = float(start_y), float(start_x)
        init_w = max(1.5, float(dist_map[start_y, start_x] * 2.0))
        pts = [(round(cy, 2), round(cx, 2), round(init_w, 2))]

        # Wake mask covering the full width of the already-drawn ink behind the pen
        wake_mask = np.zeros((h, w), dtype=bool)
        recent_centers: List[Tuple[float, float, float]] = [(cy, cx, max(1.8, float(dist_map[start_y, start_x]) + 1.0))]

        cur_dy, cur_dx = float(init_dy), float(init_dx)
        norm_init = float(np.hypot(cur_dy, cur_dx))
        if norm_init > 1e-4:
            cur_dy, cur_dx = cur_dy / norm_init, cur_dx / norm_init
        else:
            cur_dy, cur_dx = 1.0, 0.0

        step_size = 1.45

        for step in range(max_steps):
            # Commit older points (more than 3 steps behind) into the forbidden nib-width wake mask
            if len(recent_centers) > 3:
                old_y, old_x, old_r = recent_centers.pop(0)
                oy, ox = int(round(old_y)), int(round(old_x))
                r_int = int(np.ceil(old_r))
                y0, y1 = max(0, oy - r_int), min(h, oy + r_int + 1)
                x0, x1 = max(0, ox - r_int), min(w, ox + r_int + 1)
                sub_y = np.arange(y0, y1)[:, None] - old_y
                sub_x = np.arange(x0, x1)[None, :] - old_x
                wake_mask[y0:y1, x0:x1] |= ((sub_y * sub_y + sub_x * sub_x) <= (old_r * old_r))

            best_val = -1e9
            best_ny, best_nx = None, None
            best_vdy, best_vdx = None, None

            # Sample 15 forward directions strictly bounded to [-52 deg, +52 deg]
            for angle in np.linspace(-np.deg2rad(52), np.deg2rad(52), 15):
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
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

                # Check if pixel is near the pruned skeleton ridge
                y0_s, y1_s = max(0, iy - 1), min(h, iy + 2)
                x0_s, x1_s = max(0, ix - 1), min(w, ix + 2)
                on_skel_bonus = 2.2 if np.any(skel[y0_s:y1_s, x0_s:x1_s]) else 0.0

                # Strong Euler-Bernoulli inertia: penalize unnecessary turning
                momentum = (cos_a ** 2) * 3.5

                is_vis = wake_mask[iy, ix] or (global_visited is not None and global_visited[iy, ix])
                if is_vis:
                    # Only allow crossing an already-drawn stroke if we are moving nearly straight (|angle| <= 18 deg)
                    # AND there is genuine unvisited skeleton on the far side (distance 1.5*W_nib to 2.5*W_nib ahead)
                    crossing = False
                    if abs(angle) <= np.deg2rad(18):
                        local_w = max(2.5, d_val * 2.2)
                        for look_d in [local_w * 1.2, local_w * 1.8]:
                            fy = int(round(cy + cand_dy * look_d))
                            fx = int(round(cx + cand_dx * look_d))
                            if 0 <= fy < h and 0 <= fx < w and dist_map[fy, fx] >= 0.45:
                                if not wake_mask[fy, fx] and (global_visited is None or not global_visited[fy, fx]):
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
            # Smooth tangent update (exponential moving average for natural hand inertia)
            blend_dy = 0.35 * cur_dy + 0.65 * best_vdy
            blend_dx = 0.35 * cur_dx + 0.65 * best_vdx
            bnorm = float(np.hypot(blend_dy, blend_dx))
            cur_dy, cur_dx = (blend_dy / bnorm, blend_dx / bnorm) if bnorm > 1e-4 else (best_vdy, best_vdx)

            w_val = max(1.5, float(dist_map[iy, ix] * 2.0))
            pts.append((round(cy, 2), round(cx, 2), round(w_val, 2)))

            # Check loop closure: if returning close to start after >= 8 steps, close cleanly and stop
            if len(pts) >= 8:
                dist_to_start = float(np.hypot(cy - pts[0][0], cx - pts[0][1]))
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

    def _parameterize_stroke(self, raw_pts: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Applies biomechanical Gaussian trajectory smoothing, simplifies micro-oscillations,
        and computes the 40° beveled quill nib width along the smooth curve.
        """
        if len(raw_pts) <= 2:
            return raw_pts

        # 1. Iterative Gaussian-like 5-point moving average filter to remove pixel grid jitter
        coords_y = np.array([float(p[0]) for p in raw_pts], dtype=np.float64)
        coords_x = np.array([float(p[1]) for p in raw_pts], dtype=np.float64)
        widths = np.array([float(p[2]) for p in raw_pts], dtype=np.float64)

        is_closed = (len(raw_pts) >= 6 and np.hypot(coords_y[0] - coords_y[-1], coords_x[0] - coords_x[-1]) <= 2.8)

        for _ in range(2):
            if len(coords_y) >= 4:
                new_y = coords_y.copy()
                new_x = coords_x.copy()
                for i in range(1, len(coords_y) - 1):
                    new_y[i] = 0.25 * coords_y[i - 1] + 0.50 * coords_y[i] + 0.25 * coords_y[i + 1]
                    new_x[i] = 0.25 * coords_x[i - 1] + 0.50 * coords_x[i] + 0.25 * coords_x[i + 1]
                coords_y, coords_x = new_y, new_x

        if is_closed:
            coords_y[-1] = coords_y[0]
            coords_x[-1] = coords_x[0]

        pre_smoothed = [
            (float(coords_y[i]), float(coords_x[i]), float(widths[i]))
            for i in range(len(raw_pts))
        ]

        # 2. Simplify redundant collinear points while preserving smooth arcs (epsilon = 0.65 px)
        simplified = self._rdp_simplify_with_width(pre_smoothed, epsilon=0.65)

        # Ensure sufficient control points (at least ~1 point every 3.5 px) for smooth SVG spline rendering
        resampled: List[Tuple[float, float, float]] = [simplified[0]]
        for k in range(len(simplified) - 1):
            p0 = simplified[k]
            p1 = simplified[k + 1]
            seg_d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            if seg_d > 4.0:
                n_sub = int(np.ceil(seg_d / 3.2))
                for s_i in range(1, n_sub):
                    t = float(s_i) / float(n_sub)
                    resampled.append((
                        p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1]),
                        p0[2] + t * (p1[2] - p0[2])
                    ))
            resampled.append(p1)

        # 3. Compute local tangent and physical 40-deg beveled quill width
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
            nib_w = self.estimate_nib_width(tangent)
            combined_w = 0.5 * resampled[i][2] + 0.5 * nib_w
            final_pts.append((round(resampled[i][0], 2), round(resampled[i][1], 2), round(combined_w, 2)))

        return final_pts
