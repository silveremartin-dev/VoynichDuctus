"""
Biomechanical Millimeter Quill-Bernoulli Vectorizer (`CalligraphicVectorizer`).

Unifies topological minimum-bending skeleton graphs (Euler-Bernoulli) with strict
physical millimeter writing-instrument mechanics (zero pixel thresholds, zero box-relative
nib auto-scaling):
1. Fixed Physical Instrument Scale (`mm`):
   - Voynich 40° beveled goose quill (`w_nib ~ 0.45 mm`, chisel edge `~0.13 mm`, max pressed plein `<= 0.92 mm`).
   - Codex Seraphinianus continuous cursive nib (`w_nib ~ 0.42 mm`, pressure-modulated single-stroke ribbon).
   - Sub-nib physical impossibility gate: marks <= `0.55 mm` (`<= 1.22 * w_nib`) or `<= 0.35 mm^2` cannot
     physically sustain a stroke trajectory and derender strictly as stationary nib impacts (`v = 0`, `length_mm = 0`).
   - Macro physical ceiling: shapes `> 15 mm` or ink bands thicker than `3 * w_nib` never inflate the virtual nib
     beyond physical quill bounds (`is_macro_non_scribal = True`).
2. Economy of Motion (Minimum Biomechanical Action Functional in `mm`):
   - Built directly on the homotopy-preserving medial skeleton graph (preserving 100% of Euler loop cycles
     and stem-to-bowl Eulerian trails without artificial loop amputation).
   - Splices protruding `'3'`-shaped cusp tongues (`v = 0` bounce-back without air lift).
   - Chains junction segments when on-vellum bending cost in `mm` beats physical pen-lift cost `C_lift_mm`,
     penalizes anti-grain upward quill gouging, and solves exact minimum air-travel stroke sequencing.
3. Coupled 40° Bevel + Inferred Scribal Pressure `p(s)` + Analytic Loop-Hole Clearance:
   - Infers smooth muscular pressure `p(s)` along arc-length from physical ink ribbon width `2 * EDT_mm(s)`
     relative to geometric 40° bevel projection, while analytically clamping clearance around enclosed holes.
"""

import itertools
import math
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter1d, label
from skimage.morphology import skeletonize

from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor

if TYPE_CHECKING:
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration


class CalligraphicVectorizer:
    """
    Biomechanical Millimeter Quill-Bernoulli Ductus Engine.
    Derenders 2D binary ink masks into ordered `(y, x, width_px)` trajectories governed
    strictly in physical millimeters (`mm`) by nib geometry, muscular pressure, and motion economy.
    """

    _DEG_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    _ONES_3X3_U8 = np.ones((3, 3), dtype=np.uint8)
    _RIDGE_ANGLES: List[Tuple[float, float, bool]] = [
        (float(np.cos(a)), float(np.sin(a)), bool(abs(a) <= np.deg2rad(18)))
        for a in np.linspace(-np.deg2rad(65), np.deg2rad(65), 19)
    ]

    # Physical writing instrument constants in millimeters (mm)
    VOYNICH_NOMINAL_NIB_MM: float = 0.45
    SERAFINI_NOMINAL_NIB_MM: float = 0.42
    MIN_NIB_CHISEL_RATIO: float = 0.28      # Thin 40° délié edge (~0.13 mm)
    MAX_PRESSED_SPREAD_MM: float = 0.92     # Physical ceiling for heavy pressure + vellum capillary spread
    MACRO_SPAN_THRESHOLD_MM: float = 15.0   # Above 15 mm, flag as macro / non-scribal illustration element
    PEN_LIFT_ACTION_COST_MM: float = 1.45   # Equivalent hand-travel cost (mm) of lifting & re-landing the quill

    def __init__(
        self,
        nib_angle_deg: float = 40.0,
        min_nib_width: float = 1.2,
        max_nib_width: float = 5.5,
        smoothing_sigma: float = 0.85,
    ):
        self.nib_angle_rad = float(np.deg2rad(nib_angle_deg))
        self.min_nib_width = float(min_nib_width)
        self.max_nib_width = float(max_nib_width)
        self.smoothing_sigma = float(smoothing_sigma)

    def estimate_nib_width(self, tangent_angle: float) -> float:
        """
        Calculates expected stroke thickness based on tangent direction relative to nib bevel:
        w(theta) = (W_max - W_min) * |sin(theta - theta_nib)| + W_min
        """
        d_theta = tangent_angle - self.nib_angle_rad
        effective_w = (self.max_nib_width - self.min_nib_width) * np.abs(np.sin(d_theta)) + self.min_nib_width
        return float(effective_w)

    @staticmethod
    def prune_skeleton_spurs(skel: np.ndarray, dist_map: np.ndarray, min_spur_len: int = 4) -> np.ndarray:
        """
        Protrusion-aware morphological spur filter:
        Removes only parasitic degree-1 skeleton hairs caused by vellum edge roughness
        that remain trapped inside `1.22 * R_junc` of their parent junction, preserving
        100% of protruding cusps (`'3'`), serifs, and terminal hooks.
        """
        pruned = skel.copy().astype(bool)
        h, w = pruned.shape
        if int(np.sum(pruned)) <= 5:
            return pruned

        def get_degree_map(s: np.ndarray) -> np.ndarray:
            s_u8 = s.astype(np.uint8)
            return (
                cv2.filter2D(s_u8, cv2.CV_16U, CalligraphicVectorizer._DEG_KERNEL, borderType=cv2.BORDER_CONSTANT)
                * s_u8
            ).astype(np.int32)

        deg_map = get_degree_map(pruned)
        if not np.any(deg_map >= 3):
            return pruned

        rem_count = int(np.sum(pruned))
        endpoints = np.argwhere(deg_map == 1)
        for ey, ex in endpoints:
            if not pruned[ey, ex]:
                continue
            chain = [(int(ey), int(ex))]
            cy, cx = int(ey), int(ex)
            hit_junction = False
            jn_y, jn_x = cy, cx
            local_nib_r = max(float(min_spur_len), float(dist_map[ey, ex]) * 1.45)
            max_walk = int(round(local_nib_r)) + 1

            for _ in range(max_walk):
                y0, y1 = max(0, cy - 1), min(h, cy + 2)
                x0, x1 = max(0, cx - 1), min(w, cx + 2)
                nbs = [
                    (ny, nx)
                    for ny in range(y0, y1)
                    for nx in range(x0, x1)
                    if (ny != cy or nx != cx) and pruned[ny, nx] and (ny, nx) not in chain
                ]
                if not nbs:
                    break
                ny, nx = nbs[0]
                if deg_map[ny, nx] >= 3:
                    hit_junction = True
                    jn_y, jn_x = ny, nx
                    break
                chain.append((ny, nx))
                cy, cx = ny, nx

            if hit_junction and len(chain) <= max_walk and (rem_count - len(chain)) >= 4:
                r_junc = float(dist_map[jn_y, jn_x])
                tip_dist = math.hypot(float(ey - jn_y), float(ex - jn_x))
                if tip_dist > r_junc * 1.25:
                    continue
                tip_y, tip_x = chain[-1]
                y0_j, y1_j = max(0, jn_y - 1), min(h, jn_y + 2)
                x0_j, x1_j = max(0, jn_x - 1), min(w, jn_x + 2)
                rem_deg = sum(
                    1
                    for jy in range(y0_j, y1_j)
                    for jx in range(x0_j, x1_j)
                    if (jy, jx) != (jn_y, jn_x) and (jy, jx) != (tip_y, tip_x) and pruned[jy, jx]
                )
                if rem_deg >= 2:
                    for py, px in chain:
                        pruned[py, px] = False
                    rem_count -= len(chain)
                    deg_map = get_degree_map(pruned)

        return pruned

    def find_touchdown_endpoints(
        self, binary_mask: np.ndarray, skel: np.ndarray
    ) -> List[Tuple[float, int, int, float, float]]:
        """Identifies scribal touchdown endpoints (degree-1 tips or 11-o'clock loop crests)."""
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

            init_dy, init_dx = float(cy - y), float(cx - x)
            n_len = math.hypot(init_dy, init_dx)
            if n_len > 1e-4:
                endpoints.append((priority, int(y), int(x), init_dy / n_len, init_dx / n_len))
            else:
                endpoints.append((priority, int(y), int(x), 1.0, 0.2))

        if not endpoints:
            skel_pts = np.argwhere(skel)
            best_p = min(skel_pts, key=lambda p: (p[0] / max(1.0, float(h))) * 1.5 + (p[1] / max(1.0, float(w))) * 2.0)
            endpoints.append((12.0, int(best_p[0]), int(best_p[1]), 1.0, 0.3))

        endpoints.sort(key=lambda item: item[0], reverse=True)
        return endpoints

    @staticmethod
    def _polyline_arc_mm(pts: List[Tuple[float, float, float]], mm_per_px: float) -> float:
        if len(pts) < 2:
            return 0.0
        return float(
            sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))
            * mm_per_px
        )

    def _make_impact_stroke(
        self,
        pts_rc: np.ndarray,
        stroke_idx: int,
        ppm: float,
        w_nib_mm: float,
    ) -> Dict[str, Any]:
        """Builds a zero-trajectory stationary nib touchdown impact (`v = 0`, `length_mm = 0.0`)."""
        cy = float(np.mean(pts_rc[:, 0]))
        cx = float(np.mean(pts_rc[:, 1]))
        span_h_mm = (float(pts_rc[:, 0].max() - pts_rc[:, 0].min()) + 1.0) / ppm
        span_w_mm = (float(pts_rc[:, 1].max() - pts_rc[:, 1].min()) + 1.0) / ppm
        w_edge_mm = self.MIN_NIB_CHISEL_RATIO * w_nib_mm
        w_contact_mm = float(np.clip(min(max(span_h_mm, span_w_mm), w_nib_mm * 1.15), w_edge_mm, self.MAX_PRESSED_SPREAD_MM))
        w_px = round(max(0.5, w_contact_mm * ppm), 2)
        return {
            "stroke_id": f"cal_s{stroke_idx:02d}",
            "order_index": stroke_idx,
            "points": [(round(cy, 2), round(cx, 2), w_px)],
            "length": 0.0,
            "length_mm": 0.0,
            "start": (round(cy, 2), round(cx, 2)),
            "end": (round(cy, 2), round(cx, 2)),
            "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
            "mean_width": w_px,
            "mean_width_mm": round(w_contact_mm, 3),
            "mean_pressure": 1.0,
            "is_impact": True,
            "is_dot": True,
            "is_macro_non_scribal": False,
            "px_per_mm": ppm,
        }

    def extract_calligraphic_ductus(
        self,
        binary_mask: np.ndarray,
        single_stroke_mode: bool = False,
        max_strokes: Optional[int] = None,
        scale_calib: Optional["MetricScaleCalibration"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Derenders a 2D binary ink mask into biomechanically ordered `(y, x, width_px)` strokes
        governed strictly in millimeters (`mm`) by physical nib geometry, pressure, and motion economy.
        """
        if binary_mask is None or not np.any(binary_mask):
            return []

        mask_bool = binary_mask.astype(bool)
        if scale_calib is None:
            from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
            calib = MetricScaleCalibration(
                manuscript_type="seraphinianus" if single_stroke_mode else "voynich"
            )
        else:
            calib = scale_calib
        ppm = max(1e-3, float(calib.px_per_mm))
        mm_per_px = 1.0 / ppm

        w_nib_nom_mm = self.SERAFINI_NOMINAL_NIB_MM if single_stroke_mode else self.VOYNICH_NOMINAL_NIB_MM

        pts_all = np.argwhere(mask_bool)
        span_h_mm = (float(pts_all[:, 0].max() - pts_all[:, 0].min()) + 1.0) * mm_per_px
        span_w_mm = (float(pts_all[:, 1].max() - pts_all[:, 1].min()) + 1.0) * mm_per_px
        total_area_mm2 = float(len(pts_all)) * (mm_per_px ** 2)

        # Physical Law 1: Sub-nib & zero-displacement impact gate (e.g. 0.3 mm speck or stationary 0.65 mm nib blot)
        # No stroke trajectory can exist if the mark is smaller than the nib OR is a solid compact blot with ~0 net travel!
        sub_nib_span_mm = max(calib.DOT_MAX_DIAMETER_MM, 1.40 * w_nib_nom_mm)
        if (span_h_mm <= sub_nib_span_mm and span_w_mm <= sub_nib_span_mm) or total_area_mm2 <= max(calib.DOT_MAX_AREA_MM2, 0.42):
            return [self._make_impact_stroke(pts_all, 0, ppm, w_nib_nom_mm)]

        dist_full_mm = distance_transform_edt(mask_bool) * mm_per_px
        max_inscribed_d_mm = float(np.max(dist_full_mm)) * 2.0
        if max(span_h_mm, span_w_mm) <= 1.05 and (max(span_h_mm, span_w_mm) - max_inscribed_d_mm) <= 0.22:
            pad_chk = np.pad(mask_bool, 1, mode="constant", constant_values=False)
            if not np.any(binary_fill_holes(pad_chk) & (~pad_chk)):
                return [self._make_impact_stroke(pts_all, 0, ppm, w_nib_nom_mm)]
        inner_mm = dist_full_mm[mask_bool]
        med_ribbon_mm = float(np.median(inner_mm)) * 2.0 if len(inner_mm) > 0 else w_nib_nom_mm

        # Calibrate active nib width strictly within authentic physical instrument bounds [0.30 mm, 0.65 mm]
        # Even on a 300 mm x 300 mm block, w_nib_mm NEVER exceeds 0.65 mm!
        w_nib_mm = float(np.clip(0.65 * med_ribbon_mm + 0.35 * w_nib_nom_mm, 0.30, 0.65))
        is_macro = bool(
            max(span_h_mm, span_w_mm) > self.MACRO_SPAN_THRESHOLD_MM or med_ribbon_mm > 3.0 * self.VOYNICH_NOMINAL_NIB_MM
        )

        if single_stroke_mode or (max_strokes is not None and max_strokes == 1):
            st = self._extract_continuous_cursive_tour(
                mask_bool, ppm=ppm, w_nib_mm=w_nib_mm, is_macro=is_macro
            )
            return [st] if st is not None else [self._make_impact_stroke(pts_all, 0, ppm, w_nib_mm)]

        stroke_limit = min(4, max_strokes) if max_strokes is not None else 4
        labeled_mask, num_features = label(mask_bool, structure=np.ones((3, 3), dtype=bool))
        if num_features == 0:
            return []

        comp_min_area_px = max(1, int(round(calib.area_mm2_to_px2(0.035))))
        components = []
        for cid in range(1, num_features + 1):
            cm = (labeled_mask == cid)
            ca = int(np.sum(cm))
            if ca < comp_min_area_px:
                continue
            c_pts = np.argwhere(cm)
            components.append({
                "mask": cm,
                "pts": c_pts,
                "area_mm2": float(ca) * (mm_per_px ** 2),
                "span_h_mm": (float(c_pts[:, 0].max() - c_pts[:, 0].min()) + 1.0) * mm_per_px,
                "span_w_mm": (float(c_pts[:, 1].max() - c_pts[:, 1].min()) + 1.0) * mm_per_px,
            })

        if not components:
            return [self._make_impact_stroke(pts_all, 0, ppm, w_nib_mm)]

        # Prioritize largest substantive components so tiny specks never starve main stems of stroke budget
        components.sort(key=lambda c: c["area_mm2"], reverse=True)

        graph_ext = StrokeGraphExtractor()
        raw_stroke_dicts: List[Dict[str, Any]] = []

        for comp in components:
            rem_budget = stroke_limit - len(raw_stroke_dicts)
            if rem_budget <= 0:
                break

            # Detached sub-nib dot component inside a multi-component glyph (e.g. dot of 'i')
            if (
                (comp["span_h_mm"] <= sub_nib_span_mm and comp["span_w_mm"] <= sub_nib_span_mm)
                or comp["area_mm2"] <= calib.DOT_MAX_AREA_MM2
            ):
                raw_stroke_dicts.append(self._make_impact_stroke(comp["pts"], len(raw_stroke_dicts), ppm, w_nib_mm))
                continue

            comp_strokes = self._extract_component_quill_bernoulli(
                comp["mask"],
                ppm=ppm,
                w_nib_mm=w_nib_mm,
                max_strokes=rem_budget,
                is_serafini=False,
                is_macro=is_macro,
                graph_ext=graph_ext,
            )
            raw_stroke_dicts.extend(comp_strokes)

        if not raw_stroke_dicts:
            return [self._make_impact_stroke(pts_all, 0, ppm, w_nib_mm)]

        # Solve global minimum biomechanical action sequencing (top-left entry + quill pull prior + minimum air-travel)
        ordered = self._order_strokes_minimum_action(raw_stroke_dicts, mask_bool.shape, ppm=ppm)
        return ordered[:stroke_limit]

    def _clean_pinholes_and_get_holes(
        self, comp_mask: np.ndarray, ppm: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fills microscopic vellum grain pores (`<= 0.045 mm^2` or inscribed radius `<= 0.115 mm` < quarter nib),
        preserving 100% of genuine enclosed loop cavities (`o`, `a`, `e`, `d`, `8`, `sh`) and precomputing `dist_to_hole_px`.
        """
        pinhole_max_px = max(1, int(round(0.045 * (ppm ** 2))))
        max_pore_rad_px = 0.115 * ppm
        padded = np.pad(comp_mask, 2, mode="constant", constant_values=False)
        holes = binary_fill_holes(padded) & (~padded)
        lbl_h, num_h = label(holes)
        clean_comp = comp_mask.copy()
        holes_union = np.zeros_like(comp_mask, dtype=bool)
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)[2:-2, 2:-2]
            ha = int(np.sum(hm))
            if ha <= pinhole_max_px:
                clean_comp |= hm
            elif ha <= int(round(0.070 * (ppm ** 2))) and float(np.max(distance_transform_edt(hm))) <= max_pore_rad_px:
                clean_comp |= hm
            elif np.any(hm):
                holes_union |= hm
        dist_to_hole = distance_transform_edt(~holes_union) if np.any(holes_union) else None
        return clean_comp, holes_union, dist_to_hole

    def _extract_component_quill_bernoulli(
        self,
        comp_mask: np.ndarray,
        ppm: float,
        w_nib_mm: float,
        max_strokes: int,
        is_serafini: bool,
        is_macro: bool,
        graph_ext: StrokeGraphExtractor,
    ) -> List[Dict[str, Any]]:
        """
        Extracts 1..max_strokes trajectories on a connected component directly from its
        homotopy-preserving medial skeleton graph using minimum biomechanical action.
        """
        h, w = comp_mask.shape
        mm_per_px = 1.0 / ppm
        clean_comp, holes_union, dist_to_hole = self._clean_pinholes_and_get_holes(comp_mask, ppm)
        dist_map = distance_transform_edt(clean_comp)
        raw_skel = skeletonize(clean_comp)

        if not np.any(raw_skel):
            pts_c = np.argwhere(clean_comp)
            return [self._make_impact_stroke(pts_c, 0, ppm, w_nib_mm)] if len(pts_c) > 0 else []

        spur_min_px = max(2, int(round(0.38 * ppm)))
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_px)

        p_graph = graph_ext.build_pixel_graph(skel, dist_map * 2.0)
        raw_segs = graph_ext.decompose_into_strokes(p_graph)
        if not raw_segs:
            pts_c = np.argwhere(clean_comp)
            return [self._make_impact_stroke(pts_c, 0, ppm, w_nib_mm)] if len(pts_c) > 0 else []

        # 1. Splice genuine protruding '3'-cusp bounce-back tongues (>= 0.50 mm physical protrusion)
        raw_segs = self._splice_cusp_tongues(raw_segs, px_per_mm=ppm)

        # 2. Drop tiny sub-nib junction-triangle slivers (< 0.30 mm) only when main branches exist
        min_sliver_mm = 0.30
        if len(raw_segs) > 1:
            substantive = [
                s for s in raw_segs
                if s.get("type") == "loop" or self._polyline_arc_mm(s.get("points", []), mm_per_px) >= min_sliver_mm
            ]
            if substantive:
                raw_segs = substantive

        # 3. Minimum-action junction chaining: merge segments across junctions whenever
        #    on-vellum bending cost in mm < physical pen-lift action cost (1.45 mm)
        chained_segs = self._chain_segments_minimum_action(
            raw_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=max_strokes
        )

        # 4. Complete free eroded endpoints to true ink extremities (bounded narrow forward cone in mm)
        med_half_w = float(np.median(dist_map[clean_comp])) if np.any(clean_comp) else (0.5 * w_nib_mm * ppm)
        raw_polylines: List[List[Tuple[float, float, float]]] = []
        for s in chained_segs:
            pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in s.get("points", [])]
            if len(pts) >= 2:
                pts = self._extend_and_anchor_stroke(pts, clean_comp, dist_map, raw_polylines, med_half_w, ppm=ppm)
                raw_polylines.append(pts)

        # 5. Residual ink limb recovery: ONLY if an authentic uncovered stroke limb exists
        #    beyond physical nib footprint (area >= max(0.22, 1.10 * w_nib^2) mm^2 AND arc >= 1.30 * w_nib mm)
        min_res_mm2 = max(0.22, 1.10 * (w_nib_mm ** 2))
        min_res_px = max(4, int(round(min_res_mm2 * (ppm ** 2))))
        min_limb_arc_mm = max(0.55, 1.30 * w_nib_mm)
        while len(raw_polylines) < max_strokes and raw_polylines:
            pred_m = self._rasterize_strokes_mask(raw_polylines, (h, w), scale_w=1.05)
            uncov = clean_comp & (~pred_m)
            if int(np.sum(uncov)) < min_res_px:
                break
            lbl_u, n_u = label(uncov, structure=np.ones((3, 3), dtype=bool))
            if n_u == 0:
                break
            cnt_u = np.bincount(lbl_u.ravel())
            cnt_u[0] = 0
            best_uid = int(np.argmax(cnt_u))
            if int(cnt_u[best_uid]) < min_res_px:
                break
            u_mask = (lbl_u == best_uid)
            grow_iters = max(1, int(round(0.18 * ppm)))
            u_grown = (cv2.dilate(u_mask.astype(np.uint8), self._ONES_3X3_U8, iterations=grow_iters) > 0) & clean_comp
            u_skel = skeletonize(u_grown)
            if int(np.sum(u_skel)) < 2:
                break
            u_segs = graph_ext.decompose_into_strokes(graph_ext.build_pixel_graph(u_skel, dist_map * 2.0))
            if not u_segs:
                break
            u_ch = self._chain_segments_minimum_action(u_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=1)
            if not u_ch or len(u_ch[0].get("points", [])) < 2:
                break
            u_pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in u_ch[0]["points"]]
            u_pts = self._extend_and_anchor_stroke(u_pts, clean_comp, dist_map, raw_polylines, med_half_w, ppm=ppm)
            if self._polyline_arc_mm(u_pts, mm_per_px) < min_limb_arc_mm:
                break
            raw_polylines.append(u_pts)
            new_pred = self._rasterize_strokes_mask(raw_polylines, (h, w), scale_w=1.05)
            if int(np.sum(clean_comp & (~new_pred))) >= int(np.sum(uncov)) - 1:
                break

        # 6. Parameterize with coupled 40° bevel + inferred muscular pressure p(s) + loop-hole clearance
        out_strokes: List[Dict[str, Any]] = []
        for idx, poly in enumerate(raw_polylines[:max_strokes]):
            cal_pts, mean_p = self._parameterize_biomechanical_stroke(
                poly,
                clean_comp=clean_comp,
                dist_map=dist_map,
                dist_to_hole=dist_to_hole,
                ppm=ppm,
                w_nib_mm=w_nib_mm,
                is_serafini=is_serafini,
            )
            if len(cal_pts) < 2:
                continue
            arc_mm = self._polyline_arc_mm(cal_pts, mm_per_px)
            mean_w_px = float(np.mean([p[2] for p in cal_pts]))
            out_strokes.append({
                "stroke_id": f"cal_s{idx:02d}",
                "order_index": idx,
                "points": cal_pts,
                "length": float(len(cal_pts)),
                "length_mm": round(arc_mm, 3),
                "start": (cal_pts[0][0], cal_pts[0][1]),
                "end": (cal_pts[-1][0], cal_pts[-1][1]),
                "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
                "mean_width": round(mean_w_px, 2),
                "mean_width_mm": round(mean_w_px * mm_per_px, 3),
                "mean_pressure": round(mean_p, 3),
                "is_impact": False,
                "is_dot": False,
                "is_macro_non_scribal": is_macro,
                "px_per_mm": ppm,
            })

        return out_strokes

    def _extract_continuous_cursive_tour(
        self,
        mask_bool: np.ndarray,
        ppm: float,
        w_nib_mm: float,
        is_macro: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts strictly 1 continuous biomechanical cursive stroke (Codex Seraphinianus mode)
        via an Eulerian Trail / Chinese-Postman tour directly on the homotopy-preserving skeleton graph.
        """
        lbl, num_f = label(mask_bool, structure=np.ones((3, 3), dtype=bool))
        if num_f == 0:
            return None
        cc_sizes = [(cid, int(np.sum(lbl == cid))) for cid in range(1, num_f + 1)]
        dom_id = max(cc_sizes, key=lambda x: x[1])[0]
        comp_mask = (lbl == dom_id)
        min_frag_px = max(2, int(round(0.05 * (ppm ** 2))))
        for cid, csize in cc_sizes:
            if cid != dom_id and csize >= min_frag_px:
                comp_mask |= (lbl == cid)

        h, w = comp_mask.shape
        mm_per_px = 1.0 / ppm
        clean_comp, _, dist_to_hole = self._clean_pinholes_and_get_holes(comp_mask, ppm)
        dist_map = distance_transform_edt(clean_comp)
        raw_skel = skeletonize(clean_comp)
        if not np.any(raw_skel):
            return None

        spur_min_px = max(2, int(round(0.36 * ppm)))
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_px)
        graph_ext = StrokeGraphExtractor()
        raw_segs = graph_ext.decompose_into_strokes(graph_ext.build_pixel_graph(skel, dist_map * 2.0))
        if not raw_segs:
            return None

        raw_segs = self._splice_cusp_tongues(raw_segs, px_per_mm=ppm)
        chained = self._chain_segments_minimum_action(raw_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=6)
        med_half_w = float(np.median(dist_map[clean_comp])) if np.any(clean_comp) else (0.5 * w_nib_mm * ppm)

        chains: List[List[Tuple[float, float, float]]] = []
        for s in chained:
            pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in s.get("points", [])]
            if len(pts) >= 2:
                pts = self._extend_and_anchor_stroke(pts, clean_comp, dist_map, chains, med_half_w, ppm=ppm)
                chains.append(pts)

        if not chains:
            return None

        # Recover any genuine uncovered cursive flourish (>= nib footprint) before single-tour assembly
        min_res_mm2 = max(0.20, 1.05 * (w_nib_mm ** 2))
        min_res_px = max(4, int(round(min_res_mm2 * (ppm ** 2))))
        min_limb_arc_mm = max(0.52, 1.25 * w_nib_mm)
        for _ in range(2):
            pred_m = self._rasterize_strokes_mask(chains, (h, w), scale_w=1.05)
            uncov = clean_comp & (~pred_m)
            if int(np.sum(uncov)) < min_res_px:
                break
            lbl_u, n_u = label(uncov, structure=np.ones((3, 3), dtype=bool))
            if n_u == 0:
                break
            cnt_u = np.bincount(lbl_u.ravel())
            cnt_u[0] = 0
            best_uid = int(np.argmax(cnt_u))
            if int(cnt_u[best_uid]) < min_res_px:
                break
            u_mask = (lbl_u == best_uid)
            u_grown = (cv2.dilate(u_mask.astype(np.uint8), self._ONES_3X3_U8, iterations=max(1, int(round(0.18 * ppm)))) > 0) & clean_comp
            u_skel = skeletonize(u_grown)
            if int(np.sum(u_skel)) < 2:
                break
            u_segs = graph_ext.decompose_into_strokes(graph_ext.build_pixel_graph(u_skel, dist_map * 2.0))
            if not u_segs:
                break
            u_ch = self._chain_segments_minimum_action(u_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=1)
            if u_ch and len(u_ch[0].get("points", [])) >= 2:
                u_pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in u_ch[0]["points"]]
                u_ext = self._extend_and_anchor_stroke(u_pts, clean_comp, dist_map, chains, med_half_w, ppm=ppm)
                if self._polyline_arc_mm(u_ext, mm_per_px) >= min_limb_arc_mm:
                    chains.append(u_ext)
                else:
                    break

        # Assemble all chains into 1 minimum-action continuous cursive tour (splicing T-branches in-place)
        chains.sort(key=lambda ch: (min(p[0] for p in ch) * 1.2 + min(p[1] for p in ch)) * mm_per_px)
        tour = list(chains[0])
        close_tol_px = 0.35 * ppm
        if len(tour) >= 2 and math.hypot(tour[0][0] - tour[-1][0], tour[0][1] - tour[-1][1]) > close_tol_px:
            if (tour[0][0] * 1.5 + tour[0][1]) > (tour[-1][0] * 1.5 + tour[-1][1]):
                tour = tour[::-1]

        rem = chains[1:]
        while rem:
            best_ri, best_mode, best_d = 0, "append_fwd", 1e9
            for ri, rch in enumerate(rem):
                opts = [
                    ("append_fwd", math.hypot(tour[-1][0] - rch[0][0], tour[-1][1] - rch[0][1])),
                    ("append_rev", math.hypot(tour[-1][0] - rch[-1][0], tour[-1][1] - rch[-1][1])),
                    ("prepend_fwd", math.hypot(rch[-1][0] - tour[0][0], rch[-1][1] - tour[0][1])),
                    ("prepend_rev", math.hypot(rch[0][0] - tour[0][0], rch[0][1] - tour[0][1])),
                ]
                for m_s, d_s in opts:
                    if d_s < best_d:
                        best_d, best_ri, best_mode = d_s, ri, m_s
            chosen = rem.pop(best_ri)
            is_ch_loop = math.hypot(chosen[0][0] - chosen[-1][0], chosen[0][1] - chosen[-1][1]) <= close_tol_px
            d_mid_0 = min((math.hypot(bp[0] - chosen[0][0], bp[1] - chosen[0][1]), bi) for bi, bp in enumerate(tour))
            d_mid_1 = min((math.hypot(bp[0] - chosen[-1][0], bp[1] - chosen[-1][1]), bi) for bi, bp in enumerate(tour))

            if is_ch_loop and d_mid_0[0] + 0.25 * ppm < best_d:
                bi = d_mid_0[1]
                tour = tour[:bi + 1] + chosen + tour[bi + 1:]
            elif min(d_mid_0[0], d_mid_1[0]) + 0.35 * ppm < best_d and min(d_mid_0[0], d_mid_1[0]) <= 0.75 * ppm:
                if d_mid_0[0] <= d_mid_1[0]:
                    bi = d_mid_0[1]
                    exc = chosen + chosen[:-1][::-1]
                else:
                    bi = d_mid_1[1]
                    rev_c = chosen[::-1]
                    exc = rev_c + rev_c[:-1][::-1]
                tour = tour[:bi + 1] + exc + tour[bi + 1:]
            else:
                if best_mode == "append_fwd":
                    tour = tour + chosen
                elif best_mode == "append_rev":
                    tour = tour + chosen[::-1]
                elif best_mode == "prepend_fwd":
                    tour = chosen + tour
                else:
                    tour = chosen[::-1] + tour

        # Ensure natural top-to-bottom / left-to-right cursive orientation when open
        if len(tour) >= 2 and math.hypot(tour[0][0] - tour[-1][0], tour[0][1] - tour[-1][1]) > close_tol_px:
            if (tour[0][0] * 1.5 + tour[0][1] * 0.8) > (tour[-1][0] * 1.5 + tour[-1][1] * 0.8):
                tour = tour[::-1]

        cal_pts, mean_p = self._parameterize_biomechanical_stroke(
            tour,
            clean_comp=clean_comp,
            dist_map=dist_map,
            dist_to_hole=dist_to_hole,
            ppm=ppm,
            w_nib_mm=w_nib_mm,
            is_serafini=True,
        )
        if len(cal_pts) < 2:
            return None

        arc_mm = self._polyline_arc_mm(cal_pts, mm_per_px)
        mean_w_px = float(np.mean([p[2] for p in cal_pts]))
        return {
            "stroke_id": "cal_s00",
            "order_index": 0,
            "points": cal_pts,
            "length": float(len(cal_pts)),
            "length_mm": round(arc_mm, 3),
            "start": (cal_pts[0][0], cal_pts[0][1]),
            "end": (cal_pts[-1][0], cal_pts[-1][1]),
            "nib_angle_deg": float(np.rad2deg(self.nib_angle_rad)),
            "mean_width": round(mean_w_px, 2),
            "mean_width_mm": round(mean_w_px * mm_per_px, 3),
            "mean_pressure": round(mean_p, 3),
            "is_impact": False,
            "is_dot": False,
            "is_macro_non_scribal": is_macro,
            "px_per_mm": ppm,
        }

    def _chain_segments_minimum_action(
        self,
        segments: List[Dict[str, Any]],
        ppm: float,
        w_nib_mm: float,
        max_strokes: int,
    ) -> List[Dict[str, Any]]:
        """
        Chains topological skeleton segments across junctions by comparing on-vellum
        Euler-Bernoulli bending + gap cost (in mm) against the physical pen-lift cost `C_lift_mm`:
          C_join(s1 -> s2) = d_gap_mm + 0.95 * w_nib_mm * (1 - cos_theta)^1.4
        Straight X-crossings (cos_theta ~ 1.0, C_join ~ 0) merge first, leaving orthogonal crossbars
        separate; smooth stem-to-bowl Eulerian trails (`'d'`, `'b'`, `'q'`, `'8'`) fuse into 1 fluid stroke.
        """
        if len(segments) <= 1:
            return segments

        mm_per_px = 1.0 / ppm
        max_join_gap_mm = max(0.75, 1.65 * w_nib_mm)
        k_tan = max(2, int(round(0.32 * ppm)))

        work = [dict(s, points=list(s.get("points", []))) for s in segments if len(s.get("points", [])) >= 2]
        if len(work) <= 1:
            return work

        # Pass 1: Merge whenever C_join_mm < PEN_LIFT_ACTION_COST_MM and cos_theta >= -0.38
        merged = True
        while merged and len(work) > 1:
            merged = False
            best_cost = self.PEN_LIFT_ACTION_COST_MM
            best_action = None

            for i in range(len(work)):
                p1 = work[i]["points"]
                p1s, p1e = p1[0], p1[-1]
                for j in range(i + 1, len(work)):
                    p2 = work[j]["points"]
                    p2s, p2e = p2[0], p2[-1]

                    orientations = [
                        (math.hypot(p1e[0] - p2s[0], p1e[1] - p2s[1]) * mm_per_px, p1, p2),
                        (math.hypot(p1e[0] - p2e[0], p1e[1] - p2e[1]) * mm_per_px, p1, p2[::-1]),
                        (math.hypot(p2e[0] - p1s[0], p2e[1] - p1s[1]) * mm_per_px, p2, p1),
                        (math.hypot(p1s[0] - p2s[0], p1s[1] - p2s[1]) * mm_per_px, p1[::-1], p2),
                    ]
                    for gap_mm, a_pts, b_pts in orientations:
                        if gap_mm > max_join_gap_mm:
                            continue
                        cos_t = JunctionResolver._tangent_cosine(a_pts, b_pts, k=k_tan)
                        if cos_t < -0.38:
                            continue
                        turn_cost_mm = gap_mm + 0.95 * w_nib_mm * ((1.0 - cos_t) ** 1.4)
                        if turn_cost_mm < best_cost:
                            best_cost = turn_cost_mm
                            comb = a_pts + (b_pts[1:] if gap_mm < 0.08 else b_pts)
                            best_action = (i, j, comb)

            if best_action is not None:
                i, j, comb = best_action
                work[i] = {
                    "stroke_id": work[i].get("stroke_id", "s_m"),
                    "type": "segment",
                    "points": comb,
                    "start": (comb[0][0], comb[0][1]),
                    "end": (comb[-1][0], comb[-1][1]),
                    "length": float(len(comb)),
                }
                work.pop(j)
                merged = True

        # Pass 2: If still above stroke ceiling, coalesce closest touching branches at junctions before ranking by arc-length
        while max_strokes > 0 and len(work) > max_strokes:
            best_d_mm = max_join_gap_mm * 1.45
            best_pair = None
            for i in range(len(work)):
                p1 = work[i]["points"]
                for j in range(i + 1, len(work)):
                    p2 = work[j]["points"]
                    opts = [
                        (math.hypot(p1[-1][0] - p2[0][0], p1[-1][1] - p2[0][1]) * mm_per_px, p1 + p2),
                        (math.hypot(p1[-1][0] - p2[-1][0], p1[-1][1] - p2[-1][1]) * mm_per_px, p1 + p2[::-1]),
                        (math.hypot(p2[-1][0] - p1[0][0], p2[-1][1] - p1[0][1]) * mm_per_px, p2 + p1),
                        (math.hypot(p1[0][0] - p2[0][0], p1[0][1] - p2[0][1]) * mm_per_px, p1[::-1] + p2),
                    ]
                    for d_mm, comb in opts:
                        if d_mm < best_d_mm:
                            best_d_mm = d_mm
                            best_pair = (i, j, comb)
            if best_pair is not None:
                i, j, comb = best_pair
                work[i] = {
                    "stroke_id": work[i].get("stroke_id", "s_m"),
                    "type": "segment",
                    "points": comb,
                    "start": (comb[0][0], comb[0][1]),
                    "end": (comb[-1][0], comb[-1][1]),
                    "length": float(len(comb)),
                }
                work.pop(j)
            else:
                work.sort(key=lambda s: self._polyline_arc_mm(s["points"], mm_per_px), reverse=True)
                work = work[:max_strokes]
                break

        return work

    def _order_strokes_minimum_action(
        self,
        strokes: List[Dict[str, Any]],
        shape: Tuple[int, int],
        ppm: float,
    ) -> List[Dict[str, Any]]:
        """
        Solves exact minimum biomechanical action across all `K <= 4` strokes:
        - Closed pure loops start at their 11-o'clock top-left crest.
        - Open strokes evaluate forward vs reversed direction with a strong quill-grain penalty
          against pushing steeply upward (`dy_mm < -0.45 mm` against the 40° split nib).
        - Exact permutation + direction search minimizes total air-travel distance in mm between pen lifts.
        """
        mm_per_px = 1.0 / ppm
        close_tol_px = 0.35 * ppm

        def _anti_grain_cost_mm(pts: List[Tuple[float, float, float]]) -> float:
            if len(pts) < 2:
                return 0.0
            if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px and len(pts) >= 6:
                return 0.0
            dy_net_mm = (pts[-1][0] - pts[0][0]) * mm_per_px
            dx_net_mm = (pts[-1][1] - pts[0][1]) * mm_per_px
            cost = 0.0
            # Pulling downward (dy > 0) and rightward (dx > 0) glides smoothly; pushing upward digs into vellum
            if dy_net_mm < -0.25:
                cost += abs(dy_net_mm) * 2.4
            if dx_net_mm < -0.35 and dy_net_mm <= 0.25:
                cost += abs(dx_net_mm) * 1.4
            return cost

        # Normalize pure closed loops to start at 11-o'clock crest and build candidate orientations per stroke
        stroke_variants: List[List[Tuple[Dict[str, Any], float]]] = []
        for s in strokes:
            pts = list(s.get("points", []))
            if s.get("is_impact", False) or len(pts) < 2:
                stroke_variants.append([(dict(s), 0.0)])
                continue

            is_closed = len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px
            if is_closed:
                ring = pts[:-1]
                crest_i = min(range(len(ring)), key=lambda idx: ring[idx][0] * 1.5 + ring[idx][1] * 1.8)
                rot = ring[crest_i:] + ring[:crest_i] + [ring[crest_i]]
                s_loop = dict(s, points=rot, start=(rot[0][0], rot[0][1]), end=(rot[-1][0], rot[-1][1]))
                stroke_variants.append([(s_loop, 0.0)])
            else:
                c_fwd = _anti_grain_cost_mm(pts)
                rev_pts = pts[::-1]
                c_rev = _anti_grain_cost_mm(rev_pts)
                s_fwd = dict(s, points=pts, start=(pts[0][0], pts[0][1]), end=(pts[-1][0], pts[-1][1]))
                s_rev = dict(s, points=rev_pts, start=(rev_pts[0][0], rev_pts[0][1]), end=(rev_pts[-1][0], rev_pts[-1][1]))
                stroke_variants.append([(s_fwd, c_fwd), (s_rev, c_rev)])

        k_n = len(stroke_variants)
        if k_n == 1:
            best_s = min(stroke_variants[0], key=lambda item: item[1] + (item[0]["start"][0] * 0.5 + item[0]["start"][1] * 0.4) * mm_per_px)[0]
            best_s["order_index"] = 0
            best_s["stroke_id"] = "cal_s00"
            return [best_s]

        best_total = 1e12
        best_seq: List[Dict[str, Any]] = []
        for perm in itertools.permutations(range(k_n)):
            for choice_tuple in itertools.product(*[stroke_variants[idx] for idx in perm]):
                s0, g0 = choice_tuple[0]
                # Entry prior: Western hand enters from top-left of glyph box
                cost = (s0["start"][0] * 0.55 + s0["start"][1] * 0.65) * mm_per_px + g0
                for step_i in range(1, k_n):
                    s_prev = choice_tuple[step_i - 1][0]
                    s_curr, g_curr = choice_tuple[step_i]
                    air_mm = math.hypot(s_curr["start"][0] - s_prev["end"][0], s_curr["start"][1] - s_prev["end"][1]) * mm_per_px
                    cost += self.PEN_LIFT_ACTION_COST_MM + 0.65 * air_mm + g_curr
                if cost < best_total:
                    best_total = cost
                    best_seq = [dict(item[0]) for item in choice_tuple]

        for idx, s in enumerate(best_seq):
            s["order_index"] = idx
            s["stroke_id"] = f"cal_s{idx:02d}"
        return best_seq

    def _parameterize_biomechanical_stroke(
        self,
        raw_pts: List[Tuple[float, float, float]],
        clean_comp: np.ndarray,
        dist_map: np.ndarray,
        dist_to_hole: Optional[np.ndarray],
        ppm: float,
        w_nib_mm: float,
        is_serafini: bool,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        1. Enforces physical nib-scale intra-stroke bandwidth (`w_nib_mm`): freezes ONLY genuine
           velocity-reversal cusps (`cos < -0.20` measured over physical chords `>= 0.42 mm`, never 1-px grid stairs).
        2. Smooths trajectory coordinates with physical hand inertia (`sigma_mm = 0.22 mm`) and simplifies
           sub-nib jitter (`rdp_eps_mm = 0.052 mm`) without discrete per-point jumps.
        3. Infers muscular nib pressure `p(s)` along arc-length coupled with the `40°` nib bevel:
             w_mm(s) = p(s) * sqrt((w_nib * sin(theta - 40°))^2 + w_edge^2)
           strictly bounded to physical instrument limits `[w_edge_mm, MAX_PRESSED_SPREAD_MM]` and
           smoothly clamped outside interior loop cavities so holes NEVER plug.
        """
        if len(raw_pts) < 2:
            return raw_pts, 1.0

        h, w = clean_comp.shape
        mm_per_px = 1.0 / ppm

        dedup: List[Tuple[float, float, float]] = [(float(raw_pts[0][0]), float(raw_pts[0][1]), float(raw_pts[0][2]))]
        min_dedup_px = 0.020 * ppm
        for p in raw_pts[1:]:
            if math.hypot(float(p[0]) - dedup[-1][0], float(p[1]) - dedup[-1][1]) >= min_dedup_px:
                dedup.append((float(p[0]), float(p[1]), float(p[2])))
        if len(dedup) < 2:
            p_end = (float(raw_pts[-1][0]), float(raw_pts[-1][1]), float(raw_pts[-1][2]))
            if math.hypot(p_end[0] - dedup[0][0], p_end[1] - dedup[0][1]) > 1e-4:
                dedup.append(p_end)
            else:
                return raw_pts, 1.0

        n_pts = len(dedup)
        close_dist_px = 0.35 * ppm
        is_closed = n_pts >= 6 and math.hypot(dedup[0][0] - dedup[-1][0], dedup[0][1] - dedup[-1][1]) <= close_dist_px

        # Cumulative arc-length in mm to evaluate cusps strictly at >= nib-width chord scale (never 1-px staircase!)
        cum_mm = np.zeros(n_pts, dtype=np.float64)
        for i in range(1, n_pts):
            cum_mm[i] = cum_mm[i - 1] + math.hypot(dedup[i][0] - dedup[i - 1][0], dedup[i][1] - dedup[i - 1][1]) * mm_per_px

        cusp_chord_mm = max(0.42, 0.95 * w_nib_mm)
        min_anchor_sep_mm = max(0.55, 1.20 * w_nib_mm)
        cand_cusps: List[Tuple[int, float]] = []
        if cum_mm[-1] >= 2.0 * cusp_chord_mm:
            for i in range(1, n_pts - 1):
                if cum_mm[i] < cusp_chord_mm or (cum_mm[-1] - cum_mm[i]) < cusp_chord_mm:
                    continue
                i_prev = int(np.searchsorted(cum_mm, cum_mm[i] - cusp_chord_mm, side="right")) - 1
                i_prev = max(0, i_prev)
                i_next = int(np.searchsorted(cum_mm, cum_mm[i] + cusp_chord_mm, side="left"))
                i_next = min(n_pts - 1, i_next)
                v1y, v1x = dedup[i][0] - dedup[i_prev][0], dedup[i][1] - dedup[i_prev][1]
                v2y, v2x = dedup[i_next][0] - dedup[i][0], dedup[i_next][1] - dedup[i][1]
                n1, n2 = math.hypot(v1y, v1x), math.hypot(v2y, v2x)
                if n1 >= 0.35 * ppm and n2 >= 0.35 * ppm:
                    cos_t = (v1y * v2y + v1x * v2x) / (n1 * n2)
                    if cos_t < -0.20:
                        cand_cusps.append((i, cos_t))

        anchors = [0]
        for idx_c, cos_c in cand_cusps:
            if len(anchors) == 1 or (cum_mm[idx_c] - cum_mm[anchors[-1]]) >= min_anchor_sep_mm:
                anchors.append(idx_c)
            else:
                prev_idx = anchors[-1]
                prev_cos = next((c for k, c in cand_cusps if k == prev_idx), 0.0)
                if cos_c < prev_cos:
                    anchors[-1] = idx_c
        anchors.append(n_pts - 1)

        # Physical hand-inertia smoothing scale (0.22 mm) and sub-nib RDP simplification (0.052 mm)
        py_all = np.array([p[0] for p in dedup], dtype=np.float64)
        px_all = np.array([p[1] for p in dedup], dtype=np.float64)
        step_ds = float(np.median(np.hypot(np.diff(py_all), np.diff(px_all)))) if n_pts >= 3 else (0.08 * ppm)
        step_ds = max(0.04 * ppm, step_ds)
        sig_pts = max(1.0, min(4.5, (0.22 * ppm) / max(1e-3, step_ds)))
        rdp_eps_px = 0.052 * ppm
        simplified: List[Tuple[float, float, float]] = []

        if is_closed and len(anchors) == 2:
            cy_g1 = gaussian_filter1d(py_all[:-1], sigma=sig_pts, mode="wrap")
            cx_g1 = gaussian_filter1d(px_all[:-1], sigma=sig_pts, mode="wrap")
            # Taubin zero-shrinkage fairing: removes staircase jitter without shrinking tight bowls inward
            cy_a = cy_g1 + 0.58 * (cy_g1 - gaussian_filter1d(cy_g1, sigma=sig_pts, mode="wrap"))
            cx_a = cx_g1 + 0.58 * (cx_g1 - gaussian_filter1d(cx_g1, sigma=sig_pts, mode="wrap"))
            ring_sm = [(float(cy_a[k]), float(cx_a[k]), dedup[k][2]) for k in range(len(cy_a))]
            ring_sm.append(ring_sm[0])
            simplified = self._rdp_simplify_with_width(ring_sm, epsilon=rdp_eps_px)
        else:
            for a_i in range(len(anchors) - 1):
                piece = dedup[anchors[a_i]:anchors[a_i + 1] + 1]
                if len(piece) >= 4:
                    py_a = np.array([p[0] for p in piece], dtype=np.float64)
                    px_a = np.array([p[1] for p in piece], dtype=np.float64)
                    sy0, sx0, sy1, sx1 = py_a[0], px_a[0], py_a[-1], px_a[-1]
                    py_g1 = gaussian_filter1d(py_a, sigma=sig_pts, mode="nearest")
                    px_g1 = gaussian_filter1d(px_a, sigma=sig_pts, mode="nearest")
                    py_a = py_g1 + 0.55 * (py_g1 - gaussian_filter1d(py_g1, sigma=sig_pts, mode="nearest"))
                    px_a = px_g1 + 0.55 * (px_g1 - gaussian_filter1d(px_g1, sigma=sig_pts, mode="nearest"))
                    py_a[0], px_a[0], py_a[-1], px_a[-1] = sy0, sx0, sy1, sx1
                    piece = [(float(py_a[k]), float(px_a[k]), piece[k][2]) for k in range(len(piece))]
                simp_p = self._rdp_simplify_with_width(piece, epsilon=rdp_eps_px)
                simplified.extend(simp_p[1:] if simplified else simp_p)

        # Resample at half-nib physical cadence (~0.24 mm) for smooth bevel/pressure profiling
        # followed by gentle C^2 curve fairing so resampled knots follow a smooth arc rather than flat chords
        resample_step_px = max(0.8, 0.24 * ppm)
        resampled: List[Tuple[float, float, float]] = [simplified[0]]
        cusp_coords = [(dedup[a][0], dedup[a][1]) for a in anchors[1:-1]]
        for k in range(len(simplified) - 1):
            p0, p1 = simplified[k], simplified[k + 1]
            seg_d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if seg_d > resample_step_px * 1.35:
                n_sub = int(math.ceil(seg_d / resample_step_px))
                for s_i in range(1, n_sub):
                    t = float(s_i) / float(n_sub)
                    resampled.append((
                        p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1]),
                        p0[2] + t * (p1[2] - p0[2]),
                    ))
            resampled.append(p1)

        n_r = len(resampled)
        coords_yx = [[float(p[0]), float(p[1])] for p in resampled]
        if n_r >= 5 and not cusp_coords:
            ry_a = np.array([c[0] for c in coords_yx], dtype=np.float64)
            rx_a = np.array([c[1] for c in coords_yx], dtype=np.float64)
            if is_closed:
                ry_s = gaussian_filter1d(ry_a[:-1], sigma=0.75, mode="wrap")
                rx_s = gaussian_filter1d(rx_a[:-1], sigma=0.75, mode="wrap")
                for i in range(n_r - 1):
                    coords_yx[i] = [float(ry_s[i]), float(rx_s[i])]
                coords_yx[-1] = list(coords_yx[0])
            else:
                ry_s = gaussian_filter1d(ry_a, sigma=0.75, mode="nearest")
                rx_s = gaussian_filter1d(rx_a, sigma=0.75, mode="nearest")
                ry_s[0], rx_s[0], ry_s[-1], rx_s[-1] = ry_a[0], rx_a[0], ry_a[-1], rx_a[-1]
                for i in range(n_r):
                    coords_yx[i] = [float(ry_s[i]), float(rx_s[i])]

        # Coupled physical nib bevel + inferred muscular pressure p(s)
        w_edge_mm = (0.48 if is_serafini else self.MIN_NIB_CHISEL_RATIO) * w_nib_mm
        geom_w_mm = np.zeros(n_r, dtype=np.float64)
        obs_w_mm = np.zeros(n_r, dtype=np.float64)

        for i in range(n_r):
            if i == 0:
                dy, dx = coords_yx[1][0] - coords_yx[0][0], coords_yx[1][1] - coords_yx[0][1]
            elif i == n_r - 1:
                dy, dx = coords_yx[i][0] - coords_yx[i - 1][0], coords_yx[i][1] - coords_yx[i - 1][1]
            else:
                dy, dx = coords_yx[i + 1][0] - coords_yx[i - 1][0], coords_yx[i + 1][1] - coords_yx[i - 1][1]
            theta = math.atan2(dy, dx)
            sin_b = abs(math.sin(theta - self.nib_angle_rad))
            if is_serafini:
                geom_w_mm[i] = w_nib_mm * (0.80 + 0.20 * sin_b)
            else:
                geom_w_mm[i] = math.hypot(w_nib_mm * sin_b, w_edge_mm)

            iy = max(0, min(h - 1, int(round(coords_yx[i][0]))))
            ix = max(0, min(w - 1, int(round(coords_yx[i][1]))))
            edt_mm = float(dist_map[iy, ix]) * mm_per_px
            obs_w_mm[i] = max(w_edge_mm, 2.0 * edt_mm)

        # Infer normalized pressure p(s) in [0.55, 1.55] with muscular smoothing along arc-length
        raw_p = np.clip(obs_w_mm / np.maximum(1e-4, geom_w_mm), 0.55, 1.55)
        if n_r >= 4:
            p_mode = "wrap" if is_closed else "nearest"
            smooth_p = gaussian_filter1d(raw_p, sigma=1.25, mode=p_mode)
        else:
            smooth_p = raw_p

        # Combine pressure * bevel width, strictly capped by physical MAX_PRESSED_SPREAD_MM AND smooth hole clearance!
        base_w_px: List[float] = []
        hole_caps_arr = np.full(n_r, self.MAX_PRESSED_SPREAD_MM * ppm, dtype=np.float64)
        for i in range(n_r):
            iy = max(0, min(h - 1, int(round(coords_yx[i][0]))))
            ix = max(0, min(w - 1, int(round(coords_yx[i][1]))))
            if dist_to_hole is not None:
                dh_px = float(dist_to_hole[iy, ix])
                if dh_px < 0.75 * ppm:
                    hole_caps_arr[i] = max(0.10 * ppm, 2.0 * (dh_px - 0.035 * ppm))
        if dist_to_hole is not None and n_r >= 4:
            hole_caps_arr = np.minimum(
                hole_caps_arr,
                gaussian_filter1d(hole_caps_arr, sigma=0.90, mode="wrap" if is_closed else "nearest"),
            )
        hole_caps_px = [float(v) for v in hole_caps_arr]
        for i in range(n_r):
            w_mm_i = float(np.clip(smooth_p[i] * geom_w_mm[i], w_edge_mm, self.MAX_PRESSED_SPREAD_MM))
            base_w_px.append(min(w_mm_i * ppm, hole_caps_px[i]))

        # Fine silhouette IoU + hole-preservation calibration within physical bounds [0.88, 1.18]
        best_gain = 1.0
        best_obj = -1e9
        holes_mask = (dist_to_hole == 0) if dist_to_hole is not None else None
        hole_tot = int(np.sum(holes_mask)) if holes_mask is not None else 0

        for gain in (0.88, 0.98, 1.08, 1.18):
            scaled = [
                (
                    coords_yx[i][0],
                    coords_yx[i][1],
                    min(hole_caps_px[i], self.MAX_PRESSED_SPREAD_MM * ppm, base_w_px[i] * gain),
                )
                for i in range(n_r)
            ]
            pred = self._rasterize_strokes_mask([scaled], (h, w), scale_w=1.0)
            inter = int(np.sum(pred & clean_comp))
            union = int(np.sum(pred | clean_comp))
            iou = (inter / union) if union > 0 else 0.0
            hole_inv = (int(np.sum(pred & holes_mask)) / hole_tot) if hole_tot > 0 else 0.0
            obj = iou - 0.45 * hole_inv
            if obj > best_obj:
                best_obj = obj
                best_gain = gain

        w_arr = np.array(
            [
                min(
                    hole_caps_px[i],
                    self.MAX_PRESSED_SPREAD_MM * ppm,
                    max(w_edge_mm * ppm, base_w_px[i] * best_gain),
                )
                for i in range(n_r)
            ],
            dtype=np.float64,
        )
        if n_r >= 4:
            w_arr = np.minimum(
                hole_caps_arr,
                gaussian_filter1d(w_arr, sigma=0.80, mode="wrap" if is_closed else "nearest"),
            )

        final_pts: List[Tuple[float, float, float]] = [
            (round(coords_yx[i][0], 2), round(coords_yx[i][1], 2), round(float(w_arr[i]), 2))
            for i in range(n_r)
        ]
        if is_closed and len(final_pts) >= 4:
            final_pts[-1] = final_pts[0]

        return final_pts, float(np.mean(smooth_p))

    @staticmethod
    def _splice_cusp_tongues(
        segs: List[Dict[str, Any]],
        px_per_mm: float = 8.0,
    ) -> List[Dict[str, Any]]:
        """
        Preserves genuine `'3'`-shaped middle cusps (`arch_top -> cusp_tip -> arch_bottom`)
        by splicing protruding degree-1 junction tongues strictly within `[0.50 mm, 1.40 mm]`
        ONLY when the two flanking branches do NOT already form a smooth straight stem (`cos < 0.45`).
        Sub-nib `< 0.50 mm` corner bumps along `'a'`/`'o'` bowls are never spliced as cusps.
        """
        if len(segs) < 3:
            return segs
        mm_per_px = 1.0 / px_per_mm
        min_tongue_mm = 0.50
        max_tongue_mm = 1.40
        junc_tol_px = 0.28 * px_per_mm
        k_tan = max(2, int(round(0.32 * px_per_mm)))
        work = [dict(s, points=list(s.get("points", []))) for s in segs if len(s.get("points", [])) >= 2]
        if len(work) < 3:
            return segs

        merged_any = True
        while merged_any and len(work) >= 3:
            merged_any = False
            for t_idx, t_seg in enumerate(work):
                t_pts = t_seg["points"]
                arc_mm = CalligraphicVectorizer._polyline_arc_mm(t_pts, mm_per_px)
                if arc_mm < min_tongue_mm or arc_mm > max_tongue_mm:
                    continue
                for j_end in (0, -1):
                    jy, jx = float(t_pts[j_end][0]), float(t_pts[j_end][1])
                    tip_end = -1 if j_end == 0 else 0
                    ty_tip, tx_tip = float(t_pts[tip_end][0]), float(t_pts[tip_end][1])
                    if math.hypot(ty_tip - jy, tx_tip - jx) * mm_per_px < min_tongue_mm:
                        continue
                    tip_touches = any(
                        o_idx != t_idx
                        and (
                            math.hypot(ty_tip - o["points"][0][0], tx_tip - o["points"][0][1]) <= junc_tol_px
                            or math.hypot(ty_tip - o["points"][-1][0], tx_tip - o["points"][-1][1]) <= junc_tol_px
                        )
                        for o_idx, o in enumerate(work)
                    )
                    if tip_touches:
                        continue
                    neighbors = []
                    for o_idx, o in enumerate(work):
                        if o_idx == t_idx or len(o["points"]) < 3:
                            continue
                        d0 = math.hypot(jy - o["points"][0][0], jx - o["points"][0][1])
                        d1 = math.hypot(jy - o["points"][-1][0], jx - o["points"][-1][1])
                        if d0 <= junc_tol_px or d1 <= junc_tol_px:
                            neighbors.append((o_idx, 0 if d0 <= d1 else -1))
                    if len(neighbors) == 2:
                        (idx_a, end_a), (idx_b, end_b) = neighbors
                        if idx_a == idx_b:
                            continue
                        if work[idx_a].get("type") == "loop" or work[idx_b].get("type") == "loop":
                            continue
                        pts_a = list(work[idx_a]["points"])
                        pts_b = list(work[idx_b]["points"])
                        if end_a == 0:
                            pts_a = pts_a[::-1]
                        if end_b == -1:
                            pts_b = pts_b[::-1]
                        # Never splice an out-and-back spike into a closed loop ('a'/'d'/'q') or smooth stem!
                        if math.hypot(pts_a[0][0] - pts_b[-1][0], pts_a[0][1] - pts_b[-1][1]) * mm_per_px < 0.85:
                            continue
                        if JunctionResolver._tangent_cosine(pts_a, pts_b, k=k_tan) >= 0.45:
                            continue
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

    @staticmethod
    def _rdp_simplify_with_width(
        pts: List[Tuple[float, float, float]],
        epsilon: float = 0.25,
    ) -> List[Tuple[float, float, float]]:
        """Ramer-Douglas-Peucker trajectory simplification preserving closed loops and local widths."""
        if len(pts) <= 3:
            return pts

        if np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= epsilon * 8.0 and len(pts) >= 6:
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
        return [pts[0], pts[-1]]

    @staticmethod
    def _rasterize_strokes_mask(
        stroke_pts_list: List[List[Tuple[float, float, float]]],
        shape: Tuple[int, int],
        scale_w: float = 1.0,
    ) -> np.ndarray:
        """Renders candidate variable-width calligraphic strokes onto a 2D boolean mask for silhouette IoU."""
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
        ppm: float = 8.0,
    ) -> List[Tuple[float, float, float]]:
        """
        1. Snaps T-junction endpoints onto parent strokes across genuine ink (`0.0 mm` junction gap).
        2. Completes free eroded endpoints via a 2-pass geodesic forward cone bounded strictly in mm (`<= 0.85 mm`).
        """
        if len(pts) < 2:
            return pts
        h, w = clean_comp.shape
        close_tol = 0.35 * ppm
        if len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol:
            return pts

        out = list(pts)
        snap_r = min(0.85 * ppm, med_half_w * 2.2)
        prior_flat = [p for st in prior_strokes for p in st] if prior_strokes else []

        for end_flag in (0, -1):
            ey, ex, ew = float(out[end_flag][0]), float(out[end_flag][1]), float(out[end_flag][2])
            if prior_flat:
                best_p = None
                best_d = snap_r + 1.0
                for pp in prior_flat:
                    d = math.hypot(ey - pp[0], ex - pp[1])
                    if d < best_d:
                        best_d = d
                        best_p = pp
                if best_p is not None and 0.05 * ppm <= best_d <= snap_r:
                    my, mx = int(round(0.5 * (ey + best_p[0]))), int(round(0.5 * (ex + best_p[1])))
                    if 0 <= my < h and 0 <= mx < w and clean_comp[my, mx]:
                        anchor_pt = (round(float(best_p[0]), 2), round(float(best_p[1]), 2), round(float(ew), 2))
                        out = ([anchor_pt] + out) if end_flag == 0 else (out + [anchor_pt])
                        continue

            max_fwd_px = 0.55 * ppm
            max_lat_px = 0.20 * ppm
            ey, ex, ew = float(out[end_flag][0]), float(out[end_flag][1]), float(out[end_flag][2])
            k_step = min(len(out) - 1, max(1, int(round(0.35 * ppm))))
            if end_flag == 0:
                ty, tx = out[0][0] - out[k_step][0], out[0][1] - out[k_step][1]
            else:
                ty, tx = out[-1][0] - out[-1 - k_step][0], out[-1][1] - out[-1 - k_step][1]
            tn = math.hypot(ty, tx)
            if tn >= 1e-4:
                ty, tx = ty / tn, tx / tn
                py_dir, px_dir = -tx, ty
                iy0, ix0 = max(0, min(h - 1, int(round(ey)))), max(0, min(w - 1, int(round(ex))))
                local_r = min(0.50 * ppm, float(dist_map[iy0, ix0]))
                fwd_lim = min(max_fwd_px, local_r * 1.45 + 0.12 * ppm)
                lat_lim = min(max_lat_px, local_r * 0.55 + 0.06 * ppm)
                r_box = max(1, int(math.ceil(fwd_lim)))
                y_lo, y_hi = max(0, iy0 - r_box), min(h, iy0 + r_box + 1)
                x_lo, x_hi = max(0, ix0 - r_box), min(w, ix0 + r_box + 1)
                ys_sub, xs_sub = np.where(clean_comp[y_lo:y_hi, x_lo:x_hi])
                best_tip = None
                best_score = 0.08 * ppm
                for dy_i, dx_i in zip(ys_sub, xs_sub):
                    cy_i, cx_i = float(y_lo + dy_i), float(x_lo + dx_i)
                    vy, vx = cy_i - ey, cx_i - ex
                    proj = vy * ty + vx * tx
                    if proj < 0.08 * ppm or proj > fwd_lim:
                        continue
                    perp = abs(vy * py_dir + vx * px_dir)
                    if perp > lat_lim or perp > proj * 0.45:
                        continue
                    mid_y, mid_x = int(round(0.5 * (ey + cy_i))), int(round(0.5 * (ex + cx_i)))
                    q_y, q_x = int(round(0.75 * cy_i + 0.25 * ey)), int(round(0.75 * cx_i + 0.25 * ex))
                    if not (clean_comp[mid_y, mid_x] and clean_comp[q_y, q_x]):
                        continue
                    sc = proj - 0.55 * perp + 0.20 * float(dist_map[int(cy_i), int(cx_i)])
                    if sc > best_score:
                        best_score = sc
                        w_tip = max(0.12 * ppm, float(dist_map[int(cy_i), int(cx_i)]) * 1.80)
                        best_tip = (round(cy_i, 2), round(cx_i, 2), round(w_tip, 2))
                if best_tip is not None and math.hypot(best_tip[0] - ey, best_tip[1] - ex) >= 0.10 * ppm:
                    out = ([best_tip] + out) if end_flag == 0 else (out + [best_tip])

        return out
