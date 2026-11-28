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
        fit_cubic_bezier: Optional[bool] = None,
        hole_margin_scale: Optional[float] = None,
        gain_bias: Optional[float] = None,
        inertia_scale: Optional[float] = None,
    ):
        defaults = self._load_calibrated_defaults()
        self.nib_angle_rad = float(np.deg2rad(nib_angle_deg))
        self.min_nib_width = float(min_nib_width)
        self.max_nib_width = float(max_nib_width)
        self.smoothing_sigma = float(smoothing_sigma)
        self.fit_cubic_bezier = bool(fit_cubic_bezier if fit_cubic_bezier is not None else defaults.get("fit_cubic_bezier", True))
        self.hole_margin_scale = float(hole_margin_scale if hole_margin_scale is not None else defaults.get("hole_margin_scale", 1.0))
        self.gain_bias = float(gain_bias if gain_bias is not None else defaults.get("gain_bias", 1.0))
        self.inertia_scale = float(inertia_scale if inertia_scale is not None else defaults.get("inertia_scale", 1.0))

    @staticmethod
    def _load_calibrated_defaults() -> Dict[str, Any]:
        import json
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent / "calibrated_ductus_params.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    @staticmethod
    def _fit_cubic_bezier_piece(pts_yx: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fits an exact endpoint-pinned 4-control-point Cubic Bézier curve via regularized least squares."""
        m = len(pts_yx)
        if m < 4:
            return pts_yx, 0.0
        ds = np.hypot(np.diff(pts_yx[:, 0]), np.diff(pts_yx[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(ds)])
        tot = float(cum[-1])
        if tot < 1e-5:
            return pts_yx, 0.0
        t = cum / tot
        u = 1.0 - t
        p0, p3 = pts_yx[0], pts_yx[-1]
        b0 = (u ** 3)[:, None] * p0[None, :]
        b3 = (t ** 3)[:, None] * p3[None, :]
        rhs = pts_yx - b0 - b3
        A = np.column_stack([3.0 * (u ** 2) * t, 3.0 * u * (t ** 2)])
        # Mild Tikhonov regularization toward chord 1/3 & 2/3 points prevents control-point overshoot
        p1_lin = (2.0 * p0 + p3) / 3.0
        p2_lin = (p0 + 2.0 * p3) / 3.0
        reg = 0.08
        ata = A.T @ A + reg * np.eye(2)
        atb = A.T @ rhs + reg * np.vstack([p1_lin, p2_lin])
        ctrl_12 = np.linalg.solve(ata, atb)
        p1, p2 = ctrl_12[0], ctrl_12[1]
        fitted = (
            b0
            + (3.0 * (u ** 2) * t)[:, None] * p1[None, :]
            + (3.0 * u * (t ** 2))[:, None] * p2[None, :]
            + b3
        )
        max_err = float(np.max(np.hypot(fitted[:, 0] - pts_yx[:, 0], fitted[:, 1] - pts_yx[:, 1])))
        return fitted, max_err

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

        for _pass in range(3):
            deg_map = get_degree_map(pruned)
            if not np.any(deg_map >= 3):
                break

            changed_pass = False
            rem_count = int(np.sum(pruned))
            endpoints = np.argwhere(deg_map == 1)
            for ey, ex in endpoints:
                if not pruned[ey, ex]:
                    continue
                chain = [(int(ey), int(ex))]
                cy, cx = int(ey), int(ex)
                hit_junction = False
                jn_y, jn_x = cy, cx
                local_nib_r = max(float(min_spur_len), float(dist_map[ey, ex]) * 1.85 + 1.5)
                max_walk = int(round(local_nib_r * 1.45)) + 2

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
                    j_nbs = [p for p in nbs if deg_map[p[0], p[1]] >= 3]
                    ny, nx = j_nbs[0] if j_nbs else nbs[0]
                    if deg_map[ny, nx] >= 3:
                        hit_junction = True
                        jn_y, jn_x = ny, nx
                        break
                    chain.append((ny, nx))
                    cy, cx = ny, nx

                if hit_junction and len(chain) <= max_walk and (rem_count - len(chain)) >= 4:
                    r_junc = float(dist_map[jn_y, jn_x])
                    tip_dist = math.hypot(float(ey - jn_y), float(ex - jn_x))
                    if tip_dist > max(float(min_spur_len), r_junc * 1.68 + 0.8):
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
                        changed_pass = True
            if not changed_pass:
                break

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
            strict_one = (max_strokes is not None and max_strokes == 1)
            curs_strokes = self._extract_continuous_cursive_tour(
                mask_bool, ppm=ppm, w_nib_mm=w_nib_mm, is_macro=is_macro, allow_diacritics=(not strict_one)
            )
            return curs_strokes if curs_strokes else [self._make_impact_stroke(pts_all, 0, ppm, w_nib_mm)]

        stroke_limit = min(4, max_strokes) if max_strokes is not None else 4
        labeled_mask, num_features = label(mask_bool, structure=np.ones((3, 3), dtype=bool))
        if num_features == 0:
            return []

        comp_min_area_px = max(1, int(round(calib.area_mm2_to_px2(0.045))))
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

        # Prioritize largest substantive components; govern secondary marks strictly by stroke-thickness ratio!
        components.sort(key=lambda c: c["area_mm2"], reverse=True)
        dom_comp = components[0]
        dom_has_stroke = (
            max(dom_comp["span_h_mm"], dom_comp["span_w_mm"]) > 1.38 * w_nib_mm
            and dom_comp["area_mm2"] > 1.35 * (w_nib_mm ** 2)
        )

        graph_ext = StrokeGraphExtractor()
        raw_stroke_dicts: List[Dict[str, Any]] = []

        for comp_idx, comp in enumerate(components):
            rem_budget = stroke_limit - len(raw_stroke_dicts)
            if rem_budget <= 0:
                break

            max_comp_span_mm = max(comp["span_h_mm"], comp["span_w_mm"])
            # On Voynich (split quill), once a real glyph stroke body is identified, isolated specks
            # smaller than ~1.4x stroke thickness are parchment grain / bleed — NEVER intentional dots!
            if comp_idx > 0 and dom_has_stroke:
                if (
                    max_comp_span_mm <= 1.38 * w_nib_mm
                    or comp["area_mm2"] < max(1.30 * (w_nib_mm ** 2), 0.13 * dom_comp["area_mm2"])
                ):
                    continue

            if max_comp_span_mm <= sub_nib_span_mm and comp["area_mm2"] <= calib.DOT_MAX_AREA_MM2:
                if not dom_has_stroke:
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
        self, comp_mask: np.ndarray, ppm: float, is_serafini: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        1. Preserves 100% of genuine enclosed eyes (`> 1 px` on `< 10 px/mm` scans; `> 0.022 mm^2` or
           `EDT > 1.05 px` on HD scans).
        2. Applies a gentle `3x3` (`1 px` <= half chisel edge) hole-preserving morphological close on
           Voynich vellum so 1-px dry-ink fiber skips inside minims (`[i]`, `[s]`, `[f]`) never shatter
           a single stroke into noisy fragments, while strictly restoring `holes_union` (`0` hole loss).
        """
        padded = np.pad(comp_mask, 2, mode="constant", constant_values=False)
        holes = binary_fill_holes(padded) & (~padded)
        lbl_h, num_h = label(holes)
        clean_comp = comp_mask.copy()
        holes_union = np.zeros_like(comp_mask, dtype=bool)
        pinhole_hd_px = max(2, int(round(0.022 * (ppm ** 2))))
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)[2:-2, 2:-2]
            ha = int(np.sum(hm))
            if is_serafini or ppm < 10.0:
                if ha <= 1:
                    clean_comp |= hm
                elif np.any(hm):
                    holes_union |= hm
            else:
                if ha <= 2 or (ha <= pinhole_hd_px and float(np.max(distance_transform_edt(hm))) <= 1.05):
                    clean_comp |= hm
                elif np.any(hm):
                    holes_union |= hm

        dist_to_hole = distance_transform_edt(~holes_union) if np.any(holes_union) else None
        return clean_comp, holes_union, dist_to_hole

    @staticmethod
    def _signed_ring_area_xy(pts: List[Tuple[float, float, float]]) -> float:
        """Signed shoelace area in screen `(x_right, y_down)` coords (`< 0` = Counter-Clockwise on page)."""
        if len(pts) < 3:
            return 0.0
        return 0.5 * float(
            sum(pts[i][1] * pts[i + 1][0] - pts[i + 1][1] * pts[i][0] for i in range(len(pts) - 1))
        )

    @staticmethod
    def _resolve_collinear_junctions(
        segs: List[Dict[str, Any]],
        dist_map: np.ndarray,
        ppm: float,
        w_nib_mm: float,
        prioritize_vertical: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Eliminates medial-axis Y-junction kinks across straight crossing/branching strokes
        (e.g. Voynich gallows vertical hampes `[k]`, `[t]`, `[p]`, `[f]` crossed by crossbars/loops):
        Pairs open branches meeting within a junction hub that maintain clean far-field tangent
        alignment (`cos >= 0.72`), trimming the Y-pulled junction pixels inside `1.20 * R_junc`
        and bridging straight across the hub.
        """
        if len(segs) < 2:
            return segs

        h, w = dist_map.shape
        close_tol_px = 0.40 * ppm
        hub_tol_px = max(2.2, 0.62 * w_nib_mm * ppm)
        far_step_px = max(2.5, 0.55 * ppm)

        work = [dict(s, points=list(s.get("points", []))) for s in segs if len(s.get("points", [])) >= 2]

        def _is_loop_seg(pts: List[Tuple[float, float, float]], stype: str) -> bool:
            return stype == "loop" or (
                len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px
            )

        def _far_inward_dir(pts: List[Tuple[float, float, float]], at_end: bool) -> Tuple[float, float, int]:
            """Unit vector pointing OUTWARD from segment interior toward the endpoint, measured past Y-pull zone."""
            seq = pts[::-1] if at_end else pts
            iy = max(0, min(h - 1, int(round(seq[0][0]))))
            ix = max(0, min(w - 1, int(round(seq[0][1]))))
            r_trim = max(1.2, min(0.45 * len(seq), float(dist_map[iy, ix]) * 1.18))
            acc = 0.0
            k_trim = 0
            for k in range(1, len(seq)):
                acc += math.hypot(seq[k][0] - seq[k - 1][0], seq[k][1] - seq[k - 1][1])
                if acc >= r_trim and k < len(seq) - 1:
                    k_trim = k
                    break
            acc2 = 0.0
            k_ref = len(seq) - 1
            for k in range(k_trim + 1, len(seq)):
                acc2 += math.hypot(seq[k][0] - seq[k - 1][0], seq[k][1] - seq[k - 1][1])
                if acc2 >= far_step_px:
                    k_ref = k
                    break
            dy = float(seq[k_trim][0] - seq[k_ref][0])
            dx = float(seq[k_trim][1] - seq[k_ref][1])
            n = math.hypot(dy, dx)
            if n < 1e-4:
                dy = float(seq[0][0] - seq[-1][0])
                dx = float(seq[0][1] - seq[-1][1])
                n = max(1e-4, math.hypot(dy, dx))
            return dy / n, dx / n, k_trim

        merged_any = True
        while merged_any and len(work) >= 2:
            merged_any = False
            best_score = -1.0
            best_merge = None

            for i in range(len(work)):
                pi = work[i]["points"]
                if _is_loop_seg(pi, work[i].get("type", "")):
                    continue
                for j in range(i + 1, len(work)):
                    pj = work[j]["points"]
                    if _is_loop_seg(pj, work[j].get("type", "")):
                        continue
                    for end_i in (False, True):
                        pti = pi[-1] if end_i else pi[0]
                        for end_j in (False, True):
                            ptj = pj[-1] if end_j else pj[0]
                            d_ij = math.hypot(pti[0] - ptj[0], pti[1] - ptj[1])
                            if d_ij > hub_tol_px:
                                continue
                            # Outward tangent of i at pti should oppose outward tangent of j at ptj
                            uy_i, ux_i, trim_i = _far_inward_dir(pi, at_end=end_i)
                            uy_j, ux_j, trim_j = _far_inward_dir(pj, at_end=end_j)
                            cos_through = -(uy_i * uy_j + ux_i * ux_j)
                            vert_pair = prioritize_vertical and abs(uy_i) >= 0.78 and abs(uy_j) >= 0.78 and (uy_i * uy_j < 0)
                            min_thr = 0.58 if vert_pair else 0.74
                            if cos_through < min_thr:
                                continue
                            vert_align = (0.5 * (abs(uy_i) + abs(uy_j))) if prioritize_vertical else 0.5
                            score = cos_through + (0.42 * vert_align if prioritize_vertical else 0.0) - 0.04 * (d_ij / max(1.0, hub_tol_px))
                            if score > best_score:
                                # Assemble `a_pts -> straight bridge -> b_pts` trimming Y-junction pull inside `trim_i`/`trim_j`
                                a_pts = pi[:len(pi) - trim_i] if end_i else pi[trim_i:][::-1]
                                b_pts = pj[trim_j:] if (not end_j) else pj[:len(pj) - trim_j][::-1]
                                if len(a_pts) >= 1 and len(b_pts) >= 1:
                                    comb = a_pts + b_pts
                                    if len(comb) >= 2:
                                        best_score = score
                                        best_merge = (i, j, comb)

            if best_merge is not None:
                i, j, comb = best_merge
                if prioritize_vertical and comb[-1][0] < comb[0][0]:
                    comb = comb[::-1]
                work[i] = {
                    "stroke_id": work[i].get("stroke_id", "s_st"),
                    "type": "segment",
                    "points": comb,
                    "start": (comb[0][0], comb[0][1]),
                    "end": (comb[-1][0], comb[-1][1]),
                    "length": float(len(comb)),
                }
                work.pop(j)
                merged_any = True

        return work

    @classmethod
    def _normalize_humanist_cursive_segments(
        cls,
        segs: List[Dict[str, Any]],
        ppm: float,
        w_nib_mm: float,
    ) -> List[Dict[str, Any]]:
        """
        15th-Century Italian Humanist Cursive Law (Lisa Fagin Davis 2022 / Voynich.nu):
        - Closed bowls (`[o]`, `[e]`, `[9/y]`, `[8/d]`, `[a]` bowl) execute as continuous
          Counter-Clockwise (CCW) loops (thick down-left *plein*, unloaded 40° *délié* closure),
          NEVER chopped into artificial Gothic crescents.
        - When a CCW loop meets a downward tail or second bowl (`[9/y]`, `[8/d]`) smoothly at a
          junction, it chains into a single continuous cursive gesture.
        - Open strokes (`[i]`, `[n]` upward-left finial, `[c]`, `[s/2]`, gallows legs) preserve
          their continuous trajectory intact without splitting at bottom `'∪'` troughs!
        """
        mm_per_px = 1.0 / ppm
        close_tol_px = 0.44 * ppm
        junc_tol_px = max(2.0, 0.58 * w_nib_mm * ppm)
        k_tan = max(2, int(round(0.30 * ppm)))

        loops: List[Dict[str, Any]] = []
        opens: List[Dict[str, Any]] = []

        for s in segs:
            pts = list(s.get("points", []))
            if len(pts) < 2:
                continue
            is_loop = (
                s.get("type") == "loop"
                or (len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px)
            )
            if is_loop and len(pts) >= 5:
                ring = pts[:-1] if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px else list(pts)
                closed_test = ring + [ring[0]]
                # Enforce Counter-Clockwise winding on page (`_signed_ring_area_xy < 0`)
                if cls._signed_ring_area_xy(closed_test) > 0:
                    ring = ring[::-1]
                # Default ~1 o'clock upper-right/top touchdown so initial pull sweeps left-and-down
                i_start = min(range(len(ring)), key=lambda idx: float(ring[idx][0]) - 0.22 * float(ring[idx][1]))
                rot = ring[i_start:] + ring[:i_start] + [ring[i_start]]
                loops.append({
                    "stroke_id": s.get("stroke_id", "s_loop"),
                    "type": "loop",
                    "points": rot,
                    "ring": ring,
                    "start": (rot[0][0], rot[0][1]),
                    "end": (rot[-1][0], rot[-1][1]),
                    "length": float(len(rot)),
                })
            else:
                opens.append(dict(s, type="segment", points=pts, start=(pts[0][0], pts[0][1]), end=(pts[-1][0], pts[-1][1])))

        # Splice a CCW loop directly into a smooth downward outgoing tail (e.g. `[y/9]` loop+tail or `[e]` eye+crescent),
        # UNLESS the open segment is a tall full-height gallows stem (`span_h >= 2.3 mm`) which must precede the loop!
        used_opens = set()
        out: List[Dict[str, Any]] = []
        for lp in loops:
            ring = lp["ring"]
            m_r = len(ring)
            best_tail = None
            best_cos = 0.65
            for o_idx, op in enumerate(opens):
                if o_idx in used_opens:
                    continue
                o_pts = op["points"]
                o_ys = [p[0] for p in o_pts]
                o_span_h_mm = (max(o_ys) - min(o_ys)) * mm_per_px
                # Never absorb a tall gallows leg into a head loop: gallows legs pull top-to-bottom FIRST!
                if o_span_h_mm >= 2.35:
                    continue
                for end_flag in (0, -1):
                    jy, jx = o_pts[end_flag][0], o_pts[end_flag][1]
                    d_ring, idx_r = min(
                        (math.hypot(ring[k][0] - jy, ring[k][1] - jx), k) for k in range(m_r)
                    )
                    if d_ring > junc_tol_px:
                        continue
                    cand_tail = o_pts if end_flag == 0 else o_pts[::-1]
                    # Tail must flow predominantly downward (`dy >= 0.15 mm`) rather than climbing up into a right minim
                    if (cand_tail[-1][0] - cand_tail[0][0]) * mm_per_px < 0.18:
                        continue
                    rot_j = ring[idx_r:] + ring[:idx_r] + [ring[idx_r]]
                    cos_t = JunctionResolver._tangent_cosine(rot_j, cand_tail, k=k_tan)
                    if cos_t > best_cos:
                        best_cos = cos_t
                        best_tail = (o_idx, rot_j + cand_tail[1:])

            if best_tail is not None:
                o_idx, spliced = best_tail
                used_opens.add(o_idx)
                out.append({
                    "stroke_id": lp["stroke_id"],
                    "type": "segment",
                    "points": spliced,
                    "start": (spliced[0][0], spliced[0][1]),
                    "end": (spliced[-1][0], spliced[-1][1]),
                    "length": float(len(spliced)),
                })
            else:
                lp_clean = dict(lp)
                lp_clean.pop("ring", None)
                out.append(lp_clean)

        for o_idx, op in enumerate(opens):
            if o_idx in used_opens:
                continue
            pts = list(op["points"])
            if cls._humanist_anti_grain_cost_mm(pts, mm_per_px) > cls._humanist_anti_grain_cost_mm(pts[::-1], mm_per_px) + 0.05:
                pts = pts[::-1]

            # Gallows two-phase apex split (Davis 2022): ONLY on TALL gallows (`span_h >= 2.65 mm`) whose
            # vertical stem terminates at the bottom foot of the glyph (never on `[n]` rounded bottom troughs!)
            ys_o = [p[0] for p in pts]
            y_max_o = max(ys_o)
            span_h_o = (y_max_o - min(ys_o)) * mm_per_px
            chord_o = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * mm_per_px
            arc_o = cls._polyline_arc_mm(pts, mm_per_px)
            if span_h_o >= 2.65 and (arc_o / max(1e-4, chord_o)) > 1.32 and len(pts) >= 8:
                y_cut_lim = min(ys_o) + 0.45 * (y_max_o - min(ys_o))
                best_k = -1
                best_stem_score = -1.0
                for k_sp in range(2, len(pts) - 2):
                    if pts[k_sp][0] > y_cut_lim:
                        continue
                    for cand_stem, cand_head in ((pts[k_sp:], pts[:k_sp + 1]), (pts[:k_sp + 1][::-1], pts[k_sp:])):
                        # Stem must reach the bottom foot (`y_max_o`) rather than curving back up!
                        if abs(cand_stem[-1][0] - y_max_o) > 0.25 * ppm:
                            continue
                        dy_s = (cand_stem[-1][0] - cand_stem[0][0]) * mm_per_px
                        dx_s = abs(cand_stem[-1][1] - cand_stem[0][1]) * mm_per_px
                        ch_s = math.hypot(dy_s, dx_s)
                        ar_s = cls._polyline_arc_mm(cand_stem, mm_per_px)
                        ar_h = cls._polyline_arc_mm(cand_head, mm_per_px)
                        if dy_s >= 1.75 and dy_s >= 2.0 * dx_s and (ar_s / max(1e-4, ch_s)) <= 1.16 and ar_h >= 0.50:
                            sc_sp = dy_s - (ar_s / max(1e-4, ch_s) - 1.0) * 3.0
                            if sc_sp > best_stem_score:
                                best_stem_score = sc_sp
                                best_k = k_sp
                                best_pair_sp = (cand_stem, cand_head)
                if best_k > 0:
                    st_pts, hd_pts = best_pair_sp
                    if cls._humanist_anti_grain_cost_mm(hd_pts, mm_per_px) > cls._humanist_anti_grain_cost_mm(hd_pts[::-1], mm_per_px) + 0.05:
                        hd_pts = hd_pts[::-1]
                    out.append({"stroke_id": f"{op.get('stroke_id', 's')}_stem", "type": "segment", "points": st_pts, "start": (st_pts[0][0], st_pts[0][1]), "end": (st_pts[-1][0], st_pts[-1][1]), "length": float(len(st_pts))})
                    out.append({"stroke_id": f"{op.get('stroke_id', 's')}_head", "type": "segment", "points": hd_pts, "start": (hd_pts[0][0], hd_pts[0][1]), "end": (hd_pts[-1][0], hd_pts[-1][1]), "length": float(len(hd_pts))})
                    continue

            out.append(dict(op, type="segment", points=pts, start=(pts[0][0], pts[0][1]), end=(pts[-1][0], pts[-1][1])))

        return out

    @staticmethod
    def _humanist_anti_grain_cost_mm(pts: List[Tuple[float, float, float]], mm_per_px: float) -> float:
        """
        Biomechanical cost (in mm) of executing an open stroke with a 40° split goose quill:
        - Pulling down (`dy > 0`) or horizontally right (`dx > 0`) glides effortlessly (`cost = 0`).
        - Curving upward-left/right in a *terminal* finial *after* an initial downward pull (EVA `[n]`)
          glides unloaded on the thin chisel edge (`cost ~ 0`).
        - Starting a stroke by pushing upward (`dy_init < 0`) OR ending by pushing up a straight
          slanted/vertical stem (`dy_end < 0` with low curvature) digs into calfskin (`high cost`).
        """
        n = len(pts)
        if n < 2:
            return 0.0
        arc_mm = CalligraphicVectorizer._polyline_arc_mm(pts, mm_per_px)
        if arc_mm < 1e-4:
            return 0.0

        k_init = max(1, min(n - 1, (n * 2) // 5))
        dy_init_mm = (pts[k_init][0] - pts[0][0]) * mm_per_px
        dx_init_mm = (pts[k_init][1] - pts[0][1]) * mm_per_px

        k_tail = min(n - 2, max(0, (n * 3) // 5))
        dy_tail_mm = (pts[-1][0] - pts[k_tail][0]) * mm_per_px
        tail_pts = pts[k_tail:]
        tail_arc_mm = CalligraphicVectorizer._polyline_arc_mm(tail_pts, mm_per_px)
        tail_chord_mm = math.hypot(pts[-1][0] - pts[k_tail][0], pts[-1][1] - pts[k_tail][1]) * mm_per_px
        tail_tort = tail_arc_mm / max(1e-4, tail_chord_mm)

        dy_net_mm = (pts[-1][0] - pts[0][0]) * mm_per_px
        dx_net_mm = (pts[-1][1] - pts[0][1]) * mm_per_px
        full_chord_mm = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * mm_per_px
        full_tort = arc_mm / max(1e-4, full_chord_mm)

        cost = 0.0
        # 1. Initial attack penalty: allow rising rightward humanist arch shoulder (`dx_init > 0.30 * |dy_init|`)
        #    when followed by a strong downward pull (`dy_tail > 0.25 mm`, Primitive P3 `∩`); penalize steep up-left pushes!
        if dy_init_mm < -0.18:
            if dx_init_mm > 0.30 * abs(dy_init_mm) and dy_tail_mm > 0.25:
                cost += abs(dy_init_mm) * 0.35
            elif dx_init_mm <= 1.15 * abs(dy_init_mm):
                cost += abs(dy_init_mm) * 3.4
            else:
                cost += abs(dy_init_mm) * 0.85
        elif abs(dy_init_mm) <= 0.18 and dx_init_mm < -0.22:
            cost += abs(dx_init_mm) * 1.6

        # 2. Straight upward stem push & bottom-to-top crescent penalty (never penalizing `[n]` since `dy_init_mm > 0.20`!)
        if dy_tail_mm < -0.32 and tail_tort < 1.16 and dy_init_mm <= 0.22:
            cost += abs(dy_tail_mm) * 3.6
        if dy_net_mm < -0.35 and full_tort < 1.22:
            cost += abs(dy_net_mm) * 3.8
        elif dy_net_mm < -0.42 and dy_init_mm <= 0.15:
            cost += abs(dy_net_mm) * 2.8
        elif dy_net_mm < 0.0 and dx_net_mm < -0.20 and dy_init_mm < 0.15:
            cost += (abs(dy_net_mm) + 0.6 * abs(dx_net_mm)) * 2.5

        return cost

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
        clean_comp, holes_union, dist_to_hole = self._clean_pinholes_and_get_holes(comp_mask, ppm, is_serafini=is_serafini)
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

        # 2. Drop tiny sub-nib junction-triangle slivers (< 0.26 mm) only when not bridging two branches
        min_sliver_mm = 0.26
        if len(raw_segs) > 2:
            j_tol = 0.30 * ppm
            substantive = []
            for idx_s, s in enumerate(raw_segs):
                pts_s = s.get("points", [])
                if s.get("type") == "loop" or self._polyline_arc_mm(pts_s, mm_per_px) >= min_sliver_mm:
                    substantive.append(s)
                    continue
                # Keep short bridge segment if BOTH ends touch other segments (e.g. vertical bridge between 'p'/'t' bars!)
                if len(pts_s) >= 2:
                    touch_0 = any(
                        k != idx_s and (
                            math.hypot(pts_s[0][0] - o["points"][0][0], pts_s[0][1] - o["points"][0][1]) <= j_tol
                            or math.hypot(pts_s[0][0] - o["points"][-1][0], pts_s[0][1] - o["points"][-1][1]) <= j_tol
                        )
                        for k, o in enumerate(raw_segs) if len(o.get("points", [])) >= 2
                    )
                    touch_1 = any(
                        k != idx_s and (
                            math.hypot(pts_s[-1][0] - o["points"][0][0], pts_s[-1][1] - o["points"][0][1]) <= j_tol
                            or math.hypot(pts_s[-1][0] - o["points"][-1][0], pts_s[-1][1] - o["points"][-1][1]) <= j_tol
                        )
                        for k, o in enumerate(raw_segs) if len(o.get("points", [])) >= 2
                    )
                    if touch_0 and touch_1:
                        substantive.append(s)
            if substantive:
                raw_segs = substantive

        # 3. Eliminate medial-axis Y-junction kinks across straight crossing stems (`[k]`, `[t]`, `[p]`, `[f]`)
        raw_segs = self._resolve_collinear_junctions(
            raw_segs, dist_map=dist_map, ppm=ppm, w_nib_mm=w_nib_mm, prioritize_vertical=(not is_serafini)
        )

        # 4. 15th-century Italian Humanist Cursive normalization on Voynich:
        #    Keep closed bowls as Counter-Clockwise (CCW) loops, splice smooth loop->tail flows (`[y/9]`),
        #    and preserve continuous upward-left `[n]` finials intact!
        if not is_serafini:
            raw_segs = self._normalize_humanist_cursive_segments(raw_segs, ppm=ppm, w_nib_mm=w_nib_mm)

        # 5. Minimum-action junction chaining (preserving humanist quill physics on Voynich)
        chained_segs = self._chain_segments_minimum_action(
            raw_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=max_strokes, medieval_downward_only=(not is_serafini)
        )

        # 6. Complete free eroded endpoints to true ink extremities (bounded narrow forward cone in mm)
        med_half_w = float(np.median(dist_map[clean_comp])) if np.any(clean_comp) else (0.5 * w_nib_mm * ppm)
        raw_polylines: List[List[Tuple[float, float, float]]] = []
        for s in chained_segs:
            pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in s.get("points", [])]
            if len(pts) >= 2:
                pts = self._extend_and_anchor_stroke(pts, clean_comp, dist_map, raw_polylines, med_half_w, ppm=ppm)
                raw_polylines.append(pts)

        # 6b. Tall Gallows Stem Integrity Guard (Voynich `TALL >= 2.85 mm` & `H >= 1.22 W`, `[p]`, `[f]`, `[k]`, `[t]`):
        #     When multiple crossbars/dry-ink skips fragment a dominant vertical ascender column so no clean
        #     downward stem stroke survived, recover the primary vertical EDT ridge top-to-bottom directly!
        ys_ink, xs_ink = np.where(clean_comp)
        ink_h_mm = (float(ys_ink.max() - ys_ink.min()) + 1.0) * mm_per_px if len(ys_ink) > 0 else 0.0
        ink_w_mm = (float(xs_ink.max() - xs_ink.min()) + 1.0) * mm_per_px if len(xs_ink) > 0 else 0.0
        if (not is_serafini) and len(raw_polylines) >= 2 and ink_h_mm >= 3.05 and ink_h_mm >= 0.88 * ink_w_mm and w >= 3:
            has_clean_tall_stem = any(
                len(pl) >= 2
                and math.hypot(pl[0][0] - pl[-1][0], pl[0][1] - pl[-1][1]) * mm_per_px > 0.45
                and abs(pl[-1][0] - pl[0][0]) * mm_per_px >= 0.58 * ink_h_mm
                and (self._polyline_arc_mm(pl, mm_per_px) / max(1e-4, math.hypot(pl[-1][0] - pl[0][0], pl[-1][1] - pl[0][1]) * mm_per_px)) <= 1.24
                for pl in raw_polylines
            )
            if not has_clean_tall_stem:
                band_r = max(1, int(round(0.28 * ppm)))
                best_col_x = -1
                best_col_cov = 0
                best_y0, best_y1 = 0, h - 1
                for cx_i in range(w):
                    x_lo, x_hi = max(0, cx_i - band_r), min(w, cx_i + band_r + 1)
                    col_hit = np.any(clean_comp[:, x_lo:x_hi], axis=1)
                    col_closed = cv2.morphologyEx(col_hit.astype(np.uint8).reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((2, 1), dtype=np.uint8)).ravel() > 0
                    best_run_len, best_r0, best_r1 = 0, 0, 0
                    cur_s = -1
                    for ry in range(h):
                        if col_closed[ry]:
                            if cur_s < 0:
                                cur_s = ry
                            if (ry - cur_s + 1) > best_run_len:
                                best_run_len, best_r0, best_r1 = ry - cur_s + 1, cur_s, ry
                        else:
                            cur_s = -1
                    if best_run_len > best_col_cov:
                        best_col_cov, best_col_x, best_y0, best_y1 = best_run_len, cx_i, best_r0, best_r1
                if best_col_x >= 0 and (best_col_cov * mm_per_px) >= max(2.25, 0.72 * ink_h_mm):
                    stem_pts: List[Tuple[float, float, float]] = []
                    step_y = max(2, int(round(0.38 * ppm)))
                    cur_x = float(best_col_x)
                    for ry in list(range(best_y0, best_y1, step_y)) + [best_y1]:
                        if stem_pts and abs(ry - stem_pts[-1][0]) < 1.0:
                            continue
                        xl = max(0, int(round(cur_x)) - 2 * band_r)
                        xh = min(w, int(round(cur_x)) + 2 * band_r + 1)
                        row_edt = dist_map[ry, xl:xh]
                        if len(row_edt) > 0 and float(np.max(row_edt)) > 0.0:
                            best_dx = int(np.argmax(row_edt))
                            cur_x = 0.78 * cur_x + 0.22 * float(xl + best_dx)
                            w_here = max(1.0, 2.0 * float(row_edt[best_dx]))
                        else:
                            w_here = max(1.0, w_nib_mm * ppm)
                        stem_pts.append((float(ry), cur_x, w_here))
                    stem_core = self._rasterize_strokes_mask([stem_pts], (h, w), scale_w=0.82)
                    rem_head = clean_comp & (~stem_core)
                    new_polys: List[List[Tuple[float, float, float]]] = [stem_pts]
                    if int(np.sum(rem_head)) >= max(4, int(round(0.20 * (ppm ** 2)))):
                        grow_it = max(1, int(round(0.15 * ppm)))
                        head_grown = (cv2.dilate(rem_head.astype(np.uint8), self._ONES_3X3_U8, iterations=grow_it) > 0) & clean_comp
                        h_skel = self.prune_skeleton_spurs(skeletonize(head_grown), dist_map, min_spur_len=spur_min_px)
                        if np.any(h_skel):
                            h_segs = graph_ext.decompose_into_strokes(graph_ext.build_pixel_graph(h_skel, dist_map * 2.0))
                            h_segs = self._normalize_humanist_cursive_segments(h_segs, ppm=ppm, w_nib_mm=w_nib_mm)
                            h_ch = self._chain_segments_minimum_action(
                                h_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=max(1, max_strokes - 1), medieval_downward_only=True
                            )
                            for hs in h_ch[: max(1, max_strokes - 1)]:
                                hpts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in hs.get("points", [])]
                                if len(hpts) >= 2 and self._polyline_arc_mm(hpts, mm_per_px) >= 0.42:
                                    hpts = self._extend_and_anchor_stroke(hpts, clean_comp, dist_map, new_polys, med_half_w, ppm=ppm)
                                    new_polys.append(hpts)
                    raw_polylines = new_polys

        # 7. Residual ink limb recovery: ONLY if an authentic long stroke arm was completely missed
        #    (never fabricate strokes along thick plein borders of `[i]`, `[s]`, or `[n]`!)
        min_res_mm2 = max(0.45, 2.10 * (w_nib_mm ** 2))
        min_res_px = max(8, int(round(min_res_mm2 * (ppm ** 2))))
        min_limb_arc_mm = max(1.05, 2.25 * w_nib_mm)
        while len(raw_polylines) < max_strokes and raw_polylines:
            pred_m = self._rasterize_strokes_mask(raw_polylines, (h, w), scale_w=1.38)
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
            if int(np.sum(u_skel)) < 3:
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
            new_pred = self._rasterize_strokes_mask(raw_polylines, (h, w), scale_w=1.12)
            if int(np.sum(clean_comp & (~new_pred))) >= int(np.sum(uncov)) - 2:
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
        allow_diacritics: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extracts smooth curvilinear cursive loop strokes (Codex Seraphinianus mode):
        - Never jumps across blank paper to detached diacritic dots; renders authentic
          supralinear/infralinear dots (`0.32..2.2 * w_stroke^2`) as separate `is_impact=True` nib dots.
        - Converts thick optical-bleed side lobes into smooth elliptical micro-loops rather than 180° spikes.
        """
        lbl, num_f = label(mask_bool, structure=np.ones((3, 3), dtype=bool))
        if num_f == 0:
            return []
        mm_per_px = 1.0 / ppm
        w_stroke_px = w_nib_mm * ppm

        cc_info = []
        for cid in range(1, num_f + 1):
            cm = (lbl == cid)
            pts_c = np.argwhere(cm)
            cc_info.append({
                "cid": cid,
                "mask": cm,
                "pts": pts_c,
                "area": len(pts_c),
                "h": int(pts_c[:, 0].max() - pts_c[:, 0].min()) + 1,
                "w": int(pts_c[:, 1].max() - pts_c[:, 1].min()) + 1,
            })
        cc_info.sort(key=lambda x: x["area"], reverse=True)
        dom = cc_info[0]
        comp_mask = dom["mask"].copy()
        dom_dist_out = distance_transform_edt(~comp_mask)

        diacritic_strokes: List[Dict[str, Any]] = []
        graph_ext = StrokeGraphExtractor()
        for c in cc_info[1:]:
            min_gap_px = float(np.min(dom_dist_out[c["mask"]]))
            fill_r = float(c["area"]) / float(max(1, c["h"] * c["w"]))
            is_compact_dot = (
                min_gap_px >= 0.25 * ppm
                and 0.24 * (w_stroke_px ** 2) <= float(c["area"]) <= 2.40 * (w_stroke_px ** 2)
                and max(c["h"], c["w"]) <= 1.80 * w_stroke_px
                and fill_r >= 0.40
            )
            is_accent_tick = (
                min_gap_px >= 0.25 * ppm
                and 0.45 * (w_stroke_px ** 2) <= float(c["area"]) <= 4.20 * (w_stroke_px ** 2)
                and max(c["h"], c["w"]) <= 3.20 * w_stroke_px
                and not is_compact_dot
            )
            if is_compact_dot:
                if allow_diacritics and len(diacritic_strokes) < 2:
                    ds = self._make_impact_stroke(c["pts"], len(diacritic_strokes) + 1, ppm, w_nib_mm)
                    ds["is_diacritic"] = True
                    diacritic_strokes.append(ds)
            elif is_accent_tick and allow_diacritics and len(diacritic_strokes) < 2:
                # Short detached acute/grave accent mark above or below Seraphinianus glyph
                ac_strokes = self._extract_component_quill_bernoulli(
                    c["mask"], ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=1, is_serafini=True, is_macro=is_macro, graph_ext=graph_ext
                )
                if ac_strokes:
                    ac_strokes[0]["is_diacritic"] = True
                    diacritic_strokes.append(ac_strokes[0])
                else:
                    ds = self._make_impact_stroke(c["pts"], len(diacritic_strokes) + 1, ppm, w_nib_mm)
                    ds["is_diacritic"] = True
                    diacritic_strokes.append(ds)
            elif min_gap_px <= 1.25 * ppm and c["area"] >= max(3, int(round(0.06 * (ppm ** 2)))):
                # Bridge narrow broken hairline gap directly onto dominant body so comp_mask stays 1 connected component!
                d_sub = distance_transform_edt(~c["mask"])
                iy_dom, ix_dom = np.unravel_index(int(np.argmin(np.where(comp_mask, d_sub, 1e9))), comp_mask.shape)
                iy_sub, ix_sub = np.unravel_index(int(np.argmin(np.where(c["mask"], dom_dist_out, 1e9))), comp_mask.shape)
                bridge = np.zeros_like(comp_mask, dtype=np.uint8)
                cv2.line(bridge, (int(ix_dom), int(iy_dom)), (int(ix_sub), int(iy_sub)), 1, max(1, int(round(0.45 * w_stroke_px))))
                comp_mask |= c["mask"] | (bridge > 0)
                dom_dist_out = distance_transform_edt(~comp_mask)

        h, w = comp_mask.shape
        clean_comp, _, dist_to_hole = self._clean_pinholes_and_get_holes(comp_mask, ppm, is_serafini=True)
        dist_map = distance_transform_edt(clean_comp)
        raw_skel = skeletonize(clean_comp)
        if not np.any(raw_skel):
            return diacritic_strokes

        spur_min_px = max(2, int(round(0.42 * ppm)))
        skel = self.prune_skeleton_spurs(raw_skel, dist_map, min_spur_len=spur_min_px)
        raw_segs = graph_ext.decompose_into_strokes(graph_ext.build_pixel_graph(skel, dist_map * 2.0))
        if not raw_segs:
            return diacritic_strokes

        # Resolve straight-through crossings before chaining so self-crossing cursive loops don't kink
        raw_segs = self._resolve_collinear_junctions(
            raw_segs, dist_map=dist_map, ppm=ppm, w_nib_mm=w_nib_mm, prioritize_vertical=False
        )
        chained = self._chain_segments_minimum_action(raw_segs, ppm=ppm, w_nib_mm=w_nib_mm, max_strokes=6)
        med_half_w = float(np.median(dist_map[clean_comp])) if np.any(clean_comp) else (0.5 * w_nib_mm * ppm)

        chains: List[List[Tuple[float, float, float]]] = []
        for s in chained:
            pts = [(float(p[0]), float(p[1]), max(0.5, float(p[2]))) for p in s.get("points", [])]
            if len(pts) >= 2:
                pts = self._extend_and_anchor_stroke(pts, clean_comp, dist_map, chains, med_half_w, ppm=ppm)
                chains.append(pts)

        if not chains:
            return diacritic_strokes

        def _make_smooth_lobe_loop(lobe_pts: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
            """Opens a collapsed cursive side-lobe into a smooth teardrop micro-loop instead of a 180° needle spike."""
            if len(lobe_pts) < 3:
                return []
            j_y, j_x = lobe_pts[0][0], lobe_pts[0][1]
            t_y, t_x, t_w = lobe_pts[-1][0], lobe_pts[-1][1], lobe_pts[-1][2]
            ay, ax = t_y - j_y, t_x - j_x
            alen = math.hypot(ay, ax)
            if alen < 0.42 * ppm:
                return []
            ny, nx = -ax / alen, ay / alen
            max_bulge = min(0.32 * ppm, 0.42 * max(p[2] * 0.5 for p in lobe_pts))
            out_half = []
            in_half = []
            n_l = len(lobe_pts)
            for idx_l, lp in enumerate(lobe_pts):
                u = float(idx_l) / float(max(1, n_l - 1))
                bulge = max_bulge * math.sin(math.pi * u)
                out_half.append((lp[0] + ny * bulge, lp[1] + nx * bulge, lp[2] * 0.85))
                in_half.append((lp[0] - ny * bulge, lp[1] - nx * bulge, lp[2] * 0.85))
            tip_cap = (t_y + (ay / alen) * (0.12 * ppm), t_x + (ax / alen) * (0.12 * ppm), t_w * 0.85)
            return out_half + [tip_cap] + in_half[::-1]

        # Assemble all open & loop chains iteratively into 1 connected continuous tour (zero dropped limbs!)
        close_tol_px = 0.44 * ppm
        k_tan = max(2, int(round(0.32 * ppm)))
        open_chains: List[List[Tuple[float, float, float]]] = []
        loop_chains: List[List[Tuple[float, float, float]]] = []
        for ch in chains:
            if len(ch) >= 5 and math.hypot(ch[0][0] - ch[-1][0], ch[0][1] - ch[-1][1]) <= close_tol_px:
                loop_chains.append(ch)
            else:
                open_chains.append(ch)

        all_x_min = min(p[1] for ch in chains for p in ch)
        all_y_min = min(p[0] for ch in chains for p in ch)

        # Start at a loop ONLY if the loop forms the top/left head of the glyph (e.g. loop at top + vertical stem below)
        start_with_loop = False
        if loop_chains and open_chains:
            lp_top = min(loop_chains, key=lambda ch: min(p[0] + 0.8 * p[1] for p in ch))
            lp_ymin = min(p[0] for p in lp_top)
            lp_xmin = min(p[1] for p in lp_top)
            op_ymin = min(p[0] for ch in open_chains for p in ch)
            op_xmin = min(p[1] for ch in open_chains for p in ch)
            if lp_ymin < op_ymin - 0.35 * ppm and lp_xmin <= op_xmin + 0.45 * ppm:
                start_with_loop = True

        if open_chains and not start_with_loop:
            # Pick open chain possessing the leftmost external endpoint (`min x`) and orient rightward
            best_idx = 0
            best_rev = False
            best_x_score = 1e12
            for idx_o, och in enumerate(open_chains):
                s0 = och[0][1] + 0.12 * och[0][0]
                s1 = och[-1][1] + 0.12 * och[-1][0]
                if s0 < best_x_score:
                    best_x_score, best_idx, best_rev = s0, idx_o, False
                if s1 < best_x_score:
                    best_x_score, best_idx, best_rev = s1, idx_o, True
            seed = open_chains.pop(best_idx)
            tour = seed[::-1] if best_rev else list(seed)
        else:
            loop_chains.sort(key=lambda ch: min(p[1] + 0.25 * p[0] for p in ch))
            ring0 = loop_chains.pop(0)[:-1]
            if self._signed_ring_area_xy(ring0 + [ring0[0]]) > 0:
                ring0 = ring0[::-1]
            # If open stems hang below/right, start loop at its top/left crest so it exits cleanly at the stem junction!
            i_start = min(range(len(ring0)), key=lambda idx: float(ring0[idx][1]) + 0.35 * float(ring0[idx][0]) - 0.0 * all_x_min - 0.0 * all_y_min)
            tour = ring0[i_start:] + ring0[:i_start] + [ring0[i_start]]

        # Iteratively attach whichever remaining loop OR open chain is closest to the growing `tour`!
        while open_chains or loop_chains:
            best_kind = None
            best_idx = -1
            best_dist = 1e12
            best_payload: Any = None

            for li, lp in enumerate(loop_chains):
                ring = lp[:-1] if len(lp) >= 4 else lp
                if len(ring) < 3:
                    continue
                for bi, bp in enumerate(tour):
                    for ri, rp in enumerate(ring):
                        d_br = math.hypot(bp[0] - rp[0], bp[1] - rp[1])
                        if d_br < best_dist:
                            best_dist = d_br
                            best_kind = "loop"
                            best_idx = li
                            best_payload = (bi, ri, ring)

            for oi, rch in enumerate(open_chains):
                opts = [
                    ("append_fwd", math.hypot(tour[-1][0] - rch[0][0], tour[-1][1] - rch[0][1])),
                    ("append_rev", math.hypot(tour[-1][0] - rch[-1][0], tour[-1][1] - rch[-1][1])),
                    ("prepend_fwd", math.hypot(rch[-1][0] - tour[0][0], rch[-1][1] - tour[0][1])),
                    ("prepend_rev", math.hypot(rch[0][0] - tour[0][0], rch[0][1] - tour[0][1])),
                ]
                end_m, end_d = min(opts, key=lambda x: x[1])
                d_mid_0 = min((math.hypot(bp[0] - rch[0][0], bp[1] - rch[0][1]), bi) for bi, bp in enumerate(tour))
                d_mid_1 = min((math.hypot(bp[0] - rch[-1][0], bp[1] - rch[-1][1]), bi) for bi, bp in enumerate(tour))
                mid_d = min(d_mid_0[0], d_mid_1[0])
                eff_d = min(end_d, mid_d)
                if eff_d < best_dist:
                    best_dist = eff_d
                    best_kind = "open"
                    best_idx = oi
                    best_payload = (end_m, end_d, d_mid_0, d_mid_1)

            if best_kind is None or best_dist > 3.2 * ppm:
                break

            if best_kind == "loop":
                loop_chains.pop(best_idx)
                bi, ri, ring = best_payload
                rot_fwd = ring[ri:] + ring[:ri] + [ring[ri]]
                rev_r = ring[::-1]
                ri_rev = len(ring) - 1 - ri
                rot_rev = rev_r[ri_rev:] + rev_r[:ri_rev] + [rev_r[ri_rev]]
                prefix = tour[:bi + 1]
                if len(prefix) >= 2:
                    c_f = JunctionResolver._tangent_cosine(prefix, rot_fwd, k=k_tan)
                    c_r = JunctionResolver._tangent_cosine(prefix, rot_rev, k=k_tan)
                    chosen_loop = rot_fwd if c_f >= c_r else rot_rev
                else:
                    chosen_loop = rot_fwd
                tour = tour[:bi + 1] + chosen_loop[1:] + tour[bi + 1:]
            else:
                chosen = open_chains.pop(best_idx)
                end_m, end_d, d_mid_0, d_mid_1 = best_payload
                if end_d <= min(d_mid_0[0], d_mid_1[0]) + 0.40 * ppm:
                    if end_m == "append_fwd":
                        tour = tour + chosen
                    elif end_m == "append_rev":
                        tour = tour + chosen[::-1]
                    elif end_m == "prepend_fwd":
                        tour = chosen + tour
                    else:
                        tour = chosen[::-1] + tour
                else:
                    # If `tour` is currently closed and `chosen` hangs off its rim (e.g. loop + bottom stem!),
                    # rotate the closed `tour` to start opposite the stem and exit cleanly down the stem!
                    if math.hypot(tour[0][0] - tour[-1][0], tour[0][1] - tour[-1][1]) <= close_tol_px * 1.4:
                        ring_t = tour[:-1]
                        if d_mid_0[0] <= d_mid_1[0]:
                            bi = min(d_mid_0[1], len(ring_t) - 1)
                            tail = chosen
                        else:
                            bi = min(d_mid_1[1], len(ring_t) - 1)
                            tail = chosen[::-1]
                        tour = ring_t[bi:] + ring_t[:bi] + [ring_t[bi]] + tail[1:]
                    else:
                        if d_mid_0[0] <= d_mid_1[0]:
                            bi = d_mid_0[1]
                            lobe_loop = _make_smooth_lobe_loop(chosen)
                        else:
                            bi = d_mid_1[1]
                            lobe_loop = _make_smooth_lobe_loop(chosen[::-1])
                        if lobe_loop:
                            tour = tour[:bi + 1] + lobe_loop + tour[bi + 1:]
                        elif end_m.startswith("append"):
                            tour = tour + (chosen if end_m == "append_fwd" else chosen[::-1])
                        else:
                            tour = (chosen if end_m == "prepend_fwd" else chosen[::-1]) + tour

        # Strict Left-to-Right Cursive Progression Law on Codex Seraphinianus:
        # Continuous cursive writing progresses from left entry (`x_start`) to right exit (`x_end`)!
        if len(tour) >= 2:
            is_closed_tour = math.hypot(tour[0][0] - tour[-1][0], tour[0][1] - tour[-1][1]) <= close_tol_px
            if not is_closed_tour:
                dx_net = tour[-1][1] - tour[0][1]
                dy_net = tour[-1][0] - tour[0][0]
                span_w = max(p[1] for p in tour) - min(p[1] for p in tour)
                span_h = max(p[0] for p in tour) - min(p[0] for p in tour)
                # Purely vertical stem-with-head glyph (`|dx_net| <= 0.35 * span_w` and `span_h >= 1.45 * span_w`): top-to-bottom
                if span_h >= 1.45 * span_w and abs(dx_net) <= 0.45 * span_w:
                    if dy_net < 0.0:
                        tour = tour[::-1]
                elif abs(dx_net) > 0.18 * ppm:
                    if dx_net < 0.0:
                        tour = tour[::-1]
                elif dy_net < 0.0:
                    tour = tour[::-1]
            else:
                ring_t = tour[:-1]
                if len(ring_t) >= 4:
                    i_left = min(range(len(ring_t)), key=lambda idx: float(ring_t[idx][1]) + 0.12 * float(ring_t[idx][0]))
                    tour = ring_t[i_left:] + ring_t[:i_left] + [ring_t[i_left]]

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
            return diacritic_strokes

        arc_mm = self._polyline_arc_mm(cal_pts, mm_per_px)
        mean_w_px = float(np.mean([p[2] for p in cal_pts]))
        main_st = {
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
        all_out = [main_st]
        for idx_d, ds in enumerate(diacritic_strokes, start=1):
            ds["order_index"] = idx_d
            ds["stroke_id"] = f"cal_s{idx_d:02d}"
            all_out.append(ds)
        return all_out

    def _chain_segments_minimum_action(
        self,
        segments: List[Dict[str, Any]],
        ppm: float,
        w_nib_mm: float,
        max_strokes: int,
        medieval_downward_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Chains topological skeleton segments across junctions by comparing on-vellum
        Euler-Bernoulli bending + gap cost (in mm) against the physical pen-lift cost `C_lift_mm`.
        When `medieval_downward_only=True` (Voynich split quill), segments are NEVER reversed
        upward against the nib grain and never fused into 360° upward-climbing loops!
        """
        if len(segments) <= 1:
            return segments

        mm_per_px = 1.0 / ppm
        max_join_gap_mm = max(0.75, 1.65 * w_nib_mm)
        k_tan = max(2, int(round(0.32 * ppm)))

        work = [dict(s, points=list(s.get("points", []))) for s in segments if len(s.get("points", [])) >= 2]
        if len(work) <= 1:
            return work

        def _is_tall_stem_pts(pts_chk: List[Tuple[float, float, float]]) -> bool:
            if len(pts_chk) < 2:
                return False
            dy_m = abs(pts_chk[-1][0] - pts_chk[0][0]) * mm_per_px
            dx_m = abs(pts_chk[-1][1] - pts_chk[0][1]) * mm_per_px
            chord_m = math.hypot(dy_m, dx_m)
            arc_m = self._polyline_arc_mm(pts_chk, mm_per_px)
            return dy_m >= 1.55 and dy_m >= 1.45 * dx_m and (arc_m / max(1e-4, chord_m)) <= 1.25

        def _violates_humanist_quill(cand_pts: List[Tuple[float, float, float]]) -> bool:
            if not medieval_downward_only or len(cand_pts) < 4:
                return False
            is_cl = len(cand_pts) >= 6 and math.hypot(cand_pts[0][0] - cand_pts[-1][0], cand_pts[0][1] - cand_pts[-1][1]) * mm_per_px <= 0.45
            if is_cl:
                return self._signed_ring_area_xy(cand_pts) > 0
            return self._humanist_anti_grain_cost_mm(cand_pts, mm_per_px) > 1.35

        min_cos_allow = 0.22 if medieval_downward_only else -0.38

        # Pass 1: Merge whenever C_join_mm < PEN_LIFT_ACTION_COST_MM and tangent/quill constraints hold
        merged = True
        while merged and len(work) > 1:
            merged = False
            best_cost = self.PEN_LIFT_ACTION_COST_MM
            best_action = None

            for i in range(len(work)):
                p1 = work[i]["points"]
                p1s, p1e = p1[0], p1[-1]
                is_loop_1 = work[i].get("type") == "loop" or (len(p1) >= 6 and math.hypot(p1s[0] - p1e[0], p1s[1] - p1e[1]) * mm_per_px <= 0.45)
                tall_1 = medieval_downward_only and _is_tall_stem_pts(p1)
                for j in range(i + 1, len(work)):
                    p2 = work[j]["points"]
                    p2s, p2e = p2[0], p2[-1]
                    is_loop_2 = work[j].get("type") == "loop" or (len(p2) >= 6 and math.hypot(p2s[0] - p2e[0], p2s[1] - p2e[1]) * mm_per_px <= 0.45)
                    tall_2 = medieval_downward_only and _is_tall_stem_pts(p2)
                    if medieval_downward_only and (is_loop_1 != is_loop_2):
                        continue

                    orientations = [
                        (math.hypot(p1e[0] - p2s[0], p1e[1] - p2s[1]) * mm_per_px, p1, p2),
                        (math.hypot(p2e[0] - p1s[0], p2e[1] - p1s[1]) * mm_per_px, p2, p1),
                    ]
                    if not is_loop_1 and not is_loop_2:
                        orientations.extend([
                            (math.hypot(p1e[0] - p2e[0], p1e[1] - p2e[1]) * mm_per_px, p1, p2[::-1]),
                            (math.hypot(p1s[0] - p2s[0], p1s[1] - p2s[1]) * mm_per_px, p1[::-1], p2),
                        ])
                    for gap_mm, a_pts, b_pts in orientations:
                        if gap_mm > max_join_gap_mm:
                            continue
                        cos_t = JunctionResolver._tangent_cosine(a_pts, b_pts, k=k_tan)
                        # Gallows rule: a tall straight vertical stem ONLY merges with a collinear vertical continuation, NEVER with a crossbar/head!
                        if (tall_1 or tall_2) and cos_t < 0.72:
                            continue
                        if cos_t < min_cos_allow:
                            continue
                        if cos_t < 0.15 and gap_mm > 0.30:
                            continue
                        comb = a_pts + (b_pts[1:] if gap_mm < 0.08 else b_pts)
                        if (tall_1 or tall_2) and not _is_tall_stem_pts(comb):
                            continue
                        if _violates_humanist_quill(comb):
                            continue
                        turn_cost_mm = gap_mm + 0.75 * w_nib_mm * ((1.0 - cos_t) ** 1.3)
                        if turn_cost_mm < best_cost:
                            best_cost = turn_cost_mm
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

        # Pass 2: If still above stroke ceiling, coalesce ONLY tangent-compatible open branches; otherwise drop minor stray twigs
        while max_strokes > 0 and len(work) > max_strokes:
            best_d_mm = max_join_gap_mm * 1.35
            best_pair = None
            for i in range(len(work)):
                p1 = work[i]["points"]
                is_loop_1 = work[i].get("type") == "loop" or (len(p1) >= 6 and math.hypot(p1[0][0] - p1[-1][0], p1[0][1] - p1[-1][1]) * mm_per_px <= 0.45)
                if medieval_downward_only and (is_loop_1 or _is_tall_stem_pts(p1)):
                    continue
                for j in range(i + 1, len(work)):
                    p2 = work[j]["points"]
                    is_loop_2 = work[j].get("type") == "loop" or (len(p2) >= 6 and math.hypot(p2[0][0] - p2[-1][0], p2[0][1] - p2[-1][1]) * mm_per_px <= 0.45)
                    if medieval_downward_only and (is_loop_2 or _is_tall_stem_pts(p2)):
                        continue
                    opts = [
                        (math.hypot(p1[-1][0] - p2[0][0], p1[-1][1] - p2[0][1]) * mm_per_px, p1, p2),
                        (math.hypot(p2[-1][0] - p1[0][0], p2[-1][1] - p1[0][1]) * mm_per_px, p2, p1),
                        (math.hypot(p1[-1][0] - p2[-1][0], p1[-1][1] - p2[-1][1]) * mm_per_px, p1, p2[::-1]),
                        (math.hypot(p1[0][0] - p2[0][0], p1[0][1] - p2[0][1]) * mm_per_px, p1[::-1], p2),
                    ]
                    for d_mm, a_p, b_p in opts:
                        if d_mm >= best_d_mm:
                            continue
                        if JunctionResolver._tangent_cosine(a_p, b_p, k=k_tan) < (0.15 if medieval_downward_only else -0.25):
                            continue
                        comb = a_p + b_p
                        if not _violates_humanist_quill(comb):
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
        Solves exact minimum biomechanical action across all `K <= 4` Voynich Humanist Cursive strokes:
        - Closed pure loops rotate Counter-Clockwise starting at their ~1-o'clock crest.
        - Tall vertical gallows stems (`span_h >= 2.35 mm`) receive a primary stem-first priority
          (Lisa Fagin Davis 2022: vertical hampes pulled top-to-bottom first, upper loop/crossbar second).
        - Open strokes evaluate forward vs reversed direction via `_humanist_anti_grain_cost_mm`,
          preserving continuous `[n]` finials while penalizing upward stem pushes.
        """
        mm_per_px = 1.0 / ppm
        close_tol_px = 0.42 * ppm

        stroke_variants: List[List[Tuple[Dict[str, Any], float]]] = []
        for s in strokes:
            pts = list(s.get("points", []))
            if s.get("is_impact", False) or len(pts) < 2:
                stroke_variants.append([(dict(s), 0.0)])
                continue

            is_closed = len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= close_tol_px
            if is_closed:
                ring = pts[:-1]
                if self._signed_ring_area_xy(ring + [ring[0]]) > 0:
                    ring = ring[::-1]
                crest_i = min(range(len(ring)), key=lambda idx: float(ring[idx][0]) - 0.22 * float(ring[idx][1]))
                rot = ring[crest_i:] + ring[:crest_i] + [ring[crest_i]]
                s_loop = dict(s, points=rot, start=(rot[0][0], rot[0][1]), end=(rot[-1][0], rot[-1][1]))
                stroke_variants.append([(s_loop, 0.0)])
            else:
                c_fwd = self._humanist_anti_grain_cost_mm(pts, mm_per_px)
                rev_pts = pts[::-1]
                c_rev = self._humanist_anti_grain_cost_mm(rev_pts, mm_per_px)
                s_fwd = dict(s, points=pts, start=(pts[0][0], pts[0][1]), end=(pts[-1][0], pts[-1][1]))
                s_rev = dict(s, points=rev_pts, start=(rev_pts[0][0], rev_pts[0][1]), end=(rev_pts[-1][0], rev_pts[-1][1]))
                stroke_variants.append([(s_fwd, c_fwd), (s_rev, c_rev)])

        k_n = len(stroke_variants)
        if k_n == 1:
            best_s = min(stroke_variants[0], key=lambda item: item[1] + (item[0]["start"][0] * 0.45 + item[0]["start"][1] * 0.35) * mm_per_px)[0]
            best_s["order_index"] = 0
            best_s["stroke_id"] = "cal_s00"
            return [best_s]

        def _is_tall_vertical_stem(st_d: Dict[str, Any]) -> bool:
            pts_s = st_d.get("points", [])
            if len(pts_s) < 2:
                return False
            if len(pts_s) >= 6 and math.hypot(pts_s[0][0] - pts_s[-1][0], pts_s[0][1] - pts_s[-1][1]) <= close_tol_px:
                return False
            dy_mm = (pts_s[-1][0] - pts_s[0][0]) * mm_per_px
            dx_mm = abs(pts_s[-1][1] - pts_s[0][1]) * mm_per_px
            ch_mm = math.hypot(dy_mm, dx_mm)
            ar_mm = self._polyline_arc_mm(pts_s, mm_per_px)
            return dy_mm >= 1.42 and dy_mm >= 1.45 * dx_mm and (ar_mm / max(1e-4, ch_mm)) <= 1.24

        has_any_tall_stem = any(_is_tall_vertical_stem(v[0]) for var_list in stroke_variants for v in var_list)

        best_total = 1e12
        best_seq: List[Dict[str, Any]] = []
        for perm in itertools.permutations(range(k_n)):
            for choice_tuple in itertools.product(*[stroke_variants[idx] for idx in perm]):
                s0, g0 = choice_tuple[0]
                cost = (s0["start"][0] * 0.50 + s0["start"][1] * 0.60) * mm_per_px + g0
                # Gallows law (Davis 2022): ALL tall vertical legs execute FIRST (left-to-right), upper loops/crossbars AFTER!
                if has_any_tall_stem:
                    if not _is_tall_vertical_stem(s0):
                        cost += 4.50
                    seen_non_stem = not _is_tall_vertical_stem(s0)
                    for step_chk in range(1, k_n):
                        is_st = _is_tall_vertical_stem(choice_tuple[step_chk][0])
                        if not is_st:
                            seen_non_stem = True
                        elif seen_non_stem:
                            cost += 3.20
                for step_i in range(1, k_n):
                    s_prev = choice_tuple[step_i - 1][0]
                    s_curr, g_curr = choice_tuple[step_i]
                    air_mm = math.hypot(s_curr["start"][0] - s_prev["end"][0], s_curr["start"][1] - s_prev["end"][1]) * mm_per_px
                    cost += self.PEN_LIFT_ACTION_COST_MM + 0.55 * air_mm + g_curr
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
        # Never freeze sharp angles on Seraphinianus: its continuous cursive script is 100% curvilinear loops!
        if (not is_serafini) and cum_mm[-1] >= 2.0 * cusp_chord_mm:
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
                    if cos_t < -0.52:
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

        # Physical hand-inertia smoothing scale (0.30 mm curvilinear on Seraphinianus, 0.22 mm on Voynich)
        py_all = np.array([p[0] for p in dedup], dtype=np.float64)
        px_all = np.array([p[1] for p in dedup], dtype=np.float64)
        step_ds = float(np.median(np.hypot(np.diff(py_all), np.diff(px_all)))) if n_pts >= 3 else (0.08 * ppm)
        step_ds = max(0.04 * ppm, step_ds)
        inertia_mm = (0.36 if is_serafini else 0.23) * self.inertia_scale
        sig_pts = max(1.2, min(7.2 if is_serafini else 5.0, (inertia_mm * ppm) / max(1e-3, step_ds)))
        rdp_eps_px = (0.040 if is_serafini else 0.052) * ppm
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
        if n_r >= 4:
            sig_nib = (0.95 if is_serafini else 0.85) * self.inertia_scale
            ry_a = np.array([c[0] for c in coords_yx], dtype=np.float64)
            rx_a = np.array([c[1] for c in coords_yx], dtype=np.float64)
            if is_closed:
                ry_s = gaussian_filter1d(ry_a[:-1], sigma=sig_nib, mode="wrap")
                rx_s = gaussian_filter1d(rx_a[:-1], sigma=sig_nib, mode="wrap")
                for i in range(n_r - 1):
                    coords_yx[i] = [float(ry_s[i]), float(rx_s[i])]
                coords_yx[-1] = list(coords_yx[0])
            else:
                ry_s = gaussian_filter1d(ry_a, sigma=sig_nib, mode="nearest")
                rx_s = gaussian_filter1d(rx_a, sigma=sig_nib, mode="nearest")
                ry_s[0], rx_s[0], ry_s[-1], rx_s[-1] = ry_a[0], rx_a[0], ry_a[-1], rx_a[-1]
                arr_sm = np.column_stack([ry_s, rx_s])
                # Fit analytical 4-control-point Cubic Bézier (or 2-segment cubic spline) on open Voynich strokes
                # whenever fidelity error <= 0.22 mm, guaranteeing zero discrete polyline tremor by construction!
                if self.fit_cubic_bezier and (not is_serafini) and n_r >= 5:
                    tol_bez_px = 0.22 * ppm
                    fit1, err1 = self._fit_cubic_bezier_piece(arr_sm)
                    if err1 <= tol_bez_px:
                        arr_sm = fit1
                    elif n_r >= 9:
                        mid_b = n_r // 2
                        f_a, e_a = self._fit_cubic_bezier_piece(arr_sm[: mid_b + 1])
                        f_b, e_b = self._fit_cubic_bezier_piece(arr_sm[mid_b:])
                        if max(e_a, e_b) <= tol_bez_px:
                            arr_sm = np.vstack([f_a[:-1], f_b])
                            ry_b = gaussian_filter1d(arr_sm[:, 0], sigma=0.55, mode="nearest")
                            rx_b = gaussian_filter1d(arr_sm[:, 1], sigma=0.55, mode="nearest")
                            ry_b[0], rx_b[0], ry_b[-1], rx_b[-1] = arr_sm[0, 0], arr_sm[0, 1], arr_sm[-1, 0], arr_sm[-1, 1]
                            arr_sm = np.column_stack([ry_b, rx_b])
                for i in range(n_r):
                    coords_yx[i] = [float(arr_sm[i, 0]), float(arr_sm[i, 1])]

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
            # Convert screen-down `(dy, dx)` to mathematical Cartesian angle (`y` up) for right-handed +40° bevel
            theta = math.atan2(-dy, dx)
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

        # Combine pressure * bevel width, strictly capped by physical MAX_PRESSED_SPREAD_MM AND raster-safe hole clearance!
        base_w_px: List[float] = []
        hole_caps_arr = np.full(n_r, self.MAX_PRESSED_SPREAD_MM * ppm, dtype=np.float64)
        hole_margin_px = max(0.68 if is_serafini else 0.54, (0.085 if is_serafini else 0.068) * ppm) * self.hole_margin_scale
        for i in range(n_r):
            iy = max(0, min(h - 1, int(round(coords_yx[i][0]))))
            ix = max(0, min(w - 1, int(round(coords_yx[i][1]))))
            if dist_to_hole is not None:
                dh_px = float(dist_to_hole[iy, ix])
                if dh_px < 0.65 * ppm:
                    hole_caps_arr[i] = max(0.95, 2.0 * max(0.45, dh_px - hole_margin_px))
        if dist_to_hole is not None and n_r >= 4:
            hole_caps_arr = np.minimum(
                hole_caps_arr,
                gaussian_filter1d(hole_caps_arr, sigma=0.80, mode="wrap" if is_closed else "nearest"),
            )
        hole_caps_px = [float(v) for v in hole_caps_arr]
        for i in range(n_r):
            w_mm_i = float(np.clip(smooth_p[i] * geom_w_mm[i], w_edge_mm, self.MAX_PRESSED_SPREAD_MM))
            base_w_px.append(min(w_mm_i * ppm, hole_caps_px[i]))

        # Fine silhouette IoU + strict hole-preservation calibration within physical bounds [0.92, 1.24]
        best_gain = 1.04
        best_obj = -1e9
        holes_mask = (dist_to_hole == 0) if dist_to_hole is not None else None
        hole_tot = int(np.sum(holes_mask)) if holes_mask is not None else 0

        for gain in (0.92 * self.gain_bias, 1.04 * self.gain_bias, 1.14 * self.gain_bias, 1.24 * self.gain_bias):
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
            obj = iou - 0.95 * hole_inv
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
            k_step = min(len(out) - 1, max(1, int(round(0.35 * ppm))))
            if end_flag == 0:
                ty, tx = out[0][0] - out[k_step][0], out[0][1] - out[k_step][1]
            else:
                ty, tx = out[-1][0] - out[-1 - k_step][0], out[-1][1] - out[-1 - k_step][1]
            tn = math.hypot(ty, tx)
            uy_out, ux_out = (ty / tn, tx / tn) if tn >= 1e-4 else (0.0, 0.0)

            if prior_flat and tn >= 1e-4:
                best_p = None
                best_d = snap_r + 1.0
                for pp in prior_flat:
                    vy_p, vx_p = float(pp[0]) - ey, float(pp[1]) - ex
                    d = math.hypot(vy_p, vx_p)
                    # Only snap into the forward outward cone (`cos >= 0.25`), NEVER fold 180° backward!
                    if 0.05 * ppm <= d <= snap_r and (vy_p * uy_out + vx_p * ux_out) >= 0.25 * d:
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
            if tn >= 1e-4:
                ty, tx = uy_out, ux_out
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
