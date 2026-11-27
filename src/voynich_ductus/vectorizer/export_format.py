"""
Exporters for standardized vector stroke datasets (SVG with scribal metadata, JSON, GeoJSON).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional


class VectorExporter:
    """
    Exports extracted strokes into open vector formats.
    """

    @staticmethod
    def _points_to_smooth_path_d(pts: List[Tuple[float, float, float]], min_step: Optional[float] = None) -> str:
        """
        Converts discrete pixel stroke points into a smooth cubic Bézier path.
        Eliminates discrete pixel staircase jitter, producing calligraphic cursive strokes.
        """
        if not pts:
            return ""
        if len(pts) == 1:
            return f"M {pts[0][1]:.2f} {pts[0][0]:.2f}"
        if len(pts) == 2:
            return f"M {pts[0][1]:.2f} {pts[0][0]:.2f} L {pts[1][1]:.2f} {pts[1][0]:.2f}"

        # Subsample only near-duplicate points while strictly preserving sharp cusps / inflections (e.g. '3' middle tongue)
        import math
        coords = [(float(p[1]), float(p[0])) for p in pts]  # (x, y)
        is_closed = (len(coords) >= 5 and ((coords[0][0] - coords[-1][0]) ** 2 + (coords[0][1] - coords[-1][1]) ** 2) <= 9.0)
        if min_step is None:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            min_step = max(0.45, min(1.05, span * 0.018))
        min_step_sq = float(min_step * min_step)

        def _turn_cos(pa: Tuple[float, float], pb: Tuple[float, float], pc: Tuple[float, float]) -> float:
            v1x, v1y = pb[0] - pa[0], pb[1] - pa[1]
            v2x, v2y = pc[0] - pb[0], pc[1] - pb[1]
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 < 1e-5 or n2 < 1e-5:
                return 1.0
            return (v1x * v2x + v1y * v2y) / (n1 * n2)

        simplified = [coords[0]]
        for idx_c in range(1, len(coords) - 1):
            p = coords[idx_c]
            dx = p[0] - simplified[-1][0]
            dy = p[1] - simplified[-1][1]
            is_sharp = _turn_cos(coords[idx_c - 1], p, coords[idx_c + 1]) < 0.55
            if is_sharp or (dx * dx + dy * dy) >= min_step_sq:
                if (dx * dx + dy * dy) >= 1e-4:
                    simplified.append(p)
        end_pt = coords[0] if is_closed else coords[-1]
        if (end_pt[0] - simplified[-1][0]) ** 2 + (end_pt[1] - simplified[-1][1]) ** 2 >= 1e-4 or len(simplified) == 1:
            simplified.append(end_pt)

        if len(simplified) < 3:
            return " ".join([f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"] + [f"L {p[0]:.2f} {p[1]:.2f}" for p in simplified[1:]])

        # Corner-aware Catmull-Rom to Cubic Bézier spline: zero tangent overshoot across sharp cusps ('3', 'm', 'k')
        d_tokens = [f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"]
        n = len(simplified)
        if is_closed and n >= 5:
            m = n - 1
            for i in range(m):
                p0 = simplified[(i - 1) % m]
                p1 = simplified[i]
                p2 = simplified[(i + 1) % m]
                p3 = simplified[(i + 2) % m]
                t1 = 0.0 if _turn_cos(p0, p1, p2) < 0.45 else (1.0 / 6.0)
                t2 = 0.0 if _turn_cos(p1, p2, p3) < 0.45 else (1.0 / 6.0)
                cp1x = p1[0] + (p2[0] - p0[0]) * t1
                cp1y = p1[1] + (p2[1] - p0[1]) * t1
                cp2x = p2[0] - (p3[0] - p1[0]) * t2
                cp2y = p2[1] - (p3[1] - p1[1]) * t2
                d_tokens.append(f"C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}")
            d_tokens.append("Z")
        else:
            for i in range(n - 1):
                p0 = simplified[max(0, i - 1)]
                p1 = simplified[i]
                p2 = simplified[i + 1]
                p3 = simplified[min(n - 1, i + 2)]
                t1 = 0.0 if (i > 0 and _turn_cos(p0, p1, p2) < 0.45) else (1.0 / 6.0)
                t2 = 0.0 if (i + 1 < n - 1 and _turn_cos(p1, p2, p3) < 0.45) else (1.0 / 6.0)
                cp1x = p1[0] + (p2[0] - p0[0]) * t1
                cp1y = p1[1] + (p2[1] - p0[1]) * t1
                cp2x = p2[0] - (p3[0] - p1[0]) * t2
                cp2y = p2[1] - (p3[1] - p1[1]) * t2
                d_tokens.append(f"C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}")

        return " ".join(d_tokens)

    @staticmethod
    def _build_variable_width_ribbon_d(pts: List[Tuple[float, float, float]]) -> str:
        """
        Builds a smooth, swallowtail-free SVG variable-width calligraphic ribbon path along
        `pts = [(y, x, width_px), ...]` by sampling the exact Catmull-Rom cubic Bézier spine
        and clamping inner concave normal offsets against focal-radius self-intersection.
        """
        import math
        if len(pts) < 2:
            return ""

        clean: List[Tuple[float, float, float]] = [(float(pts[0][1]), float(pts[0][0]), max(0.35, float(pts[0][2])))]
        for p in pts[1:]:
            x_c, y_c, w_c = float(p[1]), float(p[0]), max(0.35, float(p[2]))
            if math.hypot(x_c - clean[-1][0], y_c - clean[-1][1]) >= 0.35:
                clean.append((x_c, y_c, w_c))
        end_cand = (float(pts[-1][1]), float(pts[-1][0]), max(0.35, float(pts[-1][2])))
        if len(clean) == 1 and math.hypot(end_cand[0] - clean[0][0], end_cand[1] - clean[0][1]) > 1e-3:
            clean.append(end_cand)
        if len(clean) < 2:
            return ""

        is_closed = len(clean) >= 5 and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) <= 3.0
        if is_closed:
            clean[-1] = clean[0]

        def _turn_cos_xy(pa: Tuple[float, float, float], pb: Tuple[float, float, float], pc: Tuple[float, float, float]) -> float:
            v1x, v1y = pb[0] - pa[0], pb[1] - pa[1]
            v2x, v2y = pc[0] - pb[0], pc[1] - pb[1]
            n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
            if n1 < 1e-5 or n2 < 1e-5:
                return 1.0
            return (v1x * v2x + v1y * v2y) / (n1 * n2)

        # Sample smooth cubic Bézier spine (4 subdivisions per segment) so ribbon matches stroke centerline 1:1
        samples: List[Tuple[float, float, float]] = []
        n_k = len(clean)
        if n_k == 2:
            samples = [clean[0], clean[1]]
        elif is_closed and n_k >= 5:
            m = n_k - 1
            for i in range(m):
                p0, p1, p2, p3 = clean[(i - 1) % m], clean[i], clean[(i + 1) % m], clean[(i + 2) % m]
                t1 = 0.0 if _turn_cos_xy(p0, p1, p2) < -0.25 else (1.0 / 6.0)
                t2 = 0.0 if _turn_cos_xy(p1, p2, p3) < -0.25 else (1.0 / 6.0)
                cp1x, cp1y = p1[0] + (p2[0] - p0[0]) * t1, p1[1] + (p2[1] - p0[1]) * t1
                cp2x, cp2y = p2[0] - (p3[0] - p1[0]) * t2, p2[1] - (p3[1] - p1[1]) * t2
                steps = 5
                for s_i in range(steps):
                    u = float(s_i) / float(steps)
                    um = 1.0 - u
                    bx = (um ** 3) * p1[0] + 3 * (um ** 2) * u * cp1x + 3 * um * (u ** 2) * cp2x + (u ** 3) * p2[0]
                    by = (um ** 3) * p1[1] + 3 * (um ** 2) * u * cp1y + 3 * um * (u ** 2) * cp2y + (u ** 3) * p2[1]
                    bw = p1[2] + u * (p2[2] - p1[2])
                    samples.append((bx, by, bw))
            samples.append(samples[0])
        else:
            for i in range(n_k - 1):
                p0, p1, p2, p3 = clean[max(0, i - 1)], clean[i], clean[i + 1], clean[min(n_k - 1, i + 2)]
                t1 = 0.0 if (i > 0 and _turn_cos_xy(p0, p1, p2) < -0.25) else (1.0 / 6.0)
                t2 = 0.0 if (i + 1 < n_k - 1 and _turn_cos_xy(p1, p2, p3) < -0.25) else (1.0 / 6.0)
                cp1x, cp1y = p1[0] + (p2[0] - p0[0]) * t1, p1[1] + (p2[1] - p0[1]) * t1
                cp2x, cp2y = p2[0] - (p3[0] - p1[0]) * t2, p2[1] - (p3[1] - p1[1]) * t2
                steps = 5
                for s_i in range(steps):
                    u = float(s_i) / float(steps)
                    um = 1.0 - u
                    bx = (um ** 3) * p1[0] + 3 * (um ** 2) * u * cp1x + 3 * um * (u ** 2) * cp2x + (u ** 3) * p2[0]
                    by = (um ** 3) * p1[1] + 3 * (um ** 2) * u * cp1y + 3 * um * (u ** 2) * cp2y + (u ** 3) * p2[1]
                    bw = p1[2] + u * (p2[2] - p1[2])
                    samples.append((bx, by, bw))
            samples.append(clean[-1])

        left_side: List[Tuple[float, float]] = []
        right_side: List[Tuple[float, float]] = []
        ns = len(samples)
        for i in range(ns):
            x, y, w = samples[i]
            r = 0.5 * w
            p_prev = samples[(i - 2) % (ns - 1)] if is_closed else samples[max(0, i - 1)]
            p_next = samples[(i + 2) % (ns - 1)] if is_closed else samples[min(ns - 1, i + 1)]
            dx, dy = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
            norm = math.hypot(dx, dy)
            if norm < 1e-5:
                nx_perp, ny_perp = 0.0, 1.0
            else:
                nx_perp, ny_perp = -dy / norm, dx / norm

            # Clamp inner concave offset against local focal radius so tight bowls never swallowtail
            r_left, r_right = r, r
            if 0 < i < ns - 1 or is_closed:
                v1x, v1y = x - p_prev[0], y - p_prev[1]
                v2x, v2y = p_next[0] - x, p_next[1] - y
                l1, l2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
                if l1 > 1e-4 and l2 > 1e-4:
                    cross = v1x * v2y - v1y * v2x
                    chord = math.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
                    kappa_signed = (2.0 * cross) / max(1e-5, l1 * l2 * max(chord, 0.25))
                    if kappa_signed > 1e-3:
                        r_left = min(r, max(0.25 * r, 0.78 / kappa_signed))
                    elif kappa_signed < -1e-3:
                        r_right = min(r, max(0.25 * r, 0.78 / abs(kappa_signed)))

            left_side.append((x + nx_perp * r_left, y + ny_perp * r_left))
            right_side.append((x - nx_perp * r_right, y - ny_perp * r_right))

        if is_closed and len(left_side) >= 4:
            tokens = [f"M {left_side[0][0]:.2f} {left_side[0][1]:.2f}"]
            for lx, ly in left_side[1:]:
                tokens.append(f"L {lx:.2f} {ly:.2f}")
            tokens.append("Z")
            tokens.append(f"M {right_side[0][0]:.2f} {right_side[0][1]:.2f}")
            for rx, ry in reversed(right_side[:-1]):
                tokens.append(f"L {rx:.2f} {ry:.2f}")
            tokens.append("Z")
            return " ".join(tokens)

        # Smooth semicircular endcaps on open strokes so the true physical nib thickness is cleanly closed
        tokens = [f"M {left_side[0][0]:.2f} {left_side[0][1]:.2f}"]
        for lx, ly in left_side[1:]:
            tokens.append(f"L {lx:.2f} {ly:.2f}")
        end_x, end_y, end_w = samples[-1]
        end_r = 0.5 * end_w
        dx_e, dy_e = samples[-1][0] - samples[-2][0], samples[-1][1] - samples[-2][1]
        ne = math.hypot(dx_e, dy_e)
        if ne > 1e-5:
            ux_e, uy_e = dx_e / ne, dy_e / ne
            px_e, py_e = -uy_e, ux_e
            for ang in (0.25 * math.pi, 0.0, -0.25 * math.pi):
                cx_cap = end_x + end_r * (math.cos(ang) * ux_e + math.sin(ang) * px_e)
                cy_cap = end_y + end_r * (math.cos(ang) * uy_e + math.sin(ang) * py_e)
                tokens.append(f"L {cx_cap:.2f} {cy_cap:.2f}")
        for rx, ry in reversed(right_side):
            tokens.append(f"L {rx:.2f} {ry:.2f}")
        st_x, st_y, st_w = samples[0]
        st_r = 0.5 * st_w
        dx_s, dy_s = samples[0][0] - samples[1][0], samples[0][1] - samples[1][1]
        ns_0 = math.hypot(dx_s, dy_s)
        if ns_0 > 1e-5:
            ux_s, uy_s = dx_s / ns_0, dy_s / ns_0
            px_s, py_s = -uy_s, ux_s
            for ang in (0.25 * math.pi, 0.0, -0.25 * math.pi):
                cx_cap = st_x + st_r * (math.cos(ang) * ux_s + math.sin(ang) * px_s)
                cy_cap = st_y + st_r * (math.cos(ang) * uy_s + math.sin(ang) * py_s)
                tokens.append(f"L {cx_cap:.2f} {cy_cap:.2f}")
        tokens.append("Z")
        return " ".join(tokens)

    @staticmethod
    def to_svg(
        strokes: List[Dict[str, Any]],
        width: int,
        height: int,
        output_path: Optional[Union[str, Path]] = None,
        include_order_colors: bool = True,
        pad: Optional[int] = None,
        px_per_mm: Optional[float] = None,
        variable_width: bool = True,
    ) -> str:
        """
        Exports strokes to an SVG string/file with true physical millimeter scaling (zero box-relative
        nib inflation) and variable-width nib + pressure ribbons.
        - A sub-nib speck (e.g. 0.3 mm x 0.3 mm) sits inside a minimum 1.6 mm physical window as a stationary nib imprint.
        - A giant macro shape (e.g. 300 mm x 300 mm) keeps the physical 0.45 mm quill width without scaling up.
        """
        import math
        colors = ["#38bdf8", "#10b981", "#f59e0b", "#c084fc", "#f43f5e", "#06b6d4", "#a855f7"]

        eff_ppm = float(px_per_mm) if (px_per_mm is not None and px_per_mm > 0) else 0.0
        if eff_ppm <= 0.0:
            for s in strokes:
                if s.get("px_per_mm", 0.0) > 0:
                    eff_ppm = float(s["px_per_mm"])
                    break

        if pad is None:
            pad_f = max(2.0, 0.35 * eff_ppm) if eff_ppm > 0 else 6.0
        else:
            pad_f = float(pad)

        raw_w = max(1.0, float(width))
        raw_h = max(1.0, float(height))

        # Enforce minimum physical viewport window (1.60 mm x 1.60 mm) when calibrated in mm
        # so a 0.3 mm speck is never zoomed 10x to masquerade as a full-sized character.
        min_view_px = (1.60 * eff_ppm) if eff_ppm > 0 else 0.0
        inner_w = max(raw_w, min_view_px)
        inner_h = max(raw_h, min_view_px)
        off_x = 0.5 * (inner_w - raw_w)
        off_y = 0.5 * (inner_h - raw_h)

        vx = -(pad_f + off_x)
        vy = -(pad_f + off_y)
        vw = max(1.0, inner_w + 2.0 * pad_f)
        vh = max(1.0, inner_h + 2.0 * pad_f)

        # Physical reference nib spine thickness in scan units (never proportional to glyph_dim!)
        phys_spine_cap = (0.24 * eff_ppm) if eff_ppm > 0 else 2.2
        phys_spine_min = (0.09 * eff_ppm) if eff_ppm > 0 else 0.75

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx:.2f} {vy:.2f} {vw:.2f} {vh:.2f}" width="100%" height="100%" style="overflow:hidden;">',
            '  <g id="voynich_strokes">'
        ]

        for i, stroke in enumerate(strokes):
            pts = stroke.get("points", [])
            if not pts:
                continue

            norm_pts: List[Tuple[float, float, float]] = []
            for p in pts:
                if isinstance(p, dict):
                    y_val = float(p.get("y", 0.0))
                    x_val = float(p.get("x", 0.0))
                    w_val = float(p.get("pressure_proxy", p.get("w", p.get("width", 1.5))))
                    norm_pts.append((y_val, x_val, w_val))
                elif isinstance(p, (tuple, list)):
                    y_val = float(p[0])
                    x_val = float(p[1])
                    w_val = float(p[2]) if len(p) > 2 else 1.5
                    norm_pts.append((y_val, x_val, w_val))

            if not norm_pts:
                continue

            color = colors[i % len(colors)] if include_order_colors else "#38bdf8"
            is_impact = bool(stroke.get("is_impact", False)) or len(norm_pts) == 1
            nib_deg = float(stroke.get("nib_angle_deg", 40.0))

            if is_impact:
                cy, cx, w_imp = norm_pts[0]
                rx = max(0.45, 0.50 * w_imp)
                ry = max(0.30, 0.30 * w_imp)
                svg_lines.append(
                    f'    <ellipse id="{stroke.get("stroke_id", f"s{i}")}" cx="{cx:.2f}" cy="{cy:.2f}" '
                    f'rx="{rx:.2f}" ry="{ry:.2f}" transform="rotate({-nib_deg:.1f} {cx:.2f} {cy:.2f})" '
                    f'fill="{color}" fill-opacity="0.65" stroke="#10b981" stroke-width="{max(0.35, phys_spine_min * 0.7):.2f}" '
                    f'data-order="{stroke.get("order_index", i)}" data-impact="true"/>'
                )
                continue

            # 1. Prominent physical variable-width nib + pressure ribbon (true supposed stroke thickness in mm!)
            avg_width = sum(p[2] for p in norm_pts) / len(norm_pts)
            if variable_width:
                ribbon_d = VectorExporter._build_variable_width_ribbon_d(norm_pts)
                if ribbon_d:
                    rib_edge_w = max(0.20, (0.035 * eff_ppm) if eff_ppm > 0 else 0.32)
                    svg_lines.append(
                        f'    <path d="{ribbon_d}" fill="{color}" fill-rule="evenodd" fill-opacity="0.68" '
                        f'stroke="{color}" stroke-opacity="0.95" stroke-width="{rib_edge_w:.2f}"/>'
                    )
                stroke_width = max(0.30, (0.055 * eff_ppm) if eff_ppm > 0 else max(phys_spine_min * 0.7, min(avg_width * 0.22, phys_spine_cap * 0.6)))
                spine_color = "#f8fafc"
                spine_opacity = "0.82"
            else:
                stroke_width = max(phys_spine_min, min(avg_width * 0.42, phys_spine_cap * 1.15))
                spine_color = color
                spine_opacity = "1.0"

            d_str = VectorExporter._points_to_smooth_path_d(norm_pts)
            start_y, start_x = norm_pts[0][0], norm_pts[0][1]

            svg_lines.append(
                f'    <path id="{stroke.get("stroke_id", f"s{i}")}" d="{d_str}" '
                f'fill="none" stroke="{spine_color}" stroke-opacity="{spine_opacity}" stroke-width="{stroke_width:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round" data-order="{stroke.get("order_index", i)}" '
                f'data-mean-width="{avg_width:.2f}"/>'
            )

            # 3. Pen-down touchdown dot (#10b981) & subtle direction chevron at pen-lift exit
            dot_r = max(0.42, min(max(stroke_width * 1.25, avg_width * 0.22), phys_spine_cap * 0.90))
            svg_lines.append(
                f'    <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="{dot_r:.2f}" fill="#10b981" opacity="0.95"/>'
            )
            if len(norm_pts) >= 3:
                end_y, end_x = norm_pts[-1][0], norm_pts[-1][1]
                prev_y, prev_x = norm_pts[-2][0], norm_pts[-2][1]
                tdx, tdy = end_x - prev_x, end_y - prev_y
                tn = math.hypot(tdx, tdy)
                if tn > 1e-4 and math.hypot(end_x - start_x, end_y - start_y) > dot_r * 2.2:
                    ux, uy = tdx / tn, tdy / tn
                    px_n, py_n = -uy, ux
                    ah = max(0.7, stroke_width * 1.45)
                    aw = max(0.5, stroke_width * 0.95)
                    ax1, ay1 = end_x - ux * ah + px_n * aw, end_y - uy * ah + py_n * aw
                    ax2, ay2 = end_x - ux * ah - px_n * aw, end_y - uy * ah - py_n * aw
                    svg_lines.append(
                        f'    <path d="M {ax1:.2f} {ay1:.2f} L {end_x:.2f} {end_y:.2f} L {ax2:.2f} {ay2:.2f}" '
                        f'fill="none" stroke="{color}" stroke-width="{max(0.4, stroke_width * 0.75):.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
                    )

        svg_lines.append("  </g>")
        svg_lines.append("</svg>")

        svg_content = "\n".join(svg_lines)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(svg_content)

        return svg_content

    @staticmethod
    def to_json(strokes: List[Dict[str, Any]], metadata: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Exports strokes with full kinematic timestamps and metadata to JSON.
        """
        payload = {
            "metadata": metadata,
            "stroke_count": len(strokes),
            "strokes": strokes
        }
        json_str = json.dumps(payload, indent=2)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str
