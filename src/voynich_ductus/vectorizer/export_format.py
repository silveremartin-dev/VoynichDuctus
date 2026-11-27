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
    def to_svg(
        strokes: List[Dict[str, Any]],
        width: int,
        height: int,
        output_path: Optional[Union[str, Path]] = None,
        include_order_colors: bool = True,
        pad: int = 6,
    ) -> str:
        """
        Exports strokes to an SVG string/file with embedded kinematic ordering and smooth cubic curves.
        Includes padding in viewBox to avoid border clipping and ensure exact 1:1 alignment with scan crops.
        """
        # Palette for distinct pen-lift strokes
        colors = ["#38bdf8", "#10b981", "#f59e0b", "#c084fc", "#f43f5e", "#06b6d4", "#a855f7"]

        vx = -pad
        vy = -pad
        vw = max(1, width + 2 * pad)
        vh = max(1, height + 2 * pad)

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx} {vy} {vw} {vh}" width="100%" height="100%" style="overflow:hidden;">',
            '  <g id="voynich_strokes">'
        ]

        for i, stroke in enumerate(strokes):
            pts = stroke.get("points", [])
            if not pts:
                continue

            # Normalize points format to List[Tuple[y, x, width]]
            norm_pts = []
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
            glyph_dim = max(width, height)
            max_allowed_w = max(0.85, min(glyph_dim * 0.068, 3.4))
            avg_width = sum(p[2] for p in norm_pts) / len(norm_pts)
            stroke_width = max(0.65, min(avg_width * 0.48, max_allowed_w))

            d_str = VectorExporter._points_to_smooth_path_d(norm_pts)
            start_y, start_x = norm_pts[0][0], norm_pts[0][1]

            svg_lines.append(
                f'    <path id="{stroke.get("stroke_id", f"s{i}")}" d="{d_str}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_width:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round" data-order="{stroke.get("order_index", i)}"/>'
            )
            # Pen-down touch point indicator (small circle on stroke start)
            dot_r = max(0.55, min(stroke_width * 0.70, glyph_dim * 0.052, 2.1))
            svg_lines.append(
                f'    <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="{dot_r:.2f}" fill="#10b981" opacity="0.95"/>'
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
