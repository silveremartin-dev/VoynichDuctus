"""
Exporters for standardized vector stroke datasets (SVG with scribal metadata, JSON, GeoJSON).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union


class VectorExporter:
    """
    Exports extracted strokes into open vector formats.
    """

    @staticmethod
    def _points_to_smooth_path_d(pts: List[Tuple[float, float, float]]) -> str:
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

        # Subsample / simplify points slightly so cubic spline doesn't wobble
        coords = [(float(p[1]), float(p[0])) for p in pts]  # (x, y)
        simplified = [coords[0]]
        for p in coords[1:-1]:
            dx = p[0] - simplified[-1][0]
            dy = p[1] - simplified[-1][1]
            if (dx * dx + dy * dy) >= 2.25:  # min 1.5px step
                simplified.append(p)
        simplified.append(coords[-1])

        if len(simplified) < 3:
            return " ".join([f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"] + [f"L {p[0]:.2f} {p[1]:.2f}" for p in simplified[1:]])

        # Catmull-Rom to Cubic Bézier spline
        d_tokens = [f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"]
        n = len(simplified)
        for i in range(n - 1):
            p0 = simplified[max(0, i - 1)]
            p1 = simplified[i]
            p2 = simplified[i + 1]
            p3 = simplified[min(n - 1, i + 2)]

            cp1x = p1[0] + (p2[0] - p0[0]) / 6.0
            cp1y = p1[1] + (p2[1] - p0[1]) / 6.0
            cp2x = p2[0] - (p3[0] - p1[0]) / 6.0
            cp2y = p2[1] - (p3[1] - p1[1]) / 6.0

            d_tokens.append(f"C {cp1x:.2f} {cp1y:.2f}, {cp2x:.2f} {cp2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}")

        return " ".join(d_tokens)

    @staticmethod
    def to_svg(strokes: List[Dict[str, Any]], width: int, height: int, output_path: Union[str, Path], include_order_colors: bool = True) -> str:
        """
        Exports strokes to an SVG string/file with embedded kinematic ordering and smooth cubic curves.
        """
        # Palette for distinct pen-lift strokes
        colors = ["#38bdf8", "#10b981", "#f59e0b", "#c084fc", "#f43f5e", "#06b6d4", "#a855f7"]

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="overflow:visible;">',
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
            avg_width = sum(p[2] for p in norm_pts) / len(norm_pts)
            stroke_width = max(2.0, min(avg_width * 1.3, 7.0))

            d_str = VectorExporter._points_to_smooth_path_d(norm_pts)
            start_y, start_x = norm_pts[0][0], norm_pts[0][1]

            svg_lines.append(
                f'    <path id="{stroke.get("stroke_id", f"s{i}")}" d="{d_str}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_width:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round" data-order="{stroke.get("order_index", i)}"/>'
            )
            # Pen-down touch point indicator (small circle on stroke start)
            dot_r = max(2.5, min(stroke_width * 0.8, 5.0))
            svg_lines.append(
                f'    <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="{dot_r:.2f}" fill="#10b981" opacity="0.9"/>'
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
    def to_json(strokes: List[Dict[str, Any]], metadata: Dict[str, Any], output_path: Union[str, Path]) -> str:
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
