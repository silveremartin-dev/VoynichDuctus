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
    def to_svg(strokes: List[Dict[str, Any]], width: int, height: int, output_path: Union[str, Path], include_order_colors: bool = True) -> str:
        """
        Exports strokes to an SVG string/file with embedded kinematic ordering.
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

            color = colors[i % len(colors)] if include_order_colors else "#38bdf8"
            avg_width = sum(p[2] if len(p) > 2 else 1.5 for p in pts) / len(pts)
            stroke_width = max(2.0, min(avg_width * 1.3, 7.0))

            # Build path data 'M x y L x y ...'
            start_y, start_x = pts[0][0], pts[0][1]
            path_d = [f"M {start_x:.2f} {start_y:.2f}"]
            for pt in pts[1:]:
                path_d.append(f"L {pt[1]:.2f} {pt[0]:.2f}")

            d_str = " ".join(path_d)
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
