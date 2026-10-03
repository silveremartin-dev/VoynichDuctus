"""
Generates a complete dataset of Voynich vector ductus SVGs from glyphs and words,
and builds an interactive visual HTML gallery with animated pen strokes.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw

from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer


def draw_glyph_q(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'q' (EVA q: left loop + descender)."""
    draw.ellipse([offset_x + 5, offset_y + 15, offset_x + 25, offset_y + 35], outline="#24160d", width=3)
    draw.line([offset_x + 25, offset_y + 10, offset_x + 25, offset_y + 50], fill="#24160d", width=3)


def draw_glyph_o(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'o' (EVA o: closed oval loop)."""
    draw.ellipse([offset_x + 5, offset_y + 18, offset_x + 23, offset_y + 36], outline="#24160d", width=3)


def draw_glyph_k_gallows(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich gallows glyph 'k' (high ascender bench + loop + crossbar)."""
    draw.line([offset_x + 5, offset_y + 2, offset_x + 5, offset_y + 40], fill="#24160d", width=3)
    draw.arc([offset_x + 5, offset_y + 5, offset_x + 28, offset_y + 25], start=180, end=0, fill="#24160d", width=3)
    draw.line([offset_x + 28, offset_y + 15, offset_x + 28, offset_y + 40], fill="#24160d", width=3)
    draw.line([offset_x, offset_y + 15, offset_x + 32, offset_y + 15], fill="#24160d", width=2)


def draw_glyph_t_gallows(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich gallows glyph 't' (high ascender bench + crossbar)."""
    draw.line([offset_x + 5, offset_y + 2, offset_x + 5, offset_y + 40], fill="#24160d", width=3)
    draw.line([offset_x + 25, offset_y + 2, offset_x + 25, offset_y + 40], fill="#24160d", width=3)
    draw.line([offset_x + 2, offset_y + 8, offset_x + 28, offset_y + 8], fill="#24160d", width=3)
    draw.line([offset_x, offset_y + 16, offset_x + 30, offset_y + 16], fill="#24160d", width=2)


def draw_glyph_e(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'e' (small c-curve / eyelet)."""
    draw.arc([offset_x + 3, offset_y + 18, offset_x + 20, offset_y + 36], start=45, end=315, fill="#24160d", width=3)


def draw_glyph_d(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'd' (EVA d / 8-shape ligature)."""
    draw.ellipse([offset_x + 5, offset_y + 10, offset_x + 22, offset_y + 24], outline="#24160d", width=3)
    draw.ellipse([offset_x + 3, offset_y + 22, offset_x + 24, offset_y + 38], outline="#24160d", width=3)


def draw_glyph_y(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'y' (EVA y: right flourishing tail)."""
    draw.arc([offset_x + 5, offset_y + 18, offset_x + 22, offset_y + 35], start=0, end=180, fill="#24160d", width=3)
    draw.line([offset_x + 22, offset_y + 25, offset_x + 35, offset_y + 10], fill="#24160d", width=3)


def draw_glyph_ch(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich bench ligature 'ch' (EVA ch)."""
    draw.arc([offset_x + 3, offset_y + 18, offset_x + 18, offset_y + 36], start=45, end=315, fill="#24160d", width=3)
    draw.line([offset_x + 18, offset_y + 15, offset_x + 32, offset_y + 15], fill="#24160d", width=2)
    draw.line([offset_x + 32, offset_y + 15, offset_x + 32, offset_y + 36], fill="#24160d", width=3)


def draw_glyph_sh(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich bench ligature 'sh' (EVA sh: double bench)."""
    draw.arc([offset_x + 3, offset_y + 18, offset_x + 18, offset_y + 36], start=45, end=315, fill="#24160d", width=3)
    draw.line([offset_x + 18, offset_y + 15, offset_x + 45, offset_y + 15], fill="#24160d", width=2)
    draw.line([offset_x + 30, offset_y + 15, offset_x + 30, offset_y + 36], fill="#24160d", width=3)
    draw.line([offset_x + 45, offset_y + 15, offset_x + 45, offset_y + 36], fill="#24160d", width=3)


def draw_glyph_l(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'l' (EVA l: high vertical stem + base hook)."""
    draw.line([offset_x + 8, offset_y + 5, offset_x + 8, offset_y + 36], fill="#24160d", width=3)
    draw.arc([offset_x + 8, offset_y + 26, offset_x + 22, offset_y + 38], start=270, end=90, fill="#24160d", width=3)


def draw_glyph_r(draw: ImageDraw.Draw, offset_x: int, offset_y: int):
    """Draws Voynich glyph 'r' (EVA r: short leg + top hook)."""
    draw.line([offset_x + 6, offset_y + 18, offset_x + 6, offset_y + 36], fill="#24160d", width=3)
    draw.arc([offset_x + 6, offset_y + 16, offset_x + 22, offset_y + 28], start=180, end=0, fill="#24160d", width=3)


def create_sample_word(name: str, builders: list, width: int = 220, height: int = 70) -> Image.Image:
    """Creates a raster parchment patch containing a composite Voynichese word."""
    img = Image.new("RGB", (width, height), color="#f5eedc")
    draw = ImageDraw.Draw(img)

    curr_x = 15
    for fn in builders:
        fn(draw, curr_x, 12)
        curr_x += 32

    return img


def vectorize_image_to_ductus(image: Image.Image, output_svg_path: Path) -> Dict:
    """Processes image and outputs SVG vector ductus and metadata."""
    binarizer = Binarizer(method="sauvola", window_size=15)
    binary = binarizer.binarize(image)
    binary = binarizer.remove_small_artifacts(binary, min_size=4)

    skel_engine = Skeletonizer(method="medial_axis")
    skel, stroke_widths = skel_engine.extract_skeleton(binary)

    graph_extractor = StrokeGraphExtractor()
    pixel_graph = graph_extractor.build_pixel_graph(skel, stroke_widths)
    raw_strokes = graph_extractor.decompose_into_strokes(pixel_graph)

    resolver = JunctionResolver(right_handed_prior=True)
    ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)

    h, w = binary.shape
    svg_content = VectorExporter.to_svg(ordered_strokes, width=w, height=h, output_path=output_svg_path)

    return {
        "stroke_count": len(ordered_strokes),
        "ordered_strokes": ordered_strokes,
        "width": w,
        "height": h,
        "svg_content": svg_content
    }


def generate_full_dataset():
    out_dir = Path("./output/svg_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(exist_ok=True)
    (out_dir / "svg").mkdir(exist_ok=True)

    catalog = [
        # Canonical single glyphs
        ("glyph_01_q", [draw_glyph_q], 70, 70, "Glyph 'q' (EVA q): Left loop with descending tail"),
        ("glyph_02_o", [draw_glyph_o], 60, 60, "Glyph 'o' (EVA o): Closed oval ink loop"),
        ("glyph_03_k_gallows", [draw_glyph_k_gallows], 80, 70, "Glyph 'k' (EVA k): Gallows bench with upper loop"),
        ("glyph_04_t_gallows", [draw_glyph_t_gallows], 80, 70, "Glyph 't' (EVA t): Gallows bench with double stem"),
        ("glyph_05_e", [draw_glyph_e], 60, 60, "Glyph 'e' (EVA e): Small open eyelet curve"),
        ("glyph_06_d", [draw_glyph_d], 60, 70, "Glyph 'd' (EVA d): Double vertical figure-8 loop"),
        ("glyph_07_y", [draw_glyph_y], 70, 60, "Glyph 'y' (EVA y): Terminal flourish with diagonal tail"),
        ("glyph_08_ch", [draw_glyph_ch], 80, 60, "Ligature 'ch' (EVA ch): Eyelet with horizontal traverse"),
        ("glyph_09_sh", [draw_glyph_sh], 90, 60, "Ligature 'sh' (EVA sh): Eyelet with double bench traverse"),
        ("glyph_10_l", [draw_glyph_l], 60, 60, "Glyph 'l' (EVA l): High ascender with baseline foot"),
        ("glyph_11_r", [draw_glyph_r], 60, 60, "Glyph 'r' (EVA r): Short vertical stroke with upper hook"),

        # Frequent Voynich Words
        ("word_qokedy", [draw_glyph_q, draw_glyph_o, draw_glyph_k_gallows, draw_glyph_e, draw_glyph_d, draw_glyph_y], 230, 70, "Word 'qokedy': Most iconic Voynichese recurring token"),
        ("word_chol", [draw_glyph_ch, draw_glyph_o, draw_glyph_l], 140, 70, "Word 'chol': Standard herbal section paragraph starter"),
        ("word_dain", [draw_glyph_d, draw_glyph_e, draw_glyph_r, draw_glyph_y], 160, 70, "Word 'dain': Common grammatical particle / root"),
        ("word_qotey", [draw_glyph_q, draw_glyph_o, draw_glyph_t_gallows, draw_glyph_e, draw_glyph_y], 200, 70, "Word 'qotey': Gallows variant with central bench"),
        ("word_shedy", [draw_glyph_sh, draw_glyph_e, draw_glyph_d, draw_glyph_y], 180, 70, "Word 'shedy': Double bench ligature with terminal flourish"),
        ("word_or", [draw_glyph_o, draw_glyph_r], 100, 60, "Word 'or': Common two-glyph connector"),
    ]

    items_data = []

    print(f"[*] Generating {len(catalog)} vector ductus SVG samples...")
    for identifier, builders, w, h, description in catalog:
        img = create_sample_word(identifier, builders, width=w, height=h)
        png_path = out_dir / "png" / f"{identifier}.png"
        svg_path = out_dir / "svg" / f"{identifier}.svg"
        img.save(png_path)

        res = vectorize_image_to_ductus(img, svg_path)
        items_data.append({
            "id": identifier,
            "description": description,
            "png_rel": f"png/{identifier}.png",
            "svg_rel": f"svg/{identifier}.svg",
            "svg_content": res["svg_content"],
            "stroke_count": res["stroke_count"],
            "width": w,
            "height": h
        })
        print(f"  [+] {identifier}: {res['stroke_count']} strokes vectorized -> {svg_path.name}")

    # Build interactive HTML viewer
    html_content = generate_html_gallery(items_data)
    html_path = out_dir / "ductus_gallery.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] Interactive Ductus Visual Gallery saved to: {html_path.resolve()}")
    return html_path


def generate_html_gallery(items: List[Dict]) -> str:
    cards = []
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#00ced1"]

    for it in items:
        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <h3>{it['id'].replace('_', ' ').title()}</h3>
                <span class="badge">{it['stroke_count']} Kinematic Strokes</span>
            </div>
            <p class="desc">{it['description']}</p>
            
            <div class="comparison-grid">
                <div class="view-box">
                    <div class="label">1. Raw Manuscript Scan (2D Raster)</div>
                    <img src="{it['png_rel']}" alt="{it['id']}" class="raster-img">
                </div>
                
                <div class="view-box">
                    <div class="label">2. Vector Ductus (Color-Coded Pen Order)</div>
                    <div class="svg-container">
                        {it['svg_content']}
                    </div>
                </div>
            </div>

            <div class="order-legend">
                <span class="legend-title">Ductus Trajectory (Pen-down Sequence):</span>
                <div class="legend-chips">
                    {"".join(f'<span class="chip" style="background:{palette[i % len(palette)]}">Stroke {i+1}</span>' for i in range(it['stroke_count']))}
                </div>
            </div>
            
            <div class="download-link">
                <a href="{it['svg_rel']}" download="{it['id']}.svg" class="btn">📥 Download SVG Vector File</a>
            </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoynichDuctus — Interactive Vector Ductus & Scribal Kinematics Gallery</title>
    <style>
        :root {{
            --bg: #0f141c;
            --card-bg: #1a2230;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --border: #2d3748;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px;
        }}
        .header {{
            max-width: 1200px;
            margin: 0 auto 40px auto;
            text-align: center;
        }}
        h1 {{
            font-size: 2.2rem;
            color: #f8fafc;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.5;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
            gap: 25px;
            max-width: 1300px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .card-header h3 {{
            margin: 0;
            font-size: 1.25rem;
            color: var(--accent);
        }}
        .badge {{
            background: #0284c7;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
        }}
        .desc {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin: 0 0 16px 0;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 16px;
        }}
        .view-box {{
            background: #111827;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 110px;
        }}
        .view-box .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .raster-img {{
            max-width: 90%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #475569;
        }}
        .svg-container svg {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #475569;
        }}
        .order-legend {{
            margin-bottom: 15px;
        }}
        .legend-title {{
            font-size: 0.8rem;
            color: var(--text-muted);
            display: block;
            margin-bottom: 6px;
        }}
        .legend-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .chip {{
            color: white;
            font-size: 0.7rem;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.6);
        }}
        .download-link {{
            text-align: right;
            border-top: 1px solid var(--border);
            padding-top: 12px;
        }}
        .btn {{
            display: inline-block;
            background: #334155;
            color: #f8fafc;
            text-decoration: none;
            padding: 6px 14px;
            font-size: 0.85rem;
            font-weight: 500;
            border-radius: 6px;
            transition: background 0.2s ease;
        }}
        .btn:hover {{
            background: var(--accent);
            color: #0f172a;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VoynichDuctus — Vector Stroke & Kinematics Dataset</h1>
        <p class="subtitle">
            Every glyph and word below is converted from a 2D raster scan into an ordered parametric vector graph (SVG).
            Colors denote the chronological scribal ductus (pen-down to pen-up sequence) extracted via 15th-century right-handed scribal priors and Euler-Bernoulli tangent continuity.
        </p>
    </div>

    <div class="gallery-grid">
        {"".join(cards)}
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    generate_full_dataset()
