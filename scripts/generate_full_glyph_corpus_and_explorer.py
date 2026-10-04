"""
Full-scale Glyph Extraction, Kinematic Ductus Vectorization, Standard Corpus Matching, and Interactive Grand Explorer.
Extracts individual isolated glyphs across Voynich Manuscript and Codex Seraphinianus,
induces canonical alphabets with standard corpus correspondences (EVA, Currier, Serafini),
indexes all spatial coordinates (x, y) across all folios, and renders the comprehensive interactive explorer
featuring interactive Zoom/Pan, Minimap, Auto-Zoom on glyph click, and 3-way visual comparisons.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np

from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.pdf_loader import PDFScanLoader
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.ingestion.transcription_reference import PaleographyYieldValidator


def clean_page_data_for_json(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_pages = []
    for p in pages_data:
        clean_lines = []
        for l in p.get("lines", []):
            clean_lines.append({
                "line_id": l["line_id"],
                "bbox": [int(x) for x in l["bbox"]]
            })
        clean_glyphs = []
        for g in p.get("glyphs", []):
            clean_glyphs.append({
                "glyph_id": g["glyph_id"],
                "page_id": g["page_id"],
                "line_id": g["line_id"],
                "word_id": g["word_id"],
                "bbox": [int(x) for x in g["bbox"]],
                "height": int(g["height"]),
                "width": int(g["width"]),
                "area": int(g["area"]),
                "fill_factor": float(g.get("fill_factor", 0.25)),
                "stroke_count": int(g.get("stroke_count", len(g.get("strokes", [])))),
                "png_rel": g["png_rel"],
                "svg_rel": g["svg_rel"],
                "svg_content": g.get("svg_content", ""),
                "canonical_type": g.get("canonical_type", "")
            })
        clean_pages.append({
            "page_id": p["page_id"],
            "page_img_rel": p.get("page_img_rel", ""),
            "image_width": p.get("image_width", 1500),
            "image_height": p.get("image_height", 2000),
            "line_count": p["line_count"],
            "glyph_count": p["glyph_count"],
            "yield_report": p.get("yield_report", {}),
            "lines": clean_lines,
            "glyphs": clean_glyphs
        })
    return clean_pages


def generate_explorer_html(
    voynich_pages_data: List[Dict[str, Any]],
    voynich_catalogue: Dict[str, Any],
    serafini_pages_data: List[Dict[str, Any]],
    serafini_catalogue: Dict[str, Any],
    output_path: Path
):
    """
    Builds the interactive full-scale paleographic explorer web application.
    """
    app_data = {
        "voynich": {
            "title": "Voynich Manuscript (Beinecke MS 408)",
            "pages": clean_page_data_for_json(voynich_pages_data),
            "catalogue": voynich_catalogue
        },
        "seraphinianus": {
            "title": "Codex Seraphinianus (Luigi Serafini, 1981)",
            "pages": clean_page_data_for_json(serafini_pages_data),
            "catalogue": serafini_catalogue
        }
    }

    app_json = json.dumps(app_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoynichDuctus — Autonomous Digital Paleography & Grand Glyph Explorer</title>
    <style>
        :root {{
            --bg-dark: #090d16;
            --panel-bg: #111827;
            --card-bg: #1f2937;
            --card-hover: #374151;
            --border: #374151;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --voynich-color: #f59e0b;
            --serafini-color: #c084fc;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }}
        header {{
            background: var(--panel-bg);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand h1 {{ font-size: 1.25rem; font-weight: 700; }}
        .brand span {{ color: var(--accent); font-size: 0.8rem; }}
        
        .controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .btn-toggle {{
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 7px 13px;
            border-radius: 6px;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-toggle.active-voynich {{ background: var(--voynich-color); color: #000; font-weight: 700; border-color: var(--voynich-color); }}
        .btn-toggle.active-serafini {{ background: var(--serafini-color); color: #000; font-weight: 700; border-color: var(--serafini-color); }}
        
        .nav-tabs {{
            display: flex;
            gap: 4px;
            background: #0b1120;
            padding: 3px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .tab-btn {{
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.82rem;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: var(--card-bg);
            color: var(--text-main);
            font-weight: 600;
        }}

        main {{
            flex: 1;
            padding: 12px 16px;
            max-width: 1920px;
            margin: 0 auto;
            width: 100%;
        }}

        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }}
        .stat-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
        }}
        .stat-card h4 {{ font-size: 0.68rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px; }}
        .stat-card .val {{ font-size: 1.25rem; font-weight: 700; color: var(--accent); }}

        /* 4-Panel Page Layout */
        .page-view-layout {{
            display: grid;
            grid-template-columns: 210px 1.4fr 1fr 1.35fr;
            gap: 12px;
            height: calc(100vh - 150px);
        }}
        .panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        .panel h3 {{
            font-size: 0.85rem;
            margin-bottom: 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--border);
            color: var(--accent);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Page List */
        .page-item {{
            padding: 8px 10px;
            border-radius: 6px;
            cursor: pointer;
            background: var(--card-bg);
            margin-bottom: 6px;
            border: 1px solid transparent;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .page-item:hover {{ border-color: var(--accent); }}
        .page-item.active {{ background: #1e3a8a; border-color: var(--accent); font-weight: 600; }}

        /* Interactive Canvas Container & Controls */
        .canvas-wrapper {{
            position: relative;
            width: 100%;
            height: 100%;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: grab;
            user-select: none;
        }}
        .canvas-wrapper.grabbing {{
            cursor: grabbing;
        }}
        #page-canvas {{
            display: block;
            transform-origin: 0 0;
            position: absolute;
            left: 0;
            top: 0;
        }}

        /* Canvas Zoom Toolbar */
        .canvas-toolbar {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            z-index: 20;
        }}
        .btn-tool {{
            background: var(--card-bg);
            color: #fff;
            border: 1px solid var(--border);
            border-radius: 4px;
            width: 26px;
            height: 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            cursor: pointer;
        }}
        .btn-tool:hover {{ background: var(--accent); color: #000; }}
        .zoom-level-text {{
            font-size: 0.72rem;
            color: var(--text-muted);
            min-width: 44px;
            text-align: center;
        }}

        /* Minimap (Mini-vue) */
        .minimap-container {{
            position: absolute;
            top: 10px;
            right: 10px;
            width: 110px;
            height: 145px;
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid var(--accent);
            border-radius: 6px;
            overflow: hidden;
            z-index: 20;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.7);
            cursor: pointer;
        }}
        #minimap-canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .minimap-viewport-box {{
            position: absolute;
            border: 2px solid #38bdf8;
            background: rgba(56, 189, 248, 0.25);
            pointer-events: none;
        }}

        /* Glyphs Grid */
        .glyphs-header-controls {{
            display: flex;
            gap: 6px;
            margin-bottom: 8px;
            align-items: center;
        }}
        .select-filter {{
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 4px 6px;
            border-radius: 6px;
            font-size: 0.75rem;
            outline: none;
        }}
        .glyphs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
            gap: 6px;
            overflow-y: auto;
            flex: 1;
            padding-right: 4px;
        }}
        .glyph-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 5px;
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .glyph-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
        .glyph-card.selected {{
            border-color: var(--accent);
            background: rgba(56, 189, 248, 0.20);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }}
        .glyph-card img {{
            width: 46px;
            height: 46px;
            object-fit: contain;
            filter: contrast(110%);
        }}
        .glyph-card .gid {{
            font-size: 0.62rem;
            color: var(--text-muted);
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }}
        .badge-type {{
            background: #374151;
            color: var(--accent);
            padding: 1px 4px;
            border-radius: 4px;
            font-size: 0.62rem;
            font-weight: 700;
            margin-top: 2px;
            cursor: pointer;
        }}
        .badge-type:hover {{
            background: var(--accent);
            color: #000;
        }}

        /* Detailed Glyph Inspector (Expanded Previews) */
        .inspector-panel {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow-y: auto;
            flex: 1;
        }}
        .inspector-trio {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }}
        .preview-box-large {{
            background: #000;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 260px;
            min-height: 260px;
            text-align: center;
            overflow: hidden;
        }}
        .preview-box-large img {{
            width: 100%;
            height: 215px;
            object-fit: contain;
            filter: contrast(140%) brightness(102%) saturate(90%);
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }}
        .preview-box-large svg {{
            width: 100%;
            height: 215px;
        }}
        .meta-list {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.74rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .meta-list strong {{ color: var(--text-muted); }}

        .corpus-box {{
            background: #0b1727;
            border: 1px solid #1e3a8a;
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.75rem;
        }}
        .corpus-title {{
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 3px;
            display: flex;
            justify-content: space-between;
        }}

        /* Catalogue Tab */
        .catalogue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 14px;
            overflow-y: auto;
            max-height: calc(100vh - 170px);
        }}
        .catalogue-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .catalogue-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(0,0,0,0.5);
        }}
        .catalogue-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .catalogue-header h3 {{
            font-size: 1.05rem;
            color: var(--accent);
        }}
        .badge-freq {{
            background: #1e293b;
            color: #38bdf8;
            border: 1px solid #38bdf8;
            padding: 2px 7px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
        .catalogue-trio {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
        }}
        .c-box {{
            background: #000;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 85px;
        }}
        .c-box img, .c-box svg {{
            max-width: 100%;
            max-height: 70px;
            object-fit: contain;
        }}
        .btn-variations {{
            background: #1e3a8a;
            color: #fff;
            border: 1px solid var(--accent);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            cursor: pointer;
            text-align: center;
            font-weight: 600;
            margin-top: 4px;
        }}
        .btn-variations:hover {{
            background: var(--accent);
            color: #000;
        }}

        /* Modal for Archetype Variations */
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            z-index: 100;
            display: none;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .modal-overlay.open {{ display: flex; }}
        .modal-content {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            width: 100%;
            max-width: 1250px;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
        }}
        .modal-header {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
        }}
        .modal-header h2 {{ font-size: 1.15rem; color: var(--accent); }}
        .modal-close {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.4rem;
            cursor: pointer;
            line-height: 1;
        }}
        .modal-close:hover {{ color: #fff; }}
        .modal-body {{
            padding: 16px;
            overflow-y: auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .variations-banner {{
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 14px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }}
        .variations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(125px, 1fr));
            gap: 8px;
            overflow-y: auto;
            max-height: 460px;
            padding-right: 4px;
        }}
        .var-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
            transition: all 0.15s;
            text-align: center;
        }}
        .var-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
            background: #283548;
        }}
        .var-card img {{
            width: 52px;
            height: 52px;
            object-fit: contain;
            margin-bottom: 4px;
        }}
        .var-card .loc {{
            font-size: 0.70rem;
            font-weight: 700;
            color: var(--accent);
        }}
        .var-card .coords {{
            font-size: 0.62rem;
            color: var(--text-muted);
            margin: 2px 0;
        }}
        .btn-locate {{
            background: #1e3a8a;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 0.62rem;
            padding: 3px 6px;
            margin-top: 3px;
            cursor: pointer;
            width: 100%;
        }}
        .btn-locate:hover {{
            background: var(--accent);
            color: #000;
            font-weight: 700;
        }}

        .yield-box {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 7px 10px;
            font-size: 0.72rem;
            margin-bottom: 6px;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>VoynichDuctus</h1>
            <span>Interactive Digital Paleography & Standard Corpus Engine</span>
        </div>
        <div class="controls">
            <button id="btn-voynich" class="btn-toggle active-voynich" onclick="setManuscript('voynich')">Voynich (Beinecke MS 408)</button>
            <button id="btn-serafini" class="btn-toggle" onclick="setManuscript('seraphinianus')">Codex Seraphinianus</button>
            <div class="nav-tabs">
                <button id="tab-pages" class="tab-btn active" onclick="setTab('pages')">Manuscript Canvas & Bounding Boxes</button>
                <button id="tab-catalogue" class="tab-btn" onclick="setTab('catalogue')">Canonical Alphabet & Corpora</button>
            </div>
        </div>
    </header>

    <main>
        <div class="stats-bar" id="stats-bar"></div>

        <!-- TAB 1: Pages & Overlays -->
        <div id="view-pages" class="page-view-layout">
            <!-- 1. Pages List -->
            <div class="panel">
                <h3>Processed Folios</h3>
                <div id="yield-panel"></div>
                <div id="pages-list" style="overflow-y:auto; flex:1;"></div>
            </div>

            <!-- 2. High-Res Canvas Overlay with Zoom & Minimap -->
            <div class="panel" style="padding: 4px;">
                <h3>
                    <span>Interactive Folio Canvas</span>
                    <span id="canvas-status" style="font-size:0.72rem; color:var(--text-muted);">Scroll to Zoom • Drag to Pan • Click box to inspect</span>
                </h3>
                <div class="canvas-wrapper" id="canvas-wrapper">
                    <canvas id="page-canvas"></canvas>

                    <!-- Minimap Mini-vue Overlay -->
                    <div class="minimap-container" id="minimap-container">
                        <canvas id="minimap-canvas"></canvas>
                        <div class="minimap-viewport-box" id="minimap-viewport"></div>
                    </div>

                    <!-- Zoom Controls & Image Enhancement Toolbar -->
                    <div class="canvas-toolbar">
                        <button class="btn-tool" onclick="zoomIn()" title="Zoom In">+</button>
                        <button class="btn-tool" onclick="zoomOut()" title="Zoom Out">-</button>
                        <button class="btn-tool" onclick="resetZoom()" title="Reset Zoom / Fit Page" style="font-size:0.7rem; width:34px;">Fit</button>
                        <button class="btn-tool" onclick="zoomActual()" title="100% Scale" style="font-size:0.7rem; width:34px;">1:1</button>
                        <span class="zoom-level-text" id="zoom-text">100%</span>
                        <select id="canvas-filter-select" class="select-filter" onchange="setCanvasFilter(this.value)" style="margin-left:4px; font-size:0.72rem; padding:2px 6px;">
                            <option value="crisp">✨ Ink Boost (Crisp)</option>
                            <option value="sharp">🔥 High-Pass Sharp</option>
                            <option value="pure">📜 Pure Ink (Binarized)</option>
                            <option value="raw">📷 Natural Scan (Raw)</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- 3. Glyphs Grid -->
            <div class="panel">
                <div class="glyphs-header-controls">
                    <h3 style="border:none; margin:0; padding:0; flex:1;" id="glyphs-title">Extracted Glyphs</h3>
                    <select id="archetype-filter" class="select-filter" onchange="filterGlyphsByArchetype(this.value)">
                        <option value="ALL">All Archetypes</option>
                    </select>
                </div>
                <div class="glyphs-grid" id="glyphs-grid"></div>
            </div>

            <!-- 4. Detailed Inspector -->
            <div class="panel">
                <h3>Glyph Ductus & Standard Corpus Comparison</h3>
                <div class="inspector-panel" id="inspector-content"></div>
            </div>
        </div>

        <!-- TAB 2: Catalogue & Standard Corpora -->
        <div id="view-catalogue" style="display: none;">
            <div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 id="catalogue-title" style="font-size:1.25rem; color:var(--accent);">Canonical Alphabet Archetypes & Standard Transliteration Corpora</h2>
                    <p style="color:var(--text-muted); font-size:0.82rem; margin-top:2px;">
                        Side-by-side comparison of induced exemplars, kinematic ductus, and official reference calligraphy (EVA / Currier / Serafini). Click any card to inspect all variations.
                    </p>
                </div>
            </div>
            <div class="catalogue-grid" id="catalogue-grid"></div>
        </div>
    </main>

    <!-- Modal for Archetype Variations -->
    <div class="modal-overlay" id="variations-modal" onclick="closeModalOnOverlay(event)">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-archetype-title">Archetype Variations & Standard Corpus Correspondence</h2>
                <button class="modal-close" onclick="closeVariationsModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="variations-banner" id="modal-banner"></div>
                <div>
                    <h4 style="font-size:0.85rem; color:var(--accent); margin-bottom:8px;" id="modal-count-header">All Extracted Occurrences Across Folios</h4>
                    <div class="variations-grid" id="modal-variations-grid"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const data = {app_json};
        let currentMs = 'voynich';
        let currentTab = 'pages';
        let currentPageIdx = 0;
        let selectedGlyph = null;
        let pageImageObj = null;
        let archetypeFilterVal = 'ALL';

        // Pan & Zoom Engine State
        let zoomScale = 1.0;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let startPanX = 0;
        let startPanY = 0;

        function setManuscript(ms) {{
            currentMs = ms;
            currentPageIdx = 0;
            selectedGlyph = null;
            archetypeFilterVal = 'ALL';
            zoomScale = 1.0;
            panX = 0;
            panY = 0;
            
            document.getElementById('btn-voynich').className = ms === 'voynich' ? 'btn-toggle active-voynich' : 'btn-toggle';
            document.getElementById('btn-serafini').className = ms === 'seraphinianus' ? 'btn-toggle active-serafini' : 'btn-toggle';
            
            render();
        }}

        function setTab(tab) {{
            currentTab = tab;
            document.getElementById('tab-pages').className = tab === 'pages' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tab-catalogue').className = tab === 'catalogue' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('view-pages').style.display = tab === 'pages' ? 'grid' : 'none';
            document.getElementById('view-catalogue').style.display = tab === 'catalogue' ? 'block' : 'none';
            render();
        }}

        function render() {{
            const msData = data[currentMs];
            const totalGlyphs = msData.pages.reduce((acc, p) => acc + p.glyph_count, 0);
            const totalLines = msData.pages.reduce((acc, p) => acc + p.line_count, 0);

            document.getElementById('stats-bar').innerHTML = `
                <div class="stat-card">
                    <h4>Manuscript</h4>
                    <div class="val" style="font-size:1.0rem; color:${{currentMs === 'voynich' ? 'var(--voynich-color)' : 'var(--serafini-color)'}};">${{msData.title}}</div>
                </div>
                <div class="stat-card">
                    <h4>Processed Pages</h4>
                    <div class="val">${{msData.pages.length}}</div>
                </div>
                <div class="stat-card">
                    <h4>Identified Lines</h4>
                    <div class="val">${{totalLines}}</div>
                </div>
                <div class="stat-card">
                    <h4>Pure Isolated Glyphs</h4>
                    <div class="val">${{totalGlyphs}}</div>
                </div>
                <div class="stat-card">
                    <h4>Canonical Alphabet</h4>
                    <div class="val">${{msData.catalogue.canonical_alphabet_size}} Archetypes</div>
                </div>
            `;

            if (currentTab === 'pages') {{
                renderPagesTab();
            }} else {{
                renderCatalogueTab();
            }}
        }}

        function renderPagesTab() {{
            const msData = data[currentMs];
            const pagesList = document.getElementById('pages-list');
            pagesList.innerHTML = msData.pages.map((p, idx) => `
                <div class="page-item ${{idx === currentPageIdx ? 'active' : ''}}" onclick="selectPage(${{idx}})">
                    <div>
                        <strong>${{p.page_id}}</strong>
                        <div style="font-size:0.72rem; color:var(--text-muted);">${{p.line_count}} lines</div>
                    </div>
                    <span class="badge-freq">${{p.glyph_count}} glyphs</span>
                </div>
            `).join('');

            const page = msData.pages[currentPageIdx];
            const yr = page.yield_report;
            const yieldPanel = document.getElementById('yield-panel');
            if (yr && yr.has_reference) {{
                yieldPanel.innerHTML = `
                    <div class="yield-box">
                        <div style="font-weight:600; color:var(--accent); margin-bottom:3px;">Census Benchmark (${{yr.folio_id}})</div>
                        <div>Lines: <strong>${{yr.detected_lines}}/${{yr.expected_lines}}</strong> (${{yr.line_yield_pct}}%)</div>
                        <div>Glyphs: <strong>${{yr.detected_glyphs}}/${{yr.expected_glyphs}}</strong> (${{yr.glyph_yield_pct}}%)</div>
                        <div>Hand: <strong>${{yr.scribe_hand}}</strong> (${{yr.currier_language}})</div>
                        <div style="color:var(--text-muted); font-size:0.68rem; margin-top:2px;">Section: ${{yr.section}}</div>
                    </div>
                `;
            }} else {{
                yieldPanel.innerHTML = '';
            }}

            populateArchetypeFilter();
            loadPageImageAndDraw();
            renderGlyphGrid();
            renderInspector();
        }}

        function populateArchetypeFilter() {{
            const cat = data[currentMs].catalogue;
            const select = document.getElementById('archetype-filter');
            const options = ['<option value="ALL">All Archetypes</option>'];
            cat.alphabet.forEach(a => {{
                const label = currentMs === 'voynich' ? `${{a.type_id}} (EVA: '${{a.corpus_match?.eva_equivalent || '?'}}')` : `${{a.type_id}} (${{a.corpus_match?.serafini_code || 'S'}})`;
                options.push(`<option value="${{a.type_id}}" ${{archetypeFilterVal === a.type_id ? 'selected' : ''}}>${{label}}</option>`);
            }});
            select.innerHTML = options.join('');
        }}

        function filterGlyphsByArchetype(val) {{
            archetypeFilterVal = val;
            renderGlyphGrid();
            drawCanvasOverlay();
        }}

        function selectPage(idx) {{
            currentPageIdx = idx;
            zoomScale = 1.0;
            panX = 0;
            panY = 0;
            renderPagesTab();
        }}

        function loadPageImageAndDraw() {{
            const page = data[currentMs].pages[currentPageIdx];
            if (page.page_img_rel) {{
                pageImageObj = new Image();
                pageImageObj.src = page.page_img_rel;
                pageImageObj.onload = () => {{
                    fitCanvasToWrapper();
                    drawCanvasOverlay();
                    drawMinimap();
                    if (selectedGlyph) {{
                        autoZoomOnGlyph(selectedGlyph);
                    }}
                }};
            }}
        }}

        function fitCanvasToWrapper() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const canvas = document.getElementById('page-canvas');
            if (!pageImageObj || !pageImageObj.naturalWidth) return;

            canvas.width = pageImageObj.naturalWidth;
            canvas.height = pageImageObj.naturalHeight;

            const scaleW = wrapper.clientWidth / canvas.width;
            const scaleH = wrapper.clientHeight / canvas.height;
            zoomScale = Math.min(scaleW, scaleH) * 0.96;
            panX = (wrapper.clientWidth - canvas.width * zoomScale) / 2;
            panY = (wrapper.clientHeight - canvas.height * zoomScale) / 2;
            applyCanvasTransform();
        }}

        let currentCanvasFilter = 'crisp';
        const canvasFilterPresets = {{
            crisp: 'contrast(145%) brightness(102%) saturate(90%)',
            sharp: 'contrast(180%) brightness(108%) saturate(75%) drop-shadow(0 0 1px rgba(0,0,0,0.6))',
            pure: 'grayscale(100%) contrast(240%) brightness(92%)',
            raw: 'none'
        }};

        function setCanvasFilter(mode) {{
            currentCanvasFilter = mode;
            applyCanvasTransform();
        }}

        function applyCanvasTransform() {{
            const canvas = document.getElementById('page-canvas');
            canvas.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{zoomScale}})`;
            canvas.style.filter = canvasFilterPresets[currentCanvasFilter] || 'none';
            document.getElementById('zoom-text').innerText = `${{Math.round(zoomScale * 100)}}%`;
            updateMinimapViewport();
        }}

        function drawCanvasOverlay() {{
            const canvas = document.getElementById('page-canvas');
            const page = data[currentMs].pages[currentPageIdx];
            if (!pageImageObj || !pageImageObj.complete || pageImageObj.naturalWidth === 0) return;

            canvas.width = pageImageObj.naturalWidth;
            canvas.height = pageImageObj.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(pageImageObj, 0, 0);

            // Draw discrete and elegant bounding boxes
            page.glyphs.forEach(g => {{
                if (archetypeFilterVal !== 'ALL' && g.canonical_type !== archetypeFilterVal) {{
                    return;
                }}

                const [y0, x0, y1, x1] = g.bbox;
                const isSelected = selectedGlyph && selectedGlyph.glyph_id === g.glyph_id;

                if (isSelected) {{
                    // Selected box: crisp glowing cyan rectangular frame (NO distracting circles!)
                    ctx.strokeStyle = '#00e5ff';
                    ctx.lineWidth = 3.5;
                    ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
                    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                }} else {{
                    // Default box: subtle, elegant semi-transparent frame
                    ctx.strokeStyle = 'rgba(56, 189, 248, 0.40)';
                    ctx.lineWidth = 1.5;
                    ctx.fillStyle = 'rgba(56, 189, 248, 0.04)';
                    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                }}
            }});
        }}

        function drawMinimap() {{
            const mCanvas = document.getElementById('minimap-canvas');
            if (!pageImageObj || !pageImageObj.naturalWidth) return;
            mCanvas.width = 110;
            mCanvas.height = 145;
            const ctx = mCanvas.getContext('2d');
            ctx.drawImage(pageImageObj, 0, 0, mCanvas.width, mCanvas.height);
            updateMinimapViewport();
        }}

        function updateMinimapViewport() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const canvas = document.getElementById('page-canvas');
            const vBox = document.getElementById('minimap-viewport');
            if (!canvas.width || !wrapper.clientWidth) return;

            const scaleX = 110 / canvas.width;
            const scaleY = 145 / canvas.height;

            const visibleW = (wrapper.clientWidth / zoomScale) * scaleX;
            const visibleH = (wrapper.clientHeight / zoomScale) * scaleY;
            const visibleX = (-panX / zoomScale) * scaleX;
            const visibleY = (-panY / zoomScale) * scaleY;

            vBox.style.width = `${{Math.max(10, Math.min(110, visibleW))}}px`;
            vBox.style.height = `${{Math.max(10, Math.min(145, visibleH))}}px`;
            vBox.style.left = `${{Math.max(0, Math.min(110 - visibleW, visibleX))}}px`;
            vBox.style.top = `${{Math.max(0, Math.min(145 - visibleH, visibleY))}}px`;
        }}

        // Auto-Zoom on glyph to a clear, legible reading magnification
        function autoZoomOnGlyph(glyph) {{
            const wrapper = document.getElementById('canvas-wrapper');
            const [y0, x0, y1, x1] = glyph.bbox;
            const centerX = (x0 + x1) / 2;
            const centerY = (y0 + y1) / 2;

            zoomScale = 2.6; // Clear line reading zoom factor
            panX = wrapper.clientWidth / 2 - centerX * zoomScale;
            panY = wrapper.clientHeight / 2 - centerY * zoomScale;

            applyCanvasTransform();
        }}

        function selectGlyph(glyph, shouldAutoZoom = true) {{
            selectedGlyph = glyph;
            
            const msData = data[currentMs];
            const pIdx = msData.pages.findIndex(p => p.page_id === glyph.page_id);
            if (pIdx !== -1 && pIdx !== currentPageIdx) {{
                currentPageIdx = pIdx;
                renderPagesTab();
                return;
            }}

            drawCanvasOverlay();
            if (shouldAutoZoom) {{
                autoZoomOnGlyph(glyph);
            }}
            renderGlyphGrid();
            renderInspector();
        }}

        // Setup Interactive Mouse Zoom & Pan
        (function setupPanZoom() {{
            const wrapper = document.getElementById('canvas-wrapper');
            
            wrapper.addEventListener('wheel', (evt) => {{
                evt.preventDefault();
                const rect = wrapper.getBoundingClientRect();
                const mouseX = evt.clientX - rect.left;
                const mouseY = evt.clientY - rect.top;

                const zoomFactor = evt.deltaY < 0 ? 1.18 : 0.85;
                const newScale = Math.max(0.2, Math.min(8.0, zoomScale * zoomFactor));

                panX = mouseX - (mouseX - panX) * (newScale / zoomScale);
                panY = mouseY - (mouseY - panY) * (newScale / zoomScale);
                zoomScale = newScale;

                applyCanvasTransform();
            }});

            wrapper.addEventListener('mousedown', (evt) => {{
                if (evt.button !== 0) return;
                isPanning = true;
                startPanX = evt.clientX - panX;
                startPanY = evt.clientY - panY;
                wrapper.classList.add('grabbing');
            }});

            window.addEventListener('mousemove', (evt) => {{
                if (!isPanning) return;
                panX = evt.clientX - startPanX;
                panY = evt.clientY - startPanY;
                applyCanvasTransform();
            }});

            window.addEventListener('mouseup', () => {{
                isPanning = false;
                wrapper.classList.remove('grabbing');
            }});

            // Click on canvas box
            wrapper.addEventListener('click', (evt) => {{
                if (isPanning) return;
                const rect = wrapper.getBoundingClientRect();
                const clickCanvasX = (evt.clientX - rect.left - panX) / zoomScale;
                const clickCanvasY = (evt.clientY - rect.top - panY) / zoomScale;

                const page = data[currentMs].pages[currentPageIdx];
                const hit = page.glyphs.find(g => {{
                    const [y0, x0, y1, x1] = g.bbox;
                    return clickCanvasX >= x0 - 6 && clickCanvasX <= x1 + 6 && clickCanvasY >= y0 - 6 && clickCanvasY <= y1 + 6;
                }});

                if (hit) {{
                    selectGlyph(hit, false);
                }}
            }});

            // Minimap click to pan
            const minimap = document.getElementById('minimap-container');
            minimap.addEventListener('click', (evt) => {{
                evt.stopPropagation();
                const rect = minimap.getBoundingClientRect();
                const mX = (evt.clientX - rect.left) / 110;
                const mY = (evt.clientY - rect.top) / 145;

                const canvas = document.getElementById('page-canvas');
                const targetX = mX * canvas.width;
                const targetY = mY * canvas.height;

                panX = wrapper.clientWidth / 2 - targetX * zoomScale;
                panY = wrapper.clientHeight / 2 - targetY * zoomScale;
                applyCanvasTransform();
            }});
        }})();

        function zoomIn() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const centerX = wrapper.clientWidth / 2;
            const centerY = wrapper.clientHeight / 2;
            const newScale = Math.min(8.0, zoomScale * 1.3);
            panX = centerX - (centerX - panX) * (newScale / zoomScale);
            panY = centerY - (centerY - panY) * (newScale / zoomScale);
            zoomScale = newScale;
            applyCanvasTransform();
        }}

        function zoomOut() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const centerX = wrapper.clientWidth / 2;
            const centerY = wrapper.clientHeight / 2;
            const newScale = Math.max(0.2, zoomScale / 1.3);
            panX = centerX - (centerX - panX) * (newScale / zoomScale);
            panY = centerY - (centerY - panY) * (newScale / zoomScale);
            zoomScale = newScale;
            applyCanvasTransform();
        }}

        function resetZoom() {{
            fitCanvasToWrapper();
        }}

        function zoomActual() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const centerX = wrapper.clientWidth / 2;
            const centerY = wrapper.clientHeight / 2;
            panX = centerX - (centerX - panX) * (1.0 / zoomScale);
            panY = centerY - (centerY - panY) * (1.0 / zoomScale);
            zoomScale = 1.0;
            applyCanvasTransform();
        }}

        function renderGlyphGrid() {{
            const page = data[currentMs].pages[currentPageIdx];
            let filteredGlyphs = page.glyphs;
            if (archetypeFilterVal !== 'ALL') {{
                filteredGlyphs = filteredGlyphs.filter(g => g.canonical_type === archetypeFilterVal);
            }}

            document.getElementById('glyphs-title').innerText = `${{page.page_id}} — ${{filteredGlyphs.length}} Glyphs`;
            
            const grid = document.getElementById('glyphs-grid');
            grid.innerHTML = filteredGlyphs.map(g => `
                <div class="glyph-card ${{selectedGlyph && selectedGlyph.glyph_id === g.glyph_id ? 'selected' : ''}}" onclick='selectGlyph(${{JSON.stringify(g)}}, true)'>
                    <img src="${{g.png_rel}}" alt="${{g.glyph_id}}">
                    <span class="gid">${{g.glyph_id.split('_').slice(-2).join('_')}}</span>
                    <span class="badge-type" onclick="event.stopPropagation(); openArchetypeVariations('${{g.canonical_type}}')">${{g.canonical_type || 'G??'}}</span>
                </div>
            `).join('');
        }}

        function renderInspector() {{
            const ins = document.getElementById('inspector-content');
            if (!selectedGlyph) {{
                ins.innerHTML = `<p style="color:var(--text-muted); margin-top:40px; text-align:center;">Select any glyph from the grid or page overlay to inspect its coordinates and vector ductus.</p>`;
                return;
            }}

            const cat = data[currentMs].catalogue;
            const arch = cat.alphabet.find(a => a.type_id === selectedGlyph.canonical_type);
            const cm = arch?.corpus_match;

            let corpusHtml = '';
            let refSvgBox = '';
            if (cm) {{
                if (currentMs === 'voynich') {{
                    corpusHtml = `
                        <div class="corpus-box">
                            <div class="corpus-title">
                                <span>Standard Corpus: EVA '${{cm.eva_equivalent}}' / Currier '${{cm.currier_equivalent}}'</span>
                                <span style="color:var(--success); font-weight:bold;">${{cm.confidence_pct}}% match</span>
                            </div>
                            <div><strong>Category:</strong> ${{cm.category}} (${{cm.name}})</div>
                            <div style="color:var(--text-muted); font-size:0.72rem; margin-top:2px;">${{cm.description}}</div>
                        </div>
                    `;
                    refSvgBox = `
                        <div class="preview-box-large">
                            <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">EVA STANDARD REFERENCE</span>
                            ${{cm.reference_svg || ''}}
                        </div>
                    `;
                }} else {{
                    corpusHtml = `
                        <div class="corpus-box">
                            <div class="corpus-title">
                                <span>Typology: ${{cm.serafini_code}}</span>
                                <span style="color:var(--success); font-weight:bold;">${{cm.confidence_pct}}% match</span>
                            </div>
                            <div><strong>Category:</strong> ${{cm.category}} (${{cm.name}})</div>
                            <div style="color:var(--text-muted); font-size:0.72rem; margin-top:2px;">${{cm.description}}</div>
                        </div>
                    `;
                    refSvgBox = `
                        <div class="preview-box-large">
                            <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">SERAFINI REFERENCE</span>
                            ${{cm.reference_svg || ''}}
                        </div>
                    `;
                }}
            }}

            ins.innerHTML = `
                <!-- 3-Way Side-by-Side Large Comparison -->
                <div class="inspector-trio">
                    <div class="preview-box-large">
                        <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">ORIGINAL SCAN CROP</span>
                        <img src="${{selectedGlyph.png_rel}}" alt="${{selectedGlyph.glyph_id}}">
                    </div>
                    <div class="preview-box-large">
                        <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">KINEMATIC DUCTUS</span>
                        ${{selectedGlyph.svg_content}}
                    </div>
                    ${{refSvgBox}}
                </div>

                ${{corpusHtml}}

                <div class="meta-list">
                    <div><strong>Glyph ID:</strong> ${{selectedGlyph.glyph_id}}</div>
                    <div><strong>Page / Line:</strong> ${{selectedGlyph.page_id}} / Line ${{selectedGlyph.line_id}}</div>
                    <div><strong>Coordinates (y0, x0, y1, x1):</strong> [${{selectedGlyph.bbox.join(', ')}}]</div>
                    <div><strong>Dimensions:</strong> ${{selectedGlyph.width}}x${{selectedGlyph.height}} px (Area: ${{selectedGlyph.area}} px²)</div>
                    <div><strong>Fill Factor:</strong> ${{selectedGlyph.fill_factor}} (1D Filiform)</div>
                    <div><strong>Stroke Count:</strong> ${{selectedGlyph.stroke_count}} kinematic strokes</div>
                    <div><strong>Archetype:</strong> <span style="color:var(--accent); font-weight:bold; cursor:pointer;" onclick="openArchetypeVariations('${{selectedGlyph.canonical_type}}')">${{selectedGlyph.canonical_type}} (Click to view all variations)</span></div>
                </div>

                <button class="btn-variations" onclick="openArchetypeVariations('${{selectedGlyph.canonical_type}}')">🔍 View All ${{arch?.total_instances_count || ''}} Variations of ${{selectedGlyph.canonical_type}}</button>
                <a href="${{selectedGlyph.svg_rel}}" download class="btn-toggle" style="text-decoration:none; margin-top:2px; text-align:center;">⬇ Download Glyph SVG</a>
            `;
        }}

        function renderCatalogueTab() {{
            const cat = data[currentMs].catalogue;
            const grid = document.getElementById('catalogue-grid');
            grid.innerHTML = cat.alphabet.map(a => {{
                const cm = a.corpus_match;
                let matchHeader = '';
                let refSvgElem = '';
                if (cm) {{
                    if (currentMs === 'voynich') {{
                        matchHeader = `<div style="color:var(--accent); font-size:0.85rem; font-weight:bold;">EVA: '${{cm.eva_equivalent}}' | Currier: '${{cm.currier_equivalent}}' <span style="font-size:0.75rem; color:var(--success); font-weight:normal;">(${{cm.confidence_pct}}% match)</span></div>`;
                    }} else {{
                        matchHeader = `<div style="color:var(--accent); font-size:0.85rem; font-weight:bold;">${{cm.serafini_code}}: ${{cm.name}} <span style="font-size:0.75rem; color:var(--success); font-weight:normal;">(${{cm.confidence_pct}}% match)</span></div>`;
                    }}
                    refSvgElem = `
                        <div class="c-box">
                            <span style="font-size:0.58rem; color:var(--text-muted); margin-bottom:2px;">STANDARD REF</span>
                            ${{cm.reference_svg || ''}}
                        </div>
                    `;
                }}

                return `
                <div class="catalogue-card" onclick="openArchetypeVariations('${{a.type_id}}')">
                    <div class="catalogue-header">
                        <h3>Type ${{a.type_id}}</h3>
                        <span class="badge-freq">${{a.frequency}} occurrences (${{a.percentage}}%)</span>
                    </div>
                    ${{matchHeader}}
                    <div class="catalogue-trio">
                        <div class="c-box">
                            <span style="font-size:0.58rem; color:var(--text-muted); margin-bottom:2px;">EXEMPLAR</span>
                            <img src="${{a.exemplar_png}}" alt="${{a.type_id}}">
                        </div>
                        <div class="c-box">
                            <span style="font-size:0.58rem; color:var(--text-muted); margin-bottom:2px;">DUCTUS</span>
                            ${{a.exemplar_svg_content}}
                        </div>
                        ${{refSvgElem}}
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.3;">
                        <div>Mean Strokes: <strong>${{a.mean_strokes}}</strong> | Dim: <strong>${{a.mean_width}}x${{a.mean_height}} px</strong></div>
                        <div style="margin-top:2px; font-size:0.70rem;">Exemplar ID: <code>${{a.exemplar_id}}</code></div>
                    </div>
                    <button class="btn-variations" onclick="event.stopPropagation(); openArchetypeVariations('${{a.type_id}}')">
                        View All ${{a.total_instances_count || a.frequency}} Variations Across Pages ➔
                    </button>
                </div>
            `;
            }}).join('');
        }}

        // Archetype Variations Modal Functionality
        function openArchetypeVariations(typeId) {{
            const cat = data[currentMs].catalogue;
            const arch = cat.alphabet.find(a => a.type_id === typeId);
            if (!arch) return;

            const cm = arch.corpus_match;
            document.getElementById('modal-archetype-title').innerText = `Archetype ${{arch.type_id}} — Morphological Variations & Standard Cross-Reference`;
            
            let bannerHtml = `
                <div style="display:flex; gap:8px; background:#000; padding:8px; border-radius:6px; border:1px solid var(--border);">
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
                        <span style="font-size:0.62rem; color:var(--text-muted);">DUCTUS</span>
                        <div style="max-height:90px;">${{arch.exemplar_svg_content}}</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
                        <span style="font-size:0.62rem; color:var(--text-muted);">STANDARD REF</span>
                        <div style="max-height:90px;">${{cm?.reference_svg || ''}}</div>
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; justify-content:center; gap:4px;">
                    <div style="font-size:1.1rem; font-weight:bold; color:var(--accent);">
                        Type ${{arch.type_id}} (${{arch.frequency}} occurrences across corpus)
                    </div>
                    <div style="font-size:0.85rem; color:#fff;">
                        ${{cm ? (currentMs === 'voynich' ? `<strong>EVA Transliteration:</strong> <span style="color:var(--voynich-color); font-size:1.1rem; font-weight:bold;">'${{cm.eva_equivalent}}'</span> | <strong>Currier:</strong> '${{cm.currier_equivalent}}'` : `<strong>Serafini Typology:</strong> <span style="color:var(--serafini-color); font-weight:bold;">${{cm.serafini_code}}</span> (${{cm.name}})`) : ''}}
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted);">
                        <strong>Category:</strong> ${{cm?.category || 'General'}} | <strong>Confidence:</strong> <span style="color:var(--success); font-weight:bold;">${{cm?.confidence_pct || 90}}%</span>
                    </div>
                    <div style="font-size:0.72rem; color:var(--text-muted);">${{cm?.description || ''}}</div>
                </div>
            `;
            document.getElementById('modal-banner').innerHTML = bannerHtml;

            const instances = arch.all_instances || [];
            document.getElementById('modal-count-header').innerText = `All ${{instances.length}} Variations Across Folios (Click any item to locate on page)`;

            const varGrid = document.getElementById('modal-variations-grid');
            varGrid.innerHTML = instances.map(inst => `
                <div class="var-card" onclick="locateInstanceFromModal('${{inst.glyph_id}}', '${{inst.page_id}}')">
                    <img src="${{inst.png_rel}}" alt="${{inst.glyph_id}}">
                    <div class="loc">${{inst.page_id}}</div>
                    <div class="coords">[${{inst.bbox[0]}}, ${{inst.bbox[1]}}]</div>
                    <div style="font-size:0.62rem; color:var(--text-muted);">${{inst.stroke_count}} stroke(s)</div>
                    <button class="btn-locate">📍 Locate on Page</button>
                </div>
            `).join('');

            document.getElementById('variations-modal').classList.add('open');
        }}

        function closeVariationsModal() {{
            document.getElementById('variations-modal').classList.remove('open');
        }}

        function closeModalOnOverlay(evt) {{
            if (evt.target.id === 'variations-modal') {{
                closeVariationsModal();
            }}
        }}

        function locateInstanceFromModal(glyphId, pageId) {{
            closeVariationsModal();
            const msData = data[currentMs];
            const pIdx = msData.pages.findIndex(p => p.page_id === pageId);
            if (pIdx !== -1) {{
                currentPageIdx = pIdx;
                if (currentTab !== 'pages') {{
                    setTab('pages');
                }} else {{
                    renderPagesTab();
                }}
                
                const page = msData.pages[pIdx];
                const g = page.glyphs.find(x => x.glyph_id === glyphId);
                if (g) {{
                    selectGlyph(g, true);
                }}
            }}
        }}

        // Initial render
        render();
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[+] Full Glyph Corpus & Interactive Explorer generated at: {output_path.resolve()}")


def main():
    base_out = Path("./output/atlas_dataset")
    
    # 1. Clean slate: wipe old directories for 100% fresh extraction
    for sub in ["voynich", "seraphinianus"]:
        sub_dir = base_out / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        (base_out / sub / "pages").mkdir(parents=True, exist_ok=True)
        (base_out / sub / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
        (base_out / sub / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

    segmenter = GlyphSegmenter()
    catalogue_builder = GlyphCatalogue(target_alphabet_size=28)

    # 1. Process Voynich Manuscript Pages
    print("\n=======================================================")
    print(" 1. FULL GLYPH EXTRACTION: VOYNICH MANUSCRIPT")
    print("=======================================================")
    voynich_pages_data = []
    all_voynich_glyphs = []

    voynich_sources = [("f001r", Path("data/scans/f001r.jpg")), ("f001v", Path("data/scans/f001v.jpg"))]
    for folio_id, vf in voynich_sources:
        if vf.exists():
            img = Image.open(vf)
            print(f"[*] Extracting all glyphs on Voynich master '{folio_id}'...")
            res = segmenter.extract_page_glyphs(folio_id, img, output_dir=base_out, subfolder="voynich")
            
            # Census yield validation
            yr = PaleographyYieldValidator.evaluate_yield(
                folio_id=folio_id,
                detected_lines=res["line_count"],
                detected_glyphs=res["glyph_count"],
                extracted_strokes=sum(len(g["strokes"]) for g in res["glyphs"])
            )
            res["yield_report"] = yr
            print(f"  [+] Identified {res['line_count']}/{yr.get('expected_lines', 28)} lines ({yr.get('line_yield_pct', 0)}%), extracted {res['glyph_count']}/{yr.get('expected_glyphs', 1100)} glyphs ({yr.get('glyph_yield_pct', 0)}%).")
            voynich_pages_data.append(res)
            all_voynich_glyphs.extend(res["glyphs"])

    # Load additional folios from Voynich PDF
    voynich_pdf = Path("data/scans/voynich/VoynichManuscript.pdf")
    if voynich_pdf.exists():
        loader_v = PDFScanLoader(voynich_pdf)
        for page_idx in [2, 3, 4, 5]:
            try:
                folio_name = f"f{page_idx:03d}"
                print(f"[*] Extracting all glyphs on Voynich folio '{folio_name}' from PDF...")
                p_img = loader_v.get_page_image(page_idx, target_min_dim=1500)
                res_v = segmenter.extract_page_glyphs(folio_name, p_img, output_dir=base_out, subfolder="voynich")
                yr_v = PaleographyYieldValidator.evaluate_yield(
                    folio_id=folio_name,
                    detected_lines=res_v["line_count"],
                    detected_glyphs=res_v["glyph_count"],
                    extracted_strokes=sum(len(g["strokes"]) for g in res_v["glyphs"])
                )
                res_v["yield_report"] = yr_v
                print(f"  [+] Identified {res_v['line_count']}/{yr_v.get('expected_lines', 20)} lines ({yr_v.get('line_yield_pct', 0)}%), extracted {res_v['glyph_count']}/{yr_v.get('expected_glyphs', 750)} glyphs ({yr_v.get('glyph_yield_pct', 0)}%).")
                voynich_pages_data.append(res_v)
                all_voynich_glyphs.extend(res_v["glyphs"])
            except Exception as e:
                print(f"[-] Error on Voynich PDF page {page_idx}: {e}")

    print(f"\n[*] Inducing Voynich Canonical Alphabet & Matching Standard Corpora (EVA/Currier)...")
    voynich_catalogue = catalogue_builder.build_catalogue(all_voynich_glyphs, corpus_type="voynich")
    catalogue_builder.export_catalogue_json(voynich_catalogue, base_out / "voynich" / "voynich_alphabet_catalogue.json")
    print(f"  [+] Discovered {voynich_catalogue['canonical_alphabet_size']} unique canonical glyph archetypes across Voynich folios.")

    # 2. Process Codex Seraphinianus Pages
    print("\n=======================================================")
    print(" 2. FULL GLYPH EXTRACTION: CODEX SERAPHINIANUS")
    print("=======================================================")
    serafini_pages_data = []
    all_serafini_glyphs = []

    serafini_pdf = Path("data/scans/seraphinianus/Codex Seraphinianus.pdf")
    if serafini_pdf.exists():
        loader_s = PDFScanLoader(serafini_pdf)
        for p_idx in [15, 20, 25, 30, 35, 40]:
            try:
                page_id = f"serafini_p{p_idx:03d}"
                print(f"[*] Extracting all glyphs on Seraphinianus page {p_idx}...")
                p_img = loader_s.get_page_image(p_idx, target_min_dim=1500)
                res_s = segmenter.extract_page_glyphs(page_id, p_img, output_dir=base_out, subfolder="seraphinianus")
                print(f"  [+] Identified {res_s['line_count']} lines, extracted {res_s['glyph_count']} pure isolated glyphs.")
                serafini_pages_data.append(res_s)
                all_serafini_glyphs.extend(res_s["glyphs"])
            except Exception as e:
                print(f"[-] Error on Seraphinianus page {p_idx}: {e}")

    print(f"\n[*] Inducing Seraphinianus Canonical Alphabet & Matching Serafinian Typology...")
    serafini_catalogue = catalogue_builder.build_catalogue(all_serafini_glyphs, corpus_type="seraphinianus")
    catalogue_builder.export_catalogue_json(serafini_catalogue, base_out / "seraphinianus" / "seraphinianus_alphabet_catalogue.json")
    print(f"  [+] Discovered {serafini_catalogue['canonical_alphabet_size']} unique canonical glyph archetypes across Seraphinianus pages.")

    # 3. Generate Interactive Visual Explorer
    explorer_path = base_out / "grand_glyph_explorer.html"
    generate_explorer_html(voynich_pages_data, voynich_catalogue, serafini_pages_data, serafini_catalogue, explorer_path)


if __name__ == "__main__":
    main()
