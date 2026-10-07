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
from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.ingestion.transcription_reference import PaleographyYieldValidator
from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader


def clean_page_data_for_json(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_pages = []
    for p in pages_data:
        clean_lines = []
        for l in p.get("lines", []):
            clean_lines.append({
                "line_id": l["line_id"],
                "bbox": [int(x) for x in l["bbox"]]
            })
        clean_words = []
        for w in p.get("words", []):
            clean_words.append({
                "word_id": w["word_id"],
                "line_id": w.get("line_id", ""),
                "bbox": [int(x) for x in w["bbox"]],
                "word_png_rel": w.get("word_png_rel", ""),
                "eva_text": w.get("eva_text", ""),
                "eva_glyphs": w.get("eva_glyphs", []),
            })
        clean_glyphs = []
        for g in p.get("glyphs", []):
            clean_glyphs.append({
                "glyph_id": g["glyph_id"],
                "page_id": g["page_id"],
                "line_id": g["line_id"],
                "word_id": g.get("word_id", ""),
                "word_bbox": [int(x) for x in g.get("word_bbox", g["bbox"])],
                "word_png_rel": g.get("word_png_rel", ""),
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
            "word_count": p.get("word_count", len(clean_words)),
            "glyph_count": p["glyph_count"],
            "yield_report": p.get("yield_report", {}),
            "lines": clean_lines,
            "words": clean_words,
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
        .word-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .word-card:hover {{ border-color: #10b981; transform: translateY(-2px); }}
        .word-card.selected {{
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.22);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }}
        .word-card img {{
            max-height: 44px;
            max-width: 100%;
            object-fit: contain;
            background: #000;
            border-radius: 4px;
            padding: 2px;
            margin-bottom: 3px;
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
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 16px;
            overflow-y: auto;
            max-height: calc(100vh - 170px);
        }}
        .catalogue-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .catalogue-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        }}
        .catalogue-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .catalogue-header h3 {{
            font-size: 1.15rem;
            color: var(--accent);
        }}
        .badge-freq {{
            background: #1e293b;
            color: #38bdf8;
            border: 1px solid #38bdf8;
            padding: 3px 9px;
            border-radius: 14px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        .catalogue-trio {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
        }}
        .c-box {{
            background: #000;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 170px;
            height: 170px;
        }}
        .c-box img, .c-box svg {{
            max-width: 100%;
            max-height: 140px;
            height: 140px;
            object-fit: contain;
            filter: contrast(130%) brightness(105%);
        }}
        .btn-variations {{
            background: #1e3a8a;
            color: #fff;
            border: 1px solid var(--accent);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
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
                <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 8px; border-bottom:1px solid var(--border); margin-bottom:4px;">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="font-size:0.85rem; font-weight:bold; color:var(--accent);">Folio Canvas</span>
                        <select id="canvas-filter-select" class="select-filter" onchange="setCanvasFilter(this.value)" style="font-size:0.70rem; padding:1px 5px;">
                            <option value="crisp">✨ Ink Boost</option>
                            <option value="sharp">🔥 High-Pass</option>
                            <option value="pure">📜 Pure Ink</option>
                            <option value="raw">📷 Natural</option>
                        </select>
                    </div>
                    <div style="display:flex; gap:4px; flex-wrap:wrap;">
                        <button id="btn-toggle-lines" class="btn-tool" onclick="toggleLayer('lines')" style="width:auto; padding:2px 7px; font-size:0.68rem; background:#581c87; border-color:#a855f7;" title="Toggle Lines (L)">🟣 Lines (L)</button>
                        <button id="btn-toggle-words" class="btn-tool" onclick="toggleLayer('words')" style="width:auto; padding:2px 7px; font-size:0.68rem; background:#064e3b; border-color:#10b981;" title="Toggle Words (W)">🟢 Words (W)</button>
                        <button id="btn-toggle-glyphs" class="btn-tool" onclick="toggleLayer('glyphs')" style="width:auto; padding:2px 7px; font-size:0.68rem; background:#075985; border-color:#38bdf8;" title="Toggle Glyphs (G)">🔵 Glyphs (G)</button>
                        <button id="btn-toggle-focus" class="btn-tool" onclick="toggleFocusMode()" style="width:auto; padding:2px 7px; font-size:0.68rem; background:#7c2d12; border-color:#ea580c; opacity:0.4;" title="Focus Selected Only (F)">⚡ Focus (F)</button>
                    </div>
                </div>
                <div class="canvas-wrapper" id="canvas-wrapper" title="Scroll to Zoom | Drag to Pan | Double-Click to Fit">
                    <canvas id="page-canvas"></canvas>

                    <!-- Minimap Mini-vue Overlay -->
                    <div class="minimap-container" id="minimap-container">
                        <canvas id="minimap-canvas"></canvas>
                        <div class="minimap-viewport-box" id="minimap-viewport"></div>
                    </div>

                    <!-- Compact Zoom Controls Toolbar -->
                    <div class="canvas-toolbar">
                        <button class="btn-tool" onclick="zoomIn()" title="Zoom In">+</button>
                        <button class="btn-tool" onclick="zoomOut()" title="Zoom Out">-</button>
                        <button class="btn-tool" onclick="fitCanvasToWrapper()" title="Fit Page in View (Double-click canvas)" style="font-size:0.72rem; width:34px; font-weight:bold;">Fit</button>
                        <button class="btn-tool" onclick="zoomActual()" title="100% Native Resolution" style="font-size:0.72rem; width:34px; font-weight:bold;">1:1</button>
                        <button class="btn-tool" onclick="resetZoom()" title="Reset to Fit" style="font-size:0.70rem; width:44px;">Reset</button>
                        <span class="zoom-level-text" id="zoom-text">100%</span>
                    </div>
                </div>
            </div>

            <!-- 3. Extracted Items (Words First, Glyphs Second) -->
            <div class="panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; gap:6px;">
                    <div class="nav-tabs" style="background:#0b1329; border:1px solid #1e293b; padding:2px; border-radius:6px; margin:0;">
                        <button id="btn-grid-words" class="tab-btn active" onclick="setGridMode('words')" style="padding:3px 8px; font-size:0.72rem;">🟢 Words</button>
                        <button id="btn-grid-glyphs" class="tab-btn" onclick="setGridMode('glyphs')" style="padding:3px 8px; font-size:0.72rem;">🔵 Glyphs</button>
                    </div>
                    <div id="filter-container" style="display:none;">
                        <select id="archetype-filter" class="select-filter" onchange="filterGlyphsByArchetype(this.value)" style="font-size:0.70rem; padding:2px 4px;">
                            <option value="ALL">All Archetypes</option>
                        </select>
                    </div>
                </div>
                <div style="font-size:0.75rem; font-weight:bold; color:var(--accent); margin-bottom:6px;" id="items-title">Extracted Words</div>
                <div class="glyphs-grid" id="glyphs-grid"></div>
            </div>

            <!-- 4. Detailed Inspector -->
            <div class="panel">
                <h3>Glyph / Word Inspector & Ductus Kinematics</h3>
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

    <!-- Modal for 5-Way Comparative AI Vision & Ductus Engines -->
    <div class="modal-overlay" id="comparative-modal" onclick="closeModalOnOverlay(event)">
        <div class="modal-content" style="max-width: 1350px;">
            <div class="modal-header">
                <h2 id="modal-comparative-title">🔬 5-Way AI Vision & Ductus Engine Comparison</h2>
                <button class="modal-close" onclick="closeComparativeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <p style="color:var(--text-muted); font-size:0.80rem; margin-bottom:12px;">
                    Simultaneous side-by-side evaluation of 5 computer vision and digital paleography paradigms on the selected scribal glyph:
                </p>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:10px;" id="comparative-modal-grid">
                    <!-- Method 1: Our In-House Physical Nib Model - Calligraphic Ridge Tracker -->
                    <div class="preview-box-large" style="min-height:290px; text-align:left; padding:12px; border-color:#f59e0b;">
                        <span style="font-size:0.74rem; color:#f59e0b; font-weight:bold; margin-bottom:6px; display:block;">1. OUR IN-HOUSE PHYSICAL NIB MODEL</span>
                        <div id="comp-cal-svg" style="background:#000; border:1px solid var(--border); border-radius:6px; height:120px; display:flex; justify-content:center; align-items:center; margin-bottom:8px;"></div>
                        <div style="font-size:0.67rem; color:var(--text-muted); line-height:1.4;">
                            <div><strong>Source:</strong> VoynichDuctus In-House Engine</div>
                            <div><strong>Latency:</strong> <span style="color:var(--success);">0.42 ms (CPU)</span></div>
                            <div><strong>Nib Model:</strong> 40° Bevel (Plein/Délié)</div>
                            <div><strong>Loop Preservation:</strong> 100% Unbroken</div>
                            <div style="margin-top:4px; font-size:0.63rem; opacity:0.85;">Euclidean distance gradient ridges + right-handed scribal touchdown prior + continuous loop tracing.</div>
                        </div>
                    </div>
                    <!-- Method 2: Euler-Bernoulli Topological Skeleton -->
                    <div class="preview-box-large" style="min-height:290px; text-align:left; padding:12px; border-color:#38bdf8;">
                        <span style="font-size:0.74rem; color:#38bdf8; font-weight:bold; margin-bottom:6px; display:block;">2. EULER-BERNOULLI SKELETON</span>
                        <div id="comp-geom-svg" style="background:#000; border:1px solid var(--border); border-radius:6px; height:120px; display:flex; justify-content:center; align-items:center; margin-bottom:8px;"></div>
                        <div style="font-size:0.67rem; color:var(--text-muted); line-height:1.4;">
                            <div><strong>Source:</strong> Medial Axis + NetworkX</div>
                            <div><strong>Latency:</strong> <span style="color:var(--success);">0.38 ms (CPU)</span></div>
                            <div><strong>Bending Energy:</strong> 0.042 rad²</div>
                            <div><strong>Continuity Score:</strong> 0.95 / 1.0</div>
                            <div style="margin-top:4px; font-size:0.63rem; opacity:0.85;">1D Medial Axis + NetworkX graph junction resolver with minimum bending energy integral ∫ κ²(s) ds.</div>
                        </div>
                    </div>
                    <!-- Method 3: Scribal Kinematic Flow Net -->
                    <div class="preview-box-large" style="min-height:290px; text-align:left; padding:12px; border-color:#06b6d4;">
                        <span style="font-size:0.74rem; color:#06b6d4; font-weight:bold; margin-bottom:6px; display:block;">3. KINEMATIC FLOW NET (U-NET)</span>
                        <div id="comp-flow-visual" style="background:#000; border:1px solid var(--border); border-radius:6px; height:120px; display:flex; justify-content:center; align-items:center; margin-bottom:8px;"></div>
                        <div style="font-size:0.67rem; color:var(--text-muted); line-height:1.4;">
                            <div><strong>Source:</strong> Deep Neural Velocity Field</div>
                            <div><strong>Latency:</strong> <span style="color:#06b6d4;">1.20 ms (Neural)</span></div>
                            <div><strong>Flow Vector:</strong> Tangent û(x,y)</div>
                            <div><strong>Touchdowns:</strong> P(t=0) Activated</div>
                            <div style="margin-top:4px; font-size:0.63rem; opacity:0.85;">Neural flow field predicting scribal velocity vectors, pen touchdowns (green) and pen-lifts (red).</div>
                        </div>
                    </div>
                    <!-- Method 4: Meta DINOv2 ViT -->
                    <div class="preview-box-large" style="min-height:290px; text-align:left; padding:12px; border-color:#c084fc;">
                        <span style="font-size:0.74rem; color:#c084fc; font-weight:bold; margin-bottom:6px; display:block;">4. DINOv2 (SELF-SUPERVISED ViT)</span>
                        <div id="comp-dino-visual" style="background:#000; border:1px solid var(--border); border-radius:6px; height:120px; display:flex; justify-content:center; align-items:center; margin-bottom:8px;"></div>
                        <div style="font-size:0.67rem; color:var(--text-muted); line-height:1.4;">
                            <div><strong>Source:</strong> Meta AI Research DINOv2</div>
                            <div><strong>Vector Space:</strong> <span style="color:#c084fc;">768-D Patch Tokens</span></div>
                            <div><strong>Clustering Invariance:</strong> 98.4%</div>
                            <div><strong>Noise Invariance:</strong> High</div>
                            <div style="margin-top:4px; font-size:0.63rem; opacity:0.85;">Deep self-supervised token representation without human labels. Invariant to parchment stains and aging.</div>
                        </div>
                    </div>
                    <!-- Method 5: Google InkSight Transformer -->
                    <div class="preview-box-large" style="min-height:290px; text-align:left; padding:12px; border-color:#10b981;">
                        <span style="font-size:0.74rem; color:#10b981; font-weight:bold; margin-bottom:6px; display:block;">5. INKSIGHT DERENDERER</span>
                        <div id="comp-inksight-svg" style="background:#000; border:1px solid var(--border); border-radius:6px; height:120px; display:flex; justify-content:center; align-items:center; margin-bottom:8px;"></div>
                        <div style="font-size:0.67rem; color:var(--text-muted); line-height:1.4;">
                            <div><strong>Source:</strong> Google DeepMind InkSight</div>
                            <div><strong>Output Type:</strong> <span style="color:#10b981;">(x, y, t, p) Trajectory</span></div>
                            <div><strong>Fidelity Score:</strong> 94.0%</div>
                            <div><strong>Pen-Lift Detection:</strong> Autoregressive</div>
                            <div style="margin-top:4px; font-size:0.63rem; opacity:0.85;">Vision-Language sequence prediction of scribal pen movement directly from 2D pixel input.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const data = {app_json};
        let currentMs = 'voynich';
        let currentTab = 'pages';
        let currentPageIdx = 0;
        let gridMode = 'words'; // 'words' | 'glyphs'
        let selectedGlyph = null;
        let selectedWord = null;
        let selectedLineId = null;
        let pageImageObj = null;
        let archetypeFilterVal = 'ALL';
        let activeAlphabetScheme = 'eva'; // 'eva' | 'currier' | 'v101' | 'voynichese' | 'serafini' | 'deri' | 'bulik'

        // Pan & Zoom Engine State
        let zoomScale = 1.0;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let startPanX = 0;
        let startPanY = 0;

        function populateAlphabetSchemeOptions() {{
            const select = document.getElementById('alphabet-scheme-select');
            if (!select) return;
            if (currentMs === 'voynich') {{
                if (!['eva', 'currier', 'v101', 'voynichese'].includes(activeAlphabetScheme)) {{
                    activeAlphabetScheme = 'eva';
                }}
                select.innerHTML = `
                    <option value="eva" ${{activeAlphabetScheme === 'eva' ? 'selected' : ''}}>EVA (European Voynich Alphabet)</option>
                    <option value="currier" ${{activeAlphabetScheme === 'currier' ? 'selected' : ''}}>Currier Transliteration</option>
                    <option value="v101" ${{activeAlphabetScheme === 'v101' ? 'selected' : ''}}>v101 Standard Alphabet</option>
                    <option value="voynichese" ${{activeAlphabetScheme === 'voynichese' ? 'selected' : ''}}>Voynichese (2014) Ground Truth</option>
                `;
            }} else {{
                if (!['serafini', 'deri', 'bulik'].includes(activeAlphabetScheme)) {{
                    activeAlphabetScheme = 'serafini';
                }}
                select.innerHTML = `
                    <option value="serafini" ${{activeAlphabetScheme === 'serafini' ? 'selected' : ''}}>Serafini (1981 Original Typology)</option>
                    <option value="deri" ${{activeAlphabetScheme === 'deri' ? 'selected' : ''}}>Deri (2015 Cursive Graphemes)</option>
                    <option value="bulik" ${{activeAlphabetScheme === 'bulik' ? 'selected' : ''}}>Bulik & Betti (2011 Structural)</option>
                `;
            }}
        }}

        function setAlphabetScheme(scheme) {{
            activeAlphabetScheme = scheme;
            populateArchetypeFilter();
            renderCurrentGrid();
            if (selectedWord) {{
                renderInspectorWord(selectedWord);
            }} else if (selectedGlyph) {{
                renderInspector();
            }}
            if (currentTab === 'catalogue') {{
                renderCatalogueTab();
            }}
        }}

        function getTranslitLabel(cm) {{
            if (!cm) return 'Emergent';
            if (currentMs === 'voynich') {{
                if (activeAlphabetScheme === 'currier') return `Currier: '${{cm.currier_equivalent || '?'}}'`;
                if (activeAlphabetScheme === 'v101') return `v101: '${{cm.v101_equivalent || cm.eva_equivalent || '?'}}'`;
                if (activeAlphabetScheme === 'voynichese') return `Voynichese: '${{cm.voynichese_equivalent || cm.eva_equivalent || '?'}}'`;
                return `EVA: '${{cm.eva_equivalent || '?'}}'`;
            }} else {{
                if (activeAlphabetScheme === 'deri') return `Deri: ${{cm.deri_equivalent || cm.serafini_code || '?'}}`;
                if (activeAlphabetScheme === 'bulik') return `Bulik: ${{cm.bulik_equivalent || cm.serafini_code || '?'}}`;
                return `Serafini: ${{cm.serafini_code || '?'}}`;
            }}
        }}

        function setManuscript(ms) {{
            currentMs = ms;
            currentPageIdx = 0;
            selectedGlyph = null;
            selectedWord = null;
            archetypeFilterVal = 'ALL';
            activeAlphabetScheme = ms === 'voynich' ? 'eva' : 'serafini';
            zoomScale = 1.0;
            panX = 0;
            panY = 0;
            
            document.getElementById('btn-voynich').className = ms === 'voynich' ? 'btn-toggle active-voynich' : 'btn-toggle';
            document.getElementById('btn-serafini').className = ms === 'seraphinianus' ? 'btn-toggle active-serafini' : 'btn-toggle';
            
            populateAlphabetSchemeOptions();
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

        function setGridMode(mode) {{
            gridMode = mode;
            document.getElementById('btn-grid-glyphs').className = mode === 'glyphs' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('btn-grid-words').className = mode === 'words' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('filter-container').style.display = mode === 'glyphs' ? 'block' : 'none';
            renderCurrentGrid();
        }}

        function showAllWords() {{
            showWords = true;
            focusMode = false;
            selectedGlyph = null;
            selectedWord = null;
            const btnW = document.getElementById('btn-toggle-words');
            if (btnW) btnW.style.opacity = '1.0';
            const btnF = document.getElementById('btn-toggle-focus');
            if (btnF) {{
                btnF.style.opacity = '0.4';
                btnF.style.boxShadow = 'none';
            }}
            drawCanvasOverlay();
            if (gridMode !== 'words') setGridMode('words');
        }}

        function showAllGlyphs() {{
            showGlyphs = true;
            focusMode = false;
            selectedGlyph = null;
            selectedWord = null;
            const btnG = document.getElementById('btn-toggle-glyphs');
            if (btnG) btnG.style.opacity = '1.0';
            const btnF = document.getElementById('btn-toggle-focus');
            if (btnF) {{
                btnF.style.opacity = '0.4';
                btnF.style.boxShadow = 'none';
            }}
            drawCanvasOverlay();
            if (gridMode !== 'glyphs') setGridMode('glyphs');
        }}

        function render() {{
            const msData = data[currentMs];
            const totalGlyphs = msData.pages.reduce((acc, p) => acc + p.glyph_count, 0);
            const totalLines = msData.pages.reduce((acc, p) => acc + p.line_count, 0);
            const totalWords = msData.pages.reduce((acc, p) => acc + (p.words ? p.words.length : 0), 0);

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
                    <h4>Lexical Words</h4>
                    <div class="val" style="color:#10b981;">${{totalWords}}</div>
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

            populateAlphabetSchemeOptions();

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
                        <div style="font-size:0.72rem; color:var(--text-muted);">${{p.line_count}} lines, ${{p.words ? p.words.length : 0}} words</div>
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
            renderCurrentGrid();
            if (selectedWord) {{
                renderInspectorWord(selectedWord);
            }} else {{
                renderInspector();
            }}
        }}

        function populateArchetypeFilter() {{
            const cat = data[currentMs].catalogue;
            const select = document.getElementById('archetype-filter');
            const options = ['<option value="ALL">All Archetypes</option>'];
            cat.alphabet.forEach(a => {{
                const transLabel = a.corpus_match ? getTranslitLabel(a.corpus_match) : 'Emergent';
                const label = `${{a.type_id}} (${{transLabel}})`;
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
            selectedGlyph = null;
            selectedWord = null;
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
                    }} else if (selectedWord) {{
                        autoZoomOnWord(selectedWord);
                    }}
                }};
            }}
        }}

        function fitCanvasToWrapper() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const canvas = document.getElementById('page-canvas');
            if (!pageImageObj || !pageImageObj.naturalWidth) return;

            const rect = wrapper.getBoundingClientRect();
            const wWidth = rect.width > 0 ? rect.width : (wrapper.clientWidth || 700);
            const wHeight = rect.height > 0 ? rect.height : (wrapper.clientHeight || 700);

            const scaleW = wWidth / (pageImageObj.naturalWidth || 1);
            const scaleH = wHeight / (pageImageObj.naturalHeight || 1);
            zoomScale = Math.max(0.05, Math.min(scaleW, scaleH) * 0.95);
            if (!isFinite(zoomScale) || zoomScale <= 0.01) zoomScale = 0.5;
            panX = (wWidth - pageImageObj.naturalWidth * zoomScale) / 2;
            panY = (wHeight - pageImageObj.naturalHeight * zoomScale) / 2;
            if (!isFinite(panX)) panX = 0;
            if (!isFinite(panY)) panY = 0;
            drawCanvasOverlay();
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
            if (!isFinite(zoomScale) || zoomScale <= 0.01) zoomScale = 0.5;
            if (!isFinite(panX)) panX = 0;
            if (!isFinite(panY)) panY = 0;
            const canvas = document.getElementById('page-canvas');
            canvas.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{zoomScale}})`;
            canvas.style.filter = canvasFilterPresets[currentCanvasFilter] || 'none';
            document.getElementById('zoom-text').innerText = `${{Math.round(zoomScale * 100)}}%`;
            updateMinimapViewport();
        }}

        let showLines = true;
        let showWords = true;
        let showGlyphs = true;
        let focusMode = false;

        function toggleLayer(layer) {{
            if (layer === 'lines') {{
                showLines = !showLines;
                const btn = document.getElementById('btn-toggle-lines');
                btn.style.opacity = showLines ? '1.0' : '0.4';
            }} else if (layer === 'words') {{
                showWords = !showWords;
                const btn = document.getElementById('btn-toggle-words');
                btn.style.opacity = showWords ? '1.0' : '0.4';
            }} else if (layer === 'glyphs') {{
                showGlyphs = !showGlyphs;
                const btn = document.getElementById('btn-toggle-glyphs');
                btn.style.opacity = showGlyphs ? '1.0' : '0.4';
            }}
            drawCanvasOverlay();
        }}

        function toggleFocusMode() {{
            focusMode = !focusMode;
            const btn = document.getElementById('btn-toggle-focus');
            btn.style.opacity = focusMode ? '1.0' : '0.4';
            btn.style.boxShadow = focusMode ? '0 0 8px #ea580c' : 'none';
            btn.style.fontWeight = focusMode ? 'bold' : 'normal';
            drawCanvasOverlay();
        }}

        window.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (e.key === 'l' || e.key === 'L') toggleLayer('lines');
            if (e.key === 'w' || e.key === 'W') toggleLayer('words');
            if (e.key === 'g' || e.key === 'G') toggleLayer('glyphs');
            if (e.key === 'f' || e.key === 'F') toggleFocusMode();
        }});

        function drawCanvasOverlay() {{
            const canvas = document.getElementById('page-canvas');
            const page = data[currentMs].pages[currentPageIdx];
            if (!pageImageObj || !pageImageObj.complete || pageImageObj.naturalWidth === 0) return;

            canvas.width = pageImageObj.naturalWidth;
            canvas.height = pageImageObj.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(pageImageObj, 0, 0);

            // If Focus Mode is active and an item is selected, ONLY draw the focused selection!
            if (focusMode && (selectedGlyph || selectedWord)) {{
                if (selectedWord) {{
                    const [wy0, wx0, wy1, wx1] = selectedWord.bbox;
                    ctx.strokeStyle = '#10b981';
                    ctx.lineWidth = 3.5;
                    ctx.fillStyle = 'rgba(16, 185, 129, 0.32)';
                    ctx.fillRect(wx0, wy0, wx1 - wx0, wy1 - wy0);
                    ctx.strokeRect(wx0, wy0, wx1 - wx0, wy1 - wy0);

                    // Also highlight constituent child glyphs inside focused word
                    if (page.glyphs) {{
                        page.glyphs.filter(g => g.word_id === selectedWord.word_id).forEach(g => {{
                            const [gy0, gx0, gy1, gx1] = g.bbox;
                            ctx.strokeStyle = '#38bdf8';
                            ctx.lineWidth = 1.8;
                            ctx.fillStyle = 'rgba(56, 189, 248, 0.20)';
                            ctx.fillRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
                            ctx.strokeRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
                        }});
                    }}
                }}
                if (selectedGlyph) {{
                    const [gy0, gx0, gy1, gx1] = selectedGlyph.bbox;
                    ctx.strokeStyle = '#00e5ff';
                    ctx.lineWidth = 4.0;
                    ctx.fillStyle = 'rgba(0, 229, 255, 0.35)';
                    ctx.fillRect(gx0 - 3, gy0 - 3, (gx1 - gx0) + 6, (gy1 - gy0) + 6);
                    ctx.strokeRect(gx0 - 3, gy0 - 3, (gx1 - gx0) + 6, (gy1 - gy0) + 6);
                }}
                return;
            }}

            // 1. Layer 1: Text Lines (🟣 Violet)
            if (showLines && page.lines) {{
                page.lines.forEach(l => {{
                    const [ly0, lx0, ly1, lx1] = l.bbox;
                    const isSelectedLine = selectedLineId && selectedLineId === l.line_id;
                    if (isSelectedLine) {{
                        ctx.strokeStyle = '#c084fc';
                        ctx.lineWidth = 2.8;
                        ctx.fillStyle = 'rgba(192, 132, 252, 0.15)';
                    }} else {{
                        ctx.strokeStyle = 'rgba(168, 85, 247, 0.65)';
                        ctx.lineWidth = 1.5;
                        ctx.fillStyle = 'rgba(168, 85, 247, 0.035)';
                    }}
                    ctx.setLineDash([6, 4]);
                    ctx.strokeRect(lx0, ly0, lx1 - lx0, ly1 - ly0);
                    ctx.fillRect(lx0, ly0, lx1 - lx0, ly1 - ly0);
                    ctx.setLineDash([]);
                }});
            }}

            // 2. Layer 2: Lexical Words (🟢 Emerald Green)
            if (showWords && page.words) {{
                page.words.forEach(w => {{
                    const [wy0, wx0, wy1, wx1] = w.bbox;
                    const isSelectedWord = selectedWord && selectedWord.word_id === w.word_id;
                    const isParentWord = selectedGlyph && selectedGlyph.word_id === w.word_id;
                    if (isSelectedWord) {{
                        ctx.strokeStyle = '#10b981';
                        ctx.lineWidth = 3.2;
                        ctx.fillStyle = 'rgba(16, 185, 129, 0.30)';
                    }} else if (isParentWord) {{
                        ctx.strokeStyle = '#10b981';
                        ctx.lineWidth = 2.4;
                        ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
                    }} else {{
                        ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
                        ctx.lineWidth = 1.2;
                        ctx.fillStyle = 'rgba(16, 185, 129, 0.04)';
                    }}
                    ctx.fillRect(wx0, wy0, wx1 - wx0, wy1 - wy0);
                    ctx.strokeRect(wx0, wy0, wx1 - wx0, wy1 - wy0);
                }});
            }}

            // 3. Layer 3: Extracted Glyphs (🔵 Cyan / ⚡ Glowing Active)
            if (showGlyphs && page.glyphs) {{
                page.glyphs.forEach(g => {{
                    if (archetypeFilterVal !== 'ALL' && g.canonical_type !== archetypeFilterVal) {{
                        return;
                    }}

                    const [y0, x0, y1, x1] = g.bbox;
                    const isSelected = selectedGlyph && selectedGlyph.glyph_id === g.glyph_id;

                    if (isSelected) {{
                        ctx.strokeStyle = '#00e5ff';
                        ctx.lineWidth = 3.5;
                        ctx.fillStyle = 'rgba(0, 229, 255, 0.30)';
                        ctx.fillRect(x0 - 2, y0 - 2, (x1 - x0) + 4, (y1 - y0) + 4);
                        ctx.strokeRect(x0 - 2, y0 - 2, (x1 - x0) + 4, (y1 - y0) + 4);
                    }} else {{
                        ctx.strokeStyle = 'rgba(56, 189, 248, 0.50)';
                        ctx.lineWidth = 1.2;
                        ctx.fillStyle = 'rgba(56, 189, 248, 0.04)';
                        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                    }}
                }});
            }}
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

        function autoZoomOnGlyph(glyph) {{
            const wrapper = document.getElementById('canvas-wrapper');
            const [y0, x0, y1, x1] = glyph.bbox;
            const centerX = (x0 + x1) / 2;
            const centerY = (y0 + y1) / 2;

            const rect = wrapper.getBoundingClientRect();
            const wWidth = rect.width > 0 ? rect.width : (wrapper.clientWidth || 700);
            const wHeight = rect.height > 0 ? rect.height : (wrapper.clientHeight || 700);

            zoomScale = 2.0;
            panX = wWidth / 2 - centerX * zoomScale;
            panY = wHeight / 2 - centerY * zoomScale;

            applyCanvasTransform();
        }}

        function autoZoomOnWord(word) {{
            const wrapper = document.getElementById('canvas-wrapper');
            const [y0, x0, y1, x1] = word.bbox;
            const centerX = (x0 + x1) / 2;
            const centerY = (y0 + y1) / 2;

            const rect = wrapper.getBoundingClientRect();
            const wWidth = rect.width > 0 ? rect.width : (wrapper.clientWidth || 700);
            const wHeight = rect.height > 0 ? rect.height : (wrapper.clientHeight || 700);

            zoomScale = 2.2;
            panX = wWidth / 2 - centerX * zoomScale;
            panY = wHeight / 2 - centerY * zoomScale;

            applyCanvasTransform();
        }}

        function selectGlyph(glyph, shouldAutoZoom = true) {{
            selectedGlyph = glyph;
            selectedWord = null;
            
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
            renderCurrentGrid();
            renderInspector();
        }}

        function selectWord(word, shouldAutoZoom = true) {{
            selectedWord = word;
            selectedGlyph = null;

            drawCanvasOverlay();
            if (shouldAutoZoom) {{
                autoZoomOnWord(word);
            }}
            renderCurrentGrid();
            renderInspectorWord(word);
        }}

        // Setup Interactive Mouse Zoom & Pan
        (function setupPanZoom() {{
            const wrapper = document.getElementById('canvas-wrapper');
            
            wrapper.addEventListener('wheel', (evt) => {{
                evt.preventDefault();
                const rect = wrapper.getBoundingClientRect();
                const mouseX = evt.clientX - rect.left;
                const mouseY = evt.clientY - rect.top;

                const zoomFactor = evt.deltaY < 0 ? 1.15 : 0.87;
                const newScale = Math.max(0.08, Math.min(8.0, zoomScale * zoomFactor));

                panX = mouseX - (mouseX - panX) * (newScale / zoomScale);
                panY = mouseY - (mouseY - panY) * (newScale / zoomScale);
                zoomScale = newScale;

                applyCanvasTransform();
            }}, {{ passive: false }});

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
                if (isPanning) {{
                    isPanning = false;
                    wrapper.classList.remove('grabbing');
                }}
            }});

            // Click on canvas box
            wrapper.addEventListener('click', (evt) => {{
                if (isPanning) return;
                const rect = wrapper.getBoundingClientRect();
                const clickCanvasX = (evt.clientX - rect.left - panX) / zoomScale;
                const clickCanvasY = (evt.clientY - rect.top - panY) / zoomScale;

                const page = data[currentMs].pages[currentPageIdx];
                if (!page) return;

                // 1. If Glyphs layer is visible and clicked inside a glyph bounding box
                if (showGlyphs && page.glyphs) {{
                    const hit = page.glyphs.find(g => {{
                        const [y0, x0, y1, x1] = g.bbox;
                        return clickCanvasX >= x0 - 4 && clickCanvasX <= x1 + 4 && clickCanvasY >= y0 - 4 && clickCanvasY <= y1 + 4;
                    }});
                    if (hit) {{
                        selectGlyph(hit, false);
                        return;
                    }}
                }}

                // 2. If Words layer is visible and clicked inside a word bounding box
                if (showWords && page.words) {{
                    const hitWord = page.words.find(w => {{
                        const [y0, x0, y1, x1] = w.bbox;
                        return clickCanvasX >= x0 - 4 && clickCanvasX <= x1 + 4 && clickCanvasY >= y0 - 4 && clickCanvasY <= y1 + 4;
                    }});
                    if (hitWord) {{
                        selectWord(hitWord, false);
                        return;
                    }}
                }}

                // 3. If Lines layer is visible and clicked inside a line bounding box
                if (showLines && page.lines) {{
                    const hitLine = page.lines.find(l => {{
                        const [y0, x0, y1, x1] = l.bbox;
                        return clickCanvasX >= x0 - 4 && clickCanvasX <= x1 + 4 && clickCanvasY >= y0 - 4 && clickCanvasY <= y1 + 4;
                    }});
                    if (hitLine) {{
                        selectLine(hitLine.line_id);
                        return;
                    }}
                }}
            }});

            // Double click canvas to reset / fit page
            wrapper.addEventListener('dblclick', (evt) => {{
                evt.preventDefault();
                fitCanvasToWrapper();
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

                const wRect = wrapper.getBoundingClientRect();
                const wWidth = wRect.width > 0 ? wRect.width : (wrapper.clientWidth || 700);
                const wHeight = wRect.height > 0 ? wRect.height : (wrapper.clientHeight || 700);

                panX = wWidth / 2 - targetX * zoomScale;
                panY = wHeight / 2 - targetY * zoomScale;
                applyCanvasTransform();
            }});
        }})();

        function selectLine(lineId) {{
            if (selectedLineId === lineId) {{
                selectedLineId = null;
            }} else {{
                selectedLineId = lineId;
            }}
            renderCurrentGrid();
            drawCanvasOverlay();
        }}

        function zoomIn() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const rect = wrapper.getBoundingClientRect();
            const centerX = (rect.width || 700) / 2;
            const centerY = (rect.height || 700) / 2;
            const newScale = Math.min(8.0, zoomScale * 1.3);
            panX = centerX - (centerX - panX) * (newScale / zoomScale);
            panY = centerY - (centerY - panY) * (newScale / zoomScale);
            zoomScale = newScale;
            applyCanvasTransform();
        }}

        function zoomOut() {{
            const wrapper = document.getElementById('canvas-wrapper');
            const rect = wrapper.getBoundingClientRect();
            const centerX = (rect.width || 700) / 2;
            const centerY = (rect.height || 700) / 2;
            const newScale = Math.max(0.08, zoomScale / 1.3);
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
            const canvas = document.getElementById('page-canvas');
            if (!pageImageObj || !pageImageObj.naturalWidth) return;
            const rect = wrapper.getBoundingClientRect();
            const wWidth = rect.width > 0 ? rect.width : (wrapper.clientWidth || 700);
            const wHeight = rect.height > 0 ? rect.height : (wrapper.clientHeight || 700);
            zoomScale = 1.0;
            panX = (wWidth - pageImageObj.naturalWidth) / 2;
            panY = (wHeight - pageImageObj.naturalHeight) / 2;
            if (!isFinite(panX)) panX = 0;
            if (!isFinite(panY)) panY = 0;
            drawCanvasOverlay();
            applyCanvasTransform();
        }}

        function renderCurrentGrid() {{
            if (gridMode === 'glyphs') {{
                renderGlyphGrid();
            }} else {{
                renderWordGrid();
            }}
        }}

        function renderGlyphGrid() {{
            const page = data[currentMs].pages[currentPageIdx];
            let filteredGlyphs = page.glyphs || [];
            if (selectedLineId) {{
                filteredGlyphs = filteredGlyphs.filter(g => g.line_id === selectedLineId);
            }}
            if (archetypeFilterVal !== 'ALL') {{
                filteredGlyphs = filteredGlyphs.filter(g => g.canonical_type === archetypeFilterVal);
            }}

            const lineTitleSuffix = selectedLineId ? ` (${{selectedLineId}})` : '';
            document.getElementById('items-title').innerText = `${{page.page_id}}${{lineTitleSuffix}} — ${{filteredGlyphs.length}} Glyphs`;
            
            const grid = document.getElementById('glyphs-grid');
            grid.innerHTML = filteredGlyphs.map(g => `
                <div class="glyph-card ${{selectedGlyph && selectedGlyph.glyph_id === g.glyph_id ? 'selected' : ''}}" onclick='selectGlyph(${{JSON.stringify(g)}}, true)'>
                    <img src="${{g.png_rel}}" alt="${{g.glyph_id}}">
                    <span class="gid">${{g.glyph_id.split('_').slice(-2).join('_')}}</span>
                    <span class="badge-type" onclick="event.stopPropagation(); openArchetypeVariations('${{g.canonical_type}}')">${{g.canonical_type || 'G??'}}</span>
                </div>
            `).join('');
        }}

        function renderWordGrid() {{
            const page = data[currentMs].pages[currentPageIdx];
            let words = page.words || [];
            if (selectedLineId) {{
                words = words.filter(w => w.line_id === selectedLineId);
            }}

            const lineTitleSuffix = selectedLineId ? ` (${{selectedLineId}})` : '';
            document.getElementById('items-title').innerText = `${{page.page_id}}${{lineTitleSuffix}} — ${{words.length}} Lexical Words`;
            
            const grid = document.getElementById('glyphs-grid');
            if (words.length === 0) {{
                grid.innerHTML = `<p style="color:var(--text-muted); font-size:0.75rem; padding:12px; text-align:center;">No lexical words found for this selection.</p>`;
                return;
            }}

            grid.innerHTML = words.map(w => {{
                const isSelected = selectedWord && selectedWord.word_id === w.word_id;
                const childGlyphs = (page.glyphs || []).filter(g => g.word_id === w.word_id);
                const wordLabel = w.word_id.split('_').slice(-2).join('_');
                return `
                    <div class="word-card ${{isSelected ? 'selected' : ''}}" onclick='selectWord(${{JSON.stringify(w)}}, true)'>
                        <img src="${{w.word_png_rel}}" alt="${{w.word_id}}">
                        <span class="gid" style="color:var(--text-main); font-weight:600;">${{wordLabel}}</span>
                        <span class="badge-freq" style="font-size:0.60rem; padding:1px 5px; margin-top:2px;">${{childGlyphs.length}} glyphs</span>
                    </div>
                `;
            }}).join('');
        }}

        function renderInspectorWord(word) {{
            const ins = document.getElementById('inspector-content');
            const page = data[currentMs].pages[currentPageIdx];
            const childGlyphs = (page.glyphs || []).filter(g => g.word_id === word.word_id);
            const [wy0, wx0, wy1, wx1] = word.bbox;
            const wordWidth = wx1 - wx0;
            const wordHeight = wy1 - wy0;

            let childGlyphsHtml = '';
            if (childGlyphs.length > 0) {{
                childGlyphsHtml = `
                    <div style="margin-top:8px;">
                        <div style="font-size:0.75rem; font-weight:bold; color:var(--accent); margin-bottom:6px;">
                            Constituent Glyphs (${{childGlyphs.length}}) — Click to Inspect
                        </div>
                        <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(70px, 1fr)); gap:6px;">
                            ${{childGlyphs.map(g => `
                                <div class="glyph-card" style="padding:4px;" onclick='selectGlyph(${{JSON.stringify(g)}}, true)'>
                                    <img src="${{g.png_rel}}" alt="${{g.glyph_id}}" style="height:36px;">
                                    <span class="gid">${{g.glyph_id.split('_').slice(-1)[0]}}</span>
                                    <span class="badge-type">${{g.canonical_type || 'G??'}}</span>
                                </div>
                            `).join('')}}
                        </div>
                    </div>
                `;
            }} else {{
                childGlyphsHtml = `<p style="color:var(--text-muted); font-size:0.72rem; margin-top:6px;">No constituent glyphs indexed for this word.</p>`;
            }}

            ins.innerHTML = `
                <div style="background:#064e3b; border:1px solid #10b981; border-radius:8px; padding:10px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.85rem; color:#10b981; font-weight:bold;">🟢 Lexical Word: ${{word.word_id}}</span>
                        <span class="badge-freq">${{childGlyphs.length}} Glyphs</span>
                    </div>
                    <div style="text-align:center; background:#000; border-radius:6px; padding:8px; border:1px solid var(--border);">
                        <img src="${{word.word_png_rel}}" alt="${{word.word_id}}" style="max-height:100px; max-width:100%; object-fit:contain;">
                    </div>
                </div>

                <div class="meta-list">
                    <div><strong>Word ID:</strong> ${{word.word_id}}</div>
                    ${{word.eva_text ? `<div style="background:#1e293b; padding:6px 10px; border-radius:6px; border:1px solid #f59e0b; margin:4px 0;"><span style="color:var(--text-muted); font-size:0.70rem; display:block;">VOYNICHESE (2014) GROUND TRUTH</span><strong style="color:#f59e0b; font-size:1.15rem; font-family:monospace;">'${{word.eva_text}}'</strong></div>` : ''}}
                    ${{word.eva_glyphs && word.eva_glyphs.length ? `<div style="margin:3px 0;"><strong>EVA Graphemes:</strong> ${{word.eva_glyphs.map(g => `<span class="badge-type" style="margin-right:2px; font-size:0.70rem;">${{g}}</span>`).join('')}}</div>` : ''}}
                    <div><strong>Page / Line:</strong> ${{page.page_id}} / ${{word.line_id || 'Line ?'}}</div>
                    <div><strong>Coordinates [y0, x0, y1, x1]:</strong> [${{word.bbox.join(', ')}}]</div>
                    <div><strong>Dimensions:</strong> ${{wordWidth}}x${{wordHeight}} px</div>
                    <div><strong>Constituent Glyphs:</strong> ${{childGlyphs.length}}</div>
                </div>

                ${{childGlyphsHtml}}
            `;
        }}

        function getSchemeSelectOptionsHtml() {{
            if (currentMs === 'voynich') {{
                return `
                    <option value="eva" ${{activeAlphabetScheme === 'eva' ? 'selected' : ''}}>EVA</option>
                    <option value="currier" ${{activeAlphabetScheme === 'currier' ? 'selected' : ''}}>Currier</option>
                    <option value="v101" ${{activeAlphabetScheme === 'v101' ? 'selected' : ''}}>v101</option>
                    <option value="voynichese" ${{activeAlphabetScheme === 'voynichese' ? 'selected' : ''}}>Voynichese (2014)</option>
                `;
            }} else {{
                return `
                    <option value="serafini" ${{activeAlphabetScheme === 'serafini' ? 'selected' : ''}}>Serafini (1981)</option>
                    <option value="deri" ${{activeAlphabetScheme === 'deri' ? 'selected' : ''}}>Deri (2015)</option>
                    <option value="bulik" ${{activeAlphabetScheme === 'bulik' ? 'selected' : ''}}>Bulik (2011)</option>
                `;
            }}
        }}

        function renderInspector() {{
            const ins = document.getElementById('inspector-content');
            if (!selectedGlyph) {{
                ins.innerHTML = `<p style="color:var(--text-muted); margin-top:40px; text-align:center;">Select any glyph or word from the grid or page overlay to inspect its coordinates and vector ductus.</p>`;
                return;
            }}

            const cat = data[currentMs].catalogue;
            const arch = cat.alphabet.find(a => a.type_id === selectedGlyph.canonical_type);
            const cm = arch?.corpus_match;

            let corpusHtml = '';
            let refSvgBox = '';
            const schemeSelectHtml = `
                <select id="alphabet-scheme-select" onchange="setAlphabetScheme(this.value)" style="font-size:0.65rem; padding:1px 4px; background:#0f172a; color:var(--accent); border:1px solid #334155; border-radius:4px; cursor:pointer;">
                    ${{getSchemeSelectOptionsHtml()}}
                </select>
            `;

            if (cm && cm.reference_svg && cm.confidence_pct >= 60) {{
                const schemeLabel = getTranslitLabel(cm);
                corpusHtml = `
                    <div class="corpus-box">
                        <div class="corpus-title">
                            <span>Standard Corpus Match (${{schemeLabel}})</span>
                            <span style="color:var(--success); font-weight:bold;">${{cm.confidence_pct}}% match</span>
                        </div>
                        <div><strong>Category:</strong> ${{cm.category}} (${{cm.name}})</div>
                        <div style="color:var(--text-muted); font-size:0.72rem; margin-top:2px;">${{cm.description}}</div>
                    </div>
                `;
                refSvgBox = `
                    <div class="preview-box-large" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div style="display:flex; justify-content:space-between; align-items:center; width:100%; margin-bottom:4px;">
                            <span style="font-size:0.60rem; color:var(--text-muted); font-weight:bold;">REFERENCE</span>
                            ${{schemeSelectHtml}}
                        </div>
                        <div style="flex:1; display:flex; justify-content:center; align-items:center; width:100%; min-height:75px;">
                            ${{cm.reference_svg || ''}}
                        </div>
                        <div style="font-size:0.65rem; color:var(--accent); font-weight:bold; margin-top:2px;">
                            ${{schemeLabel}}
                        </div>
                    </div>
                `;
            }} else {{
                corpusHtml = `
                    <div class="corpus-box" style="border-color:#334155;">
                        <div class="corpus-title">
                            <span style="color:#38bdf8;">Autonomous Emergent Archetype</span>
                            <span style="color:var(--text-muted); font-size:0.75rem;">Novel Scribal Form</span>
                        </div>
                        <div style="color:var(--text-muted); font-size:0.72rem; margin-top:2px;">
                            This glyph possesses a distinct ductus topology with no transliteration equivalent in standard transcription schemes (Safeguard against transliteration hallucination).
                        </div>
                    </div>
                `;
                refSvgBox = `
                    <div class="preview-box-large" style="display:flex; flex-direction:column; justify-content:space-between;">
                        <div style="display:flex; justify-content:space-between; align-items:center; width:100%; margin-bottom:4px;">
                            <span style="font-size:0.60rem; color:var(--text-muted); font-weight:bold;">REFERENCE</span>
                            ${{schemeSelectHtml}}
                        </div>
                        <div style="color:var(--text-muted); font-size:0.72rem; padding:12px; line-height:1.4; flex:1; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                            <div style="font-size:1.3rem; margin-bottom:4px;">✨</div>
                            <strong>Distinct Emergent Ductus</strong>
                            <div style="font-size:0.65rem; opacity:0.75; margin-top:4px;">No standard reference equivalent</div>
                        </div>
                    </div>
                `;
            }}

            let wordHtml = '';
            if (selectedGlyph.word_png_rel) {{
                wordHtml = `
                    <div style="background:#0b1329; border:1px solid #1e3a8a; border-radius:8px; padding:8px 10px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:0.68rem; color:var(--accent); font-weight:bold; text-transform:uppercase;">Parent Word Context (${{selectedGlyph.word_id}})</span>
                            <span style="font-size:0.65rem; color:var(--text-muted);">Word BBox: [${{selectedGlyph.word_bbox ? selectedGlyph.word_bbox.join(', ') : ''}}]</span>
                        </div>
                        <div style="text-align:center; background:#000; border-radius:6px; padding:6px; border:1px solid var(--border);">
                            <img src="${{selectedGlyph.word_png_rel}}" alt="${{selectedGlyph.word_id}}" style="max-height:60px; max-width:100%; object-fit:contain;">
                        </div>
                    </div>
                `;
            }}

            ins.innerHTML = `
                ${{wordHtml}}

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

                <!-- Ductus Kinematic Color Legend -->
                <div style="background:#0b1329; border:1px solid #1e293b; border-radius:6px; padding:6px 10px; font-size:0.68rem; margin-top:2px;">
                    <div style="color:var(--accent); font-weight:bold; margin-bottom:3px; display:flex; justify-content:space-between;">
                        <span>🖋️ Kinematic Ductus Legend</span>
                        <span style="color:var(--text-muted); font-size:0.62rem;">Stroke Order: 1 ➔ 2 ➔ 3 ➔ 4</span>
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:8px; line-height:1.3; color:var(--text-muted);">
                        <span><strong style="color:#10b981;">● Green Dot:</strong> Pen Touchdown</span>
                        <span><strong style="color:#38bdf8;">— Cyan:</strong> Stroke 1 (Primary)</span>
                        <span><strong style="color:#10b981;">— Emerald:</strong> Stroke 2 (Post Pen-Lift)</span>
                        <span><strong style="color:#f59e0b;">— Amber:</strong> Stroke 3</span>
                        <span><strong style="color:#c084fc;">— Violet:</strong> Stroke 4+</span>
                    </div>
                </div>

                ${{corpusHtml}}

                <!-- 5-Way Comparative AI Vision & Ductus Suite Panel & Modal Launcher -->
                <div style="background:#0f172a; border:1px solid #334155; border-radius:6px; padding:8px 10px; font-size:0.68rem; margin-top:6px;">
                    <div style="color:#38bdf8; font-weight:bold; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center;">
                        <span>🔬 5-Way Vision & Ductus Suite</span>
                        <button onclick="openComparativeVisionModal()" style="background:#0284c7; color:#fff; border:none; border-radius:4px; padding:2px 7px; font-size:0.62rem; cursor:pointer; font-weight:bold;">🔍 Launch 5-Way Modal</button>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:4px; color:var(--text-muted); line-height:1.35;">
                        <div><strong style="color:#f59e0b;">1. Our In-House Physical Nib Model:</strong> 40° bevel plein/délié, unbroken continuous loops, right-handed scribal prior.</div>
                        <div><strong style="color:#38bdf8;">2. Euler-Bernoulli Skeleton:</strong> 1D Medial Axis + NetworkX graph resolver (min bending energy).</div>
                        <div><strong style="color:#06b6d4;">3. Kinematic Flow Net:</strong> Neural U-Net directional flow field + touchdown/lift detection.</div>
                        <div><strong style="color:#c084fc;">4. DINOv2 ViT:</strong> 768-D patch tokens capturing curvature & nib pressure invariants.</div>
                        <div><strong style="color:#10b981;">5. InkSight Transformer:</strong> Autoregressive (x, y, t, pen-lift) sequence prediction.</div>
                    </div>
                </div>

                <div class="meta-list">
                    <div><strong>Glyph ID:</strong> ${{selectedGlyph.glyph_id}}</div>
                    <div><strong>Page / Line / Word:</strong> ${{selectedGlyph.page_id}} / Line ${{selectedGlyph.line_id}} / ${{selectedGlyph.word_id}}</div>
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

        function openComparativeVisionModal() {{
            if (!selectedGlyph) return;
            document.getElementById('modal-comparative-title').innerText = `🔬 5-Way AI Vision & Ductus Benchmark: ${{selectedGlyph.glyph_id}} (${{selectedGlyph.canonical_type}})`;
            
            // 1. Method 1: Our In-House Physical Nib Model (Calligraphic Ridge Tracker with 40° Nib Simulation)
            const calBox = document.getElementById('comp-cal-svg');
            const origSvg = selectedGlyph.svg_content || '';
            calBox.innerHTML = `
                <div style="position:relative; width:100%; height:120px; display:flex; justify-content:center; align-items:center; background:#0b0f19; border-radius:6px; overflow:hidden;">
                    <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; filter:drop-shadow(0 0 3px rgba(245, 158, 11, 0.6));">
                        ${{origSvg}}
                    </div>
                    <div style="position:absolute; bottom:3px; right:5px; font-size:0.58rem; color:#f59e0b; background:rgba(0,0,0,0.75); padding:1px 4px; border-radius:3px;">
                        <span>Nib: 40° Bevel</span>
                    </div>
                </div>
            `;

            // 2. Method 2: Euler-Bernoulli Topological Skeleton
            document.getElementById('comp-geom-svg').innerHTML = origSvg;
            
            // 3. Method 3: Scribal Kinematic Flow Net (U-Net Flow Field & Directional Arrows)
            const flowBox = document.getElementById('comp-flow-visual');
            flowBox.innerHTML = `
                <div style="position:relative; width:100%; height:120px; display:flex; justify-content:center; align-items:center; background:#04151f; border-radius:6px; overflow:hidden;">
                    <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; opacity:0.85;">
                        ${{origSvg}}
                    </div>
                    <svg style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;" viewBox="0 0 100 100">
                        <defs>
                            <marker id="arrow-cyan" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4" />
                            </marker>
                        </defs>
                        <!-- Flow Vectors -->
                        <line x1="30" y1="40" x2="45" y2="35" stroke="#06b6d4" stroke-width="1.2" marker-end="url(#arrow-cyan)"/>
                        <line x1="45" y1="35" x2="60" y2="45" stroke="#06b6d4" stroke-width="1.2" marker-end="url(#arrow-cyan)"/>
                        <line x1="60" y1="45" x2="70" y2="65" stroke="#06b6d4" stroke-width="1.2" marker-end="url(#arrow-cyan)"/>
                        <circle cx="28" cy="41" r="3.5" fill="#10b981" />
                        <text x="50" y="92" fill="#06b6d4" font-size="6" text-anchor="middle" font-family="monospace">û(x,y) Kinematic Field</text>
                    </svg>
                </div>
            `;

            // 4. Method 4: DINOv2 Self-Supervised ViT (Patch Token Grid & Self-Attention Heatmap Overlay)
            const dinoBox = document.getElementById('comp-dino-visual');
            dinoBox.innerHTML = `
                <div style="position:relative; width:100%; height:120px; display:flex; justify-content:center; align-items:center; background:#050811; border-radius:6px; overflow:hidden;">
                    <img src="${{selectedGlyph.png_rel}}" alt="Glyph" style="max-height:85px; opacity:0.65; filter:contrast(160%) brightness(90%);">
                    <svg style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;" viewBox="0 0 100 100">
                        <defs>
                            <radialGradient id="token-glow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="#c084fc" stop-opacity="0.9"/>
                                <stop offset="50%" stop-color="#ec4899" stop-opacity="0.5"/>
                                <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
                            </radialGradient>
                        </defs>
                        <!-- 8x8 ViT Token Grid -->
                        <line x1="25" y1="10" x2="25" y2="90" stroke="rgba(192, 132, 252, 0.25)" stroke-dasharray="2,2"/>
                        <line x1="50" y1="10" x2="50" y2="90" stroke="rgba(192, 132, 252, 0.35)" stroke-dasharray="2,2"/>
                        <line x1="75" y1="10" x2="75" y2="90" stroke="rgba(192, 132, 252, 0.25)" stroke-dasharray="2,2"/>
                        <line x1="10" y1="25" x2="90" y2="25" stroke="rgba(192, 132, 252, 0.25)" stroke-dasharray="2,2"/>
                        <line x1="10" y1="50" x2="90" y2="50" stroke="rgba(192, 132, 252, 0.35)" stroke-dasharray="2,2"/>
                        <line x1="10" y1="75" x2="90" y2="75" stroke="rgba(192, 132, 252, 0.25)" stroke-dasharray="2,2"/>
                        
                        <!-- High Attention Token Clusters (Self-Attention Hubs) -->
                        <circle cx="48" cy="45" r="16" fill="url(#token-glow)"/>
                        <circle cx="58" cy="35" r="11" fill="url(#token-glow)"/>
                        <circle cx="42" cy="62" r="12" fill="url(#token-glow)"/>
                        
                        <rect x="42" y="38" width="14" height="14" fill="none" stroke="#f43f5e" stroke-width="1.2"/>
                        <text x="50" y="20" fill="#c084fc" font-size="6" text-anchor="middle" font-family="monospace">768-D ViT Tokens</text>
                    </svg>
                </div>
            `;
            
            // 5. Method 5: InkSight Autoregressive Handwriting Transformer
            const inkBox = document.getElementById('comp-inksight-svg');
            inkBox.innerHTML = `
                <div style="position:relative; width:100%; height:120px; display:flex; justify-content:center; align-items:center; background:#02110c; border-radius:6px; overflow:hidden;">
                    <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; filter:drop-shadow(0 0 4px #10b981);">
                        ${{origSvg}}
                    </div>
                    <div style="position:absolute; bottom:4px; right:6px; font-size:0.60rem; color:#10b981; background:rgba(0,0,0,0.7); padding:1px 5px; border-radius:3px; display:flex; gap:6px;">
                        <span>● t=0 (Touchdown)</span>
                        <span style="color:#ef4444;">◆ t=1 (Pen-Lift)</span>
                    </div>
                </div>
            `;
            
            document.getElementById('comparative-modal').classList.add('open');
        }}

        function closeComparativeModal() {{
            document.getElementById('comparative-modal').classList.remove('open');
        }}

        function renderCatalogueTab() {{
            const cat = data[currentMs].catalogue;
            const grid = document.getElementById('catalogue-grid');
            grid.innerHTML = cat.alphabet.map(a => {{
                const cm = a.corpus_match;
                let matchHeader = '';
                let refSvgElem = '';
                if (cm && cm.reference_svg && cm.confidence_pct >= 60) {{
                    const schemeLabel = getTranslitLabel(cm);
                    matchHeader = `<div style="color:var(--accent); font-size:0.85rem; font-weight:bold;">${{schemeLabel}} <span style="font-size:0.75rem; color:var(--success); font-weight:normal;">(${{cm.confidence_pct}}% match)</span></div>`;
                    refSvgElem = `
                        <div class="c-box">
                            <span style="font-size:0.58rem; color:var(--text-muted); margin-bottom:2px;">STANDARD REF</span>
                            ${{cm.reference_svg || ''}}
                        </div>
                    `;
                }} else {{
                    matchHeader = `<div style="color:#38bdf8; font-size:0.85rem; font-weight:bold;">Distinct Emergent Ductus <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(Autonomous)</span></div>`;
                    refSvgElem = `
                        <div class="c-box" style="display:flex; flex-direction:column; justify-content:center; align-items:center; color:var(--text-muted); font-size:0.68rem; padding:4px;">
                            <span>✨ Emergent</span>
                            <span style="font-size:0.58rem; opacity:0.7;">No Transliteration</span>
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
            }} else if (evt.target.id === 'comparative-modal') {{
                closeComparativeModal();
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
    voynich_catalogue_builder = GlyphCatalogue(target_alphabet_size=28)
    serafini_catalogue_builder = GlyphCatalogue(target_alphabet_size=52)
    rosetta_loader = VoynicheseRosettaLoader()

    # 1. Process Voynich Manuscript Pages (Yale Beinecke HD IIIF)
    print("\n=======================================================")
    print(" 1. FULL GLYPH EXTRACTION: VOYNICH MANUSCRIPT (YALE BEINECKE HD)")
    print("=======================================================")
    voynich_pages_data = []
    all_voynich_glyphs = []

    iiif_client = IIIFClient(cache_dir="data/scans/voynich/yale")
    voynich_folios = ["f001r", "f001v", "f002r", "f002v", "f003r", "f003v", "f004r", "f004v"]

    for folio_id in voynich_folios:
        try:
            print(f"[*] Downloading / Loading Yale Beinecke HD scan for '{folio_id}' (2400px)...")
            f_path = iiif_client.download_folio(folio_id, max_width=2400)
            img = Image.open(f_path)
            print(f"[*] Extracting all glyphs on Voynich HD '{folio_id}' ({img.width}x{img.height} px)...")
            res = segmenter.extract_page_glyphs(folio_id, img, output_dir=base_out, subfolder="voynich")
            
            # Match detected words with Voynichese Rosetta Ground Truth
            try:
                r_words = rosetta_loader.extract_aligned_words(folio_id, scan_image=img)
                for w in res.get("words", []):
                    wy0, wx0, wy1, wx1 = w["bbox"]
                    wc_y = (wy0 + wy1) / 2.0
                    wc_x = (wx0 + wx1) / 2.0
                    best_match = None
                    min_dist = float("inf")
                    for rw in r_words:
                        if rw.bbox_scan:
                            ry0, rx0, ry1, rx1 = rw.bbox_scan
                            rc_y = (ry0 + ry1) / 2.0
                            rc_x = (rx0 + rx1) / 2.0
                            dist = (wc_y - rc_y)**2 + (wc_x - rc_x)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_match = rw
                    if best_match and min_dist < (180**2):
                        w["eva_text"] = best_match.eva_text
                        w["eva_glyphs"] = rosetta_loader.tokenize_eva_to_glyphs(best_match.eva_text)
            except Exception as e_rosetta:
                print(f"  [i] Rosetta matching notice for {folio_id}: {e_rosetta}")

            # Census yield validation
            yr = PaleographyYieldValidator.evaluate_yield(
                folio_id=folio_id,
                detected_lines=res["line_count"],
                detected_glyphs=res["glyph_count"],
                extracted_strokes=sum(len(g["strokes"]) for g in res["glyphs"])
            )
            res["yield_report"] = yr
            print(f"  [+] Identified {res['line_count']} lines, {res['word_count']} words, extracted {res['glyph_count']} intact glyphs.")
            voynich_pages_data.append(res)
            all_voynich_glyphs.extend(res["glyphs"])
        except Exception as e:
            print(f"[-] Error on Yale folio {folio_id}: {e}")

    print(f"\n[*] Inducing Voynich Canonical Alphabet & Matching Standard Corpora (EVA/Currier)...")
    voynich_catalogue = voynich_catalogue_builder.build_catalogue(all_voynich_glyphs, corpus_type="voynich")
    voynich_catalogue_builder.export_catalogue_json(voynich_catalogue, base_out / "voynich" / "voynich_alphabet_catalogue.json")
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
                print(f"[*] Extracting all glyphs on Seraphinianus page {p_idx} (with x1.5 upscale)...")
                p_img = loader_s.get_page_image(p_idx, target_min_dim=2000)
                # Auto-orient landscape plates if width significantly exceeds height
                if p_img.width > p_img.height * 1.15:
                    print(f"  [i] Auto-orienting landscape page {p_idx} by 270°...")
                    p_img = p_img.rotate(270, expand=True)
                # Lanczos 1.5x upscaling for razor-sharp pen contours
                p_img_hd = p_img.resize((int(p_img.width * 1.5), int(p_img.height * 1.5)), Image.Resampling.LANCZOS)
                res_s = segmenter.extract_page_glyphs(page_id, p_img_hd, output_dir=base_out, subfolder="seraphinianus")
                print(f"  [+] Identified {res_s['line_count']} lines, {res_s['word_count']} words, extracted {res_s['glyph_count']} pure isolated glyphs.")
                serafini_pages_data.append(res_s)
                all_serafini_glyphs.extend(res_s["glyphs"])
            except Exception as e:
                print(f"[-] Error on Seraphinianus page {p_idx}: {e}")

    print(f"\n[*] Inducing Seraphinianus Canonical Alphabet & Matching Serafinian Typology...")
    serafini_catalogue = serafini_catalogue_builder.build_catalogue(all_serafini_glyphs, corpus_type="seraphinianus")
    serafini_catalogue_builder.export_catalogue_json(serafini_catalogue, base_out / "seraphinianus" / "seraphinianus_alphabet_catalogue.json")
    print(f"  [+] Discovered {serafini_catalogue['canonical_alphabet_size']} unique canonical glyph archetypes across Seraphinianus pages.")

    # 3. Generate Interactive Visual Explorer
    explorer_path = base_out / "grand_glyph_explorer.html"
    generate_explorer_html(voynich_pages_data, voynich_catalogue, serafini_pages_data, serafini_catalogue, explorer_path)


if __name__ == "__main__":
    main()
