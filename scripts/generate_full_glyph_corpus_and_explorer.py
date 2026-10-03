"""
Full-scale Glyph Extraction, Kinematic Ductus Vectorization, and Interactive Grand Explorer.
Extracts individual isolated glyphs across Voynich Manuscript and Codex Seraphinianus,
induces canonical alphabets, and renders the comprehensive interactive visual explorer with full page overlays.
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
                "stroke_count": int(g["stroke_count"]),
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
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: var(--panel-bg);
            border-bottom: 1px solid var(--border);
            padding: 14px 24px;
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
            gap: 12px;
        }}
        .brand h1 {{ font-size: 1.35rem; font-weight: 700; }}
        .brand span {{ color: var(--accent); font-size: 0.85rem; }}
        
        .controls {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .btn-toggle {{
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-toggle.active-voynich {{ background: var(--voynich-color); color: #000; font-weight: 600; border-color: var(--voynich-color); }}
        .btn-toggle.active-serafini {{ background: var(--serafini-color); color: #000; font-weight: 600; border-color: var(--serafini-color); }}
        
        .nav-tabs {{
            display: flex;
            gap: 6px;
            background: #0b1120;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .tab-btn {{
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: var(--card-bg);
            color: var(--text-main);
            font-weight: 600;
        }}

        main {{
            flex: 1;
            padding: 20px;
            max-width: 1700px;
            margin: 0 auto;
            width: 100%;
        }}

        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .stat-card h4 {{ font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
        .stat-card .val {{ font-size: 1.45rem; font-weight: 700; color: var(--accent); }}

        /* 4-Panel Page Layout */
        .page-view-layout {{
            display: grid;
            grid-template-columns: 260px 420px 1fr 320px;
            gap: 16px;
            height: calc(100vh - 180px);
        }}
        .panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow-y: auto;
            padding: 14px;
            display: flex;
            flex-direction: column;
        }}
        .panel h3 {{
            font-size: 0.95rem;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
            color: var(--accent);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Page List */
        .page-item {{
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            background: var(--card-bg);
            margin-bottom: 8px;
            border: 1px solid transparent;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .page-item:hover {{ border-color: var(--accent); }}
        .page-item.active {{ background: #1e3a8a; border-color: var(--accent); font-weight: 600; }}

        /* Full Page Canvas Overlay */
        .canvas-container {{
            position: relative;
            width: 100%;
            height: 100%;
            background: #000;
            border-radius: 6px;
            overflow: auto;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }}
        #page-canvas {{
            max-width: 100%;
            height: auto;
            display: block;
        }}

        /* Glyph Grid */
        .glyphs-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(75px, 1fr));
            gap: 8px;
            overflow-y: auto;
            padding: 4px;
        }}
        .glyph-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s;
            aspect-ratio: 1;
        }}
        .glyph-card:hover {{ border-color: var(--accent); transform: scale(1.05); background: var(--card-hover); }}
        .glyph-card.selected {{ border-color: var(--accent); box-shadow: 0 0 12px rgba(56, 189, 248, 0.5); background: #0c4a6e; }}
        .glyph-card img {{ max-width: 85%; max-height: 40px; object-fit: contain; margin-bottom: 2px; }}
        .glyph-card .gid {{ font-size: 0.6rem; color: var(--text-muted); }}

        /* Inspector */
        .inspector-content {{
            display: flex;
            flex-direction: column;
            gap: 14px;
            align-items: center;
            text-align: center;
        }}
        .preview-box {{
            width: 100%;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 120px;
        }}
        .preview-box img {{ max-width: 100%; max-height: 100px; object-fit: contain; }}
        .preview-box svg {{ max-width: 100%; max-height: 100px; }}
        .meta-list {{
            width: 100%;
            text-align: left;
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.6;
        }}
        .meta-list strong {{ color: var(--text-main); }}

        /* Yield Summary Box */
        .yield-box {{
            background: #0c1524;
            border: 1px solid #1e3a8a;
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
            font-size: 0.78rem;
            line-height: 1.5;
        }}

        /* Catalogue Tab View */
        .catalogue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 14px;
        }}
        .catalogue-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .catalogue-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .catalogue-header h3 {{ font-size: 1.05rem; color: var(--accent); }}
        .catalogue-duo {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            background: #000;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        .catalogue-duo .c-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 65px;
        }}
        .catalogue-duo img {{ max-width: 85%; max-height: 55px; object-fit: contain; }}
        .catalogue-duo svg {{ max-width: 90%; max-height: 55px; }}
        .badge-freq {{ background: #1e293b; color: #38bdf8; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; }}
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>VoynichDuctus</h1>
            <span>Autonomous Paleography & Ductus Corpus Explorer</span>
        </div>
        <div class="controls">
            <button class="btn-toggle active-voynich" id="btn-voynich" onclick="setManuscript('voynich')">Voynich Manuscript</button>
            <button class="btn-toggle" id="btn-serafini" onclick="setManuscript('seraphinianus')">Codex Seraphinianus</button>
            <div class="nav-tabs">
                <button class="tab-btn active" id="tab-btn-pages" onclick="setTab('pages')">Page Overlay & Glyphs</button>
                <button class="tab-btn" id="tab-btn-catalogue" onclick="setTab('catalogue')">Canonical Alphabet</button>
            </div>
        </div>
    </header>

    <main>
        <div class="stats-bar" id="stats-bar"></div>

        <!-- TAB 1: Pages & Glyphs View -->
        <div id="view-pages" class="page-view-layout">
            <div class="panel">
                <h3>Manuscript Pages</h3>
                <div id="pages-list"></div>
                <div id="yield-panel"></div>
            </div>

            <div class="panel">
                <h3>Full Page & Bounding Boxes</h3>
                <div class="canvas-container">
                    <canvas id="page-canvas"></canvas>
                </div>
            </div>

            <div class="panel" style="flex:1;">
                <h3 id="glyphs-title">Isolated Glyphs</h3>
                <div class="glyphs-container" id="glyphs-grid"></div>
            </div>

            <div class="panel">
                <h3>Ductus Inspector</h3>
                <div class="inspector-content" id="inspector-content">
                    <p style="color:var(--text-muted); margin-top:40px;">Select any glyph from the grid or page overlay to inspect its scan crop and vector ductus.</p>
                </div>
            </div>
        </div>

        <!-- TAB 2: Discovered Canonical Alphabet Catalogue -->
        <div id="view-catalogue" style="display:none;">
            <div class="catalogue-grid" id="catalogue-grid"></div>
        </div>
    </main>

    <script>
        const data = {app_json};
        let currentMs = 'voynich';
        let currentTab = 'pages';
        let currentPageIdx = 0;
        let selectedGlyph = null;
        let pageImageObj = new Image();

        function setManuscript(ms) {{
            currentMs = ms;
            currentPageIdx = 0;
            selectedGlyph = null;
            document.getElementById('btn-voynich').className = 'btn-toggle' + (ms === 'voynich' ? ' active-voynich' : '');
            document.getElementById('btn-serafini').className = 'btn-toggle' + (ms === 'seraphinianus' ? ' active-serafini' : '');
            render();
        }}

        function setTab(tab) {{
            currentTab = tab;
            document.getElementById('tab-btn-pages').className = 'tab-btn' + (tab === 'pages' ? ' active' : '');
            document.getElementById('tab-btn-catalogue').className = 'tab-btn' + (tab === 'catalogue' ? ' active' : '');
            document.getElementById('view-pages').style.display = tab === 'pages' ? 'grid' : 'none';
            document.getElementById('view-catalogue').style.display = tab === 'catalogue' ? 'block' : 'none';
            render();
        }}

        function selectPage(idx) {{
            currentPageIdx = idx;
            selectedGlyph = null;
            renderPagesTab();
        }}

        function selectGlyph(g) {{
            selectedGlyph = g;
            renderGlyphGrid();
            renderInspector();
            drawCanvasOverlay();
        }}

        function render() {{
            const msData = data[currentMs];
            const totalGlyphs = msData.pages.reduce((sum, p) => sum + p.glyph_count, 0);
            const totalLines = msData.pages.reduce((sum, p) => sum + p.line_count, 0);

            // Render stats
            document.getElementById('stats-bar').innerHTML = `
                <div class="stat-card">
                    <h4>Manuscript</h4>
                    <div class="val" style="font-size:1.05rem; color:${{currentMs === 'voynich' ? 'var(--voynich-color)' : 'var(--serafini-color)'}};">${{msData.title}}</div>
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
                        <div style="font-size:0.75rem; color:var(--text-muted);">${{p.line_count}} lines</div>
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
                        <div style="font-weight:600; color:var(--accent); margin-bottom:4px;">Census Benchmark (${{yr.folio_id}})</div>
                        <div>Lines: <strong>${{yr.detected_lines}}/${{yr.expected_lines}}</strong> (${{yr.line_yield_pct}}%)</div>
                        <div>Glyphs: <strong>${{yr.detected_glyphs}}/${{yr.expected_glyphs}}</strong> (${{yr.glyph_yield_pct}}%)</div>
                        <div>Hand: <strong>${{yr.scribe_hand}}</strong> (${{yr.currier_language}})</div>
                        <div style="color:var(--text-muted); font-size:0.7rem; margin-top:4px;">Section: ${{yr.section}}</div>
                    </div>
                `;
            }} else {{
                yieldPanel.innerHTML = '';
            }}

            loadPageImageAndDraw();
            renderGlyphGrid();
            renderInspector();
        }}

        function loadPageImageAndDraw() {{
            const page = data[currentMs].pages[currentPageIdx];
            if (page.page_img_rel) {{
                pageImageObj = new Image();
                pageImageObj.src = page.page_img_rel;
                pageImageObj.onload = () => {{
                    drawCanvasOverlay();
                }};
            }}
        }}

        function drawCanvasOverlay() {{
            const canvas = document.getElementById('page-canvas');
            const page = data[currentMs].pages[currentPageIdx];
            if (!pageImageObj.complete || pageImageObj.naturalWidth === 0) return;

            canvas.width = pageImageObj.naturalWidth;
            canvas.height = pageImageObj.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(pageImageObj, 0, 0);

            // Draw bounding boxes for all glyphs
            page.glyphs.forEach(g => {{
                const [y0, x0, y1, x1] = g.bbox;
                const isSelected = selectedGlyph && selectedGlyph.glyph_id === g.glyph_id;

                if (isSelected) {{
                    ctx.strokeStyle = '#38bdf8';
                    ctx.lineWidth = 5;
                    ctx.fillStyle = 'rgba(56, 189, 248, 0.45)';
                    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

                    // Add glowing target marker
                    ctx.beginPath();
                    ctx.arc((x0 + x1)/2, (y0 + y1)/2, Math.max(14, (x1 - x0)/1.5), 0, 2 * Math.PI);
                    ctx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }} else {{
                    ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)';
                    ctx.lineWidth = 2;
                    ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
                    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                }}
            }});

            // Setup click handler on canvas
            canvas.onclick = function(evt) {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const clickX = (evt.clientX - rect.left) * scaleX;
                const clickY = (evt.clientY - rect.top) * scaleY;

                // Find clicked glyph
                const hit = page.glyphs.find(g => {{
                    const [y0, x0, y1, x1] = g.bbox;
                    return clickX >= x0 - 4 && clickX <= x1 + 4 && clickY >= y0 - 4 && clickY <= y1 + 4;
                }});

                if (hit) {{
                    selectGlyph(hit);
                }}
            }};
        }}


        function renderGlyphGrid() {{
            const page = data[currentMs].pages[currentPageIdx];
            document.getElementById('glyphs-title').innerText = `${{page.page_id}} — ${{page.glyph_count}} Pure Glyphs`;
            
            const grid = document.getElementById('glyphs-grid');
            grid.innerHTML = page.glyphs.map(g => `
                <div class="glyph-card ${{selectedGlyph && selectedGlyph.glyph_id === g.glyph_id ? 'selected' : ''}}" onclick='selectGlyph(${{JSON.stringify(g)}})'>
                    <img src="${{g.png_rel}}" alt="${{g.glyph_id}}">
                    <span class="gid">${{g.glyph_id.split('_').slice(-2).join('_')}}</span>
                </div>
            `).join('');
        }}

        function renderInspector() {{
            const ins = document.getElementById('inspector-content');
            if (!selectedGlyph) {{
                ins.innerHTML = `<p style="color:var(--text-muted); margin-top:40px;">Select any glyph from the grid or page overlay to inspect its scan crop and vector ductus.</p>`;
                return;
            }}
            ins.innerHTML = `
                <div class="preview-box">
                    <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">ORIGINAL SCAN CROP</span>
                    <img src="${{selectedGlyph.png_rel}}" alt="${{selectedGlyph.glyph_id}}">
                </div>
                <div class="preview-box">
                    <span style="font-size:0.65rem; color:var(--text-muted); margin-bottom:4px;">VECTOR KINEMATIC DUCTUS</span>
                    ${{selectedGlyph.svg_content}}
                </div>
                <div class="meta-list">
                    <div><strong>Glyph ID:</strong> ${{selectedGlyph.glyph_id}}</div>
                    <div><strong>Bounding Box:</strong> (${{selectedGlyph.bbox.join(', ')}})</div>
                    <div><strong>Dimensions:</strong> ${{selectedGlyph.width}}x${{selectedGlyph.height}} px (Area: ${{selectedGlyph.area}} px²)</div>
                    <div><strong>Fill Factor:</strong> ${{selectedGlyph.fill_factor}} (1D Filiform)</div>
                    <div><strong>Stroke Count:</strong> ${{selectedGlyph.stroke_count}} kinematic strokes</div>
                    <div><strong>Assigned Archetype:</strong> <span style="color:var(--accent); font-weight:bold;">${{selectedGlyph.canonical_type || 'Unclassified'}}</span></div>
                </div>
                <a href="${{selectedGlyph.svg_rel}}" download class="btn-toggle" style="text-decoration:none; margin-top:6px;">⬇ Download Glyph SVG</a>
            `;
        }}

        function renderCatalogueTab() {{
            const cat = data[currentMs].catalogue;
            const grid = document.getElementById('catalogue-grid');
            grid.innerHTML = cat.alphabet.map(a => `
                <div class="catalogue-card">
                    <div class="catalogue-header">
                        <h3>Type ${{a.type_id}}</h3>
                        <span class="badge-freq">${{a.frequency}} occurrences (${{a.percentage}}%)</span>
                    </div>
                    <div class="catalogue-duo">
                        <div class="c-box">
                            <span style="font-size:0.6rem; color:var(--text-muted); margin-bottom:4px;">EXEMPLAR CROP</span>
                            <img src="${{a.exemplar_png}}" alt="${{a.type_id}}">
                        </div>
                        <div class="c-box">
                            <span style="font-size:0.6rem; color:var(--text-muted); margin-bottom:4px;">CANONICAL DUCTUS</span>
                            ${{a.exemplar_svg_content}}
                        </div>
                    </div>
                    <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
                        <div>Mean Strokes: <strong>${{a.mean_strokes}}</strong> | Dim: <strong>${{a.mean_width}}x${{a.mean_height}} px</strong></div>
                        <div style="margin-top:4px;">Exemplar: <code>${{a.exemplar_id}}</code></div>
                    </div>
                </div>
            `).join('');
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

    # Master archival scans first
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
        for page_idx in [2, 3, 4, 5]:  # folios 2r, 2v, 3r, 3v
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

    print(f"\n[*] Inducing Voynich Canonical Alphabet from {len(all_voynich_glyphs)} isolated glyphs...")
    voynich_catalogue = catalogue_builder.build_catalogue(all_voynich_glyphs)
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

    print(f"\n[*] Inducing Seraphinianus Canonical Alphabet from {len(all_serafini_glyphs)} isolated glyphs...")
    serafini_catalogue = catalogue_builder.build_catalogue(all_serafini_glyphs)
    catalogue_builder.export_catalogue_json(serafini_catalogue, base_out / "seraphinianus" / "seraphinianus_alphabet_catalogue.json")
    print(f"  [+] Discovered {serafini_catalogue['canonical_alphabet_size']} unique canonical glyph archetypes across Seraphinianus pages.")

    # 3. Generate Interactive Visual Explorer
    explorer_path = base_out / "grand_glyph_explorer.html"
    generate_explorer_html(voynich_pages_data, voynich_catalogue, serafini_pages_data, serafini_catalogue, explorer_path)


if __name__ == "__main__":
    main()
