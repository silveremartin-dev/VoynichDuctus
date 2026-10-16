"""
Dynamic Grand Glyph Explorer Generator for VoynichDuctus.
Assembles the complete 596-page paleographic atlas (225 Voynich folios + 371 Seraphinianus pages)
into an ultra-fast, zero-waste web application with dynamic on-demand page loading.
"""

import json
from pathlib import Path
import re
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def clean_catalogue_for_embed(cat_dict: dict) -> dict:
    """Retains full archetype metadata, exemplars and transliteration alignments while omitting massive instance lists."""
    cleaned_alphabet = []
    for arch in cat_dict.get("alphabet", []):
        cleaned_alphabet.append({
            "type_id": arch.get("type_id"),
            "rank": arch.get("rank"),
            "cluster_index": arch.get("cluster_index"),
            "frequency": arch.get("frequency"),
            "percentage": arch.get("percentage"),
            "exemplar_id": arch.get("exemplar_id"),
            "exemplar_png": arch.get("exemplar_png", ""),
            "exemplar_svg": arch.get("exemplar_svg", ""),
            "exemplar_svg_content": arch.get("exemplar_svg_content", ""),
            "mean_strokes": arch.get("mean_strokes"),
            "mean_height": arch.get("mean_height"),
            "mean_width": arch.get("mean_width"),
            "corpus_match": arch.get("corpus_match"),
            "corpus_matches": arch.get("corpus_matches"),
            "total_instances_count": arch.get("total_instances_count", arch.get("frequency", 0)),
            "sample_instances": arch.get("sample_instances", [])[:15]
        })
    return {
        "total_glyphs": cat_dict.get("total_glyphs", 0),
        "canonical_alphabet_size": len(cleaned_alphabet),
        "alphabet": cleaned_alphabet
    }


def parse_page_stub_fast(json_path: Path, subfolder: str) -> dict:
    """Ultra-fast header reading without parsing megabytes of internal stroke arrays."""
    stem = json_path.stem
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(800)

    m_pid = re.search(r'"page_id":\s*"([^"]+)"', head)
    page_id = m_pid.group(1) if m_pid else stem

    m_w = re.search(r'"image_width":\s*(\d+)', head)
    m_h = re.search(r'"image_height":\s*(\d+)', head)
    m_lines = re.search(r'"line_count":\s*(\d+)', head)
    m_words = re.search(r'"word_count":\s*(\d+)', head)
    m_glyphs = re.search(r'"glyph_count":\s*(\d+)', head)

    return {
        "page_id": page_id,
        "page_img_rel": f"{subfolder}/pages/{stem}.jpg",
        "json_rel": f"{subfolder}/pages/{stem}.json",
        "image_width": int(m_w.group(1)) if m_w else 1600,
        "image_height": int(m_h.group(1)) if m_h else 2200,
        "line_count": int(m_lines.group(1)) if m_lines else 0,
        "word_count": int(m_words.group(1)) if m_words else 0,
        "glyph_count": int(m_glyphs.group(1)) if m_glyphs else 0,
        "yield_report": {}
    }


def main():
    base_out = Path("output/atlas_dataset")
    voynich_dir = base_out / "voynich"
    serafini_dir = base_out / "seraphinianus"

    print("[*] Loading Voynich Alphabet Catalogue...")
    with open(voynich_dir / "voynich_alphabet_catalogue.json", "r", encoding="utf-8") as f:
        v_cat_raw = json.load(f)
    v_cat = clean_catalogue_for_embed(v_cat_raw)
    print(f"  [+] Voynich Catalogue: {v_cat['canonical_alphabet_size']} canonical archetypes.")

    print("[*] Loading Seraphinianus Alphabet Catalogue...")
    with open(serafini_dir / "seraphinianus_alphabet_catalogue.json", "r", encoding="utf-8") as f:
        s_cat_raw = json.load(f)
    s_cat = clean_catalogue_for_embed(s_cat_raw)
    print(f"  [+] Seraphinianus Catalogue: {s_cat['canonical_alphabet_size']} canonical archetypes.")

    # 1. Index Voynich Pages
    print("[*] Indexing Voynich pages (fast header pass)...")
    v_json_files = sorted((voynich_dir / "pages").glob("*.json"))
    v_pages_stubs = []

    for idx, jf in enumerate(v_json_files):
        stub = parse_page_stub_fast(jf, "voynich")
        # Pre-load full content for f001r (page 0)
        if idx == 0:
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
            stub["yield_report"] = p_full.get("yield_report", {})
        v_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(v_pages_stubs)} Voynich folios.")

    # 2. Index Seraphinianus Pages
    print("[*] Indexing Codex Seraphinianus pages (fast header pass)...")
    s_json_files = sorted((serafini_dir / "pages").glob("*.json"))
    s_pages_stubs = []
    p30_idx = 30

    for idx, jf in enumerate(s_json_files):
        stub = parse_page_stub_fast(jf, "seraphinianus")
        if jf.stem == "serafini_p030":
            p30_idx = idx
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
        s_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(s_pages_stubs)} Seraphinianus pages (p30 index: {p30_idx}).")

    # 3. Read template and inject app data and dynamic loader
    src_template = Path("scripts/generate_full_glyph_corpus_and_explorer.py").read_text(encoding="utf-8")

    match_html = re.search(r'(<!DOCTYPE html>.*</html>)', src_template, re.DOTALL)
    if not match_html:
        raise ValueError("Could not find HTML template in scripts/generate_full_glyph_corpus_and_explorer.py")

    html_template = match_html.group(1)

    # Prepare compact APP_DATA JSON
    app_data = {
        "voynich": {
            "title": "Voynich Manuscript (Beinecke MS 408)",
            "pages": v_pages_stubs,
            "catalogue": v_cat
        },
        "seraphinianus": {
            "title": "Codex Seraphinianus (Luigi Serafini, 1981)",
            "pages": s_pages_stubs,
            "catalogue": s_cat
        }
    }
    app_data_json = json.dumps(app_data, ensure_ascii=False)

    html_out = html_template.replace("{app_json}", app_data_json)

    # 4. Enhance Processed Folios Panel with search input & badge
    old_panel_html = '<div id="pages-list" style="overflow-y:auto; flex:1;"></div>'
    new_panel_html = f"""
                <div style="margin-bottom:8px;">
                    <input id="page-search-input" type="text" placeholder="🔍 Search folio/page... (e.g. 30, f4r)" 
                           style="background:#0b1120; border:1px solid var(--border); color:#fff; font-size:0.75rem; padding:6px 10px; border-radius:6px; width:100%; box-sizing:border-box;" 
                           oninput="renderPagesTab()">
                </div>
                <div id="pages-list" style="overflow-y:auto; flex:1;"></div>
    """
    html_out = html_out.replace(old_panel_html, new_panel_html)

    # Enhance "<h3>Processed Folios</h3>" to include count badge
    html_out = html_out.replace(
        '<h3>Processed Folios</h3>',
        '<h3>Processed Folios <span id="pages-count-badge" style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;"></span></h3>'
    )

    # 5. Inject Enhanced Dynamic Loader, selectPage & renderPagesTab
    dynamic_code = f"""
        // Dynamic On-Demand Page Data Loader (with LRU / in-memory cache)
        const pageLoadCache = new Map();
        let isLoadingPage = false;

        async function ensurePageLoaded(ms, idx) {{
            const page = data[ms].pages[idx];
            if (!page) return null;
            if (page.glyphs && page.glyphs.length > 0) return page;

            const cacheKey = `${{ms}}_${{page.page_id}}`;
            if (pageLoadCache.has(cacheKey)) {{
                const cached = pageLoadCache.get(cacheKey);
                Object.assign(page, cached);
                return page;
            }}

            try {{
                isLoadingPage = true;
                const statusNotice = document.getElementById('canvas-status-notice');
                if (statusNotice) {{
                    statusNotice.style.display = 'block';
                    statusNotice.textContent = `Loading vector coordinates for ${{page.page_id}}...`;
                }}

                const resp = await fetch(page.json_rel);
                if (resp.ok) {{
                    const fullData = await resp.json();
                    page.lines = fullData.lines || [];
                    page.words = fullData.words || [];
                    page.glyphs = fullData.glyphs || [];
                    page.yield_report = fullData.yield_report || {{}};
                    pageLoadCache.set(cacheKey, {{
                        lines: page.lines,
                        words: page.words,
                        glyphs: page.glyphs,
                        yield_report: page.yield_report
                    }});
                }}
            }} catch (err) {{
                console.warn(`Dynamic load notice for ${{page.page_id}} (using direct file protocol?):`, err);
            }} finally {{
                isLoadingPage = false;
                const statusNotice = document.getElementById('canvas-status-notice');
                if (statusNotice) statusNotice.style.display = 'none';
            }}
            return page;
        }}

        async function selectPage(idx) {{
            currentPageIdx = idx;
            selectedGlyph = null;
            selectedWord = null;
            zoomScale = 1.0;
            panX = 0;
            panY = 0;
            await ensurePageLoaded(currentMs, idx);
            renderPagesTab();
        }}

        function renderPagesTab() {{
            const msData = data[currentMs];
            const filterInput = document.getElementById('page-search-input');
            const filterText = filterInput ? filterInput.value.toLowerCase().trim() : '';
            const badge = document.getElementById('pages-count-badge');
            if (badge) badge.textContent = `(${{msData.pages.length}} total)`;

            const pagesList = document.getElementById('pages-list');
            const items = [];
            for (let idx = 0; idx < msData.pages.length; idx++) {{
                const p = msData.pages[idx];
                if (filterText && !p.page_id.toLowerCase().includes(filterText)) continue;
                const wCount = (typeof p.word_count !== 'undefined') ? p.word_count : (p.words ? p.words.length : 0);
                items.push(`
                    <div class="page-item ${{idx === currentPageIdx ? 'active' : ''}}" onclick="selectPage(${{idx}})">
                        <div>
                            <strong>${{p.page_id}}</strong>
                            <div style="font-size:0.72rem; color:var(--text-muted);">${{p.line_count}} lines, ${{wCount}} words</div>
                        </div>
                        <span class="badge-freq">${{p.glyph_count}} glyphs</span>
                    </div>
                `);
            }}
            pagesList.innerHTML = items.join('');

            const page = msData.pages[currentPageIdx];
            if (!page) return;
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

        function setManuscript(ms) {{
            currentMs = ms;
            currentPageIdx = (ms === 'seraphinianus') ? {p30_idx} : 0;
            selectedGlyph = null;
            selectedWord = null;
            archetypeFilterVal = 'ALL';
            activeAlphabetScheme = ms === 'voynich' ? 'eva' : 'serafini';
            zoomScale = 1.0;
            panX = 0;
            panY = 0;
            
            const filterInput = document.getElementById('page-search-input');
            if (filterInput) filterInput.value = '';

            document.getElementById('btn-voynich').className = ms === 'voynich' ? 'btn-toggle active-voynich' : 'btn-toggle';
            document.getElementById('btn-serafini').className = ms === 'seraphinianus' ? 'btn-toggle active-serafini' : 'btn-toggle';
            
            populateAlphabetSchemeOptions();
            ensurePageLoaded(currentMs, currentPageIdx).then(() => {{
                render();
            }});
        }}
    """

    # Replace renderPagesTab, selectPage, and setManuscript in the template
    # Replace renderPagesTab
    html_out = re.sub(r'function renderPagesTab\(\)\s*\{[\s\S]*?renderInspector\(\);\s*\}\s*\}', '', html_out, count=1)
    # Replace selectPage
    html_out = re.sub(r'function selectPage\(idx\)\s*\{[^}]*\}', '', html_out, count=1)
    # Replace setManuscript
    html_out = re.sub(r'function setManuscript\(ms\)\s*\{[\s\S]*?render\(\);\s*\}', '', html_out, count=1)

    # Append dynamic_code right after `<script>`
    html_out = html_out.replace('<script>', f'<script>\n{dynamic_code}\n')

    # Add a subtle notification bar for CORS / local server info
    cors_info_html = """
        <div id="cors-helper-banner" style="background:#1e1b4b; border-bottom:1px solid #6366f1; padding:6px 16px; font-size:0.75rem; color:#c7d2fe; display:flex; justify-content:space-between; align-items:center;">
            <span>💡 <strong>Astuce de navigation locale :</strong> Pour explorer instantanément les 596 folios/pages avec chargement asynchrone des coordonnées, lancez <code>launch_explorer.bat</code> ou ouvrez <code>http://localhost:8000/grand_glyph_explorer.html</code>.</span>
            <button onclick="document.getElementById('cors-helper-banner').style.display='none'" style="background:transparent; border:none; color:#c7d2fe; cursor:pointer; font-weight:bold;">✕</button>
        </div>
    """
    html_out = html_out.replace('<body>', f'<body>\n{cors_info_html}')

    # Output file
    out_html_path = base_out / "grand_glyph_explorer.html"
    out_html_path.write_text(html_out, encoding="utf-8")
    print(f"\n[+] Generated Dynamic Grand Glyph Explorer: {out_html_path.resolve()}")
    print(f"  [+] File size: {out_html_path.stat().st_size / (1024*1024):.2f} MB")

    # Sync to GCS
    try:
        from voynich_ductus.utils.gcs_sync import sync_file_to_gcs
    except ImportError:
        pass


if __name__ == "__main__":
    main()
