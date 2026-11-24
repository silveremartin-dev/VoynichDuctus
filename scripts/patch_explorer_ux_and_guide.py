"""
Script to patch scripts/generate_full_glyph_corpus_and_explorer.py with:
1. Maximize canvas viewport width and height (2.45fr, 100vh - 105px).
2. Remove clutter from above canvas:
   - Remove #canvas-status-banner.
   - Remove #model-scope-banner and "Use ◀ / ▶ or Click to Navigate Folios".
   - Move Translit select from canvas toolbar to Panel 4 Inspector header.
3. Add tooltips (title="...") across toolbar and layers.
4. Add top-level navigation tab: "📖 Methodology & Help Guide".
5. Add rich multi-tab Methodology & Help View with 4 sub-tabs:
   - Metric Scale Calibration (mm, dmm, scale bands, nib width 0.55mm, across Voynich, Seraphinianus, Gallica).
   - Three Segmentation Paradigms (Quill 40° vs AI Neural PCA vs Hybrid Consensus, lexical word invariants, scope).
   - Proposition 4 & The Anti-Circularity Firewall (Popperian Falsification, Zero-Ouroboros).
   - Codicology & Explorer Guide (Quires, singulion ordering, layer shortcuts).
6. Update JS setTab and add setMethodologySubTab.
"""

from pathlib import Path

target_file = Path("scripts/generate_full_glyph_corpus_and_explorer.py")
content = target_file.read_text(encoding="utf-8")

# 1. Update Grid Layout CSS
old_grid = """        /* 4-Panel Page Layout */
        .page-view-layout {{
            display: grid;
            grid-template-columns: 260px 1.4fr 1fr 1.35fr;
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
        }}"""

new_grid = """        /* 4-Panel Page Layout */
        .page-view-layout {{
            display: grid;
            grid-template-columns: 215px 2.45fr 0.95fr 1.25fr;
            gap: 10px;
            height: calc(100vh - 105px);
        }}
        .panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-y: auto;
            padding: 8px;
            display: flex;
            flex-direction: column;
            position: relative;
            min-height: 0;
        }}
        .panel-canvas {{
            padding: 4px !important;
            overflow: hidden !important;
        }}

        /* Methodology & Scientific Calibration Guide */
        .methodology-container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 4px 4px 24px 4px;
        }}
        .methodology-subtabs {{
            display: flex;
            gap: 6px;
            margin-bottom: 14px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            flex-wrap: wrap;
        }}
        .methodology-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }}
        .methodology-card h3 {{
            color: var(--accent);
            font-size: 1.05rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .methodology-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
            margin-top: 10px;
        }}
        .methodology-table th, .methodology-table td {{
            padding: 8px 12px;
            border: 1px solid var(--border);
            text-align: left;
        }}
        .methodology-table th {{
            background: #0b1120;
            color: var(--accent);
            font-weight: 600;
        }}
        .methodology-table tr:nth-child(even) {{
            background: rgba(255, 255, 255, 0.02);
        }}"""

assert old_grid in content, "Error: old_grid not found"
content = content.replace(old_grid, new_grid, 1)

# 2. Canvas-wrapper flex sizing
old_canvas_wrapper = """        /* Interactive Canvas Container & Controls */
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
        }}"""

new_canvas_wrapper = """        /* Interactive Canvas Container & Controls */
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
            flex: 1;
            min-height: 0;
        }}"""

assert old_canvas_wrapper in content, "Error: old_canvas_wrapper not found"
content = content.replace(old_canvas_wrapper, new_canvas_wrapper, 1)

# 3. Top Navbar Nav Tabs
old_nav_tabs = """            <div class="nav-tabs">
                <button id="tab-pages" class="tab-btn active" onclick="setTab('pages')">Manuscript Canvas & Bounding Boxes</button>
                <button id="tab-catalogue" class="tab-btn" onclick="setTab('catalogue')">Canonical Alphabet & Corpora</button>
                <button id="tab-gallica" class="tab-btn" onclick="setTab('gallica')">🏛️ Gallica BnF Calibration</button>
            </div>"""

new_nav_tabs = """            <div class="nav-tabs">
                <button id="tab-pages" class="tab-btn active" onclick="setTab('pages')" title="Interactive high-resolution folio canvas and bounding boxes">Manuscript Canvas & Bounding Boxes</button>
                <button id="tab-catalogue" class="tab-btn" onclick="setTab('catalogue')" title="Canonical alphabet induction, archetypes and transliteration schemes">Canonical Alphabet & Corpora</button>
                <button id="tab-gallica" class="tab-btn" onclick="setTab('gallica')" title="Bibliothèque nationale de France IIIF medieval calibration suites">🏛️ Gallica BnF Calibration</button>
                <button id="tab-methodology" class="tab-btn" onclick="setTab('methodology')" title="Scientific methodology, calibration protocols, 3 segmentation models, and anti-circularity guide">📖 Methodology & Help Guide</button>
            </div>"""

assert old_nav_tabs in content, "Error: old_nav_tabs not found"
content = content.replace(old_nav_tabs, new_nav_tabs, 1)

# 4. Panel 2 Canvas HTML: Unclutter and Enlarge Canvas
old_canvas_panel = """            <!-- 2. High-Res Canvas Overlay with Zoom & Minimap -->
            <div class="panel" style="padding: 4px;">
                <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 8px; border-bottom:1px solid var(--border); margin-bottom:4px; gap:8px; flex-wrap:wrap;">
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <span style="font-size:0.85rem; font-weight:bold; color:var(--accent);" title="Folio Canvas — High-resolution visualization and parchment inspection">Folio Canvas</span>
                        <select id="canvas-filter-select" class="select-filter" onchange="setCanvasFilter(this.value)" style="font-size:0.70rem; padding:2px 6px;" title="Parchment image render mode">
                            <option value="crisp">✨ Ink Boost</option>
                            <option value="sharp">🔥 High-Pass</option>
                            <option value="pure">📜 Pure Ink</option>
                            <option value="raw">📷 Natural</option>
                        </select>
                        <div style="display:flex; align-items:center; gap:4px;">
                            <span style="font-size:0.68rem; color:var(--text-muted); font-weight:600;" title="How intact lexical words are segmented into constituent glyphs">Decomposition:</span>
                            <select id="segmentation-mode-select" class="select-filter" onchange="setSegmentationMode(this.value)" style="font-size:0.70rem; padding:2px 6px; background:#0f172a; border-color:#38bdf8; color:#38bdf8; font-weight:600;" title="Word-to-glyph segmentation and decomposition algorithm">
                                <option value="heuristic_calligraphic" title="Physical medieval quill mechanics (40° nib bevel) and supralinear diacritic decoupling">Quill Ductus (Physical 40°)</option>
                                <option value="neural_word_pca" title="Self-supervised neural word decomposer extracting horizontal principal spatial basis components">AI Decomposition (Neural PCA)</option>
                                <option value="hybrid_consensus" title="Strict mathematical consensus between physical quill kinematics and neural basis decomposition">Hybrid Consensus (Physical + AI)</option>
                            </select>
                        </div>
                        <div style="display:flex; align-items:center; gap:4px;">
                            <span style="font-size:0.68rem; color:#f59e0b; font-weight:700;">Translit:</span>
                            <select id="active-translit-scheme-select" class="select-filter" onchange="setAlphabetScheme(this.value)" style="font-size:0.70rem; padding:2px 6px; background:#0f172a; border-color:#f59e0b; color:#f59e0b; font-weight:600;" title="Transliteration alignment scheme (EVA, Currier, v101, Voynichese)">
                                <option value="eva">EVA</option>
                                <option value="currier">Currier</option>
                                <option value="v101">v101</option>
                                <option value="voynichese">Voynichese (2014)</option>
                            </select>
                        </div>
                    </div>
                    <!-- Clean Modern Layer Control Pills -->
                    <div style="display:flex; gap:5px; align-items:center; flex-wrap:wrap;">
                        <span style="font-size:0.65rem; color:var(--text-muted); margin-right:2px;">Layers:</span>
                        <button id="btn-toggle-lines" class="layer-pill layer-pill-lines active" onclick="toggleLayer('lines')" title="Toggle lines (Key L)">🟣 Lines</button>
                        <button id="btn-toggle-voynich-boxes" class="layer-pill layer-pill-voynich-boxes active" onclick="toggleLayer('voynich_boxes')" title="Toggle Voynich reference consensus boxes (Key B)">📦 Voynich Boxes</button>
                        <button id="btn-toggle-words" class="layer-pill layer-pill-words active" onclick="toggleLayer('words')" title="Toggle words (Key W)">🟡 Words</button>
                        <button id="btn-toggle-glyphs" class="layer-pill layer-pill-glyphs active" onclick="toggleLayer('glyphs')" title="Toggle glyphs (Key G)">🔵 Glyphs</button>
                        <button id="btn-toggle-focus" class="layer-pill layer-pill-focus" onclick="toggleFocusMode()" title="Isolate selected item only (Key F)">🎯 Isolate</button>
                    </div>
                </div>

                <!-- Dynamic Segmentation Status Banner -->
                <div id="canvas-status-banner" class="canvas-status-banner">
                    <span id="canvas-status-text">🖋️ Active Model: <strong style="color:#38bdf8;">Quill Ductus (Physical 40°)</strong></span>
                    <span id="canvas-status-action"></span>
                </div>
                <div style="background:#090e1a; border:1px solid #1e293b; border-radius:4px; padding:3px 8px; margin-bottom:4px; font-size:0.65rem; color:var(--text-muted); display:flex; justify-content:space-between; align-items:center;">
                    <span><strong style="color:var(--accent);">ℹ️ Model Scope:</strong> Lexical words are invariant whitespace units. The 3 models govern how words are decomposed into constituent glyphs.</span>
                    <span style="color:#94a3b8; font-size:0.62rem;">Use ◀ / ▶ or Click to Navigate Folios</span>
                </div>
                <div class="canvas-wrapper" id="canvas-wrapper" title="Scroll to Zoom | Drag to Pan | Double-Click to Fit">"""

new_canvas_panel = """            <!-- 2. High-Res Canvas Overlay with Zoom & Minimap -->
            <div class="panel panel-canvas">
                <div style="display:flex; justify-content:space-between; align-items:center; padding:3px 6px; border-bottom:1px solid var(--border); margin-bottom:3px; gap:6px; flex-wrap:wrap; flex-shrink:0;">
                    <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                        <span style="font-size:0.82rem; font-weight:bold; color:var(--accent);" title="Folio Canvas — High-resolution visualization and parchment inspection">Folio Canvas</span>
                        <select id="canvas-filter-select" class="select-filter" onchange="setCanvasFilter(this.value)" style="font-size:0.68rem; padding:2px 5px;" title="Parchment image render mode">
                            <option value="crisp" title="Contrast-enhanced iron gall ink filter">✨ Ink Boost</option>
                            <option value="sharp" title="High-pass sharpness filter">🔥 High-Pass</option>
                            <option value="pure" title="Pure binarized ink threshold">📜 Pure Ink</option>
                            <option value="raw" title="Raw uncalibrated scan">📷 Natural</option>
                        </select>
                        <div style="display:flex; align-items:center; gap:3px;">
                            <span style="font-size:0.66rem; color:var(--text-muted); font-weight:600;" title="How intact lexical words are segmented into constituent glyphs">Decomp:</span>
                            <select id="segmentation-mode-select" class="select-filter" onchange="setSegmentationMode(this.value)" style="font-size:0.68rem; padding:2px 5px; background:#0f172a; border-color:#38bdf8; color:#38bdf8; font-weight:600;" title="Word-to-glyph segmentation algorithm (Quill Ductus, Neural PCA, Hybrid Consensus)">
                                <option value="heuristic_calligraphic" title="Physical medieval quill mechanics (40° nib bevel) and supralinear diacritic decoupling">Quill Ductus (40°)</option>
                                <option value="neural_word_pca" title="Self-supervised neural word decomposer extracting horizontal principal spatial basis components">AI Neural PCA</option>
                                <option value="hybrid_consensus" title="Strict mathematical consensus between physical quill kinematics and neural basis decomposition">Hybrid Consensus</option>
                            </select>
                        </div>
                    </div>
                    <!-- Clean Modern Layer Control Pills -->
                    <div style="display:flex; gap:4px; align-items:center; flex-wrap:wrap;">
                        <span style="font-size:0.64rem; color:var(--text-muted); margin-right:1px;" title="Toggle graphical overlays on canvas">Layers:</span>
                        <button id="btn-toggle-lines" class="layer-pill layer-pill-lines active" onclick="toggleLayer('lines')" title="Toggle baseline lines overlay (Key L)">🟣 Lines</button>
                        <button id="btn-toggle-voynich-boxes" class="layer-pill layer-pill-voynich-boxes active" onclick="toggleLayer('voynich_boxes')" title="Toggle Voynich reference consensus boxes (Key B)">📦 Voynich Boxes</button>
                        <button id="btn-toggle-words" class="layer-pill layer-pill-words active" onclick="toggleLayer('words')" title="Toggle word bounding boxes (Key W)">🟡 Words</button>
                        <button id="btn-toggle-glyphs" class="layer-pill layer-pill-glyphs active" onclick="toggleLayer('glyphs')" title="Toggle extracted glyph bounding boxes (Key G)">🔵 Glyphs</button>
                        <button id="btn-toggle-focus" class="layer-pill layer-pill-focus" onclick="toggleFocusMode()" title="Isolate selected item only, dimming all others (Key F)">🎯 Isolate</button>
                    </div>
                </div>

                <div class="canvas-wrapper" id="canvas-wrapper" title="Scroll to Zoom | Drag to Pan | Double-Click to Fit">"""

assert old_canvas_panel in content, "Error: old_canvas_panel not found"
content = content.replace(old_canvas_panel, new_canvas_panel, 1)

# 5. Panel 4 Header: Incorporate Translit Select
old_panel4 = """            <!-- 4. Detailed Inspector -->
            <div class="panel">
                <h3>Glyph / Word Inspector & Ductus Kinematics</h3>
                <div class="inspector-panel" id="inspector-content"></div>
            </div>"""

new_panel4 = """            <!-- 4. Detailed Inspector -->
            <div class="panel">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); padding-bottom:5px; margin-bottom:8px; gap:8px;">
                    <h3 style="margin:0; border:none; padding:0; font-size:0.85rem; color:var(--accent);" title="Detailed inspection of selected word or glyph, physical metrics, ductus strokes and transcription">
                        Inspector &amp; Ductus
                    </h3>
                    <div style="display:flex; align-items:center; gap:4px;">
                        <span style="font-size:0.65rem; color:#f59e0b; font-weight:700;" title="Standard transliteration reference alphabet">Translit:</span>
                        <select id="active-translit-scheme-select" class="select-filter" onchange="setAlphabetScheme(this.value)" style="font-size:0.68rem; padding:2px 5px; background:#0f172a; border-color:#f59e0b; color:#f59e0b; font-weight:600;" title="Transliteration alignment scheme (EVA, Currier, v101, Voynichese 2014)">
                            <option value="eva">EVA</option>
                            <option value="currier">Currier</option>
                            <option value="v101">v101</option>
                            <option value="voynichese">Voynichese (2014)</option>
                        </select>
                    </div>
                </div>
                <div class="inspector-panel" id="inspector-content"></div>
            </div>"""

assert old_panel4 in content, "Error: old_panel4 not found"
content = content.replace(old_panel4, new_panel4, 1)

# 6. Add Methodology View right before </main>
old_main_close = """                </div>
            </div>
        </div>
    </main>"""

new_main_close = """                </div>
            </div>
        </div>

        <!-- TAB 4: Methodology, Calibration & Help Guide -->
        <div id="view-methodology" style="display: none;">
            <div class="methodology-container">
                <div style="margin-bottom:16px;">
                    <h2 style="font-size:1.35rem; color:var(--accent); display:flex; align-items:center; gap:8px;">
                        <span>📖</span> Scientific Methodology, Calibration Protocols &amp; User Guide
                    </h2>
                    <p style="color:var(--text-muted); font-size:0.85rem; margin-top:4px; line-height:1.5;">
                        Comprehensive reference documentation explaining the physical metric calibration, the three stroke segmentation paradigms, the anti-circularity firewall (Proposition 4), and codicological navigation.
                    </p>
                </div>

                <!-- Methodology Sub-Navigation Tabs -->
                <div class="methodology-subtabs">
                    <button id="btn-subtab-metric" class="tab-btn active" onclick="setMethodologySubTab('metric')" title="Physical metric scale calibration, millimetric conversions, and nib footprint thresholds">📏 1. Metric Calibration &amp; Scales</button>
                    <button id="btn-subtab-paradigms" class="tab-btn" onclick="setMethodologySubTab('paradigms')" title="Detailed comparative breakdown of Quill Ductus, Neural Word PCA, and Hybrid Consensus">🖋️ 2. Three Segmentation Paradigms</button>
                    <button id="btn-subtab-anticirc" class="tab-btn" onclick="setMethodologySubTab('anticirc')" title="Proposition 4, Popperian discordance detector, and zero-ouroboros anti-circularity firewall">🛡️ 3. Anti-Circularity &amp; Proposition 4</button>
                    <button id="btn-subtab-manual" class="tab-btn" onclick="setMethodologySubTab('manual')" title="Codicological quire structure, Singulion ordering, layer controls, and keyboard shortcuts">🧭 4. Codicology &amp; User Manual</button>
                </div>

                <!-- PANE 1: Metric Scale Calibration & Invariant Thresholds -->
                <div id="methodology-pane-metric" class="methodology-pane">
                    <div class="methodology-card">
                        <h3>📏 Physical Metric Scale Calibration Across Codices</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6;">
                            Digital paleography cannot rely on raw pixel coordinates, because digital scan resolutions vary drastically across digitization campaigns. VoynichDuctus calibrates every manuscript against physical millimeter measurements documented in library conservation records. All measurements are internally computed in <strong>decimillimeters (dmm = 0.1 mm)</strong> to maintain integer precision without floating-point drift.
                        </p>
                        <table class="methodology-table">
                            <thead>
                                <tr>
                                    <th>Manuscript / Codex</th>
                                    <th>Conservation Source</th>
                                    <th>Physical Folio Height</th>
                                    <th>Normalized Scan Height</th>
                                    <th>Calibration Scale Factor</th>
                                    <th>Ink Rheology &amp; Spread</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong style="color:var(--voynich-color);">Voynich Manuscript</strong> (Beinecke MS 408)</td>
                                    <td>Beinecke Rare Book &amp; Manuscript Library, Yale University</td>
                                    <td><strong>225 mm</strong> (2250 dmm)</td>
                                    <td>1800 px</td>
                                    <td><strong>0.125 mm/px</strong> (1.25 dmm/px | 8.0 px/mm)</td>
                                    <td>Iron-gall ink on parchment, 0.22–0.35 mm lateral diffusion</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:var(--serafini-color);">Codex Seraphinianus</strong> (Luigi Serafini, 1981)</td>
                                    <td>Franco Maria Ricci (FMR) First Edition, Milan</td>
                                    <td><strong>350 mm</strong> (3500 dmm)</td>
                                    <td>2187 px</td>
                                    <td><strong>0.160 mm/px</strong> (1.60 dmm/px | 6.25 px/mm)</td>
                                    <td>Modern synthetic pigment ribbon ink, high edge acutance</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:#f59e0b;">BnF Latin 6823</strong> (Liber de herbis)</td>
                                    <td>Bibliothèque nationale de France (Gallica IIIF)</td>
                                    <td><strong>280 mm</strong> (2800 dmm)</td>
                                    <td>2000 px</td>
                                    <td><strong>0.140 mm/px</strong> (1.40 dmm/px | 7.14 px/mm)</td>
                                    <td>14th–15th c. Italian monastic iron-gall scriptorium ink</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="methodology-card">
                        <h3>⚖️ Physical Paleographical Invariant Threshold Rules</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6; margin-bottom:10px;">
                            The physical scale calibration guarantees that segmentation parameters remain strictly invariant across scales and digitization resolutions:
                        </p>
                        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px;">
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px;">
                                <div style="color:var(--accent); font-weight:bold; font-size:0.80rem; margin-bottom:4px;">🖋️ Physical Quill Nib Width (0.55 mm)</div>
                                <div style="color:var(--text-muted); font-size:0.75rem; line-height:1.4;">
                                    Represents the typical cut tip width of a 15th-century goose quill (0.55 mm = 5.5 dmm = 4.4 px on Voynich scans).
                                </div>
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px;">
                                <div style="color:#10b981; font-weight:bold; font-size:0.80rem; margin-bottom:4px;">🎯 Nib Impact Footprint (&le; 0.55 mm)</div>
                                <div style="color:var(--text-muted); font-size:0.75rem; line-height:1.4;">
                                    Ink objects with diameter &le; 0.55 mm (or &le; 8 px, area &le; 25 px²) are classified as instantaneous pen touchdown impact dots (<code>is_impact=True</code>, <code>length=0.0</code>) rather than continuous stroke paths.
                                </div>
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px;">
                                <div style="color:#f59e0b; font-weight:bold; font-size:0.80rem; margin-bottom:4px;">↔️ Inter-Word Whitespace (&ge; 1.1 mm)</div>
                                <div style="color:var(--text-muted); font-size:0.75rem; line-height:1.4;">
                                    A physical whitespace gap &ge; 1.1 mm (11 dmm = 8.8 px) defines guaranteed lexical word separation, preventing accidental cross-word ligature bridging.
                                </div>
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px;">
                                <div style="color:#c084fc; font-weight:bold; font-size:0.80rem; margin-bottom:4px;">📊 Four Physical Scale Stratification Bands</div>
                                <div style="color:var(--text-muted); font-size:0.75rem; line-height:1.4;">
                                    Glyphs are partitioned into 4 non-overlapping scale tiers: <strong>TINY</strong> (&le; 1.3 mm, diacritics/spatters), <strong>CORE</strong> (1.3–3.4 mm, standard minims/bowls), <strong>TALL</strong> (&ge; 3.4 mm, gallows/ascenders), and <strong>WIDE</strong> (&ge; 3.4 mm, horizontal crossbars).
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- PANE 2: Three Segmentation Paradigms -->
                <div id="methodology-pane-paradigms" class="methodology-pane" style="display:none;">
                    <div class="methodology-card">
                        <h3>ℹ️ Fundamental Distinction: Word Boundaries vs Glyph Decomposition</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6;">
                            A critical architectural principle of VoynichDuctus is the <strong>strict decoupling between lexical word partitioning and intra-word glyph decomposition</strong>:
                        </p>
                        <div style="background:#090d16; border:1px solid #1e293b; border-radius:6px; padding:12px; margin-top:8px; line-height:1.6; font-size:0.78rem;">
                            <div><strong style="color:#10b981;">1. Lexical Words are Spatial Invariants:</strong> Across all 225 folios, words are defined by physical whitespace boundaries and Voronoi/watershed spatial partitioning. All 3 models preserve the exact same 37,601 word containers (0 shared parchment, zero overlaps).</div>
                            <div style="margin-top:6px;"><strong style="color:#38bdf8;">2. What the 3 Models Govern:</strong> The models do not change where words begin and end; they govern how ink inside an intact lexical word is decomposed into constituent glyph archetypes.</div>
                        </div>
                    </div>

                    <div class="methodology-card">
                        <h3>⚙️ Comparative Analysis of the Three Segmentation Paradigms</h3>
                        <table class="methodology-table">
                            <thead>
                                <tr>
                                    <th>Model Paradigm</th>
                                    <th>Application Scope</th>
                                    <th>Algorithmic Mechanism</th>
                                    <th>Calligraphic / Paleographic Prior</th>
                                    <th>Objective &amp; Strengths</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong style="color:#38bdf8;">1. Quill Ductus</strong><br>(Physical 40° Mechanics)</td>
                                    <td><span class="badge-type" style="background:#0c4a6e; color:#bae6fd;">Full Corpus (225 Folios)</span></td>
                                    <td>1D Medial Axis Transform + Euclidean Distance Transform + NetworkX Graph Junction Resolver.</td>
                                    <td>Fixed 40° nib bevel angle, pleins et déliés thickness profiling, strict physical ceiling of 1–4 strokes per character, supralinear diacritic decoupling.</td>
                                    <td>Accurately reconstructs the biomechanical movements of a medieval right-handed scribe with realistic quill pen lifts.</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:#c084fc;">2. AI Decomposition</strong><br>(Neural Spatial PCA)</td>
                                    <td><span class="badge-type" style="background:#581c87; color:#f3e8ff;">Comparative Benchmark (f1r)</span></td>
                                    <td>Deep spatial convolutional encoder generating 1D continuous latent trajectories Z(x) &isin; &Ropf;³² + Non-Negative Matrix Factorization (NMF).</td>
                                    <td>Knife-free unsupervised spatial basis induction without heuristic geometric cutting.</td>
                                    <td>Tests whether glyph boundaries emerge naturally from continuous visual representations without rule-based assumptions.</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:#10b981;">3. Hybrid Consensus</strong><br>(Physical ∩ AI)</td>
                                    <td><span class="badge-type" style="background:#064e3b; color:#a7f3d0;">Validation Benchmark (f1r)</span></td>
                                    <td>Strict geometric centroid intersection (&epsilon; &le; 3 px) between Model 1 (quill physics) and Model 2 (neural decomposition).</td>
                                    <td>Multi-model concordance: a boundary is validated only when two completely independent paradigms agree.</td>
                                    <td>Provides the highest standard of mathematical proof, isolating robust glyph centroids that are free from single-model artifacts.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- PANE 3: Anti-Circularity & Proposition 4 -->
                <div id="methodology-pane-anticirc" class="methodology-pane" style="display:none;">
                    <div class="methodology-card">
                        <h3>🐍 The Ouroboros Risk ("Le serpent qui se mord la queue")</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6;">
                            In Voynich studies, researchers frequently fall into circular reasoning: they adopt subjective 20th-century transliteration schemes (like the European Voynich Alphabet, EVA) as ground truth, segment the text into EVA letters, and then claim their computational results confirm the linguistic structure of the manuscript. This is an epistemic feedback loop where the analysis merely mirrors human subjective priors.
                        </p>
                        <div style="background:#090d16; border-left:4px solid #ef4444; border-radius:4px; padding:10px 14px; margin-top:10px; font-size:0.78rem; line-height:1.5;">
                            <strong style="color:#ef4444;">VoynichDuctus Anti-Circularity Guarantee:</strong>
                            Stroke segmentation, kinematics inference, and archetype clustering are conducted <strong>100% blind to EVA, Currier, and human transliterations</strong>. Transcription schemes are loaded exclusively downstream as empirical diagnostic comparators.
                        </div>
                    </div>

                    <div class="methodology-card">
                        <h3>🔬 Proposition 4: Popperian Falsification &amp; Discordance Engine</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6; margin-bottom:10px;">
                            Proposition 4 formulates a falsifiable Popperian test by comparing the physically extracted stroke components against standard transliteration counts for every lexical word:
                        </p>
                        <table class="methodology-table">
                            <thead>
                                <tr>
                                    <th>Diagnostic State</th>
                                    <th>Condition</th>
                                    <th>Paleographical Interpretation</th>
                                    <th>Significance for Ductus Theory</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong style="color:#10b981;">Concordance</strong></td>
                                    <td>N<sub>ductus</sub> = N<sub>EVA</sub></td>
                                    <td>Physical scribal kinematics and manual transcription identify the exact same number of discrete graphemes.</td>
                                    <td>Validates that the historical transcription accurately matches physical pen lifts for this lexical unit.</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:#f59e0b;">Sub-Glyphic Cleavage</strong></td>
                                    <td>N<sub>ductus</sub> &gt; N<sub>EVA</sub></td>
                                    <td>Physical quill mechanics reveals pen lifts or distinct strokes inside what manual schemes treat as a single letter.</td>
                                    <td>Demonstrates that compound characters (such as gallows <code>cth</code>, <code>ckh</code>) are composite ligatures composed of 3 to 4 distinct pen movements.</td>
                                </tr>
                                <tr>
                                    <td><strong style="color:#c084fc;">Graphemic Conflation</strong></td>
                                    <td>N<sub>ductus</sub> &lt; N<sub>EVA</sub></td>
                                    <td>The scribe executed in a single continuous stroke what manual schemes divide into two separate letters.</td>
                                    <td>Reveals natural cursive ductus shortcuts and continuous pen trajectories that human transcribers over-segmented.</td>
                                </tr>
                            </tbody>
                        </table>
                        <div style="background:#090d16; border:1px solid #1e293b; border-radius:6px; padding:10px; margin-top:12px; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem;">
                            <span><strong style="color:#10b981;">🛡️ Zero Dropped Ink Guarantee:</strong> 100% of the ink pixels inside every bounding box are assigned to physical strokes. Exactly 0 px of ink are ignored.</span>
                        </div>
                    </div>
                </div>

                <!-- PANE 4: Codicology & User Manual -->
                <div id="methodology-pane-manual" class="methodology-pane" style="display:none;">
                    <div class="methodology-card">
                        <h3>📜 Manuscript Codicological Structure</h3>
                        <p style="color:var(--text-muted); font-size:0.82rem; line-height:1.6;">
                            The Voynich Manuscript (Beinecke MS 408) comprises 225 folios across 20 quires, organized into 6 traditional thematic sections:
                        </p>
                        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; margin-top:10px;">
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#10b981;">🌿 Herbal Section (f1r–f66v):</strong> Large botanical watercolors with descriptive paragraphs of cursive text.
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#38bdf8;">⭐ Astronomical / Astrological (f67r1–f73v):</strong> Circular diagrams with sun, moon, stars, and zodiac medallions.
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#06b6d4;">🛁 Biological / Balneological (f75r–f84v):</strong> Dense networks of water basins, conduits, and female figures.
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#c084fc;">🌌 Cosmological (f85r1–f86v6):</strong> Famous 9-fold rosette foldout with interconnecting causeways.
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#f59e0b;">🏺 Pharmaceutical (f87r–f102v2):</strong> Apothecary jars alongside botanical roots and plants.
                            </div>
                            <div style="background:#090d16; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:0.75rem;">
                                <strong style="color:#ef4444;">📜 Recipes / Text (f103r–f116v):</strong> Over 300 short paragraphs each marked with a seven-pointed star.
                            </div>
                        </div>
                    </div>

                    <div class="methodology-card">
                        <h3>⌨️ Interactive Canvas Controls &amp; Keyboard Shortcuts</h3>
                        <table class="methodology-table">
                            <thead>
                                <tr>
                                    <th>Action / Control</th>
                                    <th>Mouse Interaction</th>
                                    <th>Keyboard Shortcut</th>
                                    <th>Description</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Zoom In / Out</strong></td>
                                    <td>Mouse Wheel / Toolbar +/- Buttons</td>
                                    <td><code>+</code> / <code>-</code></td>
                                    <td>Zooms into parchment fiber and ink edge details.</td>
                                </tr>
                                <tr>
                                    <td><strong>Pan Viewport</strong></td>
                                    <td>Click and Drag on Canvas / Minimap</td>
                                    <td>Arrow Keys</td>
                                    <td>Moves the canvas viewport smoothly across large folios.</td>
                                </tr>
                                <tr>
                                    <td><strong>Fit to Window</strong></td>
                                    <td>Double-Click Canvas / Fit Button</td>
                                    <td><code>Space</code> / <code>0</code></td>
                                    <td>Automatically centers and fits the full manuscript page in the view.</td>
                                </tr>
                                <tr>
                                    <td><strong>Toggle Lines</strong></td>
                                    <td>Click 🟣 Lines Pill</td>
                                    <td><code>L</code></td>
                                    <td>Toggles horizontal baseline text line detections.</td>
                                </tr>
                                <tr>
                                    <td><strong>Toggle Voynich Boxes</strong></td>
                                    <td>Click 📦 Voynich Boxes Pill</td>
                                    <td><code>B</code></td>
                                    <td>Toggles official consensus reference word bounding boxes.</td>
                                </tr>
                                <tr>
                                    <td><strong>Toggle Words</strong></td>
                                    <td>Click 🟡 Words Pill</td>
                                    <td><code>W</code></td>
                                    <td>Toggles spatial disjoint partitioned lexical word boxes.</td>
                                </tr>
                                <tr>
                                    <td><strong>Toggle Glyphs</strong></td>
                                    <td>Click 🔵 Glyphs Pill</td>
                                    <td><code>G</code></td>
                                    <td>Toggles extracted glyph bounding boxes and stroke trajectories.</td>
                                </tr>
                                <tr>
                                    <td><strong>Isolate Selection</strong></td>
                                    <td>Click 🎯 Isolate Pill</td>
                                    <td><code>F</code></td>
                                    <td>Dims all other words and glyphs to inspect only the selected item.</td>
                                </tr>
                                <tr>
                                    <td><strong>Folio Navigation</strong></td>
                                    <td>Prev / Next Buttons</td>
                                    <td><code>◀</code> / <code>▶</code> (Left / Right)</td>
                                    <td>Jumps to the previous or next manuscript folio in order.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </main>"""

assert old_main_close in content, "Error: old_main_close not found"
content = content.replace(old_main_close, new_main_close, 1)

# 7. JavaScript: Update setTab and add setMethodologySubTab
old_set_tab = """        function setTab(tab) {{
            currentTab = tab;
            document.getElementById('tab-pages').className = tab === 'pages' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tab-catalogue').className = tab === 'catalogue' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tab-gallica').className = tab === 'gallica' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('view-pages').style.display = tab === 'pages' ? 'grid' : 'none';
            document.getElementById('view-catalogue').style.display = tab === 'catalogue' ? 'block' : 'none';
            document.getElementById('view-gallica').style.display = tab === 'gallica' ? 'block' : 'none';
            render();
        }}"""

new_set_tab = """        function setTab(tab) {{
            currentTab = tab;
            document.getElementById('tab-pages').className = tab === 'pages' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tab-catalogue').className = tab === 'catalogue' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('tab-gallica').className = tab === 'gallica' ? 'tab-btn active' : 'tab-btn';
            const tabMeth = document.getElementById('tab-methodology');
            if (tabMeth) tabMeth.className = tab === 'methodology' ? 'tab-btn active' : 'tab-btn';
            document.getElementById('view-pages').style.display = tab === 'pages' ? 'grid' : 'none';
            document.getElementById('view-catalogue').style.display = tab === 'catalogue' ? 'block' : 'none';
            document.getElementById('view-gallica').style.display = tab === 'gallica' ? 'block' : 'none';
            const viewMeth = document.getElementById('view-methodology');
            if (viewMeth) viewMeth.style.display = tab === 'methodology' ? 'block' : 'none';
            render();
        }}

        function setMethodologySubTab(subTabId) {{
            const tabs = ['metric', 'paradigms', 'anticirc', 'manual'];
            tabs.forEach(t => {{
                const btn = document.getElementById('btn-subtab-' + t);
                const pane = document.getElementById('methodology-pane-' + t);
                if (btn) btn.className = (t === subTabId) ? 'tab-btn active' : 'tab-btn';
                if (pane) pane.style.display = (t === subTabId) ? 'block' : 'none';
            }});
        }}"""

assert old_set_tab in content, "Error: old_set_tab not found"
content = content.replace(old_set_tab, new_set_tab, 1)

target_file.write_text(content, encoding="utf-8")
print("[+] Successfully patched scripts/generate_full_glyph_corpus_and_explorer.py!")
