"""
Full-scale batch vectorization and paleographic atlas generator.
Processes real master scans of the Voynich Manuscript (Beinecke MS 408) and the Codex Seraphinianus (Luigi Serafini).
Applies color normalization, illumination flattening, and chromatic pigment separation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from voynich_ductus.ingestion.pdf_loader import PDFScanLoader
from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.ingestion.color_normalizer import ColorIlluminationNormalizer
from voynich_ductus.ingestion.transcription_reference import TranscriptionReference, PaleographyYieldValidator
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer
from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
from voynich_ductus.utils.io import BenchmarkFormatter


def process_manuscript_page(
    page_id: str,
    image: Image.Image,
    output_dir: Path,
    subfolder: str,
    normalizer: ColorIlluminationNormalizer,
    max_words_per_page: int = 40
) -> Dict[str, Any]:
    """
    Normalizes color/illumination, separates ink from colored pigments,
    slices text lines & words, and vectorizes into SVG ductus files.
    """
    print(f"[*] Processing {subfolder.upper()} page '{page_id}' (Resolution: {image.size[0]}x{image.size[1]})...")
    
    # 1. Color normalization & Chromatic pigment separation (strips green/ochre/blue paint)
    ink_mask, _ = normalizer.extract_ink_mask_chromatic(image)
    
    # Fallback to local Sauvola adaptive binarization if ink mask is too sparse
    if np.sum(ink_mask) < 2000:
        binarizer = Binarizer(method="sauvola", window_size=25, k=0.22)
        ink_mask = binarizer.binarize(image)
        ink_mask = binarizer.remove_small_artifacts(ink_mask, min_size=5)

    # 2. Text Paragraph & Line Extraction (excludes marginal drawings & header arches)
    segmenter = LineSegmenter(min_line_pitch=30, min_line_height=16, min_word_width=18, min_word_area=35)
    lines = segmenter.segment_lines(ink_mask, auto_isolate=True)
    print(f"  [+] Identified {len(lines)} clean text lines on '{page_id}'")

    skel_engine = Skeletonizer(method="medial_axis")
    graph_extractor = StrokeGraphExtractor()
    resolver = JunctionResolver(right_handed_prior=True)

    words_data = []
    all_strokes = []
    word_count = 0

    for line in lines:
        if word_count >= max_words_per_page:
            break
        raw_words = segmenter.segment_words(line["image"], line_offset=(line["bbox"][0], line["bbox"][1]), space_gap_min=12)
        for w in raw_words:
            if word_count >= max_words_per_page:
                break
            w_img = w["image"]
            if w_img.shape[0] < 10 or w_img.shape[1] < 15:
                continue

            # Vectorize word
            skel, widths = skel_engine.extract_skeleton(w_img)
            pixel_graph = graph_extractor.build_pixel_graph(skel, widths)
            raw_strokes = graph_extractor.decompose_into_strokes(pixel_graph)
            if not raw_strokes:
                continue
            ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)

            word_id = f"{page_id}_{line['line_id']}_{w['word_id']}"
            svg_filename = f"{word_id}.svg"
            png_filename = f"{word_id}.png"

            svg_path = output_dir / subfolder / "svg" / svg_filename
            png_path = output_dir / subfolder / "png" / png_filename

            # Save clean cropped PNG patch
            y0, x0, y1, x1 = w["bbox"]
            crop_patch = image.crop((x0, y0, x1, y1))
            crop_patch.save(png_path)

            # Export SVG
            wh, ww = w_img.shape
            svg_str = VectorExporter.to_svg(ordered_strokes, width=ww, height=wh, output_path=svg_path)

            words_data.append({
                "word_id": word_id,
                "page_id": page_id,
                "bbox": w["bbox"],
                "stroke_count": len(ordered_strokes),
                "strokes": ordered_strokes,
                "svg_rel": f"{subfolder}/svg/{svg_filename}",
                "png_rel": f"{subfolder}/png/{png_filename}",
                "svg_content": svg_str
            })

            all_strokes.extend(ordered_strokes)
            word_count += 1

    print(f"  [+] Vectorized {len(words_data)} intact words ({len(all_strokes)} strokes).")
    return {
        "page_id": page_id,
        "line_count": len(lines),
        "word_count": len(words_data),
        "words": words_data,
        "strokes": all_strokes
    }


def cluster_and_induce_alphabet(all_strokes: List[Dict[str, Any]], n_clusters: int = 25) -> Tuple[GlyphClusterer, np.ndarray, StrokeTokenizer]:
    """
    Induces canonical alphabet from extracted real stroke features via unsupervised clustering.
    """
    print(f"[*] Extracting 16-D geometric descriptors from {len(all_strokes)} strokes...")
    feature_extractor = GeometricFeatureExtractor()
    features = feature_extractor.extract_batch(all_strokes)

    latent_dim = min(8, features.shape[1], features.shape[0])
    projector = StrokeLatentProjector(latent_dim=latent_dim)
    latent_space = projector.fit_transform(features)

    print(f"[*] Discovering canonical alphabet via unsupervised clustering (target = ~{n_clusters} glyph primitives)...")
    clusterer = GlyphClusterer(method="agglomerative", n_clusters=min(n_clusters, len(latent_space)))
    labels = clusterer.fit_predict(latent_space)

    tokenizer = StrokeTokenizer(prefix="G")
    return clusterer, labels, tokenizer


def generate_grand_atlas_html(voynich_words: List[Dict], serafini_words: List[Dict], benchmark_results: Dict, yield_reports: List[Dict], output_path: Path):
    """
    Renders the Grand Paleography Visual Atlas comparing real Voynich and Seraphinianus words,
    along with information-theoretic diagnostics and paleographic extraction yield benchmarks.
    """
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#00ced1", "#e6ab02", "#66a61e"]

    def render_cards(word_list, max_items=36):
        cards = []
        for it in word_list[:max_items]:
            chips = "".join(f'<span class="chip" style="background:{palette[i % len(palette)]}">T{i+1}</span>' for i in range(min(it['stroke_count'], 10)))
            if it['stroke_count'] > 10:
                chips += f'<span class="chip" style="background:#475569">+{it["stroke_count"]-10}</span>'

            cards.append(f"""
            <div class="card">
                <div class="card-header">
                    <h4>{it['word_id']}</h4>
                    <span class="badge">{it['stroke_count']} strokes</span>
                </div>
                <div class="duo-view">
                    <div class="box">
                        <span class="box-lbl">Scan Crop</span>
                        <img src="{it['png_rel']}" alt="{it['word_id']}">
                    </div>
                    <div class="box svg-box">
                        <span class="box-lbl">Ductus SVG</span>
                        {it['svg_content']}
                    </div>
                </div>
                <div class="legend-row">
                    <span class="legend-lbl">Kinematic Sequence:</span>
                    <div class="chips-container">{chips}</div>
                </div>
                <a href="{it['svg_rel']}" download class="btn-dl">⬇ SVG</a>
            </div>
            """)
        return "\n".join(cards)

    table_rows = []
    for name, res in benchmark_results.items():
        ent = res["entropy"]
        mem = res["memory"]
        comp = res["compressibility"]
        mark = res["markov_automata"]["order_2"]
        diag = res["diagnostic_diagnosis"]
        table_rows.append(f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{res['token_count']}</td>
            <td>{ent['h1_char_bits']:.2f}</td>
            <td>{ent['h2_cond_char_bits']:.2f}</td>
            <td>{ent['h2_drop_ratio']*100:.1f}%</td>
            <td>{mem['hurst_exponent']:.3f}</td>
            <td>{comp['gzip_ratio']:.3f}</td>
            <td>{mark['top1_prediction_accuracy']:.2f}</td>
            <td><span class="badge-hyp">{diag['top_hypothesis']}</span></td>
        </tr>
        """)

    yield_rows = []
    for yr in yield_reports:
        if yr.get("has_reference"):
            diag_str = ", ".join(yr["diagnostics"]) if yr["diagnostics"] else "Nominal Yield"
            yield_rows.append(f"""
            <tr>
                <td><strong>{yr['folio_id']}</strong> ({yr['section']})</td>
                <td>{yr['currier_language']} / {yr['scribe_hand']}</td>
                <td>{yr['detected_lines']} / {yr['expected_lines']} ({yr['line_yield_pct']}%)</td>
                <td>{yr['detected_words']} / {yr['expected_words']} ({yr['word_yield_pct']}%)</td>
                <td>{yr['extracted_strokes']}</td>
                <td>{yr['avg_strokes_per_word']}</td>
                <td><span class="badge-hyp">{yr['status']}</span> <small style="color:var(--text-dim)">{diag_str}</small></td>
            </tr>
            """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoynichDuctus & Codex Seraphinianus — Real Master Scans Vector Paleography Atlas</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --card: #151d2e;
            --border: #233148;
            --accent: #38bdf8;
            --voynich-accent: #f59e0b;
            --serafini-accent: #a855f7;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            max-width: 1200px;
            margin: 0 auto 35px auto;
        }}
        h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
        .intro {{ color: var(--text-dim); max-width: 900px; margin: 0 auto; line-height: 1.5; }}
        
        .section-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.4rem;
            margin: 40px 0 20px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border);
        }}
        .badge-voynich {{ background: #d97706; color: white; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; }}
        .badge-serafini {{ background: #7e22ce; color: white; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .card-header h4 {{ margin: 0; font-size: 0.95rem; color: var(--accent); }}
        .badge {{ background: #1e293b; color: #94a3b8; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
        
        .duo-view {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }}
        .box {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80px;
        }}
        .box-lbl {{ font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 6px; }}
        .box img {{ max-width: 90%; max-height: 70px; border-radius: 4px; object-fit: contain; }}
        .svg-box svg {{ max-width: 100%; max-height: 70px; }}
        
        .legend-row {{ margin-bottom: 10px; }}
        .legend-lbl {{ font-size: 0.75rem; color: var(--text-dim); display: block; margin-bottom: 4px; }}
        .chips-container {{ display: flex; flex-wrap: wrap; gap: 4px; }}
        .chip {{ font-size: 0.65rem; font-weight: bold; color: white; padding: 1px 6px; border-radius: 3px; }}
        
        .btn-dl {{
            display: inline-block;
            background: #1e293b;
            color: #f1f5f9;
            text-decoration: none;
            padding: 4px 10px;
            font-size: 0.75rem;
            border-radius: 4px;
            align-self: flex-end;
            transition: background 0.2s;
        }}
        .btn-dl:hover {{ background: var(--accent); color: #0b0f19; }}
        
        .table-container {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow-x: auto;
            margin-top: 20px;
        }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
        th {{ background: #0f172a; color: var(--accent); font-weight: 600; }}
        .badge-hyp {{ background: #0369a1; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VoynichDuctus & Codex Seraphinianus — Real Master Scans Paleography Atlas</h1>
        <p class="intro">
            High-purity batch vectorization with flat-field illumination flattening and chromatic pigment separation.
            Colored illustration areas (green leaves, ochre roots, blue water) are automatically rejected to extract pure iron-gall and monochrome cursive text strokes.
        </p>
    </div>

    <div class="section-title">
        <span>Voynich Manuscript (Beinecke MS 408 — Master Scans 1r & 1v)</span>
        <span class="badge-voynich">{len(voynich_words)} Pure Words Vectorized</span>
    </div>
    <div class="grid">
        {render_cards(voynich_words)}
    </div>

    <div class="section-title">
        <span>Codex Seraphinianus (Luigi Serafini, 1981 — High-Resolution Scans)</span>
        <span class="badge-serafini">{len(serafini_words)} Pure Words Vectorized</span>
    </div>
    <div class="grid">
        {render_cards(serafini_words)}
    </div>

    <div class="section-title">
        <span>Ground-Truth Paleographic Yield Benchmark (Takahashi EVA Census vs Automated Vectorization)</span>
    </div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Folio & Section</th>
                    <th>Language / Scribe</th>
                    <th>Lines (Detected / Census)</th>
                    <th>Words (Vectorized / Census)</th>
                    <th>Extracted Strokes</th>
                    <th>Strokes / Word</th>
                    <th>Calibration Diagnostic</th>
                </tr>
            </thead>
            <tbody>
                {"".join(yield_rows)}
            </tbody>
        </table>
    </div>

    <div class="section-title">
        <span>Comparative Information-Theoretic Diagnostics Suite</span>
    </div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Corpus / Language System</th>
                    <th>Tokens</th>
                    <th>H1 (bits)</th>
                    <th>H2 Cond</th>
                    <th>H2 Drop %</th>
                    <th>Hurst (DFA)</th>
                    <th>Gzip Ratio</th>
                    <th>Markov O2 Acc</th>
                    <th>Statistical Verdict</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[+] Real Master Scans Paleography Atlas saved to: {output_path.resolve()}")


def main():
    base_out = Path("./output/atlas_dataset")
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "voynich" / "png").mkdir(parents=True, exist_ok=True)
    (base_out / "voynich" / "svg").mkdir(parents=True, exist_ok=True)
    (base_out / "seraphinianus" / "png").mkdir(parents=True, exist_ok=True)
    (base_out / "seraphinianus" / "svg").mkdir(parents=True, exist_ok=True)

    normalizer = ColorIlluminationNormalizer()

    all_voynich_words = []
    all_voynich_strokes = []
    all_serafini_words = []
    all_serafini_strokes = []
    yield_reports = []

    # 1. Process Voynich Manuscript Scans (from Yale Beinecke high-res or PDF)
    # Master archival folios
    voynich_sources = [Path("data/scans/f001r.jpg"), Path("data/scans/f001v.jpg")]
    for vf in voynich_sources:
        if vf.exists():
            img = Image.open(vf)
            folio_id = vf.stem
            res = process_manuscript_page(folio_id, img, base_out, "voynich", normalizer, max_words_per_page=35)
            all_voynich_words.extend(res["words"])
            all_voynich_strokes.extend(res["strokes"])

            # Evaluate yield vs standard paleographic ground truth census
            yr = PaleographyYieldValidator.evaluate_yield(
                folio_id=folio_id,
                detected_lines=res["line_count"],
                detected_words=res["word_count"],
                extracted_strokes=len(res["strokes"])
            )
            yield_reports.append(yr)
            if yr["has_reference"]:
                print(f"  [>] Ground Truth Benchmark: {yr['detected_lines']}/{yr['expected_lines']} lines ({yr['line_yield_pct']}%), Scribe: {yr['scribe_hand']} ({yr['currier_language']})")

    # 2. Process Codex Seraphinianus Scans (from PDF)
    serafini_pdf = Path("data/scans/seraphinianus/Codex Seraphinianus.pdf")
    if serafini_pdf.exists():
        loader_s = PDFScanLoader(serafini_pdf)
        # Process multiple text-dense pages across chapters
        for page_idx in [15, 20, 25, 30]:
            try:
                page_img = loader_s.get_page_image(page_idx, target_min_dim=1500)
                res_s = process_manuscript_page(f"serafini_p{page_idx:03d}", page_img, base_out, "seraphinianus", normalizer, max_words_per_page=20)
                all_serafini_words.extend(res_s["words"])
                all_serafini_strokes.extend(res_s["strokes"])
            except Exception as e:
                print(f"[-] Error processing Seraphinianus page {page_idx}: {e}")

    # 3. Unsupervised Alphabet Induction
    if all_voynich_strokes:
        clusterer, labels, tokenizer = cluster_and_induce_alphabet(all_voynich_strokes, n_clusters=25)
        print(f"[+] Discovered {clusterer.get_cluster_count()} canonical glyph clusters across real manuscript folios.")

    # 4. Comparative Information-Theoretic Diagnostics
    from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
    from voynich_ductus.generators.seraphinianus import SeraphinianusEngine
    from voynich_ductus.generators.baselines import BaselineGenerator

    suite = BenchmarkSuite()
    diag_corpora = {
        "Latin Herbal (15th c. Natural Language)": BaselineGenerator.get_natural_latin_sample(1200),
        "Voynich Manuscript (Timm Self-Citation Model)": TimmSelfCitationGenerator(seed=42).generate(1200),
        "Codex Seraphinianus (Asemic Cursive Model)": SeraphinianusEngine.generate_serafini_text_tokens(1200),
        "Uniform Random Noise (Memoryless)": BaselineGenerator.get_uniform_random_gibberish(1200),
    }
    bench_results = suite.compare_corpora(diag_corpora)

    # 5. Generate Grand Visual Atlas
    atlas_path = base_out / "grand_paleography_atlas.html"
    generate_grand_atlas_html(all_voynich_words, all_serafini_words, bench_results, yield_reports, atlas_path)


if __name__ == "__main__":
    main()

