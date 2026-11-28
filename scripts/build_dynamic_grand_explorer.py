"""
Dynamic Grand Glyph Explorer Generator for VoynichDuctus.
Uses the official generate_explorer_html function from scripts/generate_full_glyph_corpus_and_explorer.py
to generate a 100% syntactically valid HTML/CSS/JS application with dark theme and dynamic page loading.
"""

import json
from pathlib import Path
import re
import sys

# Ensure src and scripts are on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "scripts"))

from generate_full_glyph_corpus_and_explorer import generate_explorer_html


def _make_crop_uri_from_page(
    page_id: str,
    bbox: list,
    pages_img_dir: Optional[Path],
    img_cache: dict,
    pad: int = 4,
    polygon: Optional[list] = None,
    hole_polygons: Optional[list] = None,
) -> str:
    """Slices a compact JPEG data-URI scan thumbnail from a cached page image (punching out exterior & interior loop holes when polygons are provided)."""
    if not pages_img_dir or not page_id or not bbox or len(bbox) < 4:
        return ""
    if page_id not in img_cache:
        img_file = pages_img_dir / f"{page_id}.jpg"
        if not img_file.exists():
            img_cache[page_id] = None
        else:
            try:
                from PIL import Image
                with Image.open(img_file) as im:
                    img_cache[page_id] = im.convert("RGB")
            except Exception:
                img_cache[page_id] = None
    im_page = img_cache.get(page_id)
    if im_page is None:
        return ""
    try:
        import base64
        import io
        from PIL import Image, ImageDraw
        y0, x0, y1, x1 = [int(v) for v in bbox[:4]]
        pw, ph = im_page.size
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(pw, x1 + pad), min(ph, y1 + pad)
        if cx1 <= cx0 or cy1 <= cy0:
            return ""
        crop = im_page.crop((cx0, cy0, cx1, cy1)).copy()
        if polygon and len(polygon) >= 3:
            mask_im = Image.new("L", crop.size, 0)
            draw = ImageDraw.Draw(mask_im)
            outer_pts = [(int(pt[0]) - cx0, int(pt[1]) - cy0) for pt in polygon]
            draw.polygon(outer_pts, fill=255)
            if hole_polygons:
                for hp in hole_polygons:
                    if hp and len(hp) >= 3:
                        inner_pts = [(int(pt[0]) - cx0, int(pt[1]) - cy0) for pt in hp]
                        draw.polygon(inner_pts, fill=0)
            bg_black = Image.new("RGB", crop.size, (0, 0, 0))
            crop = Image.composite(crop, bg_black, mask_im)
        if max(crop.size) > 96:
            crop.thumbnail((96, 96))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _clean_alphabet_list(arch_list: list, pages_img_dir: Optional[Path] = None, img_cache: Optional[dict] = None) -> list:
    """Cleans an induced alphabet entry list for compact JSON embedding in the HTML explorer."""
    if img_cache is None:
        img_cache = {}
    cleaned_alphabet = []
    for arch in arch_list:
        all_inst = arch.get("all_instances", [])
        source_insts = all_inst if all_inst else arch.get("sample_instances", [])
        samples = []
        for inst in source_insts[:36]:
            if isinstance(inst, dict):
                inst_page = inst.get("page_id", "")
                inst_bbox = inst.get("bbox", [0, 0, 0, 0])
                inst_poly = inst.get("polygon", [])
                inst_holes = inst.get("hole_polygons", [])
                c_uri = _make_crop_uri_from_page(
                    inst_page, inst_bbox, pages_img_dir, img_cache, polygon=inst_poly, hole_polygons=inst_holes
                ) or inst.get("crop_uri", "")
                samples.append({
                    "glyph_id": inst.get("glyph_id", ""),
                    "page_id": inst_page,
                    "line_id": inst.get("line_id", ""),
                    "word_id": inst.get("word_id", ""),
                    "bbox": inst_bbox,
                    "polygon": inst_poly,
                    "hole_polygons": inst_holes,
                    "width": inst.get("width", 0),
                    "height": inst.get("height", 0),
                    "height_mm": inst.get("height_mm", round(inst.get("height", 0) * 0.125, 2)),
                    "width_mm": inst.get("width_mm", round(inst.get("width", 0) * 0.125, 2)),
                    "height_dmm": inst.get("height_dmm", round(inst.get("height", 0) * 1.25, 1)),
                    "width_dmm": inst.get("width_dmm", round(inst.get("width", 0) * 1.25, 1)),
                    "area_mm2": inst.get("area_mm2", 0.0),
                    "stroke_count": inst.get("stroke_count", 1),
                    "loop_count": inst.get("loop_count", arch.get("loop_count", 0)),
                    "topology_group": inst.get("topology_group", arch.get("topology_group", "")),
                    "crop_uri": c_uri,
                    "svg_content": inst.get("svg_content", inst.get("ductus_svg", "")),
                    "euler_svg_content": inst.get("euler_svg_content", ""),
                    "euler_stroke_count": inst.get("euler_stroke_count", len(inst.get("euler_strokes", []))),
                    "eva_char": inst.get("eva_char", "")
                })
            else:
                samples.append({
                    "glyph_id": str(inst),
                    "page_id": str(inst).split("_")[0] if "_" in str(inst) else "",
                    "bbox": [0, 0, 0, 0],
                    "width": arch.get("mean_width", 20),
                    "height": arch.get("mean_height", 25),
                    "stroke_count": arch.get("mean_strokes", 1),
                    "loop_count": arch.get("loop_count", 0),
                    "topology_group": arch.get("topology_group", ""),
                    "crop_uri": "",
                    "svg_content": "",
                    "eva_char": ""
                })

        ex_page = arch.get("exemplar_page_id", "") or (samples[0].get("page_id", "") if samples else "")
        ex_bbox = arch.get("exemplar_bbox", [0, 0, 0, 0])
        if (not ex_bbox or ex_bbox == [0, 0, 0, 0]) and samples:
            ex_bbox = samples[0].get("bbox", [0, 0, 0, 0])
        ex_poly = samples[0].get("polygon", []) if samples else []
        ex_holes = samples[0].get("hole_polygons", []) if samples else []
        ex_crop_uri = (
            _make_crop_uri_from_page(ex_page, ex_bbox, pages_img_dir, img_cache, polygon=ex_poly, hole_polygons=ex_holes)
            or (samples[0].get("crop_uri", "") if samples else "")
            or arch.get("exemplar_crop_uri", "")
        )

        cleaned_alphabet.append({
            "type_id": arch.get("type_id"),
            "rank": arch.get("rank"),
            "cluster_index": arch.get("cluster_index"),
            "ductus_engine": arch.get("ductus_engine", "quill_40"),
            "frequency": arch.get("frequency"),
            "folio_frequency": arch.get("folio_frequency", len(samples)),
            "percentage": arch.get("percentage"),
            "indexed_pages": arch.get("indexed_pages", []),
            "page_count": arch.get("page_count", len(arch.get("indexed_pages", []))),
            "topology_group": arch.get("topology_group", ""),
            "loop_count": arch.get("loop_count", 0),
            "exemplar_id": arch.get("exemplar_id"),
            "exemplar_page_id": ex_page,
            "exemplar_bbox": ex_bbox,
            "exemplar_crop_uri": ex_crop_uri,
            "exemplar_png": arch.get("exemplar_png", ""),
            "exemplar_svg": arch.get("exemplar_svg", ""),
            "exemplar_svg_content": arch.get("exemplar_svg_content", ""),
            "exemplar_euler_svg_content": arch.get("exemplar_euler_svg_content", ""),
            "mean_strokes": arch.get("mean_strokes"),
            "mean_height": arch.get("mean_height"),
            "mean_width": arch.get("mean_width"),
            "mean_height_mm": arch.get("mean_height_mm"),
            "mean_width_mm": arch.get("mean_width_mm"),
            "mean_height_dmm": arch.get("mean_height_dmm"),
            "mean_width_dmm": arch.get("mean_width_dmm"),
            "corpus_match": arch.get("corpus_match"),
            "corpus_matches": arch.get("corpus_matches"),
            "total_instances_count": arch.get("total_instances_count", arch.get("frequency", 0)),
            "sample_instances": samples
        })
    return cleaned_alphabet


def clean_catalogue_for_embed(cat_dict: dict, pages_img_dir: Optional[Path] = None, img_cache: Optional[dict] = None) -> dict:
    """Cleans the unified Canonical Alphabet (each card with Scan, Quill 40° Bézier, Euler-Bernoulli, Symbol) for Explorer HTML."""
    if img_cache is None:
        img_cache = {}
    cleaned_a = _clean_alphabet_list(cat_dict.get("alphabet", []), pages_img_dir=pages_img_dir, img_cache=img_cache)
    return {
        "total_glyphs": cat_dict.get("total_glyphs", 0),
        "indexed_pages_count": cat_dict.get("indexed_pages_count", 1),
        "canonical_alphabet_size": len(cleaned_a),
        "alphabet": cleaned_a,
    }


def rebuild_dual_catalogue_from_pages(
    pages_dir: Path,
    out_cat_path: Path,
    corpus_type: str,
    target_size: Optional[int] = None,
    min_px_per_mm: float = 0.0,
    preferred_stems: list = None,
) -> dict:
    """
    Rebuilds the unified Canonical Alphabet via natural Kneedle-Fisher emergence
    from all processed folio JSONs meeting physical scan density `min_px_per_mm`
    (or `preferred_stems`) and writes back `canonical_type`.
    """
    from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    folio_h_mm = MetricScaleCalibration.DEFAULT_FOLIO_HEIGHTS_MM.get(corpus_type.lower(), 225.0)
    all_json = sorted(pages_dir.glob("*.json"))
    selected_files = []
    for jf in all_json:
        if preferred_stems and jf.stem not in preferred_stems:
            continue
        stub = parse_page_stub_fast(jf, corpus_type)
        ppm = float(stub.get("image_height", 0)) / folio_h_mm
        if min_px_per_mm > 0.0 and ppm < min_px_per_mm:
            continue
        if stub.get("glyph_count", 0) > 0:
            selected_files.append(jf)

    if not selected_files:
        if out_cat_path.exists():
            with open(out_cat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

    page_files_used = []
    pool = []
    calib_ref = None
    for jf in selected_files:
        with open(jf, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        gs = pdata.get("glyphs", [])
        if not gs:
            continue
        if calib_ref is None:
            calib_ref = MetricScaleCalibration(
                manuscript_type=corpus_type,
                image_width_px=pdata.get("image_width", 2400),
                image_height_px=pdata.get("image_height", 3200),
            )
        page_files_used.append(jf)
        # Keep lightweight feature-clustering records in memory so 596 full page JSONs never exceed ~450 MB RAM
        for g in gs:
            pool.append({
                "glyph_id": g.get("glyph_id", ""),
                "page_id": g.get("page_id", jf.stem),
                "line_id": g.get("line_id", ""),
                "word_id": g.get("word_id", ""),
                "bbox": g.get("bbox", [0, 0, 0, 0]),
                "width": g.get("width", 20),
                "height": g.get("height", 20),
                "height_mm": g.get("height_mm", 0.0),
                "width_mm": g.get("width_mm", 0.0),
                "height_dmm": g.get("height_dmm", 0.0),
                "width_dmm": g.get("width_dmm", 0.0),
                "area": g.get("area", 100),
                "area_mm2": g.get("area_mm2", 0.0),
                "fill_factor": g.get("fill_factor", 0.35),
                "loop_count": g.get("loop_count", -1),
                "stroke_count": g.get("stroke_count", 1),
                "euler_stroke_count": g.get("euler_stroke_count", len(g.get("euler_strokes", []))),
                "eva_char": g.get("eva_char", ""),
                "strokes": [{"points": s.get("points", [])} for s in g.get("strokes", [])],
            })
        del pdata

    if not pool:
        return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

    builder = GlyphCatalogue(target_alphabet_size=target_size)
    cat = builder.build_catalogue(
        pool,
        corpus_type=corpus_type,
        scale_calib=calib_ref,
        stroke_key="strokes",
        tag_field="canonical_type",
    )
    cat["indexed_pages_count"] = len(page_files_used)

    # Build fast lookup maps from induced pool and identify sample/exemplar IDs needing full SVG/polygon hydration
    assigned_map = {
        g["glyph_id"]: (
            g.get("canonical_type", "G01"),
            g.get("topology_group", ""),
            int(g.get("loop_count", 0)),
            bool(g.get("is_outlier", False)),
            g.get("outlier_bin", "INLIER_CONFORMANT"),
            g.get("medoid_distance", 0.0),
        )
        for g in pool
    }
    del pool

    needed_rich_ids = set()
    for arch in cat.get("alphabet", []):
        if arch.get("exemplar_id"):
            needed_rich_ids.add(arch["exemplar_id"])
        for s_inst in arch.get("sample_instances", []):
            if isinstance(s_inst, dict) and s_inst.get("glyph_id"):
                needed_rich_ids.add(s_inst["glyph_id"])

    rich_lookup = {}
    # Stream one page JSON at a time: persist canonical_type back to disk & hydrate sample SVGs/polygons
    for jf in page_files_used:
        with open(jf, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        pdata.pop("glyphs_neural_pca", None)
        pdata.pop("glyphs_hybrid_consensus", None)
        pdata.pop("glyph_count_neural_pca", None)
        pdata.pop("glyph_count_hybrid_consensus", None)
        for g in pdata.get("glyphs", []):
            gid = g.get("glyph_id", "")
            if gid in assigned_map:
                ct, tg, lc, io, ob, md = assigned_map[gid]
                g["canonical_type"] = ct
                g["topology_group"] = tg
                g["loop_count"] = lc
                g["is_outlier"] = io
                g["outlier_bin"] = ob
                g["medoid_distance"] = md
            if gid in needed_rich_ids:
                rich_lookup[gid] = {
                    "polygon": g.get("polygon", []),
                    "hole_polygons": g.get("hole_polygons", []),
                    "svg_content": g.get("svg_content", g.get("ductus_svg", "")),
                    "euler_svg_content": g.get("euler_svg_content", ""),
                }
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(pdata, f, ensure_ascii=False)
        del pdata

    for arch in cat.get("alphabet", []):
        ex_id = arch.get("exemplar_id", "")
        if ex_id in rich_lookup:
            arch["exemplar_svg_content"] = rich_lookup[ex_id]["svg_content"]
            arch["exemplar_euler_svg_content"] = rich_lookup[ex_id]["euler_svg_content"]
        for s_inst in arch.get("sample_instances", []):
            if isinstance(s_inst, dict):
                sid = s_inst.get("glyph_id", "")
                if sid in rich_lookup:
                    s_inst["polygon"] = rich_lookup[sid]["polygon"]
                    s_inst["hole_polygons"] = rich_lookup[sid]["hole_polygons"]
                    s_inst["svg_content"] = rich_lookup[sid]["svg_content"]
                    s_inst["euler_svg_content"] = rich_lookup[sid]["euler_svg_content"]

    builder.export_catalogue_json(cat, out_cat_path)
    return cat


def compute_live_corpus_statistics(
    voynich_pages_dir: Path,
    serafini_pages_dir: Path,
    v_cat: dict,
    s_cat: dict,
    v_min_px_per_mm: float = 10.0,
    s_preferred_stems: list = None,
) -> dict:
    """
    Dynamically computes real corpus counts and information-theoretic diagnostics
    directly from all processed page JSONs and induced alphabets (zero hardcoded stats).
    """
    from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
    from voynich_ductus.generators.baselines import BaselineGenerator
    from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
    from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    def _collect_ms_stats(pages_dir: Path, subfolder: str, min_ppm: float = 0.0, stems: list = None) -> dict:
        folio_h_mm = MetricScaleCalibration.DEFAULT_FOLIO_HEIGHTS_MM.get(subfolder.lower(), 225.0)
        all_files = sorted(pages_dir.glob("*.json"))
        indexed_words = 0
        indexed_glyphs_a = 0
        processed_ids = []
        proc_words = 0
        proc_a = 0
        word_token_seqs = []

        for jf in all_files:
            stub = parse_page_stub_fast(jf, subfolder)
            indexed_words += stub.get("word_count", 0)
            indexed_glyphs_a += stub.get("glyph_count", 0)

            in_scope = True
            if stems and jf.stem not in stems:
                in_scope = False
            ppm = float(stub.get("image_height", 0)) / folio_h_mm
            if min_ppm > 0.0 and ppm < min_ppm:
                in_scope = False
            if in_scope and stub.get("glyph_count", 0) > 0:
                with open(jf, "r", encoding="utf-8") as f:
                    pd = json.load(f)
                processed_ids.append(pd.get("page_id", jf.stem))
                proc_words += len(pd.get("words", []))
                ga = pd.get("glyphs", [])
                proc_a += len(ga)

                by_w = {}
                for g in ga:
                    by_w.setdefault(g.get("word_id", ""), []).append(g)
                for w in pd.get("words", []):
                    wg = sorted(by_w.get(w.get("word_id", ""), []), key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
                    if wg:
                        word_token_seqs.append("-".join(g.get("canonical_type", "G01") for g in wg))

        return {
            "indexed_pages_total": len(all_files),
            "indexed_words_total": indexed_words,
            "indexed_glyphs_total": indexed_glyphs_a,
            "processed_pages_count": len(processed_ids),
            "processed_page_ids": processed_ids,
            "processed_words": proc_words,
            "processed_glyphs_a": proc_a,
            "word_token_seqs": word_token_seqs,
        }

    v_stats = _collect_ms_stats(voynich_pages_dir, "voynich", min_ppm=v_min_px_per_mm)
    s_stats = _collect_ms_stats(
        serafini_pages_dir,
        "seraphinianus",
        stems=s_preferred_stems,
    )

    suite = BenchmarkSuite()
    v_tokens = v_stats.pop("word_token_seqs") or ["G01-G02"]
    s_tokens = s_stats.pop("word_token_seqs") or ["G01-G02"]
    ctrl_n = max(120, len(v_tokens))

    latin_words = BaselineGenerator.get_natural_latin_sample(num_words=ctrl_n)
    timm_words = TimmSelfCitationGenerator(seed=42).generate(num_words=ctrl_n)
    rugg_words = RuggCardanGenerator(seed=42).generate(num_words=ctrl_n)
    random_words = BaselineGenerator.get_uniform_random_gibberish(num_words=ctrl_n)

    def _summarize_diag(name: str, seq: list, scope_note: str) -> dict:
        import math
        r = suite.evaluate_corpus(name, seq)
        ent = r.get("entropy", {})
        mem = r.get("memory", {})
        comp = r.get("compressibility", {})
        mkv = r.get("markov_automata", {})
        o1 = mkv.get("order_1", {}).get("top1_prediction_accuracy", 0.0)
        o2 = mkv.get("order_2", {}).get("top1_prediction_accuracy", 0.0)
        def _safe_round(val, ndigits=2):
            try:
                fv = float(val)
                return round(fv, ndigits) if math.isfinite(fv) else None
            except Exception:
                return None
        return {
            "name": name,
            "scope": scope_note,
            "token_count": len(seq),
            "h1": _safe_round(ent.get("h1_char_bits", 0.0)),
            "h2": _safe_round(ent.get("h2_cond_char_bits", 0.0)),
            "hurst": _safe_round(mem.get("hurst_exponent", 0.5)),
            "lzma_ratio": _safe_round(comp.get("lzma_ratio", comp.get("gzip_ratio", 0.0))),
            "markov_order": f"O1={o1:.0%} / O2={o2:.0%}",
        }

    diagnostics = [
        _summarize_diag(
            "Voynich (HD Ductus G-Tokens)",
            v_tokens,
            f"{v_stats['processed_pages_count']} HD folios ({v_stats['processed_words']} words)",
        ),
        _summarize_diag(
            "Codex Seraphinianus (Ribbon G-Tokens)",
            s_tokens,
            f"{s_stats['processed_pages_count']} text pages ({s_stats['processed_words']} words)",
        ),
        _summarize_diag("Natural Latin Herbal (Control)", latin_words, f"Seeded baseline ({ctrl_n} words)"),
        _summarize_diag("Timm Self-Citation (Baseline)", timm_words, f"Seeded baseline ({ctrl_n} words)"),
        _summarize_diag("Rugg Cardan Grille (Baseline)", rugg_words, f"Seeded baseline ({ctrl_n} words)"),
        _summarize_diag("Random Uniform Noise (Baseline)", random_words, f"Seeded baseline ({ctrl_n} words)"),
    ]

    v_stats["alphabet_a_size"] = v_cat.get("canonical_alphabet_size", 0)
    s_stats["alphabet_a_size"] = s_cat.get("canonical_alphabet_size", 0)

    from datetime import datetime
    return {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "voynich": v_stats,
        "seraphinianus": s_stats,
        "diagnostics": diagnostics,
    }


def parse_page_stub_fast(json_path: Path, subfolder: str) -> dict:
    """Fast header reading without parsing megabytes of internal stroke arrays."""
    stem = json_path.stem
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(4096)

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
        "image_width": int(m_w.group(1)) if m_w else 2400,
        "image_height": int(m_h.group(1)) if m_h else 3300,
        "line_count": int(m_lines.group(1)) if m_lines else 0,
        "word_count": int(m_words.group(1)) if m_words else 0,
        "glyph_count": int(m_glyphs.group(1)) if m_glyphs else 0,
        "yield_report": {},
    }


def sync_readme_live_stats(live_stats: dict, readme_path: Path) -> None:
    """Synchronizes the live corpus & diagnostics Markdown block inside README.md so zero numbers stay hardcoded."""
    if not readme_path.exists() or not live_stats:
        return
    txt = readme_path.read_text(encoding="utf-8")
    start_tag = "<!-- LIVE_CORPUS_STATS_START -->"
    end_tag = "<!-- LIVE_CORPUS_STATS_END -->"
    if start_tag not in txt or end_tag not in txt:
        return

    v = live_stats.get("voynich", {})
    s = live_stats.get("seraphinianus", {})
    diags = live_stats.get("diagnostics", [])
    v_ids = ", ".join(v.get("processed_page_ids", [])[:8]) or "—"
    s_ids = ", ".join(s.get("processed_page_ids", [])[:8]) or "—"

    lines = [
        start_tag,
        f"*Auto-synchronized from `data/processed/atlas_dataset/` on `{live_stats.get('computed_at', '')}` via `scripts/build_dynamic_grand_explorer.py`:*",
        "",
        "| Metric / Database Scope | Voynich Manuscript (Beinecke MS 408 · 225 mm) | Codex Seraphinianus (1981 · 350 mm) |",
        "| :--- | :--- | :--- |",
        f"| **Indexed Dataset Pages (Canvas Stubs)** | **{v.get('indexed_pages_total', 0):,}** folios ({v.get('indexed_words_total', 0):,} words, {v.get('indexed_glyphs_total', 0):,} glyphs) | **{s.get('indexed_pages_total', 0):,}** pages ({s.get('indexed_words_total', 0):,} words, {s.get('indexed_glyphs_total', 0):,} glyphs) |",
        f"| **Active HD Reprocessed Set** | **{v.get('processed_pages_count', 0)}** HD folios (`{v_ids}`) | **{s.get('processed_pages_count', 0)}** HD plates (`{s_ids}`) |",
        f"| **Processed Words & Glyphs (`GenerativeTypewriterDecomposer`)** | **{v.get('processed_words', 0):,}** words · **{v.get('processed_glyphs_a', 0):,}** isolated glyphs | **{s.get('processed_words', 0):,}** words · **{s.get('processed_glyphs_a', 0):,}** isolated glyphs |",
        f"| **Live Canonical Alphabet Size (100% Coverage)** | **{v.get('alphabet_a_size', 0)}** emergent archetypes | **{s.get('alphabet_a_size', 0)}** emergent archetypes |",
        "",
        "| Corpus / Seeded Control (`BenchmarkSuite`) | Evaluated Scope | $H_1$ (bits) | $H_2$ Cond. (bits) | Hurst $H$ (DFA) | LZMA Ratio | Markov Top-1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for d in diags:
        h1 = f"{d['h1']:.2f}" if d.get("h1") is not None else "—"
        h2 = f"{d['h2']:.2f}" if d.get("h2") is not None else "—"
        hu = f"{d['hurst']:.3f}" if d.get("hurst") is not None else "—"
        lz = f"{d['lzma_ratio']:.3f}" if d.get("lzma_ratio") is not None else "—"
        lines.append(
            f"| **{d.get('name', '')}** | {d.get('scope', '')} | {h1} | {h2} | {hu} | {lz} | {d.get('markov_order', '—')} |"
        )
    lines.append(end_tag)

    i0 = txt.index(start_tag)
    i1 = txt.index(end_tag) + len(end_tag)
    updated = txt[:i0] + "\n".join(lines) + txt[i1:]
    readme_path.write_text(updated, encoding="utf-8")


def main(skip_catalogue: bool = False):
    import os
    import sys
    if "--skip-catalogue" in sys.argv:
        skip_catalogue = True
    try:
        import torch
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
    except Exception:
        pass
    base_out = Path("data/processed/atlas_dataset")
    voynich_dir = base_out / "voynich"
    serafini_dir = base_out / "seraphinianus"

    if skip_catalogue and (voynich_dir / "voynich_alphabet_catalogue.json").exists():
        with open(voynich_dir / "voynich_alphabet_catalogue.json", "r", encoding="utf-8") as f:
            v_cat_raw = json.load(f)
    else:
        print("[*] Discovering Natural Emergent Canonical Alphabet from Voynich pages...")
        v_cat_raw = rebuild_dual_catalogue_from_pages(
            pages_dir=voynich_dir / "pages",
            out_cat_path=voynich_dir / "voynich_alphabet_catalogue.json",
            corpus_type="voynich",
            min_px_per_mm=0.0,
        )
    v_cat = clean_catalogue_for_embed(v_cat_raw, pages_img_dir=voynich_dir / "pages")
    print(
        f"  [+] Voynich Emergent Alphabet: {v_cat['canonical_alphabet_size']} archetypes "
        f"({v_cat['total_glyphs']} glyphs across {v_cat['indexed_pages_count']} folios)."
    )

    if skip_catalogue and (serafini_dir / "seraphinianus_alphabet_catalogue.json").exists():
        with open(serafini_dir / "seraphinianus_alphabet_catalogue.json", "r", encoding="utf-8") as f:
            s_cat_raw = json.load(f)
    else:
        print("[*] Discovering Natural Emergent Canonical Alphabet from Seraphinianus pages...")
        s_cat_raw = rebuild_dual_catalogue_from_pages(
            pages_dir=serafini_dir / "pages",
            out_cat_path=serafini_dir / "seraphinianus_alphabet_catalogue.json",
            corpus_type="seraphinianus",
            preferred_stems=None,
        )
    s_cat = clean_catalogue_for_embed(s_cat_raw, pages_img_dir=serafini_dir / "pages")
    print(
        f"  [+] Seraphinianus Canonical Alphabet: {s_cat['canonical_alphabet_size']} archetypes "
        f"({s_cat['total_glyphs']} glyphs across {s_cat['indexed_pages_count']} pages)."
    )

    # 1. Index Voynich Pages
    print("[*] Indexing Voynich pages...")
    from voynich_ductus.ingestion.codicology import natural_folio_sort_key, build_folio_codicology_registry

    v_json_files = sorted((voynich_dir / "pages").glob("*.json"), key=lambda p: natural_folio_sort_key(p.stem))
    v_pids = [jf.stem for jf in v_json_files]
    cod_reg = build_folio_codicology_registry(v_pids)
    v_pages_stubs = []

    for idx, jf in enumerate(v_json_files):
        stub = parse_page_stub_fast(jf, "voynich")
        c_info = cod_reg.get(stub["page_id"], {})
        stub["quire_id"] = c_info.get("quire", "")
        stub["section"] = c_info.get("section", "")
        stub["davis_scribe"] = c_info.get("davis_scribe", 1)
        stub["singulion_id"] = c_info.get("singulion_id", "")
        stub["conjoint_leaf"] = c_info.get("conjoint_leaf")
        stub["codex_index"] = c_info.get("codex_index", idx + 1)
        stub["ld_index"] = c_info.get("ld_index", idx + 1)
        stub["notes"] = c_info.get("notes", "")

        # Pre-load full content for f1r (page 0)
        if idx == 0:
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
            stub["glyph_count"] = len(stub["glyphs"])
            stub["yield_report"] = p_full.get("yield_report", {})
        v_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(v_pages_stubs)} Voynich folios.")

    # 2. Index Seraphinianus Pages
    print("[*] Indexing Codex Seraphinianus pages...")
    s_json_files = sorted((serafini_dir / "pages").glob("*.json"))
    s_pages_stubs = []

    for idx, jf in enumerate(s_json_files):
        stub = parse_page_stub_fast(jf, "seraphinianus")
        if jf.stem == "serafini_p030":
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
            stub["glyph_count"] = len(stub["glyphs"])
        s_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(s_pages_stubs)} Seraphinianus pages.")

    print("[*] Computing live corpus statistics & information-theoretic diagnostics...")
    live_stats = compute_live_corpus_statistics(
        voynich_pages_dir=voynich_dir / "pages",
        serafini_pages_dir=serafini_dir / "pages",
        v_cat=v_cat,
        s_cat=s_cat,
        v_min_px_per_mm=0.0,
        s_preferred_stems=None,
    )

    # 3. Call official generator (Python f-string correctly unescapes all CSS and JS braces!)
    out_html_path = base_out / "grand_glyph_explorer.html"
    print("[*] Building Grand Glyph Explorer HTML via native generator...")
    generate_explorer_html(
        voynich_pages_data=v_pages_stubs,
        voynich_catalogue=v_cat,
        serafini_pages_data=s_pages_stubs,
        serafini_catalogue=s_cat,
        output_path=out_html_path,
        live_stats=live_stats,
    )
    sync_readme_live_stats(live_stats, root_dir / "README.md")

    print(f"[+] Successfully generated: {out_html_path.resolve()}")
    print(f"  [+] File size: {out_html_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
