"""
Shared Multi-Model Page Enrichment & Gentle Multicore Worker Utilities.
Provides a single source of truth for running Decompositions A, B, and C
with physical MetricScaleCalibration (mm/0.1 mm), consensus ductus reuse,
and OS-level non-blocking gentle CPU throttling for local/cloud batch runs.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.embeddings.word_decomposition import (
    HybridConsensusWordDecomposition,
    NeuralWordDecomposition,
    cross_evaluate_options_consensus,
)
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.ingestion.spatial_partitioner import SpatialWordPartitioner
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.vectorizer.export_format import VectorExporter


def configure_gentle_worker_environment(low_priority: bool = True) -> None:
    """
    Prevents BLAS/OpenCV thread oversubscription inside multi-process workers
    and optionally lowers OS scheduling priority so parallel runs remain
    non-blocking for other concurrent workloads on the host machine.
    """
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[env_var] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    if not low_priority:
        return

    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass


def tokenize_eva_word(eva_word: str) -> List[str]:
    """Splits an EVA transliteration word into canonical paleographic graphemes."""
    s = re.sub(r"[^a-zA-Z]", "", (eva_word or "").lower())
    if not s:
        return []
    pattern = re.compile(r"(cth|ckh|cph|cfh|ch|sh|iii|ii|ee|[a-z])")
    return [m.group(0) for m in pattern.finditer(s)]


def realign_voynich_word_eva_tokens(words: List[Dict[str, Any]], glyphs: List[Dict[str, Any]]) -> None:
    """
    Aligns constituent glyphs of each word with its EVA ground-truth tokens
    using strictly monotonic left-to-right sequence mapping.
    """
    word_map = {w["word_id"]: w for w in words if "word_id" in w}
    by_word: Dict[str, List[Dict[str, Any]]] = {}
    for g in glyphs:
        wid = g.get("word_id", "")
        by_word.setdefault(wid, []).append(g)

    for wid, w_glyphs in by_word.items():
        w_glyphs.sort(key=lambda item: item.get("bbox", [0, 0, 0, 0])[1])
        w_obj = word_map.get(wid, {})
        eva_text = w_obj.get("eva_text") or (w_glyphs[0].get("ref_eva_text", "") if w_glyphs else "")
        tokens = tokenize_eva_word(eva_text)
        if not tokens or not w_glyphs:
            continue
        n_g, k_t = len(w_glyphs), len(tokens)
        for i, g in enumerate(w_glyphs):
            k = min(k_t - 1, int(i * k_t / n_g))
            g["eva_char"] = tokens[k]


def _find_matching_option_a_glyph(
    bbox: Tuple[int, int, int, int],
    opt_a_glyphs: List[Dict[str, Any]],
    tol_px: int = 1
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Finds the best overlapping Option A glyph in the same word.
    Returns (best_glyph, is_exact_match) where is_exact_match indicates that
    the bounding box matches within `tol_px` on all 4 sides, allowing direct
    reuse of the already-computed 40° quill ductus and polygon.
    """
    gy0, gx0, gy1, gx1 = bbox
    best_overlap_g = None
    best_inter = 0
    for qa in opt_a_glyphs:
        qb = qa.get("bbox", [0, 0, 0, 0])
        if (
            abs(gy0 - qb[0]) <= tol_px
            and abs(gx0 - qb[1]) <= tol_px
            and abs(gy1 - qb[2]) <= tol_px
            and abs(gx1 - qb[3]) <= tol_px
        ):
            return qa, True
        inter_x = max(0, min(gx1, qb[3]) - max(gx0, qb[1]))
        min_w = max(1, min(gx1 - gx0, qb[3] - qb[1]))
        if inter_x / min_w > 0.50 and inter_x > best_inter:
            best_inter = inter_x
            best_overlap_g = qa
    return best_overlap_g, False


def _vectorize_glyph_mask(
    g_mask: np.ndarray,
    gy0: int,
    gx0: int,
    gh: int,
    gw: int,
    is_serafini: bool,
    scale_calib: MetricScaleCalibration,
    tracker: CalligraphicVectorizer,
    segmenter: GlyphSegmenter,
) -> Tuple[List[Dict[str, Any]], str, List[List[int]]]:
    """Extracts 40° quill ductus strokes, SVG, and tight ink polygon for a glyph mask."""
    max_strokes = 1 if is_serafini else 4
    strokes = tracker.extract_calligraphic_ductus(
        g_mask, single_stroke_mode=is_serafini, max_strokes=max_strokes, scale_calib=scale_calib
    )[:max_strokes]

    if not strokes:
        strokes = [{
            "stroke_id": "cal_s00",
            "order_index": 0,
            "points": [(float(gh * 0.2), float(gw * 0.5), 2.5), (float(gh * 0.8), float(gw * 0.5), 2.5)],
            "length": 2.0,
            "start": (float(gh * 0.2), float(gw * 0.5)),
            "end": (float(gh * 0.8), float(gw * 0.5)),
            "nib_angle_deg": 40.0,
            "mean_width": 2.5
        }]

    for s in strokes:
        s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]

    svg = VectorExporter.to_svg(strokes, max(10, gw), max(10, gh))
    poly: List[List[int]] = [[gx0, gy0], [gx0 + gw, gy0], [gx0 + gw, gy0 + gh], [gx0, gy0 + gh]]
    if gh >= 4 and gw >= 4:
        try:
            computed_poly = segmenter.compute_ink_polygon(g_mask, offset=(gy0, gx0), scale_calib=scale_calib)
            if computed_poly:
                poly = computed_poly
        except Exception:
            pass
    return strokes, svg, poly


def enrich_page_multimodel(
    page_data: Dict[str, Any],
    img_bgr: np.ndarray,
    manuscript_type: str = "voynich",
    restore_rosetta_boxes: bool = True,
    recompute_option_a: bool = True,
) -> Tuple[int, int, int]:
    """
    Enriches `page_data` in-place with line-aware disjoint word partitions,
    metric-calibrated word polygons, and all 3 glyph decomposition models (A, B, C).
    Reuses Option A's vectorized ductus whenever B or C matches Option A's box
    within 1 px (avoiding redundant ridge-tracking on consensus glyphs).
    """
    is_serafini = (manuscript_type.lower() == "seraphinianus")
    H_img, W_img = img_bgr.shape[:2]
    page_id = page_data.get("page_id", "")
    raw_words = page_data.get("words", [])
    if not raw_words:
        return 0, 0, 0

    scale_calib = MetricScaleCalibration(
        manuscript_type="seraphinianus" if is_serafini else "voynich",
        image_height_px=H_img,
        image_width_px=W_img,
    )
    binarizer = Binarizer(method="sauvola")
    segmenter = GlyphSegmenter(scale_calib=scale_calib)
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)
    neural_decomposer = NeuralWordDecomposition()
    hybrid_decomposer = HybridConsensusWordDecomposition()

    if not is_serafini and restore_rosetta_boxes:
        try:
            from PIL import Image
            from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader
            rosetta_loader = VoynicheseRosettaLoader()
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            r_words = rosetta_loader.extract_aligned_words(page_id, scan_image=pil_img, prefer_js=True)
            valid_r_words = [rw for rw in r_words if rw.bbox_scan is not None]
            if len(valid_r_words) == len(raw_words):
                line_clusters: List[List[Any]] = []
                curr_cluster: List[Any] = []
                for rw in valid_r_words:
                    if not curr_cluster:
                        curr_cluster.append(rw)
                    else:
                        prev_rw = curr_cluster[-1]
                        if rw.x_xml < prev_rw.x_xml - 35 or (rw.y_xml - prev_rw.y_xml) > 25:
                            line_clusters.append(curr_cluster)
                            curr_cluster = [rw]
                        else:
                            curr_cluster.append(rw)
                if curr_cluster:
                    line_clusters.append(curr_cluster)
                for lc in line_clusters:
                    lc.sort(key=lambda w: w.bbox_scan[0])
                sorted_rw = [rw for lc in line_clusters for rw in lc]
                for w_dict, rw in zip(raw_words, sorted_rw):
                    rx0, ry0, rx1, ry1 = [int(v) for v in rw.bbox_scan]
                    rx0, ry0 = max(0, min(W_img - 1, rx0)), max(0, min(H_img - 1, ry0))
                    rx1, ry1 = max(rx0 + 1, min(W_img, rx1)), max(ry0 + 1, min(H_img, ry1))
                    w_dict["bbox"] = [ry0, rx0, ry1, rx1]
        except Exception:
            pass

    words = SpatialWordPartitioner.resolve_disjoint_word_partitions(raw_words, (H_img, W_img))
    page_data["words"] = words
    page_data["image_width"] = int(W_img)
    page_data["image_height"] = int(H_img)

    old_by_word: Dict[str, List[Dict[str, Any]]] = {}
    for g in page_data.get("glyphs", []):
        old_by_word.setdefault(g.get("word_id", ""), []).append(g)

    word_patches = []
    for w in words[:35]:
        wb = w.get("bbox", [])
        if len(wb) == 4:
            y0, x0, y1, x1 = [int(v) for v in wb]
            if y1 > y0 + 4 and x1 > x0 + 4:
                word_patches.append(img_bgr[y0:y1, x0:x1])
    if word_patches:
        neural_decomposer.fit_on_word_corpus(word_patches)

    quill_glyphs: List[Dict[str, Any]] = []
    neural_glyphs: List[Dict[str, Any]] = []
    hybrid_glyphs: List[Dict[str, Any]] = []

    for w in words:
        w_bbox = w.get("bbox")
        if not w_bbox or len(w_bbox) != 4:
            continue
        wy0, wx0, wy1, wx1 = [int(v) for v in w_bbox]
        wy0, wx0 = max(0, wy0), max(0, wx0)
        wy1, wx1 = min(H_img, wy1), min(W_img, wx1)
        if wy1 <= wy0 + 2 or wx1 <= wx0 + 2:
            continue

        w_patch = img_bgr[wy0:wy1, wx0:wx1]
        w_patch_rgb = cv2.cvtColor(w_patch, cv2.COLOR_BGR2RGB)
        try:
            if is_serafini:
                w_mask = binarizer.binarize(cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)).astype(bool)
            else:
                w_mask = segmenter.binarize_word_crop_hysteresis(w_patch_rgb).astype(bool)
            if np.sum(w_mask) < 12:
                gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
                w_mask = (gray_w < 128).astype(bool)
        except Exception:
            gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
            w_mask = (gray_w < 128).astype(bool)

        w_poly = segmenter.compute_ink_polygon(w_mask, offset=(wy0, wx0), scale_calib=scale_calib)
        if w_poly:
            w["polygon"] = w_poly

        line_id = w.get("line_id", "L00")
        word_id = w.get("word_id", f"{line_id}_W00")
        eva_text = w.get("eva_text", "")
        old_w_glyphs = old_by_word.get(word_id, [])

        # Option A: Topological & Valley (40° Quill)
        opt_a_local: List[Dict[str, Any]] = []
        opt_a_records: List[Dict[str, Any]] = []
        if recompute_option_a or not old_w_glyphs:
            assembled = segmenter.assemble_word_into_glyphs(w_mask, expected_glyph_count=None, scale_calib=scale_calib)
            a_idx = 0
            for comp in assembled:
                g_mask = comp["mask"].astype(bool)
                by0, bx0, by1, bx1 = comp["bbox_local"]
                gy0, gx0 = wy0 + int(by0), wx0 + int(bx0)
                gy1, gx1 = wy0 + int(by1), wx0 + int(bx1)
                gh, gw = g_mask.shape[:2]
                area = int(np.sum(g_mask))
                h_mm = scale_calib.px_to_mm(gh)
                w_mm = scale_calib.px_to_mm(gw)
                area_mm2 = scale_calib.area_px2_to_mm2(area)
                if area < scale_calib.min_glyph_area_px or (h_mm < 0.65 and w_mm < 0.65):
                    continue

                c_type = "G01"
                if old_w_glyphs:
                    gcx = (gx0 + gx1) / 2.0
                    best_old = min(
                        old_w_glyphs,
                        key=lambda og: abs((og.get("bbox", [0, gcx, 0, gcx])[1] + og.get("bbox", [0, gcx, 0, gcx])[3]) / 2.0 - gcx),
                    )
                    c_type = best_old.get("canonical_type", "G01")

                a_strokes, svg, poly = _vectorize_glyph_mask(
                    g_mask, gy0, gx0, gh, gw, is_serafini, scale_calib, tracker, segmenter
                )
                g_id = f"{page_id}_{word_id}_G{a_idx:02d}" if line_id in word_id else f"{page_id}_{line_id}_{word_id}_G{a_idx:02d}"
                g_rec = {
                    "glyph_id": g_id,
                    "page_id": page_id,
                    "line_id": line_id,
                    "word_id": word_id,
                    "word_bbox": [wy0, wx0, wy1, wx1],
                    "glyph_index_in_word": a_idx,
                    "bbox": [gy0, gx0, gy1, gx1],
                    "polygon": poly,
                    "height": int(gh),
                    "width": int(gw),
                    "height_mm": round(h_mm, 2),
                    "width_mm": round(w_mm, 2),
                    "height_dmm": round(h_mm * 10.0, 1),
                    "width_dmm": round(w_mm * 10.0, 1),
                    "area": area,
                    "area_mm2": round(area_mm2, 2),
                    "fill_factor": round(float(area / max(1, gh * gw)), 3),
                    "stroke_count": len(a_strokes),
                    "strokes": a_strokes,
                    "ductus_svg": svg,
                    "svg_content": svg,
                    "euler_stroke_count": len(a_strokes),
                    "euler_svg_content": svg,
                    "png_rel": "",
                    "svg_rel": "",
                    "crop_uri": "",
                    "canonical_type": c_type,
                    "eva_char": "",
                    "ref_eva_text": eva_text,
                    "segmentation_mode": "heuristic_calligraphic",
                    "diacritics": comp.get("diacritics", []),
                    "is_dot": comp.get("is_dot", False),
                }
                GlyphCatalogue.count_glyph_loops(g_rec)
                quill_glyphs.append(g_rec)
                opt_a_records.append(g_rec)
                opt_a_local.append({"bbox_local": (int(by0), int(bx0), int(by1), int(bx1)), "quill_glyph": g_rec})
                a_idx += 1
        else:
            for og in old_w_glyphs:
                gb = og.get("bbox", [wy0, wx0, wy1, wx1])
                opt_a_records.append(og)
                quill_glyphs.append(og)
                opt_a_local.append({
                    "bbox_local": (max(0, gb[0] - wy0), max(0, gb[1] - wx0), min(wy1 - wy0, gb[2] - wy0), min(wx1 - wx0, gb[3] - wx0)),
                    "quill_glyph": og,
                })

        # Option B: Spatial PCA Word Decomposition
        neural_comps = neural_decomposer.decompose_word(w_mask, scale_calib=scale_calib)
        n_idx = 0
        for comp in neural_comps:
            ng_mask = comp["mask"].astype(bool)
            by0, bx0, by1, bx1 = comp["bbox_local"]
            gy0, gx0 = wy0 + int(by0), wx0 + int(bx0)
            gy1, gx1 = wy0 + int(by1), wx0 + int(bx1)
            gh, gw = ng_mask.shape[:2]
            area = int(np.sum(ng_mask))
            h_mm = scale_calib.px_to_mm(gh)
            w_mm = scale_calib.px_to_mm(gw)
            area_mm2 = scale_calib.area_px2_to_mm2(area)
            if area < scale_calib.min_glyph_area_px or (h_mm < 0.65 and w_mm < 0.65):
                continue

            matched_g, is_exact = _find_matching_option_a_glyph((gy0, gx0, gy1, gx1), opt_a_records)
            c_type = matched_g.get("canonical_type", "G01") if matched_g else "G01"
            if is_exact and matched_g and matched_g.get("strokes"):
                n_strokes = matched_g["strokes"]
                svg = matched_g.get("svg_content", matched_g.get("ductus_svg", ""))
                poly = matched_g.get("polygon", [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]])
            else:
                n_strokes, svg, poly = _vectorize_glyph_mask(
                    ng_mask, gy0, gx0, gh, gw, is_serafini, scale_calib, tracker, segmenter
                )

            ng_rec = {
                "glyph_id": f"{page_id}_{word_id}_N{n_idx:02d}" if line_id in word_id else f"{page_id}_{line_id}_{word_id}_N{n_idx:02d}",
                "page_id": page_id,
                "line_id": line_id,
                "word_id": word_id,
                "word_bbox": [wy0, wx0, wy1, wx1],
                "glyph_index_in_word": n_idx,
                "bbox": [gy0, gx0, gy1, gx1],
                "polygon": poly,
                "height": int(gh),
                "width": int(gw),
                "height_mm": round(h_mm, 2),
                "width_mm": round(w_mm, 2),
                "height_dmm": round(h_mm * 10.0, 1),
                "width_dmm": round(w_mm * 10.0, 1),
                "area": area,
                "area_mm2": round(area_mm2, 2),
                "stroke_count": min(4, len(n_strokes)),
                "strokes": n_strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "euler_stroke_count": len(n_strokes),
                "euler_svg_content": svg,
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": "",
                "canonical_type": c_type,
                "segmentation_mode": "neural_word_pca",
                "neural_basis_id": comp.get("neural_basis_id", "PHI_00"),
            }
            GlyphCatalogue.count_glyph_loops(ng_rec)
            neural_glyphs.append(ng_rec)
            n_idx += 1

        # Option C: Hybrid Neuro-Kinematic Consensus
        hybrid_comps = hybrid_decomposer.decompose_word_hybrid(
            w_mask, opt_a_local, neural_comps, scale_calib=scale_calib
        )
        h_idx = 0
        for comp in hybrid_comps:
            hg_mask = comp["mask"].astype(bool)
            by0, bx0, by1, bx1 = comp["bbox_local"]
            gy0, gx0 = wy0 + int(by0), wx0 + int(bx0)
            gy1, gx1 = wy0 + int(by1), wx0 + int(bx1)
            gh, gw = hg_mask.shape[:2]
            area = int(np.sum(hg_mask))
            h_mm = scale_calib.px_to_mm(gh)
            w_mm = scale_calib.px_to_mm(gw)
            area_mm2 = scale_calib.area_px2_to_mm2(area)
            if area < scale_calib.min_glyph_area_px or (h_mm < 0.65 and w_mm < 0.65):
                continue

            matched_g, is_exact = _find_matching_option_a_glyph((gy0, gx0, gy1, gx1), opt_a_records)
            c_type = matched_g.get("canonical_type", "G01") if matched_g else "G01"
            if is_exact and matched_g and matched_g.get("strokes"):
                h_strokes = matched_g["strokes"]
                svg = matched_g.get("svg_content", matched_g.get("ductus_svg", ""))
                poly = matched_g.get("polygon", [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]])
            else:
                h_strokes, svg, poly = _vectorize_glyph_mask(
                    hg_mask, gy0, gx0, gh, gw, is_serafini, scale_calib, tracker, segmenter
                )

            hg_rec = {
                "glyph_id": f"{page_id}_{word_id}_H{h_idx:02d}" if line_id in word_id else f"{page_id}_{line_id}_{word_id}_H{h_idx:02d}",
                "page_id": page_id,
                "line_id": line_id,
                "word_id": word_id,
                "word_bbox": [wy0, wx0, wy1, wx1],
                "glyph_index_in_word": h_idx,
                "bbox": [gy0, gx0, gy1, gx1],
                "polygon": poly,
                "height": int(gh),
                "width": int(gw),
                "height_mm": round(h_mm, 2),
                "width_mm": round(w_mm, 2),
                "height_dmm": round(h_mm * 10.0, 1),
                "width_dmm": round(w_mm * 10.0, 1),
                "area": area,
                "area_mm2": round(area_mm2, 2),
                "stroke_count": min(4, len(h_strokes)),
                "strokes": h_strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "euler_stroke_count": len(h_strokes),
                "euler_svg_content": svg,
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": "",
                "canonical_type": c_type,
                "segmentation_mode": "hybrid_consensus",
                "hybrid_basis_id": comp.get("hybrid_basis_id", "HYB_00"),
            }
            GlyphCatalogue.count_glyph_loops(hg_rec)
            hybrid_glyphs.append(hg_rec)
            h_idx += 1

    cross_evaluate_options_consensus(quill_glyphs, neural_glyphs, hybrid_glyphs)
    if not is_serafini:
        realign_voynich_word_eva_tokens(words, quill_glyphs)
        realign_voynich_word_eva_tokens(words, neural_glyphs)
        realign_voynich_word_eva_tokens(words, hybrid_glyphs)

    page_data["glyphs"] = quill_glyphs
    page_data["glyphs_neural_pca"] = neural_glyphs
    page_data["glyphs_hybrid_consensus"] = hybrid_glyphs
    page_data["glyph_count"] = len(quill_glyphs)
    page_data["glyph_count_neural_pca"] = len(neural_glyphs)
    page_data["glyph_count_hybrid_consensus"] = len(hybrid_glyphs)
    page_data["segmentation_modes"] = ["heuristic_calligraphic", "neural_word_pca", "hybrid_consensus"]

    return len(quill_glyphs), len(neural_glyphs), len(hybrid_glyphs)
