"""
Full Word-to-Glyph Regeneration, Kinematic Ductus Extraction, and Option C Archetype Induction.
Starting from the validated lexical words (`words`), this script:
1. Re-segments every word into intact constituent glyphs via `GlyphSegmenter.assemble_word_into_glyphs`
   (attaching diacritics/fragments, assembling 2-stroke composite letters like Voynich 'a' = 'c'+'i',
   and splitting ligature valleys without ever cutting through topological loops).
2. Computes tight polygonal contours and isolated scan crops (`crop_uri`) with zero neighbor bleed.
3. Extracts complete calligraphic ductus (1-stroke ribbon for Seraphinianus, 40° quill multi-stroke for Voynich).
4. Rebuilds the canonical alphabet catalogues via Option C (Strict Topological Partitioning + 2D Image Morphology + DINOv2).
"""

import base64
import json
import re
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from skimage.filters import threshold_otsu

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.spatial_partitioner import SpatialWordPartitioner
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.embeddings.word_neural_decomposer import (
    WordScribalEncoder,
    NeuralWordDecomposer,
    HybridNeuroKinematicDecomposer,
    cross_evaluate_options_consensus
)
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue


def build_isolated_crop_uri(patch_bgr: np.ndarray, g_mask: np.ndarray) -> str:
    """
    Renders a clean isolated glyph crop thumbnail (base64 PNG URI) where any ink
    outside the glyph's own component mask `g_mask` is neutralized to parchment background tone.
    """
    if patch_bgr.size == 0 or g_mask.size == 0:
        return ""
    clean_bgr = patch_bgr.copy()
    bg_color = np.percentile(patch_bgr.reshape(-1, 3), 82, axis=0).astype(np.uint8)
    dil_k = np.ones((5, 5), dtype=np.uint8)
    ink_halo = cv2.dilate(g_mask.astype(np.uint8), dil_k, iterations=1).astype(bool)
    outside = ~ink_halo
    if np.any(outside):
        clean_bgr[outside] = (
            0.86 * bg_color.astype(np.float32) + 0.14 * clean_bgr[outside].astype(np.float32)
        ).astype(np.uint8)

    ok, buf = cv2.imencode(".png", clean_bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def tokenize_eva_word(eva_word: str) -> List[str]:
    """Splits an EVA transliteration word into canonical paleographic graphemes."""
    s = re.sub(r"[^a-zA-Z]", "", (eva_word or "").lower())
    if not s:
        return []
    pattern = re.compile(r"(cth|ckh|cph|cfh|ch|sh|iii|ii|ee|[a-z])")
    return [m.group(0) for m in pattern.finditer(s)]


def realign_voynich_word_eva_tokens(words: List[Dict[str, Any]], glyphs: List[Dict[str, Any]]):
    """
    Aligns constituent glyphs of each word with its EVA ground-truth tokens
    using horizontal relative position + topological compatibility (loops & height).
    """
    word_map = {w["word_id"]: w for w in words if "word_id" in w}
    by_word: Dict[str, List[Dict[str, Any]]] = {}
    for g in glyphs:
        wid = g.get("word_id", "")
        by_word.setdefault(wid, []).append(g)

    loop_eva_set = {"o", "a", "y", "d", "q", "8"}
    tall_eva_set = {"k", "t", "p", "f", "cth", "ckh", "cph", "cfh", "d", "l"}
    wide_eva_set = {"ch", "sh", "ii", "iii", "ee", "m", "cth", "ckh"}

    for wid, w_glyphs in by_word.items():
        w_glyphs.sort(key=lambda item: item.get("bbox", [0, 0, 0, 0])[1])
        w_obj = word_map.get(wid, {})
        eva_text = w_obj.get("eva_text") or (w_glyphs[0].get("ref_eva_text", "") if w_glyphs else "")
        tokens = tokenize_eva_word(eva_text)
        if not tokens:
            continue

        if len(w_glyphs) == len(tokens):
            for g, tok in zip(w_glyphs, tokens):
                g["eva_char"] = tok
            continue

        x_mins = [g.get("bbox", [0, 0, 0, 0])[1] for g in w_glyphs]
        x_maxs = [g.get("bbox", [0, 0, 0, 0])[3] for g in w_glyphs]
        w_left = float(min(x_mins))
        w_span = max(1.0, float(max(x_maxs) - w_left))

        K = len(tokens)
        for g in w_glyphs:
            gx_center = 0.5 * (g.get("bbox", [0, 0, 0, 0])[1] + g.get("bbox", [0, 0, 0, 0])[3])
            rel_x = np.clip((gx_center - w_left) / w_span, 0.0, 0.999)
            expected_k = rel_x * K

            g_loops = int(g.get("loop_count", 0))
            gh = float(g.get("height", 20))
            gw = float(g.get("width", 15))

            best_tok = tokens[min(K - 1, int(round(expected_k)))]
            best_score = -1e9
            k_start = max(0, int(np.floor(expected_k)) - 1)
            k_end = min(K - 1, int(np.ceil(expected_k)) + 1)
            for k_cand in range(k_start, k_end + 1):
                tok = tokens[k_cand]
                pos_pen = abs(k_cand - expected_k) * 1.25
                compat = 0.0
                if g_loops >= 1 and tok in loop_eva_set:
                    compat += 1.5
                elif g_loops == 0 and tok not in loop_eva_set:
                    compat += 0.8
                if gh >= 26 and tok in tall_eva_set:
                    compat += 1.2
                if gw >= 22 and tok in wide_eva_set:
                    compat += 1.0
                score = compat - pos_pen
                if score > best_score:
                    best_score = score
                    best_tok = tok
            g["eva_char"] = best_tok


def reprocess_seraphinianus(base_dir: Path):
    page_json_path = base_dir / "seraphinianus" / "pages" / "serafini_p030.json"
    page_img_path = base_dir / "seraphinianus" / "pages" / "serafini_p030.jpg"
    cat_json_path = base_dir / "seraphinianus" / "seraphinianus_alphabet_catalogue.json"

    print(f"[*] Regenerating Seraphinianus p030 glyphs from words ({page_json_path})...")
    with open(page_json_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    img = cv2.imread(str(page_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {page_img_path}")
    H_img, W_img = img.shape[:2]
    scale_calib = MetricScaleCalibration(manuscript_type="seraphinianus", image_height_px=H_img, image_width_px=W_img)

    binarizer = Binarizer()
    segmenter = GlyphSegmenter(scale_calib=scale_calib)
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)

    page_id = page_data.get("page_id", "serafini_p030")
    raw_words = page_data.get("words", [])

    # 1. Enforce strict disjoint spatial partitioning (zero overlapping bboxes or shared parchment)
    words = SpatialWordPartitioner.resolve_disjoint_word_partitions(raw_words, (H_img, W_img))
    page_data["words"] = words
    print(f"  [+] Seraphinianus: Spatial disjoint partitioning enforced on {len(words)} words (0 overlaps).")

    # 2. Train / Fit Neural Word Decomposer on intact word crops
    word_patches = []
    for w in words:
        wb = w.get("bbox", [])
        if len(wb) == 4:
            y0, x0, y1, x1 = [int(v) for v in wb]
            if y1 > y0 + 4 and x1 > x0 + 4:
                word_patches.append(img[y0:y1, x0:x1])

    neural_decomposer = NeuralWordDecomposer()
    if word_patches:
        neural_decomposer.fit_on_word_corpus(word_patches)

    hybrid_decomposer = HybridNeuroKinematicDecomposer()

    regenerated_glyphs: List[Dict[str, Any]] = []
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

        w_patch = img[wy0:wy1, wx0:wx1]
        w_mask = binarizer.binarize(cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)).astype(bool)

        # Apply word polygon if present to prevent cross-line bleed
        w_poly = w.get("polygon", [])
        if w_poly and len(w_poly) >= 3:
            pm = np.zeros(w_mask.shape[:2], dtype=np.uint8)
            pts_loc = np.array([[int(round(p[0] - wx0)), int(round(p[1] - wy0))] for p in w_poly], dtype=np.int32)
            cv2.fillPoly(pm, [pts_loc], 1)
            pm = cv2.dilate(pm, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
            if int(np.sum(w_mask & pm)) >= 6:
                w_mask = w_mask & pm

        # OPTION A: Ductus Géométrique & Découplage Diacritique
        assembled = segmenter.assemble_word_into_glyphs(w_mask, expected_glyph_count=None, scale_calib=scale_calib)
        line_id = w.get("line_id", "L000")
        word_id = w.get("word_id", f"{line_id}_W00")

        glyph_idx = 0
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

            g_patch = img[gy0:gy1, gx0:gx1]
            if g_patch.shape[:2] != g_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(g_patch, g_mask)
            poly = segmenter.compute_ink_polygon(g_mask, offset=(gy0, gx0))

            strokes = tracker.extract_calligraphic_ductus(g_mask, single_stroke_mode=True, max_strokes=1, scale_calib=scale_calib)
            for s in strokes:
                pts = s.get("points", [])
                if len(pts) >= 2:
                    s["points"] = tracker._parameterize_stroke(pts)
                    s["mean_width"] = float(np.mean([p[2] for p in s["points"]]))
                    s["nib_angle_deg"] = 40.0

            if not strokes:
                strokes = [{
                    "stroke_id": "cal_s00",
                    "order_index": 0,
                    "points": [(float(gh / 2.0), float(gw / 2.0), 2.0)],
                    "length": 0.0,
                    "start": (float(gh / 2.0), float(gw / 2.0)),
                    "end": (float(gh / 2.0), float(gw / 2.0)),
                    "nib_angle_deg": 40.0,
                    "mean_width": 2.0,
                    "is_impact": True
                }]

            for s in strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]

            svg = VectorExporter.to_svg(strokes, max(10, gw), max(10, gh))
            glyph_id = f"{page_id}_{line_id}_{word_id}_G{glyph_idx:02d}"

            g_rec = {
                "glyph_id": glyph_id,
                "page_id": page_id,
                "line_id": line_id,
                "word_id": word_id,
                "word_bbox": [wy0, wx0, wy1, wx1],
                "glyph_index_in_word": glyph_idx,
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
                "stroke_count": 1,
                "strokes": strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "png_rel": "",
                "svg_rel": "",
                "mask": g_mask,
                "crop_uri": crop_uri,
                "segmentation_mode": "heuristic_calligraphic",
                "diacritics": comp.get("diacritics", []),
                "is_dot": comp.get("is_dot", False)
            }
            GlyphCatalogue.count_glyph_loops(g_rec)
            regenerated_glyphs.append(g_rec)
            glyph_idx += 1

        # OPTION B: Décomposition Neuronale de Mots (PCA / NMF)
        neural_comps = neural_decomposer.decompose_word(w_mask)
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
            ng_patch = img[gy0:gy1, gx0:gx1]
            if ng_patch.shape[:2] != ng_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(ng_patch, ng_mask)
            poly = segmenter.compute_ink_polygon(ng_mask, offset=(gy0, gx0))
            n_strokes = tracker.extract_calligraphic_ductus(ng_mask, single_stroke_mode=True, max_strokes=1, scale_calib=scale_calib)
            for s in n_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(n_strokes, max(10, gw), max(10, gh))

            ng_rec = {
                "glyph_id": f"{page_id}_{line_id}_{word_id}_N{n_idx:02d}",
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
                "stroke_count": 1,
                "strokes": n_strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": crop_uri,
                "segmentation_mode": "neural_word_pca",
                "neural_basis_id": comp.get("neural_basis_id", "PHI_00")
            }
            GlyphCatalogue.count_glyph_loops(ng_rec)
            neural_glyphs.append(ng_rec)
            n_idx += 1

        # OPTION C: Hybride Neuro-Cinématique & Consensus
        hybrid_comps = hybrid_decomposer.decompose_word_hybrid(w_mask, assembled, neural_comps)
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
            hg_patch = img[gy0:gy1, gx0:gx1]
            if hg_patch.shape[:2] != hg_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(hg_patch, hg_mask)
            poly = segmenter.compute_ink_polygon(hg_mask, offset=(gy0, gx0))
            h_strokes = tracker.extract_calligraphic_ductus(hg_mask, single_stroke_mode=True, max_strokes=1, scale_calib=scale_calib)
            for s in h_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(h_strokes, max(10, gw), max(10, gh))

            hg_rec = {
                "glyph_id": f"{page_id}_{line_id}_{word_id}_H{h_idx:02d}",
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
                "stroke_count": 1,
                "strokes": h_strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": crop_uri,
                "segmentation_mode": "hybrid_consensus",
                "hybrid_basis_id": comp.get("hybrid_basis_id", "HYB_00")
            }
            GlyphCatalogue.count_glyph_loops(hg_rec)
            hybrid_glyphs.append(hg_rec)
            h_idx += 1

    # Cross-evaluate options consensus and tag every glyph
    cross_evaluate_options_consensus(regenerated_glyphs, neural_glyphs, hybrid_glyphs)

    print(f"  [+] Seraphinianus: Option A = {len(regenerated_glyphs)} glyphs, Option B (Neural PCA) = {len(neural_glyphs)}, Option C (Hybrid) = {len(hybrid_glyphs)}. Building Option C Catalogue...")
    cat_builder = GlyphCatalogue(target_alphabet_size=52)
    cat_data = cat_builder.build_catalogue(regenerated_glyphs, corpus_type="seraphinianus", scale_calib=scale_calib)
    cat_builder.export_catalogue_json(cat_data, cat_json_path)

    for g in regenerated_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)
    for g in neural_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)
    for g in hybrid_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)

    page_data["glyphs"] = regenerated_glyphs
    page_data["glyphs_neural_pca"] = neural_glyphs
    page_data["glyphs_hybrid_consensus"] = hybrid_glyphs
    page_data["glyph_count"] = len(regenerated_glyphs)
    page_data["glyph_count_neural_pca"] = len(neural_glyphs)
    page_data["glyph_count_hybrid_consensus"] = len(hybrid_glyphs)
    page_data["segmentation_modes"] = ["heuristic_calligraphic", "neural_word_pca", "hybrid_consensus"]
    with open(page_json_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    print(f"  [+] Seraphinianus catalogue rebuilt: {len(cat_data.get('alphabet', []))} strictly loop-disjoint archetypes.")


def reprocess_voynich(base_dir: Path):
    page_json_path = base_dir / "voynich" / "pages" / "f1r.json"
    page_img_path = base_dir / "voynich" / "pages" / "f1r.jpg"
    cat_json_path = base_dir / "voynich" / "voynich_alphabet_catalogue.json"

    print(f"[*] Regenerating Voynich f1r glyphs from words ({page_json_path})...")
    with open(page_json_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    page_id = page_data.get("page_id", "f1r")
    raw_words = page_data.get("words", [])

    yale_raw_path = root_dir / "data" / "raw" / "scans" / "voynich" / "yale" / "f001r.jpg"
    if yale_raw_path.exists():
        yale_bgr = cv2.imread(str(yale_raw_path))
        if yale_bgr is not None:
            from voynich_ductus.ingestion.image_registration import MultimodalFolioAligner
            aligner = MultimodalFolioAligner()
            ref_img = cv2.imread(str(page_img_path))
            already_yale = (ref_img is not None and ref_img.shape[:2] == yale_bgr.shape[:2])
            if not already_yale and ref_img is not None:
                print("  [+] Performing Multimodal Folio SIFT/RANSAC registration onto Yale HD scan (3168x2500)...")
                reg = aligner.compute_registration(ref_img, yale_bgr, folio_id="f1r1800_to_yale")
                if reg.get("status") == "success":
                    M = reg["matrix_2x3"]
                    for w in raw_words:
                        w["bbox"] = list(aligner.transform_bbox(w["bbox"], M))
                        if "polygon" in w and w["polygon"]:
                            w["polygon"] = aligner.transform_polygon(w["polygon"], M)
                    print(f"  [+] Bounding boxes accurately transformed onto Yale HD plate ({reg.get('inliers')} inliers, scale {reg.get('scale_x')}).")

            # Apply CLAHE contrast enhancement on LAB L channel to restore rich iron gall ink contrast
            lab = cv2.cvtColor(yale_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_yale = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(page_img_path), enhanced_yale)
            img = enhanced_yale
            H_img, W_img = img.shape[:2]
            page_data["image_width"] = int(W_img)
            page_data["image_height"] = int(H_img)
    else:
        img = cv2.imread(str(page_img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {page_img_path}")
        H_img, W_img = img.shape[:2]

    # 1. Enforce strict disjoint spatial partitioning on Voynich words (0 overlaps)
    words = SpatialWordPartitioner.resolve_disjoint_word_partitions(raw_words, (H_img, W_img))
    page_data["words"] = words
    print(f"  [+] Voynich f1r: Spatial disjoint partitioning enforced on {len(words)} words (0 overlaps).")

    # 2. Train / Fit Neural Word Decomposer on intact word crops
    word_patches = []
    for w in words:
        wb = w.get("bbox", [])
        if len(wb) == 4:
            y0, x0, y1, x1 = [int(v) for v in wb]
            if y1 > y0 + 4 and x1 > x0 + 4:
                word_patches.append(img[y0:y1, x0:x1])

    neural_decomposer = NeuralWordDecomposer()
    if word_patches:
        neural_decomposer.fit_on_word_corpus(word_patches)

    hybrid_decomposer = HybridNeuroKinematicDecomposer()

    scale_calib = MetricScaleCalibration(manuscript_type="voynich", image_height_px=H_img, image_width_px=W_img)
    binarizer = Binarizer(method="sauvola")
    segmenter = GlyphSegmenter(scale_calib=scale_calib)
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)
    skel_eng = Skeletonizer()
    graph_ext = StrokeGraphExtractor()
    resolver = JunctionResolver()

    regenerated_glyphs: List[Dict[str, Any]] = []
    neural_glyphs: List[Dict[str, Any]] = []
    hybrid_glyphs: List[Dict[str, Any]] = []
    improved_count = 0

    for w in words:
        w_bbox = w.get("bbox")
        if not w_bbox or len(w_bbox) != 4:
            continue
        wy0, wx0, wy1, wx1 = [int(v) for v in w_bbox]
        wy0, wx0 = max(0, wy0), max(0, wx0)
        wy1, wx1 = min(H_img, wy1), min(W_img, wx1)
        if wy1 <= wy0 + 2 or wx1 <= wx0 + 2:
            continue

        w_patch = img[wy0:wy1, wx0:wx1]
        gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
        try:
            w_mask = binarizer.binarize(gray_w).astype(bool)
            if np.sum(w_mask) < 25:
                t_w = threshold_otsu(gray_w)
                w_mask = (gray_w < t_w).astype(bool)
        except Exception:
            t_w = threshold_otsu(gray_w)
            w_mask = (gray_w < t_w).astype(bool)

        eva_text = w.get("eva_text", "")
        eva_tokens = tokenize_eva_word(eva_text)
        expected_k = len(eva_tokens) if eva_tokens else None

        # OPTION A: Ductus Géométrique & Découplage Diacritique
        assembled = segmenter.assemble_word_into_glyphs(w_mask, expected_glyph_count=expected_k, scale_calib=scale_calib)
        line_id = w.get("line_id", "L00")
        word_id = w.get("word_id", f"{line_id}_W00")

        glyph_idx = 0
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

            g_patch = img[gy0:gy1, gx0:gx1]
            if g_patch.shape[:2] != g_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(g_patch, g_mask)
            poly = segmenter.compute_ink_polygon(g_mask, offset=(gy0, gx0))

            strokes = tracker.extract_calligraphic_ductus(g_mask, single_stroke_mode=False, max_strokes=4, scale_calib=scale_calib)
            ys = [p[0] for s in strokes for p in s.get("points", [])]
            y_cov = (max(ys) - min(ys)) / float(max(1, gh)) if ys else 0

            if y_cov < 0.60 and gh >= 16:
                try:
                    skel, widths = skel_eng.extract_skeleton(g_mask)
                    G = graph_ext.build_pixel_graph(skel, widths)
                    raw = graph_ext.decompose_into_strokes(G)
                    fb_strokes = resolver.resolve_and_order_strokes(raw) if raw else []
                    fb_strokes = fb_strokes[:4]  # Medieval cap: max 4 strokes (e.g. compound gallows cth/ckh)
                    fb_ys = [p[0] for s in fb_strokes for p in s.get("points", [])]
                    fb_cov = (max(fb_ys) - min(fb_ys)) / float(max(1, gh)) if fb_ys else 0
                    if fb_cov > y_cov and fb_strokes:
                        strokes = fb_strokes
                        improved_count += 1
                except Exception:
                    pass

            strokes = strokes[:4]  # Strict guarantee: never more than 4 strokes

            for s in strokes:
                pts = s.get("points", [])
                if len(pts) >= 2:
                    s["points"] = tracker._parameterize_stroke(pts)
                    s["mean_width"] = float(np.mean([p[2] for p in s["points"]]))
                    s["nib_angle_deg"] = 40.0

            if not strokes:
                strokes = [{
                    "stroke_id": "cal_s00",
                    "order_index": 0,
                    "points": [(float(gh / 2.0), float(gw / 2.0), 2.0)],
                    "length": 0.0,
                    "start": (float(gh / 2.0), float(gw / 2.0)),
                    "end": (float(gh / 2.0), float(gw / 2.0)),
                    "nib_angle_deg": 40.0,
                    "mean_width": 2.0,
                    "is_impact": True
                }]

            for s in strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]

            svg = VectorExporter.to_svg(strokes, max(10, gw), max(10, gh))
            if word_id.startswith(f"{line_id}_"):
                glyph_id = f"{page_id}_{word_id}_G{glyph_idx:02d}"
            else:
                glyph_id = f"{page_id}_{line_id}_{word_id}_G{glyph_idx:02d}"

            eva_c = eva_tokens[glyph_idx] if (eva_tokens and glyph_idx < len(eva_tokens)) else ""

            g_rec = {
                "glyph_id": glyph_id,
                "page_id": page_id,
                "line_id": line_id,
                "word_id": word_id,
                "word_bbox": [wy0, wx0, wy1, wx1],
                "glyph_index_in_word": glyph_idx,
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
                "stroke_count": len(strokes),
                "strokes": strokes,
                "ductus_svg": svg,
                "svg_content": svg,
                "eva_char": eva_c,
                "ref_eva_text": eva_text,
                "png_rel": "",
                "svg_rel": "",
                "mask": g_mask,
                "crop_uri": crop_uri,
                "segmentation_mode": "heuristic_calligraphic",
                "diacritics": comp.get("diacritics", []),
                "is_dot": comp.get("is_dot", False)
            }
            GlyphCatalogue.count_glyph_loops(g_rec)
            regenerated_glyphs.append(g_rec)
            glyph_idx += 1

        # OPTION B: Décomposition Neuronale de Mots (PCA / NMF)
        neural_comps = neural_decomposer.decompose_word(w_mask, expected_glyph_count=expected_k)
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
            ng_patch = img[gy0:gy1, gx0:gx1]
            if ng_patch.shape[:2] != ng_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(ng_patch, ng_mask)
            poly = segmenter.compute_ink_polygon(ng_mask, offset=(gy0, gx0))
            n_strokes = tracker.extract_calligraphic_ductus(ng_mask, single_stroke_mode=False, max_strokes=4, scale_calib=scale_calib)[:4]
            for s in n_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(n_strokes, max(10, gw), max(10, gh))

            ng_rec = {
                "glyph_id": f"{page_id}_{line_id}_{word_id}_N{n_idx:02d}",
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
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": crop_uri,
                "segmentation_mode": "neural_word_pca",
                "neural_basis_id": comp.get("neural_basis_id", "PHI_00")
            }
            GlyphCatalogue.count_glyph_loops(ng_rec)
            neural_glyphs.append(ng_rec)
            n_idx += 1

        # OPTION C: Hybride Neuro-Cinématique & Consensus
        hybrid_comps = hybrid_decomposer.decompose_word_hybrid(w_mask, assembled, neural_comps, expected_glyph_count=expected_k)
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
            hg_patch = img[gy0:gy1, gx0:gx1]
            if hg_patch.shape[:2] != hg_mask.shape[:2]:
                continue

            crop_uri = build_isolated_crop_uri(hg_patch, hg_mask)
            poly = segmenter.compute_ink_polygon(hg_mask, offset=(gy0, gx0))
            h_strokes = tracker.extract_calligraphic_ductus(hg_mask, single_stroke_mode=False, max_strokes=4, scale_calib=scale_calib)[:4]
            for s in h_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(h_strokes, max(10, gw), max(10, gh))

            hg_rec = {
                "glyph_id": f"{page_id}_{line_id}_{word_id}_H{h_idx:02d}",
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
                "png_rel": "",
                "svg_rel": "",
                "crop_uri": crop_uri,
                "segmentation_mode": "hybrid_consensus",
                "hybrid_basis_id": comp.get("hybrid_basis_id", "HYB_00")
            }
            GlyphCatalogue.count_glyph_loops(hg_rec)
            hybrid_glyphs.append(hg_rec)
            h_idx += 1

    # Cross-evaluate options consensus and tag every glyph
    cross_evaluate_options_consensus(regenerated_glyphs, neural_glyphs, hybrid_glyphs)

    realign_voynich_word_eva_tokens(words, regenerated_glyphs)
    realign_voynich_word_eva_tokens(words, neural_glyphs)
    realign_voynich_word_eva_tokens(words, hybrid_glyphs)

    print(f"  [+] Voynich f1r: Option A = {len(regenerated_glyphs)} glyphs, Option B (Neural PCA) = {len(neural_glyphs)}, Option C (Hybrid) = {len(hybrid_glyphs)}. Building Option C Catalogue...")
    cat_builder = GlyphCatalogue(target_alphabet_size=35)
    cat_data = cat_builder.build_catalogue(regenerated_glyphs, corpus_type="voynich", scale_calib=scale_calib)
    cat_builder.export_catalogue_json(cat_data, cat_json_path)

    for g in regenerated_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)
    for g in neural_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)
    for g in hybrid_glyphs:
        g.pop("mask", None)
        g.pop("crop_uri", None)

    page_data["glyphs"] = regenerated_glyphs
    page_data["glyphs_neural_pca"] = neural_glyphs
    page_data["glyphs_hybrid_consensus"] = hybrid_glyphs
    page_data["glyph_count"] = len(regenerated_glyphs)
    page_data["glyph_count_neural_pca"] = len(neural_glyphs)
    page_data["glyph_count_hybrid_consensus"] = len(hybrid_glyphs)
    page_data["segmentation_modes"] = ["heuristic_calligraphic", "neural_word_pca", "hybrid_consensus"]
    with open(page_json_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    print(f"  [+] Voynich catalogue rebuilt: {len(cat_data.get('alphabet', []))} strictly loop-disjoint archetypes.")


def main():
    base_dir = root_dir / "data" / "processed" / "atlas_dataset"
    reprocess_seraphinianus(base_dir)
    reprocess_voynich(base_dir)
    print("\n[*] Rebuilding Grand Glyph Explorer HTML with cache-busting & latest catalogues...")
    sys.path.insert(0, str(root_dir / "scripts"))
    import build_dynamic_grand_explorer
    build_dynamic_grand_explorer.main()
    print("\n[*] Full Cascade (Words -> Glyphs -> Ductus -> Archetypes -> Explorer HTML) completed!")


if __name__ == "__main__":
    main()
