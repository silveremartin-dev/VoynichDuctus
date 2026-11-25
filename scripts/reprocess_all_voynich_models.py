"""
Ultra-Fast Batch Multi-Model Glyph Reprocessor for VoynichDuctus.
Computes all 3 segmentation models across Voynich folios:
1. Quill Ductus (heuristic_calligraphic, 40° nib bevel)
2. Neural Word PCA (neural_word_pca, horizontal spatial basis functions)
3. Hybrid Consensus (hybrid_consensus, neuro-kinematic arbitration)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.embeddings.word_decomposition import (
    NeuralWordDecomposition,
    HybridConsensusWordDecomposition,
    cross_evaluate_options_consensus
)
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.ingestion.codicology import natural_folio_sort_key


def reprocess_page(json_path: Path, img_path: Path, binarizer, segmenter, scale_calib, neural_decomposer, hybrid_decomposer):
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)
    with open(json_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    page_id = page_data.get("page_id", json_path.stem)
    words = page_data.get("words", [])
    if not words:
        return 0, 0, 0

    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0, 0
    H_img, W_img = img.shape[:2]

    existing_quill = page_data.get("glyphs", [])
    quill_by_word = {}
    for g in existing_quill:
        wid = g.get("word_id")
        if wid not in quill_by_word:
            quill_by_word[wid] = []
        quill_by_word[wid].append(g)

    # Fast calibration on representative word patches
    word_patches = []
    for w in words[:35]:
        wb = w.get("bbox", [])
        if len(wb) == 4:
            y0, x0, y1, x1 = [int(v) for v in wb]
            if y1 > y0 + 4 and x1 > x0 + 4:
                word_patches.append(img[y0:y1, x0:x1])

    if word_patches:
        neural_decomposer.fit_on_word_corpus(word_patches)

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
        gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
        try:
            w_mask = binarizer.binarize(gray_w).astype(bool)
            if np.sum(w_mask) < 25:
                w_mask = (gray_w < 128).astype(bool)
        except Exception:
            w_mask = (gray_w < 128).astype(bool)

        line_id = w.get("line_id", "L00")
        word_id = w.get("word_id", f"{line_id}_W00")
        w_quill = quill_by_word.get(word_id, [])

        opt_a_local = []
        for g in w_quill:
            gb = g.get("bbox", [wy0, wx0, wy1, wx1])
            opt_a_local.append({
                "bbox_local": (max(0, gb[0] - wy0), max(0, gb[1] - wx0), min(wy1 - wy0, gb[2] - wy0), min(wx1 - wx0, gb[3] - wx0)),
                "quill_glyph": g
            })

        # Model 2: Neural Word PCA
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

            matched_g = None
            for qa in opt_a_local:
                q_by0, q_bx0, q_by1, q_bx1 = qa["bbox_local"]
                inter_x = max(0, min(bx1, q_bx1) - max(bx0, q_bx0))
                min_w = max(1, min(bx1 - bx0, q_bx1 - q_bx0))
                if inter_x / min_w > 0.50:
                    matched_g = qa["quill_glyph"]
                    break

            c_type = matched_g.get("canonical_type", "G01") if matched_g else "G01"
            n_strokes = tracker.extract_calligraphic_ductus(ng_mask, single_stroke_mode=False, max_strokes=4, scale_calib=scale_calib)[:4]
            if not n_strokes:
                n_strokes = [{
                    "stroke_id": "cal_s00",
                    "order_index": 0,
                    "points": [(float(gh * 0.2), float(gw * 0.5), 2.5), (float(gh * 0.8), float(gw * 0.5), 2.5)],
                    "length": 2.0,
                    "start": (float(gh * 0.2), float(gw * 0.5)),
                    "end": (float(gh * 0.8), float(gw * 0.5)),
                    "nib_angle_deg": 40.0,
                    "mean_width": 2.5
                }]
            for s in n_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(n_strokes, max(10, gw), max(10, gh))

            poly = [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]]
            if gh >= 4 and gw >= 4:
                try:
                    poly = segmenter.compute_ink_polygon(ng_mask, offset=(gy0, gx0))
                except Exception:
                    pass

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
                "crop_uri": "",
                "canonical_type": c_type,
                "segmentation_mode": "neural_word_pca",
                "neural_basis_id": comp.get("neural_basis_id", "PHI_00")
            }
            neural_glyphs.append(ng_rec)
            n_idx += 1

        # Model 3: Hybrid Neuro-Kinematic Consensus
        hybrid_comps = hybrid_decomposer.decompose_word_hybrid(w_mask, opt_a_local, neural_comps)
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

            matched_g = None
            for qa in opt_a_local:
                q_by0, q_bx0, q_by1, q_bx1 = qa["bbox_local"]
                inter_x = max(0, min(bx1, q_bx1) - max(bx0, q_bx0))
                min_w = max(1, min(bx1 - bx0, q_bx1 - q_bx0))
                if inter_x / min_w > 0.50:
                    matched_g = qa["quill_glyph"]
                    break

            c_type = matched_g.get("canonical_type", "G01") if matched_g else "G01"
            h_strokes = tracker.extract_calligraphic_ductus(hg_mask, single_stroke_mode=False, max_strokes=4, scale_calib=scale_calib)[:4]
            if not h_strokes:
                h_strokes = [{
                    "stroke_id": "cal_s00",
                    "order_index": 0,
                    "points": [(float(gh * 0.2), float(gw * 0.5), 2.5), (float(gh * 0.8), float(gw * 0.5), 2.5)],
                    "length": 2.0,
                    "start": (float(gh * 0.2), float(gw * 0.5)),
                    "end": (float(gh * 0.8), float(gw * 0.5)),
                    "nib_angle_deg": 40.0,
                    "mean_width": 2.5
                }]
            for s in h_strokes:
                s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]
            svg = VectorExporter.to_svg(h_strokes, max(10, gw), max(10, gh))

            poly = [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]]
            if gh >= 4 and gw >= 4:
                try:
                    poly = segmenter.compute_ink_polygon(hg_mask, offset=(gy0, gx0))
                except Exception:
                    pass

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
                "crop_uri": "",
                "canonical_type": c_type,
                "segmentation_mode": "hybrid_consensus",
                "hybrid_basis_id": comp.get("hybrid_basis_id", "HYB_00")
            }
            hybrid_glyphs.append(hg_rec)
            h_idx += 1

    cross_evaluate_options_consensus(existing_quill, neural_glyphs, hybrid_glyphs)

    page_data["glyphs_neural_pca"] = neural_glyphs
    page_data["glyphs_hybrid_consensus"] = hybrid_glyphs
    page_data["glyph_count_neural_pca"] = len(neural_glyphs)
    page_data["glyph_count_hybrid_consensus"] = len(hybrid_glyphs)
    page_data["segmentation_modes"] = ["heuristic_calligraphic", "neural_word_pca", "hybrid_consensus"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)

    return len(existing_quill), len(neural_glyphs), len(hybrid_glyphs)


def main():
    pages_dir = root_dir / "data" / "processed" / "atlas_dataset" / "voynich" / "pages"
    json_files = sorted(pages_dir.glob("*.json"), key=lambda p: natural_folio_sort_key(p.stem))

    scale_calib = MetricScaleCalibration(manuscript_type="voynich")
    binarizer = Binarizer(method="sauvola")
    segmenter = GlyphSegmenter(scale_calib=scale_calib)
    neural_decomposer = NeuralWordDecomposition()
    hybrid_decomposer = HybridConsensusWordDecomposition()

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(json_files)
    print(f"[*] Reprocessing multi-model glyph sets for {min(limit, len(json_files))} Voynich folios...", flush=True)

    total_start = time.time()
    for i, jf in enumerate(json_files[:limit]):
        if jf.stem == "f1r":
            print(f"[{i+1}/{limit}] f1r already precomputed with Yale HD alignment. Skipping.", flush=True)
            continue

        img_f = pages_dir / f"{jf.stem}.jpg"
        if not img_f.exists():
            continue

        t0 = time.time()
        q_cnt, n_cnt, h_cnt = reprocess_page(jf, img_f, binarizer, segmenter, scale_calib, neural_decomposer, hybrid_decomposer)
        t_el = time.time() - t0
        print(f"[{i+1}/{limit}] {jf.stem}: Quill={q_cnt}, Neural PCA={n_cnt}, Consensus={h_cnt} in {t_el:.2f}s", flush=True)

    print(f"\n[✓] Completed multi-model reprocessing in {time.time() - total_start:.2f}s!", flush=True)

    print("\n[*] Rebuilding Grand Glyph Explorer HTML...", flush=True)
    import build_dynamic_grand_explorer
    build_dynamic_grand_explorer.main()
    print("[✓] Explorer rebuilt with all multi-model glyph sets!", flush=True)


if __name__ == "__main__":
    main()
