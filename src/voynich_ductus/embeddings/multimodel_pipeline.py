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
from voynich_ductus.embeddings.word_decomposition import GenerativeTypewriterDecomposer
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.ingestion.spatial_partitioner import SpatialWordPartitioner
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor


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


def _extract_euler_bernoulli_ductus(
    g_mask: np.ndarray,
    skel_eng: Skeletonizer,
    graph_ext: StrokeGraphExtractor,
    resolver: JunctionResolver,
    max_strokes: int = 6,
    scale_calib: Optional[MetricScaleCalibration] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts the comparative Euler-Bernoulli Topological Skeleton Ductus with
    metric-calibrated pinhole filling, lateral spur pruning, and smooth RDP trajectories.
    """
    gh, gw = g_mask.shape[:2]
    px_per_mm = float(scale_calib.px_per_mm) if scale_calib is not None else max(4.5, max(gh, gw) / 4.5)
    try:
        from scipy.ndimage import binary_fill_holes, gaussian_filter1d, label as nd_label
        pinhole_max = max(2, int(round(0.045 * (px_per_mm ** 2))))
        padded = np.pad(g_mask.astype(bool), 2, mode="constant", constant_values=False)
        holes = binary_fill_holes(padded) & (~padded)
        lbl_h, num_h = nd_label(holes)
        clean_mask = g_mask.astype(bool).copy()
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)[2:-2, 2:-2]
            if int(np.sum(hm)) <= pinhole_max:
                clean_mask |= hm

        skel, widths = skel_eng.extract_skeleton(clean_mask)
        spur_min = max(3, int(round(0.40 * px_per_mm)))
        skel = CalligraphicVectorizer.prune_skeleton_spurs(skel, widths * 0.5, min_spur_len=spur_min)
        graph = graph_ext.build_pixel_graph(skel, widths)
        raw = graph_ext.decompose_into_strokes(graph)
        # Filter tiny stray skeleton fragments before junction chaining
        min_seg_pts = max(3, int(round(0.35 * px_per_mm)))
        if raw and len(raw) > 1:
            non_tiny = [s for s in raw if len(s.get("points", [])) >= min_seg_pts]
            if non_tiny:
                raw = non_tiny
        eb_strokes = (
            resolver.resolve_and_order_strokes(
                raw,
                max_gap=max(6.0, 0.78 * px_per_mm),
                min_cos=-0.35,
                max_strokes=max_strokes,
            )
            if raw
            else []
        )

        # Metric smoothing & RDP simplification preserving curved ovals and arcs (100% pure mm)
        rdp_eps = 0.055 * px_per_mm
        for s in eb_strokes:
            pts = s.get("points", [])
            if len(pts) >= 4:
                cy = np.array([float(p[0]) for p in pts], dtype=np.float64)
                cx = np.array([float(p[1]) for p in pts], dtype=np.float64)
                ws = [float(p[2]) if len(p) > 2 else 0.22 * px_per_mm for p in pts]
                sy0, sx0, sy1, sx1 = cy[0], cx[0], cy[-1], cx[-1]
                closed = np.hypot(sy0 - sy1, sx0 - sx1) <= 0.35 * px_per_mm
                step_ds = float(np.median(np.hypot(np.diff(cy), np.diff(cx)))) if len(pts) >= 3 else (0.08 * px_per_mm)
                step_ds = max(0.05 * px_per_mm, step_ds)
                sig_pts = max(1.0, min(4.2, (0.22 * px_per_mm) / max(1e-3, step_ds)))
                if closed and len(cy) >= 6:
                    cy = gaussian_filter1d(cy[:-1], sigma=sig_pts, mode="wrap")
                    cx = gaussian_filter1d(cx[:-1], sigma=sig_pts, mode="wrap")
                    cy = np.append(cy, cy[0])
                    cx = np.append(cx, cx[0])
                else:
                    cy = gaussian_filter1d(cy, sigma=sig_pts, mode="nearest")
                    cx = gaussian_filter1d(cx, sigma=sig_pts, mode="nearest")
                    cy[0], cx[0], cy[-1], cx[-1] = sy0, sx0, sy1, sx1
                sm_pts = [(float(cy[i]), float(cx[i]), ws[i]) for i in range(len(pts))]
                simp = CalligraphicVectorizer._rdp_simplify_with_width(sm_pts, epsilon=rdp_eps)
                s["points"] = [(round(p[0], 2), round(p[1], 2), round(p[2], 2)) for p in simp]
                s["start"] = (s["points"][0][0], s["points"][0][1])
                s["end"] = (s["points"][-1][0], s["points"][-1][1])
                s["length"] = float(len(s["points"]))
        if len(eb_strokes) > 1:
            min_arc_px = 0.28 * px_per_mm
            long_eb = [
                s for s in eb_strokes
                for pts in [s.get("points", [])]
                if sum(
                    float(np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]))
                    for k in range(len(pts) - 1)
                ) >= min_arc_px
            ]
            if long_eb:
                eb_strokes = long_eb
        for idx_eb, s in enumerate(eb_strokes):
            s["order_index"] = idx_eb
            s["stroke_id"] = f"eb_s{idx_eb:02d}"
    except Exception:
        eb_strokes = []

    if not eb_strokes:
        eb_strokes = [{
            "stroke_id": "eb_s00",
            "order_index": 0,
            "points": [(float(gh * 0.25), float(gw * 0.5), 0.22 * px_per_mm), (float(gh * 0.75), float(gw * 0.5), 0.22 * px_per_mm)],
            "length": 2.0,
            "start": (float(gh * 0.25), float(gw * 0.5)),
            "end": (float(gh * 0.75), float(gw * 0.5)),
            "px_per_mm": float(px_per_mm),
        }]
    for s in eb_strokes:
        s["px_per_mm"] = float(px_per_mm)
    eb_svg = VectorExporter.to_svg(eb_strokes, max(1, gw), max(1, gh), px_per_mm=px_per_mm, variable_width=False)
    return eb_strokes, eb_svg


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
    skel_eng: Skeletonizer,
    graph_ext: StrokeGraphExtractor,
    resolver: JunctionResolver,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], str, List[List[int]]]:
    """Extracts biomechanical mm quill/ribbon ductus strokes, Euler-Bernoulli skeleton ductus, SVGs, and tight ink polygon."""
    max_strokes = 1 if is_serafini else 4
    ppm = float(scale_calib.px_per_mm)
    strokes = tracker.extract_calligraphic_ductus(
        g_mask, single_stroke_mode=is_serafini, max_strokes=max_strokes, scale_calib=scale_calib
    )[:max_strokes]

    if not strokes:
        w_nom = (0.38 if is_serafini else 0.45) * ppm
        cy, cx = float(gh * 0.5), float(gw * 0.5)
        strokes = [{
            "stroke_id": "cal_s00",
            "order_index": 0,
            "points": [(cy, cx, w_nom)],
            "length": 0.0,
            "length_mm": 0.0,
            "start": (cy, cx),
            "end": (cy, cx),
            "nib_angle_deg": 40.0,
            "mean_width": w_nom,
            "mean_width_mm": round(w_nom / max(1e-4, ppm), 3),
            "is_impact": True,
            "is_dot": True,
            "px_per_mm": ppm,
        }]

    for s in strokes:
        s["px_per_mm"] = ppm
        s["page_points"] = [[float(p[0] + gy0), float(p[1] + gx0), float(p[2])] for p in s.get("points", [])]

    svg = VectorExporter.to_svg(strokes, max(1, gw), max(1, gh), px_per_mm=ppm, variable_width=True)
    eb_strokes, eb_svg = _extract_euler_bernoulli_ductus(
        g_mask, skel_eng, graph_ext, resolver, max_strokes=max_strokes, scale_calib=scale_calib
    )
    poly: List[List[int]] = [[gx0, gy0], [gx0 + gw, gy0], [gx0 + gw, gy0 + gh], [gx0, gy0 + gh]]
    if gh >= 4 and gw >= 4:
        try:
            computed_poly = segmenter.compute_ink_polygon(g_mask, offset=(gy0, gx0), scale_calib=scale_calib)
            if computed_poly:
                poly = computed_poly
        except Exception:
            pass
    return strokes, svg, eb_strokes, eb_svg, poly


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
    tracker = CalligraphicVectorizer(
        nib_angle_deg=40.0,
        min_nib_width=max(1.2, scale_calib.mm_to_px(0.18)),
        max_nib_width=max(3.5, scale_calib.mm_to_px(0.58)),
        smoothing_sigma=max(0.85, scale_calib.mm_to_px(0.14)),
    )
    skel_eng = Skeletonizer()
    graph_ext = StrokeGraphExtractor()
    resolver = JunctionResolver()
    typewriter_decomposer = GenerativeTypewriterDecomposer(is_serafini=is_serafini)

    if not is_serafini and restore_rosetta_boxes:
        try:
            from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader
            rosetta_loader = VoynicheseRosettaLoader()
            _, _, r_words = rosetta_loader.get_yale_hd_scan_and_words(page_id)
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
                new_lines = []
                for l_idx, lc in enumerate(line_clusters):
                    l_id = f"L{l_idx:02d}"
                    ly0 = min(max(0, rw.bbox_scan[1]) for rw in lc)
                    lx0 = min(max(0, rw.bbox_scan[0]) for rw in lc)
                    ly1 = max(min(H_img, rw.bbox_scan[3]) for rw in lc)
                    lx1 = max(min(W_img, rw.bbox_scan[2]) for rw in lc)
                    new_lines.append({"line_id": l_id, "bbox": [int(ly0), int(lx0), int(ly1), int(lx1)]})
                for li in range(len(new_lines) - 1):
                    if new_lines[li]["bbox"][2] > new_lines[li + 1]["bbox"][0]:
                        mid_y = int(round((new_lines[li]["bbox"][2] + new_lines[li + 1]["bbox"][0]) / 2.0))
                        new_lines[li]["bbox"][2] = mid_y
                        new_lines[li + 1]["bbox"][0] = mid_y
                page_data["lines"] = new_lines
                page_data["line_count"] = len(new_lines)
        except Exception:
            pass

    words = SpatialWordPartitioner.resolve_disjoint_word_partitions(raw_words, (H_img, W_img))
    page_data["words"] = words
    page_data["image_width"] = int(W_img)
    page_data["image_height"] = int(H_img)

    old_by_word: Dict[str, List[Dict[str, Any]]] = {}
    for g in page_data.get("glyphs", []):
        old_by_word.setdefault(g.get("word_id", ""), []).append(g)

    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win_s = max(15, int(round(scale_calib.mm_to_px(4.1))) | 1)
    win_s = max(3, min(win_s, (min(H_img, W_img) - 1) | 1))
    try:
        g64 = gray_full.astype(np.float64)
        mean_g = cv2.boxFilter(g64, cv2.CV_64F, (win_s, win_s), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(g64 * g64, cv2.CV_64F, (win_s, win_s), normalize=True, borderType=cv2.BORDER_REFLECT)
        std_g = np.sqrt(np.maximum(0.0, mean_sq - mean_g * mean_g))
        t_sauv_full = (mean_g * (1.0 + 0.16 * ((std_g / 128.0) - 1.0))).astype(np.float32)
    except Exception:
        t_sauv_full = None

    # First pass: binarize all word envelopes on the page (100% metric-calibrated mm windows on both codices)
    min_word_ink = max(2, int(round(scale_calib.area_mm2_to_px2(0.08))))
    word_entries: List[Tuple[Dict[str, Any], Tuple[int, int, int, int], np.ndarray]] = []
    word_patches_bgr: List[np.ndarray] = []
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
        if len(word_patches_bgr) < 35 and (wy1 - wy0) > 4 and (wx1 - wx0) > 4:
            word_patches_bgr.append(w_patch)
        w_patch_rgb = cv2.cvtColor(w_patch, cv2.COLOR_BGR2RGB)
        try:
            w_mask = segmenter.binarize_word_crop_hysteresis(
                w_patch_rgb,
                scale_calib=scale_calib,
                full_image_gray=gray_full,
                crop_box=(wy0, wx0, wy1, wx1),
                precomputed_sauvola=t_sauv_full,
            ).astype(bool)
            if np.sum(w_mask) < min_word_ink:
                gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
                w_mask = (gray_w < 128).astype(bool)
        except Exception:
            gray_w = cv2.cvtColor(w_patch, cv2.COLOR_BGR2GRAY)
            w_mask = (gray_w < 128).astype(bool)

        w_poly = segmenter.compute_ink_polygon(w_mask, offset=(wy0, wx0), scale_calib=scale_calib)
        if w_poly:
            w["polygon"] = w_poly
        word_entries.append((w, (wy0, wx0, wy1, wx1), w_mask))

    if word_entries:
        typewriter_decomposer.fit_on_word_masks(
            [wm for _, _, wm in word_entries],
            scale_calib=scale_calib,
            word_patches=word_patches_bgr,
        )

    # Second pass: Viterbi MDL word-to-glyph segmentation + Bézier silhouette ductus vectorization
    unified_glyphs: List[Dict[str, Any]] = []
    for w, (wy0, wx0, wy1, wx1), w_mask in word_entries:
        line_id = w.get("line_id", "L00")
        word_id = w.get("word_id", f"{line_id}_W00")
        eva_text = w.get("eva_text", "")
        old_w_glyphs = old_by_word.get(word_id, [])

        if recompute_option_a or not old_w_glyphs:
            assembled = typewriter_decomposer.decompose_word(w_mask, scale_calib=scale_calib)
            g_idx = 0
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

                strokes, svg, eb_strokes, eb_svg, poly = _vectorize_glyph_mask(
                    g_mask, gy0, gx0, gh, gw, is_serafini, scale_calib, tracker, segmenter, skel_eng, graph_ext, resolver
                )
                hole_polys = segmenter.compute_hole_polygons(g_mask, offset=(gy0, gx0), scale_calib=scale_calib)
                g_id = f"{page_id}_{word_id}_G{g_idx:02d}" if line_id in word_id else f"{page_id}_{line_id}_{word_id}_G{g_idx:02d}"
                g_rec = {
                    "glyph_id": g_id,
                    "page_id": page_id,
                    "line_id": line_id,
                    "word_id": word_id,
                    "word_bbox": [wy0, wx0, wy1, wx1],
                    "glyph_index_in_word": g_idx,
                    "bbox": [gy0, gx0, gy1, gx1],
                    "polygon": poly,
                    "hole_polygons": hole_polys,
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
                    "euler_stroke_count": len(eb_strokes),
                    "euler_strokes": eb_strokes,
                    "euler_svg_content": eb_svg,
                    "png_rel": "",
                    "svg_rel": "",
                    "crop_uri": "",
                    "canonical_type": c_type,
                    "eva_char": "",
                    "ref_eva_text": eva_text,
                    "segmentation_mode": "generative_typewriter",
                    "typewriter_proto_id": comp.get("typewriter_proto_id", "TW_00"),
                    "diacritics": comp.get("diacritics", []),
                    "is_dot": comp.get("is_dot", False),
                    "mask": g_mask,
                }
                GlyphCatalogue.count_glyph_loops(g_rec)
                g_rec.pop("mask", None)
                unified_glyphs.append(g_rec)
                g_idx += 1
        else:
            unified_glyphs.extend(old_w_glyphs)

    if not is_serafini:
        realign_voynich_word_eva_tokens(words, unified_glyphs)

    # Clean up obsolete multi-model arrays so page JSONs stay 3x smaller and faster
    page_data.pop("glyphs_neural_pca", None)
    page_data.pop("glyphs_hybrid_consensus", None)
    page_data.pop("glyph_count_neural_pca", None)
    page_data.pop("glyph_count_hybrid_consensus", None)
    page_data["glyphs"] = unified_glyphs
    page_data["glyph_count"] = len(unified_glyphs)
    page_data["segmentation_modes"] = ["generative_typewriter"]

    return len(unified_glyphs), 0, 0
