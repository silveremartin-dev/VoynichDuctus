"""
Closed-Loop Paleographic Judge & Visual Proof-Plate Benchmark (`autotune_ductus_loop.py`).

Evaluates `CalligraphicVectorizer` on a deterministic multi-scribe reference panel of
real manuscript glyphs from Beinecke MS 408 (Voynich, Scribes 1-5 per Lisa Fagin Davis 2022)
and Codex Seraphinianus (1981), rendering side-by-side `Scan | Ribbon 40° | Score` plates
and computing a formal composite paleographic score:
  1. Generative Ribbon Fidelity: Silhouette IoU & Recall of rendered variable-width ribbon `w(s)`.
  2. Loop Cavity Clearance (`HoleOpen`): Enclosed loop bowls ('o', 'a', 'e', '8', '9', Serafini loops)
     must remain open rather than plugged by excessive stroke width.
  3. Biomechanical Smoothness (Anti-Tremor): Penalizes sub-nib sawtooth reversals and discrete jerk.
  4. Historical Kinematic Laws:
     - Voynich (15th-c. Italian Humanist Cursive): 1-4 strokes, counter-clockwise closed bowls,
       unbroken upward-left `[n]` finials, straight top-to-bottom gallows stems without Y-junction kinks,
       and heavy penalty on straight upward stem pushes against the 40° split quill.
     - Codex Seraphinianus: Strictly 1 continuous curvilinear main ribbon (+ optional detached
       diacritic dots) with mandatory Left-to-Right progression (`x_start <= x_end`).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from voynich_ductus.embeddings.multimodel_pipeline import _sanitize_glyph_mask_by_stroke_thickness
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer


# Target EVA archetypes covering core Voynich Humanist Cursive & Davis's 5-scribe tells
VOYNICH_PANEL_SPECS: List[Tuple[str, str, str]] = [
    ("f1r", "o", "Scribe1 [o] CCW loop"),
    ("f1r", "e", "Scribe1 [e] crescent/eye"),
    ("f1r", "a", "Scribe1 [a] bowl+minim"),
    ("f1r", "y", "Scribe1 [y/9] loop+tail"),
    ("f1r", "d", "Scribe1 [d/8] double-loop"),
    ("f1r", "s", "Scribe1 [s/2] curve"),
    ("f1r", "i", "Scribe1 [i] minim"),
    ("f1r", "n", "Scribe1 [n] up-left finial"),
    ("f26r", "n", "Scribe2 [n] 45deg finial"),
    ("f1r", "ch", "Scribe1 [ch] bench"),
    ("f1r", "sh", "Scribe1 [sh] plumed bench"),
    ("f1r", "k", "Scribe1 [k] bowed gallows"),
    ("f26r", "k", "Scribe2 [k] lifted bar"),
    ("f1r", "t", "Scribe1 [t] 2-loop gallows"),
    ("f1r", "p", "Scribe1 [p] 1-leg 2-loop"),
    ("f1r", "f", "Scribe1 [f] 1-leg 1-loop"),
    ("f68r1", "k", "Scribe4 [k] astro gallows"),
    ("f57r", "n", "Scribe5 [n] long low finial"),
    ("f103r", "n", "Scribe3 [n] tight finial"),
    ("f1r", "cth", "Scribe1 [cth] benched gallows"),
]


def _extract_real_glyph_entry(
    page_dir: Path,
    page_id: str,
    is_serafini: bool,
    target_eva: Optional[str] = None,
    glyph_predicate: Optional[Any] = None,
    label_text: str = "",
) -> Optional[Dict[str, Any]]:
    json_path = page_dir / f"{page_id}.json"
    jpg_path = page_dir / f"{page_id}.jpg"
    if not json_path.exists() or not jpg_path.exists():
        return None

    img_bgr = cv2.imread(str(jpg_path))
    if img_bgr is None:
        return None
    pdata = json.loads(json_path.read_text(encoding="utf-8"))
    H_img, W_img = img_bgr.shape[:2]
    calib = MetricScaleCalibration(
        manuscript_type="seraphinianus" if is_serafini else "voynich",
        image_height_px=H_img,
        image_width_px=W_img,
    )
    segmenter = GlyphSegmenter(scale_calib=calib)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    candidates = []
    for g in pdata.get("glyphs", []):
        if target_eva is not None and g.get("eva_char") != target_eva:
            continue
        if glyph_predicate is not None and not glyph_predicate(g):
            continue
        bbox = g.get("bbox", [0, 0, 0, 0])
        gy0, gx0, gy1, gx1 = [int(v) for v in bbox]
        if gy1 - gy0 < 6 or gx1 - gx0 < 4:
            continue
        # Require clean interior loop when sampling looped letters ('o', 'a', 'd', 'k', 'p', 't', 'f')
        if target_eva in ("o", "a", "d", "k", "p", "t", "f", "y") and int(g.get("loop_count", 0)) < 1:
            continue
        candidates.append(g)

    if not candidates and target_eva is not None:
        # Fallback without strict loop filter if low-res page plugged eye
        for g in pdata.get("glyphs", []):
            if g.get("eva_char") == target_eva:
                candidates.append(g)

    if not candidates:
        return None

    # Pick a clean representative near median area
    candidates.sort(key=lambda item: float(item.get("area_mm2", 1.0)))
    chosen = candidates[len(candidates) // 2]
    wy0, wx0, wy1, wx1 = [int(v) for v in chosen.get("word_bbox", chosen["bbox"])]
    wy0, wx0 = max(0, wy0), max(0, wx0)
    wy1, wx1 = min(H_img, wy1), min(W_img, wx1)
    w_patch_rgb = cv2.cvtColor(img_bgr[wy0:wy1, wx0:wx1], cv2.COLOR_BGR2RGB)
    w_mask = segmenter.binarize_word_crop_hysteresis(
        w_patch_rgb,
        scale_calib=calib,
        full_image_gray=gray_full,
        crop_box=(wy0, wx0, wy1, wx1),
    ).astype(bool)

    gy0, gx0, gy1, gx1 = [int(v) for v in chosen["bbox"]]
    ly0, lx0 = max(0, gy0 - wy0), max(0, gx0 - wx0)
    ly1, lx1 = min(w_mask.shape[0], gy1 - wy0), min(w_mask.shape[1], gx1 - wx0)
    sub_mask = w_mask[ly0:ly1, lx0:lx1].copy()

    # Apply polygon clip if present so adjacent glyph tails inside bbox are excluded
    poly = chosen.get("polygon", [])
    if poly and len(poly) >= 3:
        pmask = np.zeros_like(sub_mask, dtype=np.uint8)
        pts_local = np.array([[int(p[0]) - gx0, int(p[1]) - gy0] for p in poly], dtype=np.int32)
        cv2.fillPoly(pmask, [pts_local], 1)
        if np.any(sub_mask & (pmask > 0)):
            sub_mask &= (pmask > 0)

    clean_mask, ty0, tx0, ty1, tx1 = _sanitize_glyph_mask_by_stroke_thickness(
        sub_mask, is_serafini=is_serafini, scale_calib=calib
    )
    cy0, cx0 = gy0 + ty0, gx0 + tx0
    cy1, cx1 = gy0 + ty1, gx0 + tx1
    crop_bgr = img_bgr[max(0, cy0):min(H_img, cy1), max(0, cx0):min(W_img, cx1)].copy()

    return {
        "glyph_id": chosen.get("glyph_id", f"{page_id}_g"),
        "page_id": page_id,
        "eva_char": chosen.get("eva_char", ""),
        "label": label_text or f"{page_id} [{chosen.get('eva_char', '')}]",
        "is_serafini": is_serafini,
        "mask": clean_mask,
        "crop_bgr": crop_bgr,
        "calib": calib,
    }


def build_reference_panel(base_atlas: Path, cache_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import pickle
    if cache_dir is not None:
        cache_file = cache_dir / "reference_panel_v1.pkl"
        if cache_file.exists():
            try:
                return pickle.loads(cache_file.read_bytes())
            except Exception:
                pass

    v_dir = base_atlas / "voynich/pages"
    s_dir = base_atlas / "seraphinianus/pages"

    v_panel: List[Dict[str, Any]] = []
    for pid, eva_c, lbl in VOYNICH_PANEL_SPECS:
        ent = _extract_real_glyph_entry(v_dir, pid, is_serafini=False, target_eva=eva_c, label_text=lbl)
        if ent is not None:
            v_panel.append(ent)

    s_panel: List[Dict[str, Any]] = []
    s_specs = [
        ("serafini_p030", lambda g: int(g.get("loop_count", 0)) == 1 and float(g.get("width_mm", 0)) >= 2.2, "p030 1-loop cursive"),
        ("serafini_p030", lambda g: int(g.get("loop_count", 0)) >= 2 and float(g.get("width_mm", 0)) >= 2.8, "p030 2+loop ribbon"),
        ("serafini_p030", lambda g: int(g.get("loop_count", 0)) == 0 and float(g.get("width_mm", 0)) >= 2.4, "p030 open wave"),
        ("serafini_p030", lambda g: len(g.get("diacritics", [])) >= 1 or int(g.get("stroke_count", 1)) >= 2, "p030 diacritic glyph"),
        ("serafini_p012", lambda g: int(g.get("loop_count", 0)) == 1 and float(g.get("width_mm", 0)) >= 2.2, "p012 HD 1-loop"),
        ("serafini_p012", lambda g: int(g.get("loop_count", 0)) >= 2 and float(g.get("width_mm", 0)) >= 2.8, "p012 HD multi-loop"),
        ("serafini_p012", lambda g: float(g.get("height_mm", 0)) >= 3.8 and float(g.get("width_mm", 0)) >= 2.0, "p012 HD tall ascender"),
        ("serafini_p040", lambda g: int(g.get("loop_count", 0)) >= 1 and float(g.get("width_mm", 0)) >= 2.5, "p040 HD cursive loop"),
    ]
    for pid, pred, lbl in s_specs:
        ent = _extract_real_glyph_entry(s_dir, pid, is_serafini=True, glyph_predicate=pred, label_text=lbl)
        if ent is not None:
            s_panel.append(ent)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "reference_panel_v1.pkl").write_bytes(pickle.dumps((v_panel, s_panel)))

    return v_panel, s_panel


def evaluate_glyph_paleography(
    entry: Dict[str, Any],
    strokes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Formal Paleographic Judge scoring a single rendered glyph on [0..1]:
      - Ribbon Silhouette IoU & Recall
      - Interior Loop Hole Openness (`hole_open`)
      - Trajectory Smoothness / Anti-Tremor (`jerk_pen`)
      - Kinematic Law Compliance (`law_pen`)
    """
    mask: np.ndarray = entry["mask"]
    calib: MetricScaleCalibration = entry["calib"]
    ppm = float(calib.px_per_mm)
    mm_per_px = 1.0 / ppm
    is_serafini = bool(entry["is_serafini"])
    eva_c = entry.get("eva_char", "")
    h, w = mask.shape

    # Identify genuine interior holes (> pinhole threshold)
    pinhole_px = 1 if is_serafini else max(1, int(round(0.045 * (ppm ** 2))))
    padded = np.pad(mask, 2, mode="constant", constant_values=False)
    holes_raw = binary_fill_holes(padded)[2:-2, 2:-2] & (~mask)
    lbl_h, num_h = label(holes_raw)
    holes_mask = np.zeros_like(mask, dtype=bool)
    for hid in range(1, num_h + 1):
        hm = (lbl_h == hid)
        if int(np.sum(hm)) > pinhole_px:
            holes_mask |= hm

    pts_list = [s.get("points", []) for s in strokes if s.get("points")]
    pred_mask = CalligraphicVectorizer._rasterize_strokes_mask(pts_list, (h, w), scale_w=1.0)
    inter = int(np.sum(pred_mask & mask))
    union = int(np.sum(pred_mask | mask))
    iou = float(inter / union) if union > 0 else 0.0
    recall = float(inter / max(1, int(np.sum(mask))))

    hole_tot = int(np.sum(holes_mask))
    if hole_tot > 0:
        hole_open = 1.0 - float(int(np.sum(pred_mask & holes_mask)) / hole_tot)
    else:
        hole_open = 1.0

    # Smoothness / Anti-tremor penalty
    jerk_pen = 0.0
    law_pen = 0.0
    notes: List[str] = []

    non_dot_strokes = [
        s for s in strokes
        if not s.get("is_impact", False) and not s.get("is_diacritic", False) and len(s.get("points", [])) >= 2
    ]

    for s in non_dot_strokes:
        pts = s["points"]
        for k in range(1, len(pts) - 1):
            dy1, dx1 = pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]
            dy2, dx2 = pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]
            l1, l2 = math.hypot(dy1, dx1), math.hypot(dy2, dx2)
            if l1 > 1e-4 and l2 > 1e-4:
                cos_t = (dy1 * dy2 + dx1 * dx2) / (l1 * l2)
                if (l1 * mm_per_px) < 0.40 and (l2 * mm_per_px) < 0.40 and cos_t < 0.0:
                    jerk_pen += 0.12 * abs(cos_t)

    if is_serafini:
        if len(non_dot_strokes) != 1:
            law_pen += 0.25 * abs(len(non_dot_strokes) - 1)
            notes.append(f"main_strokes={len(non_dot_strokes)}")
        if non_dot_strokes:
            p_main = non_dot_strokes[0]["points"]
            dx_mm = (p_main[-1][1] - p_main[0][1]) * mm_per_px
            span_w_mm = (max(p[1] for p in p_main) - min(p[1] for p in p_main)) * mm_per_px
            is_closed = math.hypot(p_main[0][0] - p_main[-1][0], p_main[0][1] - p_main[-1][1]) * mm_per_px <= 0.42
            if not is_closed and span_w_mm >= 0.80 and dx_mm < -0.15:
                law_pen += 0.35
                notes.append(f"R->L({dx_mm:.2f}mm)")
            elif is_closed:
                min_x = min(p[1] for p in p_main)
                if (p_main[0][1] - min_x) * mm_per_px > 0.45 * span_w_mm:
                    law_pen += 0.15
                    notes.append("loop_start_not_left")
    else:
        # Voynich Italian Humanist Cursive laws
        if len(strokes) > 4:
            law_pen += 0.25 * (len(strokes) - 4)
            notes.append(f">4st({len(strokes)})")

        # 1. Closed single-loop signs ('o', 'e') should not be chopped into 2-4 fragments
        if eva_c == "o" and hole_tot > 0:
            if len(non_dot_strokes) != 1 and num_h <= 1:
                law_pen += 0.18
                notes.append(f"o_chopped({len(non_dot_strokes)}st)")
            elif len(non_dot_strokes) == 1 and num_h == 1:
                pts_o = non_dot_strokes[0]["points"]
                # Signed area in (x_right, y_down): CCW single loop has signed_area < 0
                s_area = sum(
                    pts_o[i][1] * pts_o[i + 1][0] - pts_o[i + 1][1] * pts_o[i][0]
                    for i in range(len(pts_o) - 1)
                )
                if s_area > 0:
                    law_pen += 0.20
                    notes.append("CW_loop")

        # 2. EVA '[n]' should preserve its single continuous down-right -> up-left finial stroke
        if eva_c == "n":
            if len(non_dot_strokes) > 1:
                law_pen += 0.22
                notes.append(f"n_chopped({len(non_dot_strokes)}st)")
            elif non_dot_strokes:
                pts_n = non_dot_strokes[0]["points"]
                # Initial pull must go downward (dy > 0), NOT start from finial tip
                k_init = min(len(pts_n) - 1, max(1, len(pts_n) // 4))
                if (pts_n[k_init][0] - pts_n[0][0]) * mm_per_px < 0.10:
                    law_pen += 0.22
                    notes.append("n_wrong_start")

        # 3. Tall Gallows ('k', 't', 'p', 'f') first vertical stem must pull top-to-bottom cleanly
        if eva_c in ("k", "t", "p", "f") and non_dot_strokes:
            s0 = non_dot_strokes[0]["points"]
            dy0_mm = (s0[-1][0] - s0[0][0]) * mm_per_px
            chord0_mm = math.hypot(s0[-1][0] - s0[0][0], s0[-1][1] - s0[0][1]) * mm_per_px
            arc0_mm = CalligraphicVectorizer._polyline_arc_mm(s0, mm_per_px)
            tort0 = arc0_mm / max(1e-3, chord0_mm)
            if dy0_mm < 1.2:
                law_pen += 0.20
                notes.append("gallows_no_downstem")
            elif tort0 > 1.30:
                law_pen += 0.15
                notes.append(f"stem_kink({tort0:.2f})")

        # 4. Universal split-quill anti-gouge: no stroke may start by pushing a straight stem steep UP or UP-LEFT
        for s in non_dot_strokes:
            pts = s["points"]
            is_cl = len(pts) >= 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) * mm_per_px <= 0.45
            if not is_cl:
                dy_net = (pts[-1][0] - pts[0][0]) * mm_per_px
                chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * mm_per_px
                arc = CalligraphicVectorizer._polyline_arc_mm(pts, mm_per_px)
                # Straight upward stem push (tortuosity < 1.35 and net climb > 0.65 mm)
                if dy_net < -0.65 and (arc / max(1e-3, chord)) < 1.35:
                    law_pen += 0.25
                    notes.append(f"up_push({dy_net:.2f}mm)")

    score = max(0.0, 0.55 * iou + 0.20 * recall + 0.25 * hole_open - jerk_pen - law_pen)
    return {
        "iou": round(iou, 3),
        "recall": round(recall, 3),
        "hole_open": round(hole_open, 3),
        "stroke_count": len(strokes),
        "jerk_pen": round(jerk_pen, 3),
        "law_pen": round(law_pen, 3),
        "score": round(score, 3),
        "notes": ",".join(notes) if notes else "OK",
    }


def render_diagnostic_plate(
    panel_results: List[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]],
    title: str,
    out_path: Path,
) -> None:
    """Renders a high-clarity visual proof plate (`Scan | True Ribbon + Ductus Arrows`) for multimodal inspection."""
    cols = 4
    n = len(panel_results)
    rows = max(1, math.ceil(n / cols))
    card_w, card_h = 340, 175
    header_h = 46
    canvas = np.full((header_h + rows * card_h, cols * card_w, 3), 24, dtype=np.uint8)

    mean_sc = float(np.mean([r[2]["score"] for r in panel_results])) if panel_results else 0.0
    mean_iou = float(np.mean([r[2]["iou"] for r in panel_results])) if panel_results else 0.0
    mean_hole = float(np.mean([r[2]["hole_open"] for r in panel_results])) if panel_results else 0.0
    hdr = f"{title} | Mean Score={mean_sc:.3f} | IoU={mean_iou:.3f} | HoleOpen={mean_hole:.3f}"
    cv2.putText(canvas, hdr, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 240, 245), 1, cv2.LINE_AA)

    palette_bgr = [
        (70, 75, 245),    # Stroke 1: Red
        (245, 150, 50),   # Stroke 2: Blue
        (95, 210, 65),    # Stroke 3: Green
        (40, 180, 250),   # Stroke 4: Amber
    ]

    for idx, (entry, strokes, metrics) in enumerate(panel_results):
        r_i, c_i = divmod(idx, cols)
        ox, oy = c_i * card_w, header_h + r_i * card_h
        cv2.rectangle(canvas, (ox + 3, oy + 3), (ox + card_w - 4, oy + card_h - 4), (42, 46, 54), -1)
        cv2.rectangle(canvas, (ox + 3, oy + 3), (ox + card_w - 4, oy + card_h - 4), (78, 84, 96), 1)

        mask = entry["mask"]
        crop_bgr = entry["crop_bgr"]
        gh, gw = mask.shape
        box_sz = 112
        scale = min((box_sz - 16) / max(1.0, float(gw)), (box_sz - 16) / max(1.0, float(gh)))
        scale = max(1.5, min(9.0, scale))
        rw, rh = max(4, int(round(gw * scale))), max(4, int(round(gh * scale)))

        # Left sub-panel: Natural Scan
        scan_box = np.full((box_sz, box_sz, 3), 215, dtype=np.uint8)
        if crop_bgr.size > 0:
            rs_scan = cv2.resize(crop_bgr, (rw, rh), interpolation=cv2.INTER_CUBIC)
            sy0, sx0 = (box_sz - rh) // 2, (box_sz - rw) // 2
            scan_box[sy0:sy0 + rh, sx0:sx0 + rw] = rs_scan
        canvas[oy + 28:oy + 28 + box_sz, ox + 12:ox + 12 + box_sz] = scan_box

        # Right sub-panel: Mask + Rendered Ribbon + Spine + Start/Direction Arrows
        rib_box = np.full((box_sz, box_sz, 3), 245, dtype=np.uint8)
        sy0, sx0 = (box_sz - rh) // 2, (box_sz - rw) // 2
        rs_mask = cv2.resize(mask.astype(np.uint8) * 255, (rw, rh), interpolation=cv2.INTER_NEAREST) > 127
        # Faint parchment-gray ground-truth ink silhouette
        sub_view = rib_box[sy0:sy0 + rh, sx0:sx0 + rw]
        sub_view[rs_mask] = (195, 198, 205)

        # Overlay variable-width ribbon per stroke
        rib_overlay = rib_box.copy()
        for s_idx, s in enumerate(strokes):
            col = palette_bgr[s_idx % len(palette_bgr)]
            pts = s.get("points", [])
            if not pts:
                continue
            if len(pts) == 1 or s.get("is_impact", False):
                py = int(round(sy0 + pts[0][0] * scale))
                px = int(round(sx0 + pts[0][1] * scale))
                rad = max(2, int(round(pts[0][2] * 0.5 * scale)))
                cv2.circle(rib_overlay, (px, py), rad, col, -1, cv2.LINE_AA)
            else:
                for k in range(len(pts) - 1):
                    p0, p1 = pts[k], pts[k + 1]
                    x0_i, y0_i = int(round(sx0 + p0[1] * scale)), int(round(sy0 + p0[0] * scale))
                    x1_i, y1_i = int(round(sx0 + p1[1] * scale)), int(round(sy0 + p1[0] * scale))
                    th = max(1, int(round(0.5 * (p0[2] + p1[2]) * scale)))
                    cv2.line(rib_overlay, (x0_i, y0_i), (x1_i, y1_i), col, th, cv2.LINE_AA)

        rib_box = cv2.addWeighted(rib_overlay, 0.52, rib_box, 0.48, 0)

        # Draw crisp centerlines, direction arrows, and numbered start dots
        for s_idx, s in enumerate(strokes):
            col = palette_bgr[s_idx % len(palette_bgr)]
            pts = s.get("points", [])
            if not pts:
                continue
            if len(pts) >= 2 and not s.get("is_impact", False):
                poly_xy = np.array(
                    [[int(round(sx0 + p[1] * scale)), int(round(sy0 + p[0] * scale))] for p in pts],
                    dtype=np.int32,
                )
                cv2.polylines(rib_box, [poly_xy], False, col, 2, cv2.LINE_AA)
                # Mid-stroke arrow showing exact direction of motion
                mid_k = max(1, len(pts) // 2)
                ax0, ay0 = poly_xy[mid_k - 1]
                ax1, ay1 = poly_xy[mid_k]
                if math.hypot(ax1 - ax0, ay1 - ay0) < 4 and mid_k + 1 < len(pts):
                    ax1, ay1 = poly_xy[min(len(pts) - 1, mid_k + 2)]
                cv2.arrowedLine(rib_box, (int(ax0), int(ay0)), (int(ax1), int(ay1)), (20, 20, 20), 2, cv2.LINE_AA, tipLength=0.45)

            st_x, st_y = int(round(sx0 + pts[0][1] * scale)), int(round(sy0 + pts[0][0] * scale))
            cv2.circle(rib_box, (st_x, st_y), 4, (30, 185, 50), -1, cv2.LINE_AA)
            cv2.circle(rib_box, (st_x, st_y), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(rib_box, str(s_idx + 1), (st_x + 4, max(11, st_y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 1, cv2.LINE_AA)

        canvas[oy + 28:oy + 28 + box_sz, ox + 134:ox + 134 + box_sz] = rib_box

        # Labels & metrics
        cv2.putText(canvas, entry["label"][:32], (ox + 10, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 240, 245), 1, cv2.LINE_AA)
        sc_col = (110, 230, 120) if metrics["score"] >= 0.72 else ((80, 195, 245) if metrics["score"] >= 0.55 else (90, 95, 245))
        cv2.putText(canvas, f"Sc:{metrics['score']:.2f}", (ox + 254, oy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sc_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"IoU:{metrics['iou']:.2f}", (ox + 254, oy + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Hol:{metrics['hole_open']:.2f}", (ox + 254, oy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"St:{metrics['stroke_count']}", (ox + 254, oy + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, metrics["notes"][:26], (ox + 12, oy + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sc_col, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_benchmark(
    tag: str = "eval",
    out_dir: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    from voynich_ductus.ingestion.codicology import extract_davis_scribal_forensics

    base_atlas = ROOT_DIR / "data/processed/atlas_dataset"
    if out_dir is None:
        out_dir = ROOT_DIR / "scratch/ductus_judge"

    v_panel, s_panel = build_reference_panel(base_atlas, cache_dir=out_dir)
    p_cfg = params or {}
    vec = CalligraphicVectorizer(
        nib_angle_deg=p_cfg.get("nib_angle_deg", 40.0),
        fit_cubic_bezier=p_cfg.get("fit_cubic_bezier", True),
        hole_margin_scale=p_cfg.get("hole_margin_scale", 1.0),
        gain_bias=p_cfg.get("gain_bias", 1.0),
        inertia_scale=p_cfg.get("inertia_scale", 1.0),
    )

    v_results = []
    forensic_glyphs = []
    for ent in v_panel:
        strokes = vec.extract_calligraphic_ductus(
            ent["mask"], single_stroke_mode=False, max_strokes=4, scale_calib=ent["calib"]
        )
        m = evaluate_glyph_paleography(ent, strokes)
        v_results.append((ent, strokes, m))
        forensic_glyphs.append({
            "glyph_id": ent["glyph_id"],
            "eva_char": ent["eva_char"],
            "strokes": strokes,
            "width_mm": ent["mask"].shape[1] / float(ent["calib"].px_per_mm),
            "height_mm": ent["mask"].shape[0] / float(ent["calib"].px_per_mm),
            "loop_count": 1 if m["hole_open"] < 1.0 or ent["eva_char"] in ("o", "a", "d", "k", "p", "t", "f") else 0,
        })

    s_results = []
    for ent in s_panel:
        strokes = vec.extract_calligraphic_ductus(
            ent["mask"], single_stroke_mode=True, max_strokes=3, scale_calib=ent["calib"]
        )
        m = evaluate_glyph_paleography(ent, strokes)
        s_results.append((ent, strokes, m))

    v_plate = out_dir / f"voynich_panel_{tag}.png"
    s_plate = out_dir / f"serafini_panel_{tag}.png"
    if not quiet:
        render_diagnostic_plate(v_results, f"Voynich Humanist Ductus ({tag})", v_plate)
        render_diagnostic_plate(s_results, f"Seraphinianus Cursive L->R ({tag})", s_plate)

    v_mean = float(np.mean([r[2]["score"] for r in v_results])) if v_results else 0.0
    s_mean = float(np.mean([r[2]["score"] for r in s_results])) if s_results else 0.0
    law_tot = float(sum(r[2]["law_pen"] for r in v_results) + sum(r[2]["law_pen"] for r in s_results))
    overall = 0.65 * v_mean + 0.35 * s_mean
    forensics = extract_davis_scribal_forensics(forensic_glyphs)

    if not quiet:
        print(f"=== Paleographic Judge [{tag}] ===")
        print(f"  Voynich ({len(v_results)} glyphs)       : Score={v_mean:.3f}")
        for ent, _, m in v_results:
            print(f"    - {ent['label']:28s} | Sc={m['score']:.3f} IoU={m['iou']:.2f} Hole={m['hole_open']:.2f} St={m['stroke_count']} [{m['notes']}]")
        print(f"  Seraphinianus ({len(s_results)} glyphs) : Score={s_mean:.3f}")
        for ent, _, m in s_results:
            print(f"    - {ent['label']:28s} | Sc={m['score']:.3f} IoU={m['iou']:.2f} Hole={m['hole_open']:.2f} St={m['stroke_count']} [{m['notes']}]")
        print(f"  Davis 5-Scribe Forensics        : {forensics}")
        print(f"  OVERALL COMPOSITE SCORE         : {overall:.4f} (law_pen_sum={law_tot:.3f})")
        print(f"  Plates saved to: {v_plate} & {s_plate}")

    return {
        "voynich_score": v_mean,
        "serafini_score": s_mean,
        "overall": overall,
        "law_pen_sum": law_tot,
        "forensics": forensics,
        "v_plate": str(v_plate),
        "s_plate": str(s_plate),
    }


def run_autonomous_autotune(rounds: int = 12, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Closed-loop coordinate-ascent & stochastic neighborhood self-calibration:
    Evaluates physical parameter candidates against the multi-scribe Paleographic Judge,
    accepts strictly improving configurations without law regressions, saves optimal
    parameters to `src/voynich_ductus/vectorizer/calibrated_ductus_params.json`, and
    renders the final calibrated proof plates.
    """
    if out_dir is None:
        out_dir = ROOT_DIR / "scratch/ductus_judge"
    cfg_out = ROOT_DIR / "src/voynich_ductus/vectorizer/calibrated_ductus_params.json"

    best_params: Dict[str, Any] = {
        "nib_angle_deg": 40.0,
        "fit_cubic_bezier": True,
        "hole_margin_scale": 1.0,
        "gain_bias": 1.0,
        "inertia_scale": 1.0,
    }
    base_res = run_benchmark(tag="autotune_init", out_dir=out_dir, params=best_params, quiet=True)
    best_score = base_res["overall"]
    best_law = base_res["law_pen_sum"]
    print(f"[*] Autonomous Closed-Loop Calibration started | Init Score={best_score:.4f} (law_pen={best_law:.3f})")

    # Deterministic coordinate grid + neighborhood sweeps
    proposals: List[Dict[str, Any]] = [
        {"hole_margin_scale": 0.88},
        {"hole_margin_scale": 0.94},
        {"hole_margin_scale": 1.08},
        {"hole_margin_scale": 1.14},
        {"gain_bias": 1.05},
        {"gain_bias": 1.10},
        {"gain_bias": 0.96},
        {"inertia_scale": 0.92},
        {"inertia_scale": 1.10},
        {"inertia_scale": 1.18},
        {"fit_cubic_bezier": False},
        {"hole_margin_scale": 1.08, "inertia_scale": 1.10, "fit_cubic_bezier": True},
        {"hole_margin_scale": 1.12, "gain_bias": 1.04, "inertia_scale": 1.10},
    ]

    for idx, delta in enumerate(proposals[: max(1, rounds)], start=1):
        cand = dict(best_params)
        cand.update(delta)
        res = run_benchmark(tag=f"round_{idx:02d}", out_dir=out_dir, params=cand, quiet=True)
        sc = res["overall"]
        lw = res["law_pen_sum"]
        accepted = (sc > best_score + 1e-4) and (lw <= best_law + 0.02)
        status = "ACCEPTED" if accepted else "rejected"
        print(
            f"  [Round {idx:02d}] {status:8s} | Score={sc:.4f} (V={res['voynich_score']:.3f}, S={res['serafini_score']:.3f}, law={lw:.2f}) | delta={delta}"
        )
        if accepted:
            best_score = sc
            best_law = lw
            best_params = cand

    cfg_out.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"[+] Saved optimal physical parameters to {cfg_out}: {best_params}")
    final_res = run_benchmark(tag="v4_autotuned", out_dir=out_dir, params=best_params, quiet=False)
    return final_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="current")
    parser.add_argument("--autotune", action="store_true", help="Run closed-loop self-calibration and persist optimal params")
    parser.add_argument("--rounds", type=int, default=14)
    args = parser.parse_args()
    if args.autotune:
        run_autonomous_autotune(rounds=args.rounds)
    else:
        run_benchmark(tag=args.tag)
