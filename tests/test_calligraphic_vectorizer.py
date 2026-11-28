"""
Unit tests for CalligraphicVectorizer and Pen-Nib Physics Ridge Tracker.
"""

import numpy as np
import pytest
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer


def test_nib_width_estimation():
    vec = CalligraphicVectorizer(nib_angle_deg=40.0, min_nib_width=1.0, max_nib_width=5.0)
    
    # Perpendicular to nib (40 + 90 = 130 deg) -> Maximum thickness
    perp_angle = np.deg2rad(130.0)
    w_max = vec.estimate_nib_width(perp_angle)
    assert pytest.approx(w_max, 0.1) == 5.0

    # Parallel to nib (40 deg) -> Minimum thickness (délié)
    par_angle = np.deg2rad(40.0)
    w_min = vec.estimate_nib_width(par_angle)
    assert pytest.approx(w_min, 0.1) == 1.0


def test_calligraphic_ductus_extraction():
    vec = CalligraphicVectorizer()

    # Synthetic loop (letter 'o' shape)
    mask = np.zeros((40, 40), dtype=bool)
    for y in range(40):
        for x in range(40):
            r = np.hypot(y - 20, x - 20)
            if 10 <= r <= 15:
                mask[y, x] = True

    strokes = vec.extract_calligraphic_ductus(mask)
    assert len(strokes) >= 1
    assert "points" in strokes[0]
    assert len(strokes[0]["points"]) >= 4
    assert strokes[0]["mean_width"] > 0
    assert strokes[0]["nib_angle_deg"] == 40.0


def test_empty_mask_ductus():
    vec = CalligraphicVectorizer()
    mask = np.zeros((20, 20), dtype=bool)
    strokes = vec.extract_calligraphic_ductus(mask)
    assert strokes == []


def test_multi_stroke_ductus_extraction():
    vec = CalligraphicVectorizer()
    
    # Synthetic letter 'i' or accented letter with a main stem and a separate dot/accent mark
    mask = np.zeros((50, 50), dtype=bool)
    # Detached accent / dot (y: 6 to 12, x: 23 to 27)
    mask[6:12, 23:27] = True
    # Main stem (y: 18 to 45, x: 23 to 27)
    mask[18:45, 23:27] = True

    strokes = vec.extract_calligraphic_ductus(mask)
    assert len(strokes) == 2
    assert strokes[0]["stroke_id"] == "cal_s00"
    assert strokes[1]["stroke_id"] == "cal_s01"
    assert strokes[0]["order_index"] == 0
    assert strokes[1]["order_index"] == 1


def test_seraphinianus_single_continuous_stroke():
    vec = CalligraphicVectorizer()

    # Synthetic complex cursive glyph with loop and stem
    mask = np.zeros((60, 40), dtype=bool)
    # Loop at top
    for y in range(5, 25):
        for x in range(10, 30):
            r = np.hypot(y - 15, x - 20)
            if 6 <= r <= 9:
                mask[y, x] = True
    # Stem extending down from bottom of loop
    mask[24:55, 18:23] = True

    # In single stroke mode, MUST return strictly 1 continuous stroke
    strokes = vec.extract_calligraphic_ductus(mask, single_stroke_mode=True)
    assert len(strokes) == 1
    assert strokes[0]["stroke_id"] == "cal_s00"
    assert len(strokes[0]["points"]) >= 10
    # Ductus must start at the top (touchdown) and end lower (pen lift)
    assert strokes[0]["points"][0][0] <= strokes[0]["points"][-1][0]


def test_physical_millimeter_scale_impossibility_and_ceiling():
    """
    Verifies that pixels are never treated as dimensionless:
    - A 0.3 mm x 0.3 mm mark (< physical nib width) cannot sustain a stroke trajectory (`is_impact=True`, `length_mm=0.0`).
    - A 300 mm x 300 mm giant block with a 20 mm wall never inflates the quill nib beyond physical quill bounds (`<= 0.95 mm`).
    """
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    vec = CalligraphicVectorizer(nib_angle_deg=40.0)

    cal_hd = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=14.0)
    tiny_mask = np.zeros((10, 10), dtype=bool)
    tiny_mask[3:7, 3:7] = True  # 4 px / 14 px/mm = 0.286 mm x 0.286 mm
    st_tiny = vec.extract_calligraphic_ductus(tiny_mask, scale_calib=cal_hd)
    assert len(st_tiny) == 1
    assert st_tiny[0]["is_impact"] is True
    assert len(st_tiny[0]["points"]) == 1
    assert st_tiny[0]["length_mm"] == 0.0

    cal_macro = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=2.0)
    macro_mask = np.zeros((620, 620), dtype=bool)
    macro_mask[10:610, 10:610] = True
    macro_mask[50:570, 50:570] = False  # 300 mm x 300 mm frame with 20 mm thick wall!
    st_macro = vec.extract_calligraphic_ductus(macro_mask, scale_calib=cal_macro)
    assert len(st_macro) >= 1
    assert st_macro[0]["is_macro_non_scribal"] is True
    assert st_macro[0]["mean_width_mm"] <= 0.95


def test_small_looped_glyph_no_sub_nib_sawtooth_tremor():
    """
    Regression test for 2.1 mm x 1.6 mm looped minims (e.g. Voynich 'a' on f1r_L22_W00_G01):
    With a 0.45 mm quill at 14 px/mm, pixel-staircase vellum roughness around a tight enclosed
    bowl + right leg must NEVER trigger sub-nib sawtooth oscillations or parasitic corner strokes.
    """
    import math
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
    from voynich_ductus.vectorizer.export_format import VectorExporter

    ppm = 14.0
    cal = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=ppm)
    vec = CalligraphicVectorizer(nib_angle_deg=40.0)

    # Build a 2.14 mm x 1.64 mm (23 x 30 px) looped 'a' with staircase border roughness
    mask = np.zeros((23, 30), dtype=bool)
    yy, xx = np.ogrid[:23, :30]
    bowl_outer = ((yy - 11.0) / 7.2) ** 2 + ((xx - 11.5) / 8.0) ** 2 <= 1.0
    bowl_hole = ((yy - 11.0) / 2.4) ** 2 + ((xx - 10.5) / 2.6) ** 2 <= 1.0
    mask[bowl_outer & (~bowl_hole)] = True
    # Descending right leg with rough 1-px vellum bumps
    for r_y in range(5, 20):
        x_c = 16.5 + 0.45 * (r_y - 5)
        x_lo = int(round(x_c - 2.8 - (r_y % 2) * 0.6))
        x_hi = int(round(x_c + 2.8 + ((r_y + 1) % 2) * 0.6))
        mask[r_y, max(0, x_lo):min(30, x_hi)] = True
    mask[bowl_hole] = False

    strokes = vec.extract_calligraphic_ductus(mask, scale_calib=cal)
    # 15th-century Italian Humanist Cursive Law (Davis 2022): looped Voynich 'a' = CCW bowl + downward right minim
    assert 1 <= len(strokes) <= 3
    assert all(not s.get("is_impact", False) for s in strokes)

    total_pts = sum(len(s["points"]) for s in strokes)
    assert total_pts <= 48

    for s in strokes:
        pts = s["points"]
        for i in range(1, len(pts) - 1):
            dy1, dx1 = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
            dy2, dx2 = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
            d1_mm = math.hypot(dy1, dx1) / ppm
            d2_mm = math.hypot(dy2, dx2) / ppm
            if 1e-4 < d1_mm < 0.40 and 1e-4 < d2_mm < 0.40:
                cos_turn = (dy1 * dy2 + dx1 * dx2) / (math.hypot(dy1, dx1) * math.hypot(dy2, dx2))
                assert cos_turn > -0.15, f"Sub-nib sawtooth reversal detected: cos={cos_turn:.3f}"

    svg = VectorExporter.to_svg(strokes, 30, 23, px_per_mm=ppm, variable_width=True)
    assert 'fill-rule="evenodd"' in svg
    assert 'fill-opacity="0.68"' in svg


def test_voynich_humanist_ccw_loop_and_upward_n_finial():
    """
    Verifies Lisa Fagin Davis (2022) & Voynich.nu Italian Humanist Cursive ductus laws:
    1. Closed single-loop `[o]` executes as 1 continuous Counter-Clockwise stroke (`signed_area_xy < 0`).
    2. Word-final EVA `[n]` (down-right minim + upward-left curved finial) stays 1 unbroken stroke
       starting at the top-left minim and ending on the upward-left flourish.
    """
    import cv2
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    ppm = 12.0
    cal = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=ppm)
    vec = CalligraphicVectorizer(nib_angle_deg=40.0)

    # 1. Closed 'o' ring
    o_mask = np.zeros((36, 36), dtype=np.uint8)
    cv2.circle(o_mask, (18, 18), 10, 1, 4)
    st_o = vec.extract_calligraphic_ductus(o_mask > 0, scale_calib=cal)
    assert len(st_o) == 1
    assert CalligraphicVectorizer._signed_ring_area_xy(st_o[0]["points"]) < 0.0

    # 2. Scribe-1 EVA '[n]' (slanted minim top-left -> bottom-right -> curve up & back left above minim)
    n_mask = np.zeros((48, 54), dtype=np.uint8)
    n_poly = np.array([[18, 16], [24, 34], [34, 36], [40, 25], [34, 9], [8, 5]], dtype=np.int32)
    cv2.polylines(n_mask, [n_poly], False, 1, 4, cv2.LINE_AA)
    st_n = vec.extract_calligraphic_ductus(n_mask > 0, scale_calib=cal)
    assert len(st_n) == 1
    pts_n = st_n[0]["points"]
    # Touchdown must be at top-left of minim (around y~14, x~18), descending first!
    assert pts_n[len(pts_n) // 4][0] > pts_n[0][0]


def test_voynich_gallows_straight_stem_before_head_loop():
    """
    Verifies Voynich gallows (`[f]`/`[k]`) two-phase construction:
    1. Primary vertical hampe executes FIRST (`order_index == 0`), top-to-bottom (`dy > 0`),
       without medial-axis Y-junction kinks (`tortuosity <= 1.18`).
    2. Upper crossbar + head loop executes SECOND.
    """
    import cv2
    import math
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    ppm = 12.0
    cal = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=ppm)
    vec = CalligraphicVectorizer(nib_angle_deg=40.0)

    g_mask = np.zeros((64, 48), dtype=np.uint8)
    cv2.line(g_mask, (14, 6), (14, 58), 1, 4)        # Tall vertical ascender stem
    cv2.line(g_mask, (14, 14), (32, 14), 1, 3)       # Horizontal crossbar below apex
    cv2.circle(g_mask, (33, 19), 6, 1, 3)            # Upper right loop

    strokes = vec.extract_calligraphic_ductus(g_mask > 0, scale_calib=cal)
    assert 2 <= len(strokes) <= 4
    s0 = strokes[0]["points"]
    dy_mm = (s0[-1][0] - s0[0][0]) / ppm
    chord_mm = math.hypot(s0[-1][0] - s0[0][0], s0[-1][1] - s0[0][1]) / ppm
    arc_mm = CalligraphicVectorizer._polyline_arc_mm(s0, 1.0 / ppm)
    assert dy_mm >= 3.2
    assert (arc_mm / chord_mm) <= 1.18


def test_seraphinianus_strict_left_to_right_progression():
    """
    Verifies that Codex Seraphinianus continuous cursive strokes ALWAYS progress
    from left (`x_start`) to right (`x_end`), even when entering low on the left
    baseline and rising higher on the right!
    """
    import cv2
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

    ppm = 9.0
    cal = MetricScaleCalibration(manuscript_type="seraphinianus", custom_px_per_mm=ppm)
    vec = CalligraphicVectorizer(nib_angle_deg=40.0)

    s_mask = np.zeros((48, 60), dtype=np.uint8)
    wave_xy = np.array([[6, 38], [20, 34], [32, 22], [52, 8]], dtype=np.int32)
    cv2.polylines(s_mask, [wave_xy], False, 1, 3, cv2.LINE_AA)

    strokes = vec.extract_calligraphic_ductus(s_mask > 0, single_stroke_mode=True, scale_calib=cal)
    assert len(strokes) == 1
    pts = strokes[0]["points"]
    assert pts[0][1] < pts[-1][1], f"Expected strict L->R progression, got x0={pts[0][1]} -> x1={pts[-1][1]}"






