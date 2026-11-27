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



