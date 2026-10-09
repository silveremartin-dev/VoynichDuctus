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

