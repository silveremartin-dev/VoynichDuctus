"""
Unit tests for Voynich codicological quire structure and Layfield & Davis (2026) page reordering.
"""

import pytest
from voynich_ductus.ingestion.codicology import (
    LAYFIELD_DAVIS_CITATION,
    VOYNICH_QUIRES,
    get_traditional_codex_order,
    get_layfield_davis_order,
    build_folio_codicology_registry,
    natural_folio_sort_key,
)


def test_layfield_davis_citation():
    assert "Layfield" in LAYFIELD_DAVIS_CITATION["authors"][0]
    assert "Davis" in LAYFIELD_DAVIS_CITATION["authors"][1]
    assert "Singulion Structure" in LAYFIELD_DAVIS_CITATION["title"]
    assert LAYFIELD_DAVIS_CITATION["year"] == 2026
    assert LAYFIELD_DAVIS_CITATION["journal"] == "Digital Medievalist"


def test_natural_sort_key():
    test_folios = ["f10r", "f1r", "f2v", "f1v", "f100r", "f67r2", "f67r1"]
    sorted_folios = sorted(test_folios, key=natural_folio_sort_key)
    assert sorted_folios == ["f1r", "f1v", "f2v", "f10r", "f67r1", "f67r2", "f100r"]


def test_quire_13_balneology_reordering():
    # Quire 13 folios in arbitrary scrambled order
    q13_folios = [
        "f75r", "f75v", "f76r", "f76v", "f77r", "f77v",
        "f78r", "f78v", "f79r", "f79v", "f80r", "f80v",
        "f81r", "f81v", "f82r", "f82v", "f83r", "f83v",
        "f84r", "f84v"
    ]
    
    # Traditional order is 75r..84v
    trad = get_traditional_codex_order(q13_folios)
    assert trad[0] == "f75r"
    assert trad[-1] == "f84v"
    
    # Layfield & Davis (2026) Table 14:
    # 77|82 -> 78|81 -> 75|84 -> 76|83 -> 79|80
    ld = get_layfield_davis_order(q13_folios)
    assert len(ld) == 20
    assert ld[:4] == ["f77r", "f77v", "f82r", "f82v"]
    assert ld[4:8] == ["f78r", "f78v", "f81r", "f81v"]
    assert ld[8:12] == ["f75r", "f75v", "f84r", "f84v"]
    assert ld[12:16] == ["f76r", "f76v", "f83r", "f83v"]
    assert ld[16:20] == ["f79r", "f79v", "f80r", "f80v"]


def test_quire_20_stars_reordering():
    q20_folios = [
        "f103r", "f103v", "f104r", "f104v", "f105r", "f105v",
        "f106r", "f106v", "f107r", "f107v", "f108r", "f108v",
        "f111r", "f111v", "f112r", "f112v", "f113r", "f113v",
        "f114r", "f114v", "f115r", "f115v", "f116r", "f116v"
    ]
    
    ld = get_layfield_davis_order(q20_folios)
    assert len(ld) == 24
    # Starts with 105|114 (initial oversized glyph on 105r)
    assert ld[:4] == ["f105r", "f105v", "f114r", "f114v"]
    assert ld[4:8] == ["f104r", "f104v", "f115r", "f115v"]
    assert ld[8:12] == ["f106r", "f106v", "f113r", "f113v"]
    assert ld[12:16] == ["f107r", "f107v", "f112r", "f112v"]
    assert ld[16:20] == ["f108r", "f108v", "f111r", "f111v"]
    # Ends with 103|116 (blank 116v at the very end)
    assert ld[20:24] == ["f103r", "f103v", "f116r", "f116v"]


def test_build_folio_codicology_registry():
    sample = ["f1r", "f1v", "f77r", "f82v", "f105r"]
    reg = build_folio_codicology_registry(sample)
    
    assert reg["f77r"]["quire"] == "Q13"
    assert reg["f77r"]["singulion_id"] == "77|82"
    assert reg["f77r"]["section"] == "Balneological (Biological)"
    assert reg["f77r"]["conjoint_leaf"] == 82
    
    assert reg["f105r"]["quire"] == "Q20"
    assert reg["f105r"]["singulion_id"] == "105|114"
    assert reg["f105r"]["section"] == "Marginal Stars / Recipes"
    assert reg["f1r"]["davis_scribe"] == 1
    assert reg["f77r"]["davis_scribe"] == 2
    assert reg["f105r"]["davis_scribe"] == 3


def test_davis_five_scribes_and_forensics():
    """Verifies Lisa Fagin Davis (2022) 5-scribe folio attributions, forensic ductus tells, and closed-loop recutting."""
    import numpy as np
    from voynich_ductus.ingestion.codicology import (
        DAVIS_FIVE_SCRIBES_CITATION,
        get_davis_scribe_attribution,
        extract_davis_scribal_forensics,
    )
    from voynich_ductus.embeddings.word_decomposition import GenerativeTypewriterDecomposer
    from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
    from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer

    assert DAVIS_FIVE_SCRIBES_CITATION["year"] == 2022
    assert get_davis_scribe_attribution("f1r") == 1
    assert get_davis_scribe_attribution("f26r") == 2
    assert get_davis_scribe_attribution("f103r") == 3
    assert get_davis_scribe_attribution("f68r1") == 4
    assert get_davis_scribe_attribution("f57r") == 5
    assert get_davis_scribe_attribution("f57v") == 1

    mock_glyphs = [
        {
            "glyph_id": "g_n",
            "eva_char": "n",
            "width_mm": 2.1,
            "height_mm": 1.8,
            "loop_count": 0,
            "strokes": [{"points": [(8.0, 12.0, 3.0), (22.0, 18.0, 3.5), (24.0, 24.0, 2.5), (10.0, 6.0, 2.0)], "px_per_mm": 10.0}],
        },
        {
            "glyph_id": "g_bridge",
            "eva_char": "cth",
            "width_mm": 6.2,
            "height_mm": 3.8,
            "loop_count": 2,
            "strokes": [],
        },
    ]
    forensics = extract_davis_scribal_forensics(mock_glyphs)
    assert forensics["n_count"] == 1
    assert forensics["n_single_stroke_ratio"] == 1.0
    assert forensics["n_mean_left_reach_mm"] > 0.0
    assert forensics["bridge_gallows_count"] == 1

    # Verify closed-loop Ductus -> Word Re-Segmentation splits an over-wide 5-minim fused compound
    cal = MetricScaleCalibration(manuscript_type="voynich", custom_px_per_mm=10.0)
    decomp = GenerativeTypewriterDecomposer(is_serafini=False)
    vec = CalligraphicVectorizer()
    wide_mask = np.zeros((24, 56), dtype=bool)
    for idx_m, cx in enumerate((5, 15, 25, 38, 48)):
        wide_mask[4:20, cx : cx + 3] = True
    wide_mask[18:20, 5:50] = True  # Thin baseline ligature connecting two multi-minim glyphs
    wide_mask[18:20, 28:35] = False
    wide_mask[19, 28:35] = True    # 1-px hairline valley at x=31
    slices = [{
        "mask": wide_mask,
        "bbox_local": (0, 0, 24, 56),
        "area": int(np.sum(wide_mask)),
        "diacritics": [],
        "is_dot": False,
        "typewriter_proto_id": "TW_00",
    }]
    recut = decomp.refine_glyph_slices_with_ductus(slices, vectorizer=vec, scale_calib=cal)
    assert len(recut) == 2

