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
