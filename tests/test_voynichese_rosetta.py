"""
Unit tests for VoynicheseRosettaLoader and Ground Truth Annotation Engine.
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader, RosettaWord


@pytest.fixture
def rosetta_loader():
    return VoynicheseRosettaLoader(
        annotations_dir="data/annotations/voynichese",
        scans_dir="data/scans/voynich/yale",
    )


def test_list_available_folios(rosetta_loader):
    folios = rosetta_loader.list_available_folios()
    assert len(folios) == 225
    assert "f1r" in folios
    assert "f116r" in folios
    # Check natural sorting order: f1r comes before f10r
    f1_idx = folios.index("f1r")
    f10_idx = folios.index("f10r")
    assert f1_idx < f10_idx


def test_load_folio_words(rosetta_loader):
    words = rosetta_loader.load_folio_words("f1r")
    assert len(words) == 210
    
    # Check word 0
    w0 = words[0]
    assert isinstance(w0, RosettaWord)
    assert w0.index == 0
    assert w0.eva_text == "fachys"
    assert w0.x_xml == 97.0
    assert w0.y_xml == 128.0
    assert w0.width_xml == 106.0
    assert w0.height_xml == 96.0

    # Check word 1
    w1 = words[1]
    assert w1.eva_text == "ykal"

    # Check dictionary serialization
    d = w0.to_dict()
    assert d["eva_text"] == "fachys"
    assert d["folio"] == "f1r"


def test_tokenize_eva_to_glyphs(rosetta_loader):
    # Test compound gallows and complex ligatures
    assert rosetta_loader.tokenize_eva_to_glyphs("fachys") == ["f", "a", "ch", "y", "s"]
    assert rosetta_loader.tokenize_eva_to_glyphs("cthaiin") == ["cth", "a", "ii", "n"]
    assert rosetta_loader.tokenize_eva_to_glyphs("sholdy") == ["sh", "o", "l", "d", "y"]
    assert rosetta_loader.tokenize_eva_to_glyphs("cfhoaiin") == ["cfh", "o", "a", "ii", "n"]
    assert rosetta_loader.tokenize_eva_to_glyphs("") == []


def test_affine_transform_computation(rosetta_loader):
    dummy_img = Image.new("RGB", (2500, 3168), color=(240, 230, 200))
    sx, sy, tx, ty = rosetta_loader.compute_scan_affine_transform("f1r", dummy_img, xml_canvas=(1090.0, 1500.0))
    
    assert sx > 1.9
    assert sy > 1.9
    assert tx >= 0
    assert ty >= 0


def test_extract_aligned_words(rosetta_loader):
    # Create a synthetic scan image
    test_img = Image.new("RGB", (2500, 3168), color=(220, 210, 190))
    aligned = rosetta_loader.extract_aligned_words("f1r", scan_image=test_img, padding_px=2)
    
    assert len(aligned) == 210
    w0 = aligned[0]
    assert w0.bbox_scan is not None
    assert len(w0.bbox_scan) == 4
    min_x, min_y, max_x, max_y = w0.bbox_scan
    assert max_x > min_x
    assert max_y > min_y
    assert w0.image_patch is not None
    assert w0.image_patch.size == (max_x - min_x, max_y - min_y)


def test_corpus_statistics(rosetta_loader):
    stats = rosetta_loader.get_full_corpus_statistics()
    assert stats["total_folios"] == 225
    assert stats["total_words"] > 30000
    assert stats["unique_words"] > 5000
    assert stats["unique_glyphs"] > 20
    assert len(stats["top_words"]) == 25
    assert len(stats["top_glyphs"]) == 25
