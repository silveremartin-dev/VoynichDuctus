"""
Unit tests for SpatialWordPartitioner, WordScribalEncoder, GenerativeTypewriterDecomposer,
Physical Nib Impact Threshold, and Supralinear Diacritic Decoupling.
"""

import numpy as np
from voynich_ductus.ingestion.spatial_partitioner import SpatialWordPartitioner
from voynich_ductus.embeddings.word_decomposition import (
    WordScribalEncoder,
    GenerativeTypewriterDecomposer,
)
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter


def test_spatial_word_partitioner_resolves_all_overlaps():
    """Verifies that overlapping candidate word bounding boxes are strictly partitioned with 0 overlaps."""
    words = [
        {"word_id": "W01", "bbox": [100, 50, 150, 120]},
        {"word_id": "W02", "bbox": [100, 100, 150, 180]},
        {"word_id": "W03", "bbox": [140, 60, 190, 130]},
    ]
    partitioned = SpatialWordPartitioner.resolve_disjoint_word_partitions(words, (300, 300))
    assert len(partitioned) == 3

    for i in range(len(partitioned)):
        b1 = partitioned[i]["bbox"]
        for j in range(i + 1, len(partitioned)):
            b2 = partitioned[j]["bbox"]
            overlaps = not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])
            assert not overlaps, f"Words {partitioned[i]['word_id']} and {partitioned[j]['word_id']} still overlap!"


def test_physical_nib_impact_point_threshold():
    """Verifies that tiny dots/specks are classified as instantaneous touchdown impacts, not curved trajectories."""
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)

    dot_mask = np.zeros((20, 20), dtype=bool)
    dot_mask[8:13, 8:13] = True

    strokes = tracker.extract_calligraphic_ductus(dot_mask, single_stroke_mode=False)
    assert len(strokes) == 1
    s = strokes[0]
    assert s.get("is_impact") is True
    assert s.get("is_dot") is True
    assert s["length"] == 0.0
    assert len(s["points"]) == 1


def test_word_scribal_encoder_and_supralinear_diacritic_decoupling():
    """Verifies reading-axis Z(x) encoding and metric supralinear diacritic decoupling."""
    word_mask = np.zeros((40, 40), dtype=bool)
    word_mask[18:36, 12:28] = True
    word_mask[4:8, 18:22] = True

    encoder = WordScribalEncoder(latent_dim=32)
    traj = encoder.encode_word_patch(word_mask)
    assert traj.ndim == 2 and traj.shape[1] == 32

    segmenter = GlyphSegmenter()
    glyphs = segmenter.assemble_word_into_glyphs(word_mask)
    assert len(glyphs) >= 1
    base_g = glyphs[0]
    assert "diacritics" in base_g
    assert len(base_g["diacritics"]) == 1
    assert base_g["diacritics"][0]["type"] == "DOT"


def test_generative_typewriter_viterbi_decomposer():
    """Verifies GenerativeTypewriterDecomposer sprite dictionary induction & Viterbi decoding."""
    w1 = np.zeros((44, 120), dtype=bool)
    rr, cc = np.ogrid[:44, :120]
    o_ring = ((rr - 24) ** 2 + (cc - 18) ** 2 <= 9 ** 2) & (~((rr - 24) ** 2 + (cc - 18) ** 2 <= 4 ** 2))
    w1 |= o_ring
    w1[16:33, 42:49] = True
    w1[16:20, 49:53] = True
    w1[29:33, 49:53] = True
    w1[16:34, 55:60] = True
    w1[6:35, 82:88] = True

    tw = GenerativeTypewriterDecomposer(is_serafini=False)
    tw.fit_on_word_masks([w1, w1, w1, w1])
    out = tw.decompose_word(w1)
    assert len(out) == 3
    for g in out:
        assert g["segmentation_type"] == "GENERATIVE_TYPEWRITER"
