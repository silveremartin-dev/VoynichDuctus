"""
Unit tests for SpatialWordPartitioner, NeuralWordDecomposer,
and Physical Nib Impact Threshold / Diacritic Decoupling.
"""

import numpy as np
import pytest
from voynich_ductus.ingestion.spatial_partitioner import SpatialWordPartitioner
from voynich_ductus.embeddings.word_neural_decomposer import (
    WordScribalEncoder,
    NeuralWordDecomposer,
    HybridNeuroKinematicDecomposer,
    cross_evaluate_options_consensus
)
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter


def test_spatial_word_partitioner_resolves_all_overlaps():
    """Verifies that overlapping candidate word bounding boxes are strictly partitioned with 0 overlaps."""
    # Create two overlapping words on the same line
    words = [
        {"word_id": "W01", "bbox": [100, 50, 150, 120]},
        {"word_id": "W02", "bbox": [100, 100, 150, 180]},
        # And another word vertically overlapping
        {"word_id": "W03", "bbox": [140, 60, 190, 130]}
    ]
    partitioned = SpatialWordPartitioner.resolve_disjoint_word_partitions(words, (300, 300))
    assert len(partitioned) == 3

    # Check that no pair of bounding boxes overlap in 2D space
    for i in range(len(partitioned)):
        b1 = partitioned[i]["bbox"]
        for j in range(i + 1, len(partitioned)):
            b2 = partitioned[j]["bbox"]
            overlaps = not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])
            assert not overlaps, f"Words {partitioned[i]['word_id']} and {partitioned[j]['word_id']} still overlap!"


def test_physical_nib_impact_point_threshold():
    """Verifies that tiny dots/specks are classified as instantaneous touchdown impacts, not curved trajectories."""
    tracker = CalligraphicVectorizer(nib_angle_deg=40.0)
    
    # 5x5 dot
    dot_mask = np.zeros((20, 20), dtype=bool)
    dot_mask[8:13, 8:13] = True
    
    strokes = tracker.extract_calligraphic_ductus(dot_mask, single_stroke_mode=False)
    assert len(strokes) == 1
    s = strokes[0]
    assert s.get("is_impact") is True
    assert s.get("is_dot") is True
    assert s["length"] == 0.0
    assert len(s["points"]) == 1


def test_neural_word_encoder_and_decomposition():
    """Verifies that WordScribalEncoder and NeuralWordDecomposer partition words along latent transitions."""
    word_mask = np.zeros((36, 100), dtype=bool)
    # Three artificial characters
    word_mask[8:28, 5:25] = True
    word_mask[10:30, 35:55] = True
    word_mask[6:26, 65:90] = True

    encoder = WordScribalEncoder(latent_dim=32)
    traj = encoder.encode_word_patch(word_mask)
    assert traj.ndim == 2
    assert traj.shape[1] == 32

    decomposer = NeuralWordDecomposer()
    glyphs = decomposer.decompose_word(word_mask)
    assert len(glyphs) >= 2
    for g in glyphs:
        assert "mask" in g
        assert "bbox_local" in g
        assert g["segmentation_type"] == "NEURAL_WORD_PCA"


def test_supralinear_diacritic_decoupling():
    """Verifies that supralinear diacritics are decoupled as attributes rather than merged into base body mask."""
    segmenter = GlyphSegmenter()
    
    # Create a word with a base letter and a floating dot above it
    word_mask = np.zeros((40, 40), dtype=bool)
    # Base letter body (rows 18 to 36)
    word_mask[18:36, 12:28] = True
    # Floating supralinear dot (rows 4 to 8, area ~16 px)
    word_mask[4:8, 18:22] = True

    glyphs = segmenter.assemble_word_into_glyphs(word_mask)
    assert len(glyphs) >= 1
    # Check that diacritic is detected
    base_g = glyphs[0]
    assert "diacritics" in base_g
    assert len(base_g["diacritics"]) == 1
    assert base_g["diacritics"][0]["type"] == "DOT"


def test_hybrid_neuro_kinematic_consensus_option_c():
    """Verifies that HybridNeuroKinematicDecomposer (Option C) fuses Option A and Option B cuts."""
    word_mask = np.zeros((30, 80), dtype=bool)
    word_mask[8:25, 4:22] = True
    word_mask[8:25, 28:50] = True
    word_mask[8:25, 56:76] = True

    opt_a_glyphs = [
        {"bbox_local": (8, 4, 25, 24)},
        {"bbox_local": (8, 28, 25, 52)},
        {"bbox_local": (8, 56, 25, 76)}
    ]
    opt_b_glyphs = [
        {"bbox_local": (8, 4, 25, 26)},
        {"bbox_local": (8, 28, 25, 54)},
        {"bbox_local": (8, 56, 25, 76)}
    ]

    hybrid = HybridNeuroKinematicDecomposer(tolerance_px=4)
    opt_c_glyphs = hybrid.decompose_word_hybrid(word_mask, opt_a_glyphs, opt_b_glyphs)
    assert len(opt_c_glyphs) >= 2
    for g in opt_c_glyphs:
        assert g["segmentation_type"] == "HYBRID_CONSENSUS_C"


def test_cross_option_consensus_and_disjoint_tagging():
    """Verifies that cross_evaluate_options_consensus tags consensus and disjoint archetypes."""
    # Glyphs that match across all 3
    g_a = [{"bbox": [10, 10, 40, 40]}, {"bbox": [10, 50, 40, 80]}]
    g_b = [{"bbox": [10, 12, 40, 42]}, {"bbox": [10, 90, 40, 120]}]  # Second glyph is disjoint to B
    g_c = [{"bbox": [10, 11, 40, 41]}, {"bbox": [10, 50, 40, 80]}]

    cross_evaluate_options_consensus(g_a, g_b, g_c, iou_thresh=0.55)

    # First glyph is corroborated by A, B, and C
    assert g_a[0]["support_badge"] == "A+B+C (Consensus)"
    assert g_a[0]["is_consensus"] is True
    assert g_a[0]["is_disjoint"] is False

    # Second glyph in B only exists in B
    assert g_b[1]["support_badge"] == "B only (Disjoint)"
    assert g_b[1]["is_disjoint"] is True

