"""Unit tests for feature extraction, clustering, and tokenization."""

import pytest
import numpy as np
from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer


def test_geometric_features_and_projection():
    strokes = [
        {"points": [(0, 0, 1.0), (10, 0, 1.0), (20, 0, 1.0)]},
        {"points": [(0, 0, 1.5), (0, 10, 1.5), (0, 20, 1.5)]},
        {"points": [(5, 5, 2.0), (10, 10, 2.0), (15, 15, 2.0)]},
        {"points": [(0, 0, 1.0), (10, 0, 1.0), (20, 0, 1.0)]},
    ]

    extractor = GeometricFeatureExtractor()
    feat_matrix = extractor.extract_batch(strokes)
    assert feat_matrix.shape == (4, 16)

    projector = StrokeLatentProjector(latent_dim=4)
    latent = projector.fit_transform(feat_matrix)
    assert latent.shape == (4, 4)


def test_clustering_and_tokenization():
    # 2 distinct clusters
    data = np.array([
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.2],
        [10.0, 10.0], [9.9, 10.1], [10.2, 9.8]
    ])

    clusterer = GlyphClusterer(method="agglomerative", n_clusters=2)
    labels = clusterer.fit_predict(data)
    assert len(labels) == 6
    assert clusterer.get_cluster_count() == 2

    tokenizer = StrokeTokenizer(prefix="G")
    word_tok = tokenizer.tokenize_word(labels[:3].tolist())
    assert "G" in word_tok


def test_glyph_catalogue():
    from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue

    catalogue_builder = GlyphCatalogue(target_alphabet_size=3)
    
    mock_glyphs = [
        {"glyph_id": "G001", "page_id": "f001r", "bbox": [10, 10, 30, 20], "height": 20, "width": 10, "stroke_count": 1, "png_rel": "g1.png", "svg_rel": "g1.svg", "strokes": [{"points": [(0, 0, 1), (10, 0, 1)]}]},
        {"glyph_id": "G002", "page_id": "f001r", "bbox": [12, 10, 33, 21], "height": 21, "width": 11, "stroke_count": 1, "png_rel": "g2.png", "svg_rel": "g2.svg", "strokes": [{"points": [(0, 0, 1), (11, 0, 1)]}]},
        {"glyph_id": "G003", "page_id": "f001v", "bbox": [50, 40, 90, 70], "height": 40, "width": 30, "stroke_count": 3, "png_rel": "g3.png", "svg_rel": "g3.svg", "strokes": [{"points": [(0, 0, 2), (0, 20, 2)], "points": [(0, 10, 2), (20, 10, 2)]}]},
        {"glyph_id": "G004", "page_id": "f001v", "bbox": [52, 40, 94, 71], "height": 42, "width": 31, "stroke_count": 3, "png_rel": "g4.png", "svg_rel": "g4.svg", "strokes": [{"points": [(0, 0, 2), (0, 21, 2)], "points": [(0, 10, 2), (21, 10, 2)]}]},
    ]

    cat = catalogue_builder.build_catalogue(mock_glyphs, corpus_type="voynich")
    assert cat["total_glyphs"] == 4
    assert cat["canonical_alphabet_size"] >= 2
    assert len(cat["alphabet"]) >= 2
    for entry in cat["alphabet"]:
        assert "corpus_match" in entry
        assert "all_instances" in entry
        assert len(entry["all_instances"]) == entry["frequency"]


def test_corpus_matcher():
    from voynich_ductus.clustering.corpus_matcher import CorpusCorrespondenceMatcher

    match_v = CorpusCorrespondenceMatcher.match_voynich_archetype(
        archetype_id="G01",
        mean_strokes=1.1,
        mean_width=15,
        mean_height=18,
        frequency_rank=1
    )
    assert match_v["eva_equivalent"] in ["o", "a", "e", "y"]
    assert match_v["confidence_pct"] > 50

    match_s = CorpusCorrespondenceMatcher.match_serafini_archetype(
        archetype_id="G01",
        mean_strokes=2.0,
        mean_width=12,
        mean_height=20,
        frequency_rank=1
    )
    assert "serafini_code" in match_s
    assert match_s["confidence_pct"] > 50


