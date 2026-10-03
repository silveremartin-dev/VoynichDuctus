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
