"""
Feature extraction, word decomposition, and latent embedding module for stroke subgraphs and geometric primitives.
"""

from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.embeddings.word_decomposition import (
    WordScribalEncoder,
    NeuralWordDecomposition,
    HybridConsensusWordDecomposition,
    NeuralWordDecomposer,
    HybridNeuroKinematicDecomposer,
    cross_evaluate_options_consensus,
)

__all__ = [
    "GeometricFeatureExtractor",
    "StrokeLatentProjector",
    "WordScribalEncoder",
    "NeuralWordDecomposition",
    "HybridConsensusWordDecomposition",
    "NeuralWordDecomposer",
    "HybridNeuroKinematicDecomposer",
    "cross_evaluate_options_consensus",
]
