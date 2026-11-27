"""
Feature extraction, unsupervised word decomposition, and latent embedding module.
"""

from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.embeddings.word_decomposition import (
    WordScribalEncoder,
    GenerativeTypewriterDecomposer,
)

__all__ = [
    "GeometricFeatureExtractor",
    "StrokeLatentProjector",
    "WordScribalEncoder",
    "GenerativeTypewriterDecomposer",
]
