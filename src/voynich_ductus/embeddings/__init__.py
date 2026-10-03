"""
Feature extraction and latent embedding module for stroke subgraphs and geometric primitives.
"""

from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector

__all__ = ["GeometricFeatureExtractor", "StrokeLatentProjector"]
