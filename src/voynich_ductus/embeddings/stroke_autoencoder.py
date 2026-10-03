"""
Latent projection and dimensionality reduction for stroke features (PCA/Autoencoder/UMAP interface).
"""

from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class StrokeLatentProjector:
    """
    Projects multidimensional stroke feature spaces into normalized latent representations.
    """

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=latent_dim)
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> "StrokeLatentProjector":
        if len(features) < self.latent_dim:
            self.latent_dim = max(1, len(features))
            self.pca = PCA(n_components=self.latent_dim)

        scaled = self.scaler.fit_transform(features)
        self.pca.fit(scaled)
        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            self.fit(features)
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
