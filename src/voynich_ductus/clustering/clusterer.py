"""
Unsupervised clustering of stroke primitives into an emergent, data-driven alphabet.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


class GlyphClusterer:
    """
    Groups stroke vectors into discrete glyph/ligature classes.
    """

    def __init__(self, method: str = "dbscan", eps: float = 1.2, min_samples: int = 3, n_clusters: int = 30):
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = {}

    def fit_predict(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Clusters latent representations into glyph IDs (0..K-1, or -1 for outliers).
        """
        if len(latent_vectors) == 0:
            return np.array([], dtype=int)

        if len(latent_vectors) < self.min_samples:
            # Not enough samples for DBSCAN
            return np.zeros(len(latent_vectors), dtype=int)

        if self.method == "dbscan":
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.labels_ = model.fit_predict(latent_vectors)
        elif self.method == "agglomerative":
            k = min(self.n_clusters, len(latent_vectors))
            model = AgglomerativeClustering(n_clusters=k)
            self.labels_ = model.fit_predict(latent_vectors)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Compute cluster centroids
        unique_labels = set(self.labels_)
        for label_id in unique_labels:
            if label_id == -1:
                continue
            mask = (self.labels_ == label_id)
            self.cluster_centers_[int(label_id)] = np.mean(latent_vectors[mask], axis=0)

        return self.labels_

    def get_cluster_count(self) -> int:
        """Returns number of discovered clusters (excluding noise -1)."""
        if self.labels_ is None:
            return 0
        return len([lbl for lbl in set(self.labels_) if lbl != -1])
