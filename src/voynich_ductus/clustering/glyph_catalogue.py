"""
Glyph Catalogue and Canonical Alphabet Induction Engine.
Synthesizes unique canonical glyph archetypes from thousands of isolated glyph extractions.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector


class GlyphCatalogue:
    """
    Groups extracted individual glyphs into an objective canonical alphabet inventory
    via unsupervised geometric clustering.
    """

    def __init__(self, target_alphabet_size: int = 28, distance_threshold: Optional[float] = None):
        self.target_alphabet_size = target_alphabet_size
        self.distance_threshold = distance_threshold
        self.feature_extractor = GeometricFeatureExtractor()
        self.projector = StrokeLatentProjector(latent_dim=8)

    def extract_glyph_features(self, glyphs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts composite 16-D geometric descriptors for each glyph based on its strokes.
        """
        glyph_feature_vectors = []
        for g in glyphs:
            strokes = g.get("strokes", [])
            if not strokes:
                # Fallback zero vector
                glyph_feature_vectors.append(np.zeros(16, dtype=np.float32))
                continue

            # Extract features for all strokes in glyph
            s_feats = self.feature_extractor.extract_batch(strokes)
            # Pool stroke features (mean + max + stroke_count descriptor)
            mean_f = np.mean(s_feats, axis=0)
            max_f = np.max(s_feats, axis=0)
            
            # Combine into robust descriptor
            composite = 0.6 * mean_f + 0.4 * max_f
            composite[0] = len(strokes) / 10.0  # Normalize stroke count
            glyph_feature_vectors.append(composite)

        return np.array(glyph_feature_vectors, dtype=np.float32)

    def build_catalogue(self, glyphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Clusters glyphs into canonical alphabet types and identifies exemplar archetypes.
        """
        if not glyphs:
            return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

        X = self.extract_glyph_features(glyphs)
        
        # Fit latent projector
        latent_dim = min(8, X.shape[1], max(2, X.shape[0] // 2))
        self.projector = StrokeLatentProjector(latent_dim=latent_dim)
        X_latent = self.projector.fit_transform(X)

        # Cluster into unique canonical glyph types
        n_clusters = min(self.target_alphabet_size, len(glyphs))
        if self.distance_threshold is not None:
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold, metric="euclidean", linkage="ward")
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

        labels = clusterer.fit_predict(X_latent)
        unique_labels = sorted(list(set(labels)))
        actual_clusters = len(unique_labels)

        # Build canonical glyph entries
        alphabet_entries = []
        for cluster_id in unique_labels:
            indices = np.where(labels == cluster_id)[0]
            cluster_glyphs = [glyphs[i] for i in indices]
            cluster_points = X_latent[indices]

            # Find medoid (exemplar closest to cluster center)
            centroid = np.mean(cluster_points, axis=0, keepdims=True)
            dists = pairwise_distances(cluster_points, centroid).flatten()
            medoid_idx = indices[np.argmin(dists)]
            exemplar_glyph = glyphs[medoid_idx]

            type_name = f"G{cluster_id + 1:02d}"

            # Tag all members with canonical type
            for g in cluster_glyphs:
                g["canonical_type"] = type_name

            alphabet_entries.append({
                "type_id": type_name,
                "cluster_index": int(cluster_id),
                "frequency": len(cluster_glyphs),
                "percentage": round(len(cluster_glyphs) / len(glyphs) * 100, 2),
                "exemplar_id": exemplar_glyph["glyph_id"],
                "exemplar_png": exemplar_glyph["png_rel"],
                "exemplar_svg": exemplar_glyph["svg_rel"],
                "exemplar_svg_content": exemplar_glyph.get("svg_content", ""),
                "mean_strokes": round(float(np.mean([g["stroke_count"] for g in cluster_glyphs])), 2),
                "mean_height": round(float(np.mean([g["height"] for g in cluster_glyphs])), 1),
                "mean_width": round(float(np.mean([g["width"] for g in cluster_glyphs])), 1),
                "sample_instances": [g["glyph_id"] for g in cluster_glyphs[:20]]
            })

        # Sort alphabet by frequency descending
        alphabet_entries.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "total_glyphs": len(glyphs),
            "canonical_alphabet_size": actual_clusters,
            "alphabet": alphabet_entries
        }

    def export_catalogue_json(self, catalogue: Dict[str, Any], output_path: Path):
        """Exports the canonical alphabet catalogue as a clean JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2, ensure_ascii=False)
