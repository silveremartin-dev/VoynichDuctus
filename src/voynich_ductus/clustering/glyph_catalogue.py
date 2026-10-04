"""
Glyph Catalogue and Canonical Alphabet Induction Engine.
Synthesizes unique canonical glyph archetypes from thousands of isolated glyph extractions,
indexes all spatial instances (x, y coordinates across all pages),
and cross-references each archetype against historical reference transliteration corpora (EVA, Currier, Serafini).
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.corpus_matcher import CorpusCorrespondenceMatcher


class GlyphCatalogue:
    """
    Groups extracted individual glyphs into an objective canonical alphabet inventory
    via unsupervised geometric clustering, preserving all instance occurrences and coordinates.
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

    def build_catalogue(self, glyphs: List[Dict[str, Any]], corpus_type: str = "voynich") -> Dict[str, Any]:
        """
        Clusters glyphs into canonical alphabet types, indexes all instances with spatial coordinates,
        and estimates correspondences with standard transcription corpora.
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

        # Build initial clusters
        raw_clusters = []
        for cluster_id in unique_labels:
            indices = np.where(labels == cluster_id)[0]
            cluster_glyphs = [glyphs[i] for i in indices]
            raw_clusters.append((cluster_id, indices, cluster_glyphs))

        # Sort raw clusters by size descending to have clean rank 1..N
        raw_clusters.sort(key=lambda item: len(item[2]), reverse=True)

        # Build canonical glyph entries
        alphabet_entries = []
        for rank, (cluster_id, indices, cluster_glyphs) in enumerate(raw_clusters, start=1):
            cluster_points = X_latent[indices]

            # Find medoid (exemplar closest to cluster center)
            centroid = np.mean(cluster_points, axis=0, keepdims=True)
            dists = pairwise_distances(cluster_points, centroid).flatten()
            medoid_idx = indices[np.argmin(dists)]
            exemplar_glyph = glyphs[medoid_idx]

            type_name = f"G{rank:02d}"

            # Tag all members with canonical type
            for g in cluster_glyphs:
                g["canonical_type"] = type_name

            mean_strokes = round(float(np.mean([g.get("stroke_count", len(g.get("strokes", []))) for g in cluster_glyphs])), 2)
            mean_height = round(float(np.mean([g.get("height", 20) for g in cluster_glyphs])), 1)
            mean_width = round(float(np.mean([g.get("width", 15) for g in cluster_glyphs])), 1)

            # Match with standard historical corpora
            if corpus_type.lower() == "seraphinianus":
                corpus_match = CorpusCorrespondenceMatcher.match_serafini_archetype(
                    archetype_id=type_name,
                    mean_strokes=mean_strokes,
                    mean_width=mean_width,
                    mean_height=mean_height,
                    frequency_rank=rank,
                    cluster_glyphs=cluster_glyphs
                )
            else:
                corpus_match = CorpusCorrespondenceMatcher.match_voynich_archetype(
                    archetype_id=type_name,
                    mean_strokes=mean_strokes,
                    mean_width=mean_width,
                    mean_height=mean_height,
                    frequency_rank=rank,
                    cluster_glyphs=cluster_glyphs
                )

            # Collect full occurrences / spatial instances across all pages
            all_instances = []
            for g in cluster_glyphs:
                all_instances.append({
                    "glyph_id": g.get("glyph_id", ""),
                    "page_id": g.get("page_id", ""),
                    "line_id": g.get("line_id", ""),
                    "word_id": g.get("word_id", ""),
                    "bbox": [int(x) for x in g.get("bbox", [0, 0, 0, 0])],
                    "width": int(g.get("width", 0)),
                    "height": int(g.get("height", 0)),
                    "stroke_count": int(g.get("stroke_count", len(g.get("strokes", [])))),
                    "png_rel": g.get("png_rel", ""),
                    "svg_rel": g.get("svg_rel", "")
                })

            alphabet_entries.append({
                "type_id": type_name,
                "rank": rank,
                "cluster_index": int(cluster_id),
                "frequency": len(cluster_glyphs),
                "percentage": round(len(cluster_glyphs) / len(glyphs) * 100, 2),
                "exemplar_id": exemplar_glyph.get("glyph_id", ""),
                "exemplar_png": exemplar_glyph.get("png_rel", ""),
                "exemplar_svg": exemplar_glyph.get("svg_rel", ""),
                "exemplar_svg_content": exemplar_glyph.get("svg_content", ""),
                "mean_strokes": mean_strokes,
                "mean_height": mean_height,
                "mean_width": mean_width,
                "corpus_match": corpus_match,
                "total_instances_count": len(all_instances),
                "all_instances": all_instances,
                "sample_instances": [g["glyph_id"] for g in cluster_glyphs[:20]]
            })

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
