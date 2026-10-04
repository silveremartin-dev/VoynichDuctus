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
        Extracts composite 32-D spatial-geometric descriptors for each glyph based on its strokes,
        aspect ratio, loop topology, and normalized 4x4 spatial density grid.
        """
        from sklearn.preprocessing import StandardScaler

        glyph_feature_vectors = []
        for g in glyphs:
            strokes = g.get("strokes", [])
            gw = max(1.0, float(g.get("width", 20)))
            gh = max(1.0, float(g.get("height", 20)))
            stroke_count = float(len(strokes)) if strokes else 1.0

            aspect = float(np.log(gw / gh))
            fill_factor = float(g.get("fill_factor", g.get("area", 20) / (gw * gh)))

            # Collect all points
            all_pts = []
            for s in strokes:
                all_pts.extend(s.get("points", []))

            if not all_pts:
                glyph_feature_vectors.append(np.zeros(32, dtype=np.float32))
                continue

            pts_arr = np.array([[p[0], p[1]] for p in all_pts], dtype=np.float32)
            min_y, min_x = np.min(pts_arr, axis=0)
            max_y, max_x = np.max(pts_arr, axis=0)
            norm_h = max(1.0, max_y - min_y)
            norm_w = max(1.0, max_x - min_x)

            # 4x4 Spatial Density Grid (16 descriptors)
            grid_4x4 = np.zeros((4, 4), dtype=np.float32)
            rel_y = np.clip((pts_arr[:, 0] - min_y) / norm_h, 0.0, 0.999)
            rel_x = np.clip((pts_arr[:, 1] - min_x) / norm_w, 0.0, 0.999)
            grid_rows = (rel_y * 4).astype(int)
            grid_cols = (rel_x * 4).astype(int)
            for r, c in zip(grid_rows, grid_cols):
                grid_4x4[r, c] += 1.0
            if len(all_pts) > 0:
                grid_4x4 /= len(all_pts)

            # Tangents
            entry_s = strokes[0].get("points", [])
            exit_s = strokes[-1].get("points", [])
            if len(entry_s) >= 2:
                dy = entry_s[1][0] - entry_s[0][0]
                dx = entry_s[1][1] - entry_s[0][1]
                entry_angle = float(np.arctan2(dy, dx))
            else:
                entry_angle = 0.0

            if len(exit_s) >= 2:
                dy_ex = exit_s[-1][0] - exit_s[-2][0]
                dx_ex = exit_s[-1][1] - exit_s[-2][1]
                exit_angle = float(np.arctan2(dy_ex, dx_ex))
            else:
                exit_angle = 0.0

            # Loop detection (distance between start and end of longest stroke)
            longest_s = max(strokes, key=lambda s: len(s.get("points", [])))
            l_pts = longest_s.get("points", [])
            if len(l_pts) >= 4:
                p0 = np.array(l_pts[0][:2])
                p_end = np.array(l_pts[-1][:2])
                is_loop = 1.0 if np.linalg.norm(p0 - p_end) <= 8.0 else 0.0
            else:
                is_loop = 0.0

            # Stroke feature extraction
            s_feats = self.feature_extractor.extract_batch(strokes)
            mean_f = np.mean(s_feats, axis=0) if len(s_feats) > 0 else np.zeros(16, dtype=np.float32)

            vec = [
                stroke_count,
                aspect,
                fill_factor,
                is_loop,
                np.sin(entry_angle),
                np.cos(entry_angle),
                np.sin(exit_angle),
                np.cos(exit_angle),
                float(mean_f[0]),  # arc_length
                float(mean_f[2]),  # tortuosity
                float(mean_f[7]),  # mean_width
                float(mean_f[8]),  # std_width
            ]
            # Add 16 spatial grid features
            vec.extend(grid_4x4.flatten().tolist())
            # Add 4 direction bins
            vec.extend(mean_f[11:15].tolist())

            glyph_feature_vectors.append(np.array(vec, dtype=np.float32))

        X_raw = np.array(glyph_feature_vectors, dtype=np.float32)
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return X_scaled

    @staticmethod
    def classify_glyph_topology(g: Dict[str, Any]) -> str:
        """
        Deterministically classifies a glyph into an invariant topological family:
        - LOOP (closed loop o, a, d, q)
        - OPEN_C (open right-facing cavity c, e)
        - MINIM (vertical single minim i, r, n)
        - GALLOWS (tall ascender or gallows crossbar k, t, p, f)
        - OTHER (flourishes, complex ligatures)
        """
        strokes = g.get("strokes", [])
        gw = max(1.0, float(g.get("width", 20)))
        gh = max(1.0, float(g.get("height", 20)))
        aspect = gw / gh
        stroke_count = len(strokes)

        all_pts = []
        for s in strokes:
            all_pts.extend(s.get("points", []))

        if not all_pts:
            return "OTHER"

        pts_arr = np.array([[p[0], p[1]] for p in all_pts], dtype=np.float32)
        min_y, min_x = np.min(pts_arr, axis=0)

        longest_s = max(strokes, key=lambda s: len(s.get("points", [])))
        l_pts = longest_s.get("points", [])

        # 1. Check for closed loop:
        if len(l_pts) >= 5:
            p0 = np.array(l_pts[0][:2])
            p_end = np.array(l_pts[-1][:2])
            dist_ends = float(np.linalg.norm(p0 - p_end))
            if dist_ends <= 8.0 or dist_ends <= 0.32 * max(gw, gh):
                return "LOOP"

        # 2. Check for Open C vs Minim:
        if stroke_count == 1:
            if len(l_pts) >= 4:
                mid_idx = len(l_pts) // 2
                x_start = l_pts[0][1]
                x_end = l_pts[-1][1]
                x_mid = l_pts[mid_idx][1]
                if x_mid < min(x_start, x_end) - 1.5:
                    return "OPEN_C"
            if aspect <= 0.65:
                return "MINIM"

        # 3. Gallows / crossbars
        if stroke_count >= 2 or gh > 1.35 * gw:
            return "GALLOWS"

        return "OTHER"

    def build_catalogue(self, glyphs: List[Dict[str, Any]], corpus_type: str = "voynich") -> Dict[str, Any]:
        """
        Clusters glyphs into canonical alphabet types using topological partitioning and
        spatial density grid descriptors, ensuring loops, crescents, minims, and gallows never mix.
        """
        if not glyphs:
            return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

        # Tag each glyph with its topological group
        for g in glyphs:
            g["topology_group"] = self.classify_glyph_topology(g)

        topo_groups = {}
        for idx, g in enumerate(glyphs):
            tg = g["topology_group"]
            if tg not in topo_groups:
                topo_groups[tg] = []
            topo_groups[tg].append(idx)

        X_scaled = self.extract_glyph_features(glyphs)

        raw_clusters = []
        cluster_id_counter = 0

        # Cluster within each topological partition
        for tg, indices in topo_groups.items():
            if len(indices) <= 2:
                # Group small partitions directly
                cluster_glyphs = [glyphs[i] for i in indices]
                raw_clusters.append((cluster_id_counter, indices, cluster_glyphs))
                cluster_id_counter += 1
                continue

            sub_X = X_scaled[indices]
            # Determine partition sub-clusters proportional to population
            k_sub = max(1, min(int(round(self.target_alphabet_size * (len(indices) / len(glyphs)))), len(indices)))
            
            if k_sub == 1:
                cluster_glyphs = [glyphs[i] for i in indices]
                raw_clusters.append((cluster_id_counter, indices, cluster_glyphs))
                cluster_id_counter += 1
            else:
                clusterer = AgglomerativeClustering(n_clusters=k_sub, metric="euclidean", linkage="ward")
                sub_labels = clusterer.fit_predict(sub_X)
                for sl in set(sub_labels):
                    sub_idx = [indices[i] for i in np.where(sub_labels == sl)[0]]
                    cluster_glyphs = [glyphs[i] for i in sub_idx]
                    raw_clusters.append((cluster_id_counter, sub_idx, cluster_glyphs))
                    cluster_id_counter += 1

        # Sort raw clusters by size descending to have clean rank 1..N
        raw_clusters.sort(key=lambda item: len(item[2]), reverse=True)

        # Build canonical glyph entries
        alphabet_entries = []
        for rank, (cluster_id, indices, cluster_glyphs) in enumerate(raw_clusters, start=1):
            cluster_points = X_scaled[indices]

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
            "canonical_alphabet_size": len(alphabet_entries),
            "alphabet": alphabet_entries
        }

    def export_catalogue_json(self, catalogue: Dict[str, Any], output_path: Path):
        """Exports the canonical alphabet catalogue as a clean JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2, ensure_ascii=False)
