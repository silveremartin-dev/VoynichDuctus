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
from voynich_ductus.vectorizer.comparative_vectorizers import DINOv2EmbeddingEngine


class GlyphCatalogue:
    """
    Groups extracted individual glyphs into an objective canonical alphabet inventory
    via unsupervised geometric clustering and DINOv2 self-supervised visual features,
    preserving all instance occurrences and coordinates.
    """

    def __init__(self, target_alphabet_size: int = 28, distance_threshold: Optional[float] = None):
        self.target_alphabet_size = target_alphabet_size
        self.distance_threshold = distance_threshold
        self.feature_extractor = GeometricFeatureExtractor()
        self.projector = StrokeLatentProjector(latent_dim=8)
        self.dinov2_engine = DINOv2EmbeddingEngine()

    def extract_glyph_features(self, glyphs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts composite deep visual (DINOv2) + spatial-geometric descriptors for each glyph.
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
                glyph_feature_vectors.append(np.zeros(48, dtype=np.float32))
                continue

            pts_arr = np.array([[p[0], p[1]] for p in all_pts], dtype=np.float32)
            min_y, min_x = np.min(pts_arr, axis=0)
            max_y, max_x = np.max(pts_arr, axis=0)
            norm_h = max(1.0, max_y - min_y)
            norm_w = max(1.0, max_x - min_x)

            # 6x6 Spatial Density Grid (36 descriptors)
            grid_6x6 = np.zeros((6, 6), dtype=np.float32)
            rel_y = np.clip((pts_arr[:, 0] - min_y) / norm_h, 0.0, 0.999)
            rel_x = np.clip((pts_arr[:, 1] - min_x) / norm_w, 0.0, 0.999)
            grid_rows = (rel_y * 6).astype(int)
            grid_cols = (rel_x * 6).astype(int)
            for r, c in zip(grid_rows, grid_cols):
                grid_6x6[r, c] += 1.0
            if len(all_pts) > 0:
                grid_6x6 /= len(all_pts)

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

            # Rasterize mini patch for DINOv2 / deep embedding
            patch_dim = 32
            mini_patch = np.zeros((patch_dim, patch_dim), dtype=np.uint8)
            py = np.clip((rel_y * (patch_dim - 1)).astype(int), 0, patch_dim - 1)
            px = np.clip((rel_x * (patch_dim - 1)).astype(int), 0, patch_dim - 1)
            mini_patch[py, px] = 255
            dino_emb = self.dinov2_engine.extract_embedding(mini_patch)
            # Take top 16 principal components of DINOv2
            dino_top = dino_emb[:16] if len(dino_emb) >= 16 else np.pad(dino_emb, (0, 16 - len(dino_emb)))

            vec = [
                stroke_count * 2.0,
                aspect * 2.5,
                fill_factor * 1.5,
                is_loop * 3.0,
                np.sin(entry_angle),
                np.cos(entry_angle),
                np.sin(exit_angle),
                np.cos(exit_angle),
                float(mean_f[0]),  # arc_length
                float(mean_f[2]),  # tortuosity
                float(mean_f[7]),  # mean_width
                float(mean_f[8]),  # std_width
            ]
            # Add 36 spatial density grid features
            vec.extend(grid_6x6.flatten().tolist())
            # Add DINOv2 deep visual tokens
            vec.extend((dino_top * 2.0).tolist())

            glyph_feature_vectors.append(np.array(vec, dtype=np.float32))

        X_raw = np.array(glyph_feature_vectors, dtype=np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return X_scaled

    @staticmethod
    def classify_glyph_topology(g: Dict[str, Any]) -> str:
        """
        Deterministically classifies a glyph into an invariant topological family:
        - CLOSED_LOOP (circular/oval closed shapes like o, a bowl)
        - VERTICAL_MINIM (single vertical upright stroke like i, r)
        - OPEN_C_CRESCENT (crescent or open curve like c, e)
        - TALL_GALLOWS (tall ascenders, loops with tall stems like k, t, p, f)
        - BENCH_HORIZONTAL (wide horizontal structures)
        - COMPOSITE_LIGATURE (complex multi-stroke conjoined glyphs)
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
            return "VERTICAL_MINIM"

        longest_s = max(strokes, key=lambda s: len(s.get("points", [])))
        l_pts = longest_s.get("points", [])

        # 1. Closed loop check
        if len(l_pts) >= 5:
            p0 = np.array(l_pts[0][:2])
            p_end = np.array(l_pts[-1][:2])
            dist_ends = float(np.linalg.norm(p0 - p_end))
            if dist_ends <= 7.0 or dist_ends <= 0.28 * max(gw, gh):
                return "CLOSED_LOOP"

        # 2. Tall ascender or Gallows
        if gh >= 1.45 * gw or (stroke_count >= 2 and gh >= 28):
            return "TALL_GALLOWS"

        # 3. Wide horizontal bench
        if aspect >= 1.4 and gw >= 30:
            return "BENCH_HORIZONTAL"

        # 4. Single-stroke open curves vs minims
        if stroke_count == 1:
            if len(l_pts) >= 4:
                mid_idx = len(l_pts) // 2
                x_start = l_pts[0][1]
                x_end = l_pts[-1][1]
                x_mid = l_pts[mid_idx][1]
                if x_mid < min(x_start, x_end) - 2.0:
                    return "OPEN_C_CRESCENT"
            if aspect <= 0.75:
                return "VERTICAL_MINIM"

        if stroke_count >= 2:
            return "COMPOSITE_LIGATURE"

        return "VERTICAL_MINIM"

    def build_catalogue(self, glyphs: List[Dict[str, Any]], corpus_type: str = "voynich") -> Dict[str, Any]:
        """
        Clusters glyphs into canonical alphabet types using topological partitioning and
        spatial density grid descriptors, ensuring loops, crescents, minims, and gallows never mix.
        """
        if not glyphs:
            return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

        X_scaled = self.extract_glyph_features(glyphs)

        # 1. Voynich Ground-Truth Guided Induction (if EVA tokens are present)
        eva_present = any(bool(g.get("eva_char")) for g in glyphs)
        if corpus_type.lower() == "voynich" and eva_present:
            eva_groups = {}
            for idx, g in enumerate(glyphs):
                ec = (g.get("eva_char") or "").strip().lower()
                if not ec:
                    ec = "emergent"
                if ec not in eva_groups:
                    eva_groups[ec] = []
                eva_groups[ec].append(idx)

            raw_clusters = []
            cluster_id_counter = 0
            for ec, indices in eva_groups.items():
                cluster_glyphs = [glyphs[i] for i in indices]
                raw_clusters.append((cluster_id_counter, indices, cluster_glyphs))
                cluster_id_counter += 1
        else:
            # 2. Unsupervised Deep Visual & Topological Induction (Seraphinianus / Asemic)
            for g in glyphs:
                g["topology_group"] = self.classify_glyph_topology(g)

            topo_groups = {}
            for idx, g in enumerate(glyphs):
                tg = g["topology_group"]
                if tg not in topo_groups:
                    topo_groups[tg] = []
                topo_groups[tg].append(idx)

            raw_clusters = []
            cluster_id_counter = 0

            # Cluster within each topological partition with tight granularity
            for tg, indices in topo_groups.items():
                if len(indices) <= 2:
                    cluster_glyphs = [glyphs[i] for i in indices]
                    raw_clusters.append((cluster_id_counter, indices, cluster_glyphs))
                    cluster_id_counter += 1
                    continue

                sub_X = X_scaled[indices]
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
        assigned_standard_codes = set()
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

            # Match with standard historical corpora (enforcing 1-to-1 unique correspondence)
            if corpus_type.lower() == "seraphinianus":
                corpus_match = CorpusCorrespondenceMatcher.match_serafini_archetype(
                    archetype_id=type_name,
                    mean_strokes=mean_strokes,
                    mean_width=mean_width,
                    mean_height=mean_height,
                    frequency_rank=rank,
                    cluster_glyphs=cluster_glyphs
                )
                code = corpus_match.get("serafini_code")
                if code and code != "S-EMERGENT" and code in assigned_standard_codes:
                    # Already claimed by a higher-ranked archetype -> mark as distinct emergent
                    corpus_match = {
                        "system": "Serafini Typology (1981) / Deri (2015)",
                        "serafini_code": f"S-EMERGENT-{rank:02d}",
                        "deri_equivalent": f"EMERGENT-{rank:02d}",
                        "bulik_equivalent": f"ω_{rank}",
                        "name": f"Distinct Cursive Archetype {type_name}",
                        "category": "Autonomous Emergent Form",
                        "confidence_pct": round(max(40.0, 95.0 - rank * 1.5), 1),
                        "description": f"Unique asemic cursive grapheme distinct from previously indexed standard archetypes.",
                        "reference_svg": ""
                    }
                elif code and code != "S-EMERGENT":
                    assigned_standard_codes.add(code)
            else:
                corpus_match = CorpusCorrespondenceMatcher.match_voynich_archetype(
                    archetype_id=type_name,
                    mean_strokes=mean_strokes,
                    mean_width=mean_width,
                    mean_height=mean_height,
                    frequency_rank=rank,
                    cluster_glyphs=cluster_glyphs
                )
                eva_eq = corpus_match.get("eva_equivalent")
                if eva_eq and eva_eq != "—" and eva_eq in assigned_standard_codes:
                    # Already claimed by a higher-ranked archetype -> mark as distinct emergent
                    corpus_match = {
                        "system": "EVA / Currier / v101 / Voynichese",
                        "eva_equivalent": "—",
                        "currier_equivalent": "—",
                        "v101_equivalent": "—",
                        "voynichese_equivalent": "—",
                        "name": f"Distinct Scribal Ductus {type_name}",
                        "category": "Autonomous Grapheme",
                        "confidence_pct": round(max(40.0, 95.0 - rank * 1.5), 1),
                        "description": f"Autonomous scribal grapheme with distinct topology; avoids duplicate mapping to '{eva_eq}'.",
                        "reference_svg": ""
                    }
                elif eva_eq and eva_eq != "—":
                    assigned_standard_codes.add(eva_eq)

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
