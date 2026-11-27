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
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

class GlyphCatalogue:
    """
    Groups extracted individual glyphs into an objective, naturally emergent canonical alphabet
    via unsupervised topological stratification, kinematic ductus features, and DINOv2 visual
    embeddings (selected automatically via Fisher separation + Kneedle rate-distortion curvature,
    with zero hardcoded target alphabet size).
    """

    def __init__(self, target_alphabet_size: Optional[int] = None, distance_threshold: Optional[float] = None):
        self.target_alphabet_size = target_alphabet_size
        self.distance_threshold = distance_threshold
        self.feature_extractor = GeometricFeatureExtractor()
        self.projector = StrokeLatentProjector(latent_dim=8)
        self.dinov2_engine = DINOv2EmbeddingEngine()
        self._dino_patch_cache: Dict[bytes, np.ndarray] = {}

    @staticmethod
    def count_glyph_loops(g: Dict[str, Any], stroke_key: str = "strokes") -> int:
        """
        Deterministically counts the exact number of closed topological loops (0, 1, 2+)
        in a glyph using both its binary image mask (Euler holes) and stroke trajectory
        self-intersections / closures.
        """
        from scipy.ndimage import binary_fill_holes, binary_closing, label as nd_label

        # Estimate scan pixel density from metric height when available so grain pinholes on HD scans
        # (~4 px = 0.02 mm^2 at 14 px/mm) are never mistaken for genuine scribal loops (>= 0.065 mm^2)
        gw = max(4, int(round(float(g.get("width", 20)))))
        gh = max(4, int(round(float(g.get("height", 20)))))
        h_mm_val = float(g.get("height_mm", 0.0))
        est_ppm = (gh / h_mm_val) if h_mm_val > 0.15 else 7.5
        min_mask_hole_px = max(4, int(round(0.065 * (est_ppm ** 2))))
        min_closed_hole_px = max(8, int(round(0.11 * (est_ppm ** 2))))
        min_vec_hole_px = max(5, int(round(0.075 * (est_ppm ** 2))))

        strokes = g.get(stroke_key) or g.get("strokes", [])

        mask_loops = 0
        g_mask = g.get("mask")
        has_live_mask = (g_mask is not None and isinstance(g_mask, np.ndarray) and g_mask.size >= 16)
        pre_loops = -1 if has_live_mask else int(g.get("loop_count", -1))

        if has_live_mask:
            bin_m = (g_mask > 0).astype(bool)
            padded = np.pad(bin_m, 3, mode="constant", constant_values=False)
            filled = binary_fill_holes(padded)
            holes = filled & (~padded)
            labeled_h, n_h = nd_label(holes)
            for h_id in range(1, n_h + 1):
                if int(np.sum(labeled_h == h_id)) >= min_mask_hole_px:
                    mask_loops += 1
            if mask_loops == 0:
                # Bridge 1px quantization gap in scan binarization
                closed_m = binary_closing(padded, structure=np.ones((2, 2), dtype=bool))
                filled_c = binary_fill_holes(closed_m)
                holes_c = filled_c & (~closed_m)
                labeled_hc, n_hc = nd_label(holes_c)
                for h_id in range(1, n_hc + 1):
                    if int(np.sum(labeled_hc == h_id)) >= min_closed_hole_px:
                        mask_loops += 1

        # Also check rasterized strokes & trajectory self-crossings
        vec_loops = 0
        if strokes:
            canvas = np.zeros((gh + 8, gw + 8), dtype=np.uint8)
            all_pts = []
            for s in strokes:
                pts = s.get("points", [])
                all_pts.extend(pts)
                if len(pts) >= 2:
                    for k in range(len(pts) - 1):
                        y0, x0 = int(round(pts[k][0])) + 4, int(round(pts[k][1])) + 4
                        y1, x1 = int(round(pts[k + 1][0])) + 4, int(round(pts[k + 1][1])) + 4
                        y0 = max(0, min(gh + 7, y0))
                        x0 = max(0, min(gw + 7, x0))
                        y1 = max(0, min(gh + 7, y1))
                        x1 = max(0, min(gw + 7, x1))
                        # Bresenham-like thick segment on canvas
                        n_steps = max(2, int(max(abs(y1 - y0), abs(x1 - x0))) + 1)
                        ys = np.linspace(y0, y1, n_steps).astype(int)
                        xs = np.linspace(x0, x1, n_steps).astype(int)
                        canvas[ys, xs] = 1
                        canvas[np.clip(ys + 1, 0, gh + 7), xs] = 1
                        canvas[ys, np.clip(xs + 1, 0, gw + 7)] = 1

            filled_v = binary_fill_holes(canvas > 0)
            holes_v = filled_v & (~(canvas > 0))
            labeled_v, n_v = nd_label(holes_v)
            for h_id in range(1, n_v + 1):
                if int(np.sum(labeled_v == h_id)) >= min_vec_hole_px:
                    vec_loops += 1

            # Check explicit trajectory self-intersection along longest stroke
            longest_s = max(strokes, key=lambda s: len(s.get("points", [])))
            l_pts = longest_s.get("points", [])
            if len(l_pts) >= 8 and vec_loops == 0:
                pts_xy = np.array([[float(p[0]), float(p[1])] for p in l_pts], dtype=np.float32)
                seg_lens = np.linalg.norm(np.diff(pts_xy, axis=0), axis=1)
                cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
                if cum_s[-1] >= 2.25 * est_ppm:
                    p0, pend = pts_xy[0], pts_xy[-1]
                    if float(np.linalg.norm(p0 - pend)) <= 0.68 * est_ppm:
                        vec_loops = max(vec_loops, 1)

        total_loops = max(pre_loops if pre_loops >= 0 else 0, mask_loops, vec_loops)
        g["loop_count"] = int(total_loops)
        return int(total_loops)

    def extract_glyph_features(self, glyphs: List[Dict[str, Any]], stroke_key: str = "strokes") -> np.ndarray:
        """
        Extracts hybrid topological + 2D image morphological + deep visual (DINOv2)
        descriptors for 100% of glyphs (zero truncation). Applies StandardScaler
        normalization first and THEN weights paleographic subspaces so loops, physical
        scale, ductus kinematics, and 16-D DINOv2 embeddings are never drowned out by
        raw 8x8 grid pixel counts.
        """
        from sklearn.preprocessing import StandardScaler
        from scipy.ndimage import binary_fill_holes

        mini_patches = []
        precomputed_partials = []
        patch_dim = 32

        for g in glyphs:
            strokes = g.get(stroke_key) or g.get("strokes", [])
            gw = max(1.0, float(g.get("width", 20)))
            gh = max(1.0, float(g.get("height", 20)))
            area = float(g.get("area", gw * gh))
            stroke_count = float(len(strokes)) if strokes else 1.0

            aspect = float(np.log(gw / gh))
            fill_factor = float(g.get("fill_factor", area / max(1.0, gw * gh)))
            loops = float(self.count_glyph_loops(g, stroke_key=stroke_key))
            # Prefer physical metric scale (mm) when available so scan resolution stays invariant
            h_mm = float(g.get("height_mm", 0.0))
            w_mm = float(g.get("width_mm", 0.0))
            a_mm2 = float(g.get("area_mm2", 0.0))
            log_h = float(np.log(max(0.2, h_mm if h_mm > 0 else gh)))
            log_w = float(np.log(max(0.2, w_mm if w_mm > 0 else gw)))
            log_area = float(np.log(max(0.05, a_mm2 if a_mm2 > 0 else area)))

            all_pts = []
            for s in strokes:
                all_pts.extend(s.get("points", []))

            # Build normalized 32x32 image patch (preserving aspect ratio in 28x28 inner box)
            mini_patch = np.zeros((patch_dim, patch_dim), dtype=np.uint8)
            g_mask = g.get("mask")
            if g_mask is not None and isinstance(g_mask, np.ndarray) and g_mask.size >= 4:
                mh, mw = g_mask.shape[:2]
                scale = min(26.0 / max(1, mh), 26.0 / max(1, mw))
                nh = max(2, min(28, int(round(mh * scale))))
                nw = max(2, min(28, int(round(mw * scale))))
                # Nearest-neighbor / area resize using numpy indexing
                ry = np.linspace(0, mh - 1, nh).astype(int)
                rx = np.linspace(0, mw - 1, nw).astype(int)
                resized = (g_mask[np.ix_(ry, rx)] > 0).astype(np.uint8) * 255
                y_off = (patch_dim - nh) // 2
                x_off = (patch_dim - nw) // 2
                mini_patch[y_off:y_off + nh, x_off:x_off + nw] = resized
            elif all_pts:
                pts_arr = np.array([[p[0], p[1]] for p in all_pts], dtype=np.float32)
                min_y, min_x = np.min(pts_arr, axis=0)
                max_y, max_x = np.max(pts_arr, axis=0)
                span_h = max(1.0, max_y - min_y)
                span_w = max(1.0, max_x - min_x)
                scale = min(26.0 / span_h, 26.0 / span_w)
                nh = max(2, min(28, int(round(span_h * scale))))
                nw = max(2, min(28, int(round(span_w * scale))))
                y_off = (patch_dim - nh) // 2
                x_off = (patch_dim - nw) // 2
                for s in strokes:
                    s_pts = s.get("points", [])
                    for k in range(len(s_pts)):
                        y0 = int(round((s_pts[k][0] - min_y) * scale)) + y_off
                        x0 = int(round((s_pts[k][1] - min_x) * scale)) + x_off
                        y0 = max(0, min(patch_dim - 1, y0))
                        x0 = max(0, min(patch_dim - 1, x0))
                        mini_patch[y0, x0] = 255
                        if k + 1 < len(s_pts):
                            y1 = int(round((s_pts[k + 1][0] - min_y) * scale)) + y_off
                            x1 = int(round((s_pts[k + 1][1] - min_x) * scale)) + x_off
                            y1 = max(0, min(patch_dim - 1, y1))
                            x1 = max(0, min(patch_dim - 1, x1))
                            n_st = max(2, max(abs(y1 - y0), abs(x1 - x0)) + 1)
                            ys = np.linspace(y0, y1, n_st).astype(int)
                            xs = np.linspace(x0, x1, n_st).astype(int)
                            mini_patch[ys, xs] = 255
                            mini_patch[np.clip(ys + 1, 0, patch_dim - 1), xs] = 255

            mini_patches.append(mini_patch)

            if not all_pts and np.sum(mini_patch) == 0:
                precomputed_partials.append(None)
                continue

            # 1. Image Morphology: 8x8 Spatial Density Grid (64 features) from mini_patch
            patch_norm = (mini_patch > 0).astype(np.float32)
            grid_8x8 = patch_norm.reshape(8, 4, 8, 4).mean(axis=(1, 3)).flatten()

            # 2. Horizontal & Vertical Projection Profiles (8 + 8 = 16 features)
            h_proj = patch_norm.reshape(8, 4, patch_dim).mean(axis=(1, 2))
            v_proj = patch_norm.reshape(patch_dim, 8, 4).mean(axis=(0, 2))

            # 3. Enclosed hole area ratio on mini_patch
            filled_p = binary_fill_holes(patch_norm > 0.5)
            holes_p = filled_p & (~(patch_norm > 0.5))
            hole_ratio = float(np.sum(holes_p)) / float(max(1, np.sum(filled_p)))

            # 4. First-class Kinematic Ductus features (stroke count, entry/exit tangents, tortuosity, orientation histogram, plein/délié width variance)
            if strokes:
                entry_s = strokes[0].get("points", [])
                exit_s = strokes[-1].get("points", [])
                if len(entry_s) >= 2:
                    entry_angle = float(np.arctan2(entry_s[1][0] - entry_s[0][0], entry_s[1][1] - entry_s[0][1]))
                else:
                    entry_angle = 0.0
                if len(exit_s) >= 2:
                    exit_angle = float(np.arctan2(exit_s[-1][0] - exit_s[-2][0], exit_s[-1][1] - exit_s[-2][1]))
                else:
                    exit_angle = 0.0
                s_feats = self.feature_extractor.extract_batch(strokes)
                mean_f = np.mean(s_feats, axis=0) if len(s_feats) > 0 else np.zeros(16, dtype=np.float32)
            else:
                entry_angle = 0.0
                exit_angle = 0.0
                mean_f = np.zeros(16, dtype=np.float32)

            partial_vec = [
                loops,                     # [0] Dominant topological loop invariant
                log_h,                     # [1] Physical height scale invariant
                log_w,                     # [2] Physical width scale invariant
                log_area,                  # [3] Physical area scale invariant
                hole_ratio,                # [4] Interior hole ratio invariant
                aspect,                    # [5] Log aspect ratio (tall vs core vs wide)
                fill_factor,               # [6] Ink density
                stroke_count,              # [7] Kinematic pen-lift count (1 vs 2 vs 3-4 strokes)
                np.sin(entry_angle),       # [8] Pen touchdown entry tangent sin
                np.cos(entry_angle),       # [9] Pen touchdown entry tangent cos
                np.sin(exit_angle),        # [10] Pen-lift exit tangent sin
                np.cos(exit_angle),        # [11] Pen-lift exit tangent cos
                float(mean_f[0]),          # [12] Log kinematic arc length
                float(mean_f[2]),          # [13] Stroke tortuosity / bending curvature
                float(mean_f[7]),          # [14] Mean nib thickness (plein/délié)
                float(mean_f[8]),          # [15] Nib thickness variance (bevel contrast)
                float(mean_f[11]),         # [16] Orientation histogram bin 1
                float(mean_f[12]),         # [17] Orientation histogram bin 2
                float(mean_f[13]),         # [18] Orientation histogram bin 3
                float(mean_f[14]),         # [19] Orientation histogram bin 4
                float(mean_f[15]),         # [20] Orientation histogram bin 5
            ]
            partial_vec.extend(grid_8x8.tolist())
            partial_vec.extend(h_proj.tolist())
            partial_vec.extend(v_proj.tolist())
            precomputed_partials.append(partial_vec)

        # Full DINOv2 batch inference with exact SHA-1 pixel-hash deduplication + persistent disk cache
        # Project the 384/768-D DINOv2 ViT embeddings down to their top 16 principal components (PCA)
        import hashlib
        import pickle
        cache_path = Path("data/intermediate/dinov2_patch_cache.pkl")
        if not getattr(self, "_dino_disk_loaded", False):
            self._dino_disk_loaded = True
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f_dc:
                        loaded_dc = pickle.load(f_dc)
                    if isinstance(loaded_dc, dict):
                        self._dino_patch_cache.update(loaded_dc)
                except Exception:
                    pass

        patch_keys = [hashlib.sha1(p.tobytes()).digest() for p in mini_patches]
        missing_idx = [i for i, k in enumerate(patch_keys) if k not in self._dino_patch_cache]
        if missing_idx:
            # Deduplicate identical missing patches before running ViT forward passes
            unique_missing = {}
            for i in missing_idx:
                k = patch_keys[i]
                if k not in unique_missing:
                    unique_missing[k] = mini_patches[i]
            u_keys = list(unique_missing.keys())
            u_patches = list(unique_missing.values())
            u_embs = self.dinov2_engine.extract_batch_embeddings(u_patches, batch_size=128)
            for k, emb in zip(u_keys, u_embs):
                self._dino_patch_cache[k] = np.asarray(emb, dtype=np.float32)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f_dc:
                    pickle.dump(self._dino_patch_cache, f_dc, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass

        dino_batch_embs = np.vstack([self._dino_patch_cache[k] for k in patch_keys]) if patch_keys else np.zeros((0, 16), dtype=np.float32)
        if len(dino_batch_embs) > 0 and dino_batch_embs.shape[1] > 16:
            from sklearn.decomposition import PCA
            n_comp = min(16, dino_batch_embs.shape[0], dino_batch_embs.shape[1])
            if n_comp >= 2:
                pca_dino = PCA(n_components=n_comp, random_state=42)
                dino_proj = pca_dino.fit_transform(dino_batch_embs)
                if n_comp < 16:
                    dino_proj = np.pad(dino_proj, ((0, 0), (0, 16 - n_comp)))
            else:
                dino_proj = dino_batch_embs[:, :16]
        else:
            dino_proj = dino_batch_embs

        n_partial_dim = 21 + 64 + 16
        glyph_feature_vectors = []
        for idx, partial in enumerate(precomputed_partials):
            if partial is None:
                glyph_feature_vectors.append(np.zeros(n_partial_dim + 16, dtype=np.float32))
                continue
            dino_emb = dino_proj[idx] if idx < len(dino_proj) else np.zeros(16, dtype=np.float32)
            dino_top = dino_emb[:16] if len(dino_emb) >= 16 else np.pad(dino_emb, (0, 16 - len(dino_emb)))
            norm_d = float(np.linalg.norm(dino_top))
            if norm_d > 1e-5:
                dino_top = dino_top / norm_d
            vec = list(partial)
            vec.extend(dino_top.tolist())
            glyph_feature_vectors.append(np.array(vec, dtype=np.float32))

        X_raw = np.array(glyph_feature_vectors, dtype=np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Apply domain subspace weights AFTER unit-variance scaling so 64 raw grid bins
        # never drown out loop topology, metric dimensions, kinematics, or DINOv2.
        w = np.ones(X_scaled.shape[1], dtype=np.float32)
        w[0] = 4.0
        w[1:4] = 2.6
        w[4:6] = 2.4
        w[6] = 1.2
        w[7] = 2.4
        w[8:12] = 1.8
        w[12:14] = 2.0
        w[14:16] = 1.4
        w[16:21] = 1.5
        w[21:85] = 0.35
        w[85:101] = 0.55
        if X_scaled.shape[1] >= 117:
            w[101:117] = 1.5
        return X_scaled * w

    @staticmethod
    def classify_glyph_topology(
        g: Dict[str, Any],
        scale_calib: Optional[MetricScaleCalibration] = None,
        stroke_key: str = "strokes"
    ) -> str:
        """
        Deterministically classifies a glyph into a strictly loop-disjoint topological family
        using physical metric scale calibration (millimeters & tenths of mm):
        0-Loop Families:
          - PUNCTUATION_OR_SPECK
          - DRAWING_DEBRIS
          - TALL_GALLOWS_LOOPLESS
          - BENCH_HORIZONTAL
          - OPEN_C_CRESCENT
          - VERTICAL_MINIM
          - COMPOSITE_LIGATURE
        1-Loop Families:
          - TALL_GALLOWS_1LOOP
          - CLOSED_LOOP
          - WIDE_LOOP_LIGATURE
        2+ Loop Families:
          - TALL_GALLOWS_MULTILOOP
          - MULTI_LOOP_COMPLEX
        """
        strokes = g.get(stroke_key) or g.get("strokes", [])
        gw = max(1.0, float(g.get("width", 20)))
        gh = max(1.0, float(g.get("height", 20)))
        aspect = gw / gh
        stroke_count = len(strokes)
        area = float(g.get("area", gw * gh))

        if scale_calib is None:
            scale_calib = g.get("scale_calib")
        if scale_calib is None:
            c_type = g.get("corpus_type") or ("seraphinianus" if "serafini" in str(g.get("page_id", "")) else "voynich")
            scale_calib = MetricScaleCalibration(manuscript_type=c_type)

        # Preserve pre-calibrated metric mm dimensions when already computed on HD scans
        h_mm = float(g.get("height_mm", 0.0)) or scale_calib.px_to_mm(gh)
        w_mm = float(g.get("width_mm", 0.0)) or scale_calib.px_to_mm(gw)
        area_mm2 = float(g.get("area_mm2", 0.0)) or scale_calib.area_px2_to_mm2(area)
        g["height_mm"] = round(h_mm, 2)
        g["width_mm"] = round(w_mm, 2)
        g["area_mm2"] = round(area_mm2, 2)
        g["height_dmm"] = round(h_mm * 10.0, 1)
        g["width_dmm"] = round(w_mm * 10.0, 1)

        # 0. Micro specks / Punctuation dots: physical touchdown footprint (diameter <= 0.55 mm, area <= 0.35 mm^2)
        if area_mm2 <= scale_calib.DOT_MAX_AREA_MM2 or (w_mm <= scale_calib.DOT_MAX_DIAMETER_MM and h_mm <= scale_calib.DOT_MAX_DIAMETER_MM):
            g["loop_count"] = 0
            return "PUNCTUATION_OR_SPECK"

        loops = GlyphCatalogue.count_glyph_loops(g, stroke_key=stroke_key)

        # 0b. Macro drawing / illustration debris (area >= 25 mm^2 or height/width >= 14 mm)
        if area_mm2 >= 25.0 or h_mm >= 14.0 or w_mm >= 14.0:
            return f"DRAWING_DEBRIS_{min(2, loops)}L"

        # Determine physical scale band: TINY (<= 1.30 mm / 13 tenths), CORE, TALL (>= 2.60 mm / 26 tenths), WIDE (>= 2.40 mm / 24 tenths)
        if area_mm2 < 0.40 or (w_mm <= 1.05 and h_mm <= scale_calib.TINY_MAX_HEIGHT_MM):
            scale_band = "TINY"
        elif (gh >= 1.35 * gw or aspect <= 0.80) and h_mm >= scale_calib.TALL_GALLOWS_MIN_HEIGHT_MM:
            scale_band = "TALL"
        elif aspect >= 1.35 and w_mm >= scale_calib.WIDE_BENCH_MIN_WIDTH_MM:
            scale_band = "WIDE"
        else:
            scale_band = "CORE"

        # 1. Strictly partition by loop count FIRST so 0-loop glyphs NEVER mix with looped glyphs
        tall_thresh = scale_calib.TALL_GALLOWS_MIN_HEIGHT_MM * 0.88
        wide_thresh = scale_calib.WIDE_BENCH_MIN_WIDTH_MM * 0.88

        if loops >= 2:
            base_family = "TALL_GALLOWS_MULTILOOP" if (gh >= 1.35 * gw and h_mm >= tall_thresh) else "MULTI_LOOP_COMPLEX"
            return f"{base_family}__{scale_band}"

        if loops == 1:
            if gh >= 1.35 * gw and h_mm >= tall_thresh:
                base_family = "TALL_GALLOWS_1LOOP"
            elif aspect >= 1.35 and w_mm >= wide_thresh:
                base_family = "WIDE_LOOP_LIGATURE"
            else:
                base_family = "CLOSED_LOOP"
            return f"{base_family}__{scale_band}"

        # Below this line, loops == 0 is strictly guaranteed!
        if gh >= 1.35 * gw and h_mm >= tall_thresh:
            return f"TALL_GALLOWS_LOOPLESS__{scale_band}"
        if stroke_count >= 2 and h_mm >= tall_thresh:
            return f"TALL_GALLOWS_LOOPLESS__{scale_band}"

        if aspect >= 1.35 and w_mm >= wide_thresh:
            return f"BENCH_HORIZONTAL__{scale_band}"

        all_pts = []
        for s in strokes:
            all_pts.extend(s.get("points", []))
        if not all_pts:
            return f"VERTICAL_MINIM__{scale_band}"

        longest_s = max(strokes, key=lambda s: len(s.get("points", [])))
        l_pts = longest_s.get("points", [])

        if stroke_count == 1:
            if len(l_pts) >= 4:
                mid_idx = len(l_pts) // 2
                x_start = l_pts[0][1]
                x_end = l_pts[-1][1]
                x_mid = l_pts[mid_idx][1]
                crescent_indent = max(2.0, scale_calib.mm_to_px(0.25))
                if x_mid < min(x_start, x_end) - crescent_indent:
                    return f"OPEN_C_CRESCENT__{scale_band}"
            if aspect <= 0.75:
                return f"VERTICAL_MINIM__{scale_band}"
            return f"OPEN_C_CRESCENT__{scale_band}"

        if stroke_count >= 2:
            return f"COMPOSITE_LIGATURE__{scale_band}"

        return f"VERTICAL_MINIM__{scale_band}"

    @staticmethod
    def _discover_emergent_clusters(
        glyphs: List[Dict[str, Any]],
        X_scaled: np.ndarray,
        topo_groups: Dict[str, List[int]],
    ) -> Tuple[List[Tuple[int, List[int], List[Dict[str, Any]]]], Dict[str, Any]]:
        """
        Parameter-free unsupervised emergent alphabet induction:
        1. Consolidates ultra-sparse sub-family tags (`< min_support`) into their strict
           `(loop_count, scale_band)` parent stratum so 1-glyph border splinters cannot
           inflate cluster counts, while keeping `0L / 1L / 2+L` and `TINY / CORE / TALL / WIDE`
           100% disjoint.
        2. Builds exact Ward dendrograms within each topological stratum and expands a global
           priority frontier ordered by the dimensionless Fisher modal separation-to-dispersion
           ratio `rho = ||mu_L - mu_R||_2 / sigma_within`.
        3. Automatically selects the emergent alphabet size `K*` at the Kneedle maximum-curvature
           elbow of the normalized rate-distortion curve `WCSS(K) / TSS` (zero hardcoded target size).
        """
        import heapq
        import math
        from scipy.cluster.hierarchy import linkage

        debris_groups: Dict[str, List[int]] = {}
        raw_lexical: Dict[str, List[int]] = {}
        for tg, idxs in topo_groups.items():
            if tg == "PUNCTUATION_OR_SPECK" or tg.startswith("DRAWING_DEBRIS"):
                debris_groups[tg] = idxs
            else:
                raw_lexical[tg] = idxs

        N_lex = sum(len(idxs) for idxs in raw_lexical.values())
        min_support = max(4, int(round(0.0035 * N_lex))) if N_lex >= 20 else max(1, N_lex // 4)

        # Merge tiny sub-family edge buckets (< min_support) into their strict (loop, scale_band) parent bucket
        lexical_strata: Dict[str, List[int]] = {}
        sparse_pool: Dict[str, List[int]] = {}
        for tg, idxs in raw_lexical.items():
            if len(idxs) >= min_support or N_lex < 20:
                lexical_strata[tg] = list(idxs)
            else:
                scale_suffix = tg.split("__")[-1] if "__" in tg else "CORE"
                loops_val = min(2, int(glyphs[idxs[0]].get("loop_count", 0)))
                parent_key = f"CONSOLIDATED_{loops_val}L__{scale_suffix}"
                sparse_pool.setdefault(parent_key, []).extend(idxs)

        for pk, s_idxs in sparse_pool.items():
            # Attach to matching loop+scale largest stratum if one exists, else keep consolidated loop+scale stratum
            loops_prefix = pk.split("__")[0].replace("CONSOLIDATED_", "")
            scale_suf = pk.split("__")[-1]
            matching = [
                k_st for k_st, v_st in lexical_strata.items()
                if k_st.endswith(f"__{scale_suf}")
                and min(2, int(glyphs[v_st[0]].get("loop_count", 0))) == int(loops_prefix[0])
            ]
            if matching:
                best_target = max(matching, key=lambda k_m: len(lexical_strata[k_m]))
                lexical_strata[best_target].extend(s_idxs)
            else:
                lexical_strata[pk] = s_idxs

        raw_clusters: List[Tuple[int, List[int], List[Dict[str, Any]]]] = []
        cid = 0

        if N_lex < 8 or len(lexical_strata) == 0:
            for _, idxs in lexical_strata.items():
                raw_clusters.append((cid, idxs, [glyphs[i] for i in idxs]))
                cid += 1
            for _, idxs in debris_groups.items():
                raw_clusters.append((cid, idxs, [glyphs[i] for i in idxs]))
                cid += 1
            return raw_clusters, {"induction_mode": "emergent_small_sample", "lexical_archetypes": len(lexical_strata)}

        all_lex_idx = [i for idxs in lexical_strata.values() for i in idxs]
        X_lex = X_scaled[all_lex_idx]
        tss = max(1e-6, float(np.sum((X_lex - np.mean(X_lex, axis=0, keepdims=True)) ** 2)))

        def _stats(g_idxs: List[int]) -> Tuple[np.ndarray, float]:
            pts = X_scaled[g_idxs]
            mu = np.mean(pts, axis=0)
            return mu, float(np.sum((pts - mu) ** 2))

        trees: Dict[str, Tuple[Dict[int, List[int]], Dict[int, Tuple[int, int]]]] = {}
        leaves: List[Tuple[str, int, List[int], float]] = []
        pq: List[Tuple[float, int, str, int, int, int, float, float, float]] = []
        uid = 0

        for tg, idxs in lexical_strata.items():
            m = len(idxs)
            _, w0 = _stats(idxs)
            if m < 2 * min_support:
                leaves.append((tg, 0, idxs, w0))
                continue
            Z = linkage(X_scaled[idxs], method="ward")
            nm = {k: [idxs[k]] for k in range(m)}
            nc = {}
            for s_i in range(m - 1):
                c1, c2 = int(Z[s_i, 0]), int(Z[s_i, 1])
                nm[m + s_i] = nm[c1] + nm[c2]
                nc[m + s_i] = (c1, c2)
            root = 2 * m - 2
            trees[tg] = (nm, nc)
            leaves.append((tg, root, nm[root], w0))

            c1, c2 = nc[root]
            m1, m2 = nm[c1], nm[c2]
            if len(m1) >= min_support and len(m2) >= min_support:
                mu1, w1 = _stats(m1)
                mu2, w2 = _stats(m2)
                sep = float(np.linalg.norm(mu1 - mu2))
                sig_w = math.sqrt(max(1e-6, (w1 + w2) / float(len(m1) + len(m2))))
                rho = sep / sig_w
                sc = rho * ((len(m1) + len(m2)) / float(N_lex)) ** 0.18
                uid += 1
                heapq.heappush(pq, (-sc, uid, tg, root, c1, c2, w1, w2, rho))

        # Expand natural Ward-Fisher frontier up to natural support horizon (no fixed target K)
        max_horizon = min(140, max(45, N_lex // max(1, min_support)))
        snapshots: List[Tuple[int, float, float, List[Tuple[str, int, List[int], float]]]] = [
            (len(leaves), sum(x[3] for x in leaves) / tss, 9.99, list(leaves))
        ]

        while pq and len(leaves) < max_horizon:
            _, _, tg_k, nid, c1, c2, w1, w2, rho = heapq.heappop(pq)
            pos = next((i for i, x in enumerate(leaves) if x[0] == tg_k and x[1] == nid), -1)
            if pos < 0:
                continue
            nm_l, nc_l = trees[tg_k]
            leaves.pop(pos)
            leaves.append((tg_k, c1, nm_l[c1], w1))
            leaves.append((tg_k, c2, nm_l[c2], w2))
            for ch in (c1, c2):
                if ch in nc_l:
                    cc1, cc2 = nc_l[ch]
                    m1, m2 = nm_l[cc1], nm_l[cc2]
                    if len(m1) >= min_support and len(m2) >= min_support:
                        mu1, cw1 = _stats(m1)
                        mu2, cw2 = _stats(m2)
                        sep = float(np.linalg.norm(mu1 - mu2))
                        sig_w = math.sqrt(max(1e-6, (cw1 + cw2) / float(len(m1) + len(m2))))
                        r_c = sep / sig_w
                        sc_c = r_c * ((len(m1) + len(m2)) / float(N_lex)) ** 0.18
                        uid += 1
                        heapq.heappush(pq, (-sc_c, uid, tg_k, ch, cc1, cc2, cw1, cw2, r_c))
            snapshots.append((len(leaves), sum(x[3] for x in leaves) / tss, rho, list(leaves)))

        if len(snapshots) >= 3:
            ks = np.array([s[0] for s in snapshots], dtype=np.float64)
            ds = np.array([s[1] for s in snapshots], dtype=np.float64)
            k_norm = (ks - ks[0]) / max(1.0, ks[-1] - ks[0])
            d_norm = (ds - ds[-1]) / max(1e-6, ds[0] - ds[-1])
            elbow_idx = int(np.argmax(1.0 - k_norm - d_norm))
        else:
            elbow_idx = len(snapshots) - 1

        chosen_k, chosen_wcss_ratio, chosen_rho, chosen_leaves = snapshots[elbow_idx]
        for _, _, member_idxs, _ in chosen_leaves:
            raw_clusters.append((cid, member_idxs, [glyphs[i] for i in member_idxs]))
            cid += 1
        for _, idxs in debris_groups.items():
            raw_clusters.append((cid, idxs, [glyphs[i] for i in idxs]))
            cid += 1

        meta = {
            "induction_mode": "emergent_kneedle_fisher",
            "lexical_archetypes": int(chosen_k),
            "debris_bins": int(len(debris_groups)),
            "kneedle_wcss_ratio": round(float(chosen_wcss_ratio), 4),
            "fisher_rho_at_elbow": round(float(chosen_rho), 3),
        }
        return raw_clusters, meta

    def build_catalogue(
        self,
        glyphs: List[Dict[str, Any]],
        corpus_type: str = "voynich",
        scale_calib: Optional[MetricScaleCalibration] = None,
        stroke_key: str = "strokes",
        tag_field: str = "canonical_type"
    ) -> Dict[str, Any]:
        """
        Induces the Canonical Alphabet via unsupervised topological stratification,
        joint kinematic + DINOv2 feature clustering, and natural Kneedle-Fisher emergence
        (when `target_alphabet_size is None`) or explicit target partitioning (for unit tests).
        """
        if not glyphs:
            return {"total_glyphs": 0, "canonical_alphabet_size": 0, "alphabet": []}

        if scale_calib is None:
            scale_calib = MetricScaleCalibration(manuscript_type=corpus_type)

        X_scaled = self.extract_glyph_features(glyphs, stroke_key=stroke_key)

        for g in glyphs:
            g["topology_group"] = self.classify_glyph_topology(g, scale_calib=scale_calib, stroke_key=stroke_key)

        topo_groups: Dict[str, List[int]] = {}
        for idx, g in enumerate(glyphs):
            topo_groups.setdefault(g["topology_group"], []).append(idx)

        emergence_meta: Dict[str, Any] = {"induction_mode": "explicit_target"}
        if self.target_alphabet_size is None:
            raw_clusters, emergence_meta = self._discover_emergent_clusters(glyphs, X_scaled, topo_groups)
        else:
            raw_clusters = []
            cluster_id_counter = 0
            non_debris_total = sum(
                len(idxs) for tg, idxs in topo_groups.items()
                if tg != "PUNCTUATION_OR_SPECK" and not tg.startswith("DRAWING_DEBRIS")
            )
            effective_total = max(1, non_debris_total)

            for tg, indices in topo_groups.items():
                if tg == "PUNCTUATION_OR_SPECK" or tg.startswith("DRAWING_DEBRIS") or len(indices) <= 2:
                    raw_clusters.append((cluster_id_counter, indices, [glyphs[i] for i in indices]))
                    cluster_id_counter += 1
                    continue

                sub_X = X_scaled[indices]
                prop = len(indices) / float(effective_total)
                k_sub = max(1, min(int(round(self.target_alphabet_size * prop)), len(indices)))

                if k_sub == 1:
                    raw_clusters.append((cluster_id_counter, indices, [glyphs[i] for i in indices]))
                    cluster_id_counter += 1
                else:
                    clusterer = AgglomerativeClustering(n_clusters=k_sub, metric="euclidean", linkage="ward")
                    sub_labels = clusterer.fit_predict(sub_X)
                    for sl in sorted(set(sub_labels)):
                        sub_idx = [indices[i] for i in np.where(sub_labels == sl)[0]]
                        raw_clusters.append((cluster_id_counter, sub_idx, [glyphs[i] for i in sub_idx]))
                        cluster_id_counter += 1

        # Sort raw clusters: genuine letterforms first (by frequency desc), debris/specks at the end
        def cluster_sort_key(item):
            c_glyphs = item[2]
            topo = c_glyphs[0].get("topology_group", "")
            is_debris = 1 if (topo == "PUNCTUATION_OR_SPECK" or topo.startswith("DRAWING_DEBRIS")) else 0
            return (is_debris, -len(c_glyphs))

        raw_clusters.sort(key=cluster_sort_key)

        # Build canonical glyph entries
        alphabet_entries = []
        assigned_standard_codes = set()
        for rank, (cluster_id, indices, cluster_glyphs) in enumerate(raw_clusters, start=1):
            cluster_points = X_scaled[indices]

            # Find consensus medoid: closest to morphological centroid with high ductus completeness
            centroid = np.mean(cluster_points, axis=0, keepdims=True)
            centroid_dists = pairwise_distances(cluster_points, centroid).flatten()

            best_local_idx = int(np.argmin(centroid_dists))
            if len(indices) > 2:
                max_cd = max(1e-6, float(np.max(centroid_dists)))
                best_score = -1e9
                for loc_i, g_cand in enumerate(cluster_glyphs):
                    norm_d = float(centroid_dists[loc_i]) / max_cd
                    if norm_d > 0.55 and loc_i != int(np.argmin(centroid_dists)):
                        continue
                    # Evaluate ductus span completeness
                    gh_c = max(1.0, float(g_cand.get("height", 20)))
                    gw_c = max(1.0, float(g_cand.get("width", 20)))
                    pts_c = [p for s in g_cand.get("strokes", []) for p in s.get("points", [])]
                    if len(pts_c) >= 2:
                        ys_c = [p[0] for p in pts_c]
                        xs_c = [p[1] for p in pts_c]
                        cov_y = min(1.0, (max(ys_c) - min(ys_c)) / gh_c)
                        cov_x = min(1.0, (max(xs_c) - min(xs_c)) / gw_c)
                        cov = 0.6 * cov_y + 0.4 * cov_x
                    else:
                        cov = 0.0
                    cand_score = cov * 0.45 - norm_d
                    if cand_score > best_score:
                        best_score = cand_score
                        best_local_idx = loc_i

            medoid_local_idx = best_local_idx
            exemplar_glyph = glyphs[indices[medoid_local_idx]]

            # Compute exact pairwise morphological distances to the selected medoid
            medoid_vec = cluster_points[[medoid_local_idx]]
            dists = pairwise_distances(cluster_points, medoid_vec).flatten()

            sorted_pairs = sorted(zip(dists, indices, cluster_glyphs), key=lambda x: x[0])
            cluster_glyphs = [p[2] for p in sorted_pairs]

            # Outlier quarantine: evaluate distance from medoid
            med_dist = float(np.median(dists)) if len(dists) > 0 else 1.0
            max_allowed_dist = self.distance_threshold if self.distance_threshold is not None else max(2.6, med_dist * 2.5)

            # Calculate geometric averages for outlier thresholds and morphological matching
            mean_strokes = round(float(np.mean([g.get("stroke_count", len(g.get("strokes", []))) for g in cluster_glyphs])), 2)
            mean_height = round(float(np.mean([g.get("height", 20) for g in cluster_glyphs])), 1)
            mean_width = round(float(np.mean([g.get("width", 15) for g in cluster_glyphs])), 1)
            cluster_loops = int(exemplar_glyph.get("loop_count", 0))
            cluster_topology = exemplar_glyph.get("topology_group", "VERTICAL_MINIM")

            inlier_count = 0
            outlier_count = 0
            rejection_bins_counts = {}

            for d_val, g_idx, g_item in sorted_pairs:
                gw = float(g_item.get("width", 20))
                gh = float(g_item.get("height", 20))
                g_loops = int(g_item.get("loop_count", 0))
                g_area = float(g_item.get("area", gw * gh))
                g_stk = float(g_item.get("stroke_count", len(g_item.get("strokes", []))))

                # Strict conformance checks:
                # 1. Exact topological loop match
                loop_deviant = (g_loops != cluster_loops)
                # 2. Tight physical scale tolerance (within 0.58x - 1.55x height and 0.55x - 1.65x width)
                scale_deviant = (gh < 0.58 * mean_height or gh > 1.55 * mean_height or
                                 gw < 0.55 * mean_width or gw > 1.65 * mean_width)
                # 3. Distance from medoid
                dist_out = bool(d_val > max_allowed_dist and len(sorted_pairs) > 3)

                is_out = bool(dist_out or scale_deviant or loop_deviant)
                g_item["medoid_distance"] = round(float(d_val), 3)
                g_item["is_outlier"] = is_out

                if is_out:
                    outlier_count += 1
                    calib_item = scale_calib or g_item.get("scale_calib") or MetricScaleCalibration(corpus_type)
                    gw_mm = calib_item.px_to_mm(gw)
                    gh_mm = calib_item.px_to_mm(gh)
                    g_area_mm2 = calib_item.area_px2_to_mm2(g_area)

                    if loop_deviant:
                        b_tag = "TOPOLOGY_LOOP_MISMATCH"
                    elif scale_deviant:
                        b_tag = "PHYSICAL_SCALE_DEVIANT"
                    elif corpus_type.lower() == "seraphinianus":
                        if g_area_mm2 > 18.0 or gw_mm > 12.0 or gh_mm > 14.0:
                            b_tag = "SERAFINI_ILLUSTRATION_BLEED"
                        elif g_area_mm2 <= calib_item.DOT_MAX_AREA_MM2 or (gw_mm <= calib_item.DOT_MAX_DIAMETER_MM and gh_mm <= calib_item.DOT_MAX_DIAMETER_MM):
                            b_tag = "SERAFINI_SCANNER_SHADOW_SPECK"
                        elif (gw / max(1.0, gh)) < 0.6 and gh_mm <= calib_item.TINY_MAX_HEIGHT_MM:
                            b_tag = "SERAFINI_DETACHED_DIACRITIC"
                        elif gw_mm > 9.5:
                            b_tag = "SERAFINI_WORD_JUNCTION_MERGE"
                        else:
                            b_tag = "SERAFINI_IRREGULAR_CURSIVE"
                    else:
                        if g_area_mm2 > 16.0 or gw_mm > 10.0 or gh_mm > 12.0 or g_item.get("paint_fraction", 0.0) > 0.35:
                            b_tag = "ILLUSTRATION_BLEED"
                        elif (gw_mm <= calib_item.DOT_MAX_DIAMETER_MM and gh_mm <= calib_item.DOT_MAX_DIAMETER_MM) or g_area_mm2 <= calib_item.DOT_MAX_AREA_MM2:
                            b_tag = "PARCHMENT_FLAW_SPECK"
                        elif (gw >= 2.2 * mean_width and gh >= 1.5 * mean_height) or g_area >= 2.8 * (mean_width * mean_height) or g_stk >= 3:
                            b_tag = "SEGMENTATION_MERGE"
                        elif g_area <= 0.28 * (mean_width * mean_height) or (gw <= 0.35 * mean_width and gh <= 0.6 * mean_height):
                            b_tag = "SEGMENTATION_FRAGMENT"
                        elif gw >= 1.8 * mean_width or gh >= 1.8 * mean_height:
                            b_tag = "EXTREME_FLOURISH"
                        else:
                            b_tag = "HAPAX_ATYPICAL"
                else:
                    inlier_count += 1
                    b_tag = "INLIER_CONFORMANT"

                g_item["outlier_bin"] = b_tag
                rejection_bins_counts[b_tag] = rejection_bins_counts.get(b_tag, 0) + 1

            type_name = f"G{rank:02d}"

            # Tag all members with canonical type on requested field
            for g in cluster_glyphs:
                g[tag_field] = type_name

            # Match with standard historical corpora (preserving reference_svg even for allographs)
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
                ref_svg_keep = corpus_match.get("reference_svg", "")
                if code and code != "S-EMERGENT" and code in assigned_standard_codes:
                    corpus_match = {
                        "system": "Serafini Typology (1981) / Deri (2015)",
                        "serafini_code": f"{code}-v{rank:02d}",
                        "deri_equivalent": f"{corpus_match.get('deri_equivalent', 'S')}-v{rank:02d}",
                        "bulik_equivalent": f"ω_{rank}",
                        "name": f"{corpus_match.get('name', 'Cursive Archetype')} (Allograph {type_name})",
                        "category": f"{corpus_match.get('category', 'Cursive')} [{cluster_loops} loop(s)]",
                        "confidence_pct": corpus_match.get("confidence_pct", 75.0),
                        "description": f"Morphologically homogeneous allograph ({cluster_topology}, {cluster_loops} topological loop(s)).",
                        "reference_svg": ref_svg_keep
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
                ref_svg_keep = corpus_match.get("reference_svg", "")
                if eva_eq and eva_eq != "—" and eva_eq in assigned_standard_codes:
                    corpus_match = {
                        "system": "EVA / Currier / v101 / Voynichese",
                        "eva_equivalent": f"{eva_eq} (allo)",
                        "currier_equivalent": corpus_match.get("currier_equivalent", "—"),
                        "v101_equivalent": corpus_match.get("v101_equivalent", "—"),
                        "voynichese_equivalent": corpus_match.get("voynichese_equivalent", "—"),
                        "name": f"{corpus_match.get('name', 'Scribal Grapheme')} ({type_name})",
                        "category": f"{corpus_match.get('category', 'Grapheme')} [{cluster_loops} loop(s)]",
                        "confidence_pct": corpus_match.get("confidence_pct", 75.0),
                        "description": f"Topologically locked allograph of '{eva_eq}' ({cluster_topology}, {cluster_loops} loop(s)).",
                        "reference_svg": ref_svg_keep
                    }
                elif eva_eq and eva_eq != "—":
                    assigned_standard_codes.add(eva_eq)

            # Compute multi-scheme independent matches
            multi_matches = CorpusCorrespondenceMatcher.match_all_schemes(
                corpus_type=corpus_type,
                archetype_id=type_name,
                mean_strokes=mean_strokes,
                mean_width=mean_width,
                mean_height=mean_height,
                frequency_rank=rank,
                cluster_glyphs=cluster_glyphs
            )

            # Collect strictly conformant representative instances interleaved across pages (up to 36)
            conformant_by_page: Dict[str, List[Dict[str, Any]]] = {}
            for g in cluster_glyphs:
                if g.get("is_outlier", False):
                    continue
                if int(g.get("loop_count", 0)) != cluster_loops:
                    continue
                inst_rec = {
                    "glyph_id": g.get("glyph_id", ""),
                    "page_id": g.get("page_id", ""),
                    "line_id": g.get("line_id", ""),
                    "word_id": g.get("word_id", ""),
                    "bbox": [int(x) for x in g.get("bbox", [0, 0, 0, 0])],
                    "polygon": g.get("polygon", []),
                    "width": int(g.get("width", 0)),
                    "height": int(g.get("height", 0)),
                    "height_mm": float(g.get("height_mm", 0.0)) or round(scale_calib.px_to_mm(float(g.get("height", 0))), 2),
                    "width_mm": float(g.get("width_mm", 0.0)) or round(scale_calib.px_to_mm(float(g.get("width", 0))), 2),
                    "height_dmm": float(g.get("height_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(float(g.get("height", 0))), 1),
                    "width_dmm": float(g.get("width_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(float(g.get("width", 0))), 1),
                    "area_mm2": float(g.get("area_mm2", 0.0)) or round(scale_calib.area_px2_to_mm2(float(g.get("area", 0))), 2),
                    "stroke_count": int(len(g.get(stroke_key) or g.get("strokes", []))),
                    "loop_count": int(g.get("loop_count", 0)),
                    "topology_group": g.get("topology_group", ""),
                    "eva_char": g.get("eva_char", ""),
                    "crop_uri": g.get("crop_uri", ""),
                    "svg_content": g.get("euler_svg_content", "") if stroke_key == "euler_strokes" else g.get("svg_content", g.get("ductus_svg", "")),
                    "euler_svg_content": g.get("euler_svg_content", ""),
                    "euler_stroke_count": int(g.get("euler_stroke_count", len(g.get("euler_strokes", [])))),
                    "png_rel": g.get("png_rel", ""),
                    "svg_rel": g.get("svg_rel", ""),
                    "is_outlier": False,
                    "outlier_bin": "INLIER_CONFORMANT",
                    "medoid_distance": g.get("medoid_distance", 0.0)
                }
                conformant_by_page.setdefault(inst_rec["page_id"], []).append(inst_rec)

            # Round-robin across pages while keeping medoid-distance order within each page
            sample_instances = []
            page_queues = list(conformant_by_page.values())
            q_idx = 0
            while len(sample_instances) < 36 and any(page_queues):
                if page_queues[q_idx]:
                    sample_instances.append(page_queues[q_idx].pop(0))
                q_idx = (q_idx + 1) % len(page_queues)
                if all(len(q) == 0 for q in page_queues):
                    break

            # Fallback: if all strictly conformant instances were filtered out, preserve medoid exemplar
            if not sample_instances and cluster_glyphs:
                eg = exemplar_glyph
                sample_instances.append({
                    "glyph_id": eg.get("glyph_id", ""),
                    "page_id": eg.get("page_id", ""),
                    "line_id": eg.get("line_id", ""),
                    "word_id": eg.get("word_id", ""),
                    "bbox": [int(x) for x in eg.get("bbox", [0, 0, 0, 0])],
                    "polygon": eg.get("polygon", []),
                    "width": int(eg.get("width", 0)),
                    "height": int(eg.get("height", 0)),
                    "height_mm": float(eg.get("height_mm", 0.0)) or round(scale_calib.px_to_mm(float(eg.get("height", 0))), 2),
                    "width_mm": float(eg.get("width_mm", 0.0)) or round(scale_calib.px_to_mm(float(eg.get("width", 0))), 2),
                    "height_dmm": float(eg.get("height_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(float(eg.get("height", 0))), 1),
                    "width_dmm": float(eg.get("width_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(float(eg.get("width", 0))), 1),
                    "area_mm2": float(eg.get("area_mm2", 0.0)) or round(scale_calib.area_px2_to_mm2(float(eg.get("area", 0))), 2),
                    "stroke_count": int(len(eg.get(stroke_key) or eg.get("strokes", []))),
                    "loop_count": int(eg.get("loop_count", 0)),
                    "topology_group": eg.get("topology_group", ""),
                    "eva_char": eg.get("eva_char", ""),
                    "crop_uri": eg.get("crop_uri", ""),
                    "svg_content": eg.get("euler_svg_content", "") if stroke_key == "euler_strokes" else eg.get("svg_content", eg.get("ductus_svg", "")),
                    "euler_svg_content": eg.get("euler_svg_content", ""),
                    "euler_stroke_count": int(eg.get("euler_stroke_count", len(eg.get("euler_strokes", [])))),
                    "png_rel": eg.get("png_rel", ""),
                    "svg_rel": eg.get("svg_rel", ""),
                    "is_outlier": False,
                    "outlier_bin": "INLIER_CONFORMANT",
                    "medoid_distance": 0.0
                })

            # Real empirical occurrence count across the processed glyphs (zero hardcoded multipliers)
            corpus_freq = len(cluster_glyphs)
            corpus_pct = round(corpus_freq / float(max(1, len(glyphs))) * 100.0, 2)
            indexed_pages = sorted({str(g.get("page_id", "")) for g in cluster_glyphs if g.get("page_id")})

            alphabet_entries.append({
                "type_id": type_name,
                "rank": rank,
                "cluster_index": int(cluster_id),
                "ductus_engine": "euler_bernoulli" if stroke_key == "euler_strokes" else "quill_40",
                "frequency": corpus_freq,
                "folio_frequency": len(cluster_glyphs),
                "percentage": corpus_pct,
                "indexed_pages": indexed_pages,
                "page_count": len(indexed_pages),
                "topology_group": cluster_topology,
                "loop_count": cluster_loops,
                "exemplar_id": exemplar_glyph.get("glyph_id", ""),
                "exemplar_page_id": exemplar_glyph.get("page_id", ""),
                "exemplar_bbox": [int(x) for x in exemplar_glyph.get("bbox", [0, 0, 0, 0])],
                "exemplar_crop_uri": exemplar_glyph.get("crop_uri", ""),
                "exemplar_png": exemplar_glyph.get("png_rel", ""),
                "exemplar_svg": exemplar_glyph.get("svg_rel", ""),
                "exemplar_svg_content": exemplar_glyph.get("svg_content", exemplar_glyph.get("ductus_svg", "")),
                "exemplar_euler_svg_content": exemplar_glyph.get("euler_svg_content", ""),
                "mean_strokes": round(float(np.mean([len(g.get(stroke_key) or g.get("strokes", [])) for g in cluster_glyphs])), 2),
                "mean_height": mean_height,
                "mean_width": mean_width,
                "mean_height_mm": float(exemplar_glyph.get("height_mm", 0.0)) or round(scale_calib.px_to_mm(mean_height), 2),
                "mean_width_mm": float(exemplar_glyph.get("width_mm", 0.0)) or round(scale_calib.px_to_mm(mean_width), 2),
                "mean_height_dmm": float(exemplar_glyph.get("height_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(mean_height), 1),
                "mean_width_dmm": float(exemplar_glyph.get("width_dmm", 0.0)) or round(scale_calib.px_to_tenths_mm(mean_width), 1),
                "inliers_count": inlier_count,
                "outliers_count": outlier_count,
                "outlier_ratio": round(outlier_count / max(1, len(cluster_glyphs)), 3),
                "rejection_bins": rejection_bins_counts,
                "corpus_match": corpus_match,
                "corpus_matches": multi_matches,
                "total_instances_count": corpus_freq,
                "folio_instances_count": len(cluster_glyphs),
                "all_instances": sample_instances,
                "sample_instances": sample_instances
            })

        return {
            "total_glyphs": len(glyphs),
            "canonical_alphabet_size": len(alphabet_entries),
            "emergence_metadata": emergence_meta,
            "alphabet": alphabet_entries,
        }

    def build_dual_catalogues(
        self,
        glyphs: List[Dict[str, Any]],
        corpus_type: str = "voynich",
        scale_calib: Optional[MetricScaleCalibration] = None
    ) -> Dict[str, Any]:
        """
        Builds two independent Canonical Alphabets on the exact same glyphs:
        1. `alphabet`: clustered using Quill 40° / Ribbon kinematics + DINOv2 + topology.
        2. `euler_alphabet`: clustered using Euler-Bernoulli skeleton graph kinematics + DINOv2 + topology.
        """
        quill_cat = self.build_catalogue(
            glyphs, corpus_type=corpus_type, scale_calib=scale_calib,
            stroke_key="strokes", tag_field="canonical_type"
        )
        euler_cat = self.build_catalogue(
            glyphs, corpus_type=corpus_type, scale_calib=scale_calib,
            stroke_key="euler_strokes", tag_field="euler_canonical_type"
        )
        quill_cat["euler_alphabet"] = euler_cat.get("alphabet", [])
        quill_cat["euler_alphabet_size"] = len(quill_cat["euler_alphabet"])
        return quill_cat

    def export_catalogue_json(self, catalogue: Dict[str, Any], output_path: Path):
        """Exports the canonical alphabet catalogue as a clean JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2, ensure_ascii=False)
