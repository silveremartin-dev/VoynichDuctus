"""
Word-Level Decomposition Engine (word_decomposition).
Unifies the 3 word-to-glyph decomposition paradigms:
1. Decomposition A (Topological & Valley Decomposition): Loops, connected components, diacritics, and ink valleys.
2. Decomposition B (Spatial PCA Decomposition): 1D horizontal latent trajectory Z(x) in R^32 and PCA transition boundaries.
3. Decomposition C (Hybrid Consensus Decomposition): Mathematical arbitration between A and B with Euler loop protection.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from sklearn.decomposition import PCA, NMF


class WordScribalEncoder:
    """
    Encodes full 2D word patches into 1D horizontal latent trajectories Z(x) in R^D.
    Captures stroke height, curvature, nib angle, and loop configurations along the reading axis.
    """

    def __init__(self, target_height: int = 48, latent_dim: int = 32):
        self.target_height = target_height
        self.latent_dim = latent_dim

    def encode_word_patch(self, word_img: np.ndarray) -> np.ndarray:
        """
        Encodes a 2D word crop into a deterministic 1D sequence of 32-D geometric & topological
        column descriptors Z(x) along reading axis X (ink mass, vertical moments, vertical run/hole
        transitions, 8-band vertical density profile, and horizontal derivative context).
        Never uses unseeded/untrained random neural weights.
        Returns array of shape (W_eff, D).
        """
        if word_img is None or word_img.size == 0:
            return np.zeros((1, self.latent_dim), dtype=np.float32)

        if word_img.ndim == 3:
            gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # On raw BGR/RGB parchment scans, darker pixels carry ink: invert to ink-positive [0, 1]
            if gray.max() > 1.0:
                gray = 1.0 - (gray / 255.0)
            gray = np.clip(gray - float(np.percentile(gray, 35)), 0.0, 1.0)
            if gray.max() > 1e-4:
                gray /= float(gray.max())
        else:
            gray = word_img.astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0

        H, W = gray.shape[:2]
        if H <= 2 or W <= 2:
            return np.zeros((1, self.latent_dim), dtype=np.float32)

        scale = float(self.target_height) / float(H)
        target_w = max(4, int(round(W * scale)))
        resized = cv2.resize(gray, (target_w, self.target_height), interpolation=cv2.INTER_AREA)

        ys = np.arange(self.target_height, dtype=np.float32)
        h_f = float(self.target_height)
        raw_cols = []
        for x in range(target_w):
            col = resized[:, x]
            total_ink = float(np.sum(col))
            desc = np.zeros(self.latent_dim, dtype=np.float32)
            if total_ink >= 1e-4:
                cy = float(np.sum(ys * col) / total_ink)
                var_y = float(np.sum(((ys - cy) ** 2) * col) / total_ink)
                active = (col > 0.18)
                ink_rows = np.where(active)[0]
                top_y = float(ink_rows[0]) if len(ink_rows) > 0 else 0.0
                bot_y = float(ink_rows[-1]) if len(ink_rows) > 0 else h_f
                span_y = max(0.0, bot_y - top_y)
                # Count vertical 0->1 transitions (2+ runs in one column signals an enclosed loop or arch!)
                runs = int(np.sum(active[1:] & (~active[:-1]))) + (1 if active[0] else 0)

                sector_means = [float(np.mean(chunk)) for chunk in np.array_split(col, 8)]
                upper_mass = float(np.mean(col[:self.target_height // 3]))
                mid_mass = float(np.mean(col[self.target_height // 3: 2 * self.target_height // 3]))
                lower_mass = float(np.mean(col[2 * self.target_height // 3:]))

                desc[0] = total_ink / h_f
                desc[1] = cy / h_f
                desc[2] = np.sqrt(max(0.0, var_y)) / h_f
                desc[3] = top_y / h_f
                desc[4] = bot_y / h_f
                desc[5] = span_y / h_f
                desc[6] = min(1.0, runs / 3.0)
                desc[7] = upper_mass
                desc[8] = mid_mass
                desc[9] = lower_mass
                for k, val in enumerate(sector_means):
                    desc[10 + k] = val
            raw_cols.append(desc)

        arr = np.array(raw_cols, dtype=np.float32)
        # Populate channels 18..31 with horizontal local gradient context (dZ/dx over 1 and 2 steps)
        if target_w >= 3:
            grad1 = np.zeros_like(arr[:, :7])
            grad2 = np.zeros_like(arr[:, :7])
            grad1[1:-1] = 0.5 * (arr[2:, :7] - arr[:-2, :7])
            if target_w >= 5:
                grad2[2:-2] = 0.5 * (arr[4:, 10:17] - arr[:-4, 10:17])
            arr[:, 18:25] = grad1
            arr[:, 25:32] = grad2
        return arr


class NeuralWordDecomposition:
    """
    Decomposition B (Spatial PCA Word Decomposition):
    Decomposes continuous words into constituent glyph masks using 1D horizontal PCA / NMF
    along latent state transition boundaries || dZ/dx ||.
    """

    def __init__(self, n_components: int = 36, window_size: int = 14):
        self.n_components = n_components
        self.window_size = window_size
        self.encoder = WordScribalEncoder(latent_dim=32)
        self.pca = PCA(n_components=min(n_components, 24))
        self.is_fitted = False

    def fit_on_word_corpus(self, word_patches: List[np.ndarray]):
        """Fits the spatial PCA dictionary on a corpus of word image patches."""
        sliding_samples = []
        for patch in word_patches:
            traj = self.encoder.encode_word_patch(patch)
            if len(traj) < self.window_size:
                continue
            for x in range(0, len(traj) - self.window_size + 1, max(1, self.window_size // 3)):
                window = traj[x:x + self.window_size].flatten()
                sliding_samples.append(window)

        if len(sliding_samples) >= self.n_components:
            X = np.array(sliding_samples, dtype=np.float32)
            n_c = min(self.n_components, X.shape[0], X.shape[1])
            self.pca = PCA(n_components=n_c)
            self.pca.fit(X)
            self.is_fitted = True

    @staticmethod
    def _build_tight_slice_masks(
        word_mask: np.ndarray,
        cuts: List[int],
        scale_calib: Optional[Any] = None
    ) -> List[np.ndarray]:
        """
        Builds per-slice boolean masks from column cuts while reassigning clipped
        overhang tips of adjacent tall letters back to their owning slice so small
        letters never stretch vertically over blank parchment.
        """
        from scipy.ndimage import label
        H, W = word_mask.shape[:2]
        if scale_calib is not None:
            px_per_mm = float(scale_calib.px_per_mm)
        else:
            px_per_mm = max(3.5, float(H) / 4.8)

        v_gap_thresh = max(4.0, 0.65 * px_per_mm)
        stray_area = max(8, int(round(0.25 * (px_per_mm ** 2))))

        n_slices = max(0, len(cuts) - 1)
        if n_slices == 0:
            return []

        slice_masks = []
        for idx in range(n_slices):
            c0, c1 = cuts[idx], cuts[idx + 1]
            sub_m = np.zeros_like(word_mask, dtype=bool)
            if c1 > c0:
                sub_m[:, c0:c1] = word_mask[:, c0:c1]
            slice_masks.append(sub_m)

        lbl_word, _ = label(word_mask, structure=np.ones((3, 3), dtype=bool))

        for s_idx in range(n_slices):
            sm = slice_masks[s_idx]
            if not np.any(sm):
                continue
            lbl_s, n_s = label(sm, structure=np.ones((3, 3), dtype=bool))
            if n_s < 2:
                continue

            comps = []
            for cid in range(1, n_s + 1):
                pts = np.argwhere(lbl_s == cid)
                r0, c0_l = pts.min(axis=0)
                r1, c1_l = pts.max(axis=0) + 1
                comps.append((cid, len(pts), int(r0), int(r1), int(c0_l), int(c1_l)))

            # Anchor = largest connected component in this slice
            main_cid, main_area, main_r0, main_r1, _, _ = max(comps, key=lambda x: x[1])
            for cid, area, r0, r1, c0_l, c1_l in comps:
                if cid == main_cid:
                    continue
                v_gap = max(0, max(main_r0 - r1, r0 - main_r1))
                if v_gap < v_gap_thresh or area >= 0.45 * main_area:
                    continue

                # Check if this disconnected upper/lower piece actually belongs to a neighbor's tall CC
                frag_mask = (lbl_s == cid)
                parent_labels = lbl_word[frag_mask]
                parent_labels = parent_labels[parent_labels > 0]
                reassigned = False
                if len(parent_labels) > 0:
                    p_id = int(np.bincount(parent_labels).argmax())
                    parent_cc = (lbl_word == p_id)
                    best_nb = -1
                    best_nb_area = area
                    for nb_idx in (s_idx - 1, s_idx + 1):
                        if 0 <= nb_idx < n_slices:
                            nb_c0, nb_c1 = cuts[nb_idx], cuts[nb_idx + 1]
                            nb_overlap = int(np.sum(parent_cc[:, nb_c0:nb_c1]))
                            if nb_overlap > best_nb_area * 1.35:
                                best_nb_area = nb_overlap
                                best_nb = nb_idx
                    if best_nb >= 0:
                        slice_masks[best_nb] |= frag_mask
                        slice_masks[s_idx] &= (~frag_mask)
                        reassigned = True

                # If it was not reassigned to a neighbor but is a tiny far-edge border speck, strip it
                if not reassigned and area <= stray_area and (r0 <= 1 or r1 >= H - 1) and v_gap >= v_gap_thresh * 1.2:
                    slice_masks[s_idx] &= (~frag_mask)

        return slice_masks

    def decompose_word(
        self,
        word_mask: np.ndarray,
        expected_glyph_count: Optional[int] = None,
        scale_calib: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Segments word into constituent glyphs by finding latent state transition boundaries
        along the horizontal trajectory and whitespace/minimum-ink valleys, using physical
        metric scale calibration (mm / 0.1 mm).
        """
        H, W = word_mask.shape[:2]
        if scale_calib is not None:
            px_per_mm = float(scale_calib.px_per_mm)
            min_pitch_px = scale_calib.ligature_valley_min_pitch_px
            min_margin_px = max(6, int(round(scale_calib.mm_to_px(1.10))))
            min_splinter_px = max(5, int(round(scale_calib.mm_to_px(0.85))))
            snap_r_px = max(2, int(round(scale_calib.mm_to_px(0.55))))
            min_hole_area = scale_calib.min_loop_hole_area_px
            min_hole_h = scale_calib.min_loop_hole_height_px
            min_hole_w = scale_calib.min_loop_hole_width_px
        else:
            px_per_mm = max(3.5, H / 4.2)
            min_pitch_px = max(8, int(round(1.45 * px_per_mm)))
            min_margin_px = max(6, int(round(1.10 * px_per_mm)))
            min_splinter_px = max(5, int(round(0.85 * px_per_mm)))
            snap_r_px = max(2, int(round(0.55 * px_per_mm)))
            min_hole_area = max(6, int(round(0.18 * (px_per_mm ** 2))))
            min_hole_h = max(3, int(round(0.50 * px_per_mm)))
            min_hole_w = max(2, int(round(0.38 * px_per_mm)))

        if W <= min_margin_px:
            pts_w = np.argwhere(word_mask)
            if len(pts_w) > 0:
                r0, c0 = pts_w.min(axis=0)
                r1, c1 = pts_w.max(axis=0) + 1
                m_out = word_mask[r0:r1, c0:c1]
                bbox_out = (int(r0), int(c0), int(r1), int(c1))
            else:
                m_out = word_mask
                bbox_out = (0, 0, H, W)
            return [{
                "mask": m_out,
                "bbox_local": bbox_out,
                "area": int(np.sum(word_mask)),
                "neural_basis_id": "PHI_00",
                "segmentation_type": "NEURAL_WORD_PCA",
                "word_decomposition": "neural_word_pca"
            }]

        # Trim tiny noise specks before neural column encoding when full word image is non-trivial
        work_mask = word_mask.astype(bool)
        if H >= 16 and W >= 16 and int(np.sum(work_mask)) > 20:
            from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
            work_mask = GlyphSegmenter._clean_word_mask_border_bleed(work_mask, scale_calib=scale_calib)

        # Protect Euler closed loops from being bisected by neural derivative spikes
        from scipy.ndimage import binary_fill_holes, label
        filled = binary_fill_holes(work_mask)
        holes = filled & (~work_mask)
        lbl_h, num_h = label(holes)
        forbidden_cols = set()
        for h_idx in range(1, num_h + 1):
            hm = (lbl_h == h_idx)
            if int(np.sum(hm)) < min_hole_area:
                continue
            h_cols = np.where(np.any(hm, axis=0))[0]
            h_rows = np.where(np.any(hm, axis=1))[0]
            if (
                len(h_cols) >= min_hole_w
                and len(h_rows) >= min_hole_h
                and (h_cols.max() - h_cols.min() + 1) >= min_hole_w
                and (h_rows.max() - h_rows.min() + 1) >= min_hole_h
            ):
                for fc in range(max(0, int(h_cols.min()) - 1), min(W, int(h_cols.max()) + 2)):
                    forbidden_cols.add(fc)

        traj = self.encoder.encode_word_patch(work_mask.astype(np.float32))
        W_eff = len(traj)
        scale_x = float(W) / float(max(1, W_eff))

        # Project sliding horizontal windows through fitted Spatial PCA when available
        diffs = np.zeros(W_eff, dtype=np.float32)
        if self.is_fitted and W_eff >= self.window_size + 2:
            half_win = self.window_size // 2
            padded_traj = np.pad(traj, ((half_win, self.window_size - half_win - 1), (0, 0)), mode="edge")
            win_list = [padded_traj[i:i + self.window_size].flatten() for i in range(W_eff)]
            latent_seq = self.pca.transform(np.array(win_list, dtype=np.float32))
            for t in range(1, W_eff - 1):
                diffs[t] = float(np.linalg.norm(latent_seq[t + 1] - latent_seq[t - 1]))
        else:
            for t in range(1, W_eff - 1):
                diffs[t] = float(np.linalg.norm(traj[t + 1] - traj[t - 1]))

        from scipy.signal import find_peaks
        min_dist_eff = max(4, int(round(min_pitch_px / max(1e-4, scale_x))))
        peaks, _ = find_peaks(diffs, distance=min_dist_eff, prominence=float(np.mean(diffs) * 0.45))

        col_proj = np.sum(work_mask, axis=0).astype(np.float32)
        raw_cols = [int(round(p * scale_x)) for p in peaks]

        # Snap each latent peak to the local minimum-ink valley nearby so cuts fall between letters
        snapped_cols = []
        for c in raw_cols:
            lo = max(min_margin_px, c - snap_r_px)
            hi = min(W - min_margin_px, c + snap_r_px + 1)
            if lo < hi:
                best_c = lo + int(np.argmin(col_proj[lo:hi]))
                snapped_cols.append(best_c)
            elif min_margin_px <= c <= W - min_margin_px:
                snapped_cols.append(c)

        # Also harvest clear vertical whitespace valleys separating disjoint components
        zero_cols = (col_proj == 0)
        in_gap = False
        g_start = 0
        gap_centers = []
        for x in range(W):
            if zero_cols[x] and not in_gap:
                in_gap = True
                g_start = x
            elif not zero_cols[x] and in_gap:
                in_gap = False
                mid = (g_start + x - 1) // 2
                if min_margin_px <= mid <= W - min_margin_px:
                    gap_centers.append(mid)

        candidate_cols = []
        for c in sorted(set(gap_centers + snapped_cols)):
            if c in forbidden_cols:
                continue
            if all(abs(c - kept) >= min_pitch_px for kept in candidate_cols):
                candidate_cols.append(c)
        boundary_cols = sorted(candidate_cols)

        if expected_glyph_count is not None and expected_glyph_count >= 1:
            target_splits = expected_glyph_count - 1
            if len(boundary_cols) > target_splits:
                scored = sorted(boundary_cols, key=lambda c: (col_proj[c], -diffs[min(W_eff - 1, int(c / scale_x))]))
                boundary_cols = sorted(scored[:target_splits])
            elif len(boundary_cols) < target_splits and expected_glyph_count > 1:
                step = W / float(expected_glyph_count)
                boundary_cols = [int(round(i * step)) for i in range(1, expected_glyph_count)]

        cuts = [0] + boundary_cols + [W]
        slice_masks = self._build_tight_slice_masks(work_mask, cuts, scale_calib=scale_calib)

        glyphs = []
        for idx, sub_m in enumerate(slice_masks):
            ink_pts = np.argwhere(sub_m)
            if len(ink_pts) == 0:
                continue
            rmin, cmin = ink_pts.min(axis=0)
            rmax, cmax = ink_pts.max(axis=0) + 1
            area = int(len(ink_pts))
            if (cmax - cmin) < min_splinter_px and len(glyphs) > 0:
                prev = glyphs[-1]
                pr0, pc0, pr1, pc1 = prev["bbox_local"]
                nr0, nc0 = min(pr0, int(rmin)), min(pc0, int(cmin))
                nr1, nc1 = max(pr1, int(rmax)), max(pc1, int(cmax))
                comb_m = np.zeros((nr1 - nr0, nc1 - nc0), dtype=bool)
                comb_m[pr0 - nr0:pr1 - nr0, pc0 - nc0:pc1 - nc0] |= prev["mask"]
                comb_m[int(rmin) - nr0:int(rmax) - nr0, int(cmin) - nc0:int(cmax) - nc0] |= sub_m[rmin:rmax, cmin:cmax]
                prev["mask"] = comb_m
                prev["bbox_local"] = (nr0, nc0, nr1, nc1)
                prev["area"] = int(np.sum(comb_m))
                continue

            glyphs.append({
                "mask": sub_m[rmin:rmax, cmin:cmax],
                "bbox_local": (int(rmin), int(cmin), int(rmax), int(cmax)),
                "area": area,
                "neural_basis_id": f"PHI_{idx:02d}",
                "segmentation_type": "NEURAL_WORD_PCA",
                "word_decomposition": "neural_word_pca"
            })

        return glyphs if glyphs else [{
            "mask": work_mask,
            "bbox_local": (0, 0, H, W),
            "area": int(np.sum(work_mask)),
            "neural_basis_id": "PHI_00",
            "segmentation_type": "NEURAL_WORD_PCA",
            "word_decomposition": "neural_word_pca"
        }]


class HybridConsensusWordDecomposition:
    """
    Decomposition C (Hybrid Neuro-Topological Consensus Word Decomposition):
    Synthesizes topological/valley boundaries (Decomposition A) and spatial PCA latent boundaries (Decomposition B)
    with kinetic ink-profile minima and topological hole protection.
    """

    def __init__(self, tolerance_px: int = 4, tolerance_mm: float = 0.50):
        self.tolerance_px = tolerance_px
        self.tolerance_mm = tolerance_mm

    def decompose_word_hybrid(
        self,
        word_mask: np.ndarray,
        option_a_glyphs: List[Dict[str, Any]],
        option_b_glyphs: List[Dict[str, Any]],
        expected_glyph_count: Optional[int] = None,
        scale_calib: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuses Decomposition A and Decomposition B into consensus boundaries using physical metric units (mm).
        """
        H, W = word_mask.shape[:2]
        if scale_calib is not None:
            tol_px = max(2, int(round(scale_calib.mm_to_px(self.tolerance_mm))))
            min_pitch_px = scale_calib.ligature_valley_min_pitch_px
            min_margin_px = max(6, int(round(scale_calib.mm_to_px(1.20))))
            min_hole_area = scale_calib.min_loop_hole_area_px
            min_hole_h = scale_calib.min_loop_hole_height_px
            min_hole_w = scale_calib.min_loop_hole_width_px
        else:
            est_px_per_mm = max(3.0, H / 4.2)
            tol_px = self.tolerance_px
            min_pitch_px = max(8, int(round(1.45 * est_px_per_mm)))
            min_margin_px = max(6, int(round(1.20 * est_px_per_mm)))
            min_hole_area = max(6, int(round(0.18 * (est_px_per_mm ** 2))))
            min_hole_h = max(3, int(round(0.50 * est_px_per_mm)))
            min_hole_w = max(2, int(round(0.38 * est_px_per_mm)))

        if W <= min_margin_px:
            return [{
                "mask": word_mask,
                "bbox_local": (0, 0, H, W),
                "area": int(np.sum(word_mask)),
                "hybrid_basis_id": "HYB_00",
                "segmentation_type": "HYBRID_CONSENSUS_C",
                "word_decomposition": "hybrid_consensus"
            }]

        cuts_a = [g["bbox_local"][3] for g in option_a_glyphs[:-1]] if len(option_a_glyphs) > 1 else []
        cuts_b = [g["bbox_local"][3] for g in option_b_glyphs[:-1]] if len(option_b_glyphs) > 1 else []

        from scipy.ndimage import binary_fill_holes, label
        filled = binary_fill_holes(word_mask)
        holes = filled & (~word_mask)
        lbl_h, num_h = label(holes)
        forbidden_cols = set()
        for h_idx in range(1, num_h + 1):
            hm = (lbl_h == h_idx)
            if np.sum(hm) < min_hole_area:
                continue
            h_cols = np.where(np.any(hm, axis=0))[0]
            h_rows = np.where(np.any(hm, axis=1))[0]
            if len(h_cols) >= min_hole_w and len(h_rows) >= min_hole_h and (h_cols.max() - h_cols.min() + 1) >= min_hole_w and (h_rows.max() - h_rows.min() + 1) >= min_hole_h:
                for fc in range(max(0, int(h_cols.min()) - 1), min(W, int(h_cols.max()) + 2)):
                    forbidden_cols.add(fc)

        col_proj = np.sum(word_mask, axis=0).astype(float)
        max_proj = np.max(col_proj) if len(col_proj) > 0 else 1.0

        consensus_cuts = []
        used_b = set()

        for ca in cuts_a:
            if ca in forbidden_cols:
                continue
            matched_b = [cb for cb in cuts_b if abs(ca - cb) <= tol_px and cb not in used_b]
            if matched_b:
                cb_best = matched_b[0]
                used_b.add(cb_best)
                consensus_cuts.append(int(round((ca + cb_best) / 2.0)))
            else:
                if col_proj[min(W - 1, ca)] <= 0.65 * max_proj:
                    consensus_cuts.append(ca)

        for cb in cuts_b:
            if cb in used_b or cb in forbidden_cols:
                continue
            if col_proj[min(W - 1, cb)] <= 0.65 * max_proj:
                consensus_cuts.append(cb)

        consensus_cuts = sorted(list(set(c for c in consensus_cuts if min_margin_px <= c <= W - min_margin_px)))

        filtered_cuts = []
        for c in consensus_cuts:
            if all(abs(c - fc) >= min_pitch_px for fc in filtered_cuts):
                filtered_cuts.append(c)
        filtered_cuts.sort()

        if expected_glyph_count is not None and expected_glyph_count >= 1:
            target_splits = expected_glyph_count - 1
            if len(filtered_cuts) > target_splits:
                scored = sorted(filtered_cuts, key=lambda c: col_proj[c])
                filtered_cuts = sorted(scored[:target_splits])

        cuts = [0] + filtered_cuts + [W]
        work_mask = word_mask.astype(bool)
        if H >= 16 and W >= 16 and int(np.sum(work_mask)) > 20:
            from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
            work_mask = GlyphSegmenter._clean_word_mask_border_bleed(work_mask, scale_calib=scale_calib)
        slice_masks = NeuralWordDecomposition._build_tight_slice_masks(work_mask, cuts, scale_calib=scale_calib)

        glyphs = []
        for idx, sub_m in enumerate(slice_masks):
            ink_pts = np.argwhere(sub_m)
            if len(ink_pts) == 0:
                continue
            rmin, cmin = ink_pts.min(axis=0)
            rmax, cmax = ink_pts.max(axis=0) + 1
            area = int(len(ink_pts))

            glyphs.append({
                "mask": sub_m[rmin:rmax, cmin:cmax],
                "bbox_local": (int(rmin), int(cmin), int(rmax), int(cmax)),
                "area": area,
                "hybrid_basis_id": f"HYB_{idx:02d}",
                "segmentation_type": "HYBRID_CONSENSUS_C",
                "word_decomposition": "hybrid_consensus"
            })

        return glyphs if glyphs else [{
            "mask": word_mask,
            "bbox_local": (0, 0, H, W),
            "area": int(np.sum(word_mask)),
            "hybrid_basis_id": "HYB_00",
            "segmentation_type": "HYBRID_CONSENSUS_C",
            "word_decomposition": "hybrid_consensus"
        }]


# Backward-compatible aliases
NeuralWordDecomposer = NeuralWordDecomposition
HybridNeuroKinematicDecomposer = HybridConsensusWordDecomposition


class GenerativeTypewriterDecomposer:
    """
    Closed-Loop Unsupervised Word-to-Glyph Decomposer (inspired by 'The Learnable Typewriter'
    [Siglidis et al., ICDAR] + Minimum Description Length Viterbi Decoding + Ductus Constraints).

    Operates 100% blind to human transliterations (EVA/Currier/v101):
    1. Consolidates all-pairs vertically stacked / interlocking strokes (overhead 'sh' loops,
       gallows crossbars, bench plumes) and decouples supralinear diacritics.
    2. Proposes primitive cuts along 2D minimum-energy ink seams between multi-loop ligatures
       and wide cursive joins while protecting structural horizontal benches/arches ('ch', 'sh', 'k', 'cth').
    3. Induces a corpus-wide bank of 24x24 soft visual character sprites {P_k} with empirical
       MDL code lengths (-log2 p_k) and physical metric priors (mm).
    4. Decodes each word via exact Viterbi Dynamic Programming so multi-stroke letters separated
       by micro pen-lifts ('c'+'i' -> 'a', 'c'+'h' -> 'ch', 'i'+'i' -> 'ii') coalesce into one
       recurring archetype while cursive ligatures ('da', 'ol', 'ko', 'okol') split cleanly.
    """

    def __init__(self, sprite_size: int = 24, max_prototypes: int = 32, is_serafini: bool = False):
        self.sprite_size = sprite_size
        self.max_prototypes = max_prototypes
        self.is_serafini = is_serafini
        self.prototypes: List[Dict[str, Any]] = []
        self.pca_decomposer = NeuralWordDecomposition()
        self.is_fitted: bool = False

    @staticmethod
    def _mask_to_sprite(mask: np.ndarray, dim: int = 24) -> np.ndarray:
        """Normalizes a binary glyph mask into an aspect-preserving float32 [0, 1] sprite."""
        pts = np.argwhere(mask)
        if len(pts) == 0:
            return np.zeros((dim, dim), dtype=np.float32)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0) + 1
        crop = mask[r0:r1, c0:c1].astype(np.float32)
        ch, cw = crop.shape
        inner = dim - 4
        scale = min(float(inner) / max(1, ch), float(inner) / max(1, cw))
        nh = max(2, min(inner, int(round(ch * scale))))
        nw = max(2, min(inner, int(round(cw * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (nw, nh), interpolation=interp)
        out = np.zeros((dim, dim), dtype=np.float32)
        yo = (dim - nh) // 2
        xo = (dim - nw) // 2
        out[yo:yo + nh, xo:xo + nw] = resized
        return out

    @staticmethod
    def _count_mask_holes_and_spans(
        mask: np.ndarray,
        px_per_mm: float,
    ) -> Tuple[int, List[Tuple[int, int, float]]]:
        """Returns (loop_count, [(x0, x1, y_center_rel), ...]) for genuine interior loops."""
        from scipy.ndimage import binary_fill_holes, binary_closing, label

        H, W = mask.shape[:2]
        min_area = max(4, int(round(0.065 * (px_per_mm ** 2))))
        min_h = max(2, int(round(0.34 * px_per_mm)))
        min_w = max(2, int(round(0.28 * px_per_mm)))

        padded = np.pad(mask.astype(bool), 2, mode="constant", constant_values=False)
        holes = (binary_fill_holes(padded) & (~padded))[2:-2, 2:-2]
        lbl_h, num_h = label(holes)
        spans: List[Tuple[int, int, float]] = []
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)
            if int(np.sum(hm)) < min_area:
                continue
            ys, xs = np.where(hm)
            if (ys.max() - ys.min() + 1) >= min_h and (xs.max() - xs.min() + 1) >= min_w:
                spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))

        if not spans and H >= 8 and W >= 8:
            # Check 1-px nearly-closed bowls
            closed_p = binary_closing(padded, structure=np.ones((2, 2), dtype=bool))
            holes_c = (binary_fill_holes(closed_p) & (~closed_p))[2:-2, 2:-2]
            lbl_c, num_c = label(holes_c)
            min_area_c = max(7, int(round(0.11 * (px_per_mm ** 2))))
            for hid in range(1, num_c + 1):
                hm = (lbl_c == hid)
                if int(np.sum(hm)) >= min_area_c:
                    ys, xs = np.where(hm)
                    if (ys.max() - ys.min() + 1) >= min_h and (xs.max() - xs.min() + 1) >= min_w:
                        spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))

        spans.sort(key=lambda s: s[0])
        return len(spans), spans

    @staticmethod
    def _estimate_skeleton_stroke_complexity(mask: np.ndarray, px_per_mm: float) -> int:
        """Fast topological estimate of required quill strokes on a candidate glyph mask."""
        from skimage.morphology import skeletonize
        from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer

        if not np.any(mask):
            return 0
        skel = skeletonize(mask.astype(bool))
        n_sk = int(np.sum(skel))
        if n_sk <= 5:
            return 1
        s_u8 = skel.astype(np.uint8)
        deg = cv2.filter2D(s_u8, cv2.CV_16U, CalligraphicVectorizer._DEG_KERNEL, borderType=cv2.BORDER_CONSTANT) * s_u8
        n_ep = int(np.sum(deg == 1))
        n_jn = int(np.sum(deg >= 3))
        # Euler graph branch heuristic: ~half of free endpoints, minimum 1
        est = max(1, int(round((n_ep + 0.35 * n_jn) / 2.0)))
        return est

    def _build_word_lattice(
        self,
        word_mask: np.ndarray,
        scale_calib: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, float]:
        """
        Builds a strictly disjoint left-to-right candidate boundary lattice combining
        topological valley boundaries and 1D Spatial PCA transition boundaries outside
        protected Euler closed loops, reassigning clipped crossbar overhangs cleanly.
        """
        from scipy.ndimage import binary_fill_holes, label
        from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter

        w_mask = (word_mask > 0).astype(bool)
        H, W = w_mask.shape[:2]
        ppm = float(scale_calib.px_per_mm) if scale_calib is not None else max(4.5, float(H) / 4.5)
        if int(np.sum(w_mask)) == 0:
            return w_mask, [], np.zeros(W, dtype=float), ppm

        work = (
            GlyphSegmenter._clean_word_mask_border_bleed(w_mask, scale_calib=scale_calib)
            if (H >= 16 and W >= 16 and int(np.sum(w_mask)) > 20)
            else w_mask
        )

        # Protect genuine closed Euler loops from being bisected
        padded = np.pad(work, 2, mode="constant", constant_values=False)
        holes = (binary_fill_holes(padded) & (~padded))[2:-2, 2:-2]
        lbl_h, num_h = label(holes)
        forbidden = set()
        min_ha = max(4, int(round(0.08 * ppm * ppm)))
        min_hh = max(2, int(round(0.35 * ppm)))
        min_hw = max(2, int(round(0.28 * ppm)))
        loop_spans: List[Tuple[int, int, float]] = []
        for hid in range(1, num_h + 1):
            hm = (lbl_h == hid)
            if int(np.sum(hm)) >= min_ha:
                ys, xs = np.where(hm)
                if (ys.max() - ys.min() + 1) >= min_hh and (xs.max() - xs.min() + 1) >= min_hw:
                    loop_spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))
                    for fc in range(max(0, int(xs.min()) - 1), min(W, int(xs.max()) + 2)):
                        forbidden.add(fc)
        loop_spans.sort()

        seg = GlyphSegmenter(scale_calib=scale_calib)
        a_comps = seg.assemble_word_into_glyphs(work, scale_calib=scale_calib)
        b_comps = self.pca_decomposer.decompose_word(work, scale_calib=scale_calib)
        raw_cuts = [int(g["bbox_local"][3]) for g in a_comps[:-1]] + [int(g["bbox_local"][3]) for g in b_comps[:-1]]

        col_proj = np.sum(work, axis=0).astype(float)
        # Ensure baseline multi-loop ligatures ('da', 'oa', 'oo', 'oko') always carry a candidate cut between loops
        for li in range(len(loop_spans) - 1):
            lx0_a, lx1_a, ly_a = loop_spans[li]
            lx0_b, lx1_b, ly_b = loop_spans[li + 1]
            if not (ly_a < 0.38 and ly_b < 0.38) and lx0_b > lx1_a + 1:
                cands = [x for x in range(lx1_a + 1, lx0_b) if x not in forbidden]
                if cands:
                    raw_cuts.append(min(cands, key=lambda x: col_proj[x]))

        min_margin = max(5, int(round(0.95 * ppm)))
        min_atom = max(5, int(round(0.80 * ppm)))
        valid_c = sorted(set(c for c in raw_cuts if min_margin <= c <= W - min_margin and c not in forbidden))

        snap_r = max(2, int(round(0.30 * ppm)))
        snapped = []
        for c in valid_c:
            lo, hi = max(min_margin, c - snap_r), min(W - min_margin, c + snap_r + 1)
            opts = [x for x in range(lo, hi) if x not in forbidden]
            if opts:
                snapped.append(min(opts, key=lambda x: col_proj[x]))

        dedup: List[int] = []
        for c in sorted(set(snapped), key=lambda x: col_proj[x]):
            if all(abs(c - kc) >= min_atom for kc in dedup):
                dedup.append(c)
        dedup.sort()

        pts = [0] + dedup + [W]
        slices = NeuralWordDecomposition._build_tight_slice_masks(work, pts, scale_calib=scale_calib)
        return work, slices, col_proj, ppm

    def fit_on_word_masks(
        self,
        word_masks: List[np.ndarray],
        scale_calib: Optional[Any] = None,
        word_patches: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Induces the unsupervised Corpus Visual Sprite Dictionary {P_k} from all word masks on a page/batch.
        """
        from sklearn.cluster import AgglomerativeClustering

        if word_patches:
            self.pca_decomposer.fit_on_word_corpus(word_patches)
        elif word_masks:
            self.pca_decomposer.fit_on_word_corpus([wm.astype(np.float32) for wm in word_masks[:35] if wm is not None and wm.size > 16])

        min_area = int(round(scale_calib.min_glyph_area_px)) if scale_calib is not None else 8
        proto_cands: List[Tuple[np.ndarray, int, float, float]] = []

        for wm in word_masks:
            if wm is None or wm.size < 16 or not np.any(wm):
                continue
            _, slices, _, ppm = self._build_word_lattice(wm, scale_calib=scale_calib)
            for s_m in slices:
                if int(np.sum(s_m)) < min_area:
                    continue
                ys, xs = np.where(s_m)
                cm = s_m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                w_mm, h_mm = cm.shape[1] / ppm, cm.shape[0] / ppm
                if 0.85 <= w_mm <= 4.0 and h_mm >= 0.85:
                    l_n, _ = self._count_mask_holes_and_spans(cm, ppm)
                    if l_n <= 1:
                        proto_cands.append((self._mask_to_sprite(cm, self.sprite_size), min(1, l_n), w_mm, h_mm))

        if len(proto_cands) < 6:
            self.prototypes = []
            self.is_fitted = False
            return

        by_l: Dict[int, List[Tuple[np.ndarray, float, float]]] = {0: [], 1: []}
        for sp, ln, wm_v, hm_v in proto_cands:
            by_l[ln].append((sp, wm_v, hm_v))

        tot_c = max(1, len(proto_cands))
        prototypes: List[Dict[str, Any]] = []
        proto_idx = 0

        for ln, grp in by_l.items():
            if len(grp) < 4:
                continue
            if len(grp) > 350:
                grp = grp[::max(1, len(grp) // 350)]
            k_t = min(self.max_prototypes // 2, max(2, len(grp) // 12))
            X = np.hstack([
                np.array([g[0].ravel() for g in grp], dtype=np.float32),
                np.array([[g[1] * 0.5, g[2] * 0.5] for g in grp], dtype=np.float32),
            ])
            lbls = AgglomerativeClustering(n_clusters=k_t, linkage="ward").fit_predict(X)
            for cid in range(k_t):
                mems = [grp[idx] for idx in range(len(grp)) if lbls[idx] == cid]
                if not mems:
                    continue
                pr = (len(mems) + 1.0) / (tot_c + float(self.max_prototypes))
                prototypes.append({
                    "proto_id": f"TW_{proto_idx:02d}",
                    "loops": ln,
                    "sprite": np.mean([m[0] for m in mems], axis=0).astype(np.float32),
                    "w_mm": float(np.mean([m[1] for m in mems])),
                    "h_mm": float(np.mean([m[2] for m in mems])),
                    "mdl": float(-np.log2(max(1e-5, pr))),
                })
                proto_idx += 1

        self.prototypes = prototypes
        self.is_fitted = len(self.prototypes) > 0

    def decompose_word(
        self,
        word_mask: np.ndarray,
        expected_glyph_count: Optional[int] = None,
        scale_calib: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decomposes `word_mask` into constituent glyph masks via exact Viterbi MDL decoding
        over the disjoint candidate boundary lattice and learned visual sprite dictionary.
        """
        min_area = int(round(scale_calib.min_glyph_area_px)) if scale_calib is not None else 8
        work, slices, col_proj, ppm = self._build_word_lattice(word_mask, scale_calib=scale_calib)
        non_empty = [s for s in slices if int(np.sum(s)) >= min_area]

        if not non_empty:
            pts_w = np.argwhere(word_mask > 0)
            if len(pts_w) > 0:
                r0, c0 = pts_w.min(axis=0)
                r1, c1 = pts_w.max(axis=0) + 1
                return [{
                    "mask": (word_mask[r0:r1, c0:c1] > 0),
                    "bbox_local": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts_w)),
                    "diacritics": [],
                    "is_dot": False,
                    "typewriter_proto_id": "TW_00",
                    "segmentation_type": "GENERATIVE_TYPEWRITER",
                    "word_decomposition": "generative_typewriter",
                }]
            return []

        dot_dim_px = max(4, int(round(0.55 * ppm)))
        dot_area_px = max(10, int(round(0.35 * (ppm ** 2))))
        M = len(non_empty)

        if M == 1:
            ys, xs = np.where(non_empty[0])
            r0, r1 = int(ys.min()), int(ys.max()) + 1
            c0, c1 = int(xs.min()), int(xs.max()) + 1
            area = int(len(xs))
            return [{
                "mask": non_empty[0][r0:r1, c0:c1].copy(),
                "bbox_local": (r0, c0, r1, c1),
                "area": area,
                "diacritics": [],
                "is_dot": bool(((r1 - r0) <= dot_dim_px and (c1 - c0) <= dot_dim_px) or area <= dot_area_px),
                "typewriter_proto_id": "TW_00",
                "segmentation_type": "GENERATIVE_TYPEWRITER",
                "word_decomposition": "generative_typewriter",
            }]

        s_info: List[Tuple[int, int, float, float, int]] = []
        for s_m in non_empty:
            ys, xs = np.where(s_m)
            cm = s_m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            ln, _ = self._count_mask_holes_and_spans(cm, ppm)
            s_info.append((int(xs.min()), int(xs.max()), cm.shape[1] / ppm, cm.shape[0] / ppm, ln))

        bounds_ink: List[Tuple[float, float]] = []
        for k in range(M - 1):
            c_mid = int(round(0.5 * (s_info[k][1] + s_info[k + 1][0])))
            lo_b, hi_b = max(0, c_mid - 2), min(work.shape[1], c_mid + 3)
            min_col_ink = float(np.min(col_proj[lo_b:hi_b])) / ppm
            gap_mm = float(s_info[k + 1][0] - s_info[k][1] - 1) / ppm
            bounds_ink.append((min_col_ink, gap_mm))

        dp = [1e12] * (M + 1)
        dp_back: List[Optional[Tuple[int, Dict[str, Any]]]] = [None] * (M + 1)
        dp[0] = 0.0
        max_span = 1 if self.is_serafini else 3

        for i in range(M):
            if dp[i] >= 1e11:
                continue
            for sp_len in range(1, max_span + 1):
                j = i + sp_len
                if j > M:
                    break
                if sp_len > 1:
                    # Paleographic morphological completeness guards (100% blind to EVA):
                    # 1. Left prefix must be an open stroke (0 loops) <= 1.85 mm ('c', 'e', 'i', 's'-plume)
                    if s_info[i][4] >= 1 or s_info[i][2] > 1.85:
                        continue
                    # 2. Never merge two closed loops across lattice slices
                    if sum(1 for r in range(i, j) if s_info[r][4] >= 1) >= 2:
                        continue
                    max_g = max(bounds_ink[r][1] for r in range(i, j - 1))
                    if max_g > 0.42:
                        continue

                comb = non_empty[i] if sp_len == 1 else np.logical_or.reduce(non_empty[i:j])
                ys, xs = np.where(comb)
                r0, r1 = int(ys.min()), int(ys.max()) + 1
                c0, c1 = int(xs.min()), int(xs.max()) + 1
                cm = comb[r0:r1, c0:c1].copy()
                w_mm, h_mm = cm.shape[1] / ppm, cm.shape[0] / ppm
                l_n, _ = self._count_mask_holes_and_spans(cm, ppm)
                if sp_len > 1:
                    if l_n >= 2 or w_mm > (4.6 if h_mm >= 2.5 else 3.2):
                        continue
                    if sp_len == 3 and (h_mm < 2.5 or w_mm > 4.9):
                        continue

                best_c = 8.0
                best_pid = "TW_00"
                if self.is_fitted and self.prototypes:
                    spr = self._mask_to_sprite(cm, self.sprite_size)
                    s_sum = float(np.sum(spr))
                    same_p = [pr for pr in self.prototypes if pr["loops"] == min(1, l_n)] or self.prototypes
                    for pr in same_p:
                        inter = float(np.sum(np.minimum(spr, pr["sprite"])))
                        uni = s_sum + float(np.sum(pr["sprite"])) - inter
                        r_err = 1.0 - (inter / uni if uni > 1e-4 else 0.0)
                        c_v = 0.65 * pr["mdl"] + 3.8 * r_err + 0.25 * abs(w_mm - pr["w_mm"])
                        if c_v < best_c:
                            best_c = c_v
                            best_pid = pr["proto_id"]
                else:
                    best_c = 5.4

                target_w = 2.50 if h_mm >= 2.55 else 1.95
                width_prior = 0.70 * ((w_mm - target_w) ** 2)
                cut_pen = (2.0 * bounds_ink[j - 1][0]) if (j < M and bounds_ink[j - 1][0] > 0) else 0.0
                internal_gap_pen = sum(max(0.0, bounds_ink[r][1]) * 5.0 for r in range(i, j - 1))
                step_c = 2.25 + best_c + width_prior + cut_pen + internal_gap_pen

                if dp[i] + step_c < dp[j]:
                    dp[j] = dp[i] + step_c
                    area_c = int(len(xs))
                    dp_back[j] = (i, {
                        "mask": cm,
                        "bbox_local": (r0, c0, r1, c1),
                        "area": area_c,
                        "diacritics": [],
                        "is_dot": bool(((r1 - r0) <= dot_dim_px and (c1 - c0) <= dot_dim_px) or area_c <= dot_area_px),
                        "typewriter_proto_id": best_pid,
                        "segmentation_type": "GENERATIVE_TYPEWRITER",
                        "word_decomposition": "generative_typewriter",
                    })

        out_glyphs: List[Dict[str, Any]] = []
        cur = M
        while cur > 0 and dp_back[cur] is not None:
            prev_i, g_obj = dp_back[cur]
            out_glyphs.append(g_obj)
            cur = prev_i
        out_glyphs.reverse()
        return out_glyphs


def compute_bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union between two bounding boxes [y0, x0, y1, x1]."""
    y0_1, x0_1, y1_1, x1_1 = b1
    y0_2, x0_2, y1_2, x1_2 = b2

    oy0 = max(y0_1, y0_2)
    ox0 = max(x0_1, x0_2)
    oy1 = min(y1_1, y1_2)
    ox1 = min(x1_1, x1_2)

    if oy1 <= oy0 or ox1 <= ox0:
        return 0.0

    inter_area = float((oy1 - oy0) * (ox1 - ox0))
    area1 = float(max(1, (y1_1 - y0_1) * (x1_1 - x0_1)))
    area2 = float(max(1, (y1_2 - y0_2) * (x1_2 - x0_2)))
    union_area = area1 + area2 - inter_area
    return inter_area / max(1.0, union_area)


def cross_evaluate_options_consensus(
    glyphs_a: List[Dict[str, Any]],
    glyphs_b: List[Dict[str, Any]],
    glyphs_c: List[Dict[str, Any]],
    iou_thresh: float = 0.55
):
    """
    Evaluates multi-decomposition consensus across Decomposition A, B, and C.
    """
    for g_list, my_opt, others in [
        (glyphs_a, "A", [("B", glyphs_b), ("C", glyphs_c)]),
        (glyphs_b, "B", [("A", glyphs_a), ("C", glyphs_c)]),
        (glyphs_c, "C", [("A", glyphs_a), ("B", glyphs_b)])
    ]:
        for g in g_list:
            b_my = g.get("bbox", [0, 0, 0, 0])
            support = [my_opt]
            for opt_name, other_list in others:
                has_match = any(compute_bbox_iou(b_my, o.get("bbox", [0, 0, 0, 0])) >= iou_thresh for o in other_list)
                if has_match:
                    support.append(opt_name)

            support_sorted = sorted(support)
            g["support_options"] = support_sorted
            if len(support_sorted) == 3:
                g["support_badge"] = "A+B+C (Consensus)"
                g["is_consensus"] = True
                g["is_disjoint"] = False
            elif len(support_sorted) == 2:
                g["support_badge"] = "+".join(support_sorted)
                g["is_consensus"] = False
                g["is_disjoint"] = False
            else:
                g["support_badge"] = f"{my_opt} only (Disjoint)"
                g["is_consensus"] = False
                g["is_disjoint"] = True
