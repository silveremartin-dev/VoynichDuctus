"""
Unified Unsupervised Word-to-Glyph Decomposition Engine (`word_decomposition`).
Implements `WordScribalEncoder` (1D reading-axis geometric trajectory Z(x)) and
`GenerativeTypewriterDecomposer` (loop-stratified visual sprite dictionary + Viterbi MDL
decoding, calibrated strictly in physical millimeters via `MetricScaleCalibration`).
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from sklearn.decomposition import PCA


class WordScribalEncoder:
    """
    Encodes 2D word patches into deterministic 1D horizontal latent trajectories Z(x) in R^D
    along reading axis X (ink mass, vertical moments, Euler run transitions, 8-band profile,
    and local reading-axis derivatives).
    """

    def __init__(self, target_height: int = 48, latent_dim: int = 32):
        self.target_height = target_height
        self.latent_dim = latent_dim

    def encode_word_patch(self, word_img: np.ndarray) -> np.ndarray:
        if word_img is None or word_img.size == 0:
            return np.zeros((1, self.latent_dim), dtype=np.float32)

        if word_img.ndim == 3:
            gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
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
                active = col > 0.18
                ink_rows = np.where(active)[0]
                top_y = float(ink_rows[0]) if len(ink_rows) > 0 else 0.0
                bot_y = float(ink_rows[-1]) if len(ink_rows) > 0 else h_f
                span_y = max(0.0, bot_y - top_y)
                runs = int(np.sum(active[1:] & (~active[:-1]))) + (1 if active[0] else 0)
                sector_means = [float(np.mean(chunk)) for chunk in np.array_split(col, 8)]

                desc[0] = total_ink / h_f
                desc[1] = cy / h_f
                desc[2] = np.sqrt(max(0.0, var_y)) / h_f
                desc[3] = top_y / h_f
                desc[4] = bot_y / h_f
                desc[5] = span_y / h_f
                desc[6] = min(1.0, runs / 3.0)
                desc[7] = float(np.mean(col[: self.target_height // 3]))
                desc[8] = float(np.mean(col[self.target_height // 3 : 2 * self.target_height // 3]))
                desc[9] = float(np.mean(col[2 * self.target_height // 3 :]))
                for k, val in enumerate(sector_means):
                    desc[10 + k] = val
            raw_cols.append(desc)

        arr = np.array(raw_cols, dtype=np.float32)
        if target_w >= 3:
            grad1 = np.zeros_like(arr[:, :7])
            grad2 = np.zeros_like(arr[:, :7])
            grad1[1:-1] = 0.5 * (arr[2:, :7] - arr[:-2, :7])
            if target_w >= 5:
                grad2[2:-2] = 0.5 * (arr[4:, 10:17] - arr[:-4, 10:17])
            arr[:, 18:25] = grad1
            arr[:, 25:32] = grad2
        return arr


class _LatentTransitionProposer:
    """
    Internal reading-axis latent transition detector (`||dZ/dx||`) calibrated in millimeters (`window_mm`).
    Supplies candidate velocity-peak cut columns to `GenerativeTypewriterDecomposer._build_word_lattice`.
    """

    def __init__(self, n_components: int = 36, window_mm: float = 1.75):
        self.n_components = n_components
        self.window_mm = float(window_mm)
        self.encoder = WordScribalEncoder(latent_dim=32)
        self.pca = PCA(n_components=min(n_components, 24))
        self._win_cols = 14
        self.is_fitted = False

    def fit_on_word_corpus(self, word_patches: List[np.ndarray], px_per_mm: float = 8.0) -> None:
        self._win_cols = max(6, int(round(self.window_mm * px_per_mm)))
        sliding_samples = []
        for patch in word_patches:
            traj = self.encoder.encode_word_patch(patch)
            if len(traj) < self._win_cols:
                continue
            for x in range(0, len(traj) - self._win_cols + 1, max(1, self._win_cols // 3)):
                sliding_samples.append(traj[x : x + self._win_cols].flatten())

        if len(sliding_samples) >= self.n_components:
            X = np.array(sliding_samples, dtype=np.float32)
            n_c = min(self.n_components, X.shape[0], X.shape[1])
            self.pca = PCA(n_components=n_c)
            self.pca.fit(X)
            self.is_fitted = True

    @staticmethod
    def build_tight_slice_masks(
        word_mask: np.ndarray,
        cuts: List[int],
        scale_calib: Optional[Any] = None,
    ) -> List[np.ndarray]:
        """
        Builds per-slice boolean masks from column cuts while reassigning clipped
        overhang tips of adjacent tall letters back to their owning slice in millimeters.
        """
        from scipy.ndimage import label

        H, W = word_mask.shape[:2]
        ppm = float(scale_calib.px_per_mm) if scale_calib is not None else max(3.5, float(H) / 4.8)
        v_gap_thresh = 0.65 * ppm
        stray_area = max(2, int(round(0.25 * (ppm ** 2))))

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

            main_cid, main_area, main_r0, main_r1, _, _ = max(comps, key=lambda x: x[1])
            for cid, area, r0, r1, _, _ in comps:
                if cid == main_cid:
                    continue
                v_gap = max(0, max(main_r0 - r1, r0 - main_r1))
                if v_gap < v_gap_thresh or area >= 0.45 * main_area:
                    continue

                frag_mask = lbl_s == cid
                parent_labels = lbl_word[frag_mask]
                parent_labels = parent_labels[parent_labels > 0]
                reassigned = False
                if len(parent_labels) > 0:
                    p_id = int(np.bincount(parent_labels).argmax())
                    parent_cc = lbl_word == p_id
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
                        slice_masks[s_idx] &= ~frag_mask
                        reassigned = True

                if not reassigned and area <= stray_area and (r0 <= 1 or r1 >= H - 1) and v_gap >= v_gap_thresh * 1.2:
                    slice_masks[s_idx] &= ~frag_mask

        return slice_masks

    def propose_cuts(self, work_mask: np.ndarray, ppm: float, forbidden_cols: set) -> List[int]:
        from scipy.signal import find_peaks

        H, W = work_mask.shape[:2]
        min_pitch = max(2, int(round(1.45 * ppm)))
        min_margin = max(2, int(round(1.10 * ppm)))
        snap_r = max(1, int(round(0.55 * ppm)))
        if W <= 2 * min_margin:
            return []

        traj = self.encoder.encode_word_patch(work_mask.astype(np.float32))
        W_eff = len(traj)
        scale_x = float(W) / float(max(1, W_eff))

        diffs = np.zeros(W_eff, dtype=np.float32)
        if self.is_fitted and W_eff >= self._win_cols + 2:
            half_win = self._win_cols // 2
            padded = np.pad(traj, ((half_win, self._win_cols - half_win - 1), (0, 0)), mode="edge")
            win_list = [padded[i : i + self._win_cols].flatten() for i in range(W_eff)]
            latent_seq = self.pca.transform(np.array(win_list, dtype=np.float32))
            for t in range(1, W_eff - 1):
                diffs[t] = float(np.linalg.norm(latent_seq[t + 1] - latent_seq[t - 1]))
        else:
            for t in range(1, W_eff - 1):
                diffs[t] = float(np.linalg.norm(traj[t + 1] - traj[t - 1]))

        min_dist_eff = max(2, int(round(min_pitch / max(1e-4, scale_x))))
        peaks, _ = find_peaks(diffs, distance=min_dist_eff, prominence=float(np.mean(diffs) * 0.45))

        col_proj = np.sum(work_mask, axis=0).astype(np.float32)
        snapped_cols = []
        for p in peaks:
            c = int(round(p * scale_x))
            lo, hi = max(min_margin, c - snap_r), min(W - min_margin, c + snap_r + 1)
            if lo < hi:
                snapped_cols.append(lo + int(np.argmin(col_proj[lo:hi])))

        zero_cols = col_proj == 0
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
                if min_margin <= mid <= W - min_margin:
                    gap_centers.append(mid)

        candidate_cols: List[int] = []
        for c in sorted(set(gap_centers + snapped_cols)):
            if c not in forbidden_cols and all(abs(c - kept) >= min_pitch for kept in candidate_cols):
                candidate_cols.append(c)
        return sorted(candidate_cols)


class GenerativeTypewriterDecomposer:
    """
    Unsupervised Word-to-Glyph Decomposer (clean-room implementation inspired by
    'The Learnable Typewriter' [Siglidis et al., ICDAR] + Viterbi Minimum Description Length):
    1. Builds a disjoint candidate cut lattice combining topological valleys, supralinear
       diacritic decoupling (<= 0.55 mm), and 1D latent velocity peaks `||dZ/dx||` while
       strictly protecting closed Euler loops.
    2. Induces a loop-stratified (`0L` / `1L`) bank of normalized visual character sprites
       with empirical MDL code lengths (`-log2 p_k`) and physical millimeter width priors.
    3. Solves exact Viterbi dynamic programming across each word.
    """

    def __init__(self, sprite_size: int = 24, max_prototypes: int = 32, is_serafini: bool = False):
        self.sprite_size = sprite_size
        self.max_prototypes = max_prototypes
        self.is_serafini = is_serafini
        self.prototypes: List[Dict[str, Any]] = []
        self._latent_proposer = _LatentTransitionProposer()
        self.is_fitted: bool = False

    @staticmethod
    def _mask_to_sprite(mask: np.ndarray, dim: int = 24) -> np.ndarray:
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
        out[yo : yo + nh, xo : xo + nw] = resized
        return out

    @staticmethod
    def _count_mask_holes_and_spans(
        mask: np.ndarray,
        px_per_mm: float,
    ) -> Tuple[int, List[Tuple[int, int, float]]]:
        from scipy.ndimage import binary_closing, binary_fill_holes, label

        H, W = mask.shape[:2]
        min_area = max(2, int(round(0.065 * (px_per_mm ** 2))))
        min_h = max(2, int(round(0.34 * px_per_mm)))
        min_w = max(2, int(round(0.28 * px_per_mm)))

        padded = np.pad(mask.astype(bool), 2, mode="constant", constant_values=False)
        holes = (binary_fill_holes(padded) & (~padded))[2:-2, 2:-2]
        lbl_h, num_h = label(holes)
        spans: List[Tuple[int, int, float]] = []
        for hid in range(1, num_h + 1):
            hm = lbl_h == hid
            if int(np.sum(hm)) < min_area:
                continue
            ys, xs = np.where(hm)
            if (ys.max() - ys.min() + 1) >= min_h and (xs.max() - xs.min() + 1) >= min_w:
                spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))

        if not spans and H >= int(round(1.0 * px_per_mm)) and W >= int(round(1.0 * px_per_mm)):
            closed_p = binary_closing(padded, structure=np.ones((2, 2), dtype=bool))
            holes_c = (binary_fill_holes(closed_p) & (~closed_p))[2:-2, 2:-2]
            lbl_c, num_c = label(holes_c)
            min_area_c = max(3, int(round(0.11 * (px_per_mm ** 2))))
            for hid in range(1, num_c + 1):
                hm = lbl_c == hid
                if int(np.sum(hm)) >= min_area_c:
                    ys, xs = np.where(hm)
                    if (ys.max() - ys.min() + 1) >= min_h and (xs.max() - xs.min() + 1) >= min_w:
                        spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))

        spans.sort(key=lambda s: s[0])
        return len(spans), spans

    def _build_word_lattice(
        self,
        word_mask: np.ndarray,
        scale_calib: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, float]:
        from scipy.ndimage import binary_fill_holes, label
        from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter

        w_mask = (word_mask > 0).astype(bool)
        H, W = w_mask.shape[:2]
        ppm = float(scale_calib.px_per_mm) if scale_calib is not None else max(4.5, float(H) / 4.5)
        if int(np.sum(w_mask)) == 0:
            return w_mask, [], np.zeros(W, dtype=float), ppm

        min_clean_dim = int(round(2.0 * ppm))
        min_clean_area = int(round(0.30 * (ppm ** 2)))
        work = (
            GlyphSegmenter._clean_word_mask_border_bleed(w_mask, scale_calib=scale_calib)
            if (H >= min_clean_dim and W >= min_clean_dim and int(np.sum(w_mask)) > min_clean_area)
            else w_mask
        )

        padded = np.pad(work, 2, mode="constant", constant_values=False)
        holes = (binary_fill_holes(padded) & (~padded))[2:-2, 2:-2]
        lbl_h, num_h = label(holes)
        forbidden = set()
        min_ha = max(2, int(round(0.08 * ppm * ppm)))
        min_hh = max(2, int(round(0.35 * ppm)))
        min_hw = max(2, int(round(0.28 * ppm)))
        loop_spans: List[Tuple[int, int, float]] = []
        for hid in range(1, num_h + 1):
            hm = lbl_h == hid
            if int(np.sum(hm)) >= min_ha:
                ys, xs = np.where(hm)
                if (ys.max() - ys.min() + 1) >= min_hh and (xs.max() - xs.min() + 1) >= min_hw:
                    loop_spans.append((int(xs.min()), int(xs.max()), float(np.mean(ys)) / max(1.0, float(H))))
                    for fc in range(max(0, int(xs.min()) - 1), min(W, int(xs.max()) + 2)):
                        forbidden.add(fc)
        loop_spans.sort()

        seg = GlyphSegmenter(scale_calib=scale_calib)
        a_comps = seg.assemble_word_into_glyphs(work, scale_calib=scale_calib)
        b_cuts = self._latent_proposer.propose_cuts(work, ppm=ppm, forbidden_cols=forbidden)
        raw_cuts = [int(g["bbox_local"][3]) for g in a_comps[:-1]] + b_cuts

        col_proj = np.sum(work, axis=0).astype(float)
        for li in range(len(loop_spans) - 1):
            lx0_a, lx1_a, ly_a = loop_spans[li]
            lx0_b, lx1_b, ly_b = loop_spans[li + 1]
            if not (ly_a < 0.38 and ly_b < 0.38) and lx0_b > lx1_a + 1:
                cands = [x for x in range(lx1_a + 1, lx0_b) if x not in forbidden]
                if cands:
                    raw_cuts.append(min(cands, key=lambda x: col_proj[x]))

        min_margin = max(2, int(round(0.95 * ppm)))
        min_atom = max(2, int(round(0.80 * ppm)))
        valid_c = sorted(set(c for c in raw_cuts if min_margin <= c <= W - min_margin and c not in forbidden))

        snap_r = max(1, int(round(0.30 * ppm)))
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
        slices = _LatentTransitionProposer.build_tight_slice_masks(work, pts, scale_calib=scale_calib)
        return work, slices, col_proj, ppm

    def fit_on_word_masks(
        self,
        word_masks: List[np.ndarray],
        scale_calib: Optional[Any] = None,
        word_patches: Optional[List[np.ndarray]] = None,
    ) -> None:
        from sklearn.cluster import AgglomerativeClustering
        from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

        calib = scale_calib or MetricScaleCalibration(manuscript_type="seraphinianus" if self.is_serafini else "voynich")
        ppm_fit = float(calib.px_per_mm)
        if word_patches:
            self._latent_proposer.fit_on_word_corpus(word_patches, px_per_mm=ppm_fit)
        elif word_masks:
            self._latent_proposer.fit_on_word_corpus(
                [wm.astype(np.float32) for wm in word_masks[:35] if wm is not None and wm.size > 16],
                px_per_mm=ppm_fit,
            )

        min_area = max(1, int(round(calib.min_glyph_area_px)))
        proto_cands: List[Tuple[np.ndarray, int, float, float]] = []

        for wm in word_masks:
            if wm is None or wm.size < 16 or not np.any(wm):
                continue
            _, slices, _, ppm = self._build_word_lattice(wm, scale_calib=calib)
            for s_m in slices:
                if int(np.sum(s_m)) < min_area:
                    continue
                ys, xs = np.where(s_m)
                cm = s_m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
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
                grp = grp[:: max(1, len(grp) // 350)]
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
        from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration

        calib = scale_calib or MetricScaleCalibration(manuscript_type="seraphinianus" if self.is_serafini else "voynich")
        min_area = max(1, int(round(calib.min_glyph_area_px)))
        work, slices, col_proj, ppm = self._build_word_lattice(word_mask, scale_calib=calib)
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

        dot_dim_px = max(1, int(round(calib.DOT_MAX_DIAMETER_MM * ppm)))
        dot_area_px = max(1, int(round(calib.DOT_MAX_AREA_MM2 * (ppm ** 2))))
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
            cm = s_m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            ln, _ = self._count_mask_holes_and_spans(cm, ppm)
            s_info.append((int(xs.min()), int(xs.max()), cm.shape[1] / ppm, cm.shape[0] / ppm, ln))

        bounds_ink: List[Tuple[float, float]] = []
        half_band = max(1, int(round(0.20 * ppm)))
        for k in range(M - 1):
            c_mid = int(round(0.5 * (s_info[k][1] + s_info[k + 1][0])))
            lo_b, hi_b = max(0, c_mid - half_band), min(work.shape[1], c_mid + half_band + 1)
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
                    if s_info[i][4] >= 1 or s_info[i][2] > 1.85:
                        continue
                    if sum(1 for r in range(i, j) if s_info[r][4] >= 1) >= 2:
                        continue
                    if max(bounds_ink[r][1] for r in range(i, j - 1)) > 0.42:
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
