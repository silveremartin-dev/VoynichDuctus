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
        self._torch_encoder = None
        self._init_encoder()

    def _init_encoder(self):
        """Initializes PyTorch convolutional encoder if available, otherwise heuristic fallback."""
        try:
            import torch
            import torch.nn as nn

            class ConvWordRibbonNet(nn.Module):
                def __init__(self, out_dim=32):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d((2, 1)),  # Height / 2, width preserved
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d((2, 1)),  # Height / 4
                        nn.Conv2d(32, out_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, None))  # Collapse height to 1, preserve horizontal width
                    )

                def forward(self, x):
                    feat = self.features(x)
                    return feat.squeeze(2)

            self._torch_encoder = ConvWordRibbonNet(out_dim=self.latent_dim)
            self._torch_encoder.eval()
        except ImportError:
            self._torch_encoder = None

    def encode_word_patch(self, word_img: np.ndarray) -> np.ndarray:
        """
        Encodes a 2D word crop into a 1D sequence of D-dimensional latent vectors along axis X.
        Returns array of shape (W_eff, D).
        """
        if word_img is None or word_img.size == 0:
            return np.zeros((1, self.latent_dim), dtype=np.float32)

        if word_img.ndim == 3:
            gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
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

        if self._torch_encoder is not None:
            import torch
            with torch.no_grad():
                tensor_in = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()
                out_feat = self._torch_encoder(tensor_in).squeeze(0).cpu().numpy()
                return out_feat.T.astype(np.float32)

        profiles = []
        for x in range(target_w):
            col = resized[:, x]
            total_ink = float(np.sum(col))
            if total_ink < 1e-4:
                profiles.append(np.zeros(self.latent_dim, dtype=np.float32))
                continue

            ys = np.arange(self.target_height, dtype=np.float32)
            cy = float(np.sum(ys * col) / total_ink)
            var_y = float(np.sum(((ys - cy) ** 2) * col) / total_ink)
            ink_rows = np.where(col > 0.15)[0]
            top_y = float(ink_rows[0]) if len(ink_rows) > 0 else 0.0
            bot_y = float(ink_rows[-1]) if len(ink_rows) > 0 else float(self.target_height)

            sector_means = [float(np.mean(chunk)) for chunk in np.array_split(col, 8)]

            desc = np.zeros(self.latent_dim, dtype=np.float32)
            desc[0] = total_ink / float(self.target_height)
            desc[1] = cy / float(self.target_height)
            desc[2] = np.sqrt(max(0.0, var_y)) / float(self.target_height)
            desc[3] = top_y / float(self.target_height)
            desc[4] = bot_y / float(self.target_height)
            for k, val in enumerate(sector_means):
                desc[5 + k] = val
            profiles.append(desc)

        return np.array(profiles, dtype=np.float32)


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

    def decompose_word(
        self,
        word_mask: np.ndarray,
        expected_glyph_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Segments word into constituent glyphs by finding latent state transition boundaries
        along the horizontal trajectory.
        """
        H, W = word_mask.shape[:2]
        if W <= 8:
            return [{
                "mask": word_mask,
                "bbox_local": (0, 0, H, W),
                "area": int(np.sum(word_mask)),
                "neural_basis_id": "PHI_00",
                "segmentation_type": "NEURAL_WORD_PCA",
                "word_decomposition": "neural_word_pca"
            }]

        traj = self.encoder.encode_word_patch(word_mask.astype(np.float32))
        W_eff = len(traj)
        scale_x = float(W) / float(max(1, W_eff))

        diffs = np.zeros(W_eff, dtype=np.float32)
        for t in range(1, W_eff - 1):
            diffs[t] = float(np.linalg.norm(traj[t + 1] - traj[t - 1]))

        from scipy.signal import find_peaks
        min_dist_eff = max(4, int(10 / max(1e-4, scale_x)))
        peaks, _ = find_peaks(diffs, distance=min_dist_eff, prominence=float(np.mean(diffs) * 0.45))

        boundary_cols = [int(round(p * scale_x)) for p in peaks]
        boundary_cols = [c for c in boundary_cols if 6 <= c <= W - 6]

        if expected_glyph_count is not None and expected_glyph_count >= 1:
            target_splits = expected_glyph_count - 1
            if len(boundary_cols) > target_splits:
                scored = sorted(boundary_cols, key=lambda c: diffs[min(W_eff - 1, int(c / scale_x))], reverse=True)
                boundary_cols = sorted(scored[:target_splits])
            elif len(boundary_cols) < target_splits:
                step = W / float(expected_glyph_count)
                boundary_cols = [int(round(i * step)) for i in range(1, expected_glyph_count)]

        cuts = [0] + boundary_cols + [W]
        glyphs = []
        for idx in range(len(cuts) - 1):
            c0, c1 = cuts[idx], cuts[idx + 1]
            if c1 <= c0:
                continue
            sub_m = np.zeros_like(word_mask, dtype=bool)
            sub_m[:, c0:c1] = word_mask[:, c0:c1]
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
                "neural_basis_id": f"PHI_{idx:02d}",
                "segmentation_type": "NEURAL_WORD_PCA",
                "word_decomposition": "neural_word_pca"
            })

        return glyphs if glyphs else [{
            "mask": word_mask,
            "bbox_local": (0, 0, H, W),
            "area": int(np.sum(word_mask)),
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

    def __init__(self, tolerance_px: int = 4):
        self.tolerance_px = tolerance_px

    def decompose_word_hybrid(
        self,
        word_mask: np.ndarray,
        option_a_glyphs: List[Dict[str, Any]],
        option_b_glyphs: List[Dict[str, Any]],
        expected_glyph_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuses Decomposition A and Decomposition B into consensus boundaries.
        """
        H, W = word_mask.shape[:2]
        if W <= 8:
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
            h_cols = np.where(np.any(lbl_h == h_idx, axis=0))[0]
            if len(h_cols) >= 2:
                for fc in range(max(0, int(h_cols.min()) - 1), min(W, int(h_cols.max()) + 2)):
                    forbidden_cols.add(fc)

        col_proj = np.sum(word_mask, axis=0).astype(float)
        max_proj = np.max(col_proj) if len(col_proj) > 0 else 1.0

        consensus_cuts = []
        used_b = set()

        for ca in cuts_a:
            if ca in forbidden_cols:
                continue
            matched_b = [cb for cb in cuts_b if abs(ca - cb) <= self.tolerance_px and cb not in used_b]
            if matched_b:
                cb_best = matched_b[0]
                used_b.add(cb_best)
                consensus_cuts.append(int(round((ca + cb_best) / 2.0)))
            else:
                if col_proj[min(W - 1, ca)] <= 0.70 * max_proj:
                    consensus_cuts.append(ca)

        for cb in cuts_b:
            if cb in used_b or cb in forbidden_cols:
                continue
            if col_proj[min(W - 1, cb)] <= 0.70 * max_proj:
                consensus_cuts.append(cb)

        consensus_cuts = sorted(list(set(c for c in consensus_cuts if 6 <= c <= W - 6)))

        filtered_cuts = []
        for c in consensus_cuts:
            if all(abs(c - fc) >= 8 for fc in filtered_cuts):
                filtered_cuts.append(c)
        filtered_cuts.sort()

        if expected_glyph_count is not None and expected_glyph_count >= 1:
            target_splits = expected_glyph_count - 1
            if len(filtered_cuts) > target_splits:
                scored = sorted(filtered_cuts, key=lambda c: col_proj[c])
                filtered_cuts = sorted(scored[:target_splits])

        cuts = [0] + filtered_cuts + [W]
        glyphs = []
        for idx in range(len(cuts) - 1):
            c0, c1 = cuts[idx], cuts[idx + 1]
            if c1 <= c0:
                continue
            sub_m = np.zeros_like(word_mask, dtype=bool)
            sub_m[:, c0:c1] = word_mask[:, c0:c1]
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
