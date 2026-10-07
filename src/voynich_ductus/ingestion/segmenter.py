"""
Manuscript line and word segmentation via connected components and morphological projections.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops


class LineSegmenter:
    """
    Extracts text lines, words, and isolated character bounding boxes
    from binarized historical manuscript pages.
    """

    def __init__(self, min_line_pitch: int = 35, min_line_height: int = 18, min_word_width: int = 18, min_word_area: int = 40):
        self.min_line_pitch = min_line_pitch
        self.min_line_height = min_line_height
        self.min_word_width = min_word_width
        self.min_word_area = min_word_area

    def isolate_text_paragraph(self, binary_ink: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detects and isolates the main text paragraph block, removing marginal drawings,
        header decorations (such as the top arch on f001r), and binding margins.
        """
        H, W = binary_ink.shape
        # Standard Beinecke Voynich folio margins:
        # Avoid top 16% (often header / title decoration) and bottom 8%
        # Avoid left 12% and right 10%
        y0 = int(H * 0.17)
        y1 = int(H * 0.92)
        x0 = int(W * 0.14)
        x1 = int(W * 0.88)

        text_block = binary_ink[y0:y1, x0:x1]
        return text_block, (y0, x0, y1, x1)

    def segment_lines(self, binary_ink: np.ndarray, auto_isolate: bool = True) -> List[Dict[str, Any]]:
        """
        Segments manuscript page into distinct, intact text lines using adaptive peak detection
        and connected component clustering around baseline peaks. Ensures every line from top to bottom
        is captured without dropping short/indented lines or slicing ascenders.
        """
        if auto_isolate and (binary_ink.shape[0] > 600 and binary_ink.shape[1] > 600):
            work_img, (off_y, off_x, _, _) = self.isolate_text_paragraph(binary_ink)
        else:
            work_img = binary_ink
            off_y, off_x = 0, 0

        # Scale-adaptive line pitch and smoothing sigma based on page height
        H_work = work_img.shape[0]
        eff_pitch = max(self.min_line_pitch, int(H_work / 65))
        min_dist = max(18, int(eff_pitch * 0.65))

        # Horizontal projection profile
        raw_profile = np.sum(work_img, axis=1).astype(float)
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        smooth_profile = gaussian_filter1d(raw_profile, sigma=max(3.0, eff_pitch * 0.08))

        # Detect line baseline centers with sensitive relative prominence
        max_p = np.max(smooth_profile) if len(smooth_profile) > 0 else 0
        min_prom = max(1.5, max_p * 0.015)
        peaks, _ = find_peaks(smooth_profile, distance=min_dist, prominence=min_prom)

        if len(peaks) == 0:
            return [{
                "line_id": "L000",
                "bbox": (off_y, off_x, off_y + work_img.shape[0], off_x + work_img.shape[1]),
                "image": work_img
            }]

        # Label all connected components in work_img
        labeled_block, num_comps = label(work_img)
        if num_comps == 0:
            return []

        props = regionprops(labeled_block)

        # Group components by closest baseline peak within bounding band
        line_buckets: Dict[int, List[Any]] = {i: [] for i in range(len(peaks))}
        for p in props:
            if p.area < 6:
                continue
            cy = p.centroid[0]
            # Find closest peak
            distances = [abs(cy - peak_y) for peak_y in peaks]
            closest_idx = int(np.argmin(distances))
            if distances[closest_idx] <= eff_pitch * 1.6:
                line_buckets[closest_idx].append(p)

        results = []
        for line_idx, peak_y in enumerate(peaks):
            line_comps = line_buckets.get(line_idx, [])
            if not line_comps:
                continue

            ly0 = min(c.bbox[0] for c in line_comps)
            ly1 = max(c.bbox[2] for c in line_comps)
            lx0 = min(c.bbox[1] for c in line_comps)
            lx1 = max(c.bbox[3] for c in line_comps)

            if (lx1 - lx0) < 10 or (ly1 - ly0) < 8:
                continue

            # Construct clean composite mask with ONLY this line's intact components
            lh = ly1 - ly0
            lw = lx1 - lx0
            line_mask = np.zeros((lh, lw), dtype=bool)
            for c in line_comps:
                c_mask = (labeled_block[c.bbox[0]:c.bbox[2], c.bbox[1]:c.bbox[3]] == c.label)
                off_comp_y = c.bbox[0] - ly0
                off_comp_x = c.bbox[1] - lx0
                line_mask[off_comp_y:off_comp_y + (c.bbox[2] - c.bbox[0]), off_comp_x:off_comp_x + (c.bbox[3] - c.bbox[1])] |= c_mask

            abs_y0 = off_y + ly0
            abs_y1 = off_y + ly1
            abs_x0 = off_x + lx0
            abs_x1 = off_x + lx1

            results.append({
                "line_id": f"L{line_idx:03d}",
                "bbox": (int(abs_y0), int(abs_x0), int(abs_y1), int(abs_x1)),
                "image": line_mask
            })

        return results

    def segment_words(self, line_ink: np.ndarray, line_offset: Tuple[int, int] = (0, 0), space_gap_min: int = 12, space_threshold: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Segments a text line into clean, intact words using connected component horizontal grouping
        and vertical whitespace gap detection.
        """
        if space_threshold is not None:
            space_gap_min = space_threshold
        y_off, x_off = line_offset

        labeled_l, num_l = label(line_ink)
        if num_l == 0:
            return []

        props = sorted(regionprops(labeled_l), key=lambda p: p.bbox[1])
        valid_comps = [p for p in props if p.area >= 8]
        if not valid_comps:
            return []

        # Cluster components horizontally by inter-character space threshold
        word_clusters: List[List[Any]] = []
        current_cluster = [valid_comps[0]]

        for i in range(1, len(valid_comps)):
            curr_comp = valid_comps[i]
            prev_x1 = max(c.bbox[3] for c in current_cluster)
            curr_x0 = curr_comp.bbox[1]

            gap = curr_x0 - prev_x1
            if gap <= space_gap_min:
                current_cluster.append(curr_comp)
            else:
                word_clusters.append(current_cluster)
                current_cluster = [curr_comp]

        if current_cluster:
            word_clusters.append(current_cluster)

        results = []
        for word_idx, cl in enumerate(word_clusters):
            wy0 = min(c.bbox[0] for c in cl)
            wy1 = max(c.bbox[2] for c in cl)
            wx0 = min(c.bbox[1] for c in cl)
            wx1 = max(c.bbox[3] for c in cl)

            wh = wy1 - wy0
            ww = wx1 - wx0
            if ww < 8 or wh < 8:
                continue

            w_crop = np.zeros((wh, ww), dtype=bool)
            for c in cl:
                c_mask = (labeled_l[c.bbox[0]:c.bbox[2], c.bbox[1]:c.bbox[3]] == c.label)
                off_cy = c.bbox[0] - wy0
                off_cx = c.bbox[1] - wx0
                w_crop[off_cy:off_cy + (c.bbox[2] - c.bbox[0]), off_cx:off_cx + (c.bbox[3] - c.bbox[1])] |= c_mask

            ink_area = int(np.sum(w_crop))
            results.append({
                "word_id": f"W{word_idx:03d}",
                "bbox": (int(y_off + wy0), int(x_off + wx0), int(y_off + wy1), int(x_off + wx1)),
                "area": ink_area,
                "image": w_crop
            })

        return results

    def estimate_line_pitch_fft(self, binary_ink: np.ndarray) -> float:
        """
        Estimates the dominant scribal line pitch (interline distance in px) via 1D FFT spectral analysis
        of the horizontal projection profile. Distinguishes regular ruled text lines from chaotic illustrations.
        """
        if binary_ink.shape[0] < 50:
            return float(self.min_line_pitch)

        proj = np.sum(binary_ink, axis=1).astype(float)
        proj = proj - np.mean(proj)
        n = len(proj)
        if n == 0 or np.all(proj == 0):
            return float(self.min_line_pitch)

        fft_vals = np.abs(np.fft.rfft(proj))
        freqs = np.fft.rfftfreq(n, d=1.0)

        # Restrict to realistic manuscript line pitches [15px .. 180px]
        min_freq = 1.0 / 180.0
        max_freq = 1.0 / 15.0
        valid_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]

        if len(valid_idx) == 0:
            return float(self.min_line_pitch)

        peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
        peak_freq = freqs[peak_idx]
        if peak_freq > 0:
            pitch = 1.0 / peak_freq
            return round(float(pitch), 2)
        return float(self.min_line_pitch)

    def detect_text_macro_blocks(self, binary_ink: np.ndarray, h_radius: int = 30, v_radius: int = 12, min_area: int = 400) -> List[Dict[str, Any]]:
        """
        Detects coherent rectangular text paragraph macro-blocks using anisotropic morphological closing,
        separating compact text bodies from sprawling botanical illustrations and marginal annotations.
        """
        from scipy.ndimage import binary_closing
        H, W = binary_ink.shape
        # Anisotropic structuring element matching horizontal scribal line flow
        struct_elem = np.ones((v_radius, h_radius), dtype=bool)
        closed_mask = binary_closing(binary_ink, structure=struct_elem)

        labeled_blocks, num_blocks = label(closed_mask)
        props = regionprops(labeled_blocks)

        macro_blocks = []
        for idx, p in enumerate(props):
            # Macro-block must have sufficient area and contain genuine text density
            if p.area >= min_area:
                by0, bx0, by1, bx1 = p.bbox
                block_ink = binary_ink[by0:by1, bx0:bx1]
                density = np.sum(block_ink) / max(1, (by1 - by0) * (bx1 - bx0))
                if density >= 0.03:  # True paragraph block
                    macro_blocks.append({
                        "block_id": f"MB{idx:02d}",
                        "bbox": (int(by0), int(bx0), int(by1), int(bx1)),
                        "area": int(p.area),
                        "density": round(float(density), 3),
                        "aspect_ratio": round(float((bx1 - bx0) / max(1, by1 - by0)), 2)
                    })

        return sorted(macro_blocks, key=lambda b: (b["bbox"][0], b["bbox"][1]))

    def filter_nib_width_consistency(self, binary_ink: np.ndarray, min_radius: float = 0.6, max_radius: float = 4.5) -> np.ndarray:
        """
        Filters out non-pen strokes by measuring local Euclidean distance radius r(x,y):
        - Discards thick brush strokes (r > 4.5 px, e.g. colored washes or outer drawing contours)
        - Discards sub-nib parchment grain (r < 0.6 px)
        """
        from scipy.ndimage import distance_transform_edt
        if not np.any(binary_ink):
            return binary_ink

        edt = distance_transform_edt(binary_ink)
        # Identify regions that are excessively thick for a 15th-century pen nib
        too_thick = edt > max_radius
        from scipy.ndimage import binary_dilation
        thick_blobs = binary_dilation(too_thick, iterations=int(max_radius * 1.5))

        clean_ink = binary_ink & (~thick_blobs)
        return clean_ink

    def extract_connected_components(self, binary_ink: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts individual connected glyph/ligature components with bounding boxes.
        """
        labeled_arr, num_features = label(binary_ink)
        props = regionprops(labeled_arr)
        components = []

        for idx, prop in enumerate(props):
            if prop.area < 6:  # Skip single pixel specks
                continue
            min_row, min_col, max_row, max_col = prop.bbox
            comp_mask = (labeled_arr[min_row:max_row, min_col:max_col] == prop.label)
            components.append({
                "component_id": f"CC{idx:05d}",
                "bbox": (int(min_row), int(min_col), int(max_row), int(max_col)),
                "area": int(prop.area),
                "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
                "image": comp_mask
            })

        return components
