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
        Segments manuscript page into distinct, intact text lines using connected component
        clustering around baseline peaks. Ensures ascenders and descenders are NEVER sliced horizontally.
        """
        if auto_isolate and (binary_ink.shape[0] > 600 and binary_ink.shape[1] > 600):
            work_img, (off_y, off_x, _, _) = self.isolate_text_paragraph(binary_ink)
        else:
            work_img = binary_ink
            off_y, off_x = 0, 0

        # Horizontal projection profile
        raw_profile = np.sum(work_img, axis=1).astype(float)
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        smooth_profile = gaussian_filter1d(raw_profile, sigma=4.0)

        # Detect line baseline centers (peaks)
        peaks, _ = find_peaks(smooth_profile, distance=self.min_line_pitch, prominence=np.max(smooth_profile) * 0.10)

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

        # Group components by closest baseline peak
        line_buckets: Dict[int, List[Any]] = {i: [] for i in range(len(peaks))}
        for p in props:
            # Filter micro artifacts
            if p.area < 6:
                continue
            cy = p.centroid[0]
            # Find closest peak
            distances = [abs(cy - peak_y) for peak_y in peaks]
            closest_idx = int(np.argmin(distances))
            # If component is within reasonable distance from peak (< 1.6 * pitch)
            if distances[closest_idx] <= self.min_line_pitch * 1.8:
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

            if (lx1 - lx0) < self.min_word_width or (ly1 - ly0) < 8:
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

    def segment_words(self, line_ink: np.ndarray, line_offset: Tuple[int, int] = (0, 0), space_gap_min: int = 14, space_threshold: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Segments a text line into clean, intact words using smoothed vertical projection gaps
        and morphological area filtering to reject parchment grain specks.
        """
        if space_threshold is not None:
            space_gap_min = space_threshold
        y_off, x_off = line_offset
        from scipy.ndimage import gaussian_filter1d
        
        col_counts = np.sum(line_ink, axis=0).astype(float)
        smooth_cols = gaussian_filter1d(col_counts, sigma=2.0)

        in_word = False
        start_x = 0
        word_spans = []
        gap = 0

        for x, val in enumerate(smooth_cols):
            if val > 1.8:
                if not in_word:
                    in_word = True
                    start_x = x
                gap = 0
            else:
                if in_word:
                    gap += 1
                    if gap >= space_gap_min:
                        in_word = False
                        end_x = x - gap + 1
                        if end_x - start_x >= self.min_word_width:
                            word_spans.append((start_x, end_x))

        if in_word and line_ink.shape[1] - start_x >= self.min_word_width:
            word_spans.append((start_x, line_ink.shape[1]))

        results = []
        for word_idx, (wx0, wx1) in enumerate(word_spans):
            word_crop = line_ink[:, wx0:wx1]
            # Area and dimension verification
            ink_area = int(np.sum(word_crop))
            if ink_area < self.min_word_area:
                continue

            row_profile = np.sum(word_crop, axis=1)
            nonzero_rows = np.where(row_profile > 0)[0]
            if len(nonzero_rows) == 0:
                continue
            wy0, wy1 = nonzero_rows[0], nonzero_rows[-1] + 1
            if wy1 - wy0 < 12:
                continue

            results.append({
                "word_id": f"W{word_idx:03d}",
                "bbox": (int(y_off + wy0), int(x_off + wx0), int(y_off + wy1), int(x_off + wx1)),
                "area": ink_area,
                "image": line_ink[wy0:wy1, wx0:wx1]
            })

        return results

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
