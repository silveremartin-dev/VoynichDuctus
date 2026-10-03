"""
Manuscript line and word segmentation via connected components and morphological projections.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops


class LineSegmenter:
    """
    Extracts text lines, words, and isolated character bounding boxes
    from binarized manuscript pages.
    """

    def __init__(self, line_smoothing_sigma: float = 4.0, min_line_height: int = 20, min_word_width: int = 15):
        self.line_smoothing_sigma = line_smoothing_sigma
        self.min_line_height = min_line_height
        self.min_word_width = min_word_width

    def isolate_text_region(self, binary_ink: np.ndarray, margin_crop_pct: float = 0.08) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops outer margins where binding, folio edges, and border noise occur.
        """
        H, W = binary_ink.shape
        y0, y1 = int(H * margin_crop_pct), int(H * (1.0 - margin_crop_pct))
        x0, x1 = int(W * margin_crop_pct), int(W * (1.0 - margin_crop_pct))
        return binary_ink[y0:y1, x0:x1], (y0, x0, y1, x1)

    def segment_lines(self, binary_ink: np.ndarray, auto_crop_margins: bool = True) -> List[Dict[str, Any]]:
        """
        Segments binary ink image into horizontal text lines using smoothed horizontal projection valleys.
        """
        if auto_crop_margins and (binary_ink.shape[0] > 400 and binary_ink.shape[1] > 400):
            work_img, (off_y, off_x, _, _) = self.isolate_text_region(binary_ink)
        else:
            work_img = binary_ink
            off_y, off_x = 0, 0

        # Horizontal projection profile
        raw_profile = np.sum(work_img, axis=1).astype(float)
        # Apply 1D Gaussian smoothing to merge ascenders/descenders into line bands
        from scipy.ndimage import gaussian_filter1d
        smoothed_profile = gaussian_filter1d(raw_profile, sigma=self.line_smoothing_sigma)

        # Dynamic valley thresholding
        threshold = np.median(smoothed_profile) * 0.4
        in_line = False
        start_y = 0
        line_spans = []

        for y, val in enumerate(smoothed_profile):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                end_y = y
                if end_y - start_y >= self.min_line_height:
                    line_spans.append((start_y, end_y))

        if in_line and len(smoothed_profile) - start_y >= self.min_line_height:
            line_spans.append((start_y, len(smoothed_profile)))

        results = []
        for line_idx, (ly0, ly1) in enumerate(line_spans):
            line_crop = work_img[ly0:ly1, :]
            # Refine horizontal column bounds
            col_profile = np.sum(line_crop, axis=0)
            nonzero_cols = np.where(col_profile > 2)[0]
            if len(nonzero_cols) == 0:
                continue
            lx0, lx1 = nonzero_cols[0], nonzero_cols[-1] + 1
            if lx1 - lx0 < self.min_word_width:
                continue

            abs_y0, abs_y1 = off_y + ly0, off_y + ly1
            abs_x0, abs_x1 = off_x + lx0, off_x + lx1

            results.append({
                "line_id": f"L{line_idx:03d}",
                "bbox": (int(abs_y0), int(abs_x0), int(abs_y1), int(abs_x1)),
                "image": binary_ink[abs_y0:abs_y1, abs_x0:abs_x1]
            })

        return results

    def segment_words(self, line_ink: np.ndarray, line_offset: Tuple[int, int] = (0, 0), space_threshold: int = 12) -> List[Dict[str, Any]]:
        """
        Segments a single line into constituent words using vertical projection gaps.
        """
        y_off, x_off = line_offset
        vert_profile = np.sum(line_ink, axis=0)

        in_word = False
        start_x = 0
        words = []
        gap = 0

        for x, val in enumerate(vert_profile):
            if val > 1:
                if not in_word:
                    in_word = True
                    start_x = x
                gap = 0
            else:
                if in_word:
                    gap += 1
                    if gap >= space_threshold:
                        in_word = False
                        end_x = x - gap + 1
                        if end_x - start_x >= self.min_word_width:
                            words.append((start_x, end_x))

        if in_word and line_ink.shape[1] - start_x >= self.min_word_width:
            words.append((start_x, line_ink.shape[1]))

        results = []
        for word_idx, (wx0, wx1) in enumerate(words):
            word_crop = line_ink[:, wx0:wx1]
            row_profile = np.sum(word_crop, axis=1)
            nonzero_rows = np.where(row_profile > 0)[0]
            if len(nonzero_rows) == 0:
                continue
            wy0, wy1 = nonzero_rows[0], nonzero_rows[-1] + 1

            results.append({
                "word_id": f"W{word_idx:03d}",
                "bbox": (int(y_off + wy0), int(x_off + wx0), int(y_off + wy1), int(x_off + wx1)),
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
