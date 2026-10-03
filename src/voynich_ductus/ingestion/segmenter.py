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

    def __init__(self, line_smoothing_sigma: float = 3.0, min_line_height: int = 15, min_word_width: int = 10):
        self.line_smoothing_sigma = line_smoothing_sigma
        self.min_line_height = min_line_height
        self.min_word_width = min_word_width

    def segment_lines(self, binary_ink: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments binary ink image into horizontal text lines using horizontal projection profile.
        
        Args:
            binary_ink: 2D boolean array (True = ink)
            
        Returns:
            List of line records with bbox (ymin, xmin, ymax, xmax) and sub-images.
        """
        # Horizontal projection profile: sum ink pixels along rows
        profile = np.sum(binary_ink, axis=1)

        # Find valleys / gaps between lines
        threshold = np.mean(profile) * 0.1
        in_line = False
        start_y = 0
        lines = []

        for y, val in enumerate(profile):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                end_y = y
                if end_y - start_y >= self.min_line_height:
                    lines.append((start_y, end_y))

        if in_line and binary_ink.shape[0] - start_y >= self.min_line_height:
            lines.append((start_y, binary_ink.shape[0]))

        results = []
        for line_idx, (y0, y1) in enumerate(lines):
            line_crop = binary_ink[y0:y1, :]
            # Find horizontal bounding box bounds
            col_profile = np.sum(line_crop, axis=0)
            nonzero_cols = np.where(col_profile > 0)[0]
            if len(nonzero_cols) == 0:
                continue
            x0, x1 = nonzero_cols[0], nonzero_cols[-1] + 1

            results.append({
                "line_id": f"L{line_idx:03d}",
                "bbox": (int(y0), int(x0), int(y1), int(x1)),
                "image": binary_ink[y0:y1, x0:x1]
            })

        return results

    def segment_words(self, line_ink: np.ndarray, line_offset: Tuple[int, int] = (0, 0), space_threshold_factor: float = 0.5) -> List[Dict[str, Any]]:
        """
        Segments a single line into constituent words using vertical projection gaps.
        """
        y_off, x_off = line_offset
        vert_profile = np.sum(line_ink, axis=0)

        in_word = False
        start_x = 0
        words = []

        # Find typical character stroke gap vs inter-word space
        zero_runs = []
        cur_zero_run = 0
        for val in vert_profile:
            if val == 0:
                cur_zero_run += 1
            else:
                if cur_zero_run > 0:
                    zero_runs.append(cur_zero_run)
                    cur_zero_run = 0

        space_threshold = np.median(zero_runs) * 1.5 if len(zero_runs) > 0 else 5
        space_threshold = max(space_threshold, 4)

        gap = 0
        for x, val in enumerate(vert_profile):
            if val > 0:
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
            # Refine vertical bounds
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
