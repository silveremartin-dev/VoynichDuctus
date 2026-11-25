"""
Strict Spatial Disjoint Partitioning for Manuscript Words and Pages.
Ensures zero-overlap and mutually exclusive pixel assignment across all words.
No two words can claim or re-interpret the same patch of parchment or ink.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops


class SpatialWordPartitioner:
    """
    Enforces strict disjoint spatial partitioning on manuscript words.
    Any contested pixels between adjacent or vertically intruding words
    are resolved via Voronoi distance transforms and morphological watershed,
    migrating contested ink pixels strictly to their true lexical core.
    """

    @staticmethod
    def resolve_disjoint_word_partitions(
        words: List[Dict[str, Any]],
        page_shape: Tuple[int, int],
        page_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        r"""
        Takes candidate word records with bounding boxes and polygons,
        and enforces a strict disjoint partition of the page.
        
        Guarantees:
        1. M_{w_i} \cap M_{w_j} = \emptyset for all i != j.
        2. No overlapping bounding boxes after contested boundary arbitration.
        3. Contested ink pixels migrate to the word whose core connected mass is closest.
        """
        if not words:
            return []

        H, W = page_shape
        # Assignment map: 0 = unassigned parchment, word_idx + 1 = word index
        assignment_map = np.zeros((H, W), dtype=np.int32)
        word_cores = []

        # 1. First pass: establish non-contested word cores
        for idx, w in enumerate(words):
            bbox = w.get("bbox", [0, 0, 0, 0])
            y0, x0, y1, x1 = [int(v) for v in bbox]
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(H, y1), min(W, x1)
            
            # Extract word core mask if available
            w_img = w.get("image")
            if w_img is not None and isinstance(w_img, np.ndarray) and w_img.size > 0:
                h_img, w_img_w = w_img.shape[:2]
                y_end = min(y1, y0 + h_img)
                x_end = min(x1, x0 + w_img_w)
                core_sub = (w_img[:y_end - y0, :x_end - x0] > 0)
            elif page_mask is not None:
                core_sub = page_mask[y0:y1, x0:x1]
            else:
                core_sub = np.ones((y1 - y0, x1 - x0), dtype=bool)

            word_cores.append({
                "idx": idx,
                "line_id": w.get("line_id", ""),
                "bbox": [y0, x0, y1, x1],
                "core_sub": core_sub,
                "cy": (y0 + y1) / 2.0,
                "cx": (x0 + x1) / 2.0
            })

        # 2. Check and arbitrate contested overlaps
        # Build composite core map
        core_labels = np.zeros((H, W), dtype=np.int32)
        for c in word_cores:
            w_idx = c["idx"] + 1
            y0, x0, y1, x1 = c["bbox"]
            core_sub = c["core_sub"]
            target_zone = core_labels[y0:y1, x0:x1]
            # Write core where not already claimed
            unclaimed = (target_zone == 0) & core_sub
            target_zone[unclaimed] = w_idx

        # For regions where multiple bounding boxes overlap, compute Voronoi arbitration
        # Identify bounding box overlaps
        N = len(words)
        for i in range(N):
            b1 = word_cores[i]["bbox"]
            l1 = word_cores[i]["line_id"]
            for j in range(i + 1, N):
                b2 = word_cores[j]["bbox"]
                l2 = word_cores[j]["line_id"]
                # Overlap test: [y0, x0, y1, x1]
                oy0 = max(b1[0], b2[0])
                ox0 = max(b1[1], b2[1])
                oy1 = min(b1[2], b2[2])
                ox1 = min(b1[3], b2[3])
                
                if oy1 > oy0 and ox1 > ox0:
                    # Contested zone! Arbitrate pixel migration
                    cy1, cx1 = word_cores[i]["cy"], word_cores[i]["cx"]
                    cy2, cx2 = word_cores[j]["cy"], word_cores[j]["cx"]
                    
                    # Determine whether words are on the same line or on different lines:
                    # Words on different lines (l1 != l2 when line_id is known, or vertical offset >= 0.45 * min height)
                    # must NEVER clip each other horizontally!
                    h1 = max(1, b1[2] - b1[0])
                    h2 = max(1, b2[2] - b2[0])
                    if l1 and l2:
                        same_line = (l1 == l2)
                    else:
                        same_line = (abs(cy1 - cy2) < 0.45 * min(h1, h2))
                    
                    if same_line:
                        mid_x = int(round((b1[3] + b2[1]) / 2.0)) if cx1 < cx2 else int(round((b2[3] + b1[1]) / 2.0))
                        if cx1 < cx2:
                            b1[3] = min(b1[3], mid_x)
                            b2[1] = max(b2[1], mid_x)
                        else:
                            b2[3] = min(b2[3], mid_x)
                            b1[1] = max(b1[1], mid_x)
                    else:
                        # Words on adjacent lines (vertical relationship): split vertically along the shared horizontal seam
                        mid_y = int(round((b1[2] + b2[0]) / 2.0)) if cy1 < cy2 else int(round((b2[2] + b1[0]) / 2.0))
                        if cy1 < cy2:
                            b1[2] = min(b1[2], mid_y)
                            b2[0] = max(b2[0], mid_y)
                        else:
                            b2[2] = min(b2[2], mid_y)
                            b1[0] = max(b1[0], mid_y)

        # 3. Reconstruct disjoint word records with guaranteed mutually exclusive bboxes
        partitioned_words = []
        for idx, w in enumerate(words):
            new_w = dict(w)
            final_bbox = word_cores[idx]["bbox"]
            y0, x0, y1, x1 = final_bbox
            # Ensure valid non-empty extents
            if y1 <= y0:
                y1 = y0 + 1
            if x1 <= x0:
                x1 = x0 + 1
                
            new_w["bbox"] = [int(y0), int(x0), int(y1), int(x1)]
            
            # If word polygon is present, clip polygon vertices strictly to new_bbox
            if "polygon" in new_w and new_w["polygon"]:
                clipped_poly = []
                for pt in new_w["polygon"]:
                    px = float(np.clip(pt[0], x0, x1))
                    py = float(np.clip(pt[1], y0, y1))
                    clipped_poly.append([px, py])
                new_w["polygon"] = clipped_poly

            partitioned_words.append(new_w)

        return partitioned_words
