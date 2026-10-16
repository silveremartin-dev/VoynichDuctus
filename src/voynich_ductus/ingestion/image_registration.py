"""
Multimodal Folio Alignment and Image Registration Engine.
Computes the exact geometric invariant (Scale, Rotation, Translation, Homography)
between reference glance plates (e.g. Voynichese 636x900 / 1272x1800) and Yale Beinecke HD scans (2400x3300),
mapping ground-truth word and glyph bounding boxes onto pristine high-resolution parchment plates.
"""

from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image


class MultimodalFolioAligner:
    """
    Solves planar registration between multi-source manuscript scans using SIFT/ORB keypoints
    and robust RANSAC affine consensus.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = "data/annotations/voynichese/registration_cache"):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def compute_registration(
        self,
        reference_img: Union[np.ndarray, Image.Image],
        target_img: Union[np.ndarray, Image.Image],
        folio_id: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Computes the affine transformation matrix M mapping reference_img coordinates (x, y)
        to target_img coordinates (x', y').

        Returns
        -------
        Dictionary containing:
        - 'matrix_2x3': 2x3 affine matrix [[a, b, tx], [c, d, ty]]
        - 'scale_x', 'scale_y', 'rotation_deg', 'tx', 'ty'
        - 'inliers', 'matches', 'inlier_ratio'
        - 'status': 'success' or 'failed'
        """
        # Check cache if folio_id provided
        if folio_id and self.cache_dir and not force_recompute:
            cache_file = self.cache_dir / f"{folio_id.lower()}_affine.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["matrix_2x3"] = np.array(data["matrix_2x3"], dtype=np.float32)
                    return data
                except Exception:
                    pass

        # Convert to numpy uint8 grayscale
        ref_gray = self._to_gray(reference_img)
        tgt_gray = self._to_gray(target_img)

        # 1. Feature detection & description (SIFT)
        kp_ref, des_ref = self.sift.detectAndCompute(ref_gray, None)
        kp_tgt, des_tgt = self.sift.detectAndCompute(tgt_gray, None)

        if des_ref is None or des_tgt is None or len(kp_ref) < 8 or len(kp_tgt) < 8:
            return {"status": "failed", "error": "Insufficient keypoints detected."}

        # 2. FLANN Matcher + Lowe's Ratio Test
        knn_matches = self.flann.knnMatch(des_ref, des_tgt, k=2)
        good_matches = [m for m, n in knn_matches if m.distance < 0.72 * n.distance]

        if len(good_matches) < 8:
            # Fallback with relaxed ratio
            good_matches = [m for m, n in knn_matches if m.distance < 0.80 * n.distance]

        if len(good_matches) < 6:
            return {"status": "failed", "error": f"Too few geometric matches ({len(good_matches)})."}

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 3. Robust RANSAC Estimation: Affine Partial (Scale + Rotation + Translation)
        M_affine, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.5
        )

        if M_affine is None:
            return {"status": "failed", "error": "RANSAC affine estimation failed to converge."}

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        inlier_ratio = float(inlier_count / len(good_matches))

        # Decompose affine parameters
        sx = float(np.sqrt(M_affine[0, 0]**2 + M_affine[1, 0]**2))
        sy = float(np.sqrt(M_affine[0, 1]**2 + M_affine[1, 1]**2))
        theta_rad = float(np.arctan2(M_affine[1, 0], M_affine[0, 0]))
        theta_deg = float(np.rad2deg(theta_rad))
        tx = float(M_affine[0, 2])
        ty = float(M_affine[1, 2])

        result = {
            "status": "success",
            "folio_id": folio_id or "unknown",
            "matrix_2x3": M_affine,
            "scale_x": round(sx, 5),
            "scale_y": round(sy, 5),
            "rotation_deg": round(theta_deg, 3),
            "tx": round(tx, 2),
            "ty": round(ty, 2),
            "inliers": inlier_count,
            "matches": len(good_matches),
            "inlier_ratio": round(inlier_ratio, 3)
        }

        # Save to cache
        if folio_id and self.cache_dir:
            cache_file = self.cache_dir / f"{folio_id.lower()}_affine.json"
            save_data = dict(result)
            save_data["matrix_2x3"] = M_affine.tolist()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2)

        return result

    @staticmethod
    def transform_point(point: Tuple[float, float], matrix_2x3: np.ndarray) -> Tuple[float, float]:
        """Transforms a single (x, y) point."""
        x, y = point
        x_new = matrix_2x3[0, 0] * x + matrix_2x3[0, 1] * y + matrix_2x3[0, 2]
        y_new = matrix_2x3[1, 0] * x + matrix_2x3[1, 1] * y + matrix_2x3[1, 2]
        return float(x_new), float(y_new)

    @staticmethod
    def transform_bbox(bbox: Tuple[int, int, int, int], matrix_2x3: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Transforms a bounding box [y0, x0, y1, x1] from reference to target coordinates.
        Returns [y0_target, x0_target, y1_target, x1_target].
        """
        y0, x0, y1, x1 = bbox
        corners = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1]
        ], dtype=np.float32).reshape(-1, 1, 2)

        corners_tgt = cv2.transform(corners, matrix_2x3).reshape(-1, 2)
        min_x = int(np.floor(np.min(corners_tgt[:, 0])))
        max_x = int(np.ceil(np.max(corners_tgt[:, 0])))
        min_y = int(np.floor(np.min(corners_tgt[:, 1])))
        max_y = int(np.ceil(np.max(corners_tgt[:, 1])))

        return min_y, min_x, max_y, max_x

    @staticmethod
    def transform_polygon(polygon: List[List[float]], matrix_2x3: np.ndarray) -> List[List[float]]:
        """Transforms a polygon contour [[x0, y0], [x1, y1], ...]."""
        if not polygon:
            return []
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt = cv2.transform(pts, matrix_2x3).reshape(-1, 2)
        return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts_tgt]

    def extract_registered_hd_patch(
        self,
        target_img: Union[np.ndarray, Image.Image],
        reference_bbox: Tuple[int, int, int, int],
        matrix_2x3: np.ndarray,
        pad: int = 6
    ) -> Optional[np.ndarray]:
        """
        Extracts the high-resolution patch from target_img corresponding to reference_bbox.
        """
        tgt_arr = np.array(target_img)
        th, tw = tgt_arr.shape[:2]

        y0, x0, y1, x1 = self.transform_bbox(reference_bbox, matrix_2x3)
        cy0 = max(0, y0 - pad)
        cx0 = max(0, x0 - pad)
        cy1 = min(th, y1 + pad)
        cx1 = min(tw, x1 + pad)

        if cy1 <= cy0 or cx1 <= cx0:
            return None

        return tgt_arr[cy0:cy1, cx0:cx1]

    @staticmethod
    def _to_gray(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img

        if arr.ndim == 2:
            return arr
        elif arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
