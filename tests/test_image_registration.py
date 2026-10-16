"""
Unit tests for MultimodalFolioAligner (SIFT/RANSAC multimodal image registration).
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw
from voynich_ductus.ingestion.image_registration import MultimodalFolioAligner


def create_synthetic_textured_document(width=600, height=800):
    """Creates a synthetic page with parchment texture and unique, randomized letterforms."""
    rng = np.random.RandomState(42)
    # Parchment background noise
    bg_noise = rng.randint(215, 240, (height, width), dtype=np.uint8)
    img = Image.fromarray(bg_noise)
    draw = ImageDraw.Draw(img)

    # Draw distinct, non-repeating features
    for idx in range(35):
        cx = int(rng.randint(60, width - 80))
        cy = int(rng.randint(60, height - 80))
        r = int(rng.randint(12, 28))
        color = int(rng.randint(20, 70))
        shape_type = idx % 4
        if shape_type == 0:
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        elif shape_type == 1:
            draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], outline=color, width=3)
        elif shape_type == 2:
            draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], outline=color, fill=color + 30)
        else:
            draw.line([(cx - r, cy), (cx + r, cy + r)], fill=color, width=4)
            draw.line([(cx, cy - r), (cx, cy + r)], fill=color, width=3)

    return np.array(img)


def test_aligner_initialization():
    aligner = MultimodalFolioAligner(cache_dir="intermediate/test_cache/registration")
    assert aligner is not None
    assert aligner.sift is not None


def test_synthetic_registration_accuracy():
    aligner = MultimodalFolioAligner(cache_dir="intermediate/test_cache/registration")
    ref_img = create_synthetic_textured_document(width=600, height=800)

    scale_gt = 1.4
    angle_gt = 1.5
    tx_gt = 60.0
    ty_gt = 40.0

    center = (ref_img.shape[1] / 2, ref_img.shape[0] / 2)
    M_rot_scale = cv2.getRotationMatrix2D(center, angle_gt, scale_gt)
    M_gt = np.copy(M_rot_scale)
    M_gt[0, 2] += tx_gt
    M_gt[1, 2] += ty_gt

    tgt_w = int(ref_img.shape[1] * scale_gt + 150)
    tgt_h = int(ref_img.shape[0] * scale_gt + 150)
    tgt_img = cv2.warpAffine(ref_img, M_gt, (tgt_w, tgt_h), borderValue=230)

    res = aligner.compute_registration(ref_img, tgt_img, folio_id="synthetic_test", force_recompute=True)
    assert res["status"] == "success"
    assert res["inliers"] >= 10
    assert abs(res["scale_x"] - scale_gt) < 0.08
    assert abs(abs(res["rotation_deg"]) - abs(angle_gt)) < 0.5

    # Test BBox transformation (interior box)
    test_bbox = (150, 150, 200, 220)
    transformed_bbox = aligner.transform_bbox(test_bbox, res["matrix_2x3"])
    assert transformed_bbox[0] >= 0
    assert transformed_bbox[1] >= 0
    assert transformed_bbox[2] > transformed_bbox[0]
    assert transformed_bbox[3] > transformed_bbox[1]

    # Test HD Patch Extraction
    patch = aligner.extract_registered_hd_patch(tgt_img, test_bbox, res["matrix_2x3"], pad=4)
    assert patch is not None
    assert patch.ndim >= 2
    assert patch.shape[0] > 0 and patch.shape[1] > 0


def test_real_voynich_folio_registration():
    glance_file = Path("data/scans/voynich/voynichese_scans/f1r.jpg")
    yale_file = Path("data/scans/voynich/yale/f001r.jpg")
    if not glance_file.exists() or not yale_file.exists():
        pytest.skip("Scans f1r/f001r not found on disk")

    aligner = MultimodalFolioAligner(cache_dir="intermediate/test_cache/registration")
    img_glance = cv2.imread(str(glance_file))
    img_yale = cv2.imread(str(yale_file))

    res = aligner.compute_registration(img_glance, img_yale, folio_id="f1r_test", force_recompute=True)
    assert res["status"] == "success"
    assert res["inliers"] >= 30
    assert 2.5 <= res["scale_x"] <= 3.2
    assert abs(res["rotation_deg"]) < 2.0
    assert res["tx"] > 300
    assert res["ty"] > 150
