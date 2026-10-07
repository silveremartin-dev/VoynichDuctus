"""Unit tests for ingestion, binarization, and segmentation."""

import pytest
import numpy as np
from PIL import Image
from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter


def test_iiif_normalization():
    assert IIIFClient.normalize_folio_name("f1r") == "f001r"
    assert IIIFClient.normalize_folio_name("67v2") == "f067v2"
    assert IIIFClient.normalize_folio_name("F100R") == "f100r"


def test_binarization():
    # Create synthetic image with a dark stripe on light background
    arr = np.ones((50, 50), dtype=np.uint8) * 200
    arr[20:30, :] = 50  # Dark ink bar

    binarizer = Binarizer(method="sauvola", window_size=15)
    binary = binarizer.binarize(arr)

    assert binary.shape == (50, 50)
    assert np.any(binary[20:30, :])  # Ink region detected as True
    assert not np.all(binary)


def test_segmenter():
    # Synthetic page with 2 lines
    img = np.zeros((100, 100), dtype=bool)
    img[15:30, 10:80] = True  # Line 1
    img[60:75, 10:80] = True  # Line 2

    segmenter = LineSegmenter(min_line_height=10)
    lines = segmenter.segment_lines(img)
    assert len(lines) == 2

    # Segment words on line 1 with a gap
    line1 = np.zeros((20, 80), dtype=bool)
    line1[2:18, 5:30] = True   # Word 1
    line1[2:18, 50:75] = True  # Word 2

    words = segmenter.segment_words(line1)
    assert len(words) == 2


def test_color_normalizer():
    from voynich_ductus.ingestion.color_normalizer import ColorIlluminationNormalizer

    normalizer = ColorIlluminationNormalizer()
    
    # Create synthetic RGB image with parchment background (warm beige), dark text ink, and green plant leaf
    h, w = 100, 100
    img_arr = np.full((h, w, 3), [220, 200, 170], dtype=np.uint8) # parchment
    img_arr[20:30, 20:80] = [40, 30, 20] # dark ink text
    img_arr[60:80, 20:80] = [40, 150, 40] # bright green leaf pigment
    
    pil_img = Image.fromarray(img_arr)
    ink_mask, clean_gray = normalizer.extract_ink_mask_chromatic(pil_img)
    
    assert ink_mask.shape == (h, w)
    # Ink area should be True (detected)
    assert np.mean(ink_mask[22:28, 22:78]) > 0.5
    # Green plant area should be rejected (mostly False)
    assert np.mean(ink_mask[62:78, 22:78]) < 0.2


def test_transcription_reference_and_yield_validator():
    from voynich_ductus.ingestion.transcription_reference import TranscriptionReference, PaleographyYieldValidator

    # Test normalization of folio IDs
    meta_f1r = TranscriptionReference.get_metadata("f1r")
    assert meta_f1r is not None
    assert meta_f1r.folio_id == "f001r"
    assert meta_f1r.expected_lines == 28
    assert meta_f1r.currier_language == "Currier A"
    assert meta_f1r.scribe_hand == "Hand 1"

    # Test yield evaluation
    yield_res = PaleographyYieldValidator.evaluate_yield(
        folio_id="f001r",
        detected_lines=19,
        detected_glyphs=203,
        extracted_strokes=577
    )
    assert yield_res["has_reference"] is True
    assert yield_res["line_yield_pct"] == 67.9
    assert yield_res["expected_lines"] == 28
    assert yield_res["detected_lines"] == 19
    assert yield_res["detected_glyphs"] == 203
    assert yield_res["expected_glyphs"] == 1180


def test_extract_page_glyphs_with_ground_truth():
    from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
    from voynich_ductus.ingestion.voynichese_rosetta import RosettaWord
    from PIL import ImageDraw

    segmenter = GlyphSegmenter()
    # Create synthetic test page image (light parchment background with drawn strokes)
    h, w = 120, 200
    img = Image.new("RGB", (w, h), color=(235, 225, 205))
    draw = ImageDraw.Draw(img)
    # Draw dark strokes for word "daiin" in word bbox [30, 40, 150, 70]
    draw.ellipse([35, 45, 55, 65], outline=(30, 25, 20), width=3) # 'd' bowl
    draw.line([(55, 38), (55, 68)], fill=(30, 25, 20), width=3)    # 'd' ascender
    draw.ellipse([65, 48, 85, 68], outline=(30, 25, 20), width=3) # 'a'
    draw.line([(95, 48), (95, 68)], fill=(30, 25, 20), width=3)    # 'i'
    draw.line([(110, 48), (110, 68)], fill=(30, 25, 20), width=3)  # 'i'
    draw.line([(125, 48), (125, 68)], fill=(30, 25, 20), width=3)  # 'n'

    r_words = [
        RosettaWord(
            folio="f1r",
            index=0,
            eva_text="daiin",
            x_xml=30,
            y_xml=40,
            width_xml=120,
            height_xml=30,
            bbox_scan=(30, 35, 150, 75),
        )
    ]

    res = segmenter.extract_page_glyphs_with_ground_truth("f001r", img, rosetta_words=r_words)
    assert res["word_count"] == 1
    assert res["words"][0]["eva_text"] == "daiin"
    assert len(res["glyphs"]) >= 1
    assert "eva_char" in res["glyphs"][0]




