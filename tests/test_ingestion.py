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
