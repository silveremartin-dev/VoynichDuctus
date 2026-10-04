import pytest
import numpy as np
from PIL import Image

from voynich_ductus.vectorizer.comparative_vectorizers import (
    ComparativeVectorizerBenchmark,
    GeometricVectorEngine,
    DINOv2EmbeddingEngine,
    InkSightVectorEngine
)
from voynich_ductus.ingestion.segmenter import LineSegmenter


def test_comparative_vectorizers():
    # Create synthetic test glyph mask (a closed loop 'o')
    mask = np.zeros((40, 40), dtype=bool)
    rr, cc = np.ogrid[:40, :40]
    circle_outer = (rr - 20)**2 + (cc - 20)**2 <= 14**2
    circle_inner = (rr - 20)**2 + (cc - 20)**2 <= 8**2
    mask = circle_outer & (~circle_inner)

    benchmark = ComparativeVectorizerBenchmark()
    res = benchmark.evaluate_glyph(mask)

    assert "geometric" in res
    assert "dinov2" in res
    assert "inksight" in res
    assert "summary" in res

    # Check geometric metrics
    assert res["geometric"]["stroke_count"] >= 1
    assert "<svg" in res["geometric"]["svg_content"]
    assert res["geometric"]["latency_ms"] >= 0

    # Check DINOv2
    assert res["dinov2"]["embedding_dim"] > 0
    assert len(res["dinov2"]["embedding_sample"]) > 0

    # Check InkSight
    assert res["inksight"]["stroke_count"] >= 1
    assert "<svg" in res["inksight"]["svg_content"]


def test_line_segmenter_new_paleographic_methods():
    segmenter = LineSegmenter()

    # Synthetic multi-line page
    page = np.zeros((300, 200), dtype=bool)
    # 5 text lines with 50px pitch
    for y in [40, 90, 140, 190, 240]:
        page[y:y+6, 20:180] = True

    pitch = segmenter.estimate_line_pitch_fft(page)
    # FFT line pitch should be close to 50px
    assert 40 <= pitch <= 60

    # Macro-block detection
    blocks = segmenter.detect_text_macro_blocks(page)
    assert len(blocks) >= 1

    # Nib width filtering
    clean = segmenter.filter_nib_width_consistency(page, max_radius=4.5)
    assert np.sum(clean) > 0
