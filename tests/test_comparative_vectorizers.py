import numpy as np

from voynich_ductus.vectorizer.comparative_vectorizers import (
    ComparativeVectorizerBenchmark,
    GeometricVectorEngine,
    DINOv2EmbeddingEngine,
)
from voynich_ductus.ingestion.segmenter import LineSegmenter


def test_comparative_vectorizers():
    mask = np.zeros((40, 40), dtype=bool)
    rr, cc = np.ogrid[:40, :40]
    circle_outer = (rr - 20) ** 2 + (cc - 20) ** 2 <= 14 ** 2
    circle_inner = (rr - 20) ** 2 + (cc - 20) ** 2 <= 8 ** 2
    mask = circle_outer & (~circle_inner)

    benchmark = ComparativeVectorizerBenchmark()
    res = benchmark.evaluate_glyph(mask)

    assert "calligraphic" in res
    assert "geometric" in res
    assert "dinov2" in res
    assert "summary" in res
    assert "inksight" not in res

    assert res["calligraphic"]["stroke_count"] >= 1
    assert "<svg" in res["calligraphic"]["svg_content"]
    assert res["geometric"]["stroke_count"] >= 1
    assert "<svg" in res["geometric"]["svg_content"]
    assert res["dinov2"]["embedding_dim"] > 0
    assert len(res["dinov2"]["embedding_sample"]) > 0


def test_line_segmenter_new_paleographic_methods():
    segmenter = LineSegmenter()

    page = np.zeros((300, 200), dtype=bool)
    for y in [40, 90, 140, 190, 240]:
        page[y : y + 6, 20:180] = True

    pitch = segmenter.estimate_line_pitch_fft(page)
    assert 40 <= pitch <= 60

    blocks = segmenter.detect_text_macro_blocks(page)
    assert len(blocks) >= 1

    clean = segmenter.filter_nib_width_consistency(page, max_radius_mm=0.56)
    assert np.sum(clean) > 0
