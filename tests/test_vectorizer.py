"""Unit tests for skeletonization, stroke graphs, and vector export."""

import pytest
import numpy as np
from pathlib import Path
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter


def test_skeleton_and_graph():
    # Synthetic "T" shape ink
    img = np.zeros((30, 30), dtype=bool)
    img[5:8, 5:25] = True   # horizontal bar
    img[8:25, 14:17] = True  # vertical stem

    skel_engine = Skeletonizer(method="medial_axis")
    skel, widths = skel_engine.extract_skeleton(img)

    assert np.any(skel)
    assert np.any(widths > 0)

    graph_extractor = StrokeGraphExtractor()
    pixel_graph = graph_extractor.build_pixel_graph(skel, widths)
    assert pixel_graph.number_of_nodes() > 0

    strokes = graph_extractor.decompose_into_strokes(pixel_graph)
    assert len(strokes) >= 1

    resolver = JunctionResolver()
    ordered = resolver.resolve_and_order_strokes(strokes)
    assert len(ordered) == len(strokes)


def test_vector_export(tmp_path: Path):
    strokes = [
        {"stroke_id": "S1", "points": [(10, 10, 2.0), (10, 20, 2.0)], "order_index": 0},
        {"stroke_id": "S2", "points": [(10, 15, 2.0), (30, 15, 2.0)], "order_index": 1}
    ]
    svg_file = tmp_path / "test.svg"
    svg_str = VectorExporter.to_svg(strokes, width=50, height=50, output_path=svg_file)
    assert "<svg" in svg_str
    assert svg_file.exists()
