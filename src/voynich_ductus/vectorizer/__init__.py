"""
Vectorization and scribal kinematics module.
Converts 2D raster ink images into biomechanically ordered stroke trajectories and topological graphs.
"""

from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.vectorizer.comparative_vectorizers import (
    ComparativeVectorizerBenchmark,
    CalligraphicVectorEngine,
    GeometricVectorEngine,
    ScribalFlowVectorEngine,
    DINOv2EmbeddingEngine,
)

__all__ = [
    "CalligraphicVectorizer",
    "Skeletonizer",
    "StrokeGraphExtractor",
    "JunctionResolver",
    "VectorExporter",
    "ComparativeVectorizerBenchmark",
    "CalligraphicVectorEngine",
    "GeometricVectorEngine",
    "ScribalFlowVectorEngine",
    "DINOv2EmbeddingEngine",
]
