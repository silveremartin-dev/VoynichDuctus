"""
Vectorization and scribal kinematics module.
Converts 2D raster ink images into ordered parametric stroke trajectories and topological graphs.
"""

from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.inksight_adapter import InkSightAdapter
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.vectorizer.comparative_vectorizers import (
    ComparativeVectorizerBenchmark,
    GeometricVectorEngine,
    DINOv2EmbeddingEngine,
    InkSightVectorEngine
)

__all__ = [
    "Skeletonizer",
    "StrokeGraphExtractor",
    "JunctionResolver",
    "InkSightAdapter",
    "VectorExporter",
    "ComparativeVectorizerBenchmark",
    "GeometricVectorEngine",
    "DINOv2EmbeddingEngine",
    "InkSightVectorEngine",
]
