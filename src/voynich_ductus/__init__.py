"""
Voynich Ductus: Objective Vector Glyphs, Ductus Kinematics & Statistical Paleography Pipeline.

A reproducible scientific pipeline designed to bypass subjective transcription biases (e.g., EVA)
by extracting topological stroke graphs directly from high-resolution manuscript scans,
inferring scribal kinematics, clustering raw geometric primitives, and subjecting the resulting
unsupervised token streams to rigorous information-theoretic diagnostics (Kolmogorov, Hurst/DFA,
Markov order, Shannon conditional entropy) against adversarial synthetic generators.
"""

__version__ = "0.1.0"
__author__ = "Voynich Ductus Contributors"

from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer
from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator

__all__ = [
    "BenchmarkSuite",
    "Skeletonizer",
    "StrokeGraphExtractor",
    "GlyphClusterer",
    "StrokeTokenizer",
    "TimmSelfCitationGenerator",
    "RuggCardanGenerator",
]
