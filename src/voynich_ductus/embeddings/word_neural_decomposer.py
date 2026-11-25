"""
Backward-compatible re-export of word_decomposition module.
All word-to-glyph decomposition classes now live in `voynich_ductus.embeddings.word_decomposition`.
"""

from voynich_ductus.embeddings.word_decomposition import (
    WordScribalEncoder,
    NeuralWordDecomposition,
    HybridConsensusWordDecomposition,
    NeuralWordDecomposer,
    HybridNeuroKinematicDecomposer,
    compute_bbox_iou,
    cross_evaluate_options_consensus,
)

__all__ = [
    "WordScribalEncoder",
    "NeuralWordDecomposition",
    "HybridConsensusWordDecomposition",
    "NeuralWordDecomposer",
    "HybridNeuroKinematicDecomposer",
    "compute_bbox_iou",
    "cross_evaluate_options_consensus",
]
