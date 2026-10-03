"""
Information-theoretic and computational linguistics diagnostic suite.
Quantifies Shannon conditional entropy, long-range memory (Hurst/DFA),
Lempel-Ziv compressibility, and Markov automaton order.
"""

from voynich_ductus.diagnostics.entropy import ShannonEntropyAnalyzer
from voynich_ductus.diagnostics.memory import LongRangeMemoryAnalyzer
from voynich_ductus.diagnostics.compressibility import CompressibilityAnalyzer
from voynich_ductus.diagnostics.markov_automata import MarkovAutomatonAnalyzer
from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite

__all__ = [
    "ShannonEntropyAnalyzer",
    "LongRangeMemoryAnalyzer",
    "CompressibilityAnalyzer",
    "MarkovAutomatonAnalyzer",
    "BenchmarkSuite",
]
