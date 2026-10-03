"""Unit tests for information-theoretic diagnostics."""

import pytest
import numpy as np
from voynich_ductus.diagnostics.entropy import ShannonEntropyAnalyzer
from voynich_ductus.diagnostics.memory import LongRangeMemoryAnalyzer
from voynich_ductus.diagnostics.compressibility import CompressibilityAnalyzer
from voynich_ductus.diagnostics.markov_automata import MarkovAutomatonAnalyzer
from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite


def test_shannon_entropy():
    # Constant string has 0 entropy
    assert ShannonEntropyAnalyzer.shannon_entropy("aaaaaa") == 0.0

    # 4 equally probable characters has log2(4) = 2.0 bits
    assert abs(ShannonEntropyAnalyzer.shannon_entropy("abcd") - 2.0) < 1e-4

    # Conditional entropy
    res = ShannonEntropyAnalyzer.analyze_text(["hello", "world", "voynich", "manuscript"])
    assert res["h1_char_bits"] > 0
    assert res["h2_cond_char_bits"] >= 0


def test_long_range_memory():
    # Long range repeated words vs random
    sample_words = ["word" + str(i % 5) for i in range(100)]
    res = LongRangeMemoryAnalyzer.analyze_memory(sample_words)
    assert "hurst_exponent" in res
    assert 0.0 <= res["hurst_exponent"] <= 1.5


def test_compressibility():
    repeated_text = "qokedy " * 500
    res = CompressibilityAnalyzer.measure(repeated_text)
    assert res["gzip_ratio"] < 0.20  # Highly compressible


def test_markov_automata():
    seq = ["A", "B", "A", "B", "A", "B", "A", "B"]
    res = MarkovAutomatonAnalyzer.evaluate_transition_predictability(seq, order=1)
    assert res["top1_prediction_accuracy"] == 1.0


def test_benchmark_suite():
    suite = BenchmarkSuite()
    corpora = {
        "Test1": ["qokedy", "dal", "qokedy", "chedy"] * 20,
        "Test2": ["lorem", "ipsum", "dolor", "sit", "amet"] * 20
    }
    results = suite.compare_corpora(corpora)
    assert "Test1" in results
    assert "Test2" in results
    assert "top_hypothesis" in results["Test1"]["diagnostic_diagnosis"]
