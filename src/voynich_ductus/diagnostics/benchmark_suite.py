"""
Benchmark Suite: executes comprehensive comparative statistical paleography diagnostics.
"""

from typing import Dict, List, Any, Union
from voynich_ductus.diagnostics.entropy import ShannonEntropyAnalyzer
from voynich_ductus.diagnostics.memory import LongRangeMemoryAnalyzer
from voynich_ductus.diagnostics.compressibility import CompressibilityAnalyzer
from voynich_ductus.diagnostics.markov_automata import MarkovAutomatonAnalyzer


class BenchmarkSuite:
    """
    Executes the 4 key formal diagnostic tests to differentiate between:
    1. Uncorrelated random gibberish
    2. Mechanical automaton / Cardan grid generator
    3. Self-citation / Timm algorithm generator
    4. Natural human language (Latin, Middle English, Italian)
    """

    def evaluate_corpus(self, name: str, corpus: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Runs complete battery of tests on a given corpus.
        """
        if isinstance(corpus, str):
            words = corpus.split()
        else:
            words = corpus

        entropy_res = ShannonEntropyAnalyzer.analyze_text(words)
        memory_res = LongRangeMemoryAnalyzer.analyze_memory(words)
        comp_res = CompressibilityAnalyzer.measure(words)
        markov_res = MarkovAutomatonAnalyzer.analyze_grammar_type(words)

        # Classification decision rules
        # Natural Language signature: Hurst >= 0.62, Gzip ratio 0.30 - 0.48, H2/H1 moderate drop
        # Mechanical Automaton signature: Gzip ratio < 0.28, Top-1 order-2 > 0.75, Hurst < 0.58
        # Timm Self-Citation signature: Gzip ratio 0.25 - 0.35, Hurst 0.60 - 0.75, Top-1 order-2 high

        hurst = memory_res["hurst_exponent"]
        gz_ratio = comp_res["gzip_ratio"]
        top1_order2 = markov_res["order_2"]["top1_prediction_accuracy"]

        hypothesis_scores = {
            "random_gibberish": 0.0,
            "mechanical_cardan_grid": 0.0,
            "timm_self_citation": 0.0,
            "natural_language": 0.0
        }

        # Evaluate Hurst (Long-range memory)
        if hurst < 0.55:
            hypothesis_scores["random_gibberish"] += 2.0
            hypothesis_scores["mechanical_cardan_grid"] += 1.5
        elif hurst >= 0.65:
            hypothesis_scores["natural_language"] += 2.0
            hypothesis_scores["timm_self_citation"] += 2.0

        # Evaluate Compression
        if gz_ratio > 0.60:
            hypothesis_scores["random_gibberish"] += 3.0
        elif gz_ratio < 0.28:
            hypothesis_scores["mechanical_cardan_grid"] += 3.0
            hypothesis_scores["timm_self_citation"] += 1.5
        elif 0.28 <= gz_ratio <= 0.48:
            hypothesis_scores["natural_language"] += 2.5
            hypothesis_scores["timm_self_citation"] += 2.0

        # Evaluate Markov Order 2 Determinism
        if top1_order2 > 0.75:
            hypothesis_scores["mechanical_cardan_grid"] += 2.0
            hypothesis_scores["timm_self_citation"] += 2.5
        else:
            hypothesis_scores["natural_language"] += 1.5
            hypothesis_scores["random_gibberish"] += 1.0

        best_hypothesis = max(hypothesis_scores, key=hypothesis_scores.get)

        return {
            "corpus_name": name,
            "token_count": len(words),
            "entropy": entropy_res,
            "memory": memory_res,
            "compressibility": comp_res,
            "markov_automata": markov_res,
            "diagnostic_diagnosis": {
                "top_hypothesis": best_hypothesis,
                "confidence_scores": hypothesis_scores
            }
        }

    def compare_corpora(self, corpora_dict: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        """
        Runs benchmark across multiple candidate or control corpora simultaneously.
        """
        results = {}
        for name, corpus in corpora_dict.items():
            results[name] = self.evaluate_corpus(name, corpus)
        return results
