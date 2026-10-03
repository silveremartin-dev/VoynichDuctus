"""
Comparative Benchmark Script:
Evaluates and contrasts information-theoretic signatures across candidate hypotheses.
"""

from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator
from voynich_ductus.generators.baselines import BaselineGenerator
from voynich_ductus.utils.io import BenchmarkFormatter


def run_benchmark():
    words_per_corpus = 1500
    print(f"[*] Generating {words_per_corpus} words per hypothesis model...")

    timm_corpus = TimmSelfCitationGenerator(seed=42).generate(words_per_corpus)
    rugg_corpus = RuggCardanGenerator(seed=42).generate(words_per_corpus)
    latin_corpus = BaselineGenerator.get_natural_latin_sample(words_per_corpus)
    markov_corpus = BaselineGenerator.get_markov_babbler(latin_corpus, order=1, num_words=words_per_corpus)
    random_corpus = BaselineGenerator.get_uniform_random_gibberish(words_per_corpus)

    corpora = {
        "Latin Herbal (15th c. Natural Language)": latin_corpus,
        "Timm Self-Citation (Voynich-like Model)": timm_corpus,
        "Rugg Cardan Grille (Mechanical Grid)": rugg_corpus,
        "Markov Babbler (Order-1 Grammar)": markov_corpus,
        "Uniform Random Gibberish (Noise)": random_corpus,
    }

    print("[*] Running formal information-theoretic diagnostics...")
    suite = BenchmarkSuite()
    results = suite.compare_corpora(corpora)

    print("\n" + "=" * 95)
    print("                      VOYNICH DUCTUS — COMPARATIVE DIAGNOSTIC MATRIX")
    print("=" * 95 + "\n")
    print(BenchmarkFormatter.to_markdown_table(results))
    print("\n" + "=" * 95)


if __name__ == "__main__":
    run_benchmark()
