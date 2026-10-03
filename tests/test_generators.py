"""Unit tests for synthetic generators."""

import pytest
from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator
from voynich_ductus.generators.baselines import BaselineGenerator


def test_timm_generator():
    gen = TimmSelfCitationGenerator(memory_buffer_size=10, mutation_rate=0.3, seed=42)
    words = gen.generate(num_words=100)
    assert len(words) == 100
    assert all(isinstance(w, str) and len(w) > 0 for w in words)
    # Check that vocabulary repeats (self-citation signature)
    assert len(set(words)) < 100


def test_rugg_cardan_generator():
    gen = RuggCardanGenerator(table_size=8, seed=42)
    words = gen.generate(num_words=100)
    assert len(words) == 100
    assert all(isinstance(w, str) and len(w) > 0 for w in words)


def test_baselines_generator():
    latin = BaselineGenerator.get_natural_latin_sample(num_words=50)
    assert len(latin) == 50
    assert "herba" in latin or "est" in latin

    noise = BaselineGenerator.get_uniform_random_gibberish(num_words=50)
    assert len(noise) == 50

    markov = BaselineGenerator.get_markov_babbler(latin, order=1, num_words=30)
    assert len(markov) == 30
