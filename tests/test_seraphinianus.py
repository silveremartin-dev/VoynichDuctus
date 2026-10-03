"""Unit tests for Codex Seraphinianus engine."""

import pytest
from voynich_ductus.generators.seraphinianus import SeraphinianusEngine


def test_seraphinianus_generator():
    tokens = SeraphinianusEngine.generate_serafini_text_tokens(num_words=100, seed=42)
    assert len(tokens) == 100
    assert all(isinstance(t, str) and len(t) > 0 for t in tokens)


def test_seraphinianus_word_image():
    img = SeraphinianusEngine.generate_serafini_word_image(num_primitives=4, has_capital=True)
    assert img.size == (240, 80)
