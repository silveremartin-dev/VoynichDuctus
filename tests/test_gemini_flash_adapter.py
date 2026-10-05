"""
Unit tests for GeminiPaleographyAdapter multimodal vision interface.
"""

from PIL import Image
import numpy as np
import pytest
from voynich_ductus.ai.gemini_flash_adapter import GeminiPaleographyAdapter


def test_gemini_adapter_glyph_inspection():
    adapter = GeminiPaleographyAdapter()
    
    # Create test glyph image (30x30 black stroke on white parchment)
    img = Image.new("RGB", (30, 30), color=(240, 230, 210))
    for i in range(5, 25):
        img.putpixel((15, i), (30, 20, 10))

    res = adapter.inspect_glyph_kinematics(img)
    assert "stroke_count" in res
    assert "entry_tangent_deg" in res
    assert "exit_tangent_deg" in res
    assert "ductus_confidence_pct" in res
    assert res["stroke_count"] >= 1


def test_gemini_adapter_scribal_comparison():
    adapter = GeminiPaleographyAdapter()
    
    img1 = Image.new("RGB", (40, 40), color=(240, 230, 210))
    img2 = Image.new("RGB", (40, 40), color=(235, 225, 205))

    res = adapter.compare_scribal_hands(img1, img2)
    assert "same_scribe_probability_pct" in res
    assert "slant_difference_deg" in res
    assert "nib_width_match" in res
    assert 0 <= res["same_scribe_probability_pct"] <= 100


def test_gemini_adapter_word_decomposition():
    adapter = GeminiPaleographyAdapter()
    word_img = Image.new("RGB", (90, 30), color=(240, 230, 210))

    spans = adapter.decompose_word_into_glyphs(word_img)
    assert len(spans) >= 1
    assert "sub_glyph_index" in spans[0]
    assert "bbox_local" in spans[0]
