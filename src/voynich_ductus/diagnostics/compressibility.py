"""
Lempel-Ziv & algorithmic compressibility metrics (Kolmogorov Complexity upper bounds).
"""

import zlib
import gzip
import bz2
import lzma
from typing import Dict, Union, List


class CompressibilityAnalyzer:
    """
    Measures algorithmic compressibility using standard lossless entropy coders (deflate, bz2, lzma).
    - Random Gibberish: Ratio ≈ 1.0 (incompressible)
    - Natural Medieval Languages (Latin, Middle English, Italian): Ratio ≈ 0.30 - 0.45
    - Cardan Grid / Simple Automaton: Ratio << 0.25 (hyper-compressible due to rigid state repetition)
    """

    @staticmethod
    def measure(text_or_tokens: Union[str, List[str]]) -> Dict[str, float]:
        if isinstance(text_or_tokens, list):
            raw_text = " ".join(text_or_tokens)
        else:
            raw_text = text_or_tokens

        raw_bytes = raw_text.encode("utf-8")
        raw_len = len(raw_bytes)

        if raw_len == 0:
            return {"raw_bytes": 0, "gzip_ratio": 1.0, "bz2_ratio": 1.0, "lzma_ratio": 1.0}

        gz_len = len(gzip.compress(raw_bytes, compresslevel=9))
        bz2_len = len(bz2.compress(raw_bytes, compresslevel=9))
        lzma_len = len(lzma.compress(raw_bytes))

        # Ratios (compressed_size / original_size)
        return {
            "raw_bytes": raw_len,
            "gzip_ratio": round(gz_len / raw_len, 4),
            "bz2_ratio": round(bz2_len / raw_len, 4),
            "lzma_ratio": round(lzma_len / raw_len, 4),
            "estimated_kolmogorov_bits_per_char": round((lzma_len * 8.0) / len(raw_text), 4) if len(raw_text) > 0 else 0.0
        }
