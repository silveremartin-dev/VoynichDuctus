"""
Converts ordered glyph clusters and word structures into token streams and symbolic transcriptions.
"""

from typing import List, Dict, Any
import numpy as np


class StrokeTokenizer:
    """
    Translates clustered strokes and segmented words into clean token sequences.
    """

    def __init__(self, prefix: str = "G"):
        self.prefix = prefix

    def format_glyph_id(self, cluster_idx: int) -> str:
        """Formats cluster integer into a standardized glyph token e.g. G042, or ? for noise."""
        if cluster_idx < 0:
            return f"{self.prefix}_UNK"
        return f"{self.prefix}{cluster_idx:02d}"

    def tokenize_word(self, stroke_cluster_ids: List[int]) -> str:
        """
        Combines ordered stroke cluster IDs into a single word token.
        e.g., [4, 12, 1] -> 'G04-G12-G01'
        """
        if not stroke_cluster_ids:
            return ""
        return "-".join(self.format_glyph_id(cid) for cid in stroke_cluster_ids)

    def tokenize_document(self, document_words: List[List[int]]) -> List[str]:
        """
        Tokenizes a full document (list of words, each containing ordered stroke cluster IDs).
        """
        tokens = []
        for word_strokes in document_words:
            w_str = self.tokenize_word(word_strokes)
            if w_str:
                tokens.append(w_str)
        return tokens
