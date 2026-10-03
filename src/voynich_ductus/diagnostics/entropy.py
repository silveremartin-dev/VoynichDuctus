"""
Shannon Entropy & Conditional Entropy Calculation for character/glyph and word distributions.
"""

from collections import Counter
from typing import List, Dict, Any, Union
import numpy as np


class ShannonEntropyAnalyzer:
    """
    Measures 1st order Shannon entropy H1 and 2nd order conditional entropy H2.
    H1 = - sum( p(x) * log2(p(x)) )
    H2 = H(X2 | X1) = H(X1, X2) - H(X1)
    """

    @staticmethod
    def shannon_entropy(sequence: Union[str, List[Any]]) -> float:
        """Computes H1 in bits per symbol."""
        if not sequence:
            return 0.0

        counts = Counter(sequence)
        total = float(len(sequence))
        entropy = 0.0

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    @staticmethod
    def conditional_entropy_order2(sequence: Union[str, List[Any]]) -> float:
        """
        Computes conditional entropy H(X_t | X_{t-1}) in bits per symbol.
        H(X_t | X_{t-1}) = - sum_{x_{t-1}, x_t} p(x_{t-1}, x_t) * log2( p(x_t | x_{t-1}) )
        """
        if len(sequence) < 2:
            return 0.0

        unigrams = Counter(sequence)
        bigrams = Counter(zip(sequence[:-1], sequence[1:]))

        total_bigrams = float(len(sequence) - 1)
        cond_entropy = 0.0

        for (u, v), bigram_count in bigrams.items():
            p_joint = bigram_count / total_bigrams
            p_cond = bigram_count / unigrams[u]
            if p_cond > 0:
                cond_entropy -= p_joint * np.log2(p_cond)

        return float(cond_entropy)

    @classmethod
    def analyze_text(cls, text_or_tokens: Union[str, List[str]]) -> Dict[str, float]:
        """
        Calculates character-level and word-level entropy metrics.
        """
        if isinstance(text_or_tokens, str):
            chars = list(text_or_tokens)
            words = text_or_tokens.split()
        else:
            # Flatten characters
            chars = list("".join(text_or_tokens))
            words = text_or_tokens

        h1_char = cls.shannon_entropy(chars)
        h2_char = cls.conditional_entropy_order2(chars)
        h1_word = cls.shannon_entropy(words)
        h2_word = cls.conditional_entropy_order2(words)

        # Entropy reduction ratio (drop in uncertainty)
        h2_drop_ratio = (h1_char - h2_char) / (h1_char + 1e-6)

        return {
            "h1_char_bits": round(h1_char, 4),
            "h2_cond_char_bits": round(h2_char, 4),
            "h2_drop_ratio": round(h2_drop_ratio, 4),
            "h1_word_bits": round(h1_word, 4),
            "h2_cond_word_bits": round(h2_word, 4),
            "vocab_size_char": len(set(chars)),
            "vocab_size_word": len(set(words))
        }
