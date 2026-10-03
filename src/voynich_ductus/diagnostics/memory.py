"""
Detrended Fluctuation Analysis (DFA) and Hurst Exponent estimation for long-range memory.
"""

from typing import List, Union, Dict
import numpy as np


class LongRangeMemoryAnalyzer:
    """
    Evaluates persistent memory across token/word sequences using Detrended Fluctuation Analysis (DFA).
    - H ≈ 0.5: Uncorrelated noise / short-term Markov process without long memory.
    - H > 0.65 - 0.80: Persistent long-range semantic correlations (hallmark of natural language).
    """

    @staticmethod
    def dfa(series: np.ndarray, min_scale: int = 4, max_scale: int = None, n_scales: int = 16) -> float:
        """
        Calculates DFA scaling exponent (Hurst parameter proxy).
        """
        N = len(series)
        if N < 32:
            return 0.5  # Not enough data for reliable estimation

        if max_scale is None:
            max_scale = N // 4

        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
        scales = scales[scales >= min_scale]

        # Integrate series: Y(k) = sum_{i=1}^k (x_i - mean)
        y = np.cumsum(series - np.mean(series))

        fluctuations = []
        valid_scales = []

        for s in scales:
            num_segments = N // s
            if num_segments < 2:
                continue

            rms_list = []
            for v in range(num_segments):
                idx = np.arange(v * s, (v + 1) * s)
                segment = y[idx]
                t = np.arange(s)
                # Linear trend fit
                poly = np.polyfit(t, segment, 1)
                trend = np.polyval(poly, t)
                rms = np.sqrt(np.mean((segment - trend) ** 2))
                rms_list.append(rms)

            if rms_list:
                fluctuations.append(np.mean(rms_list))
                valid_scales.append(s)

        if len(valid_scales) < 3:
            return 0.5

        # Log-log linear regression: log(F(s)) = H * log(s) + C
        log_s = np.log(valid_scales)
        log_f = np.log(fluctuations)

        # Fit line
        slope, _ = np.polyfit(log_s, log_f, 1)
        return float(np.clip(slope, 0.0, 1.5))

    @classmethod
    def word_recurrence_series(cls, words: List[str]) -> np.ndarray:
        """
        Converts sequence of words into a word-recurrence distance time series.
        For each word at index i, value is distance to previous occurrence of the same word.
        """
        last_seen = {}
        series = []
        for i, w in enumerate(words):
            if w in last_seen:
                dist = i - last_seen[w]
            else:
                dist = len(words) // 2
            last_seen[w] = i
            series.append(float(dist))

        return np.array(series, dtype=float)

    @classmethod
    def analyze_memory(cls, words: List[str]) -> Dict[str, float]:
        """
        Runs DFA on token recurrence time series.
        """
        series = cls.word_recurrence_series(words)
        hurst_h = cls.dfa(series)

        # Natural language classification check
        has_natural_memory = bool(hurst_h >= 0.62)

        return {
            "hurst_exponent": round(hurst_h, 4),
            "is_long_range_correlated": has_natural_memory,
            "sample_length": len(words)
        }
