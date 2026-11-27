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

        if len(valid_scales) < 3 or float(np.std(series)) < 1e-9:
            return 0.5

        fluct_arr = np.array(fluctuations, dtype=float)
        scale_arr = np.array(valid_scales, dtype=float)
        pos_mask = fluct_arr > 1e-12
        if int(np.sum(pos_mask)) < 3:
            return 0.5

        # Log-log linear regression: log(F(s)) = H * log(s) + C
        log_s = np.log(scale_arr[pos_mask])
        log_f = np.log(fluct_arr[pos_mask])

        # Fit line
        slope, _ = np.polyfit(log_s, log_f, 1)
        if not np.isfinite(slope):
            return 0.5
        return float(np.clip(slope, 0.0, 1.5))

    @classmethod
    def word_recurrence_series(cls, words: List[str]) -> np.ndarray:
        """
        Converts sequence of words into a recurrence distance time series.
        When word repetition rate is sparse (< 20%, e.g. short folio slices or random noise),
        falls back to atomic glyph/symbol recurrence intervals so hapax warm-up plateaus
        cannot artificially inflate Hurst H or collapse variance to zero.
        """
        if not words:
            return np.array([], dtype=float)
        rep_ratio = 1.0 - (len(set(words)) / float(max(1, len(words))))
        if rep_ratio < 0.20:
            from voynich_ductus.diagnostics.entropy import ShannonEntropyAnalyzer
            units = ShannonEntropyAnalyzer.extract_atomic_symbols(words)
        else:
            units = words

        last_seen = {}
        series = []
        n_u = max(2, len(units))
        for i, u in enumerate(units):
            if u in last_seen:
                dist = i - last_seen[u]
            else:
                dist = max(1, min(i + 1, n_u // 8))
            last_seen[u] = i
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
