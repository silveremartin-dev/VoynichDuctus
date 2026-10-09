"""
Standard Corpus Correspondence Matching Engine.
Estimates the mapping between unsupervised induced glyph archetypes (G01, G02, ...)
and established historical transliteration systems (EVA, Currier, Takahashi, Serafinian typology)
using invariant topological and morphological paleographic signatures, and generates canonical vector SVGs.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class CorpusCorrespondenceMatcher:
    """
    Estimates the correspondence between emergent unsupervised glyph clusters
    and historical standard paleographic transcription corpora (EVA, Currier, Takahashi).
    """

    # High-Fidelity Vector SVGs for Standard Reference Corpora
    EVA_REFERENCE_SVGS: Dict[str, str] = {
        "o": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="50" cy="65" rx="26" ry="32" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="50" cy="33" r="3.5" fill="#10b981"/>
</svg>""",
        "a": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="38" cy="66" rx="22" ry="28" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="60" y1="36" x2="60" y2="96" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="60" cy="36" r="3.5" fill="#10b981"/>
</svg>""",
        "y": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="28" y1="42" x2="52" y2="68" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 46 38 Q 58 66 62 76 Q 66 98 28 114" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="28" cy="42" r="3.5" fill="#10b981"/>
</svg>""",
        "d": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="36" cy="74" rx="20" ry="24" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="56" y1="16" x2="56" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="56" cy="16" r="3.5" fill="#10b981"/>
</svg>""",
        "e": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 68 52 C 68 36 30 36 30 64 C 30 88 72 88 72 74" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="68" cy="52" r="3.5" fill="#10b981"/>
</svg>""",
        "s": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 66 40 C 42 34 30 52 50 66 C 70 80 62 98 34 92" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="66" cy="40" r="3.5" fill="#10b981"/>
</svg>""",
        "l": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 44 98 L 44 32 C 44 14 68 14 68 32 C 68 52 44 58 44 58" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="44" cy="98" r="3.5" fill="#10b981"/>
</svg>""",
        "r": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="36" y1="42" x2="36" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 36 56 Q 56 38 74 48" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="36" cy="42" r="3.5" fill="#10b981"/>
</svg>""",
        "ch": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 36 48 C 16 48 16 82 36 82 L 62 82" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="84" y1="24" x2="84" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 60 82 C 62 62 82 62 84 82" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="36" cy="48" r="3.5" fill="#10b981"/>
</svg>""",
        "sh": """<svg viewBox="0 0 140 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 28 48 C 12 48 12 82 28 82 L 48 82" fill="none" stroke="#38bdf8" stroke-width="7" stroke-linecap="round"/>
  <path d="M 48 48 C 34 48 34 82 48 82 L 78 82" fill="none" stroke="#38bdf8" stroke-width="7" stroke-linecap="round"/>
  <line x1="104" y1="24" x2="104" y2="98" stroke="#38bdf8" stroke-width="7" stroke-linecap="round"/>
  <circle cx="28" cy="48" r="3.5" fill="#10b981"/>
</svg>""",
        "k": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="22" y1="26" x2="98" y2="26" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="36" y1="26" x2="36" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="84" y1="26" x2="84" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 36 60 C 50 44 70 44 84 60" fill="none" stroke="#38bdf8" stroke-width="6"/>
  <circle cx="22" cy="26" r="3.5" fill="#10b981"/>
</svg>""",
        "t": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="60" y1="16" x2="60" y2="102" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="26" y1="50" x2="94" y2="50" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 40 76 C 50 62 70 62 80 76" fill="none" stroke="#38bdf8" stroke-width="6"/>
  <circle cx="60" cy="16" r="3.5" fill="#10b981"/>
</svg>""",
        "p": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="22" y1="26" x2="98" y2="26" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="36" y1="26" x2="36" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="84" y1="26" x2="84" y2="98" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="60" cy="26" r="10" fill="none" stroke="#38bdf8" stroke-width="5"/>
  <circle cx="22" cy="26" r="3.5" fill="#10b981"/>
</svg>""",
        "f": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="60" y1="16" x2="60" y2="102" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <line x1="26" y1="46" x2="94" y2="46" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 60 26 C 76 16 86 34 60 46" fill="none" stroke="#38bdf8" stroke-width="6"/>
  <circle cx="60" cy="16" r="3.5" fill="#10b981"/>
</svg>""",
        "q": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="44" cy="50" rx="22" ry="22" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <path d="M 66 50 L 66 84 Q 66 112 32 112" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="44" cy="28" r="3.5" fill="#10b981"/>
</svg>""",
        "m": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 24 48 C 24 38 44 38 44 56 L 44 96 M 44 48 C 44 38 64 38 64 56 L 64 96 M 64 48 C 64 38 84 38 84 56 L 84 96" fill="none" stroke="#38bdf8" stroke-width="7" stroke-linecap="round"/>
  <circle cx="24" cy="48" r="3.5" fill="#10b981"/>
</svg>""",
        "n": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 28 48 C 28 38 50 38 50 56 L 50 96 M 50 48 C 50 38 72 38 72 56 L 72 96" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="28" cy="48" r="3.5" fill="#10b981"/>
</svg>""",
        "i": """<svg viewBox="0 0 80 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="40" y1="42" x2="40" y2="96" stroke="#38bdf8" stroke-width="8" stroke-linecap="round"/>
  <circle cx="40" cy="42" r="3.5" fill="#10b981"/>
</svg>""",
    }

    SERAFINI_REFERENCE_SVGS: Dict[str, str] = {
        "S-L01": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="36" cy="68" rx="20" ry="26" fill="none" stroke="#c084fc" stroke-width="8"/>
  <line x1="56" y1="18" x2="56" y2="98" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <circle cx="36" cy="42" r="3.5" fill="#10b981"/>
</svg>""",
        "S-C02": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="65" x2="100" y2="65" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <path d="M 40 65 C 40 42 80 42 80 65" fill="none" stroke="#c084fc" stroke-width="8"/>
</svg>""",
        "S-H03": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 28 85 C 28 50 56 50 56 85 C 56 50 84 50 84 85" fill="none" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
</svg>""",
        "S-W04": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="35" y1="35" x2="60" y2="65" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <path d="M 60 65 Q 75 90 50 105 Q 30 115 15 105" fill="none" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
</svg>""",
        "S-B05": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="25" y1="35" x2="95" y2="35" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <line x1="60" y1="35" x2="60" y2="95" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
</svg>""",
        "S-K06": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="65" r="28" fill="none" stroke="#c084fc" stroke-width="8"/>
  <line x1="20" y1="65" x2="80" y2="65" stroke="#c084fc" stroke-width="8"/>
</svg>""",
        "S-S07": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M 65 25 C 30 15 20 45 50 65 C 80 85 70 115 35 105" fill="none" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
</svg>""",
        "S-M08": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="35" y1="40" x2="35" y2="95" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <line x1="65" y1="40" x2="65" y2="95" stroke="#c084fc" stroke-width="8" stroke-linecap="round"/>
  <line x1="30" y1="40" x2="70" y2="40" stroke="#c084fc" stroke-width="6"/>
</svg>""",
        "S-T09": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <line x1="25" y1="40" x2="95" y2="40" stroke="#c084fc" stroke-width="7" stroke-linecap="round"/>
  <line x1="35" y1="40" x2="35" y2="95" stroke="#c084fc" stroke-width="7" stroke-linecap="round"/>
  <line x1="60" y1="40" x2="60" y2="95" stroke="#c084fc" stroke-width="7" stroke-linecap="round"/>
  <line x1="85" y1="40" x2="85" y2="95" stroke="#c084fc" stroke-width="7" stroke-linecap="round"/>
</svg>""",
        "S-P10": """<svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="65" r="14" fill="#c084fc"/>
  <path d="M 50 51 C 65 51 75 65 70 80 C 65 92 45 92 40 85" fill="none" stroke="#c084fc" stroke-width="5" stroke-linecap="round"/>
</svg>""",
    }

    # Voynich Standard Paleographic Profiles with Strict Topological Constraints
    VOYNICH_STANDARD_PROFILES = [
        {
            "eva": "o",
            "currier": "O",
            "v101": "o",
            "voynichese": "o",
            "name": "Single Loop Oval",
            "category": "Oval / Minim",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": True,
            "aspect_ratio": 0.85,
            "relative_freq_rank": 1,
            "description": "Closed circular or oval minim loop, highest frequency base vowel"
        },
        {
            "eva": "a",
            "currier": "A",
            "v101": "a",
            "voynichese": "a",
            "name": "Loop with Vertical Stem",
            "category": "Oval / Minim",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": True,
            "aspect_ratio": 0.95,
            "relative_freq_rank": 3,
            "description": "Minim loop with attached downward right vertical stroke"
        },
        {
            "eva": "y",
            "currier": "Y",
            "v101": "y",
            "voynichese": "y",
            "name": "Terminal Tail Descender",
            "category": "Descender",
            "strokes": 2.0,
            "min_strokes": 1,
            "max_strokes": 3,
            "has_loop": False,
            "aspect_ratio": 0.70,
            "relative_freq_rank": 2,
            "description": "Rightward stem with sweeping lower-left descender stroke"
        },
        {
            "eva": "d",
            "currier": "D",
            "v101": "d",
            "voynichese": "d",
            "name": "Benched Loop Ascender",
            "category": "Ascender",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": True,
            "aspect_ratio": 0.65,
            "relative_freq_rank": 4,
            "description": "Minim bowl with tall straight upright vertical ascender shaft"
        },
        {
            "eva": "e",
            "currier": "E",
            "v101": "e",
            "voynichese": "e",
            "name": "C-Curve Minim",
            "category": "Minim",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 0.75,
            "relative_freq_rank": 5,
            "description": "Open crescent or short horizontal stroke, frequent in sequences (ee, eee)"
        },
        {
            "eva": "s",
            "currier": "S",
            "v101": "s",
            "voynichese": "s",
            "name": "S-Curve Terminal",
            "category": "Terminal",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 0.60,
            "relative_freq_rank": 8,
            "description": "S-shaped curved stroke, frequently in word-final position"
        },
        {
            "eva": "l",
            "currier": "L",
            "v101": "l",
            "voynichese": "l",
            "name": "Looping Tall Ascender",
            "category": "Ascender",
            "strokes": 2.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": True,
            "aspect_ratio": 0.55,
            "relative_freq_rank": 6,
            "description": "Tall ascender with top loop or benched ligature"
        },
        {
            "eva": "r",
            "currier": "R",
            "v101": "r",
            "voynichese": "r",
            "name": "Shouldered Arch / Hook",
            "category": "Minim",
            "strokes": 2.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 0.78,
            "relative_freq_rank": 7,
            "description": "Short vertical stroke with rightward arched shoulder"
        },
        {
            "eva": "ch",
            "currier": "C",
            "v101": "c8",
            "voynichese": "ch",
            "name": "Single Correa Ligature",
            "category": "Ligature / Correa",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": False,
            "aspect_ratio": 1.25,
            "relative_freq_rank": 9,
            "description": "Horizontal bench 'c' connecting to a vertical upright 'h' stem"
        },
        {
            "eva": "sh",
            "currier": "H",
            "v101": "c9",
            "voynichese": "sh",
            "name": "Double-Loop Correa Ligature",
            "category": "Ligature / Correa",
            "strokes": 3.0,
            "min_strokes": 3,
            "max_strokes": 4,
            "has_loop": False,
            "aspect_ratio": 1.45,
            "relative_freq_rank": 10,
            "description": "Double-nested correa arches with an upright right ascender stem"
        },
        {
            "eva": "k",
            "currier": "K",
            "v101": "t",
            "voynichese": "k",
            "name": "Benched Gallows (High Crossbar)",
            "category": "Gallows",
            "strokes": 3.0,
            "min_strokes": 3,
            "max_strokes": 5,
            "has_loop": False,
            "aspect_ratio": 1.10,
            "relative_freq_rank": 11,
            "description": "Tall benched gallows glyph with horizontal overhead beam and twin legs"
        },
        {
            "eva": "t",
            "currier": "T",
            "v101": "k",
            "voynichese": "t",
            "name": "Pedestal Crossed Gallows",
            "category": "Gallows",
            "strokes": 3.0,
            "min_strokes": 2,
            "max_strokes": 4,
            "has_loop": False,
            "aspect_ratio": 0.95,
            "relative_freq_rank": 12,
            "description": "Central tall ascender staff with intersected mid-level crossbar"
        },
        {
            "eva": "p",
            "currier": "P",
            "v101": "p",
            "voynichese": "p",
            "name": "Double-Benched Split Gallows",
            "category": "Gallows",
            "strokes": 4.0,
            "min_strokes": 3,
            "max_strokes": 6,
            "has_loop": True,
            "aspect_ratio": 1.30,
            "relative_freq_rank": 14,
            "description": "Elaborate gallows with twin loops and center cross ties"
        },
        {
            "eva": "f",
            "currier": "F",
            "v101": "f",
            "voynichese": "f",
            "name": "Looped Flag Gallows",
            "category": "Gallows",
            "strokes": 3.0,
            "min_strokes": 3,
            "max_strokes": 5,
            "has_loop": True,
            "aspect_ratio": 1.20,
            "relative_freq_rank": 15,
            "description": "Gallows structure with looped flag ornament on horizontal crossbar"
        },
        {
            "eva": "q",
            "currier": "Q",
            "v101": "4",
            "voynichese": "q",
            "name": "Initial 4-Like Descender Hook",
            "category": "Initial / Descender",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": True,
            "aspect_ratio": 0.85,
            "relative_freq_rank": 13,
            "description": "Word-initial closed loop with descending leftward stem"
        },
        {
            "eva": "m",
            "currier": "M",
            "v101": "m",
            "voynichese": "m",
            "name": "Triple Minim Wave",
            "category": "Minim Sequence",
            "strokes": 3.0,
            "min_strokes": 3,
            "max_strokes": 4,
            "has_loop": False,
            "aspect_ratio": 1.60,
            "relative_freq_rank": 16,
            "description": "Three consecutive downstrokes with upper rounded arches"
        },
        {
            "eva": "n",
            "currier": "N",
            "v101": "n",
            "voynichese": "n",
            "name": "Double Minim Wave",
            "category": "Minim Sequence",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": False,
            "aspect_ratio": 1.20,
            "relative_freq_rank": 17,
            "description": "Twin vertical minims with linking upper arch"
        },
        {
            "eva": "i",
            "currier": "I",
            "v101": "i",
            "voynichese": "i",
            "name": "Single Vertical Minim",
            "category": "Minim",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 1,
            "has_loop": False,
            "aspect_ratio": 0.40,
            "relative_freq_rank": 18,
            "description": "Single upright vertical stroke"
        },
    ]

    SERAFINI_STANDARD_PROFILES = [
        {
            "code": "S-L01",
            "name": "Loop-Stem Ascender",
            "deri": "A-Loop",
            "bulik": "α₁",
            "category": "Syllabic Initial",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": True,
            "aspect_ratio": 0.60,
            "description": "Initial left-handed oval loop with a high right ascender staff"
        },
        {
            "code": "S-C02",
            "name": "Crescent-Bar Connector",
            "deri": "Arch-Cap",
            "bulik": "β₂",
            "category": "Medial Connector",
            "strokes": 2.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 1.10,
            "description": "Horizontal connecting bar carrying an open upward crescent"
        },
        {
            "code": "S-H03",
            "name": "Double-Hook Ligature",
            "deri": "Double-Arch",
            "bulik": "γ₃",
            "category": "Rhythmic Arch",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": False,
            "aspect_ratio": 1.35,
            "description": "Twin symmetrical curved hooks on a common baseline"
        },
        {
            "code": "S-W04",
            "name": "Wave-Descender",
            "deri": "Wave-Tail",
            "bulik": "δ₄",
            "category": "Terminal Descender",
            "strokes": 2.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 0.80,
            "description": "Undulating sinusoidal tail plunging below the baseline"
        },
        {
            "code": "S-B05",
            "name": "Benched Arch",
            "deri": "Cross-Beam",
            "bulik": "ε₅",
            "category": "Roof / Arch",
            "strokes": 2.0,
            "min_strokes": 2,
            "max_strokes": 3,
            "has_loop": False,
            "aspect_ratio": 1.15,
            "description": "Square-topped horizontal canopy with a single vertical leg"
        },
        {
            "code": "S-K06",
            "name": "Circle-Knot Glyphe",
            "deri": "O-Ring",
            "bulik": "ζ₆",
            "category": "Complex Loop",
            "strokes": 3.0,
            "min_strokes": 2,
            "max_strokes": 4,
            "has_loop": True,
            "aspect_ratio": 0.90,
            "description": "Closed circular center with crossing horizontal flourish"
        },
        {
            "code": "S-S07",
            "name": "Serpentine-S Ligature",
            "deri": "S-Cursive",
            "bulik": "η₇",
            "category": "Vertical Ligature",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": False,
            "aspect_ratio": 0.50,
            "description": "Elongated vertical s-curve spanning ascender and descender zones"
        },
        {
            "code": "S-M08",
            "name": "Twin-Tooth Comb",
            "deri": "Twin-Pole",
            "bulik": "θ₈",
            "category": "Minim Wave",
            "strokes": 3.0,
            "min_strokes": 2,
            "max_strokes": 4,
            "has_loop": False,
            "aspect_ratio": 1.05,
            "description": "Two parallel vertical teeth with an upper connecting cross-tie"
        },
        {
            "code": "S-T09",
            "name": "Triple-Crown Comb",
            "deri": "Trident-Base",
            "bulik": "ι₉",
            "category": "Comb / Crest",
            "strokes": 4.0,
            "min_strokes": 3,
            "max_strokes": 5,
            "has_loop": False,
            "aspect_ratio": 1.55,
            "description": "Three vertical strokes surmounted by a wavy horizontal ridge"
        },
        {
            "code": "S-P10",
            "name": "Spiral Terminal Dot",
            "deri": "P-Loop",
            "bulik": "κ₁₀",
            "category": "Punctuation / Marker",
            "strokes": 1.0,
            "min_strokes": 1,
            "max_strokes": 2,
            "has_loop": True,
            "aspect_ratio": 0.85,
            "description": "Curled terminal dot indicating clause or number boundary"
        },
    ]

    @classmethod
    def match_voynich_archetype(
        cls,
        archetype_id: str,
        mean_strokes: float,
        mean_width: float,
        mean_height: float,
        frequency_rank: int,
        cluster_glyphs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Estimates the closest standard corpus match (EVA / Currier / v101 / Voynichese) for an induced Voynich archetype
        using strict topological constraints (stroke count bounds, aspect ratio, and loop compatibility).
        """
        # Check if cluster has certified ground truth EVA characters from transcriptions
        if cluster_glyphs:
            eva_votes = {}
            for g in cluster_glyphs:
                ec = (g.get("eva_char") or "").strip().lower()
                if ec:
                    eva_votes[ec] = eva_votes.get(ec, 0) + 1
            if eva_votes:
                top_eva, vote_count = max(eva_votes.items(), key=lambda item: item[1])
                ratio = vote_count / max(1, len(cluster_glyphs))
                if ratio >= 0.25 or vote_count >= 3:
                    for prof in cls.VOYNICH_STANDARD_PROFILES:
                        if prof["eva"] == top_eva:
                            conf = min(0.98, max(0.75, ratio + 0.15))
                            return {
                                "system": "EVA / Currier / v101 / Voynichese",
                                "eva_equivalent": prof["eva"],
                                "currier_equivalent": prof["currier"],
                                "v101_equivalent": prof.get("v101", prof["eva"]),
                                "voynichese_equivalent": prof.get("voynichese", prof["eva"]),
                                "name": prof["name"],
                                "category": prof["category"],
                                "confidence_pct": round(conf * 100, 1),
                                "description": prof["description"],
                                "reference_svg": cls.EVA_REFERENCE_SVGS.get(prof["eva"], "")
                            }

        aspect_ratio = mean_width / max(1.0, mean_height)

        archetype_has_loop = False
        if cluster_glyphs:
            loop_scores = []
            for g in cluster_glyphs[:12]:
                ff = g.get("fill_factor", 0.25)
                if 0.18 <= ff <= 0.48 and len(g.get("strokes", [])) <= 3:
                    loop_scores.append(True)
                else:
                    loop_scores.append(False)
            archetype_has_loop = np.mean(loop_scores) >= 0.50 if loop_scores else False

        best_match = None
        best_distance = float("inf")

        for prof in cls.VOYNICH_STANDARD_PROFILES:
            if round(mean_strokes) < prof["min_strokes"]:
                stroke_pen = 4.0 * (prof["min_strokes"] - mean_strokes)
            elif round(mean_strokes) > prof["max_strokes"]:
                stroke_pen = 4.0 * (mean_strokes - prof["max_strokes"])
            else:
                stroke_pen = abs(prof["strokes"] - mean_strokes) * 1.2

            aspect_diff = abs(prof["aspect_ratio"] - aspect_ratio) * 1.8
            rank_diff = abs(prof["relative_freq_rank"] - frequency_rank) / 20.0

            loop_pen = 0.0
            if prof["has_loop"] and not archetype_has_loop and mean_strokes < 1.8:
                loop_pen = 3.5
            elif not prof["has_loop"] and archetype_has_loop and prof["strokes"] <= 1.2:
                loop_pen = 3.0

            total_dist = stroke_pen + aspect_diff + rank_diff + loop_pen

            if total_dist < best_distance:
                best_distance = total_dist
                best_match = prof

        if best_distance > 4.2:
            return {
                "system": "EVA / Currier / v101 / Voynichese",
                "eva_equivalent": "—",
                "currier_equivalent": "—",
                "v101_equivalent": "—",
                "voynichese_equivalent": "—",
                "name": "Distinct Emergent Ductus",
                "category": "Unclassified Scribal Form",
                "confidence_pct": round(max(30.0, 100.0 - best_distance * 15.0), 1),
                "description": "Emergent scribal grapheme with distinct topology; no isomorphic match in EVA scheme (anti-hallucination safeguard).",
                "reference_svg": ""
            }

        confidence = max(0.55, min(0.96, 1.0 - (best_distance * 0.16)))

        eva_char = best_match["eva"] if best_match else "o"
        ref_svg = cls.EVA_REFERENCE_SVGS.get(eva_char, "")

        return {
            "system": "EVA / Currier / v101 / Voynichese",
            "eva_equivalent": best_match["eva"],
            "currier_equivalent": best_match["currier"],
            "v101_equivalent": best_match.get("v101", best_match["eva"]),
            "voynichese_equivalent": best_match.get("voynichese", best_match["eva"]),
            "name": best_match["name"],
            "category": best_match["category"],
            "confidence_pct": round(confidence * 100, 1),
            "description": best_match["description"],
            "reference_svg": ref_svg
        }

    @classmethod
    def match_serafini_archetype(
        cls,
        archetype_id: str,
        mean_strokes: float,
        mean_width: float,
        mean_height: float,
        frequency_rank: int,
        cluster_glyphs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Estimates the closest Serafinian standard typology match using topological constraints.
        """
        aspect_ratio = mean_width / max(1.0, mean_height)

        archetype_has_loop = False
        if cluster_glyphs:
            loop_scores = []
            for g in cluster_glyphs[:12]:
                ff = g.get("fill_factor", 0.25)
                if 0.18 <= ff <= 0.48 and len(g.get("strokes", [])) <= 2:
                    loop_scores.append(True)
                else:
                    loop_scores.append(False)
            archetype_has_loop = np.mean(loop_scores) >= 0.50 if loop_scores else False

        best_match = None
        best_distance = float("inf")

        for prof in cls.SERAFINI_STANDARD_PROFILES:
            if round(mean_strokes) < prof["min_strokes"]:
                stroke_pen = 4.0 * (prof["min_strokes"] - mean_strokes)
            elif round(mean_strokes) > prof["max_strokes"]:
                stroke_pen = 4.0 * (mean_strokes - prof["max_strokes"])
            else:
                stroke_pen = abs(prof["strokes"] - mean_strokes) * 1.2

            aspect_diff = abs(prof["aspect_ratio"] - aspect_ratio) * 1.8

            loop_pen = 0.0
            if prof.get("has_loop", False) and not archetype_has_loop and mean_strokes < 1.8:
                loop_pen = 3.0
            elif not prof.get("has_loop", False) and archetype_has_loop and prof["strokes"] <= 1.5:
                loop_pen = 3.0

            total_dist = stroke_pen + aspect_diff + loop_pen

            if total_dist < best_distance:
                best_distance = total_dist
                best_match = prof

        if best_distance > 3.6:
            return {
                "system": "Serafini Typology (1981) / Deri (2015) / Bulik (2011)",
                "serafini_code": "S-EMERGENT",
                "deri_equivalent": "EMERGENT",
                "bulik_equivalent": "ω₀",
                "name": "Emergent Serafinian Ductus",
                "category": "Unclassified Cursive Glyph",
                "confidence_pct": round(max(30.0, 100.0 - best_distance * 15.0), 1),
                "description": "Distinct cursive character with no direct archetype equivalent in standard 10-class typology.",
                "reference_svg": ""
            }

        confidence = max(0.55, min(0.96, 1.0 - (best_distance * 0.18)))
        ref_svg = cls.SERAFINI_REFERENCE_SVGS.get(best_match["code"], "")

        return {
            "system": "Serafini Typology (1981) / Deri (2015) / Bulik (2011)",
            "serafini_code": best_match["code"],
            "deri_equivalent": best_match.get("deri", best_match["code"]),
            "bulik_equivalent": best_match.get("bulik", best_match["code"]),
            "name": best_match["name"],
            "category": best_match["category"],
            "confidence_pct": round(confidence * 100, 1),
            "description": best_match["description"],
            "reference_svg": ref_svg
        }
