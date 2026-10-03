"""
Codex Seraphinianus generator and synthetic stroke sampler.
Models the asemic, looped cursive morphology created by Luigi Serafini (1981).
"""

import random
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw

from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter


class SeraphinianusEngine:
    """
    Simulates and extracts vector ductus from the asemic cursive script of Codex Seraphinianus.
    Characteristics of Serafini's script:
    - Flowing, looped modern cursive with high ascenders and descenders.
    - Large decorative initial capitals with spirals/knots.
    - Base-21 numeral system and repetitive morphological roots.
    """

    # Serafinian cursive primitives
    @staticmethod
    def draw_serafini_loop_ascender(draw: ImageDraw.Draw, x: int, y: int):
        """High looped ascender (characteristic of Serafini's 'l'/'h' variants)."""
        draw.arc([x, y - 15, x + 15, y + 25], start=180, end=0, fill="#1a1a2e", width=3)
        draw.line([x + 15, y - 5, x + 5, y + 25], fill="#1a1a2e", width=3)
        draw.arc([x + 5, y + 15, x + 20, y + 27], start=270, end=90, fill="#1a1a2e", width=3)

    @staticmethod
    def draw_serafini_spiral_initial(draw: ImageDraw.Draw, x: int, y: int):
        """Flourished spiral capital initial."""
        draw.arc([x, y - 10, x + 25, y + 15], start=0, end=270, fill="#1a1a2e", width=3)
        draw.arc([x + 5, y - 5, x + 20, y + 10], start=180, end=360, fill="#1a1a2e", width=2)
        draw.line([x + 12, y + 12, x + 12, y + 30], fill="#1a1a2e", width=3)
        draw.arc([x + 12, y + 20, x + 28, y + 32], start=270, end=90, fill="#1a1a2e", width=3)

    @staticmethod
    def draw_serafini_undulating_body(draw: ImageDraw.Draw, x: int, y: int):
        """Undulating wave / 'm'/'n' variant with upper hook."""
        draw.arc([x, y + 8, x + 12, y + 26], start=180, end=0, fill="#1a1a2e", width=3)
        draw.line([x + 12, y + 17, x + 12, y + 26], fill="#1a1a2e", width=3)
        draw.arc([x + 12, y + 8, x + 24, y + 26], start=180, end=0, fill="#1a1a2e", width=3)
        draw.line([x + 24, y + 17, x + 24, y + 26], fill="#1a1a2e", width=3)

    @staticmethod
    def draw_serafini_descender_hook(draw: ImageDraw.Draw, x: int, y: int):
        """Loop descending below the baseline with backwards return."""
        draw.ellipse([x + 2, y + 8, x + 16, y + 22], outline="#1a1a2e", width=3)
        draw.line([x + 16, y + 15, x + 16, y + 38], fill="#1a1a2e", width=3)
        draw.arc([x + 4, y + 28, x + 16, y + 42], start=0, end=180, fill="#1a1a2e", width=3)

    @staticmethod
    def draw_serafini_eyelet_tie(draw: ImageDraw.Draw, x: int, y: int):
        """Small floating loop ligature."""
        draw.ellipse([x + 2, y + 12, x + 14, y + 24], outline="#1a1a2e", width=3)
        draw.line([x + 14, y + 18, x + 26, y + 18], fill="#1a1a2e", width=2)

    @classmethod
    def generate_serafini_word_image(cls, num_primitives: int = 4, has_capital: bool = False, width: int = 240, height: int = 80) -> Image.Image:
        """Generates a synthetic word in Serafini's cursive style."""
        img = Image.new("RGB", (width, height), color="#fcf9ee")
        draw = ImageDraw.Draw(img)

        primitives = [
            cls.draw_serafini_loop_ascender,
            cls.draw_serafini_undulating_body,
            cls.draw_serafini_descender_hook,
            cls.draw_serafini_eyelet_tie
        ]

        curr_x = 15
        if has_capital:
            cls.draw_serafini_spiral_initial(draw, curr_x, 25)
            curr_x += 35

        for _ in range(num_primitives):
            fn = random.choice(primitives)
            fn(draw, curr_x, 25)
            curr_x += 24

        return img

    @classmethod
    def generate_serafini_text_tokens(cls, num_words: int = 1000, seed: int = 42) -> List[str]:
        """
        Generates symbolic tokens simulating the morphological patterns of the Codex Seraphinianus.
        Serafini's text exhibits a high rate of prefix/root repetition with fluid asemic suffixes.
        """
        random.seed(seed)
        capitals = ["Ser_A", "Ser_B", "Ser_C", "Ser_D", "Ser_E", "Ser_F"]
        roots = ["lumu", "siri", "fal", "kron", "zep", "nari", "tolo", "vari", "mox", "heli"]
        suffixes = ["ina", "onis", "orum", "atis", "el", "ium", "as", "ux"]

        tokens = []
        for i in range(num_words):
            prefix = random.choice(capitals) if (i % 8 == 0) else ""
            root = random.choice(roots)
            suffix = random.choice(suffixes) if random.random() > 0.3 else ""
            tokens.append(f"{prefix}{root}{suffix}".strip())

        return tokens
