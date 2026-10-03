"""
Gordon Rugg's Cardan Grid / Renaissance Combinatorial Table Generator.
Simulates a mechanical cardan grille moved across a fixed syllable table.
"""

import random
from typing import List, Optional


class RuggCardanGenerator:
    """
    Implements Gordon Rugg's Cardan grille hypothesis:
    A table of prefixes, roots, and suffixes is read through moving window holes,
    mechanically producing pseudo-words that follow Zipf's law and rigid morphology.
    """

    def __init__(self, table_size: int = 10, num_holes: int = 3, seed: Optional[int] = 42):
        self.table_size = table_size
        self.num_holes = num_holes
        if seed is not None:
            random.seed(seed)

        # Build Renaissance combinatorial syllable table
        self.prefix_table = [["qo", "ch", "sh", "d", "y", "ol", "or", "ot", "s", "t"][i % 10] for i in range(table_size * table_size)]
        self.root_table = [["k", "t", "e", "ee", "ai", "ar", "al", "ey", "am", "or"][i % 10] for i in range(table_size * table_size)]
        self.suffix_table = [["edy", "ain", "ar", "dy", "y", "am", "or", "hy", "ol", "in"][i % 10] for i in range(table_size * table_size)]

        random.shuffle(self.prefix_table)
        random.shuffle(self.root_table)
        random.shuffle(self.suffix_table)

    def generate(self, num_words: int = 1000) -> List[str]:
        """
        Generates pseudo-words by shifting the Cardan grille across the table.
        """
        words = []
        grid_pos_x = 0
        grid_pos_y = 0

        for _ in range(num_words):
            # Read cells through grille
            p_idx = (grid_pos_y * self.table_size + grid_pos_x) % len(self.prefix_table)
            r_idx = ((grid_pos_y + 1) * self.table_size + (grid_pos_x + 2)) % len(self.root_table)
            s_idx = ((grid_pos_y + 2) * self.table_size + (grid_pos_x + 1)) % len(self.suffix_table)

            p = self.prefix_table[p_idx] if random.random() > 0.25 else ""
            r = self.root_table[r_idx]
            s = self.suffix_table[s_idx] if random.random() > 0.25 else ""

            word = f"{p}{r}{s}"
            words.append(word)

            # Step the grille
            grid_pos_x = (grid_pos_x + 1) % self.table_size
            if grid_pos_x == 0:
                grid_pos_y = (grid_pos_y + 1) % self.table_size

        return words
