"""
Baseline generators: Natural 15th-century Latin herbal corpus, uniform noise, and Markov babbler.
"""

import random
from collections import defaultdict, Counter
from typing import List, Optional


class BaselineGenerator:
    """
    Provides ground truth natural language corpora and stochastic noise baselines.
    """

    # Authentic 15th-century Latin herbal / medical botanical text excerpts (Pseudo-Apuleius / Circa Instans style)
    LATIN_HERBAL_CORPUS = """
    Artemisia herba est calida et sicca in secundo gradu cuius virtus est aperitiva et confortativa.
    Radix eius decocta in vino albo valet contra dolores stomachi et matricis fluxum provocat.
    Folia trita et cum oleo rosaceo applicata capiti dolorem leniunt et somnum inducunt.
    Betonica nascitur in locis montosis et silvaticis colligitur mense augusto ante ortum solis.
    Succus eius bibitus cum aqua calida prodest hydropicis et visum clarificat.
    Mandragora est herba frigida et humida cuius radix habet similitudinem formae humanae.
    Cortex radicis decoctus bibitur ante incisionem membrorum ut sopor inducatur nec dolor sentiatur.
    Absinthium calefacit stomachum et confortat digestionem vermes ventris expellit et febres tertianas sanat.
    Plantago habet virtutem adstringentem et refrigerativum folia eius vulneribus recentibus utiliter imponuntur.
    Ruta hortensis comesta visum acuit et venenum resistit semen eius in aqua coctum podagricis succurrit.
    Salvia confortat nervos et paralyticis prodest folia eius decocta gargarizata gingivas confortant.
    Hyoscyamus est herba venenosa sed semina eius cum melle imposita oculorum calores extinguunt.
    Chelidonia succum croceum habet qui oculos clarificat et lepras cutis extergit.
    """

    @classmethod
    def get_natural_latin_sample(cls, num_words: int = 1000) -> List[str]:
        """Returns tokenized authentic 15th-century medieval Latin corpus repeated/expanded to target word length."""
        raw_words = [w.lower().strip(".,;:!?") for w in cls.LATIN_HERBAL_CORPUS.split() if w.strip(".,;:!?")]
        out = []
        while len(out) < num_words:
            out.extend(raw_words)
        return out[:num_words]

    @staticmethod
    def get_uniform_random_gibberish(num_words: int = 1000, avg_len: int = 5, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> List[str]:
        """Generates memoryless random noise words with uniform character distribution."""
        words = []
        for _ in range(num_words):
            w_len = max(2, int(random.gauss(avg_len, 1.5)))
            w = "".join(random.choice(alphabet) for _ in range(w_len))
            words.append(w)
        return words

    @classmethod
    def get_markov_babbler(cls, source_words: List[str], order: int = 1, num_words: int = 1000) -> List[str]:
        """Trains an order-N character or word Markov babbler and generates a sequence."""
        transitions = defaultdict(Counter)
        for i in range(len(source_words) - order):
            ctx = tuple(source_words[i:i + order])
            nxt = source_words[i + order]
            transitions[ctx][nxt] += 1

        if not transitions:
            return source_words[:num_words]

        curr = list(transitions.keys())[0]
        out = list(curr)

        for _ in range(num_words - order):
            nxt_counts = transitions[curr]
            if not nxt_counts:
                curr = random.choice(list(transitions.keys()))
                nxt_counts = transitions[curr]

            candidates = list(nxt_counts.keys())
            weights = list(nxt_counts.values())
            nxt_word = random.choices(candidates, weights=weights, k=1)[0]
            out.append(nxt_word)
            curr = tuple(out[-order:])

        return out[:num_words]
