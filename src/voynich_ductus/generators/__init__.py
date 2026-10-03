"""
Adversarial and baseline synthetic text generators.
Provides ground-truth controls for testing computational paleography hypotheses:
- Torsten Timm's Self-Citation Algorithm
- Gordon Rugg's Cardan Grid / Syllable Wheel Generator
- Natural Medieval Latin and Random Baselines
"""

from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator
from voynich_ductus.generators.baselines import BaselineGenerator

__all__ = [
    "TimmSelfCitationGenerator",
    "RuggCardanGenerator",
    "BaselineGenerator",
]
