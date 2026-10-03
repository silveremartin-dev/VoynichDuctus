"""
Markov order evaluation, transition determinism, and finite-state automaton fit.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Union
import numpy as np


class MarkovAutomatonAnalyzer:
    """
    Measures the degree of determinism in token/character transitions to distinguish
    between discrete Finite State Automata (regular languages Type-3) and natural hierarchical grammars.
    """

    @staticmethod
    def evaluate_transition_predictability(sequence: Union[str, List[str]], order: int = 1) -> Dict[str, float]:
        """
        Evaluates the mean conditional entropy and top-1 transition accuracy under order-N Markov assumption.
        """
        if len(sequence) <= order:
            return {"order": order, "mean_transition_entropy": 0.0, "top1_accuracy": 1.0}

        transitions = defaultdict(Counter)
        for i in range(len(sequence) - order):
            ctx = tuple(sequence[i:i + order]) if isinstance(sequence, list) else sequence[i:i + order]
            nxt = sequence[i + order]
            transitions[ctx][nxt] += 1

        total_contexts = len(transitions)
        entropies = []
        top1_correct = 0
        total_transitions = 0

        for ctx, nxt_counts in transitions.items():
            tot = sum(nxt_counts.values())
            total_transitions += tot
            # Top-1 choice frequency
            top1_correct += max(nxt_counts.values())

            # Entropy of this context
            h = 0.0
            for count in nxt_counts.values():
                p = count / tot
                if p > 0:
                    h -= p * np.log2(p)
            entropies.append(h)

        mean_h = np.mean(entropies) if entropies else 0.0
        top1_acc = top1_correct / total_transitions if total_transitions > 0 else 0.0

        return {
            "order": order,
            "unique_contexts": total_contexts,
            "mean_transition_entropy": round(float(mean_h), 4),
            "top1_prediction_accuracy": round(float(top1_acc), 4)
        }

    @classmethod
    def analyze_grammar_type(cls, tokens: List[str]) -> Dict[str, Any]:
        """
        Runs order-1, order-2, and order-3 Markov evaluations on the token stream.
        """
        chars = list("".join(tokens))
        res_order1 = cls.evaluate_transition_predictability(chars, order=1)
        res_order2 = cls.evaluate_transition_predictability(chars, order=2)
        res_order3 = cls.evaluate_transition_predictability(chars, order=3)

        # A drop of transition entropy near 0 or very high top-1 accuracy (>0.85)
        # indicates strong local automaton / regular grammar constraints
        is_rigid_automaton = bool(res_order2["top1_prediction_accuracy"] > 0.80)

        return {
            "order_1": res_order1,
            "order_2": res_order2,
            "order_3": res_order3,
            "is_rigid_automaton_compatible": is_rigid_automaton
        }
