"""
Visualization utilities for skeletons, stroke kinematics, and DFA curves.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


class Visualizer:
    """
    Renders diagnostic plots, skeleton overlays, and stroke order diagrams.
    """

    @staticmethod
    def plot_strokes(strokes: List[Dict[str, Any]], image_shape: tuple = (100, 200), output_path: Optional[str] = None):
        """Plots ordered strokes with color gradients denoting chronological ductus."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_facecolor("#fcfaf2")

        cmap = plt.get_cmap("viridis")
        n_strokes = max(len(strokes), 1)

        for i, s in enumerate(strokes):
            pts = s.get("points", [])
            if not pts:
                continue
            color = cmap(i / n_strokes)
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            ws = [p[2] if len(p) > 2 else 2.0 for p in pts]

            ax.plot(xs, ys, color=color, linewidth=np.mean(ws) if ws else 2.0, alpha=0.85)
            # Mark start point (pen down) with a circle
            ax.scatter([xs[0]], [ys[0]], color=color, s=25, zorder=5)

        ax.set_ylim(image_shape[0], 0)  # Invert Y axis for image coordinates
        ax.set_xlim(0, image_shape[1])
        ax.set_title("Reconstructed Scribal Ductus (Pen-down to Pen-up)")
        ax.set_aspect("equal")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        return fig
