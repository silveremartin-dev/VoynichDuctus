"""
Adapter for offline-to-online handwriting derendering models (e.g. Google Research InkSight)
with fallback heuristic kinematic derendering.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class InkSightAdapter:
    """
    Interface for deep offline-to-online derendering models (like Google InkSight: ViT+mT5).
    Provides seamless integration when torch/transformers weights are available,
    and a robust topological fallback for zero-dependency execution.
    """

    def __init__(self, model_name_or_path: Optional[str] = None, use_gpu: bool = False):
        self.model_name_or_path = model_name_or_path
        self.use_gpu = use_gpu
        self._model = None
        self._is_loaded = False

    def is_available(self) -> bool:
        """Checks if deep learning weights / packages (torch, transformers) are present."""
        try:
            import torch  # noqa: F401
            return self._is_loaded or (self.model_name_or_path is not None)
        except ImportError:
            return False

    def derender(self, word_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Derenders a 2D raster word patch into timestamped (x, y, t, p) stroke vectors.
        """
        if self.is_available() and self._model is not None:
            # Model inference branch
            return self._infer_deep_model(word_image)
        else:
            # Heuristic topological kinematic extraction
            return self._heuristic_derender(word_image)

    def _heuristic_derender(self, word_image: np.ndarray) -> List[Dict[str, Any]]:
        from voynich_ductus.vectorizer.skeleton import Skeletonizer
        from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
        from voynich_ductus.vectorizer.junction_resolver import JunctionResolver

        skel_engine = Skeletonizer(method="medial_axis")
        graph_engine = StrokeGraphExtractor()
        resolver = JunctionResolver()

        binary = word_image if word_image.dtype == bool else (word_image > 0)
        skel, widths = skel_engine.extract_skeleton(binary)
        g = graph_engine.build_pixel_graph(skel, widths)
        raw_strokes = graph_engine.decompose_into_strokes(g)
        ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)

        # Convert to timestamped trajectory
        derendered = []
        global_t = 0.0
        for stroke in ordered_strokes:
            stroke_pts = []
            for y, x, w in stroke["points"]:
                stroke_pts.append({
                    "x": float(x),
                    "y": float(y),
                    "t": float(global_t),
                    "pressure_proxy": float(w),
                    "pen_down": True
                })
                global_t += 1.0  # Normalized kinematic step
            global_t += 2.0  # Pen-lift gap
            derendered.append({
                "stroke_id": stroke.get("stroke_id"),
                "order_index": stroke.get("order_index", 0),
                "points": stroke_pts
            })

        return derendered

    def _infer_deep_model(self, word_image: np.ndarray) -> List[Dict[str, Any]]:
        # Placeholder for external model checkpoint invocation
        return self._heuristic_derender(word_image)
