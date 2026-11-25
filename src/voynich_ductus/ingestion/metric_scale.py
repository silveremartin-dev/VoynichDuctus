"""
Physical Metric Scale Calibration & Conversion System.
Converts between physical metric units (millimeters, tenths of a millimeter)
and digital scan pixels across historical codices and modern editions.
Guarantees scale-invariant paleographic thresholds regardless of scan resolution (DPI).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


class MetricScaleCalibration:
    """
    Calibrates digital image resolution against authentic physical codex folio dimensions.
    Enables all paleographic filters (nib widths, dot touchdown footprints, scale bands,
    line pitch, inter-word whitespace) to be expressed in millimeters (mm) or tenths of a millimeter (0.1 mm).
    """

    # Physical folio height standards (in mm) from codicological records:
    # - Voynich Manuscript (Beinecke MS 408): ~225 mm height x ~160 mm width
    # - Codex Seraphinianus (Franco Maria Ricci 1981 folio edition): ~350 mm height x ~235 mm width
    # - Gallica BnF Latin Codices (e.g. Latin 6823, 17868 herbal manuscripts): ~280 mm height
    DEFAULT_FOLIO_HEIGHTS_MM: Dict[str, float] = {
        "voynich": 225.0,
        "seraphinianus": 350.0,
        "gallica": 280.0,
        "generic": 250.0,
    }

    # Standard paleographical physical thresholds (expressed in millimeters):
    # Dot touchdown impact footprint: nib contact with v = 0 (diameter <= 0.55 mm, area <= 0.35 mm^2)
    DOT_MAX_DIAMETER_MM: float = 0.55       # 5.5 tenths of mm
    DOT_MAX_AREA_MM2: float = 0.35          # 0.35 mm^2
    
    # Minimum legible character / stroke threshold
    MIN_GLYPH_HEIGHT_MM: float = 0.85       # 8.5 tenths of mm
    MIN_GLYPH_WIDTH_MM: float = 0.45        # 4.5 tenths of mm
    MIN_GLYPH_AREA_MM2: float = 0.12        # 0.12 mm^2

    # Physical scale stratification bands for unsupervised alphabet induction:
    TINY_MAX_HEIGHT_MM: float = 1.30        # 13 tenths of mm (dots, punctuation, specks)
    CORE_MIN_HEIGHT_MM: float = 1.30        # 13 tenths of mm (minims, small bowls, e, c, o, a)
    CORE_MAX_HEIGHT_MM: float = 3.40        # 34 tenths of mm
    TALL_GALLOWS_MIN_HEIGHT_MM: float = 3.40 # 34 tenths of mm (hastes, ascenders, d, k, t, gallows)
    WIDE_BENCH_MIN_WIDTH_MM: float = 3.40   # 34 tenths of mm (horizontal compound gallows, ligatures)

    # Quill nib & intra-word ligature decomposition physical thresholds:
    QUILL_NIB_WIDTH_MM: float = 0.45        # 4.5 tenths of mm (standard 15th-century beveled quill stroke width)
    SINGLE_GLYPH_MAX_WIDTH_MM: float = 3.50 # 35 tenths of mm (maximum width of an unsplit single character)
    LIGATURE_VALLEY_MIN_PITCH_MM: float = 1.45 # 14.5 tenths of mm (minimum center-to-center pitch between cut letters)
    LIGATURE_PEAK_WINDOW_MM: float = 1.80   # 18 tenths of mm (horizontal window for peak-to-valley ink dip ratio)
    MIN_LOOP_HOLE_AREA_MM2: float = 0.18    # 0.18 mm^2 (~12 px^2 at 8 px/mm: minimum enclosed ink loop cavity)
    MIN_LOOP_HOLE_HEIGHT_MM: float = 0.50   # 5 tenths of mm (minimum vertical opening of a genuine loop)
    MIN_LOOP_HOLE_WIDTH_MM: float = 0.38    # 3.8 tenths of mm (minimum horizontal opening of a genuine loop)
    INTRA_WORD_BRIDGE_MM: float = 2.10      # 21 tenths of mm (max inter-letter gap inside a single word hull)
    INTRA_GLYPH_BRIDGE_MM: float = 1.10     # 11 tenths of mm (max multi-stroke gap inside a single glyph hull)

    # Document layout physical metric standards:
    LINE_PITCH_MEDIEVAL_MM: float = 6.50    # 65 tenths of mm (~6.5 mm typical medieval ruling pitch)
    WORD_SPACE_GAP_MM: float = 1.10         # 11 tenths of mm (inter-word physical whitespace)
    DIACRITIC_MAX_DIST_MM: float = 0.90     # 9 tenths of mm (supralinear floating attachment zone)
    MACRO_DRAWING_MIN_HEIGHT_MM: float = 16.0 # 160 tenths of mm (illustrations vs text glyphs)
    MACRO_DRAWING_MIN_AREA_MM2: float = 30.0  # 30 mm^2

    def __init__(
        self,
        manuscript_type: str = "voynich",
        image_height_px: Optional[int] = None,
        image_width_px: Optional[int] = None,
        custom_px_per_mm: Optional[float] = None
    ):
        self.manuscript_type = (manuscript_type or "voynich").lower()
        self.folio_height_mm = self.DEFAULT_FOLIO_HEIGHTS_MM.get(self.manuscript_type, 225.0)

        if custom_px_per_mm is not None and custom_px_per_mm > 0:
            self.px_per_mm = float(custom_px_per_mm)
        elif image_height_px is not None and image_height_px > 0:
            self.px_per_mm = float(image_height_px) / float(self.folio_height_mm)
        else:
            # Fallback default: standard Voynich scan 1800 px / 225 mm = 8.0 px/mm
            self.px_per_mm = 8.0

        self.mm_per_px = 1.0 / max(1e-5, self.px_per_mm)
        self.image_height_px = image_height_px
        self.image_width_px = image_width_px

    # ------------------------------------------------------------------
    # Forward Conversions: Physical Metric Units -> Digital Pixels
    # ------------------------------------------------------------------
    def mm_to_px(self, mm_val: float) -> float:
        """Converts millimeters (mm) to pixels (px)."""
        return float(mm_val * self.px_per_mm)

    def tenths_mm_to_px(self, tenths_val: float) -> float:
        """Converts tenths of a millimeter (0.1 mm) to pixels (px)."""
        return float(tenths_val * 0.10 * self.px_per_mm)

    def area_mm2_to_px2(self, mm2_val: float) -> float:
        """Converts physical square millimeters (mm^2) to square pixels (px^2)."""
        return float(mm2_val * (self.px_per_mm ** 2))

    # ------------------------------------------------------------------
    # Backward Conversions: Digital Pixels -> Physical Metric Units
    # ------------------------------------------------------------------
    def px_to_mm(self, px_val: float) -> float:
        """Converts pixels (px) to millimeters (mm)."""
        return float(px_val * self.mm_per_px)

    def px_to_tenths_mm(self, px_val: float) -> float:
        """Converts pixels (px) to tenths of a millimeter (0.1 mm)."""
        return float(px_val * self.mm_per_px * 10.0)

    def area_px2_to_mm2(self, px2_val: float) -> float:
        """Converts square pixels (px^2) to physical square millimeters (mm^2)."""
        return float(px2_val * (self.mm_per_px ** 2))

    # ------------------------------------------------------------------
    # Dynamic Scale-Calibrated Paleographic Threshold Queries
    # ------------------------------------------------------------------
    @property
    def dot_max_span_px(self) -> float:
        """Maximum bounding box dimension (px) for an instantaneous touchdown impact point."""
        return max(4.0, self.mm_to_px(self.DOT_MAX_DIAMETER_MM))

    @property
    def dot_max_area_px(self) -> float:
        """Maximum ink area (px^2) for an instantaneous touchdown impact point."""
        return max(12.0, self.area_mm2_to_px2(self.DOT_MAX_AREA_MM2))

    @property
    def min_glyph_height_px(self) -> float:
        """Minimum physical height (px) for an authentic textual glyph."""
        return max(5.0, self.mm_to_px(self.MIN_GLYPH_HEIGHT_MM))

    @property
    def min_glyph_width_px(self) -> float:
        """Minimum physical width (px) for an authentic textual glyph."""
        return max(3.0, self.mm_to_px(self.MIN_GLYPH_WIDTH_MM))

    @property
    def min_glyph_area_px(self) -> float:
        """Minimum physical ink area (px^2) for an authentic textual glyph."""
        return max(8.0, self.area_mm2_to_px2(self.MIN_GLYPH_AREA_MM2))

    @property
    def tiny_height_px(self) -> float:
        """Threshold separating TINY scale band (diacritics, specks) from CORE letter band."""
        return self.mm_to_px(self.TINY_MAX_HEIGHT_MM)

    @property
    def tall_height_px(self) -> float:
        """Threshold separating CORE letters from TALL ascenders/gallows."""
        return self.mm_to_px(self.TALL_GALLOWS_MIN_HEIGHT_MM)

    @property
    def wide_width_px(self) -> float:
        """Threshold separating standard letters from WIDE benches/ligatures."""
        return self.mm_to_px(self.WIDE_BENCH_MIN_WIDTH_MM)

    @property
    def word_space_min_px(self) -> float:
        """Inter-word horizontal gap threshold in pixels."""
        return max(8.0, self.mm_to_px(self.WORD_SPACE_GAP_MM))

    @property
    def diacritic_max_dist_px(self) -> float:
        """Maximum vertical gap (px) for associating a supralinear diacritic to its base glyph."""
        return max(4.0, self.mm_to_px(self.DIACRITIC_MAX_DIST_MM))

    @property
    def macro_drawing_height_px(self) -> float:
        """Height threshold (px) separating text lines/glyphs from macro drawings and miniatures."""
        return self.mm_to_px(self.MACRO_DRAWING_MIN_HEIGHT_MM)

    @property
    def macro_drawing_area_px(self) -> float:
        """Area threshold (px^2) separating text lines/glyphs from macro drawings and miniatures."""
        return self.area_mm2_to_px2(self.MACRO_DRAWING_MIN_AREA_MM2)

    @property
    def quill_nib_width_px(self) -> float:
        """Physical beveled quill nib stroke width in pixels."""
        return max(2.0, self.mm_to_px(self.QUILL_NIB_WIDTH_MM))

    @property
    def single_glyph_max_width_px(self) -> int:
        """Maximum width (px) of an intact single character before ligature valley analysis."""
        return max(16, int(round(self.mm_to_px(self.SINGLE_GLYPH_MAX_WIDTH_MM))))

    @property
    def ligature_valley_min_pitch_px(self) -> int:
        """Minimum horizontal pitch (px) between consecutive character cuts inside a word."""
        return max(8, int(round(self.mm_to_px(self.LIGATURE_VALLEY_MIN_PITCH_MM))))

    @property
    def ligature_peak_window_px(self) -> int:
        """Horizontal window (px) for measuring local ink height peaks around a ligature valley."""
        return max(8, int(round(self.mm_to_px(self.LIGATURE_PEAK_WINDOW_MM))))

    @property
    def min_loop_hole_area_px(self) -> int:
        """Minimum enclosed area (px^2) for a topological hole to be certified as a scribal loop rather than vellum grain noise."""
        return max(6, int(round(self.area_mm2_to_px2(self.MIN_LOOP_HOLE_AREA_MM2))))

    @property
    def min_loop_hole_height_px(self) -> int:
        """Minimum vertical opening (px) for a certified topological loop."""
        return max(3, int(round(self.mm_to_px(self.MIN_LOOP_HOLE_HEIGHT_MM))))

    @property
    def min_loop_hole_width_px(self) -> int:
        """Minimum horizontal opening (px) for a certified topological loop."""
        return max(2, int(round(self.mm_to_px(self.MIN_LOOP_HOLE_WIDTH_MM))))

    @property
    def intra_word_bridge_px(self) -> int:
        """Morphological horizontal closing span (px) bridging disconnected letters inside a single word polygon."""
        return max(7, int(round(self.mm_to_px(self.INTRA_WORD_BRIDGE_MM)))) | 1

    @property
    def intra_glyph_bridge_px(self) -> int:
        """Morphological horizontal closing span (px) bridging multi-stroke components inside a single glyph polygon."""
        return max(5, int(round(self.mm_to_px(self.INTRA_GLYPH_BRIDGE_MM)))) | 1

    def get_summary_dict(self) -> Dict[str, Any]:
        """Returns structured metadata dictionary documenting calibration parameters."""
        return {
            "manuscript_type": self.manuscript_type,
            "folio_height_mm": round(self.folio_height_mm, 1),
            "image_height_px": self.image_height_px,
            "px_per_mm": round(self.px_per_mm, 3),
            "mm_per_px": round(self.mm_per_px, 4),
            "dot_max_diameter_mm": self.DOT_MAX_DIAMETER_MM,
            "dot_max_diameter_tenths_mm": round(self.DOT_MAX_DIAMETER_MM * 10, 1),
            "dot_max_span_px": round(self.dot_max_span_px, 1),
            "dot_max_area_px": round(self.dot_max_area_px, 1),
            "min_glyph_height_mm": self.MIN_GLYPH_HEIGHT_MM,
            "min_glyph_height_px": round(self.min_glyph_height_px, 1),
            "tiny_height_px": round(self.tiny_height_px, 1),
            "tall_height_px": round(self.tall_height_px, 1),
            "wide_width_px": round(self.wide_width_px, 1),
            "word_space_min_px": round(self.word_space_min_px, 1),
        }
