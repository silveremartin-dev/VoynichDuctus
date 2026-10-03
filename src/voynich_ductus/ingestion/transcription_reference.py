"""
Reference transcription metadata and paleographic yield validation.
Provides standard interlinear reference counts (Takahashi EVA / Currier / Landini)
to benchmark and calibrate automated extraction recall without imposing transliteration biases.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class FolioMetadata:
    folio_id: str
    expected_lines: int
    expected_words: int
    section: str
    currier_language: str  # Currier A / Currier B
    scribe_hand: str       # Davis / Stolfi scribe hand (Hand 1 to 5)
    description: str


class TranscriptionReference:
    """
    Standard ground-truth transcription metadata for Beinecke MS 408 folios.
    Used as an objective statistical benchmark for extraction recall (lines/words/glyphs).
    """

    # Ground-truth census compiled from canonical Landini / Takahashi IVTFF corpus
    FOLIO_REGISTRY: Dict[str, FolioMetadata] = {
        "f001r": FolioMetadata("f001r", expected_lines=28, expected_words=238, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Opening herbal folio with header gallows and dense text"),
        "f001v": FolioMetadata("f001v", expected_lines=28, expected_words=231, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Herbal plant illustration with left/right text flanking"),
        "f002r": FolioMetadata("f002r", expected_lines=22, expected_words=152, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Herbal page with centered single plant"),
        "f002v": FolioMetadata("f002v", expected_lines=23, expected_words=160, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Herbal page with split root drawing"),
        "f003r": FolioMetadata("f003r", expected_lines=24, expected_words=175, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Herbal page with composite foliage"),
        "f003v": FolioMetadata("f003v", expected_lines=25, expected_words=180, section="Herbal 1", currier_language="Currier A", scribe_hand="Hand 1", description="Herbal page with text block surrounding root"),
        "f067r1": FolioMetadata("f067r1", expected_lines=18, expected_words=130, section="Astronomical", currier_language="Currier A", scribe_hand="Hand 3", description="Circular astronomical diagram with concentric text rings"),
        "f067r2": FolioMetadata("f067r2", expected_lines=12, expected_words=95, section="Astronomical", currier_language="Currier A", scribe_hand="Hand 3", description="Astronomical chart with sun and star indicators"),
        "f075r": FolioMetadata("f075r", expected_lines=42, expected_words=410, section="Balneological", currier_language="Currier B", scribe_hand="Hand 2", description="Balneological dense text page with female figures in water channels"),
        "f075v": FolioMetadata("f075v", expected_lines=45, expected_words=435, section="Balneological", currier_language="Currier B", scribe_hand="Hand 2", description="Balneological text page with elaborate pipe systems"),
        "f087r": FolioMetadata("f087r", expected_lines=35, expected_words=320, section="Cosmological", currier_language="Currier B", scribe_hand="Hand 2", description="Fold-out rosette cosmological diagram labels"),
        "f103r": FolioMetadata("f103r", expected_lines=38, expected_words=380, section="Pharmaceutical", currier_language="Currier B", scribe_hand="Hand 4", description="Pharmaceutical jars with small plant roots and text paragraphs"),
        "f107r": FolioMetadata("f107r", expected_lines=32, expected_words=310, section="Recipes", currier_language="Currier B", scribe_hand="Hand 5", description="Stars and short recipe paragraphs with paragraph-initial gallows"),
        "f111r": FolioMetadata("f111r", expected_lines=36, expected_words=365, section="Recipes", currier_language="Currier B", scribe_hand="Hand 5", description="Dense multi-column recipe paragraphs"),
    }

    @classmethod
    def get_metadata(cls, folio_id: str) -> Optional[FolioMetadata]:
        """Returns the canonical reference metadata for a folio ID (e.g. 'f001r', 'f1r', 'f075r')."""
        norm_id = folio_id.lower().strip()
        if not norm_id.startswith("f"):
            norm_id = "f" + norm_id
        # Normalize digits
        match = re.match(r"f0*(\d+)([rv]\d*)?", norm_id)
        if match:
            num = int(match.group(1))
            suffix = match.group(2) or "r"
            norm_id = f"f{num:03d}{suffix}"
        
        return cls.FOLIO_REGISTRY.get(norm_id)


class PaleographyYieldValidator:
    """
    Validates automated segmentation and vectorization yield against standard interlinear counts.
    Diagnoses under-segmentation (missed words) and over-segmentation (noise/fragments).
    """

    @staticmethod
    def evaluate_yield(
        folio_id: str,
        detected_lines: int,
        detected_words: int,
        extracted_strokes: int
    ) -> Dict[str, Any]:
        """
        Calculates extraction yield and diagnostics relative to reference census.
        """
        meta = TranscriptionReference.get_metadata(folio_id)
        if not meta:
            return {
                "folio_id": folio_id,
                "has_reference": False,
                "detected_lines": detected_lines,
                "detected_words": detected_words,
                "extracted_strokes": extracted_strokes,
                "avg_strokes_per_word": round(extracted_strokes / max(1, detected_words), 2),
                "status": "UNREGISTERED_FOLIO"
            }

        line_yield = detected_lines / max(1, meta.expected_lines)
        word_yield = detected_words / max(1, meta.expected_words)
        avg_spw = extracted_strokes / max(1, detected_words)

        diagnostics = []
        if line_yield < 0.60:
            diagnostics.append(f"UNDER_SEGMENTED_LINES (Detected {detected_lines}/{meta.expected_lines}, yield: {line_yield*100:.1f}%)")
        elif line_yield > 1.30:
            diagnostics.append(f"OVER_SEGMENTED_LINES (Detected {detected_lines}/{meta.expected_lines}, yield: {line_yield*100:.1f}%)")

        if word_yield < 0.20:
            diagnostics.append(f"LOW_WORD_EXTRACTION (Detected {detected_words}/{meta.expected_words}, yield: {word_yield*100:.1f}%)")
        
        if avg_spw < 3.0:
            diagnostics.append(f"FRAGMENTED_WORDS (Avg strokes/word: {avg_spw:.1f} < 3.0)")
        elif avg_spw > 25.0:
            diagnostics.append(f"CONFLATED_PHRASES (Avg strokes/word: {avg_spw:.1f} > 25.0)")

        status = "OPTIMAL" if not diagnostics else "FLAGGED"

        return {
            "folio_id": meta.folio_id,
            "has_reference": True,
            "section": meta.section,
            "currier_language": meta.currier_language,
            "scribe_hand": meta.scribe_hand,
            "description": meta.description,
            "expected_lines": meta.expected_lines,
            "detected_lines": detected_lines,
            "line_yield_pct": round(line_yield * 100, 1),
            "expected_words": meta.expected_words,
            "detected_words": detected_words,
            "word_yield_pct": round(word_yield * 100, 1),
            "extracted_strokes": extracted_strokes,
            "avg_strokes_per_word": round(avg_spw, 2),
            "diagnostics": diagnostics,
            "status": status
        }
