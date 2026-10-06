"""
Voynichese Rosetta Stone Annotation Loader and Alignment Engine.
Parses, aligns, and projects the comprehensive 225-folio Voynichese XML dataset
onto ultra-high-resolution Yale Beinecke scans to create an objective, supervised
ground truth corpus of 35,000+ words with EVA transliterations.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image


class RosettaWord:
    """
    Represents an annotated word entry from the Voynichese ground truth corpus.
    """
    def __init__(
        self,
        folio: str,
        index: int,
        eva_text: str,
        x_xml: float,
        y_xml: float,
        width_xml: float,
        height_xml: float,
        bbox_scan: Optional[Tuple[int, int, int, int]] = None,
        image_patch: Optional[Image.Image] = None,
    ):
        self.folio = folio
        self.index = index
        self.eva_text = (eva_text or "").strip()
        self.x_xml = x_xml
        self.y_xml = y_xml
        self.width_xml = width_xml
        self.height_xml = height_xml
        self.bbox_scan = bbox_scan  # (min_x, min_y, max_x, max_y)
        self.image_patch = image_patch

    def __repr__(self) -> str:
        return f"<RosettaWord {self.folio}:w{self.index} '{self.eva_text}' xml=({self.x_xml:.0f},{self.y_xml:.0f},{self.width_xml:.0f},{self.height_xml:.0f})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folio": self.folio,
            "index": self.index,
            "eva_text": self.eva_text,
            "x_xml": self.x_xml,
            "y_xml": self.y_xml,
            "width_xml": self.width_xml,
            "height_xml": self.height_xml,
            "bbox_scan": self.bbox_scan,
        }


class VoynicheseRosettaLoader:
    """
    Loader and coordinate alignment engine for the 225-folio Voynichese XML dataset.
    """

    # Standard Voynich EVA atomic glyph token definitions (strictly ordered by length for regex matching)
    EVA_GLYPH_TOKENS = [
        "cfh", "ckh", "cph", "cth",  # compound gallows
        "ch", "sh",                   # bench-tied gallows
        "ct", "ck", "cp", "cf",
        "iii", "ee", "ii",
        "a", "o", "y", "s", "d", "r", "l", "n", "m", "g", "x", "q", "k", "t", "p", "f", "i"
    ]

    def __init__(
        self,
        annotations_dir: Union[str, Path] = "data/annotations/voynichese",
        scans_dir: Union[str, Path] = "data/scans/voynich/yale",
    ):
        self.annotations_dir = Path(annotations_dir)
        self.scans_dir = Path(scans_dir)
        self._cache_folios: Dict[str, List[RosettaWord]] = {}
        self._folio_canvas_meta: Dict[str, Tuple[float, float, int]] = {}

    def list_available_folios(self) -> List[str]:
        """
        Returns a sorted list of all available folio identifiers in the annotation directory.
        """
        if not self.annotations_dir.exists():
            return []
        xml_files = list(self.annotations_dir.glob("*.xml"))
        # Sort naturally (e.g. f1r, f2r ... f100r)
        def sort_key(p: Path):
            name = p.stem
            match = re.match(r"f(\d+)([rv]\d*)?", name)
            if match:
                num = int(match.group(1))
                side = match.group(2) or ""
                return (num, side)
            return (9999, name)
        
        sorted_files = sorted(xml_files, key=sort_key)
        return [f.stem for f in sorted_files]

    def load_folio_words(self, folio_id: str) -> List[RosettaWord]:
        """
        Parses the Voynichese XML file for a given folio and returns a list of RosettaWord objects.
        """
        # Normalize folio name (e.g., 'f001r' -> 'f1r', 'f067v2' -> 'f67v2')
        norm_name = self._normalize_folio_id(folio_id)
        
        if norm_name in self._cache_folios:
            return self._cache_folios[norm_name]

        xml_path = self.annotations_dir / f"{norm_name}.xml"
        if not xml_path.exists():
            # Try alternate zero-padded / unpadded naming
            candidates = list(self.annotations_dir.glob(f"*{norm_name}*.xml"))
            if not candidates:
                raise FileNotFoundError(f"Voynichese XML annotation not found for folio '{folio_id}' (searched {xml_path})")
            xml_path = candidates[0]

        tree = ET.parse(xml_path)
        root = tree.getroot()
        orig_w = float(root.attrib.get("width", 1090.0))
        orig_h = float(root.attrib.get("height", 1500.0))
        word_count = int(root.attrib.get("wordCount", 0))
        self._folio_canvas_meta[norm_name] = (orig_w, orig_h, word_count)

        words: List[RosettaWord] = []
        for elem in root.findall("word"):
            idx = int(elem.attrib.get("index", len(words)))
            x = float(elem.attrib.get("x", 0))
            y = float(elem.attrib.get("y", 0))
            w = float(elem.attrib.get("width", 0))
            h = float(elem.attrib.get("height", 0))
            eva = (elem.text or "").strip()

            word_obj = RosettaWord(
                folio=norm_name,
                index=idx,
                eva_text=eva,
                x_xml=x,
                y_xml=y,
                width_xml=w,
                height_xml=h,
            )
            words.append(word_obj)

        self._cache_folios[norm_name] = words
        return words

    def compute_scan_affine_transform(
        self,
        folio_id: str,
        scan_image: Image.Image,
        xml_canvas: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Computes the affine scaling and offset parameters (sx, sy, tx, ty) to map
        Voynichese XML coordinates (x_xml, y_xml) to scan image coordinates:
            x_scan = tx + x_xml * sx
            y_scan = ty + y_xml * sy
        """
        norm_name = self._normalize_folio_id(folio_id)
        if xml_canvas is None:
            if norm_name in self._folio_canvas_meta:
                xml_w, xml_h, _ = self._folio_canvas_meta[norm_name]
            else:
                xml_w, xml_h = 1090.0, 1500.0
        else:
            xml_w, xml_h = xml_canvas

        scan_w, scan_h = scan_image.size

        # Heuristic parchment registration for Yale full scans (approx 2500 x 3168 with black margin)
        # Yale full scans typically contain ~200-240px black binding border on the left and 40-60px at top.
        if scan_w >= 2000 and scan_h >= 2500:
            # High-res Yale scan
            sx = (scan_w - 300.0) / xml_w
            sy = (scan_h - 100.0) / xml_h
            tx = 230.0
            ty = 50.0
        else:
            # Generic direct proportional scaling
            sx = scan_w / xml_w
            sy = scan_h / xml_h
            tx = 0.0
            ty = 0.0

        return (sx, sy, tx, ty)

    def extract_aligned_words(
        self,
        folio_id: str,
        scan_image: Optional[Image.Image] = None,
        padding_px: int = 4,
    ) -> List[RosettaWord]:
        """
        Extracts all words for a folio with high-res bounding boxes and cropped image patches.
        """
        words = self.load_folio_words(folio_id)
        if scan_image is None:
            scan_path = self._find_scan_path(folio_id)
            if scan_path and scan_path.exists():
                scan_image = Image.open(scan_path)
            else:
                return words

        sx, sy, tx, ty = self.compute_scan_affine_transform(folio_id, scan_image)
        img_w, img_h = scan_image.size

        aligned_words: List[RosettaWord] = []
        for w in words:
            # Map XML bbox to scan coordinates
            min_x = max(0, int(tx + w.x_xml * sx - padding_px))
            min_y = max(0, int(ty + w.y_xml * sy - padding_px))
            max_x = min(img_w, int(tx + (w.x_xml + w.width_xml) * sx + padding_px))
            max_y = min(img_h, int(ty + (w.y_xml + w.height_xml) * sy + padding_px))

            bbox_scan = (min_x, min_y, max_x, max_y)
            
            # Crop image patch
            if max_x > min_x and max_y > min_y:
                patch = scan_image.crop(bbox_scan)
            else:
                patch = None

            aligned_w = RosettaWord(
                folio=w.folio,
                index=w.index,
                eva_text=w.eva_text,
                x_xml=w.x_xml,
                y_xml=w.y_xml,
                width_xml=w.width_xml,
                height_xml=w.height_xml,
                bbox_scan=bbox_scan,
                image_patch=patch,
            )
            aligned_words.append(aligned_w)

        return aligned_words

    def tokenize_eva_to_glyphs(self, eva_text: str) -> List[str]:
        """
        Decomposes an EVA word string into a sequence of canonical atomic EVA glyphs.
        E.g. 'cthaiin' -> ['cth', 'a', 'ii', 'n']
             'sholdy'  -> ['sh', 'o', 'l', 'd', 'y']
             'fachys'  -> ['f', 'a', 'ch', 'y', 's']
        """
        text = eva_text.strip().lower()
        if not text:
            return []

        # Sort tokens by descending length for greedy longest-match tokenization
        sorted_tokens = sorted(self.EVA_GLYPH_TOKENS, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(t) for t in sorted_tokens))

        pos = 0
        glyphs = []
        while pos < len(text):
            match = pattern.match(text, pos)
            if match:
                glyphs.append(match.group(0))
                pos = match.end()
            else:
                # Single character fallback
                glyphs.append(text[pos])
                pos += 1

        return glyphs

    def get_full_corpus_statistics(self) -> Dict[str, Any]:
        """
        Aggregates statistical metrics across all 225 folios in the Voynichese ground truth.
        """
        folios = self.list_available_folios()
        total_words = 0
        word_freq: Dict[str, int] = {}
        glyph_freq: Dict[str, int] = {}
        folio_word_counts: Dict[str, int] = {}

        for f in folios:
            words = self.load_folio_words(f)
            folio_word_counts[f] = len(words)
            total_words += len(words)

            for w in words:
                text = w.eva_text
                if text:
                    word_freq[text] = word_freq.get(text, 0) + 1
                    glyphs = self.tokenize_eva_to_glyphs(text)
                    for g in glyphs:
                        glyph_freq[g] = glyph_freq.get(g, 0) + 1

        return {
            "total_folios": len(folios),
            "total_words": total_words,
            "unique_words": len(word_freq),
            "unique_glyphs": len(glyph_freq),
            "top_words": sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:25],
            "top_glyphs": sorted(glyph_freq.items(), key=lambda x: x[1], reverse=True)[:25],
            "folio_word_counts": folio_word_counts,
        }

    def _normalize_folio_id(self, folio_id: str) -> str:
        """
        Standardizes folio id: 'f001r' -> 'f1r', 'f102v1' -> 'f102v1'
        """
        clean = folio_id.strip().lower()
        clean = re.sub(r"\.(xml|jpg|jpeg|png)$", "", clean)
        match = re.match(r"f0*(\d+)([rv]\d*)?", clean)
        if match:
            num = match.group(1)
            side = match.group(2) or ""
            return f"f{num}{side}"
        return clean

    def _find_scan_path(self, folio_id: str) -> Optional[Path]:
        """
        Locates the corresponding scan image file in scans_dir.
        """
        norm_name = self._normalize_folio_id(folio_id)
        # Try both f001r and f1r formatting
        match = re.match(r"f(\d+)([rv]\d*)?", norm_name)
        candidates = []
        if match:
            num = int(match.group(1))
            side = match.group(2) or ""
            padded = f"f{num:03d}{side}"
            candidates.extend([
                self.scans_dir / f"{padded}.jpg",
                self.scans_dir / f"{norm_name}.jpg",
                self.scans_dir.parent / f"{padded}.jpg",
                self.scans_dir.parent / f"{norm_name}.jpg",
            ])
        else:
            candidates.append(self.scans_dir / f"{norm_name}.jpg")

        for p in candidates:
            if p.exists():
                return p
        return None
