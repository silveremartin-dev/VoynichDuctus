"""
Voynichese Rosetta Stone Annotation Loader and Alignment Engine.
Parses, aligns, and projects the comprehensive 225-folio Voynichese XML dataset
onto ultra-high-resolution Yale Beinecke scans to create an objective, supervised
ground truth corpus of 35,000+ words with EVA transliterations.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
import re
import json
import urllib.request
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
        js_cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.scans_dir = Path(scans_dir)
        self.js_cache_dir = Path(js_cache_dir) if js_cache_dir is not None else (self.annotations_dir / "js_cache")
        self._cache_folios: Dict[str, List[RosettaWord]] = {}
        self._cache_js_folios: Dict[str, List[RosettaWord]] = {}
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

    def load_folio_words_from_js(self, folio_id: str) -> List[RosettaWord]:
        """
        Loads word annotations directly from the Voynichese .js folio script files.
        Coordinates in .js are direct pixel coordinates for the 636x900 glance image:
            https://www.voynichese.com/2/data/folio/image/glance/color/large/{folio_id}.jpg
        Downloads and caches the .js file locally under data/annotations/voynichese/js_cache/.
        """
        norm_name = self._normalize_folio_id(folio_id)
        if norm_name in self._cache_js_folios:
            return self._cache_js_folios[norm_name]

        self.js_cache_dir.mkdir(parents=True, exist_ok=True)
        js_path = self.js_cache_dir / f"{norm_name}.js"

        if not js_path.exists():
            url = f"https://www.voynichese.com/2/data/folio/script/{norm_name}.js"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                content = resp.read().decode("utf-8")
            with open(js_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(js_path, "r", encoding="utf-8") as f:
                content = f.read()

        data = json.loads(content)
        words_dict = data[0]
        boxes = data[1]

        # Record canvas metadata (native Voynichese glance image is 636x900)
        self._folio_canvas_meta[norm_name] = (636.0, 900.0, len(boxes))

        words: List[RosettaWord] = []
        for i, box in enumerate(boxes):
            # Each box is [word_dict_idx, left, top, width, height]
            word_dict_idx = int(box[0])
            left = float(box[1])
            top = float(box[2])
            width = float(box[3])
            height = float(box[4])
            eva_text = words_dict[word_dict_idx][0] if 0 <= word_dict_idx < len(words_dict) else ""

            bbox_scan = (
                int(round(left)),
                int(round(top)),
                int(round(left + width)),
                int(round(top + height)),
            )

            word_obj = RosettaWord(
                folio=norm_name,
                index=i,
                eva_text=eva_text,
                x_xml=left,
                y_xml=top,
                width_xml=width,
                height_xml=height,
                bbox_scan=bbox_scan,
            )
            words.append(word_obj)

        self._cache_js_folios[norm_name] = words
        return words

    def compute_scan_affine_transform(
        self,
        folio_id: str,
        scan_image: Image.Image,
        xml_canvas: Optional[Tuple[float, float]] = None,
        use_js: bool = True,
    ) -> Tuple[float, float, float, float]:
        """
        Computes the affine scaling and offset parameters (sx, sy, tx, ty) to map
        word coordinates (x, y) to scan image coordinates:
            x_scan = tx + x * sx
            y_scan = ty + y * sy

        For the small-image case (native voynichese scan <= 1400px wide):
        - When using JS coordinates:
          - returns (1.0, 1.0, 0.0, 0.0) for native 636x900 image
          - returns (2.0, 2.0, 0.0, 0.0) for 2x upscaled 1272x1800 image
        - When using XML coordinates (use_js=False):
          - returns scaling relative to XML canvas: sx = img_w / xml_w, sy = img_h / xml_h
        """
        norm_name = self._normalize_folio_id(folio_id)
        if xml_canvas is None:
            if norm_name in self._folio_canvas_meta:
                xml_w, xml_h, _ = self._folio_canvas_meta[norm_name]
            else:
                xml_w, xml_h = 1090.0, 1500.0
        else:
            xml_w, xml_h = xml_canvas

        img_w, img_h = scan_image.size
        # When scan_image is the native Voynichese viewport plate (width ~ 636px or upscaled 2x ~ 1272px):
        if img_w <= 1400:
            if use_js:
                if img_w == 636 and img_h == 900:
                    return (1.0, 1.0, 0.0, 0.0)
                elif img_w == 1272 and img_h == 1800:
                    return (2.0, 2.0, 0.0, 0.0)
                else:
                    return (round(float(img_w / 636.0), 4), round(float(img_h / 900.0), 4), 0.0, 0.0)
            else:
                sx = img_w / xml_w
                sy = img_h / xml_h
                return (round(float(sx), 4), round(float(sy), 4), 0.0, 0.0)


        # Certified affine transforms measured via 2D normalized cross-correlation / template-matching
        # directly between the Voynichese glance viewports and Yale Beinecke HD scans.
        EXACT_VOYNICHESE_ALIGNED_TRANSFORMS: Dict[str, Tuple[float, float, float, float]] = {
            "f1r": (1.7505, 1.8000, 330.0, 125.0),
            "f001r": (1.7505, 1.8000, 330.0, 125.0),
            "f1v": (1.7983, 1.8000, 350.0, 219.0),
            "f001v": (1.7983, 1.8000, 350.0, 219.0),
            "f2r": (1.9182, 1.9200, 208.0, 164.0),
            "f002r": (1.9182, 1.9200, 208.0, 164.0),
            "f2v": (1.7983, 1.8000, 826.0, 169.0),
            "f002v": (1.7983, 1.8000, 826.0, 169.0),
            "f3r": (1.8582, 1.8600, 231.0, 193.5),
            "f003r": (1.8582, 1.8600, 231.0, 193.5),
            "f3v": (1.7983, 1.8000, 370.0, 107.0),
            "f003v": (1.7983, 1.8000, 370.0, 107.0),
            "f4r": (1.7983, 1.8000, 246.0, 131.0),
            "f004r": (1.7983, 1.8000, 246.0, 131.0),
            "f4v": (1.7983, 1.8000, 364.0, 69.0),
            "f004v": (1.7983, 1.8000, 364.0, 69.0),
        }

        if norm_name in EXACT_VOYNICHESE_ALIGNED_TRANSFORMS:
            base_sx, base_sy, base_tx, base_ty = EXACT_VOYNICHESE_ALIGNED_TRANSFORMS[norm_name]
            if use_js:
                # Scale relative to JS glance canvas (636x900) instead of XML canvas (1090x1500)
                # and adapt to scan image resolution (baseline 2400px wide)
                js_to_xml_x = 1090.0 / 636.0
                js_to_xml_y = 1500.0 / 900.0
                res_factor = float(img_w) / 2400.0
                sx = base_sx * js_to_xml_x * res_factor
                sy = base_sy * js_to_xml_y * res_factor
                tx = base_tx * res_factor
                ty = base_ty * res_factor
                return (round(sx, 4), round(sy, 4), round(tx, 1), round(ty, 1))
            return EXACT_VOYNICHESE_ALIGNED_TRANSFORMS[norm_name]

        # Dynamic fallback based on parchment boundaries
        from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
        rgb_arr = np.array(scan_image)
        seg = GlyphSegmenter()
        py0, px0, py1, px1 = seg.detect_parchment_bounds(rgb_arr)
        pw = max(100, px1 - px0)
        ph = max(100, py1 - py0)

        sx = pw / xml_w
        sy = ph / xml_h
        tx = float(px0)
        ty = float(py0)

        return (round(float(sx), 4), round(float(sy), 4), round(float(tx), 1), round(float(ty), 1))

    def calibrate_affine_by_cross_correlation(
        self,
        folio_id: str,
        scan_image: Image.Image,
        xml_canvas: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Computes optimal affine registration parameters (sx, sy, tx, ty) between
        the Voynichese XML bounding boxes canvas and the high-resolution scan ink mask
        using 2-stage FFT cross-correlation and integral-image Nelder-Mead optimization.
        """
        import numpy as np
        from scipy.optimize import minimize
        from scipy.signal import fftconvolve
        from skimage.transform import rescale

        words = self.load_folio_words(folio_id)
        if not words:
            raise ValueError(f"No words found for folio {folio_id}")

        norm_name = self._normalize_folio_id(folio_id)
        if xml_canvas is None:
            if norm_name in self._folio_canvas_meta:
                xml_w, xml_h, _ = self._folio_canvas_meta[norm_name]
            else:
                xml_w, xml_h = 1090.0, 1500.0
        else:
            xml_w, xml_h = xml_canvas

        # Render XML word mask on XML canvas
        xml_canvas_arr = np.zeros((int(xml_h), int(xml_w)), dtype=np.float32)
        word_coords = []
        for w in words:
            wx, wy, ww, wh = float(w.x_xml), float(w.y_xml), float(w.width_xml), float(w.height_xml)
            word_coords.append((wx, wy, ww, wh))
            xml_canvas_arr[int(wy):int(wy + wh), int(wx):int(wx + ww)] = 1.0

        scan_gray = np.array(scan_image.convert("L"))
        ink = (scan_gray < 125).astype(np.float32)

        # Stage 1: Coarse search via 2D FFT cross-correlation on downsampled images (ds = 0.2)
        ds = 0.2
        ink_ds = rescale(ink, ds, anti_aliasing=False)
        xml_ds = rescale(xml_canvas_arr, ds, anti_aliasing=False)

        best_score = -1e9
        best_s = 2.10
        best_tx, best_ty = 100.0, 50.0

        for s in np.linspace(1.90, 2.25, 15):
            s_xml = rescale(xml_ds, s, anti_aliasing=False)
            if s_xml.shape[0] > ink_ds.shape[0] or s_xml.shape[1] > ink_ds.shape[1]:
                continue
            corr = fftconvolve(ink_ds, s_xml[::-1, ::-1], mode="valid")
            m_idx = np.unravel_index(np.argmax(corr), corr.shape)
            if corr[m_idx] > best_score:
                best_score = corr[m_idx]
                best_s = float(s)
                best_ty = float(m_idx[0] / ds)
                best_tx = float(m_idx[1] / ds)

        # Stage 2: Refined Nelder-Mead on integral image for O(1) word overlap lookup
        ii = np.pad(np.cumsum(np.cumsum(ink, axis=0), axis=1), ((1, 0), (1, 0)))
        H_s, W_s = ink.shape

        def loss(p):
            scale_v, off_x, off_y = p
            if scale_v < 1.80 or scale_v > 2.40:
                return 1e9
            tot = 0
            for wx, wy, ww, wh in word_coords:
                x0 = int(round(off_x + wx * scale_v))
                y0 = int(round(off_y + wy * scale_v))
                x1 = int(round(off_x + (wx + ww) * scale_v))
                y1 = int(round(off_y + (wy + wh) * scale_v))
                if 0 <= y0 < y1 <= H_s and 0 <= x0 < x1 <= W_s:
                    cnt = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
                    tot += cnt
            return -float(tot)

        init_p = [best_s, best_tx, best_ty]
        res = minimize(loss, init_p, method="Nelder-Mead", options={"maxiter": 100, "xatol": 0.005, "fatol": 1.0})
        s_opt, tx_opt, ty_opt = res.x
        return (round(float(s_opt), 3), round(float(s_opt), 3), round(float(tx_opt), 1), round(float(ty_opt), 1))

    def extract_aligned_words(
        self,
        folio_id: str,
        scan_image: Optional[Image.Image] = None,
        padding_px: int = 4,
        prefer_js: bool = True,
    ) -> List[RosettaWord]:
        """
        Extracts all words for a folio with high-res bounding boxes and cropped image patches.
        When the scan image is the native Voynichese glance image (636x900) or 2x upscale (1272x1800),
        it uses the .js pixel coordinates directly.
        When the scan image is a Yale HD scan (img_w > 1400), it falls back to the XML + affine approach.
        """
        if scan_image is None:
            scan_path = self._find_scan_path(folio_id)
            if scan_path and scan_path.exists():
                scan_image = Image.open(scan_path)
            else:
                if prefer_js:
                    try:
                        return self.load_folio_words_from_js(folio_id)
                    except Exception:
                        return self.load_folio_words(folio_id)
                else:
                    return self.load_folio_words(folio_id)

        img_w, img_h = scan_image.size
        is_voynichese_glance = (img_w <= 1400)

        use_js = False
        words: List[RosettaWord] = []
        if prefer_js and is_voynichese_glance:
            try:
                words = self.load_folio_words_from_js(folio_id)
                use_js = True
            except Exception:
                words = self.load_folio_words(folio_id)
                use_js = False
        else:
            words = self.load_folio_words(folio_id)
            use_js = False

        sx, sy, tx, ty = self.compute_scan_affine_transform(folio_id, scan_image, use_js=use_js)

        aligned_words: List[RosettaWord] = []
        for w in words:
            # Map word bbox to scan coordinates
            init_x0 = max(0, min(img_w - 1, int(round(tx + w.x_xml * sx))))
            init_y0 = max(0, min(img_h - 1, int(round(ty + w.y_xml * sy))))
            init_x1 = max(init_x0 + 1, min(img_w, int(round(tx + (w.x_xml + w.width_xml) * sx))))
            init_y1 = max(init_y0 + 1, min(img_h, int(round(ty + (w.y_xml + w.height_xml) * sy))))

            bbox_scan = (init_x0, init_y0, init_x1, init_y1)

            # Crop image patch
            if init_x1 > init_x0 and init_y1 > init_y0:
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
        clean = re.sub(r"\.(xml|jpg|jpeg|png|js)$", "", clean)
        match = re.match(r"f0*(\d+)([rv]\d*)?", clean)
        if match:
            num = match.group(1)
            side = match.group(2) or ""
            return f"f{num}{side}"
        return clean

    def _find_scan_path(self, folio_id: str) -> Optional[Path]:
        """
        Locates the corresponding scan image file in scans_dir or voynichese_scans.
        """
        norm_name = self._normalize_folio_id(folio_id)
        # Try both f001r and f1r formatting
        match = re.match(r"f(\d+)([rv]\d*)?", norm_name)
        candidates = []
        voynichese_scans = Path("data/scans/voynich/voynichese_scans")
        if match:
            num = int(match.group(1))
            side = match.group(2) or ""
            padded = f"f{num:03d}{side}"
            candidates.extend([
                voynichese_scans / f"{norm_name}.jpg",
                voynichese_scans / f"{padded}.jpg",
                self.scans_dir / f"{padded}.jpg",
                self.scans_dir / f"{norm_name}.jpg",
                self.scans_dir.parent / f"{padded}.jpg",
                self.scans_dir.parent / f"{norm_name}.jpg",
            ])
        else:
            candidates.extend([
                voynichese_scans / f"{norm_name}.jpg",
                self.scans_dir / f"{norm_name}.jpg",
            ])

        for p in candidates:
            if p.exists():
                return p
        return None
