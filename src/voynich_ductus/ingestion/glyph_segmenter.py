"""
Hierarchical Glyph Segmentation and Extraction Engine.
Extracts individual characters/grapheme primitives from manuscript pages,
filtering out drawings, illustrations, and page borders.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage.measure import regionprops

from voynich_ductus.ingestion.color_normalizer import ColorIlluminationNormalizer
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter


class GlyphSegmenter:
    """
    Decomposes full manuscript pages into hierarchical text structures:
    Page -> Text Blocks -> Lines -> Words -> Isolated Glyphs (with Vector Ductus).
    """

    def __init__(
        self,
        min_glyph_height: int = 10,
        max_glyph_height: int = 95,
        min_glyph_width: int = 6,
        max_glyph_width: int = 150,
        min_glyph_area: int = 15,
        max_drawing_area: int = 3000
    ):
        self.min_glyph_height = min_glyph_height
        self.max_glyph_height = max_glyph_height
        self.min_glyph_width = min_glyph_width
        self.max_glyph_width = max_glyph_width
        self.min_glyph_area = min_glyph_area
        self.max_drawing_area = max_drawing_area

        self.normalizer = ColorIlluminationNormalizer()
        self.binarizer = Binarizer(method="sauvola", window_size=29, k=0.20)
        self.line_segmenter = LineSegmenter(min_line_pitch=28, min_line_height=14, min_word_width=12, min_word_area=25)
        self.skel_engine = Skeletonizer(method="medial_axis")
        self.graph_extractor = StrokeGraphExtractor()
        self.resolver = JunctionResolver(right_handed_prior=True)

    def extract_clean_page_mask(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts clean text ink mask, removing chromatic paints (green/ochre/blue)
        and large drawing/border connected components.
        """
        rgb_arr = np.array(image.convert("RGB"))
        H, W, _ = rgb_arr.shape

        # 1. Local adaptive Sauvola thresholding
        sauvola_mask = self.binarizer.binarize(image)
        sauvola_mask = self.binarizer.remove_small_artifacts(sauvola_mask, min_size=6)

        # 2. Chromatic pigment detection (green/blue/ochre illustrations)
        flat_rgb = self.normalizer.flatten_illumination_fast(self.normalizer.to_rgb_array(image))
        chroma = np.max(flat_rgb, axis=2) - np.min(flat_rgb, axis=2)
        from skimage.color import rgb2hsv
        hsv = rgb2hsv(flat_rgb)
        is_green = (hsv[:, :, 0] >= 0.15) & (hsv[:, :, 0] <= 0.48) & (chroma > 0.12)
        is_blue = (hsv[:, :, 0] >= 0.50) & (hsv[:, :, 0] <= 0.75) & (chroma > 0.12)
        is_paint = is_green | is_blue | (chroma > 0.25)

        # Subtract colored paint from text ink mask
        ink_mask = sauvola_mask & (~is_paint)

        # 3. Strip page borders & binding margins (outer 8% on all edges)
        margin_y = max(20, int(H * 0.08))
        margin_x = max(20, int(W * 0.08))
        inner_mask = np.zeros_like(ink_mask)
        inner_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True

        # Cut off outer 12% corners (where folio marks, creases, or thumb marks reside)
        corner_y = int(H * 0.12)
        corner_x = int(W * 0.12)
        inner_mask[:corner_y, :corner_x] = False
        inner_mask[:corner_y, W - corner_x:] = False
        inner_mask[H - corner_y:, :corner_x] = False
        inner_mask[H - corner_y:, W - corner_x:] = False

        ink_mask = ink_mask & inner_mask

        # 4. Remove large drawing connected components (drawings, frames, large diagrams)
        labeled, num_features = label(ink_mask)
        props = regionprops(labeled)

        clean_text_mask = np.copy(ink_mask)
        for p in props:
            ph = p.bbox[2] - p.bbox[0]
            pw = p.bbox[3] - p.bbox[1]
            aspect = pw / max(1, ph)
            # Rejection criteria:
            # - Too large area (drawings/diagrams)
            # - Height or width exceeding text line scale
            # - Extreme aspect ratio (horizontal rules or vertical crease lines)
            is_drawing = p.area > self.max_drawing_area or ph > self.max_glyph_height * 1.8 or pw > self.max_glyph_width * 2.5
            is_line_smear = (aspect > 5.0 and pw > 80) or (aspect < 0.15 and ph > 80)
            if is_drawing or is_line_smear:
                clean_text_mask[labeled == p.label] = False

        return clean_text_mask, rgb_arr

    def extract_glyphs_from_line(
        self,
        line_mask: np.ndarray,
        line_offset: Tuple[int, int],
        page_rgb: np.ndarray,
        page_id: str,
        line_id: str,
        output_dir: Optional[Path] = None,
        subfolder: str = "voynich"
    ) -> List[Dict[str, Any]]:
        """
        Segments a single text line into words, and words into individual glyphs with SVG ductus.
        """
        y_off, x_off = line_offset
        words = self.line_segmenter.segment_words(line_mask, line_offset=(0, 0), space_gap_min=10)

        line_glyphs = []
        for word_idx, w in enumerate(words):
            w_img = w["image"]
            wy0, wx0, wy1, wx1 = w["bbox"]
            word_id = f"{line_id}_W{word_idx:02d}"

            # Extract connected components in the word
            labeled_w, num_w = label(w_img)
            w_props = sorted(regionprops(labeled_w), key=lambda p: p.bbox[1])  # Left-to-right order

            glyph_idx = 0
            for p in w_props:
                gh = p.bbox[2] - p.bbox[0]
                gw = p.bbox[3] - p.bbox[1]
                aspect = gw / max(1, gh)
                density = p.area / max(1, gh * gw)

                # Strict glyph dimension and density validation
                if p.area < self.min_glyph_area or gh < self.min_glyph_height or gw < self.min_glyph_width:
                    continue
                if gh > self.max_glyph_height or gw > self.max_glyph_width:
                    continue
                if aspect > 4.2 or aspect < 0.18 or density < 0.10:
                    continue


                glyph_id = f"{page_id}_{word_id}_G{glyph_idx:02d}"
                gy0 = y_off + wy0 + p.bbox[0]
                gx0 = x_off + wx0 + p.bbox[1]
                gy1 = y_off + wy0 + p.bbox[2]
                gx1 = x_off + wx0 + p.bbox[3]

                g_mask = (labeled_w[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] == p.label)

                # Vectorize glyph ductus
                try:
                    skel, widths = self.skel_engine.extract_skeleton(g_mask)
                    pixel_graph = self.graph_extractor.build_pixel_graph(skel, widths)
                    raw_strokes = self.graph_extractor.decompose_into_strokes(pixel_graph)
                    ordered_strokes = self.resolver.resolve_and_order_strokes(raw_strokes) if raw_strokes else []
                except Exception:
                    ordered_strokes = []

                if not ordered_strokes:
                    continue

                # Save crops and SVGs if output_dir provided
                png_rel, svg_rel, svg_str = "", "", ""
                if output_dir:
                    (output_dir / subfolder / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
                    (output_dir / subfolder / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

                    png_filename = f"{glyph_id}.png"
                    svg_filename = f"{glyph_id}.svg"
                    png_path = output_dir / subfolder / "glyphs" / "png" / png_filename
                    svg_path = output_dir / subfolder / "glyphs" / "svg" / svg_filename

                    # Crop patch from original RGB page image (with 2px margin)
                    pad = 2
                    cy0, cx0 = max(0, gy0 - pad), max(0, gx0 - pad)
                    cy1, cx1 = min(page_rgb.shape[0], gy1 + pad), min(page_rgb.shape[1], gx1 + pad)
                    crop_rgb = page_rgb[cy0:cy1, cx0:cx1]
                    Image.fromarray(crop_rgb).save(png_path)

                    gh_m, gw_m = g_mask.shape
                    svg_str = VectorExporter.to_svg(ordered_strokes, width=gw_m, height=gh_m, output_path=svg_path)

                    png_rel = f"{subfolder}/glyphs/png/{png_filename}"
                    svg_rel = f"{subfolder}/glyphs/svg/{svg_filename}"

                line_glyphs.append({
                    "glyph_id": glyph_id,
                    "page_id": page_id,
                    "line_id": line_id,
                    "word_id": word_id,
                    "bbox": (int(gy0), int(gx0), int(gy1), int(gx1)),
                    "height": int(gh),
                    "width": int(gw),
                    "area": int(p.area),
                    "stroke_count": len(ordered_strokes),
                    "strokes": ordered_strokes,
                    "png_rel": png_rel,
                    "svg_rel": svg_rel,
                    "svg_content": svg_str
                })
                glyph_idx += 1

        return line_glyphs

    def extract_page_glyphs(
        self,
        page_id: str,
        image: Image.Image,
        output_dir: Optional[Path] = None,
        subfolder: str = "voynich",
        max_lines: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes an entire manuscript page, extracting all lines, words, and individual glyphs.
        """
        clean_mask, page_rgb = self.extract_clean_page_mask(image)
        lines = self.line_segmenter.segment_lines(clean_mask, auto_isolate=False)

        if max_lines:
            lines = lines[:max_lines]

        all_page_glyphs = []
        for line in lines:
            l_img = line["image"]
            ly0, lx0, ly1, lx1 = line["bbox"]
            l_glyphs = self.extract_glyphs_from_line(
                line_mask=l_img,
                line_offset=(ly0, lx0),
                page_rgb=page_rgb,
                page_id=page_id,
                line_id=line["line_id"],
                output_dir=output_dir,
                subfolder=subfolder
            )
            all_page_glyphs.extend(l_glyphs)

        return {
            "page_id": page_id,
            "line_count": len(lines),
            "glyph_count": len(all_page_glyphs),
            "lines": lines,
            "glyphs": all_page_glyphs
        }
