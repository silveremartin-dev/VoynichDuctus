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
        Extracts clean text ink mask, strictly separating grayscale text ink
        from colored illustrations (green/ochre/blue/red/yellow paints) and large drawings.
        """
        rgb_arr = np.array(image.convert("RGB"))
        H, W, _ = rgb_arr.shape
        rgb_float = rgb_arr.astype(np.float32) / 255.0

        # Calculate Chroma and Brightness
        chroma = np.max(rgb_float, axis=2) - np.min(rgb_float, axis=2)
        gray = np.mean(rgb_float, axis=2)

        from skimage.color import rgb2hsv
        hsv = rgb2hsv(rgb_float)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 1. Detect background type and colored illustration pigments
        mean_page_sat = float(np.mean(sat))
        is_warm_parchment = mean_page_sat >= 0.18

        if is_warm_parchment:
            # Voynich / Medieval parchment:
            # Text ink is dark iron-gall. Illustrations are green foliage, blue water, or vivid red paint.
            is_green = (hue >= 0.18) & (hue <= 0.48) & (sat > 0.35) & (val > 0.40)
            is_blue = (hue >= 0.50) & (hue <= 0.75) & (sat > 0.35) & (val > 0.40)
            is_illustration_color = is_green | is_blue | ((chroma > 0.35) & (val > 0.45))
        else:
            # Codex Seraphinianus / Printed paper:
            # Text ink is strictly achromatic (chroma < 0.08). All colored pixels are illustrations!
            is_illustration_color = (chroma > 0.08)

        # 2. Local adaptive Sauvola thresholding for ink
        sauvola_mask = self.binarizer.binarize(image)
        sauvola_mask = self.binarizer.remove_small_artifacts(sauvola_mask, min_size=6)

        # Subtract colored illustration paints
        ink_mask = sauvola_mask & (~is_illustration_color)


        # 3. Strip page borders & binding margins (outer 8% on all edges)
        margin_y = max(20, int(H * 0.08))
        margin_x = max(20, int(W * 0.08))
        inner_mask = np.zeros_like(ink_mask)
        inner_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True

        # Cut off outer 12% corners (folio numbers, tears, binding shadows)
        corner_y = int(H * 0.12)
        corner_x = int(W * 0.12)
        inner_mask[:corner_y, :corner_x] = False
        inner_mask[:corner_y, W - corner_x:] = False
        inner_mask[H - corner_y:, :corner_x] = False
        inner_mask[H - corner_y:, W - corner_x:] = False

        ink_mask = ink_mask & inner_mask

        # 4. Remove large drawing connected components (drawings, frames, diagrams)
        labeled, num_features = label(ink_mask)
        props = regionprops(labeled)

        clean_text_mask = np.copy(ink_mask)
        for p in props:
            ph = p.bbox[2] - p.bbox[0]
            pw = p.bbox[3] - p.bbox[1]
            aspect = pw / max(1, ph)
            is_drawing = p.area > self.max_drawing_area or ph > self.max_glyph_height * 1.8 or pw > self.max_glyph_width * 2.5
            is_line_smear = (aspect > 4.5 and pw > 70) or (aspect < 0.18 and ph > 70)
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
        Enforces strict color neutrality (grayscale ink) and filamentary 1D stroke topology.
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
                fill_factor = p.area / max(1, gh * gw)

                # 1. Strict glyph dimension & scale bounds
                if p.area < 16 or gh < 11 or gw < 6:
                    continue
                if gh > self.max_glyph_height or gw > self.max_glyph_width:
                    continue
                if aspect > 4.0 or aspect < 0.18:
                    continue

                # 2. Filamentary 1D fractal stroke check (rejects solid textures, fur, ink blobs)
                # True written glyphs have open white loops / filiform centerlines (fill factor in [0.08, 0.50])
                if fill_factor < 0.08 or fill_factor > 0.50:
                    continue

                gy0 = y_off + wy0 + p.bbox[0]
                gx0 = x_off + wx0 + p.bbox[1]
                gy1 = y_off + wy0 + p.bbox[2]
                gx1 = x_off + wx0 + p.bbox[3]

                # 3. Strict Chromatic Pigment Discrimination (rejects colored illustration pixels)
                pad = 2
                cy0, cx0 = max(0, gy0 - pad), max(0, gx0 - pad)
                cy1, cx1 = min(page_rgb.shape[0], gy1 + pad), min(page_rgb.shape[1], gx1 + pad)
                g_mask = (labeled_w[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] == p.label)
                
                crop_patch_rgb = page_rgb[gy0:gy1, gx0:gx1].astype(np.float32) / 255.0
                if crop_patch_rgb.shape[0] == g_mask.shape[0] and crop_patch_rgb.shape[1] == g_mask.shape[1]:
                    ink_pixels = crop_patch_rgb[g_mask]
                    if ink_pixels.shape[0] > 0:
                        if subfolder == "seraphinianus":
                            # Strict achromatic check for Seraphinianus
                            chroma_ink = np.max(ink_pixels, axis=1) - np.min(ink_pixels, axis=1)
                            if np.mean(chroma_ink) > 0.08:
                                continue
                        else:
                            # Voynich / Medieval parchment: reject green/blue paint and bright vivid dyes
                            from skimage.color import rgb2hsv
                            ink_hsv = rgb2hsv(ink_pixels.reshape(-1, 1, 3))
                            is_paint_stroke = (ink_hsv[:, 0, 2] > 0.45) & (ink_hsv[:, 0, 1] > 0.35) & (((ink_hsv[:, 0, 0] >= 0.18) & (ink_hsv[:, 0, 0] <= 0.48)) | (ink_hsv[:, 0, 0] >= 0.50))
                            if np.mean(is_paint_stroke) > 0.35:
                                continue



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

                glyph_id = f"{page_id}_{word_id}_G{glyph_idx:02d}"

                # Save crops and SVGs if output_dir provided
                png_rel, svg_rel, svg_str = "", "", ""
                if output_dir:
                    (output_dir / subfolder / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
                    (output_dir / subfolder / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

                    png_filename = f"{glyph_id}.png"
                    svg_filename = f"{glyph_id}.svg"
                    png_path = output_dir / subfolder / "glyphs" / "png" / png_filename
                    svg_path = output_dir / subfolder / "glyphs" / "svg" / svg_filename

                    # Save crop patch
                    save_patch = page_rgb[cy0:cy1, cx0:cx1]
                    Image.fromarray(save_patch).save(png_path)

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
                    "fill_factor": round(float(fill_factor), 3),
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
        Saves full page image for interactive explorer overlay.
        """
        clean_mask, page_rgb = self.extract_clean_page_mask(image)
        lines = self.line_segmenter.segment_lines(clean_mask, auto_isolate=False)

        if max_lines:
            lines = lines[:max_lines]

        page_img_rel = ""
        if output_dir:
            pages_dir = output_dir / subfolder / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = pages_dir / f"{page_id}.jpg"
            image.save(page_img_path, quality=85)
            page_img_rel = f"{subfolder}/pages/{page_id}.jpg"

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
            "page_img_rel": page_img_rel,
            "image_width": image.width,
            "image_height": image.height,
            "line_count": len(lines),
            "glyph_count": len(all_page_glyphs),
            "lines": lines,
            "glyphs": all_page_glyphs
        }

