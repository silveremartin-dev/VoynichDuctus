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
        max_glyph_height: int = 220,
        min_glyph_width: int = 6,
        max_glyph_width: int = 220,
        min_glyph_area: int = 15,
        max_drawing_area: int = 6000
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

    def extract_clean_page_mask(self, image: Image.Image, subfolder: str = "voynich") -> Tuple[np.ndarray, np.ndarray]:
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

        from scipy.ndimage import binary_dilation

        if subfolder == "voynich":
            # Voynich / Medieval parchment:
            # Text ink is dark iron-gall. Illustrations are green foliage, blue water, or vivid red/yellow paint.
            is_green = (hue >= 0.14) & (hue <= 0.50) & (sat > 0.15) & (val > 0.18)
            is_blue = (hue >= 0.50) & (hue <= 0.78) & (sat > 0.15) & (val > 0.18)
            is_red = ((hue >= 0.85) | (hue <= 0.06)) & (sat > 0.22) & (val > 0.22)
            is_yellow_ochre = (hue >= 0.08) & (hue <= 0.16) & (sat > 0.30) & (val > 0.40)
            is_illustration_color = is_green | is_blue | is_red | is_yellow_ochre | ((chroma > 0.26) & (val > 0.30))
            is_illustration_color = binary_dilation(is_illustration_color, iterations=5)
        else:
            # Codex Seraphinianus / Printed paper:
            # Text ink is strictly achromatic (chroma < 0.035, sat < 0.04). All colored pixels are illustrations!
            is_illustration_color = (chroma > 0.035) | (sat > 0.04)
            is_illustration_color = binary_dilation(is_illustration_color, iterations=6)

        # 2. Local adaptive Sauvola thresholding for ink
        sauvola_mask = self.binarizer.binarize(image)
        sauvola_mask = self.binarizer.remove_small_artifacts(sauvola_mask, min_size=8)

        # Subtract colored illustration paints
        ink_mask = sauvola_mask & (~is_illustration_color)

        # 3. Strip page borders & binding margins (Keep right margin wide so letters are NOT cut off)
        if subfolder == "seraphinianus":
            margin_y = max(8, int(H * 0.020))
            margin_x = max(6, int(W * 0.015))
            inner_mask = np.zeros_like(ink_mask)
            inner_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True
        else:
            margin_y = max(10, int(H * 0.020))
            margin_x = max(6, int(W * 0.008))  # Only 0.8% margin: preserves all right-margin text!
            inner_mask = np.zeros_like(ink_mask)
            inner_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True
            # Cut off outer 2% corners (tears, corner spots)
            corner_y = int(H * 0.02)
            corner_x = int(W * 0.02)
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
            is_drawing = p.area > self.max_drawing_area or ph > self.max_glyph_height * 2.2 or pw > self.max_glyph_width * 3.0
            is_line_smear = (aspect > 6.0 and pw > 120) or (aspect < 0.12 and ph > 120)
            if is_drawing or is_line_smear:
                clean_text_mask[labeled == p.label] = False

        return clean_text_mask, rgb_arr

    def assemble_word_into_glyphs(self, w_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Assembles connected components within a word into intact multi-stroke glyphs.
        Groups vertically overlapping or tightly adjacent sub-strokes (e.g., bowl + ascender,
        gallows bar + leg, diacritic + minim) and recursively splits wide cursive ligatures.
        """
        labeled_w, num_w = label(w_img)
        if num_w == 0:
            return []

        props = sorted(regionprops(labeled_w), key=lambda p: (p.bbox[1] + p.bbox[3]) / 2.0)
        
        # 1. Filter out micro noise specks below quill nib thickness (minimum ~1.5mm / 8px)
        valid_comps = []
        for p in props:
            ph = p.bbox[2] - p.bbox[0]
            pw = p.bbox[3] - p.bbox[1]
            if p.area >= 10 and ph >= 6 and pw >= 3:
                valid_comps.append(p)

        if not valid_comps:
            return []

        # 2. Cluster components that belong to the same character
        clusters: List[List[Any]] = []
        for comp in valid_comps:
            cy0, cx0, cy1, cx1 = comp.bbox
            merged = False
            for cl in clusters:
                cl_x0 = min(c.bbox[1] for c in cl)
                cl_x1 = max(c.bbox[3] for c in cl)
                cl_y0 = min(c.bbox[0] for c in cl)
                cl_y1 = max(c.bbox[2] for c in cl)

                overlap = max(0, min(cl_x1, cx1) - max(cl_x0, cx0))
                gap = max(0, max(cx0 - cl_x1, cl_x0 - cx1))
                
                comb_w = max(cl_x1, cx1) - min(cl_x0, cx0)
                comb_h = max(cl_y1, cy1) - min(cl_y0, cy0)

                # Merge if overlapping or tight horizontal gap <= 4px
                if (overlap > 0 or gap <= 4) and comb_w <= max(45, int(comb_h * 1.35)):
                    cl.append(comp)
                    merged = True
                    break
            if not merged:
                clusters.append([comp])

        # 3. Recursive Ligature Splitter (conservative so full single characters are preserved)
        def split_ligature_mask(mask: np.ndarray, y0: int, x0: int) -> List[Dict[str, Any]]:
            gh, gw = mask.shape
            area = int(np.sum(mask))
            if area < 20 or gh < 12 or gw < 5:
                return []

            # Only split very wide multi-character ligatures
            if gw > 1.70 * gh and gw >= 38:
                col_proj = np.sum(mask, axis=0)
                mid_start = int(0.22 * gw)
                mid_end = int(0.78 * gw)
                if mid_end > mid_start:
                    min_col_rel = int(np.argmin(col_proj[mid_start:mid_end]))
                    min_col_idx = mid_start + min_col_rel
                    min_val = col_proj[min_col_idx]
                    max_left = np.max(col_proj[:min_col_idx]) if min_col_idx > 0 else 1
                    max_right = np.max(col_proj[min_col_idx:]) if min_col_idx < gw else 1
                    max_peak = max(max_left, max_right)

                    if max_peak > 0 and (min_val / max_peak) <= 0.35:
                        mask1 = mask[:, :min_col_idx]
                        mask2 = mask[:, min_col_idx:]
                        res1 = split_ligature_mask(mask1, y0, x0)
                        res2 = split_ligature_mask(mask2, y0, x0 + min_col_idx)
                        if res1 and res2:
                            return res1 + res2
                        elif res1:
                            return res1
                        elif res2:
                            return res2

            return [{
                "mask": mask,
                "bbox_local": (y0, x0, y0 + gh, x0 + gw),
                "area": area
            }]

        # 4. Create composite masks for each cluster and split ligatures
        assembled_glyphs = []
        for cl in clusters:
            gy0 = min(c.bbox[0] for c in cl)
            gy1 = max(c.bbox[2] for c in cl)
            gx0 = min(c.bbox[1] for c in cl)
            gx1 = max(c.bbox[3] for c in cl)
            
            gh = gy1 - gy0
            gw = gx1 - gx0
            
            # Build combined mask
            g_mask = np.zeros((gh, gw), dtype=bool)
            for c in cl:
                c_mask = (labeled_w[c.bbox[0]:c.bbox[2], c.bbox[1]:c.bbox[3]] == c.label)
                off_y = c.bbox[0] - gy0
                off_x = c.bbox[1] - gx0
                g_mask[off_y:off_y + (c.bbox[2] - c.bbox[0]), off_x:off_x + (c.bbox[3] - c.bbox[1])] |= c_mask

            split_res = split_ligature_mask(g_mask, gy0, gx0)
            assembled_glyphs.extend(split_res)

        return sorted(assembled_glyphs, key=lambda g: g["bbox_local"][1])

    def extract_glyphs_from_line(
        self,
        line_mask: np.ndarray,
        line_offset: Tuple[int, int],
        page_rgb: np.ndarray,
        page_id: str,
        line_id: str,
        output_dir: Optional[Path] = None,
        subfolder: str = "voynich"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Segments a single text line into verified words, and words into intact assembled glyphs with SVG ductus.
        Returns (line_glyphs, line_words).
        """
        y_off, x_off = line_offset
        words = self.line_segmenter.segment_words(line_mask, line_offset=(0, 0), space_gap_min=10)

        line_glyphs = []
        line_words = []
        for word_idx, w in enumerate(words):
            w_img = w["image"]
            wy0, wx0, wy1, wx1 = w["bbox"]
            word_id = f"{line_id}_W{word_idx:02d}"

            # Word absolute coordinates on page
            abs_wy0 = y_off + wy0
            abs_wx0 = x_off + wx0
            abs_wy1 = y_off + wy1
            abs_wx1 = x_off + wx1

            # Validate word lexical sanity
            w_w = abs_wx1 - abs_wx0
            w_h = abs_wy1 - abs_wy0
            if w_w < 14 or w_h < 12 or np.sum(w_img) < 30:
                continue

            # Save word crop with generous padding (6px)
            word_png_rel = ""
            if output_dir:
                (output_dir / subfolder / "words" / "png").mkdir(parents=True, exist_ok=True)
                w_pad = 6
                cwy0, cwx0 = max(0, abs_wy0 - w_pad), max(0, abs_wx0 - w_pad)
                cwy1, cwx1 = min(page_rgb.shape[0], abs_wy1 + w_pad), min(page_rgb.shape[1], abs_wx1 + w_pad)
                word_patch = page_rgb[cwy0:cwy1, cwx0:cwx1]
                word_enhanced = self.normalizer.enhance_contrast_and_sharpness(
                    word_patch, contrast_gain=1.35, unsharp_radius=1.0, unsharp_amount=1.6
                )
                word_png_filename = f"{page_id}_{word_id}.png"
                word_png_path = output_dir / subfolder / "words" / "png" / word_png_filename
                Image.fromarray(word_enhanced).save(word_png_path)
                word_png_rel = f"{subfolder}/words/png/{word_png_filename}"

            line_words.append({
                "word_id": word_id,
                "line_id": line_id,
                "bbox": (int(abs_wy0), int(abs_wx0), int(abs_wy1), int(abs_wx1)),
                "word_png_rel": word_png_rel
            })

            # Assemble intact glyphs from word
            assembled_components = self.assemble_word_into_glyphs(w_img)

            glyph_idx = 0
            for item in assembled_components:
                g_mask = item["mask"]
                gh, gw = g_mask.shape
                bbox_loc = item["bbox_local"]
                aspect = gw / max(1, gh)
                fill_factor = item["area"] / max(1, gh * gw)

                # 1. Strict glyph dimension & scale bounds (quill stroke physical bounds)
                if item["area"] < 22 or gh < 14 or gw < 6:
                    continue
                if gh > self.max_glyph_height or gw > self.max_glyph_width:
                    continue
                if aspect > 3.2 or aspect < 0.18:
                    continue

                # 2. Filamentary 1D stroke check (rejects solid textures, blobs, and dark areas)
                if fill_factor < 0.05 or fill_factor > 0.40:
                    continue

                gy0 = y_off + wy0 + bbox_loc[0]
                gx0 = x_off + wx0 + bbox_loc[1]
                gy1 = y_off + wy0 + bbox_loc[2]
                gx1 = x_off + wx0 + bbox_loc[3]

                # 3. Strict Chromatic Pigment Discrimination & Brightness Verification
                crop_pad = 6  # Generous padding around the stroke
                cy0, cx0 = max(0, gy0 - crop_pad), max(0, gx0 - crop_pad)
                cy1, cx1 = min(page_rgb.shape[0], gy1 + crop_pad), min(page_rgb.shape[1], gx1 + crop_pad)
                
                raw_patch = page_rgb[cy0:cy1, cx0:cx1]
                if raw_patch.shape[0] < 8 or raw_patch.shape[1] < 6:
                    continue

                # Reject mostly black/dark blobs (authentic parchment is bright, >115/255)
                if np.mean(raw_patch) < 115:
                    continue

                crop_patch_rgb = page_rgb[gy0:gy1, gx0:gx1].astype(np.float32) / 255.0
                if crop_patch_rgb.shape[0] == g_mask.shape[0] and crop_patch_rgb.shape[1] == g_mask.shape[1]:
                    ink_pixels = crop_patch_rgb[g_mask]
                    if ink_pixels.shape[0] > 0:
                        if subfolder == "seraphinianus":
                            chroma_ink = np.max(ink_pixels, axis=1) - np.min(ink_pixels, axis=1)
                            if np.mean(chroma_ink) > 0.06:
                                continue
                        else:
                            from skimage.color import rgb2hsv
                            ink_hsv = rgb2hsv(ink_pixels.reshape(-1, 1, 3))
                            is_paint_stroke = (
                                ((ink_hsv[:, 0, 0] >= 0.14) & (ink_hsv[:, 0, 0] <= 0.50) & (ink_hsv[:, 0, 1] > 0.14)) |
                                ((ink_hsv[:, 0, 0] >= 0.50) & (ink_hsv[:, 0, 0] <= 0.78) & (ink_hsv[:, 0, 1] > 0.14)) |
                                (((ink_hsv[:, 0, 0] >= 0.85) | (ink_hsv[:, 0, 0] <= 0.06)) & (ink_hsv[:, 0, 1] > 0.20) & (ink_hsv[:, 0, 2] > 0.25)) |
                                ((ink_hsv[:, 0, 0] >= 0.08) & (ink_hsv[:, 0, 0] <= 0.16) & (ink_hsv[:, 0, 1] > 0.28))
                            )
                            if np.mean(is_paint_stroke) > 0.08:
                                continue

                # Vectorize glyph ductus
                try:
                    skel, widths = self.skel_engine.extract_skeleton(g_mask)
                    pixel_graph = self.graph_extractor.build_pixel_graph(skel, widths)
                    raw_strokes = self.graph_extractor.decompose_into_strokes(pixel_graph)
                    ordered_strokes = self.resolver.resolve_and_order_strokes(raw_strokes) if raw_strokes else []
                except Exception:
                    ordered_strokes = []

                if not ordered_strokes or len(ordered_strokes) > 6:
                    continue

                total_stroke_len = sum(len(s.get("points", [])) for s in ordered_strokes)
                if total_stroke_len < 10:
                    continue

                # Unique hierarchical glyph identifier: Page_Line_Word_G##
                glyph_id = f"{page_id}_{line_id}_{word_id}_G{glyph_idx:02d}"

                # Save crops and SVGs if output_dir provided
                png_rel, svg_rel, svg_str = "", "", ""
                if output_dir:
                    (output_dir / subfolder / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
                    (output_dir / subfolder / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

                    png_filename = f"{glyph_id}.png"
                    svg_filename = f"{glyph_id}.svg"
                    png_path = output_dir / subfolder / "glyphs" / "png" / png_filename
                    svg_path = output_dir / subfolder / "glyphs" / "svg" / svg_filename

                    # Save crop patch with sharp stroke contrast and generous padding
                    save_patch = self.normalizer.enhance_contrast_and_sharpness(raw_patch, contrast_gain=1.35, unsharp_radius=1.0, unsharp_amount=1.6)
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
                    "word_bbox": (int(abs_wy0), int(abs_wx0), int(abs_wy1), int(abs_wx1)),
                    "word_png_rel": word_png_rel,
                    "bbox": (int(gy0), int(gx0), int(gy1), int(gx1)),
                    "height": int(gh),
                    "width": int(gw),
                    "area": int(item["area"]),
                    "fill_factor": round(float(fill_factor), 3),
                    "stroke_count": len(ordered_strokes),
                    "strokes": ordered_strokes,
                    "png_rel": png_rel,
                    "svg_rel": svg_rel,
                    "svg_content": svg_str
                })
                glyph_idx += 1

        return line_glyphs, line_words

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
        clean_mask, page_rgb = self.extract_clean_page_mask(image, subfolder=subfolder)
        lines = self.line_segmenter.segment_lines(clean_mask, auto_isolate=False)

        if max_lines:
            lines = lines[:max_lines]

        page_img_rel = ""
        if output_dir:
            pages_dir = output_dir / subfolder / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = pages_dir / f"{page_id}.jpg"
            image.save(page_img_path, quality=88)
            page_img_rel = f"{subfolder}/pages/{page_id}.jpg"

        all_page_glyphs = []
        all_page_words = []
        clean_lines = []
        for line in lines:
            l_img = line["image"]
            ly0, lx0, ly1, lx1 = line["bbox"]
            clean_lines.append({
                "line_id": line["line_id"],
                "bbox": (int(ly0), int(lx0), int(ly1), int(lx1))
            })
            l_glyphs, l_words = self.extract_glyphs_from_line(
                line_mask=l_img,
                line_offset=(ly0, lx0),
                page_rgb=page_rgb,
                page_id=page_id,
                line_id=line["line_id"],
                output_dir=output_dir,
                subfolder=subfolder
            )
            all_page_glyphs.extend(l_glyphs)
            all_page_words.extend(l_words)

        return {
            "page_id": page_id,
            "page_img_rel": page_img_rel,
            "image_width": image.width,
            "image_height": image.height,
            "line_count": len(lines),
            "word_count": len(all_page_words),
            "glyph_count": len(all_page_glyphs),
            "lines": clean_lines,
            "words": all_page_words,
            "glyphs": all_page_glyphs
        }

