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
from voynich_ductus.vectorizer.calligraphic_tracker import CalligraphicVectorizer
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
        self.calligraphic_vectorizer = CalligraphicVectorizer(nib_angle_deg=40.0)
        self.skel_engine = Skeletonizer(method="medial_axis")
        self.graph_extractor = StrokeGraphExtractor()
        self.resolver = JunctionResolver(right_handed_prior=True)

    def detect_parchment_bounds(self, rgb_arr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detects genuine parchment / paper boundary inside the scan, strictly removing
        outer black scanner beds, binding folds, background frames, and margin rulers.
        """
        H, W, _ = rgb_arr.shape
        
        # Downsample for fast, robust parchment bounding box detection
        scale = 4
        rgb_sub = rgb_arr[::scale, ::scale].astype(np.float32) / 255.0
        gray = np.mean(rgb_sub, axis=2)
        r = rgb_sub[:, :, 0]
        b = rgb_sub[:, :, 2]
        is_parchment = (gray > 0.35) & (r >= b - 0.04) & (gray < 0.98)

        # Morphological opening/closing to remove small dust outside
        from scipy.ndimage import binary_opening, binary_closing
        is_parchment = binary_opening(is_parchment, iterations=2)
        is_parchment = binary_closing(is_parchment, iterations=2)

        labeled, num_features = label(is_parchment)
        if num_features == 0:
            return (int(H * 0.02), int(W * 0.02), int(H * 0.98), int(W * 0.995))

        props = regionprops(labeled)
        largest = max(props, key=lambda p: p.area)
        py0, px0, py1, px1 = [int(v * scale) for v in largest.bbox]

        # Gentle margins: preserve full right margin to never clip ending 1-3 characters of lines
        buf_y = max(8, int((py1 - py0) * 0.015))
        buf_x0 = max(8, int((px1 - px0) * 0.015))
        return (min(H - 1, py0 + buf_y), min(W - 1, px0 + buf_x0), max(0, py1 - buf_y), min(W, px1 + 4))

    def extract_clean_page_mask(self, image: Image.Image, subfolder: str = "voynich") -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Extracts clean text ink mask, strictly separating grayscale text ink
        from colored illustrations (green/ochre/blue/red/yellow paints) and large drawings.
        """
        rgb_arr = np.array(image.convert("RGB"))
        H, W, _ = rgb_arr.shape
        rgb_float = rgb_arr.astype(np.float32) / 255.0

        # 1. Detect genuine parchment canvas bounding box
        if subfolder == "seraphinianus":
            # In Seraphinianus, use full page width/height with minimal safety border
            parchment_mask = np.ones((H, W), dtype=bool)
            parchment_mask[:4, :] = False
            parchment_mask[-4:, :] = False
            parchment_mask[:, :4] = False
            parchment_mask[:, -4:] = False
        else:
            py0, px0, py1, px1 = self.detect_parchment_bounds(rgb_arr)
            parchment_mask = np.zeros((H, W), dtype=bool)
            parchment_mask[py0:py1, px0:px1] = True

        # 2. Local adaptive Sauvola thresholding for ink
        sauvola_mask = self.binarizer.binarize(image)
        sauvola_mask = self.binarizer.remove_small_artifacts(sauvola_mask, min_size=8)
        ink_candidates = sauvola_mask & parchment_mask

        # 3. Component-level color and geometry discrimination
        from skimage.color import rgb2hsv
        hsv = rgb2hsv(rgb_float)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        chroma = np.max(rgb_float, axis=2) - np.min(rgb_float, axis=2)

        labeled, num_features = label(ink_candidates)
        props = regionprops(labeled)

        clean_text_mask = np.zeros((H, W), dtype=bool)

        for p in props:
            ph = p.bbox[2] - p.bbox[0]
            pw = p.bbox[3] - p.bbox[1]
            aspect = pw / max(1, ph)

            # Geometry filtering: reject giant drawings, full plate frames, or border smears
            max_area = 15000 if subfolder == "seraphinianus" else self.max_drawing_area
            if p.area > max_area:
                continue
            if ph > self.max_glyph_height * 3.5:
                continue

            # Pigment and illustration color filtering on component ink pixels
            c_mask = (labeled[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] == p.label)
            c_sat = sat[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]][c_mask]
            c_hue = hue[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]][c_mask]
            c_val = val[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]][c_mask]
            c_chroma = chroma[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]][c_mask]

            if len(c_sat) > 0:
                if subfolder == "seraphinianus":
                    # In Seraphinianus: strict dark quill ink vs colorful illustration filtering
                    # Dark scribal ink has sat < 0.18, chroma < 0.12, val < 0.48
                    if np.mean(c_sat) > 0.16 or np.mean(c_chroma) > 0.11 or np.mean(c_val) > 0.55:
                        continue
                    if np.percentile(c_sat, 75) > 0.22:
                        continue
                    # Reject specific saturated pigment hues (green, red, orange/yellow, blue)
                    is_pigment = (
                        ((c_hue >= 0.16) & (c_hue <= 0.48) & (c_sat > 0.10)) |  # Green
                        (((c_hue >= 0.82) | (c_hue <= 0.08)) & (c_sat > 0.10)) |  # Red / Magenta
                        ((c_hue >= 0.08) & (c_hue <= 0.16) & (c_sat > 0.12)) |  # Yellow / Orange
                        ((c_hue >= 0.48) & (c_hue <= 0.82) & (c_sat > 0.10))    # Blue / Indigo
                    )
                    if np.mean(is_pigment) > 0.04:
                        continue
                else:
                    # Voynich: reject colored paint washes
                    is_paint = (
                        ((c_hue >= 0.14) & (c_hue <= 0.50) & (c_sat > 0.16)) |  # Green
                        ((c_hue >= 0.50) & (c_hue <= 0.80) & (c_sat > 0.16)) |  # Blue
                        (((c_hue >= 0.85) | (c_hue <= 0.06)) & (c_sat > 0.18)) |  # Red
                        (c_chroma > 0.18)  # Saturated washes
                    )
                    if np.mean(is_paint) > 0.20 and p.area > 25:
                        continue

            clean_text_mask[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] |= c_mask

        # Check if text lines run vertically across the page (sideways landscape plate)
        was_rotated = False
        if subfolder == "seraphinianus":
            row_proj = np.sum(clean_text_mask, axis=1)
            col_proj = np.sum(clean_text_mask, axis=0)
            var_row = np.var(row_proj)
            var_col = np.var(col_proj)
            if var_col > 2.2 * max(1e-5, var_row):
                # Rotate 270° (90° clockwise) to bring text lines into standard horizontal reading orientation
                clean_text_mask = np.rot90(clean_text_mask, k=3)
                rgb_arr = np.rot90(rgb_arr, k=3)
                was_rotated = True

        return clean_text_mask, rgb_arr, was_rotated

    def assemble_word_into_glyphs(
        self,
        w_img: np.ndarray,
        expected_glyph_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Assembles connected components within a word into intact multi-stroke glyphs.
        Groups vertically overlapping or tightly adjacent sub-strokes (e.g., bowl + ascender,
        gallows bar + leg, diacritic + minim) and recursively splits wide cursive ligatures.
        Optionally uses expected_glyph_count as a calibrated prior from reference transcriptions.
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

        # 2. Cluster components that belong to the same character (e.g. loops, crossbars, accents)
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

                # Merge if overlapping or tight horizontal gap <= 6px (preserving characters up to 65px)
                if (overlap > 0 or gap <= 6) and comb_w <= max(65, int(comb_h * 1.6)):
                    cl.append(comp)
                    merged = True
                    break
            if not merged:
                clusters.append([comp])

        # 3. Ligature Splitter (strictly preserves single intact characters)
        def split_ligature_mask(mask: np.ndarray, y0: int, x0: int) -> List[Dict[str, Any]]:
            gh, gw = mask.shape
            area = int(np.sum(mask))
            if area < 20 or gh < 12 or gw < 5:
                return []

            # Only split extraordinarily wide multi-character ligatures
            if gw > 2.2 * gh and gw >= 60:
                col_proj = np.sum(mask, axis=0)
                mid_start = int(0.25 * gw)
                mid_end = int(0.75 * gw)
                if mid_end > mid_start:
                    min_col_rel = int(np.argmin(col_proj[mid_start:mid_end]))
                    min_col_idx = mid_start + min_col_rel
                    min_val = col_proj[min_col_idx]
                    max_left = np.max(col_proj[:min_col_idx]) if min_col_idx > 0 else 1
                    max_right = np.max(col_proj[min_col_idx:]) if min_col_idx < gw else 1
                    max_peak = max(max_left, max_right)

                    # Only split if there is an almost complete ink disconnection (<= 25% peak)
                    if max_peak > 0 and (min_val / max_peak) <= 0.25:
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

        res_glyphs = sorted(assembled_glyphs, key=lambda g: g["bbox_local"][1])

        # 5. Prior-guided calibration when expected ground truth glyph count is provided
        if expected_glyph_count is not None and expected_glyph_count > 0:
            K = expected_glyph_count
            # If under-segmented (e.g. 1 continuous cursive blob for a 4-glyph word)
            if len(res_glyphs) == 1 and K >= 2:
                sole = res_glyphs[0]
                s_mask = sole["mask"]
                sh, sw = s_mask.shape
                if sw >= K * 12:
                    col_proj = np.sum(s_mask, axis=0)
                    # Find K-1 cut points near proportional locations (1/K, 2/K ... (K-1)/K)
                    cuts = [0]
                    for k in range(1, K):
                        center_x = int(k * sw / K)
                        search_r = max(4, int(sw / (K * 3)))
                        x_start = max(1, center_x - search_r)
                        x_end = min(sw - 1, center_x + search_r)
                        if x_end > x_start:
                            cut_x = x_start + int(np.argmin(col_proj[x_start:x_end]))
                            cuts.append(cut_x)
                    cuts.append(sw)
                    calibrated = []
                    sy0, sx0, _, _ = sole["bbox_local"]
                    for k in range(len(cuts) - 1):
                        c0, c1 = cuts[k], cuts[k+1]
                        sub_m = s_mask[:, c0:c1]
                        if np.sum(sub_m) >= 15:
                            calibrated.append({
                                "mask": sub_m,
                                "bbox_local": (sy0, sx0 + c0, sy0 + sh, sx0 + c1),
                                "area": int(np.sum(sub_m))
                            })
                    if len(calibrated) >= 2:
                        res_glyphs = calibrated

            # If over-segmented (e.g. fragmented into > K + 2 pieces), merge closest adjacent pairs
            while len(res_glyphs) > K + 1 and len(res_glyphs) > 2:
                # Find pair with smallest horizontal distance
                min_gap = 1e9
                best_pair = 0
                for i in range(len(res_glyphs) - 1):
                    gap = res_glyphs[i+1]["bbox_local"][1] - res_glyphs[i]["bbox_local"][3]
                    if gap < min_gap:
                        min_gap = gap
                        best_pair = i
                # Merge best_pair and best_pair + 1
                g_a, g_b = res_glyphs[best_pair], res_glyphs[best_pair + 1]
                m_y0 = min(g_a["bbox_local"][0], g_b["bbox_local"][0])
                m_y1 = max(g_a["bbox_local"][2], g_b["bbox_local"][2])
                m_x0 = min(g_a["bbox_local"][1], g_b["bbox_local"][1])
                m_x1 = max(g_a["bbox_local"][3], g_b["bbox_local"][3])
                comb_m = np.zeros((m_y1 - m_y0, m_x1 - m_x0), dtype=bool)
                comb_m[g_a["bbox_local"][0]-m_y0:g_a["bbox_local"][2]-m_y0, g_a["bbox_local"][1]-m_x0:g_a["bbox_local"][3]-m_x0] |= g_a["mask"]
                comb_m[g_b["bbox_local"][0]-m_y0:g_b["bbox_local"][2]-m_y0, g_b["bbox_local"][1]-m_x0:g_b["bbox_local"][3]-m_x0] |= g_b["mask"]
                merged_g = {
                    "mask": comb_m,
                    "bbox_local": (m_y0, m_x0, m_y1, m_x1),
                    "area": int(np.sum(comb_m))
                }
                res_glyphs = res_glyphs[:best_pair] + [merged_g] + res_glyphs[best_pair+2:]

        return sorted(res_glyphs, key=lambda g: g["bbox_local"][1])

    @staticmethod
    def _paint_fraction(ink_pixels: np.ndarray) -> float:
        """
        Fraction of ink pixels whose HSV signature matches the Voynich pigment palette
        (green / blue / red / ochre washes) rather than iron-gall ink.
        """
        from skimage.color import rgb2hsv
        ink_hsv = rgb2hsv(ink_pixels.reshape(-1, 1, 3))
        h, s, v = ink_hsv[:, 0, 0], ink_hsv[:, 0, 1], ink_hsv[:, 0, 2]
        is_paint = (
            ((h >= 0.14) & (h <= 0.50) & (s > 0.14)) |
            ((h >= 0.50) & (h <= 0.78) & (s > 0.14)) |
            (((h >= 0.85) | (h <= 0.06)) & (s > 0.20) & (v > 0.25)) |
            ((h >= 0.08) & (h <= 0.16) & (s > 0.28))
        )
        return float(np.mean(is_paint))

    def build_glyph_record(
        self,
        g_mask: np.ndarray,
        abs_bbox: Tuple[int, int, int, int],
        page_rgb: np.ndarray,
        page_id: str,
        line_id: str,
        word_id: str,
        word_bbox: Tuple[int, int, int, int],
        word_png_rel: str,
        glyph_idx: int,
        output_dir: Optional[Path] = None,
        subfolder: str = "voynich",
        strict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Validates, vectorizes and exports a single glyph candidate.

        Parameters
        ----------
        g_mask : boolean ink mask of the candidate glyph (local coordinates).
        abs_bbox : (y0, x0, y1, x1) of the mask on the page.
        strict : when True (heuristic, transcription-free mode) the candidate must pass
            physical quill-scale bounds, a filamentary fill-factor test (0.05 <= fill <= 0.40)
            and a pigment test. When False (transcription-guided mode) the reference
            transcription already certifies that the zone contains text, so only a minimal
            ink-mass test is applied; the pigment fraction is still recorded for auditing.

        Returns
        -------
        Glyph dictionary (ids, bbox, ordered strokes, exported assets) or None if rejected.
        """
        gh, gw = g_mask.shape
        area = int(np.sum(g_mask))
        gy0, gx0, gy1, gx1 = [int(v) for v in abs_bbox]
        aspect = gw / max(1, gh)
        fill_factor = area / max(1, gh * gw)

        if strict:
            # Physical quill-scale bounds
            if area < 22 or gh < 14 or gw < 6:
                return None
            if gh > self.max_glyph_height or gw > self.max_glyph_width:
                return None
            if aspect > 3.2 or aspect < 0.18:
                return None
            # Filamentary stroke test (rejects solid stains / textures)
            if fill_factor < 0.05 or fill_factor > 0.40:
                return None
        else:
            if area < 12 or gh < 4 or gw < 2:
                return None

        crop_pad = 6
        cy0, cx0 = max(0, gy0 - crop_pad), max(0, gx0 - crop_pad)
        cy1, cx1 = min(page_rgb.shape[0], gy1 + crop_pad), min(page_rgb.shape[1], gx1 + crop_pad)
        raw_patch = page_rgb[cy0:cy1, cx0:cx1]
        if raw_patch.shape[0] < 4 or raw_patch.shape[1] < 3:
            return None

        paint_fraction = 0.0
        mean_chroma = 0.0
        crop_patch_rgb = page_rgb[gy0:gy1, gx0:gx1].astype(np.float32) / 255.0
        if crop_patch_rgb.shape[:2] == g_mask.shape:
            ink_pixels = crop_patch_rgb[g_mask]
            if ink_pixels.shape[0] > 0:
                chroma_ink = np.max(ink_pixels, axis=1) - np.min(ink_pixels, axis=1)
                mean_chroma = float(np.mean(chroma_ink))
                if subfolder == "seraphinianus":
                    paint_fraction = float(np.mean(chroma_ink > 0.06))
                else:
                    paint_fraction = self._paint_fraction(ink_pixels)

        if strict:
            # Authentic parchment is bright (> 115/255); reject dark blobs
            if np.mean(raw_patch) < 115:
                return None
            if subfolder == "seraphinianus":
                if mean_chroma > 0.055 or paint_fraction > 0.08:
                    return None
                if raw_patch.size > 0:
                    patch_f = raw_patch.astype(np.float32) / 255.0
                    patch_chr = np.max(patch_f, axis=2) - np.min(patch_f, axis=2)
                    if np.mean(patch_chr > 0.08) > 0.15:
                        return None
            elif paint_fraction > 0.08 or mean_chroma > 0.08:
                return None

        # Vectorize glyph ductus using Our Model (Kinematic Quill Tracker)
        try:
            ordered_strokes = self.calligraphic_vectorizer.extract_calligraphic_ductus(g_mask)
            if not ordered_strokes:
                skel, widths = self.skel_engine.extract_skeleton(g_mask)
                pixel_graph = self.graph_extractor.build_pixel_graph(skel, widths)
                raw_strokes = self.graph_extractor.decompose_into_strokes(pixel_graph)
                ordered_strokes = self.resolver.resolve_and_order_strokes(raw_strokes) if raw_strokes else []
        except Exception:
            ordered_strokes = []

        if not ordered_strokes:
            return None
        if strict:
            if len(ordered_strokes) > 6:
                return None
            if sum(len(s.get("points", [])) for s in ordered_strokes) < 8:
                return None

        # Unique hierarchical glyph identifier: Page_Line_Word_G##
        glyph_id = f"{page_id}_{line_id}_{word_id}_G{glyph_idx:02d}"

        png_rel, svg_rel, svg_str = "", "", ""
        if output_dir:
            (output_dir / subfolder / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
            (output_dir / subfolder / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)
            png_filename = f"{glyph_id}.png"
            svg_filename = f"{glyph_id}.svg"
            png_path = output_dir / subfolder / "glyphs" / "png" / png_filename
            svg_path = output_dir / subfolder / "glyphs" / "svg" / svg_filename

            save_patch = self.normalizer.enhance_contrast_and_sharpness(
                raw_patch, contrast_gain=1.35, unsharp_radius=1.0, unsharp_amount=1.6
            )
            Image.fromarray(save_patch).save(png_path)
            svg_str = VectorExporter.to_svg(ordered_strokes, width=gw, height=gh, output_path=svg_path)
            png_rel = f"{subfolder}/glyphs/png/{png_filename}"
            svg_rel = f"{subfolder}/glyphs/svg/{svg_filename}"

        return {
            "glyph_id": glyph_id,
            "page_id": page_id,
            "line_id": line_id,
            "word_id": word_id,
            "word_bbox": tuple(int(v) for v in word_bbox),
            "word_png_rel": word_png_rel,
            "bbox": (gy0, gx0, gy1, gx1),
            "height": int(gh),
            "width": int(gw),
            "area": area,
            "fill_factor": round(float(fill_factor), 3),
            "paint_fraction": round(float(paint_fraction), 3),
            "stroke_count": len(ordered_strokes),
            "strokes": ordered_strokes,
            "png_rel": png_rel,
            "svg_rel": svg_rel,
            "svg_content": svg_str,
        }

    def save_word_crop(
        self,
        page_rgb: np.ndarray,
        word_bbox: Tuple[int, int, int, int],
        page_id: str,
        word_id: str,
        output_dir: Optional[Path],
        subfolder: str,
        pad: int = 6,
    ) -> str:
        """Saves a contrast-enhanced word crop and returns its path relative to output_dir."""
        if not output_dir:
            return ""
        y0, x0, y1, x1 = word_bbox
        (output_dir / subfolder / "words" / "png").mkdir(parents=True, exist_ok=True)
        cy0, cx0 = max(0, y0 - pad), max(0, x0 - pad)
        cy1, cx1 = min(page_rgb.shape[0], y1 + pad), min(page_rgb.shape[1], x1 + pad)
        patch = page_rgb[cy0:cy1, cx0:cx1]
        if patch.size == 0:
            return ""
        enhanced = self.normalizer.enhance_contrast_and_sharpness(
            patch, contrast_gain=1.35, unsharp_radius=1.0, unsharp_amount=1.6
        )
        filename = f"{page_id}_{word_id}.png"
        Image.fromarray(enhanced).save(output_dir / subfolder / "words" / "png" / filename)
        return f"{subfolder}/words/png/{filename}"

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
                bbox_loc = item["bbox_local"]
                abs_bbox = (
                    y_off + wy0 + bbox_loc[0],
                    x_off + wx0 + bbox_loc[1],
                    y_off + wy0 + bbox_loc[2],
                    x_off + wx0 + bbox_loc[3],
                )
                record = self.build_glyph_record(
                    g_mask=item["mask"],
                    abs_bbox=abs_bbox,
                    page_rgb=page_rgb,
                    page_id=page_id,
                    line_id=line_id,
                    word_id=word_id,
                    word_bbox=(int(abs_wy0), int(abs_wx0), int(abs_wy1), int(abs_wx1)),
                    word_png_rel=word_png_rel,
                    glyph_idx=glyph_idx,
                    output_dir=output_dir,
                    subfolder=subfolder,
                    strict=True,
                )
                if record is None:
                    continue
                line_glyphs.append(record)
                glyph_idx += 1

            # Guarantee that every lexical word contains at least 1 constituent glyph (monoglyph fallback)
            if glyph_idx == 0 and np.sum(w_img) >= 15:
                try:
                    ordered_strokes = self.calligraphic_vectorizer.extract_calligraphic_ductus(w_img)
                    if not ordered_strokes:
                        skel, widths = self.skel_engine.extract_skeleton(w_img)
                        pixel_graph = self.graph_extractor.build_pixel_graph(skel, widths)
                        raw_strokes = self.graph_extractor.decompose_into_strokes(pixel_graph)
                        ordered_strokes = self.resolver.resolve_and_order_strokes(raw_strokes) if raw_strokes else []
                except Exception:
                    ordered_strokes = []

                glyph_id = f"{page_id}_{line_id}_{word_id}_G00"
                png_rel, svg_rel, svg_str = "", "", ""
                wh_m, ww_m = w_img.shape
                if output_dir:
                    (output_dir / subfolder / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
                    (output_dir / subfolder / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

                    png_filename = f"{glyph_id}.png"
                    svg_filename = f"{glyph_id}.svg"
                    png_path = output_dir / subfolder / "glyphs" / "png" / png_filename
                    svg_path = output_dir / subfolder / "glyphs" / "svg" / svg_filename

                    w_pad = 4
                    cy0, cx0 = max(0, abs_wy0 - w_pad), max(0, abs_wx0 - w_pad)
                    cy1, cx1 = min(page_rgb.shape[0], abs_wy1 + w_pad), min(page_rgb.shape[1], abs_wx1 + w_pad)
                    raw_patch = page_rgb[cy0:cy1, cx0:cx1]
                    save_patch = self.normalizer.enhance_contrast_and_sharpness(raw_patch, contrast_gain=1.35, unsharp_radius=1.0, unsharp_amount=1.6)
                    Image.fromarray(save_patch).save(png_path)

                    if ordered_strokes:
                        svg_str = VectorExporter.to_svg(ordered_strokes, width=ww_m, height=wh_m, output_path=svg_path)

                    png_rel = f"{subfolder}/glyphs/png/{png_filename}"
                    svg_rel = f"{subfolder}/glyphs/svg/{svg_filename}"

                line_glyphs.append({
                    "glyph_id": glyph_id,
                    "page_id": page_id,
                    "line_id": line_id,
                    "word_id": word_id,
                    "word_bbox": (int(abs_wy0), int(abs_wx0), int(abs_wy1), int(abs_wx1)),
                    "word_png_rel": word_png_rel,
                    "bbox": (int(abs_wy0), int(abs_wx0), int(abs_wy1), int(abs_wx1)),
                    "height": int(w_h),
                    "width": int(w_w),
                    "area": int(np.sum(w_img)),
                    "fill_factor": round(float(np.sum(w_img) / max(1, w_h * w_w)), 3),
                    "stroke_count": len(ordered_strokes),
                    "strokes": ordered_strokes,
                    "png_rel": png_rel,
                    "svg_rel": svg_rel,
                    "svg_content": svg_str
                })

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
        clean_mask, page_rgb, was_rotated = self.extract_clean_page_mask(image, subfolder=subfolder)
        lines = self.line_segmenter.segment_lines(clean_mask, auto_isolate=False)

        if max_lines:
            lines = lines[:max_lines]

        # Use page_rgb (which has correct orientation) for saving the explorer canvas image
        page_img_for_explorer = Image.fromarray(page_rgb)

        page_img_rel = ""
        if output_dir:
            pages_dir = output_dir / subfolder / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = pages_dir / f"{page_id}.jpg"
            page_img_for_explorer.save(page_img_path, quality=88)
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
            "image_width": page_img_for_explorer.width,
            "image_height": page_img_for_explorer.height,
            "line_count": len(lines),
            "word_count": len(all_page_words),
            "glyph_count": len(all_page_glyphs),
            "lines": clean_lines,
            "words": all_page_words,
            "glyphs": all_page_glyphs
        }

    def extract_page_glyphs_with_ground_truth(
        self,
        page_id: str,
        image: Image.Image,
        rosetta_words: List[Any],
        output_dir: Optional[Path] = None,
        subfolder: str = "voynich"
    ) -> Dict[str, Any]:
        """
        Extracts words and glyphs strictly guided by Voynichese Rosetta Ground Truth annotations:
        1. Uses certified ground truth word bounding boxes (no rogue words/glyphs in margins or drawings).
        2. Calibrates character segmentation to expected token length K = len(tokenize_eva(word)).
        3. Tags each constituent glyph with its exact historical EVA character transcription.
        """
        page_rgb = np.array(image.convert("RGB"))
        H, W, _ = page_rgb.shape

        page_img_for_explorer = Image.fromarray(page_rgb)
        page_img_rel = ""
        if output_dir:
            pages_dir = output_dir / subfolder / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = pages_dir / f"{page_id}.jpg"
            page_img_for_explorer.save(page_img_path, quality=88)
            page_img_rel = f"{subfolder}/pages/{page_id}.jpg"

        from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader
        loader = VoynicheseRosettaLoader()

        all_page_words = []
        all_page_glyphs = []
        
        # Sort ground truth words by y (lines) then x
        valid_r_words = [rw for rw in rosetta_words if rw.bbox_scan is not None]
        valid_r_words.sort(key=lambda rw: (rw.bbox_scan[1], rw.bbox_scan[0]))

        # Group words into lines by vertical proximity
        line_clusters: List[List[Any]] = []
        for rw in valid_r_words:
            rx0, ry0, rx1, ry1 = rw.bbox_scan
            rc_y = (ry0 + ry1) / 2.0
            
            placed = False
            for lc in line_clusters:
                lc_ys = [(w.bbox_scan[1] + w.bbox_scan[3]) / 2.0 for w in lc]
                mean_y = sum(lc_ys) / len(lc_ys)
                if abs(rc_y - mean_y) < 30:  # Same line pitch
                    lc.append(rw)
                    placed = True
                    break
            if not placed:
                line_clusters.append([rw])

        clean_lines = []
        for line_idx, lc in enumerate(line_clusters):
            lc.sort(key=lambda w: w.bbox_scan[0])  # Left to right
            line_id = f"L{line_idx:02d}"
            
            ly0 = min(w.bbox_scan[1] for w in lc)
            lx0 = min(w.bbox_scan[0] for w in lc)
            ly1 = max(w.bbox_scan[3] for w in lc)
            lx1 = max(w.bbox_scan[2] for w in lc)

            clean_lines.append({
                "line_id": line_id,
                "bbox": (int(ly0), int(lx0), int(ly1), int(lx1))
            })

            for word_idx, rw in enumerate(lc):
                rx0, ry0, rx1, ry1 = [int(v) for v in rw.bbox_scan]
                # Ensure valid bbox within scan boundaries
                rx0 = max(0, min(W - 1, rx0))
                ry0 = max(0, min(H - 1, ry0))
                rx1 = max(rx0 + 1, min(W, rx1))
                ry1 = max(ry0 + 1, min(H, ry1))
                
                w_w = rx1 - rx0
                w_h = ry1 - ry0
                if w_w < 4 or w_h < 4:
                    continue

                word_id = f"{line_id}_W{word_idx:02d}"
                eva_text = rw.eva_text or ""
                eva_tokens = loader.tokenize_eva_to_glyphs(eva_text)
                expected_k = max(1, len(eva_tokens)) if eva_tokens else None

                # Direct word patch crop from exact ground truth bounding box
                word_crop_rgb = page_rgb[ry0:ry1, rx0:rx1]
                w_pil = Image.fromarray(word_crop_rgb)
                w_mask = self.binarizer.binarize(w_pil)
                w_mask = self.binarizer.remove_small_artifacts(w_mask, min_size=3)

                # Fallback if local mask is sparse
                if np.sum(w_mask) < 6:
                    word_gray = np.mean(word_crop_rgb, axis=2)
                    w_mask = word_gray < 165

                # Save word crop image directly without shifting
                word_png_rel = self.save_word_crop(
                    page_rgb=page_rgb,
                    word_bbox=(ry0, rx0, ry1, rx1),
                    page_id=page_id,
                    word_id=word_id,
                    output_dir=output_dir,
                    subfolder=subfolder
                )

                all_page_words.append({
                    "word_id": word_id,
                    "line_id": line_id,
                    "bbox": (int(ry0), int(rx0), int(ry1), int(rx1)),
                    "word_png_rel": word_png_rel,
                    "eva_text": eva_text,
                    "eva_glyphs": eva_tokens
                })

                # Assemble constituent glyphs guided by expected token count K
                assembled_components = self.assemble_word_into_glyphs(
                    w_mask, expected_glyph_count=expected_k
                )

                glyph_idx = 0
                for item in assembled_components:
                    bbox_loc = item["bbox_local"]
                    abs_bbox = (
                        ry0 + bbox_loc[0],
                        rx0 + bbox_loc[1],
                        ry0 + bbox_loc[2],
                        rx0 + bbox_loc[3],
                    )
                    record = self.build_glyph_record(
                        g_mask=item["mask"],
                        abs_bbox=abs_bbox,
                        page_rgb=page_rgb,
                        page_id=page_id,
                        line_id=line_id,
                        word_id=word_id,
                        word_bbox=(int(ry0), int(rx0), int(ry1), int(rx1)),
                        word_png_rel=word_png_rel,
                        glyph_idx=glyph_idx,
                        output_dir=output_dir,
                        subfolder=subfolder,
                        strict=False,  # Certified by ground truth
                    )
                    if record is None:
                        continue

                    # Associate with ground truth EVA token
                    if eva_tokens and glyph_idx < len(eva_tokens):
                        record["eva_char"] = eva_tokens[glyph_idx]
                    elif eva_tokens:
                        record["eva_char"] = eva_tokens[-1]
                    else:
                        record["eva_char"] = ""
                    record["ref_eva_text"] = eva_text

                    all_page_glyphs.append(record)
                    glyph_idx += 1

                # Monoglyph fallback for words that didn't generate glyph records
                if glyph_idx == 0 and np.sum(w_mask) >= 8:
                    abs_bbox = (int(ry0), int(rx0), int(ry1), int(rx1))
                    record = self.build_glyph_record(
                        g_mask=w_mask,
                        abs_bbox=abs_bbox,
                        page_rgb=page_rgb,
                        page_id=page_id,
                        line_id=line_id,
                        word_id=word_id,
                        word_bbox=(int(ry0), int(rx0), int(ry1), int(rx1)),
                        word_png_rel=word_png_rel,
                        glyph_idx=0,
                        output_dir=output_dir,
                        subfolder=subfolder,
                        strict=False,
                    )
                    if record is not None:
                        record["eva_char"] = eva_tokens[0] if eva_tokens else ""
                        record["ref_eva_text"] = eva_text
                        all_page_glyphs.append(record)

        return {
            "page_id": page_id,
            "page_img_rel": page_img_rel,
            "image_width": W,
            "image_height": H,
            "line_count": len(clean_lines),
            "word_count": len(all_page_words),
            "glyph_count": len(all_page_glyphs),
            "lines": clean_lines,
            "words": all_page_words,
            "glyphs": all_page_glyphs
        }


