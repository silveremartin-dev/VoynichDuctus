"""
PDF Scan Loader and Page Renderer using pypdfium2.
Extracts high-resolution images from multi-page manuscript PDF scans.
"""

from pathlib import Path
from typing import List, Generator, Optional, Union
from PIL import Image
import pypdfium2 as pdfium


class PDFScanLoader:
    """
    Renders pages from digitized PDF scans (such as the Voynich Manuscript or Codex Seraphinianus)
    into high-resolution PIL images.
    """

    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF scan file not found: {self.pdf_path}")
        self.doc = pdfium.PdfDocument(str(self.pdf_path))

    def __len__(self) -> int:
        return len(self.doc)

    def get_page_image(self, page_index: int, target_min_dim: int = 1500) -> Image.Image:
        """
        Renders a specific page into a PIL Image, automatically scaling to reach target resolution.
        """
        if page_index < 0 or page_index >= len(self.doc):
            raise IndexError(f"Page index {page_index} out of bounds (0..{len(self.doc)-1})")

        page = self.doc[page_index]
        # Check native point size
        w, h = page.get_size()
        min_dim = min(w, h)
        scale = max(2.0, target_min_dim / max(min_dim, 1))

        # Render page
        pil_img = page.render(scale=scale).to_pil()
        return pil_img

    def iter_pages(self, start_page: int = 0, max_pages: Optional[int] = None, target_min_dim: int = 1500) -> Generator[tuple, None, None]:
        """
        Iterates over pages, yielding (page_index, PIL_Image).
        """
        end_page = len(self.doc) if max_pages is None else min(len(self.doc), start_page + max_pages)
        for idx in range(start_page, end_page):
            yield idx, self.get_page_image(idx, target_min_dim=target_min_dim)
