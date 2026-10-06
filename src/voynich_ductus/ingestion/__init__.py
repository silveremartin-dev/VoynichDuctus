"""Ingestion module for downloading and preprocessing manuscript scans."""

from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter
from voynich_ductus.ingestion.pdf_loader import PDFScanLoader
from voynich_ductus.ingestion.color_normalizer import ColorIlluminationNormalizer
from voynich_ductus.ingestion.transcription_reference import TranscriptionReference, PaleographyYieldValidator
from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter

from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader, RosettaWord

__all__ = [
    "IIIFClient",
    "Binarizer",
    "LineSegmenter",
    "PDFScanLoader",
    "ColorIlluminationNormalizer",
    "TranscriptionReference",
    "PaleographyYieldValidator",
    "GlyphSegmenter",
    "VoynicheseRosettaLoader",
    "RosettaWord",
]


