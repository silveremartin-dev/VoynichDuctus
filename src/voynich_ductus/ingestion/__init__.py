"""Ingestion module for downloading and preprocessing manuscript scans."""

from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter

__all__ = ["IIIFClient", "Binarizer", "LineSegmenter"]
