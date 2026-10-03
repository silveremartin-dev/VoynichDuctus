"""
Client for interacting with Yale Beinecke Rare Book & Manuscript Library IIIF API (MS 408).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import requests


class IIIFClient:
    """
    Downloads high-resolution folios of the Voynich Manuscript (Beinecke MS 408)
    via official Yale IIIF Presentation v3 API and Image API endpoints.
    """

    YALE_MANIFEST_URL = "https://collections.library.yale.edu/manifests/2002046"
    BEINECKE_BASE_URL = "https://collections.library.yale.edu/iiif/2"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/scans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_cache: Optional[Dict] = None

    @staticmethod
    def normalize_folio_name(folio_id: str) -> str:
        """
        Normalizes folio identifiers into standard canonical format:
        e.g., 'f1r', 'f001r', '1r', 'f67v2' -> 'f001r', 'f067v2'
        """
        f = folio_id.strip().lower()
        if not f.startswith("f"):
            f = "f" + f
        
        match = re.match(r"f(\d+)([rv])(\d*)", f)
        if match:
            num, side, sub = match.groups()
            padded_num = f"{int(num):03d}"
            return f"f{padded_num}{side}{sub}"
        return f

    def fetch_manifest(self) -> Dict:
        """Fetches and caches the Beinecke MS 408 IIIF Presentation v3 manifest."""
        if self._manifest_cache is not None:
            return self._manifest_cache
        headers = {"User-Agent": "VoynichDuctus-Research/0.1"}
        resp = requests.get(self.YALE_MANIFEST_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        self._manifest_cache = resp.json()
        return self._manifest_cache

    def get_folio_image_url(self, folio_id: str, max_width: int = 1600) -> Optional[str]:
        """
        Finds the IIIF image URL for a given folio identifier (e.g. '1r', 'f001r', 'f026v').
        """
        norm = self.normalize_folio_name(folio_id)
        # Match label without leading 'f0' (e.g. '1r', '26v')
        short_num = norm.lstrip("f").lstrip("0")
        if not short_num:
            short_num = "1r"
        
        manifest = self.fetch_manifest()
        for item in manifest.get("items", []):
            labels = item.get("label", {}).get("none", [])
            for lbl in labels:
                clean_lbl = lbl.strip().lower().lstrip("f").lstrip("0")
                if clean_lbl == short_num or lbl.strip().lower() == norm or lbl.strip().lower() == folio_id.strip().lower():
                    # Extract body service
                    try:
                        body = item["items"][0]["items"][0]["body"]
                        service_id = body.get("service", [{}])[0].get("@id")
                        if service_id:
                            return f"{service_id}/full/{max_width},/0/default.jpg"
                        return body.get("id")
                    except (KeyError, IndexError):
                        pass
        return None

    def download_folio(self, folio_id: str, output_path: Optional[str] = None, max_width: int = 1600) -> Path:
        """
        Downloads a specific folio image from Yale Beinecke archives.
        """
        norm_folio = self.normalize_folio_name(folio_id)
        out_file = Path(output_path) if output_path else self.cache_dir / f"{norm_folio}.jpg"

        if out_file.exists():
            return out_file

        url = self.get_folio_image_url(folio_id, max_width=max_width)
        if not url:
            # Fallback to direct Yale item 2 for f001r if search misses
            url = f"https://collections.library.yale.edu/iiif/2/1006076/full/{max_width},/0/default.jpg"

        headers = {"User-Agent": "VoynichDuctus-Research/0.1"}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        if resp.status_code == 200:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    f.write(chunk)
            return out_file
        else:
            raise RuntimeError(f"Failed to fetch folio {norm_folio} (HTTP {resp.status_code})")
