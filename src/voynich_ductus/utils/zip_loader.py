"""
Streaming Zip Corpus Loader.
Provides transparent, on-the-fly streaming of page JSONs, vector SVGs, and images
directly from contiguous .zip archives without requiring slow multi-gigabyte disk extractions.
Supports dual-mode fallback: loads from unzipped folder if present, or streams directly from .zip.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import zipfile
import json
import io
from PIL import Image


class StreamingZipCorpusLoader:
    """
    High-performance zero-extraction corpus loader.
    Streams page data directly from .zip archives with memory caching.
    """

    def __init__(self, base_path: Union[str, Path], subfolder: str = "voynich", cache_size: int = 32):
        self.base_path = Path(base_path)
        self.subfolder = subfolder
        self.cache_size = cache_size
        
        self.zip_path: Optional[Path] = None
        self.zip_ref: Optional[zipfile.ZipFile] = None
        self.is_archive = False
        
        # Check if base_path is a zip file
        if self.base_path.is_file() and self.base_path.suffix.lower() == ".zip":
            self.zip_path = self.base_path
            self.is_archive = True
        elif (self.base_path / f"{subfolder}_dataset.zip").exists():
            self.zip_path = self.base_path / f"{subfolder}_dataset.zip"
            self.is_archive = True
        elif (self.base_path / f"{subfolder}.zip").exists():
            self.zip_path = self.base_path / f"{subfolder}.zip"
            self.is_archive = True
            
        if self.is_archive and self.zip_path:
            self.zip_ref = zipfile.ZipFile(self.zip_path, "r")
            self._file_list = set(self.zip_ref.namelist())
        else:
            self._file_list = set()

        self._json_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_keys: List[str] = []

    def close(self):
        """Closes the underlying zip file handle if open."""
        if self.zip_ref:
            self.zip_ref.close()
            self.zip_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def list_page_ids(self) -> List[str]:
        """Returns sorted list of available folio / page IDs."""
        if self.is_archive and self.zip_ref:
            pids = set()
            for name in self._file_list:
                # Look for pages/*.json
                if "/pages/" in name and name.endswith(".json"):
                    pid = Path(name).stem
                    pids.add(pid)
            return sorted(list(pids))
        else:
            pages_dir = self.base_path / self.subfolder / "pages"
            if not pages_dir.exists():
                pages_dir = self.base_path / "pages"
            if pages_dir.exists():
                return sorted([p.stem for p in pages_dir.glob("*.json")])
            return []

    def load_page_json(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Loads and parses page JSON on the fly."""
        if page_id in self._json_cache:
            return self._json_cache[page_id]

        data = None
        if self.is_archive and self.zip_ref:
            # Candidate paths inside zip
            candidates = [
                f"{self.subfolder}/pages/{page_id}.json",
                f"pages/{page_id}.json",
                f"{page_id}.json"
            ]
            for cand in candidates:
                if cand in self._file_list:
                    with self.zip_ref.open(cand) as f:
                        data = json.load(f)
                    break
        else:
            candidates = [
                self.base_path / self.subfolder / "pages" / f"{page_id}.json",
                self.base_path / "pages" / f"{page_id}.json",
                self.base_path / f"{page_id}.json"
            ]
            for cand in candidates:
                if cand.exists():
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    break

        if data is not None:
            # Maintain LRU cache
            if len(self._cache_keys) >= self.cache_size:
                oldest = self._cache_keys.pop(0)
                self._json_cache.pop(oldest, None)
            self._json_cache[page_id] = data
            self._cache_keys.append(page_id)

        return data

    def load_page_image(self, page_id: str) -> Optional[Image.Image]:
        """Streams and decodes page scan image on the fly into PIL Image."""
        if self.is_archive and self.zip_ref:
            candidates = [
                f"{self.subfolder}/pages/{page_id}.jpg",
                f"{self.subfolder}/pages/{page_id}.png",
                f"pages/{page_id}.jpg",
                f"pages/{page_id}.png",
                f"{page_id}.jpg"
            ]
            for cand in candidates:
                if cand in self._file_list:
                    with self.zip_ref.open(cand) as f:
                        img_bytes = io.BytesIO(f.read())
                        return Image.open(img_bytes).convert("RGB")
        else:
            candidates = [
                self.base_path / self.subfolder / "pages" / f"{page_id}.jpg",
                self.base_path / self.subfolder / "pages" / f"{page_id}.png",
                self.base_path / "pages" / f"{page_id}.jpg",
                self.base_path / f"{page_id}.jpg"
            ]
            for cand in candidates:
                if cand.exists():
                    return Image.open(cand).convert("RGB")
        return None

    def load_catalogue(self) -> Optional[Dict[str, Any]]:
        """Loads canonical alphabet catalogue directly from archive or directory."""
        if self.is_archive and self.zip_ref:
            candidates = [
                f"{self.subfolder}/{self.subfolder}_alphabet_catalogue.json",
                f"{self.subfolder}_alphabet_catalogue.json",
                "alphabet_catalogue.json"
            ]
            for cand in candidates:
                if cand in self._file_list:
                    with self.zip_ref.open(cand) as f:
                        return json.load(f)
        else:
            candidates = [
                self.base_path / self.subfolder / f"{self.subfolder}_alphabet_catalogue.json",
                self.base_path / f"{self.subfolder}_alphabet_catalogue.json",
                self.base_path / "alphabet_catalogue.json"
            ]
            for cand in candidates:
                if cand.exists():
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f)
        return None
