import io
import json
import zipfile
from pathlib import Path
from PIL import Image

from voynich_ductus.utils.zip_loader import StreamingZipCorpusLoader


def test_streaming_zip_loader_with_mock_zip(tmp_path: Path):
    zip_path = tmp_path / "test_corpus.zip"
    
    # Create sample zip archive with pages/ and catalogue
    sample_page = {
        "page_id": "f001r",
        "word_count": 42,
        "glyph_count": 180,
        "words": [{"word_id": "L00_W00", "eva_text": "fachys"}]
    }
    sample_catalogue = {
        "canonical_alphabet_size": 28,
        "alphabet": [{"type_id": "G01", "frequency": 100}]
    }
    
    # Create a small dummy image
    img = Image.new("RGB", (60, 80), color=(200, 180, 150))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("voynich/pages/f001r.json", json.dumps(sample_page))
        z.writestr("voynich/pages/f001r.jpg", img_bytes.getvalue())
        z.writestr("voynich/voynich_alphabet_catalogue.json", json.dumps(sample_catalogue))
        
    loader = StreamingZipCorpusLoader(zip_path, subfolder="voynich")
    
    # Test listing page IDs
    pids = loader.list_page_ids()
    assert "f001r" in pids
    
    # Test loading page JSON on the fly
    pdata = loader.load_page_json("f001r")
    assert pdata is not None
    assert pdata["page_id"] == "f001r"
    assert pdata["word_count"] == 42
    
    # Test cache hit
    pdata2 = loader.load_page_json("f001r")
    assert pdata2 is pdata
    
    # Test loading page image on the fly
    pimg = loader.load_page_image("f001r")
    assert pimg is not None
    assert pimg.size == (60, 80)
    
    # Test loading catalogue
    cat = loader.load_catalogue()
    assert cat is not None
    assert cat["canonical_alphabet_size"] == 28
    
    loader.close()


def test_streaming_zip_loader_directory_fallback(tmp_path: Path):
    pages_dir = tmp_path / "voynich" / "pages"
    pages_dir.mkdir(parents=True)
    
    sample_page = {"page_id": "f002v", "word_count": 10}
    with open(pages_dir / "f002v.json", "w") as f:
        json.dump(sample_page, f)
        
    loader = StreamingZipCorpusLoader(tmp_path, subfolder="voynich")
    pids = loader.list_page_ids()
    assert "f002v" in pids
    
    pdata = loader.load_page_json("f002v")
    assert pdata["page_id"] == "f002v"
    loader.close()
