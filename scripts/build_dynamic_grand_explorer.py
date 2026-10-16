"""
Dynamic Grand Glyph Explorer Generator for VoynichDuctus.
Uses the official generate_explorer_html function from scripts/generate_full_glyph_corpus_and_explorer.py
to generate a 100% syntactically valid HTML/CSS/JS application with dark theme and dynamic page loading.
"""

import json
from pathlib import Path
import re
import sys

# Ensure src and scripts are on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "scripts"))

from generate_full_glyph_corpus_and_explorer import generate_explorer_html


def clean_catalogue_for_embed(cat_dict: dict) -> dict:
    """Retains full archetype metadata, exemplars and transliteration alignments while omitting massive instance lists."""
    cleaned_alphabet = []
    for arch in cat_dict.get("alphabet", []):
        cleaned_alphabet.append({
            "type_id": arch.get("type_id"),
            "rank": arch.get("rank"),
            "cluster_index": arch.get("cluster_index"),
            "frequency": arch.get("frequency"),
            "percentage": arch.get("percentage"),
            "exemplar_id": arch.get("exemplar_id"),
            "exemplar_png": arch.get("exemplar_png", ""),
            "exemplar_svg": arch.get("exemplar_svg", ""),
            "exemplar_svg_content": arch.get("exemplar_svg_content", ""),
            "mean_strokes": arch.get("mean_strokes"),
            "mean_height": arch.get("mean_height"),
            "mean_width": arch.get("mean_width"),
            "corpus_match": arch.get("corpus_match"),
            "corpus_matches": arch.get("corpus_matches"),
            "total_instances_count": arch.get("total_instances_count", arch.get("frequency", 0)),
            "sample_instances": arch.get("sample_instances", [])[:15]
        })
    return {
        "total_glyphs": cat_dict.get("total_glyphs", 0),
        "canonical_alphabet_size": len(cleaned_alphabet),
        "alphabet": cleaned_alphabet
    }


def parse_page_stub_fast(json_path: Path, subfolder: str) -> dict:
    """Fast header reading without parsing megabytes of internal stroke arrays."""
    stem = json_path.stem
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(800)

    m_pid = re.search(r'"page_id":\s*"([^"]+)"', head)
    page_id = m_pid.group(1) if m_pid else stem

    m_w = re.search(r'"image_width":\s*(\d+)', head)
    m_h = re.search(r'"image_height":\s*(\d+)', head)
    m_lines = re.search(r'"line_count":\s*(\d+)', head)
    m_words = re.search(r'"word_count":\s*(\d+)', head)
    m_glyphs = re.search(r'"glyph_count":\s*(\d+)', head)

    return {
        "page_id": page_id,
        "page_img_rel": f"{subfolder}/pages/{stem}.jpg",
        "json_rel": f"{subfolder}/pages/{stem}.json",
        "image_width": int(m_w.group(1)) if m_w else 1600,
        "image_height": int(m_h.group(1)) if m_h else 2200,
        "line_count": int(m_lines.group(1)) if m_lines else 0,
        "word_count": int(m_words.group(1)) if m_words else 0,
        "glyph_count": int(m_glyphs.group(1)) if m_glyphs else 0,
        "yield_report": {}
    }


def main():
    base_out = Path("output/atlas_dataset")
    voynich_dir = base_out / "voynich"
    serafini_dir = base_out / "seraphinianus"

    print("[*] Loading Voynich Alphabet Catalogue...")
    with open(voynich_dir / "voynich_alphabet_catalogue.json", "r", encoding="utf-8") as f:
        v_cat_raw = json.load(f)
    v_cat = clean_catalogue_for_embed(v_cat_raw)
    print(f"  [+] Voynich Catalogue: {v_cat['canonical_alphabet_size']} canonical archetypes.")

    print("[*] Loading Seraphinianus Alphabet Catalogue...")
    with open(serafini_dir / "seraphinianus_alphabet_catalogue.json", "r", encoding="utf-8") as f:
        s_cat_raw = json.load(f)
    s_cat = clean_catalogue_for_embed(s_cat_raw)
    print(f"  [+] Seraphinianus Catalogue: {s_cat['canonical_alphabet_size']} canonical archetypes.")

    # 1. Index Voynich Pages
    print("[*] Indexing Voynich pages...")
    v_json_files = sorted((voynich_dir / "pages").glob("*.json"))
    v_pages_stubs = []

    for idx, jf in enumerate(v_json_files):
        stub = parse_page_stub_fast(jf, "voynich")
        # Pre-load full content for f1r (page 0)
        if idx == 0:
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
            stub["yield_report"] = p_full.get("yield_report", {})
        v_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(v_pages_stubs)} Voynich folios.")

    # 2. Index Seraphinianus Pages
    print("[*] Indexing Codex Seraphinianus pages...")
    s_json_files = sorted((serafini_dir / "pages").glob("*.json"))
    s_pages_stubs = []

    for idx, jf in enumerate(s_json_files):
        stub = parse_page_stub_fast(jf, "seraphinianus")
        if jf.stem == "serafini_p030":
            with open(jf, "r", encoding="utf-8") as f_full:
                p_full = json.load(f_full)
            stub["lines"] = p_full.get("lines", [])
            stub["words"] = p_full.get("words", [])
            stub["glyphs"] = p_full.get("glyphs", [])
        s_pages_stubs.append(stub)

    print(f"  [+] Indexed {len(s_pages_stubs)} Seraphinianus pages.")

    # 3. Call official generator (Python f-string correctly unescapes all CSS and JS braces!)
    out_html_path = base_out / "grand_glyph_explorer.html"
    print("[*] Building Grand Glyph Explorer HTML via native generator...")
    generate_explorer_html(
        voynich_pages_data=v_pages_stubs,
        voynich_catalogue=v_cat,
        serafini_pages_data=s_pages_stubs,
        serafini_catalogue=s_cat,
        output_path=out_html_path
    )

    print(f"[+] Successfully generated: {out_html_path.resolve()}")
    print(f"  [+] File size: {out_html_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
