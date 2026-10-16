"""
Builds the Canonical Alphabet Catalogue for Codex Seraphinianus
from the 371 extracted page JSONs in output/atlas_dataset/seraphinianus/pages/.
"""

import json
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue


def main():
    pages_dir = Path("output/atlas_dataset/seraphinianus/pages")
    json_files = sorted(pages_dir.glob("*.json"))
    print(f"[*] Found {len(json_files)} Seraphinianus page JSONs.")

    sampled_glyphs = []
    total_glyphs_seen = 0

    for idx, jf in enumerate(json_files):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            glyphs = data.get("glyphs", [])
            total_glyphs_seen += len(glyphs)
            # Sample up to 25 glyphs with intact strokes per page
            valid_glyphs = [g for g in glyphs if len(g.get("strokes", [])) >= 1 and g.get("width", 0) >= 8 and g.get("height", 0) >= 10]
            sampled_glyphs.extend(valid_glyphs[:25])
        except Exception as e:
            print(f"[-] Error reading {jf.name}: {e}")

    print(f"[*] Total Seraphinianus glyphs across all pages: {total_glyphs_seen}")
    print(f"[*] Representative sample for clustering: {len(sampled_glyphs)} glyphs.")

    builder = GlyphCatalogue(target_alphabet_size=52)
    print("[*] Running topological partitioning & clustering...")
    catalogue = builder.build_catalogue(sampled_glyphs, corpus_type="seraphinianus")

    out_file = Path("output/atlas_dataset/seraphinianus/seraphinianus_alphabet_catalogue.json")
    builder.export_catalogue_json(catalogue, out_file)
    print(f"[+] Saved Seraphinianus Alphabet Catalogue to {out_file.resolve()}")
    print(f"[+] Induced {catalogue['canonical_alphabet_size']} canonical archetypes.")


if __name__ == "__main__":
    main()
