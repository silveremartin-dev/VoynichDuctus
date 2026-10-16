"""
Systematic cleanup script for VoynichDuctus:
Removes legacy non-zero-waste assets from output/atlas_dataset (glyphs/ and words/ folders),
removes temporary scratch_* images from data/,
removes test artifacts from output/,
and cleans up obsolete debug scripts from scratch/.
"""

import os
from pathlib import Path
import shutil

def remove_dir_safely(p: Path):
    if p.exists() and p.is_dir():
        count = sum(1 for _ in p.glob('**/*') if _.is_file())
        size = sum(_.stat().st_size for _ in p.glob('**/*') if _.is_file())
        shutil.rmtree(p)
        print(f"[-] Removed directory {p} ({count} files, {size/(1024*1024):.2f} MB)")
        return count, size
    return 0, 0

def remove_file_safely(p: Path):
    if p.exists() and p.is_file():
        size = p.stat().st_size
        p.unlink()
        print(f"[-] Removed file {p.name} ({size/(1024*1024):.2f} MB)")
        return 1, size
    return 0, 0

def main():
    root = Path(".")
    total_files = 0
    total_bytes = 0

    print("=====================================================")
    print(" 1. CLEANING DATA/ ROOT TEMPORARY SCRATCH IMAGES")
    print("=====================================================")
    data_dir = root / "data"
    for pattern in ["scratch_*", "test_align_*"]:
        for f in data_dir.glob(pattern):
            c, s = remove_file_safely(f)
            total_files += c
            total_bytes += s

    print("\n=====================================================")
    print(" 2. CLEANING OUTPUT/ATLAS_DATASET ZERO-WASTE VIOLATIONS")
    print("=====================================================")
    atlas_dir = root / "output" / "atlas_dataset"
    for sub in ["voynich", "seraphinianus"]:
        for obsolete in ["glyphs", "words"]:
            obs_dir = atlas_dir / sub / obsolete
            c, s = remove_dir_safely(obs_dir)
            total_files += c
            total_bytes += s

    # Remove duplicate f001r-f004v jpgs in voynich/pages
    v_pages = atlas_dir / "voynich" / "pages"
    for dup in ["f001r.jpg", "f001v.jpg", "f002r.jpg", "f002v.jpg", "f003r.jpg", "f003v.jpg", "f004r.jpg", "f004v.jpg"]:
        dup_file = v_pages / dup
        c, s = remove_file_safely(dup_file)
        total_files += c
        total_bytes += s

    print("\n=====================================================")
    print(" 3. CLEANING ROOT OUTPUT/ TEMP TEST ARTIFACTS")
    print("=====================================================")
    out_dir = root / "output"
    for f in [out_dir / "p030_alignment_check.jpg"]:
        c, s = remove_file_safely(f)
        total_files += c
        total_bytes += s

    for d in [out_dir / "test_gallica_cache", out_dir / "training_plates"]:
        c, s = remove_dir_safely(d)
        total_files += c
        total_bytes += s

    print("\n=====================================================")
    print(" 4. CLEANING SCRATCH/ TEMPORARY SCRIPTS")
    print("=====================================================")
    scratch_dir = root / "scratch"
    if scratch_dir.exists():
        for f in scratch_dir.glob("*.py"):
            c, s = remove_file_safely(f)
            total_files += c
            total_bytes += s

    # Also clean test_fast_head.py and test_serafini_align.py in scripts
    for test_s in [root / "scripts" / "test_fast_head.py", root / "scripts" / "test_serafini_align.py"]:
        c, s = remove_file_safely(test_s)
        total_files += c
        total_bytes += s

    print("\n=====================================================")
    print(f" TOTAL CLEANED: {total_files} files, {total_bytes/(1024*1024):.2f} MB freed!")
    print("=====================================================")

if __name__ == "__main__":
    main()
