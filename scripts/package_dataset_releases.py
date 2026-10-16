"""
Dataset Packaging and Release Preparation for VoynichDuctus.
Packages derived mathematical vector coordinates (JSONs) into partitioned zip archives (< 90 MB each),
strictly respecting GitHub's 100 MB per-file threshold, with complete separation between Voynich and Seraphinianus.
"""

import os
import zipfile
from pathlib import Path
from typing import List

MAX_ZIP_BYTES = 85 * 1024 * 1024  # 85 MB target threshold (well under GitHub 100 MB limit)


def package_subfolder(subfolder_name: str, src_dir: Path, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(list((src_dir / subfolder_name / "pages").glob("*.json")))
    catalogue_files = list((src_dir / subfolder_name).glob("*catalogue*.json"))
    all_files = catalogue_files + json_files

    print(f"\n[*] Packaging {subfolder_name.upper()} ({len(all_files)} JSON files)...")

    part_idx = 1
    current_zip_path = output_dir / f"{prefix}_part{part_idx:02d}.zip"
    zf = zipfile.ZipFile(current_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    current_uncompressed_est = 0
    files_in_part = 0
    total_parts = []

    for f in all_files:
        arcname = f.relative_to(src_dir).as_posix()
        zf.write(f, arcname)
        files_in_part += 1
        
        # Check size of current zip on disk
        zf.close()
        current_size = current_zip_path.stat().st_size
        if current_size >= MAX_ZIP_BYTES:
            print(f"  [+] Closed Part {part_idx:02d}: {current_zip_path.name} ({current_size / (1024*1024):.2f} MB, {files_in_part} files)")
            total_parts.append(current_zip_path)
            part_idx += 1
            current_zip_path = output_dir / f"{prefix}_part{part_idx:02d}.zip"
            zf = zipfile.ZipFile(current_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
            files_in_part = 0
        else:
            # Reopen in append mode
            zf = zipfile.ZipFile(current_zip_path, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=9)

    zf.close()
    if current_zip_path.stat().st_size > 0:
        final_size = current_zip_path.stat().st_size
        print(f"  [+] Closed Part {part_idx:02d}: {current_zip_path.name} ({final_size / (1024*1024):.2f} MB, {files_in_part} files)")
        total_parts.append(current_zip_path)

    print(f"  [✓] {subfolder_name.upper()} successfully partitioned into {len(total_parts)} archives (< 90 MB each).")
    return total_parts


def main():
    atlas_dir = Path("output/atlas_dataset")
    out_release = Path("releases/dataset_zips")

    print("=" * 80)
    print(" VOYNICHDuctus DATASET PACKAGER (GITHUB SAFE < 100MB)")
    print("=" * 80)

    # 1. Voynich Coordinates (Public domain manuscript, derived vectors)
    v_parts = package_subfolder("voynich", atlas_dir, out_release, "voynich_coords")

    # 2. Codex Seraphinianus Coordinates (Derived vectors only, NO images)
    s_parts = package_subfolder("seraphinianus", atlas_dir, out_release, "seraphini_coords")

    print("\n" + "=" * 80)
    print(f"[✓] Summary: {len(v_parts)} Voynich parts + {len(s_parts)} Seraphinianus parts generated in {out_release}")
    for p in v_parts + s_parts:
        print(f"    - {p.name:<30} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
