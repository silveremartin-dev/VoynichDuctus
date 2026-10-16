"""
Dataset Unpacker for VoynichDuctus.
Unpacks all partitioned coordinate archives from releases/dataset_zips into output/atlas_dataset.
"""

import zipfile
from pathlib import Path

def unpack_all():
    zip_dir = Path("releases/dataset_zips")
    dest_dir = Path("output/atlas_dataset")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(list(zip_dir.glob("*.zip")))
    if not zip_files:
        print(f"[-] No zip archives found in {zip_dir}")
        return
        
    print(f"[*] Found {len(zip_files)} archives to unpack into {dest_dir}...")
    for zf_path in zip_files:
        print(f"  [+] Unpacking {zf_path.name} ({zf_path.stat().st_size / (1024*1024):.2f} MB)...")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(dest_dir)
            
    print(f"\n[✓] Successfully unpacked all archives into: {dest_dir.resolve()}")

if __name__ == "__main__":
    unpack_all()
