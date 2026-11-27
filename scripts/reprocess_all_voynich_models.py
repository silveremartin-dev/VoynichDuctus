"""
Gentle Multi-Core Batch Multi-Model Glyph Reprocessor for VoynichDuctus.
Computes all 3 segmentation models (A, B, C) with MetricScaleCalibration (mm/0.1 mm)
and consensus ductus reuse across Voynich and/or Codex Seraphinianus pages.

Designed to stay non-blocking on shared local workstations:
- Defaults to half the available CPU cores (e.g. 2 workers on a 4-core CPU).
- Sets BELOW_NORMAL OS priority and single-threaded BLAS/OpenCV per worker
  so other concurrent workloads remain responsive.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from voynich_ductus.embeddings.multimodel_pipeline import (
    configure_gentle_worker_environment,
    enrich_page_multimodel,
)
from voynich_ductus.ingestion.codicology import natural_folio_sort_key


def _worker_process_page(
    json_path_str: str,
    img_path_str: str,
    manuscript_type: str,
    low_priority: bool,
    force: bool,
) -> Tuple[str, int, int, int, float, bool]:
    """ProcessPool worker processing a single page with gentle CPU priority."""
    configure_gentle_worker_environment(low_priority=low_priority)
    json_path = Path(json_path_str)
    img_path = Path(img_path_str)
    t0 = time.time()

    with open(json_path, "r", encoding="utf-8") as f:
        page_data = json.load(f)

    page_id = page_data.get("page_id", json_path.stem)

    if not force and page_data.get("glyphs") and page_data.get("segmentation_modes") == ["generative_typewriter_viterbi"]:
        already_hd = (manuscript_type != "voynich") or (int(page_data.get("image_width", 0)) >= 1800)
        if already_hd:
            return (
                json_path.stem,
                len(page_data["glyphs"]),
                0,
                0,
                time.time() - t0,
                True,
            )

    if not page_data.get("words"):
        return json_path.stem, 0, 0, 0, time.time() - t0, True

    if manuscript_type == "voynich":
        import numpy as np
        from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader
        rosetta_loader = VoynicheseRosettaLoader()
        _, pil_hd, _ = rosetta_loader.get_yale_hd_scan_and_words(page_id)
        img = cv2.cvtColor(np.array(pil_hd.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            return json_path.stem, 0, 0, 0, time.time() - t0, True

    a_cnt, b_cnt, c_cnt = enrich_page_multimodel(
        page_data=page_data,
        img_bgr=img,
        manuscript_type=manuscript_type,
        restore_rosetta_boxes=(manuscript_type == "voynich"),
        recompute_option_a=True,
    )

    # Keep scalar metadata keys at the top of the JSON for fast stub indexing
    ordered_data = {
        "page_id": page_data.get("page_id", json_path.stem),
        "page_img_rel": page_data.get("page_img_rel", f"{manuscript_type}/pages/{json_path.stem}.jpg"),
        "image_width": page_data.get("image_width", img.shape[1]),
        "image_height": page_data.get("image_height", img.shape[0]),
        "line_count": page_data.get("line_count", len(page_data.get("lines", []))),
        "word_count": page_data.get("word_count", len(page_data.get("words", []))),
        "glyph_count": page_data.get("glyph_count", a_cnt),
        "segmentation_modes": page_data.get("segmentation_modes", []),
    }
    for k, v in page_data.items():
        if k not in ordered_data:
            ordered_data[k] = v

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, ensure_ascii=False, indent=2)

    return json_path.stem, a_cnt, b_cnt, c_cnt, time.time() - t0, False


def _run_manuscript_batch(
    label: str,
    pages_dir: Path,
    json_files: list,
    manuscript_type: str,
    workers: int,
    low_priority: bool,
    force: bool,
) -> None:
    """Dispatches a manuscript batch across `workers` gentle background processes."""
    tasks = []
    for jf in json_files:
        img_f = pages_dir / f"{jf.stem}.jpg"
        if img_f.exists():
            tasks.append((str(jf), str(img_f), manuscript_type))

    if not tasks:
        print(f"[!] No matching pages found for {label}.")
        return

    print(
        f"[*] Reprocessing {len(tasks)} {label} pages across {workers} gentle worker(s) "
        f"(low_priority={low_priority}, force={force})...",
        flush=True,
    )

    completed = 0
    if workers <= 1:
        for jp, ip, ms in tasks:
            stem, a_c, b_c, c_c, dt, skipped = _worker_process_page(jp, ip, ms, low_priority, force)
            completed += 1
            tag = "CACHED" if skipped else "Done"
            print(f"[{label} {completed}/{len(tasks)}] {stem}: {tag} | A={a_c}, B={b_c}, C={c_c} in {dt:.2f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker_process_page, jp, ip, ms, low_priority, force): Path(jp).stem
                for jp, ip, ms in tasks
            }
            for fut in as_completed(futures):
                stem_hint = futures[fut]
                completed += 1
                try:
                    stem, a_c, b_c, c_c, dt, skipped = fut.result()
                    tag = "CACHED" if skipped else "Done"
                    print(f"[{label} {completed}/{len(tasks)}] {stem}: {tag} | A={a_c}, B={b_c}, C={c_c} in {dt:.2f}s", flush=True)
                except Exception as exc:
                    print(f"[{label} {completed}/{len(tasks)}] {stem_hint}: Error: {exc}", flush=True)


def main():
    cpu_cnt = os.cpu_count() or 4
    default_workers = max(1, cpu_cnt // 2)  # e.g. 2 workers on 4-core PC -> leaves 50% CPU free

    parser = argparse.ArgumentParser(description="Gentle Multi-Core Multi-Model Glyph Reprocessor")
    parser.add_argument("limit", nargs="?", type=int, default=0, help="Optional positional limit (Voynich folios)")
    parser.add_argument("--manuscripts", type=str, default="voynich,seraphinianus", help="Comma-separated: voynich,seraphinianus")
    parser.add_argument("--voynich-limit", type=int, default=0, help="Max Voynich folios (0 = all)")
    parser.add_argument("--serafini-limit", type=int, default=0, help="Max Seraphinianus pages (0 = all)")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Parallel workers (default: {default_workers} on {cpu_cnt}-core host)")
    parser.add_argument("--no-low-priority", action="store_true", help="Disable BELOW_NORMAL/nice CPU throttling")
    parser.add_argument("--force", action="store_true", help="Recompute even if A, B, C already exist on page")
    parser.add_argument("--skip-explorer", action="store_true", help="Skip rebuilding grand_glyph_explorer.html")
    args = parser.parse_args()

    low_priority = not args.no_low_priority
    configure_gentle_worker_environment(low_priority=low_priority)

    v_limit = args.voynich_limit or args.limit
    s_limit = args.serafini_limit
    ms_list = [m.strip().lower() for m in args.manuscripts.split(",") if m.strip()]

    total_start = time.time()

    if "voynich" in ms_list:
        v_pages_dir = root_dir / "data" / "processed" / "atlas_dataset" / "voynich" / "pages"
        v_files = sorted(v_pages_dir.glob("*.json"), key=lambda p: natural_folio_sort_key(p.stem))
        if v_limit > 0:
            v_files = v_files[:v_limit]
        _run_manuscript_batch("Voynich", v_pages_dir, v_files, "voynich", max(1, args.workers), low_priority, args.force)

    if "seraphinianus" in ms_list:
        s_pages_dir = root_dir / "data" / "processed" / "atlas_dataset" / "seraphinianus" / "pages"
        s_files = sorted(s_pages_dir.glob("*.json"))
        if s_limit > 0:
            s_files = s_files[:s_limit]
        _run_manuscript_batch("Seraphinianus", s_pages_dir, s_files, "seraphinianus", max(1, args.workers), low_priority, args.force)

    print(f"\n[✓] Completed multi-core reprocessing in {time.time() - total_start:.2f}s!", flush=True)

    if not args.skip_explorer:
        print("\n[*] Rebuilding Grand Glyph Explorer HTML...", flush=True)
        sys.path.insert(0, str(root_dir / "scripts"))
        import build_dynamic_grand_explorer
        build_dynamic_grand_explorer.main()
        print("[✓] Explorer rebuilt with all multi-model glyph sets!", flush=True)


if __name__ == "__main__":
    main()
