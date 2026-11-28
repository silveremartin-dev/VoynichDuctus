"""
Full-Codex Millimeter Ductus & Segmentation Reprocessing Pipeline.
Reprocesses all Voynich Manuscript folios (225) and all Codex Seraphinianus pages (371)
in `data/processed/atlas_dataset/` with resumable atomic checkpointing per page,
then dynamically rebuilds the Unified Emergent Canonical Alphabets, `live_stats`,
`grand_glyph_explorer.html`, and `README.md`.
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

CURRENT_ENGINE_STAMP = "humanist_quill_mm_v4"


def _process_single_page(args: Tuple[str, str, bool]) -> Tuple[str, str, float, int, int, str]:
    jf_str, ms_type, force = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    sys.path.insert(0, str(ROOT_DIR / "src"))
    from voynich_ductus.embeddings.multimodel_pipeline import (
        configure_gentle_worker_environment,
        enrich_page_multimodel,
    )

    configure_gentle_worker_environment(low_priority=True)
    jf = Path(jf_str)
    t0 = time.perf_counter()

    try:
        with open(jf, "r", encoding="utf-8") as f:
            head = f.read(2048)
        if not force and f'"ductus_engine_version": "{CURRENT_ENGINE_STAMP}"' in head:
            return (ms_type, jf.stem, 0.0, -1, 0, "skipped_already_v3")

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return (ms_type, jf.stem, 0.0, 0, 0, f"read_error:{exc}")

    words = data.get("words", [])
    if not words:
        # Illustration-only or blank cover page (0 words): stamp header cleanly
        ordered_empty = {
            "page_id": data.get("page_id", jf.stem),
            "ductus_engine_version": CURRENT_ENGINE_STAMP,
            "page_img_rel": data.get("page_img_rel", f"{ms_type}/pages/{jf.stem}.jpg"),
            "image_width": data.get("image_width", 1600),
            "image_height": data.get("image_height", 2200),
            "line_count": 0,
            "word_count": 0,
            "glyph_count": 0,
            "segmentation_modes": ["generative_typewriter_viterbi"],
            "lines": [],
            "words": [],
            "glyphs": [],
        }
        tmp_p = jf.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(ordered_empty, ensure_ascii=False), encoding="utf-8")
        tmp_p.replace(jf)
        return (ms_type, jf.stem, round(time.perf_counter() - t0, 2), 0, 0, "ok_empty")

    img_path = jf.parent / f"{jf.stem}.jpg"
    if not img_path.exists():
        img_path = ROOT_DIR / "data/processed/atlas_dataset" / data.get("page_img_rel", "")
    img = cv2.imread(str(img_path))
    if img is None:
        return (ms_type, jf.stem, 0.0, 0, 0, "missing_jpg")

    ca, _, _ = enrich_page_multimodel(
        data,
        img,
        manuscript_type=ms_type,
        restore_rosetta_boxes=(ms_type == "voynich" and img.shape[1] >= 2000),
        recompute_option_a=True,
    )
    data.pop("glyphs_neural_pca", None)
    data.pop("glyphs_hybrid_consensus", None)
    data.pop("glyph_count_neural_pca", None)
    data.pop("glyph_count_hybrid_consensus", None)

    ordered = {
        "page_id": data.get("page_id", jf.stem),
        "ductus_engine_version": CURRENT_ENGINE_STAMP,
        "page_img_rel": data.get("page_img_rel", f"{ms_type}/pages/{jf.stem}.jpg"),
        "image_width": int(img.shape[1]),
        "image_height": int(img.shape[0]),
        "line_count": len(data.get("lines", [])),
        "word_count": len(data.get("words", [])),
        "glyph_count": int(ca),
        "segmentation_modes": ["generative_typewriter_viterbi"],
    }
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v

    holes = sum(1 for g in ordered.get("glyphs", []) if g.get("hole_polygons"))
    tmp_path = jf.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(jf)
    return (ms_type, jf.stem, round(time.perf_counter() - t0, 2), int(ca), int(holes), "ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess all Voynich & Seraphinianus pages and rebuild Explorer.")
    parser.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--force", action="store_true", help="Force recompute even if stamped with v3")
    parser.add_argument("--skip-explorer", action="store_true", help="Only reprocess page JSONs")
    parser.add_argument("--pages", type=str, default="", help="Comma-separated page stems to reprocess selectively")
    args = parser.parse_args()

    t_start = time.perf_counter()
    base = ROOT_DIR / "data/processed/atlas_dataset"
    v_files = sorted((base / "voynich/pages").glob("*.json"))
    s_files = sorted((base / "seraphinianus/pages").glob("*.json"))
    if args.pages.strip():
        wanted = {s.strip() for s in args.pages.split(",") if s.strip()}
        v_files = [p for p in v_files if p.stem in wanted]
        s_files = [p for p in s_files if p.stem in wanted]

    tasks = [(str(p), "voynich", args.force) for p in v_files] + [
        (str(p), "seraphinianus", args.force) for p in s_files
    ]
    total = len(tasks)
    print(f"[*] Starting Full-Codex Reprocessing across {total} pages ({len(v_files)} Voynich + {len(s_files)} Seraphinianus) on {args.workers} workers...", flush=True)

    done_cnt = 0
    total_glyphs = 0
    total_holes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_single_page, t) for t in tasks]
        for fut in as_completed(futures):
            ms, stem, dt, gc, hc, status = fut.result()
            done_cnt += 1
            if gc > 0:
                total_glyphs += gc
                total_holes += hc
            if done_cnt <= 5 or done_cnt % 20 == 0 or done_cnt == total or (status not in ("ok", "ok_empty", "skipped_already_v3")):
                elapsed = time.perf_counter() - t_start
                rate = done_cnt / max(1e-3, elapsed)
                eta_min = ((total - done_cnt) / max(1e-3, rate)) / 60.0
                print(
                    f"  [{done_cnt:3d}/{total}] {ms:13s} {stem:16s} | status={status} | glyphs={gc:4d} | holes={hc:3d} | {dt:5.2f}s | elapsed={elapsed/60.0:5.1f}m | ETA={eta_min:4.1f}m",
                    flush=True,
                )

    print(
        f"[+] Full-Codex page vectorization complete in {(time.perf_counter() - t_start)/60.0:.2f} min! "
        f"(New/updated glyphs={total_glyphs:,}, loop_holes={total_holes:,})",
        flush=True,
    )

    if not args.skip_explorer:
        import build_dynamic_grand_explorer
        build_dynamic_grand_explorer.main(skip_catalogue=False)


if __name__ == "__main__":
    main()
