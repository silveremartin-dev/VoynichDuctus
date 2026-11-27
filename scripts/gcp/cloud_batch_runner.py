"""
Google Cloud Batch & High-Throughput Paleography Orchestrator.
Supports multi-core parallel execution (ProcessPoolExecutor), batch DINOv2 inference,
and bi-directional Google Cloud Storage (GCS) synchronization.
"""

import os
import sys
import argparse
import time
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np

# Ensure src/ and scripts/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.metric_scale import MetricScaleCalibration
from voynich_ductus.ingestion.pdf_loader import PDFScanLoader
from voynich_ductus.ingestion.codicology import natural_folio_sort_key, build_folio_codicology_registry
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.ingestion.transcription_reference import PaleographyYieldValidator
from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader
from voynich_ductus.embeddings.multimodel_pipeline import (
    configure_gentle_worker_environment,
    enrich_page_multimodel,
)


def process_single_voynich_folio(
    folio_id: str,
    scan_path: str,
    base_out_str: str,
    rosetta_words_serialized: Optional[List[Dict[str, Any]]] = None,
    force: bool = False
) -> Dict[str, Any]:
    """Worker function: executes zero-waste extraction on a single Voynich folio."""
    base_out = Path(base_out_str)
    pages_dir = base_out / "voynich" / "pages"
    p_json = pages_dir / f"{folio_id}.json"
    p_jpg = pages_dir / f"{folio_id}.jpg"

    if not force and p_json.exists():
        try:
            with open(p_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("glyphs") and d.get("segmentation_modes") == ["generative_typewriter_viterbi"]:
                is_preload = (folio_id.lower() in ["f001r", "f1r"])
                return {
                    "page_id": d.get("page_id", folio_id),
                    "page_img_rel": d.get("page_img_rel", f"voynich/pages/{folio_id}.jpg"),
                    "json_rel": f"voynich/pages/{folio_id}.json",
                    "image_width": d.get("image_width", 1272),
                    "image_height": d.get("image_height", 1800),
                    "line_count": d.get("line_count", len(d.get("lines", []))),
                    "word_count": d.get("word_count", len(d.get("words", []))),
                    "glyph_count": d.get("glyph_count", len(d.get("glyphs", []))),
                    "yield_report": d.get("yield_report", {}),
                    "sample_glyphs": d.get("glyphs", [])[:30],
                    "lines": d.get("lines", []) if is_preload else [],
                    "words": d.get("words", []) if is_preload else [],
                    "glyphs": d.get("glyphs", []) if is_preload else [],
                    "cached": True
                }
        except Exception:
            pass

    configure_gentle_worker_environment(low_priority=False)
    rosetta_loader = VoynicheseRosettaLoader()
    _, yale_img, yale_r_words = rosetta_loader.get_yale_hd_scan_and_words(folio_id)
    scale_calib = MetricScaleCalibration(manuscript_type="voynich", image_height_px=yale_img.height, image_width_px=yale_img.width)
    segmenter = GlyphSegmenter(scale_calib=scale_calib)

    if yale_r_words:
        res = segmenter.extract_page_glyphs_with_ground_truth(
            page_id=folio_id,
            image=yale_img,
            rosetta_words=yale_r_words,
            output_dir=base_out,
            subfolder="voynich"
        )
        print(f"  [HD] Aligned {folio_id} on Yale HD scan ({yale_img.width}x{yale_img.height}).")
    else:
        res = segmenter.extract_page_glyphs(
            page_id=folio_id,
            image=yale_img,
            output_dir=base_out,
            subfolder="voynich"
        )

    # Enrich with all 3 decomposition models (A, B, C), MetricScaleCalibration, and line-aware disjoint word partitioning
    import cv2 as _cv2
    img_bgr = _cv2.cvtColor(np.array(yale_img.convert("RGB")), _cv2.COLOR_RGB2BGR)
    enrich_page_multimodel(
        page_data=res,
        img_bgr=img_bgr,
        manuscript_type="voynich",
        restore_rosetta_boxes=False,
        recompute_option_a=True,
    )

    # Census yield evaluation
    yr = PaleographyYieldValidator.evaluate_yield(
        folio_id=folio_id,
        detected_lines=res.get("line_count", 0),
        detected_glyphs=res.get("glyph_count", 0),
        extracted_strokes=sum(len(g.get("strokes", [])) for g in res.get("glyphs", []))
    )
    res["yield_report"] = yr

    with open(p_json, "w", encoding="utf-8") as f_out:
        json.dump(res, f_out, ensure_ascii=False)

    # Return lightweight summary with representative sample across IPC boundary
    is_preload = (folio_id.lower() in ["f001r", "f1r"])
    return {
        "page_id": folio_id,
        "page_img_rel": res.get("page_img_rel", f"voynich/pages/{folio_id}.jpg"),
        "json_rel": f"voynich/pages/{folio_id}.json",
        "image_width": res.get("image_width", 1272),
        "image_height": res.get("image_height", 1800),
        "line_count": res.get("line_count", 0),
        "word_count": res.get("word_count", 0),
        "glyph_count": res.get("glyph_count", 0),
        "glyph_count_neural_pca": res.get("glyph_count_neural_pca", 0),
        "glyph_count_hybrid_consensus": res.get("glyph_count_hybrid_consensus", 0),
        "yield_report": yr,
        "sample_glyphs": res.get("glyphs", [])[:30],
        "lines": res.get("lines", []) if is_preload else [],
        "words": res.get("words", []) if is_preload else [],
        "glyphs": res.get("glyphs", []) if is_preload else [],
        "glyphs_neural_pca": res.get("glyphs_neural_pca", []) if is_preload else [],
        "glyphs_hybrid_consensus": res.get("glyphs_hybrid_consensus", []) if is_preload else [],
    }


def process_single_serafini_page(
    page_idx: int,
    pdf_path_str: str,
    base_out_str: str
) -> Dict[str, Any]:
    """Worker function: executes topological loop-preserving 3-model extraction on a single Seraphinianus page."""
    base_out = Path(base_out_str)
    page_id = f"serafini_p{page_idx:03d}"
    pages_dir = base_out / "seraphinianus" / "pages"
    p_json = pages_dir / f"{page_id}.json"
    p_jpg = pages_dir / f"{page_id}.jpg"
    
    # Check if page was already extracted with all 3 models to avoid re-work
    if p_json.exists() and p_jpg.exists():
        try:
            with open(p_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("glyphs") and d.get("segmentation_modes") == ["generative_typewriter_viterbi"]:
                is_preload = (page_id in ["serafini_p000", "serafini_p030"])
                return {
                    "page_id": page_id,
                    "page_img_rel": d.get("page_img_rel", f"seraphinianus/pages/{page_id}.jpg"),
                    "json_rel": f"seraphinianus/pages/{page_id}.json",
                    "image_width": d.get("image_width", 1600),
                    "image_height": d.get("image_height", 2206),
                    "line_count": d.get("line_count", 0),
                    "word_count": d.get("word_count", 0),
                    "glyph_count": d.get("glyph_count", 0),
                    "sample_glyphs": d.get("glyphs", [])[:15],
                    "lines": d.get("lines", []) if is_preload else [],
                    "words": d.get("words", []) if is_preload else [],
                    "glyphs": d.get("glyphs", []) if is_preload else [],
                    "cached": True
                }
        except Exception:
            pass

    configure_gentle_worker_environment(low_priority=False)
    loader_s = PDFScanLoader(Path(pdf_path_str))
    # 1600px is crystal clear for digital PDF cursive and avoids multi-gigabyte memory spikes
    p_img = loader_s.get_page_image(page_idx, target_min_dim=1600)
    
    # Auto-orient landscape plates if width exceeds height
    if p_img.width > p_img.height * 1.15:
        p_img = p_img.rotate(270, expand=True)

    scale_calib = MetricScaleCalibration(manuscript_type="seraphinianus", image_height_px=p_img.height, image_width_px=p_img.width)
    segmenter = GlyphSegmenter(scale_calib=scale_calib)
    res_s = segmenter.extract_page_glyphs(page_id, p_img, output_dir=base_out, subfolder="seraphinianus")

    import cv2 as _cv2
    img_bgr = _cv2.cvtColor(np.array(p_img.convert("RGB")), _cv2.COLOR_RGB2BGR)
    enrich_page_multimodel(
        page_data=res_s,
        img_bgr=img_bgr,
        manuscript_type="seraphinianus",
        restore_rosetta_boxes=False,
        recompute_option_a=True,
    )
    with open(p_json, "w", encoding="utf-8") as f_out:
        json.dump(res_s, f_out, ensure_ascii=False)

    is_preload = (page_id in ["serafini_p000", "serafini_p030"])
    # Return ONLY lightweight summary across IPC boundary to keep parent memory near zero
    return {
        "page_id": page_id,
        "page_img_rel": res_s.get("page_img_rel", f"seraphinianus/pages/{page_id}.jpg"),
        "json_rel": f"seraphinianus/pages/{page_id}.json",
        "image_width": res_s.get("image_width", p_img.width),
        "image_height": res_s.get("image_height", p_img.height),
        "line_count": res_s.get("line_count", 0),
        "word_count": res_s.get("word_count", 0),
        "glyph_count": res_s.get("glyph_count", 0),
        "glyph_count_neural_pca": res_s.get("glyph_count_neural_pca", 0),
        "glyph_count_hybrid_consensus": res_s.get("glyph_count_hybrid_consensus", 0),
        "sample_glyphs": res_s.get("glyphs", [])[:15],
        "lines": res_s.get("lines", []) if is_preload else [],
        "words": res_s.get("words", []) if is_preload else [],
        "glyphs": res_s.get("glyphs", []) if is_preload else [],
        "glyphs_neural_pca": res_s.get("glyphs_neural_pca", []) if is_preload else [],
        "glyphs_hybrid_consensus": res_s.get("glyphs_hybrid_consensus", []) if is_preload else [],
        "cached": False
    }


def sync_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str = "voynich_ductus/atlas_dataset", max_workers: int = 16):
    """Fast multi-threaded upload of dataset assets to Google Cloud Storage."""
    try:
        from google.cloud import storage
        from concurrent.futures import ThreadPoolExecutor
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        files_to_upload = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file = Path(root) / file
                rel_path = local_file.relative_to(local_dir).as_posix()
                blob_path = f"{gcs_prefix}/{rel_path}"
                content_type = (
                    "text/html" if file.endswith(".html")
                    else "application/json" if file.endswith(".json")
                    else "image/jpeg" if file.endswith(".jpg") or file.endswith(".jpeg")
                    else "image/png" if file.endswith(".png")
                    else "image/svg+xml" if file.endswith(".svg")
                    else "application/octet-stream"
                )
                files_to_upload.append((local_file, blob_path, content_type))

        print(f"\n[GCS] Syncing {len(files_to_upload)} files from {local_dir} to gs://{bucket_name}/{gcs_prefix} (parallel workers={max_workers})...")
        def _upload_file(item):
            loc_path, b_path, c_type = item
            blob = bucket.blob(b_path)
            blob.upload_from_filename(str(loc_path), content_type=c_type)
            return 1

        uploaded = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for _ in pool.map(_upload_file, files_to_upload):
                uploaded += 1
                if uploaded % 100 == 0 or uploaded == len(files_to_upload):
                    print(f"  [GCS] Uploaded {uploaded}/{len(files_to_upload)} objects...")

        print(f"[GCS] Successfully uploaded {uploaded} objects to gs://{bucket_name}/{gcs_prefix}")
    except Exception as e:
        print(f"[-] GCS upload notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="VoynichDuctus GCP Batch Orchestrator")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Number of parallel worker processes (default: CPU cores - 1)")
    parser.add_argument("--manuscripts", type=str, default="voynich,seraphinianus",
                        help="Comma-separated manuscripts to process: voynich, seraphinianus")
    parser.add_argument("--output-dir", type=str, default="./data/processed/atlas_dataset",
                        help="Output directory for extracted atlas and web explorer")
    parser.add_argument("--gcs-bucket", type=str, default=None,
                        help="Google Cloud Storage bucket name for final synchronization")
    parser.add_argument("--gcs-prefix", type=str, default="voynich_ductus/atlas_dataset",
                        help="GCS path prefix for uploaded assets")
    parser.add_argument("--voynich-limit", type=int, default=None,
                        help="Max Voynich folios to process (for staging / test runs)")
    parser.add_argument("--serafini-limit", type=int, default=None,
                        help="Max Seraphinianus pages to process")
    parser.add_argument("--force-voynich", action="store_true", default=False,
                        help="Force re-extraction of Voynich folios even if cached")
    args = parser.parse_args()

    t_start = time.perf_counter()
    base_out = Path(args.output_dir)
    print("=" * 65)
    print("  VOYNICHDUTUS — GCP BATCH HIGH-THROUGHPUT ORCHESTRATOR")
    print(f"  Workers: {args.workers} parallel processes | Manuscripts: {args.manuscripts}")
    print("=" * 65)

    # 1. Clean output directories
    (base_out / "voynich" / "pages").mkdir(parents=True, exist_ok=True)
    (base_out / "seraphinianus" / "pages").mkdir(parents=True, exist_ok=True)

    do_voynich = "voynich" in args.manuscripts.lower()
    do_serafini = "seraphinianus" in args.manuscripts.lower()

    voynich_pages_data = []
    all_voynich_glyphs = []
    voynich_catalogue = {"canonical_alphabet_size": 0, "alphabet": []}

    # Check if Voynich is already complete on GCS
    voy_cat_file = base_out / "voynich" / "voynich_alphabet_catalogue.json"
    if do_voynich and not args.force_voynich and args.gcs_bucket and not voy_cat_file.exists():
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            b = bucket.blob(f"{args.gcs_prefix}/voynich/voynich_alphabet_catalogue.json")
            if b.exists():
                print(f"[GCS] Found complete Voynich corpus on GCS! Downloading metadata to skip re-extraction...")
                b.download_to_filename(str(voy_cat_file))
                for pb in bucket.list_blobs(prefix=f"{args.gcs_prefix}/voynich/pages/"):
                    if pb.name.endswith(".json"):
                        fname = Path(pb.name).name
                        pb.download_to_filename(str(base_out / "voynich" / "pages" / fname))
                print(f"[GCS] Voynich metadata downloaded successfully.")
        except Exception as e:
            print(f"[GCS] Voynich sync notice: {e}")

    if do_voynich and not args.force_voynich and voy_cat_file.exists() and len(list((base_out / "voynich" / "pages").glob("*.json"))) >= 220:
        print("[*] Voynich already 100% complete! Loading catalogue and pages from cache...")
        try:
            with open(voy_cat_file, "r", encoding="utf-8") as f:
                voynich_catalogue = json.load(f)
            v_files = sorted((base_out / "voynich" / "pages").glob("*.json"), key=lambda p: natural_folio_sort_key(p.stem))
            cod_reg = build_folio_codicology_registry([jf.stem for jf in v_files])
            all_have_multimodel = True
            for p_json in v_files:
                with open(p_json, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("segmentation_modes") != ["generative_typewriter_viterbi"]:
                    all_have_multimodel = False
                p_id = d.get("page_id", p_json.stem)
                c_info = cod_reg.get(p_id, {})
                is_preload = (p_id.lower() in ["f001r", "f1r"])
                voynich_pages_data.append({
                    "page_id": p_id,
                    "page_img_rel": d.get("page_img_rel", f"voynich/pages/{p_id}.jpg"),
                    "json_rel": f"voynich/pages/{p_id}.json",
                    "image_width": d.get("image_width", 1272),
                    "image_height": d.get("image_height", 1800),
                    "line_count": d.get("line_count", len(d.get("lines", []))),
                    "word_count": d.get("word_count", len(d.get("words", []))),
                    "glyph_count": d.get("glyph_count", len(d.get("glyphs", []))),
                    "glyph_count_neural_pca": d.get("glyph_count_neural_pca", len(d.get("glyphs_neural_pca", []))),
                    "glyph_count_hybrid_consensus": d.get("glyph_count_hybrid_consensus", len(d.get("glyphs_hybrid_consensus", []))),
                    "yield_report": d.get("yield_report", {}),
                    "quire_id": c_info.get("quire", ""),
                    "section": c_info.get("section", ""),
                    "singulion_id": c_info.get("singulion_id", ""),
                    "conjoint_leaf": c_info.get("conjoint_leaf"),
                    "codex_index": c_info.get("codex_index", 0),
                    "ld_index": c_info.get("ld_index", 0),
                    "notes": c_info.get("notes", ""),
                    "lines": d.get("lines", []) if is_preload else [],
                    "words": d.get("words", []) if is_preload else [],
                    "glyphs": d.get("glyphs", []) if is_preload else [],
                    "glyphs_neural_pca": d.get("glyphs_neural_pca", []) if is_preload else [],
                    "glyphs_hybrid_consensus": d.get("glyphs_hybrid_consensus", []) if is_preload else [],
                })
            if all_have_multimodel:
                print(f"  [+] Loaded {len(voynich_pages_data)} Voynich folios (A+B+C) from cache. Skipping Voynich extraction!")
                do_voynich = False
            else:
                voynich_pages_data = []
                print("  [i] Cached Voynich folios missing B/C models; running 3-model enrichment...")
        except Exception as e:
            print(f"[-] Error loading cached Voynich data: {e}")

    # -------------------------------------------------------------
    # 2. Parallel Processing: Voynich Manuscript
    # -------------------------------------------------------------
    if do_voynich:
        print("\n[*] Initializing Voynich Manuscript Yale Beinecke HD extraction queue...")
        rosetta_loader = VoynicheseRosettaLoader()

        all_avail = rosetta_loader.list_available_folios()
        if all_avail:
            folios = all_avail
        else:
            folios = ["f001r", "f001v", "f002r", "f002v", "f003r", "f003v", "f004r", "f004v"]
        if args.voynich_limit:
            folios = folios[:args.voynich_limit]

        # Pre-sync any completed Voynich pages from GCS if available
        if args.gcs_bucket and not args.force_voynich:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(args.gcs_bucket)
                prefix = f"{args.gcs_prefix}/voynich/pages/"
                existing_blobs = list(bucket.list_blobs(prefix=prefix))
                if existing_blobs:
                    print(f"[GCS] Found {len(existing_blobs)} existing Voynich objects on GCS. Syncing to local cache...")
                    pages_dir = base_out / "voynich" / "pages"
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    for b in existing_blobs:
                        fname = Path(b.name).name
                        dest = pages_dir / fname
                        if not dest.exists():
                            b.download_to_filename(str(dest))
            except Exception as e:
                print(f"[GCS] Voynich pre-sync notice: {e}")

        # Prepare task payloads (Yale Beinecke HD scans are downloaded on demand inside workers)
        tasks = [(f_id, "", str(base_out), None, args.force_voynich) for f_id in folios]

        print(f"[*] Dispatching {len(tasks)} Voynich folios across {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_folio = {
                executor.submit(process_single_voynich_folio, t[0], t[1], t[2], t[3], t[4]): t[0]
                for t in tasks
            }
            for future in as_completed(future_to_folio):
                f_id = future_to_folio[future]
                try:
                    res = future.result()
                    voynich_pages_data.append(res)
                    all_voynich_glyphs.extend(res.get("sample_glyphs", res.get("glyphs", [])))
                    status_str = "CACHED" if res.get("cached") else "Extracted"
                    print(f"  [+] ({len(voynich_pages_data)}/{len(tasks)}) {status_str} {f_id}: {res['line_count']} lines, {res['word_count']} words, {res['glyph_count']} glyphs.")
                except Exception as e:
                    print(f"  [-] Error on {f_id}: {e}")

        # Sort pages in canonical codicological sequence and attach codicology metadata
        voynich_pages_data.sort(key=lambda p: natural_folio_sort_key(p["page_id"]))
        cod_reg = build_folio_codicology_registry([p["page_id"] for p in voynich_pages_data])
        for idx_p, vp in enumerate(voynich_pages_data):
            c_info = cod_reg.get(vp["page_id"], {})
            vp["quire_id"] = c_info.get("quire", "")
            vp["section"] = c_info.get("section", "")
            vp["singulion_id"] = c_info.get("singulion_id", "")
            vp["conjoint_leaf"] = c_info.get("conjoint_leaf")
            vp["codex_index"] = c_info.get("codex_index", idx_p + 1)
            vp["ld_index"] = c_info.get("ld_index", idx_p + 1)
            vp["notes"] = c_info.get("notes", "")

        print(f"[*] Inducing Voynich Alphabet Catalogue across {len(all_voynich_glyphs)} extracted glyphs...")
        v_scale = MetricScaleCalibration(manuscript_type="voynich", image_height_px=1800, image_width_px=1272)
        builder_v = GlyphCatalogue(target_alphabet_size=35)
        voynich_catalogue = builder_v.build_catalogue(all_voynich_glyphs, corpus_type="voynich", scale_calib=v_scale)
        builder_v.export_catalogue_json(voynich_catalogue, base_out / "voynich" / "voynich_alphabet_catalogue.json")
        print(f"  [+] Voynich Alphabet: {voynich_catalogue['canonical_alphabet_size']} canonical archetypes.")

        if args.gcs_bucket:
            print("[*] Checkpoint sync: Syncing completed Voynich corpus to GCS...")
            sync_to_gcs(base_out / "voynich", args.gcs_bucket, f"{args.gcs_prefix}/voynich")

    # -------------------------------------------------------------
    # 3. Parallel Processing: Codex Seraphinianus
    # -------------------------------------------------------------
    serafini_pages_data = []
    all_serafini_glyphs = []
    serafini_catalogue = {"canonical_alphabet_size": 0, "alphabet": []}

    if do_serafini:
        print("\n[*] Initializing Codex Seraphinianus extraction queue...")
        pdf_path = Path("data/raw/scans/seraphinianus/Codex Seraphinianus.pdf")
        if pdf_path.exists():
            loader_s = PDFScanLoader(pdf_path)
            total_pages = len(loader_s)
            if args.serafini_limit:
                pages = list(range(min(total_pages, args.serafini_limit)))
            else:
                pages = list(range(total_pages))

            # Pre-sync any completed Seraphinianus pages from GCS if available
            if args.gcs_bucket:
                try:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(args.gcs_bucket)
                    prefix = f"{args.gcs_prefix}/seraphinianus/pages/"
                    existing_blobs = list(bucket.list_blobs(prefix=prefix))
                    if existing_blobs:
                        print(f"[GCS] Found {len(existing_blobs)} existing Seraphinianus objects on GCS. Syncing to local cache...")
                        pages_dir = base_out / "seraphinianus" / "pages"
                        pages_dir.mkdir(parents=True, exist_ok=True)
                        for b in existing_blobs:
                            fname = Path(b.name).name
                            dest = pages_dir / fname
                            if not dest.exists():
                                b.download_to_filename(str(dest))
                except Exception as e:
                    print(f"[GCS] Pre-sync notice: {e}")

            print(f"[*] Dispatching {len(pages)} Seraphinianus pages across {args.workers} workers...")
            completed_count = 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_page = {
                    executor.submit(process_single_serafini_page, p, str(pdf_path), str(base_out)): p
                    for p in pages
                }
                for future in as_completed(future_to_page):
                    p_idx = future_to_page[future]
                    try:
                        res_s = future.result()
                        all_serafini_glyphs.extend(res_s.get("sample_glyphs", []))
                        completed_count += 1
                        status_str = "CACHED" if res_s.get("cached") else "Extracted"
                        print(f"  [+] ({completed_count}/{len(pages)}) {status_str} page {p_idx}: {res_s['line_count']} lines, {res_s['word_count']} words, {res_s['glyph_count']} glyphs.")
                        
                        # Checkpoint sync to GCS every 40 pages
                        if args.gcs_bucket and completed_count % 40 == 0:
                            print(f"  [*] Checkpoint sync ({completed_count}/{len(pages)}) to GCS...")
                            sync_to_gcs(base_out / "seraphinianus" / "pages", args.gcs_bucket, f"{args.gcs_prefix}/seraphinianus/pages")
                    except Exception as e:
                        print(f"  [-] Error on page {p_idx}: {e}")

            if args.gcs_bucket:
                sync_to_gcs(base_out / "seraphinianus" / "pages", args.gcs_bucket, f"{args.gcs_prefix}/seraphinianus/pages")

            print(f"[*] Inducing Seraphinianus Alphabet Catalogue across {len(all_serafini_glyphs)} sample glyphs...")
            s_scale = MetricScaleCalibration(manuscript_type="seraphinianus", image_height_px=2206, image_width_px=1600)
            builder_s = GlyphCatalogue(target_alphabet_size=52)
            serafini_catalogue = builder_s.build_catalogue(all_serafini_glyphs, corpus_type="seraphinianus", scale_calib=s_scale)
            builder_s.export_catalogue_json(serafini_catalogue, base_out / "seraphinianus" / "seraphinianus_alphabet_catalogue.json")
            print(f"  [+] Seraphinianus Alphabet: {serafini_catalogue['canonical_alphabet_size']} canonical archetypes.")

            # Load page metadata for Grand Explorer assembly without memory bloat
            print("[*] Loading Seraphinianus page metadata for Grand Explorer assembly...")
            for p_json in sorted((base_out / "seraphinianus" / "pages").glob("*.json")):
                try:
                    with open(p_json, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    p_id = d.get("page_id", p_json.stem)
                    is_preload = (p_id in ["serafini_p000", "serafini_p030"])
                    serafini_pages_data.append({
                        "page_id": p_id,
                        "page_img_rel": d.get("page_img_rel", f"seraphinianus/pages/{p_id}.jpg"),
                        "json_rel": f"seraphinianus/pages/{p_id}.json",
                        "image_width": d.get("image_width", 1600),
                        "image_height": d.get("image_height", 2206),
                        "line_count": d.get("line_count", len(d.get("lines", []))),
                        "word_count": d.get("word_count", len(d.get("words", []))),
                        "glyph_count": d.get("glyph_count", len(d.get("glyphs", []))),
                        "glyph_count_neural_pca": d.get("glyph_count_neural_pca", len(d.get("glyphs_neural_pca", []))),
                        "glyph_count_hybrid_consensus": d.get("glyph_count_hybrid_consensus", len(d.get("glyphs_hybrid_consensus", []))),
                        "yield_report": d.get("yield_report", {}),
                        "lines": d.get("lines", []) if is_preload else [],
                        "words": d.get("words", []) if is_preload else [],
                        "glyphs": d.get("glyphs", []) if is_preload else [],
                        "glyphs_neural_pca": d.get("glyphs_neural_pca", []) if is_preload else [],
                        "glyphs_hybrid_consensus": d.get("glyphs_hybrid_consensus", []) if is_preload else [],
                    })
                except Exception:
                    pass

    # -------------------------------------------------------------
    # 4. Generate Interactive Grand Explorer Application
    # -------------------------------------------------------------
    try:
        from scripts.generate_full_glyph_corpus_and_explorer import generate_explorer_html
        from scripts.build_dynamic_grand_explorer import clean_catalogue_for_embed
    except ImportError:
        from generate_full_glyph_corpus_and_explorer import generate_explorer_html
        from build_dynamic_grand_explorer import clean_catalogue_for_embed
    explorer_path = base_out / "grand_glyph_explorer.html"
    generate_explorer_html(
        voynich_pages_data,
        clean_catalogue_for_embed(voynich_catalogue),
        serafini_pages_data,
        clean_catalogue_for_embed(serafini_catalogue),
        explorer_path,
    )

    # -------------------------------------------------------------
    # 5. Optional GCS Synchronization
    # -------------------------------------------------------------
    if args.gcs_bucket:
        sync_to_gcs(base_out, args.gcs_bucket, args.gcs_prefix)

    elapsed_s = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print(f"  BATCH RUN COMPLETE in {elapsed_s:.1f}s ({elapsed_s/60:.2f} min)")
    print(f"  Total Voynich Glyphs: {len(all_voynich_glyphs)}")
    print(f"  Total Seraphinianus Glyphs: {len(all_serafini_glyphs)}")
    print(f"  Interactive Explorer: {explorer_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
