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

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voynich_ductus.ingestion.glyph_segmenter import GlyphSegmenter
from voynich_ductus.ingestion.pdf_loader import PDFScanLoader
from voynich_ductus.clustering.glyph_catalogue import GlyphCatalogue
from voynich_ductus.ingestion.transcription_reference import PaleographyYieldValidator
from voynich_ductus.ingestion.voynichese_rosetta import VoynicheseRosettaLoader


def process_single_voynich_folio(
    folio_id: str,
    scan_path: str,
    base_out_str: str,
    rosetta_words_serialized: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Worker function: executes zero-waste extraction on a single Voynich folio."""
    segmenter = GlyphSegmenter()
    base_out = Path(base_out_str)
    
    raw_img = Image.open(scan_path)
    # 2x Lanczos upscale for crisp pen stroke resolution (1272x1800)
    img = raw_img.resize((raw_img.width * 2, raw_img.height * 2), Image.Resampling.LANCZOS)
    
    # Reconstruct RosettaWord objects if provided
    r_words = None
    if rosetta_words_serialized:
        from voynich_ductus.ingestion.voynichese_rosetta import RosettaWord
        r_words = []
        for d in rosetta_words_serialized:
            r_words.append(RosettaWord(
                folio=d["folio"],
                index=d["index"],
                eva_text=d.get("eva_text", ""),
                x_xml=d.get("x_xml", 0.0),
                y_xml=d.get("y_xml", 0.0),
                width_xml=d.get("width_xml", 0.0),
                height_xml=d.get("height_xml", 0.0),
                bbox_scan=tuple(d["bbox_scan"]) if d.get("bbox_scan") else None
            ))

    if r_words:
        res = segmenter.extract_page_glyphs_with_ground_truth(
            page_id=folio_id,
            image=img,
            rosetta_words=r_words,
            output_dir=base_out,
            subfolder="voynich"
        )
    else:
        res = segmenter.extract_page_glyphs(
            page_id=folio_id,
            image=img,
            output_dir=base_out,
            subfolder="voynich"
        )

    # Census yield evaluation
    yr = PaleographyYieldValidator.evaluate_yield(
        folio_id=folio_id,
        detected_lines=res["line_count"],
        detected_glyphs=res["glyph_count"],
        extracted_strokes=sum(len(g.get("strokes", [])) for g in res["glyphs"])
    )
    res["yield_report"] = yr
    return res


def process_single_serafini_page(
    page_idx: int,
    pdf_path_str: str,
    base_out_str: str
) -> Dict[str, Any]:
    """Worker function: executes topological loop-preserving extraction on a single Seraphinianus page."""
    segmenter = GlyphSegmenter()
    base_out = Path(base_out_str)
    page_id = f"serafini_p{page_idx:03d}"
    
    loader_s = PDFScanLoader(Path(pdf_path_str))
    p_img = loader_s.get_page_image(page_idx, target_min_dim=2000)
    
    # Auto-orient landscape plates if width exceeds height
    if p_img.width > p_img.height * 1.15:
        p_img = p_img.rotate(270, expand=True)
        
    # 1.5x Lanczos upscale
    p_img_hd = p_img.resize((int(p_img.width * 1.5), int(p_img.height * 1.5)), Image.Resampling.LANCZOS)
    res_s = segmenter.extract_page_glyphs(page_id, p_img_hd, output_dir=base_out, subfolder="seraphinianus")
    return res_s


def sync_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str = "voynich_ductus/atlas_dataset"):
    """Recursively uploads the output dataset to a Google Cloud Storage bucket."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print(f"\n[GCS] Syncing {local_dir} to gs://{bucket_name}/{gcs_prefix} ...")
        uploaded_count = 0
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file = Path(root) / file
                rel_path = local_file.relative_to(local_dir).as_posix()
                blob_path = f"{gcs_prefix}/{rel_path}"
                blob = bucket.blob(blob_path)
                
                content_type = "text/html" if file.endswith(".html") else "application/json" if file.endswith(".json") else "image/png" if file.endswith(".png") else "image/svg+xml" if file.endswith(".svg") else "image/jpeg"
                blob.upload_from_filename(str(local_file), content_type=content_type)
                uploaded_count += 1
                if uploaded_count % 500 == 0:
                    print(f"  [GCS] Uploaded {uploaded_count} objects...")
        print(f"[GCS] Successfully uploaded {uploaded_count} objects to gs://{bucket_name}/{gcs_prefix}")
    except Exception as e:
        print(f"[-] GCS upload notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="VoynichDuctus GCP Batch Orchestrator")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Number of parallel worker processes (default: CPU cores - 1)")
    parser.add_argument("--manuscripts", type=str, default="voynich,seraphinianus",
                        help="Comma-separated manuscripts to process: voynich, seraphinianus")
    parser.add_argument("--output-dir", type=str, default="./output/atlas_dataset",
                        help="Output directory for extracted atlas and web explorer")
    parser.add_argument("--gcs-bucket", type=str, default=None,
                        help="Google Cloud Storage bucket name for final synchronization")
    parser.add_argument("--gcs-prefix", type=str, default="voynich_ductus/atlas_dataset",
                        help="GCS path prefix for uploaded assets")
    parser.add_argument("--voynich-limit", type=int, default=None,
                        help="Max Voynich folios to process (for staging / test runs)")
    parser.add_argument("--serafini-limit", type=int, default=None,
                        help="Max Seraphinianus pages to process")
    args = parser.parse_args()

    t_start = time.perf_counter()
    base_out = Path(args.output_dir)
    print("=" * 65)
    print("  VOYNICHDUTUS — GCP BATCH HIGH-THROUGHPUT ORCHESTRATOR")
    print(f"  Workers: {args.workers} parallel processes | Manuscripts: {args.manuscripts}")
    print("=" * 65)

    # 1. Clean output directories
    for sub in ["voynich", "seraphinianus"]:
        sub_dir = base_out / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        (base_out / sub / "pages").mkdir(parents=True, exist_ok=True)
        (base_out / sub / "words" / "png").mkdir(parents=True, exist_ok=True)
        (base_out / sub / "glyphs" / "png").mkdir(parents=True, exist_ok=True)
        (base_out / sub / "glyphs" / "svg").mkdir(parents=True, exist_ok=True)

    do_voynich = "voynich" in args.manuscripts.lower()
    do_serafini = "seraphinianus" in args.manuscripts.lower()

    voynich_pages_data = []
    all_voynich_glyphs = []
    voynich_catalogue = {"canonical_alphabet_size": 0, "alphabet": []}

    # -------------------------------------------------------------
    # 2. Parallel Processing: Voynich Manuscript
    # -------------------------------------------------------------
    if do_voynich:
        print("\n[*] Initializing Voynich Manuscript extraction queue...")
        rosetta_loader = VoynicheseRosettaLoader()
        voynichese_scans_dir = Path("data/scans/voynich/voynichese_scans")
        voynichese_scans_dir.mkdir(parents=True, exist_ok=True)

        all_avail = rosetta_loader.list_available_folios()
        if all_avail:
            folios = all_avail
        else:
            folios = ["f001r", "f001v", "f002r", "f002v", "f003r", "f003v", "f004r", "f004v"]
        if args.voynich_limit:
            folios = folios[:args.voynich_limit]

        # Prepare task payloads
        tasks = []
        for f_id in folios:
            norm_id = rosetta_loader._normalize_folio_id(f_id)
            scan_path = voynichese_scans_dir / f"{norm_id}.jpg"
            if not scan_path.exists():
                import urllib.request
                g_url = f"https://www.voynichese.com/2/data/folio/image/glance/color/large/{norm_id}.jpg"
                req = urllib.request.Request(g_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(scan_path, "wb") as f_out:
                    f_out.write(resp.read())

            # Pre-load ground truth words
            r_words_ser = None
            try:
                # Use raw glance image for rosetta coordinate lookup
                glance_im = Image.open(scan_path)
                up_im = glance_im.resize((glance_im.width * 2, glance_im.height * 2), Image.Resampling.LANCZOS)
                words = rosetta_loader.extract_aligned_words(f_id, scan_image=up_im)
                if words:
                    r_words_ser = [{
                        "folio": w.folio,
                        "index": w.index,
                        "eva_text": w.eva_text,
                        "x_xml": w.x_xml,
                        "y_xml": w.y_xml,
                        "width_xml": w.width_xml,
                        "height_xml": w.height_xml,
                        "bbox_scan": list(w.bbox_scan) if w.bbox_scan else None
                    } for w in words]
            except Exception as e:
                print(f"  [i] Rosetta pre-fetch notice for {f_id}: {e}")

            tasks.append((f_id, str(scan_path), str(base_out), r_words_ser))

        print(f"[*] Dispatching {len(tasks)} Voynich folios across {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_folio = {
                executor.submit(process_single_voynich_folio, t[0], t[1], t[2], t[3]): t[0]
                for t in tasks
            }
            for future in as_completed(future_to_folio):
                f_id = future_to_folio[future]
                try:
                    res = future.result()
                    voynich_pages_data.append(res)
                    all_voynich_glyphs.extend(res.get("glyphs", []))
                    print(f"  [+] Completed {f_id}: {res['line_count']} lines, {res['word_count']} words, {res['glyph_count']} glyphs.")
                except Exception as e:
                    print(f"  [-] Error on {f_id}: {e}")

        # Sort pages in canonical sequence
        voynich_pages_data.sort(key=lambda p: p["page_id"])
        print(f"[*] Inducing Voynich Alphabet Catalogue across {len(all_voynich_glyphs)} extracted glyphs...")
        builder_v = GlyphCatalogue(target_alphabet_size=28)
        voynich_catalogue = builder_v.build_catalogue(all_voynich_glyphs, corpus_type="voynich")
        builder_v.export_catalogue_json(voynich_catalogue, base_out / "voynich" / "voynich_alphabet_catalogue.json")
        print(f"  [+] Voynich Alphabet: {voynich_catalogue['canonical_alphabet_size']} canonical archetypes.")

    # -------------------------------------------------------------
    # 3. Parallel Processing: Codex Seraphinianus
    # -------------------------------------------------------------
    serafini_pages_data = []
    all_serafini_glyphs = []
    serafini_catalogue = {"canonical_alphabet_size": 0, "alphabet": []}

    if do_serafini:
        print("\n[*] Initializing Codex Seraphinianus extraction queue...")
        pdf_path = Path("data/scans/seraphinianus/Codex Seraphinianus.pdf")
        if pdf_path.exists():
            loader_s = PDFScanLoader(pdf_path)
            total_pages = len(loader_s)
            if args.serafini_limit:
                pages = list(range(min(total_pages, args.serafini_limit)))
            else:
                pages = list(range(total_pages))

            print(f"[*] Dispatching {len(pages)} Seraphinianus pages across {args.workers} workers...")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_page = {
                    executor.submit(process_single_serafini_page, p, str(pdf_path), str(base_out)): p
                    for p in pages
                }
                for future in as_completed(future_to_page):
                    p_idx = future_to_page[future]
                    try:
                        res_s = future.result()
                        serafini_pages_data.append(res_s)
                        all_serafini_glyphs.extend(res_s.get("glyphs", []))
                        print(f"  [+] Completed page {p_idx}: {res_s['line_count']} lines, {res_s['word_count']} words, {res_s['glyph_count']} glyphs.")
                    except Exception as e:
                        print(f"  [-] Error on page {p_idx}: {e}")

            serafini_pages_data.sort(key=lambda p: p["page_id"])
            print(f"[*] Inducing Seraphinianus Alphabet Catalogue across {len(all_serafini_glyphs)} extracted glyphs...")
            builder_s = GlyphCatalogue(target_alphabet_size=52)
            serafini_catalogue = builder_s.build_catalogue(all_serafini_glyphs, corpus_type="seraphinianus")
            builder_s.export_catalogue_json(serafini_catalogue, base_out / "seraphinianus" / "seraphinianus_alphabet_catalogue.json")
            print(f"  [+] Seraphinianus Alphabet: {serafini_catalogue['canonical_alphabet_size']} canonical archetypes.")

    # -------------------------------------------------------------
    # 4. Generate Interactive Grand Explorer Application
    # -------------------------------------------------------------
    from scripts.generate_full_glyph_corpus_and_explorer import generate_explorer_html
    explorer_path = base_out / "grand_glyph_explorer.html"
    generate_explorer_html(voynich_pages_data, voynich_catalogue, serafini_pages_data, serafini_catalogue, explorer_path)

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
