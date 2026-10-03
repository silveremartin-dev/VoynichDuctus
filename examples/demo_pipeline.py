"""
Demo script: End-to-end extraction from a synthetic historical ink patch.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer
from voynich_ductus.utils.visualizer import Visualizer


def create_synthetic_voynich_word() -> Image.Image:
    """Generates a synthetic word image resembling Voynich glyph ligatures."""
    img = Image.new("RGB", (250, 80), color="#eedfbe")  # Parchment color
    draw = ImageDraw.Draw(img)

    # Draw a sequence of glyph-like curves in iron-gall ink color (#281a10)
    # Glyph 1: Loop + stem (like 'q' / 'o')
    draw.ellipse([20, 25, 45, 55], outline="#281a10", width=3)
    draw.line([45, 20, 45, 65], fill="#281a10", width=3)

    # Glyph 2: Bench / gallows ('k' / 't')
    draw.line([60, 15, 60, 60], fill="#281a10", width=3)
    draw.arc([60, 20, 85, 45], start=180, end=0, fill="#281a10", width=3)
    draw.line([85, 32, 85, 60], fill="#281a10", width=3)
    draw.line([50, 30, 95, 30], fill="#281a10", width=2)  # Crossbar

    # Glyph 3: Small loop + tail ('e' / 'y')
    draw.ellipse([110, 30, 130, 55], outline="#281a10", width=3)
    draw.arc([125, 40, 145, 65], start=270, end=90, fill="#281a10", width=3)

    # Glyph 4: Final tail ('dy')
    draw.line([160, 25, 160, 60], fill="#281a10", width=3)
    draw.arc([160, 20, 185, 55], start=0, end=180, fill="#281a10", width=3)
    draw.line([185, 35, 210, 20], fill="#281a10", width=3)

    return img


def run_demo():
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Creating synthetic manuscript patch...")
    img = create_synthetic_voynich_word()
    sample_path = output_dir / "sample_voynich_word.png"
    img.save(sample_path)
    print(f"[+] Sample image saved to: {sample_path}")

    print("[*] 1. Binarizing parchment and ink...")
    binarizer = Binarizer(method="sauvola", window_size=15)
    binary = binarizer.binarize(img)

    print("[*] 2. Extracting medial axis skeleton & stroke width profile...")
    skel_engine = Skeletonizer(method="medial_axis")
    skel, stroke_widths = skel_engine.extract_skeleton(binary)

    print("[*] 3. Extracting topological stroke graph & resolving ductus...")
    graph_extractor = StrokeGraphExtractor()
    pixel_graph = graph_extractor.build_pixel_graph(skel, stroke_widths)
    raw_strokes = graph_extractor.decompose_into_strokes(pixel_graph)

    resolver = JunctionResolver()
    ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)
    print(f"[+] Reconstructed {len(ordered_strokes)} ordered scribal strokes.")

    print("[*] 4. Exporting vector SVG with chronological ordering...")
    svg_path = output_dir / "reconstructed_ductus.svg"
    h, w = binary.shape
    VectorExporter.to_svg(ordered_strokes, width=w, height=h, output_path=svg_path)
    print(f"[+] Vector SVG saved to: {svg_path}")

    print("[*] 5. Extracting geometric features & clustering primitives...")
    feature_extractor = GeometricFeatureExtractor()
    features = feature_extractor.extract_batch(ordered_strokes)
    projector = StrokeLatentProjector(latent_dim=min(6, len(features)))
    latent = projector.fit_transform(features)

    clusterer = GlyphClusterer(method="agglomerative", n_clusters=min(4, len(features)))
    labels = clusterer.fit_predict(latent)
    tokenizer = StrokeTokenizer(prefix="G")
    word_token = tokenizer.tokenize_word(labels.tolist())

    print(f"[+] Emergent Objective Token: {word_token}")
    print("[+] Pipeline demo completed successfully!")


if __name__ == "__main__":
    run_demo()
