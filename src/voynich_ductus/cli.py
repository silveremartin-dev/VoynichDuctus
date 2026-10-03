"""
Command-Line Interface for the Voynich Ductus Pipeline.
"""

import sys
import json
from pathlib import Path
import click
import numpy as np
from PIL import Image

from voynich_ductus.ingestion.iiif_client import IIIFClient
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.ingestion.segmenter import LineSegmenter
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.embeddings.geometric_features import GeometricFeatureExtractor
from voynich_ductus.embeddings.stroke_autoencoder import StrokeLatentProjector
from voynich_ductus.clustering.clusterer import GlyphClusterer
from voynich_ductus.clustering.tokenizer import StrokeTokenizer
from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
from voynich_ductus.generators.rugg_cardan import RuggCardanGenerator
from voynich_ductus.generators.baselines import BaselineGenerator
from voynich_ductus.utils.io import BenchmarkFormatter


@click.group()
@click.version_option()
def main():
    """Voynich Ductus: Objective Vector Glyphs, Ductus Kinematics & Statistical Paleography."""
    pass


@main.command()
@click.option("--folio", "-f", default="f001r", help="Folio identifier (e.g., f001r, f067v2).")
@click.option("--output", "-o", default=None, help="Output destination path.")
def download(folio: str, output: str):
    """Downloads high-resolution folio scans from Beinecke IIIF archives."""
    client = IIIFClient()
    click.echo(f"[*] Downloading folio {folio}...")
    try:
        out_path = client.download_folio(folio, output_path=output)
        click.echo(f"[+] Folio saved to: {out_path}")
    except Exception as e:
        click.echo(f"[-] Download error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--words", "-n", default=1200, help="Number of tokens to generate per corpus.")
@click.option("--markdown/--no-markdown", default=True, help="Print results as a Markdown table.")
@click.option("--export-json", "-j", default=None, help="Path to save JSON benchmark output.")
def benchmark(words: int, markdown: bool, export_json: str):
    """Runs comparative statistical paleography diagnostics on synthetic controls & baselines."""
    click.echo(f"[*] Generating test corpora (N = {words} words per corpus)...")

    # 1. Torsten Timm's Self-Citation
    timm_gen = TimmSelfCitationGenerator(seed=42)
    timm_corpus = timm_gen.generate(num_words=words)

    # 2. Gordon Rugg's Cardan Grid
    rugg_gen = RuggCardanGenerator(seed=42)
    rugg_corpus = rugg_gen.generate(num_words=words)

    # 3. 15th-Century Medieval Latin Herbal
    latin_corpus = BaselineGenerator.get_natural_latin_sample(num_words=words)

    # 4. Uniform Random Noise (Memoryless)
    random_corpus = BaselineGenerator.get_uniform_random_gibberish(num_words=words)

    # 5. Order-1 Markov Babbler
    markov_corpus = BaselineGenerator.get_markov_babbler(latin_corpus, order=1, num_words=words)

    corpora = {
        "Latin Herbal (15th c. Ground Truth)": latin_corpus,
        "Timm Self-Citation (Voynich-like)": timm_corpus,
        "Rugg Cardan Grille (Mechanical)": rugg_corpus,
        "Markov Babbler (Order 1)": markov_corpus,
        "Uniform Random Gibberish (Noise)": random_corpus,
    }

    suite = BenchmarkSuite()
    click.echo("[*] Computing information-theoretic diagnostics (H1, H2, DFA Hurst, Lempel-Ziv, Markov FSA)...")
    results = suite.compare_corpora(corpora)

    if markdown:
        click.echo("\n" + BenchmarkFormatter.to_markdown_table(results) + "\n")

    if export_json:
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        click.echo(f"[+] Benchmark results exported to: {export_json}")


@main.command()
@click.option("--generator", "-g", type=click.Choice(["timm", "cardan", "latin", "random", "markov"]), default="timm")
@click.option("--words", "-n", default=200, help="Number of words to generate.")
def generate(generator: str, words: int):
    """Generates synthetic text from selected ground truth or adversarial model."""
    if generator == "timm":
        gen = TimmSelfCitationGenerator()
        tokens = gen.generate(words)
    elif generator == "cardan":
        gen = RuggCardanGenerator()
        tokens = gen.generate(words)
    elif generator == "latin":
        tokens = BaselineGenerator.get_natural_latin_sample(words)
    elif generator == "random":
        tokens = BaselineGenerator.get_uniform_random_gibberish(words)
    elif generator == "markov":
        src = BaselineGenerator.get_natural_latin_sample(500)
        tokens = BaselineGenerator.get_markov_babbler(src, order=1, num_words=words)

    click.echo(" ".join(tokens))


@main.command()
@click.option("--input-image", "-i", required=True, help="Input scan or word image file.")
@click.option("--output-svg", "-s", default="output/vectorized_strokes.svg", help="Output SVG path.")
@click.option("--output-tokens", "-t", default="output/tokens.txt", help="Output tokenized text file.")
def process(input_image: str, output_svg: str, output_tokens: str):
    """Executes full offline-to-online stroke vectorization, clustering, and tokenization pipeline."""
    img_path = Path(input_image)
    if not img_path.exists():
        click.echo(f"[-] Input image not found: {img_path}", err=True)
        sys.exit(1)

    click.echo(f"[*] Processing image: {img_path}")
    raw_img = Image.open(img_path)

    # 1. Binarize
    binarizer = Binarizer(method="sauvola")
    binary_ink = binarizer.binarize(raw_img)
    binary_ink = binarizer.remove_small_artifacts(binary_ink)

    # 2. Skeletonize
    skel_engine = Skeletonizer(method="medial_axis")
    skel, stroke_widths = skel_engine.extract_skeleton(binary_ink)

    # 3. Extract Stroke Graph & Resolve Junctions
    graph_extractor = StrokeGraphExtractor()
    pixel_graph = graph_extractor.build_pixel_graph(skel, stroke_widths)
    raw_strokes = graph_extractor.decompose_into_strokes(pixel_graph)

    resolver = JunctionResolver()
    ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)
    click.echo(f"[+] Extracted {len(ordered_strokes)} ordered kinetic strokes.")

    # 4. Export SVG
    h, w = binary_ink.shape
    VectorExporter.to_svg(ordered_strokes, width=w, height=h, output_path=output_svg)
    click.echo(f"[+] Vector strokes saved to: {output_svg}")

    # 5. Extract Features & Cluster Glyphs
    if len(ordered_strokes) >= 3:
        feature_extractor = GeometricFeatureExtractor()
        features = feature_extractor.extract_batch(ordered_strokes)
        projector = StrokeLatentProjector(latent_dim=min(8, len(features)))
        latent = projector.fit_transform(features)

        clusterer = GlyphClusterer(method="dbscan", min_samples=2)
        labels = clusterer.fit_predict(latent)
        tokenizer = StrokeTokenizer()
        word_token = tokenizer.tokenize_word(labels.tolist())
        click.echo(f"[+] Discovered {clusterer.get_cluster_count()} distinct stroke clusters.")
        click.echo(f"[+] Objective tokenization: {word_token}")

        out_tok = Path(output_tokens)
        out_tok.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tok, "w", encoding="utf-8") as f:
            f.write(word_token + "\n")
        click.echo(f"[+] Tokens saved to: {output_tokens}")


if __name__ == "__main__":
    main()
