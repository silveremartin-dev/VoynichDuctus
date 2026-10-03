"""
Extracts vector ductus SVGs from Codex Seraphinianus cursive samples and runs comparative analysis.
"""

from pathlib import Path
from PIL import Image
from voynich_ductus.generators.seraphinianus import SeraphinianusEngine
from voynich_ductus.ingestion.binarization import Binarizer
from voynich_ductus.vectorizer.skeleton import Skeletonizer
from voynich_ductus.vectorizer.stroke_graph import StrokeGraphExtractor
from voynich_ductus.vectorizer.junction_resolver import JunctionResolver
from voynich_ductus.vectorizer.export_format import VectorExporter
from voynich_ductus.diagnostics.benchmark_suite import BenchmarkSuite
from voynich_ductus.utils.io import BenchmarkFormatter


def run_seraphinianus_pipeline():
    out_dir = Path("./output/seraphinianus")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(exist_ok=True)
    (out_dir / "svg").mkdir(exist_ok=True)

    print("[*] Generating Codex Seraphinianus cursive word samples...")
    samples = [
        ("serafini_word_01_capital_spiral", 4, True),
        ("serafini_word_02_flowing_cursive", 5, False),
        ("serafini_word_03_ascenders_heavy", 4, False),
        ("serafini_word_04_descenders_loops", 6, True),
    ]

    for name, n_prims, has_cap in samples:
        img = SeraphinianusEngine.generate_serafini_word_image(num_primitives=n_prims, has_capital=has_cap)
        png_path = out_dir / "png" / f"{name}.png"
        svg_path = out_dir / "svg" / f"{name}.svg"
        img.save(png_path)

        # Vectorize
        binarizer = Binarizer(method="sauvola", window_size=15)
        binary = binarizer.binarize(img)
        binary = binarizer.remove_small_artifacts(binary, min_size=4)

        skel_engine = Skeletonizer(method="medial_axis")
        skel, widths = skel_engine.extract_skeleton(binary)

        graph_extractor = StrokeGraphExtractor()
        pixel_graph = graph_extractor.build_pixel_graph(skel, widths)
        raw_strokes = graph_extractor.decompose_into_strokes(pixel_graph)

        resolver = JunctionResolver(right_handed_prior=True)
        ordered_strokes = resolver.resolve_and_order_strokes(raw_strokes)

        h, w = binary.shape
        VectorExporter.to_svg(ordered_strokes, width=w, height=h, output_path=svg_path)
        print(f"  [+] Vectorized {name}: {len(ordered_strokes)} strokes -> {svg_path.name}")

    # Run comparative benchmark including Codex Seraphinianus
    print("\n[*] Running diagnostic benchmark comparing Voynich, Latin, and Codex Seraphinianus...")
    from voynich_ductus.generators.timm_self_citation import TimmSelfCitationGenerator
    from voynich_ductus.generators.baselines import BaselineGenerator

    words = 1200
    corpora = {
        "Latin Herbal (15th c. Natural)": BaselineGenerator.get_natural_latin_sample(words),
        "Voynich-like (Timm Self-Citation)": TimmSelfCitationGenerator(seed=42).generate(words),
        "Codex Seraphinianus (Asemic Model)": SeraphinianusEngine.generate_serafini_text_tokens(words),
        "Uniform Random Noise": BaselineGenerator.get_uniform_random_gibberish(words),
    }

    suite = BenchmarkSuite()
    results = suite.compare_corpora(corpora)

    print("\n" + BenchmarkFormatter.to_markdown_table(results) + "\n")


if __name__ == "__main__":
    run_seraphinianus_pipeline()
