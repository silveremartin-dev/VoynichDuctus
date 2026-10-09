# AGENTS.md — Developer & AI Agent Collaboration Guide

## 1. System Overview & Purpose

`VoynichDuctus` is an open-source scientific pipeline for digital paleography, stroke vectorization, and computational linguistics. Its primary mission is to **decouple the analysis of the Voynich Manuscript (Beinecke MS 408) from subjective human transliteration schemes (like the EVA alphabet)**.

By converting raw high-resolution raster scans into topological stroke graphs and inferring scribal kinematics (*offline-to-online handwriting derendering*), the system discovers an **emergent, objective alphabet** via unsupervised clustering and tests it against formal information-theoretic diagnostic criteria.

---

## 2. Core Architectural Principles for AI Agents

When interacting with or extending this codebase, all AI agents must adhere to the following principles:

1. **Zero Translation Hallucination**:
   - Never attempt or claim to "translate" the Voynich manuscript into modern language without rigorous, verifiable ground truth.
   - Frame all deliverables around measurable geometric properties (ductus, stroke curvature, pen lift, thickness) and formal information-theoretic signatures (Kolmogorov complexity, Hurst exponent, conditional entropy).

2. **Decoupled Modularity**:
   - Maintain strict separation of concerns across the 6 pipeline stages: `ingestion` ➔ `vectorizer` ➔ `embeddings` ➔ `clustering` ➔ `diagnostics` ➔ `generators`.
   - Ensure every module is callable both via programmatic Python API and the CLI (`voynich_ductus.cli`).

3. **Deterministic & Reproducible Benchmarks**:
   - Any new generative model (e.g., variant of Timm's algorithm or combinatoric wheels) must be added under `voynich_ductus.generators` and exposed in `voynich_ductus.diagnostics.benchmark_suite`.
   - All tests must be deterministic when random seeds are provided.

4. **Robust Fallbacks**:
   - Provide clean heuristic / geometric fallbacks for deep learning models (e.g. `InkSightAdapter` must function smoothly with `medial_axis` + `NetworkX` even when GPU/PyTorch dependencies are unavailable).

---

## 3. Subsystems & Pipeline Specification

```
┌────────────────────────────────────────────────────────────────────────┐
│                        voynich_ductus Pipeline                         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 1. Ingestion     │ ──▶│ 2. Vectorization   │ ──▶│ 3. Embeddings       │
│ - IIIF Client    │    │ - Medial Axis Skel │    │ - 16-D Geom Features│
│ - Sauvola / Wolf │    │ - Stroke Graph (NX)│    │ - Latent Projector  │
│ - Line / Word CC │    │ - Junction Resolver│    │   (PCA / Autoenc)   │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
                                                             │
                                                             ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 6. Ground Truth  │ ◀──│ 5. Diagnostics     │ ◀──│ 4. Clustering       │
│ - Timm Self-Cite │    │ - Shannon H1 / H2  │    │ - HDBSCAN / DBSCAN  │
│ - Rugg Cardan    │    │ - Hurst / DFA      │    │ - Tokenizer         │
│ - Latin Herbal   │    │ - Lempel-Ziv LZMA  │    │   (e.g. G01-G12)    │
│ - Random Noise   │    │ - Markov FSA Order │    │                     │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
```

### 3.1 `ingestion`
- **Goal**: Fetch scans and segment parchment into ink masks.
- **Key Classes**:
  - `IIIFClient`: Handles Beinecke MS 408 URL schemes, standardizes folio IDs (e.g. `f001r`, `f067v2`).
  - `Binarizer`: Implements Sauvola ($k=0.2, R=128$), Niblack, and Wolf-Jolion local adaptive thresholding.
  - `LineSegmenter`: Projects profiles and connected components for line/word bounding boxes.

### 3.2 `vectorizer`
- **Goal**: Convert 2D pixel clusters into ordered $(x, y, t, \text{width})$ stroke trajectories.
- **Key Classes**:
  - `Skeletonizer`: 1D medial axis transform + Euclidean distance transform for stroke width profiling (*plein/délié*).
  - `StrokeGraphExtractor`: Converts 8-connectivity skeleton pixels to a `NetworkX` graph and isolates path segments between critical nodes (degree $\neq 2$).
  - `JunctionResolver`: Resolves crossing ambivalences via minimum bending energy (Euler-Bernoulli) and applies 15th-century right-handed scribal priors.
  - `VectorExporter`: Exports standardized SVG (with kinematic color mapping and custom `data-order` attributes) and structured JSON.

### 3.3 `embeddings`
- **Goal**: Continuous numerical representation of strokes.
- **Key Classes**:
  - `GeometricFeatureExtractor`: 16-D invariant descriptors (tortuosity, aspect ratio, entry/exit tangents, orientation histograms).
  - `StrokeLatentProjector`: Normalization and dimensionality reduction (PCA / VAE).

### 3.4 `clustering`
- **Goal**: Unsupervised alphabet induction and tokenization.
- **Key Classes**:
  - `GlyphClusterer`: Density-based clustering (DBSCAN / HDBSCAN / Agglomerative) discovering canonical glyph primitives without human labels.
  - `StrokeTokenizer`: Generates unbiased token sequences (e.g. `G01-G04-G12`).

### 3.5 `diagnostics`
- **Goal**: Mathematical discrimination between language, cipher, and mechanical gibberish.
- **Key Classes**:
  - `ShannonEntropyAnalyzer`: Computes $H_1$ and conditional 2nd-order entropy $H_2$.
  - `LongRangeMemoryAnalyzer`: Detrended Fluctuation Analysis (DFA) on word recurrence intervals to measure the Hurst parameter $H$.
  - `CompressibilityAnalyzer`: Lossless compression ratios (gzip, bz2, lzma) approximating Kolmogorov complexity.
  - `MarkovAutomatonAnalyzer`: Evaluates transition determinism under order-1/2/3 Markov assumptions.
  - `BenchmarkSuite`: Automated evaluation engine producing comparative Markdown tables and JSON summaries.

### 3.6 `generators`
- **Goal**: Synthetic adversarial baselines.
- **Key Classes**:
  - `TimmSelfCitationGenerator`: Torsten Timm's self-citation algorithm (sliding memory buffer + stochastic mutations).
  - `RuggCardanGenerator`: Gordon Rugg's Cardan grille across combinatorial syllable tables.
  - `BaselineGenerator`: Authentic 15th-century Latin herbal texts (Pseudo-Apuleius / Circa Instans), uniform noise, and Markov babblers.

### 3.7 `gallica & medieval calibration`
- **Goal**: Ground stroke extraction and scribal kinematics against authentic 15th-century Latin herbal codices digitized by Bibliothèque nationale de France (BnF).
- **Key Classes**:
  - `GallicaManuscriptClient`: Standardizes Gallica ARK identifiers (`ark:/12148/btv1b52501620s`, `latin_6823`, `latin_17868`) and fetches IIIF plates with transcription alignments.
  - `MedievalScribalTrainer`: Supervised & self-supervised trainer calibrating quill physical parameters (40° nib angle, pleins & déliés contrast ratio, distance ridge continuity) on medieval scribal hands.
  - `ComparativeVectorizerBenchmark`: Side-by-side evaluation suite benchmarking 5 digital paleography paradigms (Physical Nib Model, Euler-Bernoulli Skeleton, Scribal Flow Net, DINOv2 ViT, Google InkSight).

---

## 4. Guidelines for Future Agent Tasks

### When adding a new stroke extraction model:
1. Wrap the inference logic inside `voynich_ductus.vectorizer.inksight_adapter` or create a sibling adapter class.
2. Ensure it outputs standard stroke dictionaries containing `stroke_id`, `points: [(y, x, width)]`, and `order_index`.
3. Add a corresponding unit test in `tests/test_vectorizer.py` and `tests/test_comparative_vectorizers.py`.

### When adding a new diagnostic metric:
1. Place the estimator class in `voynich_ductus.diagnostics`.
2. Connect the metric into `BenchmarkSuite.evaluate_corpus()`.
3. Update `voynich_ductus.utils.io.BenchmarkFormatter.to_markdown_table()` so the CLI output includes the new column.

### Running Tests
Always verify code changes by running:
```bash
pytest tests/ -v
```

---

## 5. Coding Standards
- Python 3.9+ compatible.
- Type annotations on all public methods (`typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Optional`, `typing.Union`).
- Comprehensive docstrings explaining theoretical rationale and mathematical equations.
- Non-interactive matplotlib backend (`matplotlib.use("Agg")`) for headless execution.
