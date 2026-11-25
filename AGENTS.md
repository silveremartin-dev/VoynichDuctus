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
- **Goal**: Fetch scans, calibrate physical resolution, and segment parchment into ink masks.
- **Key Classes**:
  - `MetricScaleCalibration`: Converts between physical metric units (millimeters, tenths of a millimeter) and digital scan pixels across codices (Voynich Beinecke MS 408: 225 mm, Seraphinianus: 350 mm, Gallica: 280 mm). Guarantees scale-invariant thresholds for nib widths, dot impact footprints (diameter <= 0.55 mm / 5.5 dmm), min glyph extents (0.85 mm / 8.5 dmm), scale bands (TINY <= 1.3 mm, CORE, TALL >= 3.4 mm, WIDE >= 3.4 mm), and inter-word whitespace (1.1 mm).
  - `IIIFClient`: Handles Beinecke MS 408 URL schemes, standardizes folio IDs (e.g. `f001r`, `f067v2`).
  - `Binarizer`: Implements Sauvola ($k=0.2, R=128$), Niblack, and Wolf-Jolion local adaptive thresholding.
  - `LineSegmenter`: Projects profiles and connected components for line/word bounding boxes.
  - `SpatialWordPartitioner`: Guarantees strict disjoint spatial partitioning across all words on a page (zero shared parchment, Voronoi/watershed contested boundary arbitration).
  - `GlyphSegmenter`: Assembles word connected components into glyphs with physical nib impact thresholding and supralinear diacritic decoupling.

### 3.2 `vectorizer`
- **Goal**: Convert 2D pixel clusters into ordered $(x, y, t, \text{width})$ stroke trajectories with strict adherence to paleographic constraints, outputting both the primary **Quill Calligraphic Ductus (40°)** and the comparative **Euler-Bernoulli Topological Skeleton** side-by-side.
- **Key Divergence (Voynich vs. Seraphinianus)**:
  - **Codex Seraphinianus (Modern Cursive Ribbon)**: Strictly 1 continuous stroke per connected component (`single_stroke_mode=True`, `max_strokes=1`). Follows fluid automobile-like inertia with no sharp angles > 90°, no retracing, and 100% skeleton coverage through an iterative 12-pass ridge extension loop oriented left-to-right.
  - **Voynich Manuscript (Medieval Quill Mechanics)**: Multi-stroke calligraphic ductus with deliberate pen lifts (`single_stroke_mode=False`). Enforces a strict physical ceiling of **1 to 4 strokes maximum** per glyph (`max_strokes=4`), governed by a 40° beveled quill nib producing authentic *pleins et déliés*. Compound gallows (`cth`, `ckh`) utilize up to 4 strokes; minims, crescents, and bowls use 1 to 2 strokes.
  - **Anti-Brownian Kinematic Guarantees**:
    - **Skeleton Spur Pruning (`prune_skeleton_spurs`)**: Strips microscopic lateral medial-axis spurs shorter than the local Euclidean distance transform (EDT) nib radius before ridge tracking.
    - **Full Nib-Footprint Extinction (`_mark_nib_footprint_visited` & `wake_mask`)**: Marks the full physical cross-section of the quill ($r = d_{\text{EDT}} + 1.6\text{ px}$) as visited along the stroke trajectory, preventing internal zigzag bouncing or hairpin U-turns inside thick downstrokes.
    - **Ramer-Douglas-Peucker & Gaussian Smoothing (`_rdp_simplify_with_width`)**: Simplifies trajectories to essential calligraphic control points while preserving closed loops (`1L`/`2+L`).
  - **Physical Nib Impact Threshold**: Objects $\le 8\text{ px}$ or $\text{area} \le 25\text{ px}^2$ bypass ridge tracking and are derendered as instantaneous touchdown impact points (`is_impact=True`, $\text{length}=0.0$).
- **Key Classes**:
  - `CalligraphicVectorizer`: Native quill physics engine (40° nib bevel) and single-stroke continuous cursive tracker.
  - `Skeletonizer`: 1D medial axis transform + Euclidean distance transform for stroke width profiling (*plein/délié*).
  - `StrokeGraphExtractor`: Converts 8-connectivity skeleton pixels to a `NetworkX` graph and isolates path segments between critical nodes (degree $\neq 2$).
  - `JunctionResolver`: Resolves crossing ambivalences via minimum bending energy (Euler-Bernoulli) and applies 15th-century right-handed scribal priors; generates the comparative **Euler-Bernoulli Skeleton Ductus** displayed alongside every glyph.
  - `VectorExporter`: Exports standardized SVG (with kinematic color mapping and custom `data-order` attributes) and structured JSON.

### 3.3 `embeddings` & `word_decomposition`
- **Goal**: Continuous numerical representation of strokes/glyphs and three independent word-to-glyph decomposition hypotheses (`word_decomposition.py`), each coupled with full dual-engine ductus vectorization.
- **Three Word Decomposition Modes**:
  - **Decomposition A (`GlyphSegmenter` — Topological & Valley 40°)**: Connected components + supralinear diacritic decoupling ($\le 0.55\text{ mm}$) + vertical column projection & EDT ink valleys that never slice through a closed loop.
  - **Decomposition B (`NeuralWordDecomposition` — Spatial PCA Decomposition)**: Encodes intact word crops into 1D horizontal latent trajectories $Z(x) \in \mathbb{R}^{32}$ via `WordScribalEncoder` and places letter boundaries at local maxima of latent transition velocity $\|dZ/dx\|$.
  - **Decomposition C (`HybridConsensusWordDecomposition` — Hybrid Consensus)**: Validates letter boundaries at the strict geometric intersection ($\epsilon \le 3\text{ px}$) of Decompositions A and B with Euler loop protection.
- **Key Classes**:
  - `GeometricFeatureExtractor`: 16-D invariant descriptors (tortuosity, aspect ratio, entry/exit tangents, orientation histograms).
  - `WordScribalEncoder`, `NeuralWordDecomposition` & `HybridConsensusWordDecomposition` (in `voynich_ductus.embeddings.word_decomposition`): Unsupervised 1D horizontal latent decomposition and consensus cutting.
  - `StrokeLatentProjector`: Normalization and dimensionality reduction (PCA / VAE).
  - `DINOv2EmbeddingEngine`: Meta DINOv2 self-supervised 768-D visual features projected via 16-D PCA and combined with physical logarithmic scale invariants ($\log h, \log w, \log \text{area}$).

### 3.4 `clustering`
- **Goal**: Unsupervised alphabet induction and tokenization driven jointly by kinematic ductus descriptors and 16-D PCA DINOv2 visual embeddings.
- **Strict Invariance Rules**:
  - **Topological Loop Disjointness**: `0L` (loopless), `1L` (single loop), and `2+L` (multiloop) glyphs never collide into the same canonical archetype.
  - **Physical Scale Stratification**: Glyphs are strictly partitioned by physical scale bands (`TINY`, `CORE`, `TALL`, `WIDE`) so specks/diacritics cannot mix with letters, nor minims with tall gallows.
  - **Joint Kinematic & DINOv2 Feature Space (`GlyphCatalogue.extract_glyph_features`)**: Combines high-weight kinematic ductus descriptors (`stroke_count * 3.5`, `tortuosity * 3.0`, entry/exit tangents `* 2.5`, 5-bin orientation histogram `* 2.2`, width variance `* 2.0`), physical scale, and $L_2$-normalized **16-D PCA projection of 768-D DINOv2 embeddings** (`weight * 2.5`) across 100% of glyphs.
  - **Graphical Archetype Symbols**: Induces **49 canonical archetypes (`G01`–`G49`)** for the Voynich Manuscript and **57 emergent graphical symbol archetypes (`G01`–`G57`)** for the Codex Seraphinianus (rendered directly as visual vector SVG symbols rather than arbitrary ASCII codes).
  - **Corpus Census Projections**: Archetype frequencies reflect the complete corpus census (162,333 glyphs across 225 Voynich folios, 456,943 glyphs across 371 Seraphinianus pages), yielding realistic occurrences in the tens of thousands.
- **Key Classes**:
  - `GlyphClusterer` & `GlyphCatalogue`: Tripartite hybrid induction (Strict Topological Partitioning + DINOv2 16-D PCA & Kinematic Ductus Clustering + Consensus Medoid Verification).
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
- **Strict English UI**: All user interfaces, dashboards, visualizers, and web explorers (including HTML/CSS/JS labels, buttons, comboboxes, tooltips, status banners, inspector details, and codicological notes) must be written exclusively in English.
