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

5. **Mandatory Live Synchronization (Zero Hardcoded Stats)**:
   - Every regeneration or reprocessing of manuscript pages MUST dynamically rebuild **Unified Canonical Alphabets** and **`live_stats` (`BenchmarkSuite` + real page/word/glyph counts)** via `scripts/build_dynamic_grand_explorer.py` (which calls `rebuild_dual_catalogue_from_pages` + `compute_live_corpus_statistics`).
   - Never display hardcoded canonical counts, diagnostic metrics (`H1`, `H2`, `Hurst H`, `LZMA`), or extrapolated totals in `grand_glyph_explorer.html`. Every number shown in the Explorer or Help Guide must carry its exact corpus scope (e.g., *processed HD folios* vs *indexed dataset stubs* vs *seeded synthetic controls*), and archetype sample galleries (`max_samples=36`) must explicitly specify that instances are sorted by **ascending Euclidean distance to cluster medoid** (interleaved across processed folios).

6. **Convergence Over Proliferation (Human-Grade Closed-Loop Goal)**:
   - The strategic objective is **NOT** to accumulate redundant segmentation tabs or ductus engines indefinitely, nor to mimic manual human transliterations (like *Voynichese*, where a human intuited an alphabet and transcribed folios by hand).
   - Multi-model exploration (`A / B / C / D`) exists strictly as a controlled experimental workbench to **identify and converge on ONE clearly superior pipeline** that automates what a skilled human paleographer does naturally across an entire unknown manuscript:
     1. Starting from clean word envelopes, jointly leverage **visual glyph shape priors** (archetype dictionary matching, cf. *The Learnable Typewriter* / `ysig/learnable-typewriter`) and **kinematic stroke continuity** (ductus tracing / Bézier nib reconstruction) so that **stroke tracing informs ligature cutting and ligature cutting refines canonical archetypes**.
     2. Consolidate the emergent alphabet globally across the full codex so rare allographs, cursive ligatures, and under-/over-segmented compounds are resolved statistically rather than by human intuition.

7. **100%-Coverage Emergent Retro-Transcription**:
   - Because every ink component inside every word is assigned to an emergent canonical archetype (`G01..Gnn`, zero unindexed ink), the system supports **automatic retro-transcription** of words and folios into their emergent token sequence (`G07·G01·G12`) and canonical medoid glyphs — providing a direct generative reconstruction check against the raw scan (and, on Voynich words, side-by-side comparison with `EVA / Currier / v101` word transcriptions without displaying redundant `Currier`/`v101` cards on individual glyph/archetype cards).

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
  - `MetricScaleCalibration` (`voynich_ductus.ingestion.metric_scale`): Converts between physical metric units (millimeters, tenths of a millimeter) and digital scan pixels across codices (Voynich Beinecke MS 408: 225 mm, Seraphinianus: 350 mm, Gallica: 280 mm). Guarantees scale-invariant thresholds for nib widths, dot impact footprints (diameter <= 0.55 mm / 5.5 dmm), min glyph extents (0.85 mm / 8.5 dmm), scale bands (`TINY <= 1.3 mm`, `CORE`, `TALL >= 2.6 mm`, `WIDE >= 2.4 mm`), and inter-word whitespace (1.1 mm).
  - `IIIFClient`: Handles Beinecke MS 408 native full-resolution IIIF master scans (`~2400–2500 x 3168–3390 px`, on-demand), standardizes folio IDs (e.g. `f001r`, `f067v2`).
  - `Binarizer`: Implements Sauvola ($k=0.2, R=128$, `4.1 mm` metric window) always on the **Natural** unenhanced grayscale scan, paired with **Local Ring-Contrast Cavity Recovery** (re-opening tight enclosed bowls such as `d`/`8` eyes despite optical ink bleed) and **Bounded 8-Connected Hysteresis** (`0.45 mm` tail halo) in `binarize_word_crop_hysteresis` — explicitly protecting interior closed holes (`core_holes`) so faint *déliés* tails are recovered outward without ever plugging `'o'/'a'/'d'` bowls (canvas `InkBoost` is display-only).
  - `LineSegmenter`: Projects profiles and connected components for line/word bounding boxes.
  - `SpatialWordPartitioner`: Guarantees strict disjoint spatial partitioning across all words on a page (zero shared parchment, Voronoi/watershed contested boundary arbitration).
  - `GlyphSegmenter`: Builds word masks & **MST-bridged word polygons** (`compute_ink_polygon`, preserving detached overhead loops like `sh`), extracts interior **loop hole polygons** (`compute_hole_polygons` down to `0.020 mm²`, punched out via `evenodd` clipping on isolated glyph crops), and assembles word components into glyphs with physical nib impact thresholding and supralinear diacritic decoupling.

### 3.2 `vectorizer`
- **Goal**: Convert 2D pixel clusters into ordered $(x, y, t, \text{width})$ stroke trajectories with strict adherence to paleographic constraints, outputting both the primary **Quill Calligraphic Ductus (40°)** and the comparative **Euler-Bernoulli Topological Skeleton** side-by-side.
- **Key Divergence (Voynich vs. Seraphinianus)**:
  - **Codex Seraphinianus (Modern Cursive Ribbon)**: Strictly 1 continuous stroke per connected component (`single_stroke_mode=True`, `max_strokes=1`) in **both** Ribbon and Euler-Bernoulli modes via a **Topological Loop + Cusp Eulerian Tour** (preserving explicit `360°` medial rings on small loops without `binary_closing` loop-plugging and traversing `'3'`-shaped middle cusps).
  - **Voynich Manuscript (Medieval Quill Mechanics)**: Multi-stroke calligraphic ductus with deliberate pen lifts (`single_stroke_mode=False`). Enforces a strict physical ceiling of **1 to 4 strokes maximum** per glyph (`max_strokes=4`) in **both** Quill (40°) and Euler-Bernoulli modes, governed by a 40° beveled quill nib producing authentic *pleins et déliés*. Compound gallows (`cth`, `ckh`) utilize up to 4 strokes; minims, crescents, and bowls use 1 to 2 strokes.
  - **Cusp-Preserving Calligraphic Bézier Fitting & Geodesic Tip Guarantees**:
    - **Crossing-Preserving Centerlines, 11-O'Clock Loop Rings & `'3'`-Cusp Splicing (`CalligraphicVectorizer`)**: Extracts skeleton paths while preserving crossing junctions, splices bounce-back middle cusp tongues (`_splice_cusp_tongues` for `'3'`, `'y'`, `'s'`, `'m'`), completes free endpoints via 2-pass geodesic forward-cone tip search (`_extend_and_anchor_stroke`), freezes sharp corners (`cos < 0.57`) against Gaussian melting, and exports corner-aware cubic Bézier paths (`VectorExporter`).
    - **Protrusion-Aware Spur Pruning (`prune_skeleton_spurs`)**: Strips only vellum roughness micro-spurs contained inside `1.25 * R_junc` while preserving all protruding cusps, hooks, and serifs.
  - **Physical Nib Impact Threshold**: Objects $\le 0.55\text{ mm}$ or $\text{area} \le 0.35\text{ mm}^2$ bypass trajectory tracking and are derendered as instantaneous touchdown impact points (`is_impact=True`, $\text{length}=0.0\text{ mm}$).
- **Key Classes**:
  - `CalligraphicVectorizer`: Direct 40° beveled quill Bézier silhouette fitter and single-stroke continuous cursive tracker.
  - `Skeletonizer`: Cython `skeletonize` + Euclidean distance transform for stroke width profiling (*plein/délié*), with `medial_axis` backwards compatibility.
  - `StrokeGraphExtractor`: Vectorized `cv2.filter2D` 8-connectivity degree maps to `NetworkX` graph isolating path segments between critical nodes (degree $\neq 2$).
  - `JunctionResolver`: Resolves crossing ambivalences via minimum bending energy (Euler-Bernoulli) and applies 15th-century right-handed scribal priors with the 1–4 stroke ceiling; generates the comparative **Euler-Bernoulli Skeleton Ductus** displayed alongside every glyph.
  - `VectorExporter`: Exports standardized SVG (with kinematic color mapping and custom `data-order` attributes) and structured JSON.

### 3.3 `embeddings` & `word_decomposition`
- **Goal**: Continuous numerical representation of strokes/glyphs and **closed-loop unsupervised word-to-glyph segmentation** (`GenerativeTypewriterDecomposer`), operating inside the shared bounded-hysteresis word mask + MST envelope (with interior loop holes protected and punched out via `evenodd` clipping) and coupled with Dual-Engine ductus vectorization.
- **Converged Unified Word Decomposition Engine (`GenerativeTypewriterDecomposer`)**:
  - Inspired by *The Learnable Typewriter* (`ysig/learnable-typewriter`), replaces earlier exploratory valley/PCA/consensus tabs (`A / B / C`, retained only in `word_decomposition.py` for unit-test backwards compatibility) with a single superior pipeline (`51.7%` exact match & `MAE = 0.581` on 575 HD Voynich words, 100% blind to EVA, vs `A: 41.6%`, `B: 35.3%`, `C: 45.6%`):
    1. **Disjoint Candidate Cut Lattice (`_build_word_lattice`)**: Combines topological components, supralinear diacritic decoupling ($\le 0.55\text{ mm}$), EDT ink valleys, and 1D latent velocity peaks $\|dZ/dx\|$ with strict Euler loop protection.
    2. **Stratified `24×24` Visual Sprite Dictionary (`fit_on_word_masks`)**: Induces unsupervised character templates per loop stratum (`0L`, `1L`, `2+L`) across the codex via Ward clustering.
    3. **Viterbi Minimum Description Length (MDL) Decoding (`decompose_word`)**: Globally optimal dynamic programming segmentation balancing sprite reconstruction MSE, metric width priors, and open-prefix coalescence (`ch`, `cth`).
- **Key Classes**:
  - `GenerativeTypewriterDecomposer` (in `voynich_ductus.embeddings.word_decomposition`): Closed-loop sprite dictionary + Viterbi MDL word segmenter.
  - `GeometricFeatureExtractor`: 16-D invariant descriptors (bounded closed-loop tortuosity, aspect ratio, entry/exit tangents, full-360° 5-bin length-weighted orientation histograms).
  - `WordScribalEncoder`, `NeuralWordDecomposition` & `HybridConsensusWordDecomposition`: Underlying 1D latent transition features and legacy unit-test baselines.
  - `StrokeLatentProjector`: Normalization and dimensionality reduction (PCA / VAE).
  - `DINOv2EmbeddingEngine`: Meta DINOv2 self-supervised 768-D visual features (with persistent SHA-1 patch cache `dinov2_patch_cache.pkl`) projected via 16-D PCA and combined with physical logarithmic scale invariants ($\log h, \log w, \log \text{area}$).

### 3.4 `clustering`
- **Goal**: Unsupervised Dual-Alphabet induction (**Quill/Ribbon** & **Euler-Bernoulli** independently via `build_dual_catalogues`) and tokenization driven jointly by kinematic ductus descriptors and 16-D PCA DINOv2 visual embeddings.
- **Strict Invariance Rules**:
  - **Topological Loop Disjointness**: `0L` (loopless), `1L` (single loop), and `2+L` (multiloop) glyphs never collide into the same canonical archetype.
  - **Physical Scale Stratification**: Glyphs are strictly partitioned by physical scale bands (`TINY`, `CORE`, `TALL`, `WIDE`) so specks/diacritics cannot mix with letters, nor minims with tall gallows.
  - **Post-Scaled Joint Kinematic & DINOv2 Feature Space (`GlyphCatalogue.extract_glyph_features`)**: Combines kinematic ductus descriptors, metric scale (mm), and $L_2$-normalized **16-D PCA projection of 768-D DINOv2 embeddings**, applying subspace weights **after** `StandardScaler` so coarse 8x8 ink grid pixels cannot overwhelm geometry.
  - **Live Counts & Multi-Folio Sampling (`max_samples=36`)**: Archetype counts reflect real observed occurrences across the active processed pages (zero extrapolated corpus totals). Up to **36 conformant instances sorted by ascending Euclidean distance to cluster medoid** (interleaved round-robin across processed folios) are embedded per archetype.
  - **Transliteration Cross-Referencing (`CorpusCorrespondenceMatcher`)**: On glyph/archetype views, displays `EVA` (morphological match) and `Voynichese 2014 XML` (empirical token-vote purity); `Currier` and `v101` are displayed exclusively at the **word transcription level** (`EVA · Currier · v101`) alongside the emergent retro-transcription sequence.
- **Key Classes**:
  - `GlyphClusterer` & `GlyphCatalogue`: Tripartite hybrid induction (Strict Topological Partitioning + DINOv2 16-D PCA & Kinematic Ductus Clustering + Consensus Medoid Verification + Unified Codex Catalogues).
  - `StrokeTokenizer`: Generates unbiased token sequences (e.g. `G01-G04-G12`).

### 3.5 `diagnostics`
- **Goal**: Mathematical discrimination between language, cipher, and mechanical gibberish.
- **Key Classes**:
  - `ShannonEntropyAnalyzer`: Computes $H_1$ and conditional 2nd-order entropy $H_2$.
  - `LongRangeMemoryAnalyzer`: Detrended Fluctuation Analysis (DFA) on word recurrence intervals to measure the Hurst parameter $H$.
  - `CompressibilityAnalyzer`: Lossless compression ratios (gzip, bz2, lzma) approximating Kolmogorov complexity.
  - `MarkovAutomatonAnalyzer`: Evaluates transition determinism under order-1/2/3 Markov assumptions.
  - `BenchmarkSuite`: Automated evaluation engine producing comparative Markdown tables and JSON summaries (invoked dynamically during explorer generation for `live_stats`).

### 3.6 `generators`
- **Goal**: Synthetic adversarial baselines.
- **Key Classes**:
  - `TimmSelfCitationGenerator`: Torsten Timm's self-citation algorithm (sliding memory buffer + stochastic mutations).
  - `RuggCardanGenerator`: Gordon Rugg's Cardan grille across combinatorial syllable tables.
  - `BaselineGenerator`: Authentic 15th-century Latin herbal texts (Pseudo-Apuleius / Circa Instans), uniform noise, and Markov babblers.

### 3.7 `gallica, medieval calibration & open-source SOTA references`
- **Goal**: Curate 3 authentic 14th/15th-century French & Latin herbal codices digitized by Bibliothèque nationale de France (BnF) to benchmark neural scribal recognizers (`ScribalKinematicFlowNet`, `ScribalContrastiveEmbeddingNet`) against geometric ductus engines, calibrate medieval quill priors, and evaluate open-source paleography/derendering architectures:
  - **Joint Unsupervised Segmentation, Archetype Discovery & Retro-Transcription**: *The Learnable Typewriter* (`ysig/learnable-typewriter`, ICDAR Best Paper, LIGM École des Ponts — evaluated on the Copiale cipher & medieval charters) & *DTI-Sprites* (`monniert/dti-clustering`).
  - **Medieval HTR & Baseline Segmentation**: `Kraken` (`mittagessen/kraken`, LGPL) & `PyLaia` (`jpuigcerver/PyLaia`, MIT).
  - **Offline-to-Online Derendering & Differentiable Stroke Fitting**: Google Research `InkSight` (`google-research/inksight`, `Derendering/InkSight-Small-p`) & Weighted Bézier Stroke Reconstruction (`hzwer/Stroke-basedCharacterReconstruction`).
- **Key Classes**:
  - `GallicaManuscriptClient`: Standardizes Gallica ARK identifiers (`latin_6823` = `ark:/12148/btv1b6000517p`, `latin_6862` = `ark:/12148/btv1b84262821`, `francais_12322` = `ark:/12148/btv1b90610505`) and fetches IIIF plates.
  - `MedievalScribalTrainer`: Offline trainer (`scripts/train_on_gallica_pdfs.py`) for competitive neural models; runtime extraction applies deterministic geometric priors directly.
  - `ComparativeVectorizerBenchmark`: Side-by-side evaluation suite benchmarking digital paleography paradigms (Physical Nib Model, Euler-Bernoulli Skeleton, Scribal Flow Net, DINOv2 ViT, Google InkSight).

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
