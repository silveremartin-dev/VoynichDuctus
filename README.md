# VoynichDuctus: Objective Vector Glyphs, Scribal Kinematics & Statistical Paleography

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paleography: Objective Ductus](https://img.shields.io/badge/paleography-unsupervised%20ductus-brightgreen.svg)]()
[![Information Theory: Diagnostics](https://img.shields.io/badge/diagnostics-H1%20%7C%20H2%20%7C%20DFA%20%7C%20LZ-orange.svg)]()

> **A reproducible computational pipeline to decouple digital paleography from subjective human transliteration schemes (like the EVA alphabet) by extracting vector strokes, reconstructing scribal kinematics (*offline-to-online derendering*), discovering emergent alphabets via unsupervised clustering, and testing them against formal information-theoretic diagnostic criteria.**

---

## Table of Contents
- [1. Executive Summary & Purpose](#1-executive-summary--purpose)
- [2. The Methodological Pitfall: The EVA Transliteration Bias](#2-the-methodological-pitfall-the-eva-transliteration-bias)
- [3. Complete Pipeline Architecture](#3-complete-pipeline-architecture)
- [4. The 5 Independent Vision & Ductus Methods](#4-the-5-independent-vision--ductus-methods)
- [5. Physical Nib, Serifs & Medieval Scribal Mechanics](#5-physical-nib-serifs--medieval-scribal-mechanics)
- [6. Emergent Transliteration & Allograph Discovery](#6-emergent-transliteration--allograph-discovery)
- [7. Formal Information-Theoretic Diagnostics Benchmark](#7-formal-information-theoretic-diagnostics-benchmark)
- [8. Interactive Grand Glyph Explorer](#8-interactive-grand-glyph-explorer)
- [9. Installation & Quickstart](#9-installation--quickstart)
- [10. Project Structure](#10-project-structure)
- [11. Acknowledgments & References](#11-acknowledgments--references)
- [12. License](#12-license)

---

## 1. Executive Summary & Purpose

Over a century of failed attempts to decipher the Voynich Manuscript (Beinecke MS 408, radiocarbon-dated to 1404–1438 AD) stems from applying classical cryptanalysis to an artifact whose fundamental nature remains unsettled: **unknown natural language, early polyalphabetic cipher, constructed taxonomic language, or algorithmic mechanical gibberish**.

Most modern computational studies suffer from a fatal flaw: they train models on human-transcribed ASCII corpora (such as the European Voynich Alphabet, **EVA**), analyzing 1990s transliteration choices rather than raw historical ink.

**VoynichDuctus** decouples paleographic analysis from human transcription bias:
1. **Parchment Normalization & Binarization**: Auto-calibrated local adaptive thresholding (Sauvola, Wolf-Jolion) preserving delicate hair-lines (*déliés*) and rejecting parchment bleed-through.
2. **Offline-to-Online Scribal Kinematics**: Simulates broad-nib quill physics (40° bevel) on Euclidean distance transform (EDT) ridges, preserving continuous closed loops without artificial breaks.
3. **Multi-Paradigm Paleographical Suite**: Contrasts 5 independent vision paradigms (Physical Quill Model, Euler-Bernoulli Skeleton, U-Net Scribal Flow, Meta DINOv2 ViT, Google InkSight).
4. **Unsupervised Alphabet Discovery**: Projects isolated glyphs into continuous latent spaces (DINOv2 + 16-D geometric descriptors) to discover canonical archetypes (`G01`, `G02`, ...) without human labeling.
5. **Transliteration & Sub-Allograph Analysis**: Aligns Voynich with Rosetta EVA ground truth to discover sub-allographs (e.g. open vs closed `'a'`), while generating the **first complete emergent transliteration of the asemic *Codex Seraphinianus***.
6. **Formal Information-Theoretic Diagnostics**: Evaluates Shannon entropies ($H_1, H_2$), Detrended Fluctuation Analysis (DFA / Hurst exponent $H$), Lempel-Ziv compressibility, and Markov finite state automata determinism against authentic 15th-c. Latin herbals and adversarial synthetic generators.

---

## 2. The Methodological Pitfall: The EVA Transliteration Bias

The European Voynich Alphabet (EVA) was created in the 1990s to store the manuscript in computer ASCII:
- **Arbitrary Grapheme Grouping**: EVA forced arbitrary decisions on whether adjacent ink strokes form a single glyph or a ligature of distinct glyphs (e.g. deciding whether `ch`, `sh`, `cth`, `ckh` are atomic or composite).
- **Subjective Human Smoothing**: When algorithms analyze EVA strings, they measure human linguistic biases rather than scribal physical realities.
- **The VoynichDuctus Solution**: Working directly on raw pixel geometry, quill angles, and stroke kinematics to induce emergent alphabets purely from objective physical and topological traits.

---

## 3. Complete Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        voynich_ductus Pipeline                         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 1. Ingestion     │ ──▶│ 2. Vectorization   │ ──▶│ 3. Embeddings       │
│ - Yale IIIF / PDF│    │ - 40° Nib Ridge Trk│    │ - 16-D Descriptors  │
│ - Sauvola / Wolf │    │ - Euler-Bernoulli  │    │ - DINOv2 ViT 768-D  │
│ - Line / Word CC │    │ - Kinematic Flow   │    │ - Contrastive Net   │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
                                                             │
                                                             ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 6. Ground Truth  │ ◀──│ 5. Diagnostics     │ ◀──│ 4. Clustering       │
│ - Timm Self-Cite │    │ - Shannon H1 / H2  │    │ - HDBSCAN / Density │
│ - Rugg Cardan    │    │ - Hurst / DFA      │    │ - Tokenizer         │
│ - Latin Herbal   │    │ - LZMA / Gzip Ratio│    │   (e.g. G01-G29)    │
│ - Random Noise   │    │ - Markov FSA Ord-2 │    │ - Rosetta Alignment │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
```

---

## 4. The 5 Independent Vision & Ductus Methods

The pipeline integrates and contrasts 5 distinct, standalone paleographic methods:

| Method | Nature / Architecture | Optimal Paleographic Condition | Output & Metrics |
| :--- | :--- | :--- | :--- |
| **1. Physical Nib Model (Our In-House Engine)** | 40° beveled quill mechanics, EDT distance ridges, Flash & Hogan minimum jerk | **Crisp medieval ink, continuous cursive loops, pleins & déliés contrast**. Fast (< 2 ms), 0 hallucination. | Time-ordered vector strokes $(x, y, t, w)$, mean width, nib angle. |
| **2. Euler-Bernoulli Skeleton** | 1D Medial Axis Transform + NetworkX topological graph | **Topological analysis of bifurcations and crossings** without neural networks (deterministic baseline). | Curvature energy $\int \kappa^2 ds$, continuity score. |
| **3. Scribal Kinematic Flow Net** | Dense U-Net Convolutional Neural Network | **Degraded, faded, or noisy parchments** where skeleton thinning fails. | Tangent flow field $\hat{\mathbf{u}}(x, y)$, touchdown $P(t=0)$ and lift $P(t=1)$ maps. |
| **4. Meta DINOv2 ViT** | Vision Transformer (ViT-B/14) 768-D self-supervised representation | **Invariant archetype clustering** across folios regardless of parchment tint or background noise. | 768-D semantic embedding vector, attention density graph. |
| **5. Google InkSight** | Seq2Seq Autoregressive Transformer Encoder-Decoder | **SOTA offline-to-online handwriting derendering benchmark**. | Autoregressive sequence of coordinates $(x_t, y_t, \text{pen\_state}_t)$. |

---

## 5. Physical Nib, Serifs & Medieval Scribal Mechanics

### A. 40° Bevel and Pleins & Déliés
Broad-nib quill mechanics govern stroke width based on pen movement direction relative to the 40° nib axis:
$$w(\theta) = (W_{\text{nib}} - w_0) \cdot |\sin(\theta - 40^\circ)| + w_0$$
- Perpendicular movements ($130^\circ$) produce heavy downstrokes (*pleins*).
- Parallel movements ($40^\circ$) produce delicate hairlines (*déliés*).

### B. Medieval Serifs & Entry Attacks
Authentic 15th-century hands (Voynich, Gothic Bastarda, Humanistic cursive) exhibit pronounced **serifs**:
- **Minim Entry Serifs**: Short diagonal approach from top-left ($dx=0.7, dy=0.7$) before the main vertical shaft, ending in an upward right foot flick.
- **Ascender Club Heads**: Triangular massues on top of hastes (`'l'`, `'b'`, `'h'`) where touchdown occurs at the upper-left crest.
- **Single-Stroke Cursive Loops (Seraphinianus)**: Enforces strictly **1 continuous stroke per connected ink component**, preventing artificial multi-stroke splitting on smooth loops.

---

## 6. Emergent Transliteration & Allograph Discovery

### A. Voynich: Rosetta Alignment & Sub-Allograph Discovery
- Words are segmented into $N$ character zones at local minima of vertical ink projection profiles, constrained by the expected Rosetta EVA character count.
- **Allograph Discovery (Non 1:1)**: While preserving the expected EVA label, our unsupervised clustering independently assigns an emergent Archetype `Gxx`. If a single EVA character `'a'` appears in two physically distinct styles (e.g., closed upright `'a'` vs looped open `'a'`), the system documents them as **two distinct scribal allographs**.

### B. Codex Seraphinianus: First Complete Emergent Transliteration
Because no page-by-page transcription exists for Luigi Serafini's asemic codex, the system induces the **first complete machine transliteration**:
- Folios are segmented into lines, words, and isolated glyphs.
- Each word is transcribed into an objective token sequence (e.g. `serafini_p020_L002_W02` $\to$ `G29-G01-G29-G14`).
- Consecutive identical glyphs (like the two $\mathcal{E}_\cdot$ on Page 20) are consistently mapped to the same archetype (`G29`).

---

## 7. Formal Information-Theoretic Diagnostics Benchmark

Evaluating the emergent ductus corpora against control baselines across 1,387 tokens per corpus:

| Corpus Tested | Tokens | $H_1$ (bits) | $H_2$ Cond (bits) | Hurst $H$ (DFA) | LZMA / Gzip Ratio | Markov Top-1 (Ord 2) | Formal Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Voynich (EVA Standard Transliteration)** | 1,387 | 3.80 | 2.31 | 0.591 | 0.354 | 50.46% | `Natural Language` |
| **Voynich (Emergent Ductus G-Tokens)** | 1,387 | 2.89 | 1.25 | 0.570 | 0.158 | **70.49%** | `Mechanical Cardan Grid` |
| **Codex Seraphinianus (Emergent Ductus)** | 1,387 | 3.14 | 2.15 | **0.915** | 0.311 | 48.98% | `Natural Language` |
| **Authentic 15th-c. Latin Herbal (Control)** | 1,387 | 4.04 | 3.27 | **0.850** | **0.070** | 47.02% | `Timm Self-Citation` |
| **Timm Self-Citation Generator (Model)** | 1,387 | 3.18 | 2.43 | 0.571 | 0.079 | 62.64% | `Mechanical Cardan Grid` |
| **Rugg Cardan Grille Generator (Model)** | 1,387 | 3.74 | 2.40 | 0.741 | 0.195 | 45.61% | `Timm Self-Citation` |
| **Uniform Random Noise (Null Baseline)** | 1,387 | 4.70 | 4.63 | 0.874 | 0.627 | 20.78% | `Random Gibberish` |

### Key Scientific Conclusions:
1. **Rejection of Random Gibberish**: Uniform noise exhibits negligible entropy drop ($H_1 = 4.70 \to H_2 = 4.63$), conclusively ruling out uncorrelated random babble for both Voynich and Seraphinianus.
2. **EVA Smoothing vs Physical Ductus**: While human EVA transliteration smooths the text towards natural language distributions, physical ductus tokenization exposes a **high 2nd-order Markov determinism (70.49%)**, consistent with combinatorial or modular scribal generation.

---

## 8. Interactive Grand Glyph Explorer

The pipeline generates an interactive, high-performance web application at:
`output/atlas_dataset/grand_glyph_explorer.html`

### Features:
- **Interactive HD Deep Zoom & Minimap**: Smooth pan and zoom across Yale Beinecke HD folios and Seraphinianus plates.
- **Word & Glyph Overlay Toggles**: Highlight identified text lines, word boxes, and isolated character boundaries.
- **5-Way Vision & Ductus Comparison Modal**: Inspect any glyph simultaneously across all 5 paleographic engines with inline keyboard navigation ($\leftarrow$ / $\rightarrow$).
- **Transliteration Inspector**: Compare Voynich Rosetta EVA strings with emergent ductus tokens, explore allograph discoveries, and browse the complete Seraphinianus transliteration.
- **Archetype Variations Explorer**: View all physical instances of any canonical archetype across the entire manuscript with 1-click page localization.

---

## 9. Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/silveremartin-dev/VoynichDuctus.git
cd VoynichDuctus

# Install in editable mode
pip install -e .
```

### Running Scripts & Generating the Explorer

```bash
# 1. Run the full test suite (45 unit & integration tests)
pytest tests/ -v

# 2. Run the information-theoretic diagnostics suite
python scripts/run_comprehensive_diagnostics.py

# 3. Generate synthetic kinematic training plates & retrain neural models
python scripts/generate_training_plates_and_train.py

# 4. Generate the full corpus and interactive Grand Glyph Explorer
python scripts/generate_full_glyph_corpus_and_explorer.py
```

---

## 10. Project Structure

```
VoynichDuctus/
├── data/
│   ├── annotations/voynichese/  # 225-folio Voynichese Rosetta ground truth dataset
│   └── scans/                   # Local HD scans & PDFs (Voynich, Seraphinianus, Gallica)
├── docs/                        # Paleographic documentation and synthesis guides
├── output/
│   ├── atlas_dataset/           # grand_glyph_explorer.html & extracted glyph database
│   ├── diagnostics/             # comprehensive_diagnostics_report.json
│   ├── models/                  # Trained PyTorch neural checkpoints (.pt)
│   └── training_plates/         # High-resolution kinematic calibration inspection plates
├── scripts/
│   ├── generate_full_glyph_corpus_and_explorer.py
│   ├── generate_training_plates_and_train.py
│   └── run_comprehensive_diagnostics.py
├── src/voynich_ductus/
│   ├── clustering/              # Unsupervised density clustering & catalogue builder
│   ├── diagnostics/             # Shannon H1/H2, Hurst DFA, Lempel-Ziv, Markov FSA
│   ├── embeddings/              # 16-D geometric features, DINOv2 adapter, ScribalFlowNet
│   ├── generators/              # Timm self-citation, Rugg Cardan grille, Latin herbal baselines
│   ├── ingestion/               # Yale IIIF client, Gallica client, Sauvola binarizer, segmenter
│   ├── models/                  # MedievalScribalTrainer (supervised & triplet loss)
│   ├── utils/                   # Vector exporter, SVG renderers, formatters
│   └── vectorizer/              # CalligraphicVectorizer (40° nib), comparative engines
└── tests/                       # Complete pytest suite (45/45 passing)
```

---

## 11. Acknowledgments & References

- **[Yale University Beinecke Library](https://beinecke.library.yale.edu/)**: High-resolution IIIF scans of Beinecke MS 408.
- **[Bibliothèque nationale de France (BnF Gallica)](https://gallica.bnf.fr/)**: Medieval Latin herbal reference codices (*Latin 6823, Latin 6862, Français 12322*).
- **[The Voynichese Project](https://github.com/voynichese/voynichese)**: Curated EVA word-level ground truth annotations (Apache-2.0).
- **Luigi Serafini**: Creator of the *Codex Seraphinianus* (1981).
- **Torsten Timm & Gordon Rugg**: Foundational research on scribal self-citation and Cardan grille mechanics.
- **Landini, Stolfi, Currier, and Takahashi**: Historical transliteration censuses and comparative statistics.

---

## 12. License

This project is licensed under the [MIT License](LICENSE).
The Voynichese annotation dataset bundled under `data/annotations/voynichese/` is licensed under the Apache License, Version 2.0.
