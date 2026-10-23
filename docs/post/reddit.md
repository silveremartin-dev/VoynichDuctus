# Reddit launch post — VoynichDuctus

## Suggested subreddits

- **r/MachineLearning**: requires the `[P]` tag in the title. The angle is unsupervised alphabet induction (DINOv2 + geometric features + density clustering) and offline-to-online handwriting derendering.
- **r/voynich**: the core community. Expect expert pushback on EVA, segmentation and the statistics, which is exactly the feedback I need.
- **r/DigitalHumanities**: computational paleography and reproducible tooling for manuscript studies.
- **r/computervision**: binarization, skeletonization/stroke graphs, SIFT/RANSAC registration, stroke-order recovery.
- **r/Paleography** (or r/linguistics, if the post is reframed around the information-theoretic diagnostics): scribal ductus, allographs, medieval hands.

---

**Title:** [P] VoynichDuctus: an open-source pipeline that analyzes the Voynich Manuscript from raw ink strokes instead of the EVA transliteration

---

Hi all,

I've been working on an open-source Python pipeline called **VoynichDuctus** and I'd like some critical eyes on it.

### Why

Most computational work on the Voynich Manuscript (Beinecke MS 408, radiocarbon-dated 1404–1438) runs statistics on the **EVA transliteration**, an ASCII scheme from the 1990s. EVA has to make arbitrary calls, for example whether `ch`, `sh`, `cth` or `ckh` are one glyph or several, and whether two visually different forms count as the same letter. Statistics computed on EVA therefore partly measure those transcription decisions.

My question: what happens if you skip the transliteration and build tokens directly from the **physical strokes** on the page?

### What it does

The pipeline has 6 decoupled stages. Each one can be called from Python or from a CLI:

1. **Ingestion**: Yale IIIF scans, local adaptive binarization (Sauvola, Wolf-Jolion, Niblack), line and word segmentation. SIFT/RANSAC affine registration maps existing word-level coordinates (Voynichese project, Apache-2.0) onto the Yale HD scans for all 225 folios.
2. **Vectorization (offline-to-online derendering)**: I compare 5 approaches:
   - a physical broad-nib model (40° nib angle, stroke width `w(θ) = (W − w0)·|sin(θ − 40°)| + w0`, run on distance-transform ridges);
   - a deterministic medial-axis + NetworkX stroke graph, with junctions resolved by minimum bending energy (`Σ(1 − cos Δθ)`) and right-handed scribal priors (top-left touchdown, CCW bowls);
   - a small U-Net "scribal flow" model trained on synthetic plates;
   - DINOv2 ViT embeddings;
   - an adapter for Google InkSight, which falls back to the geometric pipeline when PyTorch/GPU isn't available.
3. **Embeddings**: 16-D geometric descriptors (tortuosity, aspect ratio, entry/exit tangents, orientation histograms) concatenated with DINOv2 features. I added DINOv2 because the hand-crafted features alone collapsed visually different glyphs into the same cluster.
4. **Clustering**: density clustering into emergent archetypes (`G01`…), with an outlier quarantine bin for parchment flaws, illustration bleed and hapax forms. Glyphs are aligned with EVA labels so allographs can be tracked (for example two physically distinct forms of EVA `a`).
5. **Diagnostics**: Shannon H1, conditional H2, Hurst exponent via DFA, gzip/bz2/LZMA compression ratios, and order-1/2/3 Markov determinism.
6. **Generators / baselines**: a Torsten Timm-style self-citation generator, a Gordon Rugg-style Cardan grille, an authentic 15th-c. Latin herbal text, and uniform noise. All are deterministic given a seed.

I also run the same pipeline on the **Codex Seraphinianus** (a known asemic script) as a reference. No Seraphinianus page images are distributed, only derived vector/coordinate JSON.

### Scale

From the batch run (Cloud Run Jobs):
- Voynich: 225/225 folios, **200,272 glyphs**, 35 archetypes
- Seraphinianus: 371/371 pages, **5,258 glyphs**, 55 archetypes
- An interactive standalone HTML explorer (deep zoom, word/glyph overlays, 5-way vectorizer comparison per glyph, archetype browser)
- 52 pytest tests

### Results so far (1,387 tokens per corpus)

| Corpus | H1 | H2 (cond.) | Hurst (DFA) | LZMA/gzip | Markov top-1 (order 2) |
|---|---|---|---|---|---|
| Voynich, EVA | 3.80 | 2.31 | 0.591 | 0.354 | 50.46% |
| Voynich, ductus tokens | 2.89 | 1.25 | 0.570 | 0.158 | 70.49% |
| Seraphinianus, ductus tokens | 3.14 | 2.15 | 0.915 | 0.311 | 48.98% |
| Latin herbal (control) | 4.04 | 3.27 | 0.850 | 0.070 | 47.02% |
| Timm self-citation | 3.18 | 2.43 | 0.571 | 0.079 | 62.64% |
| Rugg Cardan grille | 3.74 | 2.40 | 0.741 | 0.195 | 45.61% |
| Uniform noise | 4.70 | 4.63 | 0.874 | 0.627 | 20.78% |

What I'm comfortable saying:
- Neither Voynich token stream looks like uncorrelated noise. The noise baseline barely drops from H1 to H2 (4.70 → 4.63).
- Changing the tokenization from EVA to ductus tokens changes the statistics a lot. In particular, order-2 Markov determinism goes from ~50% to ~70%. In other words, conclusions drawn from EVA depend on the tokenization.

### What this is NOT / known limitations

- **It is not a decipherment or a translation**, and I make no claim about meaning.
- **The rule-based "diagnosis" label is not reliable yet.** In the same benchmark it labels the *authentic Latin control* as "Timm self-citation", the Timm generator as "Cardan grid", and the Rugg generator as "Timm". So I don't treat its label for the Voynich ductus tokens ("Cardan grid") as evidence of anything.
- 1,387 tokens per corpus is small, and Hurst estimates on short series are noisy. One example: the uniform noise baseline gets H ≈ 0.87, where you would expect about 0.5. That points to a problem with my DFA setup or the series construction, and I still need to debug it.
- Some numbers in older internal reports come from earlier runs and don't match the table above. The README table is the current one.
- The ductus tokens are lower-entropy partly *because* my segmentation merges cursive sequences (e.g. `aiin`) and because of the number of clusters. The tokenization itself is a modeling choice, just a different one from EVA.
- Stroke order is inferred using priors (a right-handed 15th-c. scribe), not observed.

### Feedback I'm looking for

- Paleographers: are the ductus priors (nib angle, loop direction, minim splitting) reasonable for this hand?
- Stats/ML people: better ways to compare token streams of different alphabet sizes (e.g. normalized entropies, controlled-vocabulary resampling), and a sound DFA setup for short symbolic series.
- Voynich researchers: glyph segmentation errors you spot in the explorer, especially around gallows and benches.
- Anyone: tell me which baselines are missing.

Repo (MIT): https://github.com/silveremartin-dev/VoynichDuctus

Thanks for reading. I'm happy to be told where I'm wrong.

— Silvère Martin-Michiellot
