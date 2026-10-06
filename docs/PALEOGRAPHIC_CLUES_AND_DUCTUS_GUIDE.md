# Paleographic Clues, Scribal Priors & Ductus Reconstruction Guide
*Systematic Reference for Autonomous Stroke Derendering in VoynichDuctus*

---

## 1. Executive Summary & Core Objective

The central mission of **VoynichDuctus** is the **objective, unsupervised derendering of scribal handwriting** (*offline-to-online trajectory recovery*) on historical manuscripts (such as Beinecke MS 408 and the Codex Seraphinianus) without relying on subjective human transliteration alphabets (EVA, Currier, v101).

To achieve robust extraction and reconstruct natural calligraphic ductus without false positives, the system integrates a comprehensive set of **physical, chromatic, topological, and kinematic clues** established through rigorous paleographical principles.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      HIERARCHICAL MULTI-SCALE EXTRACTION                         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 1. Page-Level Illumination & Chromatic Subtraction                         │
   │    - Low-chroma text ink vs. vivid watercolor pigments (green/blue/red)    │
   │    - Projection profile variance for automatic orientation (portrait/land) │
   └────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 2. Hierarchical 3-Level Text Decomposition                                 │
   │    - Level 1: Continuous text lines (top-to-bottom page traversal)         │
   │    - Level 2: Discrete lexical words (inter-word spacing gaps >= 10px)     │
   │    - Level 3: Intact assembled glyphs (sub-stroke clustering & split)      │
   └────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 3. Physical Dimensionality & Morphological Guards                          │
   │    - Quill nib thickness & minimum physical size (>= 2 mm / >= 10 px)      │
   │    - Fill factor boundaries: 1D filiform (0.05 <= F <= 0.40) vs 2D blobs   │
   └────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 4. Kinematic Ductus Reconstruction & Calligraphic Smoothing                │
   │    - Medial axis skeletonization + Euclidean distance transform            │
   │    - Micro-spur pruning (< 5 px skeleton branch artifacts)                 │
   │    - Right-handed scribal priors: Downstrokes (top->down), bars (left->rt) │
   │    - Euler-Bernoulli minimum bending energy across junctions               │
   │    - Catmull-Rom to Cubic Bézier spline paths (no pixel staircase jitter)  │
   └────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 5. Topological Invariant Partitioning & Alphabet Induction                 │
   │    - Invariant buckets: Closed LOOP ('o'), Crescent ('c'), MINIM ('i'),    │
   │      Gallows ('k','t'), Multi-loop ('8')                                   │
   │    - 32-D spatial density grid + Agglomerative Ward clustering             │
   └────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Comprehensive Inventory of Paleographic Clues & Heuristics

### A. Chromatic & Pigment Separation Clues
1. **Monochrome Ink vs. Polychrome Illustrations**:
   - **Text Ink Signature**: Scribal ink (iron-gall, carbon black, sepia) is strictly achromatic with very low chroma ($\Delta(RGB) \le 0.16$) and low saturation ($S \le 0.20$), characterized by high local contrast against parchment.
   - **Illustration Signature**: Botanical foliage (green chlorophyll, $H \in [0.14, 0.50]$), cosmological water/stars (blue azurite/lapis, $H \in [0.50, 0.78]$), red dyes/rubrics ($H \ge 0.85 \lor H \le 0.06$), and ochre roots ($H \in [0.08, 0.16]$) exhibit high chroma ($> 0.24$) and high saturation ($S > 0.20$).
   - **Action**: All chromatic illustration pixels are subtracted with morphological dilation before text segmentation, isolating pure writing from surrounding drawings.

2. **Parchment vs. Ink Brightness Floor**:
   - Genuine text crops must possess a light parchment/paper background ($\mu_{\text{patch}} \ge 115 / 255$). Crops with near-zero background luminance represent margin tear shadows, holes, or scanning artifacts and are systematically discarded.

---

### B. Hierarchical 3-Level Text Structure
Rather than segmenting raw connected components across the whole page (which risks capturing leaf veins or drawing details), the pipeline enforces a **3-stage top-down reduction**:
1. **Lines**: Continuous horizontal baseline ridges detected across the entire page (from top header to bottom footer).
2. **Words**: Horizontal connected-component bounding boxes separated by natural scribal word gaps ($\text{gap} \ge 10\text{ px}$).
3. **Glyphs**: Discrete characters extracted strictly within word boundaries, merging vertically stacked strokes (ascender + bowl, gallows bar + leg) and splitting wide cursive ligatures ($W \ge 1.7 H$) at high-prominence column valleys.

---

### C. Physical Scale, Quill Nib Geometry & Filiform Guards
1. **Minimum Physical Resolution ($ \ge 2\text{ mm}$)**:
   - A quill nib or reed pen has an inherent physical stroke width of $0.4 - 1.2\text{ mm}$. A legible scribal character or diacritic accent cannot physically exist below $\approx 2\text{ mm}$ ($\ge 10\text{ px}$ at 2400-3000px page scan scale). Micro-specks ($< 10\text{ px}$ height, $< 6\text{ px}$ width, or area $< 15\text{ px}^2$) are filtered as parchment grain or ink splatter.
2. **1D Filiform vs. 2D Dark Blobs (Fill Factor)**:
   - True handwriting consists of one-dimensional curvilinear ink trajectories written on a blank surface. The bounding box fill factor $\Phi = \frac{\text{Ink Area}}{W \times H}$ must strictly fall within the filiform band:
     $$0.05 \le \Phi \le 0.40$$
   - Any region with $\Phi > 0.40$ is a solid dark blob, shadow patch, or miniature drawing, NOT a legible paleographic glyph.

---

### D. Automatic Page & Line Orientation Detection
- **Sideways / Landscape Plates**: Certain manuscript folios (e.g. Codex Seraphinianus page 40) are scanned in portrait orientation but formatted in landscape (vertical text columns).
- **Projection Variance Ratio**:
  $$\mathcal{R}_{\text{orient}} = \frac{\text{Var}(P_{\text{col}})}{\text{Var}(P_{\text{row}})}$$
  - For normal horizontal text: $\mathcal{R}_{\text{orient}} < 1.0$ (strong horizontal line peaks).
  - For vertical/sideways text: $\mathcal{R}_{\text{orient}} \ge 1.8$ (strong vertical column peaks).
  - **Action**: When $\mathcal{R}_{\text{orient}} \ge 1.8$, the page is automatically rotated by $270^\circ$ before segmentation so reading proceeds horizontally left-to-right.

---

### E. Kinematic Ductus & Calligraphic Smoothing
1. **Right-Handed Scribal Mechanics**:
   - **Downstroke Prior**: Ascenders, descenders, and vertical minims are pulled downwards (high pen pressure $P(t)$, movement $\Delta y > 0$).
   - **Crossbar / Ligature Prior**: Gallows horizontal crossbars and cursive baseline ligatures are pushed left-to-right ($\Delta x > 0$).
   - **Touchdown Point**: Every independent pen stroke begins with a touch-down point (marked with a green dot `●`).
2. **Pruning Skeleton Spurs & Branch Artifacts**:
   - Medial axis transform generates tiny spurious orthogonal branches at thick stroke junctions. Any branch $< 5\text{ px}$ is pruned before graph decomposition, preventing artificial "Parkinsonian jitter".
3. **Collinear Stroke Chaining**:
   - Sub-strokes meeting at simple junctions or corners are chained along their tangent continuity vectors (Euler-Bernoulli minimum bending energy), yielding natural stroke counts (typically 1 to 3 continuous strokes per character).
4. **Cubic Bézier / Spline Curves**:
   - Raw skeleton points step orthogonally or diagonally pixel-by-pixel. The vectorizer converts these points into smooth Catmull-Rom cubic Bézier curves (`C cp1x cp1y cp2x cp2y x y` in SVG), eliminating discrete pixel staircase effects and rendering silky-smooth calligraphy.

---

### F. Topological Invariants & Unsupervised Alphabet Induction
To prevent clustering open crescents (`C`) with closed loops (`O`):
1. **Euler Number & Closed Topology ($\chi$)**:
   - $\chi = 0$: Closed loop topology (`O`, `8`, `g`, `p` bowl).
   - $\chi = 1$: Open tree / curve topology (`C`, `I`, `S`, `L`).
2. **Topological Buckets**:
   - Every glyph is strictly partitioned into:
     - `LOOP`: Closed loop structures ($\ge 75\%$ closed).
     - `OPEN_C`: Open rightward/leftward crescents.
     - `MINIM`: Vertical single downstrokes ($H \gg W$).
     - `GALLOWS`: Tall ascenders with horizontal traverse bars.
     - `OTHER`: Complex composite ligatures.
3. **Spatial Density Embedding**:
   - 32-D spatial grid features normalized with `StandardScaler` are clustered via Agglomerative Ward clustering within each topological bucket, guaranteeing that open and closed glyph forms never collide into the same canonical archetype.

---

### G. Alphabet Parsimony Priors & Historical Census Calibration
1. **Parsimony of Human Alphabetic Systems**:
   - Natural human alphabets and scribal scripts (e.g. Latin, Gothic cursive, Italian humanistic, Cyrillic, Greek, Serafinian) consistently operate with **a compact set of a few dozen canonical archetypes** (typically 20 to 35 core letters), each exhibiting continuous allographic variations (different ink flow, pen angle, ligature attachments).
   - The induction algorithm uses a target alphabet capacity $K \approx 25-30$ to ensure we induce an authentic canonical inventory rather than treating every subtle stroke variation as a distinct letter.
2. **Page-by-Page Calibration Against Historical Censuses**:
   - The algorithm's line/word/glyph counts are systematically calibrated against established historical transcriptions (e.g., Takahashi, Landini-Stolfi, Currier for Voynich; Serafini typology for Codex Seraphinianus).
   - If manual census yields 30 lines and ~200 words on folio `f001v`, our automated line projection prominence and word gap thresholds are calibrated so that detected counts match historical ground truth ($\pm 10\%$).

---

### I. Native Calligraphic Ridge Tracker & Pen-Nib Physics
1. **Beveled Quill Nib Mechanics**:
   - A physical medieval quill cut with a fixed bevel angle $\theta_{\text{nib}} \approx 35^\circ - 45^\circ$ naturally produces *pleins et déliés* governed by:
     $$w(\theta) = W_{\text{nib}} \cdot |\sin(\theta - \theta_{\text{nib}})| + w_0$$
   - Strokes perpendicular to the bevel produce maximum thickness $W_{\text{nib}}$, while strokes parallel to the bevel yield hairline strokes $w_0$.
2. **Euclidean Distance Ridge Propagation**:
   - Instead of generic graph skeletonization that branches arbitrarily at intersections, `CalligraphicVectorizer` propagates continuous centerline trajectories along the ridges of the Euclidean distance map $D(y, x)$, prioritizing forward momentum and preserving nested loops as single continuous strokes.

---

### J. Scribal Kinematic Flow Net & Contrastive Triplet Embedding Net
1. **Directional Flow U-Net (`ScribalKinematicFlowNet`)**:
   - A lightweight neural architecture predicting the continuous unit vector field $\vec{u}(x, y) = (\cos \theta, \sin \theta)$, along with spatial probability maps for pen touchdowns $P(t=0)$ and pen-lifts $P(t=1)$.
2. **Triplet Contrastive Representation (`ScribalContrastiveEmbeddingNet`)**:
   - Deep 128-D embedding network trained with Triplet Margin Loss:
     $$\mathcal{L}(a, p, n) = \max(0, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha)$$
   - Enforces invariance to parchment discoloration, ink fading, and slant variations.

---

### K. Multimodal Paleographic Supervision (Gemini 2.5 Flash)
1. **Automated Expert Paleography Validation (`GeminiPaleographyAdapter`)**:
   - Visual inspection of ambiguous ligatures and cursive joints directly through multimodal vision queries.
   - Comparative scribal hand authentication (analyzing slant consistency and nib angle).
   - Zero-shot cursive word decomposition based on Minimum Description Length (MDL) principles.

---

### L. Voynichese Rosetta Stone Ground Truth Integration (`VoynicheseRosettaLoader`)
1. **Full-Corpus Ground Truth Integration**:
   - Ingestion of 225 curated XML annotation folios covering the complete Beinecke MS 408 corpus ($>35,000$ segmented word bounding boxes with verified EVA transliterations).
2. **High-Resolution Affine Coordinate Projection**:
   - Maps normalized low-resolution annotation bounding boxes $(x_{\text{xml}}, y_{\text{xml}}, w_{\text{xml}}, h_{\text{xml}})$ onto native high-definition Yale Beinecke scans ($2500 \times 3168\text{ px}$) using affine registration:
     $$X_{\text{scan}} = t_x + x_{\text{xml}} \cdot s_x, \quad Y_{\text{scan}} = t_y + y_{\text{xml}} \cdot s_y$$
3. **Atomic EVA Glyph Tokenization**:
   - Greedy longest-match tokenization decomposing complex compound words and gallows ligatures (`cth`, `ckh`, `cfh`, `cph`, `ch`, `sh`, `iii`, `ee`, `ii`, `aiin`) into atomic ductus training targets.
4. **Supervised Paleographic Benchmarking**:
   - Enables direct quantitative validation of unsupervised alphabet induction clusters against standardized paleographical ground truth without inducing transliteration hallucinations.

---

## 3. Summary of Visual Encoding & Ductus Kinematics Legend

| Visual Element | Representation | Kinematic Meaning |
| :--- | :--- | :--- |
| **Green Dot (`●`)** | `#10b981` | Initial pen touchdown point (nib contact with parchment) |
| **Cyan Stroke (`—`)** | `#38bdf8` | Stroke 1: Primary ductus trajectory |
| **Emerald Stroke (`—`)** | `#10b981` | Stroke 2: Secondary stroke after pen-lift |
| **Amber Stroke (`—`)** | `#f59e0b` | Stroke 3: Tertiary crossbar or diacritic |
| **Violet Stroke (`—`)** | `#c084fc` | Stroke 4+: Extended cursive ligature flourish |
| **Violet Dotted Box** | `rgba(168, 85, 247, 0.65)` | Detected text line baseline span |
| **Green Box** | `rgba(16, 185, 129, 0.45)` | Lexical parent word bounding box |
| **Cyan Solid Box** | `rgba(56, 189, 248, 0.50)` | Isolated, verified single glyph bounding box |


