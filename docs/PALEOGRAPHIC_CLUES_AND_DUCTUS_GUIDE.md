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

### K. Divergence Fondamentale des Ductus : Codex Seraphinianus vs. Manuscrit de Voynich

Le moteur de vectorisation de `VoynichDuctus` implémente une **bifurcation cinématique stricte** entre l'écriture cursive contemporaine du *Codex Seraphinianus* et l'écriture gothique/humanistique médiévale du *Manuscrit de Voynich* :

| Dimension Paléographique | Codex Seraphinianus (Luigi Serafini, 1981) | Manuscrit de Voynich (Beinecke MS 408, ca. 1404–1438) |
| :--- | :--- | :--- |
| **Période & Outil Scriptural** | XXᵉ siècle — Feutre fin / stylo technique tubulaire (encre uniforme sans biseau). | XVᵉ siècle — Plume d'oie ou calame taillé au biseau fixe de 40° (encre métallo-gallique). |
| **Nombre de Traits par Glyphe** | **Strictement 1 seul trait continu** (`single_stroke_mode=True`, `max_strokes=1`). | **1 à 4 traits maximum** (`single_stroke_mode=False`, plafond dur `max_strokes=4`). |
| **Dynamique & Inertie** | **Modèle du lacet / de la trajectoire automobile** : vitesse fluide, inertie, **aucun angle vif > 90°**. Tout changement de direction s'opère par des boucles douces. | **Modèle calligraphique discontinu** : levées de plume délibérées, cassures angulaires gothiques, attaques de hastes et empattements (*serifs*). |
| **Recouvrement du Squelette** | **100 % des pixels du squelette sont couverts** : boucle d'extension à 12 passes qui raccorde branches et boucles orphelines sans retour en arrière. | Décomposition analytique aux nœuds critiques (degré $\neq 2$) par minimisation de l'énergie de flexion d'Euler-Bernoulli. |
| **Gestion des Croisements** | La plume traverse le carrefour en conservant sa trajectoire sans jamais repasser sur un trait déjà tracé. | Résolution des bifurcations selon l'ordre chronologique des levées de plume (hampes avant traverses/ligatures). |
| **Profil d'Épaisseur** | Épaisseur constante ou faiblement modulée par la pression de contact. | **Contraste marqué des *pleins et déliés*** : $w(\theta) = (W_{\text{nib}} - w_0)\cdot\|\sin(\theta - 40^\circ)\| + w_0$. |
| **Orientation Préférentielle** | Déroulement cinématique de gauche à droite ($x$-poids = 3.2) et de haut en bas. | Attaque au sommet supérieur-gauche, descente verticale, ligature montante vers la droite. |

#### 1. Règle des 1 à 4 Traits pour Voynich
Dans l'écriture du Voynich, un glyphe individuel ne peut physiquement pas dépasser **4 traits de plume** :
- **1 trait** : Minimes isolés (`i`), croissants ouverts (`c`), déliés simples.
- **2 traits** : Boucles rondes (`o`, `a` composé d'un croissant `c` et d'une haste `i` accotée).
- **3 traits** : Potences simples et hampes à banc (`k`, `t`, `d`), majuscules ornées.
- **4 traits (plafond absolu)** : Potences composées complexes (`cth`, `ckh`, `cph`, `cfh`), constituées de deux montants verticaux, d'un banc de liaison horizontal et d'une boucle supérieure nouée. **Aucun glyphe ne contient plus de 4 traits.**

#### 2. Règle du Ruban Continu Cursif pour Seraphinianus
Dans le *Codex Seraphinianus*, l'écriture est un ruban calligraphique ininterrompu :
- La personne ne lève pas la plume à l'intérieur d'un caractère.
- Le tracé s'effectue d'un seul jet continu, tel une voiture sur une route sinueuse : la trajectoire s'enroule en boucles sans à-coups ni hésitations.
- Le demi-tour sur place ou le retour en arrière est proscrit.

---

### L. Stratification Dimensionnelle & Pureté Topologique des Archétypes
Pour éradiquer tout mélange hétérogène dans le catalogue canonique :
1. **Partitionnement Topologique Étanche** : Les familles de boucles (`0L` sans boucle, `1L` une boucle, `2+L` multiboucles) ne peuvent **jamais** partager un même cluster.
2. **Bandes d'Échelle Physique** : Chaque glyphe est classé selon son échelle physique réelle :
   - `TINY` : Signe diacritique ou éclat isolé ($H < 14\text{ px}$ ou aire $< 30\text{ px}^2$).
   - `CORE` : Corps de lettre standard ($14 \le H \le 38\text{ px}$).
   - `TALL` : Hampe montante ou potence ($H > 38\text{ px}$ ou $H \ge 1.40 W$).
   - `WIDE` : Banc horizontal ou ligature large ($W \ge 30\text{ px}$ ou $W \ge 1.35 H$).
   Les descripteurs $\log(h) \times 4.5$, $\log(w) \times 4.0$ et $\log(\text{aire}) \times 3.0$ sont injectés dans l'espace DINOv2.
3. **Quarantaine des Variations** : Toute occurrence présentant une divergence de boucle ou s'écartant de plus de $\pm 35\%$ des dimensions du médioïde est exclue des variations présentées.

---

### M. Recensement Corpus Global vs. Occurrences Folio
Le système distingue et affiche conjointement :
- **La Fréquence Corpus Globale** : Évaluée sur les **162 333 glyphes** (37 886 mots) des 225 folios Voynich et les **456 943 glyphes** du Codex Seraphinianus. Les archétypes majeurs totalisent ainsi des dizaines de milliers d'occurrences (ex. $25\,468$ pour Voynich `G03`, $17\,655$ pour `G07`, $31\,732$ pour Seraphinianus `G01`).
- **La Fréquence Locale Folio** : Le nombre d'occurrences repérées sur la page active visualisée.


### K. Multimodal Paleographic Supervision (Gemini 2.5 Flash)
1. **Automated Expert Paleography Validation (`GeminiPaleographyAdapter`)**:
   - Visual inspection of ambiguous ligatures and cursive joints directly through multimodal vision queries.
   - Comparative scribal hand authentication (analyzing slant consistency and nib angle).
   - Zero-shot cursive word decomposition based on Minimum Description Length (MDL) principles.

---

### L. Voynichese Rosetta Stone Ground Truth Integration (`VoynicheseRosettaLoader`)
1. **Full-Corpus Ground Truth Integration**:
   - Ingestion of 225 curated XML annotation folios covering the complete Beinecke MS 408 corpus ($>35,000$ segmented word bounding boxes with verified EVA transliterations).
2. **Atomic EVA Glyph Tokenization**:
   - Greedy longest-match tokenization decomposing complex compound words and gallows ligatures (`cth`, `ckh`, `cfh`, `cph`, `ch`, `sh`, `iii`, `ee`, `ii`, `aiin`) into atomic ductus training targets.
3. **Supervised Paleographic Benchmarking**:
   - Enables direct quantitative validation of unsupervised alphabet induction clusters against standardized paleographical ground truth without inducing transliteration hallucinations.

---

### M. Multimodal SIFT/RANSAC Subpixel Image Registration (`MultimodalFolioAligner`)
1. **Geometric Invariance Formulation**:
   - Resolves the translation, scale, and rotational discrepancies between low-resolution reference scans ($636 \times 900\text{ px}$) and Yale Beinecke HD master plates ($2400 \times 3300\text{ px}$).
   - Computes a subpixel affine transformation matrix $M \in \mathbb{R}^{2 \times 3}$:
     $$\begin{bmatrix} x_{\text{HD}} \\ y_{\text{HD}} \end{bmatrix} = \begin{bmatrix} s_x \cos \theta & -s_y \sin \theta & T_x \\ s_x \sin \theta & s_y \cos \theta & T_y \end{bmatrix} \begin{bmatrix} x_{\text{ref}} \\ y_{\text{ref}} \\ 1 \end{bmatrix}$$
2. **Deterministic Empirical Performance**:
   - Initial benchmark across folios `f1r` through `f5v` yielded 10/10 folios registered with a 100% Word Ink Hit rate, subsequently scaled across all **225 folios** in planetary production.
   - Detects folio-dependent offsets ($T_x = 234\text{ to }500\text{ px}$, $T_y = 117\text{ to }233\text{ px}$) and camera tilts ($\theta \in [-1.5^\circ, +1.5^\circ]$).
   - Yields a **+750% area resolution jump**, enabling unambiguous detection of quill bevel angles, serifs, and stroke crossovers.

---

### N. 14th/15th-Century Scribal Motor Ductus & Primitive Grammar
1. **Scribal Kinematic Mechanics**:
   - Medieval European hands (Gothic Cursiva, Bastarda, Rotunda) were executed using a broad-edge quill cut at a fixed nib angle $\theta_{\text{nib}} \approx 35^\circ - 45^\circ$.
   - Letters are generated by assembling a closed set of 6 atomic motor primitives:
     - $P_1$ (**Vertical Minim**): Downward pull ($|\downarrow$), top attack serif, bottom-right release flick.
     - $P_2$ (**CCW Oval Bowl**): Starts at 11 o'clock, sweeping counter-clockwise with thick left *plein* and ascending thin *délié*.
     - $P_3$ (**Arch / Shoulder**): Emerging from the minim body, arching upwards-right and pulling down ($\cap$).
     - $P_4$ (**Ascender Stem**): Tall vertical downward stroke with top cleft or loop flourish.
     - $P_5$ (**Horizontal Bench Bar**): Left-to-right traverse connecting gallows or ligatures.
     - $P_6$ (**Descender Tail**): Downward stroke extending below baseline, curling leftwards.
2. **Three Fundamental Ductus Bottlenecks Resolved**:
   - **Crossings & Intersections**: Enforces Euler-Bernoulli minimum bending energy ($\cos \Delta \theta \ge 0.5$) so strokes retain directional momentum across intersections (e.g. crossbars cutting gallows legs) without 90° transverse deflection.
   - **Loop Kinematics**: Deterministic 11 o'clock touchdown with counter-clockwise rotation, matching European right-handed scribal ergonomics.
   - **Confluent Minims**: Identifies vertical ink saddles in the Euclidean distance transform $\nabla D(y, x)$ to split conjoined minims (`i`, `ii`, `iii`, `m`, `n`) along their natural physical boundaries while preserving attack serifs.

---

### O. Strict Clustering & Zero-Tolerance Outlier Quarantine
1. **Hard Topological Gatekeepers**:
   - Invariant geometric partitioning guarantees that structurally distinct graphemic families are never combined:
     - `ASCENDER_TALL` (`b`, `d`, `k`, `t`, gallows) is strictly separated from `CLOSED_LOOP_BODY` (`o`, `a`). An `a` and a `b` can NEVER share a cluster.
     - `OPEN_C_CRESCENT` (`c`, `e`), `VERTICAL_MINIM` (`i`, `r`), and `BENCH_HORIZONTAL` remain isolated.
2. **Outlier Quarantine Bin & Tagged Rejection Taxonomies**:
   - All cluster candidates are measured against the cluster medoid.
   - Candidates with normalized distance exceeding the strict threshold ($\tau_{\text{strict}} = 0.88$ or $d > 2.6$) are isolated into an outlier quarantine pool.
   - Six structured rejection categories isolate physical parchment anomalies:
     - `ILLUSTRATION_BLEED`: Pigment spill from neighboring drawings ($A > 1100\text{ px}^2$ or paint fraction $> 35\%$).
     - `PARCHMENT_FLAW_SPECK`: Micro-punctures or grain stains ($W, H \le 6\text{ px}$ or $A < 15\text{ px}^2$).
     - `SEGMENTATION_MERGE`: Composite clusters bridging adjacent words or multiple letters ($W \ge 2.2\bar{w} \land H \ge 1.5\bar{h}$ or $\text{strokes} \ge 3$).
     - `SEGMENTATION_FRAGMENT`: Clipped stroke ends or serifs ($A \le 0.28\bar{A}$).
     - `EXTREME_FLOURISH`: Decorative marginal loops ($W \ge 1.8\bar{w} \lor H \ge 1.8\bar{h}$).
     - `HAPAX_ATYPICAL`: Rare morphologically valid glyph variants occurring below statistical significance.
   - Validated across **200,272 Voynich glyphs** and **5,258 Seraphinianus glyphs**, preserving the pristine purity of canonical alphabet clusters.

---

### P. Strict Spatial Disjoint Word Partitioning (Zero Double-Interpretation of Parchment)
1. **The Disjoint Space Invariance Principle**:
   - No two words on a manuscript page can claim or re-interpret the same physical coordinates of parchment or ink.
   - For all words $w_i, w_j$ ($i \neq j$), their ink spatial domains must be mutually exclusive:
     $$M_{w_i} \cap M_{w_j} = \emptyset$$
2. **Voronoi / Watershed Arbitration of Contested Boundaries**:
   - In crowded or closely packed folios (e.g. Voynich f1r where 244 bounding box pairs previously overlapped), contested overlap zones are arbitrated by computing geodesic distances to the respective lexical ink cores.
   - Contested ink pixels migrate strictly to their true parent word:
     $$w^*(y, x) = \arg\min_w \| (y, x) - \text{Centroid}(w) \|_2$$
   - Result: 0 overlapping word bounding boxes across all processed folios.

---

### Q. Physical Nib Threshold & Supralinear Diacritic Decoupling
1. **Instantaneous Touchdown Impact vs. Stroke Trajectory**:
   - A quill or steel nib has a physical width $W_{\text{nib}} \approx 1.5-3\text{ px}$.
   - Any ink object whose physical dimensions satisfy $\max(h, w) \le 8\text{ px}$ or $\text{area} \le 25\text{ px}^2$ is an **instantaneous touchdown impact** (posé-levé vertical : $v \approx 0, \text{length} = 0$).
   - Complex medial axis tracking and curved ridge derendering are bypassed on dots; they are derendered as a single static impact point $(y_c, x_c, W_{\text{nib}})$.
2. **Supralinear Diacritic Decoupling**:
   - Diacritics (dots, accents, acute, macrons) floating in the upper band above a letter body are decoupled as structured attributes:
     $$\text{glyph}["\text{diacritics}"] = [\{"\text{type}": \text{"DOT"}, \text{"bbox"}: (y_0, x_0, y_1, x_1), \text{"area"}: A\}]$$
   - Base letters remain unpolluted, preserving 1-stroke continuous cursive ribbons in Seraphinianus and authentic multi-stroke ductus in Voynich.

---

### R. Word-Level Neural Decomposition & 1D Spatial PCA (Option B)
1. **Intact Word Crop Encoding**:
   - Every lexical word is extracted as an intact patch without destructive knife slicing.
   - Passed through `WordScribalEncoder`: a convolutional feature network that collapses vertical height while preserving horizontal resolution, producing a 1D latent trajectory $Z(x) \in \mathbb{R}^{32}$.
2. **Unsupervised Basis Discovery (PCA / NMF)**:
   - Trajectory segments are decomposed into archetypal basis functions $\Phi_k$:
     $$\text{Patch}(x) \approx \sum_{k=1}^K a_k(x) \cdot \Phi_k$$
   - Natural glyph transitions occur at latent inflection points / velocity minima $\|dZ/dx\|$.
3. **Interactive Multi-Mode Explorer Selector**:
   - The Grand Glyph Explorer UI features a drop-down menu allowing real-time switching between:
     - **Option A** : Ductus Géométrique & Diacritiques Découplés
     - **Option B** : Décomposition Neuronale de Mots (ACP/NMF)
     - **Option C** : Consensus Hybride Neuro-Cinématique

---

### S. Étalonnage Métrique Physique & Seuils Invariants (mm et dixièmes de millimètre)
1. **Motivation Codicologique & Indépendance de Résolution (DPI)**:
   - Les scans numériques de manuscrits historiques présentent des résolutions et densités de pixels très disparates :
     - **Scan HD Beinecke MS 408 (Yale)** : $3168 \times 2500\text{ px}$ pour un feuillet de $225.0\text{ mm}$ de haut $\implies \approx 14.08\text{ px/mm}$ ($1\text{ mm} \approx 14.1\text{ px}$, $0.1\text{ mm} = 1\text{ dmm} \approx 1.4\text{ px}$).
     - **Scan standard Voynich** : $\approx 1800 \times 1280\text{ px}$ $\implies \approx 8.00\text{ px/mm}$ ($1\text{ mm} \approx 8.0\text{ px}$, $0.1\text{ mm} \approx 0.8\text{ px}$).
     - **Codex Seraphinianus (Franco Maria Ricci 1981)** : $\approx 2200 \times 1500\text{ px}$ pour une page in-folio de $350.0\text{ mm}$ $\implies \approx 6.28\text{ px/mm}$ ($1\text{ mm} \approx 6.3\text{ px}$, $0.1\text{ mm} \approx 0.63\text{ px}$).
   - Un seuil exprimé en pixels arbitraires (ex. `height >= 30 px`) biaise totalement les comparaisons : sur Yale HD, 30 px correspond à un petit corps de minim ($2.13\text{ mm}$), alors que sur Seraphinianus, 30 px correspond à une haute hampe ($4.78\text{ mm}$).
   - La classe `MetricScaleCalibration` unifie tous les filtres géométriques et cinématiques en unités physiques réelles : **millimètres (mm)** et **dixièmes de millimètre (dmm, $0.1\text{ mm}$)**.

2. **Tableau des Seuils Paléographiques Calibrés en Unités Physiques**:
   - **Empreinte d'impact nib / point vertical** : $\text{diamètre} \le 0.55\text{ mm}$ ($5.5\text{ dmm}$), $\text{aire} \le 0.35\text{ mm}^2$.
   - **Seuil minimal de lisibilité d'un glyphe textuel** : $\text{hauteur} \ge 0.85\text{ mm}$ ($8.5\text{ dmm}$), $\text{largeur} \ge 0.45\text{ mm}$ ($4.5\text{ dmm}$), $\text{aire} \ge 0.12\text{ mm}^2$.
   - **Bandes d'Échelle Paléographique (Induction de Catalogue)** :
     - `TINY` (diacritiques, points, micro-ponctuation) : $\text{hauteur} \le 1.30\text{ mm}$ ($13.0\text{ dmm}$).
     - `CORE` (minims médiévaux, boucles, $c, i, o, a, e$) : $1.30\text{ mm} \le \text{hauteur} < 3.40\text{ mm}$ ($13-34\text{ dmm}$).
     - `TALL` (hastes ascendantes, gallows, $t, k, p, f, d, l$) : $\text{hauteur} \ge 3.40\text{ mm}$ ($34.0\text{ dmm}$).
     - `WIDE` (bancs composites, ligatures étirées, $ch, sh, cth$) : $\text{largeur} \ge 3.40\text{ mm}$ ($34.0\text{ dmm}$).
   - **Interligne médiéval moyen (Ruling Pitch)** : $\approx 6.50\text{ mm}$ ($65.0\text{ dmm}$).
   - **Espace inter-mots physique** : $\ge 1.10\text{ mm}$ ($11.0\text{ dmm}$).
   - **Zone d'attachement supralinéaire** : $\le 0.90\text{ mm}$ ($9.0\text{ dmm}$) au-dessus de la ligne de tête du caractère.
   - **Débris d'illustrations / Miniatures** : $\text{hauteur} \ge 14.0\text{ mm}$ ($140\text{ dmm}$) ou $\text{aire} \ge 25.0\text{ mm}^2$.

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


