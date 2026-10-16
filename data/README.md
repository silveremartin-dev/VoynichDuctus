# VoynichDuctus — Input Data Directory (`data/`)

This directory is reserved strictly for **raw input data**, canonical paleographical references, and source annotations.

---

## 1. Directory Structure

```
data/
├── annotations/
│   └── voynichese/            # Transliteration ground-truth and word bounding boxes (Apache-2.0)
│       └── js_cache/          # Cached folio coordinate databases
└── scans/
    ├── voynich/               # High-resolution Voynich Manuscript scans
    │   ├── yale/              # Yale Beinecke MS 408 IIIF plates (2400x3300)
    │   └── voynichese_scans/  # Reference glance plates (636x900)
    ├── seraphinianus/         # Local PDF / plate inputs for Codex Seraphinianus
    └── gallica/               # Authenticated 15th-century Latin herbal plates (BnF Gallica)
```

---

## 2. Public Scan Sources & Archive.org Links

### A. Codex Seraphinianus (Luigi Serafini, 1981)
- **Status**: Under exclusive copyright (Luigi Serafini / Franco Maria Ricci).
- **Redistribution Policy**: Scanned raster pages of the *Codex Seraphinianus* **are NOT distributed in this repository**. Only derived mathematical stroke coordinates (`.json`) are published for non-commercial scientific research.
- **Archive.org Preservation Copy**:
  - Direct Link: [Luigi Serafini — Codex Seraphinianus on Archive.org](https://archive.org/details/Luigi.Serafini..Codex.Seraphinianus_201608/page/n315/mode/2up)
  - Users wishing to run the visual pipeline locally can download the PDF scan from Archive.org and place it in `data/scans/seraphinianus/Codex Seraphinianus.pdf`.

### B. Voynich Manuscript (Beinecke MS 408, 15th Century)
- **Status**: Public Domain.
- **Archive.org Master Scan**: [The Voynich Manuscript on Archive.org](https://archive.org/details/TheVoynichManuscript)
- **Yale Beinecke Digital Library**: [Beinecke MS 408 High-Res Collection](https://beinecke.library.yale.edu/)

### C. Medieval Latin Herbals (BnF Gallica)
- **Status**: Public Domain (Digitized by Bibliothèque nationale de France).
- Codices: *Latin 6823, Latin 6862, Français 12322*. Available via the Gallica IIIF API.
