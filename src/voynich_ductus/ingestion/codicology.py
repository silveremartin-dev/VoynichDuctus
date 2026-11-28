"""
Codicology and manuscript page ordering module for VoynichDuctus.

Implements structural quire models and page ordering schemes for the Voynich Manuscript (Beinecke MS 408):
1. Traditional Codex Foliation: Sequential leaf order (f1r to f116v) reflecting the 15th-century binding.
2. Layfield & Davis (2026) Singulion Model: Reconstructed ordering from the seminal study:
   Colin Layfield and Lisa Fagin Davis, "Singulion Structure and the Voynich Manuscript",
   Digital Medievalist, Vol. 19 (2026).

Theoretical background:
The study demonstrates via Latent Semantic Analysis (LSA) and physical codicology that the manuscript
was originally produced as independent single-bifolio booklets ("singulions") read sequentially (Ar -> Av -> Br -> Bv)
rather than nested quires. Subsequent mis-binding disrupted textual continuity and split illustrations across gutters.
Combinatorial permutation optimization restored optimal textual transitions for Quire 13 (+12.1% facing score)
and Quire 20 (+20.0% facing score).
"""

from typing import Dict, Any, List, Tuple, Optional, Iterable
import re

LAYFIELD_DAVIS_CITATION = {
    "title": "Singulion Structure and the Voynich Manuscript",
    "authors": ["Colin Layfield", "Lisa Fagin Davis"],
    "journal": "Digital Medievalist",
    "volume": 19,
    "year": 2026,
    "doi": "https://doi.org/10.16995/dm.11942",
    "summary": (
        "Demonstrates through Latent Semantic Analysis (LSA) and physical codicology that "
        "the Voynich Manuscript was not conceived as nested quires, but as individual, "
        "sequential folded bifolia ('singulions'). It was subsequently mis-bound in the 15th century, "
        "disrupting textual and visual continuity across internal gutters. Reordering quires as singulions "
        "— notably Quire 13 (Balneological, +12.1% LSA) and Quire 20 (Marginal Stars, +20.0% LSA) — "
        "restores continuity and aligns illustration boundaries."
    )
}

# Formal specification of Beinecke MS 408 quires and bifolia
# Each quire defines:
# - section: thematic content
# - traditional_bifolia: outer-to-inner nesting of bifolia pairs
# - layfield_davis_bifolia: sequence of singulions according to Layfield & Davis (2026)
VOYNICH_QUIRES: List[Dict[str, Any]] = [
    {
        "quire": "Q01",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(1, 8), (2, 7), (3, 6), (4, 5)],
        "layfield_davis_bifolia": [(1, 8), (2, 7), (3, 6), (4, 5)],
    },
    {
        "quire": "Q02",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(9, 16), (10, 15), (11, 14), (13,)],
        "layfield_davis_bifolia": [(9, 16), (10, 15), (11, 14), (13,)],
    },
    {
        "quire": "Q03",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(17, 24), (18, 23), (19, 22), (20, 21)],
        "layfield_davis_bifolia": [(17, 24), (18, 23), (19, 22), (20, 21)],
    },
    {
        "quire": "Q04",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(25, 32), (26, 31), (27, 30), (28, 29)],
        "layfield_davis_bifolia": [(25, 32), (26, 31), (27, 30), (28, 29)],
    },
    {
        "quire": "Q05",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(33, 40), (34, 39), (35, 38), (36, 37)],
        "layfield_davis_bifolia": [(33, 40), (34, 39), (35, 38), (36, 37)],
    },
    {
        "quire": "Q06",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(41, 48), (42, 47), (43, 46), (44, 45)],
        "layfield_davis_bifolia": [(41, 48), (42, 47), (43, 46), (44, 45)],
    },
    {
        "quire": "Q07",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(49, 56), (50, 55), (51, 54), (52, 53)],
        "layfield_davis_bifolia": [(49, 56), (50, 55), (51, 54), (52, 53)],
    },
    {
        "quire": "Q08",
        "section": "Botanical (Herbal 1)",
        "traditional_bifolia": [(57, 66), (58, 65)],
        "layfield_davis_bifolia": [(57, 66), (58, 65)],
    },
    {
        "quire": "Q09",
        "section": "Cosmological / Astrological",
        "traditional_bifolia": [(67,), (68,)],
        "layfield_davis_bifolia": [(67,), (68,)],
    },
    {
        "quire": "Q10",
        "section": "Cosmological / Astrological",
        "traditional_bifolia": [(69,), (70,)],
        "layfield_davis_bifolia": [(69,), (70,)],
    },
    {
        "quire": "Q11",
        "section": "Cosmological / Zodiac",
        "traditional_bifolia": [(71,), (72,), (73,)],
        "layfield_davis_bifolia": [(71,), (72,), (73,)],
    },
    {
        "quire": "Q13",
        "section": "Balneological (Biological)",
        # Traditional nested quire order: outer 75|84, 76|83, 77|82, 78|81, inner 79|80
        "traditional_bifolia": [(75, 84), (76, 83), (77, 82), (78, 81), (79, 80)],
        # Layfield & Davis (2026, Table 14): optimal LSA transition sequence:
        # 77|82 -> 78|81 -> 75|84 -> 76|83 -> 79|80 (+12.1% facing score)
        "layfield_davis_bifolia": [(77, 82), (78, 81), (75, 84), (76, 83), (79, 80)],
        "optimization_notes": "Optimal LSA sequence puts pool illustrations 78|81 and 75|84 adjacent (facing score +12.1%).",
    },
    {
        "quire": "Q14",
        "section": "Rosettes Foldout",
        "traditional_bifolia": [(85,), (86,)],
        "layfield_davis_bifolia": [(85,), (86,)],
    },
    {
        "quire": "Q15",
        "section": "Pharmaceutical / Herbal 2",
        "traditional_bifolia": [(87, 90), (88, 89)],
        "layfield_davis_bifolia": [(87, 90), (88, 89)],
    },
    {
        "quire": "Q17",
        "section": "Pharmaceutical / Herbal 2",
        "traditional_bifolia": [(93, 96), (94, 95)],
        "layfield_davis_bifolia": [(93, 96), (94, 95)],
    },
    {
        "quire": "Q19",
        "section": "Pharmaceutical / Herbal 2",
        "traditional_bifolia": [(99, 102), (100, 101)],
        "layfield_davis_bifolia": [(99, 102), (100, 101)],
    },
    {
        "quire": "Q20",
        "section": "Marginal Stars / Recipes",
        # Traditional nested quire order: outer 103|116, 104|115, 105|114, 106|113, 107|112, inner 108|111
        "traditional_bifolia": [(103, 116), (104, 115), (105, 114), (106, 113), (107, 112), (108, 111)],
        # Layfield & Davis (2026, Table 14): optimal LSA transition sequence:
        # 105|114 -> 104|115 -> 106|113 -> 107|112 -> 108|111 -> 103|116 (+20.0% facing score)
        "layfield_davis_bifolia": [(105, 114), (104, 115), (106, 113), (107, 112), (108, 111), (103, 116)],
        "optimization_notes": "Optimal LSA sequence begins with oversize initial glyph on f105r and ends with blank f116v (+20.0% facing score).",
    },
]


def natural_folio_sort_key(folio_id: str) -> Tuple[int, int, int, str]:
    """
    Computes a sorting tuple for natural codicological foliation:
    f1r < f1v < f2r < ... < f67r1 < f67r2 < ... < f116v.
    """
    m = re.match(r"^f0*(\d+)([rv]?)(\d*)", folio_id.lower().strip())
    if m:
        num = int(m.group(1))
        side = 0 if m.group(2) == "r" else (1 if m.group(2) == "v" else 2)
        sub = int(m.group(3)) if m.group(3) else 0
        return (num, side, sub, folio_id)
    return (9999, 0, 0, folio_id)


def get_traditional_codex_order(folio_ids: Iterable[str]) -> List[str]:
    """Sorts folios according to natural Beinecke MS 408 codex foliation (f1r -> f116v)."""
    return sorted(folio_ids, key=natural_folio_sort_key)


def _extract_folio_number(folio_id: str) -> Optional[int]:
    m = re.match(r"^f0*(\d+)", folio_id.lower().strip())
    return int(m.group(1)) if m else None


def get_layfield_davis_order(folio_ids: Iterable[str]) -> List[str]:
    """
    Sorts folios according to the Layfield & Davis (2026) Singulion model:
    - Bifolia treated as sequential 4-page units (Ar -> Av -> Br -> Bv)
    - Quire 13 and Quire 20 rearranged according to Table 14 optimal permutations.
    """
    folio_list = list(folio_ids)
    folio_set = set(folio_list)

    # Build map from folio number to list of matching page IDs sorted naturally
    pages_by_num: Dict[int, List[str]] = {}
    for p in folio_list:
        num = _extract_folio_number(p)
        if num is not None:
            pages_by_num.setdefault(num, []).append(p)

    for num in pages_by_num:
        pages_by_num[num].sort(key=natural_folio_sort_key)

    ld_ordered: List[str] = []
    seen = set()

    for q in VOYNICH_QUIRES:
        for bifolio in q["layfield_davis_bifolia"]:
            for fol_num in bifolio:
                for pid in pages_by_num.get(fol_num, []):
                    if pid in folio_set and pid not in seen:
                        ld_ordered.append(pid)
                        seen.add(pid)

    # Append any remaining pages (fallback)
    remaining = [p for p in folio_list if p not in seen]
    if remaining:
        ld_ordered.extend(sorted(remaining, key=natural_folio_sort_key))

    return ld_ordered


def build_folio_codicology_registry(folio_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """
    Builds a full codicological lookup registry for all folios in the manuscript,
    including quire, singulion pair, traditional index, and Layfield-Davis index.
    """
    trad_order = get_traditional_codex_order(folio_ids)
    ld_order = get_layfield_davis_order(folio_ids)

    trad_idx_map = {pid: i + 1 for i, pid in enumerate(trad_order)}
    ld_idx_map = {pid: i + 1 for i, pid in enumerate(ld_order)}

    # Map folio number to quire and bifolium info
    bifolio_info: Dict[int, Dict[str, Any]] = {}
    for q in VOYNICH_QUIRES:
        q_id = q["quire"]
        section = q["section"]
        notes = q.get("optimization_notes", "")
        for bifolio in q["layfield_davis_bifolia"]:
            if len(bifolio) == 2:
                s_id = f"{bifolio[0]}|{bifolio[1]}"
                conjoint = {bifolio[0]: bifolio[1], bifolio[1]: bifolio[0]}
            else:
                s_id = f"{bifolio[0]} (singleton)"
                conjoint = {}
            for num in bifolio:
                bifolio_info[num] = {
                    "quire": q_id,
                    "section": section,
                    "singulion_id": s_id,
                    "conjoint_leaf": conjoint.get(num),
                    "notes": notes,
                }

    registry: Dict[str, Dict[str, Any]] = {}
    for pid in folio_ids:
        num = _extract_folio_number(pid)
        info = bifolio_info.get(num, {
            "quire": "Unknown",
            "section": "Unknown",
            "singulion_id": "",
            "conjoint_leaf": None,
            "notes": ""
        })
        registry[pid] = {
            "folio_id": pid,
            "folio_number": num,
            "quire": info["quire"],
            "section": info["section"],
            "singulion_id": info["singulion_id"],
            "conjoint_leaf": info["conjoint_leaf"],
            "davis_scribe": get_davis_scribe_attribution(pid),
            "codex_index": trad_idx_map.get(pid, 0),
            "ld_index": ld_idx_map.get(pid, 0),
            "notes": info["notes"],
        }

    return registry


DAVIS_FIVE_SCRIBES_CITATION = {
    "title": "Voynich Paleography",
    "authors": ["Lisa Fagin Davis"],
    "proceedings": "Proceedings of the 1st International Conference on the Voynich Manuscript (CEUR Vol. 3313)",
    "year": 2022,
    "url": "https://ceur-ws.org/Vol-3313/keynote2.pdf",
    "summary": (
        "Establishes five distinct 15th-century humanist-influenced scribal hands (Scribes 1-5) across "
        "Beinecke MS 408 via forensic ductus criteria on word-final EVA [n] finials, [k] and [f] gallows "
        "crossbars/loops, and multi-gallows bridge ligatures."
    ),
}

# Explicit Herbal (f1-f56) Scribe 2 & Scribe 5 leaves per Currier / Lisa Fagin Davis (2022, pp. 4-6 & Voynich.nu table)
_HERBAL_SCRIBE_2_FOLIOS = {
    26, 31, 33, 34, 39, 40, 43, 46, 50, 55
}
_HERBAL_SCRIBE_5_FOLIOS = {41, 48}


def get_davis_scribe_attribution(folio_id: str) -> int:
    """
    Returns Lisa Fagin Davis's (2022) canonical scribe attribution (`1..5`) for any Voynich folio ID.
    Accounts for mixed Herbal bifolia, Quire 8 splits (`f57v`=1, `f57r`/`f66`=5, `f58`/`f65`=3),
    Astrological foldouts (`f67`-`f73`=4), Balneological (`f75`-`f84`=2), Rosettes (`f85`/`f86`),
    Pharmaceutical/Recipes (`f87`-`f102`), and Starred Recipes (`f103`-`f116`).
    """
    fid = folio_id.lower().strip()
    num = _extract_folio_number(fid)
    if num is None:
        return 1

    if fid == "f57v":
        return 1
    if fid == "f57r" or num == 66:
        return 5
    if num in (58, 65):
        return 3
    if 1 <= num <= 56:
        if num in _HERBAL_SCRIBE_5_FOLIOS:
            return 5
        if num in _HERBAL_SCRIBE_2_FOLIOS:
            return 2
        return 1
    if 67 <= num <= 74:
        return 4
    if 75 <= num <= 84:
        return 2
    if num in (85, 86):
        return 4 if ("v" in fid or "ros" in fid) else 2
    if num in (94, 95):
        return 3
    if 87 <= num <= 102:
        return 1
    if 103 <= num <= 116:
        return 3
    return 1


def extract_davis_scribal_forensics(glyphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quantitatively measures Lisa Fagin Davis's (2022, p. 3 & §5.2) forensic scribal "tells"
    directly from vectorized `CalligraphicVectorizer` strokes on a folio or corpus slice:
      1. Word-final `[n]` finial geometry: bottom turn angle (deg) & endpoint left-reach / closure ratio.
      2. Gallows `[k]` & `[f]` crossbar execution: lifted multi-stroke horizontal bar ratio vs bowed apex turn.
      3. Bridge-Gallows ligature count (§5.2, Fig. 5: wide multi-ascender connected gallows `width_mm >= 5.2`).
    """
    import math

    n_turn_angles: List[float] = []
    n_left_reach_mm: List[float] = []
    n_unbroken_ratio_hits = 0
    n_total = 0

    gallows_lifted_hits = 0
    gallows_bowed_hits = 0
    gallows_total = 0
    bridge_gallows_ids: List[str] = []

    for g in glyphs:
        eva_c = (g.get("eva_char") or "").lower()
        strokes = [s for s in g.get("strokes", []) if not s.get("is_impact", False) and len(s.get("points", [])) >= 2]
        w_mm = float(g.get("width_mm", 0.0))
        h_mm = float(g.get("height_mm", 0.0))
        loops = int(g.get("loop_count", 0))

        # §5.2 Bridge-Gallows ligature detection (wide tall multi-leg/loop gallows span)
        if h_mm >= 3.0 and w_mm >= 5.2 and loops >= 1:
            bridge_gallows_ids.append(g.get("glyph_id", ""))

        if not strokes:
            continue

        ppm = float(strokes[0].get("px_per_mm", 10.0)) or 10.0
        mm_per_px = 1.0 / ppm

        if eva_c == "n":
            n_total += 1
            if len(strokes) == 1:
                n_unbroken_ratio_hits += 1
            pts = strokes[0]["points"]
            if len(pts) >= 3:
                i_bot = max(range(len(pts)), key=lambda idx: float(pts[idx][0]))
                if 1 <= i_bot <= len(pts) - 2:
                    i_in = max(0, i_bot - max(1, len(pts) // 5))
                    i_out = min(len(pts) - 1, i_bot + max(1, len(pts) // 5))
                    v_in = (pts[i_bot][0] - pts[i_in][0], pts[i_bot][1] - pts[i_in][1])
                    v_out = (pts[i_out][0] - pts[i_bot][0], pts[i_out][1] - pts[i_bot][1])
                    l1, l2 = math.hypot(*v_in), math.hypot(*v_out)
                    if l1 > 1e-4 and l2 > 1e-4:
                        cos_a = max(-1.0, min(1.0, (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (l1 * l2)))
                        n_turn_angles.append(float(math.degrees(math.acos(cos_a))))
                # Positive left_reach_mm means finial finishes to the LEFT of initial minim touchdown (Scribe 1 & 5)
                n_left_reach_mm.append(float(pts[0][1] - pts[-1][1]) * mm_per_px)

        elif eva_c in ("k", "f"):
            gallows_total += 1
            # Lifted horizontal bar (Scribe 2 & 4) vs bowed apex bar
            has_horiz_bar = False
            for st in strokes[1:]:
                p_st = st["points"]
                dy_b = abs(p_st[-1][0] - p_st[0][0]) * mm_per_px
                dx_b = abs(p_st[-1][1] - p_st[0][1]) * mm_per_px
                ch_b = math.hypot(dy_b, dx_b)
                ar_b = sum(math.hypot(p_st[k + 1][0] - p_st[k][0], p_st[k + 1][1] - p_st[k][1]) for k in range(len(p_st) - 1)) * mm_per_px
                if dx_b >= 0.45 and dy_b <= 0.35 * dx_b and (ar_b / max(1e-4, ch_b)) <= 1.12:
                    has_horiz_bar = True
                    break
            if has_horiz_bar:
                gallows_lifted_hits += 1
            else:
                gallows_bowed_hits += 1

    return {
        "n_count": n_total,
        "n_single_stroke_ratio": round(n_unbroken_ratio_hits / max(1, n_total), 3) if n_total else 0.0,
        "n_mean_turn_angle_deg": round(sum(n_turn_angles) / len(n_turn_angles), 1) if n_turn_angles else None,
        "n_mean_left_reach_mm": round(sum(n_left_reach_mm) / len(n_left_reach_mm), 3) if n_left_reach_mm else None,
        "kf_gallows_count": gallows_total,
        "kf_lifted_straight_bar_ratio": round(gallows_lifted_hits / max(1, gallows_total), 3) if gallows_total else 0.0,
        "kf_bowed_bar_ratio": round(gallows_bowed_hits / max(1, gallows_total), 3) if gallows_total else 0.0,
        "bridge_gallows_count": len(bridge_gallows_ids),
        "bridge_gallows_ids": bridge_gallows_ids[:12],
    }
