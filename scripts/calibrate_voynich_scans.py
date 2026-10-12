import sys
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from voynich_ductus.ingestion.binarization import Binarizer

binarizer = Binarizer(method="sauvola")

folio_pairs = [
    ("f001r", "f1r"),
    ("f001v", "f1v"),
    ("f002r", "f2r"),
    ("f002v", "f2v"),
    ("f003r", "f3r"),
    ("f003v", "f3v"),
    ("f004r", "f4r"),
    ("f004v", "f4v")
]

print("CALIBRATION_START")
for folio_id, xml_id in folio_pairs:
    scan = Image.open(f"data/scans/voynich/yale/{folio_id}.jpg")
    ink = binarizer.binarize(scan)
    root = ET.parse(f"data/annotations/voynichese/{xml_id}.xml").getroot()
    words = root.findall("word")
    sample_words = words[::max(1, len(words) // 25)]

    best_score = -1.0
    best_params = None

    if folio_id == "f001r":
        ty_range = np.linspace(480, 560, 9)
        tx_range = np.linspace(80, 140, 7)
    else:
        ty_range = np.linspace(50, 180, 14)
        tx_range = np.linspace(40, 120, 9)

    sx_range = np.linspace(2.20, 2.45, 6)
    sy_range = np.linspace(2.00, 2.25, 6)

    for sx in sx_range:
        for sy in sy_range:
            for tx in tx_range:
                for ty in ty_range:
                    hits = 0
                    total_area = 0
                    for w in sample_words:
                        x0 = int(tx + float(w.attrib["x"]) * sx)
                        y0 = int(ty + float(w.attrib["y"]) * sy)
                        x1 = int(tx + (float(w.attrib["x"]) + float(w.attrib["width"])) * sx)
                        y1 = int(ty + (float(w.attrib["y"]) + float(w.attrib["height"])) * sy)
                        if 0 <= y0 < y1 <= ink.shape[0] and 0 <= x0 < x1 <= ink.shape[1]:
                            patch = ink[y0:y1, x0:x1]
                            hits += np.count_nonzero(patch)
                            total_area += patch.size
                    score = hits / max(1, total_area)
                    if score > best_score:
                        best_score = score
                        best_params = (round(float(sx), 3), round(float(sy), 3), round(float(tx), 1), round(float(ty), 1))

    print(f'"{folio_id}": {{"sx": {best_params[0]}, "sy": {best_params[1]}, "tx": {best_params[2]}, "ty": {best_params[3]}}},  # score={best_score:.4f}')

