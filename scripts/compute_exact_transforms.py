import sys
sys.path.insert(0, 'src')
from PIL import Image
import numpy as np
from skimage.feature import match_template
from skimage.transform import rescale
from voynich_ductus.ingestion.iiif_client import IIIFClient
from pathlib import Path

iiif = IIIFClient(cache_dir='data/scans/voynich/yale')
mapping = [
    ('f001r', 'f1r'), ('f001v', 'f1v'),
    ('f002r', 'f2r'), ('f002v', 'f2v'),
    ('f003r', 'f3r'), ('f003v', 'f3v'),
    ('f004r', 'f4r'), ('f004v', 'f4v')
]

for fid, norm in mapping:
    g_img = Image.open(f'data/scans/voynich/glance/{norm}.jpg')
    gw, gh = g_img.size
    y_path = iiif.download_folio(fid, max_width=2400)
    y_img = Image.open(y_path)
    
    g_gray = np.array(g_img.convert('L')).astype(np.float32) / 255.0
    y_gray = np.array(y_img.convert('L')).astype(np.float32) / 255.0
    
    y_sub = rescale(y_gray, 0.5, anti_aliasing=True)
    
    py0, px0 = int(gh * 0.35), int(gw * 0.30)
    py1, px1 = int(gh * 0.60), int(gw * 0.70)
    patch = g_gray[py0:py1, px0:px1]
    
    best_corr = -1
    best_s = 1.8
    best_loc = (0, 0)
    for s in np.linspace(1.5, 2.1, 13):
        p_sc = rescale(patch, s, anti_aliasing=True)
        if p_sc.shape[0] >= y_sub.shape[0] or p_sc.shape[1] >= y_sub.shape[1]:
            continue
        c = match_template(y_sub, p_sc)
        mv = np.max(c)
        if mv > best_corr:
            best_corr = mv
            best_s = s
            best_loc = np.unravel_index(np.argmax(c), c.shape)
            
    full_scale = best_s * 2.0
    ty = best_loc[0] * 2.0 - py0 * full_scale
    tx = best_loc[1] * 2.0 - px0 * full_scale
    
    xml_w, xml_h = (1090.0 if norm == 'f1r' else 1061.0), 1500.0
    sx = (gw / xml_w) * full_scale
    sy = (gh / xml_h) * full_scale
    print(f'"{fid}": ({sx:.4f}, {sy:.4f}, {tx:.1f}, {ty:.1f}),  # corr={best_corr:.3f}')
