#!/usr/bin/env python3
"""
Road Outline & Centreline Extractor
------------------------------------
Extracts clean road geometry from a top-down map screenshot
(Snazzy Maps / Google Maps dark-green "no-labels" style).

Pipeline
--------
  1. Colour segmentation  — HSV threshold isolates light grey-green road pixels
  2. Morphological clean  — close (bridge gaps) -> open (remove noise)
  3. Blob filter          — drops circular / tiny non-road artifacts by
                            circularity score  (circles ~1.0, roads ~0.01)
  4. Contour extraction   — TC89_KCOS + Douglas-Peucker smoothing
  5. Skeletonization      — Zhang-Suen medial axis at half-resolution for speed
  6. Dashed divider       — short dash / wide gap pattern painted on skeleton
  7. Render x2            — (a) outlines only  (b) outlines + dashed centreline

Colour assumptions (measured from image)
-----------------------------------------
  Roads      : HSV  H in [73,92]   S in [10,52]  V >= 108  (light grey-green)
  Background : HSV  H ~ 81         S ~ 50         V ~ 96   (darker green)
  Water/bldg : V < 60                                       (near-black)

Usage
-----
  python road_pipeline.py                   # processes all images in ./tests/
  python road_pipeline.py tests/mymap.png   # single image

Outputs are written to ./outputs/ and named after the input file:
  <stem>_outlines.png   -- white background, black road edges only
  <stem>_final.png      -- white background, black edges + neon-yellow dashes

Input map source: https://snazzymaps.com/style/72543/assassins-creed-iv
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# Parameters

ROAD_H = (73, 92);  ROAD_S = (10, 52);  ROAD_V = (108, 255)

CLOSE_K = 5;  OPEN_K = 5;  MIN_AREA = 200
CIRC_THRESHOLD = 0.45;  CIRC_MAX_AREA = 5000

SKEL_SCALE = 0.5
DASH_LEN = 12;  GAP_LEN = 30

COL_BG   = (255, 255, 255)
COL_EDGE = (0,   0,   0)
COL_DASH = (0, 165, 255)
EDGE_THICK = 2

def segment_roads(img):
    denoised = cv2.bilateralFilter(img, 7, 40, 40)
    hsv      = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lo       = np.array([ROAD_H[0], ROAD_S[0], ROAD_V[0]], dtype=np.uint8)
    hi       = np.array([ROAD_H[1], ROAD_S[1], ROAD_V[1]], dtype=np.uint8)
    mask     = cv2.inRange(hsv, lo, hi)
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K,  OPEN_K))
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    return mask

def remove_blobs(mask):
    clean = mask.copy()
    lbl   = label(mask > 0)
    for r in regionprops(lbl):
        circ = (4 * np.pi * r.area) / (r.perimeter ** 2 + 1e-6)
        if (r.area < CIRC_MAX_AREA and circ > CIRC_THRESHOLD) or r.area < MIN_AREA:
            for coord in r.coords:
                clean[coord[0], coord[1]] = 0
    return clean

def get_contours(mask):
    raw, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    return [cv2.approxPolyDP(c, 1.2, True) for c in raw]

def build_skeleton(mask):
    H, W  = mask.shape
    sw, sh = int(W * SKEL_SCALE), int(H * SKEL_SCALE)
    small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    skel  = skeletonize(small.astype(bool)).astype(np.uint8) * 255
    skel  = cv2.resize(skel, (W, H), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(skel, np.ones((2, 2), np.uint8))

def build_dashes(skel):
    H, W   = skel.shape
    ys, xs = np.where(skel > 0)
    dashed = np.zeros((H, W), np.uint8)
    period = DASH_LEN + GAP_LEN
    for i, (x, y) in enumerate(zip(xs, ys)):
        if (i % period) < DASH_LEN:
            dashed[y, x] = 255
    return cv2.dilate(dashed, np.ones((2, 2), np.uint8))

def render(shape, contours, dashes=None):
    H, W   = shape[:2]
    canvas = np.full((H, W, 3), COL_BG, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, COL_EDGE, EDGE_THICK)
    if dashes is not None:
        canvas[dashes > 0] = COL_DASH
    return canvas

def process_image(input_path, outputs_dir):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  [ERROR] Cannot read: {input_path}")
        return

    stem         = input_path.stem
    out_outlines = outputs_dir / f"{stem}_outlines.png"
    out_final    = outputs_dir / f"{stem}_final.png"

    print(f"\n  Input  : {input_path}  ({img.shape[1]}x{img.shape[0]} px)")

    mask     = segment_roads(img)
    clean    = remove_blobs(mask)
    contours = get_contours(clean)
    skel     = build_skeleton(clean)
    dashes   = build_dashes(skel)

    cv2.imwrite(str(out_outlines), render(img.shape, contours))
    print(f"  Saved  : {out_outlines.resolve()}")

    cv2.imwrite(str(out_final), render(img.shape, contours, dashes))
    print(f"  Saved  : {out_final.resolve()}")

def main():
    script_dir  = Path(__file__).resolve().parent
    tests_dir   = script_dir / "tests"
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        process_image(Path(sys.argv[1]), outputs_dir)
        return

    if not tests_dir.exists():
        print(f"[ERROR] tests/ folder not found at: {tests_dir}")
        print("Create it and place your map images inside.")
        return

    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    sources = sorted(p for p in tests_dir.iterdir() if p.suffix.lower() in exts)

    if not sources:
        print("No images found in tests/")
        return

    print(f"Found {len(sources)} image(s) in {tests_dir}")
    for src in sources:
        process_image(src, outputs_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()
