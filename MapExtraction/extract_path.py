import cv2
import numpy as np
import sys
import json
try:
    from skimage.morphology import skeletonize
except ImportError:
    # Fallback if skimage is not available in the specific environment
    def skeletonize(img):
        return cv2.ximgproc.thinning(img) if hasattr(cv2, 'ximgproc') else img

def main(mask_path, output_json):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read mask")
        sys.exit(1)
    
    # 1. Threshold
    binary = (img > 50).astype(np.uint8) * 255
    
    # 2. Skeletonize/Thinning
    # Using cv2.ximgproc.thinning if available, otherwise skimage
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    except:
        skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary

    y, x = np.where(skeleton > 0)
    
    if len(x) < 2:
        print("Not enough points found in skeleton")
        with open(output_json, 'w') as f:
            json.dump([], f)
        return

    # 3. Simple greedy ordering to form a path
    pts = list(zip(x, y))
    ordered = [pts.pop(0)]
    
    while pts:
        last = ordered[-1]
        # Find nearest point
        # We limit the search distance to avoid jumps between disconnected segments
        dists = [(p[0]-last[0])**2 + (p[1]-last[1])**2 for p in pts]
        idx = np.argmin(dists)
        if dists[idx] > 400: # Max jump ~20 pixels
             break
        ordered.append(pts.pop(idx))
    
    # 4. Downsample if path is too dense
    if len(ordered) > 50:
        step = max(1, len(ordered) // 50)
        ordered = ordered[::step]

    res = [{"x": int(p[0]), "y": int(p[1])} for p in ordered]
    with open(output_json, 'w') as f:
        json.dump(res, f)
    print(f"Extracted {len(res)} points from mask.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_path.py [mask_path] [output_json]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
