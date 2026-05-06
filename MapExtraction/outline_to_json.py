import cv2
import numpy as np
import json
import os
from pathlib import Path
from skimage.morphology import skeletonize

def convert_outline_png_to_json(image_path, output_json_path, mask_path=None):
    # 1. Load image (try IMREAD_UNCHANGED to handle alpha)
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Handle transparency and color
    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        # BGRA: Composite over white
        alpha = img_raw[:,:,3] / 255.0
        bgr = img_raw[:,:,:3]
        img_bgr = (bgr * alpha[:,:,np.newaxis] + 255 * (1 - alpha[:,:,np.newaxis])).astype(np.uint8)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif len(img_raw.shape) == 3:
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        img = img_raw

    # 2. Preprocess: Robust thresholding
    # We want binary where 1 is the outline.
    # We'll calculate the mean. If mean > 127, it's likely white background.
    mean_val = np.mean(img)
    if mean_val > 127:
        # Light background, dark outlines
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        # Dark background, light outlines
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # Filter out very large noise if it covers the whole screen
    if np.mean(binary) > 200: # Mostly full
        # This might be pure noise or inverted incorrectly, try simple adaptive
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2b. Apply manual edits if mask is provided
    if mask_path and os.path.exists(mask_path):
        mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_bgr is not None:
            # Resize mask to match image if necessary
            if mask_bgr.shape[:2] != binary.shape:
                mask_bgr = cv2.resize(mask_bgr, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # OpenCV is BGR.
            # Red marks (Eraser) -> mask_bgr[:,:,2] is high.
            # Green marks (Pencil) -> mask_bgr[:,:,1] is high.
            
            # Apply Red edits: Erase (set binary to 0)
            binary[mask_bgr[:,:,2] > 100] = 0
            
            # Apply Green edits: Add (set binary to 255)
            # We use a slightly higher threshold for green to avoid noise
            binary[mask_bgr[:,:,1] > 100] = 255
            
            print(f"Applied manual edits (Red/Green) from {mask_path}")

    # 3. Find Edge Contours
    # Use RETR_CCOMP to get a 2-level hierarchy:
    # - external boundaries (level 0)
    # - internal holes (level 1)
    # This helps us avoid "double lines" when the input outlines have thickness.
    # We only want the outer boundary of each "line" in the image.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Simplify contours
    edges = []
    if hierarchy is not None:
        hierarchy = hierarchy[0] # Flatten
        for i, cnt in enumerate(contours):
            # Only keep "external" contours (those with no parent)
            # hierarchy[i] is [Next, Previous, First_Child, Parent]
            if hierarchy[i][3] == -1:
                # Reduce epsilon for more detail (0.0003 for high detail)
                epsilon = 0.0003 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                # Keep even small segments if they contribute to the structure
                if len(points) >= 2:
                    edges.append(points)

    # 4. JSON Structure
    data = {
        "map_style": "custom_extraction",
        "h": img.shape[0],
        "w": img.shape[1],
        "image_scale": 1.0,
        "divider": [], 
        "all_dividers": [],
        "edges": {
            "all_segments": edges
        }
    }
    
    # 5. Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    # 6. Create AND Save Visualization on a CLEAN background (White)
    # Create white background of same size
    vis = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
    # Draw edges in blue
    for edge in edges:
        pts = np.array(edge, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (255, 100, 0), 2)
        
    vis_path = os.path.join(os.path.dirname(image_path), f"vis_{os.path.basename(image_path)}")
    cv2.imwrite(vis_path, vis)
    
    print(f"Successfully converted {image_path} to {output_json_path}")
    print(f"Captured {len(edges)} edge segments.")
    print(f"Visualization saved to {vis_path}")
    
    return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python outline_to_json.py [input_png] [output_json] [optional_mask_png]")
        sys.exit(1)
    
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    mask_p = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_outline_png_to_json(in_path, out_path, mask_p)
