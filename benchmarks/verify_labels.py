import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def verify_benchmark_outputs(target_dir):
    """
    Verifies the integrity of generated benchmark samples, 
    labels, and numpy arrays in the specified directory.
    """
    base_path = Path(target_dir)
    csv_path = base_path / "dataset.csv"
    
    print(f"--- DopplerSim Benchmark Verification ---")
    print(f"Target Directory: {base_path.absolute()}")
    
    if not csv_path.exists():
        print(f"[ERROR] dataset.csv not found at {csv_path}")
        return

    # Load labels
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded dataset.csv with {len(df)} samples.")
    print(f"\nLabel Distribution:")
    if 'trajectory_type' in df.columns:
        print(df['trajectory_type'].value_counts())
    
    # Check individual samples
    print(f"\nSample Integrity Check:")
    results = []
    
    for idx, row in df.iterrows():
        # 1. Try to get sample name directly from CSV columns
        # Priority: 'sample_id' (full name like sample_0000001) then 'sample_index'
        raw_id = row.get('sample_id', row.get('sample_index', None))
        
        possible_names = []
        if raw_id is not None:
            if isinstance(raw_id, str) and raw_id.startswith("sample_"):
                possible_names.append(raw_id)
                # Also try extract number for re-padding just in case
                try:
                    num = int(raw_id.replace("sample_", ""))
                    possible_names.extend([f"sample_{num}", f"sample_{num:07d}", f"sample_{num:03d}"])
                except: pass
            else:
                try:
                    num = int(raw_id)
                    possible_names.extend([f"sample_{num}", f"sample_{num:07d}", f"sample_{num:03d}"])
                except: pass
        
        # Always include the current loop index as a fallback
        possible_names.extend([f"sample_{idx}", f"sample_{idx+1}", f"sample_{idx:07d}", f"sample_{idx+1:07d}"])
        
        # De-duplicate while preserving order
        seen = set()
        unique_names = [x for x in possible_names if not (x in seen or seen.add(x))]
        
        sample_dir = None
        for name in unique_names:
            # 1. Check direct subfolder
            d = base_path / name
            if d.exists() and d.is_dir():
                sample_dir = d
                break
                
            # 2. Check within audio_clips
            d = base_path / "audio_clips" / name
            if d.exists() and d.is_dir():
                sample_dir = d
                break
            
            # 3. Check within batch specific naming
            d = base_path / "audio_clips" / "batch_test_batch" / name
            if d.exists() and d.is_dir():
                sample_dir = d
                break
        
        # Final fallback: Recursive search for the folder name
        if not sample_dir:
            for name in unique_names:
                matches = list(base_path.glob(f"**/{name}"))
                if matches and matches[0].is_dir():
                    sample_dir = matches[0]
                    break

        status = {"name": sample_dir.name if sample_dir else unique_names[0], "exists": sample_dir is not None}
        
        if sample_dir and sample_dir.exists():
            # Check for files
            files = os.listdir(sample_dir)
            status["wav"] = any(f.endswith(".wav") for f in files)
            status["png"] = any(f.endswith(".png") for f in files)
            status["npy_mask"] = "segmentation_mask.npy" in files
            
            # Extract ground truth from row
            status["gt"] = {
                "Speed": f"{row.get('speed_mps', 'N/A')} m/s",
                "Traj": row.get('trajectory_type', 'N/A'),
                "CPA": f"{row.get('cpa_distance_m', 'N/A')}m",
                "CPA Time": f"{row.get('cpa_time_sec', 'N/A')}s",
                "Sources": row.get('num_sources', 1),
                "Interaction": "Yes" if row.get('is_crossing', False) else "No"
            }

            # Map to Benchmark ID (Heuristic)
            b_id = "B?"
            traj = str(row.get('trajectory_type', '')).lower()
            sources = int(row.get('num_sources', 1))
            crossing = bool(row.get('is_crossing', False))
            
            if sources > 1:
                b_id = "B9 (Interaction)" if crossing else "B8 (Multi-source)"
            elif traj == "straight": b_id = "B1 (Straight)"
            elif traj == "curved": b_id = "B2 (Curved)"
            elif traj == "intersection": b_id = "B3 (Intersection)"
            elif traj == "stop_and_go" or traj == "braking": b_id = "B4 (Stop/Go)"
            elif traj == "roundabout": b_id = "B5 (Roundabout)"
            
            if status["npy_mask"]: b_id = "B6 (Segmentation)"
            
            status["benchmark"] = b_id

            # Verify .NPY Mask if exists (B6)
            if status["npy_mask"]:
                try:
                    mask = np.load(sample_dir / "segmentation_mask.npy")
                    status["mask_shape"] = mask.shape
                    status["mask_active_pct"] = (np.sum(mask) / len(mask)) * 100
                except Exception as e:
                    status["mask_error"] = str(e)
            
            # Verify Spectrogram .NPY if exists
            spec_files = [f for f in files if f.endswith(".npy") and "spec" in f]
            if spec_files:
                try:
                    spec = np.load(sample_dir / spec_files[0])
                    status["spec_shape"] = spec.shape
                except Exception as e:
                    status["spec_error"] = str(e)
        
        results.append(status)

    # Print Report
    print(f"{'Status':<8} | {'Sample ID':<16} | {'Benchmark':<18} | {'Details'}")
    print("-" * 80)
    for r in results:
        status_label = "[OK]" if r["exists"] else "[ERROR]"
        if r["exists"]:
            details = f"{r['gt']['Speed']}, {r['gt']['Traj']}, CPA:{r['gt']['CPA']}"
            if r.get("npy_mask"):
                details += f" | Mask: {r['mask_active_pct']:.1f}% active"
            
            print(f"{status_label:<8} | {r['name']:<16} | {r.get('benchmark', 'N/A'):<18} | {details}")
            
            if not r.get("wav") or not r.get("png"):
                missing = []
                if not r.get("wav"): missing.append("WAV")
                if not r.get("png"): missing.append("PNG")
                print(f"         [!] MISSING: {', '.join(missing)}")
            
            if "mask_error" in r: print(f"         [!] Mask Error: {r['mask_error']}")
        else:
            print(f"{status_label:<8} | {r['name']:<16} | {'N/A':<18} | Directory missing.")

    print(f"\nVerification Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark generation outputs.")
    parser.add_argument("path", nargs="?", default="benchmarks/test_output", 
                        help="Path to the batch output folder (e.g., output/batch_123)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.path):
        verify_benchmark_outputs(args.path)
    else:
        print(f"[ERROR] Directory not found: {args.path}")
        print("Please specify a valid path: python benchmarks/verify_labels.py output/batch_XYZ")
