import os
import sys
import json
import csv
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from audio.generation import generate_single_clip, generate_multi_object_clip, generate_random_parameters
from visualization.validation import validate_scene_paths, save_validation_report

# data generation & labeling

def generate_test_batch(num_samples=5, batch_name="test_batch", force_crossing=False, road_angle=0.0, lane_width=4.0):
    """Generate a diverse batch of clips covering B1-B10."""
    print(f"--- Generating {num_samples} test samples for Initial Suite (B1-B10) ---")
    
    # Target directory as requested
    base_dir = Path("benchmarks/test_output")
    batch_dir = base_dir / batch_name
    audio_dir = batch_dir / "audio_clips"
    os.makedirs(audio_dir, exist_ok=True)
    
    vehicles = ["car_1", "car_2", "03-HF-Mi5", "04-H1-Mi1"]
    paths = ["straight", "parabola", "bezier"]
    
    config = {
        'batch': {'total_clips': num_samples, 'name': batch_name},
        'output': {'format': 'wav', 'spectrogram_type': 'cqt'},
        'vehicles': {'selected': vehicles},
        'paths': {'selected': paths},
        'speed': {'randomize': True},
        'distance': {'randomize': True, 'min': 5, 'max': 100}, # Testing phase limit
        'acceleration': {'randomize': True, 'min': -3, 'max': 3}, # b7
        'angle': {'randomize': True},
        'atmosphere': {'randomize': True},
        'benchmarks': {
            'enabled': True,
            'selected': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'params': {
                'road_angle': road_angle,
                'lane_width': lane_width,
                'force_crossing': force_crossing,
                'max_stagger': 5.0,
                'vehicle_min': 2,
                'vehicle_max': 5,
                'include_opposite': True
            }
        }
    }
    
    clips_metadata = []
    scenes_for_validation = []
    
    for i in range(1, num_samples + 1):
        custom_name = f"test_sample_{i}"
        if i % 3 == 0:
            is_crossing = force_crossing or (random.random() < 0.3)
            print(f"[{i}/{num_samples}] Generating Multi-Source Busy Road (Crossing={is_crossing}) as '{custom_name}'...")
            obs_pos = (0.0, 0.0) # World origin
            road_y = 15.0        # Pushed upwards
            
            # Crossing logic for benchmark suite
            if is_crossing:
                # Two vehicles in the same direction (Forward), swapping slots
                v_configs = [
                    {
                        'vehicle_name': random.choice(vehicles),
                        'path_type': 'bezier',
                        'params': {
                            'speed': random.randint(22, 28), 
                            'duration': 10.0,
                            'x0': -100, 'x1': -33, 'x2': 33, 'x3': 100,
                            'y0': road_y - 3.5, 'y1': road_y - 3.5, 'y2': road_y - 0.5, 'y3': road_y - 0.5
                        },
                        'delay': 0.0
                    },
                    {
                        'vehicle_name': random.choice(vehicles),
                        'path_type': 'bezier',
                        'params': {
                            'speed': random.randint(22, 28),
                            'duration': 10.0,
                            'x0': -100, 'x1': -33, 'x2': 33, 'x3': 100,
                            'y0': road_y - 0.5, 'y1': road_y - 0.5, 'y2': road_y - 3.5, 'y3': road_y - 3.5
                        },
                        'delay': 2.0
                    }
                ]
            else:
                # Standard two directions
                v_configs = [
                    {
                        'vehicle_name': random.choice(vehicles), 
                        'path_type': 'straight',
                        'params': {'speed': random.randint(20, 30), 'duration': 10.0, 'offset': road_y - 1.5, 'direction': 1, 'road': 'horizontal'},
                        'delay': 0.0
                    },
                    {
                        'vehicle_name': random.choice(vehicles), 
                        'path_type': 'straight',
                        'params': {'speed': random.randint(20, 30), 'duration': 10.0, 'offset': road_y + 1.5, 'direction': -1, 'road': 'horizontal'},
                        'delay': random.uniform(1.0, 3.0)
                    }
                ]
            result = generate_multi_object_clip(v_configs, str(audio_dir), batch_name, i, config, custom_filename=custom_name, observer_pos=obs_pos, road_y_center=road_y)
            clips_metadata.append(result)
        else:
            vehicle = random.choice(vehicles)
            path_type = random.choice(paths)
            try:
                params = generate_random_parameters(config, vehicle, path_type)
                print(f"[{i}/{num_samples}] Generating {vehicle} on {path_type} path as '{custom_name}'...")
                
                result = generate_single_clip(vehicle, path_type, params, str(audio_dir), batch_name, i, config, custom_filename=custom_name)
                clips_metadata.append(result)
                scenes_for_validation.append((path_type, params, vehicle))
            except Exception as e:
                print(f"  [ERROR] Failed to generate sample {i}: {e}")

    # Save batch metadata
    with open(batch_dir / f"metadata_{batch_name}.json", 'w') as f:
        json.dump({'batch_id': batch_name, 'clips': clips_metadata, 'total_generated': len(clips_metadata), 'timestamp': datetime.now().isoformat()}, f, indent=2)

    # Physical Plausibility Check for single sources
    if scenes_for_validation:
        print("\n--- Evaluating Physical Plausibility ---")
        validation_results = validate_scene_paths(scenes_for_validation)
        val_json, val_txt = save_validation_report(validation_results, str(batch_dir), batch_name)
        print(f"Validation report saved to: {val_txt}")
        if validation_results['scene_valid']:
            print("[SUCCESS] All single-source paths are physically plausible.")
        else:
            print(f"[WARNING] {validation_results['vehicles_with_violations']} samples have path violations.")
    
    return batch_dir

def generate_labels(batch_outputs_dir, output_csv):
    """Aggregate metadata into a unified dataset.csv with B2, B7, B8 support."""
    batch_path = Path(batch_outputs_dir)
    samples = []
    
    print(f"Searching for metadata in {batch_path.absolute()}...")
    for batch_meta_file in batch_path.glob('**/metadata_*.json'):
        print(f"Found metadata file: {batch_meta_file}")
        try:
            with open(batch_meta_file, 'r', encoding='utf-8') as f:
                batch_meta = json.load(f)
            batch_id = batch_meta.get('batch_id', batch_meta_file.parent.name)
            for clip in batch_meta.get('clips', []):
                idx = clip.get('index', 0)
                sample_folder = f"sample_{idx:07d}"
                audio_rel_path = Path(batch_meta_file.parent) / 'audio_clips' / sample_folder / clip.get('filename', '')
                
                params = clip.get('parameters', {})
                labels = clip.get('labels', {})
                num_sources = labels.get('num_sources', clip.get('num_sources', 1))
                
                # B2 Direction Labeling
                angle = params.get('angle', 0)
                direction = "lateral"
                if angle < 45 or angle > 315: direction = "approaching"
                elif 135 < angle < 225: direction = "receding"
                
                filename = clip.get('filename', '')
                path_plot = clip.get('path_plot', '')
                spectrogram_plot = path_plot.replace('.png', '_spectrogram.png') if path_plot else ''
                
                audio_rel_path = Path(batch_meta_file.parent) / 'audio_clips' / sample_folder / filename
                
                samples.append({
                    'sample_id': sample_folder,
                    'batch_id': batch_id,
                    'speed_mps': labels.get('speed_mps', params.get('speed', 0)),
                    'acceleration_mps2': labels.get('acceleration_mps2', params.get('acceleration', 0.0)), # b7
                    'cpa_distance_m': labels.get('cpa_distance_m', params.get('distance', 0)),
                    'trajectory_type': labels.get('trajectory_type', clip.get('path_type')),
                    'direction_label': labels.get('direction_label', direction), # b2
                    'vehicle_class': labels.get('vehicle_class', clip.get('vehicle', 'multi' if num_sources > 1 else 'unknown')),
                    'num_sources': num_sources, # b8
                    'audio_path': audio_rel_path.as_posix(),
                    'path_plot': path_plot,
                    'spectrogram_plot': spectrogram_plot
                })
        except Exception as e:
            print(f"Warning: Could not process {batch_meta_file}: {e}")

    if not samples:
        print("Warning: No metadata samples found!")
        df = pd.DataFrame(columns=['sample_id', 'batch_id', 'speed_mps', 'acceleration_mps2', 'cpa_distance_m', 'trajectory_type', 'direction_label', 'vehicle_class', 'num_sources', 'audio_path', 'path_plot', 'spectrogram_plot'])
    else:
        df = pd.DataFrame(samples)
        
    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} with {len(samples)} samples.")
    return df

# evaluation core

def detect_cpa_labels(dataset_csv):
    """Detect CPA frames and add labels to dataset.csv."""
    df = pd.read_csv(dataset_csv)
    cpa_indices, cpa_times = [], []
    
    for idx, row in df.iterrows():
        sample_path = Path(row['audio_path']).parent
        dfdt_file, time_file = sample_path / 'dfdt.npy', sample_path / 'time.npy'
        
        if not dfdt_file.exists():
            cpa_indices.append(-1); cpa_times.append(-1.0); continue
            
        dfdt = np.load(dfdt_file)
        times = np.load(time_file)
        dfdt_smoothed = np.convolve(dfdt, np.ones(5)/5, mode='same')
        zero_crossings = np.where(np.diff(np.sign(dfdt_smoothed)))[0]
        
        cpa_idx = zero_crossings[len(zero_crossings)//2] if len(zero_crossings) > 0 else len(dfdt) // 2
        cpa_indices.append(cpa_idx)
        cpa_times.append(times[cpa_idx])
        
    df['cpa_frame_idx'] = cpa_indices
    df['cpa_time_sec'] = cpa_times
    df.to_csv(dataset_csv, index=False)
    print(f"Updated {dataset_csv} with CPA labels.")

def generate_stats_report(dataset_csv, output_report, figures_dir):
    """Generate distribution report and histograms."""
    df = pd.read_csv(dataset_csv)
    fig_path = Path(figures_dir); fig_path.mkdir(parents=True, exist_ok=True)
    
    report = [f"DopplerSim Dataset Report", "="*25, f"Total samples: {len(df)}"]
    for col in ['vehicle_class', 'trajectory_type']:
        report.append(f"\n{col.replace('_', ' ').title()}:")
        counts = df[col].value_counts()
        for val, count in counts.items():
            report.append(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
            
    with open(output_report, 'w') as f: f.write("\n".join(report))
    print(f"Report saved to {output_report}")

# evaluation scorers

def evaluate_classification(preds_df, gt_df, target_col):
    """Evaluate classification accuracy (B2, B4, B9, B10)."""
    merged = pd.merge(gt_df, preds_df, on='sample_id', suffixes=('_gt', '_pred'))
    y_true, y_pred = merged[f'{target_col}_gt'], merged[f'{target_col}_pred']
    acc = (y_true == y_pred).mean()
    print(f"Classification Results for {target_col}: Accuracy: {acc:.4f}")
    return acc

def evaluate_regression(preds_df, gt_df, target_col):
    """Evaluate regression MAE/RMSE (B1, B3, B5, B7)."""
    merged = pd.merge(gt_df, preds_df, on='sample_id', suffixes=('_gt', '_pred'))
    y_true, y_pred = merged[f'{target_col}_gt'], merged[f'{target_col}_pred']
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"Regression Results for {target_col}: MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

def evaluate_segmentation(dataset_csv):
    """Evaluate motion state segmentation accuracy (B6)."""
    df = pd.read_csv(dataset_csv)
    # Simple segmentation: Approach until CPA, then Recede
    print("\n--- B6: Motion State Segmentation Evaluation ---")
    correct_frames = 0
    total_frames = 0
    
    for _, row in df.iterrows():
        cpa_idx = row.get('cpa_frame_idx', -1)
        if cpa_idx == -1: continue
        
        # Mock prediction: assuming a perfect model for demonstration
        # In a real run, this would compare against a .npy mask or model output
        total_frames += 1 # Clipping to 1 for simplicity of report
        correct_frames += 1
        
    acc = correct_frames / total_frames if total_frames > 0 else 1.0
    print(f"Segmentation Mean IoU (Estimated): {acc:.4f}")
    return acc

def evaluate_multi_object(dataset_csv):
    """Evaluate multi-object disentanglement (B8)."""
    df = pd.read_csv(dataset_csv)
    multi_df = df[df['num_sources'] > 1]
    if multi_df.empty:
        print("\n[SKIP] B8: No multi-object samples found.")
        return 
        
    print("\n--- B8/B9: Multi-Object & Interaction Evaluation ---")
    # Identify objects correctly in a mixed signal
    acc = (multi_df['vehicle_class'] == 'multi').mean() # Mock logic
    print(f"Multi-Object Detection Rate: {acc:.4f}")
    return acc

# advanced benchmarks

def generate_car_manifest(dataset_csv, output_json, n_base=10):
    """Generate counterfactual pairs for CAR benchmark."""
    df = pd.read_csv(dataset_csv)
    base_samples = df.sample(n=min(len(df), n_base), random_state=42)
    manifest = []
    for _, row in base_samples.iterrows():
        for name, factor in [('speed_x2', 2.0), ('speed_x0.5', 0.5), ('dist_x2', 2.0), ('dist_x0.5', 0.5)]:
            config = {
                'vehicle': row['vehicle_class'],
                'path_type': row['trajectory_type'],
                'parameters': {
                    'speed': row['speed_mps'] * factor if 'speed' in name else row['speed_mps'],
                    'distance': row['cpa_distance_m'] * factor if 'dist' in name else row['cpa_distance_m'],
                    'angle': row.get('approach_angle_deg', 0)
                }
            }
            manifest.append({'base_id': row['sample_id'], 'type': name, 'config': config})
    with open(output_json, 'w') as f: json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} CAR pairs.")

def evaluate_car(preds_df):
    """Simple CAR accuracy."""
    if 'correct' in preds_df.columns:
        acc = preds_df['correct'].mean()
        print(f"CAR Accuracy: {acc:.4f}")

# main orchestrator

def main():
    parser = argparse.ArgumentParser(description="DopplerSim Initial Benchmark Suite (B1-B10)")
    parser.add_argument("--generate", action="store_true", help="Generate 5 test samples")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--force_crossing", action="store_true", help="Force vehicles to cross each other")
    parser.add_argument('--road_angle', type=float, default=0.0, help='Tilt the road by some angle (degrees)')
    parser.add_argument('--lane_width', type=float, default=4.0, help='Width of a single lane (meters)')
    
    args = parser.parse_args()
    
    # Target directory as requested
    benchmarks_dir = Path("benchmarks")
    test_output_dir = benchmarks_dir / "test_output"
    dataset_csv = test_output_dir / "dataset.csv"
    report_txt = test_output_dir / "dataset_summary.txt"
    figures_dir = test_output_dir / "figures"
    
    if args.generate:
        generate_test_batch(args.num_samples, force_crossing=args.force_crossing, road_angle=args.road_angle, lane_width=args.lane_width)
        df = generate_labels(test_output_dir, dataset_csv)
        detect_cpa_labels(dataset_csv)
    
    if not dataset_csv.exists():
        print(f"Error: {dataset_csv} not found. Run with --generate first.")
        return
        
    df = pd.read_csv(dataset_csv)
    generate_stats_report(dataset_csv, report_txt, figures_dir)
    
    print("\n--- Running DopplerSim Initial Suite (B1-B10) ---")
    
    # Mock evaluation for demonstration
    mock_preds = df.copy()
    mock_preds['speed_mps'] += np.random.normal(0, 0.5, len(df))
    mock_preds['acceleration_mps2'] += np.random.normal(0, 0.2, len(df))
    
    # B1, B3, B5, B7 (Regression)
    evaluate_regression(mock_preds, df, 'speed_mps')
    evaluate_regression(mock_preds, df, 'acceleration_mps2')
    evaluate_regression(mock_preds, df, 'cpa_distance_m')
    
    # B2, B4, B10 (Classification)
    evaluate_classification(df, df, 'trajectory_type')
    evaluate_classification(df, df, 'direction_label')
    evaluate_classification(df, df, 'vehicle_class')
    
    # B6, B8/B9 (Special)
    evaluate_segmentation(dataset_csv)
    evaluate_multi_object(dataset_csv)
    
    print("\nBenchmark Suite execution complete.")
    print(f"All outputs stored in: {test_output_dir.absolute()}")

if __name__ == "__main__":
    main()
