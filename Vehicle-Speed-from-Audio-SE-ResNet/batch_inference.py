import os
import time
import numpy as np
import tensorflow as tf
from src.config import Config
from src.data_loader import get_tf_dataset
from src.models import build_se_resnet
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats

# ==============================================================================
# USER CONFIGURATION (HARDCODED PATHS)
# ==============================================================================
BASE_DIR = r"D:\Antigravity\vs13-model"
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "Vehicle-Speed-from-Audio-SE-ResNet", "checkpoints")

# Datasets to evaluate
DATASET_PATHS = {
    "RealData": os.path.join(BASE_DIR, "RealData"),
    "SimulatedData": os.path.join(BASE_DIR, "SimulatedData"),
    "MixedData": [os.path.join(BASE_DIR, "RealData"), os.path.join(BASE_DIR, "SimulatedData")]
}

# Models to evaluate
MODEL_DIRS = {
    "Mixed_model": os.path.join(CHECKPOINT_ROOT, "Mixed_model"),
    "SimulatedData_model": os.path.join(CHECKPOINT_ROOT, "SimulatedData_model"),
    "RealData_model": os.path.join(CHECKPOINT_ROOT, "RealData_model")
}

# Output Directory
RESULTS_DIR = os.path.join(BASE_DIR, "Vehicle-Speed-from-Audio-SE-ResNet", "additional", "test_results")
# ==============================================================================

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

def run_single_inference(data_dirs, model_dir, model_name):
    """Runs the full 10-fold ensemble inference on a dataset using a specific model."""
    # Ensure data_dirs is a list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
        
    # 1. Update Config based on model training domain
    if "SimulatedData" in model_name and "Extended" not in model_name:
        model_source = "SimulatedData"
    elif "Extended" in model_name:
        model_source = "ExtendedSimulatedData"
    elif "Mixed" in model_name:
        model_source = "MixedData"
    else:
        model_source = "RealData"
        
    Config.update_for_dataset(model_source)
    
    # 2. Scan dataset (potentially multiple directories)
    all_paths = []
    all_speeds = []
    for d_dir in data_dirs:
        paths, speeds = get_all_audio_paths_and_labels(d_dir)
        all_paths.extend(paths)
        all_speeds.extend(speeds)
    
    if len(all_paths) == 0:
        return None, None, {}
    
    speeds = np.array(all_speeds)

    # 3. Calculate Stats & Build Dataset
    stats = calculate_global_stats(all_paths)
    ds = get_tf_dataset(all_paths, speeds, stats, is_training=False)
    
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    # 4. Ensemble Loop
    fold_predictions = []
    ext = ".keras" # Custom models use .keras
    
    for fold in range(1, Config.N_FOLDS + 1):
        weight_path = os.path.join(model_dir, f"fold_{fold}_best{ext}")
        if not os.path.exists(weight_path):
            continue
            
        model = build_se_resnet(input_shape)
        model.load_weights(weight_path)
        preds = model.predict(ds, verbose=0)
        fold_predictions.append(preds.flatten())
        
        del model
        tf.keras.backend.clear_session()

    if not fold_predictions:
        return None, None, {}

    # 5. Aggregate Results
    ensemble_preds = np.mean(fold_predictions, axis=0)
    rmse = np.sqrt(np.mean((speeds - ensemble_preds) ** 2))
    mae = np.mean(np.abs(speeds - ensemble_preds))
    
    # 6. Per-Vehicle breakdown
    vehicle_metrics = {}
    for i, path in enumerate(all_paths):
        # Extract vehicle class from folder name
        v_class = os.path.basename(os.path.dirname(path))
        if v_class not in vehicle_metrics:
            vehicle_metrics[v_class] = {'gt': [], 'pred': []}
        vehicle_metrics[v_class]['gt'].append(speeds[i])
        vehicle_metrics[v_class]['pred'].append(ensemble_preds[i])
        
    v_results = {}
    for v_class, data in vehicle_metrics.items():
        v_rmse = np.sqrt(np.mean((np.array(data['gt']) - np.array(data['pred'])) ** 2))
        v_mae = np.mean(np.abs(np.array(data['gt']) - np.array(data['pred'])))
        v_results[v_class] = {'rmse': v_rmse, 'mae': v_mae}
        
    return rmse, mae, v_results

def format_grid(data_dict, models, datasets, metric_key='rmse'):
    """Formats a 3x3 grid as a string."""
    header = f"{'Source Dataset':<25} | " + " | ".join([f"{m:<20}" for m in models])
    lines = [header, "-" * len(header)]
    
    for d in datasets:
        row = f"{d:<25} | "
        row_vals = []
        for m in models:
            val = data_dict[m][d]
            if val is None:
                row_vals.append(f"{'N/A':<20}")
            elif isinstance(val, dict):
                v = val.get(metric_key)
                row_vals.append(f"{v:<20.4f}" if v is not None else f"{'N/A':<20}")
            else:
                row_vals.append(f"{val:<20.4f}")
        row += " | ".join(row_vals)
        lines.append(row)
    
    # Add Averages
    avg_row = f"{'Average (per column)':<25} | "
    avg_vals = []
    for m in models:
        col_vals = []
        for d in datasets:
            val = data_dict[m][d]
            if val is None:
                continue
            if isinstance(val, dict):
                v = val.get(metric_key)
                if v is not None:
                    col_vals.append(v)
            else:
                col_vals.append(val)
        
        if col_vals:
            avg_vals.append(f"{np.mean(col_vals):<20.4f}")
        else:
            avg_vals.append(f"{'N/A':<20}")
            
    avg_row += " | ".join(avg_vals)
    lines.append("-" * len(header))
    lines.append(avg_row)
    
    return "\n".join(lines)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    models = list(MODEL_DIRS.keys())
    datasets = list(DATASET_PATHS.keys())
    
    # Store results: results[model][dataset] = {rmse: X, mae: Y, per_car: {car: {rmse: X, mae: Y}}}
    results = {m: {d: None for d in datasets} for m in models}
    
    print("[START] Beginning 3x3 Batch Inference Grid...")
    
    for m_name, m_dir in MODEL_DIRS.items():
        for d_name, d_dir in DATASET_PATHS.items():
            print(f"\n[EXEC] Model: {m_name} | Dataset: {d_name}")
            rmse, mae, v_results = run_single_inference(d_dir, m_dir, m_name)
            
            results[m_name][d_name] = {
                'rmse': rmse,
                'mae': mae,
                'v_results': v_results
            }

    # 1. Save average_results.txt
    avg_file = os.path.join(RESULTS_DIR, "average_results.txt")
    with open(avg_file, 'w') as f:
        f.write("=== 3x3 CROSS-EVALUATION GRID: OVERALL AVERAGES ===\n\n")
        f.write("RMSE (km/h) GRID\n")
        f.write(format_grid(results, models, datasets, 'rmse'))
        f.write("\n\n")
        f.write("MAE (km/h) GRID\n")
        f.write(format_grid(results, models, datasets, 'mae'))
    
    print(f"\n[INFO] Overall averages saved to: {avg_file}")

    # 2. Save per_car_results.txt
    car_file = os.path.join(RESULTS_DIR, "per_car_results.txt")
    # Identify all car classes across all runs
    all_cars = set()
    for m in models:
        for d in datasets:
            if results[m][d] and results[m][d]['v_results']:
                all_cars.update(results[m][d]['v_results'].keys())
    all_cars = sorted(list(all_cars))

    with open(car_file, 'w') as f:
        f.write("=== 3x3 CROSS-EVALUATION GRID: PER-CAR RESULTS ===\n\n")
        for car in all_cars:
            f.write(f"VEHICLE CLASS: {car}\n")
            f.write("-" * 30 + "\n")
            
            # Create a temporary results dict for this car to reuse format_grid
            car_results = {m: {d: results[m][d]['v_results'].get(car, {'rmse': 0, 'mae': 0}) for d in datasets} for m in models}
            
            f.write("RMSE (km/h)\n")
            f.write(format_grid(car_results, models, datasets, 'rmse'))
            f.write("\n\n")
            f.write("MAE (km/h)\n")
            f.write(format_grid(car_results, models, datasets, 'mae'))
            f.write("\n\n" + "="*80 + "\n\n")

    print(f"[INFO] Per-car results saved to: {car_file}")
    print("\n[DONE] Batch inference complete.")

if __name__ == "__main__":
    main()
