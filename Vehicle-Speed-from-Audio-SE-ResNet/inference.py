import argparse
import os
import time
import numpy as np

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.config import Config
from src.data_loader import get_tf_dataset
from src.models import build_se_resnet
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats

def run_ensemble_inference(data_dir, weights_dir, args):
    # 1. data ingestion
    print(f"[INFO] Scanning dataset at {data_dir}...")
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    Config.update_for_dataset(dataset_name)
    
    paths, speeds = get_all_audio_paths_and_labels(data_dir)
    
    if len(paths) == 0:
        raise ValueError("Dataset empty or path incorrect.")

    print(f"[INFO] Processing {len(paths)} files...")
    stats = calculate_global_stats(paths)
    
    # create optimized dataset (parallel prefetch)
    ds = get_tf_dataset(paths, speeds, stats, is_training=False)
    
    # 2. architecture setup
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    # 3. ensemble loop
    fold_predictions = []
    fold_times = []
    found_weights = False

    # Interactive Model Selection
    print("\n" + "="*45)
    print("MODEL SELECTION")
    print("="*45)
    print("Select the model to run inference with:")
    print("1. Pretrained Model")
    print("2. MatchedData Model")
    print("3. SimulatedData Model (Work In Progress)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3) [default: 1]: ").strip()
        except EOFError:
            choice = '1'
            
        if not choice or choice == '1':
            current_weights_dir = Config.PRETRAINED_DIR
            ext = "_weights.weights.h5"
            print(f"[INFO] Using Pretrained Model at: {current_weights_dir}")
            break
        elif choice == '2':
            current_weights_dir = os.path.join("checkpoints", "MatchedData_model")
            ext = ".keras"
            print(f"[INFO] Using MatchedData Model at: {current_weights_dir}")
            break
        elif choice == '3':
            current_weights_dir = os.path.join("checkpoints", "SimulatedData_model")
            ext = ".keras"
            print(f"[INFO] Using SimulatedData Model at: {current_weights_dir}")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")

    for fold in range(1, Config.N_FOLDS + 1):
        # Match filenames: fold_1_best.keras OR fold_1_best_weights.weights.h5
        if ext == ".keras":
            weight_path = os.path.join(current_weights_dir, f"fold_{fold}_best{ext}")
        else:
            weight_path = os.path.join(current_weights_dir, f"fold_{fold}_best{ext}")
        
        if not os.path.exists(weight_path):
            if fold == 1: # Only warn once for missing weights
                print(f"[WARN] Weight file missing: {weight_path}")
            continue
            
        found_weights = True
        start_fold = time.perf_counter()
        
        # build architecture and load weights
        model = build_se_resnet(input_shape)
        model.load_weights(weight_path)
        
        # execution
        preds = model.predict(ds, verbose=0)
        fold_predictions.append(preds.flatten())
        
        fold_duration = time.perf_counter() - start_fold
        fold_times.append(fold_duration)
        
        print(f"   -> Fold {fold:02d} Loaded | Processing Time: {fold_duration:.2f}s")
        
        # manual gc and session clear to prevent O(N) memory leak
        del model
        tf.keras.backend.clear_session()

    if not found_weights:
        raise FileNotFoundError(f"Checkpoints not found in {weights_dir}")

    # 4. aggregation and latency metrics
    fold_predictions = np.array(fold_predictions)
    ensemble_preds = np.mean(fold_predictions, axis=0)
    
    total_inf_time = sum(fold_times)
    avg_per_sample_ms = (total_inf_time / (len(speeds) * len(fold_times))) * 1000
    
    # 5. metrics calculation
    rmse = np.sqrt(np.mean((speeds - ensemble_preds) ** 2))
    mae = np.mean(np.abs(speeds - ensemble_preds))
    
    # Per-vehicle analysis
    vehicle_metrics = {}
    
    # Extract vehicle classes from paths
    for i, path in enumerate(paths):
        vehicle_class = os.path.basename(os.path.dirname(path))
        if vehicle_class not in vehicle_metrics:
            vehicle_metrics[vehicle_class] = {'gt': [], 'pred': []}
        vehicle_metrics[vehicle_class]['gt'].append(speeds[i])
        vehicle_metrics[vehicle_class]['pred'].append(ensemble_preds[i])

    # Prepare report string
    report_lines = []
    report_lines.append("="*45)
    report_lines.append("ENSEMBLE EVALUATION REPORT")
    report_lines.append(f"Timestamp:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset Path:     {os.path.abspath(data_dir)}")
    report_lines.append(f"Weights Path:     {os.path.abspath(current_weights_dir)}")
    report_lines.append("-" * 45)
    report_lines.append(f"Total Samples:    {len(speeds)}")
    report_lines.append(f"Avg Latency/File: {avg_per_sample_ms:.2f} ms (CPU)")
    report_lines.append(f"Final RMSE:       {rmse:.4f} km/h")
    report_lines.append(f"Final MAE:        {mae:.4f} km/h")
    report_lines.append("-" * 45)
    report_lines.append(f"{'Vehicle Class':<20} | {'Samples':<8} | {'RMSE (km/h)':<10}")
    report_lines.append("-" * 45)
    
    for v_class, data in vehicle_metrics.items():
        v_rmse = np.sqrt(np.mean((np.array(data['gt']) - np.array(data['pred'])) ** 2))
        report_lines.append(f"{v_class:<20} | {len(data['gt']):<8} | {v_rmse:.4f}")
    report_lines.append("="*45 + "\n")

    report_str = "\n".join(report_lines)
    print(report_str)

    # Save to file if requested or by default in additional/test_results
    if hasattr(args, 'save_results') and args.save_results:
        results_dir = os.path.join("additional", "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename based on dataset name
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        filename = f"eval_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report_str)
        print(f"[INFO] Results saved to: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SE-ResNet Ensemble Inference")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weights_dir', type=str, default='checkpoints')
    parser.add_argument('--save_results', action='store_true', default=True, help="Save results to additional/test_results")
    
    args = parser.parse_args()
    run_ensemble_inference(args.data_dir, args.weights_dir, args)
