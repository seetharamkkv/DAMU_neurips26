import argparse
import os
import sys
from src.config import Config
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats
from src.train_engine import run_cross_validation

def main():
    parser = argparse.ArgumentParser(description="Train SE-ResNet for Vehicle Speed Estimation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the VS13 dataset root directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        sys.exit(1)
        
    print(f"Scanning dataset at {args.data_dir}...")
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    Config.update_for_dataset(dataset_name)
    
    paths, speeds = get_all_audio_paths_and_labels(args.data_dir)
    
    if len(paths) == 0:
        print("Error: No audio files found. Check directory structure.")
        sys.exit(1)
        
    print(f"Found {len(paths)} samples.")
    
    # Calculate stats for Z-score (Critical step from paper)
    stats = calculate_global_stats(paths)
    
    # Run Training with dataset-specific checkpointing
    run_cross_validation(paths, speeds, stats, dataset_name=dataset_name)

if __name__ == "__main__":
    main()