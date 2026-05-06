import os
from src.utils import get_all_audio_paths_and_labels

data_dir = "../vs13"
try:
    paths, labels = get_all_audio_paths_and_labels(data_dir)
    print(f"Successfully loaded {len(paths)} samples.")
    if len(paths) > 0:
        print(f"Example path: {paths[0]}")
        print(f"Example label: {labels[0]}")
except Exception as e:
    print(f"Error: {e}")
