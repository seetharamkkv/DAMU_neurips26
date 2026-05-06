import os
import librosa
from src.utils import get_all_audio_paths_and_labels

data_dir = "../vs13"
try:
    paths, labels = get_all_audio_paths_and_labels(data_dir)
    print(f"Successfully loaded {len(paths)} samples.")
    if len(paths) > 0:
        path = paths[0]
        print(f"Attempting to load: {path}")
        audio, sr = librosa.load(path, sr=22050, duration=1.0)
        print(f"Successfully loaded audio. Shape: {audio.shape}, SR: {sr}")
except Exception as e:
    print(f"Error: {e}")
