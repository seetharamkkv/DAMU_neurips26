import os
import re
import numpy as np
import librosa
import json
from .config import Config

def get_all_audio_paths_and_labels(data_root):
    """
    Recursively scans the data_root for directories containing 'Train_valid_split.txt'.
    This supports both flat (vs13) and nested (AdditionalExtendedSimulatedData) structures.
    """
    all_paths = []
    all_speeds = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(data_root):
        if 'Train_valid_split.txt' in files:
            vehicle_path = root
            split_file_path = os.path.join(vehicle_path, 'Train_valid_split.txt')

            with open(split_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # format: filename split_type (e.g., Mazda3_50 train)
                        base_name = parts[0]
                        wav_file = os.path.join(vehicle_path, base_name + '.wav')
                        
                        if not os.path.exists(wav_file): 
                            continue
                        
                        # Extract speed from filename (e.g., Class_50.wav -> 50 or car_1_100.0.wav -> 100.0)
                        # We look for the speed after the LAST underscore
                        match = re.match(r'(.+)_([\d.]+)\.wav', os.path.basename(wav_file))
                        if match:
                            speed = float(match.group(2))
                            all_paths.append(wav_file)
                            all_speeds.append(speed)

    return all_paths, np.array(all_speeds)

def calculate_global_stats(audio_paths, save_path=None):
    """
    Calculates mean and std of Mel Spectrograms across the dataset 
    for Z-score normalization.
    """
    print(f"Calculating stats for {len(audio_paths)} files...")
    mel_sums = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    mel_sum_sqs = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    total_frames = 0

    for i, path in enumerate(audio_paths):
        if i % 50 == 0: print(f"Processing {i}/{len(audio_paths)}...")
        try:
            audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
            # Pad/Crop
            if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                audio = audio[:Config.AUDIO_LENGTH_SAMPLES]
            else:
                audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), 'constant')
            
            mel = librosa.feature.melspectrogram(
                y=audio, sr=Config.SAMPLE_RATE, 
                n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            mel_sums += np.sum(mel_db, axis=1, keepdims=True)
            mel_sum_sqs += np.sum(mel_db**2, axis=1, keepdims=True)
            total_frames += mel_db.shape[1]
        except Exception as e:
            print(f"Error processing {path}: {e}")

    mel_mean = (mel_sums / total_frames).astype(np.float32)
    mel_std = np.sqrt(mel_sum_sqs / total_frames - mel_mean**2).astype(np.float32)
    mel_std[mel_std < 1e-8] = 1e-8 # Prevent divide by zero

    stats = {
        "mean": mel_mean.tolist(),
        "std": mel_std.tolist()
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(stats, f)
            
    return stats