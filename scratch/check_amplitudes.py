import librosa
import numpy as np
import os

def check_amplitude(directory, n_samples=5):
    print(f"Checking directory: {directory}")
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.wav'):
                files.append(os.path.join(root, f))
    
    if not files:
        print("No wav files found.")
        return

    selected = files[:n_samples]
    for f in selected:
        try:
            y, sr = librosa.load(f, sr=None)
            peak = np.max(np.abs(y))
            mean_abs = np.mean(np.abs(y))
            print(f"File: {os.path.basename(f)} | Peak Amp: {peak:.6f} | Mean Abs Amp: {mean_abs:.6f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

print("--- RealData Amplitude ---")
check_amplitude('d:/Antigravity/vs13-model/RealData')
print("\n--- ExtendedSimulatedData Amplitude ---")
check_amplitude('d:/Antigravity/vs13-model/ExtendedSimulatedData')
print("\n--- ExtendedSimulatedData Amplitude ---")
check_amplitude('d:/Antigravity/vs13-model/ExtendedSimulatedData')
