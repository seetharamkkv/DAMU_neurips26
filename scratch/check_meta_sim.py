import librosa
f = 'd:/Antigravity/vs13-model/ExtendedSimulatedData/KiaSportage/KiaSportage_100.wav'
y, sr = librosa.load(f, sr=None)
print(f"Duration: {len(y)/sr}s | SR: {sr} | Max Amp: {y.max()} | Min Amp: {y.min()}")
