import os
checkpoint_dir = "checkpoints"
weights = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
print(f"Found {len(weights)} weight files.")
for w in sorted(weights):
    size = os.path.getsize(os.path.join(checkpoint_dir, w))
    print(f"  {w}: {size / (1024*1024):.2f} MB")
