# Checkpoint Management and Fallback System

This document explains the organization of model checkpoints and the automated fallback logic implemented in the SE-ResNet pipeline.

## 1. Directory Structure

Checkpoints are organized within the `checkpoints/` directory to separate pretrained weights from dataset-specific trained models.

```text
checkpoints/
│
├── pre-trained_weights/          # Official SE-ResNet ensemble weights (.h5)
│   ├── fold_1_best_weights.weights.h5
│   └── ...
│
├── SimulatedData_model/          # Custom models trained on SimulatedData (.keras)
│   ├── fold_1_best.keras
│   └── ...
│
├── RealData_model/               # Custom models trained on RealData (.keras)
│   ├── fold_1_best.keras
│   └── ...
│
└── <DatasetName>_model/          # Automatically created for any new dataset
```

## 2. Automated Training Workflow

When you run `main.py`, the system automatically detects the dataset name from the `--data_dir` path and creates a corresponding subfolder.

- **Command:** `python main.py --data_dir ./path/to/MyDataset`
- **Result:** Checkpoints are saved in `checkpoints/MyDataset_model/` as `.keras` files.

## 3. Inference & Fallback Logic

The `inference.py` script follows a priority-based loading strategy to ensure the best available model is used for a given dataset.

### Priority Order:
1. **Dataset-Specific Model:** Checks if `checkpoints/<DatasetName>_model/` exists and contains `.keras` files.
2. **User-Specified Directory:** If the `--weights_dir` flag is used with a path other than the default `checkpoints`.
3. **Pretrained Fallback:** If no custom model is found, it automatically falls back to the official weights in `checkpoints/pre-trained_weights/`.

### Detection Example:
If you evaluate `SimulatedData`:
- It looks for `checkpoints/SimulatedData_model/`.
- If found, it prints: `[INFO] Using custom trained model found at: checkpoints/SimulatedData_model/`
- If NOT found, it prints:
  ```text
  [INFO] Custom trained model for 'SimulatedData' not found.
  [INFO] Falling back to pretrained weights at: checkpoints/pre-trained_weights/
  ```

## 4. How to Add New Datasets

1. Prepare your data following the naming convention (`Vehicle_Speed.wav`).
2. Run `python main.py --data_dir /path/to/new_dataset`.
3. The system will create `checkpoints/new_dataset_model/`.
4. Subsequent runs of `inference.py --data_dir /path/to/new_dataset` will automatically use your new model.

## 5. File Formats
- **.keras**: Modern Native Keras format used for all new training runs.
- **.h5**: Legacy/Pretrained weights format. The system is backward compatible with these files.
