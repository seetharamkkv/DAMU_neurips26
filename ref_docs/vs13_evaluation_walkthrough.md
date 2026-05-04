# Acoustic Vehicle Speed Estimation: VS13 Evaluation Walkthrough

## 1. Repository Overview
This repository implements a **Squeeze-and-Excitation Residual Network (SE-ResNet)** for estimating vehicle speeds from single-channel audio recordings. It uses an ensemble of 10 models (folds) to achieve high-precision results (target RMSE ~7.3 km/h).

## 2. Dataset Connection & Integration
The `vs13` dataset has been integrated into the evaluation pipeline with minimal configuration.

### Dataset Structure
The dataset follows the expected format:
- **Root Directory**: `RealData/`
- **Hierarchy**: Each subfolder (e.g., `KiaSportage`, `VWPassat`) contains audio clips for that specific vehicle.
- **Labeling**: Labels are extracted from the filename format `{Class}_{Speed}.wav`.
- **Discovery**: The `Train_valid_split.txt` file in each folder serves as the manifest for identifying valid audio clips.

### Configuration Changes
The following minimal changes were made to `inference.py` to support the `vs13` dataset and provide deeper insights:
1.  **Weight Path Alignment**: Updated the weight filename pattern to match the local naming convention (`fold_X_best_weights.weights.h5`).
2.  **Per-Vehicle Reporting**: Instrumented the inference loop to track performance per vehicle class, allowing for a granular accuracy breakdown.

## 3. Running Evaluation

### Prerequisites
It is assumed that a Virtual Environment (`venv`) has already been created and the requirements from `requirements.txt` have been installed.

#### Git Bash Activation
To use the existing environment in Git Bash, activate it using:
```bash
source venv/Scripts/activate
```

### Run Full Evaluation
Once the environment is active, run the following command from the `Vehicle-Speed-from-Audio-SE-ResNet` directory:
```bash
python inference.py --data_dir "../RealData" --weights_dir checkpoints/
```

### Run Small Verification Test
To verify the pipeline works on a subset (2 samples):
```bash
python inference.py --data_dir additional/data/test_data --weights_dir checkpoints/
```

> [!TIP]
> If you prefer not to activate the environment, you can run the script directly using the venv's python executable:
> ```bash
> ./venv/Scripts/python inference.py --data_dir "../RealData" --weights_dir checkpoints/
> ```

## 4. Additional Utilities
A new `additional/` folder has been created to store auxiliary scripts and data used during the setup:
- **`additional/scripts/`**: Contains verification scripts for data loading, audio processing, and weight validation.
- **`additional/data/test_data/`**: A minimal subset of the VS13 dataset (2 samples) for quick pipeline verification.
- **`additional/test_results/`**: Automatically stores evaluation reports in `.txt` format, including timestamps, dataset paths, and per-vehicle metrics.

## 5. Performance Metrics & Expected Outputs

### Metrics Explained
- **Global RMSE (Root Mean Square Error)**: The primary metric for regression accuracy. It measures the average magnitude of error in km/h. Lower is better.
- **Per-Vehicle RMSE**: Provides insight into how well the model generalizes across different engine sounds and vehicle profiles.
- **Latency**: Measures the average processing time per audio file (ms).

### Expected Output Format
```text
[INFO] Executing Ensemble Inference...
   -> Fold 01 Loaded | Processing Time: 1.42s
   ...
   -> Fold 10 Loaded | Processing Time: 1.38s

=============================================
ENSEMBLE EVALUATION COMPLETE
Total Samples:    398
Avg Latency/File: 140.50 ms (CPU)
Final RMSE:       7.2941 km/h
---------------------------------------------
Vehicle Class        | Samples  | RMSE (km/h)
---------------------------------------------
KiaSportage          | 66       | 6.8421
NissanQashqai        | 64       | 7.1234
...
=============================================
```

## 5. Evaluation Scope & Limitations

### What is evaluated?
- **Regression Accuracy**: Yes (RMSE).
- **Per-Vehicle Metrics**: Yes (Breaking down RMSE by subfolder).
- **Aggregate Performance**: Yes.

### What is NOT evaluated?
- **Classification Accuracy**: The model predicts continuous speed values, not discrete classes.
- **Confusion Matrices**: Not applicable to regression tasks.
- **MAE (Mean Absolute Error)**: Not currently in the default output, though easily added.

### Limitations
- **Fixed Input Duration**: The model expects 10-second clips (resampled to 22.05 kHz).
- **Hardware Dependency**: Inference time varies based on CPU/GPU availability.
- **Batching**: The current script processes the ensemble sequentially per file, which is robust but not optimized for maximum throughput.

## 6. Recommendations for Interpretation
- **Vehicle Bias**: If a specific vehicle shows a much higher RMSE, it may indicate a domain mismatch (e.g., specific engine frequency signatures not well-represented in the training data).
- **Environmental Noise**: The pretrained model is sensitive to background noise. High RMSE on real-world data vs. simulator data usually indicates noise interference.
