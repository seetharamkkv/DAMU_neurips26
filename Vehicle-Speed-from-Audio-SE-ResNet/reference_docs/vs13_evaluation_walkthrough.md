# Acoustic Vehicle Speed Estimation: VS13 Evaluation Walkthrough

## 1. Repository Overview
This repository implements an **SE-ResNet** validation baseline for estimating vehicle speeds from single-channel audio recordings. This walkthrough provides the instructions necessary to generate the 3x3 cross-dataset inference grid.

## 2. Dataset Connection & Integration
The evaluation pipeline automatically processes three data conditions (`RealData`, `SimulatedData`, and `MixedData`) against three model checkpoint families.

### Cross-Domain Grid Evaluator
We use `batch_inference.py` to evaluate across datasets and models. This script:
1.  **Iterates all Data Conditions**: Automatically loads and computes normalization stats for `RealData`, `SimulatedData`, and `MixedData`.
2.  **Iterates all Models**: Automatically loads the corresponding 10-fold ensemble checkpoints (`RealData_model`, `SimulatedData_model`, `MixedData_model`).
3.  **Generates 3x3 Grids**: Produces per-vehicle and aggregate matrices for both **RMSE** and **MAE**.

## 3. Running the Evaluation

### Prerequisites
Activate your Virtual Environment:
```bash
source venv/Scripts/activate
```

### Run Batch Evaluation
Run the following command from the `Vehicle-Speed-from-Audio-SE-ResNet` directory to generate the full cross-domain report:
```bash
python batch_inference.py
```
*(By default, this looks for datasets in `../Datasets/` and checkpoints in `checkpoints/`).*

### Run Single-Model Interactive Evaluation
If you only want to test a single model/dataset pair, you can use the interactive script:
```bash
python inference.py --data_dir "../Datasets/RealData" --weights_dir checkpoints/
```

## 4. Output Generation

### Output Formats
- **Console Output**: The script will output the resulting 3x3 grids for RMSE and MAE directly to the console.
- **Detailed Logs**: A breakdown by individual vehicle class is also provided to allow for granular inspection.
