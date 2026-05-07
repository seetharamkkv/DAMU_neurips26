# SE-ResNet Training Walkthrough (Experiment 2)

This guide provides step-by-step instructions for reproducing the training runs used in **Experiment 2 (SE-ResNet speed regression on VS13)** of the DopplerSim NeurIPS 2026 submission. The pipeline trains models on three distinct data conditions (`RealData`, `SimulatedData`, and `MixedData`) to isolate the effects of acoustic domain shift.

---

## 1. Prerequisites

### Environment Setup
Ensure you have the required dependencies installed:
```bash
cd Vehicle-Speed-from-Audio-SE-ResNet
pip install -r requirements.txt
```

### Dataset Structure
The experiment relies on three primary data conditions as described in the paper:
- **`RealData`**: Ground-truth field recordings (192 clips).
- **`SimulatedData`**: DopplerSim synthetic pass-bys matched in vehicle type and speed.
- **`MixedData`**: A combined dataset representing the union of `RealData` and `ExtendedSimulatedData` to provide broader kinematic coverage.

---

## 2. Dynamic Sample Rate Handling

The system automatically detects and applies the correct sample rate (SR) during training, adapting the spectrogram extraction window accordingly:
- **`RealData`**: 16,000 Hz
- **`MixedData` / `ExtendedSimulatedData`**: 22,050 Hz
- **`SimulatedData`**: 22,050 Hz

---

## 3. Training Commands

To reproduce the cross-domain models (or train on any new dataset), simply run the following command from the project root and point it to the relevant dataset folder:

```bash
python main.py --data_dir "../Datasets/<DatasetName>"
```

The training script automatically creates and targets a dataset-specific checkpoint folder (e.g., `checkpoints/RealData_model/`, `checkpoints/SimulatedData_model/`, or `checkpoints/MixedData_model/`). 

*(Note: To train the `MixedData_model`, ensure the target dataset directory contains the union of both `RealData` and `ExtendedSimulatedData` clips).*

---

## 4. Checkpoint & Output Management

Each training run performs **10-Fold Cross-Validation**, generating 10 best-weight files in Native Keras format (`fold_1_best.keras` ... `fold_10_best.keras`). 

---

## 5. Monitoring Progress

During training, the console provides real-time feedback:
1. **Config Update:** Confirmation of the sample rate being used.
2. **Global Stats:** Z-score normalization statistics computation (using only the training split).
3. **Live Progress:** Real-time metrics (Loss, RMSE) with Cosine Decay learning rate scheduling.

---

## 6. Inference Using Your New Models

To evaluate the cross-domain performance of your models on all datasets simultaneously, use the `batch_inference.py` script. See the [VS13 Evaluation Walkthrough](vs13_evaluation_walkthrough.md) for details on generating the 3x3 evaluation grid reported in the paper.
