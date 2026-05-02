# SE-ResNet Training Walkthrough

This guide provides step-by-step instructions for retraining the SE-ResNet model on **RealData**, **MatchedData**, and **SimulatedData**. The pipeline is designed to be fully automated, handling sample rate adjustments and checkpoint organization based on the dataset name.

---

## 1. Prerequisites

### Environment Setup
Ensure you have the required dependencies installed:
```bash
cd Vehicle-Speed-from-Audio-SE-ResNet
pip install -r requirements.txt
```

### Dataset Structure
Each dataset must follow the standard repository format:
- **Audio Files:** Named as `VehicleName_Speed.wav`.
- **Split Files:** Each vehicle subfolder must contain a `Train_valid_split.txt` file.

---

## 2. Dynamic Sample Rate Handling

The system automatically detects and applies the correct sample rate (SR) during training:
- **RealData:** 16,000 Hz
- **MatchedData:** 22,050 Hz
- **SimulatedData:** 22,050 Hz
- **Any Other Dataset:** 22,050 Hz (Default)

> [!NOTE]
> The input shape of the model (number of frames) will automatically scale based on the sample rate to maintain a consistent 10-second audio window.

---

## 3. Training Commands

Run the following commands from the `Vehicle-Speed-from-Audio-SE-ResNet` root directory.

### A. Training on RealData
```bash
python main.py --data_dir "D:/Antigravity/vs13-model/RealData"
```
- **Target:** `checkpoints/RealData_model/`
- **SR:** 16,000 Hz

### B. Training on MatchedData
```bash
python main.py --data_dir "D:/Antigravity/vs13-model/MatchedData"
```
- **Target:** `checkpoints/MatchedData_model/`
- **SR:** 22,050 Hz

### C. Training on SimulatedData
```bash
python main.py --data_dir "D:/Antigravity/vs13-model/SimulatedData"
```
- **Target:** `checkpoints/SimulatedData_model/`
- **SR:** 22,050 Hz

---

## 4. Checkpoint & Output Management

### Folders
After training starts, the system creates dataset-specific subfolders inside `checkpoints/`:
- `checkpoints/RealData_model/`
- `checkpoints/MatchedData_model/`
- `checkpoints/SimulatedData_model/`

### File Format
Each training run performs **10-Fold Cross-Validation**, generating 10 best-weight files in the Native Keras format:
- `fold_1_best.keras` ... `fold_10_best.keras`

---

## 5. Monitoring Progress

During training, the console will provide real-time feedback:
1. **Config Update:** Confirmation of the sample rate being used.
2. **Global Stats:** Progress of the Z-score normalization calculation.
3. **Live Progress Bar:** Each epoch now displays a **real-time progress bar** (`verbose=1`), showing:
   - Current Epoch and Batch progress.
   - Live Training Loss and RMSE.
   - ETA and step latency.
4. **Validation Metrics:** Displayed at the end of every epoch.
5. **Final Summary:** The mean ensemble RMSE across all 10 folds.

---

## 6. Inference Using Your New Models

To evaluate your newly trained models, use `inference.py`. The system will automatically prioritize your custom model over the pretrained weights.

**Example for MatchedData:**
```bash
python inference.py --data_dir "D:/Antigravity/vs13-model/MatchedData"
```
The console will report:
`[INFO] Using custom trained model found at: checkpoints/MatchedData_model/`

---

## 7. Customization (src/config.py)

If you need to tune the training, you can modify these values in `src/config.py`:
- `BATCH_SIZE`: Adjust based on GPU memory.
- `EPOCHS`: Default is 150.
- `PATIENCE`: Early stopping patience (default 30).
- `LEARNING_RATE`: Initial learning rate for the Cosine Decay scheduler.
