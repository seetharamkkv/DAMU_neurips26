# DopplerNet-Validation-Benchmark

A comprehensive validation and retraining suite for acoustic vehicle speed estimation models. This repository integrates real-world acoustic data with high-fidelity simulations from DopplerNet to benchmark and refine the SE-ResNet speed estimation architecture.

## Core Repositories
- **Model Architecture:** [Vehicle-Speed-from-Audio-SE-ResNet](https://github.com/vafaeim/Vehicle-Speed-from-Audio-SE-ResNet)
- **Kinematics Engine:** [DopplerNet Simulator](https://github.com/rohitharumugams/DopplerNet)

---

## Repository Structure

- **`Vehicle-Speed-from-Audio-SE-ResNet/`**: The core deep learning codebase (SE-ResNet).
- **`RealData/`**: The VS13 benchmark dataset (16kHz).
- **`ExtendedSimulatedData/`**: Purely synthetic audio generated via the DopplerNet physics engine (22.05kHz).
- **`SimulatedData/`**: Hybrid dataset aligning simulation parameters with real-world VS13 metadata (22.05kHz).
- **`AdditionalExtendedSimulatedData/`**: Auxiliary datasets for stress-testing and edge-case validation.
- **`ref_docs/`**: Detailed technical documentation, training walkthroughs, and evaluation reports.

---

## Quick Start

### 1. Installation
```bash
# Enter the model directory
cd Vehicle-Speed-from-Audio-SE-ResNet

# Install dependencies
pip install -r requirements.txt
```

### 2. Training (Retraining from Scratch)
The pipeline automatically handles dataset-specific sample rates (16kHz for RealData, 22.05kHz for others) and organizes checkpoints.

```bash
# Train on ExtendedSimulatedData
python main.py --data_dir "../ExtendedSimulatedData"

# Train on RealData
python main.py --data_dir "../RealData"
```
*Checkpoints are saved to `checkpoints/<dataset>_model/`.*

### 3. Evaluation (Inference)
The inference engine includes a smart fallback system: it uses custom-trained models if available, otherwise defaults to the pretrained ensemble weights.

```bash
# Evaluate the model on ExtendedSimulatedData
python inference.py --data_dir "../ExtendedSimulatedData"
```

---

## Features & Improvements

- **Dynamic Sample Rate Logic:** Automatically switches processing parameters based on the source dataset (Real vs. Simulated).
- **Checkpoint Management:** Isolated subfolders for each training run to prevent weight overwriting.
- **Ensemble Inference:** 10-fold cross-validation engine for high-precision speed estimation.
- **Advanced Normalization:** Global Z-score normalization calculated per dataset to handle varying acoustic environments.

---

## Documentation
For more detailed information, please refer to the `ref_docs/` folder:
- [Training Walkthrough](ref_docs/training_walkthrough.md)
- [Checkpoint Management & Fallback](ref_docs/checkpoint_management.md)
- [VS13 Evaluation Guide](ref_docs/vs13_evaluation_walkthrough.md)

---

## Authors
**Seetharam Killivalavan** & **Rohith Arumugam Suresh**  
School of Computer Science <br>
*Carnegie Mellon University*

## Acknowledgments
Carnegie Mellon University, Language Technologies Institute  
Bradley Warren and Professor Bhiksha Raj for research guidance.
