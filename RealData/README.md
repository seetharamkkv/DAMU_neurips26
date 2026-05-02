# VS13 Vehicle Speed Dataset

## Overview
The **VS13 Dataset** is a specialized collection of audio-video recordings designed for research in **vehicle speed estimation**. It contains 400 high-definition video recordings of 13 different vehicle models passing a stationary camera at constant speeds, ranging from **30 km/h to 105 km/h**.

This dataset is particularly valuable for developing and benchmarking algorithms based on the **Doppler Effect** and acoustic signal processing.

- **Original Source:** Djukanović, S. et al. *"A dataset for audio-video based vehicle speed estimation."*
- **Kaggle Mirror:** [VS13 Dataset on Kaggle](https://www.kaggle.com/datasets/vafaeii/vs13-dataset)
- **Reference Paper:** [Download PDF](https://slobodan.ucg.ac.me/science/vs13/paper.pdf)

---

## Dataset Structure

The dataset is organized by vehicle model. Each folder contains:
- **`.wav` files:** Audio extracted from the original MP4 recordings (Full HD, 30 fps, 10s length).
- **`.txt` files:** Annotation files containing ground-truth speed and pass-by timing.

### Supported Vehicles (13 Models)
- Citroen C4 Picasso
- Kia Sportage
- Mazda 3 Skyactive
- Mercedes AMG 550
- Mercedes GLA 200D
- Nissan Qashqai
- Opel Insignia
- Peugeot 208, 3008, 307
- Renault Captur, Scenic
- VW Passat B7

---

## Annotation Format
Each recording has a corresponding `.txt` file with two space-separated values:

```text
<speed_kmh> <t_passby_s>
```

| Parameter | Unit | Description |
| :--- | :--- | :--- |
| **`speed_kmh`** | km/h | The constant speed maintained by the vehicle (via cruise control). |
| **`t_passby_s`** | seconds | The relative time from the start of the file identifying the **Closest Point of Approach (CPA)**. |

> [!IMPORTANT]
> The second value in the annotation file represents **Time in seconds**, not distance. This was determined by visual screening of the frame where the vehicle begins to exit the camera view.

**Example (`CitroenC4Picasso_101.txt`):**
```text
101.0 7.49
```
*Meaning: The vehicle was traveling at 101.0 km/h and passed the camera at 7.49 seconds into the clip.*

---

## Recording Setup
- **Camera:** GoPro Hero5 Session.
- **Position:** Installed by the road at a distance of approx. **0.5 m** and a height of **1.2 m**.
- **Variety:** Recordings were taken from both sides of the road and at different angles to ensure robustness.
- **Consistency:** All vehicles used on-board cruise control to maintain stable speeds.

---

## Reference
For more details about the dataset, methodology, and experimental setup, refer to the original paper:
> Mišić, S., Bjelica, M. *Vehicle Speed Estimation Based on the Doppler Effect of Audio Signals*.  
> Available at: https://slobodan.ucg.ac.me/science/vs13/paper.pdf
The paper contains the complete author list, publication details, and technical background associated with this dataset.

---

## Notes
- This dataset is an unofficial mirror hosted on Kaggle. The authoritative source and associated paper are linked above.
- The dataset contains 13 samples and is suited for research and prototyping. Data augmentation may be necessary for large-scale training tasks.