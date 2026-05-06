# Doppler Effect Simulator (DopplerSim)

A Flask-based system for generating realistic Doppler-shifted vehicle audio clips for research purposes, dataset creation, and acoustic modeling experimentation. Supports batch, overlapping, and single-clip generation using straight-line, parabolic, and Bezier paths with physically accurate Doppler and distance attenuation.

---

## Features

### Core Capabilities
- **Realistic Doppler Shift**: Simulation using acoustic wave physics with sample-level resampling at SR = 44,100 Hz.
- **Multiple Vehicle Trajectories**:
  - **Straight Line**: Standard pass-by with configurable closest point of approach (CPA).
  - **Parabolic**: Curved path simulation.
  - **Bezier Curve**: Complex multi-point cubic trajectories.
- **Batch Overlap (Busy Road)**: Simulate multiple vehicles with staggered starts and lane offsets to create complex acoustic scenes.
- **Drone Support**: specialized support for drone sound libraries and flight dynamics.
- **Adaptive Physics**: Physically correct 1/R spherical spreading for distance-based amplitude shaping.
- **Automated Visualizations**: Generates path plots and spectrograms for every audio clip generated.

### UI Functionality
- **Multi-Mode Web Interface**:
  - **Batch Generation**: Large scale dataset creation with randomized or user-defined parameters.
  - **Batch Overlap**: Complex scene generation for multi-target tracking research.
  - **Spectrograms**: dedicated tool for analyzing sound files and generating visual high-resolution spectrograms.
  - **Single Clip**: Instant preview mode for parameter tuning.
- **Vehicle Management**: Library upload, validation (3.0s duration check), and live management.
- **Real-time Plotting**: Live preview of vehicle paths on a canvas.

---

## Project Structure

```
DopplerSim/
├── core/
│   ├── config.py            # Environment, dirs, and default parameter ranges
│   ├── progress.py          # Batch progress tracking
│   └── sampler.py           # Cyclic parameter sampling
├── physics/
│   ├── straight_line.py     # Straight line Doppler model logic
│   ├── parabola.py          # Parabolic path logic
│   └── bezier.py            # Bezier curve logic
├── audio/
│   ├── audio_utils.py       # Doppler, amplitude, and resampling utilities
│   └── generation.py        # Batch generation and distribution logic
├── visualization/
│   ├── graphs.py            # Shared plotting logic for paths and stats
│   ├── plot_utils.py        # Low-level trajectory plotting
│   └── validation.py        # Path validation and scene checks
├── routes/                  # Flask Route Blueprints
│   ├── batch_routes.py
│   ├── simulate_routes.py
│   └── vehicle_routes.py
├── static/                  # Audio, Metadata, and Image storage
├── templates/               # UI components
├── app_batch.py             # Main Flask entry point
├── requirements.txt         # Project dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip and virtual environment support

### Setup

```bash
# Enter the project directory
cd DopplerSim

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
python3 app_batch.py
```
The server will start at: `http://localhost:5050`

---

## Operational Modes

### 1. Batch Generation
Used to create large-scale datasets for machine learning.
1. Select path types (Multiple can be selected).
2. Choose sound source (Cars/Drones).
3. Set ranges for speed, distance, and angle.
4. Define total clips and distribution mode (Automatic or Manual).
5. **Output**: Folders containing WAVs, metadata.json, path plots, and spectrograms.

### 2. Batch Overlap (Busy Road Simulation)
Simulates realistic environments with multiple vehicles.
1. Define number of scenes.
2. Set range of vehicles per scene.
3. Configure lane width and maximum stagger (delay between vehicle starts).
4. **Output**: A "mixed_audio.wav" per scene along with individual vehicle tracks and a combined path plot.

### 3. Spectrogram Analysis
Analysis tool for sound libraries.
- Upload any audio file to generate a high-quality spectrogram.
- View and analyze the frequency distribution of vehicle sounds before generation.

### 4. Single Clip Generation
Instant simulator for testing specific parameters.
- Control every aspect of a single vehicle's path.
- Play and download results immediately.

### 5. Benchmark Suite (B1-B10)
DopplerSim introduces a suite of ten foundational tasks designed to evaluate whether models understand motion as a physical process:
- **B1: Speed Estimation**: Predict continuous velocity (mps) or discrete speed bins.
- **B2: Direction-of-Travel**: Classify relative motion (approaching, receding, lateral).
- **B3: Distance-of-Closest-Approach**: Estimate the minimum source-sensor distance (m) over a clip.
- **B4: Trajectory Shape**: Classify the trajectory family (straight, curved, circular, etc.).
- **B5: Time-to-Event**: Predict time remaining until closest approach, scene exit, or crossing event.
- **B6: Motion State Segmentation**: Perform framewise labeling of motion states (approaching, nearest point, receding, etc.).
- **B7: Acceleration / Deceleration**: Estimate acceleration magnitude or classify motion state (accelerating, decelerating, constant).
- **B8: Multi-Object Disentanglement**: Resolve multiple concurrent sources and estimate their individual attributes.
- **B9: Crossing and Interaction**: Classify scene-level interactions (crossing, overtaking, masking, convoy).
- **B10: Source Identity**: Identify source class or identity (e.g., vehicle model) invariant to motion condition.

**Usage**: Use the **Benchmark Mode** in the Batch Generation tab or run:
```bash
python benchmarks/benchmark_suite.py --generate --num_samples 5
```

---

## External Dataset: VS13

DopplerSim includes specialized support for the **VS13 Vehicle Speed Dataset**, a collection of recordings designed for research in acoustic vehicle speed estimation.

Although the original VS13 dataset contains recordings from 13 vehicle models, this work utilizes recordings from only the following 6 vehicles:
- **Kia Sportage**
- **Nissan Qashqai**
- **Peugeot 3008**
- **Peugeot 307**
- **Renault Scenic**
- **VW Passat B7**

Additional dataset details:
- **Coverage**: Vehicle recordings at speeds ranging from 30 km/h to 105 km/h.
- **Format**: `.wav` audio files with corresponding `.txt` ground-truth annotations containing speed and CPA timing information.

---

## Synthesis Pipeline

The DopplerSim engine follows a signal-flow oriented architecture that transforms stationary source recordings into physically consistent pass-by audio.

1. **Waveform Preparation**: Monophonic source recordings are resampled to the engine rate (22,050 Hz), peak-normalized, and extended with overlap crossfades if the requested duration exceeds the library length.
2. **Motion Computation**: The active trajectory model (Straight, Parabola, Bezier, or Map) determines the source position $\mathbf{p}(t)$, tangent velocity $\mathbf{v}(t)$, and radial velocity $v_r(t)$ at every output sample.
3. **Driving Curve Generation**: The engine evaluates the instantaneous frequency ratio $\rho(t)$ and a non-negative gain $g(t)$. The gain is built from softened geometric spreading and a convective level term.
4. **Retarded-Time Alignment**: For accelerated paths, ratio and gain sequences are re-interpolated onto an approximate arrival-time grid to align kinematic changes with the sound's travel time to the observer.
5. **Audio Synthesis**: The source audio is processed through a variable-rate resampler (time-domain warp) using cubic interpolation, then multiplied by the gain sequence.

---

## Physics & Kinematics

DopplerSim utilizes a physically-grounded synthesis engine to model the acoustic transformation of moving sources. The following sections detail the core mathematical framework.

### 1. Atmospheric Speed of Sound
The speed of sound $c$ is calculated based on ambient temperature $T$ (°C) and relative humidity $RH$ (%):
- **Dry Air Base**: $c_{dry}(T) = 331.3 \sqrt{1 + \frac{T}{273.15}}$
- **Humidity Correction**: $c(T, RH) = c_{dry}(T) + 0.6 \frac{RH}{100}$

### 2. Kinematic Path Modeling
Sources follow a planar curve $\mathbf{p}(t)$ with velocity $\mathbf{v}(t)$ tangent to the path.
- **Position Interpolation**: $\mathbf{p}(t) = (1 - \lambda(t)) \mathbf{q}_j + \lambda(t) \mathbf{q}_{j+1}$
- **Tangential Speed with Acceleration**: $s(\Delta t) = v_0 \Delta t + \frac{1}{2} a (\Delta t)^2$
- **Radial Velocity**: $v_r(t) = \frac{\mathbf{v}(t) \cdot (\mathbf{p}(t) - \mathbf{o})}{\|\mathbf{p}(t) - \mathbf{o}\|}$  
  *(where $\mathbf{o}$ is the observer position)*

### 3. Acoustic Wave Modeling
The received waveform $y(t)$ is generated by warping the source signal $s(t)$ and applying gain $g(t)$:

#### Doppler Warping (Frequency Ratio)
The emitted-to-received frequency ratio $\rho(t)$ is governed by the standard kinematic Doppler expression:
$$\rho(t) = \frac{f'(t)}{f_0} = \frac{c}{c + v_r(t)}$$

#### Gain and Attenuation
The raw gain $g_{raw}(t)$ combines geometric spreading and convective effects:
- **Geometric Spreading**: $A_{sp}(t) = \frac{1}{\sqrt{\|\mathbf{p}(t) - \mathbf{o}\|^2 + R_{nf}^2}}$  
  *(with near-field radius $R_{nf} = 6m$)*
- **Convective Factor**: $A_{conv}(t) = \left(\frac{c}{c + v_r(t)}\right)^{1.1}$
- **Total Gain**: $g_{raw}(t) = (G_0 A_{sp}(t) A_{conv}(t))^\gamma$  
  *(Default constants: $G_0 = 10$, $\gamma = 0.7$)*

### 4. Propagation & Timing
- **Retarded-Time Alignment**: When tangential acceleration is non-zero, emission ($t_{emit}$) and observation ($t_{obs}$) times are approximately related to align kinematic changes with sound travel time:
  $$t_{obs} \approx t_{emit} + \frac{r(t_{emit}) - r_{cpa}}{c}$$
- **Discrete Output**: The final resampled waveform is computed using cubic interpolation:
  $$y[n] = g[n] \tilde{x}[n]$$

---

## Publication Reference

This codebase supports the following research paper:

**Dynamic Audio Motion Understanding: Benchmarking Physical Motion Inference from Sound**  
*Submitted for NeurIPS 2026 (Evaluations and Datasets Track)*
