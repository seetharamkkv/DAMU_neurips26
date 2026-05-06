import os

# Thread environment variables (must be set before numpy imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# directory structure

UPLOAD_FOLDER = 'static/vehicle_sounds'
DRONE_SOUNDS_FOLDER = 'static/drone_sounds'
OUTPUT_FOLDER = 'static/batch_outputs'
SINGLE_OUTPUT_FOLDER = 'static/single_outputs'
SPECTROGRAM_FOLDER = os.path.join('static', 'spectrograms')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRONE_SOUNDS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SINGLE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# persistence files

SAMPLER_STATE_FILE = "sampler_state.json"
PROGRESS_FILE = "generation_progress.json"

# default ranges for randomization
# NOTE: enforce constraints:
# - parabola_a is positive so it opens upwards
# - Bezier y will be sampled positive in code
# - angle limited to [-45, 45]

DEFAULT_RANGES = {
    'speed': {
        'car': (1, 50),
        'train': (1, 55),
        'drone': (1, 30),
        'motorcycle': (1, 45),
        'default': (1, 50)
    },
    'distance': (0.5, 1000),
    # duration range is now unused (we force 10s everywhere),
    # but we keep it for reference
    'duration': (10, 10),
    'angle': (-45, 45),
    # for Bezier: allow negative x, but y will be kept positive in code
    'bezier_coords': (-150, 150),
    # positive only so parabola opens towards +y
    'parabola_a': (5, 20),       # will be divided by 10000 => 0.0005 to 0.0020
    'parabola_h': (10, 50),       # always positive height
    # Atmospheric effects
    'temperature': (-20, 50),     # Celsius
    'humidity': (0, 100)          # Relative humidity (%)
}
