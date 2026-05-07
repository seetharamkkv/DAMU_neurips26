"""
DopplerSim – Flask entry point.

All business logic lives in the modules below.
This file only sets up the Flask app, registers Blueprints, and provides
the home route and server startup block.
"""

# config.py sets OMP/OPENBLAS/MKL env vars and creates all required directories.
import core.config  # noqa: F401 – side-effects (env vars + makedirs)

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os

from routes.vehicle_routes import vehicle_bp
from routes.batch_routes import batch_bp
from routes.simulate_routes import simulate_bp
from routes.mixed_routes import mixed_bp
from routes.vs13_routes import vs13_bp
from routes.auto_compare_routes import auto_compare_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(vehicle_bp)
app.register_blueprint(batch_bp)
app.register_blueprint(simulate_bp)
app.register_blueprint(mixed_bp)
app.register_blueprint(vs13_bp)
app.register_blueprint(auto_compare_bp)

# Serve MapExtraction outputs via a clean relative route
@app.route('/map_outputs/<path:filename>')
def serve_map_outputs(filename):
    outputs_dir = os.path.join(os.getcwd(), 'MapExtraction', 'outputs')
    return send_from_directory(outputs_dir, filename)


@app.route('/')
def home():
    return render_template('index_batch.html')


if __name__ == '__main__':
    from core.config import UPLOAD_FOLDER, OUTPUT_FOLDER, SINGLE_OUTPUT_FOLDER
    print("=" * 60)
    print("Doppler Effect Batch Simulator (WITH PATH VALIDATION)")
    print("=" * 60)
    print(f"Vehicle sounds folder: {UPLOAD_FOLDER}")
    print(f"Batch output folder (default root): {OUTPUT_FOLDER}")
    print(f"Single-clip output folder: {SINGLE_OUTPUT_FOLDER}")
    print(f"Server starting on http://0.0.0.0:5050")
    print("=" * 60)
    print("\nNEW FEATURES:")
    print("  [OK] Path validation enabled by default")
    print("  [OK] Detects road boundary violations")
    print("  [OK] Detects median/centerline crossings")
    print("  [OK] Generates validation reports (JSON + TXT)")
    print("  [OK] Batch validation statistics")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5050)


# Reset everything (when needed)
# Delete these two files:
# sampler_state.json
# generation_progress.json
