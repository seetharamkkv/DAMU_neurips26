import os
import json
import math
import random
import time
import traceback
import base64
import csv
from collections import Counter
import numpy as np
import librosa
from datetime import datetime

from flask import Blueprint, request, jsonify, send_from_directory

from core.config import OUTPUT_FOLDER
from core.sampler import SAMPLERS
from core.progress import load_progress, save_progress
from audio.generation import (
    validate_batch_config,
    calculate_distribution,
    generate_random_parameters,
    generate_single_clip,
    generate_multi_object_clip,
    mix_audio_clips,
    generate_statistics
)
from visualization.plot_utils import save_combined_path_plot, save_spectrogram_to_file
from visualization.validation import validate_scene_paths, save_validation_report
from audio.audio_utils import save_audio, SR
# Deferred import of MapExtraction to avoid crash if cv2 is missing
# from MapExtraction.outline_to_json import convert_outline_png_to_json

batch_bp = Blueprint('batch', __name__)

LINEAR_OVERLAP_SPEEDS_MPS = [
    8.6, 9.7, 10.6, 11.4, 12.2, 13.1, 13.9, 14.7, 15.3, 16.1,
    16.9, 17.8, 18.3, 18.9, 19.4, 20.0, 20.3, 21.1, 21.7, 22.2,
    23.1, 23.9, 24.7, 25.3, 26.1, 26.9, 27.8
]


def upload_batch_to_gcs():
    """Upload batch outputs to GCS. Returns True on success, False on failure."""
    import subprocess
    import time
    import shutil

    local_output_dir = "static/batch_outputs"
    if not os.path.isdir(local_output_dir) or not os.listdir(local_output_dir):
        print("No local batch outputs found, skipping GCS upload.")
        return False

    # Check if gsutil is installed
    if not shutil.which("gsutil"):
        print("Note: 'gsutil' not found in PATH. Skipping cloud backup. Local outputs are preserved in static/batch_outputs.")
        return False

    run_id = str(int(time.time()))
    destination = f"gs://vehicle_audio_source_sep/batch_outputs/run_{run_id}/"
    print(f"Starting batch upload to GCS: {destination} ...")

    # Try running gsutil directly (works on Windows/Linux if in PATH)
    try:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{local_output_dir}/*", destination],
            check=False,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"GCS upload failed: {result.stderr.strip()}. Local outputs preserved.")
            return False
        return True
    except Exception as e:
        print(f"Error during GCS upload attempt: {e}. Local outputs preserved.")
        return False


def cleanup_local_outputs():
    import shutil
    import os

    shutil.rmtree("static/batch_outputs", ignore_errors=True)
    os.makedirs("static/batch_outputs", exist_ok=True)
    print("Local batch outputs cleanup completed.")


def _save_spectrogram_npy(audio_array, output_path):
    """Save a lightweight STFT power spectrogram as .npy (no plotting)."""
    n_fft = 2048
    hop_length = 256
    stft = librosa.stft(audio_array, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann")
    spec_power = (np.abs(stft) ** 2).astype(np.float32)
    np.save(output_path, spec_power)
    return os.path.basename(output_path)


def _build_linear_overlap_scene(config, selected_vehicles, clip_index, scene_dir, total_clips=None):
    from audio.generation import get_doppler_audio_array

    overlap_cfg = config.get('linear_overlap', {}) or {}
    delay_min = float(overlap_cfg.get('delay_min', 0.3))
    delay_max = float(overlap_cfg.get('delay_max', 2.0))
    observer_distance = float(overlap_cfg.get('observer_distance', 18.0))
    generate_png_spectrograms = bool(overlap_cfg.get('generate_png_spectrograms', False))
    delay_min, delay_max = min(delay_min, delay_max), max(delay_min, delay_max)

    vehicle_min = int(overlap_cfg.get('vehicle_min', 3))
    vehicle_max = int(overlap_cfg.get('vehicle_max', 5))
    vehicle_min, vehicle_max = min(vehicle_min, vehicle_max), max(vehicle_min, vehicle_max)
    vehicle_min = max(2, vehicle_min)
    vehicle_max = min(len(LINEAR_OVERLAP_SPEEDS_MPS), vehicle_max)
    num_vehicles = random.randint(vehicle_min, vehicle_max)

    scene_vehicles = random.sample(selected_vehicles, k=num_vehicles)
    # Enforce unique speeds and assign in decreasing order.
    scene_speeds = sorted(random.sample(LINEAR_OVERLAP_SPEEDS_MPS, k=num_vehicles), reverse=True)

    # Sequential delays so each next vehicle starts later than previous.
    per_vehicle_start_delays = []
    running_delay = 0.0
    for i in range(num_vehicles):
        if i == 0:
            running_delay = 0.0
        else:
            running_delay += random.uniform(delay_min, delay_max)
        per_vehicle_start_delays.append(running_delay)

    scene_paths_data = []
    delayed_individual_clips = []
    vehicles_meta = []

    clip_len_samples = int(10.0 * SR)
    clip_time_duration = 10.0

    for i, (vehicle_name, speed_value, start_delay) in enumerate(
        zip(scene_vehicles, scene_speeds, per_vehicle_start_delays), start=1
    ):
        base_params = generate_random_parameters(config, vehicle_name, 'straight', force_symmetric=True)
        params = dict(base_params)
        params['speed'] = float(speed_value)
        params['distance'] = float(observer_distance)
        params['angle'] = 0.0
        params['duration'] = clip_time_duration
        params['apply_propagation_delay'] = False

        audio_arr, _, _ = get_doppler_audio_array(vehicle_name, 'straight', params)
        if len(audio_arr) != clip_len_samples:
            # Hard guarantee: every output clip is exactly 10s.
            fixed = np.zeros(clip_len_samples, dtype=np.float32)
            n = min(len(audio_arr), clip_len_samples)
            fixed[:n] = audio_arr[:n]
            audio_arr = fixed

        delay_samples = int(round(start_delay * SR))
        delayed_audio = np.zeros(clip_len_samples, dtype=np.float32)
        if delay_samples < clip_len_samples:
            src_len = clip_len_samples - delay_samples
            delayed_audio[delay_samples:] = audio_arr[:src_len]
        delayed_individual_clips.append(delayed_audio)

        v_file = f"vehicle_{i:02d}_{vehicle_name}.wav"
        v_path = os.path.join(scene_dir, v_file)
        save_audio(delayed_audio, v_path)

        v_spec_npy_file = f"vehicle_{i:02d}_{vehicle_name}_spec.npy"
        _save_spectrogram_npy(delayed_audio, os.path.join(scene_dir, v_spec_npy_file))
        v_spec_png_file = None
        if generate_png_spectrograms:
            v_spec_png_file = f"vehicle_{i:02d}_{vehicle_name}_spec.png"
            save_spectrogram_to_file(
                delayed_audio,
                SR,
                v_spec_png_file,
                os.path.join(scene_dir, v_spec_png_file)
            )

        # Plot on global timeline [0, 10] with this source active from its start delay.
        plot_params = dict(params)
        plot_params['duration'] = clip_time_duration
        plot_params['cpa_time'] = float(start_delay) + (clip_time_duration / 2.0)
        plot_params['plot_t_start'] = float(start_delay)
        plot_params['plot_t_end'] = clip_time_duration
        scene_paths_data.append(('straight', plot_params, vehicle_name))

        cpa_time_global = float(plot_params['cpa_time'])
        x_at_t0 = float(speed_value * (0.0 - cpa_time_global))
        x_at_t10 = float(speed_value * (clip_time_duration - cpa_time_global))
        y_const = float(observer_distance)
        vehicles_meta.append({
            'id': i,
            'vehicle': vehicle_name,
            'audio_file': v_file,
            'spectrogram_npy_file': v_spec_npy_file,
            'spectrogram_png_file': v_spec_png_file,
            'start_delay_s': float(start_delay),
            'start_time_s': float(start_delay),
            'speed_mps': float(speed_value),
            'positions': {
                't0': {'x_m': x_at_t0, 'y_m': y_const},
                't10': {'x_m': x_at_t10, 'y_m': y_const}
            },
            'parameters': params
        })

        # Keep UI progress alive during heavy scene generation.
        if total_clips and total_clips > 0:
            partial = (clip_index - 1) + (0.92 * (i / max(1, num_vehicles)))
            save_progress(total_clips, round(partial, 3))

    mixed_audio = np.sum(np.stack(delayed_individual_clips, axis=0), axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mixed_audio))) if mixed_audio.size else 0.0
    if peak > 0.98:
        mixed_audio = mixed_audio * (0.98 / peak)

    mixed_audio_file = "mixed_audio.wav"
    mixed_audio_path = os.path.join(scene_dir, mixed_audio_file)
    save_audio(mixed_audio, mixed_audio_path)

    mixed_spec_npy_file = "mixed_audio_spec.npy"
    _save_spectrogram_npy(mixed_audio, os.path.join(scene_dir, mixed_spec_npy_file))
    mixed_spec_png_file = None
    if generate_png_spectrograms:
        mixed_spec_png_file = "mixed_audio_spec.png"
        save_spectrogram_to_file(
            mixed_audio,
            SR,
            mixed_spec_png_file,
            os.path.join(scene_dir, mixed_spec_png_file)
        )

    path_plot_name = save_combined_path_plot(
        scene_paths_data,
        scene_dir,
        "scene",
        lane_width=4.0,
        include_opposite=False,
        road_y_center=observer_distance,
        observer_pos=(0.0, 0.0),
        road_shape='straight',
        absolute=True,
        show_road_guides=False,
        path_alpha=0.52,
        path_linewidth=18.0,
        fig_width=22.0,
        fig_height=6.6
    )

    return {
        'scene_id': f"{clip_index:04d}",
        'mode': 'linear_overlap',
        'clip_duration_s': clip_time_duration,
        'observer_distance_m': float(observer_distance),
        'delay_range_s': [float(delay_min), float(delay_max)],
        'speed_pool_mps': LINEAR_OVERLAP_SPEEDS_MPS,
        'num_vehicles': len(vehicles_meta),
        'mixed_audio_file': mixed_audio_file,
        'mixed_spectrogram_npy_file': mixed_spec_npy_file,
        'mixed_spectrogram_png_file': mixed_spec_png_file,
        'path_graph_file': path_plot_name or "scene_combined_path.png",
        'vehicles': vehicles_meta,
        'scene_parameters': {
            'temperature_range_c': [
                float(config.get('atmosphere', {}).get('temp_min', 15.0)),
                float(config.get('atmosphere', {}).get('temp_max', 35.0))
            ],
            'humidity_range_percent': [
                float(config.get('atmosphere', {}).get('hum_min', 30.0)),
                float(config.get('atmosphere', {}).get('hum_max', 70.0))
            ],
            'angle_deg_forced': 0.0,
            'path_type_forced': 'straight'
        },
        'timestamp': datetime.now().isoformat()
    }


def _run_linear_overlap_batch(config, start_time):
    base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
    os.makedirs(base_output_root, exist_ok=True)

    custom_name = config.get('batch', {}).get('name', '').strip()
    if custom_name:
        safe_batch_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_batch_name = safe_batch_name.replace(' ', '_')
        batch_id = safe_batch_name
        batch_dir = os.path.join(base_output_root, batch_id)
    else:
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(base_output_root, f'batch_{batch_id}')
    os.makedirs(batch_dir, exist_ok=True)

    total_clips = int(config['batch']['total_clips'])
    selected_vehicles = config.get('vehicles', {}).get('selected', [])
    overlap_cfg = config.get('linear_overlap', {})
    requested_vehicle_min = int(overlap_cfg.get('vehicle_min', 3))
    requested_vehicle_max = int(overlap_cfg.get('vehicle_max', 5))
    requested_vehicle_min, requested_vehicle_max = min(requested_vehicle_min, requested_vehicle_max), max(requested_vehicle_min, requested_vehicle_max)
    if not selected_vehicles:
        return jsonify({'error': 'No vehicles selected'}), 400

    if requested_vehicle_min < 2:
        return jsonify({'error': 'Linear Overlap Mode needs min vehicles >= 2.'}), 400
    if requested_vehicle_max > len(LINEAR_OVERLAP_SPEEDS_MPS):
        return jsonify({'error': f'Linear Overlap Mode supports at most {len(LINEAR_OVERLAP_SPEEDS_MPS)} vehicles (unique speed constraint).'}), 400
    if len(selected_vehicles) < requested_vehicle_max:
        return jsonify({'error': f'Select at least {requested_vehicle_max} vehicles for Linear Overlap Mode (max overlap bound).'}), 400

    SAMPLERS.clear()
    save_progress(total_clips, 0)

    scene_metadata = []
    for clip_index in range(1, total_clips + 1):
        scene_id = f"{clip_index:04d}"
        scene_dir = os.path.join(batch_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        meta = _build_linear_overlap_scene(
            config,
            selected_vehicles,
            clip_index,
            scene_dir,
            total_clips=total_clips
        )
        with open(os.path.join(scene_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        scene_metadata.append(meta)
        save_progress(total_clips, clip_index)

    metadata_file = os.path.join(batch_dir, f"metadata_{batch_id}.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'batch_id': batch_id,
            'mode': 'linear_overlap',
            'total_scenes': total_clips,
            'scenes': scene_metadata
        }, f, indent=2)

    elapsed_time = time.time() - start_time
    formatted_time = f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"

    gcs_ok = upload_batch_to_gcs()
    if gcs_ok:
        cleanup_local_outputs()

    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'total_generated': total_clips,
        'elapsed_time': elapsed_time,
        'formatted_time': formatted_time,
        'batch_directory': batch_dir,
        'metadata_file': metadata_file,
        'log_file': '',
        'stats_file': ''
    })


@batch_bp.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    """Generate batch of Doppler simulations"""
    try:
        config = request.get_json()

        start_time = time.time()

        if bool(config.get('linear_overlap', {}).get('enabled', False)):
            # Force mode constraints regardless of generic simulator controls.
            config.setdefault('paths', {})['selected'] = ['straight']
            config.setdefault('duration', {})['randomize'] = False
            config['duration']['min'] = 10.0
            config['duration']['max'] = 10.0
            config.setdefault('angle', {})['randomize'] = False
            config['angle']['min'] = 0.0
            config['angle']['max'] = 0.0

        # Validate configuration
        validation_error = validate_batch_config(config)
        if validation_error:
            return jsonify({'error': validation_error}), 400

        if bool(config.get('linear_overlap', {}).get('enabled', False)):
            return _run_linear_overlap_batch(config, start_time)

        # Base output root (respect UI "Save Path" if provided)
        base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
        os.makedirs(base_output_root, exist_ok=True)

        # Create batch directory
        custom_name = config.get('batch', {}).get('name', '').strip()
        if custom_name:
            # Sanitize custom name
            safe_batch_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_batch_name = safe_batch_name.replace(' ', '_')
            batch_id = safe_batch_name
            batch_dir = os.path.join(base_output_root, batch_id)
        else:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_dir = os.path.join(base_output_root, f'batch_{batch_id}')

        audio_dir = os.path.join(batch_dir, 'audio_clips')
        os.makedirs(audio_dir, exist_ok=True)

        # batch generation - use exact number from input
        # Get the exact number of clips requested by the user
        total_clips = int(config['batch']['total_clips'])

        # Reset samplers for fresh uniform state space coverage each batch
        SAMPLERS.clear()
        save_progress(total_clips, 0)

        # Get distribution (per-vehicle & per-path counts)
        distribution = calculate_distribution(config, total_clips)
        vehicle_dist = distribution['vehicles']
        path_dist = distribution['paths']

        # Build flat lists of vehicles and paths with exact counts
        vehicle_list = []
        for v, count in vehicle_dist.items():
            vehicle_list.extend([v] * int(count))

        path_list = []
        for p, count in path_dist.items():
            path_list.extend([p] * int(count))

        # Helper to fix length to exactly total_clips
        def fix_list(lst, allowed_values):
            lst = list(lst)
            if not allowed_values:
                # Critical fallback if no items are selected in the UI
                allowed_values = ['car_1']
            if not lst:
                # If empty, fill with the first allowed value
                return [allowed_values[0]] * total_clips
            if len(lst) > total_clips:
                return lst[:total_clips]
            if len(lst) < total_clips:
                while len(lst) < total_clips:
                    lst.append(random.choice(allowed_values))
            return lst

        vehicle_list = fix_list(vehicle_list, list(vehicle_dist.keys()))
        path_list = fix_list(path_list, list(path_dist.keys()))

        # Now both lists have length  total_clips
        random.shuffle(vehicle_list)
        random.shuffle(path_list)

        clips_metadata = []
        generation_log = []
        clip_index = 1

        for i in range(total_clips):
            vehicle_name = vehicle_list[i]
            path_type = path_list[i]

            try:
                #  Benchmark B8 (Multi-source), B9 (Interaction), B10 (Recognition) 
                bench_cfg = config.get('benchmarks', {})
                selected_bencharks = bench_cfg.get('selected', [])
                
                # Check for multi-source benchmarks
                multi_bench_active = any(b in selected_bencharks for b in ['B8', 'B9', 'B10'])
                single_bench_active = any(b in selected_bencharks for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
                bench_params_early = bench_cfg.get('params', {})
                intersection_active = (
                    'B9' in selected_bencharks
                    and bool(bench_params_early.get('intersection_benchmark', False))
                )
                # Only interleave single-source on odd indices when both single
                # AND multi benchmarks are selected; otherwise honour the selection.
                if bench_cfg.get('enabled', False) and multi_bench_active:
                    if single_bench_active:
                        is_multi_source = intersection_active or (i % 2 == 0)
                    else:
                        is_multi_source = True
                else:
                    is_multi_source = False

                if is_multi_source:
                    # Multi-source mode (Realistic Busy Road)
                    bench_params = bench_cfg.get('params', {})
                    selected_benchmarks = bench_cfg.get('selected', [])
                    
                    # Extract Busy Road parameters from benchmark settings
                    lane_width = float(bench_params.get('lane_width', 4.0)) # Width of ONE lane
                    include_opposite = bench_params.get('include_opposite', True)
                    max_stagger = float(bench_params.get('max_stagger', 5.0))
                    v_min = int(bench_params.get('vehicle_min', 2))
                    v_max = int(bench_params.get('vehicle_max', 5))
                    # Keep multi-source controls explicit to avoid NameError in loop logic.
                    num_sources = random.randint(v_min, v_max)
                    is_force_crossing = bool(
                        bench_params.get('force_crossing', False)
                        or bench_params.get('is_crossing', False)
                    )
                    # Distribute selected road shapes equally across scenes.
                    available_shapes = bench_params.get('road_shapes', ['straight', 'parabola', 'bezier'])
                    if not available_shapes:
                        available_shapes = ['straight']
                    road_shape = available_shapes[i % len(available_shapes)]
                    if road_shape == 'parabola':
                        road_curve_a = random.choice([-1, 1]) * random.uniform(3e-5, 2e-4)
                    else:
                        road_curve_a = 0.0
                    road_bezier_bulge = random.uniform(0.4, 1.2) if road_shape == 'bezier' else 0.0

                    # Calculate road_y_center to maintain safe distance (10m) from nearest edge
                    road_y_center = lane_width + 10.0
                    observer_pos = (0.0, 0.0)
                    
                    v_configs = []

                    #  B9: Plus-shaped intersection benchmark 
                    intersection_mode = (
                        'B9' in selected_bencharks
                        and bool(bench_params.get('intersection_benchmark', False))
                    )
                    if intersection_mode:
                        # Intersection centered at origin; primary road along x-axis,
                        # secondary road at the configured intersection angle.
                        road_curve_a = 0.0
                        road_y_center = 0.0
                        half_arm = float(bench_params.get('intersection_half_arm', 90.0))
                        lane_half = lane_width / 2.0
                        ia_min = float(bench_params.get('intersection_angle_min', 30.0))
                        ia_max = float(bench_params.get('intersection_angle_max', 150.0))
                        # Evenly cover the range across scenes via linear spacing.
                        if total_clips > 1:
                            intersection_angle = ia_min + (ia_max - ia_min) * (i / (total_clips - 1))
                        else:
                            intersection_angle = (ia_min + ia_max) / 2.0

                        # Observer placed 50% beyond the Q1 vertex of the
                        # intersection (where primary upper edge meets secondary edge),
                        # guaranteeing it is outside both roads.
                        _ia_rad = math.radians(intersection_angle)
                        _ia_sin = math.sin(_ia_rad)
                        _ia_cos = math.cos(_ia_rad)
                        vertex_x = lane_half * (1.0 + _ia_cos) / max(1e-6, _ia_sin)
                        vertex_y = lane_half
                        observer_pos = (1.5 * vertex_x, 1.5 * vertex_y)

                        exits_by_approach = {
                            'W': ['E', 'N', 'S'],
                            'E': ['W', 'S', 'N'],
                            'S': ['N', 'W', 'E'],
                            'N': ['S', 'E', 'W'],
                        }

                        # Primary road (E-W) stays along x-axis;
                        # secondary road (N-S) is rotated by intersection_angle.
                        _sec_cos = _ia_cos
                        _sec_sin = _ia_sin

                        def _arm_point(arm, dist_from_center, lane_pos):
                            """(x, y) at `dist_from_center` along `arm`, offset by `lane_pos`."""
                            if arm == 'W': return (-dist_from_center, lane_pos)
                            if arm == 'E': return (dist_from_center, lane_pos)
                            if arm == 'N':
                                return (dist_from_center * _sec_cos - lane_pos * _sec_sin,
                                        dist_from_center * _sec_sin + lane_pos * _sec_cos)
                            return (-dist_from_center * _sec_cos - lane_pos * _sec_sin,
                                    -dist_from_center * _sec_sin + lane_pos * _sec_cos)

                        def _intersection_waypoints(start_arm, end_arm, half_span, lh, entry_lane, exit_lane):
                            """Build waypoints with natural lane-following and drift."""
                            pts = []
                            n_arm = 10

                            # Approach: gradual drift from a starting lateral position
                            # toward entry_lane at the intersection edge.
                            approach_start = entry_lane + random.uniform(-lh * 0.3, lh * 0.3)
                            approach_start = max(-lh * 0.85, min(lh * 0.85, approach_start))
                            for k in range(n_arm):
                                frac = k / max(1, n_arm - 1)
                                d = half_span + frac * (lh - half_span)
                                lat = approach_start + frac * (entry_lane - approach_start)
                                lat += random.uniform(-0.25, 0.25) * (1.0 - frac)
                                lat = max(-lh * 0.9, min(lh * 0.9, lat))
                                pts.append(_arm_point(start_arm, d, lat))

                            entry_edge = _arm_point(start_arm, lh, entry_lane)
                            exit_edge = _arm_point(end_arm, lh, exit_lane)

                            is_straight = (
                                frozenset((start_arm, end_arm)) == frozenset(('W', 'E'))
                                or frozenset((start_arm, end_arm)) == frozenset(('N', 'S'))
                            )

                            n_turn = 6
                            if is_straight:
                                drift = random.uniform(-0.3, 0.3)
                                for k in range(1, n_turn + 1):
                                    t = k / (n_turn + 1)
                                    x = entry_edge[0] + t * (exit_edge[0] - entry_edge[0])
                                    y = entry_edge[1] + t * (exit_edge[1] - entry_edge[1])
                                    s_curve = drift * math.sin(t * math.pi)
                                    x += random.uniform(-0.15, 0.15) + s_curve
                                    y += random.uniform(-0.15, 0.15) + s_curve
                                    pts.append((x, y))
                            else:
                                pull_factor = random.uniform(0.15, 0.35)
                                mid_x = (entry_edge[0] + exit_edge[0]) * pull_factor
                                mid_y = (entry_edge[1] + exit_edge[1]) * pull_factor
                                for k in range(1, n_turn + 1):
                                    t = k / (n_turn + 1)
                                    bx = (1-t)**2 * entry_edge[0] + 2*(1-t)*t * mid_x + t**2 * exit_edge[0]
                                    by = (1-t)**2 * entry_edge[1] + 2*(1-t)*t * mid_y + t**2 * exit_edge[1]
                                    bx += random.uniform(-0.15, 0.15)
                                    by += random.uniform(-0.15, 0.15)
                                    pts.append((bx, by))

                            # Depart: drift away from exit_lane naturally.
                            depart_end = exit_lane + random.uniform(-lh * 0.3, lh * 0.3)
                            depart_end = max(-lh * 0.85, min(lh * 0.85, depart_end))
                            for k in range(n_arm):
                                frac = k / max(1, n_arm - 1)
                                d = lh + frac * (half_span - lh)
                                lat = exit_lane + frac * (depart_end - exit_lane)
                                lat += random.uniform(-0.25, 0.25) * frac
                                lat = max(-lh * 0.9, min(lh * 0.9, lat))
                                pts.append(_arm_point(end_arm, d, lat))

                            return pts

                        sel = config.get('vehicles', {}).get('selected', [vehicle_name])
                        temp = 20
                        hum = 50
                        duration = 10.0
                        speeds = [random.randint(
                            int(config.get('speed', {}).get('min', 15)),
                            int(config.get('speed', {}).get('max', 35)),
                        ) for _ in range(max(1, num_sources))]

                        # Pre-assign unique lane positions per arm so no two
                        # vehicles from the same arm share the same lateral slot.
                        inner = lane_half * 0.75
                        arm_lane_slots = {}
                        def _get_lane_slot(arm, count_in_arm):
                            if arm not in arm_lane_slots:
                                arm_lane_slots[arm] = 0
                            idx = arm_lane_slots[arm]
                            arm_lane_slots[arm] += 1
                            if count_in_arm <= 1:
                                return random.uniform(-inner * 0.5, inner * 0.5)
                            step = (2.0 * inner) / count_in_arm
                            return -inner + step * (idx + 0.5) + random.uniform(-step * 0.15, step * 0.15)

                        approach_order = ['W', 'E', 'S', 'N']
                        # Build arm assignments first to count vehicles per arm.
                        arm_assignments = []
                        for s_idx in range(max(1, num_sources)):
                            start_arm = approach_order[s_idx % 4] if s_idx < 4 else random.choice(approach_order)
                            end_arm = random.choices(
                                exits_by_approach[start_arm],
                                weights=[0.30, 0.40, 0.30],
                                k=1,
                            )[0]
                            arm_assignments.append((start_arm, end_arm))

                        start_counts = Counter(a[0] for a in arm_assignments)
                        end_counts = Counter(a[1] for a in arm_assignments)

                        v_configs = []
                        for s_idx, (start_arm, end_arm) in enumerate(arm_assignments):
                            entry_lane = _get_lane_slot(start_arm, start_counts[start_arm])
                            exit_lane = _get_lane_slot('exit_' + end_arm, end_counts[end_arm])
                            waypoints = _intersection_waypoints(
                                start_arm, end_arm, half_arm, lane_half,
                                entry_lane, exit_lane,
                            )
                            v_configs.append({
                                'vehicle_name': random.choice(sel),
                                'path_type': 'map_path',
                                'params': {
                                    'points': waypoints,
                                    'speed': speeds[s_idx],
                                    'duration': duration,
                                    'temperature': temp,
                                    'humidity': hum,
                                },
                                'delay': random.uniform(0, max_stagger),
                                'is_crossing': is_force_crossing,
                                'direction': 1,
                            })

                        result = generate_multi_object_clip(
                            v_configs, audio_dir, batch_id, clip_index, config,
                            observer_pos=observer_pos, road_curve_a=road_curve_a,
                            road_y_center=road_y_center,
                            intersection_angle=intersection_angle
                        )
                    else:
                        # Force even vehicle count for equal distribution across lanes.
                        if num_sources % 2 != 0:
                            num_sources += 1
                        half = num_sources // 2

                        # Lane geometry: road centered at road_y_center
                        # Forward lane (dir=+1): [road_y_center - lane_width, road_y_center]
                        # Opposite lane (dir=-1): [road_y_center, road_y_center + lane_width]
                        fwd_y_min = road_y_center - lane_width
                        fwd_y_max = road_y_center
                        opp_y_min = road_y_center
                        opp_y_max = road_y_center + lane_width

                        edge_buffer = 0.4
                        clamp_lo_fwd = fwd_y_min + edge_buffer
                        clamp_hi_fwd = fwd_y_max - edge_buffer
                        clamp_lo_opp = opp_y_min + edge_buffer
                        clamp_hi_opp = opp_y_max - edge_buffer

                        fwd_center = (fwd_y_min + fwd_y_max) / 2.0
                        opp_center = (opp_y_min + opp_y_max) / 2.0
                        usable_half = (lane_width / 2.0) - edge_buffer

                        # Build ordered list: alternate fwd / opp.
                        assignments = []
                        for k in range(half):
                            assignments.append((fwd_center, fwd_y_min, fwd_y_max, clamp_lo_fwd, clamp_hi_fwd, 1))
                            assignments.append((opp_center, opp_y_min, opp_y_max, clamp_lo_opp, clamp_hi_opp, -1))

                        # Scene-level crossing policy:
                        # Force one paired crossing in a randomly chosen lane
                        # whenever that lane has at least two vehicles.
                        forced_cross_lane = random.choice([1, -1]) if half >= 2 else None

                        # Spread vehicles across discrete lateral slots per lane so
                        # trajectories are visually and physically separated.
                        per_lane_total = {1: half, -1: half}
                        per_lane_seen = {1: 0, -1: 0}
                        # Target cadence: ~1 in 3 paths intersect per lane, with
                        # randomized phase so it is not always the same slot index.
                        lane_cross_phase = {1: random.randint(0, 2), -1: random.randint(0, 2)}
                        # Keep one pending "cross partner" per lane so two nearby
                        # vehicles can form an actual intersection/overtake pair.
                        lane_cross_anchor = {1: None, -1: None}

                        def _lane_slot(clamp_lo, clamp_hi, direction):
                            total = max(1, per_lane_total[direction])
                            idx = per_lane_seen[direction]
                            per_lane_seen[direction] += 1
                            width = max(0.2, clamp_hi - clamp_lo)
                            # Keep at least ~0.9 m center-to-center when possible.
                            nominal_gap = 0.9
                            max_slots_fit = max(1, int(width / nominal_gap))
                            n_slots = min(total, max_slots_fit)
                            if n_slots <= 1:
                                y_slot = 0.5 * (clamp_lo + clamp_hi) + random.uniform(-0.06, 0.06)
                                return y_slot, idx, total
                            # If too many vehicles for available width, wrap on slots
                            # and add tiny noise so paths are still distinguishable.
                            slot_idx = idx % n_slots
                            frac = (slot_idx + 0.5) / n_slots
                            y_slot = clamp_lo + frac * width
                            return y_slot + random.uniform(-0.05, 0.05), idx, total

                        for s_idx, (lane_center, lane_y_min, lane_y_max, clamp_lo, clamp_hi, direction) in enumerate(assignments):
                            v_name = random.choice(config.get('vehicles', {}).get('selected', [vehicle_name]))
                            s_min = int(config.get('speed', {}).get('min', 15))
                            s_max = int(config.get('speed', {}).get('max', 35))
                            speed = random.randint(s_min, s_max)

                            # Assign a lane slot (with small jitter) to avoid overlap.
                            lane_offset_raw, lane_idx, lane_total = _lane_slot(clamp_lo, clamp_hi, direction)
                            lane_offset = max(clamp_lo, min(clamp_hi, lane_offset_raw))

                            # Prefer road-following trajectories; keep parabola very rare.
                            p_type = random.choices(['parabola', 'bezier'], weights=[0.05, 0.95])[0]
                            is_forced_cross_pair = (
                                forced_cross_lane is not None
                                and direction == forced_cross_lane
                                and lane_idx in (0, 1)
                                and lane_total >= 2
                            )
                            if is_forced_cross_pair:
                                # Crossing pair must be Bezier so paths can intersect.
                                p_type = 'bezier'

                            # Ensure duration respects road limit to avoid bezier scaling issues
                            road_limit = 100.0
                            max_dur = (2.0 * road_limit) / speed
                            duration = min(10.0, max_dur * 0.98)

                            v_params = {
                                'speed': speed,
                                'duration': duration,
                                'temperature': 20,
                                'humidity': 50,
                            }
                            v_params['road_curve_blend'] = 1.0
                            v_params['road_angle_offset'] = 0.0
                            v_params['global_curve_scale'] = 1.0

                            if p_type == 'parabola':
                                v_params['h'] = lane_offset
                                span = abs(speed) * duration
                                half_span = max(1.0, span / 2.0)
                                available_down = lane_offset - clamp_lo
                                available_up = clamp_hi - lane_offset
                                # Keep vertical sag very small (realistic lane-following),
                                # unlike aggressive U-shapes that drift toward median.
                                target_dev = min(0.7, max(0.15, usable_half * 0.28))
                                desired_mag = target_dev / max(1.0, half_span ** 2)
                                max_a_up = max(0.0, available_up) / max(1.0, half_span ** 2)
                                max_a_down = max(0.0, available_down) / max(1.0, half_span ** 2)
                                up_cap = min(desired_mag * random.uniform(0.4, 1.0), max_a_up)
                                down_cap = min(desired_mag * random.uniform(0.4, 1.0), max_a_down)
                                if up_cap > 0 and down_cap > 0:
                                    v_params['a'] = up_cap if random.random() < 0.5 else -down_cap
                                elif up_cap > 0:
                                    v_params['a'] = up_cap
                                elif down_cap > 0:
                                    v_params['a'] = -down_cap
                                else:
                                    v_params['a'] = 0.0

                            elif p_type == 'bezier':
                                # Mostly lane-following trajectories with minor deviations.
                                forced_cross = is_forced_cross_pair
                                crossing_event = forced_cross or (
                                    lane_total >= 2
                                    and ((lane_idx + lane_cross_phase[direction]) % 3 == 0)
                                )
                                if forced_cross:
                                    maneuver = 'cross'
                                else:
                                    maneuver = random.choices(
                                        ['lane_follow', 'overtake', 'weave', 'cross'],
                                        weights=[0.66, 0.17, 0.09, 0.08] if crossing_event else [0.88, 0.05, 0.07, 0.0],
                                    )[0]
                                # Longitudinal staggering based on lane occupancy and lane width:
                                # - more vehicles in lane => tighter but still separated spacing
                                # - wider lane => allow slightly larger staggering envelope
                                half_span = min(100.0, (speed * duration) / 2.0)
                                lane_center_idx = 0.5 * (lane_total - 1)
                                spacing_gain = max(0.6, (lane_width / 4.0) ** 0.5)
                                travel_span = max(20.0, 2.0 * half_span * 0.80)
                                x_spacing = (travel_span / max(1, lane_total)) * spacing_gain
                                x_spacing = max(6.0, min(30.0, x_spacing))
                                x_jitter = random.uniform(-0.12 * x_spacing, 0.12 * x_spacing)
                                x_shift = (lane_idx - lane_center_idx) * x_spacing + x_jitter
                                # Max lateral drift from lane center in meters.
                                drift = min(0.65, max(0.12, usable_half * 0.22))
                                if crossing_event:
                                    drift = min(usable_half * 0.80, drift * 1.35)
                                # Use per-vehicle lane slot baseline so vehicles in the same
                                # lane do not collapse onto the same trajectory.
                                base_y = lane_offset

                                if maneuver == 'cross':
                                    # Smooth lane-change style arc that can intersect another path
                                    # without introducing sharp turns.
                                    amp = min(usable_half * 0.90, max(0.24, drift * 1.25))
                                    anchor = lane_cross_anchor[direction]
                                    if anchor is None:
                                        side = random.choice([-1, 1])
                                        lane_cross_anchor[direction] = {
                                            'side': side,
                                            'x_shift': x_shift,
                                        }
                                    else:
                                        # Pair with previous cross vehicle in this lane:
                                        # opposite sweep direction + nearly same x-shift so
                                        # trajectories overlap and intersect near mid-clip.
                                        side = -anchor['side']
                                        x_shift = anchor['x_shift'] + random.uniform(-0.08 * x_spacing, 0.08 * x_spacing)
                                        lane_cross_anchor[direction] = None

                                    v_params['y0'] = base_y - side * amp * 0.70
                                    v_params['y1'] = base_y - side * amp * 0.22
                                    v_params['y2'] = base_y + side * amp * 0.22
                                    v_params['y3'] = base_y + side * amp * 0.70
                                elif maneuver == 'overtake':
                                    # Mild overtake-like arc that stays lane-local.
                                    start_y = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                                    swing = random.choice([-1, 1]) * random.uniform(drift * 0.45, drift * 0.80)
                                    v_params['y0'] = start_y
                                    v_params['y1'] = base_y + swing
                                    v_params['y2'] = base_y + swing * random.uniform(0.35, 0.70)
                                    v_params['y3'] = start_y + random.uniform(-drift * 0.2, drift * 0.2)
                                elif maneuver == 'weave':
                                    # Gentle S-curve around lane center.
                                    v_params['y0'] = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                                    v_params['y3'] = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                                    side = random.choice([-1, 1])
                                    v_params['y1'] = base_y + side * random.uniform(drift * 0.35, drift * 0.75)
                                    v_params['y2'] = base_y - side * random.uniform(drift * 0.30, drift * 0.65)
                                else:
                                    # Default: follow lane with very minor steering variation.
                                    v_params['y0'] = base_y + random.uniform(-drift * 0.35, drift * 0.35)
                                    v_params['y3'] = base_y + random.uniform(-drift * 0.35, drift * 0.35)
                                    v_params['y1'] = base_y + random.uniform(-drift * 0.55, drift * 0.55)
                                    v_params['y2'] = base_y + random.uniform(-drift * 0.55, drift * 0.55)

                                for key in ['y0', 'y1', 'y2', 'y3']:
                                    v_params[key] = max(clamp_lo, min(clamp_hi, v_params[key]))

                                if direction == 1:
                                    v_params['x0'], v_params['x3'] = -half_span + x_shift, half_span + x_shift
                                else:
                                    v_params['x0'], v_params['x3'] = half_span + x_shift, -half_span + x_shift
                                v_params['x1'] = v_params['x0'] + (v_params['x3'] - v_params['x0']) * 0.33
                                v_params['x2'] = v_params['x0'] + (v_params['x3'] - v_params['x0']) * 0.66

                            delay = random.uniform(0, max_stagger)

                            v_configs.append({
                                'vehicle_name': v_name,
                                'path_type': p_type,
                                'params': v_params,
                                'delay': delay,
                                'is_crossing': is_force_crossing,
                                'offset': lane_offset,
                                'direction': direction,
                                'speed': speed,
                            })
                        
                        result = generate_multi_object_clip(
                            v_configs, audio_dir, batch_id, clip_index, config, 
                            observer_pos=observer_pos, road_curve_a=road_curve_a,
                            road_y_center=road_y_center,
                            road_shape=road_shape,
                            road_bezier_bulge=road_bezier_bulge
                        )
                else:
                    # Standard single-source mode
                    params = generate_random_parameters(config, vehicle_name, path_type)
                    result = generate_single_clip(
                        vehicle_name, path_type, params,
                        audio_dir, batch_id, clip_index, config
                    )

                clips_metadata.append(result)
                generation_log.append(f"Generated clip {clip_index}/{total_clips}: {result['filename']}")
                print(f"Generated clip {clip_index}/{total_clips}")
                save_progress(total_clips, clip_index)
                clip_index += 1

            except Exception as e:
                error_message = f"Error generating clip {clip_index}/{total_clips}: {str(e)}"
                traceback.print_exc()
                generation_log.append(error_message)
                print(error_message)
                save_progress(total_clips, clip_index)
                continue

        # Save metadata
        metadata_file = os.path.join(batch_dir, f'metadata_{batch_id}.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'config': config,
                'clips': clips_metadata,
                'total_generated': len(clips_metadata),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # NEW: Save ground-truth dataset.csv
        dataset_file = os.path.join(batch_dir, 'dataset.csv')
        csv_headers = [
            'sample_id', 'batch_id', 'filename', 'vehicle_class', 'trajectory_type',
            'speed_mps', 'acceleration_mps2', 'direction_label', 'cpa_distance_m', 'cpa_time_sec',
            'num_sources', 'is_crossing'
        ]
        
        with open(dataset_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for clip in clips_metadata:
                labels = clip.get('labels', {})
                writer.writerow({
                    'sample_id': clip.get('sample_dir', ''),
                    'batch_id': batch_id,
                    'filename': clip.get('filename', ''),
                    'vehicle_class': labels.get('vehicle_class', ''),
                    'trajectory_type': labels.get('trajectory_type', ''),
                    'speed_mps': labels.get('speed_mps', 0.0),
                    'acceleration_mps2': labels.get('acceleration_mps2', 0.0),
                    'direction_label': labels.get('direction_label', 0),
                    'cpa_distance_m': labels.get('cpa_distance_m', 0.0),
                    'cpa_time_sec': labels.get('cpa_time_sec', 5.0),
                    'num_sources': labels.get('num_sources', 1),
                    'is_crossing': labels.get('is_crossing', False)
                })

        # Save log
        log_file = os.path.join(batch_dir, f'generation_log_{batch_id}.txt')
        with open(log_file, 'w') as f:
            f.write('\n'.join(generation_log))

        # Generate statistics
        stats_text = generate_statistics(clips_metadata, config)
        stats_file = os.path.join(batch_dir, f'statistics_{batch_id}.txt')
        with open(stats_file, 'w') as f:
            f.write(stats_text)

        elapsed_time = time.time() - start_time
        formatted_time = f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        print(f"Batch generation finished in {formatted_time}")

        gcs_ok = upload_batch_to_gcs()
        if gcs_ok:
            cleanup_local_outputs()

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'total_generated': len(clips_metadata),
            'elapsed_time': elapsed_time,
            'formatted_time': formatted_time,
            'batch_directory': batch_dir,
            'metadata_file': metadata_file,
            'log_file': log_file,
            'stats_file': stats_file
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@batch_bp.route('/api/batch_overlap_generate', methods=['POST'])
def batch_overlap_generate():
    """Generate scenes with multiple overlapping vehicles (busy road simulation) WITH VALIDATION"""
    try:
        config = request.get_json()
        start_time = time.time()

        # Root output directory
        base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
        custom_name = config.get('batch', {}).get('name', 'overlap_batch').strip()
        if not custom_name:
            custom_name = f"overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        safe_root_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        root_dir = os.path.join(base_output_root, safe_root_name)
        os.makedirs(root_dir, exist_ok=True)

        num_datasets = int(config.get('batch', {}).get('total_scenes', 10))
        vehicle_min = int(config.get('overlap', {}).get('vehicle_min', 1))
        vehicle_max = int(config.get('overlap', {}).get('vehicle_max', 20))
        lane_width = float(config.get('overlap', {}).get('lane_width', 4.0))
        max_stagger = float(config.get('overlap', {}).get('max_stagger', 5.0))
        include_opposite = config.get('overlap', {}).get('include_opposite', False)
        force_crossing = config.get('overlap', {}).get('force_crossing', False)

        # NEW: Get validation settings from config (with defaults)
        enable_validation = config.get('validation', {}).get('enabled', True)
        validation_tolerance = float(config.get('validation', {}).get('tolerance', 0.5))
        road_angle = float(config.get('overlap', {}).get('road_angle', 0.0))
        
        # Inject road_angle into benchmarks params for generation consistency
        if 'benchmarks' not in config:
            config['benchmarks'] = {'enabled': True, 'params': {}}
        config['benchmarks']['params']['road_angle'] = road_angle

        selected_vehicles = config.get('vehicles', {}).get('selected', [])
        if not selected_vehicles:
            return jsonify({'error': 'No vehicles selected'}), 400

        selected_paths = config.get('paths', {}).get('selected', ['straight'])

        path_mixing_mode = config.get('overlap', {}).get('path_mixing_mode', 'same')

        SAMPLERS.clear()
        save_progress(num_datasets, 0)

        # NEW: Track validation statistics
        validation_stats = {
            'total_scenes': 0,
            'valid_scenes': 0,
            'invalid_scenes': 0,
            'total_vehicles': 0,
            'valid_vehicles': 0,
            'invalid_vehicles': 0
        }

        for scene_idx in range(1, num_datasets + 1):
            scene_id = f"{scene_idx:04d}"
            scene_dir = os.path.join(root_dir, scene_id)
            os.makedirs(scene_dir, exist_ok=True)

            num_vehicles = random.randint(vehicle_min, vehicle_max)

            # Determine path logic for this scene
            scene_path_mixing = path_mixing_mode
            if scene_path_mixing == 'both':
                scene_path_mixing = random.choice(['same', 'mixed'])

            scene_path_type = None
            if scene_path_mixing == 'same':
                scene_path_type = random.choice(selected_paths)

            clips_metadata = []
            clips_with_delays = []
            scene_paths_data = []

            for v_idx in range(1, num_vehicles + 1):
                vehicle_name = random.choice(selected_vehicles)

                # Assign path for this specific vehicle
                if scene_path_mixing == 'same':
                    path_type = scene_path_type
                else:
                    path_type = random.choice(selected_paths)

                # Base parameters for this vehicle
                params = generate_random_parameters(config, vehicle_name, path_type, force_symmetric=True)
                
                # Ensure duration matches road limit for Bezier/Parabola to avoid scaling
                speed = params.get('speed', 25.0)
                road_limit = 100.0
                max_dur = (2.0 * road_limit) / speed
                duration = min(params.get('duration', 10.0), max_dur * 0.98)
                params['duration'] = duration

                # Dynamic Centering: Ensure nearest edge is 10m away
                observer_pos = (0.0, 0.0)
                road_y_center = lane_width + 10.0
                y_median = 0.0
                road_y_min = -lane_width
                road_y_max = lane_width
                y_median = 0.0
                road_y_min = -lane_width
                road_y_max = lane_width

                # Force direction for crossing vehicles
                if force_crossing and v_idx <= 2:
                    is_opposite = False
                elif include_opposite:
                    # Randomly decide direction (roughly equal spread)
                    is_opposite = (v_idx % 2 == 0)
                else:
                    is_opposite = False

                if include_opposite:
                    if is_opposite:
                        # Lane 2 (Opposite): between median and upper road edge
                        lane_y_min, lane_y_max = y_median, road_y_max
                    else:
                        # Lane 1 (Forward): between lower road edge and median
                        lane_y_min, lane_y_max = road_y_min, y_median
                else:
                    # Single direction road: use entire width
                    lane_y_min, lane_y_max = road_y_min, road_y_max

                # Discrete lane assignment
                num_lanes = max(1, int(lane_width / 4.0))
                # Distribute based on v_idx to avoid huddling
                l_idx = (v_idx - 1) % num_lanes
                lane_sub_offset = (l_idx + 0.5) * (lane_width / num_lanes)
                
                if include_opposite:
                    if is_opposite:
                        lane_offset = lane_sub_offset
                    else:
                        lane_offset = -lane_width + lane_sub_offset
                else:
                    # Single direction: distribute across full road
                    lane_offset = -lane_width + (v_idx % (2*num_lanes) + 0.5) * (2*lane_width / (2*num_lanes))

                # For reverse traffic, we need to flip the direction
                if path_type == 'straight':
                    # Direction: Angle 180 (Right->Left) for opposite, 0 (Left->Right) for forward
                    base_angle = 180 if is_opposite else 0
                    params['angle'] = base_angle + road_angle

                    # Set distance to the lane offset (this is the y-coordinate)
                    # Add a base offset to ensure minimum distance from observer
                    params['distance'] = max(1.0, road_y_center + lane_offset)

                elif path_type == 'parabola':
                    # Direction: Negative speed (Right->Left) for opposite, Positive for forward
                    if is_opposite:
                        params['speed'] = -abs(params['speed'])
                    else:
                        params['speed'] = abs(params['speed'])

                    # Update: Limit curve to stay within the vehicle's specific lane boundaries
                    # y(t) = a * x(t)^2 + h. Since a > 0, y increases from h.
                    # We need h + a * x_max^2 <= lane_y_max
                    h = road_y_center + lane_offset
                    available_up = (road_y_center + lane_y_max) - h

                    span = params['speed'] * params['duration']
                    x_max = abs(span / 2.0)

                    # Leave a small buffer from the edge
                    max_curve_height = max(0.1, available_up - 0.2)
                    max_a = max_curve_height / (x_max ** 2) if x_max > 0 else 0.0001

                    # Use the original 'a' but clamp it to stay within road bounds
                    params['a'] = min(params['a'], max_a)

                    # Set center height to the lane offset with base offset
                    params['h'] = max(1.0, road_y_center + lane_offset)
                    params['distance'] = params['h']
                    params['angle_deg'] = road_angle

                elif path_type == 'bezier':
                    # force crossing logic
                    if force_crossing and v_idx <= 2:
                        y_lane_center = (lane_y_min + lane_y_max) / 2
                        # Swap offsets: car 1 goes from bottom to top, car 2 goes from top to bottom
                        if v_idx == 1:
                            y0_rel, y3_rel = lane_y_min + 0.5, lane_y_max - 0.5
                        else:
                            y0_rel, y3_rel = lane_y_max - 0.5, lane_y_min + 0.5
                        
                        params['y0'] = road_y_center + y0_rel
                        params['y1'] = params['y0']
                        params['y2'] = road_y_center + y3_rel
                        params['y3'] = road_y_center + y3_rel
                    else:
                        # Update: keep the bezier curve shape but constrain to this vehicle's lane
                        y_coords = [params['y0'], params['y1'], params['y2'], params['y3']]
                        y_min_curr = min(y_coords)
                        y_max_curr = max(y_coords)
                        current_span = y_max_curr - y_min_curr

                        # Available space in the assigned lane from its chosen center (lane_offset)
                        available_up = lane_y_max - lane_offset
                        available_down = lane_offset - lane_y_min
                        max_half_span = max(0.05, min(available_up, available_down) - 0.2)
                        lane_space = max_half_span * 2

                        if current_span > lane_space and current_span > 0:
                            scale_factor = lane_space / current_span
                            y_center = (y_min_curr + y_max_curr) / 2
                            params['y0'] = y_center + (params['y0'] - y_center) * scale_factor
                            params['y1'] = y_center + (params['y1'] - y_center) * scale_factor
                            params['y2'] = y_center + (params['y2'] - y_center) * scale_factor
                            params['y3'] = y_center + (params['y3'] - y_center) * scale_factor

                        # Shift the curve to the vehicle's lane position
                        base_height = road_y_center + lane_offset
                        y_coords_new = [params['y0'], params['y1'], params['y2'], params['y3']]
                        y_center_new = (min(y_coords_new) + max(y_coords_new)) / 2
                        offset = base_height - y_center_new

                        params['y0'] += offset
                        params['y1'] += offset
                        params['y2'] += offset
                        params['y3'] += offset

                    # Direction: Reverse the path by swapping endpoints and control points
                    if is_opposite:
                        params['x0'], params['x3'] = params['x3'], params['x0']
                        params['x1'], params['x2'] = params['x2'], params['x1']

                    # Clamp absolute results to lane boundaries to be 100% sure
                    y_bound_min = road_y_center + lane_y_min + 0.1
                    y_bound_max = road_y_center + lane_y_max - 0.1
                    params['y0'] = max(y_bound_min, min(y_bound_max, params['y0']))
                    params['y1'] = max(y_bound_min, min(y_bound_max, params['y1']))
                    params['y2'] = max(y_bound_min, min(y_bound_max, params['y2']))
                    params['y3'] = max(y_bound_min, min(y_bound_max, params['y3']))
                    
                    # Align with road angle
                    params['angle_deg'] = road_angle

                    # Scale x-coordinates to match the new duration/speed exactly (ensures phys_scale=1)
                    half_span = (speed * duration) / 2.0
                    x_mid = (params['x0'] + params['x3']) / 2.0
                    current_x_span = abs(params['x3'] - params['x0'])
                    if current_x_span > 0:
                        x_scale = (2.0 * half_span) / current_x_span
                        for k in ['x0', 'x1', 'x2', 'x3']:
                            params[k] = x_mid + (params[k] - x_mid) * x_scale

                delay = random.uniform(0, max_stagger)

                # Generate audio
                from audio.generation import get_doppler_audio_array
                audio_arr, freq_ratios, amplitudes = get_doppler_audio_array(vehicle_name, path_type, params)

                # Save individual audio
                v_filename = f"vehicle_{v_idx:02d}_{vehicle_name}.wav"
                v_audio_path = os.path.join(scene_dir, v_filename)
                save_audio(audio_arr, v_audio_path)

                # NEW: Auto-generate spectrogram for individual car
                v_spec_path = v_audio_path.replace('.wav', '_spec.png')
                save_spectrogram_to_file(audio_arr, SR, v_filename, v_spec_path)

                clips_with_delays.append((audio_arr, delay))
                scene_paths_data.append((path_type, params, vehicle_name))

                clips_metadata.append({
                    'id': v_idx,
                    'vehicle': vehicle_name,
                    'filename': v_filename,
                    'spectrogram': os.path.basename(v_spec_path),
                    'delay_s': delay,
                    'parameters': params
                })

            # new: validate paths before saving
            if enable_validation:
                validation_results = validate_scene_paths(
                    scenes_data=scene_paths_data,
                    lane_width=lane_width,
                    include_opposite=include_opposite,
                    tolerance=validation_tolerance,
                    y_shift=0.0, # compute_path_points already returns world coords
                    road_y_center=road_y_center,
                    road_angle=road_angle
                )

                # Save validation reports
                save_validation_report(validation_results, scene_dir, scene_id)

                # Update statistics
                validation_stats['total_scenes'] += 1
                validation_stats['total_vehicles'] += validation_results['total_vehicles']
                validation_stats['valid_vehicles'] += (
                    validation_results['total_vehicles'] -
                    validation_results['vehicles_with_violations']
                )
                validation_stats['invalid_vehicles'] += validation_results['vehicles_with_violations']

                if validation_results['scene_valid']:
                    validation_stats['valid_scenes'] += 1
                    print(f"[OK] Scene {scene_id}: All {num_vehicles} vehicle paths valid")
                else:
                    validation_stats['invalid_scenes'] += 1
                    print(f"[FAIL] Scene {scene_id}: {validation_results['vehicles_with_violations']}/{num_vehicles} "
                          f"vehicles have violations")
            else:
                print(f"Generated scene {scene_idx}/{num_datasets} (validation disabled)")

            # Align delays so the first vehicle always starts at t=0
            if clips_with_delays:
                min_delay = min(d for _, d in clips_with_delays)
                aligned_clips = []
                for i, (arr, d) in enumerate(clips_with_delays):
                    aligned_delay = max(0.0, d - min_delay)
                    aligned_clips.append((arr, aligned_delay))
                    clips_metadata[i]['delay_s'] = aligned_delay
                clips_with_delays = aligned_clips

            # Enforce exact target duration
            target_duration = float(config.get('batch', {}).get('duration', 10.0))

            # Mix and save combined audio
            mixed_audio = mix_audio_clips(clips_with_delays, target_duration_s=target_duration)
            mixed_audio_path = os.path.join(scene_dir, "mixed_audio.wav")
            save_audio(mixed_audio, mixed_audio_path)

            # NEW: Auto-generate spectrogram for mixed scene
            mixed_spec_path = os.path.join(scene_dir, "mixed_audio_spec.png")
            save_spectrogram_to_file(mixed_audio, SR, "mixed_audio.wav", mixed_spec_path)

            # Save combined plot
            save_combined_path_plot(scene_paths_data, scene_dir, "scene", lane_width=lane_width, 
                                    include_opposite=include_opposite, road_y_center=road_y_center, 
                                    observer_pos=observer_pos, absolute=True)

            # Save metadata (with validation results if enabled)
            metadata = {
                'scene_id': scene_id,
                'num_vehicles': num_vehicles,
                'path_type': scene_path_type if scene_path_mixing == 'same' else 'mixed',
                'vehicles': clips_metadata,
                'timestamp': datetime.now().isoformat()
            }

            if enable_validation:
                metadata['validation'] = {
                    'enabled': True,
                    'scene_valid': validation_results['scene_valid'],
                    'vehicles_with_violations': validation_results['vehicles_with_violations']
                }

            with open(os.path.join(scene_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            save_progress(num_datasets, scene_idx)

        elapsed_time = time.time() - start_time
        formatted_time = f"{elapsed_time:.2f}s"

        # NEW: Print validation summary
        if enable_validation:
            print("\n" + "=" * 70)
            print("BATCH VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Total Scenes: {validation_stats['total_scenes']}")
            print(f"Valid Scenes: {validation_stats['valid_scenes']} "
                  f"({validation_stats['valid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)")
            print(f"Invalid Scenes: {validation_stats['invalid_scenes']} "
                  f"({validation_stats['invalid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)")
            print()
            print(f"Total Vehicles: {validation_stats['total_vehicles']}")
            print(f"Valid Vehicles: {validation_stats['valid_vehicles']} "
                  f"({validation_stats['valid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)")
            print(f"Invalid Vehicles: {validation_stats['invalid_vehicles']} "
                  f"({validation_stats['invalid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)")
            print("=" * 70)

            # Save batch validation summary
            summary_file = os.path.join(root_dir, "validation_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("BATCH VALIDATION SUMMARY\n")
                f.write("=" * 70 + "\n")
                f.write(f"Total Scenes: {validation_stats['total_scenes']}\n")
                f.write(f"Valid Scenes: {validation_stats['valid_scenes']} "
                       f"({validation_stats['valid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)\n")
                f.write(f"Invalid Scenes: {validation_stats['invalid_scenes']} "
                       f"({validation_stats['invalid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)\n")
                f.write(f"\nTotal Vehicles: {validation_stats['total_vehicles']}\n")
                f.write(f"Valid Vehicles: {validation_stats['valid_vehicles']} "
                       f"({validation_stats['valid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)\n")
                f.write(f"Invalid Vehicles: {validation_stats['invalid_vehicles']} "
                       f"({validation_stats['invalid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)\n")

        response_data = {
            'success': True,
            'batch_id': safe_root_name,
            'root_directory': root_dir,
            'total_generated': num_datasets,
            'elapsed_time': elapsed_time,
            'formatted_time': formatted_time
        }

        if enable_validation:
            response_data['validation'] = validation_stats

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@batch_bp.route('/api/map_overlap_generate', methods=['POST'])
def map_overlap_generate():
    """Generate overlap batch based on map geometry from JSON"""
    try:
        config = request.get_json()
        start_time = time.time()
        
        map_name = config.get('map_name') # e.g. PennAve_LibertyAve_1
        num_scenes = int(config.get('batch', {}).get('total_scenes', 10))
        veh_per_scene = int(config.get('overlap', {}).get('vehicles_per_scene', 5))
        
        # Load map geometry
        map_json_path = os.path.join(os.getcwd(), "MapExtraction", f"road_edge_structure_{map_name}.json")
        if not os.path.exists(map_json_path):
            return jsonify({'error': f'Map geometry file not found: {map_json_path}'}), 400
            
        with open(map_json_path, 'r') as f:
            map_data = json.load(f)
            
        # Extract lanes or dividers
        lanes = []
        road_geom = {}
        
        if 'divider' in map_data:
            lanes.append(map_data['divider'])
            road_geom['divider'] = [[p[0]/10.0, p[1]/10.0] for p in map_data['divider']]
            
        if 'edges' in map_data:
            if 'outer' in map_data['edges']:
                road_geom['outer_edge'] = [[p[0]/10.0, p[1]/10.0] for p in map_data['edges']['outer']]
            if 'inner' in map_data['edges']:
                road_geom['inner_edge'] = [[p[0]/10.0, p[1]/10.0] for p in map_data['edges']['inner']]
                
        # If scenes_data_compatible exists, use those as actual lane trajectories
        if 'scenes_data_compatible' in map_data:
            for scene in map_data['scenes_data_compatible']:
                if 'points' in scene:
                    lanes.append(scene['points'])
        
        if not lanes:
            return jsonify({'error': 'No road lanes found in map geometry'}), 400
            
        # Create output dir
        base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
        batch_id = f"map_batch_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        root_dir = os.path.join(base_output_root, batch_id)
        os.makedirs(root_dir, exist_ok=True)
        
        selected_vehicles = config.get('vehicles', {}).get('selected', [])
        if not selected_vehicles:
            # Fallback to all vehicles
            from core.config import UPLOAD_FOLDER
            selected_vehicles = [os.path.splitext(f)[0] for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.wav', '.mp3'))]
        
        save_progress(num_scenes, 0)
        
        for scene_idx in range(1, num_scenes + 1):
            scene_id = f"{scene_idx:04d}"
            scene_dir = os.path.join(root_dir, scene_id)
            os.makedirs(scene_dir, exist_ok=True)
            
            clips_with_delays = []
            clips_metadata = []
            
            for v_idx in range(1, veh_per_scene + 1):
                vehicle_name = random.choice(selected_vehicles)
                lane_points = random.choice(lanes)
                
                # Scaling pixel to meter (10:1 ratio for physics stability)
                points_m = [[p[0]/10.0, p[1]/10.0] for p in lane_points]
                
                params = {
                    'points': points_m,
                    'speed': random.uniform(10, 30),
                    'duration': 10.0,
                    'temperature': 20,
                    'humidity': 50
                }
                
                from audio.generation import get_doppler_audio_array
                audio_arr, _, _ = get_doppler_audio_array(vehicle_name, 'map_path', params)
                
                delay = random.uniform(0, 5.0)
                clips_with_delays.append((audio_arr, delay))
                
                v_filename = f"veh_{v_idx:02d}_{vehicle_name}.wav"
                v_filepath = os.path.join(scene_dir, v_filename)
                save_audio(audio_arr, v_filepath)
                
                # Generate spectrogram for individual car
                from visualization.plot_utils import save_spectrogram_to_file
                v_spec_path = v_filepath.replace('.wav', '_spec.png')
                save_spectrogram_to_file(audio_arr, SR, v_filename, v_spec_path)
                
                clips_metadata.append({
                    'id': v_idx,
                    'vehicle': vehicle_name,
                    'filename': v_filename,
                    'spectrogram': os.path.basename(v_spec_path),
                    'delay': delay,
                    'points_count': len(points_m),
                    'parameters': params
                })
                
            mixed = mix_audio_clips(clips_with_delays)
            mixed_audio_path = os.path.join(scene_dir, "mixed.wav")
            save_audio(mixed, mixed_audio_path)
            
            # Generate spectrogram for mixed scene
            mixed_spec_path = mixed_audio_path.replace('.wav', '_spec.png')
            save_spectrogram_to_file(mixed, SR, "mixed.wav", mixed_spec_path)
            
            # Generate combined path plot with REAL road geometry
            from visualization.plot_utils import save_combined_path_plot
            scene_paths_data = [('map_path', c['parameters'], c['vehicle']) for c in clips_metadata]
            save_combined_path_plot(scene_paths_data, scene_dir, "scene", road_geometry=road_geom)
            
            with open(os.path.join(scene_dir, "metadata.json"), 'w') as f:
                json.dump({
                    'scene': scene_id, 
                    'vehicles': clips_metadata,
                    'mixed_spectrogram': os.path.basename(mixed_spec_path),
                    'combined_path_plot': "scene_combined_path.png"
                }, f, indent=2)
                
            save_progress(num_scenes, scene_idx)
            
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'root_directory': root_dir,
            'total_generated': num_scenes
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@batch_bp.route('/api/progress', methods=['GET'])
def get_progress():
    """Get current generation progress"""
    return jsonify(load_progress())


@batch_bp.route('/api/get_outline_image')
def get_outline_image():
    """Serve local outline PNGs to the frontend"""
    filename = request.args.get('filename')
    if not filename:
        return "Filename required", 400
    
    # Check if filename is safe
    if '..' in filename:
        return "Invalid filename", 400
        
    outputs_dir = os.path.join(os.getcwd(), "MapExtraction", "outputs")
    # Also check tests if needed
    if not os.path.exists(os.path.join(outputs_dir, filename)):
        tests_dir = os.path.join(os.getcwd(), "MapExtraction", "tests")
        if os.path.exists(os.path.join(tests_dir, filename)):
            return send_from_directory(tests_dir, filename)
            
    return send_from_directory(outputs_dir, filename)


@batch_bp.route('/api/list_outlines', methods=['GET'])
def list_outlines():
    """List PNG files in MapExtraction/outputs"""
    try:
        outputs_dir = os.path.join(os.getcwd(), "MapExtraction", "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)
        files = [f for f in os.listdir(outputs_dir) if f.endswith('.png')]
        return jsonify({'files': sorted(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@batch_bp.route('/api/convert_outline', methods=['POST'])
def convert_outline():
    """Convert PNG outline to JSON and generate visualization"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400

        input_path = os.path.join(os.getcwd(), "MapExtraction", "outputs", filename)
        output_json_name = f"road_edge_structure_{os.path.splitext(filename)[0]}.json"
        output_json_path = os.path.join(os.getcwd(), "MapExtraction", output_json_name)
        
        # Handle optional erasure mask from frontend (base64)
        data = request.get_json()
        erasure_mask_b64 = data.get('erasure_mask')
        mask_path = None
        
        if erasure_mask_b64:
            try:
                # Remove header if present (e.g. "data:image/png;base64,")
                if ',' in erasure_mask_b64:
                    erasure_mask_b64 = erasure_mask_b64.split(',')[1]
                
                mask_data = base64.b64decode(erasure_mask_b64)
                mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
                mask_path = os.path.join(os.getcwd(), "MapExtraction", "outputs", mask_filename)
                
                with open(mask_path, 'wb') as f:
                    f.write(mask_data)
                print(f"Saved erasure mask to {mask_path}")
            except Exception as e:
                print(f"Failed to save erasure mask: {e}")
                # Don't fail the whole request if mask saving fails, just proceed without it
                mask_path = None

        # Run conversion as a subprocess
        import subprocess
        import sys
        python_exe = sys.executable
        script_path = os.path.join(os.getcwd(), "MapExtraction", "outline_to_json.py")
        
        try:
            cmd = [python_exe, script_path, input_path, output_json_path]
            if mask_path:
                cmd.append(mask_path)
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Outline conversion output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'Conversion failed: {e.stderr}'}), 500
        except Exception as e:
            return jsonify({'error': f'Subprocess error: {str(e)}'}), 500

        # Load the generated data to return to the frontend
        try:
            with open(output_json_path, 'r') as f:
                result_data = json.load(f)
        except Exception as e:
            return jsonify({'error': f'Failed to load generated JSON: {str(e)}'}), 500
        
        # The visualization is saved by the function (we'll update it to do so)
        vis_filename = f"vis_{os.path.splitext(filename)[0]}.png"
        vis_path = os.path.join(os.getcwd(), "MapExtraction", "outputs", vis_filename)
        
        return jsonify({
            'success': True,
            'json_path': output_json_path,
            'vis_url': f"/map_outputs/{vis_filename}",
            'data': result_data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@batch_bp.route('/api/generate_road_swarm', methods=['POST'])
def generate_road_swarm():
    """Generate a multi-vehicle swarm that follows a custom road path with boundary clamping"""
    try:
        config = request.get_json()
        map_id = config.get('map_id')
        path_points = config.get('path_points', []) # List of {x, y} in pixels
        observer_pos_px = config.get('observer_pos') # {x, y} in pixels
        num_vehicles_range = config.get('num_vehicles_range', [3, 3])
        speed_range = config.get('speed_range', [20.0, 30.0])
        lane_width = float(config.get('lane_width', 4.0))
        max_stagger = float(config.get('max_stagger', 1.0))
        include_opposite = config.get('include_opposite', True)
        duration = float(config.get('duration', 5.0))
        selected_vehicles = config.get('vehicles', {}).get('selected', [])
        
        erasure_mask_b64 = config.get('erasure_mask')
        shade_mask_b64 = config.get('shade_mask')

        # 0. Path Extraction from Shade Mask (if path_points empty)
        if (not path_points or len(path_points) < 2) and shade_mask_b64:
            try:
                # Save mask to temp file
                import subprocess
                import tempfile
                import io
                from PIL import Image
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
                    mask_data_b64 = shade_mask_b64
                    if ',' in mask_data_b64: mask_data_b64 = mask_data_b64.split(',')[1]
                    mask_data = base64.b64decode(mask_data_b64)
                    tmp_mask.write(mask_data)
                    tmp_mask_path = tmp_mask.name
                
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
                    tmp_json_path = tmp_json.name
                
                # Run extraction script
                import sys
                python_exe = sys.executable
                script_path = os.path.join(os.getcwd(), "MapExtraction", "extract_path.py")
                
                cmd = [python_exe, script_path, tmp_mask_path, tmp_json_path]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Load the path
                with open(tmp_json_path, 'r') as f:
                    path_points = json.load(f)
                
                # Cleanup
                if os.path.exists(tmp_mask_path): os.remove(tmp_mask_path)
                if os.path.exists(tmp_json_path): os.remove(tmp_json_path)
                
                print(f"Extraction successful: {len(path_points)} points.")
            except Exception as e:
                print(f"Path extraction failed: {e}")
                traceback.print_exc()
                # If extraction fails, we still need path_points to be valid below

        if not path_points or len(path_points) < 2:
            return jsonify({'error': 'Please shade a road path on the map or select a simulation path.'}), 400

        # 1. Load map geometry and masks for boundary clamping
        from PIL import Image
        import io
        import numpy as np

        road_edges = []
        
        # A. Load JSON edges if available
        map_json_path = os.path.join(os.getcwd(), "MapExtraction", f"road_edge_structure_{map_id}.json")
        if os.path.exists(map_json_path):
            with open(map_json_path, 'r') as f:
                map_data = json.load(f)
                if 'edges' in map_data:
                    for e_type in ['outer', 'inner', 'all_segments']:
                        if e_type in map_data['edges']:
                            if e_type == 'all_segments':
                                for seg in map_data['edges'][e_type]:
                                    road_edges.extend([[p[0]/10.0, p[1]/10.0] for p in seg])
                            else:
                                road_edges.extend([[p[0]/10.0, p[1]/10.0] for p in map_data['edges'][e_type]])

        # B. Process Shading Mask as primary/additional road boundary
        if shade_mask_b64:
            if ',' in shade_mask_b64: shade_mask_b64 = shade_mask_b64.split(',')[1]
            mask_data = base64.b64decode(shade_mask_b64)
            mask_img = Image.open(io.BytesIO(mask_data)).convert('L')
            mask_arr = np.array(mask_img)
            
            # Find boundaries: pixels with value > 0 that have a 0-neighbor
            # We use a simple shift-based edge detection
            binary = (mask_arr > 50).astype(np.uint8)
            edges = binary ^ np.roll(binary, 1, axis=0) | binary ^ np.roll(binary, 1, axis=1) | \
                    binary ^ np.roll(binary, -1, axis=0) | binary ^ np.roll(binary, -1, axis=1)
            
            y_coords, x_coords = np.where(edges & binary) # Only boundaries inside the shaded area
            for x, y in zip(x_coords, y_coords):
                road_edges.append([x/10.0, y/10.0])

        # C. Filter edges using Erasure Mask if provided
        if erasure_mask_b64 and road_edges:
            if ',' in erasure_mask_b64: erasure_mask_b64 = erasure_mask_b64.split(',')[1]
            mask_data = base64.b64decode(erasure_mask_b64)
            e_mask_img = Image.open(io.BytesIO(mask_data)).convert('L')
            e_mask_arr = np.array(e_mask_img)
            
            filtered_edges = []
            for p in road_edges:
                px, py = int(p[0]*10), int(p[1]*10)
                if 0 <= py < e_mask_arr.shape[0] and 0 <= px < e_mask_arr.shape[1]:
                    if e_mask_arr[py, px] < 128: # Not erased
                        filtered_edges.append(p)
                else:
                    filtered_edges.append(p)
            road_edges = filtered_edges

        tree = None
        if road_edges:
            from scipy.spatial import KDTree
            tree = KDTree(road_edges)

        # 2. Normalize and interpolate the base path
        base_points = np.array([[p['x']/10.0, p['y']/10.0] for p in path_points])
        
        obs_m = (0, 0)
        if observer_pos_px:
            obs_m = (observer_pos_px['x']/10.0, observer_pos_px['y']/10.0)
            
        # 3. Generate swarm(s)
        from audio.generation import get_doppler_audio_array, mix_audio_clips
        from audio.audio_utils import save_audio
        
        batch_id = config.get('batch_name') or f"road_swarm_{map_id}_{int(time.time())}"
        total_scenes = int(config.get('total_scenes', 1))
        batch_dir = os.path.join(OUTPUT_FOLDER, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        if not selected_vehicles:
            from core.config import UPLOAD_FOLDER
            selected_vehicles = [os.path.splitext(f)[0] for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.wav', '.mp3'))]
        
        if not selected_vehicles:
            return jsonify({'error': 'No vehicle sounds available in static/vehicle_sounds'}), 400

        # Choose shift method
        shift_method = config.get('shift_method', 'resample')
        
        # Initialize progress for batch Mode
        if total_scenes > 1:
            save_progress(total_scenes, 0)
            
        all_scenes_info = []

        for scene_idx in range(total_scenes):
            clips_with_delays = []
            swarm_metadata = []
            
            num_vehicles = random.randint(num_vehicles_range[0], num_vehicles_range[1])
            for i in range(num_vehicles):
                vehicle_name = random.choice(selected_vehicles)
                
                # 1. Handle direction and stagger
                is_opposite = include_opposite and (random.random() > 0.5)
                # Random speed from range
                curr_speed = random.uniform(speed_range[0], speed_range[1])
                
                stagger = random.uniform(0, max_stagger) if i > 0 else 0
                
                # Start with a random lateral target (to simulate lanes)
                target_offset = random.uniform(-lane_width, lane_width)
                
                active_base_points = base_points if not is_opposite else base_points[::-1]
                shifted_path = []
                for j in range(len(active_base_points)):
                    p = active_base_points[j]
                    
                    # Perpendicular direction
                    if j < len(active_base_points) - 1:
                        p_next = active_base_points[j+1]
                        dx, dy = p_next[0] - p[0], p_next[1] - p[1]
                    else:
                        p_prev = active_base_points[j-1]
                        dx, dy = p[0] - p_prev[0], p[1] - p_prev[1]
                    
                    mag = np.sqrt(dx**2 + dy**2) + 1e-9
                    nx, ny = -dy/mag, dx/mag
                    
                    # Dynamic boundary check
                    actual_offset = target_offset
                    if tree:
                        # Check distance from original path to the edge
                        dist_to_edge, _ = tree.query(p)
                        # Stay at least 0.4m away from edges
                        max_safe = max(0.1, dist_to_edge - 0.4)
                        actual_offset = np.clip(target_offset, -max_safe, max_safe)
                    
                    shifted_path.append([p[0] + actual_offset * nx, p[1] + actual_offset * ny])

                # 2. Speed and delay
                params = {
                    'points': shifted_path,
                    'speed': curr_speed,
                    'duration': duration,
                    'observer_pos': obs_m
                }
                
                phase_offset = random.uniform(0, 30.0) # Decorrelate same-type vehicles
                pitch_shift = random.uniform(0.9, 1.1) # Tonal decorrelation
                audio_arr, _, _ = get_doppler_audio_array(vehicle_name, 'map_trajectory', params, method=shift_method, phase_offset=phase_offset, pitch_shift=pitch_shift)
                clips_with_delays.append((audio_arr, stagger))
                
                swarm_metadata.append({
                    'id': i,
                    'vehicle': vehicle_name,
                    'speed': curr_speed,
                    'delay': stagger,
                    'path': [[p[0]*10, p[1]*10] for p in shifted_path], # Return pixels for visualization
                    'is_opposite': is_opposite
                })

            # 3. Mix and Save Scene
            mixed_audio = mix_audio_clips(clips_with_delays)
            scene_name = f"scene_{scene_idx:03d}" if total_scenes > 1 else "road_swarm_mixed"
            mixed_filename = f"{scene_name}.wav"
            mixed_path = os.path.join(batch_dir, mixed_filename)
            save_audio(mixed_audio, mixed_path)

            # Save metadata for this scene
            scene_meta_path = os.path.join(batch_dir, f"{scene_name}.json")
            scene_data = {
                'scene_id': scene_idx,
                'duration': duration,
                'speed_range': speed_range,
                'swarm': swarm_metadata,
                'vehicles': swarm_metadata,
                'timestamp': datetime.now().isoformat(),
                'audio_url': f"/static/batch_outputs/{batch_id}/{mixed_filename}"
            }
            with open(scene_meta_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
            
            all_scenes_info.append(scene_data)
            
            # Update progress
            if total_scenes > 1:
                save_progress(total_scenes, scene_idx + 1)

        if total_scenes == 1:
            return jsonify({
                'success': True,
                'mixed_audio_file': os.path.join(batch_dir, "road_swarm_mixed.wav"),
                'session_dir': batch_dir,
                'download_url': all_scenes_info[0]['audio_url'],
                'swarm': {
                    'metadata': all_scenes_info[0]['vehicles'],
                    'duration': duration,
                    'observer_pos': obs_m
                }
            })
        else:
            return jsonify({
                'success': True,
                'batch_id': batch_id,
                'batch_dir': batch_dir,
                'total_scenes': total_scenes,
                'message': f"Generated {total_scenes} scenes in {batch_id}"
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
