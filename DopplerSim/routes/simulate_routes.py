import os
import numpy as np
import librosa
import traceback
import io
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file
import soundfile as sf
from scipy import signal

from audio.audio_utils import (
    apply_doppler_to_audio_fixed,
    save_audio,
    extend_audio_with_overlap,
    SR
)
from core.config import UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER, SINGLE_OUTPUT_FOLDER
from audio.generation import generate_single_clip

simulate_bp = Blueprint('simulate', __name__)

def _apply_subtle_air_noise(audio, sr, strength_pct, center_freq_hz=1200.0):
    """
    Add a light, natural air-noise layer.
    strength_pct in [0, 100], intentionally mapped to a gentle mix.
    """
    if audio is None or len(audio) == 0:
        return audio

    strength = float(np.clip(strength_pct, 0.0, 100.0))
    if strength <= 0.0:
        return audio

    n = len(audio)
    white = np.random.normal(0.0, 1.0, n).astype(np.float32)

    # Shape noise in frequency-domain with:
    # 1) broadband floor (fills spectrogram to the top),
    # 2) user-centered emphasis (so the frequency control remains meaningful).
    nyq = 0.5 * float(sr)
    center_hz = float(np.clip(center_freq_hz, 80.0, min(5000.0, nyq * 0.95)))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    spectrum = np.fft.rfft(white)

    # Wide Gaussian emphasis around user-selected center.
    sigma_hz = max(250.0, 0.32 * center_hz)
    peak = np.exp(-0.5 * ((freqs - center_hz) / sigma_hz) ** 2)

    # Broadband floor stays present across nearly all bins, with a tiny high-frequency
    # boost so upper bins are not visually empty in spectrograms.
    broad_floor = 0.42 + 0.20 * np.sqrt(np.clip(freqs / max(nyq, 1e-6), 0.0, 1.0))

    # Keep centered control influential, but avoid local-only energy islands.
    profile = broad_floor + 0.60 * peak

    # Very gentle top-end rolloff for naturalness (not too aggressive).
    natural_rolloff = 1.0 / np.sqrt(1.0 + (freqs / 10000.0) ** 2)
    profile *= natural_rolloff
    profile = np.clip(profile, 0.0, None)

    air = np.fft.irfft(spectrum * profile, n=n).astype(np.float32)

    # Very slow amplitude drift for a more natural bed.
    t = np.linspace(0.0, n / float(sr), n, endpoint=False)
    drift = 0.88 + 0.12 * np.sin(2.0 * np.pi * 0.12 * t + np.random.uniform(0, 2 * np.pi))
    air *= drift.astype(np.float32)

    rms = np.sqrt(np.mean(air**2) + 1e-12)
    air /= rms

    # Keep subtle, but ensure visible broadband bed at practical strengths.
    mix_gain = 0.07 * ((strength / 100.0) ** 1.25)
    mixed = audio.astype(np.float32) + (mix_gain * air)
    return np.clip(mixed, -1.0, 1.0)


@simulate_bp.route('/simulate', methods=['POST'])
def simulate_single():
    """
    Single-clip Doppler simulation endpoint for the single-clip UI.
    Returns a WAV file blob that the frontend plays directly.
    """
    try:
        # Basic inputs from form
        path_type = request.form.get('path', 'straight')
        vehicle_type = request.form.get('vehicle_type', 'car')

        # Match single-mode UI "Output Duration" (`audio_duration`, default 5 in template).
        _dur_raw = request.form.get('audio_duration')
        try:
            duration = float(_dur_raw) if _dur_raw not in (None, '') else 10.0
        except (TypeError, ValueError):
            duration = 10.0
        duration = float(max(0.5, min(300.0, duration)))

        # Use lower-case name to match uploaded vehicle files (car.wav, train.wav, etc.)
        vehicle_name = vehicle_type.lower()

        # Common parameters
        params = {
            'duration': duration,
            # Avoid leading silence in single-sim downloads/playback.
            'apply_propagation_delay': False
        }

        # Path-specific parameters (manual mode – you control signs here)
        if path_type == 'straight':
            speed = float(request.form.get('speed', 20.0))
            h = float(request.form.get('h', 10.0))       # closest distance
            angle = float(request.form.get('angle', 0.0))

            params['speed'] = speed
            params['distance'] = h
            params['angle'] = angle

        elif path_type == 'parabola':
            speed = float(request.form.get('speed', 15.0))
            a = float(request.form.get('a', 0.1))
            h = float(request.form.get('h', 10.0))

            params['speed'] = speed
            params['a'] = a
            params['h'] = h
            # store something reasonable for filename/stats distance
            params['distance'] = h

        elif path_type == 'bezier':
            speed = float(request.form.get('speed', 20.0))

            params['speed'] = speed
            params['x0'] = float(request.form.get('x0', -30))
            params['y0'] = float(request.form.get('y0', 20))
            params['x1'] = float(request.form.get('x1', -10))
            params['y1'] = float(request.form.get('y1', 5))
            params['x2'] = float(request.form.get('x2', 10))
            params['y2'] = float(request.form.get('y2', 5))
            params['x3'] = float(request.form.get('x3', 30))
            params['y3'] = float(request.form.get('y3', 20))
            # nominal distance just for filename
            params['distance'] = 10.0

        else:
            return jsonify({'error': f'Unknown path type: {path_type}'}), 400

        # Minimal config reused from batch code
        config = {
            'output': {'format': 'wav'}
        }

        single_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        index = 1

        result = generate_single_clip(
            vehicle_name=vehicle_name,
            path_type=path_type,
            params=params,
            output_dir=SINGLE_OUTPUT_FOLDER,
            batch_id=single_id,
            index=index,
            config=config
        )

        # generate_single_clip stores outputs inside sample_<id>/Common.
        sample_dir = result.get('sample_dir')
        file_path = None
        if sample_dir:
            candidate = os.path.join(SINGLE_OUTPUT_FOLDER, sample_dir, 'Common', result['filename'])
            if os.path.exists(candidate):
                file_path = candidate

        # Fallback for any legacy layout that writes directly to SINGLE_OUTPUT_FOLDER.
        if file_path is None:
            candidate = os.path.join(SINGLE_OUTPUT_FOLDER, result['filename'])
            if os.path.exists(candidate):
                file_path = candidate

        if file_path is None:
            return jsonify({'error': 'Audio generation failed - output file not created'}), 500

        add_air_noise = str(request.form.get('add_air_noise', '')).lower() in ('1', 'true', 'on', 'yes')
        air_noise_strength = float(request.form.get('air_noise_strength', 8.0))
        air_noise_frequency_hz = float(request.form.get('air_noise_frequency_hz', 1200.0))

        if not add_air_noise:
            return send_file(file_path, mimetype='audio/wav')

        audio, sr = librosa.load(file_path, sr=SR, mono=True)
        audio_with_noise = _apply_subtle_air_noise(
            audio, sr, air_noise_strength, air_noise_frequency_hz
        )
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_with_noise, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return send_file(wav_buffer, mimetype='audio/wav', download_name='single_simulation.wav')

    except FileNotFoundError as e:
        return jsonify({'error': f'Audio file not found: {str(e)}'}), 404
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Simulation error: {str(e)}'}), 500


@simulate_bp.route('/api/simulate_intersection', methods=['POST'])
def simulate_intersection():
    """
    Multi-vehicle intersection Doppler simulation endpoint.
    Receives JSON with intersection layout and vehicle list.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON body'}), 400

        vehicles_config = data.get('vehicles', [])
        intersection_cfg = data.get('intersection', {})
        
        obs_pos = (
            float(intersection_cfg.get('obs_x', 10.0)),
            float(intersection_cfg.get('obs_y', 10.0))
        )
        duration = float(data.get('duration', 10.0))
        c_sound = float(data.get('c_sound', 343.0))

        if not vehicles_config:
            return jsonify({'error': 'No vehicles provided'}), 400

        from physics.intersection import calculate_intersection_doppler
        from audio.generation import mix_audio_clips
        from audio.audio_utils import extend_audio_with_overlap

        # Calculate physics for all vehicles
        physics_results = calculate_intersection_doppler(
            vehicles_config,
            observer_pos=obs_pos,
            duration_s=duration,
            c_sound=c_sound
        )

        mixed_clips = []
        vehicle_meta = {}

        for v_cfg in vehicles_config:
            v_id = v_cfg['id']
            v_type = v_cfg.get('type', 'car').lower()
            
            # Find audio file
            vehicle_file = None
            folders_to_check = [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]
            
            # 1. Exact match
            for folder in folders_to_check:
                for ext in ['.wav', '.mp3']:
                    test_path = os.path.join(folder, f'{v_type}{ext}')
                    if os.path.exists(test_path):
                        vehicle_file = test_path
                        break
                if vehicle_file: break
            
            # 2. Starts with (e.g. car_1.wav for type='car')
            if not vehicle_file:
                for folder in folders_to_check:
                    if os.path.exists(folder):
                        files = [f for f in os.listdir(folder) if f.lower().startswith(v_type)]
                        if files:
                            vehicle_file = os.path.join(folder, files[0])
                            break
            
            # 3. Global fallback
            if not vehicle_file:
                for folder in folders_to_check:
                    if os.path.exists(folder):
                        files = [f for f in os.listdir(folder) if f.lower().endswith(('.wav', '.mp3'))]
                        if files:
                            vehicle_file = os.path.join(folder, files[0])
                            break

            if not vehicle_file:
                continue

            # Load and process audio
            audio_full, sr = librosa.load(vehicle_file, sr=SR, mono=True)
            audio = extend_audio_with_overlap(audio_full, duration * 2.0, SR)
            
            v_physics = physics_results[v_id]
            doppler_audio = apply_doppler_to_audio_fixed(
                audio, v_physics['freq_ratios'], v_physics['amplitudes']
            )
            
            # Ensure exact length
            target_samples = int(SR * duration)
            if len(doppler_audio) > target_samples:
                doppler_audio = doppler_audio[:target_samples]
            else:
                doppler_audio = np.pad(doppler_audio, (0, target_samples - len(doppler_audio)))
                
            mixed_clips.append((doppler_audio, 0.0)) # All vehicles share the same timeline
            
            # Prepare metadata for frontend visualization
            vehicle_meta[v_id] = {
                'positions': v_physics['positions'].tolist(), # [ [x...], [y...] ]
                'freq_ratios': v_physics['freq_ratios'].tolist(),
                'type': v_type
            }

        if not mixed_clips:
            return jsonify({'error': 'Failed to generate any vehicle audio'}), 500

        # Mix all clips
        final_audio = mix_audio_clips(mixed_clips)
        
        # Save result
        sim_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"intersection_{sim_id}.wav"
        filepath = os.path.join(SINGLE_OUTPUT_FOLDER, filename)
        save_audio(final_audio, filepath)
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/single_outputs/{filename}',
            'physics': vehicle_meta,
            'settings': {
                'obs_pos': obs_pos,
                'duration': duration
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Simulation error: {str(e)}'}), 500
