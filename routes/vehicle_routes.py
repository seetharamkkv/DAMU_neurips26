import os
import uuid
import time
import librosa
import numpy as np

from flask import Blueprint, request, jsonify

from audio.audio_utils import SR
from core.config import UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER, SPECTROGRAM_FOLDER
from visualization.plot_utils import save_spectrogram_to_file, save_audio_comparison_plot

import soundfile as sf

vehicle_bp = Blueprint('vehicle', __name__)


@vehicle_bp.route('/api/upload_vehicle', methods=['POST'])
def upload_vehicle():
    """Upload vehicle audio file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        vehicle_name = request.form.get('vehicle_name', 'unnamed')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate audio file
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            return jsonify({'error': 'Invalid audio format. Use WAV, MP3, OGG, or FLAC'}), 400

        # Save temporarily to check duration
        temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{uuid.uuid4()}.wav')
        file.save(temp_path)

        # Load and check duration
        try:
            audio, sr = librosa.load(temp_path, sr=SR, mono=True)
            duration = len(audio) / SR

            if not (2.5 <= duration <= 3.5):
                os.remove(temp_path)
                return jsonify({'error': f'Audio duration must be 3±0.5 seconds. Got {duration:.2f}s'}), 400

            # Save with proper name
            safe_name = "".join(c for c in vehicle_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f'{safe_name}.wav'
            final_path = os.path.join(UPLOAD_FOLDER, filename)

            # Convert to WAV format
            sf.write(final_path, audio, SR)
            os.remove(temp_path)

            return jsonify({
                'success': True,
                'filename': filename,
                'vehicle_name': safe_name,
                'duration': duration
            })

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/list_vehicles', methods=['GET'])
def list_vehicles():
    """List all vehicle sounds from static/vehicle_sounds and static/drone_sounds"""
    try:
        # Optional filter by source type
        source_filter = request.args.get('source', 'all')  # 'vehicle', 'drone', or 'all'

        vehicles = []

        # Scan both directories
        folders_to_scan = [
            (UPLOAD_FOLDER, 'vehicle'),
            (DRONE_SOUNDS_FOLDER, 'drone')
        ]

        for folder, source_type in folders_to_scan:
            # Skip if filtering and this source doesn't match
            if source_filter != 'all' and source_filter != source_type:
                continue

            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        filepath = os.path.join(folder, filename)
                        try:
                            audio, sr = librosa.load(filepath, sr=SR, mono=True)
                            duration = len(audio) / SR
                            # Remove any audio extension
                            vehicle_name = filename
                            for ext in ['.wav', '.mp3', '.ogg', '.flac', '.WAV', '.MP3', '.OGG', '.FLAC']:
                                vehicle_name = vehicle_name.replace(ext, '')
                            vehicles.append({
                                'name': vehicle_name,
                                'filename': filename,
                                'duration': round(duration, 2),
                                'source': source_type,
                                'folder': folder
                            })
                        except Exception:
                            pass

        return jsonify({'vehicles': vehicles})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/delete_vehicle/<filename>', methods=['DELETE'])
def delete_vehicle(filename):
    """Delete a vehicle sound"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/generate_spectrogram', methods=['POST'])
def generate_spectrogram():
    """Generate a spectrogram PNG for a given vehicle sound"""
    try:
        config = request.get_json()
        vehicle_name = config.get('vehicle_name')
        source = config.get('source', 'all')
        max_y_freq = 1250  # Enforced upper frequency limit for all spectrograms

        if not vehicle_name:
            return jsonify({'error': 'No vehicle name provided'}), 400

        # Find vehicle file
        vehicle_file = None
        folders_to_check = []
        if source == 'vehicle' or source == 'car':
            folders_to_check = [UPLOAD_FOLDER]
        elif source == 'drone':
            folders_to_check = [DRONE_SOUNDS_FOLDER]
        else:
            folders_to_check = [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]

        for folder in folders_to_check:
            for ext in ['.wav', '.mp3', '.ogg', '.flac']:
                test_path = os.path.join(folder, f'{vehicle_name}{ext}')
                if os.path.exists(test_path):
                    vehicle_file = test_path
                    break
            if vehicle_file:
                break

        if not vehicle_file:
            return jsonify({'error': f"Vehicle sound '{vehicle_name}' not found"}), 404

        # Load audio
        y, sr = librosa.load(vehicle_file, sr=SR)

        # Save to PNG
        file_id = f"{vehicle_name}_{int(time.time())}"
        plot_filename = f"spectrogram_{file_id}.png"
        plot_path = os.path.join(SPECTROGRAM_FOLDER, plot_filename)

        save_spectrogram_to_file(
            y, sr, f'Spectrogram: {vehicle_name}', plot_path,
            max_y_freq=max_y_freq, include_amplitude_bar=True
        )

        return jsonify({
            'success': True,
            'spectrogram_url': f'/static/spectrograms/{plot_filename}'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/upload_generate_spectrogram', methods=['POST'])
def upload_generate_spectrogram():
    """Upload an audio file and generate a spectrogram"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        max_y_freq = 1250  # Enforced upper frequency limit for all spectrograms

        # Save temporarily
        temp_filename = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(SPECTROGRAM_FOLDER, temp_filename)
        file.save(temp_path)

        # Load and generate
        y, sr = librosa.load(temp_path, sr=SR)

        plot_filename = f"spectrogram_{int(time.time())}.png"
        plot_path = os.path.join(SPECTROGRAM_FOLDER, plot_filename)

        save_spectrogram_to_file(
            y, sr, f'Spectrogram: {file.filename}', plot_path,
            max_y_freq=max_y_freq, include_amplitude_bar=True
        )

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': True,
            'spectrogram_url': f'/static/spectrograms/{plot_filename}'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/compare_audio_clips', methods=['POST'])
def compare_audio_clips():
    """Upload two clips and return side-by-side comparative analysis."""
    temp_paths = []
    try:
        file_a = request.files.get('file_a')
        file_b = request.files.get('file_b')
        if file_a is None or file_b is None:
            return jsonify({'error': 'Please upload both file_a and file_b'}), 400
        if file_a.filename == '' or file_b.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400

        max_y_freq = 1250  # Enforced upper frequency limit for all spectrograms

        temp_a = os.path.join(SPECTROGRAM_FOLDER, f"cmp_a_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav")
        temp_b = os.path.join(SPECTROGRAM_FOLDER, f"cmp_b_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav")
        file_a.save(temp_a)
        file_b.save(temp_b)
        temp_paths.extend([temp_a, temp_b])

        y_a, _ = librosa.load(temp_a, sr=SR, mono=True)
        y_b, _ = librosa.load(temp_b, sr=SR, mono=True)
        if len(y_a) == 0 or len(y_b) == 0:
            return jsonify({'error': 'One of the uploaded files appears empty or unreadable'}), 400

        n_fft = 2048
        hop_length = 256

        rms_a = librosa.feature.rms(y=y_a, frame_length=n_fft, hop_length=hop_length)[0]
        rms_b = librosa.feature.rms(y=y_b, frame_length=n_fft, hop_length=hop_length)[0]
        n_env = min(len(rms_a), len(rms_b))
        rms_a = rms_a[:n_env]
        rms_b = rms_b[:n_env]

        rms_a_norm = rms_a / (np.max(rms_a) + 1e-9)
        rms_b_norm = rms_b / (np.max(rms_b) + 1e-9)
        amp_overlap = float(np.sum(np.minimum(rms_a_norm, rms_b_norm)) / (np.sum(np.maximum(rms_a_norm, rms_b_norm)) + 1e-9) * 100.0)

        if n_env > 1 and (np.std(rms_a_norm) > 1e-9) and (np.std(rms_b_norm) > 1e-9):
            env_corr = float(np.corrcoef(rms_a_norm, rms_b_norm)[0, 1])
        else:
            env_corr = 0.0
        env_corr_pct = float(np.clip((env_corr + 1.0) * 50.0, 0.0, 100.0))

        stft_a = np.abs(librosa.stft(y_a, n_fft=n_fft, hop_length=hop_length))
        stft_b = np.abs(librosa.stft(y_b, n_fft=n_fft, hop_length=hop_length))
        spec_a = np.mean(stft_a, axis=1)
        spec_b = np.mean(stft_b, axis=1)
        spec_a_norm = spec_a / (np.sum(spec_a) + 1e-9)
        spec_b_norm = spec_b / (np.sum(spec_b) + 1e-9)
        spectral_overlap = float(np.sum(np.minimum(spec_a_norm, spec_b_norm)) * 100.0)

        zcr_a = float(np.mean(librosa.feature.zero_crossing_rate(y_a)))
        zcr_b = float(np.mean(librosa.feature.zero_crossing_rate(y_b)))
        centroid_a = float(np.mean(librosa.feature.spectral_centroid(y=y_a, sr=SR)))
        centroid_b = float(np.mean(librosa.feature.spectral_centroid(y=y_b, sr=SR)))
        peak_a = float(np.max(np.abs(y_a)))
        peak_b = float(np.max(np.abs(y_b)))
        rms_global_a = float(np.sqrt(np.mean(y_a ** 2)))
        rms_global_b = float(np.sqrt(np.mean(y_b ** 2)))

        freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
        dom_freq_a = float(freqs[int(np.argmax(spec_a))]) if len(spec_a) else 0.0
        dom_freq_b = float(freqs[int(np.argmax(spec_b))]) if len(spec_b) else 0.0

        overall_similarity = float(np.clip((spectral_overlap * 0.55) + (amp_overlap * 0.30) + (env_corr_pct * 0.15), 0.0, 100.0))

        plot_filename = f"comparison_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        plot_path = os.path.join(SPECTROGRAM_FOLDER, plot_filename)
        ok = save_audio_comparison_plot(
            y_a, y_b, SR,
            f"A: {file_a.filename}",
            f"B: {file_b.filename}",
            plot_path,
            max_y_freq=max_y_freq
        )
        if not ok:
            return jsonify({'error': 'Failed to generate comparison plot'}), 500

        return jsonify({
            'success': True,
            'comparison_plot_url': f'/static/spectrograms/{plot_filename}',
            'metrics': {
                'duration_a_sec': float(len(y_a) / SR),
                'duration_b_sec': float(len(y_b) / SR),
                'rms_a': rms_global_a,
                'rms_b': rms_global_b,
                'peak_a': peak_a,
                'peak_b': peak_b,
                'zcr_a': zcr_a,
                'zcr_b': zcr_b,
                'spectral_centroid_a_hz': centroid_a,
                'spectral_centroid_b_hz': centroid_b,
                'dominant_freq_a_hz': dom_freq_a,
                'dominant_freq_b_hz': dom_freq_b,
                'spectral_overlap_percent': spectral_overlap,
                'amplitude_overlap_percent': amp_overlap,
                'envelope_correlation_percent': env_corr_pct,
                'overall_similarity_percent': overall_similarity
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        for p in temp_paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
