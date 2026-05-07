import os
import re
import time
import json
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Blueprint, request, jsonify
from audio.audio_utils import SR
from visualization.plot_utils import save_automated_comparison_plot

auto_compare_bp = Blueprint('auto_compare', __name__)

def parse_filename(filename):
    """
    Given KiaSportage_31.wav or KiaSportage_31.0.wav,
    extract carname and speed.
    """
    if not filename.lower().endswith('.wav'):
        return None, None, None
    base = filename[:-4]  # remove .wav
    parts = base.rsplit('_', 1)
    if len(parts) != 2:
        return None, None, None
    carname = parts[0]
    speed_str = parts[1]
    try:
        speed = float(speed_str)
        norm_carname = carname.replace(' ', '').replace('_', '').replace('-', '').lower()
        return carname, norm_carname, speed
    except ValueError:
        return None, None, None

@auto_compare_bp.route('/api/auto_compare/get_pairs', methods=['POST'])
def get_pairs():
    data = request.get_json() or {}
    dataset_a = data.get('dataset_a', '../Datasets/RealData')
    dataset_b = data.get('dataset_b', '../Datasets/SimulatedData')
    out_dir = data.get('out_dir', 'static/spectrograms/comparison_outputs')

    if not os.path.exists(dataset_a) or not os.path.exists(dataset_b):
        return jsonify({'error': 'One or both dataset paths do not exist.'}), 400

    # Ensure output directory exists to save missing_comparisons.txt
    os.makedirs(out_dir, exist_ok=True)

    # Scan Dataset A
    a_items = {}
    a_unparsed = []
    for root, _, files in os.walk(dataset_a):
        for f in files:
            orig_carname, norm_carname, speed = parse_filename(f)
            if norm_carname is not None and speed is not None:
                a_items[(norm_carname, speed)] = {
                    'orig_carname': orig_carname,
                    'path': os.path.join(root, f)
                }
            elif f.lower().endswith('.wav'):
                a_unparsed.append(os.path.join(root, f))

    # Scan Dataset B
    b_items = {}
    b_unparsed = []
    for root, _, files in os.walk(dataset_b):
        for f in files:
            _, norm_carname, speed = parse_filename(f)
            if norm_carname is not None and speed is not None:
                b_items[(norm_carname, speed)] = os.path.join(root, f)
            elif f.lower().endswith('.wav'):
                b_unparsed.append(os.path.join(root, f))

    # Find matches and missing
    pairs = []
    a_missing_in_b = []
    
    for (norm_carname, speed), data_a in a_items.items():
        if (norm_carname, speed) in b_items:
            path_b = b_items[(norm_carname, speed)]
            pairs.append({
                'carname': data_a['orig_carname'],
                'speed': speed,
                'path_a': data_a['path'],
                'path_b': path_b
            })
        else:
            a_missing_in_b.append(data_a['path'])
            
    b_missing_in_a = []
    for (norm_carname, speed), path_b in b_items.items():
        if (norm_carname, speed) not in a_items:
            b_missing_in_a.append(path_b)

    # Sort pairs to be deterministic
    pairs.sort(key=lambda x: (x['carname'], x['speed']))
    
    # Write missing comparisons to a text file
    missing_txt_path = os.path.join(out_dir, 'missing_comparisons.txt')
    try:
        with open(missing_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== Missing from SimulatedData (Dataset B) ===\n")
            for path in sorted(a_missing_in_b):
                f.write(path + "\n")
            
            f.write("\n=== Missing from RealData (Dataset A) ===\n")
            for path in sorted(b_missing_in_a):
                f.write(path + "\n")
                
            f.write("\n=== Unparseable WAV files in Dataset A ===\n")
            for path in sorted(a_unparsed):
                f.write(path + "\n")
                
            f.write("\n=== Unparseable WAV files in Dataset B ===\n")
            for path in sorted(b_unparsed):
                f.write(path + "\n")
    except Exception as e:
        print(f"Failed to write missing comparisons: {e}")

    return jsonify({
        'success': True,
        'total_pairs': len(pairs),
        'pairs': pairs,
        'missing_txt_path': missing_txt_path
    })


@auto_compare_bp.route('/api/auto_compare/process_pair', methods=['POST'])
def process_pair():
    data = request.get_json() or {}
    path_a = data.get('path_a')
    path_b = data.get('path_b')
    carname = data.get('carname')
    speed = data.get('speed')
    out_dir = data.get('out_dir', 'static/spectrograms/comparison_outputs')

    if not all([path_a, path_b, carname, speed is not None]):
        return jsonify({'error': 'Missing required pair information'}), 400

    try:
        y_a, _ = librosa.load(path_a, sr=SR, mono=True)
        y_b, _ = librosa.load(path_b, sr=SR, mono=True)
    except Exception as e:
        return jsonify({'error': f'Failed to load audio: {str(e)}'}), 500

    if len(y_a) == 0 or len(y_b) == 0:
        return jsonify({'error': 'One of the audio files is empty'}), 400

    # Compute metrics
    n_fft = 2048
    hop_length = 256

    rms_a = librosa.feature.rms(y=y_a, frame_length=n_fft, hop_length=hop_length)[0]
    rms_b = librosa.feature.rms(y=y_b, frame_length=n_fft, hop_length=hop_length)[0]
    n_env = min(len(rms_a), len(rms_b))
    rms_a_norm = rms_a[:n_env] / (np.max(rms_a[:n_env]) + 1e-9)
    rms_b_norm = rms_b[:n_env] / (np.max(rms_b[:n_env]) + 1e-9)

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

    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    dom_freq_a = float(freqs[int(np.argmax(spec_a))]) if len(spec_a) else 0.0
    dom_freq_b = float(freqs[int(np.argmax(spec_b))]) if len(spec_b) else 0.0

    overall_similarity = float(np.clip((spectral_overlap * 0.55) + (amp_overlap * 0.30) + (env_corr_pct * 0.15), 0.0, 100.0))

    metrics = {
        'Duration (RealData)': float(len(y_a) / SR),
        'Duration (SimulatedData)': float(len(y_b) / SR),
        'Dominant Frequency (RealData)': dom_freq_a,
        'Dominant Frequency (SimulatedData)': dom_freq_b,
        'Envelope Correlation': env_corr_pct,
        'Spectral Overlap': spectral_overlap,
        'Overall Match Score': overall_similarity
    }

    # Format the speed for display/filename. If it's an integer, format as int.
    speed_disp = int(speed) if isinstance(speed, (int, float)) and float(speed).is_integer() else speed
    filename_base = f"{carname}_{speed_disp}"
    
    # Ensure vehicle folder exists in output dir
    vehicle_out_dir = os.path.join(out_dir, carname)
    os.makedirs(vehicle_out_dir, exist_ok=True)
    
    out_path = os.path.join(vehicle_out_dir, f"{filename_base}.png")

    ok = save_automated_comparison_plot(
        y_a, y_b, SR,
        os.path.basename(path_a),
        os.path.basename(path_b),
        out_path,
        metrics,
        max_y_freq=1250
    )

    if not ok:
        return jsonify({'error': 'Failed to save comparison plot'}), 500

    # Save metrics in individual per-clip JSON
    clip_json_path = os.path.join(vehicle_out_dir, f"{filename_base}.json")
    # Convert paths to relative for portability if they are under the project root's parent
    # (assuming DopplerSim is inside vs13-model, and datasets are peers)
    cwd = os.getcwd()
    try:
        rel_path_a = os.path.relpath(path_a, cwd)
        rel_path_b = os.path.relpath(path_b, cwd)
    except Exception:
        rel_path_a = path_a
        rel_path_b = path_b

    clip_data = {
        'clip_id': filename_base,
        'vehicle': carname,
        'speed': speed_disp,
        'metrics': metrics,
        'path_a': rel_path_a,
        'path_b': rel_path_b,
        'timestamp': int(time.time())
    }
    
    try:
        with open(clip_json_path, 'w', encoding='utf-8') as f:
            json.dump(clip_data, f, indent=4)
    except Exception as e:
        print(f"Failed to save clip JSON: {e}")

    return jsonify({
        'success': True,
        'carname': carname,
        'speed': speed_disp,
        'image_path': out_path
    })



from scipy.stats import pearsonr

@auto_compare_bp.route('/api/auto_compare/finalize', methods=['POST'])
def finalize_auto_compare():
    data = request.get_json() or {}
    out_dir = data.get('out_dir', 'static/spectrograms/comparison_outputs')
    
    # Aggregate results from all per-clip JSONs
    all_entries = []
    per_car_stats = {}
    per_speed_stats = {}
    
    if not os.path.exists(out_dir):
        return jsonify({'error': 'Output directory does not exist'}), 400
        
    for root, dirs, files in os.walk(out_dir):
        if 'averages' in root: continue
        for f in files:
            if f.endswith('.json'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as jf:
                        entry = json.load(jf)
                        if 'metrics' not in entry or 'vehicle' not in entry:
                            continue
                        
                        all_entries.append(entry)
                        car_name = entry['vehicle']
                        if car_name not in per_car_stats:
                            per_car_stats[car_name] = []
                        per_car_stats[car_name].append(entry)
                        
                        speed = entry.get('speed')
                        if speed is not None:
                            if speed not in per_speed_stats:
                                per_speed_stats[speed] = []
                            per_speed_stats[speed].append(entry)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    if not all_entries:
        return jsonify({'error': 'No metrics found to aggregate'}), 400

    metrics_keys = [
        'Duration (RealData)', 'Duration (SimulatedData)', 
        'Dominant Frequency (RealData)', 'Dominant Frequency (SimulatedData)', 
        'Envelope Correlation', 'Spectral Overlap', 
        'Overall Match Score'
    ]

    def compute_detailed_stats(entries):
        if not entries: return {}
        res = {}
        for k in metrics_keys:
            vals = [e['metrics'].get(k, 0.0) for e in entries]
            res[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals))
            }
        return res

    overall_stats = compute_detailed_stats(all_entries)
    
    averages_dir = os.path.join(out_dir, 'averages')
    os.makedirs(averages_dir, exist_ok=True)

    # 1. overall_averages.txt
    with open(os.path.join(averages_dir, 'overall_averages.txt'), 'w', encoding='utf-8') as f:
        f.write("OVERALL AVERAGES (Across all cars and speeds)\n")
        f.write("--------------------------------------------------\n")
        for k in metrics_keys:
            f.write(f"{k:35}: {overall_stats[k]['mean']:.4f}\n")

    # 2. per_vehicle_averages.txt
    with open(os.path.join(averages_dir, 'per_vehicle_averages.txt'), 'w', encoding='utf-8') as f:
        f.write("PER-VEHICLE AVERAGES\n")
        f.write("--------------------------------------------------\n")
        for car in sorted(per_car_stats.keys()):
            car_avg = compute_detailed_stats(per_car_stats[car])
            f.write(f"Vehicle: {car} ({len(per_car_stats[car])} clips)\n")
            for k in metrics_keys:
                f.write(f"  - {k:33}: {car_avg[k]['mean']:.4f}\n")
            f.write("\n")

    # 3. per_speed_averages.txt
    with open(os.path.join(averages_dir, 'per_speed_averages.txt'), 'w', encoding='utf-8') as f:
        f.write("PER-SPEED AVERAGES\n")
        f.write("--------------------------------------------------\n")
        for speed in sorted(per_speed_stats.keys()):
            speed_avg = compute_detailed_stats(per_speed_stats[speed])
            f.write(f"Speed: {speed} km/h ({len(per_speed_stats[speed])} clips)\n")
            for k in metrics_keys:
                f.write(f"  - {k:33}: {speed_avg[k]['mean']:.4f}\n")
            f.write("\n")

    # 4. distribution_stats.txt (mean +/- std + histogram data)
    with open(os.path.join(averages_dir, 'distribution_stats.txt'), 'w', encoding='utf-8') as f:
        f.write("SCORE DISTRIBUTIONS (Mean +/- Std)\n")
        f.write("--------------------------------------------------\n")
        for k in metrics_keys:
            s = overall_stats[k]
            f.write(f"{k:35}: {s['mean']:.2f} +/- {s['std']:.2f} (Range: {s['min']:.2f} - {s['max']:.2f})\n")
        
        f.write("\nMATCH SCORE HISTOGRAM DATA (Bins: 0-10, 10-20, ..., 90-100)\n")
        f.write("--------------------------------------------------\n")
        scores = [e['metrics'].get('Overall Match Score', 0.0) for e in all_entries]
        hist, bins = np.histogram(scores, bins=np.linspace(0, 100, 11))
        for i in range(len(hist)):
            f.write(f"{bins[i]:3.0f} - {bins[i+1]:3.0f}% : {hist[i]:3d} clips\n")

    # 5. speed_correlation.txt
    with open(os.path.join(averages_dir, 'speed_correlation.txt'), 'w', encoding='utf-8') as f:
        f.write("CORRELATION WITH SPEED\n")
        f.write("--------------------------------------------------\n")
        speeds = [e.get('speed', 0.0) for e in all_entries]
        for k in metrics_keys:
            vals = [e['metrics'].get(k, 0.0) for e in all_entries]
            if len(set(speeds)) > 1 and len(set(vals)) > 1:
                corr, pval = pearsonr(speeds, vals)
                f.write(f"{k:35}: r={corr:.4f}, p={pval:.4f}\n")
            else:
                f.write(f"{k:35}: Insufficient variation for correlation\n")
        
        f.write("\nSPEED-PERFORMANCE ANALYSIS\n")
        f.write("--------------------------------------------------\n")
        sorted_speeds = sorted(per_speed_stats.keys())
        if len(sorted_speeds) >= 2:
            low_speed = sorted_speeds[0]
            high_speed = sorted_speeds[-1]
            low_avg = compute_detailed_stats(per_speed_stats[low_speed])['Overall Match Score']['mean']
            high_avg = compute_detailed_stats(per_speed_stats[high_speed])['Overall Match Score']['mean']
            diff = high_avg - low_avg
            f.write(f"Average Match Score at {low_speed} km/h (min): {low_avg:.2f}%\n")
            f.write(f"Average Match Score at {high_speed} km/h (max): {high_avg:.2f}%\n")
            f.write(f"Total Change: {diff:+.2f}%\n")
            if diff < -2.0:
                f.write("CONCLUSION: Performance shows significant degradation at higher speeds.\n")
            elif diff > 2.0:
                f.write("CONCLUSION: Performance shows unexpected improvement at higher speeds.\n")
            else:
                f.write("CONCLUSION: Performance remains relatively stable across the speed range.\n")
        else:
            f.write("Insufficient speed variation for degradation analysis.\n")

    # 6. Generate Score Histogram PNG
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=np.linspace(0, 100, 11), color='#238636', edgecolor='white', alpha=0.8)
        plt.title('Distribution of Overall Match Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Match Score (%)', fontsize=12)
        plt.ylabel('Number of Clips', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlim(0, 100)
        
        # Add mean/std text to plot
        mean_val = overall_stats['Overall Match Score']['mean']
        std_val = overall_stats['Overall Match Score']['std']
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}%')
        plt.legend()
        
        hist_path = os.path.join(averages_dir, 'score_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to generate histogram plot: {e}")

    return jsonify({
        'success': True,
        'report_dir': averages_dir,
        'histogram_path': os.path.join(averages_dir, 'score_histogram.png'),
        'overall_avg': {k: overall_stats[k]['mean'] for k in metrics_keys}
    })
