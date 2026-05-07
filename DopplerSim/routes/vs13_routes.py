import os
import json
import csv
import threading
import time
import random
import traceback
import shutil
import numpy as np
from flask import Blueprint, request, jsonify
from core.config import OUTPUT_FOLDER
from audio.generation import generate_single_clip, generate_statistics
from audio.audio_utils import SR

vs13_bp = Blueprint('vs13', __name__)

# Global progress state for VS13 Mode
vs13_progress = {
    'total_target': 0,
    'generated_so_far': 0,
    'current_car': '',
    'current_speed': 0,
    'is_running': False,
    'log_line': ''
}

@vs13_bp.route('/api/generate_vs13_batch', methods=['POST'])
def generate_vs13_batch():
    global vs13_progress
    if vs13_progress['is_running']:
        return jsonify({'error': 'A VS13 batch generation is already in progress'}), 400

    config = request.json
    
    # Validation
    if not config.get('vehicles', {}).get('selected'):
        return jsonify({'error': 'No vehicles selected'}), 400
    if not config.get('paths', {}).get('selected'):
        return jsonify({'error': 'No paths selected'}), 400

    # Start generation in background
    thread = threading.Thread(target=run_vs13_generation, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

@vs13_bp.route('/api/vs13_progress')
def get_vs13_progress():
    return jsonify(vs13_progress)

def run_vs13_generation(config):
    global vs13_progress
    try:
        vs13_progress['is_running'] = True
        vs13_progress['generated_so_far'] = 0
        
        from audio.generation import calculate_distribution, generate_random_parameters
        
        # Create a unique batch root
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        custom_name = config.get('batch', {}).get('name', '').strip()
        if custom_name:
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            batch_id = f"vs13_{safe_name}_{timestamp}"
        else:
            batch_id = f"vs13_{timestamp}"
            
        base_output_dir = config.get('batch', {}).get('save_path', OUTPUT_FOLDER)
        batch_root = os.path.join(base_output_dir, batch_id)
        os.makedirs(batch_root, exist_ok=True)

        dist_mode = config.get('batch', {}).get('dist_mode', 'sweep')
        selected_vehicles = config['vehicles']['selected']
        selected_paths = config['paths']['selected']
        
        # Determine the work list (vehicle, path, speed_kmph)
        work_list = []
        
        if dist_mode == 'sweep':
            # VS13 Special: Every vehicle, every km/h once (now with custom step)
            speed_min = float(config['speed']['min'])
            speed_max = float(config['speed']['max'])
            speed_step = float(config.get('batch', {}).get('speed_step', 1.0))
            
            # Use numpy to generate the range to handle decimals correctly
            speeds_kmph = np.arange(speed_min, speed_max + (speed_step * 0.1), speed_step).tolist()
            # Ensure the last point doesn't exceed speed_max due to float precision
            speeds_kmph = [s for s in speeds_kmph if s <= speed_max + 1e-9]
            
            for car_name in selected_vehicles:
                for speed in speeds_kmph:
                    # For sweep, we cycle through selected paths
                    path = selected_paths[speeds_kmph.index(speed) % len(selected_paths)]
                    work_list.append((car_name, path, speed))
        else:
            # Standard Batch Logic: Random sampling to hit total_clips
            total_clips = int(config['batch']['total_clips'])
            distribution = calculate_distribution(config, total_clips)
            
            v_dist = distribution['vehicles']
            p_dist = distribution['paths']
            
            v_list = []
            for v, count in v_dist.items(): v_list.extend([v] * int(count))
            p_list = []
            for p, count in p_dist.items(): p_list.extend([p] * int(count))
            
            # Pad/Trim to total_clips
            while len(v_list) < total_clips: v_list.append(random.choice(selected_vehicles))
            while len(p_list) < total_clips: p_list.append(random.choice(selected_paths))
            v_list = v_list[:total_clips]
            p_list = p_list[:total_clips]
            
            random.shuffle(v_list)
            random.shuffle(p_list)
            
            speed_min = config['speed']['min']
            speed_max = config['speed']['max']
            
            for i in range(total_clips):
                # Random speed in km/h
                speed = random.uniform(speed_min, speed_max)
                work_list.append((v_list[i], p_list[i], speed))

        vs13_progress['total_target'] = len(work_list)
        vs13_progress['log_line'] = f"Planning to generate {len(work_list)} clips..."
        
        car_data = {car: [] for car in selected_vehicles}
        
        clips_metadata = []
        for i, (car_name, path_type, speed_kmph) in enumerate(work_list):
            vs13_progress['current_car'] = car_name
            vs13_progress['current_speed'] = round(speed_kmph, 2)
            
            car_dir = os.path.join(batch_root, car_name)
            os.makedirs(car_dir, exist_ok=True)
            additional_dir = os.path.join(car_dir, "Additional_files")
            os.makedirs(additional_dir, exist_ok=True)
            
            # Internal conversion to m/s
            speed_mps = speed_kmph / 3.6
            
            # Build params based on ranges
            base_params = generate_random_parameters(config, car_name, path_type)
            params = dict(base_params)
            params['speed'] = speed_mps
            params['duration'] = 10.0 # VS13 standard
            params['apply_propagation_delay'] = False
            
            disp_speed = int(speed_kmph) if float(speed_kmph).is_integer() else round(speed_kmph, 2)
            base_filename = f"{car_name}_{disp_speed}"
            
            include_samples = config.get('batch', {}).get('include_sample_folders', False)
            
            try:
                clip_meta = generate_single_clip(
                    vehicle_name=car_name,
                    path_type=path_type,
                    params=params,
                    output_dir=additional_dir,
                    batch_id=car_name,
                    index=i,
                    config=config,
                    custom_filename=base_filename
                )
                clip_meta['vehicle'] = car_name # Ensure vehicle name is in meta
                clips_metadata.append(clip_meta)
                
                # Move .wav to car_dir
                sample_folder = clip_meta['sample_dir']
                source_wav = os.path.join(additional_dir, sample_folder, "Essential", clip_meta['filename'])
                dest_wav = os.path.join(car_dir, f"{base_filename}.wav")
                if os.path.exists(source_wav):
                    shutil.copy2(source_wav, dest_wav)
                
                # Copy spectrogram if generated
                source_spec = os.path.join(additional_dir, sample_folder, "Essential", f"{base_filename}_spectrogram.png")
                dest_spec = os.path.join(car_dir, f"{base_filename}_spectrogram.png")
                if os.path.exists(source_spec):
                    shutil.copy2(source_spec, dest_spec)
                
                # If not including samples, cleanup the sample folder
                if not include_samples:
                    full_sample_path = os.path.join(additional_dir, sample_folder)
                    if os.path.exists(full_sample_path):
                        shutil.rmtree(full_sample_path)
                
                # Create .txt
                cpa_time = clip_meta['labels'].get('cpa_time_sec', 5.0)
                annotation_path = os.path.join(car_dir, f"{base_filename}.txt")
                with open(annotation_path, 'w') as af:
                    af.write(f"{float(speed_kmph):.2f} {cpa_time:.2f}")
                
                car_data[car_name].append(base_filename)
                vs13_progress['log_line'] = f"Generated {car_name} at {speed_kmph:.1f} km/h"
                
            except Exception as e:
                vs13_progress['log_line'] = f"Error: {str(e)}"
                print(traceback.format_exc())
            
            vs13_progress['generated_so_far'] += 1

        # Post-process each car folder
        from audio.generation import generate_statistics
        for car_name, sample_ids in car_data.items():
            if not sample_ids: continue
            
            car_dir = os.path.join(batch_root, car_name)
            additional_dir = os.path.join(car_dir, "Additional_files")
            car_clips = [c for c in clips_metadata if c.get('vehicle') == car_name]
            
            # 1. Train/Valid Split (80/20)
            random.seed(42)
            shuffled = list(sample_ids)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.8)
            train_set = set(shuffled[:split_idx])
            
            split_path = os.path.join(car_dir, "Train_valid_split.txt")
            with open(split_path, 'w') as sf:
                for sid in sorted(sample_ids):
                    tag = "train" if sid in train_set else "valid"
                    sf.write(f"{sid} {tag}\n")
            
            # 2. metadata.json
            with open(os.path.join(additional_dir, f"metadata_{car_name}.json"), 'w') as f:
                json.dump({'car': car_name, 'clips': car_clips}, f, indent=2)
                
            # 3. dataset.csv
            with open(os.path.join(additional_dir, "dataset.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'filename', 'speed_kmph', 'speed_mps', 'acceleration_mps2', 'distance_m', 'angle_deg'])
                for clip in car_clips:
                    p = clip['parameters']
                    writer.writerow([
                        clip['sample_dir'], clip['filename'], 
                        round(p['speed']*3.6, 2), p['speed'], 
                        p.get('acceleration', 0.0),
                        p.get('distance', p.get('h')), p.get('angle', 0)
                    ])

            # 4. statistics.txt
            stats_text = generate_statistics(car_clips, config)
            with open(os.path.join(additional_dir, f"statistics_{car_name}.txt"), 'w') as f:
                f.write(stats_text)
                    
            vs13_progress['log_line'] = f"[OK] Finalized {car_name} split and metadata"

        vs13_progress['log_line'] = "[OK] VS13 Generation Complete."

    except Exception as e:
        vs13_progress['log_line'] = f"Critical Error: {str(e)}"
        print(traceback.format_exc())
    finally:
        vs13_progress['is_running'] = False
