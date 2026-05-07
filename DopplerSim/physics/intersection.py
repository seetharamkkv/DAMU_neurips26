import numpy as np
from audio.audio_utils import SR, apply_distance_fade

NEAR_FIELD_RADIUS = 6.0  # m – broader near-field for smoother pass-by envelope

def calculate_intersection_doppler(vehicles, observer_pos=(10, 10), duration_s=10.0, c_sound=343.0):
    """
    Calculate Doppler shift for multiple vehicles in an intersection.
    
    Parameters
    vehicles : list of dict
        Each dict contains:
        - 'id': unique identifier
        - 'road': 'horizontal' or 'vertical'
        - 'direction': 1 (L->R or B->T) or -1 (R->L or T->B)
        - 'speed': speed in m/s
        - 'arrival_time': time when vehicle reaches the center (0,0)
        - 'offset': lane offset (distance from center of road)
    observer_pos : tuple (x, y)
        Position of the microphone.
    duration_s : float
        Total duration (seconds).
    c_sound : float
        Speed of sound.
        
    Returns
    dict: { vehicle_id: { 'freq_ratios': np.ndarray, 'amplitudes': np.ndarray, 'positions': np.ndarray } }
    """
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)
    
    results = {}
    obs_x, obs_y = observer_pos
    
    for v_cfg in vehicles:
        v_id = v_cfg['id']
        speed = v_cfg['speed']
        t_arr = v_cfg['arrival_time']
        direction = v_cfg['direction']
        offset = v_cfg.get('offset', 0.0)
        
        # Position relative to arrival time: p(t) = v * (t - t_arr)
        dt = t - t_arr
        
        if v_cfg['road'] == 'horizontal':
            # Horizontal road: y = offset, x moves
            x = direction * speed * dt
            y = np.full_like(x, offset)
        else:
            # Vertical road: x = offset, y moves
            y = direction * speed * dt
            x = np.full_like(y, offset)
            
        # Coordinates relative to observer
        rx = x - obs_x
        ry = y - obs_y
        
        # Distance to observer
        r = np.sqrt(rx**2 + ry**2)
        r_safe = np.maximum(r, 1e-9)
        
        # Velocity vector
        if v_cfg['road'] == 'horizontal':
            vx = direction * speed
            vy = 0.0
        else:
            vx = 0.0
            vy = direction * speed
            
        # Radial velocity v_r = (v · r_vec) / |r_vec|
        v_r = (vx * rx + vy * ry) / r_safe
        
        freq_ratios = c_sound / (c_sound + v_r)
        spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
        convective_amp = (c_sound / (c_sound + v_r))**1.1
        amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
        
        # Smooth fade-in/out to prevent abrupt spawning (per vehicle)
        amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
        
        # Normalize amplitudes relative to some global max if needed,
        # but here we keep them raw for mixing (1/R law).
        
        results[v_id] = {
            'freq_ratios': freq_ratios.astype(np.float32),
            'amplitudes': amplitudes.astype(np.float32),
            'positions': np.vstack([x, y]).astype(np.float32)
        }
        
    return results
