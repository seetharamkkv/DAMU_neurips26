import numpy as np
from audio.audio_utils import SR, apply_distance_fade

C_SOUND_STANDARD = 343.0  # m/s
NEAR_FIELD_RADIUS = 6.0  # m – broader near-field to avoid sharp CPA peaks


def calculate_straight_line_doppler(speed_mps, min_distance_m, angle_deg, duration_s, c_sound=343.0):
    """
    Straight-line pass-by with angle and near-field-safe amplitude.
    (Constant velocity variant)
    """
    return calculate_straight_line_accelerated_doppler(speed_mps, 0.0, min_distance_m, angle_deg, duration_s, c_sound)


def calculate_straight_line_accelerated_doppler(speed_v0_mps, accel_mps2, min_distance_m, angle_deg, duration_s, c_sound=343.0):
    """
    Straight-line pass-by with constant acceleration (B7).
    
    Parameters
    speed_v0_mps : float
        Initial speed at t=0 (m/s).
    accel_mps2 : float
        Constant acceleration (m/s^2).
    min_distance_m : float
        Closest distance from path to observer (meters).
    angle_deg : float
        Direction of motion angle.
    duration_s : float
        Total duration (seconds).
    """
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)
    t0 = duration_s / 2.0
    dt = t - t0

    # Instantaneous speed: v(t) = v0 + a * t
    # However, let's define v0 as the speed AT t=t0 (CPA) for consistency with min_distance
    v_t = speed_v0_mps + accel_mps2 * dt
    
    # Position: p(t) = p_c + integral of v(t) * u
    # p(t) = p_c + u * (v0 * dt + 0.5 * a * dt^2)
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])
    
    p_c = min_distance_m * n
    # Displacement along path relative to p_c (t=t0)
    s_t = speed_v0_mps * dt + 0.5 * accel_mps2 * dt**2
    
    p = p_c[:, None] + u[:, None] * s_t[None, :]
    r = np.linalg.norm(p, axis=0)
    r_safe = np.maximum(r, 1e-9)
    
    # Velocity vector at each time t
    v_vec = u[:, None] * v_t[None, :]
    # Radial velocity v_r = (v_vec · p) / |p|
    v_r = np.sum(v_vec * p, axis=0) / r_safe
    
    freq_ratios = c_sound / (c_sound + v_r)
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**1.1
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
