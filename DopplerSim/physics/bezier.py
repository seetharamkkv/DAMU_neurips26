import numpy as np
from audio.audio_utils import SR, apply_distance_fade

C_SOUND = 343.0  # m/s
NEAR_FIELD_RADIUS = 6.0  # m – broader near-field for smoother pass-by envelope


def _cubic_bezier(t, p0, p1, p2, p3):
    """
    Standard cubic Bezier position.
    t: array-like in [0,1] or scalar.
    p*: floats (for x or y component).
    """
    t = np.asarray(t)
    one_minus_t = 1.0 - t
    return (one_minus_t**3) * p0 + 3 * (one_minus_t**2) * t * p1 + \
           3 * one_minus_t * (t**2) * p2 + (t**3) * p3


def _cubic_bezier_derivative(t, p0, p1, p2, p3):
    """
    Derivative of cubic Bezier w.r.t parameter t.
    """
    t = np.asarray(t)
    one_minus_t = 1.0 - t
    return 3 * (one_minus_t**2) * (p1 - p0) + \
           6 * one_minus_t * t * (p2 - p1) + \
           3 * (t**2) * (p3 - p2)


def _rotate_point_xy(x, y, angle_deg):
    if angle_deg == 0.0 or angle_deg == 0:
        return x, y
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def _rotate_vector_xy(vx, vy, angle_deg):
    if angle_deg == 0.0 or angle_deg == 0:
        return vx, vy
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t


def _scaled_bezier_geometry(speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, duration_s, n_samples):
    """Build scaled Bezier geometry exactly as used by Doppler physics."""
    n_samples = max(4, int(n_samples))
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    T = float(duration_s)
    if T <= 0:
        T = 1.0
    tau = t / T

    dx_dtau_init = _cubic_bezier_derivative(tau, x0, x1, x2, x3)
    dy_dtau_init = _cubic_bezier_derivative(tau, y0, y1, y2, y3)
    vx_init = dx_dtau_init / T
    vy_init = dy_dtau_init / T
    speed_init = np.sqrt(vx_init**2 + vy_init**2)
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / mean_speed_init

    xs0, xs1, xs2, xs3 = x0 * phys_scale, x1 * phys_scale, x2 * phys_scale, x3 * phys_scale
    ys0, ys1, ys2, ys3 = y0 * phys_scale, y1 * phys_scale, y2 * phys_scale, y3 * phys_scale

    x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
    y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    dx_dtau = _cubic_bezier_derivative(tau, xs0, xs1, xs2, xs3)
    dy_dtau = _cubic_bezier_derivative(tau, ys0, ys1, ys2, ys3)
    vx = dx_dtau / T
    vy = dy_dtau / T
    return x, y, vx, vy


def sample_bezier_path_xy(speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, duration_s, n_points, angle_deg=0.0):
    """(x, y) samples that match calculate_bezier_doppler geometry."""
    x, y, _vx, _vy = _scaled_bezier_geometry(
        speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, duration_s, n_points
    )
    if angle_deg:
        x, y = _rotate_point_xy(x, y, angle_deg)
    return x, y


def calculate_bezier_doppler(speed_mps,
                             x0, x1, x2, x3,
                             y0, y1, y2, y3,
                             duration_s, c_sound=343.0, angle_deg=0.0, accel_mps2=0.0):
    """
    Cubic Bezier path with near-field-safe amplitude.

    Observer at origin (0,0).
    Spatial path is B(tau) for tau in [0,1]. We map physical time t in [0, T]
    linearly to tau, and then scale the Bezier derivative so that the *average*
    speed magnitude is approximately speed_mps.

    If angle_deg is provided, the entire path is rotated by this angle (in degrees)
    around the origin.

    Parameters
    speed_mps : float
        Desired average speed along the Bezier curve (m/s).
    x0..x3, y0..y3 : float
        Control points for the cubic Bezier in meters.
    duration_s : float
        Total duration (seconds).

    Returns
    freq_ratios : np.ndarray
        Length N (N = SR * duration_s), instantaneous Doppler frequency ratio f'/f0.
    amplitudes : np.ndarray
        Length N, amplitude envelope ~ 1 / sqrt(r^2 + r0^2) (normalized to max 1).
    """
    # Number of samples and time axis
    num_samples = int(round(SR * duration_s))
    if num_samples < 4:
        num_samples = 4

    # Use scaled Bezier control points, then apply acceleration-aware timing.
    n = num_samples
    T = float(duration_s) if float(duration_s) > 0 else 1.0
    t = np.linspace(0.0, T, n, endpoint=False)
    dt = max(1e-9, T / max(1, n))

    dx_dtau_init = _cubic_bezier_derivative(t / T, x0, x1, x2, x3)
    dy_dtau_init = _cubic_bezier_derivative(t / T, y0, y1, y2, y3)
    vx_init = dx_dtau_init / T
    vy_init = dy_dtau_init / T
    speed_init = np.sqrt(vx_init**2 + vy_init**2)
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / max(1e-6, mean_speed_init)

    xs0, xs1, xs2, xs3 = x0 * phys_scale, x1 * phys_scale, x2 * phys_scale, x3 * phys_scale
    ys0, ys1, ys2, ys3 = y0 * phys_scale, y1 * phys_scale, y2 * phys_scale, y3 * phys_scale

    # B7 constant-acceleration speed law + integrated progression.
    v_t = np.maximum(1e-3, float(speed_mps) + float(accel_mps2) * t)
    s_t = np.cumsum(v_t) * dt
    total_s = max(1e-9, float(s_t[-1]))
    tau = np.clip(s_t / total_s, 0.0, 1.0)
    dtaudt = v_t / total_s

    x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
    y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    dx_dtau = _cubic_bezier_derivative(tau, xs0, xs1, xs2, xs3)
    dy_dtau = _cubic_bezier_derivative(tau, ys0, ys1, ys2, ys3)
    vx_raw = dx_dtau * dtaudt
    vy_raw = dy_dtau * dtaudt

    # Rotate path if angle is non-zero
    if angle_deg:
        x, y = _rotate_point_xy(x, y, angle_deg)
        vx_raw, vy_raw = _rotate_vector_xy(vx_raw, vy_raw, angle_deg)

    # Distance to observer
    r = np.sqrt(x**2 + y**2)

    # Use true distance for Doppler geometry, with small epsilon
    eps = 1e-9
    r_safe = np.maximum(r, eps)

    # Radial velocity v_r = (v · r_hat) = (v · p) / |p|
    v_dot_r = vx_raw * x + vy_raw * y
    v_r = v_dot_r / r_safe

    # Clamp radial velocity to keep Doppler ratios realistic,
    # similar behaviour to straight-line (no insane sweeps).
    max_vr = min(0.9 * c_sound, 1.2 * abs(speed_mps))
    v_r = np.clip(v_r, -max_vr, max_vr)

    # Doppler ratio
    freq_ratios = c_sound / (c_sound + v_r)

    # Combined amplitude with master gain and gamma compression for audibility
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**1.1
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
