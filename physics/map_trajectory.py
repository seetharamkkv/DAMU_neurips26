import numpy as np
from audio.audio_utils import SR, apply_distance_fade

NEAR_FIELD_RADIUS = 6.0
# Long fades (e.g. 1 s) were meant to soften spawn/despawn but audibly duck the whole last second.
# Short tapers remove edge clicks without a noticeable “drop off a cliff” at clip end.
EDGE_FADE_S = 0.1


def sample_map_path_xy(points, speed_mps, duration_s, n_points):
    """
    (x, y) samples on the polyline at constant speed ``speed_mps`` along arclength.
    s(t) = min(speed_mps * t, L); t ∈ [0, duration) with n_points steps (endpoint=False),
    same law as :func:`calculate_map_trajectory_doppler`.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("points must be an (N, 2) array with N >= 1")

    speed_mps = float(speed_mps)
    if speed_mps < 0.0:
        speed_mps = 0.0
    duration_s = float(duration_s)
    if duration_s <= 0:
        duration_s = 1.0
    n_points = max(2, int(n_points))

    if len(points) == 1:
        return np.full(n_points, points[0, 0]), np.full(n_points, points[0, 1])

    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_path_len = float(cumulative_dist[-1])
    if total_path_len < 1e-9:
        return np.full(n_points, points[0, 0]), np.full(n_points, points[0, 1])

    t = np.linspace(0.0, duration_s, n_points, endpoint=False)
    query_dist = np.minimum(speed_mps * t, total_path_len)
    px = np.interp(query_dist, cumulative_dist, points[:, 0])
    py = np.interp(query_dist, cumulative_dist, points[:, 1])
    return px, py


def calculate_map_trajectory_doppler(points, speed_mps, duration_s, observer_pos=(0, 0), c_sound=343.0):
    """
    Calculate Doppler shift for a custom trajectory defined by point list.
    Points are (x, y) in meters.

    The source moves at constant speed ``speed_mps`` along the polyline (in order of
    the points). Arclength as a function of time is s(t) = min(speed_mps * t, L) where
    L is the total path length. If the path is shorter than speed_mps * duration_s,
    the source stays at the final point for the remainder of the clip.
    Velocity must be the **path tangent** times speed — not finite differences of the
    interpolated (x(t), y(t)) samples. FD mis-aligns v with p_rel and breaks radial
    velocity on curves and U-turns (wrong “always receding” perception after CPA).
    """
    pts = np.asarray(points, dtype=float)
    num_samples = int(round(SR * duration_s))
    px, py = sample_map_path_xy(pts, speed_mps, duration_s, num_samples)
    p = np.vstack([px, py])  # (2, N)

    obs = np.array(observer_pos, dtype=float).reshape(2, 1)
    p_rel = p - obs
    r = np.linalg.norm(p_rel, axis=0)

    speed_mps = float(speed_mps)
    _duration_s = float(duration_s)

    if pts.shape[0] < 2:
        v_r = np.zeros(num_samples, dtype=float)
    else:
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_path_len = float(np.sum(seg_lens))
        if total_path_len < 1e-9:
            v_r = np.zeros(num_samples, dtype=float)
        else:
            cumulative_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)

            t_axis = np.linspace(0.0, _duration_s, num_samples, endpoint=False)
            dist_along_time = speed_mps * t_axis
            # Still translating along polyline (same cutoff as sample_map_path_xy)
            moving = dist_along_time < total_path_len - 1e-9

            s_query = np.minimum(dist_along_time, total_path_len)
            # Segment containing arclength s: cumulative_dist[k] <= s <= cumulative_dist[k+1]
            k = np.searchsorted(cumulative_dist, s_query, side='right') - 1
            k = np.clip(k, 0, pts.shape[0] - 2)

            a_pt = pts[k]
            b_pt = pts[k + 1]
            tang = b_pt - a_pt
            tnorm = np.linalg.norm(tang, axis=1, keepdims=True)
            tnorm = np.maximum(tnorm, 1e-12)
            tang_unit = tang / tnorm

            vx = speed_mps * tang_unit[:, 0]
            vy = speed_mps * tang_unit[:, 1]
            vx = np.where(moving, vx, 0.0)
            vy = np.where(moving, vy, 0.0)

            v = np.vstack([vx, vy])
            # Radial velocity (outward positive): v_r = d|r|/dt = (v · p_rel) / |p_rel|
            v_r = np.sum(v * p_rel, axis=0) / np.maximum(r, 1e-9)

    freq_ratios = c_sound / (c_sound + v_r)
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**1.1
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7

    # Brief edge tapers only (full 1 s fade made the last ~1 s sound unnaturally quiet).
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=EDGE_FADE_S)

    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
