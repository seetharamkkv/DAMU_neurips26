import io
import os
import traceback
import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-GUI backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

import librosa
import librosa.display

from audio.audio_utils import SR
from physics.map_trajectory import sample_map_path_xy
from physics.parabola import sample_parabola_path_xy
from physics.bezier import sample_bezier_path_xy


def compute_path_points(path_type, params, n_points=200, **kwargs):
    """Compute (x, y) path points for plotting"""
    duration = params.get('duration', 10.0)
    obs_pos = kwargs.get('observer_pos', (0, 0))
    is_absolute = kwargs.get('absolute', False)

    if path_type == 'straight':
        v = params['speed']
        
        # Check if this is an intersection-style straight path
        if 'road' in params:
            road = params['road']
            direction = params.get('direction', 1)
            arrival_time = params.get('arrival_time', duration / 2.0)
            offset = params.get('offset', 0.0)
            
            t = np.linspace(0.0, duration, n_points)
            dt = t - arrival_time
            
            if road == 'horizontal':
                x = direction * v * dt
                # If absolute, this is already world-X. But to center on observer_x:
                if is_absolute:
                    x = x + obs_pos[0]
                y = np.full_like(x, offset)
            else:
                y = direction * v * dt
                if is_absolute:
                    y = y + obs_pos[1]
                x = np.full_like(y, offset)
            
            # Adjust for observer position
            if not is_absolute:
                # obs_pos for intersection defaults to (10,10) if not provided
                default_obs = (10, 10) if 'road' in params else (0, 0)
                curr_obs = kwargs.get('observer_pos', default_obs)
                x = x - curr_obs[0]
                y = y - curr_obs[1]
            
            closest = None
        else:
            # Pass-by pass logic
            h = params.get('distance', 30.0) # Fallback to 30m if distance missing
            angle = params.get('angle', 0.0)
            # Optional explicit plotting time window for truncated clips.
            t_start = float(params.get('plot_t_start', 0.0))
            t_end = float(params.get('plot_t_end', duration))
            if t_end <= t_start:
                t_end = t_start + 1e-3
            t = np.linspace(t_start, t_end, n_points)
            t0 = float(params.get('cpa_time', duration / 2.0))
            dt = t - t0

            theta = np.deg2rad(angle)
            u = np.array([np.cos(theta), np.sin(theta)])
            n = np.array([-np.sin(theta), np.cos(theta)])

            p_c = h * n
            v_vec = u * v
            p = p_c[:, None] + v_vec[:, None] * dt[None, :]

            x = p[0, :]
            y = p[1, :]

            if is_absolute:
                x = x + obs_pos[0]
                y = y + obs_pos[1]

            cx, cy = p_c
            if is_absolute:
                cx += obs_pos[0]
                cy += obs_pos[1]
            closest = (cx, cy)

    elif path_type == 'parabola':
        # Must match calculate_parabola_doppler: τ ∈ [-1,1], half-span refinement, y = a·x² + h, then angle_deg about origin
        v = float(params['speed'])
        a = float(params['a'])
        h = float(params['h'])
        angle_deg = float(params.get('angle_deg', 0.0))
        x, y = sample_parabola_path_xy(v, a, h, float(duration), n_points, angle_deg=angle_deg)
        if is_absolute:
            x = x + obs_pos[0]
            y = y + obs_pos[1]
        closest = None

    elif path_type == 'bezier':
        speed = float(params.get('speed', 20.0))
        x0 = float(params.get('x0', 0))
        x1 = float(params.get('x1', 0))
        x2 = float(params.get('x2', 0))
        x3 = float(params.get('x3', 0))
        y0 = float(params.get('y0', 0))
        y1 = float(params.get('y1', 0))
        y2 = float(params.get('y2', 0))
        y3 = float(params.get('y3', 0))
        angle_deg = float(params.get('angle_deg', 0.0))
        # Match calculate_bezier_doppler geometry: speed-rescaled control points
        # and optional angle_deg rotation about the origin.
        x, y = sample_bezier_path_xy(
            speed, x0, x1, x2, x3, y0, y1, y2, y3, float(duration), n_points, angle_deg=angle_deg
        )

        if is_absolute:
            x = x + obs_pos[0]
            y = y + obs_pos[1]
        else:
            x = x - obs_pos[0]
            y = y - obs_pos[1]
        closest = None

    elif path_type in ('map_path', 'map_trajectory'):
        points = np.array(params['points'])
        speed = float(params.get('speed', 30.0))
        x, y = sample_map_path_xy(points, speed, duration, n_points)
        if not is_absolute:
            x = x - obs_pos[0]
            y = y - obs_pos[1]
        closest = None
    else:
        # fallback: trivial horizontal line
        x = np.linspace(-10, 10, n_points)
        y = np.zeros_like(x)
        closest = None

    # APPLY GLOBAL ROAD CURVATURE (Realistic Roads Fix)
    y_in = np.asarray(y, dtype=float).copy()
    road_curve_a = kwargs.get('road_curve_a', 0.0)
    curve_blend = float(params.get('road_curve_blend', 1.0))
    global_curve_scale = float(params.get('global_curve_scale', 1.0))
    local_curve_a = float(params.get('road_curve_a', 0.0))
    path_curve_a = (road_curve_a * global_curve_scale + local_curve_a) * curve_blend
    if path_curve_a != 0:
        # In absolute mode, the curve is centered on the observer's X position
        x_ref = x - obs_pos[0] if is_absolute else x
        road_y0 = float(kwargs.get('road_y_center', 0.0))
        y_road = road_y0 + path_curve_a * (x_ref ** 2)
        # Shallow-road / monotonic-x refit (not a true Frenet parallel offset, which would
        # be constant d along arclength in the normal; here y is a Cartesian "lane" model):
        # y = y_road + y_in - chord(y_road),  chord = (1-t)r0 + t*r1 on x in [x0, x1].
        # = y_road + w + (1-t)δ0 + t*δ1 with w = y_in - chord(y_in) and δi = y_in[i] - y_road[i]
        # at the ends, so wobble w is kept and end offsets δ match the old intrinsic path.
        # Avoids stacking a second x² (intrinsic + a_road) that would fight the centerline.
        use_refit = (
            kwargs.get('refit_to_road', True)
            and is_absolute
            and path_type in ('parabola', 'bezier')
            and len(x) > 1
        )
        if use_refit:
            # Stable lane-relative refit:
            # keep each vehicle's median offset from the road centerline, plus
            # its local wobble around its own median. This avoids large mid-span
            # over/under-shoot that can violate boundaries.
            lane_rel = float(np.median(y_in - y_road))
            wobble = y_in - float(np.median(y_in))
            y = y_road + lane_rel + wobble
        else:
            y = y_in + path_curve_a * (x_ref ** 2)

    # APPLY GLOBAL ROAD TILT (pivot about road_y_center; same for all path types)
    road_angle = float(kwargs.get('road_angle', 0.0))
    road_angle += float(params.get('road_angle_offset', 0.0))
    if road_angle != 0:
        theta = np.deg2rad(road_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate around road centerline anchor instead of origin to avoid large visual drift.
        pivot_x = obs_pos[0] if is_absolute else 0.0
        pivot_y = kwargs.get('road_y_center', obs_pos[1] if is_absolute else 0.0)
        x_rel, y_rel = x - pivot_x, y - pivot_y
        x_rot = x_rel * cos_t - y_rel * sin_t
        y_rot = x_rel * sin_t + y_rel * cos_t
        x, y = x_rot + pivot_x, y_rot + pivot_y

    # ABSOLUTE PLOT SAFETY CLAMP (keeps trajectories inside road band)
    if is_absolute and kwargs.get('clamp_to_road_band', False):
        # lane_width is single-lane half-road width in generation flow.
        # Clamp to full strip road band: center ± lane_width.
        lane_width = float(kwargs.get('lane_width', 4.0))
        road_y_center = float(kwargs.get('road_y_center', 0.0))
        road_curve_clamp = kwargs.get('road_curve_a', 0.0)
        x_ref_clamp = x - obs_pos[0] if is_absolute else x
        center_at_x = road_y_center + road_curve_clamp * (x_ref_clamp ** 2)
        y_min = center_at_x - lane_width
        y_max = center_at_x + lane_width
        y = np.clip(y, y_min, y_max)

    return x, y, closest


def polyline_length_m(x, y):
    """Total Euclidean length along sampled polyline (meters)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def render_simulator_path_summary_png(path_type, params, display_path_label=None):
    """
    Render a PNG summary of the active path for single/custom simulation mode.
    """
    plot_kwargs = {'observer_pos': (0.0, 0.0), 'absolute': False}
    x, y, closest = compute_path_points(path_type, params, n_points=600, **plot_kwargs)

    if path_type in ('map_path', 'map_trajectory') and 'points' in params:
        pts = np.asarray(params['points'], dtype=float)
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    else:
        L = polyline_length_m(x, y)

    T = float(params.get('duration', 0.0))
    v = float(params.get('speed', 0.0))
    label = display_path_label or path_type.replace('_', ' ')

    fig = plt.figure(figsize=(9.0, 6.0), dpi=120)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 0.20],
        hspace=0.08,
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.06,
    )
    ax = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])

    ax.plot(x, y, '-', color='#1f77b4', linewidth=2.0, label='Trajectory', zorder=3)
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size >= 1:
        ax.scatter([xa[0]], [ya[0]], c='#2ca02c', s=50, zorder=5, label='Start', edgecolors='white', linewidths=0.55)
        if xa.size >= 2 and not (np.isclose(xa[0], xa[-1]) and np.isclose(ya[0], ya[-1])):
            ax.scatter([xa[-1]], [ya[-1]], c='#ff7f0e', s=50, zorder=5, label='End', edgecolors='white', linewidths=0.55)
    ax.scatter([0.0], [0.0], c='#d62728', s=52, zorder=6, label='Observer (mic)', edgecolors='white', linewidths=0.6)

    if closest is not None:
        cx, cy = closest
        ax.plot([0.0, cx], [0.0, cy], linestyle='--', color='gray', alpha=0.65, linewidth=1.0, zorder=2)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ (m)', fontsize=10)
    ax.set_ylabel(r'$y$ (m)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which='major', linestyle=':', alpha=0.45, color='#888888', zorder=0)
    ax.set_facecolor('#f6f8fa')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.92, edgecolor='#dddddd')
    fig.suptitle('DopplerSim path summary', fontsize=12, fontweight='600', y=0.97)

    def _fmt_len(m):
        return f'{m:.1f}' if m >= 100 else f'{m:.2f}'

    def _fmt_time(s):
        return f'{s:.2f}'

    def _fmt_spd(s_):
        return f'{s_:.1f}' if s_ >= 10 else f'{s_:.2f}'

    metrics_text = (
        f'·  {label}  ·\n\n'
        rf'·  $L={_fmt_len(L)}\,\mathrm{{m}}$ · $v={_fmt_spd(v)}\,\mathrm{{m/s}}$ · duration $T=L/v={_fmt_time(T)}\,\mathrm{{s}}$  ·'
    )
    ax_metrics.axis('off')
    ax_metrics.text(
        0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
        ha='center', va='center', fontsize=10, color='#24292f', linespacing=1.35
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.14)
    plt.close(fig)
    buf.seek(0)
    return buf


def save_path_plot(path_type, params, output_dir, base_name):
    """
    Save a PNG path graph for this clip with realistic road aesthetics.
    """
    try:
        x, y, closest = compute_path_points(path_type, params, n_points=200)
        plot_path = os.path.join(output_dir, f"{base_name}.png")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Build legend label ...
        label_parts = [f"Path ({path_type.capitalize()})"]
        for k in ['speed', 'distance', 'offset', 'angle', 'temperature', 'humidity']:
            if k in params:
                val = params[k]
                lbl = {'speed': 'v', 'distance': 'd', 'offset': 'off', 'angle': 'θ'}.get(k, k[0])
                unit = {'speed': 'm/s', 'distance': 'm', 'offset': 'm', 'angle': '°'}.get(k, '')
                label_parts.append(f"{lbl}={val}{unit}")
        
        full_label = ", ".join(label_parts)

        # Path
        ax.plot(x, y, linewidth=1.4, color='#1f77b4', label=full_label, zorder=5)

        # Observer at origin
        ax.scatter([0], [0], marker='.', s=30, color='red', label="Observer", zorder=10)

        if closest is not None:
            cx, cy = closest
            ax.plot([0, cx], [0, cy], linestyle='--', linewidth=1, color='gray', alpha=0.6, zorder=4)

        ax.axis('equal')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.grid(True, which="major", linestyle=':', alpha=0.4, zorder=0)
        ax.set_facecolor('#fafafa')
        
        # Legend at bottom
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return os.path.basename(plot_path)

    except Exception as e:
        print(f"Failed to save path plot for {base_name}: {e}")
        return None


def save_combined_path_plot(scenes_data, output_dir, base_name, **kwargs):
    """
    Save a PNG graph with all vehicle paths in a scene.
    Focuses on clear trajectory visualization and readable axis scaling.

    kwargs ``lane_width`` is interpreted as single-lane width / half-road-width
    in strip-road mode, so road edges are drawn at centerline ± lane_width.
    """
    try:
        plot_path = os.path.join(output_dir, f"{base_name}_combined_path.png")
        obs_pos = kwargs.get('observer_pos', (0, 0))
        intersection_mode = bool(kwargs.get('intersection_benchmark', False))
        road_shape = kwargs.get('road_shape', 'straight')
        road_curve_a = float(kwargs.get('road_curve_a', 0.0))
        plot_kwargs = dict(kwargs)
        plot_kwargs['road_curve_a'] = road_curve_a
        # Keep plotting physics-faithful: do not apply a visual-only clamp here,
        # otherwise displayed paths can diverge from trajectories used in audio.
        plot_kwargs['clamp_to_road_band'] = False
        
        fig_w = float(kwargs.get('fig_width', 15.5))
        fig_h = float(kwargs.get('fig_height', 7.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        show_road_guides = bool(kwargs.get('show_road_guides', True))
        path_alpha = float(kwargs.get('path_alpha', 0.65))
        path_linewidth = float(kwargs.get('path_linewidth', 1.8))

        # Precompute paths once (used for both rendering and viewport).
        sampled_paths = []
        for i, (path_type, params, vehicle_name) in enumerate(scenes_data):
            x, y, _ = compute_path_points(path_type, params, n_points=200, **plot_kwargs)
            sampled_paths.append((i, path_type, params, vehicle_name, x, y))

        # MINIMAL ROAD GUIDES (no decorative background)
        lane_width = float(kwargs.get('lane_width', 4.0))
        lane_half = lane_width
        road_y_center = kwargs.get('road_y_center', 0.0)
        road_angle = kwargs.get('road_angle', 0.0)

        if intersection_mode:
            half_arm = float(kwargs.get('intersection_half_arm', 90.0))
            int_angle = float(kwargs.get('intersection_angle', 90.0))
            # Primary road (E-W): horizontal
            if show_road_guides:
                ax.plot([-half_arm, half_arm], [lane_half, lane_half], color='#666666', linewidth=1.0, label='Road Edge (Primary)', zorder=1)
                ax.plot([-half_arm, half_arm], [-lane_half, -lane_half], color='#666666', linewidth=1.0, zorder=1)
                ax.plot([-half_arm, half_arm], [0.0, 0.0], color='#888888', linestyle='--', linewidth=0.9, label='Median (Primary)', zorder=1)
            # Secondary road at intersection_angle from x-axis
            _ia_rad = np.deg2rad(int_angle)
            _ia_cos, _ia_sin = np.cos(_ia_rad), np.sin(_ia_rad)
            # Direction along secondary arm and its perpendicular (for lane offset)
            def _sec_line(d_start, d_end, lateral):
                """Line from d_start to d_end along the secondary axis, offset by lateral."""
                x0 = d_start * _ia_cos - lateral * _ia_sin
                y0 = d_start * _ia_sin + lateral * _ia_cos
                x1 = d_end * _ia_cos - lateral * _ia_sin
                y1 = d_end * _ia_sin + lateral * _ia_cos
                return [x0, x1], [y0, y1]
            if show_road_guides:
                ax.plot(*_sec_line(-half_arm, half_arm, lane_half), color='#666666', linewidth=1.0, label='Road Edge (Secondary)', zorder=1)
                ax.plot(*_sec_line(-half_arm, half_arm, -lane_half), color='#666666', linewidth=1.0, zorder=1)
                ax.plot(*_sec_line(-half_arm, half_arm, 0.0), color='#888888', linestyle='--', linewidth=0.9, label='Median (Secondary)', zorder=1)
            # Dummy series for viewport computation
            x_upper = x_lower = x_med = np.array([-half_arm, half_arm])
            y_upper = np.array([lane_half, lane_half])
            y_lower = np.array([-lane_half, -lane_half])
            y_med = np.array([0.0, 0.0])
        else:
            if sampled_paths:
                path_x = np.concatenate([p[4] for p in sampled_paths])
                path_y = np.concatenate([p[5] for p in sampled_paths])
                x_min_path = float(np.min(path_x))
                x_max_path = float(np.max(path_x))
                x_span = max(60.0, x_max_path - x_min_path)
                x_pad = max(15.0, 0.1 * x_span)
                x_road = np.linspace(x_min_path - x_pad, x_max_path + x_pad, 500)
                if 'road_y_center' not in kwargs:
                    road_y_center = float(np.median(path_y))
            else:
                x_road = np.linspace(-120.0, 120.0, 500)

            x_rel = x_road - obs_pos[0]

            # Build a centerline that can be straight/parabolic/bezier/auto-fit.
            if road_shape == 'bezier':
                t = np.linspace(0.0, 1.0, x_road.size)
                y0 = road_y_center
                y3 = road_y_center
                bulge = float(kwargs.get('road_bezier_bulge', 0.6))
                y1 = road_y_center + bulge
                y2 = road_y_center - bulge
                y_median = (
                    ((1 - t) ** 3) * y0
                    + 3 * ((1 - t) ** 2) * t * y1
                    + 3 * (1 - t) * (t ** 2) * y2
                    + (t ** 3) * y3
                )
            elif road_shape == 'parabola' or abs(road_curve_a) > 0.0:
                y_median = road_y_center + road_curve_a * (x_rel ** 2)
            elif sampled_paths and kwargs.get('auto_fit_road', False):
                # Fit a smooth quadratic centerline from vehicle paths.
                path_x = np.concatenate([p[4] for p in sampled_paths])
                path_y = np.concatenate([p[5] for p in sampled_paths])
                if np.std(path_x) > 1e-6:
                    poly = np.polyfit(path_x, path_y, 2)
                    y_median = np.polyval(poly, x_road)
                else:
                    y_median = np.full_like(x_road, road_y_center, dtype=float)
            else:
                y_median = np.full_like(x_road, road_y_center, dtype=float)

            # Edges at ±lane_width along local normal (lane_width is half-road-width here).
            dy_dx = np.gradient(y_median, x_road)
            denom = np.sqrt(1.0 + dy_dx ** 2)
            n_x = -dy_dx / denom
            n_y = 1.0 / denom

            x_upper = x_road + lane_half * n_x
            y_upper = y_median + lane_half * n_y
            x_lower = x_road - lane_half * n_x
            y_lower = y_median - lane_half * n_y
            x_med = x_road
            y_med = y_median

            if road_angle != 0.0:
                theta = np.deg2rad(road_angle)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                pivot_x = obs_pos[0]
                pivot_y = road_y_center

                def rotate(px, py):
                    px_rel, py_rel = px - pivot_x, py - pivot_y
                    rx = px_rel * cos_t - py_rel * sin_t
                    ry = px_rel * sin_t + py_rel * cos_t
                    return rx + pivot_x, ry + pivot_y

                x_upper, y_upper = rotate(x_upper, y_upper)
                x_lower, y_lower = rotate(x_lower, y_lower)
                x_med, y_med = rotate(x_med, y_med)

            if show_road_guides:
                ax.plot(x_upper, y_upper, color='#666666', linewidth=1.0, label='Road Edge', zorder=1)
                ax.plot(x_lower, y_lower, color='#666666', linewidth=1.0, zorder=1)
                ax.plot(x_med, y_med, color='#888888', linestyle='--', linewidth=0.9, label='Median', zorder=1)

        # plot vehicle paths
        dotted_extension_m = float(kwargs.get('dotted_extension_m', 22.0))
        for i, path_type, params, vehicle_name, x, y in sampled_paths:
            # Determine arrow based on directional intent in UNROTATED frame
            # For straight/parabola it's usually speed/direction, for bezier it's x3 > x0
            is_forward = True
            if path_type == 'straight' and params.get('direction', 1) == -1:
                is_forward = False
            elif path_type == 'straight' and params.get('angle', 0) == 180:
                is_forward = False
            elif path_type == 'parabola' and params.get('speed', 1) < 0:
                is_forward = False
            elif path_type == 'bezier' and params.get('x3', 0) < params.get('x0', 0):
                is_forward = False
                
            arrow = " →" if is_forward else " ←"
            speed_val = params.get('speed', None)
            dist_val = params.get('distance', params.get('h', params.get('offset', None)))
            speed_txt = f"{float(speed_val):.1f} m/s" if speed_val is not None else "n/a"
            dist_txt = f"{abs(float(dist_val)):.1f} m" if dist_val is not None else "n/a"
            legend_label = f"V{i+1}: {vehicle_name} (v={speed_txt}, d={dist_txt}){arrow}"
            line, = ax.plot(x, y, linewidth=path_linewidth, label=legend_label, alpha=path_alpha, zorder=5)

            # Add small dotted extrapolations before/after the solid segment so
            # each trajectory visually continues beyond the active span.
            if len(x) >= 2 and dotted_extension_m > 0.0:
                x0, y0 = float(x[0]), float(y[0])
                x1, y1 = float(x[1]), float(y[1])
                xn_1, yn_1 = float(x[-2]), float(y[-2])
                xn, yn = float(x[-1]), float(y[-1])

                start_dx, start_dy = x1 - x0, y1 - y0
                end_dx, end_dy = xn - xn_1, yn - yn_1
                start_norm = max(1e-9, np.hypot(start_dx, start_dy))
                end_norm = max(1e-9, np.hypot(end_dx, end_dy))

                ux_s, uy_s = start_dx / start_norm, start_dy / start_norm
                ux_e, uy_e = end_dx / end_norm, end_dy / end_norm

                n_ext = 24
                t = np.linspace(0.0, 1.0, n_ext)
                x_pre = x0 - ux_s * dotted_extension_m * t
                y_pre = y0 - uy_s * dotted_extension_m * t
                x_post = xn + ux_e * dotted_extension_m * t
                y_post = yn + uy_e * dotted_extension_m * t

                ext_color = line.get_color()
                ax.plot(
                    x_pre, y_pre,
                    linestyle='--', dashes=(2.0, 2.0),
                    linewidth=1.0, color=ext_color, alpha=max(0.35, path_alpha - 0.25),
                    zorder=4, label='_nolegend_'
                )
                ax.plot(
                    x_post, y_post,
                    linestyle='--', dashes=(2.0, 2.0),
                    linewidth=1.0, color=ext_color, alpha=max(0.35, path_alpha - 0.25),
                    zorder=4, label='_nolegend_'
                )

        # Observer
        ax.scatter([obs_pos[0]], [obs_pos[1]], marker='.', s=30, color='red', label='Observer', zorder=10)

        observer_distance_m = kwargs.get('observer_distance_m', None)
        if observer_distance_m is not None:
            d = float(observer_distance_m)
            x0 = float(obs_pos[0])
            y0 = float(obs_pos[1])
            y1 = y0 + d
            ax.plot([x0, x0], [y0, y1], linestyle=':', linewidth=1.2, color='#555555', alpha=0.85, zorder=2)
            ax.text(
                x0 + 2.5, (y0 + y1) / 2.0,
                f"Distance = {d:.1f} m",
                fontsize=10, color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='#cccccc')
            )

        # adaptive viewport
        # Scale view from trajectory data so y-axis stays informative.
        x_all = [obs_pos[0]]
        y_all = [obs_pos[1]]
        
        # Include all path points
        for _, _, _, _, px, py in sampled_paths:
            x_all.extend(px)
            y_all.extend(py)
        # Keep road guides visible and make lane width visually meaningful.
        x_all.extend(x_upper)
        x_all.extend(x_lower)
        x_all.extend(x_med)
        y_all.extend(y_upper)
        y_all.extend(y_lower)
        y_all.extend(y_med)
        if intersection_mode:
            lh = lane_half
            ha = float(kwargs.get('intersection_half_arm', 90.0))
            # Primary road extents
            x_all.extend([-ha, ha])
            y_all.extend([lh, -lh])
            # Secondary road extents (at intersection_angle)
            _vp_rad = np.deg2rad(float(kwargs.get('intersection_angle', 90.0)))
            _vp_cos, _vp_sin = np.cos(_vp_rad), np.sin(_vp_rad)
            for _d in [-ha, ha]:
                for _lat in [-lh, lh, 0.0]:
                    x_all.append(_d * _vp_cos - _lat * _vp_sin)
                    y_all.append(_d * _vp_sin + _lat * _vp_cos)
        
        x_min, x_max = min(x_all), max(x_all)
        y_min, y_max = min(y_all), max(y_all)

        if intersection_mode:
            # Dedicated square viewport for + intersections; do not reuse strip-road
            # y-capping logic, which flattens the scene.
            ha = float(kwargs.get('intersection_half_arm', 90.0))
            pad = max(8.0, 0.12 * ha)
            x_low = min(x_min, -ha, obs_pos[0]) - pad
            x_high = max(x_max, ha, obs_pos[0]) + pad
            y_low = min(y_min, -ha, obs_pos[1]) - pad
            y_high = max(y_max, ha, obs_pos[1]) + pad
            span = max(x_high - x_low, y_high - y_low)
            cx = 0.5 * (x_low + x_high)
            cy = 0.5 * (y_low + y_high)
            ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
            ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
            ax.set_aspect('equal', adjustable='box')
        else:
            x_pad = (x_max - x_min) * 0.15
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

            # Y-axis: show full road width with generous padding so lanes
            # are clearly visible regardless of the x-axis span.
            road_edge_lo = float(np.min(y_lower))
            road_edge_hi = float(np.max(y_upper))
            road_span = road_edge_hi - road_edge_lo
            y_pad_abs = max(road_span * 0.6, 3.0)
            y_low = road_edge_lo - y_pad_abs
            y_high = road_edge_hi + y_pad_abs

            # Keep observer visible if it is near the road.
            y_low = min(y_low, obs_pos[1] - 1.0)
            y_high = max(y_high, obs_pos[1] + 1.0)

            ax.set_ylim(y_low, y_high)

        ax.set_xlabel("x (meters)", fontsize=14)
        ax.set_ylabel("y (meters)", fontsize=14)
        
        ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Set X-axis gap to 30m
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        # Keep background plain for easier visual inspection.
        ax.set_facecolor('#fafafa')

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.basename(plot_path)

    except Exception as e:
        print(f"Failed to save combined path plot: {e}")
        traceback.print_exc()
        return None


def save_spectrogram_to_file(y, sr, title, out_path, max_y_freq=1250, include_amplitude_bar=False):
    """
    Generate and save a high-resolution spectrogram PNG to a specific path.
    """
    try:
        if include_amplitude_bar:
            fig, (ax, ax_amp) = plt.subplots(
                2, 1, figsize=(12, 6.5), sharex=True,
                gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.12}
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 4.8))

        # Use a more reasonable hop_length for long clips to avoid memory blowup
        n_fft = 4096
        hop_length = 512
        win_length = 4096
        window = "hann"
        stft = librosa.stft(
            y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
        )
        S_power = np.abs(stft) ** 2
        D = librosa.power_to_db(S_power, ref=np.max)

        # Improve contrast so tonal structure is more visible and less "muddy".
        vmax = float(np.max(D))
        vmin = vmax - 80.0
        librosa.display.specshow(
            D,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='magma',
            rasterized=True,
            vmin=vmin,
            vmax=vmax
        )
        max_y_freq = float(max_y_freq) if max_y_freq else 1250.0
        if max_y_freq <= 0:
            max_y_freq = 1250.0
        ax.set_ylim(0, max_y_freq)
        # Keep 5 equal y-axis portions regardless of selected max.
        y_ticks = np.linspace(0, max_y_freq, 6)
        ax.set_yticks(y_ticks)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        if include_amplitude_bar:
            # Frame-wise RMS amplitude bars aligned to the same time axis.
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            if len(times) > 1:
                bar_width = float(times[1] - times[0]) * 0.9
            else:
                bar_width = float(hop_length) / float(sr)
            ax_amp.bar(times, rms, width=bar_width, color='#58a6ff', edgecolor='none', alpha=0.9)
            ax_amp.set_ylabel('Amp')
            ax_amp.set_xlabel('Time (s)')
            ax_amp.set_ylim(0, max(1e-6, float(np.max(rms)) * 1.15))
            ax_amp.grid(True, axis='y', linestyle=':', alpha=0.35)

        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to save spectrogram to {out_path}: {e}")
        return False


def save_audio_comparison_plot(y_a, y_b, sr, title_a, title_b, out_path, max_y_freq=1250):
    """
    Save a side-by-side comparative plot:
    - Top row: spectrogram A and B
    - Bottom row: amplitude bar graph A and B
    """
    try:
        # Do not use sharex='row': that links A and B on the same row, forcing one time span for both
        # clips (e.g. 0–10 s when A is 10 s and B is 9.6 s). Share x only within each column.
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 7), sharex=False,
            gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.18, 'wspace': 0.14}
        )
        ax_spec_a, ax_spec_b = axes[0]
        ax_amp_a, ax_amp_b = axes[1]

        n_fft = 4096
        hop_length = 512
        win_length = 4096
        window = "hann"
        max_y_freq = float(max_y_freq) if max_y_freq else 1250.0
        if max_y_freq <= 0:
            max_y_freq = 1250.0

        def _draw_spectrogram(ax, y, title):
            stft = librosa.stft(
                y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
            )
            s_power = np.abs(stft) ** 2
            d_db = librosa.power_to_db(s_power, ref=np.max)
            vmax = float(np.max(d_db))
            vmin = vmax - 80.0
            librosa.display.specshow(
                d_db,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='hz',
                ax=ax,
                cmap='magma',
                rasterized=True,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_ylim(0, max_y_freq)
            ax.set_yticks(np.linspace(0, max_y_freq, 6))
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

        def _draw_amplitude(ax, y, color):
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            bar_width = (float(times[1] - times[0]) * 0.9) if len(times) > 1 else (float(hop_length) / float(sr))
            ax.bar(times, rms, width=bar_width, color=color, edgecolor='none', alpha=0.9)
            ax.set_ylabel('Amp')
            ax.set_xlabel('Time (s)')
            ax.set_ylim(0, max(1e-6, float(np.max(rms)) * 1.15))
            ax.grid(True, axis='y', linestyle=':', alpha=0.35)

        _draw_spectrogram(ax_spec_a, y_a, title_a)
        _draw_spectrogram(ax_spec_b, y_b, title_b)
        _draw_amplitude(ax_amp_a, y_a, '#58a6ff')
        _draw_amplitude(ax_amp_b, y_b, '#f0883e')

        ax_amp_a.sharex(ax_spec_a)
        ax_amp_b.sharex(ax_spec_b)
        dur_a = float(len(y_a)) / float(sr)
        dur_b = float(len(y_b)) / float(sr)
        ax_spec_a.set_xlim(0.0, max(dur_a, 1e-6))
        ax_spec_b.set_xlim(0.0, max(dur_b, 1e-6))

        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to save audio comparison plot to {out_path}: {e}")
        return False


def save_automated_comparison_plot(y_a, y_b, sr, title_a, title_b, out_path, metrics, max_y_freq=1250):
    """
    Save a side-by-side comparative plot with metrics overlay at the bottom.
    """
    try:
        fig = plt.figure(figsize=(14, 8.5))
        
        # Grid layout: 2 cols, 3 rows (spectrogram, amplitude, text)
        gs = fig.add_gridspec(3, 2, height_ratios=[4, 1, 1], hspace=0.3, wspace=0.14)
        
        ax_spec_a = fig.add_subplot(gs[0, 0])
        ax_spec_b = fig.add_subplot(gs[0, 1])
        ax_amp_a = fig.add_subplot(gs[1, 0])
        ax_amp_b = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[2, :])
        
        # Hide text axes
        ax_text.axis('off')

        n_fft = 4096
        hop_length = 512
        win_length = 4096
        window = "hann"
        max_y_freq = float(max_y_freq) if max_y_freq else 1250.0
        if max_y_freq <= 0:
            max_y_freq = 1250.0

        def _draw_spectrogram(ax, y, title):
            stft = librosa.stft(
                y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
            )
            s_power = np.abs(stft) ** 2
            d_db = librosa.power_to_db(s_power, ref=np.max)
            vmax = float(np.max(d_db))
            vmin = vmax - 80.0
            librosa.display.specshow(
                d_db,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='hz',
                ax=ax,
                cmap='magma',
                rasterized=True,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_ylim(0, max_y_freq)
            ax.set_yticks(np.linspace(0, max_y_freq, 6))
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

        def _draw_amplitude(ax, y, color):
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            bar_width = (float(times[1] - times[0]) * 0.9) if len(times) > 1 else (float(hop_length) / float(sr))
            ax.bar(times, rms, width=bar_width, color=color, edgecolor='none', alpha=0.9)
            ax.set_ylabel('Amp')
            ax.set_xlabel('Time (s)')
            ax.set_ylim(0, max(1e-6, float(np.max(rms)) * 1.15))
            ax.grid(True, axis='y', linestyle=':', alpha=0.35)

        _draw_spectrogram(ax_spec_a, y_a, title_a)
        _draw_spectrogram(ax_spec_b, y_b, title_b)
        _draw_amplitude(ax_amp_a, y_a, '#58a6ff')
        _draw_amplitude(ax_amp_b, y_b, '#f0883e')

        ax_amp_a.sharex(ax_spec_a)
        ax_amp_b.sharex(ax_spec_b)
        dur_a = float(len(y_a)) / float(sr)
        dur_b = float(len(y_b)) / float(sr)
        ax_spec_a.set_xlim(0.0, max(dur_a, 1e-6))
        ax_spec_b.set_xlim(0.0, max(dur_b, 1e-6))
        
        # Add metrics text
        metrics_text = (
            f"Duration (RealData): {metrics.get('Duration (RealData)', 0):.2f} s    |    "
            f"Duration (SimulatedData): {metrics.get('Duration (SimulatedData)', 0):.2f} s    |    "
            f"Dominant Freq (RealData): {metrics.get('Dominant Frequency (RealData)', 0):.1f} Hz    |    "
            f"Dominant Freq (SimulatedData): {metrics.get('Dominant Frequency (SimulatedData)', 0):.1f} Hz\n"
            f"Envelope Correlation: {metrics.get('Envelope Correlation', 0):.1f}%    |    "
            f"Spectral Overlap: {metrics.get('Spectral Overlap', 0):.1f}%    |    "
            f"Overall Match Score: {metrics.get('Overall Match Score', 0):.1f}%"
        )
        
        # Render text in the text axes
        ax_text.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=1'))

        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to save automated comparison plot to {out_path}: {e}")
        return False
