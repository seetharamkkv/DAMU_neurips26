import os
import json
import shutil
import random
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d

from audio.audio_utils import (
    apply_doppler_to_audio_fixed,
    apply_doppler_to_audio_fixed_alternative,
    apply_doppler_to_audio_fixed_advanced,
    save_audio,
    SR,
    apply_retarded_time_correction
)
from audio.audio_utils import extend_audio_with_overlap

from physics.straight_line import calculate_straight_line_doppler, calculate_straight_line_accelerated_doppler
from physics.parabola import calculate_parabola_doppler, sample_parabola_path_xy
from physics.bezier import calculate_bezier_doppler, sample_bezier_path_xy
from physics.map_trajectory import calculate_map_trajectory_doppler
from physics.intersection import calculate_intersection_doppler

from core.config import DEFAULT_RANGES, UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER
from core.sampler import SAMPLERS, CyclicIntegerSampler
from visualization.plot_utils import save_path_plot


# spectral enrichment helpers

def _preflight_vehicle_spectrum(y, sr):
    """
    Vehicle WAVs are often rumble-heavy; STFT peak bin then sits ~40–80 Hz while field recordings
    often peak ~120–220 Hz (motor + tire band). Gentle HP + motor-band emphasis shifts centroid up
    without replacing the Doppler core.
    """
    x = np.asarray(y, dtype=np.float32)
    if len(x) < 64:
        return x
    nyq = sr / 2.0
    hp_hz = 45.0
    try:
        sos_hp = signal.butter(2, hp_hz, btype='high', output='sos', fs=float(sr))
        x_hp = signal.sosfilt(sos_hp, x)
        # Less raw sub bleed — roadside captures usually sound “fuller” ~120–200 Hz than library WAVs.
        x_mix = 0.86 * x_hp + 0.14 * x
    except TypeError:
        try:
            sos_hp = signal.butter(2, min(0.99, hp_hz / nyq), btype='high', output='sos')
            x_hp = signal.sosfilt(sos_hp, x)
            x_mix = 0.86 * x_hp + 0.14 * x
        except ValueError:
            x_mix = x.copy()
    except ValueError:
        x_mix = x.copy()
    lo_hz, hi_hz = 95.0, min(440.0, sr / 2.0 - 1.0)
    if hi_hz > lo_hz + 20.0:
        try:
            sos_b = signal.butter(2, [lo_hz, hi_hz], btype='band', output='sos', fs=float(sr))
            band = signal.sosfilt(sos_b, x_mix)
            x_mix = x_mix + 0.30 * band.astype(np.float32)
        except TypeError:
            try:
                sos_b = signal.butter(2, [lo_hz / nyq, hi_hz / nyq], btype='band', output='sos')
                band = signal.sosfilt(sos_b, x_mix)
                x_mix = x_mix + 0.30 * band.astype(np.float32)
            except ValueError:
                pass
        except ValueError:
            pass
    # Typical handheld pass-by: extra energy in ~110–260 Hz (engine/exhaust body vs thin sub line).
    road_lo, road_hi = 110.0, min(260.0, sr / 2.0 - 2.0)
    if road_hi > road_lo + 25.0:
        try:
            sos_r = signal.butter(2, [road_lo, road_hi], btype='band', output='sos', fs=float(sr))
            road_body = signal.sosfilt(sos_r, x_mix)
            x_mix = x_mix + 0.16 * road_body.astype(np.float32)
        except (TypeError, ValueError):
            try:
                sos_r = signal.butter(
                    2, [road_lo / nyq, road_hi / nyq], btype='band', output='sos'
                )
                road_body = signal.sosfilt(sos_r, x_mix)
                x_mix = x_mix + 0.16 * road_body.astype(np.float32)
            except ValueError:
                pass
    return np.clip(x_mix, -2.0, 2.0).astype(np.float32)


def _pink_noise(n_samples, rng):
    """Generate approximate pink noise using low-order IIR filtering."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    # Simple Paul Kellet style approximation.
    b = np.array([0.049922, -0.095993, 0.050612, -0.004408], dtype=np.float32)
    a = np.array([1.0, -2.494956, 2.017265, -0.522190], dtype=np.float32)
    pink = signal.lfilter(b, a, white)
    peak = np.max(np.abs(pink)) + 1e-8
    return (pink / peak).astype(np.float32)


def _apply_subtle_air_noise(audio, sr, strength_pct, center_freq_hz=1200.0):
    """
    Add a subtle broadband environmental air-noise bed.
    Used by batch generation when enabled in Atmosphere settings.
    """
    x = np.asarray(audio, dtype=np.float32)
    if len(x) == 0:
        return x

    strength = float(np.clip(strength_pct, 0.0, 100.0))
    if strength <= 0.0:
        return x

    n = len(x)
    white = np.random.normal(0.0, 1.0, n).astype(np.float32)
    nyq = 0.5 * float(sr)
    center_hz = float(np.clip(center_freq_hz, 80.0, min(5000.0, nyq * 0.95)))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    spectrum = np.fft.rfft(white)

    sigma_hz = max(250.0, 0.32 * center_hz)
    peak = np.exp(-0.5 * ((freqs - center_hz) / sigma_hz) ** 2)
    broad_floor = 0.42 + 0.20 * np.sqrt(np.clip(freqs / max(nyq, 1e-6), 0.0, 1.0))
    profile = broad_floor + 0.60 * peak
    natural_rolloff = 1.0 / np.sqrt(1.0 + (freqs / 10000.0) ** 2)
    profile = np.clip(profile * natural_rolloff, 0.0, None)

    air = np.fft.irfft(spectrum * profile, n=n).astype(np.float32)
    t = np.linspace(0.0, n / float(sr), n, endpoint=False)
    drift = 0.88 + 0.12 * np.sin(2.0 * np.pi * 0.12 * t + np.random.uniform(0.0, 2.0 * np.pi))
    air *= drift.astype(np.float32)

    rms = np.sqrt(np.mean(air ** 2) + 1e-12)
    air /= rms

    mix_gain = 0.07 * ((strength / 100.0) ** 1.25)
    mixed = x + (mix_gain * air)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _apply_harmonic_jitter(audio, sr, rng, amount=1.0):
    """
    Apply subtle low-frequency jitter to playback rate.
    Full-sample warp is accurate but very slow on long clips; we use a cheap mix:
    short clips keep warping, long clips use slow wow/flutter + light tremolo only.
    """
    n = len(audio)
    if n < 4:
        return audio

    t = np.arange(n, dtype=np.float32) / float(sr)
    amount = float(np.clip(amount, 0.0, 1.0))
    jitter_hz = rng.uniform(1.0, 5.0)
    # Keep shallow: deep tremolo reads as ripple on short-time RMS / envelope plots.
    base_depth = rng.uniform(0.0025, 0.009) if n > 96_000 else rng.uniform(0.004, 0.014)
    jitter_depth = amount * base_depth
    phase = rng.uniform(0.0, 2.0 * np.pi)
    # Slow wow/flutter (no per-sample noise — that drove huge interp cost).
    rate_curve = 1.0 + jitter_depth * np.sin(2.0 * np.pi * jitter_hz * t + phase)
    rate_curve += 0.35 * jitter_depth * np.sin(2.0 * np.pi * (jitter_hz * 2.1) * t + phase * 1.3)
    rate_curve = np.clip(rate_curve, 0.985, 1.015).astype(np.float64)

    # Long clips: skip O(n) interp warp (main CPU bottleneck); tremolo is negligible cost.
    if n > 96_000:
        wobble = rate_curve.astype(np.float32)
        return (audio.astype(np.float32) * wobble).astype(np.float32)

    src_pos = np.cumsum(rate_curve)
    src_pos -= src_pos[0]
    if src_pos[-1] > (n - 1):
        src_pos *= (n - 1) / max(src_pos[-1], 1e-8)
    src_pos = np.clip(src_pos, 0.0, n - 1.0)
    x_idx = np.arange(n, dtype=np.float64)
    return np.interp(src_pos, x_idx, np.asarray(audio, dtype=np.float64)).astype(np.float32)


def _apply_distance_dependent_lowpass(audio, envelope, sr):
    """
    High frequencies attenuate more at larger distance (far = darker).
    Fast path: blend two IIR lowpasses by envelope instead of STFT (which was a major bottleneck).
    Near pass-by uses a higher corner so spectrograms keep mid/high “splash” when CPA is close.
    """
    x = np.asarray(audio, dtype=np.float32)
    env = np.clip(np.asarray(envelope, dtype=np.float32), 0.0, 1.0)
    # Sublinear mix: “far” frames still carry more mid content (less all-bass average spectrum).
    env_w = np.power(env, 0.52).astype(np.float32)
    nyq = sr / 2.0
    # Far: not pure sub; near: full air band.
    w_far = min(0.99, 1280.0 / nyq)
    w_near = min(0.99, 7200.0 / nyq)
    sos_lo = signal.butter(2, w_far, btype='low', output='sos')
    sos_hi = signal.butter(2, w_near, btype='low', output='sos')
    x_far = signal.sosfilt(sos_lo, x)
    x_near = signal.sosfilt(sos_hi, x)
    out = (1.0 - env_w) * x_far + env_w * x_near
    return out.astype(np.float32)


def _smooth_short_term_level(out, sr, win_fast_ms=8.0, win_slow_ms=220.0, max_gain=1.14):
    """
    Reduce jagged amplitude traces: additive noise and wow/flutter create fast RMS swings.
    Gently pull short-term RMS toward a slower envelope (similar to visualizing Hilbert / long STFT).
    Wider slow window ≈ field-recorded “broad hump” vs coherent LF ripple in short-time RMS plots.
    """
    x = np.asarray(out, dtype=np.float64)
    wf = max(3, int((win_fast_ms * 1e-3) * sr))
    ws = max(wf + 1, int((win_slow_ms * 1e-3) * sr))
    p_fast = uniform_filter1d(x * x, size=wf, mode='nearest')
    p_slow = uniform_filter1d(x * x, size=ws, mode='nearest')
    r_fast = np.sqrt(p_fast + 1e-16)
    r_slow = np.sqrt(p_slow + 1e-16)
    gain = r_slow / (r_fast + 1e-8)
    gain = np.clip(gain, 1.0 / max_gain, max_gain)
    wg = max(3, int(0.022 * sr))
    gain = uniform_filter1d(gain, size=wg, mode='nearest')
    return (x * gain).astype(np.float32)


def _shape_start_envelope(env, sr, hold_s=0.9, ramp_s=1.6, knee=0.07):
    """
    Make early approach flatter (as in field recordings) before the rise becomes obvious.
    """
    e = np.clip(np.asarray(env, dtype=np.float32), 0.0, None)
    if hold_s <= 0.0 and ramp_s <= 0.0 and knee <= 0.0:
        peak = np.max(e) + 1e-8
        return (e / peak).astype(np.float32)
    n = len(e)
    if n < 8:
        return e
    e = e / (np.max(e) + 1e-8)

    hold_n = int(max(0.0, hold_s) * sr)
    ramp_n = int(max(1e-3, ramp_s) * sr)
    hold_n = min(hold_n, n)
    ramp_end = min(n, hold_n + ramp_n)

    out = e.copy()
    if hold_n > 0:
        base = float(np.mean(out[: max(1, hold_n)]))
        out[:hold_n] = base
    if ramp_end > hold_n:
        t = np.linspace(0.0, 1.0, ramp_end - hold_n, endpoint=False, dtype=np.float32)
        s = t * t * (3.0 - 2.0 * t)  # smoothstep
        out[hold_n:ramp_end] = (1.0 - s) * out[hold_n] + s * out[hold_n:ramp_end]

    # Keep very low levels near floor from rising too quickly.
    k = float(np.clip(knee, 0.0, 0.4))
    out = np.maximum(out - k, 0.0) / max(1e-8, (1.0 - k))
    out = uniform_filter1d(out, size=max(3, int(0.05 * sr)), mode='nearest')
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _stabilize_cpa_hump(out, env, sr, strength=0.30, win_fast_ms=30.0, win_slow_ms=360.0):
    """
    Stabilize RMS ripple near CPA where coherent harmonics are strongest.
    This applies extra slow-vs-fast leveling only near the envelope peak so
    tails keep their natural texture.
    """
    x = np.asarray(out, dtype=np.float64)
    e = np.clip(np.asarray(env, dtype=np.float64), 0.0, 1.0)
    if len(x) < 8 or len(e) != len(x):
        return x.astype(np.float32)

    wf = max(3, int((win_fast_ms * 1e-3) * sr))
    ws = max(wf + 1, int((win_slow_ms * 1e-3) * sr))
    p_fast = uniform_filter1d(x * x, size=wf, mode='nearest')
    p_slow = uniform_filter1d(x * x, size=ws, mode='nearest')
    r_fast = np.sqrt(p_fast + 1e-16)
    r_slow = np.sqrt(p_slow + 1e-16)

    # Only act strongly around CPA; leave approach/recede mostly untouched.
    cpa_mask = np.power(e, 2.4)
    cpa_mask = uniform_filter1d(cpa_mask, size=max(3, int(0.12 * sr)), mode='nearest')
    raw_gain = r_slow / (r_fast + 1e-8)
    blended_gain = 1.0 + strength * cpa_mask * (raw_gain - 1.0)
    # Tighter limits + slower control reduce audible pumping near peak.
    blended_gain = np.clip(blended_gain, 0.90, 1.12)
    w_g = max(3, int(0.08 * sr))
    blended_gain = uniform_filter1d(blended_gain, size=w_g, mode='nearest')
    return (x * blended_gain).astype(np.float32)


def _add_light_reverb(audio, sr, rng, amount=1.0):
    """Add 2-3 short roadside reflections (10-50 ms)."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-4:
        return audio.astype(np.float32)
    n = len(audio)
    out = audio.astype(np.float32).copy()
    n_taps = int(rng.integers(2, 4))
    for _ in range(n_taps):
        delay_ms = rng.uniform(10.0, 50.0)
        gain = amount * rng.uniform(0.08, 0.22)
        d = int(delay_ms * 1e-3 * sr)
        if d <= 0 or d >= n:
            continue
        out[d:] += gain * audio[:-d]
    return out


def _enrich_spectral_realism(doppler_audio, amplitudes, sr, rng, params=None):
    """
    Add realism layers while preserving primary Doppler kinematics.
    Tuned for field-recording-like level (lower peak than raw synthesis) and CPA broadband energy.
    Optional params: mic_peak_target, envelope_smooth_fast_ms, envelope_smooth_slow_ms,
    envelope_smooth_max_gain, realism_broadband_scale (multiplier on textured noise layers),
    cpa_hump_stabilize_strength, start_flat_hold_s, start_flat_ramp_s, start_flat_knee,
    wow_flutter_amount, roadside_reverb_amount.
    """
    p = params or {}
    # Typical handheld roadside peaks ~0.04–0.07 in normalized plots; match real A/B dashboards.
    target_peak = float(p.get('mic_peak_target', 0.056))
    bb_scale = float(np.clip(float(p.get('realism_broadband_scale', 1.0)), 0.25, 3.0))
    env_fast = float(p.get('envelope_smooth_fast_ms', 8.0))
    env_slow = float(p.get('envelope_smooth_slow_ms', 220.0))
    env_max = float(p.get('envelope_smooth_max_gain', 1.14))
    wow_flutter_amount = float(np.clip(float(p.get('wow_flutter_amount', 0.38)), 0.0, 1.0))
    roadside_reverb_amount = float(np.clip(float(p.get('roadside_reverb_amount', 0.42)), 0.0, 1.0))
    # Keep a realistic pass-by contour by default:
    # flatter onset, gradual rise, and less early over-activation.
    cpa_hump_stabilize_strength = float(np.clip(float(p.get('cpa_hump_stabilize_strength', 0.0)), 0.0, 1.0))
    start_flat_hold_s = float(p.get('start_flat_hold_s', 0.95))
    start_flat_ramp_s = float(p.get('start_flat_ramp_s', 1.7))
    start_flat_knee = float(p.get('start_flat_knee', 0.06))

    n = len(doppler_audio)
    if n == 0:
        return doppler_audio

    nyq = sr / 2.0

    # Envelope proxy from physics amplitude — smooth so additive noise does not shred the pass-by shape.
    env = np.asarray(amplitudes, dtype=np.float32)
    if len(env) != n:
        env = np.interp(np.linspace(0, len(env) - 1, n), np.arange(len(env)), env)
    env = np.clip(env, 0.0, None)
    env = env / (np.max(env) + 1e-8)
    sg_w = 101 if n >= 101 else max(5, (n // 2) * 2 + 1)
    env = signal.savgol_filter(env, sg_w, 2, mode='interp')
    env = np.clip(env, 0.0, 1.0).astype(np.float32)
    env = _shape_start_envelope(
        env, sr, hold_s=start_flat_hold_s, ramp_s=start_flat_ramp_s, knee=start_flat_knee
    )
    env_sq = (env * env).astype(np.float32)
    # Sharper CPA weighting for broadband splash (still smooth — env already Savitzky–Golay filtered).
    env_splash = np.power(env, 1.65).astype(np.float32)

    out = doppler_audio.astype(np.float32).copy()

    # Body emphasis (~100–450 Hz): closer to real roadside spectra than bass-only synthesis.
    try:
        sos_body = signal.butter(
            2,
            [max(30.0 / nyq, 1e-5), min(450.0 / nyq, 0.99)],
            btype='band',
            output='sos',
        )
        body = signal.sosfilt(sos_body, out)
        out = out + rng.uniform(0.11, 0.19) * body.astype(np.float32)
    except ValueError:
        pass

    # (1) Coarse engine/road texture (mostly stays with tonal path through distance color).
    white = rng.standard_normal(n).astype(np.float32)
    lo_f = 50.0 / nyq
    hi_f = min(2400.0 / nyq, 0.99)
    if hi_f > lo_f:
        b_bp, a_bp = signal.butter(4, [lo_f, hi_f], btype='bandpass')
        band_noise = signal.lfilter(b_bp, a_bp, white).astype(np.float32)
        band_noise /= (np.max(np.abs(band_noise)) + 1e-8)
        texture_level = bb_scale * rng.uniform(0.055, 0.11)
        out += texture_level * env * band_noise

    # (2) Wow/flutter before distance-dependent coloring (kinematics on sustained spectrum).
    out = _apply_harmonic_jitter(out, sr, rng, amount=wow_flutter_amount)

    # (3) Near/far on core signal only — do NOT lowpass the CPA broadband layers added next,
    # or the vertical “splash” in the spectrogram disappears.
    out = _apply_distance_dependent_lowpass(out, env, sr)

    # (4) “Orangish” mid fill: pink-ish band 180–2200 Hz — persistent road body + scatter (scene brightness).
    mid_lo = 180.0 / nyq
    mid_hi = min(2200.0 / nyq, 0.99)
    if mid_hi > mid_lo:
        b_mid, a_mid = signal.butter(3, [mid_lo, mid_hi], btype='bandpass')
        mid_noise = signal.lfilter(b_mid, a_mid, _pink_noise(n, rng)).astype(np.float32)
        mid_noise /= (np.max(np.abs(mid_noise)) + 1e-8)
        out += bb_scale * rng.uniform(0.12, 0.20) * (0.28 + 0.72 * env) * mid_noise

    # (5) CPA broadband air / tire / wind — peaks at closest approach; fills 0.3–8 kHz in the plot.
    for lo_hz, hi_hz, gain_lo, gain_hi in (
        (240.0, 2600.0, 0.19, 0.33),
        (1600.0, min(8500.0, nyq * 0.98), 0.14, 0.26),
    ):
        lo_b = lo_hz / nyq
        hi_b = min(hi_hz / nyq, 0.99)
        if hi_b <= lo_b:
            continue
        b_a, a_a = signal.butter(3, [lo_b, hi_b], btype='bandpass')
        layer = signal.lfilter(b_a, a_a, rng.standard_normal(n).astype(np.float32))
        layer /= (np.max(np.abs(layer)) + 1e-8)
        g = bb_scale * rng.uniform(gain_lo, gain_hi)
        out += g * env_splash * layer.astype(np.float32)

    # (6) Scene noise floor — real mics show textured grain across the band, not empty purple.
    floor_db = rng.uniform(-44.0, -34.0)
    floor_amp = 10.0 ** (floor_db / 20.0)
    ambient = (0.35 * rng.standard_normal(n).astype(np.float32) + 0.65 * _pink_noise(n, rng))
    ambient /= (np.max(np.abs(ambient)) + 1e-8)
    out += bb_scale * floor_amp * ambient

    # (7) Light reverberation (spreads energy slightly in time/frequency).
    out = _add_light_reverb(out, sr, rng, amount=roadside_reverb_amount)

    # (8) Level stability: noise + tremolo create fast RMS spikes; gentle slow-vs-fast leveling.
    out = _smooth_short_term_level(
        out, sr, win_fast_ms=env_fast, win_slow_ms=env_slow, max_gain=env_max
    )
    if cpa_hump_stabilize_strength > 1e-4:
        out = _stabilize_cpa_hump(
            out, env, sr, strength=cpa_hump_stabilize_strength, win_fast_ms=26.0, win_slow_ms=340.0
        )

    # Match typical recorded mic peak (~0.05–0.12) unless overridden.
    peak = np.max(np.abs(out)) + 1e-8
    out = out * (target_peak / peak)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _broaden_doppler_curves(freq_ratios, amplitudes, broaden_factor=1.3):
    """
    Spread Doppler change over more time (slower approach/recede transition).
    broaden_factor > 1.0 makes the Doppler sweep wider in time.
    """
    fr = np.asarray(freq_ratios, dtype=np.float32)
    amp = np.asarray(amplitudes, dtype=np.float32)
    n = min(len(fr), len(amp))
    if n < 8 or broaden_factor <= 1.0:
        return fr, amp

    fr = fr[:n]
    amp = amp[:n]

    # Stronger, deterministic broadening:
    # 1) smooth in time,
    # 2) flatten the CPA peak slightly.
    # This preserves overall shape while avoiding a narrow, spiky pass-by.
    win = int(max(9, (broaden_factor * n * 0.06)))
    if win % 2 == 0:
        win += 1
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 5:
        return fr, amp

    fr_s = signal.savgol_filter(fr, window_length=win, polyorder=2, mode='interp').astype(np.float32)
    amp_s = signal.savgol_filter(amp, window_length=win, polyorder=2, mode='interp').astype(np.float32)

    # Gentle compression widens perceived envelope around CPA.
    amp_s = np.clip(amp_s, 0.0, None)
    amp_max = np.max(amp_s) + 1e-8
    amp_n = amp_s / amp_max
    # Keep close to linear so early approach does not jump up too quickly.
    gamma = 0.98  # <1 broadens/softens; values near 1 preserve a natural gradual onset
    amp_b = np.power(amp_n, gamma) * amp_max

    # Slow envelope smoothing (~350 ms) — uniform_filter1d is O(n); direct convolve was O(n * window).
    smooth_len = int(max(9, min(0.35 * SR, n)))
    if smooth_len > n:
        smooth_len = n
    amp_b = uniform_filter1d(amp_b.astype(np.float32), size=max(1, smooth_len), mode='nearest')

    # Keep frequency ratios physically plausible and close to baseline range.
    fr_lo = float(np.min(fr))
    fr_hi = float(np.max(fr))
    fr_b = np.clip(fr_s, fr_lo, fr_hi).astype(np.float32)
    return fr_b, amp_b.astype(np.float32)


def _enforce_passby_envelope_shape(
    amplitudes,
    strength=0.82,
    attack_gamma=1.9,
    release_gamma=1.45,
    edge_floor=0.09,
):
    """
    Enforce a realistic pass-by amplitude contour:
    near-flat start, gradual rise to CPA, gradual fall, near-flat tail.
    """
    amp = np.asarray(amplitudes, dtype=np.float32)
    n = len(amp)
    if n < 8:
        return amp

    amp = np.clip(amp, 0.0, None)
    amax = float(np.max(amp))
    if amax <= 1e-8:
        return amp

    s = float(np.clip(strength, 0.0, 1.0))
    atk_g = float(np.clip(attack_gamma, 1.05, 3.5))
    rel_g = float(np.clip(release_gamma, 1.05, 3.5))
    floor = float(np.clip(edge_floor, 0.0, 0.35))

    a_norm = amp / amax
    peak_idx = int(np.argmax(a_norm))
    peak_idx = int(np.clip(peak_idx, 1, n - 2))

    i = np.arange(n, dtype=np.float32)
    left_len = float(max(1, peak_idx))
    right_len = float(max(1, (n - 1) - peak_idx))

    left_t = np.clip(i / left_len, 0.0, 1.0)
    right_t = np.clip((i - peak_idx) / right_len, 0.0, 1.0)

    attack = np.power(left_t, atk_g)
    release = np.power(1.0 - right_t, rel_g)
    target = np.where(i <= peak_idx, attack, release)

    # Keep boundaries slightly above zero while still mostly flat.
    target = floor + (1.0 - floor) * target
    target = np.clip(target, 0.0, 1.0).astype(np.float32)

    # Stabilize tiny local bumps while preserving broad shape.
    win = max(5, (n // 120) * 2 + 1)
    target = signal.savgol_filter(target, window_length=win, polyorder=2, mode='interp').astype(np.float32)
    target = np.clip(target, 0.0, 1.0)

    shaped = (1.0 - s) * a_norm + s * target
    return (np.clip(shaped, 0.0, 1.0) * amax).astype(np.float32)


# batch config validation

def validate_batch_config(config):
    """Validate batch configuration (GLOBAL target only)"""
    batch = config.get('batch', {})
    total_clips = batch.get('total_clips')

    if not total_clips or total_clips < 1:
        return "Total clips must be at least 1"

    vehicles = config.get('vehicles', {}).get('selected', [])
    if not vehicles:
        return "No vehicles selected"

    paths = config.get('paths', {}).get('selected', [])
    if not paths:
        return "No path types selected"

    # DO NOT validate distribution totals anymore because batching is continuous

    return None


# distribution calculation

def calculate_distribution(config, current_batch_size):
    """Calculate vehicle and path distribution for THIS batch"""
    total_clips = current_batch_size
    mode = config['batch'].get('mode', 'auto')

    if mode == 'manual':
        return config['batch']['distribution']

    vehicles = config['vehicles']['selected']
    paths = config['paths']['selected']

    clips_per_vehicle = total_clips // len(vehicles)
    clips_per_path = total_clips // len(paths)

    vehicle_dist = {v: clips_per_vehicle for v in vehicles}
    path_dist = {p: clips_per_path for p in paths}

    for i in range(total_clips % len(vehicles)):
        vehicle_dist[vehicles[i]] += 1

    for i in range(total_clips % len(paths)):
        path_dist[paths[i]] += 1

    return {
        'vehicles': vehicle_dist,
        'paths': path_dist
    }


# random parameter generation

def generate_random_parameters(config, vehicle_name, path_type, force_symmetric=False):
    params = {}

    def get_sampler(key, lo, hi):
        if key not in SAMPLERS:
            SAMPLERS[key] = CyclicIntegerSampler(lo, hi)
        return SAMPLERS[key].next()

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # speed
    gmin, gmax = DEFAULT_RANGES['speed'].get(
        vehicle_name.lower(),
        DEFAULT_RANGES['speed']['default']
    )

    if config.get('speed', {}).get('randomize', True):
        umin = float(config['speed'].get('min', gmin))
        umax = float(config['speed'].get('max', gmax))
        lo = clamp(umin, gmin, gmax)
        hi = clamp(umax, gmin, gmax)
        if lo > hi:
            lo, hi = hi, lo
        if abs(lo - hi) < 1e-4:
            params['speed'] = lo
        else:
            params['speed'] = get_sampler(f"speed_{vehicle_name}", int(lo), int(hi))
    else:
        params['speed'] = clamp(float(config['speed'].get('value', 30)), gmin, gmax)

    # distance
    dmin, dmax = DEFAULT_RANGES['distance']

    if config.get('distance', {}).get('randomize', True):
        umin = float(config['distance'].get('min', dmin))
        umax = float(config['distance'].get('max', dmax))
        lo = clamp(umin, dmin, dmax)
        hi = clamp(umax, dmin, dmax)
        if lo > hi:
            lo, hi = hi, lo
        if abs(lo - hi) < 1e-4:
            params['distance'] = lo
        else:
            params['distance'] = float(get_sampler("distance", int(lo), int(hi)))
    else:
        params['distance'] = clamp(float(config['distance'].get('value', 30)), dmin, dmax)

    # DURATION (batch / UI; do not override with a fixed 10 s)
    dur_cfg = config.get('duration', 10.0)
    if isinstance(dur_cfg, dict):
        if dur_cfg.get('randomize', False):
            d_lo = float(dur_cfg.get('min', 10.0))
            d_hi = float(dur_cfg.get('max', 15.0))
            if d_lo > d_hi:
                d_lo, d_hi = d_hi, d_lo
            params['duration'] = float(random.uniform(d_lo, d_hi))
        else:
            params['duration'] = float(dur_cfg.get('value', dur_cfg.get('min', 10.0)))
    else:
        params['duration'] = float(dur_cfg)
    params['duration'] = max(0.5, min(300.0, params['duration']))

    # straight
    if path_type == 'straight':
        amin, amax = DEFAULT_RANGES['angle']
        if config.get('angle', {}).get('randomize', True):
            lo = clamp(float(config['angle'].get('min', amin)), amin, amax)
            hi = clamp(float(config['angle'].get('max', amax)), amin, amax)
            if abs(lo - hi) < 1e-4:
                params['angle'] = lo
            else:
                params['angle'] = float(get_sampler("angle", int(min(lo, hi)), int(max(lo, hi))))
        else:
            params['angle'] = clamp(float(config['angle'].get('value', 0)), amin, amax)

    # parabola
    elif path_type == 'parabola':
        a_lo, a_hi = DEFAULT_RANGES['parabola_a']
        a_int = get_sampler("parabola_a", a_lo, a_hi)
        params['a'] = a_int / 10000.0
        # For parabola, 'h' is the vertex height, which is the closest distance
        params['h'] = params['distance']

    # bezier
    elif path_type == 'bezier':
        # Ensure centering by basing x-span on speed * duration
        span = params['speed'] * params['duration']
        half_span = 0.5 * span
        
        # Determine direction
        if random.random() > 0.5:
            x_start, x_end = -half_span, half_span
        else:
            x_start, x_end = half_span, -half_span
            
        params['x0'] = x_start
        params['x3'] = x_end
        
        # Sample intermediate control points to keep path interesting but generally centered
        # x1, x2 can be anywhere in [min(x0,x3), max(x0,x3)]
        params['x1'] = get_sampler("bx1_gen", int(min(x_start, x_end)), int(max(x_start, x_end)))
        params['x2'] = get_sampler("bx2_gen", int(min(x_start, x_end)), int(max(x_start, x_end)))
        
        # For y, we want the "dip" or "peak" of the curve to be near the distance param
        # We set y0, y3 to be larger than the closest approach
        dist = params['distance']
        # y_offset: how much the endpoints are pushed back (makes it look like a pass-by)
        y_offset = get_sampler("by_offset", 10, 50) 
        
        params['y0'] = dist + y_offset
        params['y3'] = dist + y_offset
        # Intermediate y control the "bulge" - keep them near dist
        params['y1'] = dist + get_sampler("by1_gen", -2, 5)
        params['y2'] = dist + get_sampler("by2_gen", -2, 5)

    # acceleration (b7)
    bench_cfg = config.get('benchmarks', {})
    selected_benchmarks = bench_cfg.get('selected', [])
    bench_params = bench_cfg.get('params', {})
    b7_enabled = ('B7' in selected_benchmarks) and bool(bench_params.get('enable_acceleration', False))
    use_accel_randomization = b7_enabled or (
        not bench_cfg.get('enabled', False) and config.get('acceleration', {}).get('randomize', False)
    )
    if use_accel_randomization:
        # Default to a small range if not specified
        acc_cfg = config.get('acceleration', {})
        amin, amax = acc_cfg.get('min', -5), acc_cfg.get('max', 5)
        lo, hi = int(min(amin, amax)), int(max(amin, amax))
        params['acceleration'] = get_sampler("acceleration", lo, hi)
    else:
        params['acceleration'] = config.get('acceleration', {}).get('value', 0.0)

    # atmosphere
    tmin, tmax = DEFAULT_RANGES['temperature']
    hmin, hmax = DEFAULT_RANGES['humidity']

    if config.get('atmosphere', {}).get('randomize', True):
        utmin = int(config.get('atmosphere', {}).get('temp_min', tmin))
        utmax = int(config.get('atmosphere', {}).get('temp_max', tmax))
        params['temperature'] = get_sampler("temperature", clamp(utmin, tmin, tmax), clamp(utmax, tmin, tmax))
        
        uhmin = int(config.get('atmosphere', {}).get('hum_min', hmin))
        uhmax = int(config.get('atmosphere', {}).get('hum_max', hmax))
        params['humidity'] = get_sampler("humidity", clamp(uhmin, hmin, hmax), clamp(uhmax, hmin, hmax))
    else:
        params['temperature'] = clamp(int(config.get('atmosphere', {}).get('temperature', 20)), tmin, tmax)
        params['humidity'] = clamp(int(config.get('atmosphere', {}).get('humidity', 50)), hmin, hmax)

    # benchmark constraints
    bench_cfg = config.get('benchmarks', {})
    if bench_cfg.get('enabled', False):
        selected_benchmarks = bench_cfg.get('selected', [])
        bench_params = bench_cfg.get('params', {})
        
        # B5: Time-to-Event Prediction (Target CPA Time)
        if 'B5' in selected_benchmarks:
            params['target_cpa_time'] = float(bench_params.get('cpa_time', 5.0))
            # Clip must span CPA + margin; extend user duration only when necessary.
            # (Old max(10, cpa+2) overwrote e.g. 9.5 s with 10 s whenever B5 was on.)
            min_duration_for_cpa = params['target_cpa_time'] + 2.0
            params['duration'] = float(max(float(params['duration']), min_duration_for_cpa))
            
        # B6: Motion State Segmentation (CPA Window)
        if 'B6' in selected_benchmarks:
            params['cpa_window'] = bench_params.get('cpa_window', 1.0)
            
        # B8: Multi-Object Resolution
        if 'B8' in selected_benchmarks:
            params['num_sources'] = bench_params.get('num_sources', 2)
            
        # B9: Interaction Modeling (Crossing Event)
        if 'B9' in selected_benchmarks:
            params['is_crossing'] = bench_params.get('is_crossing', True)
            if params['is_crossing'] and path_type != 'straight':
                # Force straight or intersection logic elsewhere?
                # For now just set the flag
                pass

    return params


# core audio generation

def get_doppler_audio_array(vehicle_name, path_type, params, method='resample', phase_offset=0.0, pitch_shift=1.0):
    """
    Core logic to generate Doppler-shifted audio array.
    """
    from audio.audio_utils import get_speed_of_sound
    
    # Calculate speed of sound once based on temperature and humidity
    c_sound = get_speed_of_sound(params.get('temperature', 20), params.get('humidity', 50))
    
    # Load vehicle audio
    vehicle_file = None
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
        for folder in folders_to_check:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        vehicle_file = os.path.join(folder, filename)
                        break
            if vehicle_file:
                break

    if not vehicle_file:
        raise FileNotFoundError(f"No audio files found for vehicle '{vehicle_name}'")

    audio_full, sr = librosa.load(vehicle_file, sr=SR, mono=True)
    
    # Normalize source to 1.0 peak to ensure audibility
    max_amp = np.max(np.abs(audio_full))
    if max_amp > 1e-6:
        audio_full = audio_full / max_amp
    target_samples = int(SR * params['duration'])
    # Road / scene tilt (degrees): multi-vehicle and other callers set angle_deg, e.g. from road_angle
    angle_deg = float(params.get('angle_deg', 0.0))

    # Ensure source audio is long enough for pitch-up Doppler shifts
    # (Doppler up-shift consumes source samples faster, so we need a buffer)
    audio = extend_audio_with_overlap(audio_full, params['duration'] * 2.0, SR, start_offset_s=phase_offset)

    # REALISTIC ROADS FIX: Global Curvature Integration
    road_curve_a = params.get('road_curve_a', 0.0)
    accel = float(params.get('acceleration', 0.0))
    
    if road_curve_a != 0:
        # If the road is curved, we use Map Trajectory physics as it handles arbitrary (x,y)
        from visualization.plot_utils import compute_path_points
        # Pivot tilt: prefer explicit scene `road_angle` (multi-scene) over per-path angle_deg
        pivot_angle = float(params.get('road_angle', angle_deg))
        x_pts, y_pts, _ = compute_path_points(
            path_type, params, n_points=500,
            road_curve_a=road_curve_a,
            road_angle=pivot_angle,
            road_y_center=float(params.get('road_y_center', 0.0)),
            observer_pos=params.get('observer_pos', (0, 0)),
            absolute=True,
        )
        points = np.stack([x_pts, y_pts], axis=1)
        freq_ratios, amplitudes = calculate_map_trajectory_doppler(
            points, params['speed'], params['duration'], 
            observer_pos=params.get('observer_pos', (0, 0)),
            c_sound=c_sound
        )
    else:
        # Standard physics modules
        if path_type == 'straight':
            if accel != 0:
                freq_ratios, amplitudes = calculate_straight_line_accelerated_doppler(
                    params['speed'], accel, params['distance'], params.get('angle', 0), params['duration'], c_sound=c_sound
                )
            else:
                freq_ratios, amplitudes = calculate_straight_line_doppler(
                    params['speed'], params['distance'], params.get('angle', 0), params['duration'], c_sound=c_sound
                )
        elif path_type == 'parabola':
            freq_ratios, amplitudes = calculate_parabola_doppler(
                params['speed'], params['a'], params['h'], params['duration'],
                c_sound=c_sound, angle_deg=angle_deg, accel_mps2=accel
            )
        elif path_type == 'bezier':
            freq_ratios, amplitudes = calculate_bezier_doppler(
                params['speed'], params['x0'], params['x1'], params['x2'], params['x3'],
                params['y0'], params['y1'], params['y2'], params['y3'], params['duration'],
                c_sound=c_sound, angle_deg=angle_deg, accel_mps2=accel
            )
        elif path_type in ('map_trajectory', 'map_path'):
            freq_ratios, amplitudes = calculate_map_trajectory_doppler(
                params['points'], params['speed'], params['duration'], 
                observer_pos=params.get('observer_pos', (0, 0)),
                c_sound=c_sound
            )
        else:
            freq_ratios = np.ones(target_samples)
            amplitudes = np.ones(target_samples)

    # Apply retarded-time correction (Warp) to align simulation t=0 with observer t=0.
    # This solves the "blank gap" issue by properly warping the audio from the start.
    try:
        num = len(freq_ratios)
        obs = np.array(params.get('observer_pos', (0.0, 0.0)), dtype=float).reshape(2, 1)
        r = None
        
        if path_type == 'straight':
            t = np.linspace(0.0, params['duration'], num, endpoint=False)
            t0 = params['duration'] / 2.0
            dt = t - t0
            v0 = float(params['speed'])
            angle = np.deg2rad(float(params.get('angle', 0.0)))
            u = np.array([np.cos(angle), np.sin(angle)])
            n = np.array([-np.sin(angle), np.cos(angle)])
            p_c = float(params.get('distance', 10.0)) * n
            s_t = v0 * dt + 0.5 * accel * dt**2
            p = p_c[:, None] + u[:, None] * s_t[None, :]
            r = np.linalg.norm(p - obs, axis=0)
        elif path_type == 'parabola':
            x, y = sample_parabola_path_xy(
                params['speed'], params['a'], params['h'], params['duration'], num,
                angle_deg=float(params.get('angle_deg', 0.0))
            )
            p = np.vstack([x, y])
            r = np.linalg.norm(p - obs, axis=0)
        elif path_type == 'bezier':
            x, y = sample_bezier_path_xy(
                params['speed'],
                params['x0'], params['x1'], params['x2'], params['x3'],
                params['y0'], params['y1'], params['y2'], params['y3'],
                params['duration'], num,
                angle_deg=float(params.get('angle_deg', 0.0))
            )
            p = np.vstack([x, y])
            r = np.linalg.norm(p - obs, axis=0)
        elif path_type in ('map_trajectory', 'map_path'):
            pts = np.asarray(params.get('points', []), dtype=float)
            if pts.ndim == 2 and pts.shape[0] > 0:
                # Arclength sampling for map paths
                from physics.map_trajectory import sample_map_path_xy
                x, y = sample_map_path_xy(pts, params['speed'], params['duration'], num)
                p = np.vstack([x, y])
                r = np.linalg.norm(p - obs, axis=0)

        if r is not None:
            # Warp it properly: align to 'start' (dist[0]) so clip begins at t=0
            freq_ratios, amplitudes = apply_retarded_time_correction(
                freq_ratios, amplitudes, r, c_sound=c_sound, alignment='start'
            )
    except Exception as e:
        print(f"Warning: Retarded-time warp failed: {e}")
        pass

    # Force tonal separation among identical vehicle models
    freq_ratios = freq_ratios * pitch_shift
    # Moderate broadening: enough to avoid spiky CPA, without lifting early approach too much.
    doppler_broaden = float(params.get('doppler_broaden', 1.12))
    freq_ratios, amplitudes = _broaden_doppler_curves(freq_ratios, amplitudes, doppler_broaden)
    # Enforce a realistic pass-by contour (flatter onset/tail + gradual attack/release).
    amplitudes = _enforce_passby_envelope_shape(
        amplitudes,
        strength=float(params.get('passby_envelope_strength', 0.82)),
        attack_gamma=float(params.get('passby_attack_gamma', 1.9)),
        release_gamma=float(params.get('passby_release_gamma', 1.45)),
        edge_floor=float(params.get('passby_edge_floor', 0.09)),
    )

    if method == 'spectral':
        doppler_audio = apply_doppler_to_audio_fixed_alternative(audio, freq_ratios, amplitudes)
    elif method == 'phase':
        doppler_audio = apply_doppler_to_audio_fixed_advanced(audio, freq_ratios, amplitudes)
    else:
        doppler_audio = apply_doppler_to_audio_fixed(audio, freq_ratios, amplitudes)

    # FIXED: Propagation delay is now handled by retarded-time warp above.
    # We disable the legacy shift block to avoid "blank spots" and ensure clip starts at t=0.
    if False and bool(params.get('apply_propagation_delay', False)):
        try:
            obs = np.array(params.get('observer_pos', (0.0, 0.0)), dtype=float)
            r0 = None
            # ... legacy shift logic disabled ...
            pass
        except Exception:
            pass

    # Ensure exact length
    if len(doppler_audio) > target_samples:
        doppler_audio = doppler_audio[:target_samples]
    elif len(doppler_audio) < target_samples:
        padded = np.zeros(target_samples)
        padded[:len(doppler_audio)] = doppler_audio
        doppler_audio = padded

    # Spectral realism enrichment while preserving Doppler macro-structure.
    # Per-source variability comes from clip-local randomized enrichment levels.
    if not bool(params.get('clean_audio', False)):
        # Shift spectral centroid toward roadside/engine band (vehicle WAVs often peak ~40–80 Hz otherwise).
        if bool(params.get('balance_vehicle_spectrum', True)):
            doppler_audio = _preflight_vehicle_spectrum(doppler_audio, SR)

        seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        doppler_audio = _enrich_spectral_realism(doppler_audio, amplitudes, SR, rng, params)

    return doppler_audio, freq_ratios, amplitudes


# numpy feature output
def save_numpy_outputs(doppler_audio, sample_dir, spectrogram_type='cqt', config=None, base_name='spectrogram', essential_dir=None, params=None, freq_limit=1250):
    """
    Compute and save per-frame feature arrays for one clip.
    All arrays share the same frame count T, computed with HOP_LENGTH=512.
    """
    if config is None:
        config = {}
    HOP_LENGTH = 512
    dt = HOP_LENGTH / SR

    #  1. Spectrogram (84, T) 
    if spectrogram_type == 'cqt':
        C = librosa.cqt(doppler_audio, sr=SR, hop_length=HOP_LENGTH,
                        n_bins=84, bins_per_octave=12)
        mag = np.abs(C)
        spec = np.log(1.0 + mag).astype(np.float32)          # (84, t)
        freq_bins = librosa.cqt_frequencies(84,
                        fmin=librosa.note_to_hz('C1'),
                        bins_per_octave=12).astype(np.float32)
        spec_filename = f'{spectrogram_type}.npy'

    elif spectrogram_type == 'stft':
        N_FFT = 2048
        S = librosa.stft(doppler_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag = np.abs(S)[:84, :]
        spec = np.log1p(mag).astype(np.float32)
        freq_bins = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[:84].astype(np.float32)
        spec_filename = f'{spectrogram_type}.npy'

    elif spectrogram_type == 'mel':
        S = librosa.feature.melspectrogram(
                y=doppler_audio, sr=SR, n_mels=84, hop_length=HOP_LENGTH)
        spec = np.log(1.0 + S).astype(np.float32)             # (84, t)
        freq_bins = librosa.mel_frequencies(
                n_mels=84, fmin=0.0, fmax=SR / 2.0).astype(np.float32)
        spec_filename = f'{spectrogram_type}.npy'

    else:
        raise ValueError(f"Unknown spectrogram_type: '{spectrogram_type}'. "
                         "Choose 'cqt', 'stft', or 'mel'.")

    T = spec.shape[1]

    #  2. frequency.npy (T,) [Normalized 0..1] 
    dominant_bin = np.argmax(spec, axis=0)                    # (t,)
    frequency_hz = freq_bins[dominant_bin].astype(np.float32)
    # frequency → [0, 1] normalized by Nyquist
    frequency = frequency_hz / (SR / 2.0)

    #  3. dfdt.npy (T,) [Normalized -1..1] 
    dfdt_raw = np.zeros(T, dtype=np.float32)
    # dfdt = (freq[t] - freq[t-1]) / dt
    dfdt_raw[1:] = (frequency_hz[1:] - frequency_hz[:-1]) / dt
    # dfdt → [-1, 1] normalized by max absolute value
    max_dfdt = np.max(np.abs(dfdt_raw)) + 1e-8
    dfdt = dfdt_raw / max_dfdt

    #  4. rms.npy (T,) [Normalized 0..1] 
    rms_raw = librosa.feature.rms(
                  y=doppler_audio,
                  frame_length=HOP_LENGTH * 2,
                  hop_length=HOP_LENGTH)[0]
    
    # Align to T (rms may be ±1 frame different)
    if len(rms_raw) > T:
        rms_raw = rms_raw[:T]
    elif len(rms_raw) < T:
        rms_raw = np.pad(rms_raw, (0, T - len(rms_raw)))
    
    # rms → [0, 1] normalized by max amplitude
    rms = rms_raw / (np.max(rms_raw) + 1e-8)
    rms = rms.astype(np.float32)

    #  5. spec_topk.npy (T, 3, 2) [Freq Normalized 0..1] 
    K = 3
    spec_topk = np.zeros((T, K, 2), dtype=np.float32)
    for t in range(T):
        frame = spec[:, t]
        idx = np.argsort(frame)[-K:][::-1]
        for k in range(K):
            bin_idx = idx[k]
            spec_topk[t, k, 0] = freq_bins[bin_idx] / (SR / 2.0)
            spec_topk[t, k, 1] = frame[bin_idx]

    #  6. time.npy (T,) tied to real clip duration 
    clip_duration_s = len(doppler_audio) / float(SR)
    time_arr = np.linspace(0.0, clip_duration_s, T, endpoint=False, dtype=np.float32)

    #  Consistency Check 
    assert len(frequency) == T, f"Frequency length {len(frequency)} mismatch with T={T}"
    assert len(dfdt) == T, f"dfdt length {len(dfdt)} mismatch with T={T}"
    assert len(rms) == T, f"RMS length {len(rms)} mismatch with T={T}"

    kinematics = None

    #  Save .npy files 
    common_dir = os.path.join(sample_dir, 'Common')
    os.makedirs(common_dir, exist_ok=True)
    
    np.save(os.path.join(common_dir, spec_filename), spec)
    np.save(os.path.join(common_dir, 'frequency.npy'), frequency)
    np.save(os.path.join(common_dir, 'dfdt.npy'), dfdt)
    np.save(os.path.join(common_dir, 'rms.npy'), rms)
    np.save(os.path.join(common_dir, 'spec_topk.npy'), spec_topk)
    np.save(os.path.join(common_dir, 'time.npy'), time_arr)
    if params is not None:
        v0 = float(params.get('speed', 0.0))
        acc = float(params.get('acceleration', 0.0))
        v_t = np.maximum(1e-3, v0 + acc * time_arr)
        dist = float(params.get('distance', params.get('h', 1.0)))
        alpha_eff = np.abs(acc * dist / max(v0 * v0, 1e-6))
        kinematics = np.column_stack([
            time_arr.astype(np.float32),
            v_t.astype(np.float32),
            np.full_like(time_arr, acc, dtype=np.float32),
            np.full_like(time_arr, float(alpha_eff), dtype=np.float32)
        ])
        np.save(os.path.join(common_dir, 'kinematics.npy'), kinematics)

    #  Save to Essential folder if provided 
    if essential_dir:
        os.makedirs(essential_dir, exist_ok=True)
        np.save(os.path.join(essential_dir, spec_filename), spec)
        np.save(os.path.join(essential_dir, 'time.npy'), time_arr)
        if params is not None:
            np.save(os.path.join(essential_dir, 'kinematics.npy'), kinematics)
        _save_numpy_visualization(
            doppler_audio, spec, frequency, dfdt, rms, spec_topk, time_arr,
            spectrogram_type, essential_dir, generate_diagnostics=False, base_name=base_name,
            freq_limit=freq_limit
        )

    return {
        'spec': spec,
        'frequency': frequency,
        'dfdt': dfdt,
        'rms': rms,
        'time': time_arr,
        'kinematics': kinematics,
        'spec_topk': spec_topk,
        'spec_filename': spec_filename
    }


def _save_numpy_visualization(doppler_audio, spec, frequency, dfdt, rms,
                               spec_topk, time_arr, spectrogram_type, sample_dir,
                               generate_diagnostics=True, base_name='spectrogram',
                               freq_limit=1250):
    """
    Save separate white-background PNGs inside sample_dir.
    spectrogram.png is ALWAYS saved.
    frequency/dfdt/rms/spec_topk.png are saved only if generate_diagnostics is True.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display

    def _white_ax(ax):
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#aaaaaa')
        ax.tick_params(colors='black', labelsize=9)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('#222222')

    #  1. spectrogram.png  (High contrast, magma cmap, truncated Y) 
    try:
        fig, ax = plt.subplots(figsize=(12, 4.8), facecolor='white')
        n_fft = 4096
        hop_length = 512
        win_length = 4096
        stft = librosa.stft(doppler_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Improve contrast (match Audio Comparison logic)
        vmax = float(np.max(D))
        vmin = vmax - 80.0
        
        librosa.display.specshow(D, sr=SR, x_axis='time', y_axis='hz',
                                 ax=ax, hop_length=hop_length, cmap='magma',
                                 vmin=vmin, vmax=vmax, rasterized=True)
        
        ax.set_ylim(0, freq_limit)
        ax.set_yticks(np.linspace(0, freq_limit, 6))
        ax.set_title(base_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        fig.savefig(os.path.join(sample_dir, f'{base_name}_spectrogram.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] spectrogram.png failed: {e}')

    if not generate_diagnostics:
        return

    #  2. frequency.png 
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
        _white_ax(ax)
        ax.plot(time_arr, frequency, color='#1f77b4', linewidth=1.2)
        ax.set_title('Dominant Frequency per Frame (Normalized 0..1)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (norm)')
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(os.path.join(sample_dir, 'frequency.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] frequency.png failed: {e}')

    #  3. dfdt.png 
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
        _white_ax(ax)
        ax.plot(time_arr, dfdt, color='#d62728', linewidth=1.2)
        ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--')
        ax.set_title('Rate of Change of Frequency (Normalized -1..1)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('df/dt (norm)')
        ax.set_ylim(-1.05, 1.05)
        fig.savefig(os.path.join(sample_dir, 'dfdt.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] dfdt.png failed: {e}')

    #  4. rms.png 
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
        _white_ax(ax)
        ax.plot(time_arr, rms, color='#2ca02c', linewidth=1.2)
        ax.set_title('RMS Energy per Frame (Normalized 0..1)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS (norm)')
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(os.path.join(sample_dir, 'rms.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] rms.png failed: {e}')

    #  5. spec_topk.png 
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
        _white_ax(ax)
        colors_topk = ['#ff7f0e', '#9467bd', '#e377c2']
        for rank in range(3):
            ax.plot(time_arr, spec_topk[:, rank, 0],
                    color=colors_topk[rank], linewidth=1.2,
                    label=f'Top-{rank + 1} freq')
        ax.set_title('Top-3 Frequency Components (Normalized Freq)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (norm)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(sample_dir, 'spec_topk.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    except Exception as e:
        print(f'[warn] spec_topk.png failed: {e}')


def save_benchmark_datasets(sample_dir, features, labels, params, config):
    """
    Create specialized folders for B1-B7 benchmarks.
    Only benchmarks listed in config['benchmarks']['selected'] are created.
    """
    bench_cfg = config.get('benchmarks', {})
    if not bench_cfg.get('enabled', False):
        return

    selected = bench_cfg.get('selected', [])
    
    # helper to save label and subset features
    def save_set(b_id, folder_name, b_labels, b_features):
        if b_id not in selected:
            return
        b_dir = os.path.join(sample_dir, folder_name)
        os.makedirs(b_dir, exist_ok=True)
        # Save labels
        for k, v in b_labels.items():
            np.save(os.path.join(b_dir, f'label_{k}.npy'), np.array(v))
        # Save features (duplicates)
        for k, v in b_features.items():
            np.save(os.path.join(b_dir, f'{k}.npy'), v)

    # B1: Speed Estimation
    # Extra: acceleration
    save_set('B1', 'B1_Speed', 
             {'speed': labels['speed_mps'], 'acceleration': params.get('acceleration', 0.0)}, 
             {'frequency': features['frequency']})

    # B2: Direction-of-Travel
    # Extra: approach_angle
    save_set('B2', 'B2_Direction', 
             {'direction': labels['direction_label'], 'angle': params.get('angle', 0.0)}, 
             {'dfdt': features['dfdt']})

    # B3: Distance-of-Closest-Approach
    save_set('B3', 'B3_Distance', 
             {'distance': labels['cpa_distance_m']}, 
             {'rms': features['rms']})

    # B4: Trajectory Shape
    save_set('B4', 'B4_Trajectory', 
             {'trajectory': labels['trajectory_type']}, 
             {features['spec_filename'].replace('.npy', ''): features['spec']})

    # B5: Time-to-Event
    # Extra: time_to_cpa (relative time to CPA at each frame)
    cpa_time = labels['cpa_time_sec']
    time_arr = features['time']
    time_to_cpa = cpa_time - time_arr
    save_set('B5', 'B5_Time_To_Event', 
             {'cpa_time': cpa_time}, 
             {'time': time_arr, 'dfdt': features['dfdt'], 'time_to_cpa': time_to_cpa})

    # B7: Acceleration/Deceleration
    b7_features = {
        'time': features['time'],
        'frequency': features['frequency'],
        'dfdt': features['dfdt'],
    }
    if features.get('kinematics') is not None:
        b7_features['kinematics'] = features['kinematics']
    save_set(
        'B7', 'B7_Acceleration',
        {'acceleration_mps2': labels.get('acceleration_mps2', params.get('acceleration', 0.0))},
        b7_features
    )


def generate_single_clip(vehicle_name, path_type, params, output_dir, batch_id, index, config, custom_filename=None):
    """Generate a single clip: organized into Common/, Essential/, and Benchmark folders"""
    doppler_audio, freq_ratios, amplitudes = get_doppler_audio_array(vehicle_name, path_type, params)
    atm_cfg = config.get('atmosphere', {}) if isinstance(config, dict) else {}
    if bool(atm_cfg.get('add_air_noise', False)):
        noise_strength = float(atm_cfg.get('air_noise_strength', 8.0))
        noise_freq_hz = float(atm_cfg.get('air_noise_frequency_hz', 1200.0))
        doppler_audio = _apply_subtle_air_noise(doppler_audio, SR, noise_strength, noise_freq_hz)

    #  Create per-sample folder and sub-folders 
    sample_dir = os.path.join(output_dir, f'sample_{index:07d}')
    common_dir = os.path.join(sample_dir, 'Common')
    essential_dir = os.path.join(sample_dir, 'Essential')
    
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(essential_dir, exist_ok=True)

    #  Audio 
    output_format = config.get('output', {}).get('format', 'wav')
    
    # Create a clean base name
    if custom_filename:
        base_name = custom_filename
    else:
        # Use km/h for the fallback metadata name
        speed_kmph = round(params['speed'] * 3.6, 2)
        base_name = f"{vehicle_name}_{speed_kmph}_{params['distance']}m_{index:07d}"
        
    filename = f"{base_name}.{output_format}"
    
    # Save to both Common and Essential
    for d in [common_dir, essential_dir]:
        filepath = os.path.join(d, filename)
        if output_format == 'mp3':
            wav_path = filepath.replace('.mp3', '_temp.wav')
            save_audio(doppler_audio, wav_path)
            os.rename(wav_path, filepath.replace('.mp3', '.wav'))
        else:
            save_audio(doppler_audio, filepath)
    
    # Adjust filename for return if it became wav from mp3
    if output_format == 'mp3':
        filename = filename.replace('.mp3', '.wav')

    #  Path plot 
    # save_path_plot already handled dir creation
    include_samples = config.get('batch', {}).get('include_sample_folders', True)
    generate_spec = config.get('output', {}).get('generate_spectrogram', False)

    if include_samples:
        save_path_plot(path_type, params, common_dir, base_name)
        save_path_plot(path_type, params, essential_dir, base_name)
    
    #  Numpy feature arrays + visualization (saves to Common/ and Essential/)
    spectrogram_type = config.get('output', {}).get('spectrogram_type', 'cqt')
    freq_limit = config.get('output', {}).get('freq_limit', 1250)
    
    # If skipping samples, we still need some basic features for validation if needed,
    # but the user wants to reduce compute. save_numpy_outputs generates the spectrogram.
    if include_samples or generate_spec:
        features = save_numpy_outputs(
            doppler_audio, sample_dir, spectrogram_type, config,
            base_name=base_name, essential_dir=essential_dir, params=params,
            freq_limit=freq_limit
        )
    else:
        # Minimal features just for labeling
        features = {
            'time': np.linspace(0, params['duration'], int(params['duration'] * 100)),
            'frequency': np.zeros(100) # Placeholder
        }

    #  Benchmark Labels & B6 Mask 
    # Calculate ground-truth labels
    speed_mps = params.get('speed', 0.0)
    
    # Direction: 0 for Left-to-Right (Approaching then Receding)
    # 1 for Right-to-Left (Receding then Approaching)
    angle = params.get('angle', 0.0)
    direction_label = 0 if (angle < 90 or angle > 270) else 1
    
    cpa_distance = params.get('distance', params.get('h', 0.0))
    
    # Calculate CPA Time
    cpa_time = params.get('target_cpa_time', 5.0)
    
    num_sources = params.get('num_sources', 1)
    is_crossing = params.get('is_crossing', False)

    labels = {
        'speed_mps': speed_mps,
        'acceleration_mps2': float(params.get('acceleration', 0.0)),
        'direction_label': direction_label,
        'cpa_distance_m': cpa_distance,
        'trajectory_type': path_type,
        'cpa_time_sec': cpa_time,
        'num_sources': num_sources,
        'is_crossing': is_crossing,
        'vehicle_class': vehicle_name
    }

    if include_samples:
        # B7 sanity: accelerated clips should not be spectrally flat.
        if abs(float(params.get('acceleration', 0.0))) > 1e-9:
            if float(np.std(features['frequency'])) < 1e-4:
                raise ValueError("Flat frequency evolution for accelerated clip; regenerate sample.")

        #  Save Benchmark-specific Datasets (B1-B5) 
        save_benchmark_datasets(sample_dir, features, labels, params, config)

        # B6 Motion State Segmentation Mask
        if config.get('benchmarks', {}).get('enabled', False) and 'B6' in config.get('benchmarks', {}).get('selected', []):
            window = params.get('cpa_window', 1.0)
            time_arr = features['time']
            # Mask is 1 if within window of CPA
            mask = np.abs(time_arr - cpa_time) <= (window / 2.0)
            b6_dir = os.path.join(sample_dir, 'B6_Segmentation')
            os.makedirs(b6_dir, exist_ok=True)
            np.save(os.path.join(b6_dir, 'segmentation_mask.npy'), mask.astype(np.bool_))

    return {
        'filename': filename,
        'index': index,
        'vehicle': vehicle_name,
        'path_type': path_type,
        'parameters': params,
        'labels': labels,
        'freq_ratio_range': {
            'min': 1.0, # Placeholder if skipped
            'max': 1.0
        },
        'path_plot': f"{base_name}.png",
        'sample_dir': f'sample_{index:07d}'
    }



# audio mixing

def mix_audio_clips(clips_with_delays, target_duration_s=None):
    """
    Mix multiple audio arrays with staggered start times.
    If target_duration_s is provided, the output will be exactly that length.
    """
    if not clips_with_delays:
        return np.array([])

    if target_duration_s is not None:
        max_end_sample = int(target_duration_s * SR)
    else:
        max_end_sample = 0
        for audio, delay_s in clips_with_delays:
            delay_samples = int(delay_s * SR)
            end_sample = delay_samples + len(audio)
            if end_sample > max_end_sample:
                max_end_sample = end_sample

    mixed = np.zeros(max_end_sample)
    for audio, delay_s in clips_with_delays:
        delay_samples = int(delay_s * SR)
        if delay_samples < max_end_sample:
            # Only mix the part that fits within max_end_sample
            available_space = max_end_sample - delay_samples
            to_mix = audio[:available_space]
            mixed[delay_samples : delay_samples + len(to_mix)] += to_mix

    # peak normalization
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.9

    return mixed


# statistics

def generate_statistics(clips_metadata, config):
    """Generate statistics summary with safe guards for empty/missing metadata."""
    stats = []
    stats.append("=" * 60)
    stats.append("BATCH GENERATION STATISTICS")
    stats.append("=" * 60)
    stats.append("")

    total_clips = len(clips_metadata)
    stats.append(f"Total Clips Generated: {total_clips}")
    stats.append("")

    if not total_clips:
        stats.append("No clips generated. Statistics unavailable.")
        return '\n'.join(stats)

    # Vehicle distribution
    stats.append("Vehicle Distribution:")
    vehicles = {}
    for clip in clips_metadata:
        v = clip.get('vehicle', 'unknown')
        vehicles[v] = vehicles.get(v, 0) + 1
    for v, count in sorted(vehicles.items()):
        stats.append(f"  {v}: {count} clips ({count / total_clips * 100:.1f}%)")
    stats.append("")

    # Path distribution
    stats.append("Path Type Distribution:")
    paths = {}
    for clip in clips_metadata:
        p = clip.get('path_type', 'unknown')
        paths[p] = paths.get(p, 0) + 1
    for p, count in sorted(paths.items()):
        stats.append(f"  {p}: {count} clips ({count / total_clips * 100:.1f}%)")
    stats.append("")

    # Helper for safe stats
    def format_stats(label, values, unit):
        if not values:
            stats.append(f"{label}: N/A")
            return
        stats.append(f"{label}:")
        stats.append(f"  Min: {min(values):.1f} {unit}")
        stats.append(f"  Max: {max(values):.1f} {unit}")
        stats.append(f"  Mean: {np.mean(values):.1f} {unit}")
        stats.append(f"  Median: {np.median(values):.1f} {unit}")

    # Speed statistics
    speeds = [clip['parameters']['speed'] for clip in clips_metadata if clip.get('parameters') and 'speed' in clip['parameters']]
    format_stats("Speed Statistics", speeds, "m/s")
    stats.append("")

    # Acceleration statistics
    accelerations = [clip['parameters']['acceleration'] for clip in clips_metadata if clip.get('parameters') and 'acceleration' in clip['parameters']]
    if any(a != 0.0 for a in accelerations):
        format_stats("Acceleration Statistics", accelerations, "m/s^2")
        stats.append("")

    # Distance statistics (handles multi-object offset as fallback)
    distances = []
    for clip in clips_metadata:
        params = clip.get('parameters', {})
        if 'distance' in params:
            distances.append(params['distance'])
        elif 'offset' in params: # B8/B9 fallback
            distances.append(abs(params['offset']))
        elif 'h' in params: # Parabola fallback
            distances.append(params['h'])
            
    format_stats("Distance Statistics", distances, "m")
    stats.append("")

    # Duration statistics
    durations = [clip['parameters']['duration'] for clip in clips_metadata if clip.get('parameters') and 'duration' in clip['parameters']]
    format_stats("Duration Statistics", durations, "s")
    stats.append("")

    # Per-clip listing
    stats.append("Per-Clip Listing:")
    stats.append("-" * 60)
    for idx, clip in enumerate(clips_metadata, start=1):
        params = clip.get('parameters', {})
        speed = params.get('speed', 'N/A')
        accel = params.get('acceleration', 0.0)
        vehicle = clip.get('vehicle', 'unknown')
        
        if isinstance(speed, (int, float)):
            if vehicle == 'multi':
                num = clip.get('num_sources', 1)
                line = f"  Clip {idx:4d}: {speed:6.1f} m/s (avg) | {num} sources (multi)"
                stats.append(line)
                # List individual vehicle speeds
                for v in clip.get('vehicles', []):
                    v_name = v.get('vehicle_name', 'unknown')
                    v_speed = v.get('speed', 0.0)
                    v_accel = v.get('acceleration', 0.0)
                    v_line = f"    - {v_name:15s}: {v_speed:6.1f} m/s"
                    if abs(v_accel) > 1e-6:
                        v_line += f" | Accel: {v_accel:5.1f} m/s^2"
                    stats.append(v_line)
            else:
                line = f"  Clip {idx:4d}: {speed:6.1f} m/s"
                if abs(accel) > 1e-6:
                    line += f" | Accel: {accel:5.1f} m/s^2"
                line += f"  ({vehicle})"
                stats.append(line)
        else:
            stats.append(f"  Clip {idx:4d}: {speed}  ({vehicle})")
    stats.append("")

    stats.append("=" * 60)
    return '\n'.join(stats)


def generate_multi_object_clip(vehicles_configs, output_dir, batch_name, index, config, observer_pos=(0.0, 0.0), custom_filename=None, road_curve_a=0.0, road_y_center=0.0, road_shape='parabola', road_bezier_bulge=0.0, intersection_angle=90.0):
    """
    Generate a clip with multiple sound sources using realistic staggered arrival logic (B8/B9/B10).
    Replaces the flawed intersection model with the 'Busy Road' overlap model.
    """
    from audio.audio_utils import SR, save_audio
    from visualization.plot_utils import save_combined_path_plot

    # Safe duration extraction
    duration = config.get('duration', 10.0)
    if isinstance(duration, dict):
        duration = float(duration.get('value', duration.get('min', 10.0)))
    else:
        duration = float(duration)

    road_angle = config.get('benchmarks', {}).get('params', {}).get('road_angle', 0.0)

    # Setup sample directory
    sample_id = f'sample_{index:07d}'
    sample_dir = os.path.join(output_dir, sample_id)
    common_dir = os.path.join(sample_dir, 'Common')
    essential_dir = os.path.join(sample_dir, 'Essential')
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(essential_dir, exist_ok=True)

    clips_with_delays = []
    individual_files = []
    scenes_data = []
    v_identities = []

    for i, v_cfg in enumerate(vehicles_configs):
        v_name = v_cfg.get('vehicle_name', 'car_1')
        v_identities.append(v_name)
        
        path_type = v_cfg.get('path_type', 'bezier')
        
        # Determine the vehicle's Y-offset and direction from the provided params or config
        # y_offset is the world coordinate of the path's "center" or "offset"
        v_params_input = v_cfg.get('params', {})
        y_offset = v_params_input.get('offset', v_params_input.get('h', v_params_input.get('y0', 5.0)))
        direction = v_cfg.get('direction', 1)
        
        # For physics, we need parameters RELATIVE to the observer at origin
        # observer_pos is (x_obs, y_obs)
        physics_params = v_params_input.copy()
        
        if path_type == 'straight':
            # Straight physics uses 'distance' as CPA distance
            physics_params['distance'] = abs(y_offset - observer_pos[1])
            # Tilt the entire road by road_angle
            physics_params['angle'] = road_angle + (180 if direction == -1 else 0)
        
        elif path_type == 'parabola':
            # Parabola physics uses 'h' as CPA distance from origin
            physics_params['h'] = abs(y_offset - observer_pos[1])
            physics_params['angle_deg'] = road_angle
            
        elif path_type == 'bezier':
            # Bezier physics uses x0..x3, y0..y3 relative to (0,0)
            physics_params['x0'] = v_params_input['x0'] - observer_pos[0]
            physics_params['x1'] = v_params_input['x1'] - observer_pos[0]
            physics_params['x2'] = v_params_input['x2'] - observer_pos[0]
            physics_params['x3'] = v_params_input['x3'] - observer_pos[0]
            physics_params['y0'] = v_params_input['y0'] - observer_pos[1]
            physics_params['y1'] = v_params_input['y1'] - observer_pos[1]
            physics_params['y2'] = v_params_input['y2'] - observer_pos[1]
            physics_params['y3'] = v_params_input['y3'] - observer_pos[1]
            physics_params['angle_deg'] = road_angle
        
        # Pass road configuration to individual physics
        physics_params['road_curve_a'] = road_curve_a
        physics_params['observer_pos'] = observer_pos
        physics_params['road_y_center'] = road_y_center
        physics_params['road_angle'] = float(road_angle)
        if path_type in ('parabola', 'bezier') and abs(road_curve_a) > 0.0:
            # Map trajectory from compute_path: use world params (same as combined path plot) and
            # pivot rotation only (no second rotation about the origin in sample_bezier/parabola).
            physics_params['angle_deg'] = 0.0
            map_params = v_params_input.copy()
            map_params['road_curve_a'] = road_curve_a
            map_params['road_y_center'] = road_y_center
            map_params['road_angle'] = float(road_angle)
            map_params['observer_pos'] = observer_pos
            map_params['angle_deg'] = 0.0
            map_params['road_curve_blend'] = v_params_input.get('road_curve_blend', 1.0)
            map_params['global_curve_scale'] = v_params_input.get('global_curve_scale', 1.0)
            for _k in ['speed', 'duration', 'temperature', 'humidity']:
                if _k in physics_params and _k not in map_params:
                    map_params[_k] = physics_params[_k]
            dop_params = map_params
        else:
            dop_params = physics_params

        # Logic for arrival time as a stagger delay
        delay = v_cfg.get('delay', 0.0)
        if 'arrival_time' in v_cfg and 'delay' not in v_cfg:
            delay = max(0.0, v_cfg['arrival_time'] - 5.0)

        # Generate audio array for this vehicle using relative physics_params
        audio_arr, _, _ = get_doppler_audio_array(v_name, path_type, dop_params)
        
        # Save individual vehicle audio
        v_clean_name = "".join(c for c in v_name if c.isalnum() or c in ('-', '_')).strip()
        v_filename = f"v{i+1}_{v_clean_name}.wav"
        save_audio(audio_arr, os.path.join(common_dir, v_filename))
        save_audio(audio_arr, os.path.join(essential_dir, v_filename))
        individual_files.append(v_filename)

        clips_with_delays.append((audio_arr, delay))
        # For plotting, we pass the ORIGINAL world-coordinate params
        scenes_data.append((path_type, v_params_input, v_name))

    # Align delays so the first vehicle always starts at t=0
    if clips_with_delays:
        min_delay = min(d for _, d in clips_with_delays)
        aligned_clips = []
        for i, (arr, d) in enumerate(clips_with_delays):
            aligned_delay = max(0.0, d - min_delay)
            aligned_clips.append((arr, aligned_delay))
            # Optional: update v_cfg if needed
            vehicles_configs[i]['delay'] = aligned_delay
        clips_with_delays = aligned_clips

    # Mix audio
    mixed_audio = mix_audio_clips(clips_with_delays, target_duration_s=duration)
    atm_cfg = config.get('atmosphere', {}) if isinstance(config, dict) else {}
    if bool(atm_cfg.get('add_air_noise', False)):
        noise_strength = float(atm_cfg.get('air_noise_strength', 8.0))
        noise_freq_hz = float(atm_cfg.get('air_noise_frequency_hz', 1200.0))
        mixed_audio = _apply_subtle_air_noise(mixed_audio, SR, noise_strength, noise_freq_hz)

    # Filename & Identification
    meta_name = f"multi_object_{index:07d}"
    if custom_filename:
        base_name = f"({custom_filename}_){meta_name}"
    else:
        base_name = meta_name
    
    filename = f'{base_name}.wav'
    
    # Save mixed audio
    save_audio(mixed_audio, os.path.join(common_dir, filename))
    save_audio(mixed_audio, os.path.join(essential_dir, filename))

    # Save shared features (spectrogram, etc.)
    spectrogram_type = config.get('output', {}).get('spectrogram_type', 'cqt')
    features = save_numpy_outputs(
        mixed_audio, sample_dir, spectrogram_type, config,
        base_name=base_name, essential_dir=essential_dir, params=None
    )

    # Save combined path plot
    try:
        bench_params = config.get('benchmarks', {}).get('params', {})
        plot_kwargs = {
            'observer_pos': observer_pos, 
            'lane_width': bench_params.get('lane_width', 4.0),
            'road_curve_a': road_curve_a,
            'road_angle': road_angle,
            'road_y_center': road_y_center,
            'road_shape': road_shape,
            'road_bezier_bulge': road_bezier_bulge,
            'absolute': True,
            'intersection_benchmark': bool(bench_params.get('intersection_benchmark', False)),
            'intersection_half_arm': float(bench_params.get('intersection_half_arm', 90.0)),
            'intersection_angle': intersection_angle,
        }
        save_combined_path_plot(scenes_data, common_dir, base_name, **plot_kwargs)
        save_combined_path_plot(scenes_data, essential_dir, base_name, **plot_kwargs)
        # Rename to target base name for metadata simplicity
        os.rename(os.path.join(common_dir, f"{base_name}_combined_path.png"), os.path.join(common_dir, f"{base_name}.png"))
        os.rename(os.path.join(essential_dir, f"{base_name}_combined_path.png"), os.path.join(essential_dir, f"{base_name}.png"))
    except Exception as e:
        print(f"Warning: Multi-object path plot failed: {e}")

    # Benchmark Labels
    is_crossing = any(v.get('is_crossing', False) for v in vehicles_configs)
    labels = {
        'speed_mps': float(np.mean([v.get('speed', 25.0) for v in vehicles_configs])),
        'direction_label': -1, # Mixed
        'cpa_distance_m': float(np.min([abs(v.get('offset', 5.0)) for v in vehicles_configs])),
        'trajectory_type': 'busy_road',
        'num_sources': len(vehicles_configs),
        'is_crossing': is_crossing,
        'vehicle_class': 'multi',
        'v_identities': v_identities,
        'cpa_time_sec': float(np.mean([v.get('delay', 0.0) + 5.0 for v in vehicles_configs]))
    }

    # B1-B5 labels
    save_benchmark_datasets(sample_dir, features, labels, {}, config)

    # B8-B10: Multi-source benchmark folders
    per_vehicle_meta = []
    for i, v_cfg in enumerate(vehicles_configs):
        per_vehicle_meta.append({
            'vehicle_name': v_cfg.get('vehicle_name', f'vehicle_{i}'),
            'speed': float(v_cfg.get('speed', 25.0)),
            'acceleration': float(v_cfg.get('acceleration', 0.0)),
            'offset': float(v_cfg.get('offset', 5.0)),
            'direction': int(v_cfg.get('direction', 1)),
            'delay': float(v_cfg.get('delay', 0.0)),
            'path_type': v_cfg.get('path_type', 'bezier'),
            'is_crossing': bool(v_cfg.get('is_crossing', False)),
            'audio_file': individual_files[i] if i < len(individual_files) else None,
        })

    bench_cfg = config.get('benchmarks', {})
    if bench_cfg.get('enabled', False):
        selected = bench_cfg.get('selected', [])

        if 'B8' in selected:
            b8_dir = os.path.join(sample_dir, 'B8_MultiObject')
            os.makedirs(b8_dir, exist_ok=True)
            np.save(os.path.join(b8_dir, 'label_num_sources.npy'), np.array(len(vehicles_configs)))
            np.save(os.path.join(b8_dir, 'label_per_vehicle_speeds.npy'),
                    np.array([m['speed'] for m in per_vehicle_meta]))
            np.save(os.path.join(b8_dir, 'label_per_vehicle_offsets.npy'),
                    np.array([m['offset'] for m in per_vehicle_meta]))
            np.save(os.path.join(b8_dir, 'label_per_vehicle_delays.npy'),
                    np.array([m['delay'] for m in per_vehicle_meta]))
            np.save(os.path.join(b8_dir, 'frequency.npy'), features['frequency'])
            np.save(os.path.join(b8_dir, f"{features['spec_filename'].replace('.npy', '')}.npy"), features['spec'])
            with open(os.path.join(b8_dir, 'vehicles_meta.json'), 'w') as _f:
                json.dump(per_vehicle_meta, _f, indent=2)

        if 'B9' in selected:
            b9_dir = os.path.join(sample_dir, 'B9_Interaction')
            os.makedirs(b9_dir, exist_ok=True)
            np.save(os.path.join(b9_dir, 'label_is_crossing.npy'), np.array(is_crossing))
            np.save(os.path.join(b9_dir, 'label_num_sources.npy'), np.array(len(vehicles_configs)))
            np.save(os.path.join(b9_dir, 'label_per_vehicle_directions.npy'),
                    np.array([m['direction'] for m in per_vehicle_meta]))
            np.save(os.path.join(b9_dir, 'label_per_vehicle_offsets.npy'),
                    np.array([m['offset'] for m in per_vehicle_meta]))
            np.save(os.path.join(b9_dir, 'label_per_vehicle_delays.npy'),
                    np.array([m['delay'] for m in per_vehicle_meta]))
            np.save(os.path.join(b9_dir, 'dfdt.npy'), features['dfdt'])
            with open(os.path.join(b9_dir, 'vehicles_meta.json'), 'w') as _f:
                json.dump(per_vehicle_meta, _f, indent=2)

        if 'B10' in selected:
            b10_dir = os.path.join(sample_dir, 'B10_SourceIdentity')
            os.makedirs(b10_dir, exist_ok=True)
            np.save(os.path.join(b10_dir, 'label_vehicle_identities.npy'),
                    np.array(v_identities))
            np.save(os.path.join(b10_dir, 'label_num_sources.npy'), np.array(len(vehicles_configs)))
            np.save(os.path.join(b10_dir, f"{features['spec_filename'].replace('.npy', '')}.npy"), features['spec'])
            for i, v_file in enumerate(individual_files):
                src = os.path.join(common_dir, v_file)
                dst = os.path.join(b10_dir, v_file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            with open(os.path.join(b10_dir, 'vehicles_meta.json'), 'w') as _f:
                json.dump(per_vehicle_meta, _f, indent=2)

    return {
        'filename': filename,
        'index': index,
        'vehicle': 'multi',
        'path_type': 'busy_road',
        'parameters': vehicles_configs[0],
        'vehicles': per_vehicle_meta, # Detailed list for statistics
        'labels': labels,
        'path_plot': f"{base_name}.png",
        'sample_dir': sample_id,
        'num_sources': len(vehicles_configs),
        'individual_files': individual_files
    }
