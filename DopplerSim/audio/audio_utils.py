import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.interpolate import interp1d

SOUND_DURATION = 5  # Default seconds - will be overridden by user input
SR = 22050  # Sample rate

def get_speed_of_sound(temp_c, humidity_rh=50.0):
    """
    Calculate speed of sound in air (m/s).
    Formula: c = 331.3 * sqrt(1 + T/273.15)
    Humidity adjustment: c_wet = c_dry * (1 + 0.16 * e/p)
    Simplified humidity impact: c increases by ~0.1-0.6 m/s at high humidity.
    """
    # Base velocity at 0C
    c_base = 331.3
    
    # Temperature effect (most significant)
    # T in Kelvin: 273.15 + temp_c
    # c = 331.3 * sqrt(T / 273.15)
    c_dry = c_base * np.sqrt(1 + temp_c / 273.15)
    
    # Humidity effect (minor)
    # Approx change: 0.1 m/s per 10% RH at 20C
    # We'll use a linear approximation for humidity since it's a minor factor here
    humidity_factor = (humidity_rh / 100.0) * 0.6 
    
    return c_dry + humidity_factor

def load_original_audio(audio_type='horn', duration=5):
    """Load the original audio based on vehicle type and duration"""
    audio_files = {
        'car': 'static/horn.mp3',
        'train': 'static/train.mp3',
        'flight': 'static/flight.mp3',
        'drone': 'static/drone.mp3'  # < added drone mapping
    }

    filename = audio_files.get(audio_type, 'static/horn.mp3')

    try:
        # Load without duration limit first to get the full file
        y, original_sr = librosa.load(filename, sr=SR, mono=True)
        original_duration = len(y) / SR

        print(f"Loaded {audio_type} audio from {filename}")
        print(f"Original duration: {original_duration:.2f}s, Requested: {duration}s")

        if original_duration >= duration:
            # If original is longer than requested, just trim it
            target_samples = int(SR * duration)
            y = y[:target_samples]
            print(f"Trimmed to {duration}s")
        else:
            # If original is shorter, repeat and stitch with overlaps
            y = extend_audio_with_overlap(y, duration, SR)
            print(f"Extended to {duration}s with seamless overlaps")

        return y

    except Exception as e:
        print(f"Could not load {filename}: {e}")
        print(f"Generating fallback {audio_type} sound (duration: {duration}s)...")

        # Generate fallback sounds based on type
        t = np.linspace(0, duration, int(SR * duration))

        if audio_type == 'car':
            # Car horn - multiple harmonics
            audio = (np.sin(2 * np.pi * 440 * t) +
                     0.7 * np.sin(2 * np.pi * 880 * t) +
                     0.4 * np.sin(2 * np.pi * 1320 * t) +
                     0.2 * np.sin(2 * np.pi * 660 * t))
            envelope = np.exp(-t * 0.3) * 0.5 + 0.5

        elif audio_type == 'train':
            # Train - lower frequencies with rhythm
            audio = (np.sin(2 * np.pi * 220 * t) +
                     0.8 * np.sin(2 * np.pi * 110 * t) +
                     0.6 * np.sin(2 * np.pi * 330 * t) +
                     0.3 * np.sin(2 * np.pi * 440 * t))
            # Add rhythmic component for train-like sound
            rhythm = 1 + 0.3 * np.sin(2 * np.pi * 8 * t)
            audio *= rhythm
            envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)

        elif audio_type == 'flight':
            # Flight - higher frequencies with turbine-like sound
            audio = (0.6 * np.sin(2 * np.pi * 800 * t) +
                     0.8 * np.sin(2 * np.pi * 1200 * t) +
                     0.4 * np.sin(2 * np.pi * 1600 * t) +
                     0.3 * np.sin(2 * np.pi * 400 * t))
            # Add turbine-like modulation
            turbine = 1 + 0.2 * np.sin(2 * np.pi * 15 * t)
            audio *= turbine
            envelope = 0.9 + 0.1 * np.sin(2 * np.pi * 3 * t)

        elif audio_type == 'drone':
            # Drone fallback: rotor buzz with harmonics and amplitude modulation
            # multiple rotor harmonics and vibratory texture
            base_freq = 120.0  # rotor fundamental (~120 Hz) - tuneable
            audio = np.zeros_like(t)
            harmonics = [1, 2, 3, 4, 6]  # include some non-integer-like components
            for h in harmonics:
                # slight inharmonic detune for realism
                detune = 1.0 + np.random.uniform(-0.002, 0.002)
                audio += (0.6 / h) * np.sin(2 * np.pi * base_freq * h * detune * t)

            # Add higher-frequency blades/air interaction
            audio += 0.2 * np.sin(2 * np.pi * 800 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t))

            # Collective/throttle modulation to create rise/fall and micro-variations
            throttle = 1.0 + 0.15 * np.sin(2 * np.pi * 1.5 * t) + 0.05 * np.sin(2 * np.pi * 4.7 * t)
            # Rotor jitter (small, higher-frequency amplitude changes)
            jitter = 1.0 + 0.02 * np.sin(2 * np.pi * 60 * t)
            envelope = 0.8 * throttle * jitter
            # Slight low-pass to simulate rotor body resonance (softening)
            b, a = signal.butter(2, 2000.0 / (SR / 2.0), btype='low')
            audio = signal.lfilter(b, a, audio)

        else:
            # Default fallback (same as car)
            audio = (np.sin(2 * np.pi * 440 * t) +
                     0.7 * np.sin(2 * np.pi * 880 * t) +
                     0.4 * np.sin(2 * np.pi * 1320 * t) +
                     0.2 * np.sin(2 * np.pi * 660 * t))
            envelope = np.exp(-t * 0.3) * 0.5 + 0.5

        return audio * envelope


def extend_audio_with_overlap(original_audio, target_duration, sample_rate, start_offset_s=0.0):
    """
    Extend audio to target duration by repeating with smooth overlaps

    Args:
        original_audio: numpy array of audio samples
        target_duration: desired duration in seconds
        sample_rate: audio sample rate
        start_offset_s: offset in seconds to start the sample from (for decorrelation)

    Returns:
        numpy array: extended audio with seamless transitions
    """
    original_length = len(original_audio)
    original_duration = original_length / sample_rate
    target_length = int(sample_rate * target_duration)

    if original_length >= target_length:
        return original_audio[:target_length]

    # Calculate overlap parameters
    overlap_duration = 0.12  # slightly longer overlap for smoother transitions
    overlap_samples = int(sample_rate * overlap_duration)
    overlap_samples = min(overlap_samples, original_length // 3)  # Don't overlap more than ~33% of original

    print(f"  Using {overlap_samples} samples ({overlap_samples/sample_rate*1000:.0f}ms) overlap")

    # Create extended audio array
    extended_audio = np.zeros(target_length)

    # Apply random start offset
    offset_samples = int(start_offset_s * sample_rate) % original_length
    if offset_samples > 0:
        original_for_chunk = np.roll(original_audio, -offset_samples)
    else:
        original_for_chunk = original_audio.copy()

    # Optional short fade-in at the very start to avoid clicks
    fade_in_length = min(int(sample_rate * 0.02), original_length // 10)  # 20ms or 10% of original
    if fade_in_length > 0:
        first_chunk = original_for_chunk.copy()
        fade_in = np.linspace(0, 1, fade_in_length)
        first_chunk[:fade_in_length] *= fade_in
    else:
        first_chunk = original_for_chunk

    # Copy first iteration completely
    first_copy_len = min(original_length, target_length)
    extended_audio[:first_copy_len] = first_chunk[:first_copy_len]
    current_position = first_copy_len

    # Add subsequent iterations with crossfade overlaps
    iteration = 1
    while current_position < target_length:
        remaining_samples = target_length - current_position

        if remaining_samples <= 0:
            break

        # Start position for this iteration (with overlap)
        start_pos = max(current_position - overlap_samples, 0)

        # How much of the original audio to use
        samples_to_use = min(original_length, remaining_samples + overlap_samples)
        end_pos = start_pos + samples_to_use

        if end_pos > target_length:
            samples_to_use = target_length - start_pos
            end_pos = target_length

        # Create crossfade window for overlap region
        if overlap_samples > 0 and (end_pos - start_pos) > overlap_samples:
            # Fade out existing audio in overlap
            fade_out = np.linspace(1, 0, overlap_samples)
            fade_in = np.linspace(0, 1, overlap_samples)

            overlap_end = start_pos + overlap_samples

            # Fade out existing content
            extended_audio[start_pos:overlap_end] *= fade_out

            # Add faded-in new content
            new_audio_overlap = original_audio[:overlap_samples] * fade_in
            extended_audio[start_pos:overlap_end] += new_audio_overlap

            # Add rest of new audio (non-overlapping part)
            non_overlap_samples = samples_to_use - overlap_samples
            if non_overlap_samples > 0 and overlap_end < end_pos:
                extended_audio[overlap_end:end_pos] = original_audio[overlap_samples:overlap_samples + non_overlap_samples]
        else:
            # No overlap - just concatenate
            extended_audio[start_pos:end_pos] = original_audio[:samples_to_use]

        # Update position for next iteration
        current_position = end_pos
        iteration += 1

    print(f"  Extended audio using {iteration} iterations with smooth crossfades")

    # Apply gentle fade out at the very end to avoid clicks
    fade_length = min(int(sample_rate * 0.05), target_length // 20)  # 50ms or 5% of duration
    if fade_length > 0:
        fade_out_final = np.linspace(1, 0, fade_length)
        extended_audio[-fade_length:] *= fade_out_final

    return extended_audio

# Keep backward compatibility
def load_original_horn():
    """Backward compatibility function"""
    return load_original_audio('car', 5)

def apply_true_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    Apply TRUE Doppler shift that will show proper frequency sweeps in spectrogram.
    This creates strong, visible frequency modulation over time.

    FIXED: Proper time-domain mapping that maintains perfect sync with physics timeline.
    The output length exactly matches len(freq_ratios), ensuring audio and animation sync.
    """
    target_length = len(freq_ratios)
    
    # Ensure original audio is long enough
    # If original audio is shorter than target, we need to extend it
    if len(original_audio) < target_length:
        # Pad with zeros to match target length
        original_audio = np.pad(original_audio, (0, target_length - len(original_audio)))
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes to match target length
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    # Apply minimal smoothing to prevent jitter but preserve Doppler shape
    if len(freq_curve) > 50:
        # Small odd-length window
        window_size = min(21, max(5, (len(freq_curve) // 500) * 2 + 1))
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        freq_curve = np.convolve(freq_curve, kernel, mode='same')
        amp_curve = np.convolve(amp_curve, kernel, mode='same')
    
    print("=" * 60)
    print("APPLYING TRUE DOPPLER SHIFT FOR SPECTROGRAM VISIBILITY")
    print("=" * 60)
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    print(f"Frequency variation: {np.std(freq_curve):.3f}")
    print(f"Target output length: {target_length} samples ({target_length/SR:.2f}s)")
    print(f"Original audio length: {len(original_audio)} samples ({len(original_audio)/SR:.2f}s)")
    
    # NEW APPROACH: Proper time-domain resampling
    # For each output sample, calculate where to sample from the input
    # freq_ratio > 1 means higher pitch = faster playback = advance faster through input
    # freq_ratio < 1 means lower pitch = slower playback = advance slower through input
    
    # The key insight: we want the OUTPUT to have exactly target_length samples
    # spanning the full duration, and we sample from INPUT based on freq_ratios
    
    # Build the input sample position for each output sample
    # Start at position 0, and advance based on freq_ratio at each step
    input_positions = np.zeros(target_length)
    
    for i in range(1, target_length):
        # The frequency ratio tells us how fast to advance through the input
        # freq_ratio = 1.0 means advance 1 sample per output sample (normal speed)
        # freq_ratio = 1.1 means advance 1.1 samples per output sample (10% faster = higher pitch)
        # freq_ratio = 0.9 means advance 0.9 samples per output sample (10% slower = lower pitch)
        step = freq_curve[i]
        input_positions[i] = input_positions[i-1] + step
    
    # Fix: Do not forcefully scale the entire curve to the max input position,
    # as that artificially pitch shifts the audio by ignoring the true Doppler integral.
    # Only scale if we genuinely run out of input buffer to prevent index bounds errors.
    if input_positions[-1] > 0:
        max_input_pos = len(original_audio) - 1
        if input_positions[-1] > max_input_pos:
            input_positions = input_positions * (max_input_pos / input_positions[-1])
    
    # Clip to valid range just in case
    input_positions = np.clip(input_positions, 0, len(original_audio) - 1)
    
    # Sample original audio at these positions using cubic interpolation for higher quality
    resampler = interp1d(np.arange(len(original_audio)), original_audio, kind='cubic', bounds_error=False, fill_value=(original_audio[0], original_audio[-1]))
    output = resampler(input_positions)
    
    # Apply amplitude modulation
    output *= amp_curve
    
    # Verify the effect strength
    print(f"Input position range: 0 to {input_positions[-1]:.1f} (max: {len(original_audio)-1})")
    print(f"Output length: {len(output)} samples (target: {target_length})")
    print(f"Expected strong but smoother frequency sweeps in spectrogram")
    
    return output

def apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    Enhanced spectral method with stronger frequency shifts for visible spectrogram sweeps.
    """
    target_length = len(freq_ratios)
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    # STFT parameters for good time-frequency resolution
    n_fft = 1024  # Smaller for better time resolution
    hop_length = 256  # Smaller hop for smoother frequency transitions
    
    print("=" * 60)
    print("APPLYING SPECTRAL DOPPLER SHIFT")
    print("=" * 60)
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    
    # Compute STFT of original audio
    stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    # Create output STFT
    output_stft = np.zeros_like(stft, dtype=complex)
    
    # Process each time frame with stronger frequency shifting
    n_frames = stft.shape[1]
    for frame_idx in range(n_frames):
        # Get time position for this frame
        time_sample = int(frame_idx * hop_length)
        
        # Get frequency ratio at this time
        if time_sample < len(freq_curve):
            freq_ratio = freq_curve[time_sample]
            amplitude = amp_curve[time_sample]
        else:
            freq_ratio = freq_curve[-1]
            amplitude = amp_curve[-1]
        
        current_magnitude = magnitude[:, frame_idx]
        current_phase = phase[:, frame_idx]
        
        new_magnitude = np.zeros_like(current_magnitude)
        new_phase = np.zeros_like(current_phase)
        
        # Shift each frequency bin
        for freq_idx in range(len(freqs)):
            if freq_idx == 0:  # Skip DC component
                new_magnitude[0] = current_magnitude[0]
                new_phase[0] = current_phase[0]
                continue
                
            old_freq = freqs[freq_idx]
            new_freq = old_freq * freq_ratio
            
            # Find target frequency bin(s)
            if new_freq > 0 and new_freq < freqs[-1]:
                target_idx = np.interp(new_freq, freqs, np.arange(len(freqs)))
                
                lower_idx = int(np.floor(target_idx))
                upper_idx = int(np.ceil(target_idx))
                
                if lower_idx < len(new_magnitude) and upper_idx < len(new_magnitude):
                    weight = target_idx - lower_idx
                    
                    new_magnitude[lower_idx] += current_magnitude[freq_idx] * (1 - weight)
                    new_magnitude[upper_idx] += current_magnitude[freq_idx] * weight
                    
                    new_phase[lower_idx] = current_phase[freq_idx]
                    new_phase[upper_idx] = current_phase[freq_idx]
        
        output_stft[:, frame_idx] = new_magnitude * np.exp(1j * new_phase) * amplitude
    
    output = librosa.istft(output_stft, hop_length=hop_length, length=target_length)
    
    print("Spectral processing complete - should show clear frequency sweeps")
    
    return output

def apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes):
    """
    Enhanced phase modulation method for maximum frequency sweep visibility.
    This creates the strongest and clearest frequency sweeps in spectrograms.
    
    FIXED: Removed artificial 2x amplification to maintain realistic Doppler effect.
    """
    target_length = len(freq_ratios)
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    print("=" * 60)
    print("APPLYING PHASE MODULATION DOPPLER SHIFT")
    print("=" * 60)
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    
    # Instantaneous frequency deviation from unity
    freq_deviation = freq_curve - 1.0
    
    dt = 1.0 / SR
    
    # Create phase trajectory - this directly controls instantaneous frequency
    phase_trajectory = np.zeros(target_length)
    for i in range(1, target_length):
        instantaneous_freq = 1.0 + freq_deviation[i]
        phase_trajectory[i] = phase_trajectory[i-1] + 2 * np.pi * instantaneous_freq * dt
    
    modulation_signal = np.exp(1j * phase_trajectory)
    
    analytic_signal = signal.hilbert(original_audio)
    modulated_signal = analytic_signal * modulation_signal
    
    output = np.real(modulated_signal)
    
    output *= amp_curve
    
    freq_variation = np.std(freq_deviation)
    print(f"Frequency deviation std: {freq_variation:.3f}")
    print(f"Phase trajectory range: {np.min(phase_trajectory):.1f} to {np.max(phase_trajectory):.1f}")
    
    return output

def apply_doppler_to_audio_fixed(original_audio, freq_ratios, amplitudes):
    """
    Main function that applies TRUE Doppler shift with visible spectrogram sweeps
    """
    result = apply_true_doppler_shift(original_audio, freq_ratios, amplitudes)
    return result

def apply_doppler_to_audio_fixed_alternative(original_audio, freq_ratios, amplitudes):
    """
    Alternative using spectral method
    """
    result = apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes)
    return result

def apply_doppler_to_audio_fixed_advanced(original_audio, freq_ratios, amplitudes):
    """
    Advanced method using phase modulation
    """
    result = apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes)
    return result


def apply_retarded_time_correction(freq_ratios, amplitudes, distances, c_sound=343.0, alignment='start'):
    """
    Apply observer-time alignment using a retarded-time approximation:
    t_obs = t_emit + (r - r_ref)/c.
    
    If alignment='start', r_ref = distances[0], so simulation starts at t_obs = 0.
    If alignment='cpa', r_ref = min(distances), so CPA occurs at its original t_emit.
    """
    freq = np.asarray(freq_ratios, dtype=np.float32)
    amp = np.asarray(amplitudes, dtype=np.float32)
    dist = np.asarray(distances, dtype=np.float32)
    n = min(len(freq), len(amp), len(dist))
    if n < 3:
        return freq, amp

    freq = freq[:n]
    amp = amp[:n]
    dist = np.maximum(dist[:n], 1e-6)
    dt = 1.0 / float(SR)
    t_emit = np.arange(n, dtype=np.float32) * dt
    
    if alignment == 'start':
        r_ref = float(dist[0])
    else:
        r_ref = float(np.min(dist))
        
    t_obs = t_emit + (dist - r_ref) / max(1e-6, float(c_sound))

    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    freq = freq[order]
    amp = amp[order]

    # Deduplicate possible equal timestamps for stable interpolation.
    t_unique, unique_idx = np.unique(t_obs, return_index=True)
    freq_unique = freq[unique_idx]
    amp_unique = amp[unique_idx]
    if len(t_unique) < 3:
        return freq_ratios, amplitudes

    out_t = np.linspace(float(t_unique[0]), float(t_unique[-1]), n, endpoint=False, dtype=np.float32)
    freq_corr = np.interp(out_t, t_unique, freq_unique).astype(np.float32)
    amp_corr = np.interp(out_t, t_unique, amp_unique).astype(np.float32)
    return freq_corr, amp_corr

def normalize_amplitudes(amplitudes):
    """Normalize amplitudes to [0, 1] range"""
    if amplitudes:
        max_amp = max(amplitudes)
        if max_amp > 0:
            return [a / max_amp for a in amplitudes]
    return amplitudes

def save_audio(audio_data, output_path):
    """
    Save audio data to file WITHOUT per-clip normalization.
    This preserves relative loudness differences between clips (e.g., different heights).
    """
    # Hard clip to valid range but do not rescale the whole clip
    audio_data = np.clip(audio_data, -1.0, 1.0)
    sf.write(output_path, audio_data, SR)
    return len(audio_data) / SR


def analyze_doppler_effect(original_audio, processed_audio, freq_ratios):
    """
    Analyze the Doppler effect to verify it's working correctly
    """
    print("\n" + "="*50)
    print("DOPPLER EFFECT ANALYSIS")
    print("="*50)
    
    # Compute spectrograms
    n_fft = 2048
    hop_length = 512
    
    orig_stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length)
    proc_stft = librosa.stft(processed_audio, n_fft=n_fft, hop_length=hop_length)
    
    orig_mag = np.abs(orig_stft)
    proc_mag = np.abs(proc_stft)
    
    # Find dominant frequency over time
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    orig_dominant_freqs = []
    proc_dominant_freqs = []
    
    for frame in range(orig_mag.shape[1]):
        # Original audio dominant frequency
        orig_peak_idx = np.argmax(orig_mag[:, frame])
        orig_dominant_freqs.append(freqs[orig_peak_idx])
        
        # Processed audio dominant frequency
        proc_peak_idx = np.argmax(proc_mag[:, frame])
        proc_dominant_freqs.append(freqs[proc_peak_idx])
    
    orig_dominant_freqs = np.array(orig_dominant_freqs)
    proc_dominant_freqs = np.array(proc_dominant_freqs)
    
    # Calculate frequency ratio over time
    actual_freq_ratios = proc_dominant_freqs / (orig_dominant_freqs + 1e-10)
    
    print(f"Expected frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"Actual frequency ratio range: {np.min(actual_freq_ratios):.3f} to {np.max(actual_freq_ratios):.3f}")
    
    # Check if we see the expected frequency sweep
    freq_variation = np.std(actual_freq_ratios)
    print(f"Frequency variation (std): {freq_variation:.3f}")
    
    if freq_variation > 0.05:
        print("GOOD: Significant frequency variation detected - should see sweeps in spectrogram")
    else:
        print("PROBLEM: Little frequency variation - spectrogram may appear flat")
    
    return orig_dominant_freqs, proc_dominant_freqs, actual_freq_ratios

def test_doppler_with_analysis():
    """
    Test function with detailed analysis
    """
    print("Testing Doppler shift with spectrogram analysis...")
    
    # Create test audio
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration))
    
    # Multi-harmonic test signal (like a horn)
    test_audio = (np.sin(2 * np.pi * 440 * t) + 
                  0.7 * np.sin(2 * np.pi * 880 * t) + 
                  0.4 * np.sin(2 * np.pi * 1320 * t))
    
    # Create realistic Doppler frequency ratios (approaching then receding)
    num_points = 100
    t_curve = np.linspace(0, 1, num_points)
    
    # Simulate vehicle passing by (high freq -> low freq)
    freq_ratios = 1.3 * np.exp(-((t_curve - 0.5) / 0.2)**2) + 0.7
    amplitudes = 1.0 / (((t_curve - 0.5) * 40)**2 + 1)
    
    print(f"Test frequency ratios: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    
    # Apply Doppler effect
    result = apply_doppler_to_audio_fixed(test_audio, freq_ratios.tolist(), amplitudes.tolist())
    
    # Analyze results
    analyze_doppler_effect(test_audio, result, freq_ratios)
    
    return test_audio, result, freq_ratios

if __name__ == "__main__":
    # Run test with analysis
    test_doppler_with_analysis()
def apply_distance_fade(amplitudes, fade_duration_s=1.0, sr=SR):
    """
    Apply a smooth fade-in and fade-out to the amplitude envelope 
    to prevent vehicles from 'spawning out of the blue'.
    
    Parameters
    ----------
    amplitudes : np.ndarray
        Array of amplitude values.
    fade_duration_s : float
        Duration of the fade in seconds.
    sr : int
        Sample rate.
        
    Returns
    -------
    np.ndarray
        Tapered amplitude envelope.
    """
    num_samples = len(amplitudes)
    fade_len = int(fade_duration_s * sr)
    
    # Clip fade length to avoid overlapping fades if the clip is very short
    fade_len = min(fade_len, num_samples // 2)
    
    if fade_len <= 0:
        return amplitudes
        
    # Create smooth fade ramps (half-cosine)
    # Using (1 - cos(theta)) / 2 for fade in
    t = np.linspace(0, np.pi, fade_len)
    fade_in = (1 - np.cos(t)) / 2.0
    fade_out = fade_in[::-1]
    
    # Apply to amplitudes
    new_amps = amplitudes.copy()
    new_amps[:fade_len] *= fade_in
    new_amps[-fade_len:] *= fade_out
    
    return new_amps
