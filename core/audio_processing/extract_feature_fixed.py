import numpy as np
import librosa

SR        = 22050
DURATION  = 10.0
N_SAMPLES = int(SR*DURATION)
WIN_LEN   = 1024
HOP_LEN   = 512
N_MELS    = 128
FMIN      = 20
FMAX      = SR//2            # Nyquist
PRE_EMPHASIS_ALPHA = 0.97

# Model expected shape
EXPECTED_FRAMES = 431
EXPECTED_SHAPE = (EXPECTED_FRAMES, N_MELS)

def hann(N):
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N-1))

def nextpow2(n):
    return 1<<(n-1).bit_length()

def _bit_reverse_traverse(a):
    n = a.shape[0]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

def _manual_fft(x):
    """
    Complex 1-D FFT, radix-2, length must be power-of-two.
    """
    x = np.asarray(x, dtype=np.complex64)
    n = x.shape[0]
    levels = int(np.log2(n))
    _bit_reverse_traverse(x)

    # Cooley-Tukey butterflies
    for L in range(1, levels+1):
        le = 2**L
        le2 = le//2
        u = np.exp(-2j*np.pi*np.arange(le2)/le)
        for k in range(0, n, le):
            for j in range(le2):
                t = u[j]*x[k+j+le2]
                x[k+j+le2] = x[k+j] - t
                x[k+j]     = x[k+j] + t
    return x

def manual_rfft(sig, n=None):
    """
    Real FFT that returns positive-frequency half (like np.fft.rfft).
    """
    if n is None:
        n = len(sig)
    # zero-pad & make power-of-2
    n = 1 << int(np.ceil(np.log2(n)))
    padded = np.zeros(n, dtype=np.float32)
    padded[:len(sig)] = sig
    c = _manual_fft(padded.astype(np.complex64))
    return c[:n//2+1]

def custom_stft(signal, win_len=WIN_LEN, hop_len=HOP_LEN):
    """Compute the Short-Time Fourier Transform (STFT) of a signal."""
    pad_len = win_len // 2
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='reflect')

    win = hann(win_len)
    n_frames = 1 + (len(padded_signal)-win_len)//hop_len
    fft_len  = nextpow2(win_len)
    spec = np.zeros((n_frames, fft_len//2+1), dtype=np.complex128)

    for t in range(n_frames):
        start = t * hop_len
        end = start + win_len
        chunk = padded_signal[start:end] * win
        fft = manual_rfft(chunk, fft_len)
        spec[t] = fft
    return spec.T          

def hz_to_mel(f):
    return 2595*np.log10(1+f/700)

def mel_to_hz(m):
    return 700*(10**(m/2595)-1)

def ensure_exact_duration(signal, target_samples=N_SAMPLES):
    """
    Ensure audio signal has exactly the target number of samples.
    
    Args:
        signal: Input audio signal
        target_samples: Target number of samples (default: 220,500 for 10s)
    
    Returns:
        numpy array: Signal with exactly target_samples length
    """
    current_length = len(signal)
    
    if current_length == target_samples:
        return signal
    elif current_length > target_samples:
        # Truncate if too long
        return signal[:target_samples]
    else:
        # Pad with zeros if too short
        padding = target_samples - current_length
        return np.pad(signal, (0, padding), mode='constant', constant_values=0)

def ensure_exact_shape(features, target_shape=EXPECTED_SHAPE):
    """
    Ensure mel-spectrogram has exactly the expected shape (431, 128).
    
    Args:
        features: Input mel-spectrogram features
        target_shape: Target shape (frames, mel_bands)
    
    Returns:
        numpy array: Features with exactly target_shape
    """
    target_frames, target_mels = target_shape
    current_frames, current_mels = features.shape
    
    # Handle mel bands dimension (should always be 128)
    if current_mels != target_mels:
        raise ValueError(f"Mel bands mismatch: got {current_mels}, expected {target_mels}")
    
    # Handle time frames dimension
    if current_frames == target_frames:
        return features
    elif current_frames > target_frames:
        # Truncate if too many frames
        return features[:target_frames, :]
    else:
        # Pad with zeros if too few frames
        padding_frames = target_frames - current_frames
        padding = np.zeros((padding_frames, target_mels))
        return np.vstack([features, padding])

def wav_to_logmelspec(signal, sr=22050, ensure_shape=True):
    """
    Convert audio signal to log mel-spectrogram with guaranteed shape.
    
    Args:
        signal: Input audio signal
        sr: Sample rate
        ensure_shape: Whether to ensure exact (431, 128) shape
    
    Returns:
        numpy array: Log mel-spectrogram with shape (431, 128) if ensure_shape=True
    """
    # Ensure exactly 10 seconds of audio (220,500 samples)
    if ensure_shape:
        signal = ensure_exact_duration(signal, N_SAMPLES)
    
    # Extract mel-spectrogram using custom implementation
    spec = custom_stft(signal)                            # (freq, time)
    power  = (np.abs(spec))**2
    fb     = librosa.filters.mel(sr=sr, n_fft=WIN_LEN, n_mels=N_MELS, fmin=0, fmax=sr/2)
    mel    = fb @ power                      # (n_mels, n_frames)
    logmel = np.log(mel + 1e-6)

    logmel_norm = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    features = logmel_norm.T  # Transpose to (n_frames, n_mels)
    
    # Ensure exact shape for model compatibility
    if ensure_shape:
        features = ensure_exact_shape(features, EXPECTED_SHAPE)
    
    return features

def wav_to_logmelspec_original(signal, sr=22050):
    spec = custom_stft(signal)                            
    power  = (np.abs(spec))**2
    fb     = librosa.filters.mel(sr=sr, n_fft=WIN_LEN, n_mels=N_MELS, fmin=0, fmax=sr/2)
    mel    = fb @ power                      
    logmel = np.log(mel + 1e-6)
    logmel_norm = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    return logmel_norm.T

def validate_model_input_shape(features):
    """
    Validate that features have the correct shape for the model.
    
    Args:
        features: Feature array to validate
    
    Returns:
        bool: True if shape is correct
    
    Raises:
        ValueError: If shape is incorrect
    """
    if features.shape != EXPECTED_SHAPE:
        raise ValueError(f"Invalid feature shape: {features.shape}, expected {EXPECTED_SHAPE}")
    
    return True