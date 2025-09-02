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


def wav_to_logmelspec(signal,sr=22050):

    # emphasized_signal = pre_emphasis(signal)
    spec = custom_stft(signal)                            # (freq, time)
    power  = (np.abs(spec))**2
    fb     = librosa.filters.mel(sr=sr, n_fft=WIN_LEN, n_mels=N_MELS, fmin=0, fmax=sr/2)
    # This is matrix multiplication (same as np.dot(fb, power)), introduced in Python 3.5+.
    mel    = fb @ power                      # (n_mels, n_frames)
    logmel = np.log(mel + 1e-6)

    logmel_norm = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    return logmel_norm.T