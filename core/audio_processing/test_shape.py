import numpy as np

# Constants from extract_feature.py
SR = 22050
DURATION = 10.0
N_SAMPLES = int(SR * DURATION)  # 220,500 samples
WIN_LEN = 1024
HOP_LEN = 512
N_MELS = 128

# Calculate expected frames
pad_len = WIN_LEN // 2  # 512
padded_length = N_SAMPLES + 2 * pad_len  # 220,500 + 1024 = 221,524

# Frame calculation from custom_stft
n_frames = 1 + (padded_length - WIN_LEN) // HOP_LEN
print(f"Padded length: {padded_length}")
print(f"Calculation: 1 + ({padded_length} - {WIN_LEN}) // {HOP_LEN}")
print(f"Calculation: 1 + {padded_length - WIN_LEN} // {HOP_LEN}")
print(f"Calculation: 1 + {(padded_length - WIN_LEN) // HOP_LEN}")
print(f"Expected frames: {n_frames}")
print(f"Expected shape: ({n_frames}, {N_MELS})")

if n_frames == 431:
    print("PERFECT! Matches model input (431, 128)")
else:
    print(f"MISMATCH! Need {431 - n_frames} more frames")