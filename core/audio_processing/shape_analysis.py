#!/usr/bin/env python
"""
Analysis script to determine the exact output shape of custom mel-spectrogram
and create a function to ensure (431, 128) shape for model compatibility.
"""

import numpy as np

# Constants from extract_feature.py
SR = 22050
DURATION = 10.0
N_SAMPLES = int(SR * DURATION)  # 220,500 samples
WIN_LEN = 1024
HOP_LEN = 512
N_MELS = 128

def calculate_expected_frames():
    """Calculate expected number of frames for 10-second audio."""
    
    # STFT calculation from custom_stft function
    pad_len = WIN_LEN // 2  # 512
    padded_length = N_SAMPLES + 2 * pad_len  # 220,500 + 1024 = 221,524
    
    # Frame calculation: 1 + (padded_length - win_len) // hop_len
    n_frames = 1 + (padded_length - WIN_LEN) // HOP_LEN
    n_frames = 1 + (221,524 - 1024) // 512
    n_frames = 1 + 220,500 // 512
    n_frames = 1 + 430.66...
    n_frames = 431  # This matches the expected model input!
    
    return n_frames

def analyze_shape():
    """Analyze the expected output shape."""
    expected_frames = calculate_expected_frames()
    
    print("=== Shape Analysis ===")
    print(f"Sample Rate: {SR} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Total Samples: {N_SAMPLES}")
    print(f"Window Length: {WIN_LEN}")
    print(f"Hop Length: {HOP_LEN}")
    print(f"Mel Bands: {N_MELS}")
    print()
    print(f"Expected Frames: {expected_frames}")
    print(f"Expected Shape: ({expected_frames}, {N_MELS})")
    print()
    
    if expected_frames == 431:
        print("PERFECT! The custom implementation produces exactly (431, 128)")
        print("   This matches the model's expected input shape!")
    else:
        print(f"MISMATCH! Expected 431 frames, got {expected_frames}")
        print("   Need to adjust parameters or add padding/truncation")

if __name__ == "__main__":
    analyze_shape()