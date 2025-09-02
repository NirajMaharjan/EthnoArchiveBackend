import librosa
from pydub import AudioSegment
import os
import numpy as np
from .extract_feature_fixed import wav_to_logmelspec

from io import BytesIO
from django.core.files.uploadedfile import UploadedFile
from rest_framework.exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"mp3", "m4a", "aac", "flac", "ogg", "opus", "wav"}

def validate_audio_file(uploaded_file):
    """
    Validate uploaded audio file
    
    Args:
        uploaded_file: Django UploadedFile object
        
    Returns:
        bool: True if valid, raises ValidationError if not
    """
    if not uploaded_file:
        raise ValidationError("No file provided")
    
    # Check file extension
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (limit to 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if uploaded_file.size > max_size:
        raise ValidationError(f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB")
    
    return True

def convert_to_wav(audio_segment_or_path):
    """
    Convert ANY supported audio to a standard WAV (16-bit/22.05kHz/mono).
    
    Args:
        audio_segment_or_path: Either a pydub AudioSegment or file path string
    
    Returns:
        BytesIO: A seek(0)'d buffer containing the .wav bytes.
    """
    try:
        # Handle both AudioSegment objects and file paths
        if isinstance(audio_segment_or_path, str):
            # It's a file path
            audio = AudioSegment.from_file(audio_segment_or_path)
        elif isinstance(audio_segment_or_path, AudioSegment):
            # It's already an AudioSegment
            audio = audio_segment_or_path
        else:
            # Try to load it as a file
            audio = AudioSegment.from_file(audio_segment_or_path)
        
        # Convert to the target spec
        audio = (
            audio
            .set_frame_rate(22050)      # 22.05 kHz
            .set_channels(1)             # mono
            .set_sample_width(2)         # 16-bit (2 bytes = 16 bits)
        )

        # Export to memory buffer
        wav_buffer = BytesIO()
        audio.export(
            wav_buffer,
            format="wav",
            codec="pcm_s16le"
        )
        wav_buffer.seek(0)

        return wav_buffer
        
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {str(e)}")
        raise ValidationError(f"Could not process audio file: {str(e)}")

def load_audio_file(file_path, target_sr=22050):
    """
    Load audio file using librosa
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        audio_data, sr = librosa.load(file_path, sr=target_sr)
        logger.info(f"Loaded audio file: {file_path}, shape: {audio_data.shape}, sr: {sr}")
        return audio_data, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise

def normalize_audio(audio):
    """
    Normalize audio amplitude to [-1, 1]
    
    Args:
        audio: numpy array of audio data
        
    Returns:
        numpy array: normalized audio data
    """
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def extract_audio_features(audio_chunk, feature_type="custom_mel"):
    """
    Extract features from audio chunk for ML processing with guaranteed shape.
    
    Args:
        audio_chunk: numpy array of audio data
        feature_type: type of features to extract ("custom_mel", "mel_spectrogram", "mfcc", "mel")
        
    Returns:
        numpy array: extracted features with guaranteed shape for model compatibility
    """
    try:
        if feature_type == "custom_mel" or feature_type == "mel_spectrogram":
            # Use custom mel spectrogram extraction with shape guarantee (DEFAULT)
            logger.info("Using custom mel-spectrogram feature extraction with shape guarantee")
            return wav_to_logmelspec(audio_chunk, ensure_shape=True)
            
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
            
    except Exception as e:
        logger.error(f"Error extracting {feature_type} features: {str(e)}")
        raise

def calculate_audio_statistics(audio_data):
    """
    Calculate basic statistics about audio data
    
    Args:
        audio_data: numpy array of audio data
        
    Returns:
        dict: audio statistics
    """
    return {
        'duration': len(audio_data) / 22050,  # assuming 22050 Hz
        'sample_count': len(audio_data),
        'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
        'max_amplitude': float(np.max(np.abs(audio_data))),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=22050)[0])),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=22050)[0]))
    }

def detect_silence(audio_data, threshold=0.01, min_silence_duration=0.5, sr=22050):
    """
    Detect silent regions in audio
    
    Args:
        audio_data: numpy array of audio data
        threshold: amplitude threshold for silence detection
        min_silence_duration: minimum duration (seconds) to consider as silence
        sr: sample rate
        
    Returns:
        list: list of (start_time, end_time) tuples for silent regions
    """
    # Calculate frame-wise energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    
    # Normalize energy
    energy = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Find silent frames
    silent_frames = energy < threshold
    
    # Convert to time segments
    silent_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        time = i * hop_length / sr
        
        if is_silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = time
        elif not is_silent and in_silence:
            # End of silence
            in_silence = False
            silence_duration = time - silence_start
            if silence_duration >= min_silence_duration:
                silent_regions.append((silence_start, time))
    
    # Handle case where audio ends in silence
    if in_silence:
        silence_duration = len(audio_data) / sr - silence_start
        if silence_duration >= min_silence_duration:
            silent_regions.append((silence_start, len(audio_data) / sr))
    
    return silent_regions

def trim_silence(audio_data, threshold=0.01, sr=22050):
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio_data: numpy array of audio data
        threshold: amplitude threshold for silence detection
        sr: sample rate
        
    Returns:
        numpy array: trimmed audio data
    """
    # Find non-silent regions
    non_silent = np.abs(audio_data) > threshold
    
    if not np.any(non_silent):
        # Entire audio is silent
        return audio_data
    
    # Find first and last non-silent samples
    first_non_silent = np.argmax(non_silent)
    last_non_silent = len(audio_data) - np.argmax(non_silent[::-1]) - 1
    
    return audio_data[first_non_silent:last_non_silent + 1]

def resample_audio(audio_data, original_sr, target_sr):
    """
    Resample audio to target sample rate
    
    Args:
        audio_data: numpy array of audio data
        original_sr: original sample rate
        target_sr: target sample rate
        
    Returns:
        numpy array: resampled audio data
    """
    if original_sr == target_sr:
        return audio_data
    
    return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)