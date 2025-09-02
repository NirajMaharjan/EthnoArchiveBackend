
# import librosa
# import numpy as np

# def chunk_audio(audio_data, sr, chunk_duration=10, min_chunk_duration=8):
#     """
#     Takes preprocessed audio array, returns chunks
#     Discards chunks shorter than min_chunk_duration seconds
#     """
#     chunk_samples = chunk_duration * sr
#     min_chunk_samples = min_chunk_duration * sr
    
#     chunks = []
    
#     for i in range(0, len(audio_data), chunk_samples):
#         chunk = audio_data[i:i+chunk_samples]
#         chunk_duration_actual = len(chunk) / sr
        
#         # Discard if less than 8 seconds
#         if len(chunk) < min_chunk_samples:
#             print(f"Discarding chunk {len(chunks)} - duration: {chunk_duration_actual:.2f}s (< {min_chunk_duration}s)")
#             continue
        
#         # Pad to full 10 seconds if between 8-10 seconds
#         if len(chunk) < chunk_samples:
#             padding_needed = chunk_samples - len(chunk)
#             chunk = np.pad(chunk, (0, padding_needed), mode='constant', constant_values=0)
#             print(f"Padded chunk {len(chunks)} from {chunk_duration_actual:.2f}s to {chunk_duration}s")
        
#         chunks.append({
#             'audio_data': chunk,
#             'start_time': i / sr,
#             'end_time': (i + len(chunk)) / sr,
#             'original_duration': chunk_duration_actual
#         })
    
#     print(f"Created {len(chunks)} valid chunks from audio")
#     return chunks

from io import BytesIO
from pydub import AudioSegment
from ..models import AudioFile
from django.core.files.base import ContentFile

def create_chunks(audio_archive, chunk_duration=10, min_chunk_duration=8):
    audio = AudioSegment.from_file(audio_archive.file.path)
    chunk_ms = chunk_duration * 1000
    min_ms = min_chunk_duration * 1000

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        
        if len(chunk) < min_ms:
            continue  # Skip short chunks
            
        if len(chunk) < chunk_ms:
            # Pad with silence
            silence_needed = chunk_ms - len(chunk)
            chunk += AudioSegment.silent(duration=silence_needed)

        # Create WAV file in memory
        buffer = BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)
        
        # Create ContentFile
        wav_file = ContentFile(buffer.getvalue())
        wav_file.name = f"chunk_{audio_archive.id}_{i:03d}.wav"

        # Save to database
        AudioFile.objects.create(
            archive=audio_archive,
            file=wav_file,
            start_time=start / 1000,
            end_time=(start + chunk_ms) / 1000
        )