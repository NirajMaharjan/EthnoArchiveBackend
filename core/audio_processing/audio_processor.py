import os
import uuid
import librosa
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from django.core.files.base import ContentFile
from django.conf import settings
from .audio_utils import convert_to_wav, normalize_audio, validate_audio_file, calculate_audio_statistics
from .extract_feature_fixed import wav_to_logmelspec, validate_model_input_shape, EXPECTED_SHAPE
from .ml_interface import InstrumentClassifier
from .aggregation import aggregate_chunk_predictions
from ..models import AudioArchive, AudioFile, DetectionResult, Instrument
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.classifier = InstrumentClassifier(feature="mel_spectrogram")
        self.chunk_duration = 10  # seconds
        self.min_chunk_duration = 8  # seconds
        
    def process_audio_file(self, audio_archive):
        """
        Complete audio processing pipeline:
        1. Load and preprocess audio
        2. Create chunks
        3. Extract features and classify each chunk
        4. Aggregate results
        """
        try:
            logger.info(f"Starting processing for audio archive {audio_archive.id}")
            
            # Step 1: Load and preprocess the audio
            audio_data, audio_stats = self._load_and_preprocess_audio(audio_archive)
            
            # Step 2: Create chunks and save to database
            chunk_records = self._create_audio_chunks(audio_archive, audio_data)
            
            # Step 3: Process each chunk and get predictions
            chunk_results = []
            for chunk_record in chunk_records:
                result = self._process_single_chunk(chunk_record)
                if result:
                    chunk_results.append(result)
            
            # Step 4: Aggregate results across all chunks
            aggregated_results = aggregate_chunk_predictions(chunk_results)
            
            logger.info(f"Processing completed for audio archive {audio_archive.id}")
            return {
                'success': True,
                'archive_id': audio_archive.id,
                'total_chunks': len(chunk_records),
                'processed_chunks': len(chunk_results),
                'audio_stats': audio_stats,
                'chunk_results': chunk_results,
                'aggregated_results': aggregated_results
            }
            
        except Exception as e:
            logger.error(f"Error processing audio archive {audio_archive.id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'archive_id': audio_archive.id
            }
    
    def _load_and_preprocess_audio(self, audio_archive):
        """Load audio file and convert to standard format"""
        try:
            # Validate the audio file first
            validate_audio_file(audio_archive.file)
            
            # Load audio using pydub
            audio = AudioSegment.from_file(audio_archive.file.path)
            
            # Convert to standard format (22050 Hz, mono, 16-bit)
            audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
            
            # Calculate basic audio statistics
            # Convert to numpy array for statistics
            wav_buffer = convert_to_wav(audio)
            y, sr = librosa.load(wav_buffer, sr=22050)
            audio_stats = calculate_audio_statistics(y)
            
            logger.info(f"Audio loaded: duration={audio_stats['duration']:.2f}s, "
                       f"samples={audio_stats['sample_count']}, "
                       f"rms_energy={audio_stats['rms_energy']:.4f}")
            
            return audio, audio_stats
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise
    
    def _create_audio_chunks(self, audio_archive, audio_data):
        """Create 10-second chunks from audio and save to database"""
        chunk_ms = self.chunk_duration * 1000
        min_ms = self.min_chunk_duration * 1000
        chunk_records = []
        
        total_duration = len(audio_data) / 1000  # Convert to seconds
        logger.info(f"Creating chunks from {total_duration:.2f}s audio")
        
        for i, start in enumerate(range(0, len(audio_data), chunk_ms)):
            chunk = audio_data[start:start + chunk_ms]
            
            # Skip chunks shorter than minimum duration
            if len(chunk) < min_ms:
                logger.info(f"Skipping chunk {i} - too short ({len(chunk)/1000:.2f}s)")
                continue
            
            # Pad short chunks with silence
            original_duration = len(chunk) / 1000
            if len(chunk) < chunk_ms:
                silence_needed = chunk_ms - len(chunk)
                chunk += AudioSegment.silent(duration=silence_needed)
                logger.info(f"Padded chunk {i} from {original_duration:.2f}s to {self.chunk_duration}s")
            
            # Export chunk to WAV format in memory
            buffer = BytesIO()
            chunk.export(buffer, format="wav")
            buffer.seek(0)
            
            # Create ContentFile and save to database
            wav_file = ContentFile(buffer.getvalue())
            wav_file.name = f"chunk_{audio_archive.id}_{i:03d}.wav"
            
            chunk_record = AudioFile.objects.create(
                archive=audio_archive,
                file=wav_file,
                start_time=start / 1000,
                end_time=min((start + len(chunk)) / 1000, total_duration)
            )
            
            chunk_records.append(chunk_record)
            logger.info(f"Created chunk {i}: {chunk_record.start_time:.2f}s - {chunk_record.end_time:.2f}s")
        
        logger.info(f"Created {len(chunk_records)} chunks total")
        return chunk_records
    
    def _process_single_chunk(self, chunk_record):
        """Process a single audio chunk and get classification results using custom mel-spectrogram"""
        try:
            logger.info(f"Processing chunk {chunk_record.id}")
            
            # Load audio chunk
            y, sr = librosa.load(chunk_record.file.path, sr=22050)
            y = normalize_audio(y)
            
            # Extract custom mel-spectrogram features with guaranteed shape
            mel_features = wav_to_logmelspec(y, sr=22050, ensure_shape=True)
            
            # Validate that we have the exact shape expected by the model
            try:
                validate_model_input_shape(mel_features)
                logger.info(f"Feature shape validation passed: {mel_features.shape}")
            except ValueError as e:
                logger.error(f"Feature shape validation failed: {e}")
                return None
            
            logger.info(f"Extracted custom mel-spectrogram features: shape={mel_features.shape} (guaranteed {EXPECTED_SHAPE})")
            
            # Get predictions from the model using custom mel-spectrogram
            predictions = self.classifier.make_prediction(mel_features, feature="mel_spectrogram")
            
            # OPTIMIZED: Save only detected instruments (confidence > threshold)
            detection_records = []
            detection_threshold = getattr(settings, 'DETECTION_THRESHOLD', 0.8)
            
            for instrument_name in predictions['detected_instruments']:
                confidence = predictions['probabilities'][instrument_name]
                
                # Only save if instrument is actually detected (above threshold)
                if confidence > detection_threshold:
                    # Get or create instrument
                    instrument, created = Instrument.objects.get_or_create(
                        name=instrument_name,
                        defaults={'description': f'Traditional instrument: {instrument_name}'}
                    )
                    
                    if created:
                        logger.info(f"Created new instrument: {instrument_name}")
                    
                    # Create detection result only for detected instruments
                    detection_result = DetectionResult.objects.create(
                        audio_file=chunk_record,
                        instrument=instrument,
                        confidence=confidence
                    )
                    detection_records.append(detection_result)
            
            logger.info(f"Chunk {chunk_record.id} processed: "
                       f"detected={predictions['detected_instruments']}, "
                       f"stored_records={len(detection_records)} (optimized storage)")
            
            return {
                'chunk_id': chunk_record.id,
                'start_time': chunk_record.start_time,
                'end_time': chunk_record.end_time,
                'predictions': predictions,  # Keep full predictions for aggregation
                'detected_instruments': predictions['detected_instruments'],
                'detection_records': [dr.id for dr in detection_records],
                'feature_shape': mel_features.shape
            }
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_record.id}: {str(e)}")
            return None
    
    def get_processing_status(self, archive_id):
        """Get the current processing status of an audio archive"""
        try:
            archive = AudioArchive.objects.get(id=archive_id)
            chunks = AudioFile.objects.filter(archive=archive)
            
            total_chunks = chunks.count()
            processed_chunks = chunks.filter(results__isnull=False).distinct().count()
            
            return {
                'archive_id': archive_id,
                'total_chunks': total_chunks,
                'processed_chunks': processed_chunks,
                'progress': (processed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                'status': 'completed' if processed_chunks == total_chunks else 'processing'
            }
            
        except AudioArchive.DoesNotExist:
            return {
                'archive_id': archive_id,
                'error': 'Archive not found',
                'status': 'error'
            }
        except Exception as e:
            return {
                'archive_id': archive_id,
                'error': str(e),
                'status': 'error'
            }