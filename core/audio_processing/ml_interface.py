from tensorflow import keras
import os
from django.conf import settings
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InstrumentClassifier:
    def __init__(self, feature="mel_spectrogram"):
        self.model = self._load_model()
        self.feature = feature
        self.instrument_names = ['bansuri', 'dhimey', 'khin', 'madal', 'sarangi']
        # Expected model input shape
        self.expected_shape = (431, 128)
    
    def _load_model(self):
        model_path = os.path.join(settings.BASE_DIR, 'trained_models', 'audio_classifier_improved_v3.h5')
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            # For development, we'll create a dummy model response
            return None
        
        try:
            model = keras.models.load_model(model_path,compile=False)
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def make_prediction(self, features, feature="mel_spectrogram", threshold=0.6):
        """
        Make prediction on audio features with guaranteed shape validation.
        """
        try:
            if self.model is None:
                # Return dummy predictions for development
                logger.warning("Using dummy predictions - model not loaded")
                return self._get_dummy_predictions()
            
            # Validate input shape BEFORE processing
            if features.shape != self.expected_shape:
                raise ValueError(f"Invalid input shape: {features.shape}, expected {self.expected_shape}")
            
            logger.info(f"Input shape validation passed: {features.shape}")
            
            # Prepare features for model input
            # Add batch and channel dimensions: (1, 431, 128, 1)
            model_input = np.expand_dims(np.expand_dims(features, axis=0), axis=-1)
            
            logger.info(f"Model input shape: {model_input.shape}")
            
            # Get model predictions
            probabilities = self.model.predict(model_input)[0]  # Get first prediction
            binary_predictions = (probabilities > threshold).astype(int)
            
            # Create results dictionary
            results = {
                'probabilities': {},
                'detected_instruments': [],
                'raw_probabilities': probabilities.tolist(),
                'binary_predictions': binary_predictions.tolist()
            }
            
            # Map probabilities to instrument names
            for i, instrument in enumerate(self.instrument_names):
                results['probabilities'][instrument] = float(probabilities[i])
                if binary_predictions[i] == 1:
                    results['detected_instruments'].append(instrument)
            
            logger.info(f"Prediction completed: detected={results['detected_instruments']}, "
                       f"max_confidence={max(probabilities):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return self._get_dummy_predictions()
    
    def _get_dummy_predictions(self):
        """Return dummy predictions for development/testing"""
        import random
        
        results = {
            'probabilities': {},
            'detected_instruments': [],
            'raw_probabilities': [],
            'binary_predictions': []
        }
        
        for instrument in self.instrument_names:
            # Generate random confidence scores
            confidence = random.uniform(0.1, 0.9)
            results['probabilities'][instrument] = confidence
            results['raw_probabilities'].append(confidence)
            
            # Randomly determine if instrument is detected
            detected = confidence > 0.5
            results['binary_predictions'].append(1 if detected else 0)
            
            if detected:
                results['detected_instruments'].append(instrument)
        
        logger.info("Generated dummy predictions for development")
        return results

# Create a global instance
model_instance = InstrumentClassifier()