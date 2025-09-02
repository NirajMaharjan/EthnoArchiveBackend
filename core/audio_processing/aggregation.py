import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def aggregate_chunk_predictions(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate predictions from multiple audio chunks
    
    Args:
        chunk_results: List of prediction results from individual chunks
        
    Returns:
        Dictionary containing aggregated results
    """
    if not chunk_results:
        logger.warning("No chunk results provided for aggregation")
        return {}
    
    # Initialize aggregation dictionaries
    instrument_confidences = {}
    instrument_detections = {}
    total_chunks = len(chunk_results)
    
    # Collect all instrument names
    all_instruments = set()
    for result in chunk_results:
        if 'predictions' in result and 'probabilities' in result['predictions']:
            all_instruments.update(result['predictions']['probabilities'].keys())
    
    if not all_instruments:
        logger.warning("No instruments found in chunk results")
        return {}
    
    # Aggregate confidences and detection counts
    for instrument in all_instruments:
        confidences = []
        detection_count = 0
        chunk_confidences = []
        
        for result in chunk_results:
            if 'predictions' in result and 'probabilities' in result['predictions']:
                confidence = result['predictions']['probabilities'].get(instrument, 0.0)
                confidences.append(confidence)
                chunk_confidences.append({
                    'chunk_id': result.get('chunk_id'),
                    'start_time': result.get('start_time'),
                    'end_time': result.get('end_time'),
                    'confidence': confidence
                })
                
                if instrument in result.get('detected_instruments', []):
                    detection_count += 1
        
        if confidences:
            instrument_confidences[instrument] = {
                'mean_confidence': float(np.mean(confidences)),
                'max_confidence': float(np.max(confidences)),
                'min_confidence': float(np.min(confidences)),
                'std_confidence': float(np.std(confidences)),
                'median_confidence': float(np.median(confidences)),
                'chunk_confidences': chunk_confidences
            }
            
            instrument_detections[instrument] = {
                'detection_count': detection_count,
                'detection_rate': detection_count / total_chunks,
                'detected': detection_count > 0,
                'strong_detection': detection_count > (total_chunks * 0.3)  # Detected in >30% of chunks
            }
    
    # Determine primary detected instruments (detected in >30% of chunks)
    primary_instruments = [
        instrument for instrument, data in instrument_detections.items()
        if data['detection_rate'] > 0.3
    ]
    
    # Sort instruments by mean confidence
    sorted_instruments = sorted(
        all_instruments,
        key=lambda x: instrument_confidences[x]['mean_confidence'],
        reverse=True
    )
    
    # Calculate overall statistics
    all_confidences = []
    for instrument_data in instrument_confidences.values():
        all_confidences.extend([conf['confidence'] for conf in instrument_data['chunk_confidences']])
    
    overall_stats = {
        'total_predictions': len(all_confidences),
        'mean_confidence': float(np.mean(all_confidences)) if all_confidences else 0,
        'max_confidence': float(np.max(all_confidences)) if all_confidences else 0,
        'min_confidence': float(np.min(all_confidences)) if all_confidences else 0
    }
    
    # Create summary
    summary = {
        'top_instrument': sorted_instruments[0] if sorted_instruments else None,
        'top_confidence': instrument_confidences[sorted_instruments[0]]['mean_confidence'] if sorted_instruments else 0,
        'instruments_detected': len(primary_instruments),
        'total_instruments_found': len([inst for inst, data in instrument_detections.items() if data['detected']]),
        'confidence_distribution': _calculate_confidence_distribution(instrument_confidences)
    }
    
    return {
        'total_chunks': total_chunks,
        'instrument_confidences': instrument_confidences,
        'instrument_detections': instrument_detections,
        'primary_instruments': primary_instruments,
        'sorted_by_confidence': sorted_instruments,
        'overall_stats': overall_stats,
        'summary': summary
    }

def _calculate_confidence_distribution(instrument_confidences: Dict[str, Any]) -> Dict[str, int]:
    """Calculate distribution of confidence levels"""
    distribution = {
        'very_high': 0,  # >0.8
        'high': 0,       # 0.6-0.8
        'medium': 0,     # 0.4-0.6
        'low': 0,        # 0.2-0.4
        'very_low': 0    # <0.2
    }
    
    for instrument_data in instrument_confidences.values():
        confidence = instrument_data['mean_confidence']
        if confidence > 0.8:
            distribution['very_high'] += 1
        elif confidence > 0.6:
            distribution['high'] += 1
        elif confidence > 0.4:
            distribution['medium'] += 1
        elif confidence > 0.2:
            distribution['low'] += 1
        else:
            distribution['very_low'] += 1
    
    return distribution



