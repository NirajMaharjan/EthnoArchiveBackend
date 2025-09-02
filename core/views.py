from django.shortcuts import render
from django.core.files.base import File
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import viewsets, status
from .models import Instrument, AudioArchive, DetectionResult, AudioFile
from .serializers import (
    InstrumentSerializer, 
    AudioArchiveSerializer, 
    DetectionResultSerializer, 
    AudioFileSerializer,
    AudioProcessingResultSerializer
)
from .audio_processing.audio_processor import AudioProcessor
import logging

logger = logging.getLogger(__name__)



@api_view(['POST'])
def upload_and_process_audio(request):
    """Complete audio upload and processing endpoint"""
    if 'file' not in request.FILES:
        return Response({
            'success': False,
            'error': 'No file provided'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        uploaded_file = request.FILES['file']
        
        # Validate file type
        allowed_extensions = ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            return Response({
                'success': False,
                'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create audio archive record
        serializer = AudioArchiveSerializer(data={'file': uploaded_file})
        if not serializer.is_valid():
            return Response({
                'success': False,
                'error': 'Invalid file data',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the audio archive
        archive = serializer.save(
            user=request.user if request.user.is_authenticated else None
        )
        
        logger.info(f"Audio file uploaded: {archive.id}")
        
        # Process the audio file
        processor = AudioProcessor()
        processing_result = processor.process_audio_file(archive)
        
        if not processing_result['success']:
            return Response({
                'success': False,
                'error': 'Audio processing failed',
                'details': processing_result.get('error', 'Unknown error')
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Prepare response data
        response_data = {
            'success': True,
            'message': 'Audio uploaded and processed successfully',
            'archive_id': archive.id,
            'filename': uploaded_file.name,
            'processing_results': processing_result,
            'visualization_data': _prepare_visualization_data(processing_result)
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error in upload_and_process_audio: {str(e)}")
        return Response({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def _prepare_visualization_data(processing_result):
    """Prepare data for frontend visualization"""
    if not processing_result.get('aggregated_results'):
        return {}
    
    aggregated = processing_result['aggregated_results']
    
    # Prepare data for confidence chart
    confidence_data = []
    detection_data = []
    
    for instrument, conf_data in aggregated['instrument_confidences'].items():
        confidence_data.append({
            'instrument': instrument,
            'mean_confidence': round(conf_data['mean_confidence'], 3),
            'max_confidence': round(conf_data['max_confidence'], 3),
            'min_confidence': round(conf_data['min_confidence'], 3)
        })
    
    # Prepare data for detection rate chart
    for instrument, det_data in aggregated['instrument_detections'].items():
        detection_data.append({
            'instrument': instrument,
            'detection_rate': round(det_data['detection_rate'], 3),
            'detection_count': det_data['detection_count'],
            'detected': det_data['detected']
        })
    
    # Timeline data for chunk-by-chunk results
    timeline_data = []
    for chunk_result in processing_result.get('chunk_results', []):
        timeline_data.append({
            'start_time': chunk_result['start_time'],
            'end_time': chunk_result['end_time'],
            'detected_instruments': chunk_result['detected_instruments'],
            'top_confidence': max(chunk_result['predictions']['probabilities'].values()) if chunk_result['predictions']['probabilities'] else 0
        })
    
    return {
        'confidence_chart': confidence_data,
        'detection_chart': detection_data,
        'timeline_chart': timeline_data,
        'summary': aggregated.get('summary', {}),
        'total_chunks': processing_result.get('total_chunks', 0)
    }

@api_view(['GET'])
@permission_classes([AllowAny])
def get_processing_results(request, archive_id):
    """Get detailed processing results for a specific audio archive"""
    try:
        archive = AudioArchive.objects.get(id=archive_id)
        
        # Get all chunks for this archive
        chunks = AudioFile.objects.filter(archive=archive).prefetch_related('results__instrument')
        
        # Get all detection results
        detection_results = DetectionResult.objects.filter(
            audio_file__archive=archive
        ).select_related('instrument', 'audio_file')
        
        # Organize results
        chunk_data = []
        for chunk in chunks:
            chunk_results = chunk.results.all()
            chunk_data.append({
                'chunk_id': chunk.id,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'detections': [
                    {
                        'instrument': result.instrument.name,
                        'confidence': result.confidence
                    }
                    for result in chunk_results
                ]
            })
        
        return Response({
            'success': True,
            'archive_id': archive_id,
            'chunks': chunk_data,
            'total_chunks': len(chunk_data)
        })
        
    except AudioArchive.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Audio archive not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def archives_feed(request):
    try:
        limit = int(request.GET.get('limit', 20))
        archives = AudioArchive.objects.select_related('user').prefetch_related(
            'audiofile_set__results__instrument'
        ).order_by('-uploaded_at')[:limit]
        
        feed_data = []
        for archive in archives:
            # Get all detection results for this archive
            all_results = DetectionResult.objects.filter(
                audio_file__archive=archive
            ).select_related('instrument')
            
            # Aggregate instrument data
            instrument_data = {}
            for result in all_results:
                instrument_name = result.instrument.name
                if instrument_name not in instrument_data:
                    instrument_data[instrument_name] = {
                        'count': 0,
                        'max_confidence': 0
                    }
                
                instrument_data[instrument_name]['count'] += 1
                instrument_data[instrument_name]['max_confidence'] = max(
                    instrument_data[instrument_name]['max_confidence'],
                    result.confidence
                )
            
            # Find top instrument
            top_instrument = None
            top_confidence = 0
            if instrument_data:
                top_instrument = max(
                    instrument_data.keys(),
                    key=lambda x: instrument_data[x]['max_confidence']
                )
                top_confidence = instrument_data[top_instrument]['max_confidence']
            
            feed_data.append({
                'archive_id': archive.id,
                'username': archive.user.username if archive.user else 'Anonymous',
                'file_url': archive.file.url if archive.file else None,
                'uploaded_at': archive.uploaded_at.isoformat(),
                'top_instrument': top_instrument,
                'top_confidence': round(top_confidence, 3) if top_confidence else None,
                'detection_count': len(all_results),
                'instruments': [
                    {
                        'instrument': name,
                        'count': data['count'],
                        'max_confidence': round(data['max_confidence'], 3)
                    }
                    for name, data in instrument_data.items()
                ]
            })
        
        return Response({
            'success': True,
            'results': feed_data,
            'total': len(feed_data)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def user_archives(request):
    try:
        limit = int(request.GET.get('limit', 50))
        archives = AudioArchive.objects.filter(
            user=request.user
        ).select_related('user').prefetch_related(
            'audiofile_set__results__instrument'
        ).order_by('-uploaded_at')[:limit]
        
        user_data = []
        for archive in archives:
            # Get all detection results for this archive
            all_results = DetectionResult.objects.filter(
                audio_file__archive=archive
            ).select_related('instrument')
            
            # Aggregate instrument data
            instrument_data = {}
            for result in all_results:
                instrument_name = result.instrument.name
                if instrument_name not in instrument_data:
                    instrument_data[instrument_name] = {
                        'count': 0,
                        'max_confidence': 0
                    }
                
                instrument_data[instrument_name]['count'] += 1
                instrument_data[instrument_name]['max_confidence'] = max(
                    instrument_data[instrument_name]['max_confidence'],
                    result.confidence
                )
            
            # Find top instrument
            top_instrument = None
            top_confidence = 0
            if instrument_data:
                top_instrument = max(
                    instrument_data.keys(),
                    key=lambda x: instrument_data[x]['max_confidence']
                )
                top_confidence = instrument_data[top_instrument]['max_confidence']
            
            user_data.append({
                'archive_id': archive.id,
                'username': archive.user.username if archive.user else 'Anonymous',
                'file_url': archive.file.url if archive.file else None,
                'uploaded_at': archive.uploaded_at.isoformat(),
                'top_instrument': top_instrument,
                'top_confidence': round(top_confidence, 3) if top_confidence else None,
                'detection_count': len(all_results),
                'instruments': [
                    {
                        'instrument': name,
                        'count': data['count'],
                        'max_confidence': round(data['max_confidence'], 3)
                    }
                    for name, data in instrument_data.items()
                ]
            })
        
        return Response({
            'success': True,
            'results': user_data,
            'total': len(user_data)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
def delete_archive(request, archive_id):
    try:
        archive = AudioArchive.objects.get(id=archive_id, user=request.user)
        archive.delete()
        
        return Response({
            'success': True,
            'message': 'Archive deleted successfully'
        })
        
    except AudioArchive.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Archive not found or access denied'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)