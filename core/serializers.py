from rest_framework import serializers
from .models import Instrument, AudioArchive, AudioFile, DetectionResult

class InstrumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Instrument
        fields = '__all__'

class AudioArchiveSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioArchive
        fields = '__all__'
        read_only_fields = ('uploaded_at',)

class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = '__all__'
        read_only_fields = ('created_at',)

class DetectionResultSerializer(serializers.ModelSerializer):
    instrument_name = serializers.CharField(source='instrument.name', read_only=True)
    
    class Meta:
        model = DetectionResult
        fields = '__all__'
        read_only_fields = ('detected_at',)

class ChunkResultSerializer(serializers.Serializer):
    chunk_id = serializers.IntegerField()
    start_time = serializers.FloatField()
    end_time = serializers.FloatField()
    detected_instruments = serializers.ListField(child=serializers.CharField())
    predictions = serializers.DictField()

class AggregatedResultSerializer(serializers.Serializer):
    total_chunks = serializers.IntegerField()
    instrument_confidences = serializers.DictField()
    instrument_detections = serializers.DictField()
    primary_instruments = serializers.ListField(child=serializers.CharField())
    sorted_by_confidence = serializers.ListField(child=serializers.CharField())
    summary = serializers.DictField()

class AudioProcessingResultSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    archive_id = serializers.IntegerField()
    filename = serializers.CharField()
    total_chunks = serializers.IntegerField()
    processed_chunks = serializers.IntegerField()
    chunk_results = ChunkResultSerializer(many=True)
    aggregated_results = AggregatedResultSerializer()
    visualization_data = serializers.DictField()

class VisualizationDataSerializer(serializers.Serializer):
    confidence_chart = serializers.ListField()
    detection_chart = serializers.ListField()
    timeline_chart = serializers.ListField()
    summary = serializers.DictField()
    total_chunks = serializers.IntegerField()