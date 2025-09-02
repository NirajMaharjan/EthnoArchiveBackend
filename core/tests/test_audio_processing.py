import pytest
from core.audio_processing.audio_processor import AudioProcessor
from core.models import AudioArchive
from django.contrib.auth.models import User
import os

@pytest.fixture
def audio_processor():
    return AudioProcessor()

@pytest.fixture
def test_audio_archive():
    user = User.objects.create_user(username='testuser', password='testpassword')
    # Create a dummy audio file for testing
    with open('test.mp3', 'w') as f:
        f.write('dummy content')
    
    archive = AudioArchive.objects.create(user=user, file='test.mp3')
    return archive

@pytest.mark.django_db
class TestAudioProcessor:
    def test_process_audio_file(self, audio_processor, test_audio_archive):
        result = audio_processor.process_audio_file(test_audio_archive)
        
        os.remove('test.mp3')
        
        assert result['success'] is True
        assert 'chunk_results' in result
        assert 'aggregated_results' in result

    def test_process_audio_chunk(self, audio_processor):
        # This test would require a real audio chunk and a trained model.
        # For now, we'll just test the method's existence.
        assert hasattr(audio_processor, '_process_audio_chunk')

    def test_aggregate_results(self, audio_processor):
        # This test would require some sample chunk results.
        # For now, we'll just test the method's existence.
        assert hasattr(audio_processor, '_aggregate_results')
