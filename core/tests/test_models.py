import pytest
from django.contrib.auth.models import User
from core.models import Instrument, AudioArchive, AudioFile, DetectionResult

@pytest.mark.django_db
class TestModels:
    def test_instrument_creation(self):
        instrument = Instrument.objects.create(name='Guitar', description='A fretted musical instrument.')
        assert instrument.name == 'Guitar'
        assert str(instrument) == 'Guitar'

    def test_audio_archive_creation(self):
        user = User.objects.create_user(username='testuser', password='testpassword')
        archive = AudioArchive.objects.create(user=user, file='path/to/audio.mp3')
        assert archive.user == user
        assert str(archive) == f"AudioArchive {archive.id} by testuser"

    def test_audio_file_creation(self):
        user = User.objects.create_user(username='testuser', password='testpassword')
        archive = AudioArchive.objects.create(user=user, file='path/to/audio.mp3')
        audio_file = AudioFile.objects.create(archive=archive, file='path/to/chunk.mp3', start_time=0.0, end_time=5.0)
        assert audio_file.archive == archive
        assert str(audio_file) == f"AudioFile {audio_file.id} for Archive {archive.id}"

    def test_detection_result_creation(self):
        user = User.objects.create_user(username='testuser', password='testpassword')
        archive = AudioArchive.objects.create(user=user, file='path/to/audio.mp3')
        audio_file = AudioFile.objects.create(archive=archive, file='path/to/chunk.mp3', start_time=0.0, end_time=5.0)
        instrument = Instrument.objects.create(name='Piano')
        result = DetectionResult.objects.create(audio_file=audio_file, instrument=instrument, confidence=0.95)
        assert result.instrument == instrument
        assert result.confidence == 0.95
        assert str(result) == "Piano detected in AudioFile"
