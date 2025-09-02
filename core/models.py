from django.db import models
from django.contrib.auth import get_user_model

from django.contrib.auth import get_user_model
User = get_user_model()   

# Instrument table
class Instrument(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name



#full clip
class AudioArchive(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(upload_to="archives/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

#chunks
class AudioFile(models.Model):
    archive = models.ForeignKey(AudioArchive, on_delete=models.CASCADE)
    file = models.FileField(upload_to="chunks/")  # Store actual WAV files
    start_time = models.FloatField()
    end_time = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


#change null = false
class DetectionResult(models.Model):
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE, related_name="results",null=True)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE,null=True)
    confidence = models.FloatField()
    detected_at = models.DateTimeField(auto_now_add=True)

