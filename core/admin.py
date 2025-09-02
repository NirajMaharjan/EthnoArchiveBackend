from django.contrib import admin
from .models import Instrument, AudioArchive, DetectionResult,AudioFile

# Register your models here.
admin.site.register(Instrument)
admin.site.register(AudioArchive)
admin.site.register(AudioFile)
admin.site.register(DetectionResult)