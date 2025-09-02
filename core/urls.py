from rest_framework import routers
from django.urls import path, include
from .views import (
    upload_and_process_audio,
    get_processing_results,
    archives_feed,
    user_archives,
    delete_archive
)
from . import auth_views



urlpatterns = [    
    path('upload-and-process/', upload_and_process_audio, name='upload-and-process'),
    path('results/<int:archive_id>/', get_processing_results, name='get-results'),
    path('archives/feed/', archives_feed, name='archives-feed'),
    path('archives/user/', user_archives, name='user-archives'),
    path('archives/<int:archive_id>/delete/', delete_archive, name='delete-archive'),
    path('auth/register/', auth_views.register, name='register'),
    path('auth/login/', auth_views.login, name='login'),
    path('auth/profile/', auth_views.profile, name='profile'),
]