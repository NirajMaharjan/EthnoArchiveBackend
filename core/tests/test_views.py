import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth.models import User
from core.models import AudioArchive
import os

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def test_user():
    user = User.objects.create_user(username='testuser', password='testpassword')
    return user

@pytest.fixture
def authenticated_client(api_client, test_user):
    api_client.force_authenticate(user=test_user)
    return api_client

@pytest.mark.django_db
class TestAuthViews:
    def test_register(self, api_client):
        url = reverse('register')
        data = {'username': 'newuser', 'email': 'newuser@example.com', 'password': 'newpassword'}
        response = api_client.post(url, data)
        assert response.status_code == status.HTTP_201_CREATED

    def test_login(self, api_client, test_user):
        url = reverse('login')
        data = {'username': 'testuser', 'password': 'testpassword'}
        response = api_client.post(url, data)
        assert response.status_code == status.HTTP_200_OK
        assert 'token' in response.data

@pytest.mark.django_db
class TestAudioArchiveViews:
    def test_upload_and_process_audio(self, authenticated_client):
        url = reverse('upload-and-process')
        # Create a dummy audio file for testing
        with open('test.mp3', 'w') as f:
            f.write('dummy content')
        
        with open('test.mp3', 'rb') as f:
            data = {'file': f}
            response = authenticated_client.post(url, data, format='multipart')
        
        os.remove('test.mp3')
        
        assert response.status_code == status.HTTP_200_OK
        assert response.data['success'] is True
        assert 'archive_id' in response.data

    def test_get_processing_results(self, api_client, authenticated_client):
        # First, upload a file to create an archive
        url_upload = reverse('upload-and-process')
        with open('test.mp3', 'w') as f:
            f.write('dummy content')
        
        with open('test.mp3', 'rb') as f:
            data = {'file': f}
            response_upload = authenticated_client.post(url_upload, data, format='multipart')
        
        os.remove('test.mp3')
        
        archive_id = response_upload.data['archive_id']
        
        # Now, get the results
        url_results = reverse('get-results', kwargs={'archive_id': archive_id})
        response_results = api_client.get(url_results)
        
        assert response_results.status_code == status.HTTP_200_OK
        assert response_results.data['success'] is True
        assert response_results.data['archive_id'] == archive_id

    def test_archives_feed(self, api_client):
        url = reverse('archives-feed')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['success'] is True

    def test_user_archives(self, authenticated_client):
        url = reverse('user-archives')
        response = authenticated_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['success'] is True

    def test_delete_archive(self, authenticated_client):
        # First, upload a file to create an archive
        url_upload = reverse('upload-and-process')
        with open('test.mp3', 'w') as f:
            f.write('dummy content')
        
        with open('test.mp3', 'rb') as f:
            data = {'file': f}
            response_upload = authenticated_client.post(url_upload, data, format='multipart')
        
        os.remove('test.mp3')
        
        archive_id = response_upload.data['archive_id']
        
        # Now, delete the archive
        url_delete = reverse('delete-archive', kwargs={'archive_id': archive_id})
        response_delete = authenticated_client.delete(url_delete)
        
        assert response_delete.status_code == status.HTTP_200_OK
        assert response_delete.data['success'] is True
