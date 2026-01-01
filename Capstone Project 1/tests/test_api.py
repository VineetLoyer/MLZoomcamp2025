"""
Integration tests for the MBTI Prediction Flask API.

Tests the /predict and /health endpoints with valid and invalid inputs.

Requirements: 6.2, 6.4, 6.5
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predict import app, model_loaded
from src.utils import VALID_MBTI_TYPES


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def skip_if_models_not_loaded():
    """Skip test if models are not loaded."""
    if not model_loaded:
        pytest.skip("Models not loaded - skipping integration test")


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200_when_models_loaded(self, client, skip_if_models_not_loaded):
        """Health endpoint should return 200 when models are loaded."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
    
    def test_health_returns_json(self, client, skip_if_models_not_loaded):
        """Health endpoint should return JSON response."""
        response = client.get('/health')
        assert response.content_type == 'application/json'


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_valid_text(self, client, skip_if_models_not_loaded):
        """Predict endpoint should return valid MBTI type for valid input."""
        response = client.post(
            '/predict',
            json={'text': 'I enjoy spending time alone reading books and thinking about abstract concepts.'},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        
        # Check response structure
        assert 'mbti_type' in data
        assert 'confidence' in data
        
        # Check MBTI type is valid
        assert data['mbti_type'] in VALID_MBTI_TYPES
    
    def test_predict_returns_confidence_scores(self, client, skip_if_models_not_loaded):
        """Predict endpoint should return confidence scores for all dimensions."""
        response = client.post(
            '/predict',
            json={'text': 'I love meeting new people and going to parties with friends.'},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        
        # Check confidence structure
        confidence = data['confidence']
        assert 'IE' in confidence
        assert 'NS' in confidence
        assert 'TF' in confidence
        assert 'JP' in confidence
        
        # Check each dimension has both labels
        assert 'I' in confidence['IE'] and 'E' in confidence['IE']
        assert 'N' in confidence['NS'] and 'S' in confidence['NS']
        assert 'T' in confidence['TF'] and 'F' in confidence['TF']
        assert 'J' in confidence['JP'] and 'P' in confidence['JP']
    
    def test_predict_confidence_sums_to_one(self, client, skip_if_models_not_loaded):
        """Confidence scores for each dimension should sum to 1.0."""
        response = client.post(
            '/predict',
            json={'text': 'I prefer logical analysis over emotional decisions in my work.'},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        
        confidence = data['confidence']
        
        # Check each dimension sums to 1.0 (within floating point tolerance)
        assert abs(confidence['IE']['I'] + confidence['IE']['E'] - 1.0) < 0.001
        assert abs(confidence['NS']['N'] + confidence['NS']['S'] - 1.0) < 0.001
        assert abs(confidence['TF']['T'] + confidence['TF']['F'] - 1.0) < 0.001
        assert abs(confidence['JP']['J'] + confidence['JP']['P'] - 1.0) < 0.001
    
    def test_predict_confidence_values_in_range(self, client, skip_if_models_not_loaded):
        """All confidence values should be between 0.0 and 1.0."""
        response = client.post(
            '/predict',
            json={'text': 'I like to plan everything in advance and stick to schedules.'},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        
        confidence = data['confidence']
        
        for dim in ['IE', 'NS', 'TF', 'JP']:
            for label, value in confidence[dim].items():
                assert 0.0 <= value <= 1.0, f"Confidence {dim}[{label}]={value} out of range"


class TestPredictErrorHandling:
    """Tests for error handling in /predict endpoint."""
    
    def test_predict_missing_text_field(self, client, skip_if_models_not_loaded):
        """Should return 400 when text field is missing."""
        response = client.post(
            '/predict',
            json={'content': 'some text'},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Missing required field: text' in data['error']
    
    def test_predict_empty_text(self, client, skip_if_models_not_loaded):
        """Should return 400 when text is empty."""
        response = client.post(
            '/predict',
            json={'text': ''},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'empty' in data['error'].lower()
    
    def test_predict_whitespace_only_text(self, client, skip_if_models_not_loaded):
        """Should return 400 when text is only whitespace."""
        response = client.post(
            '/predict',
            json={'text': '   \n\t   '},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_text_too_short(self, client, skip_if_models_not_loaded):
        """Should return 400 when text is too short."""
        response = client.post(
            '/predict',
            json={'text': 'short'},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'too short' in data['error'].lower()
    
    def test_predict_invalid_json(self, client, skip_if_models_not_loaded):
        """Should return 400 for invalid JSON."""
        response = client.post(
            '/predict',
            data='not valid json',
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_predict_non_string_text(self, client, skip_if_models_not_loaded):
        """Should return 400 when text is not a string."""
        response = client.post(
            '/predict',
            json={'text': 12345},
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data


class TestPredictLongText:
    """Tests for handling longer text inputs."""
    
    def test_predict_long_text(self, client, skip_if_models_not_loaded):
        """Should handle longer text inputs successfully."""
        long_text = """
        I really enjoy spending my weekends reading books about philosophy and science.
        I find that I need a lot of alone time to recharge after social events.
        When making decisions, I tend to rely on logic and analysis rather than feelings.
        I prefer to have a detailed plan before starting any project.
        I often think about abstract concepts and theoretical possibilities.
        """ * 3  # Repeat to make it longer
        
        response = client.post(
            '/predict',
            json={'text': long_text},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['mbti_type'] in VALID_MBTI_TYPES
