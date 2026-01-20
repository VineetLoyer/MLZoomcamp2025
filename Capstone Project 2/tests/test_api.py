"""
Integration tests for the Flask API endpoints.

Tests the /predict and /health endpoints with various inputs.

Requirements: 5.1, 5.2, 5.3
"""

import pytest
import os
import sys
import tempfile
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.model_serializer import ModelSerializer


# Sample valid input features
VALID_FEATURES = {
    'Aroma': 7.5,
    'Flavor': 7.8,
    'Aftertaste': 7.2,
    'Acidity': 7.5,
    'Body': 7.3,
    'Balance': 7.4,
    'Uniformity': 10.0,
    'Clean_Cup': 10.0,
    'Sweetness': 10.0,
    'Cupper_Points': 7.5
}

FEATURE_NAMES = list(VALID_FEATURES.keys())


@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    # Create and save a simple model
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100) * 20 + 70  # Scores between 70-90
    
    model = Ridge(random_state=42)
    model.fit(X, y)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    metadata = {
        'feature_names': FEATURE_NAMES,
        'target_column': 'Total_Cup_Points'
    }
    
    serializer = ModelSerializer()
    serializer.save_model(model, filepath, preprocessor=scaler, metadata=metadata)
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


@pytest.fixture
def app(temp_model_path, monkeypatch):
    """Create Flask test client with temporary model."""
    # Set the model path environment variable
    monkeypatch.setenv('MODEL_PATH', temp_model_path)
    
    # Import predict module after setting env var
    # Need to reset the pipeline global
    import predict
    predict.pipeline = None
    predict.MODEL_PATH = temp_model_path
    
    predict.app.config['TESTING'] = True
    return predict.app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get('/health')
        data = response.get_json()
        
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
    
    def test_health_returns_feature_names(self, client):
        """Test that health endpoint returns feature names."""
        response = client.get('/health')
        data = response.get_json()
        
        assert 'feature_names' in data
        assert data['feature_names'] is not None
        assert len(data['feature_names']) == 10


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_with_valid_input(self, client):
        """Test prediction with valid input features."""
        response = client.post(
            '/predict',
            json=VALID_FEATURES,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'prediction' in data
        assert isinstance(data['prediction'], (int, float))
    
    def test_predict_returns_valid_range(self, client):
        """Test that prediction is within valid range [0, 100]."""
        response = client.post(
            '/predict',
            json=VALID_FEATURES,
            content_type='application/json'
        )
        
        data = response.get_json()
        assert 0 <= data['prediction'] <= 100
    
    def test_predict_with_missing_features(self, client):
        """Test prediction with missing required features."""
        incomplete_features = {'Aroma': 7.5, 'Flavor': 7.8}
        
        response = client.post(
            '/predict',
            json=incomplete_features,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
        assert data['error_code'] == 'INVALID_INPUT'
    
    def test_predict_with_non_numeric_values(self, client):
        """Test prediction with non-numeric feature values."""
        invalid_features = VALID_FEATURES.copy()
        invalid_features['Aroma'] = 'not_a_number'
        
        response = client.post(
            '/predict',
            json=invalid_features,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
        assert data['error_code'] == 'INVALID_INPUT'
    
    def test_predict_with_empty_body(self, client):
        """Test prediction with empty request body."""
        response = client.post(
            '/predict',
            json={},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
    
    def test_predict_without_json_content_type(self, client):
        """Test prediction without JSON content type."""
        response = client.post(
            '/predict',
            data='not json',
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['error_code'] == 'INVALID_CONTENT_TYPE'
    
    def test_predict_with_null_values(self, client):
        """Test prediction with null feature values."""
        null_features = VALID_FEATURES.copy()
        null_features['Aroma'] = None
        
        response = client.post(
            '/predict',
            json=null_features,
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
    
    def test_predict_with_extra_features(self, client):
        """Test prediction with extra features (should still work)."""
        extra_features = VALID_FEATURES.copy()
        extra_features['Extra_Feature'] = 5.0
        
        response = client.post(
            '/predict',
            json=extra_features,
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 response for unknown endpoint."""
        response = client.get('/unknown')
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['error_code'] == 'NOT_FOUND'
    
    def test_405_for_wrong_method_on_predict(self, client):
        """Test 405 response for GET on /predict."""
        response = client.get('/predict')
        
        assert response.status_code == 405
        data = response.get_json()
        assert data['error_code'] == 'METHOD_NOT_ALLOWED'
    
    def test_405_for_wrong_method_on_health(self, client):
        """Test 405 response for POST on /health."""
        response = client.post('/health')
        
        assert response.status_code == 405
