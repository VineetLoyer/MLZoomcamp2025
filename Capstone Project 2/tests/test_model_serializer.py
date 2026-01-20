"""
Tests for the ModelSerializer module.
"""

import pytest
import os
import tempfile
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.model_serializer import ModelSerializer, save_model, load_model


@pytest.fixture
def sample_data():
    """Generate sample regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data
    model = Ridge(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def serializer():
    """Create a ModelSerializer instance."""
    return ModelSerializer()


@pytest.fixture
def temp_filepath():
    """Create a temporary file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


class TestModelSerializer:
    """Tests for ModelSerializer class."""
    
    def test_save_and_load_model(self, serializer, trained_model, temp_filepath):
        """Test saving and loading a model."""
        serializer.save_model(trained_model, temp_filepath)
        
        loaded_model, preprocessor, metadata = serializer.load_model(temp_filepath)
        
        assert loaded_model is not None
        assert preprocessor is None
        assert metadata == {}
    
    def test_save_and_load_with_preprocessor(self, serializer, trained_model, temp_filepath, sample_data):
        """Test saving and loading model with preprocessor."""
        X, y = sample_data
        scaler = StandardScaler()
        scaler.fit(X)
        
        serializer.save_model(trained_model, temp_filepath, preprocessor=scaler)
        
        loaded_model, loaded_preprocessor, _ = serializer.load_model(temp_filepath)
        
        assert loaded_model is not None
        assert loaded_preprocessor is not None
    
    def test_save_and_load_with_metadata(self, serializer, trained_model, temp_filepath):
        """Test saving and loading model with metadata."""
        metadata = {'rmse': 0.5, 'features': ['a', 'b', 'c']}
        
        serializer.save_model(trained_model, temp_filepath, metadata=metadata)
        
        _, _, loaded_metadata = serializer.load_model(temp_filepath)
        
        assert loaded_metadata == metadata
    
    def test_load_nonexistent_file(self, serializer):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            serializer.load_model('nonexistent_file.pkl')
    
    def test_load_model_only(self, serializer, trained_model, temp_filepath):
        """Test loading only the model."""
        serializer.save_model(trained_model, temp_filepath)
        
        loaded_model = serializer.load_model_only(temp_filepath)
        
        assert loaded_model is not None
    
    def test_predictions_match_after_roundtrip(self, serializer, trained_model, temp_filepath, sample_data):
        """Test that predictions match after save/load roundtrip."""
        X, y = sample_data
        original_predictions = trained_model.predict(X)
        
        serializer.save_model(trained_model, temp_filepath)
        loaded_model, _, _ = serializer.load_model(temp_filepath)
        
        loaded_predictions = loaded_model.predict(X)
        
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_save_and_load_functions(self, trained_model, temp_filepath, sample_data):
        """Test save_model and load_model convenience functions."""
        X, y = sample_data
        
        save_model(trained_model, temp_filepath)
        loaded_model, _, _ = load_model(temp_filepath)
        
        original_predictions = trained_model.predict(X)
        loaded_predictions = loaded_model.predict(X)
        
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
