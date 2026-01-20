"""
Tests for the PredictionPipeline module.
"""

import pytest
import os
import tempfile
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.prediction_pipeline import (
    PredictionPipeline,
    ModelNotLoadedError,
    InvalidInputError,
)
from src.model_serializer import ModelSerializer


# Sample feature names
FEATURE_NAMES = [
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
    'Balance', 'Uniformity', 'Clean_Cup', 'Sweetness', 'Cupper_Points'
]

# Sample valid input
VALID_INPUT = {
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


class TestPredictionPipeline:
    """Tests for PredictionPipeline class."""
    
    def test_load_model(self, temp_model_path):
        """Test loading a model from file."""
        pipeline = PredictionPipeline()
        pipeline.load(temp_model_path)
        
        assert pipeline.is_loaded is True
        assert pipeline.model is not None
    
    def test_load_model_in_constructor(self, temp_model_path):
        """Test loading model via constructor."""
        pipeline = PredictionPipeline(temp_model_path)
        
        assert pipeline.is_loaded is True
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        pipeline = PredictionPipeline()
        
        with pytest.raises(FileNotFoundError):
            pipeline.load('nonexistent_model.pkl')
    
    def test_feature_names(self, temp_model_path):
        """Test that feature names are loaded from metadata."""
        pipeline = PredictionPipeline(temp_model_path)
        
        assert pipeline.feature_names == FEATURE_NAMES
    
    def test_predict_with_valid_input(self, temp_model_path):
        """Test prediction with valid input."""
        pipeline = PredictionPipeline(temp_model_path)
        
        prediction = pipeline.predict(VALID_INPUT)
        
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 100
    
    def test_predict_without_loading(self):
        """Test that prediction without loading raises error."""
        pipeline = PredictionPipeline()
        
        with pytest.raises(ModelNotLoadedError):
            pipeline.predict(VALID_INPUT)
    
    def test_predict_with_missing_features(self, temp_model_path):
        """Test that missing features raises error."""
        pipeline = PredictionPipeline(temp_model_path)
        
        incomplete_input = {'Aroma': 7.5, 'Flavor': 7.8}
        
        with pytest.raises(InvalidInputError):
            pipeline.predict(incomplete_input)
    
    def test_predict_with_non_numeric_values(self, temp_model_path):
        """Test that non-numeric values raise error."""
        pipeline = PredictionPipeline(temp_model_path)
        
        invalid_input = VALID_INPUT.copy()
        invalid_input['Aroma'] = 'not_a_number'
        
        with pytest.raises(InvalidInputError):
            pipeline.predict(invalid_input)
    
    def test_predict_with_null_values(self, temp_model_path):
        """Test that null values raise error."""
        pipeline = PredictionPipeline(temp_model_path)
        
        null_input = VALID_INPUT.copy()
        null_input['Aroma'] = None
        
        with pytest.raises(InvalidInputError):
            pipeline.predict(null_input)
    
    def test_validate_input_returns_errors(self, temp_model_path):
        """Test input validation returns error list."""
        pipeline = PredictionPipeline(temp_model_path)
        
        errors = pipeline.validate_input({'Aroma': 7.5})
        
        assert len(errors) > 0
        assert 'Missing required features' in errors[0]
    
    def test_validate_input_valid_returns_empty(self, temp_model_path):
        """Test valid input returns empty error list."""
        pipeline = PredictionPipeline(temp_model_path)
        
        errors = pipeline.validate_input(VALID_INPUT)
        
        assert len(errors) == 0
    
    def test_predict_batch(self, temp_model_path):
        """Test batch prediction."""
        pipeline = PredictionPipeline(temp_model_path)
        
        inputs = [VALID_INPUT, VALID_INPUT]
        predictions = pipeline.predict_batch(inputs)
        
        assert len(predictions) == 2
        assert all(isinstance(p, float) for p in predictions)
    
    def test_preprocess_returns_array(self, temp_model_path):
        """Test that preprocess returns numpy array."""
        pipeline = PredictionPipeline(temp_model_path)
        
        X = pipeline.preprocess(VALID_INPUT)
        
        assert isinstance(X, np.ndarray)
        assert X.shape == (1, 10)
