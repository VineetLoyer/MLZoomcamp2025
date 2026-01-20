"""Prediction Pipeline module for Coffee Quality Prediction."""

import os
import numpy as np
from typing import Any, Dict, List, Optional

from .model_serializer import ModelSerializer


class ModelNotLoadedError(Exception):
    """Raised when prediction is attempted before model is loaded."""
    pass


class InvalidInputError(Exception):
    """Raised when input features fail validation."""
    pass


class PredictionPipeline:
    """Pipeline for loading trained models and making predictions."""
    
    DEFAULT_FEATURES = [
        'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
        'Balance', 'Uniformity', 'Clean_Cup', 'Sweetness', 'Cupper_Points'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the PredictionPipeline, optionally loading a model."""
        self.model = None
        self.preprocessor = None
        self.metadata: Dict[str, Any] = {}
        self._loaded = False
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        """Load model and preprocessor from pickle file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        serializer = ModelSerializer()
        self.model, self.preprocessor, self.metadata = serializer.load_model(model_path)
        self._loaded = True
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    @property
    def feature_names(self) -> List[str]:
        """Get the list of feature names expected by the model."""
        if self.metadata and 'feature_names' in self.metadata:
            return self.metadata['feature_names']
        return self.DEFAULT_FEATURES
    
    def validate_input(self, features: Dict[str, Any]) -> List[str]:
        """Validate input features, returning list of error messages."""
        errors = []
        expected_features = self.feature_names
        
        missing = [f for f in expected_features if f not in features]
        if missing:
            errors.append(f"Missing required features: {missing}")
        
        for name, value in features.items():
            if name in expected_features:
                if value is None:
                    errors.append(f"Feature '{name}' cannot be None")
                elif not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"Feature '{name}' must be numeric, got: {type(value).__name__}")
        
        return errors
    
    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess input features for prediction."""
        errors = self.validate_input(features)
        if errors:
            raise InvalidInputError("; ".join(errors))
        
        expected_features = self.feature_names
        feature_values = [float(features.get(name)) for name in expected_features]
        
        X = np.array([feature_values])
        
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        
        return X
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Preprocess features and return prediction."""
        if not self._loaded:
            raise ModelNotLoadedError("Model has not been loaded. Call load() first.")
        
        X = self.preprocess(features)
        prediction = self.model.predict(X)[0]
        prediction = max(0.0, min(100.0, float(prediction)))
        
        return prediction
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Make predictions for multiple samples."""
        if not self._loaded:
            raise ModelNotLoadedError("Model has not been loaded. Call load() first.")
        
        return [self.predict(features) for features in features_list]
