"""Model Serializer module for Coffee Quality Prediction."""

import pickle
import os
from typing import Any, Dict, Optional, Tuple
from sklearn.base import BaseEstimator


class ModelSerializer:
    """Serializes and deserializes trained models with preprocessors."""
    
    def save_model(
        self,
        model: BaseEstimator,
        filepath: str,
        preprocessor: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model to pickle file with optional preprocessor and metadata."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        package = {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata or {}
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(package, f)
        except Exception as e:
            raise IOError(f"Failed to save model to {filepath}: {str(e)}")
    
    def load_model(self, filepath: str) -> Tuple[BaseEstimator, Optional[Any], Dict[str, Any]]:
        """Load model from pickle file, returning (model, preprocessor, metadata)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                package = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {str(e)}")
        
        if isinstance(package, dict) and 'model' in package:
            model = package['model']
            preprocessor = package.get('preprocessor')
            metadata = package.get('metadata', {})
        else:
            model = package
            preprocessor = None
            metadata = {}
        
        return model, preprocessor, metadata
    
    def load_model_only(self, filepath: str) -> BaseEstimator:
        """Load only the model from pickle file."""
        model, _, _ = self.load_model(filepath)
        return model


def save_model(
    model: BaseEstimator,
    filepath: str,
    preprocessor: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to save model to pickle file."""
    serializer = ModelSerializer()
    serializer.save_model(model, filepath, preprocessor, metadata)


def load_model(filepath: str) -> Tuple[BaseEstimator, Optional[Any], Dict[str, Any]]:
    """Convenience function to load model from pickle file."""
    serializer = ModelSerializer()
    return serializer.load_model(filepath)
