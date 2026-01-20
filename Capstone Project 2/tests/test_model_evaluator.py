"""
Tests for the ModelEvaluator module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from src.model_evaluator import ModelEvaluator


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
def evaluator():
    """Create a ModelEvaluator instance."""
    return ModelEvaluator()


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_evaluate_returns_metrics(self, evaluator, trained_model, sample_data):
        """Test that evaluate returns all required metrics."""
        X, y = sample_data
        metrics = evaluator.evaluate(trained_model, X, y)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
    
    def test_evaluate_metrics_validity(self, evaluator, trained_model, sample_data):
        """Test that metrics are within valid ranges."""
        X, y = sample_data
        metrics = evaluator.evaluate(trained_model, X, y)
        
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1
    
    def test_cross_validate_returns_metrics(self, evaluator, sample_data):
        """Test that cross_validate returns mean metrics."""
        X, y = sample_data
        model = Ridge(random_state=42)
        
        metrics = evaluator.cross_validate(model, X, y, cv=3)
        
        assert 'rmse_mean' in metrics
        assert 'mae_mean' in metrics
        assert 'r2_mean' in metrics
    
    def test_cross_validate_with_std(self, evaluator, sample_data):
        """Test that cross_validate returns std when requested."""
        X, y = sample_data
        model = Ridge(random_state=42)
        
        metrics = evaluator.cross_validate(model, X, y, cv=3, return_std=True)
        
        assert 'rmse_std' in metrics
        assert 'mae_std' in metrics
        assert 'r2_std' in metrics
    
    def test_cross_validate_without_std(self, evaluator, sample_data):
        """Test that cross_validate doesn't return std when not requested."""
        X, y = sample_data
        model = Ridge(random_state=42)
        
        metrics = evaluator.cross_validate(model, X, y, cv=3, return_std=False)
        
        assert 'rmse_std' not in metrics
        assert 'mae_std' not in metrics
        assert 'r2_std' not in metrics
    
    def test_compare_models(self, evaluator, sample_data):
        """Test comparing multiple models."""
        X, y = sample_data
        
        model1 = Ridge(alpha=0.1, random_state=42)
        model1.fit(X, y)
        
        model2 = Ridge(alpha=10.0, random_state=42)
        model2.fit(X, y)
        
        models = [('ridge_0.1', model1), ('ridge_10', model2)]
        results = evaluator.compare_models(models, X, y)
        
        assert 'ridge_0.1' in results
        assert 'ridge_10' in results
        assert 'rmse' in results['ridge_0.1']
