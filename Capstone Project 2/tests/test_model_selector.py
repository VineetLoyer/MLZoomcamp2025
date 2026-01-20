"""
Tests for the ModelSelector module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from src.model_selector import ModelSelector


@pytest.fixture
def sample_data():
    """Generate sample regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def trained_models(sample_data):
    """Create trained models for testing."""
    X, y = sample_data
    
    model1 = Ridge(alpha=0.1, random_state=42)
    model1.fit(X, y)
    
    model2 = Ridge(alpha=10.0, random_state=42)
    model2.fit(X, y)
    
    model3 = Lasso(alpha=0.1, random_state=42)
    model3.fit(X, y)
    
    return [('ridge_0.1', model1), ('ridge_10', model2), ('lasso_0.1', model3)]


@pytest.fixture
def selector():
    """Create a ModelSelector instance."""
    return ModelSelector()


class TestModelSelector:
    """Tests for ModelSelector class."""
    
    def test_select_best_model_rmse(self, selector, trained_models, sample_data):
        """Test selecting best model by RMSE."""
        X, y = sample_data
        name, model = selector.select_best_model(trained_models, X, y, metric='rmse')
        
        assert name is not None
        assert model is not None
        assert name in ['ridge_0.1', 'ridge_10', 'lasso_0.1']
    
    def test_select_best_model_r2(self, selector, trained_models, sample_data):
        """Test selecting best model by R²."""
        X, y = sample_data
        name, model = selector.select_best_model(trained_models, X, y, metric='r2')
        
        assert name is not None
        assert model is not None
    
    def test_select_best_model_invalid_metric(self, selector, trained_models, sample_data):
        """Test that invalid metric raises ValueError."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Invalid metric"):
            selector.select_best_model(trained_models, X, y, metric='invalid')
    
    def test_select_best_model_empty_list(self, selector, sample_data):
        """Test that empty models list raises ValueError."""
        X, y = sample_data
        with pytest.raises(ValueError, match="cannot be empty"):
            selector.select_best_model([], X, y)
    
    def test_get_selection_history(self, selector, trained_models, sample_data):
        """Test that selection history is recorded."""
        X, y = sample_data
        selector.select_best_model(trained_models, X, y, metric='rmse')
        
        history = selector.get_selection_history()
        assert len(history) == 1
        assert 'metric' in history[0]
        assert 'best_model' in history[0]
    
    def test_get_model_rankings(self, selector, trained_models, sample_data):
        """Test getting model rankings."""
        X, y = sample_data
        rankings = selector.get_model_rankings(trained_models, X, y, metric='rmse')
        
        assert len(rankings) == 3
        assert rankings[0]['rank'] == 1
        assert rankings[1]['rank'] == 2
        assert rankings[2]['rank'] == 3
