"""
Tests for the ModelTrainer module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Generate sample regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def trainer():
    """Create a ModelTrainer instance."""
    return ModelTrainer(random_state=42)


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_train_linear_model_ridge(self, trainer, sample_data):
        """Test training Ridge regression model."""
        X, y = sample_data
        model = trainer.train_linear_model(X, y, model_type='ridge')
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_linear_model_lasso(self, trainer, sample_data):
        """Test training Lasso regression model."""
        X, y = sample_data
        model = trainer.train_linear_model(X, y, model_type='lasso')
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_linear_model_linear(self, trainer, sample_data):
        """Test training LinearRegression model."""
        X, y = sample_data
        model = trainer.train_linear_model(X, y, model_type='linear')
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_linear_model_invalid_type(self, trainer, sample_data):
        """Test that invalid model type raises ValueError."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Invalid model_type"):
            trainer.train_linear_model(X, y, model_type='invalid')
    
    def test_train_tree_model_randomforest(self, trainer, sample_data):
        """Test training RandomForest model."""
        X, y = sample_data
        model = trainer.train_tree_model(X, y, model_type='randomforest', n_estimators=10)
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_tree_model_xgboost(self, trainer, sample_data):
        """Test training XGBoost model."""
        X, y = sample_data
        model = trainer.train_tree_model(X, y, model_type='xgboost', n_estimators=10)
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_tree_model_lightgbm(self, trainer, sample_data):
        """Test training LightGBM model."""
        X, y = sample_data
        model = trainer.train_tree_model(X, y, model_type='lightgbm', n_estimators=10)
        
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_tree_model_invalid_type(self, trainer, sample_data):
        """Test that invalid model type raises ValueError."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Invalid model_type"):
            trainer.train_tree_model(X, y, model_type='invalid')
    
    def test_tune_hyperparameters(self, trainer, sample_data):
        """Test hyperparameter tuning with GridSearchCV."""
        X, y = sample_data
        from sklearn.linear_model import Ridge
        
        model = Ridge()
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        
        best_model = trainer.tune_hyperparameters(model, X, y, param_grid, cv=3)
        
        assert best_model is not None
        assert hasattr(best_model, 'alpha')
        predictions = best_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_get_default_param_grid(self, trainer):
        """Test getting default parameter grids."""
        ridge_grid = trainer.get_default_param_grid('ridge')
        assert 'alpha' in ridge_grid
        
        xgb_grid = trainer.get_default_param_grid('xgboost')
        assert 'n_estimators' in xgb_grid
        assert 'max_depth' in xgb_grid
        
        unknown_grid = trainer.get_default_param_grid('unknown')
        assert unknown_grid == {}
