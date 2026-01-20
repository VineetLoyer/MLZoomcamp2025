"""Model Trainer module for Coffee Quality Prediction."""

import numpy as np
from typing import Dict, Any, List
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    """Trains machine learning models for regression tasks."""
    
    VALID_LINEAR_MODELS = ['linear', 'ridge', 'lasso']
    VALID_TREE_MODELS = ['randomforest', 'xgboost', 'lightgbm']
    
    def __init__(self, random_state: int = 42):
        """Initialize the ModelTrainer."""
        self.random_state = random_state
    
    def train_linear_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'ridge',
        **kwargs
    ) -> BaseEstimator:
        """Train a linear regression model (linear, ridge, or lasso)."""
        model_type = model_type.lower()
        if model_type not in self.VALID_LINEAR_MODELS:
            raise ValueError(f"Invalid model_type '{model_type}'. Valid types are: {self.VALID_LINEAR_MODELS}")
        
        if model_type == 'linear':
            model = LinearRegression(**kwargs)
        elif model_type == 'ridge':
            model = Ridge(random_state=self.random_state, **kwargs)
        else:
            model = Lasso(random_state=self.random_state, **kwargs)
        
        model.fit(X_train, y_train)
        return model

    def train_tree_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'xgboost',
        **kwargs
    ) -> BaseEstimator:
        """Train a tree-based regression model (randomforest, xgboost, or lightgbm)."""
        model_type = model_type.lower()
        if model_type not in self.VALID_TREE_MODELS:
            raise ValueError(f"Invalid model_type '{model_type}'. Valid types are: {self.VALID_TREE_MODELS}")
        
        if model_type == 'randomforest':
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **kwargs)
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1, **kwargs)
        else:
            model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1, **kwargs)
        
        model.fit(X_train, y_train)
        return model
    
    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error',
        n_jobs: int = -1
    ) -> BaseEstimator:
        """Tune hyperparameters using GridSearchCV."""
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def get_default_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """Get default hyperparameter grid for a given model type."""
        model_type = model_type.lower()
        
        param_grids = {
            'ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'randomforest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        return param_grids.get(model_type, {})
