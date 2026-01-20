"""Model Evaluator module for Coffee Quality Prediction."""

import numpy as np
from typing import Dict, List
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    """Evaluates machine learning models using regression metrics."""
    
    def evaluate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance, returning RMSE, MAE, and R²."""
        y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        return_std: bool = True
    ) -> Dict[str, float]:
        """Perform cross-validation and return mean metrics."""
        rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        results = {
            'rmse_mean': float(np.mean(rmse_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'r2_mean': float(np.mean(r2_scores))
        }
        
        if return_std:
            results['rmse_std'] = float(np.std(rmse_scores))
            results['mae_std'] = float(np.std(mae_scores))
            results['r2_std'] = float(np.std(r2_scores))
        
        return results
    
    def compare_models(
        self,
        models: List[tuple],
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'rmse'
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same dataset."""
        results = {}
        for name, model in models:
            metrics = self.evaluate(model, X, y)
            results[name] = metrics
        return results
