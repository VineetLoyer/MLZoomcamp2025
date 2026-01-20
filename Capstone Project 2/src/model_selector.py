"""Model Selector module for Coffee Quality Prediction."""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator
from .model_evaluator import ModelEvaluator


class ModelSelector:
    """Selects the best model from a set of trained models."""
    
    VALID_METRICS = ['rmse', 'mae', 'r2']
    MINIMIZE_METRICS = ['rmse', 'mae']
    
    def __init__(self):
        """Initialize the ModelSelector."""
        self.evaluator = ModelEvaluator()
        self._selection_history: List[Dict] = []
    
    def select_best_model(
        self,
        models: List[Tuple[str, BaseEstimator]],
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'rmse'
    ) -> Tuple[str, BaseEstimator]:
        """Select the best model based on validation performance."""
        metric = metric.lower()
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric '{metric}'. Valid metrics are: {self.VALID_METRICS}")
        
        if not models:
            raise ValueError("Models list cannot be empty.")
        
        results = []
        for name, model in models:
            metrics = self.evaluator.evaluate(model, X_val, y_val)
            results.append({'name': name, 'model': model, 'metrics': metrics})
        
        if metric in self.MINIMIZE_METRICS:
            results.sort(key=lambda x: x['metrics'][metric])
        else:
            results.sort(key=lambda x: x['metrics'][metric], reverse=True)
        
        best = results[0]
        
        self._selection_history.append({
            'metric': metric,
            'best_model': best['name'],
            'best_score': best['metrics'][metric],
            'all_results': [{'name': r['name'], 'score': r['metrics'][metric]} for r in results]
        })
        
        return best['name'], best['model']
    
    def get_selection_history(self) -> List[Dict]:
        """Get history of model selections."""
        return self._selection_history.copy()
    
    def get_model_rankings(
        self,
        models: List[Tuple[str, BaseEstimator]],
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'rmse'
    ) -> List[Dict]:
        """Get ranked list of models by performance."""
        metric = metric.lower()
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric '{metric}'. Valid metrics are: {self.VALID_METRICS}")
        
        results = []
        for name, model in models:
            metrics = self.evaluator.evaluate(model, X_val, y_val)
            results.append({'name': name, 'rmse': metrics['rmse'], 'mae': metrics['mae'], 'r2': metrics['r2']})
        
        if metric in self.MINIMIZE_METRICS:
            results.sort(key=lambda x: x[metric])
        else:
            results.sort(key=lambda x: x[metric], reverse=True)
        
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
