"""Feature Scaler module for Coffee Quality Prediction."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureScaler:
    """Scales numerical features using StandardScaler or MinMaxScaler."""
    
    VALID_SCALER_TYPES = ['standard', 'minmax']
    
    def __init__(self, scaler_type: str = 'standard'):
        """Initialize the FeatureScaler with specified scaler type."""
        if scaler_type not in self.VALID_SCALER_TYPES:
            raise ValueError(f"Invalid scaler_type '{scaler_type}'. Valid types are: {self.VALID_SCALER_TYPES}")
        
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self._fitted = False
        self._feature_names: List[str] = []
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> 'FeatureScaler':
        """Fit the scaler on the training data."""
        if isinstance(X, pd.DataFrame):
            if columns is None:
                columns = X.select_dtypes(include=[np.number]).columns.tolist()
            self._feature_names = columns
            data = X[columns].values
        else:
            data = X
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.scaler.fit(data)
        self._fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using the fitted scaler."""
        if not self._fitted:
            raise ValueError("FeatureScaler has not been fitted. Call fit() or fit_transform() first.")
        
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            data = X[self._feature_names].values
            scaled_data = self.scaler.transform(data)
            result[self._feature_names] = scaled_data
            return result
        else:
            return self.scaler.transform(X)
    
    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit the scaler and transform the data in one step."""
        self.fit(X, columns)
        return self.transform(X)
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform scaled data back to original scale."""
        if not self._fitted:
            raise ValueError("FeatureScaler has not been fitted. Call fit() or fit_transform() first.")
        
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            data = X[self._feature_names].values
            original_data = self.scaler.inverse_transform(data)
            result[self._feature_names] = original_data
            return result
        else:
            return self.scaler.inverse_transform(X)
    
    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names that were scaled."""
        return self._feature_names.copy()
    
    @property
    def mean_(self) -> Optional[np.ndarray]:
        """Return mean values (only for StandardScaler)."""
        if self.scaler_type == 'standard' and self._fitted:
            return self.scaler.mean_
        return None
    
    @property
    def scale_(self) -> Optional[np.ndarray]:
        """Return scale values."""
        if self._fitted:
            return self.scaler.scale_
        return None
