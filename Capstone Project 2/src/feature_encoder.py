"""Feature Encoder module for Coffee Quality Prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:
    """Encodes categorical features to numerical representations."""
    
    def __init__(self):
        """Initialize the FeatureEncoder."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False
        self._categorical_cols: List[str] = []
    
    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None
    ) -> 'FeatureEncoder':
        """Fit the encoder on the training data."""
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self._categorical_cols = categorical_cols
        self.label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                values = df[col].astype(str).values
                encoder.fit(values)
                self.label_encoders[col] = encoder
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders."""
        if not self._fitted:
            raise ValueError("FeatureEncoder has not been fitted. Call fit() or fit_transform() first.")
        
        result = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in result.columns:
                values = result[col].astype(str).values
                encoded = np.zeros(len(values), dtype=int)
                
                for i, val in enumerate(values):
                    if val in encoder.classes_:
                        encoded[i] = encoder.transform([val])[0]
                    else:
                        encoded[i] = -1
                
                result[col] = encoded
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fit the encoder and transform the data in one step."""
        self.fit(df, categorical_cols)
        return self.transform(df)
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse transform encoded columns back to original values."""
        if not self._fitted:
            raise ValueError("FeatureEncoder has not been fitted. Call fit() or fit_transform() first.")
        
        result = df.copy()
        
        if columns is None:
            columns = list(self.label_encoders.keys())
        
        for col in columns:
            if col in self.label_encoders and col in result.columns:
                encoder = self.label_encoders[col]
                values = result[col].values
                decoded = []
                for val in values:
                    if val >= 0 and val < len(encoder.classes_):
                        decoded.append(encoder.inverse_transform([val])[0])
                    else:
                        decoded.append('unknown')
                result[col] = decoded
        
        return result
    
    def get_encoding_map(self, column: str) -> Dict[str, int]:
        """Get the encoding mapping for a specific column."""
        if column not in self.label_encoders:
            raise ValueError(f"Column '{column}' was not encoded.")
        
        encoder = self.label_encoders[column]
        return {cls: i for i, cls in enumerate(encoder.classes_)}
    
    @property
    def categorical_columns(self) -> List[str]:
        """Return list of categorical columns that were encoded."""
        return self._categorical_cols.copy()
