"""Data Cleaner module for Coffee Quality Prediction."""

import pandas as pd
import numpy as np
from typing import List, Optional


class DataCleaner:
    """Cleans and preprocesses coffee quality data."""
    
    VALID_STRATEGIES = ['median', 'mean', 'drop', 'mode']
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle missing values using specified strategy (median, mean, mode, drop)."""
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Valid strategies are: {self.VALID_STRATEGIES}"
            )
        
        result = df.copy()
        
        if columns is None:
            columns = result.columns.tolist()
        
        if strategy == 'drop':
            result = result.dropna(subset=columns)
        elif strategy == 'median':
            for col in columns:
                if col in result.columns and result[col].dtype in ['float64', 'int64']:
                    result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mean':
            for col in columns:
                if col in result.columns and result[col].dtype in ['float64', 'int64']:
                    result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'mode':
            for col in columns:
                if col in result.columns and not result[col].mode().empty:
                    result[col] = result[col].fillna(result[col].mode().iloc[0])
        
        return result
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """Remove outliers using z-score or IQR method."""
        if method not in ['zscore', 'iqr']:
            raise ValueError(f"Invalid method '{method}'. Use 'zscore' or 'iqr'.")
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if not np.issubdtype(result[col].dtype, np.number):
                raise ValueError(f"Column '{col}' is not numeric.")
        
        if method == 'zscore':
            mask = pd.Series([True] * len(result), index=result.index)
            for col in columns:
                col_data = result[col]
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    mask = mask & (z_scores <= threshold)
            result = result[mask]
        
        elif method == 'iqr':
            mask = pd.Series([True] * len(result), index=result.index)
            for col in columns:
                col_data = result[col]
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                mask = mask & (col_data >= lower_bound) & (col_data <= upper_bound)
            result = result[mask]
        
        return result
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by replacing dots with underscores."""
        result = df.copy()
        result.columns = [col.replace('.', '_') for col in result.columns]
        return result
