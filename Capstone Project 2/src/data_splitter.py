"""Data Splitter module for Coffee Quality Prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Splits data into train, validation, and test sets."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the DataSplitter with random state for reproducibility."""
        self.random_state = random_state
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        stratify: bool = False
    ) -> Tuple:
        """Split data into train, validation, and test sets."""
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total:.2f}")
        
        if any(s <= 0 for s in [train_size, val_size, test_size]):
            raise ValueError("All split sizes must be positive.")
        
        stratify_y = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        val_relative = val_size / (train_size + val_size)
        stratify_y_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=self.random_state,
            stratify=stratify_y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train, validation, and test sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(
            X, y, train_size, val_size, test_size, stratify
        )
        
        train_df = X_train.copy()
        train_df[target_column] = y_train
        
        val_df = X_val.copy()
        val_df[target_column] = y_val
        
        test_df = X_test.copy()
        test_df[target_column] = y_test
        
        return train_df, val_df, test_df
    
    def get_split_sizes(
        self,
        n_samples: int,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2
    ) -> Tuple[int, int, int]:
        """Calculate the number of samples in each split."""
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        return n_train, n_val, n_test
