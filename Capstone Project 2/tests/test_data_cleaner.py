"""
Unit tests for DataCleaner module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaner import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner class."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a DataCleaner instance."""
        return DataCleaner()
    
    @pytest.fixture
    def df_with_missing(self):
        """Create DataFrame with missing values."""
        return pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'string_col': ['a', 'b', None, 'd', 'e'],
            'complete_col': [10, 20, 30, 40, 50]
        })
    
    @pytest.fixture
    def df_with_outliers(self):
        """Create DataFrame with outliers."""
        return pd.DataFrame({
            'normal': [10, 11, 12, 13, 14, 15, 100],  # 100 is outlier
            'clean': [1, 2, 3, 4, 5, 6, 7]
        })
    
    def test_handle_missing_median(self, cleaner, df_with_missing):
        """Test median imputation for missing values."""
        result = cleaner.handle_missing_values(df_with_missing, strategy='median')
        
        # Numeric column should have no NaN
        assert result['numeric_col'].isna().sum() == 0
        # Median of [1, 2, 4, 5] is 3.0
        assert result['numeric_col'].iloc[2] == 3.0
    
    def test_handle_missing_mean(self, cleaner, df_with_missing):
        """Test mean imputation for missing values."""
        result = cleaner.handle_missing_values(df_with_missing, strategy='mean')
        
        assert result['numeric_col'].isna().sum() == 0
        # Mean of [1, 2, 4, 5] is 3.0
        assert result['numeric_col'].iloc[2] == 3.0
    
    def test_handle_missing_drop(self, cleaner, df_with_missing):
        """Test dropping rows with missing values."""
        result = cleaner.handle_missing_values(df_with_missing, strategy='drop')
        
        # Should have fewer rows
        assert len(result) < len(df_with_missing)
        # No missing values
        assert result.isna().sum().sum() == 0
    
    def test_handle_missing_mode(self, cleaner, df_with_missing):
        """Test mode imputation for missing values."""
        result = cleaner.handle_missing_values(df_with_missing, strategy='mode')
        
        # String column should have no NaN
        assert result['string_col'].isna().sum() == 0
    
    def test_handle_missing_invalid_strategy(self, cleaner, df_with_missing):
        """Test invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            cleaner.handle_missing_values(df_with_missing, strategy='invalid')
    
    def test_remove_outliers_zscore(self, cleaner, df_with_outliers):
        """Test outlier removal using z-score method."""
        result = cleaner.remove_outliers(
            df_with_outliers, 
            columns=['normal'], 
            threshold=2.0
        )
        
        # Should have fewer rows (outlier removed)
        assert len(result) < len(df_with_outliers)
        # 100 should be removed
        assert 100 not in result['normal'].values
    
    def test_remove_outliers_iqr(self, cleaner, df_with_outliers):
        """Test outlier removal using IQR method."""
        result = cleaner.remove_outliers(
            df_with_outliers,
            columns=['normal'],
            threshold=1.5,
            method='iqr'
        )
        
        # Should have fewer rows
        assert len(result) < len(df_with_outliers)
    
    def test_remove_outliers_invalid_method(self, cleaner, df_with_outliers):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError):
            cleaner.remove_outliers(
                df_with_outliers,
                columns=['normal'],
                method='invalid'
            )
    
    def test_remove_outliers_invalid_column(self, cleaner, df_with_outliers):
        """Test non-existent column raises ValueError."""
        with pytest.raises(ValueError):
            cleaner.remove_outliers(
                df_with_outliers,
                columns=['nonexistent']
            )
    
    def test_clean_column_names(self, cleaner):
        """Test cleaning column names."""
        df = pd.DataFrame({
            'Column.Name': [1, 2],
            'Another.Column.Name': [3, 4]
        })
        
        result = cleaner.clean_column_names(df)
        
        assert 'Column_Name' in result.columns
        assert 'Another_Column_Name' in result.columns
