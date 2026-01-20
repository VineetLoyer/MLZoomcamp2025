"""
Unit tests for FeatureEncoder module.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_encoder import FeatureEncoder


class TestFeatureEncoder:
    """Tests for FeatureEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create a FeatureEncoder instance."""
        return FeatureEncoder()
    
    @pytest.fixture
    def df_categorical(self):
        """Create DataFrame with categorical columns."""
        return pd.DataFrame({
            'country': ['Brazil', 'Colombia', 'Ethiopia', 'Brazil'],
            'method': ['washed', 'natural', 'washed', 'honey'],
            'score': [85.0, 87.5, 90.0, 82.0]
        })
    
    def test_fit_transform(self, encoder, df_categorical):
        """Test fit_transform encodes categorical columns."""
        result = encoder.fit_transform(df_categorical, ['country', 'method'])
        
        # Encoded columns should be numeric
        assert result['country'].dtype in [np.int64, np.int32, int]
        assert result['method'].dtype in [np.int64, np.int32, int]
        
        # Non-categorical column should be unchanged
        assert result['score'].dtype == np.float64
    
    def test_transform_after_fit(self, encoder, df_categorical):
        """Test transform works after fitting."""
        encoder.fit(df_categorical, ['country', 'method'])
        result = encoder.transform(df_categorical)
        
        assert result['country'].dtype in [np.int64, np.int32, int]
        assert result['method'].dtype in [np.int64, np.int32, int]
    
    def test_transform_without_fit_raises(self, encoder, df_categorical):
        """Test transform without fit raises ValueError."""
        with pytest.raises(ValueError):
            encoder.transform(df_categorical)
    
    def test_auto_detect_categorical(self, encoder, df_categorical):
        """Test automatic detection of categorical columns."""
        result = encoder.fit_transform(df_categorical)
        
        # Should detect 'country' and 'method' as categorical
        assert 'country' in encoder.categorical_columns
        assert 'method' in encoder.categorical_columns
        assert 'score' not in encoder.categorical_columns
    
    def test_inverse_transform(self, encoder, df_categorical):
        """Test inverse transform recovers original values."""
        encoded = encoder.fit_transform(df_categorical, ['country', 'method'])
        decoded = encoder.inverse_transform(encoded)
        
        # Should recover original values
        assert list(decoded['country']) == list(df_categorical['country'])
        assert list(decoded['method']) == list(df_categorical['method'])
    
    def test_unseen_category_handling(self, encoder, df_categorical):
        """Test handling of unseen categories during transform."""
        encoder.fit(df_categorical, ['country'])
        
        new_df = pd.DataFrame({
            'country': ['Kenya'],  # Unseen category
            'score': [88.0]
        })
        
        result = encoder.transform(new_df)
        
        # Unseen category should be encoded as -1
        assert result['country'].iloc[0] == -1
    
    def test_get_encoding_map(self, encoder, df_categorical):
        """Test getting encoding map for a column."""
        encoder.fit_transform(df_categorical, ['country'])
        
        encoding_map = encoder.get_encoding_map('country')
        
        assert isinstance(encoding_map, dict)
        assert 'Brazil' in encoding_map
        assert 'Colombia' in encoding_map
    
    def test_get_encoding_map_invalid_column(self, encoder, df_categorical):
        """Test getting encoding map for non-encoded column raises."""
        encoder.fit_transform(df_categorical, ['country'])
        
        with pytest.raises(ValueError):
            encoder.get_encoding_map('score')
