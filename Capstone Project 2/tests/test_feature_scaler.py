"""
Unit tests for FeatureScaler module.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_scaler import FeatureScaler


class TestFeatureScaler:
    """Tests for FeatureScaler class."""
    
    @pytest.fixture
    def standard_scaler(self):
        """Create a StandardScaler instance."""
        return FeatureScaler(scaler_type='standard')
    
    @pytest.fixture
    def minmax_scaler(self):
        """Create a MinMaxScaler instance."""
        return FeatureScaler(scaler_type='minmax')
    
    @pytest.fixture
    def sample_array(self):
        """Create sample numpy array."""
        return np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'category': ['a', 'b', 'c', 'd', 'e']
        })
    
    def test_standard_scaler_fit_transform(self, standard_scaler, sample_array):
        """Test StandardScaler fit_transform."""
        result = standard_scaler.fit_transform(sample_array)
        
        # Mean should be approximately 0
        assert np.abs(result.mean(axis=0)).max() < 1e-10
        # Std should be approximately 1
        assert np.abs(result.std(axis=0) - 1).max() < 1e-10
    
    def test_minmax_scaler_fit_transform(self, minmax_scaler, sample_array):
        """Test MinMaxScaler fit_transform."""
        result = minmax_scaler.fit_transform(sample_array)
        
        # Values should be in [0, 1] range
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_transform_without_fit_raises(self, standard_scaler, sample_array):
        """Test transform without fit raises ValueError."""
        with pytest.raises(ValueError):
            standard_scaler.transform(sample_array)
    
    def test_dataframe_scaling(self, standard_scaler, sample_df):
        """Test scaling DataFrame columns."""
        result = standard_scaler.fit_transform(sample_df, columns=['feature1', 'feature2'])
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Numeric columns should be scaled
        assert np.abs(result['feature1'].mean()) < 1e-10
        assert np.abs(result['feature2'].mean()) < 1e-10
        
        # Non-numeric column should be unchanged
        assert list(result['category']) == list(sample_df['category'])
    
    def test_inverse_transform(self, standard_scaler, sample_array):
        """Test inverse transform recovers original values."""
        scaled = standard_scaler.fit_transform(sample_array)
        recovered = standard_scaler.inverse_transform(scaled)
        
        # Should recover original values
        np.testing.assert_array_almost_equal(recovered, sample_array)
    
    def test_invalid_scaler_type(self):
        """Test invalid scaler type raises ValueError."""
        with pytest.raises(ValueError):
            FeatureScaler(scaler_type='invalid')
    
    def test_feature_names_property(self, standard_scaler, sample_df):
        """Test feature_names property."""
        standard_scaler.fit_transform(sample_df, columns=['feature1', 'feature2'])
        
        assert 'feature1' in standard_scaler.feature_names
        assert 'feature2' in standard_scaler.feature_names
    
    def test_mean_property(self, standard_scaler, sample_array):
        """Test mean_ property for StandardScaler."""
        standard_scaler.fit_transform(sample_array)
        
        assert standard_scaler.mean_ is not None
        assert len(standard_scaler.mean_) == 2
    
    def test_scale_property(self, standard_scaler, sample_array):
        """Test scale_ property."""
        standard_scaler.fit_transform(sample_array)
        
        assert standard_scaler.scale_ is not None
        assert len(standard_scaler.scale_) == 2
