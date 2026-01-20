"""
Unit tests for DataSplitter module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_splitter import DataSplitter


class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    @pytest.fixture
    def splitter(self):
        """Create a DataSplitter instance."""
        return DataSplitter(random_state=42)
    
    @pytest.fixture
    def sample_X(self):
        """Create sample feature matrix."""
        return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                        [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
    
    @pytest.fixture
    def sample_y(self):
        """Create sample target vector."""
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': range(200, 300)
        })
    
    def test_split_sizes(self, splitter, sample_X, sample_y):
        """Test split produces correct sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            sample_X, sample_y,
            train_size=0.6, val_size=0.2, test_size=0.2
        )
        
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(sample_X)
        
        # Check approximate proportions
        assert len(X_train) >= 5  # ~60%
        assert len(X_val) >= 1    # ~20%
        assert len(X_test) >= 1   # ~20%
    
    def test_split_reproducibility(self, sample_X, sample_y):
        """Test split is reproducible with same random state."""
        splitter1 = DataSplitter(random_state=42)
        splitter2 = DataSplitter(random_state=42)
        
        result1 = splitter1.split(sample_X, sample_y)
        result2 = splitter2.split(sample_X, sample_y)
        
        # Should produce identical splits
        np.testing.assert_array_equal(result1[0], result2[0])  # X_train
        np.testing.assert_array_equal(result1[3], result2[3])  # y_train
    
    def test_split_different_random_state(self, sample_X, sample_y):
        """Test different random states produce different splits."""
        splitter1 = DataSplitter(random_state=42)
        splitter2 = DataSplitter(random_state=123)
        
        result1 = splitter1.split(sample_X, sample_y)
        result2 = splitter2.split(sample_X, sample_y)
        
        # Should produce different splits (with high probability)
        # At least one element should differ
        assert not np.array_equal(result1[0], result2[0])
    
    def test_split_invalid_sizes(self, splitter, sample_X, sample_y):
        """Test invalid split sizes raise ValueError."""
        with pytest.raises(ValueError):
            splitter.split(sample_X, sample_y, train_size=0.5, val_size=0.3, test_size=0.3)
    
    def test_split_negative_size(self, splitter, sample_X, sample_y):
        """Test negative split size raises ValueError."""
        with pytest.raises(ValueError):
            splitter.split(sample_X, sample_y, train_size=-0.1, val_size=0.5, test_size=0.6)
    
    def test_split_dataframe(self, splitter, sample_df):
        """Test splitting DataFrame."""
        train_df, val_df, test_df = splitter.split_dataframe(
            sample_df, target_column='target'
        )
        
        # Should return DataFrames
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        
        # Total rows should match
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(sample_df)
        
        # Target column should be present
        assert 'target' in train_df.columns
        assert 'target' in val_df.columns
        assert 'target' in test_df.columns
    
    def test_get_split_sizes(self, splitter):
        """Test calculating split sizes."""
        n_train, n_val, n_test = splitter.get_split_sizes(
            n_samples=100,
            train_size=0.6, val_size=0.2, test_size=0.2
        )
        
        assert n_train == 60
        assert n_val == 20
        assert n_test == 20
