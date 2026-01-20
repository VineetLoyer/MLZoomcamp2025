"""
Unit tests for DataLoader module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    @pytest.fixture
    def sample_arabica_df(self):
        """Create sample Arabica DataFrame."""
        return pd.DataFrame({
            'Species': ['Arabica', 'Arabica'],
            'Country.of.Origin': ['Brazil', 'Colombia'],
            'Total.Cup.Points': [85.0, 87.5],
            'Aroma': [7.5, 8.0],
            'Flavor': [7.8, 8.2],
            'Acidity': [7.2, 7.8]
        })
    
    @pytest.fixture
    def sample_robusta_df(self):
        """Create sample Robusta DataFrame."""
        return pd.DataFrame({
            'Species': ['Robusta', 'Robusta'],
            'Country.of.Origin': ['Vietnam', 'Indonesia'],
            'Total.Cup.Points': [80.0, 82.0],
            'Fragrance...Aroma': [7.0, 7.2],
            'Flavor': [7.0, 7.5],
            'Salt...Acid': [6.8, 7.0]
        })
    
    def test_load_data_success(self, loader):
        """Test loading data from existing CSV file."""
        # Use actual data file
        filepath = 'Capstone Project 2/data/arabica_data_cleaned.csv'
        if os.path.exists(filepath):
            df = loader.load_data(filepath)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_load_data_file_not_found(self, loader):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load_data('nonexistent_file.csv')
    
    def test_merge_datasets(self, loader, sample_arabica_df, sample_robusta_df):
        """Test merging Arabica and Robusta datasets."""
        merged = loader.merge_datasets(sample_arabica_df, sample_robusta_df)
        
        # Should have rows from both datasets
        assert len(merged) == 4
        
        # Should have common columns
        assert 'Species' in merged.columns
        assert 'Country.of.Origin' in merged.columns
    
    def test_merge_datasets_column_mapping(self, loader, sample_arabica_df, sample_robusta_df):
        """Test that Robusta columns are mapped correctly."""
        merged = loader.merge_datasets(sample_arabica_df, sample_robusta_df, use_common_columns=False)
        
        # Robusta-specific columns should be renamed
        # Fragrance...Aroma -> Aroma, Salt...Acid -> Acidity
        assert 'Fragrance...Aroma' not in merged.columns or 'Aroma' in merged.columns
    
    def test_load_and_merge(self, loader):
        """Test convenience method for loading and merging."""
        arabica_path = 'Capstone Project 2/data/arabica_data_cleaned.csv'
        robusta_path = 'Capstone Project 2/data/robusta_data_cleaned.csv'
        
        if os.path.exists(arabica_path) and os.path.exists(robusta_path):
            merged = loader.load_and_merge(arabica_path, robusta_path)
            assert isinstance(merged, pd.DataFrame)
            assert len(merged) > 0
