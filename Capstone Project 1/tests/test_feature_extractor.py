"""
Unit tests for FeatureExtractor class.

Tests cover:
- TF-IDF vectorization output shape and values
- Vocabulary building
- Transform on new data
- Save/load functionality
- Edge cases
"""

import os
import tempfile
import pytest
import numpy as np
from scipy import sparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extractor import FeatureExtractor


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""
    
    def test_default_initialization(self):
        """Test default parameter values."""
        fe = FeatureExtractor()
        assert fe.max_features == 5000
        assert fe.ngram_range == (1, 2)
        assert fe.stop_words == 'english'
        assert fe.is_fitted is False
        assert fe.vocabulary_size == 0
    
    def test_custom_initialization(self):
        """Test custom parameter values."""
        fe = FeatureExtractor(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=2,
            max_df=0.9
        )
        assert fe.max_features == 1000
        assert fe.ngram_range == (1, 3)
        assert fe.stop_words is None
        assert fe.min_df == 2
        assert fe.max_df == 0.9


class TestFeatureExtractorFitTransform:
    """Tests for fit_transform method."""
    
    def test_fit_transform_returns_sparse_matrix(self):
        """Test that fit_transform returns a sparse matrix."""
        fe = FeatureExtractor(max_features=100)
        texts = [
            "hello world this is a test",
            "another sample text here",
            "machine learning is great"
        ]
        result = fe.fit_transform(texts)
        assert sparse.issparse(result)
        assert isinstance(result, sparse.csr_matrix)
    
    def test_fit_transform_correct_shape(self):
        """Test output matrix has correct shape."""
        fe = FeatureExtractor(max_features=100, stop_words=None)
        texts = ["text one document", "text two document", "text three document", "text four document"]
        result = fe.fit_transform(texts)
        assert result.shape[0] == 4  # Number of documents
        assert result.shape[1] <= 100  # Max features
    
    def test_fit_transform_sets_fitted_flag(self):
        """Test that fit_transform sets is_fitted to True."""
        fe = FeatureExtractor(stop_words=None)
        assert fe.is_fitted is False
        fe.fit_transform(["sample text document", "another sample document"])
        assert fe.is_fitted is True

    
    def test_fit_transform_values_in_range(self):
        """Test that TF-IDF values are in valid range [0, 1]."""
        fe = FeatureExtractor(max_features=100)
        texts = [
            "hello world test document",
            "another sample text here",
            "machine learning algorithms"
        ]
        result = fe.fit_transform(texts)
        # Convert to dense for easier checking
        dense = result.toarray()
        assert np.all(dense >= 0.0)
        assert np.all(dense <= 1.0)
    
    def test_fit_transform_vocabulary_size(self):
        """Test vocabulary size is set after fitting."""
        fe = FeatureExtractor(max_features=100)
        texts = ["word1 word2", "word3 word4", "word5 word6"]
        fe.fit_transform(texts)
        assert fe.vocabulary_size > 0
        assert fe.vocabulary_size <= 100


class TestFeatureExtractorTransform:
    """Tests for transform method."""
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform raises error if not fitted."""
        fe = FeatureExtractor()
        with pytest.raises(ValueError, match="has not been fitted"):
            fe.transform(["test text"])
    
    def test_transform_after_fit(self):
        """Test transform works after fitting."""
        fe = FeatureExtractor(max_features=100, stop_words=None)
        train_texts = ["training text document one", "training text document two"]
        fe.fit_transform(train_texts)
        
        test_texts = ["test text document one", "test text document two"]
        result = fe.transform(test_texts)
        
        assert sparse.issparse(result)
        assert result.shape[0] == 2
    
    def test_transform_same_features_as_fit(self):
        """Test transform produces same number of features as fit."""
        fe = FeatureExtractor(max_features=100)
        train_texts = ["hello world test", "sample document here"]
        X_train = fe.fit_transform(train_texts)
        
        test_texts = ["new test document"]
        X_test = fe.transform(test_texts)
        
        assert X_train.shape[1] == X_test.shape[1]


class TestFeatureExtractorSaveLoad:
    """Tests for save and load methods."""
    
    def test_save_unfitted_raises_error(self):
        """Test that saving unfitted extractor raises error."""
        fe = FeatureExtractor()
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Cannot save unfitted"):
                fe.save(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_save_and_load_roundtrip(self):
        """Test that save and load produce equivalent extractor."""
        fe = FeatureExtractor(max_features=100, ngram_range=(1, 2))
        texts = ["hello world test", "sample text here", "another document"]
        X_original = fe.fit_transform(texts)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        try:
            fe.save(path)
            fe_loaded = FeatureExtractor.load(path)
            
            # Check parameters are preserved
            assert fe_loaded.max_features == fe.max_features
            assert fe_loaded.ngram_range == fe.ngram_range
            assert fe_loaded.is_fitted is True
            assert fe_loaded.vocabulary_size == fe.vocabulary_size
            
            # Check transform produces same results
            X_loaded = fe_loaded.transform(texts)
            assert (X_original != X_loaded).nnz == 0  # Sparse matrices equal
        finally:
            os.remove(path)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureExtractor.load("/nonexistent/path/file.pkl")


class TestFeatureExtractorProperties:
    """Tests for property methods."""
    
    def test_feature_names_empty_before_fit(self):
        """Test feature_names is empty before fitting."""
        fe = FeatureExtractor()
        assert fe.feature_names == []
    
    def test_feature_names_after_fit(self):
        """Test feature_names returns vocabulary after fitting."""
        fe = FeatureExtractor(max_features=100)
        texts = ["hello world", "sample text"]
        fe.fit_transform(texts)
        
        names = fe.feature_names
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)
    
    def test_get_top_features(self):
        """Test get_top_features returns correct number of features."""
        fe = FeatureExtractor(max_features=100)
        texts = ["hello world test", "sample text here", "another document"]
        fe.fit_transform(texts)
        
        top_5 = fe.get_top_features(5)
        assert len(top_5) <= 5
        assert all(isinstance(name, str) for name in top_5)
    
    def test_get_top_features_before_fit(self):
        """Test get_top_features returns empty list before fitting."""
        fe = FeatureExtractor()
        assert fe.get_top_features(10) == []
    
    def test_repr(self):
        """Test string representation."""
        fe = FeatureExtractor(max_features=1000)
        repr_str = repr(fe)
        assert "FeatureExtractor" in repr_str
        assert "max_features=1000" in repr_str
        assert "not fitted" in repr_str
        
        fe.fit_transform(["test text document", "another sample document"])
        repr_str = repr(fe)
        assert "fitted" in repr_str


class TestFeatureExtractorEdgeCases:
    """Tests for edge cases."""
    
    def test_single_document(self):
        """Test with small number of documents."""
        fe = FeatureExtractor(max_features=100, stop_words=None, max_df=1.0)
        texts = ["single document with some words here"]
        result = fe.fit_transform(texts)
        assert result.shape[0] == 1
    
    def test_empty_text_in_list(self):
        """Test handling of empty strings in text list."""
        fe = FeatureExtractor(max_features=100)
        texts = ["hello world", "", "sample text"]
        result = fe.fit_transform(texts)
        assert result.shape[0] == 3
    
    def test_max_features_limit(self):
        """Test that max_features limits vocabulary size."""
        fe = FeatureExtractor(max_features=5)
        # Create texts with many unique words
        texts = [
            "word1 word2 word3 word4 word5",
            "word6 word7 word8 word9 word10",
            "word11 word12 word13 word14 word15"
        ]
        result = fe.fit_transform(texts)
        assert result.shape[1] <= 5
    
    def test_ngram_range_unigrams_only(self):
        """Test with unigrams only."""
        fe = FeatureExtractor(max_features=100, ngram_range=(1, 1))
        texts = ["hello world", "sample text"]
        fe.fit_transform(texts)
        # With unigrams only, no bigrams should be in vocabulary
        names = fe.feature_names
        assert all(' ' not in name for name in names)
    
    def test_ngram_range_with_bigrams(self):
        """Test with bigrams included."""
        fe = FeatureExtractor(max_features=100, ngram_range=(1, 2), stop_words=None)
        texts = ["hello world test", "hello world again"]
        fe.fit_transform(texts)
        names = fe.feature_names
        # Should have some bigrams (containing space)
        bigrams = [name for name in names if ' ' in name]
        assert len(bigrams) > 0
