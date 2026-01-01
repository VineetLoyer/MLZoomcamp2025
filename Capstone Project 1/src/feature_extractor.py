"""
Feature Extractor for MBTI Personality Prediction.

This module provides TF-IDF based feature extraction functionality
for transforming preprocessed text into numerical features for ML models.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    Feature extractor that transforms preprocessed text into TF-IDF vectors.
    
    Provides configurable TF-IDF vectorization with support for:
    - Configurable max_features to limit vocabulary size
    - Configurable ngram_range for capturing word sequences
    - Serialization for model persistence
    
    Attributes:
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Tuple specifying min and max n-gram lengths
        stop_words: Stop words to remove ('english' or custom list)
        tfidf_vectorizer: The underlying TfidfVectorizer instance
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        stop_words: Union[str, List[str]] = 'english',
        min_df: int = 1,
        max_df: float = 0.95
    ):
        """
        Initialize the FeatureExtractor.
        
        Args:
            max_features: Maximum number of features to extract (default: 5000)
            ngram_range: Tuple of (min_n, max_n) for n-gram range (default: (1, 2))
            stop_words: Stop words to remove (default: 'english')
            min_df: Minimum document frequency for terms (default: 1)
            max_df: Maximum document frequency for terms (default: 0.95)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
            norm='l2'  # L2 normalization
        )
        
        self._is_fitted = False

    
    @property
    def is_fitted(self) -> bool:
        """Check if the vectorizer has been fitted."""
        return self._is_fitted
    
    @property
    def vocabulary_size(self) -> int:
        """Get the size of the fitted vocabulary."""
        if not self._is_fitted:
            return 0
        return len(self.tfidf_vectorizer.vocabulary_)
    
    @property
    def feature_names(self) -> List[str]:
        """Get the feature names (vocabulary terms)."""
        if not self._is_fitted:
            return []
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            self for method chaining
        """
        self.tfidf_vectorizer.fit(texts)
        self._is_fitted = True
        return self
    
    def fit_transform(self, texts: List[str]) -> sparse.csr_matrix:
        """
        Fit the vectorizer and transform training texts to TF-IDF vectors.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Sparse matrix of shape (n_samples, n_features) with TF-IDF values
        """
        result = self.tfidf_vectorizer.fit_transform(texts)
        self._is_fitted = True
        return result
    
    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """
        Transform texts to TF-IDF vectors using fitted vectorizer.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Sparse matrix of shape (n_samples, n_features) with TF-IDF values
            
        Raises:
            ValueError: If vectorizer has not been fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "FeatureExtractor has not been fitted. "
                "Call fit() or fit_transform() first."
            )
        return self.tfidf_vectorizer.transform(texts)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            path: File path to save the vectorizer
            
        Raises:
            ValueError: If vectorizer has not been fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "Cannot save unfitted FeatureExtractor. "
                "Call fit() or fit_transform() first."
            )
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all relevant state
        state = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'stop_words': self.stop_words,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            '_is_fitted': self._is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            path: File path to load the vectorizer from
            
        Returns:
            Loaded FeatureExtractor instance
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"No saved vectorizer found at: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance and restore state
        extractor = cls(
            max_features=state['max_features'],
            ngram_range=state['ngram_range'],
            stop_words=state['stop_words'],
            min_df=state['min_df'],
            max_df=state['max_df']
        )
        extractor.tfidf_vectorizer = state['tfidf_vectorizer']
        extractor._is_fitted = state['_is_fitted']
        
        return extractor
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get the top N features by IDF score.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of top feature names
        """
        if not self._is_fitted:
            return []
        
        # Get IDF scores
        idf_scores = self.tfidf_vectorizer.idf_
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Sort by IDF score (higher = more discriminative)
        sorted_indices = np.argsort(idf_scores)[::-1][:n]
        
        return [feature_names[i] for i in sorted_indices]
    
    def __repr__(self) -> str:
        """String representation of the FeatureExtractor."""
        status = "fitted" if self._is_fitted else "not fitted"
        vocab_size = self.vocabulary_size if self._is_fitted else "N/A"
        return (
            f"FeatureExtractor("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"status={status}, "
            f"vocabulary_size={vocab_size})"
        )
