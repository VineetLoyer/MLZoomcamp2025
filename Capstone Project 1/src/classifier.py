"""
MBTI Classifier for Personality Prediction.

This module provides the MBTIClassifier class that wraps 4 binary classifiers
(one for each MBTI dimension: I/E, N/S, T/F, J/P) and provides unified
prediction and serialization interfaces.

Requirements: 6.1
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import sparse

try:
    from .utils import decode_mbti, VALID_MBTI_TYPES
except ImportError:
    from utils import decode_mbti, VALID_MBTI_TYPES


# MBTI dimensions in order
DIMENSIONS = ['IE', 'NS', 'TF', 'JP']

# Dimension labels for human-readable output
DIMENSION_LABELS = {
    'IE': {'0': 'I', '1': 'E'},
    'NS': {'0': 'N', '1': 'S'},
    'TF': {'0': 'T', '1': 'F'},
    'JP': {'0': 'J', '1': 'P'}
}


class MBTIClassifier:
    """
    Multi-label classifier for MBTI personality prediction.
    
    Uses 4 binary classifiers, one for each MBTI dimension:
    - IE: Introvert (0) / Extrovert (1)
    - NS: Intuitive (0) / Sensing (1)
    - TF: Thinking (0) / Feeling (1)
    - JP: Judging (0) / Perceiving (1)
    
    Attributes:
        classifiers: Dictionary mapping dimension names to trained classifiers
    """
    
    def __init__(self):
        """Initialize the MBTIClassifier with empty classifiers."""
        self.classifiers: Dict[str, object] = {
            'IE': None,
            'NS': None,
            'TF': None,
            'JP': None
        }
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if all classifiers have been fitted/loaded."""
        return self._is_fitted and all(
            clf is not None for clf in self.classifiers.values()
        )
    
    def set_classifier(self, dimension: str, classifier: object) -> None:
        """
        Set a classifier for a specific dimension.
        
        Args:
            dimension: One of 'IE', 'NS', 'TF', 'JP'
            classifier: A trained sklearn classifier
            
        Raises:
            ValueError: If dimension is not valid
        """
        if dimension not in DIMENSIONS:
            raise ValueError(
                f"Invalid dimension: {dimension}. "
                f"Must be one of: {DIMENSIONS}"
            )
        self.classifiers[dimension] = classifier
        
        # Check if all classifiers are now set
        if all(clf is not None for clf in self.classifiers.values()):
            self._is_fitted = True
    
    def predict(self, X: Union[sparse.csr_matrix, np.ndarray]) -> List[str]:
        """
        Predict MBTI types by combining 4 binary predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            List of 4-letter MBTI type strings
            
        Raises:
            ValueError: If classifiers have not been fitted/loaded
        """
        if not self.is_fitted:
            raise ValueError(
                "MBTIClassifier has not been fitted. "
                "Load classifiers using load() first."
            )
        
        # Get predictions for each dimension
        predictions = {}
        for dim in DIMENSIONS:
            predictions[dim] = self.classifiers[dim].predict(X)
        
        # Combine predictions into MBTI types
        n_samples = X.shape[0]
        mbti_types = []
        
        for i in range(n_samples):
            encoded = {
                dim: int(predictions[dim][i])
                for dim in DIMENSIONS
            }
            mbti_type = decode_mbti(encoded)
            mbti_types.append(mbti_type)
        
        return mbti_types
    
    def predict_proba(
        self, 
        X: Union[sparse.csr_matrix, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Return probability scores for each dimension.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Dictionary with structure:
            {
                'IE': {'I': array, 'E': array},
                'NS': {'N': array, 'S': array},
                'TF': {'T': array, 'F': array},
                'JP': {'J': array, 'P': array}
            }
            Each array has shape (n_samples,) with probability values.
            
        Raises:
            ValueError: If classifiers have not been fitted/loaded
        """
        if not self.is_fitted:
            raise ValueError(
                "MBTIClassifier has not been fitted. "
                "Load classifiers using load() first."
            )
        
        result = {}
        
        for dim in DIMENSIONS:
            clf = self.classifiers[dim]
            
            # Get probability predictions
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
                # proba shape is (n_samples, 2) for binary classification
                # Column 0 = probability of class 0, Column 1 = probability of class 1
                label_0 = DIMENSION_LABELS[dim]['0']
                label_1 = DIMENSION_LABELS[dim]['1']
                result[dim] = {
                    label_0: proba[:, 0],
                    label_1: proba[:, 1]
                }
            else:
                # Fallback for classifiers without predict_proba
                # Use decision function if available
                if hasattr(clf, 'decision_function'):
                    decision = clf.decision_function(X)
                    # Convert to pseudo-probabilities using sigmoid
                    prob_1 = 1 / (1 + np.exp(-decision))
                    prob_0 = 1 - prob_1
                else:
                    # Last resort: use hard predictions
                    pred = clf.predict(X)
                    prob_1 = pred.astype(float)
                    prob_0 = 1 - prob_1
                
                label_0 = DIMENSION_LABELS[dim]['0']
                label_1 = DIMENSION_LABELS[dim]['1']
                result[dim] = {
                    label_0: prob_0,
                    label_1: prob_1
                }
        
        return result
    
    def predict_single(
        self, 
        X: Union[sparse.csr_matrix, np.ndarray]
    ) -> Dict:
        """
        Predict MBTI type for a single sample with confidence scores.
        
        Args:
            X: Feature matrix of shape (1, n_features)
            
        Returns:
            Dictionary with structure:
            {
                'mbti_type': 'INTJ',
                'confidence': {
                    'IE': {'I': 0.75, 'E': 0.25},
                    'NS': {'N': 0.82, 'S': 0.18},
                    'TF': {'T': 0.68, 'F': 0.32},
                    'JP': {'J': 0.71, 'P': 0.29}
                }
            }
        """
        mbti_types = self.predict(X)
        proba = self.predict_proba(X)
        
        # Format confidence scores for single sample
        confidence = {}
        for dim in DIMENSIONS:
            label_0 = DIMENSION_LABELS[dim]['0']
            label_1 = DIMENSION_LABELS[dim]['1']
            confidence[dim] = {
                label_0: float(proba[dim][label_0][0]),
                label_1: float(proba[dim][label_1][0])
            }
        
        return {
            'mbti_type': mbti_types[0],
            'confidence': confidence
        }
    
    def save(self, directory: Union[str, Path]) -> None:
        """
        Save all classifiers to disk.
        
        Args:
            directory: Directory path to save classifier files
            
        Raises:
            ValueError: If classifiers have not been fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "Cannot save unfitted MBTIClassifier. "
                "Fit or load classifiers first."
            )
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for dim, clf in self.classifiers.items():
            clf_path = directory / f'classifier_{dim}.pkl'
            with open(clf_path, 'wb') as f:
                pickle.dump(clf, f)
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'MBTIClassifier':
        """
        Load classifiers from disk.
        
        Args:
            directory: Directory path containing classifier files
            
        Returns:
            Loaded MBTIClassifier instance
            
        Raises:
            FileNotFoundError: If classifier files are not found
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        classifier = cls()
        
        for dim in DIMENSIONS:
            clf_path = directory / f'classifier_{dim}.pkl'
            
            if not clf_path.exists():
                raise FileNotFoundError(
                    f"Classifier file not found: {clf_path}"
                )
            
            with open(clf_path, 'rb') as f:
                clf = pickle.load(f)
            
            classifier.classifiers[dim] = clf
        
        classifier._is_fitted = True
        return classifier
    
    def __repr__(self) -> str:
        """String representation of the MBTIClassifier."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"MBTIClassifier(status={status})"
