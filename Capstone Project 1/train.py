#!/usr/bin/env python
"""
MBTI Personality Prediction - Training Script

This script implements the full training pipeline:
1. Load data from CSV
2. Preprocess text data
3. Extract TF-IDF features
4. Train binary classifiers for each MBTI dimension
5. Save model artifacts to models/ directory

Usage:
    python train.py [--data-path DATA_PATH] [--models-dir MODELS_DIR]

Requirements: 5.1, 5.2, 5.3
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Add src to path for importing project modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from utils import encode_mbti, VALID_MBTI_TYPES

# Constants
DIMENSIONS = ['IE', 'NS', 'TF', 'JP']
RANDOM_STATE = 42


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the MBTI dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and duplicates.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning data...")
    original_size = len(df)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Validate MBTI types
    df = df[df['type'].isin(VALID_MBTI_TYPES)]
    
    print(f"Removed {original_size - len(df)} invalid/duplicate rows")
    print(f"Final dataset size: {len(df)}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing and create binary labels.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with preprocessed text and binary labels
    """
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    
    # Preprocess all posts
    df = df.copy()
    df['posts_clean'] = df['posts'].apply(preprocessor.preprocess_posts)
    
    # Create binary labels for each dimension
    print("Creating binary labels...")
    encoded = df['type'].apply(encode_mbti)
    
    for dim in DIMENSIONS:
        df[dim] = encoded.apply(lambda x: x[dim])
    
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df: Preprocessed DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Splitting data into train/val/test sets...")
    
    # Create combined label for stratification
    df = df.copy()
    df['combined_label'] = df['type']
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['combined_label'],
        random_state=RANDOM_STATE
    )
    
    # Second split: separate validation from training
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df['combined_label'],
        random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def extract_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FeatureExtractor]:
    """
    Extract TF-IDF features from text data.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        max_features: Maximum number of TF-IDF features
        
    Returns:
        Tuple of (X_train, X_val, X_test, feature_extractor)
    """
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    
    feature_extractor = FeatureExtractor(max_features=max_features)
    
    # Fit on training data and transform all sets
    X_train = feature_extractor.fit_transform(train_df['posts_clean'].tolist())
    X_val = feature_extractor.transform(val_df['posts_clean'].tolist())
    X_test = feature_extractor.transform(test_df['posts_clean'].tolist())
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    return X_train, X_val, X_test, feature_extractor


def train_classifiers(
    X_train: np.ndarray,
    y_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: pd.DataFrame
) -> Tuple[Dict[str, object], Dict[str, Dict]]:
    """
    Train binary classifiers for each MBTI dimension.
    
    Uses Logistic Regression for most dimensions and Gradient Boosting
    for JP dimension (based on notebook experimentation results).
    
    Args:
        X_train: Training features
        y_train: Training labels DataFrame with dimension columns
        X_val: Validation features
        y_val: Validation labels DataFrame
        
    Returns:
        Tuple of (classifiers dict, metrics dict)
    """
    print("Training classifiers for each dimension...")
    
    classifiers = {}
    metrics = {}
    
    for dim in DIMENSIONS:
        print(f"\n  Training {dim} classifier...")
        
        # Use Logistic Regression for all dimensions
        # (fast training with good performance)
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        model_name = 'Logistic Regression'
        
        # Train
        clf.fit(X_train, y_train[dim].values)
        
        # Evaluate on validation set
        y_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val[dim].values, y_pred)
        val_f1 = f1_score(y_val[dim].values, y_pred)
        
        classifiers[dim] = clf
        metrics[dim] = {
            'model_name': model_name,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }
        
        print(f"    {model_name}: Val Accuracy={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    return classifiers, metrics


def evaluate_on_test(
    classifiers: Dict[str, object],
    X_test: np.ndarray,
    y_test: pd.DataFrame
) -> List[Dict]:
    """
    Evaluate trained classifiers on test set.
    
    Args:
        classifiers: Dictionary of trained classifiers
        X_test: Test features
        y_test: Test labels DataFrame
        
    Returns:
        List of test metrics for each dimension
    """
    print("\nEvaluating on test set...")
    
    test_metrics = []
    
    for dim in DIMENSIONS:
        clf = classifiers[dim]
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test[dim].values, y_pred)
        f1 = f1_score(y_test[dim].values, y_pred)
        
        test_metrics.append({
            'dimension': dim,
            'accuracy': acc,
            'f1': f1
        })
        
        print(f"  {dim}: Test Accuracy={acc:.4f}, Test F1={f1:.4f}")
    
    return test_metrics


def save_artifacts(
    classifiers: Dict[str, object],
    feature_extractor: FeatureExtractor,
    val_metrics: Dict[str, Dict],
    test_metrics: List[Dict],
    models_dir: str
) -> None:
    """
    Save model artifacts to disk.
    
    Args:
        classifiers: Dictionary of trained classifiers
        feature_extractor: Fitted feature extractor
        val_metrics: Validation metrics for each dimension
        test_metrics: Test metrics for each dimension
        models_dir: Directory to save artifacts
    """
    print(f"\nSaving artifacts to {models_dir}...")
    
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature extractor (vectorizer)
    feature_extractor.save(models_path / 'vectorizer.pkl')
    print("  Saved vectorizer.pkl")
    
    # Save each classifier
    for dim, clf in classifiers.items():
        clf_path = models_path / f'classifier_{dim}.pkl'
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"  Saved classifier_{dim}.pkl")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'Logistic Regression',  # Primary model type
        'dimensions': DIMENSIONS,
        'test_metrics': test_metrics,
        'final_models': [
            {
                'dimension': dim,
                'model_name': val_metrics[dim]['model_name'],
                'val_f1': val_metrics[dim]['val_f1']
            }
            for dim in DIMENSIONS
        ]
    }
    
    with open(models_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Saved metadata.json")
    
    print("\nTraining complete!")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train MBTI personality prediction models'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/mbti_1.csv',
        help='Path to the MBTI dataset CSV file'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save model artifacts'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum number of TF-IDF features'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MBTI Personality Prediction - Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data(args.data_path)
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Preprocess data
    df = preprocess_data(df)
    
    # Step 4: Split data
    train_df, val_df, test_df = split_data(df)
    
    # Step 5: Extract features
    X_train, X_val, X_test, feature_extractor = extract_features(
        train_df, val_df, test_df,
        max_features=args.max_features
    )
    
    # Prepare label DataFrames
    y_train = train_df[DIMENSIONS]
    y_val = val_df[DIMENSIONS]
    y_test = test_df[DIMENSIONS]
    
    # Step 6: Train classifiers
    classifiers, val_metrics = train_classifiers(X_train, y_train, X_val, y_val)
    
    # Step 7: Evaluate on test set
    test_metrics = evaluate_on_test(classifiers, X_test, y_test)
    
    # Step 8: Save artifacts
    save_artifacts(
        classifiers,
        feature_extractor,
        val_metrics,
        test_metrics,
        args.models_dir
    )
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
