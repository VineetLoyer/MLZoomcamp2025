#!/usr/bin/env python
"""Training script for Coffee Quality Prediction."""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_scaler import FeatureScaler
from data_splitter import DataSplitter
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from model_serializer import ModelSerializer

RANDOM_STATE = 42
TARGET_COLUMN = 'Total_Cup_Points'

SENSORY_FEATURES = [
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
    'Balance', 'Uniformity', 'Clean_Cup', 'Sweetness', 'Cupper_Points'
]

BEST_PARAMS = {
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 0.1},
    'randomforest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }
}


def load_and_prepare_data(data_dir: str) -> pd.DataFrame:
    """Load and prepare the coffee quality dataset."""
    print("Loading data...")
    loader = DataLoader()
    
    arabica_path = os.path.join(data_dir, 'arabica_data_cleaned.csv')
    robusta_path = os.path.join(data_dir, 'robusta_data_cleaned.csv')
    
    arabica_df = loader.load_data(arabica_path)
    robusta_df = loader.load_data(robusta_path)
    df = loader.merge_datasets(arabica_df, robusta_df)
    
    print(f"  Loaded {len(df)} samples")
    
    cleaner = DataCleaner()
    df = cleaner.clean_column_names(df)
    
    return df


def preprocess_data(df: pd.DataFrame, feature_cols: list):
    """Preprocess the data for model training."""
    print("Preprocessing data...")
    
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"  Using {len(available_features)} features: {available_features}")
    
    df_clean = df[available_features + [TARGET_COLUMN]].dropna()
    print(f"  Clean dataset: {len(df_clean)} samples")
    
    X = df_clean[available_features].values
    y = df_clean[TARGET_COLUMN].values
    
    splitter = DataSplitter(random_state=RANDOM_STATE)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, train_size=0.6, val_size=0.2, test_size=0.2
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    scaler = FeatureScaler(scaler_type='standard')
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, scaler, available_features)


def train_models(X_train, y_train, X_val, y_val):
    """Train multiple models with tuned hyperparameters."""
    print("\nTraining models...")
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    evaluator = ModelEvaluator()
    trained_models = []
    
    print("  Training Ridge Regression...")
    ridge_model = trainer.train_linear_model(
        X_train, y_train, model_type='ridge', **BEST_PARAMS['ridge']
    )
    ridge_metrics = evaluator.evaluate(ridge_model, X_val, y_val)
    trained_models.append(('Ridge', ridge_model))
    print(f"    RMSE: {ridge_metrics['rmse']:.4f}, R²: {ridge_metrics['r2']:.4f}")
    
    print("  Training Lasso Regression...")
    lasso_model = trainer.train_linear_model(
        X_train, y_train, model_type='lasso', **BEST_PARAMS['lasso']
    )
    lasso_metrics = evaluator.evaluate(lasso_model, X_val, y_val)
    trained_models.append(('Lasso', lasso_model))
    print(f"    RMSE: {lasso_metrics['rmse']:.4f}, R²: {lasso_metrics['r2']:.4f}")
    
    print("  Training Random Forest...")
    rf_model = trainer.train_tree_model(
        X_train, y_train, model_type='randomforest', **BEST_PARAMS['randomforest']
    )
    rf_metrics = evaluator.evaluate(rf_model, X_val, y_val)
    trained_models.append(('RandomForest', rf_model))
    print(f"    RMSE: {rf_metrics['rmse']:.4f}, R²: {rf_metrics['r2']:.4f}")
    
    print("  Training XGBoost...")
    xgb_model = trainer.train_tree_model(
        X_train, y_train, model_type='xgboost', **BEST_PARAMS['xgboost']
    )
    xgb_metrics = evaluator.evaluate(xgb_model, X_val, y_val)
    trained_models.append(('XGBoost', xgb_model))
    print(f"    RMSE: {xgb_metrics['rmse']:.4f}, R²: {xgb_metrics['r2']:.4f}")
    
    return trained_models


def select_and_evaluate_best_model(trained_models, X_val, y_val, X_test, y_test):
    """Select the best model and evaluate on test set."""
    print("\nSelecting best model...")
    evaluator = ModelEvaluator()
    
    best_name = None
    best_model = None
    best_rmse = float('inf')
    
    for name, model in trained_models:
        metrics = evaluator.evaluate(model, X_val, y_val)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_name = name
            best_model = model
    
    val_metrics = evaluator.evaluate(best_model, X_val, y_val)
    print(f"  Best model: {best_name}")
    print(f"  Validation metrics:")
    print(f"    RMSE: {val_metrics['rmse']:.4f}")
    print(f"    MAE: {val_metrics['mae']:.4f}")
    print(f"    R²: {val_metrics['r2']:.4f}")
    
    test_metrics = evaluator.evaluate(best_model, X_test, y_test)
    print(f"\n  Test set metrics:")
    print(f"    RMSE: {test_metrics['rmse']:.4f}")
    print(f"    MAE: {test_metrics['mae']:.4f}")
    print(f"    R²: {test_metrics['r2']:.4f}")
    
    return best_name, best_model, test_metrics


def save_model(model, scaler, feature_names, metrics, model_path: str):
    """Save the trained model with preprocessor and metadata."""
    print(f"\nSaving model to {model_path}...")
    
    serializer = ModelSerializer()
    
    metadata = {
        'feature_names': feature_names,
        'target_column': TARGET_COLUMN,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        'random_state': RANDOM_STATE
    }
    
    serializer.save_model(model, model_path, preprocessor=scaler, metadata=metadata)
    print("  Model saved successfully!")
    print(f"  Metadata: {metadata}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Coffee Quality Prediction Model')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/model.pkl',
        help='Path to save the trained model (default: models/model.pkl)'
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    model_path = os.path.join(script_dir, args.model_path)
    
    print("=" * 60)
    print("Coffee Quality Prediction - Model Training")
    print("=" * 60)
    
    df = load_and_prepare_data(data_dir)
    
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names) = preprocess_data(df, SENSORY_FEATURES)
    
    trained_models = train_models(X_train, y_train, X_val, y_val)
    
    best_name, best_model, test_metrics = select_and_evaluate_best_model(
        trained_models, X_val, y_val, X_test, y_test
    )
    
    save_model(best_model, scaler, feature_names, test_metrics, model_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
