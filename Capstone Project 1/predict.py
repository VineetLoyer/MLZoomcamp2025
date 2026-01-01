#!/usr/bin/env python
"""
MBTI Personality Prediction - Flask Prediction Service

This module provides a REST API for predicting MBTI personality types
from text input using trained machine learning models.

Endpoints:
    POST /predict - Predict MBTI type from text
    GET /health - Health check endpoint

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import os
import sys
from pathlib import Path

from flask import Flask, request, jsonify

# Add src to path for importing project modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from classifier import MBTIClassifier

# Configuration
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
MIN_TEXT_LENGTH = 10

# Initialize Flask app
app = Flask(__name__)

# Global model components (loaded on startup)
preprocessor = None
feature_extractor = None
classifier = None
model_loaded = False


def load_models():
    """
    Load model artifacts on startup.
    
    Loads:
    - TextPreprocessor (stateless, just instantiate)
    - FeatureExtractor (fitted TF-IDF vectorizer)
    - MBTIClassifier (4 binary classifiers)
    """
    global preprocessor, feature_extractor, classifier, model_loaded
    
    models_path = Path(MODELS_DIR)
    
    try:
        # Initialize preprocessor (stateless)
        preprocessor = TextPreprocessor()
        
        # Load fitted vectorizer
        vectorizer_path = models_path / 'vectorizer.pkl'
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        feature_extractor = FeatureExtractor.load(vectorizer_path)
        
        # Load classifiers
        classifier = MBTIClassifier.load(models_path)
        
        model_loaded = True
        print(f"Models loaded successfully from {models_path}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        model_loaded = False
        raise


def validate_input(data):
    """
    Validate input data for prediction request.
    
    Args:
        data: Request JSON data
        
    Returns:
        Tuple of (is_valid, error_message, text)
    """
    if data is None:
        return False, "Invalid JSON format", None
    
    if 'text' not in data:
        return False, "Missing required field: text", None
    
    text = data['text']
    
    if not isinstance(text, str):
        return False, "Field 'text' must be a string", None
    
    if not text or not text.strip():
        return False, "Text cannot be empty", None
    
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False, f"Text too short for reliable prediction (minimum {MIN_TEXT_LENGTH} characters)", None
    
    return True, None, text


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict MBTI personality type from text.
    
    Request Body:
        {"text": "sample text to analyze..."}
        
    Response:
        {
            "mbti_type": "INTJ",
            "confidence": {
                "IE": {"I": 0.75, "E": 0.25},
                "NS": {"N": 0.82, "S": 0.18},
                "TF": {"T": 0.68, "F": 0.32},
                "JP": {"J": 0.71, "P": 0.29}
            }
        }
        
    Error Response:
        {"error": "error message"}
    """
    # Check if models are loaded
    if not model_loaded:
        return jsonify({"error": "Model not available"}), 503
    
    # Parse and validate input
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400
    
    is_valid, error_msg, text = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    try:
        # Preprocess text (same as training)
        cleaned_text = preprocessor.preprocess_posts(text)
        
        # Check if text is empty after preprocessing
        if not cleaned_text or not cleaned_text.strip():
            return jsonify({
                "error": "Text contains no meaningful content after preprocessing"
            }), 400
        
        # Extract features
        features = feature_extractor.transform([cleaned_text])
        
        # Get prediction with confidence scores
        result = classifier.predict_single(features)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Response:
        {"status": "healthy"} if models are loaded
        {"status": "unhealthy", "reason": "..."} otherwise
    """
    if model_loaded:
        return jsonify({"status": "healthy"}), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "reason": "Models not loaded"
        }), 503


# Load models when module is imported
try:
    load_models()
except Exception as e:
    print(f"Warning: Could not load models on startup: {e}")


if __name__ == '__main__':
    # Run the Flask development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting MBTI Prediction Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
