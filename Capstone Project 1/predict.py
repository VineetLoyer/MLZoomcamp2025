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

from flask import Flask, request, jsonify, render_template_string

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


# HTML template for the web demo
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBTI Personality Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 10px; font-size: 2em; }
        .subtitle { color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 30px; }
        .card { background: white; border-radius: 16px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        textarea { 
            width: 100%; height: 150px; padding: 15px; border: 2px solid #e0e0e0;
            border-radius: 10px; font-size: 16px; resize: vertical; margin-bottom: 20px;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        button { 
            width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102,126,234,0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result { margin-top: 25px; padding: 20px; background: #f8f9fa; border-radius: 10px; display: none; }
        .mbti-type { font-size: 48px; font-weight: bold; text-align: center; color: #667eea; margin-bottom: 20px; }
        .dimension { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #e0e0e0; }
        .dimension:last-child { border-bottom: none; }
        .dim-label { font-weight: 600; color: #333; }
        .confidence-bar { width: 200px; height: 24px; background: #e0e0e0; border-radius: 12px; overflow: hidden; position: relative; }
        .confidence-fill { height: 100%; border-radius: 12px; transition: width 0.5s; }
        .confidence-text { position: absolute; width: 100%; text-align: center; line-height: 24px; font-size: 12px; font-weight: 600; }
        .error { color: #dc3545; text-align: center; margin-top: 15px; }
        .footer { text-align: center; margin-top: 20px; color: rgba(255,255,255,0.7); font-size: 14px; }
        .footer a { color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 MBTI Personality Predictor</h1>
        <p class="subtitle">Predict your Myers-Briggs personality type from text</p>
        <div class="card">
            <textarea id="text" placeholder="Enter some text about yourself, your thoughts, or how you communicate... (minimum 10 characters)"></textarea>
            <button id="predict-btn" onclick="predict()">Predict My Personality</button>
            <div id="error" class="error"></div>
            <div id="result" class="result">
                <div id="mbti-type" class="mbti-type"></div>
                <div id="dimensions"></div>
            </div>
        </div>
        <p class="footer">Built for <a href="https://github.com/DataTalksClub/machine-learning-zoomcamp" target="_blank">ML Zoomcamp</a> Capstone Project</p>
    </div>
    <script>
        async function predict() {
            const text = document.getElementById('text').value;
            const btn = document.getElementById('predict-btn');
            const error = document.getElementById('error');
            const result = document.getElementById('result');
            
            error.textContent = '';
            result.style.display = 'none';
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await response.json();
                
                if (data.error) {
                    error.textContent = data.error;
                } else {
                    document.getElementById('mbti-type').textContent = data.mbti_type;
                    const dims = document.getElementById('dimensions');
                    dims.innerHTML = '';
                    const labels = { IE: ['Introvert', 'Extrovert'], NS: ['Intuitive', 'Sensing'], TF: ['Thinking', 'Feeling'], JP: ['Judging', 'Perceiving'] };
                    const colors = { IE: '#667eea', NS: '#28a745', TF: '#fd7e14', JP: '#dc3545' };
                    for (const [dim, conf] of Object.entries(data.confidence)) {
                        const keys = Object.keys(conf);
                        const winner = conf[keys[0]] > conf[keys[1]] ? keys[0] : keys[1];
                        const pct = Math.round(conf[winner] * 100);
                        dims.innerHTML += `
                            <div class="dimension">
                                <span class="dim-label">${labels[dim][0]} / ${labels[dim][1]}</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${pct}%; background: ${colors[dim]};"></div>
                                    <span class="confidence-text">${winner}: ${pct}%</span>
                                </div>
                            </div>`;
                    }
                    result.style.display = 'block';
                }
            } catch (e) {
                error.textContent = 'Failed to connect to server';
            }
            btn.disabled = false;
            btn.textContent = 'Predict My Personality';
        }
    </script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def home():
    """Serve the web demo interface."""
    return render_template_string(HTML_TEMPLATE)


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
