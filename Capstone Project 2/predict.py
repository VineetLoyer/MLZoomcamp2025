#!/usr/bin/env python
"""Flask API for Coffee Quality Prediction."""

import os
import sys

from flask import Flask, request, jsonify, render_template_string

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.prediction_pipeline import (
    PredictionPipeline,
    ModelNotLoadedError,
    InvalidInputError,
)

MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model.pkl')
)

app = Flask(__name__)
pipeline = None


def get_pipeline() -> PredictionPipeline:
    """Get or initialize the prediction pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = PredictionPipeline(MODEL_PATH)
    return pipeline


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        p = get_pipeline()
        return jsonify({
            'status': 'healthy',
            'model_loaded': p.is_loaded,
            'feature_names': p.feature_names if p.is_loaded else None
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Predict coffee quality from input features."""
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_CONTENT_TYPE',
            'message': 'Content-Type must be application/json'
        }), 400
    
    try:
        features = request.get_json()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_JSON',
            'message': f'Invalid JSON: {str(e)}'
        }), 400
    
    if not isinstance(features, dict):
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_INPUT',
            'message': 'Request body must be a JSON object with feature values'
        }), 400
    
    if not features:
        return jsonify({
            'status': 'error',
            'error_code': 'EMPTY_INPUT',
            'message': 'Request body cannot be empty'
        }), 400
    
    try:
        p = get_pipeline()
        prediction = p.predict(features)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        }), 200
        
    except ModelNotLoadedError as e:
        return jsonify({
            'status': 'error',
            'error_code': 'MODEL_NOT_LOADED',
            'message': str(e)
        }), 503
        
    except InvalidInputError as e:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_INPUT',
            'message': str(e)
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_code': 'PREDICTION_ERROR',
            'message': f'Prediction failed: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'error_code': 'NOT_FOUND',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'status': 'error',
        'error_code': 'METHOD_NOT_ALLOWED',
        'message': 'Method not allowed for this endpoint'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'status': 'error',
        'error_code': 'INTERNAL_ERROR',
        'message': 'Internal server error'
    }), 500


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coffee Quality Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #6F4E37 0%, #3E2723 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 10px; font-size: 2em; }
        .subtitle { color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 30px; }
        .card { background: white; border-radius: 16px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
        .section-title { font-size: 18px; font-weight: 600; color: #6F4E37; margin-bottom: 20px; border-bottom: 2px solid #D7CCC8; padding-bottom: 10px; }
        .features-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 25px; }
        @media (max-width: 600px) { .features-grid { grid-template-columns: 1fr; } }
        .feature-item { }
        .feature-label { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .feature-name { font-weight: 500; color: #5D4037; }
        .feature-value { font-weight: 600; color: #6F4E37; min-width: 40px; text-align: right; }
        input[type="range"] { 
            width: 100%; height: 8px; border-radius: 4px; background: #D7CCC8;
            -webkit-appearance: none; appearance: none; cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%;
            background: #6F4E37; cursor: pointer; border: 2px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        input[type="range"]::-moz-range-thumb {
            width: 20px; height: 20px; border-radius: 50%;
            background: #6F4E37; cursor: pointer; border: 2px solid white;
        }
        button { 
            width: 100%; padding: 15px; background: linear-gradient(135deg, #6F4E37 0%, #5D4037 100%);
            color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s; margin-top: 10px;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(111,78,55,0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result { margin-top: 25px; padding: 25px; background: linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%); border-radius: 12px; display: none; text-align: center; }
        .score-container { margin-bottom: 20px; }
        .score-label { font-size: 16px; color: #5D4037; margin-bottom: 10px; }
        .score { font-size: 72px; font-weight: bold; color: #6F4E37; }
        .score-max { font-size: 24px; color: #8D6E63; }
        .quality-badge { display: inline-block; padding: 8px 20px; border-radius: 20px; font-weight: 600; font-size: 16px; margin-top: 10px; }
        .quality-excellent { background: #4CAF50; color: white; }
        .quality-good { background: #8BC34A; color: white; }
        .quality-average { background: #FFC107; color: #333; }
        .quality-below { background: #FF9800; color: white; }
        .quality-poor { background: #f44336; color: white; }
        .error { color: #dc3545; text-align: center; margin-top: 15px; padding: 10px; background: #ffebee; border-radius: 8px; }
        .footer { text-align: center; margin-top: 20px; color: rgba(255,255,255,0.7); font-size: 14px; }
        .footer a { color: #FFCC80; }
        .info-text { font-size: 13px; color: #8D6E63; margin-top: 5px; text-align: center; }
        .preset-buttons { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .preset-btn { 
            flex: 1; min-width: 120px; padding: 10px; background: #EFEBE9; border: 2px solid #D7CCC8;
            border-radius: 8px; cursor: pointer; font-size: 14px; color: #5D4037;
            transition: all 0.2s;
        }
        .preset-btn:hover { background: #D7CCC8; border-color: #6F4E37; }
    </style>
</head>
<body>
    <div class="container">
        <h1>☕ Coffee Quality Predictor</h1>
        <p class="subtitle">Predict Total Cup Points from sensory attributes</p>
        <div class="card">
            <div class="section-title">Quick Presets</div>
            <div class="preset-buttons">
                <button type="button" class="preset-btn" onclick="setPreset('excellent')">🏆 Excellent</button>
                <button type="button" class="preset-btn" onclick="setPreset('good')">👍 Good</button>
                <button type="button" class="preset-btn" onclick="setPreset('average')">📊 Average</button>
                <button type="button" class="preset-btn" onclick="setPreset('reset')">🔄 Reset</button>
            </div>
            
            <div class="section-title">Sensory Attributes (0-10 scale)</div>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">🌸 Aroma</span>
                        <span class="feature-value" id="aroma-val">7.5</span>
                    </div>
                    <input type="range" id="aroma" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('aroma')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">😋 Flavor</span>
                        <span class="feature-value" id="flavor-val">7.5</span>
                    </div>
                    <input type="range" id="flavor" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('flavor')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">✨ Aftertaste</span>
                        <span class="feature-value" id="aftertaste-val">7.5</span>
                    </div>
                    <input type="range" id="aftertaste" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('aftertaste')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">🍋 Acidity</span>
                        <span class="feature-value" id="acidity-val">7.5</span>
                    </div>
                    <input type="range" id="acidity" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('acidity')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">💪 Body</span>
                        <span class="feature-value" id="body-val">7.5</span>
                    </div>
                    <input type="range" id="body" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('body')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">⚖️ Balance</span>
                        <span class="feature-value" id="balance-val">7.5</span>
                    </div>
                    <input type="range" id="balance" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('balance')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">🎯 Uniformity</span>
                        <span class="feature-value" id="uniformity-val">10.0</span>
                    </div>
                    <input type="range" id="uniformity" min="0" max="10" step="0.1" value="10.0" oninput="updateValue('uniformity')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">💎 Clean Cup</span>
                        <span class="feature-value" id="clean_cup-val">10.0</span>
                    </div>
                    <input type="range" id="clean_cup" min="0" max="10" step="0.1" value="10.0" oninput="updateValue('clean_cup')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">🍯 Sweetness</span>
                        <span class="feature-value" id="sweetness-val">10.0</span>
                    </div>
                    <input type="range" id="sweetness" min="0" max="10" step="0.1" value="10.0" oninput="updateValue('sweetness')">
                </div>
                <div class="feature-item">
                    <div class="feature-label">
                        <span class="feature-name">👨‍🍳 Cupper Points</span>
                        <span class="feature-value" id="cupper_points-val">7.5</span>
                    </div>
                    <input type="range" id="cupper_points" min="0" max="10" step="0.1" value="7.5" oninput="updateValue('cupper_points')">
                </div>
            </div>
            
            <button id="predict-btn" onclick="predict()">☕ Predict Quality Score</button>
            <p class="info-text">Total Cup Points range: 0-100 (sum of all attributes)</p>
            
            <div id="error" class="error" style="display: none;"></div>
            <div id="result" class="result">
                <div class="score-container">
                    <div class="score-label">Predicted Total Cup Points</div>
                    <span id="score" class="score">0</span><span class="score-max">/100</span>
                </div>
                <div id="quality-badge" class="quality-badge"></div>
            </div>
        </div>
        <p class="footer">Built for <a href="https://github.com/DataTalksClub/machine-learning-zoomcamp" target="_blank">ML Zoomcamp</a> Capstone Project</p>
    </div>
    <script>
        const features = ['aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance', 'uniformity', 'clean_cup', 'sweetness', 'cupper_points'];
        
        function updateValue(id) {
            const val = parseFloat(document.getElementById(id).value).toFixed(1);
            document.getElementById(id + '-val').textContent = val;
        }
        
        function setPreset(type) {
            const presets = {
                excellent: { aroma: 8.5, flavor: 8.5, aftertaste: 8.2, acidity: 8.3, body: 8.0, balance: 8.2, uniformity: 10, clean_cup: 10, sweetness: 10, cupper_points: 8.5 },
                good: { aroma: 7.5, flavor: 7.5, aftertaste: 7.2, acidity: 7.3, body: 7.2, balance: 7.3, uniformity: 10, clean_cup: 10, sweetness: 10, cupper_points: 7.5 },
                average: { aroma: 6.5, flavor: 6.5, aftertaste: 6.2, acidity: 6.3, body: 6.2, balance: 6.3, uniformity: 8, clean_cup: 8, sweetness: 8, cupper_points: 6.5 },
                reset: { aroma: 7.5, flavor: 7.5, aftertaste: 7.5, acidity: 7.5, body: 7.5, balance: 7.5, uniformity: 10, clean_cup: 10, sweetness: 10, cupper_points: 7.5 }
            };
            const vals = presets[type];
            for (const [key, val] of Object.entries(vals)) {
                document.getElementById(key).value = val;
                document.getElementById(key + '-val').textContent = val.toFixed(1);
            }
        }
        
        function getQualityBadge(score) {
            if (score >= 85) return { text: '🏆 Excellent (Specialty)', class: 'quality-excellent' };
            if (score >= 80) return { text: '👍 Very Good', class: 'quality-good' };
            if (score >= 75) return { text: '📊 Good', class: 'quality-average' };
            if (score >= 70) return { text: '⚠️ Below Average', class: 'quality-below' };
            return { text: '❌ Poor', class: 'quality-poor' };
        }
        
        async function predict() {
            const btn = document.getElementById('predict-btn');
            const error = document.getElementById('error');
            const result = document.getElementById('result');
            
            error.style.display = 'none';
            result.style.display = 'none';
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            const data = {
                Aroma: parseFloat(document.getElementById('aroma').value),
                Flavor: parseFloat(document.getElementById('flavor').value),
                Aftertaste: parseFloat(document.getElementById('aftertaste').value),
                Acidity: parseFloat(document.getElementById('acidity').value),
                Body: parseFloat(document.getElementById('body').value),
                Balance: parseFloat(document.getElementById('balance').value),
                Uniformity: parseFloat(document.getElementById('uniformity').value),
                Clean_Cup: parseFloat(document.getElementById('clean_cup').value),
                Sweetness: parseFloat(document.getElementById('sweetness').value),
                Cupper_Points: parseFloat(document.getElementById('cupper_points').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const res = await response.json();
                
                if (res.status === 'error') {
                    error.textContent = res.message;
                    error.style.display = 'block';
                } else {
                    const score = res.prediction;
                    document.getElementById('score').textContent = score.toFixed(1);
                    const badge = getQualityBadge(score);
                    const badgeEl = document.getElementById('quality-badge');
                    badgeEl.textContent = badge.text;
                    badgeEl.className = 'quality-badge ' + badge.class;
                    result.style.display = 'block';
                }
            } catch (e) {
                error.textContent = 'Failed to connect to server';
                error.style.display = 'block';
            }
            btn.disabled = false;
            btn.textContent = '☕ Predict Quality Score';
        }
    </script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def home():
    """Serve the web demo interface."""
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9696))
    
    print(f"Starting Coffee Quality Prediction API on port {port}...")
    print(f"Model path: {MODEL_PATH}")
    
    try:
        p = get_pipeline()
        print(f"Model loaded successfully. Features: {p.feature_names}")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
