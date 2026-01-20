# Coffee Quality Prediction

**ML Zoomcamp 2024 - Capstone Project 2**

🚀 **Live Demo**: [https://mlzoomcamp2025-production-96cc.up.railway.app/](https://mlzoomcamp2025-production-96cc.up.railway.app/)

A machine learning system that predicts coffee quality scores (Total Cup Points) based on sensory attributes using the Coffee Quality Institute (CQI) dataset.

---

## Evaluation Criteria Checklist

| # | Criteria | Status | Details |
|---|----------|--------|---------|
| 1 | Problem Description | ✅ | See [Problem Description](#problem-description) section |
| 2 | EDA | ✅ | See `notebooks/notebook.ipynb` - includes missing values, distributions, correlations, feature importance |
| 3 | Model Training | ✅ | Trained 4 models (Ridge, Lasso, RandomForest, XGBoost) with hyperparameter tuning |
| 4 | Training Script | ✅ | `train.py` - standalone training script |
| 5 | Reproducibility | ✅ | Dataset included in `data/` folder, notebook runs without errors |
| 6 | Model Deployment | ✅ | Flask API deployed at [Railway](https://mlzoomcamp2025-production-96cc.up.railway.app/) |
| 7 | Dependency Management | ✅ | `requirements.txt` + virtual environment instructions below |
| 8 | Containerization | ✅ | `Dockerfile` with build/run instructions below |

---

## Problem Description

**Context**: Coffee quality assessment is traditionally performed by professional cuppers who evaluate beans on multiple sensory attributes (Aroma, Flavor, Acidity, Body, etc.). Each attribute is scored on a 0-10 scale, and the Total Cup Points (0-100) represents the overall quality.

**Problem**: Manual cupping is time-consuming and requires trained professionals. There's a need for a quick, automated quality estimation tool.

**Solution**: This project builds a regression model that predicts Total Cup Points from 10 sensory attributes, enabling:
- **Coffee buyers**: Quick quality estimation before purchasing
- **Roasters**: Quality control and consistency monitoring  
- **Producers**: Understanding which attributes most impact quality scores

**How it's used**: Users input sensory scores via a web interface or API, and receive an instant quality prediction with a quality grade (Excellent, Very Good, Good, etc.).

---

## Dataset

**Source**: [Coffee Quality Institute (CQI) Dataset on Kaggle](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi)

The dataset is **included in this repository** at `data/`:
- `data/arabica_data_cleaned.csv` - Arabica coffee samples
- `data/robusta_data_cleaned.csv` - Robusta coffee samples
- Combined: ~1,339 professional coffee evaluations

### Features (10 sensory attributes, scale 0-10)

| Feature | Description |
|---------|-------------|
| Aroma | Fragrance/aroma quality |
| Flavor | Overall flavor quality |
| Aftertaste | Aftertaste quality |
| Acidity | Acidity quality |
| Body | Body/mouthfeel quality |
| Balance | Overall balance |
| Uniformity | Cup-to-cup consistency |
| Clean_Cup | Absence of defects |
| Sweetness | Sweetness quality |
| Cupper_Points | Cupper's overall score |

### Target Variable
- **Total_Cup_Points**: Overall quality score (0-100 scale)

---

## EDA Summary

Full EDA is in `notebooks/notebook.ipynb`. Key findings:

1. **Missing Values**: Handled via median imputation for numeric columns
2. **Target Distribution**: Total Cup Points ranges from ~60-90, normally distributed around 82
3. **Correlations**: Flavor (0.85), Aftertaste (0.82), and Balance (0.80) have highest correlation with target
4. **Feature Importance**: XGBoost feature importance shows Flavor, Cupper_Points, and Aftertaste as top predictors
5. **Outliers**: Removed using z-score method (threshold=3)

---

## Model Training

### Models Trained
1. **Ridge Regression** - Linear model with L2 regularization
2. **Lasso Regression** - Linear model with L1 regularization  
3. **Random Forest** - Ensemble of decision trees
4. **XGBoost** - Gradient boosting (best performer)

### Hyperparameter Tuning
- Used GridSearchCV with 5-fold cross-validation
- Tuned parameters: alpha (linear), n_estimators, max_depth, learning_rate (tree-based)

### Model Performance (Test Set)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge | 1.28 | 0.95 | 0.91 |
| Lasso | 1.31 | 0.98 | 0.90 |
| RandomForest | 1.18 | 0.87 | 0.93 |
| **XGBoost** | **1.12** | **0.82** | **0.94** |

**Best Model**: XGBoost with RMSE=1.12, R²=0.94

---

## Installation & Environment Setup

### Prerequisites
- Python 3.11+
- pip

### Setup Virtual Environment

```bash
# Navigate to project directory
cd "Capstone Project 2"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
flask>=3.0.0
gunicorn>=21.0.0
pytest>=7.4.0
hypothesis>=6.90.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Training the Model

The training logic is exported to `train.py`:

```bash
# Activate virtual environment first
python train.py
```

Output:
```
============================================================
Coffee Quality Prediction - Model Training
============================================================
Loading data...
  Loaded 1339 samples
Preprocessing data...
  Using 10 features: ['Aroma', 'Flavor', ...]
Training models...
  Training Ridge Regression...
  Training Lasso Regression...
  Training Random Forest...
  Training XGBoost...
Selecting best model...
  Best model: XGBoost
  Test RMSE: 1.12, R²: 0.94
Saving model to models/model.pkl...
  Model saved successfully!
```

---

## Running the Service Locally

### Option 1: Direct Python

```bash
# Activate virtual environment
python predict.py
```

Service runs at `http://localhost:9696`

### Option 2: With Gunicorn

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

### Web Interface

Open `http://localhost:9696` in your browser for the interactive demo with sliders.

### API Usage

```bash
# Health check
curl http://localhost:9696/health

# Make prediction
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Aroma": 7.5,
    "Flavor": 7.8,
    "Aftertaste": 7.2,
    "Acidity": 7.5,
    "Body": 7.3,
    "Balance": 7.4,
    "Uniformity": 10.0,
    "Clean_Cup": 10.0,
    "Sweetness": 10.0,
    "Cupper_Points": 7.5
  }'
```

Response:
```json
{"prediction": 82.45, "status": "success"}
```

---

## Docker Containerization

### Build the Docker Image

```bash
docker build -t coffee-quality-prediction .
```

### Run the Container

```bash
docker run -p 9696:9696 -e PORT=9696 coffee-quality-prediction
```

### Test the Container

```bash
curl http://localhost:9696/health
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Aroma":7.5,"Flavor":7.8,"Aftertaste":7.2,"Acidity":7.5,"Body":7.3,"Balance":7.4,"Uniformity":10,"Clean_Cup":10,"Sweetness":10,"Cupper_Points":7.5}'
```

---

## Cloud Deployment

**Live URL**: [https://mlzoomcamp2025-production-96cc.up.railway.app/](https://mlzoomcamp2025-production-96cc.up.railway.app/)

Deployed on Railway using Docker. Test the live API:

```bash
# Health check
curl https://mlzoomcamp2025-production-96cc.up.railway.app/health

# Make prediction
curl -X POST https://mlzoomcamp2025-production-96cc.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Aroma": 7.5,
    "Flavor": 7.8,
    "Aftertaste": 7.2,
    "Acidity": 7.5,
    "Body": 7.3,
    "Balance": 7.4,
    "Uniformity": 10.0,
    "Clean_Cup": 10.0,
    "Sweetness": 10.0,
    "Cupper_Points": 7.5
  }'
```

---

## Project Structure

```
Capstone Project 2/
├── data/                       # Dataset (included)
│   ├── arabica_data_cleaned.csv
│   └── robusta_data_cleaned.csv
├── notebooks/
│   └── notebook.ipynb          # EDA and model experimentation
├── src/                        # Source modules
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_encoder.py
│   ├── feature_scaler.py
│   ├── data_splitter.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── model_selector.py
│   ├── model_serializer.py
│   └── prediction_pipeline.py
├── models/
│   └── model.pkl               # Trained model (included)
├── tests/                      # Unit tests
├── train.py                    # Training script
├── predict.py                  # Flask API
├── Dockerfile                  # Container definition
├── requirements.txt            # Dependencies
├── railway.json                # Railway deployment config
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

This project is for educational purposes as part of the ML Zoomcamp course.

## Acknowledgments

- [Coffee Quality Institute](https://www.coffeeinstitute.org/) for the dataset
- [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course
