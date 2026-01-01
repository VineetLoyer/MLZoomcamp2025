# MBTI Personality Prediction from Text

An NLP-based machine learning system that predicts a person's MBTI (Myers-Briggs Type Indicator) personality type from their written text.

## Problem Description

The Myers-Briggs Type Indicator (MBTI) is a popular personality assessment that classifies individuals into 16 personality types based on four dimensions:
- **I/E**: Introversion vs Extroversion
- **N/S**: Intuition vs Sensing
- **T/F**: Thinking vs Feeling
- **J/P**: Judging vs Perceiving

This project builds a machine learning model that analyzes linguistic patterns, word choices, and writing styles to predict MBTI personality types from text data.

## Solution Approach

We use a multi-label binary classification approach with 4 separate classifiers (one for each MBTI dimension) rather than a single 16-class classifier. This approach typically yields better results due to reduced class imbalance.

### Pipeline Overview
1. **Data Preprocessing**: Clean text by removing URLs, special characters, MBTI mentions
2. **Feature Extraction**: TF-IDF vectorization with n-grams
3. **Model Training**: Train binary classifiers for each dimension
4. **Prediction Service**: Flask-based REST API for serving predictions

## Project Structure

```
Capstone Project 1/
├── data/               # Dataset files
├── models/             # Trained model artifacts
├── notebooks/          # Jupyter notebooks for EDA
├── src/                # Source code
│   ├── __init__.py
│   ├── preprocessor.py # Text preprocessing
│   ├── feature_extractor.py
│   ├── classifier.py   # MBTI classifier
│   └── utils.py        # Utility functions
├── tests/              # Test files
├── train.py            # Training script
├── predict.py          # Flask prediction service
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.9+
- pip or uv package manager

### Setup

1. Clone the repository and navigate to the project folder:
```bash
cd "Capstone Project 1"
```

2. Create a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/macOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the MBTI dataset from Kaggle. See [Data Download Instructions](#data-download-instructions) below.

### Data Download Instructions

1. Go to [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
2. Download `mbti_1.csv`
3. Place the file in the `data/` folder

## Usage

### Training the Model

Run the training script to train the model:
```bash
python train.py
```

This will:
- Load and preprocess the data
- Train classifiers for each MBTI dimension
- Save model artifacts to `models/`

### Running the Prediction Service

Start the Flask prediction service:
```bash
python predict.py
```

The service will be available at `http://localhost:9696`

### API Endpoints

**POST /predict**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love spending time alone reading books and thinking about abstract concepts..."}'
```

Response:
```json
{
  "mbti_type": "INTJ",
  "confidence": {
    "IE": {"I": 0.75, "E": 0.25},
    "NS": {"N": 0.82, "S": 0.18},
    "TF": {"T": 0.68, "F": 0.32},
    "JP": {"J": 0.71, "P": 0.29}
  }
}
```

**GET /health**
```bash
curl http://localhost:9696/health
```

### Running Tests

```bash
pytest tests/ -v
```

## Docker

The prediction service can be containerized for consistent deployment across environments.

### Prerequisites

- Docker installed and running
- Trained model artifacts in `models/` directory

### Build the Container

```bash
docker build -t mbti-predictor .
```

This will:
- Use Python 3.11 slim base image
- Install all required dependencies
- Download necessary NLTK data
- Copy source code and trained model artifacts
- Configure gunicorn as the production WSGI server

### Run the Container

Run the container in detached mode:
```bash
docker run -d --name mbti-service -p 9696:9696 mbti-predictor
```

Or run interactively to see logs:
```bash
docker run -p 9696:9696 mbti-predictor
```

The service will be available at `http://localhost:9696`

### Test the Container

Check if the service is healthy:
```bash
curl http://localhost:9696/health
```

Expected response:
```json
{"status": "healthy"}
```

Make a prediction:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love spending time alone reading books and thinking about abstract concepts. I prefer deep conversations over small talk and enjoy analyzing complex problems."}'
```

Expected response:
```json
{
  "mbti_type": "INTP",
  "confidence": {
    "IE": {"I": 0.82, "E": 0.18},
    "NS": {"N": 0.66, "S": 0.34},
    "TF": {"T": 0.73, "F": 0.27},
    "JP": {"J": 0.29, "P": 0.71}
  }
}
```

### Container Management

Stop the container:
```bash
docker stop mbti-service
```

Remove the container:
```bash
docker rm mbti-service
```

View container logs:
```bash
docker logs mbti-service
```

### Environment Variables

The container supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 9696 | Port the service listens on |
| `MODELS_DIR` | /app/models | Directory containing model artifacts |

Example with custom port:
```bash
docker run -d -p 8080:8080 -e PORT=8080 mbti-predictor
```

## Cloud Deployment (Railway)

The service can be deployed to [Railway](https://railway.app/) for cloud hosting.

### Prerequisites

- Railway account (sign up at https://railway.app/)
- Railway CLI installed: `npm install -g @railway/cli`
- Git repository with your project

### Deployment Steps

1. **Login to Railway CLI:**
```bash
railway login
```

2. **Initialize a new Railway project:**
```bash
railway init
```

3. **Deploy the service:**
```bash
railway up
```

4. **Generate a public URL:**
```bash
railway domain
```

Railway will automatically:
- Detect the Dockerfile
- Build the container
- Deploy and expose the service
- Provide a public URL (e.g., `https://mbti-predictor-production.up.railway.app`)

### Testing the Deployed Service

Once deployed, test the service using the Railway-provided URL:

```bash
# Health check
curl https://your-app.up.railway.app/health

# Make a prediction
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love spending time alone reading books and thinking about abstract concepts."}'
```

### Environment Variables

Set environment variables in Railway dashboard or via CLI:
```bash
railway variables set PORT=9696
railway variables set MODELS_DIR=/app/models
```

### Monitoring

View logs in Railway dashboard or via CLI:
```bash
railway logs
```

## Model Performance

Test set performance metrics for each MBTI dimension:

| Dimension | Accuracy | F1-Score | Model |
|-----------|----------|----------|-------|
| I/E (Introvert/Extrovert) | 78.1% | 0.565 | Logistic Regression |
| N/S (Intuitive/Sensing) | 82.7% | 0.463 | Logistic Regression |
| T/F (Thinking/Feeling) | 81.7% | 0.828 | Logistic Regression |
| J/P (Judging/Perceiving) | 71.2% | 0.786 | Logistic Regression |

The model uses balanced class weights to handle class imbalance in the dataset.

## Features

- Text preprocessing with URL removal, MBTI mention filtering, and normalization
- TF-IDF feature extraction with n-gram support
- Multi-label binary classification (4 separate classifiers)
- REST API with confidence scores for each dimension
- Docker containerization for easy deployment
- Comprehensive test suite with unit tests

## Acknowledgments

- Dataset: [MBTI Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type) by Mitchell J
- Course: [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club

## License

This project is for educational purposes as part of the ML Zoomcamp Capstone Project.
