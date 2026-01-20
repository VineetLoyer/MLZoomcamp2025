# Coffee Quality Prediction

A machine learning system that predicts coffee quality scores (Total Cup Points) based on sensory attributes and bean characteristics using the Coffee Quality Institute (CQI) dataset.

## Problem Description

Coffee quality assessment is traditionally performed by professional cuppers who evaluate beans on multiple sensory attributes. This project builds a predictive model that can estimate the overall quality score (Total Cup Points on a 0-100 scale) from individual sensory measurements, enabling:

- **Coffee buyers**: Quick quality estimation before purchasing
- **Roasters**: Quality control and consistency monitoring
- **Producers**: Understanding which attributes most impact quality scores

### Solution Approach

1. **Data Processing**: Load and clean CQI dataset (Arabica + Robusta samples)
2. **Feature Engineering**: Use sensory attributes (Aroma, Flavor, Acidity, etc.) as predictors
3. **Model Training**: Compare linear models (Ridge, Lasso) and tree-based models (Random Forest, XGBoost)
4. **Model Selection**: Select best model based on validation RMSE
5. **Deployment**: Flask REST API containerized with Docker

## Dataset

The [Coffee Quality Institute (CQI) dataset](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi) contains ~1,300 professional coffee evaluations.

### Features Used

| Feature | Description | Scale |
|---------|-------------|-------|
| Aroma | Fragrance/aroma quality | 0-10 |
| Flavor | Overall flavor quality | 0-10 |
| Aftertaste | Aftertaste quality | 0-10 |
| Acidity | Acidity quality | 0-10 |
| Body | Body/mouthfeel quality | 0-10 |
| Balance | Overall balance | 0-10 |
| Uniformity | Cup-to-cup consistency | 0-10 |
| Clean_Cup | Absence of defects | 0-10 |
| Sweetness | Sweetness quality | 0-10 |
| Cupper_Points | Cupper's overall score | 0-10 |

### Target Variable

- **Total_Cup_Points**: Overall quality score (0-100 scale)

### Download Dataset

```bash
# Option 1: Using Kaggle CLI
pip install kaggle
kaggle datasets download -d volpatto/coffee-quality-database-from-cqi -p data/
unzip data/coffee-quality-database-from-cqi.zip -d data/

# Option 2: Manual download
# Visit https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi
# Download and extract to data/ folder
```

After download, ensure these files exist:
- `data/arabica_data_cleaned.csv`
- `data/robusta_data_cleaned.csv`

## Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Setup

```bash
# Clone or navigate to project directory
cd "Capstone Project 2"

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- pandas, numpy - Data processing
- scikit-learn, xgboost, lightgbm - Machine learning
- flask, gunicorn - Web API
- pytest, hypothesis - Testing
- matplotlib, seaborn - Visualization

## Usage

### Training the Model

Train the model using the provided training script:

```bash
python train.py
```

Options:
```bash
python train.py --data-dir data --model-path models/model.pkl
```

The script will:
1. Load and merge Arabica/Robusta datasets
2. Clean and preprocess data
3. Train Ridge, Lasso, Random Forest, and XGBoost models
4. Select the best model based on validation RMSE
5. Save the model to `models/model.pkl`

Expected output:
```
============================================================
Coffee Quality Prediction - Model Training
============================================================
Loading data...
  Loaded 1339 samples
Preprocessing data...
  Using 10 features: ['Aroma', 'Flavor', ...]
  Training set: 803 samples
  Validation set: 267 samples
  Test set: 269 samples

Training models...
  Training Ridge Regression...
    RMSE: 1.2345, R²: 0.9123
  ...

Selecting best model...
  Best model: XGBoost
  Test set metrics:
    RMSE: 1.1234
    MAE: 0.8765
    R²: 0.9234

Saving model to models/model.pkl...
  Model saved successfully!
```

### Starting the Prediction Service

Run the Flask API locally:

```bash
python predict.py
```

The service starts on port 9696 by default. Configure with environment variables:
```bash
PORT=8080 MODEL_PATH=models/model.pkl python predict.py
```

### Web Interface

Open your browser and navigate to `http://localhost:9696` to access the interactive web demo:

- Use sliders to adjust sensory attributes (Aroma, Flavor, Acidity, etc.)
- Click preset buttons for quick examples (Excellent, Good, Average)
- Click "Predict Quality Score" to get the predicted Total Cup Points
- Results show the score and a quality badge (Excellent, Very Good, Good, etc.)

### Making Predictions

Send a POST request to the `/predict` endpoint:

```bash
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
{
  "prediction": 82.45,
  "status": "success"
}
```

### Health Check

```bash
curl http://localhost:9696/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_names": ["Aroma", "Flavor", "Aftertaste", ...]
}
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t coffee-quality-prediction .
```

### Run the Container

```bash
docker run -p 9696:9696 coffee-quality-prediction
```

### Test the Containerized Service

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

### Docker Compose (Optional)

Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  coffee-prediction:
    build: .
    ports:
      - "9696:9696"
    environment:
      - PORT=9696
      - MODEL_PATH=/app/models/model.pkl
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9696/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up --build
```

## Project Structure

```
Capstone Project 2/
├── data/
│   ├── arabica_data_cleaned.csv
│   ├── robusta_data_cleaned.csv
│   └── README.md
├── notebooks/
│   └── notebook.ipynb          # EDA and model experimentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── data_cleaner.py         # Data cleaning and preprocessing
│   ├── feature_encoder.py      # Categorical encoding
│   ├── feature_scaler.py       # Feature scaling
│   ├── data_splitter.py        # Train/val/test splitting
│   ├── model_trainer.py        # Model training
│   ├── model_evaluator.py      # Model evaluation metrics
│   ├── model_selector.py       # Best model selection
│   ├── model_serializer.py     # Model save/load
│   └── prediction_pipeline.py  # Inference pipeline
├── models/
│   └── model.pkl               # Trained model
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_cleaner.py
│   ├── test_feature_encoder.py
│   ├── test_feature_scaler.py
│   ├── test_model_trainer.py
│   ├── test_model_evaluator.py
│   ├── test_prediction_pipeline.py
│   └── test_api.py
├── train.py                    # Training script
├── predict.py                  # Flask API
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## API Reference

### POST /predict

Predict coffee quality score from sensory attributes.

**Request:**
```json
{
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
}
```

**Response (Success):**
```json
{
  "prediction": 82.45,
  "status": "success"
}
```

**Response (Error):**
```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "Missing required field: Aroma"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_names": ["Aroma", "Flavor", ...]
}
```

## Model Performance

The best model (typically XGBoost or Random Forest) achieves:

| Metric | Value |
|--------|-------|
| RMSE | ~1.1-1.3 |
| MAE | ~0.8-1.0 |
| R² | ~0.92-0.94 |

## Cloud Deployment

The service can be deployed to various cloud platforms. Below are instructions for Railway and Render.

### Deploy to Railway

[Railway](https://railway.app/) offers easy Docker-based deployments with a generous free tier.

#### Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app/)

#### Deployment Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/coffee-quality-prediction.git
   git push -u origin main
   ```

2. **Create a new project on Railway**
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect the `railway.json` configuration

3. **Configure environment variables** (optional)
   - In Railway dashboard, go to your service → Variables
   - Add any custom environment variables if needed:
     - `PORT`: Railway sets this automatically
     - `MODEL_PATH`: `/app/models/model.pkl` (default)

4. **Deploy**
   - Railway will automatically build and deploy your Docker container
   - Once deployed, click "Generate Domain" to get a public URL

5. **Test your deployment**
   ```bash
   # Replace with your Railway URL
   curl https://your-app.railway.app/health
   
   curl -X POST https://your-app.railway.app/predict \
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

### Deploy to Render

[Render](https://render.com/) provides free web service hosting with Docker support.

#### Prerequisites
- GitHub account
- Render account (sign up at https://render.com/)

#### Deployment Steps

1. **Push your code to GitHub** (if not already done)

2. **Create a new Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select "Docker" as the environment

3. **Configure the service**
   - Name: `coffee-quality-prediction`
   - Region: Choose closest to your users
   - Branch: `main`
   - Plan: Free (or paid for better performance)

4. **Set environment variables**
   - `PORT`: `9696`
   - `MODEL_PATH`: `/app/models/model.pkl`

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your Docker container
   - Your service URL will be: `https://coffee-quality-prediction.onrender.com`

6. **Test your deployment**
   ```bash
   curl https://coffee-quality-prediction.onrender.com/health
   ```

### Deploy to AWS (Elastic Beanstalk)

For production deployments, AWS Elastic Beanstalk provides scalable hosting.

#### Prerequisites
- AWS account
- AWS CLI installed and configured
- EB CLI installed (`pip install awsebcli`)

#### Deployment Steps

1. **Initialize Elastic Beanstalk**
   ```bash
   cd "Capstone Project 2"
   eb init -p docker coffee-quality-prediction
   ```

2. **Create environment and deploy**
   ```bash
   eb create coffee-quality-env
   ```

3. **Open your application**
   ```bash
   eb open
   ```

4. **View logs if needed**
   ```bash
   eb logs
   ```

### Deployment Configuration Files

The project includes configuration files for multiple platforms:

| File | Platform | Description |
|------|----------|-------------|
| `railway.json` | Railway | Railway deployment configuration |
| `render.yaml` | Render | Render Blueprint specification |
| `Procfile` | Heroku/Render | Process file for web dynos |
| `Dockerfile` | All | Docker container definition |

### Monitoring Your Deployment

After deployment, monitor your service:

1. **Health Check**: Access `/health` endpoint to verify service status
2. **Logs**: Check platform-specific logs for errors
3. **Metrics**: Monitor response times and error rates

### Troubleshooting

Common deployment issues:

1. **Model file not found**
   - Ensure `models/model.pkl` exists and is committed to git
   - Check `MODEL_PATH` environment variable

2. **Port binding errors**
   - Cloud platforms set `PORT` automatically
   - Don't hardcode port numbers in production

3. **Memory issues**
   - XGBoost/LightGBM models can be memory-intensive
   - Consider upgrading to a paid tier if needed

4. **Slow cold starts**
   - First request after idle period may be slow
   - Use health checks to keep service warm

## License

This project is for educational purposes as part of the ML Zoomcamp course.

## Acknowledgments

- [Coffee Quality Institute](https://www.coffeeinstitute.org/) for the dataset
- [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course
