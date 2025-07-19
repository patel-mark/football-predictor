from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
import logging
import tempfile
import shutil
import requests
from src.predict import predict_fixtures
from src.data_ingestion import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("football-predictor-api")

# Heroku-compatible paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Create temporary directory for Heroku's ephemeral filesystem
TEMP_DIR = Path(tempfile.mkdtemp(prefix="football-predictor-"))

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Public URL for model files (replace with your actual URL)
MODEL_URL = "https://github.com/patel-mark/football-predictor/raw/main/models/xgb_artifacts.pkl"

app = FastAPI(
    title="Football Match Predictor API",
    description="API for predicting football match outcomes and expected goals (xG)",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Predictions",
            "description": "Endpoints for match predictions"
        },
        {
            "name": "Monitoring",
            "description": "Health checks and system monitoring"
        }
    ]
)

class FixtureRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResult(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_home_xg: float
    predicted_away_xg: float
    total_goals: int
    confidence: str

def download_model(url: str, save_path: Path):
    """Download model from URL if not exists"""
    try:
        if not save_path.exists():
            logger.info(f"Downloading model from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Model saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Model download failed: {str(e)}")
        return False

@app.on_event("startup")
async def load_artifacts():
    """Load model artifacts on startup with download fallback"""
    try:
        logger.info("Loading model artifacts...")
        model_path = MODEL_DIR / "xgb_artifacts.pkl"
        
        # Download model if missing
        if not model_path.exists():
            logger.warning("Model file not found locally")
            if not download_model(MODEL_URL, model_path):
                raise FileNotFoundError("Model download failed")
                
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
            
        app.state.artifacts = joblib.load(model_path)
        logger.info("Artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        # Set to None but don't crash - will fail on prediction instead
        app.state.artifacts = None

@app.on_event("shutdown")
def cleanup_tempdir():
    """Cleanup temporary directory on shutdown"""
    try:
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logger.warning(f"Temp directory cleanup failed: {str(e)}")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Service health check"""
    model_loaded = hasattr(app.state, "artifacts") and app.state.artifacts is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "message": "Football predictor API is running",
        "model_loaded": model_loaded,
        "temp_dir": str(TEMP_DIR),
        "base_dir": str(BASE_DIR)
    }

@app.post("/predict/single", response_model=PredictionResult, tags=["Predictions"])
def predict_single_fixture(fixture: FixtureRequest):
    """
    Predict outcome for a single fixture
    
    - **home_team**: Home team name (e.g., "Arsenal")
    - **away_team**: Away team name (e.g., "Chelsea")
    """
    try:
        # Verify model is loaded
        if not hasattr(app.state, "artifacts") or app.state.artifacts is None:
            raise RuntimeError("Model not loaded - cannot make predictions")
            
        # Create in-memory DataFrame
        fixtures_df = pd.DataFrame({
            'Home_Team': [fixture.home_team],
            'Away_Team': [fixture.away_team]
        })
        
        # Create temp file path in Heroku-compatible temp dir
        temp_path = TEMP_DIR / 'predict_request.csv'
        fixtures_df.to_csv(temp_path, index=False)
        
        # Get prediction
        prediction = predict_fixtures(str(temp_path)).iloc[0]
        
        # Determine confidence level
        prob_diff = abs(prediction['Home_Win_Prob'] - prediction['Away_Win_Prob'])
        confidence = "High" if prob_diff > 0.3 else "Medium" if prob_diff > 0.15 else "Low"
        
        return {
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "home_win_prob": prediction['Home_Win_Prob'],
            "draw_prob": prediction['Draw_Prob'],
            "away_win_prob": prediction['Away_Win_Prob'],
            "predicted_home_xg": prediction['Predicted_Home_xG'],
            "predicted_away_xg": prediction['Predicted_Away_xG'],
            "total_goals": int(prediction['Total_Goals']),
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Predictions"])
async def predict_batch_fixtures(file: UploadFile = File(...)):
    """
    Predict outcomes for multiple fixtures from a CSV file
    
    The CSV file must contain columns:
    - Home_Team
    - Away_Team
    """
    try:
        # Verify model is loaded
        if not hasattr(app.state, "artifacts") or app.state.artifacts is None:
            raise RuntimeError("Model not loaded - cannot make predictions")
            
        # Create temp file path
        temp_path = TEMP_DIR / file.filename
        
        # Save uploaded file directly to temp location
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get predictions
        predictions_df = predict_fixtures(str(temp_path))
        
        # Format response
        results = []
        for _, row in predictions_df.iterrows():
            prob_diff = abs(row['Home_Win_Prob'] - row['Away_Win_Prob'])
            confidence = "High" if prob_diff > 0.3 else "Medium" if prob_diff > 0.15 else "Low"
            
            results.append({
                "home_team": row['Home_Team'],
                "away_team": row['Away_Team'],
                "home_win_prob": row['Home_Win_Prob'],
                "draw_prob": row['Draw_Prob'],
                "away_win_prob": row['Away_Win_Prob'],
                "predicted_home_xg": row['Predicted_Home_xG'],
                "predicted_away_xg": row['Predicted_Away_xG'],
                "total_goals": int(row['Total_Goals']),
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/predict/upcoming", response_model=List[PredictionResult], tags=["Predictions"])
def predict_upcoming_fixtures():
    """Predict upcoming fixtures from the predefined fixtures.csv file"""
    try:
        # Verify model is loaded
        if not hasattr(app.state, "artifacts") or app.state.artifacts is None:
            raise RuntimeError("Model not loaded - cannot make predictions")
            
        # Get config to find new fixtures path
        *_, config = load_data()
        fixtures_path = BASE_DIR / "data" / "raw" / "fixtures.csv"
        
        if not fixtures_path.exists():
            logger.warning(f"Fixtures file not found at {fixtures_path}, using default")
            fixtures_path = BASE_DIR / "fixtures.csv"
            
        predictions_df = predict_fixtures(str(fixtures_path))
        
        # Format response
        results = []
        for _, row in predictions_df.iterrows():
            prob_diff = abs(row['Home_Win_Prob'] - row['Away_Win_Prob'])
            confidence = "High" if prob_diff > 0.3 else "Medium" if prob_diff > 0.15 else "Low"
            
            results.append({
                "home_team": row['Home_Team'],
                "away_team": row['Away_Team'],
                "home_win_prob": row['Home_Win_Prob'],
                "draw_prob": row['Draw_Prob'],
                "away_win_prob": row['Away_Win_Prob'],
                "predicted_home_xg": row['Predicted_Home_xG'],
                "predicted_away_xg": row['Predicted_Away_xG'],
                "total_goals": int(row['Total_Goals']),
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        logger.error(f"Upcoming fixtures prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fixtures processing error: {str(e)}")

@app.get("/")
def root():
    model_loaded = hasattr(app.state, "artifacts") and app.state.artifacts is not None
    return {
        "message": "Football Predictor API",
        "status": "operational" if model_loaded else "degraded - model missing",
        "docs": "/docs",
        "health_check": "/health",
        "endpoints": {
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "upcoming_fixtures": "/predict/upcoming"
        }
    }