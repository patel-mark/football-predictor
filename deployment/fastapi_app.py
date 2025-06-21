from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
import logging
from src.predict import predict_fixtures
from src.data_ingestion import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("football-predictor-api")

# Set base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.getenv('DATA_DIR', str(BASE_DIR / 'data'))
TEMP_DIR = os.path.join(DATA_DIR, 'temp')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.on_event("startup")
async def load_artifacts():
    """Load model artifacts on startup"""
    try:
        # Load artifacts (models will be loaded on demand during prediction)
        logger.info("Loading model artifacts...")
        app.state.artifacts = joblib.load('models/xgb_artifacts.pkl')
        logger.info("Artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        raise RuntimeError("Model artifacts could not be loaded")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "message": "Football predictor API is running",
        "model_loaded": hasattr(app.state, "artifacts"),
        "directories": {
            "data": DATA_DIR,
            "temp": TEMP_DIR,
            "uploads": UPLOAD_DIR
        }
    }

@app.post("/predict/single", response_model=PredictionResult, tags=["Predictions"])
def predict_single_fixture(fixture: FixtureRequest):
    """
    Predict outcome for a single fixture
    
    - **home_team**: Home team name (e.g., "Arsenal")
    - **away_team**: Away team name (e.g., "Chelsea")
    """
    try:
        # Create temp file path
        temp_path = os.path.join(TEMP_DIR, 'predict_request.csv')
        
        # Create a temporary DataFrame
        fixtures_df = pd.DataFrame({
            'Home_Team': [fixture.home_team],
            'Away_Team': [fixture.away_team]
        })
        
        # Save to temp CSV
        fixtures_df.to_csv(temp_path, index=False)
        
        # Get prediction
        prediction = predict_fixtures(temp_path).iloc[0]
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Predictions"])
async def predict_batch_fixtures(file: UploadFile = File(...)):
    """
    Predict outcomes for multiple fixtures from a CSV file
    
    The CSV file must contain columns:
    - Home_Team
    - Away_Team
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get predictions
        predictions_df = predict_fixtures(file_path)
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/upcoming", response_model=List[PredictionResult], tags=["Predictions"])
def predict_upcoming_fixtures():
    """Predict upcoming fixtures from the predefined fixtures.csv file"""
    try:
        # Get config to find new fixtures path
        *_, config = load_data()
        predictions_df = predict_fixtures(config['data_paths']['new_fixtures'])
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Football Predictor API",
        "docs": "/docs",
        "health_check": "/health",
        "endpoints": {
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "upcoming_fixtures": "/predict/upcoming"
        }
    }