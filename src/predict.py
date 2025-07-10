import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from .utils import standardize_team_names
from .feature_engineering import create_feature_vector
from .data_ingestion import load_data

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent

def predict_fixtures(fixtures_csv_path, model_dir=None):
    """
    Predict fixtures from CSV file
    - Uses absolute paths for Heroku compatibility
    - Handles missing directories
    - Removed file saving (ephemeral filesystem)
    """
    # Set default model directory
    if model_dir is None:
        model_dir = BASE_DIR / "models"
    
    # Verify model directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load artifacts and config
    artifacts_path = model_dir / "xgb_artifacts.pkl"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    
    artifacts = joblib.load(artifacts_path)
    team_stats = artifacts['team_stats']
    feature_columns = artifacts['feature_columns']
    
    # Load config (only need config, ignore other returns)
    *_, config = load_data()  # Correct unpacking for 4 values
    
    # Load all fold models
    models = []
    for i in range(5):
        try:
            model_path = model_dir / f"xgb_model_fold{i+1}.pkl"
            if not model_path.exists():
                print(f"Warning: Model not found: {model_path}")
                continue
            model = joblib.load(model_path)
            models.append(model)
        except Exception as e:
            print(f"Error loading model fold {i+1}: {str(e)}")
            continue
            
    if not models:
        raise ValueError("No valid models found in the models directory")
    
    # Prepare new fixtures
    if not os.path.exists(fixtures_csv_path):
        raise FileNotFoundError(f"Fixtures CSV not found: {fixtures_csv_path}")
    
    new_fixtures = pd.read_csv(fixtures_csv_path)
    
    # Standardize team names using config mapping
    team_mapping = config['team_mapping']
    new_fixtures = standardize_team_names(new_fixtures, 'Home_Team', team_mapping)
    new_fixtures = standardize_team_names(new_fixtures, 'Away_Team', team_mapping)

    # Create features
    new_fixtures['features'] = new_fixtures.apply(
        lambda row: create_feature_vector(row, team_stats, feature_columns),
        axis=1
    )
    X_new = np.array(new_fixtures['features'].tolist())

    # Ensemble predictions
    class_probs = np.zeros((X_new.shape[0], 3))
    home_xg = np.zeros(X_new.shape[0])
    away_xg = np.zeros(X_new.shape[0])

    for model in models:
        class_probs += model['classifier'].predict_proba(X_new)
        home_xg += model['regressor_home'].predict(X_new)
        away_xg += model['regressor_away'].predict(X_new)

    num_models = len(models)
    result_df = pd.DataFrame({
        'Home_Team': new_fixtures['Home_Team'],
        'Away_Team': new_fixtures['Away_Team'],
        'Home_Win_Prob': class_probs[:, 2] / num_models,
        'Draw_Prob': class_probs[:, 1] / num_models,
        'Away_Win_Prob': class_probs[:, 0] / num_models,
        'Predicted_Home_xG': home_xg / num_models,
        'Predicted_Away_xG': away_xg / num_models
    })

    # Add calculated columns
    result_df['Total_Goals'] = (np.round(result_df['Predicted_Home_xG']) + 
                              np.round(result_df['Predicted_Away_xG']))
    result_df['Prob_Diff'] = abs(result_df['Home_Win_Prob'] - result_df['Away_Win_Prob'])
    
    # Sort by probability difference
    result_df = result_df.sort_values('Prob_Diff', ascending=False)

    # Removed file saving - Heroku has ephemeral filesystem
    print("Predictions generated successfully")
    
    return result_df

if __name__ == "__main__":
    # Demo prediction with fallback paths
    try:
        fixtures_path = BASE_DIR / "data" / "raw" / "fixtures.csv"
        if not fixtures_path.exists():
            fixtures_path = BASE_DIR / "fixtures.csv"
            
        predictions = predict_fixtures(str(fixtures_path))
        print(predictions.head())
    except Exception as e:
        print(f"Prediction failed: {str(e)}")