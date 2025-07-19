import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from .utils import standardize_team_names
from .feature_engineering import create_feature_vector
from .data_ingestion import load_data

BASE_DIR = Path(__file__).resolve().parent.parent

def predict_fixtures(fixtures_csv_path, model_dir=None):
    """Predict fixtures from CSV file with flexible model loading"""
    # Set model directory
    model_dir = model_dir or BASE_DIR / "models"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load artifacts
    artifacts_path = model_dir / "xgb_artifacts.pkl"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    
    try:
        artifacts = joblib.load(artifacts_path)
        team_stats = artifacts['team_stats']
        feature_columns = artifacts['feature_columns']
    except Exception as e:
        raise ValueError(f"Failed to load artifacts: {str(e)}")

    # Load config
    try:
        *_, config = load_data()
        team_mapping = config['team_mapping']
    except Exception as e:
        raise ValueError(f"Failed to load config: {str(e)}")

    # Load models - try multiple approaches
    models = []
    
    # Approach 1: Check for models in artifacts
    if 'model' in artifacts:
        models.append(artifacts['model'])
    elif all(k in artifacts for k in ['classifier', 'regressor_home', 'regressor_away']):
        models.append({
            'classifier': artifacts['classifier'],
            'regressor_home': artifacts['regressor_home'],
            'regressor_away': artifacts['regressor_away']
        })
    
    # Approach 2: Load individual fold models from separate files
    if not models:
        for i in range(1, 6):
            model_path = model_dir / f"xgb_model_fold{i}.pkl"
            if model_path.exists():
                try:
                    models.append(joblib.load(model_path))
                    print(f"Successfully loaded model from: {model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load {model_path}: {str(e)}")

    if not models:
        raise ValueError("No valid models found in artifacts or model files")

    # Load and process fixtures
    try:
        new_fixtures = pd.read_csv(fixtures_csv_path)
        new_fixtures = standardize_team_names(new_fixtures, 'Home_Team', team_mapping)
        new_fixtures = standardize_team_names(new_fixtures, 'Away_Team', team_mapping)
        
        # Create features
        new_fixtures['features'] = new_fixtures.apply(
            lambda row: create_feature_vector(row, team_stats, feature_columns),
            axis=1
        )
        X_new = np.array(new_fixtures['features'].tolist())
    except Exception as e:
        raise ValueError(f"Fixture processing failed: {str(e)}")

    # Make predictions
    try:
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

        result_df['Total_Goals'] = (np.round(result_df['Predicted_Home_xG']) + 
                                  np.round(result_df['Predicted_Away_xG']))
        result_df['Prob_Diff'] = abs(result_df['Home_Win_Prob'] - result_df['Away_Win_Prob'])
        
        return result_df.sort_values('Prob_Diff', ascending=False)

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Find fixtures file
        fixtures_path = next(
            path for path in [
                BASE_DIR / "data" / "raw" / "fixtures.csv",
                BASE_DIR / "fixtures.csv",
                Path("fixtures.csv")
            ] if path.exists()
        )
        
        predictions = predict_fixtures(str(fixtures_path))
        print("Predictions generated successfully:")
        print(predictions.head())
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        print("\nDebugging info:")
        print(f"Models directory contents: {os.listdir(BASE_DIR / 'models')}")
        if (BASE_DIR / "models" / "xgb_artifacts.pkl").exists():
            try:
                artifacts = joblib.load(BASE_DIR / "models" / "xgb_artifacts.pkl")
                print(f"Artifacts keys: {artifacts.keys()}")
            except Exception as e:
                print(f"Failed to inspect artifacts: {str(e)}")