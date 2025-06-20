import joblib
import numpy as np
import pandas as pd
from .utils import standardize_team_names
from .data_ingestion import load_data
from .feature_engineering import create_feature_vector  # Add this import

def predict_fixtures(fixtures_csv_path, model_dir='models'):
    # Load artifacts and config
    artifacts = joblib.load(f'{model_dir}/xgb_artifacts.pkl')
    team_stats = artifacts['team_stats']
    feature_columns = artifacts['feature_columns']
    
    # Load config separately
    _, _, config = load_data()  # Correct unpacking
    
    # Load all fold models
    models = []
    for i in range(5):
        try:
            model = joblib.load(f'{model_dir}/xgb_model_fold{i+1}.pkl')
            models.append(model)
        except FileNotFoundError:
            print(f"Warning: Model for fold {i+1} not found. Skipping.")
            continue
            
    if not models:
        raise ValueError("No models found in the models directory")

    # Prepare new fixtures
    new_fixtures = pd.read_csv(fixtures_csv_path)
    new_fixtures = standardize_team_names(new_fixtures, 'Home_Team', config['team_mapping'])
    new_fixtures = standardize_team_names(new_fixtures, 'Away_Team', config['team_mapping'])

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

    # Save predictions
    result_df.to_csv('data/processed/predictions.csv', index=False)
    
    return result_df

if __name__ == "__main__":
    # Get config to find new fixtures path
    _, _, config = load_data()  # Correct unpacking
    predictions = predict_fixtures(config['data_paths']['new_fixtures'])
    print(predictions)