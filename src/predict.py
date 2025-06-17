import joblib
import numpy as np
import pandas as pd
from .utils import standardize_team_names, create_feature_vector
from .data_ingestion import load_data

def predict_fixtures(fixtures_csv_path, model_dir='models'):
    # -------------------- Ensemble Prediction Function -------------------- #
    def predict_fixtures(fixtures_csv_path):
        artifacts = joblib.load('xgb_artifacts.pkl')
        team_stats = artifacts['team_stats']
        feature_columns = artifacts['feature_columns']

        # Load all fold models
        models = [joblib.load(f'xgb_model_fold{i+1}.pkl') for i in range(5)]

        # Prepare new fixtures
        new_fixtures = pd.read_csv(fixtures_csv_path)
        new_fixtures = standardize_team_names(new_fixtures, 'Home_Team')
        new_fixtures = standardize_team_names(new_fixtures, 'Away_Team')

        new_fixtures['features'] = new_fixtures.apply(
                lambda row: (
                    list(team_stats[row['Home_Team'].strip()].values()) +  
                    list(team_stats[row['Away_Team'].strip()].values())    
                ), axis=1
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

        result_df = pd.DataFrame({
            'Home_Team': new_fixtures['Home_Team'],
            'Away_Team': new_fixtures['Away_Team'],
            'Home_Win_Prob': class_probs[:, 2] / len(models),
            'Draw_Prob': class_probs[:, 1] / len(models),
            'Away_Win_Prob': class_probs[:, 0] / len(models),
            'Predicted_Home_xG': home_xg / len(models),
            'Predicted_Away_xG': away_xg / len(models)
        })
        
        # Add calculated columns
        result_df['Total_Goals'] = (np.round(result_df['Predicted_Home_xG']) + 
                                np.round(result_df['Predicted_Away_xG']))
        result_df['Prob_Diff'] = abs(result_df['Home_Win_Prob'] - result_df['Away_Win_Prob'])
        
        # Sort by probability difference
        result_df = result_df.sort_values('Prob_Diff', ascending=False)
        return result_df
    
    pass

if __name__ == "__main__":
    _, config = load_data()
    predictions = predict_fixtures(config['data_paths']['new_fixtures'])
    predictions.to_csv('data/processed/predictions.csv', index=False)