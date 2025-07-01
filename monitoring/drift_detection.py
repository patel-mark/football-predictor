# monitoring/drift_detection.py
from scipy.stats import ks_2samp
import numpy as np
import joblib
import pandas as pd

def detect_drift(new_fixtures_path='data/raw/new_fixtures.csv'):
    # Load training data stats
    artifacts = joblib.load('models/xgb_artifacts.pkl')
    training_team_stats = artifacts['team_stats']
    
    # Calculate baseline feature distributions
    baseline_features = []
    for team, stats in training_team_stats.items():
        baseline_features.append(list(stats.values()))
    baseline_features = np.array(baseline_features)
    
    # Load new fixtures
    new_fixtures = pd.read_csv(new_fixtures_path)
    new_features = []
    for _, row in new_fixtures.iterrows():
        home_features = [training_team_stats.get(row['Home_Team'], {}).get(col, 0) 
                         for col in artifacts['feature_columns']]
        away_features = [training_team_stats.get(row['Away_Team'], {}).get(col, 0) 
                         for col in artifacts['feature_columns']]
        new_features.append(home_features + away_features)
    new_features = np.array(new_features)
    
    # Compare distributions
    drift_results = {}
    for i in range(baseline_features.shape[1]):
        stat, p_value = ks_2samp(baseline_features[:, i], new_features[:, i])
        drift_results[f'feature_{i}'] = {
            'ks_statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }
    
    return drift_results