import numpy as np
from .utils import create_feature_vector

def engineer_features(fixtures_df, team_stats, config):
    fixtures_df = fixtures_df.copy()
    
    # Create match features
    fixtures_df['features'] = fixtures_df.apply(
        lambda row: create_feature_vector(row, team_stats, config['feature_columns']), 
        axis=1
    )
    
    # Create target variable
    fixtures_df['result'] = fixtures_df.apply(
        lambda row: 2 if row['Home_Score'] > row['Away_Score'] else 1 if row['Home_Score'] == row['Away_Score'] else 0,
        axis=1
    )
    
    # Prepare data arrays
    X = np.array(fixtures_df['features'].tolist())
    y_class = fixtures_df['result'].values
    y_reg = fixtures_df[['Home_xG', 'Away_xG']].values
    
    return X, y_class, y_reg

def create_feature_vector(row, team_stats, feature_columns):
    home_features = [team_stats[row['Home_Team'].strip()][col] for col in feature_columns]
    away_features = [team_stats[row['Away_Team'].strip()][col] for col in feature_columns]
    return home_features + away_features