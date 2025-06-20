import pandas as pd
from .utils import clean_numeric_values, standardize_team_names, preprocess_stats

def preprocess_data(stats_df, fixtures_df, config):
    # Clean numeric values
    numeric_cols = config['feature_columns'] + ['Home_xG', 'Away_xG']
    for df in [stats_df, fixtures_df]:
        if df is None:  # Handle case when stats_df might be None for test data
            continue
        for col in numeric_cols:
            if col in df.columns:
                df.loc[:, col] = df[col].apply(clean_numeric_values)
    
    # Handle missing values
    fixtures_df = fixtures_df.dropna(subset=['Home_xG', 'Away_xG']).copy()
    
    # Standardize team names
    team_mapping = config['team_mapping']
    stats_df = standardize_team_names(stats_df, 'team', team_mapping)
    fixtures_df = standardize_team_names(fixtures_df, 'Home_Team', team_mapping)
    fixtures_df = standardize_team_names(fixtures_df, 'Away_Team', team_mapping)
    
    # Remove matches where home and away are the same team
    fixtures_df = fixtures_df.loc[fixtures_df['Home_Team'] != fixtures_df['Away_Team']].copy()
    
    # Preprocess stats
    stats_df = preprocess_stats(stats_df, config['feature_columns'])
    
    # Create team stats dictionary
    team_stats = stats_df.set_index('team')[config['feature_columns']].to_dict('index')
    
    return stats_df, fixtures_df, team_stats