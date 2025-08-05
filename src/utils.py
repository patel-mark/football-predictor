import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

def clean_numeric_values(value):
    if isinstance(value, str):
        cleaned = value.replace(',', '').replace(' ', '')
        if cleaned.replace('.', '', 1).isdigit():
            return float(cleaned)
    return value

def temporal_train_test_split(fixtures_df, test_size=0.2):
    """
    Split fixtures chronologically based on date
    """
    # Ensure we have a datetime column
    if 'Date' not in fixtures_df.columns:
        raise ValueError("Data must contain 'Date' column for temporal split")
    
    # Convert to datetime and sort
    fixtures_df = fixtures_df.copy()
    fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'])
    fixtures_df = fixtures_df.sort_values('Date')
    
    # Calculate split index
    split_idx = int(len(fixtures_df) * (1 - test_size))
    
    # Split data
    train_df = fixtures_df.iloc[:split_idx]
    test_df = fixtures_df.iloc[split_idx:]
    
    return train_df, test_df

def standardize_team_names(df, column_name, team_mapping):
    df = df.copy()
    df.loc[:, column_name] = df[column_name].replace(team_mapping).str.strip()
    return df

def preprocess_stats(stats_df, feature_columns):
    stats_df = stats_df.copy()
    imputer = SimpleImputer(strategy='median')
    stats_df.loc[:, feature_columns] = imputer.fit_transform(stats_df[feature_columns])
    return stats_df

def create_feature_vector(row, team_stats, feature_columns):
    home_features = [team_stats[row['Home_Team'].strip()][col] for col in feature_columns]
    away_features = [team_stats[row['Away_Team'].strip()][col] for col in feature_columns]
    return home_features + away_features

# -------------------- TeamBasedGroupKFold Splitter -------------------- #
class TeamBasedGroupKFold:
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, fixtures_df):
        unique_teams = pd.concat([fixtures_df['Home_Team'], fixtures_df['Away_Team']]).unique()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for _, test_teams_idx in kf.split(unique_teams):
            test_teams = unique_teams[test_teams_idx]
            val_mask = (
                fixtures_df['Home_Team'].isin(test_teams) |
                fixtures_df['Away_Team'].isin(test_teams)
            )
            # Convert boolean mask to positional indices
            val_idx = val_mask.to_numpy().nonzero()[0] 
            train_idx = (~val_mask).to_numpy().nonzero()[0]
            yield train_idx, val_idx