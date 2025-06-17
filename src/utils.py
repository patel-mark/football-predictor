import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def clean_numeric_values(value):
    if isinstance(value, str):
        cleaned = value.replace(',', '').replace(' ', '')
        if cleaned.replace('.', '', 1).isdigit():
            return float(cleaned)
    return value

def standardize_team_names(df, column_name, team_mapping):
    df = df.copy()
    df.loc[:, column_name] = df[column_name].replace(team_mapping).str.strip()
    return df
    #df.loc[:, column_name] = df[column_name].apply(
    #   lambda x: team_mapping.get(x.strip(), x)
    #return df
def preprocess_stats(stats_df, feature_columns):
    stats_df = stats_df.copy()
    stats_df.loc[:, 'total_progression'] = stats_df['progressive_carries'] + stats_df['progressive_passes']
    imputer = SimpleImputer(strategy='median')
    stats_df.loc[:, feature_columns] = imputer.fit_transform(stats_df[feature_columns])
    return stats_df

def create_feature_vector(row, team_stats, feature_columns):
    home_features = [team_stats[row['Home_Team'].strip()][col] for col in feature_columns]
    away_features = [team_stats[row['Away_Team'].strip()][col] for col in feature_columns]
    return home_features + away_features