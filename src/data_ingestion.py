import pandas as pd
import yaml
from .utils import clean_numeric_values, standardize_team_names

def load_data(config_path='config/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    stats_df = pd.read_csv(config['data_paths']['raw_stats'])
    fixtures_df = pd.read_csv(config['data_paths']['raw_fixtures'])
    
    return stats_df, fixtures_df, config