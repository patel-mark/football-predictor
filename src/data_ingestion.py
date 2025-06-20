import pandas as pd
import yaml
from .utils import temporal_train_test_split

def load_data(config_path='config/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    stats_df = pd.read_csv(config['data_paths']['raw_stats'])
    all_fixtures = pd.read_csv(config['data_paths']['raw_fixtures'])
    
    # Split fixtures temporally
    train_fixtures, test_fixtures = temporal_train_test_split(
        all_fixtures, 
        test_size=config['split']['test_size']
    )
    
    # Save test set
    test_fixtures.to_csv(config['data_paths']['test_fixtures'], index=False)
    
    return stats_df, train_fixtures, test_fixtures, config