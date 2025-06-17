from src.data_ingestion import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.train import train_model
from src.predict import predict_fixtures

def full_pipeline():
    _, _, config = load_data()
    stats_df, fixtures_df, team_stats = preprocess_data(*load_data())
    X, y_class, y_reg = engineer_features(fixtures_df, team_stats, config)
    train_model()
    predict_fixtures(config['data_paths']['new_fixtures'])