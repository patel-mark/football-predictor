import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score, f1_score
from .feature_engineering import create_feature_vector
from .data_ingestion import load_data
from .preprocessing import preprocess_data

def evaluate_model():
    # Load config and artifacts
    stats_df, _, test_fixtures, config = load_data()
    artifacts = joblib.load('models/xgb_artifacts.pkl')
    team_stats = artifacts['team_stats']
    feature_columns = artifacts['feature_columns']
    
    # Load training dates for reference
    try:
        training_dates = joblib.load('models/training_dates.pkl')
        print(f"Training Period: {training_dates['start']} to {training_dates['end']}")
    except FileNotFoundError:
        print("Warning: Training dates not found")
        training_dates = {'start': 'Unknown', 'end': 'Unknown'}

    # Preprocess TEST data using training-derived parameters
    _, test_fixtures, _ = preprocess_data(stats_df, test_fixtures, config)
    
    # Create features from TEST data
    test_fixtures['features'] = test_fixtures.apply(
        lambda row: create_feature_vector(row, team_stats, feature_columns),
        axis=1
    )
    X_test = np.array(test_fixtures['features'].tolist())
    
    # Create target variable for test data
    test_fixtures['result'] = test_fixtures.apply(
        lambda row: 2 if row['Home_Score'] > row['Away_Score'] else 1 if row['Home_Score'] == row['Away_Score'] else 0,
        axis=1
    )
    y_class_test = test_fixtures['result'].values
    y_reg_test = test_fixtures[['Home_xG', 'Away_xG']].values
    
    # Load models
    models = []
    for i in range(5):
        try:
            model = joblib.load(f'models/xgb_model_fold{i+1}.pkl')
            models.append(model)
        except FileNotFoundError:
            print(f"Warning: Model for fold {i+1} not found. Skipping.")
            continue
            
    if not models:
        raise ValueError("No models found in the models directory")
    
    # Ensemble predictions
    class_probs = np.zeros((X_test.shape[0], 3))
    home_xg_pred = np.zeros(X_test.shape[0])
    away_xg_pred = np.zeros(X_test.shape[0])
    
    for model in models:
        class_probs += model['classifier'].predict_proba(X_test)
        home_xg_pred += model['regressor_home'].predict(X_test)
        away_xg_pred += model['regressor_away'].predict(X_test)
    
    num_models = len(models)
    y_class_pred = np.argmax(class_probs / num_models, axis=1)
    home_xg_pred /= num_models
    away_xg_pred /= num_models
    
    # Print date information
    print("\n" + "="*60)
    print(f"Temporal Evaluation Report")
    print(f"Training Period: {training_dates['start']} to {training_dates['end']}")
    print(f"Test Period: {test_fixtures['Date'].min()} to {test_fixtures['Date'].max()}")
    print(f"Test Matches: {len(test_fixtures)}")
    print("="*60)
    
    # Evaluation metrics
    print("\nClassification Performance:")
    print(classification_report(y_class_test, y_class_pred))
    print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred):.4f}")
    print(f"Macro F1: {f1_score(y_class_test, y_class_pred, average='macro'):.4f}")
    
    print("\nRegression Performance:")
    print(f"Home xG MAE: {mean_absolute_error(y_reg_test[:, 0], home_xg_pred):.4f}")
    print(f"Away xG MAE: {mean_absolute_error(y_reg_test[:, 1], away_xg_pred):.4f}")
    
    # Calculate baseline MAE (mean prediction)
    home_xg_baseline = np.mean(y_reg_test[:, 0])
    away_xg_baseline = np.mean(y_reg_test[:, 1])
    home_mae_baseline = mean_absolute_error(y_reg_test[:, 0], [home_xg_baseline]*len(y_reg_test))
    away_mae_baseline = mean_absolute_error(y_reg_test[:, 1], [away_xg_baseline]*len(y_reg_test))
    
    print(f"\nBaseline MAE (Mean Prediction):")
    print(f"Home xG: {home_mae_baseline:.4f}")
    print(f"Away xG: {away_mae_baseline:.4f}")
    
    print("\n" + "="*60)
    
    # Return metrics for monitoring
    return {
        'accuracy': accuracy_score(y_class_test, y_class_pred),
        'f1_macro': f1_score(y_class_test, y_class_pred, average='macro'),
        'home_xg_mae': mean_absolute_error(y_reg_test[:, 0], home_xg_pred),
        'away_xg_mae': mean_absolute_error(y_reg_test[:, 1], away_xg_pred),
        'test_start_date': test_fixtures['Date'].min(),
        'test_end_date': test_fixtures['Date'].max(),
        'test_matches': len(test_fixtures)
    }

if __name__ == "__main__":
    evaluate_model()