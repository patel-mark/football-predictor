# src/evaluate.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error

def evaluate_model(test_csv_path='data/raw/test_fixtures.csv'):
    # Load artifacts
    artifacts = joblib.load('models/xgb_artifacts.pkl')
    team_stats = artifacts['team_stats']
    feature_columns = artifacts['feature_columns']
    
    # Load test data
    test_fixtures = pd.read_csv(test_csv_path)
    
    # Preprocess and feature engineering
    test_fixtures['features'] = test_fixtures.apply(
        lambda row: create_feature_vector(row, team_stats, feature_columns),
        axis=1
    )
    X_test = np.array(test_fixtures['features'].tolist())
    y_class_test = test_fixtures['result'].values
    y_reg_test = test_fixtures[['Home_xG', 'Away_xG']].values
    
    # Load models
    models = [joblib.load(f'models/xgb_model_fold{i+1}.pkl') for i in range(5)]
    
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
    
    # Evaluation metrics
    print("Classification Report:")
    print(classification_report(y_class_test, y_class_pred))
    
    print("\nRegression Metrics:")
    print(f"Home xG MAE: {mean_absolute_error(y_reg_test[:, 0], home_xg_pred):.3f}")
    print(f"Away xG MAE: {mean_absolute_error(y_reg_test[:, 1], away_xg_pred):.3f}")
    
    # Return metrics for monitoring
    return {
        'accuracy': accuracy_score(y_class_test, y_class_pred),
        'f1_macro': f1_score(y_class_test, y_class_pred, average='macro'),
        'home_xg_mae': mean_absolute_error(y_reg_test[:, 0], home_xg_pred),
        'away_xg_mae': mean_absolute_error(y_reg_test[:, 1], away_xg_pred)
    }

if __name__ == "__main__":
    evaluate_model()