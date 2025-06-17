import optuna
import joblib
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from .utils import TeamBasedGroupKFold
from .data_ingestion import load_data
from .preprocessing import preprocess_data
from .feature_engineering import engineer_features

def create_objective(X_train, y_class_train, fixtures_df, train_idx):
    def objective(trial):
        params = {
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 20
        }

        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_class_train), y=y_class_train)

        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', xgb.XGBClassifier(**params))
        ])

        # Inner cross-validation
        scores = []
        inner_cv = TeamBasedGroupKFold(n_splits=3, random_state=42)
        
        # Create subset of fixtures for this fold's training data
        fold_fixtures = fixtures_df.iloc[train_idx].reset_index(drop=True)
        
        for inner_train_idx, inner_val_idx in inner_cv.split(fold_fixtures):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_class_train[inner_train_idx]

            # Apply SMOTE and sample weights
            X_res, y_res = SMOTE(random_state=42).fit_resample(X_inner_train, y_inner_train)
            sample_weights_res = class_weights[y_res]

            pipeline.named_steps['clf'].fit(
                X_res, y_res,
                sample_weight=sample_weights_res,
                eval_set=[(X_train[inner_val_idx], y_class_train[inner_val_idx])],
                verbose=False
            )

            y_pred = pipeline.predict(X_train[inner_val_idx])
            scores.append(f1_score(y_class_train[inner_val_idx], y_pred, average='macro'))

        return np.mean(scores)
    return objective

def train_model():
    # Load and preprocess data
    stats_df, fixtures_df, config = load_data()
    _, fixtures_df, team_stats = preprocess_data(stats_df, fixtures_df, config)
    X, y_class, y_reg = engineer_features(fixtures_df, team_stats, config)
    
    # Initialize the custom cross-validator
    group_kfold = TeamBasedGroupKFold(n_splits=5, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(fixtures_df)):
        print(f"Training fold {fold+1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
        y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]

        # Optuna optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        objective_func = create_objective(X_train, y_class_train, fixtures_df, train_idx)
        study.optimize(objective_func, n_trials=50)

        best_params = study.best_params

        # Train classifier with best params
        clf = xgb.XGBClassifier(
            **best_params,
            early_stopping_rounds=20,
            eval_metric="mlogloss"
        )

        # Apply SMOTE to training data
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_class_train)

        # Train with early stopping
        clf.fit(
            X_res, y_res,
            eval_set=[(X_val, y_class_val)],
            verbose=False
        )

        # Create classifier for calibration (without early stopping params)
        calib_clf_params = {k: v for k, v in best_params.items() 
                           if k not in ['early_stopping_rounds', 'eval_metric']}
        calib_clf = xgb.XGBClassifier(**calib_clf_params)
        calib_clf.fit(X_res, y_res)

        # Calibrate probabilities
        calibrated_clf = CalibratedClassifierCV(
            calib_clf,
            method='sigmoid',
            cv=KFold(n_splits=5, shuffle=True, random_state=42)
        )
        calibrated_clf.fit(X_train, y_class_train)

        # Train regression models
        reg_params = {k: v for k, v in best_params.items() 
                     if k not in ['objective', 'num_class', 'eval_metric', 'early_stopping_rounds']}
        
        reg_home = xgb.XGBRegressor(**reg_params).fit(X_res, y_reg_train[:, 0])
        reg_away = xgb.XGBRegressor(**reg_params).fit(X_res, y_reg_train[:, 1])

        # Save model artifacts
        joblib.dump({
            'classifier': calibrated_clf,
            'regressor_home': reg_home,
            'regressor_away': reg_away,
            'feature_columns': config['feature_columns'],
            'team_stats': team_stats,
            'best_params': best_params
        }, f'models/xgb_model_fold{fold+1}.pkl')

        # Validation metrics
        y_pred = calibrated_clf.predict(X_val)
        print(f"Fold {fold+1} - Val Accuracy: {accuracy_score(y_class_val, y_pred):.4f}")
        print(f"Fold {fold+1} - Val F1: {f1_score(y_class_val, y_pred, average='macro'):.4f}")

    # Save global artifacts
    joblib.dump({
        'team_stats': team_stats,
        'feature_columns': config['feature_columns']
    }, 'models/xgb_artifacts.pkl')

if __name__ == "__main__":
    train_model()