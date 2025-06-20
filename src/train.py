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
from sklearn.neighbors import NearestNeighbors
from .utils import TeamBasedGroupKFold
from .data_ingestion import load_data
from .preprocessing import preprocess_data
from .feature_engineering import engineer_features
import gc

def combined_resample(X, y_class, y_reg):
    """Resamples data while preserving all targets"""
    smote = SMOTE(random_state=42)
    
    # Resample classification data
    X_res, y_class_res = smote.fit_resample(X, y_class)
    
    # Create regression targets for synthetic samples
    n_original = X.shape[0]
    n_synthetic = X_res.shape[0] - n_original
    
    if n_synthetic > 0:
        # Find nearest neighbors for synthetic samples
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X)
        _, indices = nn.kneighbors(X_res[n_original:])
        
        # Create regression targets (use nearest neighbor's targets)
        synthetic_y_reg = y_reg[indices.flatten()]
        y_reg_res = np.vstack([y_reg, synthetic_y_reg])
    else:
        y_reg_res = y_reg
    
    return X_res, y_class_res, y_reg_res

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
            X_res_inner, y_res_inner = SMOTE(random_state=42).fit_resample(X_inner_train, y_inner_train)
            sample_weights_res = class_weights[y_res_inner]

            pipeline.named_steps['clf'].fit(
                X_res_inner, y_res_inner,
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

        # Apply SMOTE to training data with regression target preservation
        X_res, y_class_res, y_reg_res = combined_resample(X_train, y_class_train, y_reg_train)
        
        # Validation check
        assert X_res.shape[0] == len(y_class_res) == len(y_reg_res), \
            f"Data size mismatch: Features({X_res.shape[0]}), " \
            f"Class targets({len(y_class_res)}), Reg targets({len(y_reg_res)})"
        
        print(f"Resampled training data from {len(X_train)} to {len(X_res)} samples")

        # Train with early stopping
        clf.fit(
            X_res, y_class_res,
            eval_set=[(X_val, y_class_val)],
            verbose=False
        )

        # Create classifier for calibration
        calib_clf_params = {k: v for k, v in best_params.items() 
                           if k not in ['early_stopping_rounds', 'eval_metric']}
        calib_clf = xgb.XGBClassifier(**calib_clf_params)
        calib_clf.fit(X_res, y_class_res)

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
        reg_params.update({
            'tree_method': 'hist',
            'objective': 'reg:squarederror'
        })
        
        # Additional validation for regression targets
        assert len(X_res) == len(y_reg_res), \
            f"Regression target size mismatch: Features({len(X_res)}), Targets({len(y_reg_res)})"
        
        reg_home = xgb.XGBRegressor(**reg_params).fit(X_res, y_reg_res[:, 0])
        reg_away = xgb.XGBRegressor(**reg_params).fit(X_res, y_reg_res[:, 1])

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
        acc = accuracy_score(y_class_val, y_pred)
        f1 = f1_score(y_class_val, y_pred, average='macro')
        print(f"Fold {fold+1} - Val Accuracy: {acc:.4f}")
        print(f"Fold {fold+1} - Val F1: {f1:.4f}")
        
        # Memory cleanup
        del clf, calib_clf, calibrated_clf, reg_home, reg_away
        gc.collect()

    # Save global artifacts
    joblib.dump({
        'team_stats': team_stats,
        'feature_columns': config['feature_columns']
    }, 'models/xgb_artifacts.pkl')
    print("Training completed. Models saved to models/ directory")

if __name__ == "__main__":
    train_model()