import warnings
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
import joblib
from scipy.stats import randint, uniform
from sklearn.model_selection import PredefinedSplit
from .utils import TeamBasedGroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)


def create_objective(X_train, y_class_train, fixtures_df):
    # -------------------- Optuna Objective Function -------------------- #
    def create_objective(X_train, y_class_train, y_reg_train):
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
                'early_stopping_rounds': 20  # Moved to constructor parameters
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
            for inner_train_idx, inner_val_idx in inner_cv.split(fixtures_df.iloc[train_idx]):
                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_class_train[inner_train_idx]

                # Apply SMOTE and sample weights
                X_res, y_res = SMOTE(random_state=42).fit_resample(X_inner_train, y_inner_train)
                sample_weights_res = class_weights[y_res]

                # Modified fit call
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
    pass

def train_model():
    stats_df, fixtures_df, config = load_data()
    stats_df, fixtures_df, team_stats = preprocess_data(stats_df, fixtures_df, config)
    X, y_class, y_reg = engineer_features(fixtures_df, team_stats, config)
    
    # -------------------- Training Execution -------------------- #
    if __name__ == "__main__":
        team_stats, fixtures_df, feature_columns = load_and_preprocess_data()

        # Feature engineering
        fixtures_df['features'] = fixtures_df.apply(
            lambda row: (
                list(team_stats[row['Home_Team'].strip()].values()) +  # Fixed closing )
                list(team_stats[row['Away_Team'].strip()].values())    # Fixed closing )
            ), axis=1
        )

        fixtures_df['result'] = fixtures_df.apply(
            lambda row: 2 if row['Home_Score'] > row['Away_Score'] else 1 if row['Home_Score'] == row['Away_Score'] else 0,
            axis=1
        )

        X = np.array(fixtures_df['features'].tolist())
        y_class = fixtures_df['result'].values
        y_reg = fixtures_df[['Home_xG', 'Away_xG']].values

        group_kfold = TeamBasedGroupKFold(n_splits=5, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(fixtures_df)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
            y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]

            # Corrected Optuna optimization call
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(
                create_objective(X_train, y_class_train, y_reg_train),
                n_trials=50
            )

            best_params = study.best_params

            # Now use best_params
            # Instantiate classifier with early stopping and eval_metric in the constructor
            clf = xgb.XGBClassifier(
                **best_params,
                early_stopping_rounds=20,
                eval_metric="mlogloss"
            )

            # ======== CRITICAL FIX ========
            # 1. Apply SMOTE to training data
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_class_train)
            # ==============================

            # 2. Train classifier with early stopping (parameters passed in constructor)
            clf.fit(
                X_res, y_res,
                eval_set=[(X_val, y_class_val)],
                # early_stopping_rounds=20, # REMOVE - passed in constructor
                verbose=False
                # eval_metric="mlogloss" # REMOVE - passed in constructor
            )

            # 3. Create a clean classifier copy without early stopping for calibration
            # Early stopping should not be used during calibration fit
            calib_clf_params = {k: v for k, v in best_params.items() if k not in ['early_stopping_rounds', 'eval_metric']}
            calib_clf = xgb.XGBClassifier(**calib_clf_params)
            calib_clf.fit(X_res, y_res) # Train on SMOTE data without early stopping parameters

            # 4. Calibrate using original training splits
            calibrated_clf = CalibratedClassifierCV(
                calib_clf,
                method='sigmoid',
                cv=KFold(n_splits=5, shuffle=True, random_state=42)
            )
            calibrated_clf.fit(X_train, y_class_train)

            # 5. Train regression models - Use best_params, early stopping and eval_metric usually not needed for regressor fit unless you specifically want it.
            reg_home_params = {k: v for k, v in best_params.items() if k not in ['objective', 'num_class', 'eval_metric', 'early_stopping_rounds']}
            reg_away_params = {k: v for k, v in best_params.items() if k not in ['objective', 'num_class', 'eval_metric', 'early_stopping_rounds']}

            y_reg_train_res = y_reg_train[np.arange(len(X_res)) % len(y_reg_train)]

            reg_home = xgb.XGBRegressor(**reg_home_params).fit(X_res, y_reg_train_res[:, 0])
            reg_away = xgb.XGBRegressor(**reg_away_params).fit(X_res, y_reg_train_res[:, 1])
    # Save artifacts
    joblib.dump({
        'team_stats': team_stats,
        'feature_columns': config['feature_columns']
    }, 'models/xgb_artifacts.pkl')
    
if __name__ == "__main__":
    train_model()