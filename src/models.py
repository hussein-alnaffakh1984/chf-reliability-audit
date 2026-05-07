"""ML model factories for CHF prediction."""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def make_random_forest(seed=42, n_estimators=300):
    """Random Forest regressor with default hyperparameters."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=seed,
    )


def make_xgboost(seed=42, n_estimators=1000):
    """XGBoost regressor with default hyperparameters."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=30,
        eval_metric='rmse',
    )


def train_xgb_with_val(X_tr, y_tr, X_va, y_va, seed=42):
    """Train XGBoost with early stopping on validation set."""
    model = make_xgboost(seed=seed)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


# Note: MLP and BNN are PyTorch-based and have more setup;
# see notebooks/02_baseline_models.ipynb for the full implementation.
