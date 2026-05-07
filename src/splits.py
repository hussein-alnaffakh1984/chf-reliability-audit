"""Data split functions for the three reliability scenarios."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_splits(X, y, df, seed=42):
    """Build the three train/calibration/test splits.

    Args:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples,)
        df: original DataFrame (must contain 'Reference ID' and 'Pressure' columns)
        seed: random seed for reproducibility

    Returns:
        dict with keys 'S1_random', 'S2_holdout_refs', 'S3_high_P'.
        Each value is a tuple (X_tr, y_tr, X_cal, y_cal, X_te, y_te).
    """
    rng = np.random.RandomState(seed)

    # --- S1: Random split (interpolation baseline) ---
    X_tv1, X_te1, y_tv1, y_te1 = train_test_split(
        X, y, test_size=0.15, random_state=seed
    )
    X_tr1, X_va1, y_tr1, y_va1 = train_test_split(
        X_tv1, y_tv1, test_size=0.176, random_state=seed
    )

    # --- S2: Hold out 10 reference IDs (cross-source transfer) ---
    unique_refs = df['Reference ID'].unique()
    test_refs = rng.choice(unique_refs, size=10, replace=False)
    mask_s2 = df['Reference ID'].isin(test_refs).values
    X_tv2 = X[~mask_s2]; y_tv2 = y[~mask_s2]
    X_te2 = X[mask_s2];  y_te2 = y[mask_s2]
    X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
        X_tv2, y_tv2, test_size=0.15, random_state=seed
    )

    # --- S3: High-pressure extrapolation (P > 16 MPa) ---
    mask_s3 = df['Pressure'].values > 16000
    X_tv3 = X[~mask_s3]; y_tv3 = y[~mask_s3]
    X_te3 = X[mask_s3];  y_te3 = y[mask_s3]
    X_tr3, X_va3, y_tr3, y_va3 = train_test_split(
        X_tv3, y_tv3, test_size=0.15, random_state=seed
    )

    return {
        'S1_random':       (X_tr1, y_tr1, X_va1, y_va1, X_te1, y_te1),
        'S2_holdout_refs': (X_tr2, y_tr2, X_va2, y_va2, X_te2, y_te2),
        'S3_high_P':       (X_tr3, y_tr3, X_va3, y_va3, X_te3, y_te3),
    }
