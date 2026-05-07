"""Split Conformal Prediction for distribution-free uncertainty."""
import numpy as np


def conformal_quantile(residuals, alpha=0.10):
    """Compute the conformity quantile q_hat at miscoverage level alpha."""
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(residuals, q_level))


def evaluate_conformal_coverage(model, X_cal, y_cal, X_te, y_te, alpha=0.10):
    """Compute the empirical coverage of conformal intervals on a test set."""
    y_cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)
    q_hat = conformal_quantile(residuals, alpha=alpha)

    y_te_pred = model.predict(X_te)
    in_interval = np.abs(y_te - y_te_pred) <= q_hat
    coverage = float(in_interval.mean())

    return {
        'q_hat': q_hat,
        'empirical_coverage': coverage,
        'target_coverage': 1 - alpha,
        'calibration_gap': coverage - (1 - alpha),
        'failures': int((~in_interval).sum()),
        'n_test': int(len(y_te)),
    }
