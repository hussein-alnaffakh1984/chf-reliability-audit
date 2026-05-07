"""Statistical tests for variance amplification and ablations."""
import numpy as np
from scipy import stats


def cohens_d(x, y):
    """Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled


def cohens_d_interpretation(d):
    """Interpret a Cohen's d magnitude."""
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    """Bootstrap confidence interval for the mean of values."""
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return lower, upper


def bootstrap_amplification(s1_vals, s3_vals, n_boot=10000, ci=95, seed=42):
    """Bootstrap CI for the variance amplification factor sigma(S3) / sigma(S1)."""
    rng = np.random.RandomState(seed)
    amps = []
    for _ in range(n_boot):
        s1_boot = rng.choice(s1_vals, size=len(s1_vals), replace=True)
        s3_boot = rng.choice(s3_vals, size=len(s3_vals), replace=True)
        s1_std = s1_boot.std()
        if s1_std > 1e-9:
            amps.append(s3_boot.std() / s1_std)
    amps = np.array(amps)
    return {
        'median': float(np.median(amps)),
        'ci_lower': float(np.percentile(amps, (100 - ci) / 2)),
        'ci_upper': float(np.percentile(amps, 100 - (100 - ci) / 2)),
    }


def variance_amplification_tests(s1_vals, s3_vals):
    """Run Levene's test, F-test, and bootstrap CI for variance amplification."""
    s1 = np.asarray(s1_vals)
    s3 = np.asarray(s3_vals)

    s1_var = s1.var(ddof=1)
    s3_var = s3.var(ddof=1)
    F = s3_var / s1_var if s1_var > 0 else float('inf')
    df1, df2 = len(s3) - 1, len(s1) - 1
    F_p = 1 - stats.f.cdf(F, df1, df2)

    levene_W, levene_p = stats.levene(s1, s3, center='median')
    boot = bootstrap_amplification(s1, s3)

    return {
        's1_std': float(s1.std()),
        's3_std': float(s3.std()),
        'amplification_factor': float(s3.std() / s1.std()) if s1.std() > 0 else float('inf'),
        'levene_W': float(levene_W),
        'levene_p': float(levene_p),
        'F_ratio': float(F),
        'F_p': float(F_p),
        'bootstrap': boot,
    }
