# Reliability of Machine Learning Predictions for Critical Heat Flux

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24208750.svg)](https://doi.org/10.5281/zenodo.24208750)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **An empirical study of generalization, calibration, and reproducibility on the OECD/NEA Phase-1 Critical Heat Flux benchmark.**

This repository contains all code, data references, and trained models for our paper investigating the reliability of machine learning (ML) for Critical Heat Flux (CHF) prediction in water-cooled nuclear reactors.

## 📰 Paper

> Al-Naffakh, H.A.H., Radhi, A.D., Abood, M.M., Majeed, F.A., Khalf, A.M., & Ghazali, R. (2026). Reliability of machine learning methods for prediction of critical heat flux: an OECD/NEA benchmark study. *Kerntechnik*, in press. https://doi.org/10.1515/kern-2026-0078

## 🎯 Key Findings

1. **Conformal Coverage Breakdown** — empirical coverage drops from 90.4% (random splits) to 62-63% (under distribution shift).

2. **Variance Amplification under Shift** — neural-network methods exhibit ~140× variance amplification on extrapolation splits (F-test p < 10⁻⁶), while tree-based methods amplify by less than 11×.

3. **Physics-aware features do not mitigate failure modes** — augmenting raw inputs with non-dimensional groups (Re, We, ρᵣ) produces no statistically significant improvement (Welch's t-test p > 0.83).

## 🔬 Reproducibility

### Requirements

```bash
pip install -r requirements.txt
```

### Quick Start

```python
import pandas as pd
import xgboost as xgb
from src.splits import build_splits

# Load data
df = pd.read_parquet('data/chf_clean.parquet')  # see data/README.md
FEATURES = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
X = df[FEATURES].values
y = df['CHF'].values

# Build the 3 splits
splits = build_splits(X, y, df, seed=42)

# Train XGBoost on S1 (random split)
X_tr, y_tr, X_va, y_va, X_te, y_te = splits['S1_random']
model = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                         early_stopping_rounds=30)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

# Predict
y_pred = model.predict(X_te)
```

### Reproducing All Experiments

Run the notebooks in order:

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `01_data_preparation.ipynb` | Download and clean NEA Phase-1 data | 2 min |
| `02_baseline_models.ipynb` | RF/XGB/MLP/BNN × 3 splits × 5 seeds | 30 min (GPU) |
| `03_conformal_prediction.ipynb` | CP coverage analysis | 5 min |
| `04_physics_ablation.ipynb` | Physics-aware feature ablation | 10 min |
| `05_statistical_tests.ipynb` | Levene, F-test, Bootstrap CIs | 1 min |

## 📊 Main Results

### Table 1 — Test-set performance (mean ± std across 5 seeds)

| Model | S1: R² / RMSE% | S2: R² / RMSE% | S3: R² / RMSE% |
|-------|-----------------|-----------------|-----------------|
| Random Forest | 0.983 ± 0.001 / 13.9 ± 0.2 | 0.886 ± 0.061 / 41.0 ± 11.8 | 0.599 ± 0.004 / 41.7 ± 0.3 |
| **XGBoost** | **0.986 ± 0.001 / 13.1 ± 0.9** | 0.905 ± 0.044 / 34.2 ± 8.2 | 0.648 ± 0.009 / 38.4 ± 0.6 |
| MLP | 0.972 ± 0.001 / 15.1 ± 0.2 | 0.929 ± 0.024 / 24.3 ± 5.9 | 0.652 ± 0.196 / 47.8 ± 7.6 |
| BNN | 0.970 ± 0.002 / 17.1 ± 1.0 | 0.925 ± 0.031 / 26.7 ± 6.0 | 0.551 ± 0.149 / 41.2 ± 4.1 |

### Table 2 — Conformal Prediction (target = 90%)

| Split | Empirical Coverage | Coverage Failures |
|-------|--------------------|-------------------|
| S1 Random | 90.3% | 9.7% |
| S2 Hold-out refs | 62.2% | 37.8% |
| S3 High-pressure | 63.4% | 36.6% |

## 📂 Data

The OECD/NEA Phase-1 CHF benchmark dataset (24,579 measurements from 60 reference experiments) is publicly available from the U.S. Nuclear Regulatory Commission. See [data/README.md](data/README.md) for download instructions.

## 📜 Citation

If you use this code or our findings in your research, please cite:

```bibtex
@article{AlNaffakh2026CHF,
  title   = {Reliability of machine learning methods for prediction of critical heat flux: an OECD/NEA benchmark study},
  author  = {Al-Naffakh, Hussein Ali Hussein and Radhi, Ahmed Dheyaa and Abood, Mohammad Malik and Majeed, Fouad A. and Khalf, Ayaat Maan and Ghazali, Rozaida},
  journal = {Kerntechnik},
  year    = {2026},
  note    = {In press},
  doi     = {10.1515/kern-2026-0078}
}
```

## 📄 License

This work is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Acknowledgements

We thank the OECD/NEA AI/ML Task Force for releasing the Phase-1 CHF benchmark, and the U.S. NRC for providing the original data compilation.

## 📧 Contact

For questions, please open an issue or contact: Hussein Ali Hussein Al-Naffakh (corresponding author), hussein.alnaffakh@alkafeel.edu.iq
