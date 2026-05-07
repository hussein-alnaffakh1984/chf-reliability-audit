# Data

## OECD/NEA Phase-1 CHF Benchmark

The dataset used in this study is publicly available from:

**U.S. Nuclear Regulatory Commission (NRC)** — NUREG/KM-0011, ML22264A009
- Direct download: https://www.nrc.gov/docs/ML2226/ML22264A009.pdf

**Mirror (preprocessed)**:
- GitHub: https://github.com/zhouzhouwen/cutting-edge-artificial-intelligence-methods-for-prediction-of-critical-heat-flux

## Specifications

- **N = 24,579** measurements
- **60 reference experiments** spanning 1960s–2018
- **Geometry:** vertical, uniformly heated tubes
- **Working fluid:** water

## Variables

| Variable | Symbol | Unit | Range |
|----------|--------|------|-------|
| Tube diameter | D | m | 0.001 – 0.020 |
| Heated length | L | m | 0.05 – 20 |
| Pressure | P | kPa | 100 – 20,000 |
| Mass flux | G | kg/m²/s | 8 – 7,964 |
| Outlet quality | xₑ | – | -0.5 – 1.0 |
| **CHF (target)** | q_CHF | kW/m² | 50 – 16,339 |

## Note

We do not redistribute the dataset directly to respect NRC licensing. Use `notebooks/01_data_preparation.ipynb` to download and preprocess automatically.
