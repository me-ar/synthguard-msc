# SynthGuard — Privacy-Preserving Synthetic Financial Data

MSc Data Science Project — University of Liverpool

## Overview
SynthGuard generates privacy-preserving synthetic financial data using differential privacy (MWEM). It includes a membership inference attack to measure privacy leakage and a Streamlit dashboard for interactive use.

## Installation
```bash
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app.py
```

## Run the research pipeline
```bash
python 01_eda.py
python 02_train_ctgan.py
python 03_evaluate.py
python 04_mia_attack.py
python 05_dp_synthesis.py
python 06_compare_results.py
python 07_epsilon_sweep.py
```

## Dataset
Uses the Credit Card Fraud Detection dataset from Kaggle. Download it and place it at:
`Desertation/data/creditcard.csv`

## Research findings
- Baseline CTGAN quality score: 80.96%, MIA rate: 5.01%
- DP synthesis (ε=1.0) quality score: 96.05%, MIA rate: 0.00%
