import pandas as pd
import numpy as np
import os

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

real = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")
synthetic = pd.read_csv(f"{BASE}/results/synthetic_baseline.csv")

print(f"Real data shape:      {real.shape}")
print(f"Synthetic data shape: {synthetic.shape}")

try:
    from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

    metadata = {
        "columns": {
            col: {"sdtype": "categorical"} if col == "Class"
            else {"sdtype": "numerical"}
            for col in real.columns
        }
    }

    print("\nRunning quality evaluation...")
    report = QualityReport()
    report.generate(real, synthetic, metadata, verbose=True)
    score = report.get_score()
    print(f"\nOverall Quality Score: {score:.4f} ({score*100:.2f}%)")

    print("\nRunning diagnostic...")
    diag = DiagnosticReport()
    diag.generate(real, synthetic, metadata, verbose=True)

except ImportError:
    print("\nsdmetrics not available — using JS divergence fallback...")
    from scipy.spatial.distance import jensenshannon

    scores = []
    for col in real.columns:
        combined = pd.concat([real[col], synthetic[col]])
        bins = np.linspace(combined.min(), combined.max(), 21)
        real_hist, _ = np.histogram(real[col], bins=bins, density=True)
        synth_hist, _ = np.histogram(synthetic[col], bins=bins, density=True)
        real_hist += 1e-10
        synth_hist += 1e-10
        js = jensenshannon(real_hist, synth_hist)
        scores.append(1 - js)

    score = float(np.mean(scores))
    print(f"\nOverall Quality Score (JS similarity): {score:.4f} ({score*100:.2f}%)")

with open(f"{BASE}/results/quality_score.txt", 'w') as f:
    f.write(f"Baseline CTGAN Quality Score: {score:.4f}\n")

print("\nEvaluation complete.")