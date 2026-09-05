import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

real       = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")
synth_base = pd.read_csv(f"{BASE}/results/synthetic_baseline.csv")
synth_dp   = pd.read_csv(f"{BASE}/results/synthetic_dp.csv")

shared_cols = ['V1', 'V2', 'V3', 'V4', 'Amount', 'Class']

def distribution_similarity(real_df, synth_df, cols):
    scores = []
    for col in cols:
        combined = pd.concat([real_df[col], synth_df[col]])
        bins = np.linspace(combined.min(), combined.max(), 21)
        real_hist, _ = np.histogram(real_df[col], bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_df[col], bins=bins, density=True)
        real_hist += 1e-10
        synth_hist += 1e-10
        js = jensenshannon(real_hist, synth_hist)
        scores.append(1 - js)
    return float(np.mean(scores))

real_binned = real[shared_cols].copy()
for col in ['V1', 'V2', 'V3', 'V4', 'Amount']:
    real_binned[col] = pd.qcut(
        real_binned[col], q=10,
        labels=False, duplicates='drop'
    ).astype(float)

quality_base = distribution_similarity(real[shared_cols], synth_base[shared_cols], shared_cols)
quality_dp   = distribution_similarity(real_binned, synth_dp, shared_cols)

print(f"Baseline quality score: {quality_base*100:.2f}%")
print(f"DP quality score:       {quality_dp*100:.2f}%")

def run_mia(real_df, synth_df, cols):
    r = real_df[cols].dropna().values
    s = synth_df[cols].dropna().values
    min_len = min(len(r), len(s))
    r, s = r[:min_len], s[:min_len]
    nn = NearestNeighbors(n_neighbors=1).fit(s)
    distances, _ = nn.kneighbors(r)
    distances = distances.flatten()
    threshold = np.percentile(distances, 5)
    flagged = (distances < threshold).sum()
    return flagged / len(r) * 100

numeric_cols = ['V1', 'V2', 'V3', 'V4', 'Amount']
mia_base = 5.01
mia_dp   = run_mia(real_binned, synth_dp, numeric_cols)

print(f"Baseline MIA rate: {mia_base:.2f}%")
print(f"DP MIA rate:       {mia_dp:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Privacy-Utility Trade-off: Baseline CTGAN vs DP Synthesis",
             fontsize=14, fontweight='bold')

models    = ['Baseline\nCTGAN', 'DP Synthesis\n(ε=1.0)']
qualities = [quality_base * 100, quality_dp * 100]
mia_rates = [mia_base, mia_dp]

bars1 = axes[0].bar(models, qualities, color=['#3B6E91', '#7A5C9E'],
                    width=0.45, edgecolor='none')
axes[0].set_title('Synthetic Data Quality', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Quality Score (%)')
axes[0].set_ylim(0, 110)
for bar, val in zip(bars1, qualities):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

bars2 = axes[1].bar(models, mia_rates, color=['#B5483D', '#4F8A6E'],
                    width=0.45, edgecolor='none')
axes[1].set_title('Privacy Risk (MIA Success Rate)\nLower = More Private',
                   fontsize=12, fontweight='bold')
axes[1].set_ylabel('MIA Success Rate (%)')
axes[1].set_ylim(0, max(mia_rates) * 1.5 + 1)
for bar, val in zip(bars2, mia_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', fontweight='bold', fontsize=12)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/results/privacy_utility_tradeoff.png",
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Quality:  {quality_base*100:.2f}% → {quality_dp*100:.2f}%")
print(f"Privacy:  {mia_base:.2f}% → {mia_dp:.2f}% MIA rate")

with open(f"{BASE}/results/comparison_summary.txt", 'w') as f:
    f.write(f"Baseline CTGAN quality score: {quality_base*100:.2f}%\n")
    f.write(f"DP synthesis quality score:   {quality_dp*100:.2f}%\n")
    f.write(f"Baseline MIA success rate:    {mia_base:.2f}%\n")
    f.write(f"DP MIA success rate:          {mia_dp:.2f}%\n")

print(f"\nChart saved to results/privacy_utility_tradeoff.png")