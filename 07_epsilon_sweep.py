import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from snsynth import Synthesizer

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

# ── Load and prepare data ──────────────────────────────────
real = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")

selected_columns = ['V1', 'V2', 'V3', 'V4', 'V14', 'V17', 'Amount', 'Class']
real_reduced = real[selected_columns].copy()

for col in ['V1', 'V2', 'V3', 'V4', 'V14', 'V17', 'Amount']:
    real_reduced[col] = pd.qcut(
        real_reduced[col], q=10,
        labels=False, duplicates='drop'
    ).astype(int)

# ── Quality metric ─────────────────────────────────────────
shared_cols = ['V1', 'V2', 'V3', 'V4', 'Amount', 'Class']

def distribution_similarity(real_df, synth_df, cols):
    scores = []
    for col in cols:
        combined = pd.concat([real_df[col], synth_df[col]])
        bins = np.linspace(combined.min(), combined.max(), 21)
        r, _ = np.histogram(real_df[col], bins=bins, density=True)
        s, _ = np.histogram(synth_df[col], bins=bins, density=True)
        r += 1e-10
        s += 1e-10
        scores.append(1 - jensenshannon(r, s))
    return float(np.mean(scores))

# ── MIA metric ─────────────────────────────────────────────
numeric_cols = ['V1', 'V2', 'V3', 'V4', 'Amount']

def run_mia(real_df, synth_df, cols):
    r = real_df[cols].dropna().values
    s = synth_df[cols].dropna().values
    min_len = min(len(r), len(s))
    r, s = r[:min_len], s[:min_len]
    nn = NearestNeighbors(n_neighbors=1).fit(s)
    distances, _ = nn.kneighbors(r)
    distances = distances.flatten()
    threshold = np.percentile(distances, 5)
    return (distances < threshold).sum() / len(r) * 100

# ── Epsilon sweep ──────────────────────────────────────────
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
quality_scores = []
mia_rates = []

real_binned = real_reduced.copy()

print(f"{'Epsilon':<10} {'Quality %':<15} {'MIA Rate %':<15}")
print("-" * 40)

for eps in epsilons:
    print(f"\nTraining DP synthesizer with epsilon={eps}...")
    try:
        dp_synth = Synthesizer.create('mwem', epsilon=eps, split_factor=4)
        dp_synth.fit(real_reduced, preprocessor_eps=min(0.5, eps * 0.4))
        dp_synthetic = dp_synth.sample(len(real_reduced))

        q = distribution_similarity(real_binned, dp_synthetic, shared_cols)
        m = run_mia(real_binned, dp_synthetic, numeric_cols)

        quality_scores.append(q * 100)
        mia_rates.append(m)

        print(f"{eps:<10} {q*100:<15.2f} {m:<15.2f}")

        dp_synthetic.to_csv(
            f"{BASE}/results/synthetic_dp_eps{str(eps).replace('.','_')}.csv",
            index=False
        )
    except Exception as e:
        print(f"Failed for epsilon={eps}: {e}")
        quality_scores.append(None)
        mia_rates.append(None)

# ── Plot results ───────────────────────────────────────────
valid = [(e, q, m) for e, q, m in zip(epsilons, quality_scores, mia_rates)
         if q is not None and m is not None]
eps_vals = [v[0] for v in valid]
q_vals   = [v[1] for v in valid]
m_vals   = [v[2] for v in valid]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Effect of Privacy Budget (ε) on Quality and Privacy Leakage",
             fontsize=13, fontweight='bold')

axes[0].plot(eps_vals, q_vals, color='#3B6E91', marker='o',
             linewidth=2, markersize=8)
axes[0].set_title('Data Quality vs Privacy Budget', fontweight='bold')
axes[0].set_xlabel('Epsilon (ε) — Privacy Budget')
axes[0].set_ylabel('Quality Score (%)')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(eps_vals, m_vals, color='#B5483D', marker='o',
             linewidth=2, markersize=8)
axes[1].set_title('Privacy Leakage vs Privacy Budget\nLower = More Private',
                   fontweight='bold')
axes[1].set_xlabel('Epsilon (ε) — Privacy Budget')
axes[1].set_ylabel('MIA Success Rate (%)')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/results/epsilon_sweep.png",
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

# ── Save results table ─────────────────────────────────────
results_df = pd.DataFrame({
    'Epsilon': eps_vals,
    'Quality_Score_%': q_vals,
    'MIA_Success_Rate_%': m_vals
})
results_df.to_csv(f"{BASE}/results/epsilon_sweep_results.csv", index=False)

print("\n" + "=" * 40)
print("EPSILON SWEEP COMPLETE")
print("=" * 40)
print(results_df.to_string(index=False))
print(f"\nResults saved to results/epsilon_sweep_results.csv")
print(f"Chart saved to results/epsilon_sweep.png")