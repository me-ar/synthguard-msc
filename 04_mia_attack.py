import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

real = pd.read_csv(f"{BASE}/data/creditcard_sample.csv").drop(columns=['Class'])
synthetic = pd.read_csv(f"{BASE}/results/synthetic_baseline.csv").drop(columns=['Class'])

print(f"Real records:      {len(real)}")
print(f"Synthetic records: {len(synthetic)}")

print("\nRunning Membership Inference Attack...")
nn = NearestNeighbors(n_neighbors=1).fit(synthetic.values)
distances, _ = nn.kneighbors(real.values)
distances = distances.flatten()

threshold = np.percentile(distances, 5)
flagged = (distances < threshold).sum()

print(f"\nTotal real records:       {len(real)}")
print(f"Flagged as privacy risk:  {flagged}")
print(f"MIA success rate:         {flagged / len(real) * 100:.2f}%")
print(f"Distance threshold used:  {threshold:.4f}")

plt.figure(figsize=(10, 4))
plt.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='none')
plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2,
            label=f'Risk threshold ({threshold:.2f})')
plt.xlabel('Distance to Nearest Synthetic Record')
plt.ylabel('Frequency')
plt.title('Membership Inference Attack — Distance Distribution\n(Baseline CTGAN)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{BASE}/results/mia_distances.png", dpi=200, bbox_inches='tight')
plt.show()

with open(f"{BASE}/results/mia_result.txt", 'w') as f:
    f.write(f"MIA Success Rate (Baseline CTGAN): {flagged / len(real) * 100:.2f}%\n")
    f.write(f"Records flagged: {flagged} / {len(real)}\n")
    f.write(f"Distance threshold: {threshold:.4f}\n")

print("\nAttack complete. Results saved to results folder.")