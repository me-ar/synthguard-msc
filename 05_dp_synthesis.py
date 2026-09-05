import pandas as pd
import numpy as np
import os
from snsynth import Synthesizer

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

real = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")
print(f"Original shape: {real.shape}")

selected_columns = ['V1', 'V2', 'V3', 'V4', 'V14', 'V17', 'Amount', 'Class']
real_reduced = real[selected_columns].copy()

for col in ['V1', 'V2', 'V3', 'V4', 'V14', 'V17', 'Amount']:
    real_reduced[col] = pd.qcut(
        real_reduced[col], q=10,
        labels=False, duplicates='drop'
    ).astype(int)

print(f"Reduced and binned shape: {real_reduced.shape}")
print(real_reduced.head())

epsilon = 1.0
print(f"\nTraining DP synthesizer (epsilon={epsilon})...")
dp_synth = Synthesizer.create('mwem', epsilon=epsilon, split_factor=4)
dp_synth.fit(real_reduced, preprocessor_eps=0.5)

dp_synthetic = dp_synth.sample(len(real_reduced))
dp_synthetic.to_csv(f"{BASE}/results/synthetic_dp.csv", index=False)

print(f"\nDone!")
print(f"DP synthetic shape: {dp_synthetic.shape}")
print(f"DP class balance:\n{dp_synthetic['Class'].value_counts()}")