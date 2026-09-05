import pandas as pd
import os
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/models", exist_ok=True)
os.makedirs(f"{BASE}/results", exist_ok=True)

df = pd.read_csv(f"{BASE}/data/creditcard.csv")
print(f"Full dataset shape: {df.shape}")

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=5000, random_state=42)
df_sample = pd.concat([fraud, legit]).reset_index(drop=True)

print(f"Training sample shape: {df_sample.shape}")
print(f"Class balance:\n{df_sample['Class'].value_counts()}")

df_sample.to_csv(f"{BASE}/data/creditcard_sample.csv", index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_sample)
metadata.update_column(column_name='Class', sdtype='categorical')

print("\nTraining CTGAN... this will take 10-20 minutes.")
synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=True)
synthesizer.fit(df_sample)

synthesizer.save(f"{BASE}/models/ctgan_baseline.pkl")
print("Model saved.")

synthetic_df = synthesizer.sample(num_rows=len(df_sample))
synthetic_df.to_csv(f"{BASE}/results/synthetic_baseline.csv", index=False)

print(f"\nDone!")
print(f"Synthetic data shape: {synthetic_df.shape}")
print(f"Synthetic class balance:\n{synthetic_df['Class'].value_counts()}")