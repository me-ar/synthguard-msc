import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

df = pd.read_csv(f"{BASE}/data/creditcard.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape:            {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass balance:\n{df['Class'].value_counts()}")
print(f"\nFraud percentage: {round(df['Class'].mean() * 100, 4)}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Class'].value_counts().plot(kind='bar', ax=axes[0],
    color=['steelblue', 'crimson'], edgecolor='none')
axes[0].set_title('Transaction Class Distribution')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[0].set_ylabel('Count')
axes[1].pie(df['Class'].value_counts(),
    labels=['Legitimate', 'Fraud'],
    autopct='%1.2f%%', colors=['steelblue', 'crimson'])
axes[1].set_title('Class Balance')
plt.tight_layout()
plt.savefig(f"{BASE}/results/class_distribution.png", dpi=200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6, label='Legitimate', color='steelblue', ax=ax)
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6, label='Fraud', color='crimson', ax=ax)
ax.set_xlabel('Transaction Amount')
ax.set_ylabel('Frequency')
ax.set_title('Transaction Amount Distribution by Class')
ax.legend()
plt.tight_layout()
plt.savefig(f"{BASE}/results/amount_distribution.png", dpi=200, bbox_inches='tight')
plt.show()

print("\nEDA complete. Charts saved to results folder.")