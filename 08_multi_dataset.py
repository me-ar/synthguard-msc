import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from snsynth import Synthesizer

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

DATASETS = [
    {
        "name": "Credit Card Fraud",
        "file": "creditcard.csv",
        "label_col": "Class",
        "positive_label": 1,
        "separator": ",",
        "description": "European credit card transactions (fraud detection)"
    },
    {
        "name": "Credit Card Default",
        "file": "UCI_Credit_Card.csv",
        "label_col": "default.payment.next.month",
        "positive_label": 1,
        "separator": ",",
        "description": "Taiwan credit card default prediction"
    },
    {
        "name": "Bank Marketing",
        "file": "bank-additional-full.csv",
        "label_col": "y",
        "positive_label": "yes",
        "separator": ";",
        "description": "Bank telemarketing campaign subscription"
    },
]

def js_similarity(real_df, synth_df, cols):
    scores = []
    for col in cols:
        try:
            combined = pd.concat([real_df[col], synth_df[col]])
            bins = np.linspace(combined.min(), combined.max(), 21)
            r, _ = np.histogram(real_df[col], bins=bins, density=True)
            s, _ = np.histogram(synth_df[col], bins=bins, density=True)
            r += 1e-10
            s += 1e-10
            scores.append(1 - jensenshannon(r, s))
        except Exception:
            pass
    return float(np.mean(scores)) * 100 if scores else 0.0

def run_mia(real_df, synth_df, cols):
    try:
        r = real_df[cols].dropna().values
        s = synth_df[cols].dropna().values
        min_len = min(len(r), len(s))
        r, s = r[:min_len], s[:min_len]
        nn = NearestNeighbors(n_neighbors=1).fit(s)
        distances, _ = nn.kneighbors(r)
        distances = distances.flatten()
        threshold = np.percentile(distances, 5)
        return (distances < threshold).sum() / len(r) * 100
    except Exception as e:
        print(f"  MIA failed: {e}")
        return -1.0

def prepare_sample(df, label_col, pos_label, n_legit=5000):
    pos = df[df[label_col] == pos_label]
    neg = df[df[label_col] != pos_label]
    n = min(n_legit, len(neg))
    neg_sample = neg.sample(n=n, random_state=42)
    sample = pd.concat([pos, neg_sample]).reset_index(drop=True)
    print(f"  Sample: {len(sample)} rows ({len(pos)} positive + {n} negative)")
    return sample

def get_numeric_cols(df, label_col, max_cols=8):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)
    return num_cols[:max_cols]

def encode_categoricals(df, label_col):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df

all_results = []

print("\n" + "="*60)
print("MULTI-DATASET PRIVACY PIPELINE")
print("="*60)

for ds in DATASETS:
    print(f"\n{'='*60}")
    print(f"DATASET: {ds['name']}")
    print(f"{'='*60}")

    filepath = f"{BASE}/data/{ds['file']}"
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath} - skipping.")
        continue

    df = pd.read_csv(filepath, sep=ds['separator'])
    id_cols = [c for c in df.columns if c.lower() in ['id', 'unnamed: 0']]
    df.drop(columns=id_cols, inplace=True, errors='ignore')

    label_col = ds['label_col']
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    vc = df[label_col].value_counts()
    print(f"  Class balance:\n{vc.to_string()}")
    pos_pct = (df[label_col] == ds['positive_label']).mean() * 100
    print(f"  Positive class rate: {pos_pct:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc.plot(kind='bar', ax=axes[0], color=['steelblue', 'crimson'], edgecolor='none')
    axes[0].set_title(f"{ds['name']} - Class Distribution")
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    axes[1].pie(vc, labels=vc.index.astype(str),
                autopct='%1.2f%%', colors=['steelblue', 'crimson'])
    axes[1].set_title(f"{ds['name']} - Class Balance")
    plt.tight_layout()
    safe_name = ds['name'].lower().replace(' ', '_')
    plt.savefig(f"{BASE}/results/{safe_name}_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  EDA chart saved.")

    df_encoded = encode_categoricals(df, label_col)

    if df[label_col].dtype == object:
        cats = pd.Categorical(df[label_col]).categories.tolist()
        pos_label_encoded = cats.index(ds['positive_label'])
    else:
        pos_label_encoded = ds['positive_label']

    df_sample = prepare_sample(df_encoded, label_col, pos_label_encoded)

    print(f"  Training CTGAN...")
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_sample)
        metadata.update_column(column_name=label_col, sdtype='categorical')

        synth = CTGANSynthesizer(metadata, epochs=100, verbose=False)
        synth.fit(df_sample)
        synthetic_base = synth.sample(num_rows=len(df_sample))
        synthetic_base.to_csv(f"{BASE}/results/{safe_name}_synthetic.csv", index=False)
        print(f"  CTGAN done. Synthetic shape: {synthetic_base.shape}")
    except Exception as e:
        print(f"  CTGAN failed: {e}")
        continue

    num_cols = get_numeric_cols(df_sample, label_col)
    quality_base = js_similarity(df_sample, synthetic_base, num_cols)
    print(f"  Baseline quality score: {quality_base:.2f}%")

    mia_base = run_mia(df_sample[num_cols], synthetic_base[num_cols], num_cols)
    print(f"  Baseline MIA rate: {mia_base:.2f}%")

    print(f"  Training DP synthesizer (epsilon=1.0)...")
    try:
        selected = num_cols[:7] + [label_col]
        real_reduced = df_sample[selected].copy()

        for col in num_cols[:7]:
            try:
                real_reduced[col] = pd.qcut(
                    real_reduced[col], q=10,
                    labels=False, duplicates='drop'
                ).astype(int)
            except Exception:
                real_reduced[col] = pd.cut(
                    real_reduced[col], bins=10,
                    labels=False
                ).fillna(0).astype(int)

        dp_synth = Synthesizer.create('mwem', epsilon=1.0, split_factor=4)
        dp_synth.fit(real_reduced, preprocessor_eps=0.5)
        dp_synthetic = dp_synth.sample(len(real_reduced))
        dp_synthetic.to_csv(f"{BASE}/results/{safe_name}_synthetic_dp.csv", index=False)

        quality_dp = js_similarity(real_reduced, dp_synthetic, num_cols[:7])
        mia_dp = run_mia(real_reduced[num_cols[:7]], dp_synthetic[num_cols[:7]], num_cols[:7])
        print(f"  DP quality score: {quality_dp:.2f}%")
        print(f"  DP MIA rate: {mia_dp:.2f}%")

    except Exception as e:
        print(f"  DP synthesis failed: {e}")
        quality_dp = -1.0
        mia_dp = -1.0

    all_results.append({
        "Dataset": ds['name'],
        "Records": len(df),
        "Positive Rate (%)": round(pos_pct, 2),
        "Baseline Quality (%)": round(quality_base, 2),
        "Baseline MIA (%)": round(mia_base, 2),
        "DP Quality (%)": round(quality_dp, 2),
        "DP MIA (%)": round(mia_dp, 2),
    })

    print(f"  {ds['name']} complete.")

print(f"\n{'='*60}")
print("MASTER RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
results_df.to_csv(f"{BASE}/results/multi_dataset_results.csv", index=False)

valid = results_df[
    (results_df["Baseline Quality (%)"] > 0) &
    (results_df["DP Quality (%)"] > 0)
]

if len(valid) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Multi-Dataset Privacy-Utility Trade-off", fontsize=14, fontweight='bold')

    x = np.arange(len(valid))
    width = 0.35
    datasets = valid["Dataset"].tolist()

    b1 = axes[0].bar(x - width/2, valid["Baseline Quality (%)"],
                     width, label='Baseline CTGAN', color='#3B6E91', edgecolor='none')
    b2 = axes[0].bar(x + width/2, valid["DP Quality (%)"],
                     width, label='DP Synthesis (e=1.0)', color='#7A5C9E', edgecolor='none')
    axes[0].set_title('Data Quality Score', fontweight='bold')
    axes[0].set_ylabel('Quality Score (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=10, ha='right')
    axes[0].set_ylim(0, 115)
    axes[0].legend()
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f'{bar.get_height():.1f}%',
                     ha='center', fontsize=9, fontweight='bold')
    for bar in b2:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f'{bar.get_height():.1f}%',
                     ha='center', fontsize=9, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    b3 = axes[1].bar(x - width/2, valid["Baseline MIA (%)"],
                     width, label='Baseline CTGAN', color='#B5483D', edgecolor='none')
    b4 = axes[1].bar(x + width/2, valid["DP MIA (%)"],
                     width, label='DP Synthesis (e=1.0)', color='#4F8A6E', edgecolor='none')
    axes[1].set_title('Privacy Risk (MIA Rate)\nLower = More Private', fontweight='bold')
    axes[1].set_ylabel('MIA Success Rate (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=10, ha='right')
    axes[1].legend()
    for bar in b3:
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f'{bar.get_height():.2f}%',
                     ha='center', fontsize=9, fontweight='bold')
    for bar in b4:
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f'{bar.get_height():.2f}%',
                     ha='center', fontsize=9, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{BASE}/results/multi_dataset_comparison.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Master comparison chart saved.")

print("All datasets complete.")
print(f"Results saved to: {BASE}/results/multi_dataset_results.csv")