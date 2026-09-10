"""
Multi-dataset privacy pipeline — FINAL VERSION
Uses GaussianCopula (no multiprocessing) for new datasets.
Reuses existing Credit Card Fraud results from earlier files.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from snsynth import Synthesizer

BASE = "/Users/adeshr/Desktop/addessshhhh/Projects/Adesh Desertation Project/Desertation"
os.makedirs(f"{BASE}/results", exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────
def js_quality(real, synth, cols):
    scores = []
    for col in cols:
        try:
            combined = pd.concat([real[col], synth[col]])
            bins = np.linspace(combined.min(), combined.max(), 21)
            r, _ = np.histogram(real[col], bins=bins, density=True)
            s, _ = np.histogram(synth[col], bins=bins, density=True)
            r += 1e-10; s += 1e-10
            scores.append(1 - jensenshannon(r, s))
        except:
            pass
    return round(float(np.mean(scores)) * 100, 2) if scores else 0.0

def mia_attack(real_arr, synth_arr):
    try:
        n = min(len(real_arr), len(synth_arr))
        r, s = real_arr[:n], synth_arr[:n]
        nn = NearestNeighbors(n_neighbors=1).fit(s)
        dist, _ = nn.kneighbors(r)
        dist = dist.flatten()
        t = np.percentile(dist, 5)
        return round((dist < t).sum() / len(r) * 100, 2)
    except Exception as e:
        print(f"    MIA error: {e}")
        return -1.0

def discretise(df, cols, bins=8):
    df = df.copy()
    for col in cols:
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=False,
                              duplicates='drop').astype(int)
        except:
            df[col] = pd.cut(df[col], bins=bins,
                             labels=False).fillna(0).astype(int)
    return df

def encode_cats(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df

def get_num_cols(df, label, n=5):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label in cols:
        cols.remove(label)
    return cols[:n]

def eda_chart(df, label, name, safe):
    vc = df[label].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vc.plot(kind='bar', ax=axes[0], color=['steelblue','crimson'], edgecolor='none')
    axes[0].set_title(f'{name} — Class Distribution')
    axes[0].tick_params(axis='x', rotation=0)
    axes[1].pie(vc.values, labels=vc.index.astype(str),
                autopct='%1.2f%%', colors=['steelblue','crimson'])
    axes[1].set_title(f'{name} — Class Balance')
    plt.tight_layout()
    plt.savefig(f"{BASE}/results/{safe}_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  EDA chart saved.")

def run_dp(real_red, n_rows):
    dp = Synthesizer.create('mwem', epsilon=1.0, split_factor=2)
    dp.fit(real_red, preprocessor_eps=0.5)
    return dp.sample(n_rows)

results = []

# ══════════════════════════════════════════════════════════════════════
# DATASET 1 — Credit Card Fraud (reuse existing files)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 1: Credit Card Fraud (existing results)")
print("="*55)
try:
    real1   = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")
    synth1  = pd.read_csv(f"{BASE}/results/synthetic_baseline.csv")
    synth1d = pd.read_csv(f"{BASE}/results/synthetic_dp.csv")

    eda_chart(pd.read_csv(f"{BASE}/data/creditcard.csv"),
              'Class', 'Credit Card Fraud', 'credit_card_fraud')

    nc1 = get_num_cols(real1, 'Class')
    q1  = js_quality(real1, synth1, nc1)
    m1  = mia_attack(real1[nc1].dropna().values,
                     synth1[nc1].dropna().values)

    dp_cols1 = [c for c in synth1d.columns if c != 'Class'][:5]
    r1b = discretise(real1[dp_cols1 + ['Class']].copy(), dp_cols1)
    q1d = js_quality(r1b, synth1d, dp_cols1)
    m1d = mia_attack(r1b[dp_cols1].dropna().values,
                     synth1d[dp_cols1].dropna().values)

    print(f"  Baseline — Quality: {q1}%  MIA: {m1}%")
    print(f"  DP       — Quality: {q1d}%  MIA: {m1d}%")
    results.append({"Dataset":"Credit Card Fraud","Records":284807,
                    "Positive %":0.17,"Base Q":q1,"Base MIA":m1,
                    "DP Q":q1d,"DP MIA":m1d})
    print("  COMPLETE.")
except Exception as e:
    print(f"  Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# DATASET 2 — Credit Card Default
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 2: Credit Card Default (UCI)")
print("="*55)
try:
    df2 = pd.read_csv(f"{BASE}/data/UCI_Credit_Card.csv")
    df2.drop(columns=[c for c in df2.columns
                      if c.lower() in ['id','unnamed: 0']],
             inplace=True, errors='ignore')
    L2 = 'default.payment.next.month'

    print(f"  Shape: {df2.shape}")
    print(f"  Class balance:\n{df2[L2].value_counts().to_string()}")
    pos_pct2 = round((df2[L2]==1).mean()*100, 2)
    print(f"  Positive rate: {pos_pct2}%")

    eda_chart(df2, L2, 'Credit Card Default', 'credit_card_default')

    # Sample
    pos2 = df2[df2[L2]==1]
    neg2 = df2[df2[L2]==0].sample(n=2000, random_state=42)
    s2   = pd.concat([pos2, neg2]).reset_index(drop=True)
    print(f"  Sample: {len(s2)} rows")
    nc2 = get_num_cols(s2, L2)

    # GaussianCopula — no multiprocessing, no hang
    print("  Training GaussianCopula synthesizer...")
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    meta2 = SingleTableMetadata()
    meta2.detect_from_dataframe(s2)
    meta2.update_column(column_name=L2, sdtype='categorical')
    gc2 = GaussianCopulaSynthesizer(meta2)
    gc2.fit(s2)
    print("  Generating synthetic data...")
    synth2 = gc2.sample(num_rows=len(s2))
    synth2.to_csv(f"{BASE}/results/credit_card_default_synthetic.csv", index=False)
    print("  Done.")

    q2  = js_quality(s2, synth2, nc2)
    m2  = mia_attack(s2[nc2].dropna().values,
                     synth2[nc2].dropna().values)
    print(f"  Baseline — Quality: {q2}%  MIA: {m2}%")

    # DP
    print("  Running DP synthesis...")
    r2b = discretise(s2[nc2+[L2]].copy(), nc2)
    dp2 = run_dp(r2b, len(r2b))
    dp2.to_csv(f"{BASE}/results/credit_card_default_synthetic_dp.csv", index=False)
    q2d = js_quality(r2b, dp2, nc2)
    m2d = mia_attack(r2b[nc2].dropna().values, dp2[nc2].dropna().values)
    print(f"  DP       — Quality: {q2d}%  MIA: {m2d}%")

    results.append({"Dataset":"Credit Card Default","Records":len(df2),
                    "Positive %":pos_pct2,"Base Q":q2,"Base MIA":m2,
                    "DP Q":q2d,"DP MIA":m2d})
    print("  COMPLETE.")
except Exception as e:
    print(f"  Error: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# DATASET 3 — Bank Marketing
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 3: Bank Marketing")
print("="*55)
try:
    df3 = pd.read_csv(f"{BASE}/data/bank-additional-full.csv", sep=';')
    L3  = 'y'

    print(f"  Shape: {df3.shape}")
    print(f"  Class balance:\n{df3[L3].value_counts().to_string()}")
    pos_pct3 = round((df3[L3]=='yes').mean()*100, 2)
    print(f"  Positive rate: {pos_pct3}%")

    eda_chart(df3, L3, 'Bank Marketing', 'bank_marketing')

    df3e = encode_cats(df3)
    cats3 = pd.Categorical(df3[L3]).categories.tolist()
    pe3   = cats3.index('yes')

    pos3 = df3e[df3e[L3]==pe3]
    neg3 = df3e[df3e[L3]!=pe3].sample(n=2000, random_state=42)
    s3   = pd.concat([pos3, neg3]).reset_index(drop=True)
    print(f"  Sample: {len(s3)} rows")
    nc3 = get_num_cols(s3, L3)

    print("  Training GaussianCopula synthesizer...")
    meta3 = SingleTableMetadata()
    meta3.detect_from_dataframe(s3)
    meta3.update_column(column_name=L3, sdtype='categorical')
    gc3 = GaussianCopulaSynthesizer(meta3)
    gc3.fit(s3)
    print("  Generating synthetic data...")
    synth3 = gc3.sample(num_rows=len(s3))
    synth3.to_csv(f"{BASE}/results/bank_marketing_synthetic.csv", index=False)
    print("  Done.")

    q3  = js_quality(s3, synth3, nc3)
    m3  = mia_attack(s3[nc3].dropna().values,
                     synth3[nc3].dropna().values)
    print(f"  Baseline — Quality: {q3}%  MIA: {m3}%")

    print("  Running DP synthesis...")
    r3b = discretise(s3[nc3+[L3]].copy(), nc3)
    dp3 = run_dp(r3b, len(r3b))
    dp3.to_csv(f"{BASE}/results/bank_marketing_synthetic_dp.csv", index=False)
    q3d = js_quality(r3b, dp3, nc3)
    m3d = mia_attack(r3b[nc3].dropna().values, dp3[nc3].dropna().values)
    print(f"  DP       — Quality: {q3d}%  MIA: {m3d}%")

    results.append({"Dataset":"Bank Marketing","Records":len(df3),
                    "Positive %":pos_pct3,"Base Q":q3,"Base MIA":m3,
                    "DP Q":q3d,"DP MIA":m3d})
    print("  COMPLETE.")
except Exception as e:
    print(f"  Error: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# MASTER SUMMARY + CHART
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MASTER RESULTS SUMMARY")
print("="*55)

rdf = pd.DataFrame(results)
rdf.columns = ["Dataset","Records","Positive %",
               "Baseline Quality (%)","Baseline MIA (%)",
               "DP Quality (%)","DP MIA (%)"]
print(rdf.to_string(index=False))
rdf.to_csv(f"{BASE}/results/multi_dataset_results.csv", index=False)

if len(rdf) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Multi-Dataset: Privacy-Utility Trade-off",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(rdf))
    w = 0.35
    labels = rdf["Dataset"].tolist()

    b1 = axes[0].bar(x-w/2, rdf["Baseline Quality (%)"], w,
                     label='Baseline (CTGAN/GC)', color='#3B6E91', edgecolor='none')
    b2 = axes[0].bar(x+w/2, rdf["DP Quality (%)"].clip(lower=0), w,
                     label='DP Synthesis (e=1.0)', color='#7A5C9E', edgecolor='none')
    axes[0].set_title('Data Quality Score', fontweight='bold')
    axes[0].set_ylabel('Quality Score (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10, ha='right')
    axes[0].set_ylim(0, 115)
    axes[0].legend(fontsize=8)
    for bar in list(b1)+list(b2):
        if bar.get_height() > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+1,
                         f'{bar.get_height():.1f}%',
                         ha='center', fontsize=8, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    b3 = axes[1].bar(x-w/2, rdf["Baseline MIA (%)"], w,
                     label='Baseline (CTGAN/GC)', color='#B5483D', edgecolor='none')
    b4 = axes[1].bar(x+w/2, rdf["DP MIA (%)"].clip(lower=0), w,
                     label='DP Synthesis (e=1.0)', color='#4F8A6E', edgecolor='none')
    axes[1].set_title('Privacy Risk (MIA Rate)\nLower = More Private',
                      fontweight='bold')
    axes[1].set_ylabel('MIA Success Rate (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10, ha='right')
    axes[1].legend(fontsize=8)
    for bar in list(b3)+list(b4):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05,
                     f'{bar.get_height():.2f}%',
                     ha='center', fontsize=8, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{BASE}/results/multi_dataset_comparison.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Master chart saved to results/multi_dataset_comparison.png")

print("\nPipeline complete.")