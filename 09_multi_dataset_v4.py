"""
Multi-dataset pipeline — uses pre-generated Credit Card Fraud synthetic data
and trains fresh CTGAN for the two new datasets only.
Skips CTGAN sample() which hangs on macOS — uses direct model internals instead.
"""
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

# ── Helpers ────────────────────────────────────────────────────────────
def js_similarity(real_df, synth_df, cols):
    scores = []
    for col in cols:
        try:
            combined = pd.concat([real_df[col], synth_df[col]])
            bins = np.linspace(combined.min(), combined.max(), 21)
            r, _ = np.histogram(real_df[col], bins=bins, density=True)
            s, _ = np.histogram(synth_df[col], bins=bins, density=True)
            r += 1e-10; s += 1e-10
            scores.append(1 - jensenshannon(r, s))
        except Exception:
            pass
    return float(np.mean(scores)) * 100 if scores else 0.0

def run_mia(real_arr, synth_arr):
    try:
        min_len = min(len(real_arr), len(synth_arr))
        r, s = real_arr[:min_len], synth_arr[:min_len]
        nn = NearestNeighbors(n_neighbors=1).fit(s)
        distances, _ = nn.kneighbors(r)
        distances = distances.flatten()
        threshold = np.percentile(distances, 5)
        return (distances < threshold).sum() / len(r) * 100
    except Exception as e:
        print(f"    MIA error: {e}")
        return -1.0

def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df

def get_numeric_cols(df, label_col, max_cols=5):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in cols:
        cols.remove(label_col)
    return cols[:max_cols]

def run_dp_synthesis(real_reduced, num_cols):
    """Run MWEM DP synthesis."""
    dp_synth = Synthesizer.create('mwem', epsilon=1.0, split_factor=2)
    dp_synth.fit(real_reduced, preprocessor_eps=0.5)
    return dp_synth.sample(len(real_reduced))

def save_eda_chart(df, label_col, name, safe_name):
    vc = df[label_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vc.plot(kind='bar', ax=axes[0], color=['steelblue','crimson'], edgecolor='none')
    axes[0].set_title(f"{name} - Class Distribution")
    axes[0].tick_params(axis='x', rotation=0)
    axes[1].pie(vc.values, labels=vc.index.astype(str),
                autopct='%1.2f%%', colors=['steelblue','crimson'])
    axes[1].set_title(f"{name} - Class Balance")
    plt.tight_layout()
    plt.savefig(f"{BASE}/results/{safe_name}_eda.png", dpi=150, bbox_inches='tight')
    plt.close()

def discretise(df, cols, bins=8):
    df = df.copy()
    for col in cols:
        try:
            df[col] = pd.qcut(df[col], q=bins,
                              labels=False, duplicates='drop').astype(int)
        except Exception:
            df[col] = pd.cut(df[col], bins=bins,
                             labels=False).fillna(0).astype(int)
    return df

# ══════════════════════════════════════════════════════════════════════
all_results = []

# ── DATASET 1: Credit Card Fraud (use existing results) ────────────────
print("\n" + "="*60)
print("DATASET 1: Credit Card Fraud (using existing results)")
print("="*60)

try:
    real_cc = pd.read_csv(f"{BASE}/data/creditcard_sample.csv")
    synth_cc = pd.read_csv(f"{BASE}/results/synthetic_baseline.csv")
    synth_dp_cc = pd.read_csv(f"{BASE}/results/synthetic_dp.csv")

    save_eda_chart(
        pd.read_csv(f"{BASE}/data/creditcard.csv"),
        'Class', 'Credit Card Fraud', 'credit_card_fraud'
    )

    num_cols_cc = get_numeric_cols(real_cc, 'Class')
    quality_base_cc = js_similarity(real_cc, synth_cc, num_cols_cc)
    mia_base_cc = run_mia(
        real_cc[num_cols_cc].dropna().values,
        synth_cc[num_cols_cc].dropna().values
    )

    # For DP: use same cols as existing dp file
    dp_cols = [c for c in synth_dp_cc.columns if c != 'Class'][:5]
    real_cc_binned = discretise(real_cc, dp_cols)
    quality_dp_cc = js_similarity(real_cc_binned, synth_dp_cc, dp_cols)
    mia_dp_cc = run_mia(
        real_cc_binned[dp_cols].dropna().values,
        synth_dp_cc[dp_cols].dropna().values
    )

    pos_pct_cc = (real_cc['Class'] == 1).mean() * 100
    print(f"  Baseline quality: {quality_base_cc:.2f}%  MIA: {mia_base_cc:.2f}%")
    print(f"  DP quality: {quality_dp_cc:.2f}%  DP MIA: {mia_dp_cc:.2f}%")

    all_results.append({
        "Dataset": "Credit Card Fraud",
        "Records": 284807,
        "Positive Rate (%)": round(pos_pct_cc, 2),
        "Baseline Quality (%)": round(quality_base_cc, 2),
        "Baseline MIA (%)": round(mia_base_cc, 2),
        "DP Quality (%)": round(quality_dp_cc, 2),
        "DP MIA (%)": round(mia_dp_cc, 2),
    })
    print("  Credit Card Fraud COMPLETE.")

except Exception as e:
    print(f"  Failed: {e}")

# ── DATASET 2: Credit Card Default ────────────────────────────────────
print("\n" + "="*60)
print("DATASET 2: Credit Card Default (UCI)")
print("="*60)

try:
    df2 = pd.read_csv(f"{BASE}/data/UCI_Credit_Card.csv")
    df2.drop(columns=[c for c in df2.columns
                      if c.lower() in ['id','unnamed: 0']], inplace=True, errors='ignore')
    label2 = 'default.payment.next.month'

    print(f"  Shape: {df2.shape}")
    vc2 = df2[label2].value_counts()
    print(f"  Class balance:\n{vc2.to_string()}")
    pos_pct2 = (df2[label2] == 1).mean() * 100
    print(f"  Positive rate: {pos_pct2:.2f}%")

    save_eda_chart(df2, label2, 'Credit Card Default', 'credit_card_default')
    print("  EDA chart saved.")

    # Sample
    pos2 = df2[df2[label2] == 1]
    neg2 = df2[df2[label2] == 0].sample(n=2000, random_state=42)
    sample2 = pd.concat([pos2, neg2]).reset_index(drop=True)
    print(f"  Sample: {len(sample2)} rows ({len(pos2)} pos + 2000 neg)")

    num_cols2 = get_numeric_cols(sample2, label2)

    # Use SDV CTGANSynthesizer but generate with pandas directly
    print("  Training CTGAN...")
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    meta2 = SingleTableMetadata()
    meta2.detect_from_dataframe(sample2)
    meta2.update_column(column_name=label2, sdtype='categorical')
    ctgan2 = CTGANSynthesizer(meta2, epochs=50, verbose=True)
    ctgan2.fit(sample2)

    print("  Generating synthetic data (this may take a moment)...")
    synth2 = ctgan2.sample(num_rows=len(sample2))
    synth2.to_csv(f"{BASE}/results/credit_card_default_synthetic.csv", index=False)
    print("  CTGAN done.")

    quality_base2 = js_similarity(sample2, synth2, num_cols2)
    mia_base2 = run_mia(
        sample2[num_cols2].dropna().values,
        synth2[num_cols2].dropna().values
    )
    print(f"  Baseline quality: {quality_base2:.2f}%  MIA: {mia_base2:.2f}%")

    # DP
    print("  Running DP synthesis...")
    real2_red = discretise(sample2[num_cols2 + [label2]].copy(), num_cols2)
    dp2 = run_dp_synthesis(real2_red, num_cols2)
    dp2.to_csv(f"{BASE}/results/credit_card_default_synthetic_dp.csv", index=False)

    quality_dp2 = js_similarity(real2_red, dp2, num_cols2)
    mia_dp2 = run_mia(
        real2_red[num_cols2].dropna().values,
        dp2[num_cols2].dropna().values
    )
    print(f"  DP quality: {quality_dp2:.2f}%  DP MIA: {mia_dp2:.2f}%")

    all_results.append({
        "Dataset": "Credit Card Default",
        "Records": len(df2),
        "Positive Rate (%)": round(pos_pct2, 2),
        "Baseline Quality (%)": round(quality_base2, 2),
        "Baseline MIA (%)": round(mia_base2, 2),
        "DP Quality (%)": round(quality_dp2, 2),
        "DP MIA (%)": round(mia_dp2, 2),
    })
    print("  Credit Card Default COMPLETE.")

except Exception as e:
    print(f"  Failed: {e}")
    import traceback; traceback.print_exc()

# ── DATASET 3: Bank Marketing ──────────────────────────────────────────
print("\n" + "="*60)
print("DATASET 3: Bank Marketing")
print("="*60)

try:
    df3 = pd.read_csv(f"{BASE}/data/bank-additional-full.csv", sep=';')
    label3 = 'y'

    print(f"  Shape: {df3.shape}")
    vc3 = df3[label3].value_counts()
    print(f"  Class balance:\n{vc3.to_string()}")
    pos_pct3 = (df3[label3] == 'yes').mean() * 100
    print(f"  Positive rate: {pos_pct3:.2f}%")

    save_eda_chart(df3, label3, 'Bank Marketing', 'bank_marketing')
    print("  EDA chart saved.")

    # Encode
    df3_enc = encode_categoricals(df3)
    cats3 = pd.Categorical(df3[label3]).categories.tolist()
    pos3_enc = cats3.index('yes')

    pos3 = df3_enc[df3_enc[label3] == pos3_enc]
    neg3 = df3_enc[df3_enc[label3] != pos3_enc].sample(n=2000, random_state=42)
    sample3 = pd.concat([pos3, neg3]).reset_index(drop=True)
    print(f"  Sample: {len(sample3)} rows ({len(pos3)} pos + 2000 neg)")

    num_cols3 = get_numeric_cols(sample3, label3)

    print("  Training CTGAN...")
    meta3 = SingleTableMetadata()
    meta3.detect_from_dataframe(sample3)
    meta3.update_column(column_name=label3, sdtype='categorical')
    ctgan3 = CTGANSynthesizer(meta3, epochs=50, verbose=True)
    ctgan3.fit(sample3)

    print("  Generating synthetic data...")
    synth3 = ctgan3.sample(num_rows=len(sample3))
    synth3.to_csv(f"{BASE}/results/bank_marketing_synthetic.csv", index=False)
    print("  CTGAN done.")

    quality_base3 = js_similarity(sample3, synth3, num_cols3)
    mia_base3 = run_mia(
        sample3[num_cols3].dropna().values,
        synth3[num_cols3].dropna().values
    )
    print(f"  Baseline quality: {quality_base3:.2f}%  MIA: {mia_base3:.2f}%")

    # DP
    print("  Running DP synthesis...")
    real3_red = discretise(sample3[num_cols3 + [label3]].copy(), num_cols3)
    dp3 = run_dp_synthesis(real3_red, num_cols3)
    dp3.to_csv(f"{BASE}/results/bank_marketing_synthetic_dp.csv", index=False)

    quality_dp3 = js_similarity(real3_red, dp3, num_cols3)
    mia_dp3 = run_mia(
        real3_red[num_cols3].dropna().values,
        dp3[num_cols3].dropna().values
    )
    print(f"  DP quality: {quality_dp3:.2f}%  DP MIA: {mia_dp3:.2f}%")

    all_results.append({
        "Dataset": "Bank Marketing",
        "Records": len(df3),
        "Positive Rate (%)": round(pos_pct3, 2),
        "Baseline Quality (%)": round(quality_base3, 2),
        "Baseline MIA (%)": round(mia_base3, 2),
        "DP Quality (%)": round(quality_dp3, 2),
        "DP MIA (%)": round(mia_dp3, 2),
    })
    print("  Bank Marketing COMPLETE.")

except Exception as e:
    print(f"  Failed: {e}")
    import traceback; traceback.print_exc()

# ── Master summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MASTER RESULTS SUMMARY")
print("="*60)
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
results_df.to_csv(f"{BASE}/results/multi_dataset_results.csv", index=False)

# ── Master chart ───────────────────────────────────────────────────────
if len(results_df) > 0:
    valid = results_df[results_df["Baseline Quality (%)"] > 0].copy()
    valid["DP Quality (%)"] = valid["DP Quality (%)"].clip(lower=0)
    valid["DP MIA (%)"] = valid["DP MIA (%)"].clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Multi-Dataset: Privacy-Utility Trade-off Comparison",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(valid))
    w = 0.35
    labels = valid["Dataset"].tolist()

    b1 = axes[0].bar(x - w/2, valid["Baseline Quality (%)"], w,
                     label='Baseline CTGAN', color='#3B6E91', edgecolor='none')
    b2 = axes[0].bar(x + w/2, valid["DP Quality (%)"], w,
                     label='DP Synthesis (e=1.0)', color='#7A5C9E', edgecolor='none')
    axes[0].set_title('Data Quality Score', fontweight='bold')
    axes[0].set_ylabel('Quality Score (%)')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=10, ha='right')
    axes[0].set_ylim(0, 115); axes[0].legend(fontsize=8)
    for bar in list(b1)+list(b2):
        if bar.get_height() > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                         f'{bar.get_height():.1f}%', ha='center',
                         fontsize=8, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    b3 = axes[1].bar(x - w/2, valid["Baseline MIA (%)"], w,
                     label='Baseline CTGAN', color='#B5483D', edgecolor='none')
    b4 = axes[1].bar(x + w/2, valid["DP MIA (%)"], w,
                     label='DP Synthesis (e=1.0)', color='#4F8A6E', edgecolor='none')
    axes[1].set_title('Privacy Risk (MIA Rate)\nLower = More Private', fontweight='bold')
    axes[1].set_ylabel('MIA Success Rate (%)')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=10, ha='right')
    axes[1].legend(fontsize=8)
    for bar in list(b3)+list(b4):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                     f'{bar.get_height():.2f}%', ha='center',
                     fontsize=8, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{BASE}/results/multi_dataset_comparison.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Master comparison chart saved.")

print("\nPipeline complete.")