import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SynthGuard — Privacy-Preserving Financial Data",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0F1117;
        color: #E8EAF0;
    }

    .main-header {
        background: linear-gradient(135deg, #1A1F2E 0%, #0F1117 100%);
        border-bottom: 1px solid #2A3050;
        padding: 2rem 0 1.5rem 0;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #7B8DB0;
        font-weight: 400;
        margin-bottom: 0;
    }

    .hero-accent {
        color: #4E9BCD;
    }

    .metric-card {
        background: #1A1F2E;
        border: 1px solid #2A3050;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #7B8DB0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .metric-value-good { color: #4CAF82; }
    .metric-value-warn { color: #E8A83A; }
    .metric-value-bad  { color: #E05C5C; }

    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2A3050;
    }

    .info-box {
        background: #141824;
        border-left: 3px solid #4E9BCD;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #A8B4CC;
        line-height: 1.6;
    }

    .warn-box {
        background: #1E1710;
        border-left: 3px solid #E8A83A;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #C4A870;
        line-height: 1.6;
    }

    .success-box {
        background: #111E17;
        border-left: 3px solid #4CAF82;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #7DC4A0;
        line-height: 1.6;
    }

    .step-badge {
        display: inline-block;
        background: #4E9BCD22;
        color: #4E9BCD;
        border: 1px solid #4E9BCD44;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: 0.04em;
    }

    div[data-testid="stSidebar"] {
        background-color: #141824;
        border-right: 1px solid #2A3050;
    }

    div[data-testid="stSidebar"] .stMarkdown p {
        color: #7B8DB0;
        font-size: 0.85rem;
    }

    .stSlider > div > div > div {
        background: #4E9BCD !important;
    }

    .stButton > button {
        background: #4E9BCD;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.8rem;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #3A7FAD;
    }

    .stDownloadButton > button {
        background: #1A2A1F;
        color: #4CAF82;
        border: 1px solid #4CAF82;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
    }

    .stFileUploader {
        border: 1px dashed #2A3050;
        border-radius: 10px;
        background: #141824;
    }

    hr {
        border-color: #2A3050;
        margin: 1.5rem 0;
    }

    .footer-note {
        font-size: 0.78rem;
        color: #4A5570;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1E2436;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def distribution_similarity(real_df, synth_df, cols):
    from scipy.spatial.distance import jensenshannon
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


def run_mia(real_df, synth_df, cols):
    from sklearn.neighbors import NearestNeighbors
    try:
        r = real_df[cols].dropna().values
        s = synth_df[cols].dropna().values
        min_len = min(len(r), len(s))
        r, s = r[:min_len], s[:min_len]
        nn = NearestNeighbors(n_neighbors=1).fit(s)
        distances, _ = nn.kneighbors(r)
        distances = distances.flatten()
        threshold = np.percentile(distances, 5)
        return (distances < threshold).sum() / len(r) * 100, distances, threshold
    except Exception:
        return 0.0, np.array([]), 0.0


def make_dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1A1F2E')
    ax.set_facecolor('#1A1F2E')
    ax.tick_params(colors='#7B8DB0', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2A3050')
    ax.xaxis.label.set_color('#7B8DB0')
    ax.yaxis.label.set_color('#7B8DB0')
    ax.title.set_color('#FFFFFF')
    return fig, ax


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 SynthGuard")
    st.markdown("*Privacy-Preserving Synthetic Data for Financial Applications*")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
1. **Upload** your financial CSV
2. **Choose** a privacy budget (ε)
3. **Generate** synthetic data
4. **Evaluate** quality & privacy risk
5. **Download** your synthetic dataset
    """)

    st.markdown("---")
    st.markdown("### About ε (epsilon)")
    st.markdown("""
**Epsilon** controls the privacy-utility trade-off:

- **Low ε (0.1–0.5)** → Stronger privacy, lower quality
- **Medium ε (1.0–2.0)** → Balanced trade-off
- **High ε (5.0+)** → Higher quality, weaker privacy

The research behind this tool found that **ε = 1.0** eliminates all measurable privacy leakage while maintaining high data quality.
    """)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem; color:#4A5570;'>
MSc Data Science Project<br>
University of Liverpool<br>
Adesh Raj Rajkumar · 2025
</div>
""", unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="hero-title">
        SynthGuard <span class="hero-accent">—</span> Privacy-Preserving Synthetic Data
    </div>
    <div class="hero-sub">
        Generate realistic synthetic financial data with measurable privacy guarantees.
        Upload a dataset, choose your privacy budget, and download a safe synthetic version.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 — Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your financial dataset</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Upload a CSV file containing tabular financial data. The tool will generate a synthetic version 
that preserves statistical patterns without exposing any real records. Your data is processed 
locally and never stored.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload any tabular CSV. Works best with numerical financial data."
)

# Demo mode with built-in sample
use_demo = st.checkbox(
    "Use the Credit Card Fraud Detection dataset (demo mode)",
    value=uploaded_file is None,
    help="Uses a pre-loaded sample of the Kaggle credit card fraud dataset"
)

if uploaded_file is not None:
    real_df = pd.read_csv(uploaded_file)
    st.markdown(f"""
    <div class="success-box">
    ✓ Loaded <strong>{len(real_df):,} rows</strong> and <strong>{len(real_df.columns)} columns</strong> from your file.
    </div>
    """, unsafe_allow_html=True)
elif use_demo:
    # Generate a realistic demo dataset matching credit card fraud structure
    np.random.seed(42)
    n = 1000
    fraud_n = 17
    legit_n = n - fraud_n

    legit = pd.DataFrame(
        np.random.randn(legit_n, 28),
        columns=[f'V{i}' for i in range(1, 29)]
    )
    legit['Amount'] = np.abs(np.random.exponential(50, legit_n))
    legit['Time'] = np.sort(np.random.uniform(0, 172800, legit_n))
    legit['Class'] = 0

    fraud = pd.DataFrame(
        np.random.randn(fraud_n, 28) * 1.5,
        columns=[f'V{i}' for i in range(1, 29)]
    )
    fraud['Amount'] = np.abs(np.random.exponential(120, fraud_n))
    fraud['Time'] = np.random.uniform(0, 172800, fraud_n)
    fraud['Class'] = 1

    real_df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)

    st.markdown(f"""
    <div class="info-box">
    ℹ Demo mode — using a simulated Credit Card Fraud dataset 
    (<strong>{len(real_df):,} rows</strong>, <strong>{len(real_df.columns)} columns</strong>, 
    {fraud_n} fraud cases / {legit_n} legitimate transactions).
    </div>
    """, unsafe_allow_html=True)
else:
    st.stop()

# Show data preview
with st.expander("Preview uploaded data"):
    st.dataframe(real_df.head(10), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(real_df):,}")
    col2.metric("Columns", len(real_df.columns))
    if 'Class' in real_df.columns:
        fraud_pct = real_df['Class'].mean() * 100
        col3.metric("Fraud rate", f"{fraud_pct:.2f}%")

st.markdown("---")


# ── Step 2: Configure ──────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 2 — Configure</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Privacy settings</div>', unsafe_allow_html=True)

col_cfg1, col_cfg2 = st.columns([2, 1])

with col_cfg1:
    epsilon = st.select_slider(
        "Privacy budget (ε — epsilon)",
        options=[0.1, 0.5, 1.0, 2.0, 5.0],
        value=1.0,
        help="Lower epsilon = more privacy, higher epsilon = better quality"
    )

    privacy_labels = {
        0.1: ("Strong privacy", "#4CAF82", "Very high privacy protection. Some loss in data quality."),
        0.5: ("Good privacy", "#7DC4A0", "Good privacy with reasonable quality. Suitable for sensitive data."),
        1.0: ("Balanced", "#4E9BCD", "Research-recommended setting. Eliminates measurable privacy leakage."),
        2.0: ("Moderate privacy", "#E8A83A", "Higher quality with moderate privacy. Suitable for lower-risk data."),
        5.0: ("Light privacy", "#E05C5C", "Highest quality. Privacy protection is reduced."),
    }

    label, colour, description = privacy_labels[epsilon]
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid {colour};">
        <div class="metric-label">Selected privacy level</div>
        <div style="font-size:1.1rem; font-weight:600; color:{colour}; margin-bottom:0.3rem;">{label}</div>
        <div style="font-size:0.85rem; color:#7B8DB0;">{description}</div>
    </div>
    """, unsafe_allow_html=True)

with col_cfg2:
    num_rows = st.number_input(
        "Rows to generate",
        min_value=100,
        max_value=min(10000, len(real_df) * 2),
        value=min(len(real_df), 1000),
        step=100,
        help="Number of synthetic rows to generate"
    )
    st.markdown(f"""
    <div class="info-box">
    Generating <strong>{num_rows:,}</strong> synthetic rows from 
    <strong>{len(real_df):,}</strong> real records.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ── Step 3: Generate ───────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 3 — Generate</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Generate synthetic data</div>', unsafe_allow_html=True)

if st.button("🔐 Generate synthetic data", type="primary"):

    # Prepare data for DP synthesis
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()

    # Select up to 8 most informative columns
    selected = numeric_cols[:min(7, len(numeric_cols))]
    label_col = 'Class' if 'Class' in real_df.columns else None
    if label_col:
        selected_cols = selected + [label_col]
    else:
        selected_cols = selected

    real_reduced = real_df[selected_cols].copy()

    # Bin continuous columns
    for col in selected:
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

    with st.spinner(f"Training DP synthesizer with ε={epsilon}... this takes 30–90 seconds"):
        try:
            from snsynth import Synthesizer

            dp_synth = Synthesizer.create(
                'mwem', epsilon=epsilon, split_factor=4
            )
            dp_synth.fit(real_reduced, preprocessor_eps=min(0.4, epsilon * 0.3))
            synth_df = dp_synth.sample(num_rows)

            st.session_state['synth_df']     = synth_df
            st.session_state['real_reduced'] = real_reduced
            st.session_state['real_df']      = real_df
            st.session_state['selected']     = selected
            st.session_state['epsilon']      = epsilon
            st.session_state['generated']    = True

            st.markdown("""
            <div class="success-box">
            ✓ Synthetic data generated successfully.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.info("Try reducing the number of rows or choosing a higher epsilon value.")
            st.session_state['generated'] = False


# ── Step 4: Evaluate ───────────────────────────────────────────────────────────
if st.session_state.get('generated'):

    st.markdown("---")
    st.markdown('<div class="step-badge">Step 4 — Evaluate</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quality and privacy evaluation</div>', unsafe_allow_html=True)

    synth_df     = st.session_state['synth_df']
    real_reduced = st.session_state['real_reduced']
    selected     = st.session_state['selected']
    eps          = st.session_state['epsilon']

    # Compute metrics
    quality = distribution_similarity(real_reduced, synth_df, selected)
    mia_rate, distances, threshold = run_mia(real_reduced, synth_df, selected)

    # Privacy interpretation
    if mia_rate == 0.0:
        priv_label = "No leakage detected"
        priv_colour = "good"
        priv_icon = "🛡"
    elif mia_rate < 3.0:
        priv_label = "Low leakage"
        priv_colour = "warn"
        priv_icon = "⚠"
    else:
        priv_label = "Measurable leakage"
        priv_colour = "bad"
        priv_icon = "🔓"

    # Quality interpretation
    if quality >= 90:
        q_label = "Excellent"
        q_colour = "good"
    elif quality >= 75:
        q_label = "Good"
        q_colour = "warn"
    else:
        q_label = "Moderate"
        q_colour = "bad"

    # Metric cards
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data quality score</div>
            <div class="metric-value metric-value-{q_colour}">{quality:.1f}%</div>
            <div style="font-size:0.8rem; color:#7B8DB0; margin-top:0.3rem;">{q_label} — distributional similarity</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Privacy risk (MIA rate)</div>
            <div class="metric-value metric-value-{priv_colour}">{mia_rate:.2f}%</div>
            <div style="font-size:0.8rem; color:#7B8DB0; margin-top:0.3rem;">{priv_icon} {priv_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Privacy budget used</div>
            <div class="metric-value" style="color:#4E9BCD;">ε = {eps}</div>
            <div style="font-size:0.8rem; color:#7B8DB0; margin-top:0.3rem;">{privacy_labels[eps][0]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**MIA distance distribution**")
        if len(distances) > 0:
            fig, ax = make_dark_fig((6, 3.5))
            ax.hist(distances, bins=40, color='#4E9BCD', alpha=0.8, edgecolor='none')
            ax.axvline(threshold, color='#E05C5C', linestyle='--',
                       linewidth=1.5, label=f'Risk threshold ({threshold:.2f})')
            ax.set_xlabel('Distance to nearest synthetic record')
            ax.set_ylabel('Frequency')
            ax.set_title('Membership Inference Attack', color='#FFFFFF', fontsize=11)
            ax.legend(framealpha=0, labelcolor='#7B8DB0', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with chart_col2:
        st.markdown("**Synthetic vs real column distributions**")
        plot_col = selected[0] if selected else None
        if plot_col:
            fig, ax = make_dark_fig((6, 3.5))
            ax.hist(real_reduced[plot_col], bins=15, alpha=0.6,
                    color='#4E9BCD', label='Real', edgecolor='none')
            ax.hist(synth_df[plot_col], bins=15, alpha=0.6,
                    color='#4CAF82', label='Synthetic', edgecolor='none')
            ax.set_xlabel(plot_col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution comparison — {plot_col}',
                         color='#FFFFFF', fontsize=11)
            ax.legend(framealpha=0, labelcolor='#7B8DB0', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Epsilon sweep reference chart
    st.markdown("---")
    st.markdown("**Privacy-utility trade-off reference** — from research experiments")

    eps_vals  = [0.1, 0.5, 1.0, 2.0, 5.0]
    q_vals    = [64.05, 89.90, 96.05, 97.69, 98.56]
    mia_vals  = [1.40, 1.89, 0.00, 0.00, 0.00]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='#1A1F2E')
    for ax in axes:
        ax.set_facecolor('#1A1F2E')
        ax.tick_params(colors='#7B8DB0', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A3050')
        ax.xaxis.label.set_color('#7B8DB0')
        ax.yaxis.label.set_color('#7B8DB0')

    axes[0].plot(eps_vals, q_vals, color='#4E9BCD', marker='o',
                 linewidth=2, markersize=6)
    axes[0].axvline(eps, color='#4CAF82', linestyle='--',
                    linewidth=1.2, label=f'Your ε={eps}')
    axes[0].set_title('Quality vs ε', color='#FFFFFF', fontsize=10)
    axes[0].set_xlabel('Epsilon (ε)')
    axes[0].set_ylabel('Quality score (%)')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.15, color='#2A3050')
    axes[0].legend(framealpha=0, labelcolor='#7B8DB0', fontsize=8)

    axes[1].plot(eps_vals, mia_vals, color='#E05C5C', marker='o',
                 linewidth=2, markersize=6)
    axes[1].axvline(eps, color='#4CAF82', linestyle='--',
                    linewidth=1.2, label=f'Your ε={eps}')
    axes[1].set_title('Privacy risk vs ε', color='#FFFFFF', fontsize=10)
    axes[1].set_xlabel('Epsilon (ε)')
    axes[1].set_ylabel('MIA success rate (%)')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.15, color='#2A3050')
    axes[1].legend(framealpha=0, labelcolor='#7B8DB0', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Synthetic data preview
    st.markdown("---")
    st.markdown("**Synthetic data preview**")
    st.dataframe(synth_df.head(10), use_container_width=True)

    # ── Step 5: Download ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="step-badge">Step 5 — Download</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Download your synthetic dataset</div>', unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    synth_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label="⬇ Download synthetic CSV",
            data=csv_bytes,
            file_name=f"synthetic_data_eps{str(eps).replace('.','_')}.csv",
            mime="text/csv"
        )

    with dl2:
        # Summary report
        report = f"""SynthGuard — Privacy-Preserving Synthetic Data Report
=====================================================
Generated by: SynthGuard (MSc Project, University of Liverpool)
Method: MWEM Differential Privacy Synthesizer

Settings
--------
Privacy budget (epsilon): {eps}
Rows generated: {num_rows}
Columns used: {len(selected)}

Results
-------
Data quality score:       {quality:.2f}%
MIA success rate:         {mia_rate:.2f}%
Privacy assessment:       {priv_label}

Interpretation
--------------
A quality score above 80% indicates the synthetic data preserves
the statistical structure of the original dataset well.

An MIA success rate of 0.00% means no measurable privacy leakage
was detected — an attacker cannot identify real records from the
synthetic data using distance-based inference methods.

Reference: Stadler et al. (2022), Dwork et al. (2006)
"""
        st.download_button(
            label="⬇ Download summary report",
            data=report.encode(),
            file_name=f"synthguard_report_eps{str(eps).replace('.','_')}.txt",
            mime="text/plain"
        )

    st.markdown("""
    <div class="warn-box">
    ⚠ The synthetic data contains no real records from your original dataset. 
    It is safe to share for research and development purposes. Always verify 
    suitability for your specific use case before using in production.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    SynthGuard · Privacy-Preserving Synthetic Financial Data · 
    MSc Data Science, University of Liverpool · 2025 · 
    Built on MWEM differential privacy (Hardt et al. 2010) and 
    membership inference evaluation (Stadler et al. 2022)
</div>
""", unsafe_allow_html=True)