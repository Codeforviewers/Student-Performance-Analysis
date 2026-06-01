import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Attempt to import DoubleML for live causal analysis
try:
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor
    DOUBLEML_AVAILABLE = True
except ImportError:
    DOUBLEML_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="EduInsight | Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
        height: 100%;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prediction-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: #FFFFFF;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    .prediction-score {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #334155;
    }
    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-weight: 500;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        border-bottom: 3px solid #4F46E5;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & MAPPINGS (Identical to training pipeline in 5.30best.py)
# =============================================================================
ORDINAL_MAPPINGS = {
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
}

NOMINAL_COLS = ['School_Type', 'Gender', 'Internet_Access',
                'Extracurricular_Activities', 'Learning_Disabilities']

NUMERIC_COLS = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                  'Tutoring_Sessions', 'Physical_Activity']

FEATURE_RANGES = {
    'Hours_Studied': (1, 44, 20, "hours/week"),
    'Attendance': (60, 100, 80, "%"),
    'Previous_Scores': (50, 100, 75, "points"),
    'Sleep_Hours': (4, 10, 7, "hours/night"),
    'Tutoring_Sessions': (0, 8, 1, "sessions/week"),
    'Physical_Activity': (0, 6, 3, "days/week"),
}

# Pre-computed DoubleML ATE results (from research pipeline)
PRECOMPUTED_CAUSAL = {
    'Hours_Studied':     {'ate': 0.292, 'ci_lower': 0.282, 'ci_upper': 0.302, 'unit': 'per hour/week'},
    'Attendance':        {'ate': 0.198, 'ci_lower': 0.185, 'ci_upper': 0.211, 'unit': 'per %'},
    'Previous_Scores':   {'ate': 0.048, 'ci_lower': 0.040, 'ci_upper': 0.056, 'unit': 'per point'},
    'Parental_Involvement': {'ate': 1.021, 'ci_lower': 0.850, 'ci_upper': 1.192, 'unit': 'per level'},
    'Sleep_Hours':       {'ate': 0.354, 'ci_lower': 0.152, 'ci_upper': 0.556, 'unit': 'per hour'},
    'Tutoring_Sessions': {'ate': 0.448, 'ci_lower': 0.302, 'ci_upper': 0.594, 'unit': 'per session'},
    'Teacher_Quality':   {'ate': 0.685, 'ci_lower': 0.512, 'ci_upper': 0.858, 'unit': 'per level'},
}

# =============================================================================
# DATA & MODEL LOADING FUNCTIONS
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models with graceful fallback."""
    models = {'ridge': None, 'gb': None, 'ridge_loaded': False, 'gb_loaded': False}
    try:
        models['ridge'] = joblib.load('best_ridge_model_ordinal_onehot.pkl')
        models['ridge_loaded'] = True
    except Exception:
        pass
    try:
        models['gb'] = joblib.load('gb_model_for_cate.pkl')
        models['gb_loaded'] = True
    except Exception:
        pass
    return models

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """Replicate exact preprocessing pipeline from 5.30best.py."""
    try:
        df = pd.read_csv('StudentPerformanceFactors.csv')
    except FileNotFoundError:
        return None, None, None, None, None

    # 1. Outlier removal (identical to training)
    df_clean = df[df['Exam_Score'] <= 100].copy()

    # 2. Missing value imputation
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 3. Ordinal encoding for ordered categorical variables
    for col, mapping in ORDINAL_MAPPINGS.items():
        df_clean[col] = df_clean[col].map(mapping)

    # 4. Feature separation
    feature_cols = NUMERIC_COLS + list(ORDINAL_MAPPINGS.keys()) + NOMINAL_COLS
    X = df_clean[feature_cols].copy()
    y = df_clean['Exam_Score'].copy()

    # 5. One-Hot encoding (drop_first=False to match training script)
    X_encoded = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=False)

    # 6. Train-test split (random_state=42, test_size=0.2 — must match training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, df_clean

@st.cache_resource(show_spinner=False)
def train_gb_fallback(X_train, y_train):
    """Train Gradient Boosting for CATE if pickle is missing."""
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
    )
    gb.fit(X_train, y_train)
    return gb

@st.cache_data(show_spinner=False)
def run_doubleml_live(X_encoded, y, treatment_col):
    """Execute DoubleML PLR with cross-fitting (cached per treatment)."""
    data_dml = pd.concat([X_encoded.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    x_cols = [c for c in X_encoded.columns if c != treatment_col]

    dml_data = DoubleMLData(
        data=data_dml,
        y_col='Exam_Score',
        d_cols=treatment_col,
        x_cols=x_cols
    )
    # Nuisance learners matching the research script
    ml_g = RandomForestRegressor(n_estimators=200, max_depth=8,
                                 random_state=42, n_jobs=-1)
    ml_m = RandomForestRegressor(n_estimators=200, max_depth=8,
                                 random_state=42, n_jobs=-1)
    dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=5)
    dml_plr.fit()

    ate = float(dml_plr.coef[0])
    ci = dml_plr.confint()
    return {
        'ate': ate,
        'ci_lower': float(ci.iloc[0, 0]),
        'ci_upper': float(ci.iloc[0, 1]),
    }

# =============================================================================
# LOAD RESOURCES
# =============================================================================
models = load_models()
data_package = load_and_preprocess_data()
X_train = X_test = y_train = y_test = df_clean = None
if data_package[0] is not None:
    X_train, X_test, y_train, y_test, df_clean = data_package

# Fallback GB training if pickle missing
if models['gb'] is None and X_train is not None:
    models['gb'] = train_gb_fallback(X_train, y_train)
    models['gb_loaded'] = True

# =============================================================================
# SIDEBAR — STUDENT PROFILE INPUT
# =============================================================================
with st.sidebar:
    st.markdown("<h1 style='text-align:center; margin-bottom:0;'>🎓</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-top:0; color:#4F46E5;'>EduInsight</h2>", unsafe_allow_html=True)
    st.caption("Student Performance Intelligence System")
    st.markdown("---")

    st.subheader("🔧 Student Profile")

    # Numeric sliders
    input_numeric = {}
    for col, (mn, mx, default, unit) in FEATURE_RANGES.items():
        input_numeric[col] = st.slider(
            f"{col.replace('_', ' ')} ({unit})",
            min_value=mn, max_value=mx, value=default,
            help=f"Valid range: {mn}–{mx} {unit}"
        )

    # Ordinal categorical selectors
    input_ordinal = {}
    for col, mapping in ORDINAL_MAPPINGS.items():
        opts = list(mapping.keys())
        default_idx = 1 if len(opts) > 1 else 0
        input_ordinal[col] = st.selectbox(
            col.replace('_', ' '),
            opts,
            index=default_idx,
            help="Select level"
        )

    # Nominal categorical selectors
    input_nominal = {}
    nominal_defaults = {
        'School_Type': ['Public', 'Private'],
        'Gender': ['Male', 'Female'],
        'Internet_Access': ['Yes', 'No'],
        'Extracurricular_Activities': ['Yes', 'No'],
        'Learning_Disabilities': ['No', 'Yes'],
    }
    for col, opts in nominal_defaults.items():
        input_nominal[col] = st.selectbox(
            col.replace('_', ' '),
            opts,
            index=0
        )

    st.markdown("---")
    st.caption("Adjust inputs and click **Predict** in the Performance tab.")

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown('<div class="main-header">EduInsight Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Academic Performance Prediction · Causal Effect Analysis · Personalized Intervention</div>', unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab_pred, tab_causal, tab_intervention = st.tabs([
    "📊 Academic Performance Prediction",
    "⚖️ Causal Effect Analysis",
    "🎯 Personalized Intervention"
])

# ============================================================================
# TAB 1: ACADEMIC PERFORMANCE PREDICTION
# ============================================================================
with tab_pred:
    st.markdown("### Predict Exam Score & Identify Key Drivers")

    # Intro + Action row
    c_intro, c_btn, c_avg = st.columns([2, 1, 1])
    with c_intro:
        st.markdown("""
        <div class="insight-box">
        <b>Model:</b> Ridge Regression (α=50) with Ordinal + One-Hot encoding achieves 
        <b>R² = 0.825</b> on the hold-out test set. Predictions are unbiased and 
        coefficients are directly interpretable as feature contributions.
        </div>
        """, unsafe_allow_html=True)
    with c_btn:
        st.write("")
        st.write("")
        predict_btn = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)
    with c_avg:
        if y_test is not None:
            st.metric("Class Average", f"{float(np.mean(y_test)):.1f}",
                      help="Mean exam score in the test set")
        else:
            st.metric("Class Average", "N/A")

    if predict_btn:
        if models['ridge'] is None:
            st.error("❌ Ridge model not found. Ensure `best_ridge_model_ordinal_onehot.pkl` is in the working directory.")
        else:
            # ---- Replicate exact inference preprocessing ----
            input_df = pd.DataFrame([{**input_numeric, **input_ordinal, **input_nominal}])

            # Ordinal encoding
            for col, mapping in ORDINAL_MAPPINGS.items():
                input_df[col] = input_df[col].map(mapping)

            # One-Hot encoding
            input_enc = pd.get_dummies(input_df, columns=NOMINAL_COLS, drop_first=False)

            # Align to training columns (critical for model compatibility)
            if X_train is not None:
                input_enc = input_enc.reindex(columns=X_train.columns, fill_value=0)

            # Predict
            pred_raw = float(models['ridge'].predict(input_enc)[0])
            prediction = float(np.clip(pred_raw, 0, 100))

            # Feature contributions (coefficient × feature value)
            coefs = models['ridge'].coef_
            contrib = input_enc.values[0] * coefs
            contrib_df = pd.DataFrame({
                'Feature': input_enc.columns,
                'Contribution': contrib,
                'Abs_Contribution': np.abs(contrib)
            }).sort_values('Abs_Contribution', ascending=False).head(12)

            # ---- LAYOUT: Score + Drivers ----
            left_col, right_col = st.columns([1, 2])

            with left_col:
                # Color-coded score
                if prediction >= 80:
                    score_hex, status_msg, status_emoji = "#34D399", "Strong Performance", "✅"
                elif prediction >= 60:
                    score_hex, status_msg, status_emoji = "#FBBF24", "Moderate Performance", "📋"
                else:
                    score_hex, status_msg, status_emoji = "#F87171", "At-Risk Alert", "⚠️"

                st.markdown(f"""
                <div class="prediction-hero">
                    <div style="font-size:0.9rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7;">
                        Predicted Exam Score
                    </div>
                    <div class="prediction-score" style="color:{score_hex};">{prediction:.1f}</div>
                    <div style="font-size:1rem; opacity:0.8; margin-top:0.25rem;">out of 100 points</div>
                    <div style="margin-top:1rem; font-size:0.95rem; background:rgba(255,255,255,0.1); 
                                display:inline-block; padding:0.35rem 0.9rem; border-radius:999px;">
                        {status_emoji} {status_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Risk insights
                if prediction < 60:
                    st.error("**Intervention Recommended:** Low predicted score. Prioritize attendance and 1-on-1 tutoring.")
                elif prediction < 80:
                    st.warning("**Growth Opportunity:** Targeted improvements in top drivers can lift score into the next band.")
                else:
                    st.success("**Maintain Trajectory:** Continue current study habits and peer collaboration.")

                # Top 3 driver summary
                st.markdown("#### 🔑 Top Impact Drivers")
                for _, row in contrib_df.head(3).iterrows():
                    arrow = "↑" if row['Contribution'] > 0 else "↓"
                    st.markdown(f"- **{row['Feature']}**: {arrow} {row['Contribution']:+.2f} pts")

            with right_col:
                # Contribution waterfall-like bar chart
                fig_contrib = px.bar(
                    contrib_df.sort_values('Contribution', ascending=True),
                    x='Contribution',
                    y='Feature',
                    orientation='h',
                    color='Contribution',
                    color_continuous_scale=['#EF4444', '#F59E0B', '#10B981'],
                    title="Feature Contribution to Predicted Score",
                    template="plotly_white",
                    height=520
                )
                fig_contrib.update_layout(
                    yaxis_title="", xaxis_title="Contribution (points)",
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                    title_font_size=16, title_x=0.5,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

                # Comparison to class average
                if y_test is not None:
                    avg_score = float(np.mean(y_test))
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Bar(
                        x=['Your Student', 'Class Average'],
                        y=[prediction, avg_score],
                        marker_color=[score_hex, '#94A3B8'],
                        text=[f"{prediction:.1f}", f"{avg_score:.1f}"],
                        textposition='outside',
                        textfont_size=14,
                        showlegend=False
                    ))
                    fig_cmp.update_layout(
                        template="plotly_white", height=280,
                        yaxis_title="Exam Score", margin=dict(l=20, r=20, t=40, b=20),
                        title="Comparison to Test-Set Average", title_x=0.5,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

# ============================================================================
# TAB 2: CAUSAL EFFECT ANALYSIS
# ============================================================================
with tab_causal:
    st.markdown("### DoubleML Causal Effect Estimation")
    st.markdown("""
    <div class="insight-box">
    <b>Methodology:</b> <b>Double Machine Learning (DoubleML)</b> with Partially Linear Regression (PLR) 
    estimates <b>Average Treatment Effects (ATE)</b> via Neyman-orthogonal scores and 5-fold cross-fitting. 
    This controls for high-dimensional confounders and delivers robust, asymptotically normal causal 
    estimates — superior to simple regression coefficients.
    </div>
    """, unsafe_allow_html=True)

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        treatment_opts = ['Hours_Studied', 'Attendance', 'Previous_Scores',
                          'Parental_Involvement', 'Sleep_Hours', 'Tutoring_Sessions',
                          'Teacher_Quality']
        selected_treatments = st.multiselect(
            "Select Treatment Variables",
            treatment_opts,
            default=['Hours_Studied', 'Attendance', 'Parental_Involvement'],
            help="Variables must exist as columns in the encoded feature matrix (numeric or ordinal)."
        )
    with c2:
        mode = st.radio(
            "Mode",
            ["Pre-computed Results", "Live DoubleML"],
            index=0,
            help="Pre-computed: instant display from research pipeline. Live: runs DoubleMLPLR (~20–40s)."
        )
    with c3:
        run_live = False
        if mode == "Live DoubleML":
            st.write("")
            st.write("")
            run_live = st.button("▶️ Run DoubleML", type="primary", use_container_width=True)

    # Gather results
    results_rows = []

    if mode == "Live DoubleML" and run_live:
        if not DOUBLEML_AVAILABLE:
            st.error("❌ DoubleML not installed. Run: `pip install DoubleML`")
        elif X_train is None:
            st.error("❌ Dataset not found.")
        else:
            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            prog = st.progress(0)
            for idx, treat in enumerate(selected_treatments):
                if treat in X_full.columns:
                    with st.spinner(f"Estimating ATE for **{treat}**... ({idx+1}/{len(selected_treatments)})"):
                        res = run_doubleml_live(X_full, y_full, treat)
                    results_rows.append({
                        'Treatment': treat,
                        'ATE': res['ate'],
                        'CI_Lower': res['ci_lower'],
                        'CI_Upper': res['ci_upper'],
                        'Significant': "✅ Yes" if res['ci_lower'] > 0 or res['ci_upper'] < 0 else "❌ No"
                    })
                else:
                    st.warning(f"⚠️ {treat} not found in encoded matrix (One-Hotted?). Skipped.")
                prog.progress((idx + 1) / len(selected_treatments))
            prog.empty()
            st.success("✅ DoubleML analysis complete!")
    else:
        # Pre-computed path
        for treat in selected_treatments:
            if treat in PRECOMPUTED_CAUSAL:
                pc = PRECOMPUTED_CAUSAL[treat]
                results_rows.append({
                    'Treatment': treat,
                    'ATE': pc['ate'],
                    'CI_Lower': pc['ci_lower'],
                    'CI_Upper': pc['ci_upper'],
                    'Significant': "✅ Yes" if pc['ci_lower'] > 0 else "❌ No"
                })

    # Visualization
    if results_rows:
        res_df = pd.DataFrame(results_rows)

        # Metric cards
        st.markdown("#### 📊 Average Treatment Effects")
        metric_cols = st.columns(len(res_df))
        for i, (_, row) in enumerate(res_df.iterrows()):
            with metric_cols[i]:
                st.metric(
                    label=row['Treatment'].replace('_', ' '),
                    value=f"{row['ATE']:+.3f}",
                    delta=f"95% CI: [{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]"
                )

        st.markdown("---")

        # Forest Plot — MANUAL error bars for MAXIMUM compatibility with ALL Plotly versions
        fig_forest = go.Figure()

        # 1. Draw horizontal CI lines manually (avoids error_x API entirely)
        for _, row in res_df.iterrows():
            y_label = row['Treatment'].replace('_', ' ').title()
            fig_forest.add_trace(go.Scatter(
                x=[row['CI_Lower'], row['CI_Upper']],
                y=[y_label, y_label],
                mode='lines',
                line=dict(color='rgba(100, 116, 139, 0.5)', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Small end caps
            fig_forest.add_trace(go.Scatter(
                x=[row['CI_Lower'], row['CI_Lower'], row['CI_Upper'], row['CI_Upper']],
                y=[y_label, y_label, y_label, y_label],
                mode='markers',
                marker=dict(
                    color='rgba(100, 116, 139, 0.5)',
                    size=8,
                    symbol='line-ns',
                    line=dict(width=2)
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 2. Draw ATE center points
        point_colors = ['#10B981' if row['CI_Lower'] > 0 else '#EF4444' if row['CI_Upper'] < 0 else '#64748B'
                        for _, row in res_df.iterrows()]

        fig_forest.add_trace(go.Scatter(
            x=res_df['ATE'].tolist(),
            y=res_df['Treatment'].apply(lambda x: x.replace('_', ' ').title()).tolist(),
            mode='markers+text',
            marker=dict(
                color=point_colors,
                size=16,
                line=dict(width=2, color='white')
            ),
            text=[f"{v:+.3f}" for v in res_df['ATE'].tolist()],
            textposition="middle right",
            textfont=dict(size=11, color='#334155'),
            name='ATE',
            hovertemplate='<b>%{y}</b><br>ATE: %{x:+.3f}<extra></extra>'
        ))

        # Zero-effect reference line
        fig_forest.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=2,
                             annotation_text="Zero Effect", annotation_position="top")

        fig_forest.update_layout(
            title="DoubleML Forest Plot: ATE with 95% Confidence Intervals",
            xaxis_title="Average Treatment Effect (ATE) on Exam Score",
            yaxis_title="", template="plotly_white", height=420,
            margin=dict(l=20, r=20, t=60, b=40), title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_forest, use_container_width=True)

        # Interpretation
        strongest = res_df.loc[res_df['ATE'].abs().idxmax()]
        st.markdown("#### 📖 Interpretation")
        st.markdown(f"""
        - **{strongest['Treatment'].replace('_', ' ').title()}** exhibits the strongest causal impact 
          (ATE = **{strongest['ATE']:+.3f}** points, 95% CI: [{strongest['CI_Lower']:.3f}, {strongest['CI_Upper']:.3f}]).
        - Effects whose entire confidence interval lies to the right of zero (red dashed line) are 
          **statistically significant** at the 5% level — we can confidently assert a positive causal effect.
        - DoubleML removes regularization bias by orthogonalizing the estimation equations, making these 
          estimates valid for policy decisions.
        """)

        with st.expander("View Detailed Numeric Results"):
            disp = res_df.copy()
            for c in ['ATE', 'CI_Lower', 'CI_Upper']:
                disp[c] = disp[c].apply(lambda x: f"{x:+.4f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Select treatment variables and choose an analysis mode to view causal estimates.")

# ============================================================================
# TAB 3: PERSONALIZED INTERVENTION
# ============================================================================
with tab_intervention:
    st.markdown("### Personalized Intervention Recommendation (CATE)")
    st.markdown("""
    <div class="insight-box">
    <b>Methodology:</b> We estimate <b>Conditional Average Treatment Effects (CATE)</b> using a 
    Gradient Boosting surrogate model. For each student in the test set, we simulate a counterfactual 
    (increasing a treatment variable by Δ) and rank students by their predicted individual gain. 
    This enables precision education: allocating scarce intervention resources to students who benefit most.
    </div>
    """, unsafe_allow_html=True)

    if X_test is None or models['gb'] is None:
        st.error("❌ Test data or Gradient Boosting model unavailable. Cannot generate recommendations.")
    else:
        # Control panel
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1])
        with ctrl1:
            interv_var = st.selectbox(
                "Intervention Variable",
                ['Hours_Studied', 'Attendance', 'Tutoring_Sessions',
                 'Sleep_Hours', 'Physical_Activity'],
                index=0,
                help="Which lever to pull?"
            )
        with ctrl2:
            interv_delta = st.slider(
                "Magnitude (Δ)",
                min_value=1, max_value=10, value=5,
                help=f"Increase {interv_var} by this amount"
            )
        with ctrl3:
            top_n = st.slider(
                "Top N Students",
                min_value=5, max_value=50, value=10
            )

        st.write("")
        calc_btn = st.button("🎯 Calculate CATE & Rank Students", type="primary", use_container_width=True)

        if calc_btn:
            if interv_var not in X_test.columns:
                st.error(f"❌ {interv_var} not found in test set. It may have been One-Hot encoded during preprocessing.")
            else:
                # ---- CATE Computation ----
                base_pred = models['gb'].predict(X_test)
                X_counter = X_test.copy()
                mn, mx, _, _ = FEATURE_RANGES[interv_var]
                X_counter[interv_var] = np.clip(X_counter[interv_var] + interv_delta, mn, mx)
                cf_pred = models['gb'].predict(X_counter)
                benefit = cf_pred - base_pred

                # Summary KPIs
                avg_gain = float(np.mean(benefit))
                max_gain = float(np.max(benefit))
                pct_pos = float(np.mean(benefit > 0) * 100)
                std_gain = float(np.std(benefit))

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Average Gain", f"{avg_gain:+.2f} pts",
                         help=f"Mean predicted gain from +{interv_delta} {interv_var.replace('_', ' ')}")
                k2.metric("Max Gain", f"{max_gain:+.2f} pts", help="Highest individual predicted gain")
                k3.metric("Benefiting", f"{pct_pos:.0f}%", help="Share of students with positive predicted gain")
                k4.metric("Heterogeneity (SD)", f"{std_gain:.2f}", help="Standard deviation of gains (larger = more personalization potential)")

                st.markdown("---")

                # Distribution histogram
                fig_hist = px.histogram(
                    pd.DataFrame({"Gain": benefit}),
                    x="Gain", nbins=45,
                    title=f"Distribution of Predicted Gains (+{interv_delta} {interv_var.replace('_', ' ').title()})",
                    labels={"Gain": "Predicted Score Gain (points)", "count": "Students"},
                    template="plotly_white",
                    color_discrete_sequence=["#6366F1"],
                    opacity=0.85
                )
                fig_hist.add_vline(x=avg_gain, line_dash="dash", line_color="#EF4444", line_width=2,
                                   annotation_text=f"Mean: {avg_gain:+.2f}", annotation_position="top")
                fig_hist.update_layout(
                    height=360, margin=dict(l=20, r=20, t=50, b=20), title_x=0.5,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    bargap=0.05
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Top-N detailed table + bar chart
                top_idx = np.argsort(benefit)[-top_n:][::-1]
                top_df = pd.DataFrame({
                    'Student_ID': [f"S{idx}" for idx in top_idx],
                    'Current_Score': base_pred[top_idx].round(1),
                    'Counterfactual_Score': cf_pred[top_idx].round(1),
                    'Predicted_Gain': benefit[top_idx].round(2),
                    interv_var: X_test.iloc[top_idx][interv_var].values,
                    'Attendance': X_test.iloc[top_idx]['Attendance'].values,
                    'Previous_Scores': X_test.iloc[top_idx]['Previous_Scores'].values,
                })

                left, right = st.columns([2, 3])
                with left:
                    st.markdown(f"#### 🏆 Top {top_n} Beneficiaries")
                    st.dataframe(
                        top_df.style.background_gradient(
                            subset=['Predicted_Gain'], cmap='Greens'
                        ).format({
                            'Current_Score': "{:.1f}",
                            'Counterfactual_Score': "{:.1f}",
                            'Predicted_Gain': "{:.2f}",
                            interv_var: "{:.0f}",
                            'Attendance': "{:.0f}",
                            'Previous_Scores': "{:.0f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=420
                    )

                with right:
                    fig_top = px.bar(
                        top_df.sort_values('Predicted_Gain', ascending=True),
                        x='Predicted_Gain',
                        y='Student_ID',
                        orientation='h',
                        color='Predicted_Gain',
                        color_continuous_scale='Viridis',
                        title=f"Top {top_n} Students by Predicted Gain",
                        template="plotly_white",
                        height=500
                    )
                    fig_top.update_layout(
                        yaxis_title="", xaxis_title="Predicted Score Gain (points)",
                        coloraxis_showscale=False,
                        margin=dict(l=20, r=20, t=50, b=20), title_x=0.5,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

                # Segmented strategy insights
                st.markdown("---")
                st.markdown("#### 🔍 Segmented Strategy Insights")

                # Quintile-based segmentation by baseline score
                q20 = np.percentile(base_pred, 20)
                q40 = np.percentile(base_pred, 40)
                q60 = np.percentile(base_pred, 60)
                q80 = np.percentile(base_pred, 80)

                seg_masks = [
                    (base_pred < q20, "Bottom 20% (Lowest Baseline)"),
                    ((base_pred >= q20) & (base_pred < q40), "20–40%"),
                    ((base_pred >= q40) & (base_pred < q60), "40–60%"),
                    ((base_pred >= q60) & (base_pred < q80), "60–80%"),
                    (base_pred >= q80, "Top 20% (Highest Baseline)"),
                ]

                seg_list = []
                for mask, label in seg_masks:
                    seg_list.append({
                        'Segment': label,
                        'Avg_Gain': float(np.mean(benefit[mask])),
                        'N_Students': int(np.sum(mask)),
                        'Pct_Positive': float(np.mean(benefit[mask] > 0) * 100)
                    })
                seg_df = pd.DataFrame(seg_list)

                c_seg, c_txt = st.columns([3, 2])
                with c_seg:
                    fig_seg = px.bar(
                        seg_df,
                        x='Segment',
                        y='Avg_Gain',
                        color='Avg_Gain',
                        color_continuous_scale='RdYlGn',
                        text=seg_df['Avg_Gain'].apply(lambda x: f"{x:+.2f}"),
                        title="Average Treatment Effect by Baseline Performance Quintile",
                        template="plotly_white",
                        height=380
                    )
                    fig_seg.update_traces(textposition='outside', textfont_size=12)
                    fig_seg.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(l=20, r=20, t=50, b=60),
                        title_x=0.5,
                        xaxis_tickangle=-25,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_seg, use_container_width=True)

                with c_txt:
                    best_seg = seg_df.loc[seg_df['Avg_Gain'].idxmax()]
                    st.markdown(f"""
                    <div class="insight-box">
                    <b>Precision Education Insight</b><br><br>
                    The <b>{best_seg['Segment']}</b> segment shows the highest 
                    average treatment effect (<b>{best_seg['Avg_Gain']:+.2f}</b> points). 
                    <br><br>
                    <b>Recommendation:</b> Prioritize {interv_var.replace('_', ' ').lower()} 
                    interventions for this cohort to maximize educational ROI and equity.
                    </div>
                    """, unsafe_allow_html=True)

                    # Show table
                    st.markdown("<b>Segment Summary</b>", unsafe_allow_html=True)
                    st.dataframe(
                        seg_df.style.background_gradient(subset=['Avg_Gain'], cmap='Greens').format({
                            'Avg_Gain': "{:.2f}",
                            'N_Students': "{:.0f}",
                            'Pct_Positive': "{:.0f}%"
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=280
                    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <b>EduInsight</b> | Streamlit · Ridge (α=50) · DoubleML PLR · Gradient Boosting CATE<br>
    Encoding: Ordinal (ordered) + One-Hot (nominal) | Best Model R² = 0.825 | RMSE = 1.523
</div>
""", unsafe_allow_html=True)
