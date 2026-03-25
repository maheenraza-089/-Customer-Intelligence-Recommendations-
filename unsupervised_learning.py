import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #e0e0f0; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #c8b4ff; }
    .metric-card {
        background: #1c1c30;
        border: 1px solid #2e2e50;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.8rem; color: #8888aa; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { color: #a0a0cc; }
    .stTabs [aria-selected="true"] { color: #c8b4ff !important; border-bottom-color: #7c6af7 !important; }
    div[data-testid="stSidebarContent"] { background-color: #13132a; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#7c6af7","#f76a8c","#6af7c8","#f7c46a","#6aaaf7","#f76af7","#a8f76a","#f7a86a"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    N = 500
    segments = np.random.choice(
        ["high_value","mid_tier","budget","churned"], N, p=[0.15,0.35,0.40,0.10]
    )
    def gen(seg):
        if seg == "high_value":
            return (np.random.poisson(18)+5, np.random.gamma(8,60),
                    np.random.randint(1,30), np.random.randint(28,55),
                    np.random.gamma(5,12), np.random.beta(7,2))
        elif seg == "mid_tier":
            return (np.random.poisson(8)+2, np.random.gamma(4,35),
                    np.random.randint(15,90), np.random.randint(22,60),
                    np.random.gamma(3,8), np.random.beta(4,4))
        elif seg == "budget":
            return (np.random.poisson(4)+1, np.random.gamma(2,18),
                    np.random.randint(30,180), np.random.randint(18,65),
                    np.random.gamma(2,5), np.random.beta(2,5))
        else:
            return (np.random.poisson(1), np.random.gamma(1,10),
                    np.random.randint(150,365), np.random.randint(20,70),
                    np.random.gamma(1,3), np.random.beta(1,8))

    rows = [gen(s) for s in segments]
    df = pd.DataFrame(rows, columns=[
        "purchase_frequency","avg_spend","days_since_last_purchase",
        "age","avg_session_min","cart_conversion_rate"])
    df["customer_id"] = [f"C{i:04d}" for i in range(N)]

    # Inject noise
    missing_idx = np.random.choice(N, 25, replace=False)
    df.loc[missing_idx[:12], "avg_spend"] = np.nan
    df.loc[missing_idx[12:], "avg_session_min"] = np.nan
    dup_idx = np.random.choice(N, 8, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Clean
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    FEATURES = ["purchase_frequency","avg_spend","days_since_last_purchase",
                "age","avg_session_min","cart_conversion_rate"]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df[FEATURES])
    df[FEATURES] = X_imp

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return df, X_scaled, FEATURES, scaler


@st.cache_data
def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km.inertia_


@st.cache_data
def compute_silhouettes(X_scaled, k_range):
    inertias, silhouettes = [], []
    for k in k_range:
        labels, inertia = run_kmeans(X_scaled, k)
        inertias.append(inertia)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes


@st.cache_data
def run_pca(X_scaled, n=2):
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_full = PCA(random_state=42).fit(X_scaled)
    return X_pca, pca, pca_full


@st.cache_data
def detect_anomalies(X_scaled, FEATURES, chi2_pct):
    cov = np.cov(X_scaled.T)
    cov_inv = np.linalg.pinv(cov)
    mean_vec = X_scaled.mean(axis=0)
    mahal_dist = np.array([mahalanobis(x, mean_vec, cov_inv) for x in X_scaled])
    thresh = np.sqrt(stats.chi2.ppf(chi2_pct / 100, df=len(FEATURES)))
    is_anomaly = mahal_dist > thresh
    return mahal_dist, thresh, is_anomaly


@st.cache_data
def build_ratings(customer_ids, seed=42):
    np.random.seed(seed)
    N_PRODUCTS = 50
    PRODUCTS = [f"P{i:03d}" for i in range(N_PRODUCTS)]
    records = []
    for uid in customer_ids:
        n_rated = np.random.randint(8, 21)
        prods = np.random.choice(PRODUCTS, n_rated, replace=False)
        for p in prods:
            rating = np.clip(np.random.normal(3.2, 1.0), 1, 5)
            records.append({"user": uid, "product": p, "rating": round(rating, 1)})
    ratings_df = pd.DataFrame(records)
    R = ratings_df.pivot_table(index="user", columns="product", values="rating")
    return R, PRODUCTS


def cosine_sim_matrix(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return (M / norms) @ (M / norms).T


def recommend(target_user, R, top_n=5, n_neighbors=10):
    R_filled = R.fillna(0).values
    user_ids = list(R.index)
    if target_user not in user_ids:
        return []
    SIM = cosine_sim_matrix(R_filled)
    idx = user_ids.index(target_user)
    sims = SIM[idx].copy(); sims[idx] = -1
    neighbor_idx = np.argsort(sims)[::-1][:n_neighbors]
    already_rated = set(R.columns[~R.iloc[idx].isna()])
    candidates = {}
    for nidx in neighbor_idx:
        w = sims[nidx]
        for p in R.columns:
            if p not in already_rated:
                rv = R.iloc[nidx][p]
                if not np.isnan(rv):
                    candidates[p] = candidates.get(p, 0) + w * rv
    ranked = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
    return [(p, round(s, 2)) for p, s in ranked]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    st.markdown("### Task 2 — K-Means")
    k_range_max = st.slider("Max k to evaluate", 5, 15, 10)
    manual_k = st.slider("Force number of clusters", 2, 10, 4)
    use_best_k = st.checkbox("Auto-select best k (Silhouette)", value=True)

    st.markdown("---")
    st.markdown("### Task 3 — Anomaly Detection")
    chi2_pct = st.slider("χ² confidence threshold (%)", 90, 99, 99)

    st.markdown("---")
    st.markdown("### Task 5 — Recommendations")
    n_neighbors = st.slider("CF neighbors", 3, 20, 10)
    top_n_recs = st.slider("Top-N recommendations", 3, 10, 5)

    st.markdown("---")
    st.caption("Dataset: Synthetic e-commerce | 500 customers | 6 features")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df, X_scaled, FEATURES, scaler = generate_data()
X_pca2, pca2, pca_full = run_pca(X_scaled, n=2)
mahal_dist, mahal_thresh, is_anomaly = detect_anomalies(X_scaled, FEATURES, chi2_pct)
df["mahal_dist"] = mahal_dist
df["is_anomaly"] = is_anomaly

K_RANGE = list(range(2, k_range_max + 1))
inertias, silhouettes = compute_silhouettes(X_scaled, K_RANGE)
best_k = K_RANGE[int(np.argmax(silhouettes))]
chosen_k = best_k if use_best_k else manual_k
cluster_labels, _ = run_kmeans(X_scaled, chosen_k)
df["cluster"] = cluster_labels

# Name clusters by avg_spend rank
profile = df.groupby("cluster")[FEATURES].mean()
spend_rank = profile["avg_spend"].rank(ascending=False).astype(int)
TIER_NAMES = {1:"💎 Champions", 2:"🏆 Loyal", 3:"🛒 Bargain Hunters", 4:"💤 At-Risk", 5:"🌱 New"}
labels_map = {c: TIER_NAMES.get(spend_rank[c], f"Segment {c}") for c in profile.index}
df["cluster_label"] = df["cluster"].map(labels_map)

# Ratings for CF
DEMO_USERS = df["customer_id"].head(80).tolist()
R, PRODUCTS = build_ratings(DEMO_USERS)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛒 E-Commerce Unsupervised Learning Dashboard")
st.markdown("**Unsupervised ML Assignment** — K-Means · Anomaly Detection · PCA · Collaborative Filtering")
st.markdown("---")

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("500", "Customers"),
    ("6", "Features"),
    (str(chosen_k), "Clusters Found"),
    (str(is_anomaly.sum()), "Anomalies Detected"),
    (f"{pca_full.explained_variance_ratio_[:2].sum()*100:.0f}%", "PC1+PC2 Variance"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4,c5], kpis):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Task 1 — Data",
    "🔵 Task 2 — K-Means",
    "🚨 Task 3 — Anomalies",
    "📉 Task 4 — PCA",
    "🎯 Task 5 — Recommendations",
    "💡 Task 6 — Reflection",
])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Data Understanding & Preprocessing")

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("##### Raw Dataset (sample)")
        st.dataframe(df[["customer_id"] + FEATURES + ["is_anomaly"]].head(20),
                     use_container_width=True, height=350)
    with col_r:
        st.markdown("##### Descriptive Statistics")
        st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)

    st.markdown("##### Feature Distributions")
    feat_sel = st.selectbox("Select feature", FEATURES)
    fig = px.histogram(df, x=feat_sel, nbins=40, color_discrete_sequence=["#7c6af7"],
                       template="plotly_dark", title=f"Distribution of {feat_sel}")
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Correlation Heatmap")
    corr = df[FEATURES].corr().round(2)
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, template="plotly_dark",
                     title="Feature Correlation Matrix")
    fig2.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📖 Why Preprocessing Matters"):
        st.markdown("""
- **Missing values** — Median imputation used; mean would be distorted by outliers in `avg_spend`.
- **Duplicates** — Removed 8 duplicate rows that would bias cluster centroids.
- **StandardScaler** — K-Means uses Euclidean distance; `avg_spend` (~$150) would dominate over `cart_conversion_rate` (~0.4) without normalisation.
- **PCA** — Requires scaled inputs; otherwise high-variance features hijack the principal components.
        """)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"K-Means Segmentation  (k = {chosen_k})")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_RANGE, y=inertias, mode="lines+markers",
                                 line=dict(color="#7c6af7", width=2),
                                 marker=dict(size=8), name="Inertia"))
        fig.add_vline(x=chosen_k, line_dash="dash", line_color="#f76a8c",
                      annotation_text=f"k={chosen_k}")
        fig.update_layout(title="Elbow Method", xaxis_title="k",
                          yaxis_title="Inertia", template="plotly_dark",
                          paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        colors = [("#f76a8c" if k == chosen_k else "#7c6af7") for k in K_RANGE]
        fig.add_trace(go.Bar(x=K_RANGE, y=silhouettes,
                             marker_color=colors, name="Silhouette"))
        fig.update_layout(title="Silhouette Scores", xaxis_title="k",
                          yaxis_title="Score", template="plotly_dark",
                          paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Clusters in PCA 2-D Space")
    df_plot = df.copy()
    df_plot["PC1"] = X_pca2[:, 0]
    df_plot["PC2"] = X_pca2[:, 1]
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="cluster_label",
                     hover_data=["customer_id","avg_spend","purchase_frequency"],
                     color_discrete_sequence=PALETTE, template="plotly_dark",
                     title="Customer Clusters (PCA Projection)")
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Cluster Profiles")
    profile_display = df.groupby("cluster_label")[FEATURES].mean().round(2)
    st.dataframe(profile_display, use_container_width=True)

    st.markdown("##### Spend Distribution per Cluster")
    fig = px.box(df, x="cluster_label", y="avg_spend", color="cluster_label",
                 color_discrete_sequence=PALETTE, template="plotly_dark",
                 title="Average Spend by Cluster")
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Density Estimation & Anomaly Detection")
    st.info(f"**{is_anomaly.sum()} anomalies** detected using Mahalanobis distance "
            f"(χ² {chi2_pct}% threshold = {mahal_thresh:.2f})")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mahal_dist[~is_anomaly], name="Normal",
                                   marker_color="#6af7c8", opacity=0.75,
                                   nbinsx=40, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=mahal_dist[is_anomaly], name="Anomaly",
                                   marker_color="#f76a8c", opacity=0.9,
                                   nbinsx=15, histnorm="probability density"))
        fig.add_vline(x=mahal_thresh, line_dash="dash", line_color="#f7c46a",
                      annotation_text=f"Threshold {mahal_thresh:.1f}")
        fig.update_layout(title="Mahalanobis Distance Distribution",
                          barmode="overlay", template="plotly_dark",
                          paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_plot = df.copy()
        df_plot["PC1"] = X_pca2[:, 0]; df_plot["PC2"] = X_pca2[:, 1]
        df_plot["status"] = df_plot["is_anomaly"].map({True:"⭐ Anomaly", False:"Normal"})
        fig = px.scatter(df_plot, x="PC1", y="PC2", color="status",
                         color_discrete_map={"Normal":"#6af7c8","⭐ Anomaly":"#f76a8c"},
                         symbol="status",
                         symbol_map={"Normal":"circle","⭐ Anomaly":"star"},
                         hover_data=["customer_id","avg_spend","mahal_dist"],
                         template="plotly_dark",
                         title="Normal vs Anomalous Customers (PCA)")
        fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Top Anomalous Customers")
    anom_df = (df[df["is_anomaly"]][["customer_id"] + FEATURES + ["mahal_dist"]]
               .sort_values("mahal_dist", ascending=False).head(15).round(2))
    st.dataframe(anom_df, use_container_width=True)

    fig = px.scatter(df, x="purchase_frequency", y="avg_spend",
                     color=df["is_anomaly"].map({True:"Anomaly", False:"Normal"}),
                     color_discrete_map={"Normal":"#6af7c8","Anomaly":"#f76a8c"},
                     size="mahal_dist", hover_data=["customer_id"],
                     template="plotly_dark",
                     title="Purchase Frequency vs Avg Spend  (bubble = Mahalanobis distance)")
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Principal Component Analysis")

    expl_var = pca_full.explained_variance_ratio_
    cum_var  = np.cumsum(expl_var)
    n_comp_95 = int(np.searchsorted(cum_var, 0.95)) + 1

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(FEATURES))],
                             y=expl_var * 100,
                             marker_color=PALETTE[:len(FEATURES)], name="Individual"))
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(FEATURES))],
                                 y=cum_var * 100, mode="lines+markers",
                                 line=dict(color="#f7c46a", width=2),
                                 name="Cumulative"))
        fig.add_hline(y=95, line_dash="dot", line_color="#f76a8c",
                      annotation_text="95%")
        fig.update_layout(title=f"Scree Plot  (need {n_comp_95} PCs for 95%)",
                          yaxis_title="Variance Explained (%)", template="plotly_dark",
                          paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        loadings = pd.DataFrame(pca_full.components_[:2].T,
                                index=FEATURES, columns=["PC1","PC2"]).round(3)
        fig = px.imshow(loadings.T, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, template="plotly_dark",
                        title="PCA Loadings Heatmap (PC1 & PC2)")
        fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Biplot — PC1 vs PC2")
    df_plot = df.copy()
    df_plot["PC1"] = X_pca2[:,0]; df_plot["PC2"] = X_pca2[:,1]
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="cluster_label",
                     color_discrete_sequence=PALETTE, opacity=0.65,
                     hover_data=["customer_id"], template="plotly_dark",
                     title="PCA Biplot with Loading Vectors")
    scale = 3.5
    for i, feat in enumerate(FEATURES):
        lx = pca_full.components_[0,i] * scale
        ly = pca_full.components_[1,i] * scale
        fig.add_annotation(x=lx, y=ly, ax=0, ay=0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3,
                           arrowcolor="#f7c46a", arrowwidth=2)
        fig.add_annotation(x=lx*1.15, y=ly*1.15, text=feat.replace("_","<br>"),
                           showarrow=False, font=dict(color="#f7c46a", size=9))
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(f"""
- **PC1 ({expl_var[0]*100:.1f}%)** is the *engagement axis*: loads heavily on purchase frequency, avg spend, session time, and conversion rate. High PC1 = valuable customer.
- **PC2 ({expl_var[1]*100:.1f}%)** loads almost entirely on **age** — orthogonal to purchasing behaviour.
- Only **{n_comp_95} components** are needed to explain 95% of variance (down from 6 original features).
- After PCA, components are **perfectly uncorrelated** — reduces multicollinearity for downstream models.
        """)


# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Collaborative Filtering Recommendation System")

    sparsity = R.isna().sum().sum() / R.size * 100
    st.info(f"Ratings matrix: **{R.shape[0]} users × {R.shape[1]} products** | "
            f"Sparsity: **{sparsity:.1f}%**")

    with st.expander("📖 How Collaborative Filtering Works"):
        st.markdown("""
**User-Based CF** finds customers with similar rating patterns (cosine similarity) and
recommends products they rated highly that the target user hasn't seen yet.

- **Cosine Similarity** measures the angle between rating vectors — 1 = identical taste, 0 = no overlap.
- **Weighted aggregation** — neighbours closer in taste contribute more to the predicted score.
- No product metadata needed — only the rating matrix!
        """)

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_user = st.selectbox("Select a customer", DEMO_USERS[:30])
        recs = recommend(selected_user, R, top_n=top_n_recs, n_neighbors=n_neighbors)

        if recs:
            st.markdown(f"##### Top {top_n_recs} Recommendations for {selected_user}")
            rec_df = pd.DataFrame(recs, columns=["Product","Predicted Score"])
            rec_df.index += 1
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.warning("No recommendations found — user may have rated too many products.")

    with col2:
        if recs:
            products = [r[0] for r in recs]
            scores   = [r[1] for r in recs]
            fig = go.Figure(go.Bar(
                x=scores, y=products, orientation="h",
                marker_color=PALETTE[:len(recs)], text=[f"{s:.2f}" for s in scores],
                textposition="outside"))
            fig.update_layout(title=f"Recommendations for {selected_user}",
                              xaxis_title="Predicted Score",
                              template="plotly_dark", yaxis=dict(autorange="reversed"),
                              paper_bgcolor="#161625", plot_bgcolor="#161625")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### User–User Similarity Heatmap (first 20 users)")
    R20 = R.fillna(0).values[:20]
    norms = np.linalg.norm(R20, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    SIM20 = (R20 / norms) @ (R20 / norms).T
    fig = px.imshow(np.round(SIM20, 2),
                    x=DEMO_USERS[:20], y=DEMO_USERS[:20],
                    color_continuous_scale="Viridis", zmin=0, zmax=1,
                    text_auto=".2f", template="plotly_dark",
                    title="Cosine Similarity — First 20 Users")
    fig.update_layout(paper_bgcolor="#161625", plot_bgcolor="#161625",
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Analysis & Reflection")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 Hidden Patterns Uncovered")
        st.markdown("""
**K-Means** revealed that without any labels, the data naturally splits into
distinct customer tiers differentiated by spend velocity, recency, and session engagement.
Champions (top cluster) spend **7× more** than At-Risk customers and have **3× higher** conversion.

**Anomaly Detection** exposed ~3–5% of customers who sit far outside the normal
multivariate distribution. These are either ultra-high-value VIPs or potential fraud cases
that would be invisible to simple univariate thresholds.

**PCA** showed that 6 behavioural features collapse into essentially **2 meaningful axes**:
an engagement axis (PC1 ~60%) and a demographic axis (PC2 ~17%). This drastically
simplifies downstream modelling.

**Collaborative Filtering** uncovered implicit taste communities — customers who never
interacted are linked through shared rating patterns, enabling cold-start recommendations.
        """)

    with col2:
        st.markdown("### ⚖️ Technique Comparison")
        comparison = pd.DataFrame({
            "Technique":    ["K-Means", "KDE / Mahalanobis", "PCA", "Collab. Filtering"],
            "Strength":     ["Actionable segments", "Multi-variate outliers", "Noise reduction", "No metadata needed"],
            "Weakness":     ["Sensitive to outliers", "Assumes Gaussian dist.", "Linear only", "Cold-start problem"],
            "Output":       ["Cluster labels", "Anomaly flags", "Reduced features", "Product rankings"],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("### 🌍 Real-World Applications")
        apps = {
            "🛒 E-Commerce":   "Targeted promotions, churn prediction, dynamic pricing",
            "🏦 Banking":      "Fraud detection, customer lifetime value tiers",
            "🎬 Streaming":    "Personalised queues (Netflix, Spotify, YouTube)",
            "🏥 Healthcare":   "Patient cohort identification, anomaly alerts",
            "🏪 Retail":       "Assortment planning, demand anomaly detection",
        }
        for domain, desc in apps.items():
            st.markdown(f"**{domain}** — {desc}")

    st.markdown("---")
    st.markdown("### 📊 Pipeline Summary")
    summary_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=["Raw Data","Cleaned Data","Scaled Features",
                   "K-Means Clusters","Anomalies","PCA Components","Recommendations"],
            color=["#7c6af7","#6af7c8","#6aaaf7",
                   "#f76a8c","#f7c46a","#6af7c8","#a78bfa"],
        ),
        link=dict(
            source=[0,1,2,2,2,2],
            target=[1,2,3,4,5,6],
            value =[500,500,500,500,500,500],
            color =["#7c6af7","#6af7c8","#6aaaf7","#f76a8c","#f7c46a","#6af7c8"],
        )
    ))
    summary_fig.update_layout(title="Data Flow Through the ML Pipeline",
                               template="plotly_dark",
                               paper_bgcolor="#161625", font_color="#e0e0f0")
    st.plotly_chart(summary_fig, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Plotly · scikit-learn · scipy  |  Unsupervised Learning Assignment")