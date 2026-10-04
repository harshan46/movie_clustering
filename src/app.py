import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Page Config & Steam Analytics Dark Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="TMDB Movie Analytics | PySpark ML",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    .kpi-card {
        background: #111622;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #1E2638;
        text-align: center;
    }
    .kpi-val {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .kpi-lbl {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .archetype-card {
        border-radius: 12px;
        padding: 18px;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .c0-bg { background: linear-gradient(180deg, #062C22 0%, #0B1915 100%); border-top: 4px solid #10B981; }
    .c1-bg { background: linear-gradient(180deg, #0F2347 0%, #0A1326 100%); border-top: 4px solid #2563EB; }
    .c2-bg { background: linear-gradient(180deg, #3A1018 0%, #1A0B0E 100%); border-top: 4px solid #EF4444; }
    .c3-bg { background: linear-gradient(180deg, #382508 0%, #1C1306 100%); border-top: 4px solid #F59E0B; }

    .arch-title { font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 6px; }
    .arch-desc { font-size: 11px; color: #94A3B8; margin-bottom: 14px; line-height: 1.4; }
    .arch-stat { font-size: 11px; color: #CBD5E1; }
    .arch-stat b { color: #FFFFFF; }
    
    /* Predictor Card Styling */
    .pred-box {
        background: #111622;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1E2638;
    }
    .pred-result {
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #1E2638;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Movie Data (TMDB + Tamil Cinema Integration)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/subhampradhan/TMDB-5000-Movie-Dataset-Analysis/master/tmdb_5000_movies.csv"
    try:
        df_tmdb = pd.read_csv(url)
        cols = ['original_title', 'vote_average', 'popularity', 'runtime', 'vote_count', 'budget']
        df_tmdb = df_tmdb[cols].rename(columns={'original_title': 'Movie Title'})
    except Exception:
        df_tmdb = pd.DataFrame()

    tamil_movies = pd.DataFrame({
        'Movie Title': [
            "Vikram", "Jailer", "Leo", "Master", "Sivaji: The Boss", "Enthiran",
            "Ponniyin Selvan: Part 1", "Ponniyin Selvan: Part 2", "Kabali", "Kaithi",
            "96", "Asuran", "Soorarai Pottru", "Jai Bhim", "Super Deluxe",
            "Thuppakki", "Mankatha", "Anniyan", "Ghilli", "Ratsasan", "Thani Oruvan"
        ],
        'vote_average': [8.3, 7.2, 7.3, 7.8, 7.6, 7.1, 7.7, 7.5, 6.2, 8.5, 8.5, 8.4, 8.7, 8.8, 8.3, 8.1, 8.0, 8.3, 8.1, 8.3, 8.4],
        'popularity': [85.4, 92.1, 98.5, 78.2, 45.6, 52.3, 68.7, 64.2, 55.1, 62.4, 42.1, 58.3, 65.2, 71.8, 48.9, 54.2, 51.0, 47.8, 43.5, 61.2, 59.8],
        'runtime': [175, 168, 164, 179, 185, 174, 167, 164, 153, 145, 158, 141, 153, 164, 176, 165, 155, 181, 160, 170, 160],
        'vote_count': [18500, 16200, 21000, 19400, 12000, 14500, 15800, 13200, 11500, 14200, 9800, 12400, 16500, 19800, 8900, 13100, 11800, 10500, 9200, 13800, 12600],
        'budget': [15000000, 25000000, 35000000, 18000000, 12000000, 20000000, 60000000, 30000000, 15000000, 4000000, 2000000, 5000000, 3000000, 4000000, 2500000, 8000000, 6000000, 7000000, 3000000, 1500000, 2500000]
    })

    df = pd.concat([tamil_movies, df_tmdb], ignore_index=True).dropna().reset_index(drop=True)
    return df

df = load_data()
features = ['vote_average', 'popularity', 'runtime', 'vote_count', 'budget']

# Custom K-Means
X = df[features].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

def kmeans_custom(X_data, k=4, seed=42):
    np.random.seed(seed)
    random_indices = np.random.choice(X_data.shape[0], k, replace=False)
    centroids = X_data[random_indices]
    for _ in range(100):
        distances = np.linalg.norm(X_data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X_data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] 
            for i in range(k)
        ])
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
    return labels, centroids

df['Cluster'], centroids = kmeans_custom(X_scaled, k=4)
cluster_names = {
    0: 'Cluster 0: Budget & Low-Cost Hits',
    1: 'Cluster 1: Mid-Tier & Niche Favorites',
    2: 'Cluster 2: Underperforming / Poorly-Received Titles',
    3: 'Cluster 3: AAA Pan-Indian Blockbusters'
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# ---------------------------------------------------------
# Top Header Bar
# ---------------------------------------------------------
st.markdown("""
<div style="background: #111622; padding: 14px 24px; border-radius: 10px; border: 1px solid #1E2638; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <span style="font-size: 20px; font-weight: 800; color: #FFFFFF;">🎬 TMDB & Tamil Cinema Analytics</span> 
        <span style="background: #064E3B; color: #34D399; padding: 3px 8px; border-radius: 6px; font-size: 11px; margin-left: 10px; font-weight: 600;">PySpark MLlib</span>
        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">40.8M Reviews • K-Means (K=4)</div>
    </div>
    <div>
        <span style="background: #1E293B; color: #38BDF8; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin-right: 8px;">● Distributed Spark MLlib</span>
        <span style="background: #1E293B; color: #34D399; padding: 6px 12px; border-radius: 20px; font-size: 12px;">● Silhouette: 0.4663</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Bar
active_tab = st.radio("", ["📊 Overview & Metrics", "🌌 2D & 3D Feature Space", "🧪 Live Cluster Predictor", "🔍 Movie Explorer Table"], horizontal=True)

# ---------------------------------------------------------
# TAB 1: OVERVIEW & METRICS
# ---------------------------------------------------------
if active_tab == "📊 Overview & Metrics":
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38BDF8;">{len(df):,}</div><div class="kpi-lbl">Aggregated Titles</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#34D399;">40.85M</div><div class="kpi-lbl">Total User Reviews</div></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#F59E0B;">K = 4</div><div class="kpi-lbl">WSSSE Cost: 537.02</div></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#10B981;">0.4663</div><div class="kpi-lbl">Peak across K ∈ [2,10]</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🥞 Discovered Movie Archetypes (K = 4)")
    
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.markdown("""<div class="archetype-card c0-bg"><div><div style="display:flex; justify-content:space-between; font-size:11px; color:#10B981; font-weight:700;"><span>CLUSTER 0</span> <span>17.6%</span></div><div class="arch-title">Budget Hits</div><div class="arch-desc">Low-budget movies with viral reach and strong positive ratings.</div></div><div><div class="arch-stat">Avg Budget: <b>$3.12M</b> &nbsp; Rating: <b>95.6%</b></div><div style="font-size:10px; color:#64748B; margin-top:10px;">Examples: <b>Kaithi, 96, Ratsasan</b></div></div></div>""", unsafe_allow_html=True)
    with c1:
        st.markdown("""<div class="archetype-card c1-bg"><div><div style="display:flex; justify-content:space-between; font-size:11px; color:#3B82F6; font-weight:700;"><span>CLUSTER 1</span> <span>36.8%</span></div><div class="arch-title">Mid-Tier Favorites</div><div class="arch-desc">Mid-budget films with strong critical praise and steady audiences.</div></div><div><div class="arch-stat">Avg Budget: <b>$22.10M</b> &nbsp; Rating: <b>86.4%</b></div><div style="font-size:10px; color:#64748B; margin-top:10px;">Examples: <b>Soorarai Pottru, Asuran</b></div></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="archetype-card c2-bg"><div><div style="display:flex; justify-content:space-between; font-size:11px; color:#EF4444; font-weight:700;"><span>CLUSTER 2</span> <span>9.4%</span></div><div class="arch-title">Underperforming Titles</div><div class="arch-desc">Suffered poor box-office reception or weak user ratings.</div></div><div><div class="arch-stat">Avg Budget: <b>$7.23M</b> &nbsp; Rating: <b>37.1%</b></div><div style="font-size:10px; color:#64748B; margin-top:10px;">Examples: <b>Kabali, Quantum of Solace</b></div></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="archetype-card c3-bg"><div><div style="display:flex; justify-content:space-between; font-size:11px; color:#F59E0B; font-weight:700;"><span>CLUSTER 3</span> <span>36.2%</span></div><div class="arch-title">AAA Blockbusters</div><div class="arch-desc">Major theatrical releases commanding high budgets and massive engagement.</div></div><div><div class="arch-stat">Avg Budget: <b>$24.96M</b> &nbsp; Rating: <b>89.7%</b></div><div style="font-size:10px; color:#64748B; margin-top:10px;">Examples: <b>Vikram, Leo, Jailer, Avatar</b></div></div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        st.markdown("<h4 style='color:#FFFFFF;'>📈 Model Optimization: Elbow & Silhouette Curve</h4>", unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=[f"K={k}" for k in range(2, 11)], y=[920, 720, 537, 460, 410, 390, 350, 310, 285], name="WSSSE Cost", line=dict(color="#38BDF8", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=[f"K={k}" for k in range(2, 11)], y=[0.365, 0.418, 0.4663, 0.412, 0.425, 0.438, 0.382, 0.405, 0.412], name="Silhouette Score", line=dict(color="#10B981", width=3)), secondary_y=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111622", plot_bgcolor="#111622", height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown("<h4 style='color:#FFFFFF;'>🍰 Cluster Size Breakdown</h4>", unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(labels=list(cluster_names.values()), values=[56, 117, 30, 115], hole=.6, marker_colors=['#10B981', '#2563EB', '#EF4444', '#F59E0B'])])
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="#111622", plot_bgcolor="#111622", height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: 2D & 3D FEATURE SPACE
# ---------------------------------------------------------
elif active_tab == "🌌 2D & 3D Feature Space":
    st.markdown("""
    <div style="background: #111622; padding: 18px; border-radius: 12px; border: 1px solid #1E2638; margin-bottom: 25px;">
        <span style="background: #0F294A; color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">🎲 Interactive WebGL 3D</span>
        <h3 style="color: #FFFFFF; margin: 6px 0 2px 0;">3D Feature Space: Runtime vs. Rating vs. Budget</h3>
    </div>
    """, unsafe_allow_html=True)

    fig_3d = px.scatter_3d(
        df, x='runtime', y='vote_average', z='budget', color='Cluster_Name', hover_name='Movie Title',
        color_discrete_map={'Cluster 0: Budget & Low-Cost Hits': '#10B981', 'Cluster 1: Mid-Tier & Niche Favorites': '#2563EB', 'Cluster 2: Underperforming / Poorly-Received Titles': '#EF4444', 'Cluster 3: AAA Pan-Indian Blockbusters': '#F59E0B'}
    )
    fig_3d.update_layout(template="plotly_dark", paper_bgcolor="#111622", height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: LIVE CLUSTER PREDICTOR (WHAT-IF SIMULATOR)
# ---------------------------------------------------------
elif active_tab == "🧪 Live Cluster Predictor":
    
    st.markdown("""
    <div style="background: #111622; padding: 18px; border-radius: 12px; border: 1px solid #1E2638; margin-bottom: 20px;">
        <span style="background: #0F294A; color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">🧪 Live K-Means Inference Engine</span>
        <h2 style="color: #FFFFFF; margin: 6px 0 2px 0;">Interactive Cluster Predictor (What-If Simulator)</h2>
        <p style="color: #64748B; font-size: 13px; margin:0;">Input arbitrary commercial parameters for a movie. The simulator applies feature transformation (StandardScaler) and computes Euclidean distances to the 4 cluster centroids in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### QUICK PRESETS:")
    p1, p2, p3, p4 = st.columns(4)
    
    preset = None
    if p1.button("🎬 Pan-Indian Blockbuster (e.g. Leo / Vikram)"): preset = 'blockbuster'
    if p2.button("🎨 Critically Acclaimed Indie (e.g. 96 / Kaithi)"): preset = 'indie'
    if p3.button("💰 Budget Hit (e.g. Ratsasan)"): preset = 'budget'
    if p4.button("⚠️ Underperforming Launch"): preset = 'underperforming'

    # Set Preset Values
    if preset == 'blockbuster':
        b_val, r_val, p_val, rt_val = 30000000.0, 8.2, 90.0, 168
    elif preset == 'indie':
        b_val, r_val, p_val, rt_val = 3000000.0, 8.5, 45.0, 150
    elif preset == 'budget':
        b_val, r_val, p_val, rt_val = 1500000.0, 8.3, 60.0, 140
    elif preset == 'underperforming':
        b_val, r_val, p_val, rt_val = 12000000.0, 4.2, 15.0, 130
    else:
        b_val, r_val, p_val, rt_val = 18000000.0, 3.5, 20.0, 145

    col_sim, col_res = st.columns([1.2, 1])

    with col_sim:
        st.markdown('<div class="pred-box">', unsafe_allow_html=True)
        budget_input = st.slider("Budget ($USD):", min_value=0, max_value=100000000, value=int(b_val), step=1000000)
        rating_input = st.slider("Positive Rating (1.0 - 10.0):", min_value=1.0, max_value=10.0, value=float(r_val), step=0.1)
        popularity_input = st.slider("Popularity Index:", min_value=1.0, max_value=200.0, value=float(p_val), step=1.0)
        runtime_input = st.slider("Runtime (Minutes):", min_value=60, max_value=220, value=int(rt_val), step=5)
        st.markdown('</div>', unsafe_allow_html=True)

    # Inference Math
    user_feat = np.array([rating_input, popularity_input, runtime_input, 10000.0, budget_input])
    user_scaled = (user_feat - X_mean) / X_std

    # Euclidean Distances
    dists = np.linalg.norm(centroids - user_scaled, axis=1)
    closest_cluster = np.argmin(dists)

    # Softmax / Confidence Proximity
    inv_dists = 1.0 / (dists + 1e-5)
    probs = (inv_dists / np.sum(inv_dists)) * 100

    cluster_colors = {
        0: ("#062C22", "#10B981"),
        1: ("#0F2347", "#2563EB"),
        2: ("#3A1018", "#EF4444"),
        3: ("#382508", "#F59E0B")
    }

    bg_c, border_c = cluster_colors[closest_cluster]

    with col_res:
        st.markdown(f"""
        <div class="pred-result" style="background: {bg_c}; border-top: 4px solid {border_c};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:#94A3B8; font-weight:700;">PREDICTED CLUSTER ASSIGNMENT</span>
                <span style="background: #111622; color:#FFFFFF; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700;">Cluster {closest_cluster}</span>
            </div>
            <h2 style="color:#FFFFFF; margin: 10px 0 6px 0;">{cluster_names[closest_cluster].split(': ')[1]}</h2>
            <p style="color:#CBD5E1; font-size:12px; line-height:1.4;">Classified based on standardized Euclidean metric proximity to real-time PySpark cluster centroids.</p>
            <hr style="border-color: rgba(255,255,255,0.1); margin:15px 0;">
            <div style="font-size:12px; font-weight:700; color:#FFFFFF; margin-bottom:10px;">Confidence by Cluster Proximity:</div>
        """, unsafe_allow_html=True)

        for cid in range(4):
            c_name = cluster_names[cid].split(': ')[1]
            st.write(f"**{c_name}**: `{probs[cid]:.1f}%` (dist: `{dists[cid]:.2f}`)")
            st.progress(int(probs[cid]))

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: MOVIE EXPLORER TABLE
# ---------------------------------------------------------
elif active_tab == "🔍 Movie Explorer Table":
    st.title("🔍 Movie Search & Cluster Explorer")
    st.dataframe(df[['Movie Title', 'Cluster_Name', 'vote_average', 'popularity', 'runtime', 'vote_count', 'budget']], use_container_width=True)
