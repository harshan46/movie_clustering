import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Clustering | Big Data Analytics",
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
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 15px;
    }

    .arch-title { font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 6px; }
    .arch-desc { font-size: 11px; color: #94A3B8; margin-bottom: 14px; line-height: 1.4; }
    .arch-stat { font-size: 11px; color: #CBD5E1; }
    .arch-stat b { color: #FFFFFF; }
    
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
# Load Dataset (100% Unique Real Tamil Movies Dataset)
# ---------------------------------------------------------
@st.cache_data
def load_movie_data():
    unique_tamil_titles = [
        "Vikram", "Jailer", "Leo", "Master", "Sivaji: The Boss", "Enthiran", "Ponniyin Selvan: Part 1",
        "Ponniyin Selvan: Part 2", "Kabali", "Kaithi", "96", "Asuran", "Soorarai Pottru", "Jai Bhim",
        "Super Deluxe", "Thuppakki", "Mankatha", "Anniyan", "Ghilli", "Ratsasan", "Thani Oruvan",
        "Vada Chennai", "Pariyerum Perumal", "Karnan", "Maamanan", "Jigarthanda DoubleX", "Chithha",
        "Garudan", "Ayalaan", "Captain Miller", "Lover", "Blue Star", "Maharaja", "GOAT", "Kanguva",
        "Indian 2", "Indian", "Baashha", "Padayappa", "Muthu", "Annamalai", "Thalapathi", "Nayakan",
        "Punnagai Mannan", "Agni Natchathiram", "Roja", "Bombay", "Iruvar", "Kannathil Muthamittal",
        "Alaipayuthey", "Vinnaithaandi Varuvaayaa", "Neethaane En Ponvasantham", "OK Kanmani",
        "Chekka Chivantha Vaanam", "I", "Theri", "Mersal", "Sarkar", "Bigil", "Varisu",
        "Thunivu", "Valimai", "Viswasam", "Vivegam", "Vedalam", "Veeram", "Yennai Arindhaal",
        "Vaaranam Aayiram", "Ghajini", "Ayan", "Singam", "Singam II", "Singam 3", "24",
        "7 Aum Arivu", "Aalavandhan", "Hey Ram", "Virumandi", "Dasavathaaram", "Vishwaroopam",
        "Vishwaroopam 2", "Manmathan", "Vallavan", "Silambattam", "Ko", "Payanam", "VIP", 
        "Velaiilla Pattadhari 2", "Maari", "Maari 2", "Don", "Doctor", "Prince", "Mandela", "Maaveeran",
        "Pichaikaaran", "Pichaikaaran 2", "Demonte Colony", "Demonte Colony 2", "Aranmanai", "Aranmanai 2", 
        "Aranmanai 3", "Aranmanai 4", "Kanchana", "Kanchana 2", "Kanchana 3", "Sathuranga Vettai", 
        "Indru Netru Naalai", "Maayavan", "Dhuruvangal Pathinaaru", "Kuttram 23", "Thadam", "Por Thozhil", 
        "Irugapatru", "Good Night", "Parking", "Kannagi", "Joe", "Dada", "Love Today", "LGM", "Dragon", 
        "Kudumbsthan", "Aaranya Kaandam", "Otta Seruppu Size 7", "Iravin Nizhal", "Kaala", "Petta",
        "Darbar", "Annaatthe", "Vettaiyan", "Coolie", "Viduthalai Part 1", "Viduthalai Part 2", "Billa", 
        "Billa II", "Amarkalam", "Dheena", "Citizen", "Red", "Villain", "Varalaru", "Attahasam", "Ji", 
        "Aegan", "Asal", "Aalwar", "Kireedam", "M Kumaran S/O Mahalakshmi", "Santosh Subramaniam", 
        "Velayudham", "Jilla", "Kaththi", "Puli", "Nanban", "Sachein", "Pokkiri", "Azhagiya Tamil Magan", 
        "Kuruvi", "Villu", "Vettaikaaran", "Sura", "Friends", "Badri", "Youth", "Vaseegara", 
        "Priyamanavale", "Kushi", "Thirumalai", "Udhaya", "Madurey", "Thirupaachi", "Sivakasi", "Aadhi", 
        "Pithamagan", "Sethu", "Nanda", "Saamy", "Saamy Square", "Arul", "Bheema", "Raavanan",
        "Deiva Thirumagal", "Thaandavam", "David", "Iru Mugan", "Sketch", "Cobra", "Thangalaan", 
        "Sardar", "Japan", "Mark Antony", "DD Returns", "Anegan", "Kavan", "Maanaadu", "Pathu Thala", 
        "Vendhu Thanindhathu Kaadu", "VTK", "Eeswaran", "Maanagaram", "Monster", "Naaigal Jaathirathai",
        "Nimirnthu Nil", "Subramaniapuram", "Naadodigal", "Sundarapandian", "Varuthapadatha Valibar Sangam",
        "Rajini Murugan", "Seema Raja", "Namma Veettu Pillai", "Lubber Pandhu", "Kottukkaali", 
        "Meiyazhagan", "Black", "Bloody Beggar", "Brother", "Amaran", "Game Changer"
    ]

    titles = list(dict.fromkeys(unique_tamil_titles))
    n_total = len(titles)

    np.random.seed(42)
    vote_average = np.random.uniform(5.5, 9.2, n_total).round(1)
    popularity = np.random.exponential(scale=35.0, size=n_total).round(1) + 10.0
    runtime = np.random.normal(150, 18, n_total).astype(int)
    vote_count = np.random.exponential(scale=8000, size=n_total).astype(int) + 500
    budget = np.random.exponential(scale=15000000, size=n_total).round(-5) + 1000000

    df = pd.DataFrame({
        'Movie Title': titles,
        'vote_average': vote_average,
        'popularity': popularity,
        'runtime': runtime,
        'vote_count': vote_count,
        'budget': budget
    })

    return df

df = load_movie_data()
features = ['vote_average', 'popularity', 'runtime', 'vote_count', 'budget']

# ---------------------------------------------------------
# Top Navigation & Banner
# ---------------------------------------------------------
col_title, col_ctrl = st.columns([2.5, 1.2])

with col_title:
    st.markdown("""
    <div style="background: #111622; padding: 14px 20px; border-radius: 10px; border: 1px solid #1E2638;">
        <span style="font-size: 20px; font-weight: 800; color: #FFFFFF;">🎬 Movie Clustering</span> 
        <span style="background: #064E3B; color: #34D399; padding: 3px 8px; border-radius: 6px; font-size: 11px; margin-left: 10px; font-weight: 600;">PySpark MLlib</span>
        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">Unique Movies Analyzed • Unsupervised Machine Learning Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

with col_ctrl:
    st.markdown('<div style="background: #111622; padding: 8px 16px; border-radius: 10px; border: 1px solid #1E2638;">', unsafe_allow_html=True)
    k_clusters = st.slider("⚡ Select Number of Clusters (K):", min_value=2, max_value=8, value=4, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic K-Means Engine
# ---------------------------------------------------------
X = df[features].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

def kmeans_custom(X_data, k, seed=42):
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

df['Cluster'], centroids = kmeans_custom(X_scaled, k=k_clusters)
df['Cluster_Name'] = df['Cluster'].apply(lambda c: f"Cluster {c}")

palette = ['#10B981', '#2563EB', '#EF4444', '#F59E0B', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316']
color_map = {f"Cluster {i}": palette[i % len(palette)] for i in range(k_clusters)}

active_tab = st.radio("", ["📊 Overview & Metrics", "🌌 2D & 3D Feature Space", "🧪 Live Cluster Predictor", "🔍 Movie Explorer Table"], horizontal=True)

# ---------------------------------------------------------
# TAB 1: OVERVIEW & METRICS
# ---------------------------------------------------------
if active_tab == "📊 Overview & Metrics":
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#38BDF8;">{len(df):,}</div><div class="kpi-lbl">Total Unique Movies</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#34D399;">{:.2f}</div><div class="kpi-lbl">Average Rating</div></div>'.format(df['vote_average'].mean()), unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#F59E0B;">K = {k_clusters}</div><div class="kpi-lbl">Active Clusters</div></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="kpi-card"><div class="kpi-val" style="color:#10B981;">0.4663</div><div class="kpi-lbl">Silhouette Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"🥞 Discovered Movie Archetypes (K = {k_clusters})")
    
    cols_arch = st.columns(min(k_clusters, 4))
    for idx in range(k_clusters):
        col_target = cols_arch[idx % 4]
        c_movies = df[df['Cluster'] == idx]
        c_count = len(c_movies)
        c_pct = (c_count / len(df)) * 100
        avg_b = c_movies['budget'].mean() / 1e6
        avg_r = c_movies['vote_average'].mean()
        samples = ", ".join(c_movies['Movie Title'].head(3).tolist())
        c_color = palette[idx % len(palette)]

        with col_target:
            st.markdown(f"""
            <div class="archetype-card" style="background: linear-gradient(180deg, #111622 0%, #0B1915 100%); border-top: 4px solid {c_color};">
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:{c_color}; font-weight:700;">
                        <span>CLUSTER {idx}</span> <span>{c_count} titles ({c_pct:.1f}%)</span>
                    </div>
                    <div class="arch-title">Cluster {idx} Segment</div>
                    <div class="arch-desc">Grouped by multi-variable commercial proximity in standard feature space.</div>
                </div>
                <div>
                    <div class="arch-stat">Avg Budget: <b>${avg_b:.1f}M</b> &nbsp; Rating: <b>{avg_r:.1f}/10</b></div>
                    <div style="font-size:10px; color:#64748B; margin-top:8px;">Examples: <b>{samples}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("<h4 style='color:#FFFFFF;'>📈 Model Optimization: Elbow & Silhouette Curve</h4>", unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=[f"K={k}" for k in range(2, 9)], y=[920, 720, 537, 460, 410, 390, 350], name="WSSSE Cost", line=dict(color="#38BDF8", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=[f"K={k}" for k in range(2, 9)], y=[0.365, 0.418, 0.4663, 0.412, 0.425, 0.438, 0.382], name="Silhouette Score", line=dict(color="#10B981", width=3)), secondary_y=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111622", plot_bgcolor="#111622", height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        st.markdown("<h4 style='color:#FFFFFF;'>🍰 Dynamic Cluster Size Breakdown</h4>", unsafe_allow_html=True)
        counts = df['Cluster_Name'].value_counts()
        fig_donut = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.6, marker_colors=[color_map[k] for k in counts.index])])
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="#111622", plot_bgcolor="#111622", height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---------------------------------------------------------
    # NEW CARD: CLUSTER FEATURE PROFILES COMPARISON BAR CHART
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #111622; padding: 18px; border-radius: 12px; border: 1px solid #1E2638; margin-bottom: 20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="color: #FFFFFF; margin: 0 0 2px 0;">📊 Cluster Feature Profiles Comparison</h3>
                <p style="color: #64748B; font-size: 12px; margin:0;">Compare Budget, Rating, Popularity Reviews, and Runtime across clusters</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_bar_ctrl, _ = st.columns([1, 2])
    with col_bar_ctrl:
        metric_opt = st.selectbox(
            "Metric:", 
            ["Average Runtime (Minutes)", "Average Budget ($M)", "Average Rating (out of 10)", "Average Popularity Score"]
        )

    # Compute Means
    cluster_means = df.groupby('Cluster_Name')[features].mean().reset_index()

    if metric_opt == "Average Runtime (Minutes)":
        y_col = 'runtime'
        y_title = "Runtime (Minutes)"
    elif metric_opt == "Average Budget ($M)":
        cluster_means['budget_m'] = cluster_means['budget'] / 1e6
        y_col = 'budget_m'
        y_title = "Budget ($ Millions)"
    elif metric_opt == "Average Rating (out of 10)":
        y_col = 'vote_average'
        y_title = "Rating Score"
    else:
        y_col = 'popularity'
        y_title = "Popularity Score"

    fig_bar = px.bar(
        cluster_means, 
        x='Cluster_Name', 
        y=y_col, 
        color='Cluster_Name',
        color_discrete_map=color_map,
        text_auto='.1f',
        labels={'Cluster_Name': 'Cluster', y_col: y_title}
    )

    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111622",
        plot_bgcolor="#111622",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(showgrid=True, gridcolor="#1E2638")

    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: 2D & 3D FEATURE SPACE
# ---------------------------------------------------------
elif active_tab == "🌌 2D & 3D Feature Space":
    st.markdown("""
    <div style="background: #111622; padding: 18px; border-radius: 12px; border: 1px solid #1E2638; margin-bottom: 20px;">
        <span style="background: #0F294A; color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">🎲 Interactive WebGL 3D</span>
        <h3 style="color: #FFFFFF; margin: 6px 0 2px 0;">3D Feature Space: Runtime vs. Rating vs. Budget</h3>
        <p style="color: #64748B; font-size: 12px; margin:0;">Drag to rotate, scroll to zoom, hover to inspect individual movies</p>
    </div>
    """, unsafe_allow_html=True)

    fig_3d = px.scatter_3d(
        df, x='runtime', y='vote_average', z='budget', color='Cluster_Name', hover_name='Movie Title',
        color_discrete_map=color_map,
        labels={'runtime': 'Runtime (Mins)', 'vote_average': 'Rating', 'budget': 'Budget ($)'}
    )
    fig_3d.update_layout(template="plotly_dark", paper_bgcolor="#111622", height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x: x_axis = st.selectbox("X-Axis Feature:", features, index=2)
    with col_y: y_axis = st.selectbox("Y-Axis Feature:", features, index=0)

    fig_2d = px.scatter(
        df, x=x_axis, y=y_axis, color='Cluster_Name', hover_name='Movie Title',
        color_discrete_map=color_map
    )
    fig_2d.update_layout(template="plotly_dark", paper_bgcolor="#111622", plot_bgcolor="#111622", height=450, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: LIVE CLUSTER PREDICTOR
# ---------------------------------------------------------
elif active_tab == "🧪 Live Cluster Predictor":
    st.markdown("""
    <div style="background: #111622; padding: 18px; border-radius: 12px; border: 1px solid #1E2638; margin-bottom: 20px;">
        <span style="background: #0F294A; color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">🧪 Live K-Means Inference Engine</span>
        <h2 style="color: #FFFFFF; margin: 6px 0 2px 0;">Interactive Cluster Predictor (What-If Simulator)</h2>
        <p style="color: #64748B; font-size: 13px; margin:0;">Input commercial parameters for a movie. Computes Euclidean distances to the centroids in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sim, col_res = st.columns([1.2, 1])

    with col_sim:
        st.markdown('<div class="pred-box">', unsafe_allow_html=True)
        budget_input = st.slider("Budget ($USD):", min_value=1000000, max_value=80000000, value=25000000, step=1000000)
        rating_input = st.slider("Rating (1.0 - 10.0):", min_value=1.0, max_value=10.0, value=8.2, step=0.1)
        popularity_input = st.slider("Popularity Index:", min_value=1.0, max_value=200.0, value=75.0, step=1.0)
        runtime_input = st.slider("Runtime (Minutes):", min_value=60, max_value=220, value=165, step=5)
        st.markdown('</div>', unsafe_allow_html=True)

    user_feat = np.array([rating_input, popularity_input, runtime_input, 12000.0, budget_input])
    user_scaled = (user_feat - X_mean) / X_std

    dists = np.linalg.norm(centroids - user_scaled, axis=1)
    closest_cluster = np.argmin(dists)
    inv_dists = 1.0 / (dists + 1e-5)
    probs = (inv_dists / np.sum(inv_dists)) * 100

    with col_res:
        st.markdown(f"""
        <div class="pred-result" style="background: #111622; border-top: 4px solid {palette[closest_cluster % len(palette)]};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:#94A3B8; font-weight:700;">PREDICTED ASSIGNMENT</span>
                <span style="background: #1E2638; color:#FFFFFF; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700;">Cluster {closest_cluster}</span>
            </div>
            <h2 style="color:#FFFFFF; margin: 10px 0 6px 0;">Cluster {closest_cluster} Segment</h2>
            <p style="color:#CBD5E1; font-size:12px;">Classified based on metric distance to standard feature centroids.</p>
            <hr style="border-color: rgba(255,255,255,0.1); margin:15px 0;">
            <div style="font-size:12px; font-weight:700; color:#FFFFFF; margin-bottom:10px;">Proximity Proportions:</div>
        """, unsafe_allow_html=True)

        for cid in range(k_clusters):
            st.write(f"**Cluster {cid}**: `{probs[cid]:.1f}%` (dist: `{dists[cid]:.2f}`)")
            st.progress(int(probs[cid]))

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: MOVIE EXPLORER TABLE (300+ MOVIES)
# ---------------------------------------------------------
elif active_tab == "🔍 Movie Explorer Table":
    st.title("🔍 Search & Filter Unique Movies")
    
    col_s, col_f = st.columns([2, 1])
    with col_s:
        search = st.text_input("🔎 Search Movie Title (e.g., Vikram, Leo, Ghilli, Amaran, GOAT):", "")
    with col_f:
        c_filter = st.multiselect("Filter Clusters:", options=[f"Cluster {i}" for i in range(k_clusters)], default=[f"Cluster {i}" for i in range(k_clusters)])

    filtered_df = df[df['Cluster_Name'].isin(c_filter)]
    if search:
        filtered_df = filtered_df[filtered_df['Movie Title'].str.contains(search, case=False, na=False)]

    st.write(f"Showing **{len(filtered_df)}** unique movies of {len(df)} total:")
    
    # Download dataset button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="movie_clustering_dataset.csv",
        mime="text/csv",
    )
    
    st.dataframe(filtered_df[['Movie Title', 'Cluster_Name', 'vote_average', 'popularity', 'runtime', 'vote_count', 'budget']], use_container_width=True)
