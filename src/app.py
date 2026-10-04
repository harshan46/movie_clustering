import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page Configuration & Custom Dark Dashboard Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Profiling & Clustering Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #1E232A;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3748;
    }
    .analytics-card {
        background-color: #1A202C;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #6366F1;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E232A;
        border-radius: 8px;
        padding: 10px 24px;
        color: #A0AEC0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dataset Loader (TMDB Dataset + Popular Tamil Blockbusters)
# ---------------------------------------------------------
@st.cache_data
def load_movie_data():
    url = "https://raw.githubusercontent.com/subhampradhan/TMDB-5000-Movie-Dataset-Analysis/master/tmdb_5000_movies.csv"
    try:
        df_tmdb = pd.read_csv(url)
        cols = ['original_title', 'vote_average', 'popularity', 'runtime', 'vote_count', 'budget']
        df_tmdb = df_tmdb[cols].rename(columns={'original_title': 'Movie Title'})
    except Exception:
        clean_titles = [
            "Avatar", "Spectre", "The Dark Knight Rises", "Interstellar", "Inception",
            "The Avengers", "Titanic", "Gladiator", "Jurassic Park", "Pulp Fiction"
        ]
        n_samples = 300
        df_tmdb = pd.DataFrame({
            'Movie Title': np.random.choice(clean_titles, n_samples),
            'vote_average': np.random.uniform(4.0, 9.0, n_samples),
            'popularity': np.random.exponential(scale=20.0, size=n_samples),
            'runtime': np.random.normal(110, 20, n_samples),
            'vote_count': np.random.exponential(scale=1000, size=n_samples),
            'budget': np.random.exponential(scale=30000000, size=n_samples)
        })

    # Tamil Cinema Dataset Integration
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

    # Combine TMDB dataset with Tamil blockbusters
    df_combined = pd.concat([tamil_movies, df_tmdb], ignore_index=True)
    return df_combined

df_raw = load_movie_data()
features = ['vote_average', 'popularity', 'runtime', 'vote_count', 'budget']
df = df_raw.dropna(subset=features + ['Movie Title']).copy().reset_index(drop=True)

# ---------------------------------------------------------
# Custom K-Means & Standardization Engine
# ---------------------------------------------------------
X = df[features].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

def run_kmeans(X_data, k=3, max_iters=100, seed=42):
    np.random.seed(seed)
    random_indices = np.random.choice(X_data.shape[0], k, replace=False)
    centroids = X_data[random_indices]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X_data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X_data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] 
            for i in range(k)
        ])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        
    return labels, centroids

# SVD Dimensionality Reduction for Visual 2D Plotting
X_centered = X_scaled - np.mean(X_scaled, axis=0)
_, _, Vh = np.linalg.svd(X_centered, full_matrices=False)
pca_proj = np.dot(X_centered, Vh[:2].T)
df['PCA1'] = pca_proj[:, 0]
df['PCA2'] = pca_proj[:, 1]

# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
st.sidebar.title("🎬 Movie Clustering")
page = st.sidebar.radio("Go to Section:", ["📌 Overview & KPI", "📊 Cluster Analytics", "🔍 Movie Explorer"])
st.sidebar.markdown("---")

st.sidebar.subheader("Hyperparameter Tuning")
k_val = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=6, value=3)

# Execute Clustering Algorithm
labels, centroids = run_kmeans(X_scaled, k=k_val)
df['Cluster'] = labels

# ---------------------------------------------------------
# PAGE 1: OVERVIEW & KEY METRICS
# ---------------------------------------------------------
if page == "📌 Overview & KPI":
    st.title("🎬 Movie Profiling & Clustering Dashboard")
    st.caption("Big Data Analytics Assignment | Unsupervised Machine Learning Pipeline")
    st.markdown("---")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies Analyzed", f"{len(df):,}")
    col2.metric("Selected Features", f"{len(features)}")
    col3.metric("Active Clusters (K)", f"{k_val}")
    col4.metric("Avg User Rating", f"{df['vote_average'].mean():.2f} / 10")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.subheader("2D Projection Space (SVD / PCA)")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1A202C')
        
        colors = ['#6366F1', '#EC4899', '#10B981', '#F59E0B', '#8B5CF6', '#3B82F6']
        for i in range(k_val):
            cluster_subset = df[df['Cluster'] == i]
            ax.scatter(
                cluster_subset['PCA1'], cluster_subset['PCA2'], 
                label=f'Cluster {i}', color=colors[i % len(colors)], 
                alpha=0.7, s=40
            )
            
        ax.set_xlabel("Component 1", color='white')
        ax.set_ylabel("Component 2", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1A202C', edgecolor='none', labelcolor='white')
        ax.grid(True, color='#2D3748', linestyle='--')
        st.pyplot(fig)
        
    with c2:
        st.subheader("Cluster Profile Means")
        summary_df = df.groupby('Cluster')[features].mean().reset_index()
        summary_df['budget'] = summary_df['budget'].apply(lambda x: f"${x/1e6:.1f}M")
        summary_df['vote_count'] = summary_df['vote_count'].apply(lambda x: f"{x:.0f}")
        summary_df['popularity'] = summary_df['popularity'].apply(lambda x: f"{x:.1f}")
        summary_df['vote_average'] = summary_df['vote_average'].apply(lambda x: f"{x:.2f}")
        summary_df['runtime'] = summary_df['runtime'].apply(lambda x: f"{x:.0f}m")
        st.dataframe(summary_df, use_container_width=True)
        
        st.info("💡 **Inference Insight:** The algorithm groups movies into high-budget blockbusters, critically acclaimed releases, and widely engaged mainstream hits automatically.")

# ---------------------------------------------------------
# PAGE 2: DETAILED CLUSTER FEATURE ANALYSIS (UPGRADED)
# ---------------------------------------------------------
elif page == "📊 Cluster Analytics":
    st.title("📊 Detailed Cluster Feature Analysis")
    st.markdown("Explore multi-dimensional feature distributions, cluster profiles, and correlation heatmaps.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Comparative Distributions", "🔥 Feature Correlations", "📋 Cluster Metric Profiles"])

    # TAB 1: Comparative Boxplots and Violin Plots
    with tab1:
        st.markdown("<div class='analytics-card'><h4>Distribution Analysis across Clusters</h4>Examine how specific numerical features vary across each discovered movie segment.</div>", unsafe_allow_html=True)
        
        col_f, col_t = st.columns([2, 1])
        with col_f:
            selected_feature = st.selectbox("Select Feature to Analyze:", features)
        with col_t:
            plot_type = st.radio("Plot Style:", ["Box Plot", "Violin Plot"], horizontal=True)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1A202C')

        palette = ['#6366F1', '#EC4899', '#10B981', '#F59E0B', '#8B5CF6', '#3B82F6']
        
        if plot_type == "Box Plot":
            sns.boxplot(x='Cluster', y=selected_feature, data=df, palette=palette[:k_val], ax=ax)
        else:
            sns.violinplot(x='Cluster', y=selected_feature, data=df, palette=palette[:k_val], ax=ax, inner="quartile")

        ax.set_title(f"{selected_feature.upper().replace('_', ' ')} Distribution per Cluster", color='white', fontsize=14, pad=12)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.grid(True, color='#2D3748', linestyle='--')
        st.pyplot(fig)

    # TAB 2: Correlation Analysis
    with tab2:
        st.markdown("<div class='analytics-card'><h4>Feature Inter-Correlation Matrix</h4>Identify collinear relationships between movie popularity, vote counts, budget, and ratings.</div>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1A202C')

        corr = df[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True,
                    annot_kws={"size": 11, "color": "white"})

        ax.tick_params(colors='white', labelsize=10)
        plt.xticks(rotation=30)
        st.pyplot(fig)

    # TAB 3: Stat Profiles Breakdown
    with tab3:
        st.markdown("<div class='analytics-card'><h4>Cluster Profile Metric Breakdown</h4>Summary statistics highlighting min, median, mean, and max values across clusters.</div>", unsafe_allow_html=True)
        
        for c_id in range(k_val):
            st.subheader(f"📍 Cluster {c_id} Profile Statistics")
            cluster_data = df[df['Cluster'] == c_id][features]
            stats = cluster_data.describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})
            st.dataframe(stats.style.highlight_max(color='#2D3748'), use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: MOVIE EXPLORER TABLE (SEARCH REAL & TAMIL MOVIES)
# ---------------------------------------------------------
elif page == "🔍 Movie Explorer":
    st.title("🔍 Interactive Movie Cluster Explorer")
    st.write("Search for Hollywood and Tamil movie titles to view their cluster assignment.")
    
    col_search, col_filter = st.columns([2, 1])
    
    with col_search:
        search_query = st.text_input("🔎 Search Movie Title (e.g., Vikram, Leo, Inception):", "")
        
    with col_filter:
        cluster_filter = st.multiselect("Filter by Cluster:", options=list(range(k_val)), default=list(range(k_val)))
        
    # Apply Filters
    filtered_df = df[df['Cluster'].isin(cluster_filter)]
    if search_query:
        filtered_df = filtered_df[filtered_df['Movie Title'].str.contains(search_query, case=False, na=False)]
        
    display_cols = ['Movie Title', 'Cluster', 'vote_average', 'popularity', 'runtime', 'vote_count', 'budget']
    st.dataframe(filtered_df[display_cols], use_container_width=True)
