import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Clustering App", layout="wide")

st.title("🎬 Movie Profiling & Clustering App")
st.write("An unsupervised machine learning application using **K-Means Clustering** and **SVD/PCA**.")

# 1. Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/subhampradhan/TMDB-5000-Movie-Dataset-Analysis/master/tmdb_5000_movies.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'vote_average': np.random.uniform(4.0, 9.0, n_samples),
            'popularity': np.random.exponential(scale=20.0, size=n_samples),
            'runtime': np.random.normal(110, 20, n_samples),
            'vote_count': np.random.exponential(scale=1000, size=n_samples),
            'budget': np.random.exponential(scale=30000000, size=n_samples)
        })
    return df

df = load_data()

# 2. Sidebar Controls
st.sidebar.header("Cluster Settings")
k_clusters = st.sidebar.slider("Select Number of Clusters (K):", min_value=2, max_value=6, value=3)

features = ['vote_average', 'popularity', 'runtime', 'vote_count', 'budget']
df_clean = df[features].dropna().reset_index(drop=True)

# 3. Standardization
X = df_clean[features].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

# 4. Custom K-Means Algorithm
def kmeans_custom(X_data, k, max_iters=100, seed=42):
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
        
    return labels

labels = kmeans_custom(X_scaled, k=k_clusters)
df_clean['Cluster'] = labels

# 5. Dimensionality Reduction (SVD)
X_centered = X_scaled - np.mean(X_scaled, axis=0)
_, _, Vh = np.linalg.svd(X_centered, full_matrices=False)
pca_proj = np.dot(X_centered, Vh[:2].T)
df_clean['PCA1'] = pca_proj[:, 0]
df_clean['PCA2'] = pca_proj[:, 1]

# 6. Display Output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Cluster Profile Summary (Mean Values)")
    summary = df_clean.groupby('Cluster')[features].mean()
    st.dataframe(summary.style.highlight_max(axis=0))

with col2:
    st.subheader("2D Visualization (SVD Projection)")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(k_clusters):
        c_data = df_clean[df_clean['Cluster'] == i]
        ax.scatter(c_data['PCA1'], c_data['PCA2'], label=f'Cluster {i}', color=colors[i % len(colors)], alpha=0.7)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
