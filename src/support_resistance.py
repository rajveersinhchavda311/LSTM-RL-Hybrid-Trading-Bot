def detect_support_resistance(df, window=20):
    highs = df['High'].rolling(window).max()
    lows = df['Low'].rolling(window).min()
    return highs, lows

def detect_fractals(df, window=5):
    highs = df['High'].rolling(window, center=True).max()
    lows = df['Low'].rolling(window, center=True).min()
    return highs, lows

from sklearn.cluster import KMeans
import numpy as np

def kmeans_support_resistance(df, n_clusters=5):
    # Combine high and low prices for clustering
    price_levels = np.concatenate([df['High'].values, df['Low'].values]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(price_levels)
    levels = sorted(kmeans.cluster_centers_.flatten())
    return levels