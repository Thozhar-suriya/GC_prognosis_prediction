## Libraries
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
from scipy import stats

##Data
df = pd.read_csv("cox_features.csv")
print(df.head())
features = df.iloc[:,1:10]
print(features)

##Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(scaled_features)

# Define range of cluster numbers to try
cluster_range = range(2, 11)

# Perform k-means clustering with k=2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(scaled_features)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the scatter plot
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.title("K-means Clustering (k=2)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
