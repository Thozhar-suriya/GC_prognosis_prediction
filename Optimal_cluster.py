import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

##Data
df = pd.read_csv("extracted_features.csv")
print(df.head())
features = df.iloc[:,1:101]
print(features)

##Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(scaled_features)

# Initialize variables to store scores
silhouette_scores = []
calinski_scores = []

# Define a range of cluster numbers to test
cluster_range = range(2, 11)

# Perform clustering and calculate scores for each cluster number
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(scaled_features)
    silhouette = silhouette_score(scaled_features, labels)
    calinski = calinski_harabasz_score(scaled_features, labels)
    silhouette_scores.append(silhouette)
    calinski_scores.append(calinski)

# Create subplots with shared x-axis
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Plot the silhouette score on the left side (ax1)
ax1.plot(cluster_range, silhouette_scores, label='Silhouette Score', linestyle='dashed', marker='o', color='k')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Silhouette Score', color='k')
ax1.tick_params('y', colors='k')

# Plot the Calinski-Harabasz score on the right side (ax2)
ax2.plot(cluster_range, calinski_scores, label='Calinski-Harabasz Score', linestyle='dotted', marker='v', color='k')
ax2.set_ylabel('Calinski-Harabasz Score', color='k')
ax2.tick_params('y', colors='k')

# Add legends and a title
plt.title('Optimal cluster number')
plt.grid(True)
plt.show()