"""Implement a simple usage of a KNN model to cluster points from a 2D dataset"""
from functions import save_fig, load_dataset
from kmeans_class import KMeans

# Loads the dataset
df = load_dataset()

# Get the two variables we are going to use to cluster
df = df[["flipper_length_mm", "bill_length_mm"]]

# Initialize and fits the model
k_means = KMeans()
k_means.fit(df)

print(k_means.centroids)
print(k_means.clusters)

# Plot the clusters and save the figure
k_means.plot_clusters(df, "clusters", "initial clusters")

# Runs a silhouette analysis for k different clusters
best_k, silhouette_scores = k_means.silhouette_analysis(df, max_k=10)
k_means.plot_silhouette_scores(silhouette_scores, max_k=10)

# Fits the model again but now using the best k found from the above analysis 
best_kmeans = KMeans(k=best_k)
best_kmeans.fit(df)

print(best_kmeans.centroids)
print(best_kmeans.clusters)

# Plot and save the figure showing the best arrangement of clusters
best_kmeans.plot_clusters(df, "best_clusters", "best clusters")