"""KMeans class for fitting a KNN model"""
import numpy as np
import random
import matplotlib.pyplot as plt
from functions import save_fig
from sklearn.metrics import silhouette_score

class KMeans:

    def __init__(self, k=3, tolerance=1e-4, max_iter=1000):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iter
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self, df):
        """
        First step of the algorithm: initializes k clusters.
        The first one is set at random and the next k-1 via probabilistic approach
        known as KMeans++

        Parameters:

        df : pd.DataFrame
        -------
        Returns
        
        numpy array containing the centroids
        """

        # Transform the DataFrame to a numpy array for calculations
        # Initialize an empty list to save the centroids
        df = df.to_numpy()
        centroids = []

        # Initialiwe randomly the first centroid
        first_centroid = random.choice(df)
        centroids.append(first_centroid)
        
        # The next k-1 centroids will be chosen from a probabilistic approach
        # For evey centroid already chosed, calculates the distance between the centroid to every point
        # in the Dataframe and use its mean as a probability weight to chose the next centroid
        # Repeat untill all centroids have been chosen.
        for _ in range(self.k-1):

            distances = np.sum([np.linalg.norm(centroid - df, axis=1) for centroid in centroids], axis=0)
            distances /= np.sum(distances)

            next_centroid, = np.random.choice(range(len(df)), 1, p=distances)
            centroids.append(df[next_centroid])

        return np.array(centroids)

    def assign_clusters(self, df, centroids):
        """
        Second step of the algorithm: assign every point to a centroid

        Parameters:

        df : pd.DataFrame
        centroids : numpy.ndarray
        -------
        Returns:
        numpy array containing the assaigned clusters
        """

        # Transform the Dataframe to a numpy array for calculations
        # Initialize a list to hold the index of the nearest centroid for each data point
        df = df.to_numpy()
        cluster_assignments = []
    
        # Iterate over each data point in the dataset
        for point in df:
            # Compute the distance from the point to each centroid
            distances = np.linalg.norm(point - centroids, axis=1)
        
            # Find the index of the closest centroid as the minimum distance found before
            closest_centroid_index = np.argmin(distances)
        
            # Append the index to the cluster assignments
            cluster_assignments.append(closest_centroid_index)
    
        return np.array(cluster_assignments)

    def update_centroids(self, df):
        """
        Third step of the algorithm: update the centroids.
        Computes the mean distance between every points of a given cluster,
        and use it as the new position for the centroid of that cluster.

        Parameters:

        df : pd.DataFrame
        -------
        Returns
        A numpy array containing the new centroids
        """

        # Initialize an empty list to save the centroid's new position
        new_centroids = []

        # Iterate over k clusters to find the centroid's new position
        for i in range(self.k):

            # Get from the DataFrame the rows that belongs to the cluster i
            cluster_points = df[self.clusters == i]

            # If the given cluster have at least one point assigned to it,
            # upodates its position using the mean ofthe position of its points.
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
            # if not, updates its position randomly
            else:
                new_centroid = df.iloc[np.random.choice(len(df))]
            
            new_centroids.append(new_centroid)
        
        return np.array(new_centroids)

    def fit(self, df):
        """
        Fit method to iterate over the above steps until finding all cluster's
        final positions given some convergency criteria.
        In this case, the criteria is to stop when the new position of the each cluster
        don't differ from the old position by a given tolerance level.

        Parameters:
        
        df : pd.DataFrame
        """
        
        # Initialize centroids
        centroids = self.initialize_centroids(df)
    
        # Iterate until convergence or maximum iterations
        for _ in range(self.max_iterations):
            # Step 1: Assign each point to the nearest centroid
            self.clusters = self.assign_clusters(df, centroids)
        
            # Step 2: Compute new centroids
            new_centroids = self.update_centroids(df)
        
            # Check for convergence (if centroids don't change significantly)
            if np.linalg.norm(new_centroids - centroids) < self.tolerance:
                break
        
            # Update centroids for next iteration
            self.centroids = new_centroids
    
    def plot_clusters(self, df, fig_name:str, title:str):
        """
        Function to plot the clusters.

        Parameters:
        df : pd.DataFrame
        fig_name : string
        title : string
        """
               
        df = df.to_numpy()
        
        plt.figure(figsize=(8, 6))

        for i in range(self.k):
            points = df[self.clusters == i]
            plt.scatter(points[:, 0], points[:, 1], s=30, label=f'Cluster {i+1}')

        centroids = np.array(self.centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

        plt.xlabel("Flipper Length (mm)")
        plt.ylabel("Bill Length (mm)")
        plt.title(f"K-Means Clustering of Penguins Dataset - {title}")
        plt.legend()
        save_fig(f"{fig_name}")
        plt.show()
    
    def silhouette_analysis(self, df, max_k):
        """
        Function to run a silhouette analysis for finding the best number of clusters

        Parameters:
        df : pd.DataFrame
        max_k : int
        -------
        Returns:

        best_k : int containing the best number of clusters k
        silhouette_scores : list containing the silhouette scores for every k
        """

        # Initializes empty list to save the scores at every step
        silhouette_scores = []
        # The number of clusters starting with minimum of 2
        k_values = range(2, max_k+1)

        for k in k_values:

            self.k = k
            # Fits KMeans for every k
            self.fit(df)

            # Calculates the silhouette score and save it
            score = silhouette_score(df, self.clusters)
            silhouette_scores.append(score)
            print(f"Silhouette Score for k={k}: {score}")

        # Choses the best k as the k_value corresponding to the biggest score
        best_k = k_values[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)

        print(f"\nBest k: {best_k} with a Silhouette Score of {best_score}")
        return best_k, silhouette_scores

    def plot_silhouette_scores(self, silhouette_scores, max_k):
        """
        Function to plot the silhouette score for each k

        Parameters:
        silhouette scores : list of scores 
        max_k : int
        """

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters k")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        save_fig("sillhouette_scores")
        plt.show()