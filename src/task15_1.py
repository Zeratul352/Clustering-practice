import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# K-Means Clustering Implementation
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k  # Number of clusters
        self.max_iters = max_iters  # Max iterations to converge
        self.centroids = []  # Cluster centroids
        self.labels = []  # Labels for data points


    def fit(self, data, initial_centroids=None):
        """
        Fits K-Means to the provided data.
        """
        self.clusters = {}

        if not initial_centroids:
            for i in range(self.k):
                self.centroids.append([random.random() * 5, random.random() * 5])

                cluster = {
                    'index': i,
                    'center': self.centroids[i],
                    'points': []
                }
                self.clusters[i] = cluster
        else:
            self.centroids = initial_centroids
            for i in range(self.k):


                cluster = {
                    'index': i,
                    'center': self.centroids[i],
                    'points': []
                }
                self.clusters[i] = cluster

        for point in data:
            label = self.predict(point)
            self.clusters[label]['points'].append(point)
            self.update()


        # Assign final labels
        self.labels = [self.get_cluster_label(point) for point in data]


    def predict(self, point):
        """
        Predicts the cluster label for a given point.
        """
        distances = []
        for center in self.centroids:
            distances.append(KMeans.euclidean_distance(point, center))

        cls = distances.index(min(distances))

        return cls

    def update(self):
        for i in range(len(self.clusters)):
            if len(self.clusters[i]['points']):
                self.centroids[i] = KMeans.compute_centroid(self.clusters[i])
                self.clusters[i]['center'] = self.centroids[i]
    def get_cluster_label(self, point):
        """
        Helper function to determine a cluster label.
        """
        distances = []
        for center in self.centroids:
            distances.append(KMeans.euclidean_distance(point, center))

        return distances.index(min(distances))


    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Computes Euclidean distance between two points.
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def compute_centroid(cluster):
        """
        Computes the centroid of a cluster.
        """

        x = 0
        y = 0
        for point in cluster['points']:
            x += point[0]
            y += point[1]
        x /= (len(cluster['points']))
        y /= (len(cluster['points']))
        return [x, y]
        #return np.mean(cluster, axis=0).tolist()


# Visualization function
def plot_clusters(data, labels, centroids):
    """
    Visualizes clustered data points.
    """
    data = np.array(data)
    labels = np.array(labels)
    centroids = np.array(centroids)

    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Annual Income ($1000)')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Input Data
    data = [
        [15, 39], [15, 81], [16, 6], [16, 77], [17, 40],
        [18, 6], [18, 94], [19, 3], [19, 72], [20, 44]
    ]
    #x, y = make_blobs(n_samples=500,n_features=2,centers=3,random_state=23)

    # Step 1: Train K-Means
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    #kmeans.fit(x.tolist())
    # Step 2: Visualize Results
    print("Centroids:", kmeans.centroids)
    print("Labels:", kmeans.labels)
    plot_clusters(data, np.array(kmeans.labels), kmeans.centroids)
