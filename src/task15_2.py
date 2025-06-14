import numpy as np
import random
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # Maximum distance to consider a neighbor
        self.min_samples = min_samples  # Minimum points to form a cluster
        self.labels = []  # Cluster labels for each point
        self.noise_label = -1  # Label for noise points

    def fit(self, data):
        """
        Perform DBSCAN clustering on the input data.
        """
        self.clusters = {}
        #random sample to start class
        self.data_copy = data.copy()
        i = 0
        #noise class
        self.clusters[-1] = []
        while len(self.data_copy) > 0:
            core = random.sample(self.data_copy, 1)[0]
            index = data.index(core)
            self.clusters[i] = []
            region = self.region_query(data, index)
            if len(region) +1>= self.min_samples:
                self.clusters[i].append(core)
            else:
                self.clusters[-1].append(core)
                self.data_copy.remove(core)
                continue
            self.expand_cluster(data, data.index(core), self.region_query(data, data.index(core)), i)
            i += 1


        for dot in data:
            for k in range(-1, len(self.clusters) - 1):
                if dot in self.clusters[k]:
                    self.labels.append(k)

        return self.labels






    def expand_cluster(self, data, point_idx, neighbors, cluster_id):
        """
        Expand the cluster from the core point.
        """
        point = data[point_idx]
        self.clusters[cluster_id].append(point)
        self.data_copy.remove(point)
        for dot in neighbors:
            if dot in self.data_copy:
                self.expand_cluster(data, data.index(dot), self.region_query(data, data.index(dot)), cluster_id)


    def region_query(self, data, point_idx):
        """
        Find all points within `eps` distance of the given point.
        """
        neighbors = []
        for point in data:
            if point == data[point_idx]:
                continue
            if DBSCAN.euclidean_distance(point, data[point_idx]) < self.eps:
                neighbors.append(point)
        return neighbors


    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Visualization function for clusters
def plot_clusters(data, labels):
    """
    Visualize DBSCAN results.
    """
    import matplotlib.pyplot as plt
    data = np.array(data)
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            # Noise points
            color = "red"
            label_name = "Noise"
        else:
            color = np.random.rand(3,)
            label_name = f"Cluster {label}"

        cluster_points = data[np.array(labels) == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=label_name)

    plt.title("DBSCAN Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


# Main Execution
if __name__ == "__main__":
    data = [
        [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]
    ]
    eps = 2
    min_samples = 2

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    print("Labels:", dbscan.labels)  # Output cluster and noise labels
    plot_clusters(data, dbscan.labels)  # Visualize results
