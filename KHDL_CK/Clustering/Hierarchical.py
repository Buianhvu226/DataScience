import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
file_path = r"C:\KHDL_CK\Clean\Dat_new\train_data.csv"
rawdata = pd.read_csv(file_path)

# Select relevant features
selected_features = ['price($)', 'Reference number', 'Scope of delivery', 'Year of production', 'name', 'Location', 'Case material', 'Movement']
data = rawdata[selected_features].values

# Apply Gaussian Mixture Models (GMM)
gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(data)

# Print the means and covariances of the components
print("Means of each component:")
print(gmm.means_)
print("Covariances of each component:")
print(gmm.covariances_)

# Plot the clustering results in 2D
def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Cluster {label+1}')
    plt.scatter(centroids[:, 0], centroids[:, 3], marker='*', s=300, c='black', label='Centroids')
    plt.xlabel('price($)')
    plt.ylabel('name')
    plt.title('GMM Clustering Results')
    plt.legend()
    plt.show()

# Plot 2D clusters
plot_clusters(data, labels, gmm.means_)

# Plot the clustering results in 3D
def plot_clusters_3d(data, labels, centroids):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], label=f'Cluster {label+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=300, c='black', label='Centroids')
    ax.set_xlabel('price($)')
    ax.set_ylabel('name')
    ax.set_zlabel('Movement')
    plt.title('GMM Clustering Results - 3D')
    plt.legend()
    plt.show()

# Plot 3D clusters
plot_clusters_3d(data, labels, gmm.means_)
