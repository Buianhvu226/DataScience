import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

# Load dataset
rawdata = pd.read_csv("C:\KHDL_CK\Clean\Dat_new\\train_data.csv")

# Select relevant features
selected_features = ['price($)', 'Reference number', 'Scope of delivery', 'Year of production', 'name', 'Location', 'Case material', 'Movement']
data = rawdata[selected_features].values

# Initialize GMM models
gmm_models = [GaussianMixture(n_components=i, random_state=42).fit(data) for i in range(4, 11)]

# Calculate Silhouette scores for each GMM model
Sil_score = [round(metrics.silhouette_score(data, gmm.predict(data)), 5) for gmm in gmm_models]

# Print results
print("Gaussian Mixture Models (GMM) clustering results:")
for i, model in enumerate(gmm_models):
    print(f"Number of components: {i+2}")
    print("Means:")
    print(model.means_)
    print("Covariances:")
    print(model.covariances_)
    print("Silhouette Score:", Sil_score[i])
    print("-------------------------")

# Choose the best number of components based on Silhouette Score
best_components = np.argmax(Sil_score) + 4
best_gmm_model = gmm_models[best_components - 4]

# Plot clustering results

# Function to plot clusters in 2D
def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(10, 6))
    for i in range(len(centroids)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
        
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black', label='Centroids')
    plt.title('Gaussian Mixture Models (GMM) Clustering Results')
    plt.xlabel('price($)')
    plt.ylabel('name')
    plt.legend()
    plt.show()

# Plot the clustering results in 2D
plot_clusters(data[:, :2], best_gmm_model.predict(data), best_gmm_model.means_[:, :2])

# Function to plot clusters in 3D
def plot_clusters_3d(data, labels, centroids):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(centroids)):
        ax.scatter(data[labels == i, 0], data[labels == i, 1], data[labels == i, 2], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=300, c='black', label='Centroids')
    ax.set_xlabel('price($)')
    ax.set_ylabel('name')
    ax.set_zlabel('Movement')
    plt.title('Gaussian Mixture Models (GMM) Clustering Results - 3D')
    plt.legend()
    plt.show()

# Plot the clustering results in 3D
plot_clusters_3d(data, best_gmm_model.predict(data), best_gmm_model.means_)
