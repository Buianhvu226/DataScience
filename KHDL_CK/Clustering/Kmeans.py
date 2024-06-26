import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import StandardScaler

# Load dataset
rawdata = pd.read_csv("C:\KHDL_CK\Clean\patekphilippe_clean_data_pure.csv")

# selected_features = ['price($)', 'name', 'Case area(mm2)', 'Scope of delivery', 'Movement', 'Reference number', 'Condition', 'Case material', 'Year of production']
selected_features = ['price($)', 'name', 'Reference number', 'Year of production']
label_encoder = LabelEncoder()
rawdata['name'] = label_encoder.fit_transform(rawdata['name'])
rawdata['Reference number'] = label_encoder.fit_transform(rawdata['Reference number'])


# Select features and convert to numpy array
data = rawdata[selected_features].values

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Initialize KMeans models
cls = [KMeans(n_clusters=i, max_iter=300, random_state=42).fit(data) for i in range(4, 20)]

# Calculate Silhouette scores for each KMeans model
Sil_score = [round(metrics.silhouette_score(data, i.labels_), 5) for i in cls]

# Print results
print("KMeans clustering results:")
for i, model in enumerate(cls):
    print(f"Number of clusters (K): {i+2}")
    print("Centroids:")
    print(model.cluster_centers_)
    print("Silhouette Score:", Sil_score[i])
    print("-------------------------")

# Choose the best K based on Silhouette Score
best_K = np.argmax(Sil_score) + 4 

# Get the best KMeans model
best_model = cls[best_K - 4]

# Plot clustering results

# Function to plot clusters in 2D
# Giá, Điểm đánh giá
def plot_clusters(data, centroids, labels):
    """
    Plot clustering results in 2D.

    Parameters:
    - data: Data points.
    - centroids: Centroids of clusters.
    - labels: Cluster labels of data points.
    """
    plt.figure(figsize=(10, 6))
    for i in range(len(centroids)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
        
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black', label='Centroids')
    plt.title('KMeans Clustering Results')
    plt.xlabel('price($)')
    plt.ylabel('name')
    plt.legend()
    plt.show()

# Plot the clustering results in 2D

plot_clusters(data[:, :2], best_model.cluster_centers_[:, :2], best_model.labels_)







# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function to plot clusters in 3D
# Giá, Điểm đánh giá, Số lượt đánh giá
def plot_clusters_3d_1(data, centroids, labels):
    """
    Plot clustering results in 3D.

    Parameters:
    - data: Data points.
    - centroids: Centroids of clusters.
    - labels: Cluster labels of data points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(centroids)):
        ax.scatter(data[labels == i, 0], data[labels == i, 1], data[labels == i, 2], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=300, c='black', label='Centroids')
    ax.set_xlabel('price($)')
    ax.set_ylabel('name')
    ax.set_zlabel('Reference number')
    plt.title('KMeans Clustering Results - 3D')
    plt.legend()
    plt.show()


# Plot the clustering results in 3D
plot_clusters_3d_1(data, best_model.cluster_centers_, best_model.labels_)

# # Function to plot clusters in 3D
# # Điểm đánh giá, Khoảng cách, Số lượng dịch vụ
def plot_clusters_3d_2(data, centroids, labels):
    """
    Plot clustering results in 3D.

    Parameters:
    - data: Data points.
    - centroids: Centroids of clusters.
    - labels: Cluster labels of data points.
    """
    # ['price($)', 'Reference number', 'Scope of delivery', 'Year of production', 'name', 'Location', 'Case material', 'Movement']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(centroids)):
        ax.scatter(data[labels == i, 0], data[labels == i, 1], data[labels == i, 3], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 3], marker='*', s=300, c='black', label='Centroids')
    ax.set_xlabel('price($)')
    ax.set_ylabel('name')
    ax.set_zlabel('Year of production')
    plt.title('KMeans Clustering Results - 3D')
    plt.legend()
    plt.show()

# Plot the clustering results in 3D
plot_clusters_3d_2(data, best_model.cluster_centers_, best_model.labels_)
