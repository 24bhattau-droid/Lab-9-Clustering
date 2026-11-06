# Lab 9: K-Means Clustering and Hierarchical Clustering
# Dataset: Iris dataset from scikit-learn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dataset shape:", X.shape)
print("\nFirst five rows:")
print(X.head())

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("\nK-Means cluster counts:")
print(pd.Series(kmeans_labels).value_counts().sort_index())
print("K-Means silhouette score:",
      round(silhouette_score(X_scaled, kmeans_labels), 3))

# Step 4: Apply Hierarchical (Agglomerative) clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

print("\nHierarchical cluster counts:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())
print("Hierarchical silhouette score:",
      round(silhouette_score(X_scaled, hierarchical_labels), 3))

# Step 5: Visualize K-Means clusters
plt.figure(figsize=(7, 5))
plt.scatter(
    X["petal length (cm)"],
    X["petal width (cm)"],
    c=kmeans_labels,
    s=45
)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("K-Means Clustering on Iris Dataset")
plt.tight_layout()
plt.show()

# Step 6: Visualize Hierarchical clusters
plt.figure(figsize=(7, 5))
plt.scatter(
    X["petal length (cm)"],
    X["petal width (cm)"],
    c=hierarchical_labels,
    s=45
)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Hierarchical Clustering on Iris Dataset")
plt.tight_layout()
plt.show()
