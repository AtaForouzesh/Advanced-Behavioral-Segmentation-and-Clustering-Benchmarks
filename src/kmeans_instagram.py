import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('Instagram visits clustering.csv')

x = data.iloc[:, [1, 2]].values

kmeans = KMeans(n_clusters=5, random_state=42, init="k-means++")
kmeans.fit(x)
y_ = kmeans.predict(x)

plt.scatter(x[y_ == 0, 0], x[y_ == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_ == 1, 0], x[y_ == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_ == 2, 0], x[y_ == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_ == 3, 0], x[y_ == 3, 1], s=100, c='orange', label='Cluster 4')
plt.scatter(x[y_ == 4, 0], x[y_ == 4, 1], s=100, c='purple', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', label='Centroids')
plt.xlabel('Instagram visit score')
plt.ylabel('Spending Rank')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
