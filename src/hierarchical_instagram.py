import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv('Instagram visits clustering.csv')

x = data.iloc[:, [1, 2]].values
x = np.array(x)

dendrogram(linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Instagram visit score')
plt.ylabel('Euclidean Distances')
plt.show()


hierarchi = AgglomerativeClustering(
    n_clusters=5, metric='euclidean', linkage='ward')

y_ = hierarchi.fit_predict(x)

plt.scatter(x[y_ == 0, 0], x[y_ == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_ == 1, 0], x[y_ == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_ == 2, 0], x[y_ == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_ == 3, 0], x[y_ == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_ == 4, 0], x[y_ == 4, 1], s=100,
            c='magenta', label='Cluster 5')
plt.title('Clusters of Instagram Visits')
plt.xlabel('Instagram visit score')
plt.ylabel('Spending Rank')
plt.legend()
plt.show()
