from src.data.dataIngestion import normalise_table
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import matplotlib.pyplot as plt
from src.data import dataIngestion as di



def get_matrix(df):
    return df.values

def give_num_clusters(matrix ,min_cluster, max_cluster):
    distortions = []
    N_clusters = range(min_cluster, max_cluster)
    for n in N_clusters:
        kmeans = KMeans(init = 'k-means++', n_clusters = n , n_init  = 100)
        kmeans.fit(matrix)
        distortions.append(sum(np.min(cdist(matrix, kmeans.cluster_centers_,'euclidean'), axis = 1)) / matrix.shape[0])
    kn = KneeLocator(list(N_clusters), distortions, S = 0.1, curve = 'convex', direction = 'decreasing')
    fig, ax = plt.subplots()
    ax.plot(N_clusters, distortions, 'bx-')
    ax.set_xlabel('N')
    ax.set_ylabel('Distortion')
    ax.set_title('The Elbow Method showing the optimal customer clusters')
    ax.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow")
    return {'Best_N': kn.knee, 'Plot':plt.tight_layout()}


def get_df_with_labels(num_cluster, rfmtable):
	'''RFM table should be normalized already'''
	matrix = get_matrix(rfmtable)
	kmeans = KMeans(init = 'k-means++', n_clusters = num_cluster , n_init  = 100)
	kmeans.fit(matrix)
	#clusters = kmeans.predict(matrix)
	labels = kmeans.labels_
	rfmtable['cluster'] = labels
	return rfmtable

def plot_clusters(df):
    fig, ax = plt.subplots()
    ax.scatter(df['recency'], df['monetary'], c=df['cluster'], s=df['frequency'], cmap='viridis')
    return {'Plot':plt.tight_layout()}