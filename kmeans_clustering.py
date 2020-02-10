from dataIngestion import normalise_table
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import matplotlib.pyplot as plt
import dataIngestion as di



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
    ax.set_title('The Elbow Method showing the optimal k')
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
	cluster1=df.loc[df['cluster'] == 0]
	cluster2=df.loc[df['cluster'] == 1]
	cluster3=df.loc[df['cluster'] == 2]

	scatter1 = dict(
	    mode = "markers",
	    name = "Cluster 1",
	    type = "scatter3d",
	    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
	    marker = dict( size=2, color='green')
	)
	scatter2 = dict(
	    mode = "markers",
	    name = "Cluster 2",
	    type = "scatter3d",
	    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
	    marker = dict( size=2, color='blue')
	)
	scatter3 = dict(
	    mode = "markers",
	    name = "Cluster 3",
	    type = "scatter3d",
	    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
	    marker = dict( size=2, color='red')
	)
	cluster1 = dict(
	    alphahull = 5,
	    name = "Cluster 1",
	    opacity = .1,
	    type = "mesh3d",
	    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
	    color='green', showscale = True
	)
	cluster2 = dict(
	    alphahull = 5,
	    name = "Cluster 2",
	    opacity = .1,
	    type = "mesh3d",
	    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
	    color='blue', showscale = True
	)
	cluster3 = dict(
	    alphahull = 5,
	    name = "Cluster 3",
	    opacity = .1,
	    type = "mesh3d",
	    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
	    color='red', showscale = True
	)
	layout = dict(
     autosize = False,
     width=1000,
     height=800,
     title = 'Interactive Cluster Shapes in 3D',
     scene = dict(
         xaxis = dict( zeroline=True, title='Recency'),
         yaxis = dict( zeroline=True, title='Frequency'),
         zaxis = dict( zeroline=True, title='Monetary'),
         )
	)
	fig = dict(data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout)
	return fig

