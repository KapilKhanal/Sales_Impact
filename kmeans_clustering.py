from dataIngestion import normalise_col

def get_matrix(df):
	return df.as_matrix()

 def give_num_clusters(matrix ,min_cluster,max_cluster):
 	distortions = []
 	N_clusters = range(min_cluster,max_cluster)
 	for n in N_clusters:
 		kmeans = Kmeans(init = 'k-means++', n_clusters = n , n_init  = 100)
 		kmeans.fit(matrix)
 		clusters = kmeans.predict(matrix)
 		distortions.append(sum(np.min(cdist(matrix, kmeans.cluster_centers_,'euclidean'),axis = 1))/matrix.shape[0])
 	kn = KneeLocator(list(N_clusters),distortions,S = 0.1,curve ='convex',direction = 'decreasing' )
 	return {'best_n':kn.knee,'plot':kn.plot_knee()}
 	


def get_df_with_labels(num_cluster,rfmtable):
	'''RFM table should be normalized already'''
	matrix = get_matrix(rfmtable)
	kmeans = Kmeans(init = 'k-means++', n_clusters = num_cluster , n_init  = 100)
	kmeans.fit(matrix)
	labels = kmeans.label_
	rfmtable['cluster'] = labels
	rfmtable.reset_index(inplace= True)
return rfmtable





