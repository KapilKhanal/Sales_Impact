


import numpy as np
import pandas as pd
import time, warnings
import datetime as dt


import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


import dataIngestion as di
import config as config
import kmeans_clustering as kc
import causalImpact as cimpact
import RFM as rfm

def main():
	original = di.read_file(config.NAME_DATA)
	df = di.remove_na(original,config.COLS_WITH_NA)
	df = di.remove_negative(df,config.NEGATIVE_COL)
	now = dt.date(config.REFERENCE_DATE)
	rfmtable = rfm.calculate_rfm(df,GROUP_BY_COL,LIST_COL_AGG,now)
	df_normalized = di.normalise_col(rfmtable)
	#Kmeans stuff
	matrix = kc.get_matrix(df_normalized)
	num_cluster = kc.give_num_clusters(matrix ,config.MIN_CLUSTER,config.MAX_CLUSTER)['best_n']
	rfm_with_label = kc.get_df_with_labels(num_cluster,rfmtable)
	#merging to original
	merged_df = di.join_rfm_orginial(df,rfm_with_label,config.JOIN_ON_COL)
	before_ci_df = di.give_cluster_df(merged_df,config.CLUSTER_WANT)

	#CI stuff
	ci = cimpact.causalImpact(before_ci_df)
	cimpact.plot_ci(ci)




if __name__ == '__main__':
	main()