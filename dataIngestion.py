import typing
import logging
import numpy as np
import pandas as pd
def read_file(file,date_col):
	sales_file = pd.read_csv(file,parse_dates = [date_col])
	return sales_file

def remove_na(df, cols_with_na):
	df = df.copy()
	df.dropna(subset = cols_with_na,how = 'all',inplace = True)
	return df

def remove_negative(df,negative_col):
	df = df.copy()
	df = df[df[negative_col]>0]
	return df

def normalise_table(rfmtable):
	rfmtable = rfmtable.copy()
	rfmtable['recency'] = np.log(rfmtable['recency']+0.1)
	rfmtable['frequency'] = np.log(rfmtable['frequency'])
	rfmtable['monetary'] = np.log(rfmtable['monetary']+0.1)
	rfmtable = rfmtable.set_index('CustomerID')
	return rfmtable

def join_rfm_orginial(original,rfm,on_col):
	df = original.merge(rfm,how = 'left',on = [on_col])
	df = df.sort_values('cluster',ascending=True)
	return df

def give_cluster_df(merged_df_original,cluster):
	df_cluster = merged_df_original[merged_df_original['cluster'] == cluster]
	return df_cluster

def row_per_date_df(df_cluster):
	df = df_cluster.sort_values('date', ascending=True)
	#df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
	df = df.groupby(['date'],as_index =True).agg({'TotalCost':sum})
	return df
