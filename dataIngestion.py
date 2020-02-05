import typing
import logging
import numpy as np
def read_excel(file):
	sales_file = pd.read_excel(file)
	return sales_file


def remove_na(df:pd.DataFrame, cols_with_na:list)->pd.DataFrame:
	df = df.copy()
	df.dropna(subset = cols_with_na,how = 'all',inplace = True)
	return df

def remove_negative(df,negative_col:String):
	df = df.copy()
	df = df[df[negative_col]>0]
	return df


def normalise_col(rfmtable):
	rfmtable = rfmtable.copy()
	rfmtable['recency'] = np.log(rfmtable['recency']+0.1)
	rfmtable['frequency'] = np.log(rfmtable['frequency'])
	rfmtable['monetary'] = np.log(rfmtable['monetary']+0.1)
	return rfmtable

def join_rfm_orginial(original,rfm,on_col):
	df = original.merge(rfm,how = 'left',on = [on_col])
	return df










