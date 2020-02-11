import config
from causalimpact import CausalImpact
from dataIngestion import row_per_date_df
import pandas as pd
import numpy as np
import datetime as dt


def give_pre_post_df(df, pre, post):
	pre_period = [pd.to_datetime(np.min(df.index.values)),pre]
	post_period = [post,pd.to_datetime(np.max(df.index.values))]
	return {'pre_period':pre_period,'post_period':post_period}

def causal_impact(df,pre,post):
	df = row_per_date_df(df)
	pre_period = give_pre_post_df(df,pre,post)['pre_period']
	print("printing pre preiod")
	print(pre_period)
	post_period = give_pre_post_df(df,pre,post)['post_period']
	print()
	ci = CausalImpact(df,pre_period,post_period,prior_level_sd=None)
	return ci

def plot_ci(ci):
	plot = ci.plot(figsize=(15, 12))
	report = ci.summary(output = 'report')
	return {'Plot':plot, 'Report':report}

def get_st_dates(df):
	df = row_per_date_df(df)
	# Pre
	min_date = pd.to_datetime(np.min(df.index.values))
	min_date_obj = min_date.to_pydatetime()
	three_months = dt.timedelta(3*365/12)
	intervention = min_date_obj + three_months # for st
	max_date = pd.to_datetime(np.max(df.index.values))
	max_date_obj = max_date.to_pydatetime()
	one_week = dt.timedelta(1*365/52)	
	post_intervention = intervention + one_week
	return {'stpre':intervention,
         	'stpost':post_intervention,
          	'stmax': max_date_obj}