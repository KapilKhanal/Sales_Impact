import config
from causalimpact import CausalImpact
from dataIngestion import row_per_date_df
import pandas as pd
import numpy as np

def give_pre_post_df(df):
	pre_period = [pd.to_datetime(np.min(df.index.values)),pd.to_datetime(config.INTERVENTION_DATE)]
	post_period = [pd.to_datetime(config.EXPERIMENT_DATE),pd.to_datetime(np.max(df.index.values))]
	return {'pre_period':pre_period,'post_period':post_period}

def causal_impact(df):
	df = row_per_date_df(df)
	pre_period = give_pre_post_df(df)['pre_period']
	print("printing pre preiod")
	print(pre_period)
	post_period = give_pre_post_df(df)['post_period']
	print()
	ci = CausalImpact(df,pre_period,post_period,prior_level_sd=None)
	return ci

def plot_ci(ci):
	plot = ci.plot(figsize=(15, 12))
	report = ci.summary(output = 'report')
	return {'Plot':plot, 'Report':report}
