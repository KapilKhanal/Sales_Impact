from config import *
from dataIngestion import row_per_date_df

def give_pre_post_df(df):
	
	pre_period = [pd.to_datetime(np.min(df.index.values)),pd.to_datetime(INTERVENTION_DATE)]
	post_period = [pd.to_datetime(INTERVENTION_DATE),pd.to_datetime(np.max(df.index.values))]
	return {'pre_period':pre_period,'post_period':post_period}

def causal_impact(df):
	df = row_per_date_df(df)
	pre_period = give_pre_post_df(df)['pre_period']
	post_period = give_pre_post_df(df)['post_period']
	ci = CausalImpact(df,pre_period,post_period,prior_level_sd = None)
	return ci

def plot_ci(ci):
	ci.plot()
	print('printing report \n')
	print(ci.summary('report'))


