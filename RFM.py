

def calculate_rfm(df,groupby_col , list_col_agg:dict , reference_date ):
	'''Takes in df dataframe groupsby groupby_col {Customer id} and looks for a dictionary of columns provided
	as list_col_agg where key is recency for column with date , frequency for num of sales 
	and monetary for total sales'''
	orders = df.copy()
	rfmTable = orders.groupby(groupby_col).agg({list_col_agg['recency']: lambda x: (reference_date- x.max()).days, # Recency
                                        list_col_agg['frequency']: lambda x:len(x),      # Frequency
                                        list_col_agg['monetary']: 'sum'}) # Monetary Value

	rfmTable['order_date'] = rfmTable['order_date'].astype(int)
	rfmTable.rename(columns={list_col_agg['recency']: 'recency', 
                         list_col_agg['frequency']: 'frequency', 
                         list_col_agg['monetary']: 'monetary_value'}, inplace=True)
	rfmTable = rfmTable.reset_index()
	return rfmTable






