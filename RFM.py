

def calculate_rfm(df,groupby_col , list_col_agg:dict , reference_date ):
	'''Takes in df dataframe groupsby groupby_col {Customer id} and looks for a dictionary of columns provided
	as list_col_agg where key is recency for column with date , frequency for num of sales 
	and monetary for total sales'''
	orders = df.copy()
	orders['date'] = orders['InvoiceDate'].dt.date
	rfmTable = orders.groupby([groupby_col]).agg({list_col_agg['recency']: lambda x: max((reference_date- x).dt.days), # Recency
                                        list_col_agg['frequency']: lambda x:len(x),      # Frequency
                                        list_col_agg['monetary']: 'sum'}) # Monetary Value

	#rfmTable['order_date'] = rfmTable['order_date'].astype(int)
	rfmTable.rename(columns={list_col_agg['recency']: 'recency', 
                         list_col_agg['frequency']: 'frequency', 
                         list_col_agg['monetary']: 'monetary'}, inplace=True)
	rfmTable = rfmTable.reset_index()
	return rfmTable






