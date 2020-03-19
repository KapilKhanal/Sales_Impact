NAME_DATA = "Sales_df.csv"
DATE_COL = 'InvoiceDate'
COLS_WITH_NA = ['CustomerID']
NEGATIVE_COL = 'Quantity'
GROUP_BY_COL = 'CustomerID'
LIST_COL_AGG = {'recency':'date','frequency':"InvoiceNo" , 'monetary':'TotalCost'}
N_CLUSTERS = 5
REFERENCE_YEAR = 2011
REFERENCE_Month = 12
REFERENCE_day = 9 #end of the sale period
INTERVENTION_DATE = '20111002'
EXPERIMENT_DATE = '20111010'
JOIN_ON_COL = 'CustomerID'

MIN_CLUSTER = 2
MAX_CLUSTER = 10
# Cluster_df date range shoule contain Intervention and Experiment DATE.
# Streamlit should let people choose cluster first then select dates as Experiment and Intervention
CLUSTER_WANT = 2
