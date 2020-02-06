NAME_DATA = "Online_Retail.xlxs"
COLS_WITH_NA = ['CustomerID']
NEGATIVE_COL = 'Quantity'
GROUP_BY_COL = 'CustomerID'
LIST_COL_AGG = {'recency':'date','frequency':"InvoiceNo" , 'monetary':'TotalCost'}
N_CLUSTERS = 5
REFERENCE_DATE = '2011/12/09' #end of the sale period
INTERVENTION_DATE = '20111002'
JOIN_ON_COL = 'CustomerID'

MIN_CLUSTER = 2
MAX_CLUSTER = 10

CLUSTER_WANT = 1


