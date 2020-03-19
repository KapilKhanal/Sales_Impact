from queries import joinall, countall
from creating_tables import create_connection
import pandas as pd

def create_task():
    conn = create_connection(r"../data/SalesDatabase.db")
    cur = conn.cursor()
    cur.execute(joinall)
    df = cur.fetchmany(7)
    print("after")
    return df

def create_count():
    conn = create_connection(r"../data/SalesDatabase.db")
    cur = conn.cursor()
    cur.execute(countall)
    df = cur.fetchall()
    return df

if __name__ == "__main__":
    df = create_count()
    df = pd.DataFrame(df, columns=['TransactionID', 'InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country', 'StockCode', 'Quantity', 'CustomerID', 'Country', 'StockCode', 'Description', 'UnitPrice'])
    df.to_csv('../data/Joined_df.csv', index=False)
    
