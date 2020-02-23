import csv
import sqlite3
from queries import *
from sqlite3 import Error
import os


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def process_customer_table(cur, file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        rows = [row[1:] for row in reader]
        cur.execute(drop_customers_table)
        cur.execute(create_customers_table)
        cur.executemany('INSERT INTO customerstable VALUES (?, ?)',rows)
        print("processed customer table")

def process_transaction_table(cur, file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        cur.execute(drop_transactions_table)
        cur.execute(create_transactions_table)
        cur.executemany('INSERT INTO transactionstable VALUES (?,?,?,?,?,?,?)', rows)
        print("processed transactions table")   

def process_items_table(cur, file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        rows = [row[1:] for row in reader]
        cur.execute(drop_items_table)
        cur.execute(create_items_table)
        cur.executemany('INSERT INTO itemstable VALUES (?, ?, ?)', rows)    
        print("processed items table")


if __name__ == '__main__':
    p = os.path.join(os.path.dirname(__file__),'../data/SalesDatabase.db')
    conn = create_connection(p)
    cur = conn.cursor()
    process_customer_table(cur, '../data/CustomerTable.csv')
    conn.commit()
    process_transaction_table(cur, '../data/TransactionsTable.csv')
    conn.commit()
    process_items_table(cur, '../data/ItemsTable.csv')
    conn.commit()