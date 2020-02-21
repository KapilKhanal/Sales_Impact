import csv
import sqlite3
from queries import *
from sqlite3 import Error


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
        next(reader)
        rows = [row for row in reader]
        print(rows[1:10])
        cur.execute(drop_customers_table)
        cur.execute(create_customers_table)
        for row in rows:
            cur.execute('INSERT INTO customerstable VALUES (?, ?)',row[1:])
        print("processed customer table")

def process_transaction_table(cur, file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cur.execute(drop_transactions_table)
        cur.execute(create_transactions_table)
        for row in reader:
            cur.execute('INSERT INTO transactionstable VALUES (?,?,?,?,?,?)', row)
        print("processed transactions table")   

def process_items_table(cur, file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cur.execute(drop_items_table)
        cur.execute(create_items_table)
        for row in reader:
            cur.execute('INSERT INTO itemstable VALUES (?, ?, ?)', row[1:])
        print("processed items table")


if __name__ == '__main__':
    conn = create_connection(r"SalesDatabase.db")
    cur = conn.cursor()
    process_customer_table(cur, 'CustomerTable.csv')
    conn.commit()
    process_transaction_table(cur, 'TransactionsTable.csv')
    conn.commit()
    process_items_table(cur, 'ItemsTable.csv')
    conn.commit()