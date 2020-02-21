drop_customers_table = "DROP TABLE IF EXISTS customerstable"
drop_transactions_table = "DROP TABLE IF EXISTS transactionstable"
drop_items_table = "DROP TABLE IF EXISTS itemstable"

# Making Database
create_customers_table = ("""
CREATE TABLE customerstable(
    CustomerID VARCHAR(12),
    Country VARCHAR(100),
    PRIMARY KEY(CustomerID, Country)
)""")

create_transactions_table = ("""
CREATE TABLE transactionstable(
    TransactionID VARCHAR(100) PRIMARY KEY,
    InvoiceNo VARCHAR(100),
    InvoiceDate VARCHAR(100),
    CustomerID INTEGER,
    StockCode VARCHAR(100),
    Quantity INTEGER
)""")

create_items_table = ("""
CREATE TABLE itemstable(
    StockCode VARCHAR(100),
    Description VARCHAR(100),
    UnitPrice REAL
)""")

# Joins
joinall = """
SELECT *
FROM
(SELECT *
FROM transactionstable as T
LEFT JOIN customerstable as C ON
T.CustomerID = C.CustomerID) as TC
LEFT JOIN itemstable as I ON
TC.StockCode = I.StockCode
"""
countall = """
SELECT *
FROM
(SELECT *
FROM transactionstable as T
LEFT JOIN customerstable as C ON
T.CustomerID = C.CustomerID) as TC
LEFT JOIN itemstable as I ON
TC.StockCode = I.StockCode
"""



