from queries import joinall, countall
from creating_tables import create_connection

def create_task():
    conn = create_connection(r"SalesDatabase.db")
    cur = conn.cursor()
    cur.execute(joinall)
    df = cur.fetchmany(7)
    print("after")
    return df

def create_count():
    conn = create_connection(r"SalesDatabase.db")
    cur = conn.cursor()
    cur.execute(countall)
    df = cur.fetchall()
    print(len(df))
    return df

if __name__ == "__main__":
    df = create_count()