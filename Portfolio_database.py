import sqlite3
def create_database():
    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolio(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        quantity INTEGER,
        price REAL,
    )
    """)
def add_purchase_date_column():

    conn = sqlite3.connect("portfolio.db")
    cur = conn.cursor()
        
    cur.execute("""
        ALTER TABLE portfolio
        ADD COLUMN purchase_date DATE
        """)
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    add_purchase_date_column()
