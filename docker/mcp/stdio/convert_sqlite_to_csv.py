
# ------------- IDENTIFY THE STRUCTURE OF THE SQLITE DATABASE -------------
# import sqlite3

# db_path = "compacted.db"
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# try:
#     cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
#     rows = cursor.fetchall()
#     for name, schema in rows:
#         print(f"Table: {name}\nSchema: {schema}\n")
# except sqlite3.DatabaseError as e:
#         print("❌ Not a valid SQLite database:", e)
# finally:
#     conn.close()

# ------------- CONVERT SQLITE DATABASE TO CSV FILES -------------
import sqlite3
import pandas as pd
import os

DB_PATH = "compacted.db"
OUTPUT_DIR = "csv_export"
ROW_LIMIT = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Only export Invoice and Invoice_Line tables
tables = ["Invoice", "Invoice_Line"]

for table in tables:
    query = f"SELECT * FROM {table} LIMIT {ROW_LIMIT}"
    df = pd.read_sql_query(query, conn)
    
    csv_path = os.path.join(OUTPUT_DIR, f"{table}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Exported {len(df)} rows from '{table}' → {csv_path}")

conn.close()
print("\nAll done! CSVs are ready in the 'csv_export' folder.")

