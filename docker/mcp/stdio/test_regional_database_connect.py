import pymssql
import os
from dotenv import load_dotenv

load_dotenv()

# Your config
SQL_SERVER_HOST = os.getenv("SQL_SERVER_HOST", "localhost\\SQLEXPRESS")
SQL_SERVER_DATABASE = os.getenv("SQL_SERVER_DATABASE", "InvoiceDB")
SQL_SERVER_USE_WINDOWS_AUTH = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "true").lower() == "true"
SQL_SERVER_USERNAME = os.getenv("SQL_SERVER_USERNAME", None)
SQL_SERVER_PASSWORD = os.getenv("SQL_SERVER_PASSWORD", None)

print(f"Attempting connection to: {SQL_SERVER_HOST}")
print(f"Database: {SQL_SERVER_DATABASE}")
print(f"Windows Auth: {SQL_SERVER_USE_WINDOWS_AUTH}")

try:
    if SQL_SERVER_USE_WINDOWS_AUTH:
        conn = pymssql.connect(
            server=SQL_SERVER_HOST,
            database=SQL_SERVER_DATABASE,
            timeout=300,  # 5 mins timeout
            login_timeout=300,  # 5 minutes for connection
            as_dict=True
        )

    else:
        conn = pymssql.connect(
            server=SQL_SERVER_HOST,
            user=SQL_SERVER_USERNAME,
            password=SQL_SERVER_PASSWORD,
            database=SQL_SERVER_DATABASE,
            timeout=300,  # 5 mins timeout
            login_timeout=300,  # 5 minutes for connection
            as_dict=True
        )
    
    print("✅ Connection successful!")
    
    # Test query
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
    row = cursor.fetchone()
    print(f"✅ Query successful: {row}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    