import os
import pyodbc
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Load environment variables
SQL_SERVER_HOST = os.getenv("SQL_SERVER_HOST", "localhost\\SQLEXPRESS")
SQL_SERVER_DATABASE = os.getenv("SQL_SERVER_DATABASE", "InvoiceDB")
SQL_SERVER_DRIVER = os.getenv("SQL_SERVER_DRIVER", "ODBC Driver 17 for SQL Server")
SQL_SERVER_USE_WINDOWS_AUTH = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "true").lower() == "true"

SQL_SERVER_USERNAME = os.getenv("SQL_SERVER_USERNAME", None)
SQL_SERVER_PASSWORD = os.getenv("SQL_SERVER_PASSWORD", None)

def test_connection():
    try:
        # Build connection string
        if SQL_SERVER_USE_WINDOWS_AUTH:
            connection_string = (
                f"DRIVER={{{SQL_SERVER_DRIVER}}};"
                f"SERVER={SQL_SERVER_HOST};"
                f"DATABASE={SQL_SERVER_DATABASE};"
                f"Trusted_Connection=yes;"
            )
        else:
            connection_string = (
                f"DRIVER={{{SQL_SERVER_DRIVER}}};"
                f"SERVER={SQL_SERVER_HOST};"
                f"DATABASE={SQL_SERVER_DATABASE};"
                f"UID={SQL_SERVER_USERNAME};"
                f"PWD={SQL_SERVER_PASSWORD};"
            )
        
        print("Attempting to connect...")
        print(f"Server: {SQL_SERVER_HOST}")
        print(f"Database: {SQL_SERVER_DATABASE}")
        
        # Connect to database
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # Test query - selects one row from a system table
        cursor.execute("SELECT TOP 1 DB_NAME() AS DatabaseName, GETDATE() AS CurrentDateTime")
        
        # Fetch and print the result
        row = cursor.fetchone()
        if row:
            print("\n✓ Connection successful!")
            print(f"Database: {row.DatabaseName}")
            print(f"Server Time: {row.CurrentDateTime}")
        
        # Close connection
        cursor.close()
        conn.close()
        
    except pyodbc.Error as e:
        print(f"\n✗ Connection failed!")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error!")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()