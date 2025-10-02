import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","..", ".env"))
print(f"THIS IS THE .ENV PATH: {env_path}")
load_dotenv(dotenv_path=env_path)

host = os.getenv("SQL_SERVER_HOST")
database = os.getenv("SQL_SERVER_DATABASE")
driver = os.getenv("SQL_SERVER_DRIVER")
use_windows_auth = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "False").lower() == "true"

username = os.getenv("SQL_SERVER_USERNAME")
password = os.getenv("SQL_SERVER_PASSWORD")

# Build connection string
if use_windows_auth:
    # For Windows Authentication
    conn_str = f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};Trusted_Connection=yes;"
else:
    # For SQL Authentication
    conn_str = f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};UID={username};PWD={password}"

# Connect to SQL Server
try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Test query
    cursor.execute("SELECT @@VERSION;")
    row = cursor.fetchone()
    print("Connected to SQL Server:")
    print(row[0])

    cursor.close()
    conn.close()
except Exception as e:
    print("Connection failed:", e)