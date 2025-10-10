import pymssql
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

# Your config
SQL_SERVER_HOST = "10.19.50.53\\UIPORCH"
SQL_SERVER_DATABASE = os.getenv("SQL_SERVER_DATABASE", "Nodinite")
SQL_SERVER_USE_WINDOWS_AUTH = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "false").lower() == "true"
SQL_SERVER_USERNAME = os.getenv("SQL_SERVER_USERNAME", None)
SQL_SERVER_PASSWORD = os.getenv("SQL_SERVER_PASSWORD", None)

# List of common SQL Server ports to try
PORTS_TO_TRY = [1433, 1434, 1435, 49152, 49153, 49154, 49155, 49156]

print(f"Testing connection to: {SQL_SERVER_HOST}")
print(f"Database: {SQL_SERVER_DATABASE}")
print(f"Windows Auth: {SQL_SERVER_USE_WINDOWS_AUTH}")
print(f"Username: {SQL_SERVER_USERNAME if SQL_SERVER_USERNAME else 'N/A'}")
print("\n" + "="*60 + "\n")

for port in PORTS_TO_TRY:
    print(f"Trying port {port}...")
    
    try:
        if SQL_SERVER_USE_WINDOWS_AUTH:
            conn = pymssql.connect(
                server=SQL_SERVER_HOST,
                database=SQL_SERVER_DATABASE,
                port=port,
                timeout=10,
                login_timeout=10,
                as_dict=True
            )
        else:
            conn = pymssql.connect(
                server=SQL_SERVER_HOST,
                user=SQL_SERVER_USERNAME,
                password=SQL_SERVER_PASSWORD,
                database=SQL_SERVER_DATABASE,
                port=port,
                timeout=10,
                login_timeout=10,
                as_dict=True
            )
        
        print(f"✅ ✅ ✅ CONNECTION SUCCESSFUL ON PORT {port}! ✅ ✅ ✅\n")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION as version, @@SERVERNAME as server")
        row = cursor.fetchone()
        print(f"Server Info: {row}")
        
        cursor.execute("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
        row = cursor.fetchone()
        print(f"✅ Query successful!")
        print(f"Sample row: {row}")
        
        cursor.close()
        conn.close()
        
        print(f"\n🎯 WORKING PORT FOUND: {port}")
        print("="*60)
        break
        
    except Exception as e:
        print(f"❌ Port {port} failed: {str(e)[:100]}")
        print()

print("\nPort scan complete.")