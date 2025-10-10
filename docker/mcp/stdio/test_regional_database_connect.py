import pymssql
import os
import traceback
import time
from dotenv import load_dotenv

load_dotenv()

# Your config
SQL_SERVER_HOST = "10.19.50.53\\UIPORCH"
SQL_SERVER_DATABASE = os.getenv("SQL_SERVER_DATABASE", "Nodinite")
SQL_SERVER_USE_WINDOWS_AUTH = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "false").lower() == "true"
SQL_SERVER_USERNAME = os.getenv("SQL_SERVER_USERNAME", None)
SQL_SERVER_PASSWORD = os.getenv("SQL_SERVER_PASSWORD", None)
SQL_SERVER_PORT = 1433  # Port that works! 49154!

MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds between retries

print(f"Attempting connection to: {SQL_SERVER_HOST}:{SQL_SERVER_PORT}")
print(f"Database: {SQL_SERVER_DATABASE}")
print(f"Windows Auth: {SQL_SERVER_USE_WINDOWS_AUTH}")
print(f"Username: {SQL_SERVER_USERNAME if SQL_SERVER_USERNAME else 'N/A'}")
print(f"Max retries: {MAX_RETRIES}")
print("="*60)

connection_successful = False

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Connecting...")
    
    try:
        if SQL_SERVER_USE_WINDOWS_AUTH:
            print("⚠️ Warning: Windows Auth from Docker may not work with remote servers!")
            conn = pymssql.connect(
                server=SQL_SERVER_HOST,
                database=SQL_SERVER_DATABASE,
                port=SQL_SERVER_PORT,
                timeout=300,
                login_timeout=60,
                as_dict=True
            )
        else:
            conn = pymssql.connect(
                server=SQL_SERVER_HOST,
                user=SQL_SERVER_USERNAME,
                password=SQL_SERVER_PASSWORD,
                database=SQL_SERVER_DATABASE,
                port=SQL_SERVER_PORT,
                timeout=300,
                login_timeout=60,
                as_dict=True
            )
        
        print("✅ Connection successful!")
        
        # Test query to verify connection is stable
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
        row = cursor.fetchone()
        print(f"✅ Query successful!")
        print(f"Sample row keys: {list(row.keys()) if row else 'No data'}")
        
        cursor.close()
        conn.close()
        
        connection_successful = True
        print(f"\n🎯 Stable connection established on attempt {attempt}!")
        break
        
    except Exception as e:
        print(f"❌ Attempt {attempt} failed: {str(e)[:150]}")
        
        if attempt < MAX_RETRIES:
            print(f"⏳ Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            print("\n❌ All retry attempts exhausted!")
            print("\nFull error details from last attempt:")
            traceback.print_exc()

if connection_successful:
    print("\n" + "="*60)
    print("✅ Connection test completed successfully!")
else:
    print("\n" + "="*60)
    print("❌ Connection test failed after all retries!")