import pyodbc

# Configuration (must match your bash script)
DB_HOST = "localhost"
DB_PORT = 1433
DB_DATABASE = "Compacted"
DB_USERNAME = "sa"
DB_PASSWORD = "YourPassword123!"

def test_connection():
    print("\n🔍 Testing MSSQL Connection...\n")

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_HOST},{DB_PORT};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USERNAME};"
        f"PWD={DB_PASSWORD};"
        f"TrustServerCertificate=yes;"
    )

    try:
        print("Connecting to SQL Server...")
        conn = pyodbc.connect(conn_str, timeout=5)
        print("✅ Connection established.")

        cursor = conn.cursor()
        cursor.execute("SELECT DB_NAME() AS CurrentDatabase;")
        row = cursor.fetchone()
        print(f"📘 Current Database: {row.CurrentDatabase}")

        cursor.execute("SELECT 1 AS TestResult;")
        row = cursor.fetchone()
        print(f"✅ Test Query Result: {row.TestResult}")

        conn.close()
        print("\n🎯 Connection test passed successfully.")

    except pyodbc.Error as e:
        print("\n❌ Connection failed:")
        print(str(e))
        print("\n💡 Check if the Docker container is running and ports are correct.")
    except Exception as e:
        print("\n❌ Unexpected error:")
        print(str(e))

if __name__ == "__main__":
    test_connection()
