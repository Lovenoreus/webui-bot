import pymssql

def test_sql_connection():
    """
    Simple test to connect to SQL Server and print one row
    """
    try:
        # Connect to SQL Server instance vs837\UIPORCH with hardcoded credentials
        print("Testing connection to vs837\\UIPORCH...")
        
        conn = pymssql.connect(
            server='10.19.50.53\\UIPORCH',
            database='Nodinite',
            user='vll\lono05',                    # Hardcoded username
            password='x'       # Hardcoded password - CHANGE THIS
        )
        
        print("✓ Connected successfully!")
        
        # Execute simple query to get one row
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice_Line]")
        
        # Get column names
        columns = [column[0] for column in cursor.description]
        
        # Get one row
        row = cursor.fetchone()
        
        if row:
            print("\n--- Sample Row from Invoice_Line ---")
            for col, val in zip(columns, row):
                print(f"{col}: {val}")
        else:
            print("No data found in table")
            
        cursor.close()
        conn.close()
        print("\n✓ Connection test successful!")
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")

if __name__ == "__main__":
    test_sql_connection()