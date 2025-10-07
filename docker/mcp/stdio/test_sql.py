import pyodbc

server = 'localhost,1433'
database = 'SampleDB'
username = 'sa'
password = 'YourPassword123!'
driver = '{ODBC Driver 17 for SQL Server}'

try:
    conn = pyodbc.connect(
        f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )
    print("✅ Connected to SQL Server successfully!")

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Employees;")

    for row in cursor.fetchall():
        print(row)

except Exception as e:
    print("❌ Error:", e)
