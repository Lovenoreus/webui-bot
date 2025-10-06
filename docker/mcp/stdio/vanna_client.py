from vanna.local import LocalContext_OpenAI
import os
import logging
import vanna
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())


if hasattr(vanna, "telemetry"):
    vanna.telemetry.capture = lambda *a, **kw: None
    
logging.getLogger("vanna.telemetry").setLevel(logging.CRITICAL)


vn = LocalContext_OpenAI({"api_key": os.getenv("OPENAI_API_KEY")})

vn.train(sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME AS supplier_name,
    i.SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID AS supplier_id,
    i.CUSTOMER_PARTY_NAME AS customer_name,
    SUM(CAST(il.INVOICED_LINE_EXTENSION_AMOUNT AS FLOAT)) AS total_sales,
    COUNT(il.INVOICE_LINE_ID) AS num_items
FROM Invoice_Line il
JOIN Invoice i 
    ON il.INVOICE_ID = i.INVOICE_ID
WHERE il.INVOICED_LINE_EXTENSION_AMOUNT IS NOT NULL
GROUP BY supplier_name, supplier_id, customer_name
ORDER BY total_sales DESC
LIMIT 10;
""")

# Teach Vanna your schema and an example query with date filtering
vn.train(sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME AS supplier_name,
    SUM(CAST(il.INVOICED_LINE_EXTENSION_AMOUNT AS FLOAT)) AS total_sales
FROM Invoice_Line il
JOIN Invoice i 
    ON il.INVOICE_ID = i.INVOICE_ID
WHERE strftime('%Y', i.ISSUE_DATE) = '2024'
GROUP BY supplier_name
ORDER BY total_sales DESC
LIMIT 10;
""")

# Determine the full path to compacted.db (same folder as this script)
db_path = os.path.join(os.path.dirname(__file__), "compacted.db")

# Connect to your local SQLite database
vn.connect_to_sqlite(db_path)

# Ask a natural language question
vn.ask("How many invoices have the contact person ANJO119?")




