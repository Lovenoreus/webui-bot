# -------------------- External Libraries --------------------
# from dotenv import load_dotenv, find_dotenv
#
# # -------------------- User-defined Modules --------------------
# import config

# load_dotenv(find_dotenv())


def get_vanna_question_sql_pairs(remote=False):
    """
    Get just the question-SQL training pairs for Vanna
    
    Args:
        remote (bool): If True, returns SQL Server specific pairs, otherwise SQLite pairs
    
    Returns:
        list: List of dictionaries with 'question' and 'sql' keys
    """
    if remote:
        # SQL Server specific training pairs
        return [
            {
                "question": "How much did we pay for ISTAT CREATINI CARTRIDGE?",
                "sql": """
                    SELECT 
                        SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%ISTAT CREATINI CARTRIDGE%')
                """
            },
            {
                "question": "What companies sell the product ALINITY M BOTTLE ETHANOL U?",
                "sql": """
                    SELECT DISTINCT 
                        i.SUPPLIER_PARTY_NAME
                    FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
                    INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                        ON i.INVOICE_ID = il.INVOICE_ID
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%ALINITY M BOTTLE ETHANOL U%')
                """
            },
            {
                "question": "What products were delivered to thoraxradiologi?",
                "sql": """
                    SELECT DISTINCT 
                        il.ITEM_NAME
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
                        ON il.INVOICE_ID = i.INVOICE_ID
                    WHERE LOWER(i.DELIVERY_PARTY_NAME) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_ADDRESS_LINE) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%thoraxradiologi%')
                """
            },

            {
                "question": "How many invoices do we have for buying pipettes?",
                "sql": """
                    SELECT 
                        COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%pipette%')
                """
            },

            {
            "question": "How many invoices do we have for screws?",
            "sql": """
                SELECT 
                    COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
                FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%screw%')
            """
            },



            {
                "question": "How many invoices are there in total?",
                "sql": "SELECT COUNT(*) AS total_invoices FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]"
            },
            {
                "question": "Show me all invoices from Abbott Scandinavia",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott Scandinavia%')"
            },
            {
                "question": "What is the total amount of all invoices?",
                "sql": "SELECT SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]"
            },
            {
                "question": "List all invoices for Region Västerbotten",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(CUSTOMER_PARTY_NAME) LIKE LOWER('%Region Västerbotten%')"
            },
            {
                "question": "Show me invoices issued in 2023",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE ISSUE_DATE LIKE '2023%'"
            },
            {
                "question": "What are the top 10 suppliers by invoice count?",
                "sql": "SELECT TOP 10 SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME ORDER BY invoice_count DESC"
            },
            {
                "question": "Show me invoices with tax amount greater than 40000",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE TAX_AMOUNT > 40000"
            },
            {
                "question": "List all invoice line items for invoice 0000470081",
                "sql": "SELECT INVOICE_LINE_ID, ITEM_NAME, INVOICED_QUANTITY, PRICE_AMOUNT, INVOICED_LINE_EXTENSION_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE INVOICE_ID = '0000470081'"
            },
            {
                "question": "Show me the total quantity and amount for all PP ALNTY items",
                "sql": "SELECT SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%')"
            },
            {
                "question": "Which customers have the highest total invoice amounts?",
                "sql": "SELECT TOP 5 CUSTOMER_PARTY_NAME, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY CUSTOMER_PARTY_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me invoices due in the next 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE DUE_DATE >= CAST(GETDATE() AS DATE) AND DUE_DATE <= DATEADD(DAY, 30, CAST(GETDATE() AS DATE))"
            },
            {
                "question": "What is the average invoice amount by supplier?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount, COUNT(*) AS invoice_count FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 1 ORDER BY avg_amount DESC"
            },
            {
                "question": "Show me medical reagent items and their total sales",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_sales FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%RGT%') GROUP BY ITEM_NAME ORDER BY total_sales DESC"
            },
            {
                "question": "List invoices with delivery to Umeå",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DELIVERY_LOCATION_CITY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%UMEÅ%')"
            },
            {
                "question": "Show me invoices in SEK currency only",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, DOCUMENT_CURRENCY_CODE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE DOCUMENT_CURRENCY_CODE = 'SEK'"
            },
            {
                "question": "Find all invoices with payment terms of 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, PAYMENT_TERMS_NOTE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(PAYMENT_TERMS_NOTE) LIKE LOWER('%30%')"
            },
            {
                "question": "Show me items with unit code 'EA' (each)",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, AVG(PRICE_AMOUNT) AS avg_price FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE INVOICED_QUANTITY_UNIT_CODE = 'EA' GROUP BY ITEM_NAME ORDER BY total_quantity DESC"
            },
            {
                "question": "List invoices with tax rate of 25%",
                "sql": "SELECT DISTINCT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il ON i.INVOICE_ID = il.INVOICE_ID WHERE il.ITEM_TAXCAT_PERCENT = 25.0"
            },
            {
                "question": "Show me the monthly invoice totals for 2023",
                "sql": "SELECT SUBSTRING(ISSUE_DATE, 1, 7) AS month, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE ISSUE_DATE LIKE '2023%' GROUP BY SUBSTRING(ISSUE_DATE, 1, 7) ORDER BY month"
            },
            {
                "question": "Find invoices where tax amount is more than 20% of the total",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT, (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 100) AS tax_percentage FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE TAX_AMOUNT > (LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 0.2)"
            },
            {
                "question": "Show me suppliers with more than 5 invoices",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 5 ORDER BY total_spent DESC"
            },
            {
                "question": "List all unique item names containing 'test' or 'kit'",
                "sql": "SELECT DISTINCT ITEM_NAME, COUNT(*) AS frequency FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%test%') OR LOWER(ITEM_NAME) LIKE LOWER('%kit%') GROUP BY ITEM_NAME ORDER BY frequency DESC"
            }
        ]
    else:
        # SQLite specific training pairs
        return [
            {
                "question": "How many invoices are there in total?",
                "sql": "SELECT COUNT(*) AS total_invoices FROM Invoice"
            },
            {
                "question": "Show me all invoices from Abbott Scandinavia",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott Scandinavia%')"
            },
            {
                "question": "What is the total amount of all invoices?",
                "sql": "SELECT SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice"
            },
            {
                "question": "List all invoices for Region Västerbotten",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(CUSTOMER_PARTY_NAME) LIKE LOWER('%Region Västerbotten%')"
            },
            {
                "question": "Show me invoices issued in 2023",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE ISSUE_DATE LIKE '2023%'"
            },
            {
                "question": "What are the top 10 suppliers by invoice count?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count FROM Invoice GROUP BY SUPPLIER_PARTY_NAME ORDER BY invoice_count DESC LIMIT 10"
            },
            {
                "question": "Show me invoices with tax amount greater than 40000",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE TAX_AMOUNT > 40000"
            },
            {
                "question": "List all invoice line items for invoice 0000470081",
                "sql": "SELECT INVOICE_LINE_ID, ITEM_NAME, INVOICED_QUANTITY, PRICE_AMOUNT, INVOICED_LINE_EXTENSION_AMOUNT FROM Invoice_Line WHERE INVOICE_ID = '0000470081'"
            },
            {
                "question": "Show me the total quantity and amount for all PP ALNTY items",
                "sql": "SELECT SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%')"
            },
            {
                "question": "Which customers have the highest total invoice amounts?",
                "sql": "SELECT CUSTOMER_PARTY_NAME, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice GROUP BY CUSTOMER_PARTY_NAME ORDER BY total_amount DESC LIMIT 5"
            },
            {
                "question": "Show me invoices due in the next 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE DUE_DATE >= date('now') AND DUE_DATE <= date('now', '+30 days')"
            },
            {
                "question": "What is the average invoice amount by supplier?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount, COUNT(*) AS invoice_count FROM Invoice GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 1 ORDER BY avg_amount DESC"
            },
            {
                "question": "Show me medical reagent items and their total sales",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_sales FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%RGT%') GROUP BY ITEM_NAME ORDER BY total_sales DESC"
            },
            {
                "question": "List invoices with delivery to Umeå",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DELIVERY_LOCATION_CITY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%UMEÅ%')"
            },
            {
                "question": "Show me invoices in SEK currency only",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, DOCUMENT_CURRENCY_CODE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE DOCUMENT_CURRENCY_CODE = 'SEK'"
            },
            {
                "question": "Find all invoices with payment terms of 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, PAYMENT_TERMS_NOTE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(PAYMENT_TERMS_NOTE) LIKE LOWER('%30%')"
            },
            {
                "question": "Show me items with unit code 'EA' (each)",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, AVG(PRICE_AMOUNT) AS avg_price FROM Invoice_Line WHERE INVOICED_QUANTITY_UNIT_CODE = 'EA' GROUP BY ITEM_NAME ORDER BY total_quantity DESC"
            },
            {
                "question": "List invoices with tax rate of 25%",
                "sql": "SELECT DISTINCT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice i INNER JOIN Invoice_Line il ON i.INVOICE_ID = il.INVOICE_ID WHERE il.ITEM_TAXCAT_PERCENT = 25.0"
            },
            {
                "question": "Show me the monthly invoice totals for 2023",
                "sql": "SELECT substr(ISSUE_DATE, 1, 7) AS month, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice WHERE ISSUE_DATE LIKE '2023%' GROUP BY substr(ISSUE_DATE, 1, 7) ORDER BY month"
            },
            {
                "question": "Find invoices where tax amount is more than 20% of the total",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT, (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 100) AS tax_percentage FROM Invoice WHERE TAX_AMOUNT > (LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 0.2)"
            },
            {
                "question": "Show me suppliers with more than 5 invoices",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent FROM Invoice GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 5 ORDER BY total_spent DESC"
            },
            {
                "question": "List all unique item names containing 'test' or 'kit'",
                "sql": "SELECT DISTINCT ITEM_NAME, COUNT(*) AS frequency FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%test%') OR LOWER(ITEM_NAME) LIKE LOWER('%kit%') GROUP BY ITEM_NAME ORDER BY frequency DESC"
            }
        ]


def get_vanna_training(remote=False):
    """
    Get complete Vanna training data including DDL, documentation, and question-SQL pairs
    
    Args:
        remote (bool): If True, returns SQL Server specific data, otherwise SQLite data
    
    Returns:
        list: [invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, training_pairs]
    """
    if remote:
        print(f"Using remote database schema: [Nodinite].[dbo]")
        
        invoice_ddl = """
            ## Database Information
            - **Database Type**: Microsoft SQL Server
            - **Dialect**: T-SQL (Transact-SQL)
            - **Database Name**: Nodinite
            - **Schema**: dbo
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[dbo].[TableName]

            Always use column aliases for aggregate functions and expressions. 
            Example: SELECT COUNT(*) AS count, SUM(amount) AS total_amount
    
            CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] (
                INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
                ISSUE_DATE NVARCHAR(10) NOT NULL,
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
                SUPPLIER_PARTY_NAME NVARCHAR(255),
                SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),
                SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
                SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),
                SUPPLIER_PARTY_CITY NVARCHAR(100),
                SUPPLIER_PARTY_COUNTRY NVARCHAR(2),
                SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
                SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100),
                SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
                SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
                SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
                SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
                CUSTOMER_PARTY_ID NVARCHAR(50),
                CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),
                CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
                CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
                CUSTOMER_PARTY_NAME NVARCHAR(255),
                CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),
                CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),
                CUSTOMER_PARTY_COUNTRY NVARCHAR(2),
                CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
                CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),
                CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
                CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
                CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
                DUE_DATE NVARCHAR(10),
                DOCUMENT_CURRENCY_CODE NVARCHAR(3),
                DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),
                DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
                DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),
                DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),
                DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
                DELIVERY_LOCATION_COUNTRY NVARCHAR(2),
                DELIVERY_PARTY_NAME NVARCHAR(255),
                ACTUAL_DELIVERY_DATE NVARCHAR(10),
                TAX_AMOUNT_CURRENCY NVARCHAR(3),
                TAX_AMOUNT DECIMAL(18,2),
                PERIOD_START_DATE NVARCHAR(10),
                PERIOD_END_DATE NVARCHAR(10),
                LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,2),
                BUYER_REFERENCE NVARCHAR(100),
                PROJECT_REFERENCE_ID NVARCHAR(100),
                INVOICE_TYPE_CODE NVARCHAR(10),
                NOTE NVARCHAR(MAX),
                TAX_POINT_DATE NVARCHAR(10),
                ACCOUNTING_COST NVARCHAR(100),
                ORDER_REFERENCE_ID NVARCHAR(100),
                ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
                PAYMENT_TERMS_NOTE NVARCHAR(MAX),
                BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
                BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(10),
                CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                ETL_LOAD_TS NVARCHAR(30)
            );
            """

        invoice_line_ddl = """
            ## Database Information
            - **Database Type**: Microsoft SQL Server
            - **Dialect**: T-SQL (Transact-SQL)
            - **Database Name**: Nodinite
            - **Schema**: dbo
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[dbo].[TableName]

            Always use column aliases for aggregate functions and expressions. 
            Example: SELECT COUNT(*) AS count, SUM(amount) AS total_amount
    
            CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] (
                INVOICE_ID NVARCHAR(50) NOT NULL,
                ISSUE_DATE NVARCHAR(10) NOT NULL,
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
                INVOICE_LINE_ID NVARCHAR(50) NOT NULL,
                ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(50),
                ACCOUNTING_COST NVARCHAR(100),
                INVOICED_QUANTITY DECIMAL(18,4),
                INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),
                INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,2),
                INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3),
                INVOICE_PERIOD_START_DATE NVARCHAR(10),
                INVOICE_PERIOD_END_DATE NVARCHAR(10),
                INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(10),
                INVOICE_LINE_NOTE NVARCHAR(MAX),
                ITEM_DESCRIPTION NVARCHAR(MAX),
                ITEM_NAME NVARCHAR(255),
                ITEM_TAXCAT_ID NVARCHAR(10),
                ITEM_TAXCAT_PERCENT DECIMAL(5,2),
                ITEM_BUYERS_ID NVARCHAR(100),
                ITEM_SELLERS_ITEM_ID NVARCHAR(100),
                ITEM_STANDARD_ITEM_ID NVARCHAR(100),
                ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),
                ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(50),
                PRICE_AMOUNT DECIMAL(18,2),
                PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),
                PRICE_BASE_QUANTITY DECIMAL(18,4),
                PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),
                PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,2),
                PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),
                ETL_LOAD_TS NVARCHAR(30),
                PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
                FOREIGN KEY (INVOICE_ID) REFERENCES [Nodinite].[dbo].[Invoice](INVOICE_ID)
            );
            """

        invoice_doc = """
            # Invoice Table Documentation

            ## Business Context
            The Invoice table stores header-level information for invoices received from suppliers.
            Each invoice has a unique INVOICE_ID and contains supplier, customer, delivery, and financial information.

            ## Key Fields Explanation
            - INVOICE_ID: Unique identifier for each invoice (e.g., '0000470081')
            - ISSUE_DATE: Date the invoice was issued (format: YYYY-MM-DD)
            - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the supplier (e.g., '5560466137' for Abbott Scandinavia)
            - SUPPLIER_PARTY_NAME: Name of the supplier company
            - CUSTOMER_PARTY_NAME: Name of the customer (often Swedish regions like 'Region Västerbotten')
            - CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the customer
            - DOCUMENT_CURRENCY_CODE: Currency used (typically 'SEK' for Swedish Krona)
            - TAX_AMOUNT: Total VAT/tax amount on the invoice
            - LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT: Total amount excluding tax
            - LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT: Total excluding tax (can differ from line extension due to charges/allowances)
            - LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT: Total including tax
            - LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT: Final amount to be paid
            - BUYER_REFERENCE: Customer's internal reference/cost center
            - ORDER_REFERENCE_ID: Reference to purchase order
            - INVOICE_TYPE_CODE: Type of invoice (e.g., '380' = standard commercial invoice)
            - PAYMENT_TERMS_NOTE: Payment terms description (e.g., '30 | Dröjsmålsränta %' means 30 days payment terms with late payment interest)
            - DELIVERY_LOCATION_*: Fields describing where goods/services were delivered
            
            ## Important Notes
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

        invoice_line_doc = """
            # Invoice_Line Table Documentation

            ## Business Context
            The Invoice_Line table stores individual line items for each invoice.
            Each line represents a specific product or service being invoiced.
            One invoice (INVOICE_ID) can have multiple lines (INVOICE_LINE_ID).

            ## Key Fields Explanation
            - INVOICE_ID: Links to the parent Invoice table
            - INVOICE_LINE_ID: Unique line number within the invoice (e.g., '1', '2', '10', '11')
            - ITEM_NAME: Description of the product/service (e.g., 'PP ALNTY I HAVAB IGM RGT' = medical test reagent)
            - INVOICED_QUANTITY: Quantity of items (e.g., 26.000)
            - INVOICED_QUANTITY_UNIT_CODE: Unit of measurement (e.g., 'EA' = Each/piece)
            - PRICE_AMOUNT: Unit price per item
            - INVOICED_LINE_EXTENSION_AMOUNT: Total line amount (quantity × price, before tax)
            - ITEM_TAXCAT_ID: Tax category identifier (typically 'S' for standard VAT)
            - ITEM_TAXCAT_PERCENT: VAT percentage (typically 25.000 for Sweden)
            - ITEM_SELLERS_ITEM_ID: Supplier's product code (e.g., '2R2897', '7P8797')
            - INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID: Currency (typically 'SEK')

            ## Important Notes
            - Invoice line totals should sum to the invoice header LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
            - Tax is calculated per line: INVOICED_LINE_EXTENSION_AMOUNT × (ITEM_TAXCAT_PERCENT / 100)
            - NULL values in ORDER_LINE_REFERENCE_LINE_ID, ACCOUNTING_COST are common
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """
    else:
        print("Using local database schema")

        invoice_ddl = """
        ## Database Information
        - **Database Type**: SQLite
        - **Dialect**: You must generate SQLite-compatible SQL syntax

        CREATE TABLE Invoice (
            INVOICE_ID TEXT NOT NULL PRIMARY KEY,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            SUPPLIER_PARTY_NAME TEXT,
            SUPPLIER_PARTY_STREET_NAME TEXT,
            SUPPLIER_PARTY_ADDITIONAL_STREET_NAME TEXT,
            SUPPLIER_PARTY_POSTAL_ZONE TEXT,
            SUPPLIER_PARTY_CITY TEXT,
            SUPPLIER_PARTY_COUNTRY TEXT,
            SUPPLIER_PARTY_ADDRESS_LINE TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM TEXT,
            SUPPLIER_PARTY_CONTACT_NAME TEXT,
            SUPPLIER_PARTY_CONTACT_EMAIL TEXT,
            SUPPLIER_PARTY_CONTACT_PHONE TEXT,
            SUPPLIER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ID TEXT,
            CUSTOMER_PARTY_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_NAME TEXT,
            CUSTOMER_PARTY_STREET_NAME TEXT,
            CUSTOMER_PARTY_POSTAL_ZONE TEXT,
            CUSTOMER_PARTY_COUNTRY TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT,
            CUSTOMER_PARTY_CONTACT_NAME TEXT,
            CUSTOMER_PARTY_CONTACT_EMAIL TEXT,
            CUSTOMER_PARTY_CONTACT_PHONE TEXT,
            DUE_DATE TEXT,
            DOCUMENT_CURRENCY_CODE TEXT,
            DELIVERY_LOCATION_STREET_NAME TEXT,
            DELIVERY_LOCATION_ADDITIONAL_STREET_NAME TEXT,
            DELIVERY_LOCATION_CITY_NAME TEXT,
            DELIVERY_LOCATION_POSTAL_ZONE TEXT,
            DELIVERY_LOCATION_ADDRESS_LINE TEXT,
            DELIVERY_LOCATION_COUNTRY TEXT,
            DELIVERY_PARTY_NAME TEXT,
            ACTUAL_DELIVERY_DATE TEXT,
            TAX_AMOUNT_CURRENCY TEXT,
            TAX_AMOUNT REAL,
            PERIOD_START_DATE TEXT,
            PERIOD_END_DATE TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT REAL,
            BUYER_REFERENCE TEXT,
            PROJECT_REFERENCE_ID TEXT,
            INVOICE_TYPE_CODE TEXT,
            NOTE TEXT,
            TAX_POINT_DATE TEXT,
            ACCOUNTING_COST TEXT,
            ORDER_REFERENCE_ID TEXT,
            ORDER_REFERENCE_SALES_ORDER_ID TEXT,
            PAYMENT_TERMS_NOTE TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE TEXT,
            CONTRACT_DOCUMENT_REFERENCE_ID TEXT,
            DESPATCH_DOCUMENT_REFERENCE_ID TEXT,
            ETL_LOAD_TS TEXT
        );
        """

        invoice_line_ddl = """
        ## Database Information
        - **Database Type**: SQLite
        - **Dialect**: You must generate SQLite-compatible SQL syntax

        CREATE TABLE Invoice_Line (
            INVOICE_ID TEXT NOT NULL,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            INVOICE_LINE_ID TEXT NOT NULL,
            ORDER_LINE_REFERENCE_LINE_ID TEXT,
            ACCOUNTING_COST TEXT,
            INVOICED_QUANTITY REAL,
            INVOICED_QUANTITY_UNIT_CODE TEXT,
            INVOICED_LINE_EXTENSION_AMOUNT REAL,
            INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID TEXT,
            INVOICE_PERIOD_START_DATE TEXT,
            INVOICE_PERIOD_END_DATE TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_ID TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE TEXT,
            INVOICE_LINE_NOTE TEXT,
            ITEM_DESCRIPTION TEXT,
            ITEM_NAME TEXT,
            ITEM_TAXCAT_ID TEXT,
            ITEM_TAXCAT_PERCENT REAL,
            ITEM_BUYERS_ID TEXT,
            ITEM_SELLERS_ITEM_ID TEXT,
            ITEM_STANDARD_ITEM_ID TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID TEXT,
            PRICE_AMOUNT REAL,
            PRICE_AMOUNT_CURRENCY_ID TEXT,
            PRICE_BASE_QUANTITY REAL,
            PRICE_BASE_QUANTITY_UNIT_CODE TEXT,
            PRICE_ALLOWANCE_CHARGE_AMOUNT REAL,
            PRICE_ALLOWANCE_CHARGE_INDICATOR TEXT,
            ETL_LOAD_TS TEXT,
            PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
            FOREIGN KEY (INVOICE_ID) REFERENCES Invoice(INVOICE_ID)
        );
        """

        invoice_doc = """
            # Invoice Table Documentation (SQLite Version)

            ## Business Context
            The Invoice table stores header-level information for invoices received from suppliers.
            Each invoice has a unique INVOICE_ID and contains supplier, customer, delivery, and financial information.

            ## Key Fields Explanation
            - Same as SQL Server version, but using SQLite data types (TEXT, REAL)
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

        invoice_line_doc = """
            # Invoice_Line Table Documentation (SQLite Version)

            ## Business Context
            The Invoice_Line table stores individual line items for each invoice.
            Each line represents a specific product or service being invoiced.
            One invoice (INVOICE_ID) can have multiple lines (INVOICE_LINE_ID).

            ## Key Fields Explanation
            - Same as SQL Server version, but using SQLite data types (TEXT, REAL)
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

    # Get the question-SQL training pairs
    training_pairs = get_vanna_question_sql_pairs(remote)

    return [invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, training_pairs]