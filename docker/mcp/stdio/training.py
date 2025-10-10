# # -------------------- External Libraries --------------------
# from dotenv import load_dotenv, find_dotenv
#
# # -------------------- User-defined Modules --------------------
# import config


# load_dotenv(find_dotenv())


# Get training
def get_vanna_training(remote=False):
    if remote:
        print(f"Using remote database schema: [Nodinite].[ods]")
        #
        invoice_ddl = """
            ## Database Information
            - **Database Type**: Microsoft SQL Server
            - **Dialect**: T-SQL (Transact-SQL)
            - **Database Name**: Nodinite
            - **Schema**: ods
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[ods].[TableName]
    
            CREATE TABLE [Nodinite].[ods].[Invoice] (
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
            - **Schema**: ods
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[ods].[TableName]
    
            CREATE TABLE [Nodinite].[ods].[Invoice_Line] (
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
                FOREIGN KEY (INVOICE_ID) REFERENCES [Nodinite].[ods].[Invoice](INVOICE_ID)
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

            ## Sample Data (Real Examples from Abbott Scandinavia)
            Invoice 0000470081 issued 2023-05-16:
            - Supplier: Abbott Scandinavia (5560466137), Solna
            - Customer: Region Västerbotten (2321000222)
            - Delivery: Universitetssjukhuset, UMEÅ
            - Currency: SEK
            - Line Extension Amount: 173,333.59 SEK
            - Tax Amount: 43,333.42 SEK
            - Total Payable: 216,667.01 SEK
            - Payment Terms: 30 days
            - Note: 'CPR Statistik April 2023'

            Invoice 0000470109 issued 2023-05-22:
            - Supplier: Abbott Scandinavia (5560466137)
            - Customer: Region Västerbotten (2321000222)
            - Delivery: Godsmottagning, UMEÅ
            - Total Payable: 13,750.00 SEK

            ## Common Suppliers
            - Abbott Scandinavia (5560466137): Medical supplies and equipment
            - Visma Draftit AB: Software and IT services
            - JA Hotel Karlskrona: Hospitality services

            ## Common Customers
            - Region Västerbotten (2321000222)
            - Region Skåne
            - Stockholms Stad
            - Västra Götaland
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

            ## Sample Data (Real Examples from Abbott Scandinavia Invoice 0000470081)

            Line 1:
            - Item: PP ALNTY I HAVAB IGM RGT (medical test)
            - Quantity: 26 EA
            - Unit Price: 66.29 SEK
            - Line Total: 1,723.54 SEK
            - Tax: 25%
            - Seller Item ID: 2R2897

            Line 10:
            - Item: PP ALNTY I ANTI-HBC RGT
            - Quantity: 443 EA
            - Unit Price: 19.51 SEK
            - Line Total: 8,642.93 SEK
            - Tax: 25%
            - Seller Item ID: 7P8797

            Line 11:
            - Item: PP ALNTY I ANTI HBS RGT
            - Quantity: 170 EA
            - Unit Price: 35.10 SEK
            - Line Total: 5,967.00 SEK

            Line 12:
            - Item: PP ALNTY I ANTI-HCV RGT
            - Quantity: 1,726 EA
            - Unit Price: 31.37 SEK
            - Line Total: 54,144.62 SEK

            Line 13:
            - Item: PP ALNTY I HIV COMBO RGT
            - Quantity: 1,928 EA
            - Unit Price: 18.49 SEK
            - Line Total: 35,648.72 SEK

            ## Common Product Types
            - Medical test reagents (PP ALNTY I series)
            - Hotel accommodation services
            - Software licenses
            - Training services
            - Medical supplies

            ## Important Notes
            - Invoice line totals should sum to the invoice header LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
            - Tax is calculated per line: INVOICED_LINE_EXTENSION_AMOUNT × (ITEM_TAXCAT_PERCENT / 100)
            - NULL values in ORDER_LINE_REFERENCE_LINE_ID, ACCOUNTING_COST are common
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
    
            ## Sample Data (Real Examples from Abbott Scandinavia)
            Invoice 0000470081 issued 2023-05-16:
            - Supplier: Abbott Scandinavia (5560466137), Solna
            - Customer: Region Västerbotten (2321000222)
            - Delivery: Universitetssjukhuset, UMEÅ
            - Currency: SEK
            - Line Extension Amount: 173,333.59 SEK
            - Tax Amount: 43,333.42 SEK
            - Total Payable: 216,667.01 SEK
            - Payment Terms: 30 days
            - Note: 'CPR Statistik April 2023'
    
            Invoice 0000470109 issued 2023-05-22:
            - Supplier: Abbott Scandinavia (5560466137)
            - Customer: Region Västerbotten (2321000222)
            - Delivery: Godsmottagning, UMEÅ
            - Total Payable: 13,750.00 SEK
    
            ## Common Suppliers
            - Abbott Scandinavia (5560466137): Medical supplies and equipment
            - Visma Draftit AB: Software and IT services
            - JA Hotel Karlskrona: Hospitality services
    
            ## Common Customers
            - Region Västerbotten (2321000222)
            - Region Skåne
            - Stockholms Stad
            - Västra Götaland
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
    
            ## Sample Data (Real Examples from Abbott Scandinavia Invoice 0000470081)
    
            Line 1:
            - Item: PP ALNTY I HAVAB IGM RGT (medical test)
            - Quantity: 26 EA
            - Unit Price: 66.29 SEK
            - Line Total: 1,723.54 SEK
            - Tax: 25%
            - Seller Item ID: 2R2897
    
            Line 10:
            - Item: PP ALNTY I ANTI-HBC RGT
            - Quantity: 443 EA
            - Unit Price: 19.51 SEK
            - Line Total: 8,642.93 SEK
            - Tax: 25%
            - Seller Item ID: 7P8797
    
            Line 11:
            - Item: PP ALNTY I ANTI HBS RGT
            - Quantity: 170 EA
            - Unit Price: 35.10 SEK
            - Line Total: 5,967.00 SEK
    
            Line 12:
            - Item: PP ALNTY I ANTI-HCV RGT
            - Quantity: 1,726 EA
            - Unit Price: 31.37 SEK
            - Line Total: 54,144.62 SEK
    
            Line 13:
            - Item: PP ALNTY I HIV COMBO RGT
            - Quantity: 1,928 EA
            - Unit Price: 18.49 SEK
            - Line Total: 35,648.72 SEK
    
            ## Common Product Types
            - Medical test reagents (PP ALNTY I series)
            - Hotel accommodation services
            - Software licenses
            - Training services
            - Medical supplies
    
            ## Important Notes
            - Invoice line totals should sum to the invoice header LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
            - Tax is calculated per line: INVOICED_LINE_EXTENSION_AMOUNT × (ITEM_TAXCAT_PERCENT / 100)
            - NULL values in ORDER_LINE_REFERENCE_LINE_ID, ACCOUNTING_COST are common
            """

    return [invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc]
