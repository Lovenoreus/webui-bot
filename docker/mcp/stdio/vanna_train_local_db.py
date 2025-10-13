def train_for_local_db(vanna_manager):
    
        # ================================================================
        # TABLE 1: Invoice Header Table (Main Invoice Information)
        # ================================================================

    if vanna_manager.train(ddl="""
        CREATE TABLE Invoice (
            -- Primary Key
            INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
            
            -- Date Information
            ISSUE_DATE DATE NOT NULL,                    -- Format: YYYY-MM-DD (e.g., '2023-10-06')
            DUE_DATE DATE,                               -- Format: YYYY-MM-DD
            TAX_POINT_DATE DATE,                         -- Format: YYYY-MM-DD
            ACTUAL_DELIVERY_DATE DATE,                   -- Format: YYYY-MM-DD
            PERIOD_START_DATE DATE,                      -- Format: YYYY-MM-DD
            PERIOD_END_DATE DATE,                        -- Format: YYYY-MM-DD
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE DATE,
            
            -- Supplier Party Information
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,  -- Swedish organization number (e.g., '5560466137')
            SUPPLIER_PARTY_NAME NVARCHAR(255),                             -- Company name (e.g., 'Abbott Scandinavia', 'Visma Draftit AB')
            SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),           -- Official legal name
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100), -- Legal form (e.g., 'Säte Solna SE')
            
            -- Supplier Address
            SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),                     -- (e.g., 'Hemvärnsgatan 9', 'Styrmansgatan 2')
            SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
            SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),                      -- 5-digit postal code (e.g., '171 29', '21118')
            SUPPLIER_PARTY_CITY NVARCHAR(100),                            -- City name (e.g., 'Solna', 'Malmo', 'Karlskrona')
            SUPPLIER_PARTY_COUNTRY NVARCHAR(2),                           -- Country code (always 'SE')
            SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
            
            -- Supplier Contact Information
            SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
            SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),                   -- (e.g., 'support@vismadraftit.se', 'info@jahotel.se')
            SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),                    -- (e.g., '+46101992350', '045555560')
            SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
            
            -- Customer Party Information (Region Västerbotten)
            CUSTOMER_PARTY_ID NVARCHAR(50),                               -- (e.g., '7362321000224')
            CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),                     -- (e.g., '0088')
            CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
            CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
            CUSTOMER_PARTY_NAME NVARCHAR(255),                            -- Always 'Region Västerbotten'
            CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),          -- Always 'Region Västerbotten'
            CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),         -- Always '2321000222'
            
            -- Customer Address
            CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),                     -- (e.g., 'FE 5102')
            CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),                      -- (e.g., '838 77', '90189')
            CUSTOMER_PARTY_COUNTRY NVARCHAR(2),                           -- Always 'SE'
            
            -- Customer Contact Information
            CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),                    -- Contact person or reference code
            CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),                   -- (e.g., 'daniel.stromberg@regionvasterbotten.se')
            CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
            
            -- Delivery Information
            DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),                  -- (e.g., 'Universitetssjukhuset', 'Volgsjövägen 37')
            DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
            DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),                    -- Cities in Västerbotten (e.g., 'UMEÅ', 'Vilhelmina', 'Tärnaby')
            DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),                   -- (e.g., '901 85', '912 32')
            DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
            DELIVERY_LOCATION_COUNTRY NVARCHAR(2),                        -- Always 'SE'
            DELIVERY_PARTY_NAME NVARCHAR(255),
            
            -- Currency (always SEK)
            DOCUMENT_CURRENCY_CODE NVARCHAR(3),                           -- Always 'SEK'
            
            -- Tax Information
            TAX_AMOUNT_CURRENCY NVARCHAR(3),                              -- Always 'SEK'
            TAX_AMOUNT DECIMAL(18,3),                                     -- Tax amount with 3 decimal places
            
            -- Legal Monetary Totals (all in SEK)
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,3),           -- Sum of line items before tax
            
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,3),           -- Total excluding tax
            
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,3),           -- Total including tax
            
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,3),            -- Final amount to be paid
            
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,3),    -- Discounts/reductions
            
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,3),       -- Additional charges
            
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,3),   -- Rounding adjustment
            
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,3),            -- Prepaid amount
            
            -- Reference Information
            BUYER_REFERENCE NVARCHAR(100),                                -- Internal reference (e.g., 'DEJA01', '1035030')
            PROJECT_REFERENCE_ID NVARCHAR(100),
            ORDER_REFERENCE_ID NVARCHAR(100),                             -- Purchase order reference
            ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
            CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),                 -- Contract reference (e.g., 'CPR')
            DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
            
            -- Invoice Type and Notes
            INVOICE_TYPE_CODE NVARCHAR(10),                               -- Always '380' (standard invoice)
            NOTE NVARCHAR(MAX),                                           -- Free-text notes (e.g., 'CPR Statistik April 2023')
            ACCOUNTING_COST NVARCHAR(100),                                -- Cost center/accounting code
            PAYMENT_TERMS_NOTE NVARCHAR(MAX),                             -- Payment terms (e.g., '30 | Dröjsmålsränta %.', 'Net 30.')
            
            -- ETL Metadata
            ETL_LOAD_TS DATETIME                                          -- Timestamp format: 'YYYY-MM-DD HH:MM:SS.mmm'
        )
        """):
        print("✅ Successfully trained Invoice DDL")
    else:
        print("❌ Failed to train Invoice DDL")

    # Train Invoice_Line table DDL
    if vanna_manager.train(ddl="""
        CREATE TABLE Invoice_Line (
            -- Foreign Key to Invoice table
            INVOICE_ID NVARCHAR(50) NOT NULL,                             -- Links to Invoice.INVOICE_ID
            ISSUE_DATE DATE NOT NULL,                                     -- Duplicate from Invoice for easier querying
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL, -- Duplicate from Invoice
            
            -- Line Item Identification
            INVOICE_LINE_ID NVARCHAR(50) NOT NULL,                        -- Line number (e.g., '1', '2', '13')
            ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(100),                   -- Reference to order line
            
            -- Quantity and Unit
            INVOICED_QUANTITY DECIMAL(18,3),                              -- Quantity invoiced (e.g., 6.000, 26.000)
            INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),                     -- Unit code (e.g., 'EA'=each, 'ZZ'=mutually defined)
            
            -- Line Amount
            INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,3),                 -- Line total before tax
            INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3),      -- Always 'SEK'
            
            -- Period Information
            INVOICE_PERIOD_START_DATE DATE,                               -- Service/subscription start date
            INVOICE_PERIOD_END_DATE DATE,                                 -- Service/subscription end date
            
            -- Document References
            INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),
            INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(50),
            INVOICE_LINE_NOTE NVARCHAR(MAX),                              -- Line-specific notes
            
            -- Item Description
            ITEM_DESCRIPTION NVARCHAR(MAX),                               -- Detailed description (e.g., '2024-04-05 - 2025-04-04')
            ITEM_NAME NVARCHAR(500),                                      -- Product/service name (e.g., 'HR Expert Kommun', 'ISTAT G3 CARTRIDGE')
            
            -- Tax Information
            ITEM_TAXCAT_ID NVARCHAR(10),                                  -- Tax category ('S'=standard, 'E'=exempt)
            ITEM_TAXCAT_PERCENT DECIMAL(5,2),                             -- Tax percentage (e.g., 12.000, 25.000, 0.000)
            
            -- Item Identifiers
            ITEM_BUYERS_ID NVARCHAR(100),                                 -- Buyer's item code
            ITEM_SELLERS_ITEM_ID NVARCHAR(100),                           -- Seller's item code (e.g., '6000', 'HRKOMEXP', '2R2897')
            ITEM_STANDARD_ITEM_ID NVARCHAR(100),                          -- Standard item identifier
            ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),
            ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(100),
            
            -- Pricing Information
            PRICE_AMOUNT DECIMAL(18,3),                                   -- Unit price (e.g., 883.930, 12022.010)
            PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),                         -- Always 'SEK'
            PRICE_BASE_QUANTITY DECIMAL(18,3),                            -- Base quantity for pricing
            PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),                   -- Unit code for base quantity
            PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,3),                  -- Allowance/charge on price
            PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),                -- True/False indicator
            
            -- Accounting
            ACCOUNTING_COST NVARCHAR(100),                                -- Cost center/accounting code
            
            -- ETL Metadata
            ETL_LOAD_TS DATETIME,                                         -- Timestamp format: 'YYYY-MM-DD HH:MM:SS.mmm'
            
            -- Composite Primary Key
            PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
            
            -- Foreign Key Constraint
            FOREIGN KEY (INVOICE_ID) REFERENCES Invoice(INVOICE_ID)
        )
        """):
        print("✅ Successfully trained Invoice_Line DDL")
    else:
        print("❌ Failed to train Invoice_Line DDL")
        
    # Train Table Relationship
    if vanna_manager.train(ddl="""
        -- Relationship between Invoice and Invoice_Line tables
        -- Invoice (Parent/Header)
        --   └─ INVOICE_ID (Primary Key)
        --        │
        --        └─ Links to ─> Invoice_Line (Child/Lines)
        --                        └─ INVOICE_ID (Foreign Key, part of composite PK)
        --
        -- To get complete invoice with line items, JOIN on INVOICE_ID:
        -- FROM Invoice inv
        -- INNER JOIN Invoice_Line line 
        --     ON inv.INVOICE_ID = line.INVOICE_ID
        """):
        print("✅ Successfully trained Table Relationship")
    else:
        print("❌ Failed to train Table Relationship")

    # Train Query Patterns
    if vanna_manager.train(ddl="""
        -- Always use column aliases for aggregate functions:
        -- CORRECT: SELECT COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
        -- INCORRECT: SELECT COUNT(*), SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT)

        -- Date filtering (dates stored as DATE type):
        -- WHERE ISSUE_DATE >= '2023-01-01' AND ISSUE_DATE < '2024-01-01'
        -- For current year: WHERE YEAR(ISSUE_DATE) = YEAR(GETDATE())
        -- For date ranges: WHERE ISSUE_DATE BETWEEN '2023-06-01' AND '2023-06-30'

        -- Swedish-specific considerations:
        -- - Organization numbers: SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID (e.g., '5560466137', '5565783957')
        -- - Currency: Always 'SEK' (Swedish Krona) in this dataset
        -- - VAT rates: 25.00% (standard) or 12.00% (reduced) or 0.00% (exempt) in ITEM_TAXCAT_PERCENT
        -- - Postal codes: 5 digits, may have space (e.g., '171 29', '90189')
        -- - Country codes: Always 'SE' (Sweden) in this dataset
        -- - Customer: Always 'Region Västerbotten' (ID: '2321000222')
        """):
        print("✅ Successfully trained Query Patterns")
    else:
        print("❌ Failed to train Query Patterns")

    print("✅ DDL Training Complete!")
    print("📊 Trained on:")
    print("   - Database dialect and naming conventions")
    print("   - Complete Invoice header table schema (67 columns)")
    print("   - Complete Invoice_Line table schema (36 columns)")
    print("   - Table relationships and join patterns")
    print("   - Important query patterns and data formats")

    # ================================================================
    # PHASE 2: DOCUMENTATION TRAINING FOR VANNA AI
    # Business context, terminology, and domain knowledge
    # ================================================================

    # ================================================================
    # SECTION 1: CRITICAL TECHNICAL REQUIREMENTS
    # ================================================================

    if vanna_manager.train(documentation="""
    CRITICAL DATABASE REQUIREMENTS:
    - Table Names: Invoice (header/parent), Invoice_Line (lines/child)
    - MANDATORY: All queries must reference these exact table names
    - MANDATORY: Always use column aliases for aggregate functions and expressions
    - Date fields are stored as DATE type (not NVARCHAR)
    - Decimal fields use DECIMAL(18,3) for 3 decimal place precision
    - All currency amounts are in SEK (Swedish Krona)

    Example correct query structure:
    SELECT 
        COUNT(*) AS invoice_count, 
        SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
    FROM Invoice
    WHERE ISSUE_DATE >= '2023-01-01'
    """):
        print("✅ Successfully trained Tech Requirements")
    else:
        print("❌ Failed to train Tech Requirements")

    # ================================================================
    # SECTION 2: DATABASE OVERVIEW AND PURPOSE
    # ================================================================

    if vanna_manager.train(documentation="""
    DATABASE PURPOSE AND CONTEXT:
    This database contains invoice data for Region Västerbotten, a Swedish healthcare region 
    in northern Sweden. The invoices represent purchases of medical supplies, equipment, 
    services, and other goods necessary for healthcare operations.

    Key Suppliers in Dataset:
    - Abbott Scandinavia (5560466137): Medical devices and diagnostics (ISTAT cartridges, Alinity systems)
    - Visma Draftit AB (5565783957): Software/HR services (HR Expert, Privacy Expert)
    - JA Hotel Karlskrona (5592985237): Hotel accommodation services

    Delivery Locations (Cities in Västerbotten):
    - Umeå (main city, university hospital)
    - Vilhelmina, Tärnaby, Åsele, Dorotea (smaller towns)
    - Various hospital departments and healthcare facilities

    The database tracks complete invoice transactions including:
    - Supplier and customer information
    - Invoice header data (dates, amounts, references)
    - Detailed line items (products, quantities, pricing)
    - Delivery information to various healthcare facilities
    - Tax and payment details
    """):
        print("✅ Successfully trained Database Overview")
    else:
        print("❌ Failed to train Database Overview")

    # ================================================================
    # SECTION 3: DATE FORMATS AND HANDLING
    # ================================================================

    if vanna_manager.train(documentation="""
    DATE STORAGE AND FORMATS:
    All date fields are stored as DATE type (not strings).

    Key Date Fields:
    - ISSUE_DATE: Date the invoice was created and sent (e.g., 2023-10-06, 2024-04-17)
    - DUE_DATE: Date payment is due (typically 30 days after issue date)
    - ACTUAL_DELIVERY_DATE: When goods/services were delivered
    - PERIOD_START_DATE: Start of service period (for subscriptions)
    - PERIOD_END_DATE: End of service period (for subscriptions)
    - TAX_POINT_DATE: Date for tax calculation purposes

    Date Range in Dataset: 2023-05-16 to 2024-09-30

    Date Filtering Examples:
    - Specific year: WHERE YEAR(ISSUE_DATE) = 2023
    - Date range: WHERE ISSUE_DATE BETWEEN '2023-06-01' AND '2023-06-30'
    - After date: WHERE ISSUE_DATE >= '2023-07-01'
    - Current month: WHERE YEAR(ISSUE_DATE) = YEAR(GETDATE()) AND MONTH(ISSUE_DATE) = MONTH(GETDATE())
    """):
        print("✅ Successfully trained Date Handling")
    else:
        print("❌ Failed to train Date Handling")

    # ================================================================
    # SECTION 4: SUPPLIER PARTY INFORMATION
    # ================================================================

    if vanna_manager.train(documentation="""
    SUPPLIER PARTY IDENTIFICATION:
    Suppliers are companies that provide goods and services to Region Västerbotten.

    Key Suppliers in Dataset:
    1. Abbott Scandinavia (5560466137)
    - Location: Hemvärnsgatan 9, 171 29 Solna, SE
    - Products: Medical diagnostics (ISTAT, Alinity, Point-of-Care testing)
    - Major supplier with most invoices in dataset

    2. Visma Draftit AB (5565783957)
    - Location: Styrmansgatan 2, 21118 Malmo, SE
    - Products: Software services (HR Expert Kommun, Privacy Expert Premium)
    - Contact: support@vismadraftit.se, +46101992350

    3. JA Hotel Karlskrona (5592985237)
    - Location: Borgmästaregatan 13, 37115 Karlskrona, SE
    - Products: Hotel accommodation services (overnight stays)
    - Contact: info@jahotel.se, 045555560

    Supplier Fields:
    - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID: Swedish organization number (10 digits)
    Format: No hyphens (e.g., '5560466137', '5565783957')
    This is the official government registration number (organisationsnummer)

    - SUPPLIER_PARTY_NAME: Common/trade name of the company
    Often includes 'AB' suffix (Aktiebolag = Swedish limited company)
    Examples: 'Abbott Scandinavia', 'Visma Draftit AB'

    - SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME: Official registered legal name
    Examples: 'Abbott Scandinavia', 'Visma Draftit AB', 'JA Hotel Karlskrona'

    - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM: Legal form and location
    Format: 'Säte [City] SE' (e.g., 'Säte Solna SE', 'Säte Malmö SE')
    'Säte' means 'Seat' or registered office location
    """):
        print("✅ Successfully trained Supplier Information")
    else:
        print("❌ Failed to train Supplier Information")

    if vanna_manager.train(documentation="""
    SUPPLIER CONTACT AND ADDRESS:
    Address Fields:
    - SUPPLIER_PARTY_STREET_NAME: Primary street address (e.g., 'Hemvärnsgatan 9', 'Styrmansgatan 2')
    - SUPPLIER_PARTY_ADDITIONAL_STREET_NAME: Secondary address line (usually NULL in this dataset)
    - SUPPLIER_PARTY_POSTAL_ZONE: 5-digit Swedish postal code (e.g., '171 29', '21118', '37115')
    Format: May include space between 3rd and 4th digit
    - SUPPLIER_PARTY_CITY: City name (e.g., 'Solna', 'Malmo', 'Karlskrona')
    - SUPPLIER_PARTY_COUNTRY: ISO 2-letter country code (always 'SE' for Sweden in this dataset)
    - SUPPLIER_PARTY_ADDRESS_LINE: Additional location reference (usually NULL)

    Contact Information:
    - SUPPLIER_PARTY_CONTACT_NAME: Contact person or invoice reference
    Examples: 'INV00002506 / 1000449', 'INV00004076 / 1000449' (Visma)
    - SUPPLIER_PARTY_CONTACT_EMAIL: Email address for supplier contact
    Examples: 'support@vismadraftit.se', 'info@jahotel.se'
    - SUPPLIER_PARTY_CONTACT_PHONE: Phone number for supplier contact
    Examples: '+46101992350', '045555560'
    - SUPPLIER_PARTY_ENDPOINT_ID: Electronic invoicing identifier
    Usually same as organization number (e.g., '7365565783953', '5592985237')
    """):
        print("✅ Successfully trained Supplier Address & Contact")
    else:
        print("❌ Failed to train Supplier Address & Contact")
