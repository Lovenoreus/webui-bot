

    if vanna_manager.train(ddl="""
    CREATE TABLE Invoice (
        -- Primary Key
        INVOICE_ID TEXT NOT NULL PRIMARY KEY,
        
        -- Date Information
        ISSUE_DATE TEXT NOT NULL,  -- Format: YYYY-MM-DD (e.g., '2025-06-11')
        DUE_DATE TEXT,              -- Format: YYYY-MM-DD
        TAX_POINT_DATE TEXT,        -- Format: YYYY-MM-DD
        ACTUAL_DELIVERY_DATE TEXT,  -- Format: YYYY-MM-DD
        PERIOD_START_DATE TEXT,     -- Format: YYYY-MM-DD
        PERIOD_END_DATE TEXT,       -- Format: YYYY-MM-DD
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE TEXT,
        
        -- Supplier Party Information
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,  -- Swedish organization number (e.g., '5560466137')
        SUPPLIER_PARTY_NAME TEXT,                              -- Company common name (e.g., 'Instrumenta AB')
        SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,              -- Official legal name
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM TEXT,    -- Legal form and location
        
        -- Supplier Address
        SUPPLIER_PARTY_STREET_NAME TEXT,
        SUPPLIER_PARTY_ADDITIONAL_STREET_NAME TEXT,
        SUPPLIER_PARTY_POSTAL_ZONE TEXT,     -- 5-digit postal code (e.g., '41502')
        SUPPLIER_PARTY_CITY TEXT,            -- City name (e.g., 'Stockholm', 'Solna')
        SUPPLIER_PARTY_COUNTRY TEXT,         -- Country code (e.g., 'SE')
        SUPPLIER_PARTY_ADDRESS_LINE TEXT,
        
        -- Supplier Contact Information
        SUPPLIER_PARTY_CONTACT_NAME TEXT,
        SUPPLIER_PARTY_CONTACT_EMAIL TEXT,
        SUPPLIER_PARTY_CONTACT_PHONE TEXT,
        SUPPLIER_PARTY_ENDPOINT_ID TEXT,
        
        -- Customer Party Information (Region Västerbotten)
        CUSTOMER_PARTY_ID TEXT,
        CUSTOMER_PARTY_ID_SCHEME_ID TEXT,
        CUSTOMER_PARTY_ENDPOINT_ID TEXT,
        CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID TEXT,
        CUSTOMER_PARTY_NAME TEXT,                    -- Often care unit or dept (e.g., 'Region Västerbotten | REF 1050103')
        CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
        CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT,
        
        -- Customer Address
        CUSTOMER_PARTY_STREET_NAME TEXT,
        CUSTOMER_PARTY_POSTAL_ZONE TEXT,
        CUSTOMER_PARTY_COUNTRY TEXT,          -- Usually 'SE', but can be 'NO' or 'FI'
        
        -- Customer Contact Information
        CUSTOMER_PARTY_CONTACT_NAME TEXT,
        CUSTOMER_PARTY_CONTACT_EMAIL TEXT,
        CUSTOMER_PARTY_CONTACT_PHONE TEXT,
        
        -- Delivery Information
        DELIVERY_LOCATION_STREET_NAME TEXT,
        DELIVERY_LOCATION_ADDITIONAL_STREET_NAME TEXT,
        DELIVERY_LOCATION_CITY_NAME TEXT,   -- Northern Sweden cities (e.g., 'Kiruna')
        DELIVERY_LOCATION_POSTAL_ZONE TEXT,
        DELIVERY_LOCATION_ADDRESS_LINE TEXT,
        DELIVERY_LOCATION_COUNTRY TEXT,
        DELIVERY_PARTY_NAME TEXT,
        
        -- Currency
        DOCUMENT_CURRENCY_CODE TEXT,          -- Usually 'SEK'
        
        -- Tax Information
        TAX_AMOUNT_CURRENCY TEXT,             -- Usually 'SEK'
        TAX_AMOUNT REAL,
        
        -- Legal Monetary Totals
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT REAL,           -- Sum of line items before tax
        
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT REAL,           -- Total excluding tax
        
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT REAL,           -- Total including tax
        
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT REAL,            -- Final amount to be paid
        
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT REAL,    -- Discounts/reductions
        
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT REAL,       -- Additional charges
        
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT REAL,   -- Rounding adjustment
        
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT REAL,            -- Prepaid amount
        
        -- Reference Information
        BUYER_REFERENCE TEXT,               -- Internal reference from region
        PROJECT_REFERENCE_ID TEXT,          -- Project identifier
        ORDER_REFERENCE_ID TEXT,            -- Purchase order reference
        ORDER_REFERENCE_SALES_ORDER_ID TEXT,
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID TEXT,
        CONTRACT_DOCUMENT_REFERENCE_ID TEXT,
        DESPATCH_DOCUMENT_REFERENCE_ID TEXT,
        
        -- Invoice Type and Notes
        INVOICE_TYPE_CODE TEXT,              -- 380=standard, 381=credit note, 383=prepayment
        NOTE TEXT,                           -- Free-text notes from supplier
        ACCOUNTING_COST TEXT,                -- Cost center/accounting code
        PAYMENT_TERMS_NOTE TEXT,             -- Payment terms description
        
        -- ETL Metadata
        ETL_LOAD_TS TEXT                     -- Timestamp when loaded to warehouse
    )
    """):
        print("✅ Successfully trained Invoice DDL")

    else:
        print("❌ Failed to train Invoice DDL")

    # Train Vanna with DDL and documentation
    print("\n📚 Training Vanna with DDL and documentation...")
    print("Training with Invoice DDL...")

    # if vanna_manager.train(ddl=invoice_ddl):
    #     print("✅ Successfully trained Invoice DDL")

    # else:
    #     print("❌ Failed to train Invoice DDL")

    print("Training with Invoice_Line DDL...")
    if vanna_manager.train(ddl="""
    CREATE TABLE Invoice_Line (
        -- Composite Primary Key
        INVOICE_ID TEXT NOT NULL,            -- Links to Invoice.INVOICE_ID
        INVOICE_LINE_ID TEXT NOT NULL,       -- Unique line identifier within invoice
        PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
        
        -- Date Information
        ISSUE_DATE TEXT NOT NULL,            -- Format: YYYY-MM-DD (e.g., '2025-12-31')
        INVOICE_PERIOD_START_DATE TEXT,      -- Format: YYYY-MM-DD
        INVOICE_PERIOD_END_DATE TEXT,        -- Format: YYYY-MM-DD
        
        -- Supplier Reference
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,  -- Swedish org number (e.g., '5590081294')
        
        -- Line Item References
        ORDER_LINE_REFERENCE_LINE_ID TEXT,   -- Reference to purchase order line
        ACCOUNTING_COST TEXT,                -- Cost center code for this line
        INVOICE_LINE_DOCUMENT_REFERENCE_ID TEXT,       -- Related document reference
        INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE TEXT,  -- 130=Delivery note, 751=Work order
        
        -- Quantity and Amount Information
        INVOICED_QUANTITY REAL,              -- Quantity invoiced (e.g., 27.680)
        INVOICED_QUANTITY_UNIT_CODE TEXT,    -- Unit of measure (e.g., 'EA', 'HUR', 'XHG')
        INVOICED_LINE_EXTENSION_AMOUNT REAL,           -- Line total excluding tax
        INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID TEXT, -- Usually 'SEK'
        
        -- Item Description
        ITEM_NAME TEXT,                      -- Short item name/title
        ITEM_DESCRIPTION TEXT,               -- Detailed description of goods/services
        INVOICE_LINE_NOTE TEXT,              -- Additional line-level notes
        
        -- Tax Information
        ITEM_TAXCAT_ID TEXT,                 -- Tax category (e.g., 'S' for standard rate)
        ITEM_TAXCAT_PERCENT REAL,            -- Tax percentage (typically 25.00 for Swedish VAT)
        
        -- Item Identifiers
        ITEM_BUYERS_ID TEXT,                 -- Region's internal article number
        ITEM_SELLERS_ITEM_ID TEXT,           -- Supplier's article number
        ITEM_STANDARD_ITEM_ID TEXT,          -- Global identifier (GTIN/EAN)
        
        -- Item Classification
        ITEM_COMMODITYCLASS_CLASSIFICATION TEXT,        -- Classification code (CPV, UNSPSC)
        ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID TEXT, -- Classification system ('MP', 'STI')
        
        -- Pricing Information
        PRICE_AMOUNT REAL,                   -- Unit price excluding tax
        PRICE_AMOUNT_CURRENCY_ID TEXT,       -- Usually 'SEK'
        PRICE_BASE_QUANTITY REAL,            -- Base quantity for pricing (often 1.0000)
        PRICE_BASE_QUANTITY_UNIT_CODE TEXT,  -- Unit for base quantity
        PRICE_ALLOWANCE_CHARGE_AMOUNT REAL,  -- Discount or charge amount
        PRICE_ALLOWANCE_CHARGE_INDICATOR INTEGER,        -- 0=allowance/discount, 1=charge
        
        -- ETL Metadata
        ETL_LOAD_TS TEXT                     -- Timestamp when loaded (e.g., '2025-10-12 14:35:42.123')
    )
    """):
        print("✅ Successfully trained Invoice_Line DDL")

    else:
        print("❌ Failed to train Invoice_Line DDL")


    if vanna_manager.train(ddl="""
    -- Relationship between Invoice and InvoiceLine tables
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


    if vanna_manager.train(ddl="""
    -- Always use column aliases for aggregate functions:
    -- CORRECT: SELECT COUNT(*) AS invoice_count, SUM(amount) AS total_amount
    -- INCORRECT: SELECT COUNT(*), SUM(amount)

    -- Date filtering (dates stored as TEXT in YYYY-MM-DD format):
    -- WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'

    -- Swedish-specific considerations:
    -- - Organization numbers: SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID (e.g., '5560466137')
    -- - Currency: Usually 'SEK' (Swedish Krona)
    -- - VAT rate: Usually 25.00% in ITEM_TAXCAT_PERCENT
    -- - Postal codes: 5 digits (e.g., '41502')
    -- - Country codes: ISO 2-letter ('SE', 'NO', 'FI')
    """):
        print("✅ Successfully trained Query Patterns")

    else:
        print("❌ Failed to train Query Patterns")

    print("✅ DDL Training Complete!")
    print("📊 Trained on:")
    print("   - Database dialect and naming conventions")
    print("   - Complete Invoice header table schema (67 columns)")
    print("   - Complete InvoiceLine table schema (36 columns)")
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
    - Database Type: SQLite
    - Dialect: Standard SQL (SQLite)
    - Database File: sqlite_invoices_full.db
    - Schema: N/A (SQLite does not use schemas)
    - MANDATORY: ALL table references MUST use table names exactly as created (case-sensitive where applicable)
    - MANDATORY: Always use column aliases for aggregate functions and expressions
    Example: SELECT COUNT(*) AS count, SUM(amount) AS total_amount
    """):
        print("✅ Successfully trained Tech Requirements")

    else:
        print("❌ Failed to train Tech Requirements")

    # ================================================================
    # SECTION 2: DATABASE OVERVIEW AND PURPOSE
    # ================================================================

    if vanna_manager.train(documentation="""
    DATABASE PURPOSE AND CONTEXT:
    This SQLite database contains invoice data for Region Västerbotten, a Swedish healthcare region 
    in northern Sweden. The invoices represent purchases of medical supplies, equipment, 
    services, and other goods necessary for healthcare operations. Suppliers include medical 
    device companies, pharmaceutical suppliers, service providers, and equipment vendors.

    The database tracks complete invoice transactions including:
    - Supplier and customer information
    - Invoice header data (dates, amounts, references)
    - Detailed line items (products, quantities, pricing)
    - Delivery information
    - Tax and payment details
    """):
        print("✅ Successfully trained Tech Requirements")

    else:
        print("❌ Failed to train Tech Requirements")

    # ================================================================
    # SECTION 3: DATE FORMATS AND HANDLING
    # ================================================================

    if vanna_manager.train(documentation="""
    DATE STORAGE AND FORMATS:
    All date fields are stored as TEXT in the format YYYY-MM-DD.

    Examples:
    - ISSUE_DATE: '2025-06-11' (the date the invoice was created and sent)
    - DUE_DATE: '2025-05-02' (the date payment is due)
    - ACTUAL_DELIVERY_DATE: '2024-01-07' (when goods/services were delivered)
    - PERIOD_START_DATE: '2025-01-01' (start of invoiced period for subscriptions/services)
    - PERIOD_END_DATE: '2025-01-31' (end of invoiced period)

    When filtering by dates, use standard string comparison:
    WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'

    When working with current date in SQLite, use:
    strftime('%Y-%m-%d', 'now')
    """):
        print("✅ Successfully trained Tech Requirements")

    else:
        print("❌ Failed to train Tech Requirements")

    # ================================================================
    # SECTION 4: SUPPLIER PARTY INFORMATION
    # ================================================================

    if vanna_manager.train(documentation="""
    SUPPLIER PARTY IDENTIFICATION:
    Suppliers are companies that provide goods and services to Region Västerbotten.

    Key Supplier Fields:
    - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID: Swedish organization number (organisationsnummer)
    Format: 10 digits, no hyphens (e.g., '5560466137', '5590081294')
    This is the official government registration number for Swedish companies.

    - SUPPLIER_PARTY_NAME: Common/trade name of the company
    Often includes 'AB' suffix (Aktiebolag = Swedish limited company)
    Examples: 'Instrumenta AB', 'Hotel Botnia AB', 'Medtronic AB'

    - SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME: Official registered legal name
    Usually longer and more formal than SUPPLIER_PARTY_NAME
    Almost always includes 'AB' suffix

    - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM: Legal form and location
    Format examples: 'Säte Askim SE', 'Säte Stockholm SE'
    'Säte' means 'Seat' or registered office location
    """):
        print("✅ Successfully trained Tech Requirements")

    else:
        print("❌ Failed to train Tech Requirements")

    if vanna_manager.train(documentation="""
    SUPPLIER CONTACT AND ADDRESS:
    - SUPPLIER_PARTY_STREET_NAME: Primary street address (e.g., 'Styrmansgatan 2', 'Box 102')
    - SUPPLIER_PARTY_ADDITIONAL_STREET_NAME: Alternative or secondary address line
    - SUPPLIER_PARTY_POSTAL_ZONE: 5-digit Swedish postal code (e.g., '41502', '11234')
    - SUPPLIER_PARTY_CITY: City name (e.g., 'Stockholm', 'Göteborg', 'Solna', 'Uppsala')
    - SUPPLIER_PARTY_COUNTRY: ISO 2-letter country code (typically 'SE' for Sweden)
    - SUPPLIER_PARTY_ADDRESS_LINE: Additional location reference (e.g., 'Sweden', 'Kundtjänst', 'Bemanning')

    Contact Information:
    - SUPPLIER_PARTY_CONTACT_NAME: Contact person or reference (e.g., 'Andersson, Albin', phone numbers)
    - SUPPLIER_PARTY_CONTACT_EMAIL: Email address for supplier contact
    - SUPPLIER_PARTY_CONTACT_PHONE: Phone number for supplier contact
    - SUPPLIER_PARTY_ENDPOINT_ID: Electronic invoicing identifier (e.g., '9164770787')
    """):
        print("✅ Successfully trained Tech Requirements")

    else:
        print("❌ Failed to train Tech Requirements")

#TODO: End of the block 
    # ================================================================
    # SECTION 5: CUSTOMER PARTY INFORMATION (REGION VÄSTERBOTTEN)
    # ================================================================

