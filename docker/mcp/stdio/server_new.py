# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Optional
import os

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- User-defined Modules --------------------
import config
from query_engine import QueryEngine
from instructions import SQLITE_INVOICE_PROMPT, SQLSERVER_INVOICE_PROMPT
# from vanna_engine import initialize_vanna_engine, vanna_manager, generate_sql
from vanna_engine import VannaModelManager
from training import get_vanna_training, get_vanna_question_sql_pairs, normalize_for_comprehensive_search, get_comprehensive_search_instructions
from models import (
    QueryDatabaseRequest,
    GreetRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
)

load_dotenv(find_dotenv())

# Debug flag
DEBUG = True

app = FastAPI(
    title="Invoice SQL MCP Server",
    description="MCP Tools Server with LLM SQL Generation for Invoice Management"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def get_database_server_url():
    """Determine the correct database server URL based on environment"""
    if os.environ.get('DOCKER_CONTAINER') or os.path.exists('/.dockerenv'):
        return config.MCP_DOCKER_DATABASE_SERVER_URL

    else:
        return config.MCP_DATABASE_SERVER_URL


# Database configuration
DATABASE_CHOICE = config.DATABASE_CHOICE
USE_REMOTE = DATABASE_CHOICE == "remote"

if USE_REMOTE:
    DATABASE_SERVER_URL = f"SQL Server: {config.SQL_SERVER_HOST}/{config.SQL_SERVER_DATABASE}"

else:
    DATABASE_SERVER_URL = get_database_server_url()

if DEBUG:
    print(f"[MCP DEBUG] Database mode: {DATABASE_CHOICE}")
    print(f"[MCP DEBUG] Database info: {DATABASE_SERVER_URL}")

# Invoice-specific configuration
INVOICE_TABLES = [
    'Invoice',
    'Invoice_Line'
]

INVOICE_TABLE_VARIATIONS = {
    'invoice': ['invoice', 'bill', 'receipt', 'payment'],
    'invoice_line': ['line', 'item', 'product', 'service']
}


if USE_REMOTE:
    INVOICE_SYSTEM_PROMPT = SQLSERVER_INVOICE_PROMPT

else:
    INVOICE_SYSTEM_PROMPT = SQLITE_INVOICE_PROMPT


if DEBUG:
    print(f"[MCP DEBUG] Database mode: {DATABASE_CHOICE}")
    print(f"[MCP DEBUG] Using prompt: {'SQL Server' if USE_REMOTE else 'SQLite'}")
    print(f"[MCP DEBUG] Database info: {DATABASE_SERVER_URL}")


# Initialize QueryEngine with invoice configuration
query_engine = QueryEngine(
    database_url="http://database_server:8762",
    tables=INVOICE_TABLES,
    table_variations=INVOICE_TABLE_VARIATIONS,
    system_prompt=INVOICE_SYSTEM_PROMPT,
    debug=DEBUG,
    use_remote=USE_REMOTE
)


# Initialize Vanna Manager at module level
print("Initializing Vanna Manager...")
# Custom path
vanna_manager = VannaModelManager(
    chroma_path=config.CHROMA_DB_PATH,
    clear_existing=True  # Clear the path before setup.
)

# Initialize on module load
print("Initializing Vanna...")
vanna_manager.initialize_vanna()

print(f"✅ Vanna initialized with provider: {vanna_manager.current_provider}")

# Train with your schema
print("📚 Training Vanna with schema...")

print(f"Remote choice for Vanna is: {USE_REMOTE}")

# Get the training data.
invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, synonym_instructions, training_pairs = get_vanna_training(remote=USE_REMOTE)


# MANUAL TRAINING. TO SEND TO TRAINING.
# CRITICAL: Database dialect and naming conventions
if vanna_manager.train(ddl="""
-- Database: Nodinite
-- Schema: dbo
-- Dialect: T-SQL (Microsoft SQL Server)
-- CRITICAL REQUIREMENT: ALL table references MUST use three-part names: [Nodinite].[dbo].[TableName]
"""):
    print("✅ Successfully trained Schema DDL")

# ================================================================
# TABLE 1: Invoice Header Table (Main Invoice Information)
# ================================================================

if vanna_manager.train(ddl="""
CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] (
    -- Primary Key
    INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
    
    -- Date Information
    ISSUE_DATE NVARCHAR(10) NOT NULL,  -- Format: YYYY-MM-DD (e.g., '2025-06-11')
    DUE_DATE NVARCHAR(10),              -- Format: YYYY-MM-DD
    TAX_POINT_DATE NVARCHAR(10),        -- Format: YYYY-MM-DD
    ACTUAL_DELIVERY_DATE NVARCHAR(10),  -- Format: YYYY-MM-DD
    PERIOD_START_DATE NVARCHAR(10),     -- Format: YYYY-MM-DD
    PERIOD_END_DATE NVARCHAR(10),       -- Format: YYYY-MM-DD
    BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(10),
    
    -- Supplier Party Information
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,  -- Swedish organization number (e.g., '5560466137')
    SUPPLIER_PARTY_NAME NVARCHAR(255),                             -- Company common name (e.g., 'Instrumenta AB')
    SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),           -- Official legal name
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100), -- Legal form and location
    
    -- Supplier Address
    SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),
    SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
    SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),     -- 5-digit postal code (e.g., '41502')
    SUPPLIER_PARTY_CITY NVARCHAR(100),           -- City name (e.g., 'Stockholm', 'Solna')
    SUPPLIER_PARTY_COUNTRY NVARCHAR(2),          -- Country code (e.g., 'SE')
    SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
    
    -- Supplier Contact Information
    SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
    SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
    SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
    SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
    
    -- Customer Party Information (Region Västerbotten)
    CUSTOMER_PARTY_ID NVARCHAR(50),
    CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),
    CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
    CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
    CUSTOMER_PARTY_NAME NVARCHAR(255),                    -- Often care unit or dept (e.g., 'Region Västerbotten | REF 1050103')
    CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
    CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),
    
    -- Customer Address
    CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),
    CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),
    CUSTOMER_PARTY_COUNTRY NVARCHAR(2),          -- Usually 'SE', but can be 'NO' or 'FI'
    
    -- Customer Contact Information
    CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
    CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
    CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
    
    -- Delivery Information
    DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),
    DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
    DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),   -- Northern Sweden cities (e.g., 'Kiruna')
    DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),
    DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
    DELIVERY_LOCATION_COUNTRY NVARCHAR(2),
    DELIVERY_PARTY_NAME NVARCHAR(255),
    
    -- Currency
    DOCUMENT_CURRENCY_CODE NVARCHAR(3),          -- Usually 'SEK'
    
    -- Tax Information
    TAX_AMOUNT_CURRENCY NVARCHAR(3),             -- Usually 'SEK'
    TAX_AMOUNT DECIMAL(18,2),
    
    -- Legal Monetary Totals
    LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,2),           -- Sum of line items before tax
    
    LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,2),           -- Total excluding tax
    
    LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,2),           -- Total including tax
    
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,2),            -- Final amount to be paid
    
    LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,2),    -- Discounts/reductions
    
    LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,2),       -- Additional charges
    
    LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,2),   -- Rounding adjustment
    
    LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
    LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,2),            -- Prepaid amount
    
    -- Reference Information
    BUYER_REFERENCE NVARCHAR(100),               -- Internal reference from region
    PROJECT_REFERENCE_ID NVARCHAR(100),          -- Project identifier
    ORDER_REFERENCE_ID NVARCHAR(100),            -- Purchase order reference
    ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
    BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
    CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),
    DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
    
    -- Invoice Type and Notes
    INVOICE_TYPE_CODE NVARCHAR(10),              -- 380=standard, 381=credit note, 383=prepayment
    NOTE NVARCHAR(MAX),                          -- Free-text notes from supplier
    ACCOUNTING_COST NVARCHAR(100),               -- Cost center/accounting code
    PAYMENT_TERMS_NOTE NVARCHAR(MAX),            -- Payment terms description
    
    -- ETL Metadata
    ETL_LOAD_TS NVARCHAR(30)                     -- Timestamp when loaded to warehouse
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
CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] (
    -- Composite Primary Key
    INVOICE_ID NVARCHAR(50) NOT NULL,            -- Links to LLM_OnPrem_Invoice_kb.INVOICE_ID
    INVOICE_LINE_ID NVARCHAR(50) NOT NULL,       -- Unique line identifier within invoice
    PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
    
    -- Date Information
    ISSUE_DATE NVARCHAR(10) NOT NULL,            -- Format: YYYY-MM-DD (e.g., '2025-12-31')
    INVOICE_PERIOD_START_DATE NVARCHAR(10),      -- Format: YYYY-MM-DD
    INVOICE_PERIOD_END_DATE NVARCHAR(10),        -- Format: YYYY-MM-DD
    
    -- Supplier Reference
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,  -- Swedish org number (e.g., '5590081294')
    
    -- Line Item References
    ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(50),   -- Reference to purchase order line
    ACCOUNTING_COST NVARCHAR(100),               -- Cost center code for this line
    INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),       -- Related document reference
    INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(10),  -- 130=Delivery note, 751=Work order
    
    -- Quantity and Amount Information
    INVOICED_QUANTITY DECIMAL(18,3),             -- Quantity invoiced (e.g., 27.680)
    INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),    -- Unit of measure (e.g., 'EA', 'HUR', 'XHG')
    INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,2),           -- Line total excluding tax
    INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3), -- Usually 'SEK'
    
    -- Item Description
    ITEM_NAME NVARCHAR(500),                     -- Short item name/title
    ITEM_DESCRIPTION NVARCHAR(MAX),              -- Detailed description of goods/services
    INVOICE_LINE_NOTE NVARCHAR(MAX),             -- Additional line-level notes
    
    -- Tax Information
    ITEM_TAXCAT_ID NVARCHAR(10),                 -- Tax category (e.g., 'S' for standard rate)
    ITEM_TAXCAT_PERCENT DECIMAL(18,2),           -- Tax percentage (typically 25.00 for Swedish VAT)
    
    -- Item Identifiers
    ITEM_BUYERS_ID NVARCHAR(100),                -- Region's internal article number
    ITEM_SELLERS_ITEM_ID NVARCHAR(100),          -- Supplier's article number
    ITEM_STANDARD_ITEM_ID NVARCHAR(100),         -- Global identifier (GTIN/EAN)
    
    -- Item Classification
    ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),        -- Classification code (CPV, UNSPSC)
    ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(50), -- Classification system ('MP', 'STI')
    
    -- Pricing Information
    PRICE_AMOUNT DECIMAL(18,2),                  -- Unit price excluding tax
    PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),        -- Usually 'SEK'
    PRICE_BASE_QUANTITY DECIMAL(18,4),           -- Base quantity for pricing (often 1.0000)
    PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),  -- Unit for base quantity
    PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,2), -- Discount or charge amount
    PRICE_ALLOWANCE_CHARGE_INDICATOR BIT,        -- false=allowance/discount, true=charge
    
    -- ETL Metadata
    ETL_LOAD_TS NVARCHAR(30)                     -- Timestamp when loaded (e.g., '2025-10-12 14:35:42.123')
)
"""):
    print("✅ Successfully trained Invoice_Line DDL")

else:
    print("❌ Failed to train Invoice_Line DDL")


if vanna_manager.train(ddl="""
-- Relationship between Invoice and InvoiceLine tables
-- LLM_OnPrem_Invoice_kb (Parent/Header)
--   └─ INVOICE_ID (Primary Key)
--        │
--        └─ Links to ─> LLM_OnPrem_InvoiceLine_kb (Child/Lines)
--                        └─ INVOICE_ID (Foreign Key, part of composite PK)
--
-- To get complete invoice with line items, JOIN on INVOICE_ID:
-- FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] inv
-- INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] line 
--     ON inv.INVOICE_ID = line.INVOICE_ID
"""):

    print("✅ Successfully trained Table Relationship")

else:
    print("❌ Failed to train Table Relationship")


if vanna_manager.train(ddl="""
-- Always use column aliases for aggregate functions:
-- CORRECT: SELECT COUNT(*) AS invoice_count, SUM(amount) AS total_amount
-- INCORRECT: SELECT COUNT(*), SUM(amount)

-- Date filtering (dates stored as NVARCHAR in YYYY-MM-DD format):
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
- Database Type: Microsoft SQL Server
- Dialect: T-SQL (Transact-SQL)
- Database Name: Nodinite
- Schema: dbo
- MANDATORY: ALL table references MUST use full three-part names: [Nodinite].[dbo].[TableName]
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
This database contains invoice data for Region Västerbotten, a Swedish healthcare region 
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
All date fields are stored as NVARCHAR(10) in the format YYYY-MM-DD.

Examples:
- ISSUE_DATE: '2025-06-11' (the date the invoice was created and sent)
- DUE_DATE: '2025-05-02' (the date payment is due)
- ACTUAL_DELIVERY_DATE: '2024-01-07' (when goods/services were delivered)
- PERIOD_START_DATE: '2025-01-01' (start of invoiced period for subscriptions/services)
- PERIOD_END_DATE: '2025-01-31' (end of invoiced period)

When filtering by dates, use standard string comparison:
WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'

When working with current date, use CONVERT for proper formatting:
CONVERT(NVARCHAR(10), GETDATE(), 23) produces 'YYYY-MM-DD' format
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

# ================================================================
# SECTION 5: CUSTOMER PARTY INFORMATION (REGION VÄSTERBOTTEN)
# ================================================================

if vanna_manager.train(documentation="""
CUSTOMER PARTY IDENTIFICATION:
The customer is Region Västerbotten, a Swedish regional healthcare authority.
Invoices are typically directed to specific departments, care units, or cost centers.

Key Customer Fields:
- CUSTOMER_PARTY_NAME: Often identifies specific care units or departments
  Common format: 'Region Västerbotten | REF XXXXXXX'
  Examples: 'Region Västerbotten | REF 1050103', 'Region Västerbotten 3025'
  The REF number identifies the specific unit or department

- CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID: Legal identifier for the regional unit
  Format: Numeric identifier (e.g., '7362321000224')

- CUSTOMER_PARTY_ID: Unit-specific identification number
  Various formats (e.g., '0012998853')

- CUSTOMER_PARTY_ENDPOINT_ID: Electronic invoicing identifier for the region
  Format: Numeric (e.g., '9164770787')
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

if vanna_manager.train(documentation="""
CUSTOMER LOCATION AND CONTACT:
Region Västerbotten is located in northern Sweden (Norrland).

Address Fields:
- CUSTOMER_PARTY_STREET_NAME: Care unit name or street address
  Examples: 'VLL REF 2037040', 'Lasarettsvägen 29'
  VLL = Västerbottens Läns Landsting (former name of Region Västerbotten)

- CUSTOMER_PARTY_POSTAL_ZONE: Postal code, sometimes with country prefix
  Format: 'SE-831 07' or '90185'

- CUSTOMER_PARTY_COUNTRY: ISO 2-letter country code
  Usually 'SE' (Sweden), but can be 'NO' (Norway) or 'FI' (Finland) for border services

Contact Information:
- CUSTOMER_PARTY_CONTACT_NAME: Regional contact person or reference number
  Examples: 'Anna Andersson', '52702', '250314-214'

- CUSTOMER_PARTY_CONTACT_EMAIL: Official email address
  Format: firstname.lastname@regionvasterbotten.se
  Example: 'hanna.henriksson@regionvasterbotten.se'

- CUSTOMER_PARTY_CONTACT_PHONE: Work or mobile phone number for responsible person
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 6: DELIVERY INFORMATION
# ================================================================

if vanna_manager.train(documentation="""
DELIVERY LOCATION AND DETAILS:
Tracks where goods or services were delivered within the region.

Delivery Location:
- DELIVERY_LOCATION_STREET_NAME: Street address or department name for delivery
- DELIVERY_LOCATION_ADDITIONAL_STREET_NAME: Additional location information
- DELIVERY_LOCATION_CITY_NAME: City in northern Sweden or Västerbotten
  Common cities: Umeå, Skellefteå, Lycksele, Storuman, Vilhelmina, Åsele, 
  Dorotea, Malå, Norsjö, Bjurholm, Vindeln, Robertsfors, Vännäs, Kiruna

- DELIVERY_LOCATION_POSTAL_ZONE: 5-digit Swedish postal code for delivery location
- DELIVERY_LOCATION_ADDRESS_LINE: Department name or specific location within facility
- DELIVERY_LOCATION_COUNTRY: ISO 2-letter country code (usually 'SE')

- DELIVERY_PARTY_NAME: Name of department or recipient responsible for receiving delivery
- ACTUAL_DELIVERY_DATE: Date when goods/services were actually delivered (format: YYYY-MM-DD)
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 7: CURRENCY AND MONETARY AMOUNTS
# ================================================================

if vanna_manager.train(documentation="""
CURRENCY AND AMOUNT FIELDS:
All monetary amounts are stored as DECIMAL(18,2) with two decimal places.
Currency codes follow ISO 4217 standard.

DOCUMENT_CURRENCY_CODE: The currency for the entire invoice
- Typically 'SEK' (Swedish Krona)
- All amount fields in the invoice use this currency

MONETARY AMOUNT BREAKDOWN:
1. LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT: Sum of all line item amounts BEFORE tax, discounts, and charges
   This is the subtotal of all invoice lines

2. LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT: Total invoice amount EXCLUDING tax
   After applying allowances and charges

3. TAX_AMOUNT: Total tax/VAT amount applied to the invoice
   Currency specified in TAX_AMOUNT_CURRENCY (usually 'SEK')

4. LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT: Total invoice amount INCLUDING tax

5. LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT: Total discounts or reductions applied

6. LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT: Total additional charges applied

7. LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT: Amount already prepaid by the region before this invoice

8. LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT: Rounding adjustment (typically small amounts)

9. LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT: FINAL AMOUNT DUE to be paid by the customer
   This is the most important amount field - what the region actually owes

Amount Calculation Formula:
PAYABLE_AMOUNT = TAX_INCL_AMOUNT - PREPAID_AMOUNT + ROUNDING_AMOUNT
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 8: TAX INFORMATION (SWEDISH VAT)
# ================================================================

if vanna_manager.train(documentation="""
SWEDISH TAX (VAT/MOMS) INFORMATION:
Sweden uses 'moms' (mervärdesskatt) which is Value Added Tax (VAT).

Tax Fields in Invoice Header:
- TAX_AMOUNT: Total tax amount for the entire invoice
- TAX_AMOUNT_CURRENCY: Currency code for tax (usually 'SEK')
- TAX_POINT_DATE: Date when tax becomes chargeable (may differ from issue date)

Tax Fields in Invoice Lines:
- ITEM_TAXCAT_ID: Tax category code
  Common values: 'S' = Standard rate, 'Z' = Zero rate, 'E' = Exempt

- ITEM_TAXCAT_PERCENT: Tax percentage rate
  Standard Swedish VAT is 25.00%
  Reduced rates: 12% (food, hotels), 6% (books, newspapers, transport)
  Medical supplies may have different rates depending on classification

Tax Calculation:
Line Amount Excluding Tax × (ITEM_TAXCAT_PERCENT / 100) = Tax Amount
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 9: INVOICE TYPES AND CODES
# ================================================================

if vanna_manager.train(documentation="""
INVOICE TYPE CODES (INVOICE_TYPE_CODE):
Standard UNCL 1001 invoice type codes:

- 380: Standard commercial invoice (most common)
- 381: Credit note (refund or correction reducing amount owed)
- 383: Prepayment invoice (advance payment before delivery)
- 384: Corrected invoice
- 386: Prepayment credit note
- 389: Self-billed invoice
- 390: Delcredere invoice
- 393: Factored invoice
- 394: Lease invoice
- 395: Consignment invoice

Most invoices in this database will be type 380 (standard) or 381 (credit note).
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 10: REFERENCE FIELDS
# ================================================================

if vanna_manager.train(documentation="""
REFERENCE AND IDENTIFIER FIELDS:

ORDER REFERENCES:
- ORDER_REFERENCE_ID: Purchase order number from Region Västerbotten's ordering system
  Links the invoice back to the original purchase order
- ORDER_REFERENCE_SALES_ORDER_ID: Supplier's internal sales order reference

PROJECT AND ACCOUNTING:
- PROJECT_REFERENCE_ID: Project identifier if invoice relates to a specific project
- BUYER_REFERENCE: Internal reference from the region (requester or department identifier)
- ACCOUNTING_COST: Cost center or accounting code used by the region for budget tracking
  Used for internal financial reporting and cost allocation

DOCUMENT REFERENCES:
- BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID: Reference to a previous invoice
  Used for credit notes, corrections, or when referencing another invoice

- CONTRACT_DOCUMENT_REFERENCE_ID: Contract number under which goods/services were provided

- DESPATCH_DOCUMENT_REFERENCE_ID: Delivery note or dispatch document reference
  Links invoice to the physical delivery documentation

NOTES:
- NOTE: Free-text field for supplier comments
  Often contains contact person names, dates, special instructions
  
- PAYMENT_TERMS_NOTE: Description of payment terms
  Examples: 'Net 30 days', '30 dagar netto', 'Betalningsvillkor: 30 dagar'
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

# ================================================================
# SECTION 11: INVOICE LINE ITEM DETAILS
# ================================================================

if vanna_manager.train(documentation="""
INVOICE LINE ITEM STRUCTURE:
Each invoice can have multiple line items in the LLM_OnPrem_InvoiceLine_kb table.

LINE IDENTIFICATION:
- INVOICE_LINE_ID: Unique identifier for the line within the invoice
  Format: Often sequential numbers ('1', '2', '3') or codes ('10517')
  
- ORDER_LINE_REFERENCE_LINE_ID: Links to specific line on the purchase order
  Connects invoice line back to what was originally ordered

QUANTITY AND UNITS:
- INVOICED_QUANTITY: Quantity of goods or services on this line
  Format: DECIMAL(18,3) allows for fractional quantities (e.g., 27.680)
  
- INVOICED_QUANTITY_UNIT_CODE: Unit of measure
  Common codes:
  * 'EA' = Each (individual items)
  * 'HUR' = Hour (time-based services)
  * 'XHG' = Piece/Unit
  * 'MTR' = Meter
  * 'KGM' = Kilogram
  * 'LTR' = Liter
  * 'SET' = Set
  * 'PCE' = Piece

LINE AMOUNTS:
- INVOICED_LINE_EXTENSION_AMOUNT: Total for this line EXCLUDING tax
  Calculation: INVOICED_QUANTITY × PRICE_AMOUNT (after discounts)
  
- INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID: Currency code (usually 'SEK')
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

if vanna_manager.train(documentation="""
ITEM DESCRIPTIONS AND NAMES:
- ITEM_NAME: Short name or title of the item
  Examples:
  * Medical: 'SAX METZENBAUM NOIR VÅGSLIPN.KRÖKT 200MM'
  * Generic: 'Ert ordernr: 3308-3309'
  * Service: 'Ambulanstransport'
  
- ITEM_DESCRIPTION: Detailed description of goods or services
  Medical supplies example: '[61-4005] H/S Elliptosphere kateter (5 Fr) Art: 61-4005 10/fp'
  Transport example: '19690518-8196 / Karlsson, Albin körd från Kyrkogatan 12, Malå till Malå Vårdcentral 939 31 MALÅ, 1.0km'
  
  Often includes:
  * Article numbers
  * Patient identifiers (anonymized)
  * Detailed specifications
  * Transport routes and distances
  * Packaging quantities

- INVOICE_LINE_NOTE: Additional free-text notes for this specific line
  Example format:
  'Vår referens B2306190015
   Ert referensnr NIMA01
   Lasarettsvägen 29
   Skellefteå
   Tillhand. Parkeringsautomat'
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

if vanna_manager.train(documentation="""
ITEM IDENTIFIERS AND CLASSIFICATION:
Multiple identification systems for items:

ITEM IDENTIFIERS:
- ITEM_BUYERS_ID: Region Västerbotten's internal article/item number
  Format: 'REG-ITEM-00123' or similar internal codes
  
- ITEM_SELLERS_ITEM_ID: Supplier's own article/SKU number
  Format: 'SUP-ITEM-789', manufacturer part numbers
  
- ITEM_STANDARD_ITEM_ID: Global standardized identifier
  Format: GTIN (Global Trade Item Number) or EAN (European Article Number)
  Example: '7311234567890' (13-digit barcode number)

ITEM CLASSIFICATION:
- ITEM_COMMODITYCLASS_CLASSIFICATION: Classification code
  Format: Numeric codes like '50532300'
  Used for categorizing types of products/services
  
- ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID: Classification system identifier
  Common values:
  * 'MP' = UNSPSC (United Nations Standard Products and Services Code)
  * 'STI' = CPV (Common Procurement Vocabulary)
  * Other standardized classification systems
"""):
    print("✅ Successfully trained Tech Requirements")

else:
    print("❌ Failed to train Tech Requirements")

if vanna_manager.train(documentation="""
PRICING INFORMATION (INVOICE LINE LEVEL):
Detailed pricing breakdown for each line item:

UNIT PRICE:
- PRICE_AMOUNT: Unit price of the item EXCLUDING tax and before allowances
  Format: DECIMAL(18,2) - e.g., 1250.00 SEK per unit
  
- PRICE_AMOUNT_CURRENCY_ID: Currency for the price (usually 'SEK')

BASE QUANTITY FOR PRICING:
- PRICE_BASE_QUANTITY: Quantity used as basis for the unit price
  Usually 1.0000 (price per single unit)
  Sometimes other values for bulk pricing (e.g., price per 10 units)
  
- PRICE_BASE_QUANTITY_UNIT_CODE: Unit of measure for base quantity
  Examples: 'EA', 'HUR', 'KGM'

DISCOUNTS AND CHARGES:
- PRICE_ALLOWANCE_CHARGE_AMOUNT: Amount of discount or additional charge
  Format: DECIMAL(18,2) - e.g., 50.00 for 50 SEK discount
  
- PRICE_ALLOWANCE_CHARGE_INDICATOR: Type of adjustment
  * false (0) = Allowance/Discount (reduces price)
  * true (1) = Charge (increases price)

Example Calculation:
If PRICE_AMOUNT = 1000.00, INVOICED_QUANTITY = 5.0, and 
PRICE_ALLOWANCE_CHARGE_AMOUNT = 100.00 (discount), then:
INVOICED_LINE_EXTENSION_AMOUNT = (1000.00 × 5.0) - 100.00 = 4900.00
"""):
    print("✅ Successfully trained Pricing information")

else:
    print("❌ Failed to train Pricing information")

# ================================================================
# SECTION 12: DOCUMENT REFERENCES IN LINES
# ================================================================

if vanna_manager.train(documentation="""
INVOICE LINE DOCUMENT REFERENCES:
Links invoice lines to supporting documentation:

- INVOICE_LINE_DOCUMENT_REFERENCE_ID: Reference ID for related document
  Examples: 'DN-1000456', 'WO-2024-5678'
  
- INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE: Type of referenced document
  Common UNCL 1001 codes:
  * '130' = Delivery note / Goods receipt
  * '751' = Work order / Service order
  * '50' = Price/sales catalogue
  * '270' = Self billed invoice
  * '916' = Related document

These references help trace the invoice line back to delivery confirmations or work orders.
"""):
    print("✅ Successfully trained Document references")

else:
    print("❌ Failed to train Document references")

# ================================================================
# SECTION 13: PERIOD-BASED INVOICING
# ================================================================

if vanna_manager.train(documentation="""
INVOICING PERIODS (for recurring services and subscriptions):

Invoice Header Level:
- PERIOD_START_DATE: Start date of the overall invoiced period
- PERIOD_END_DATE: End date of the overall invoiced period
Format: YYYY-MM-DD (e.g., '2024-01-07' to '2024-01-31')

Invoice Line Level:
- INVOICE_PERIOD_START_DATE: Start date for this specific line item's service period
- INVOICE_PERIOD_END_DATE: End date for this specific line item's service period

USE CASES:
- Monthly service fees (e.g., January 1-31)
- Subscription services
- Leasing/rental fees
- Time-based service contracts
- Recurring maintenance agreements

For one-time purchases or immediate services, these fields may be NULL.
For recurring services, the period helps identify which month/timeframe is being billed.
"""):
    print("✅ Successfully trained Period-based invoicing")

else:
    print("❌ Failed to train Period-based invoicing")

# ================================================================
# SECTION 14: ETL AND DATA WAREHOUSE FIELDS
# ================================================================

if vanna_manager.train(documentation="""
ETL (EXTRACT, TRANSFORM, LOAD) METADATA:

- ETL_LOAD_TS: Timestamp when the record was loaded into the data warehouse
  Format: 'YYYY-MM-DD HH:MM:SS.mmm'
  Example: '2025-10-12 14:35:42.123'
  
This field tracks when the data was inserted/updated in the Nodinite database.
It does NOT represent when the invoice was created (use ISSUE_DATE for that).

Useful for:
- Auditing data loading processes
- Identifying recently loaded invoices
- Troubleshooting ETL issues
- Incremental data processing
"""):
    print("✅ Successfully trained ETL metadata")

else:
    print("❌ Failed to train ETL metadata")

# ================================================================
# SECTION 15: COMMON QUERY PATTERNS AND BEST PRACTICES
# ================================================================

if vanna_manager.train(documentation="""
COMMON QUERY PATTERNS FOR THIS DATABASE:

1. SUPPLIER ANALYSIS:
   - Total spending by supplier
   - Invoice count by supplier
   - Average invoice amount per supplier
   - Top suppliers by volume or amount

2. TIME-BASED ANALYSIS:
   - Invoices by month/quarter/year
   - Trends over time
   - Overdue invoices (DUE_DATE < current date)
   - Invoices issued in a specific period

3. PRODUCT/ITEM ANALYSIS:
   - Most frequently ordered items
   - Total spending by item category
   - Average price per item
   - Quantity trends for specific items

4. DEPARTMENT/COST CENTER ANALYSIS:
   - Spending by customer department
   - Invoice volume by care unit
   - Budget tracking by ACCOUNTING_COST

5. FINANCIAL REPORTING:
   - Total payable amounts
   - Tax summaries
   - Outstanding payments
   - Credit notes vs. standard invoices

6. INVOICE + LINE DETAILS:
   Always JOIN both tables to get complete information:
   FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] inv
   INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] line 
       ON inv.INVOICE_ID = line.INVOICE_ID
"""):
    print("✅ Successfully trained Common query patterns")

else:
    print("❌ Failed to train Common query patterns")

if vanna_manager.train(documentation="""
BEST PRACTICES FOR QUERY CONSTRUCTION:

1. ALWAYS use three-part table names: [Nodinite].[dbo].[TableName]

2. ALWAYS use column aliases for aggregates:
   ✓ SELECT COUNT(*) AS invoice_count
   ✗ SELECT COUNT(*)

3. Date filtering with NVARCHAR dates:
   ✓ WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'
   ✓ WHERE ISSUE_DATE BETWEEN '2025-01-01' AND '2025-12-31'

4. Currency filtering (when needed):
   WHERE DOCUMENT_CURRENCY_CODE = 'SEK'

5. Handling NULL values:
   Use ISNULL() or COALESCE() for NULL amounts
   Example: ISNULL(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT, 0)

6. String matching for names/descriptions:
   Use LIKE with wildcards: WHERE SUPPLIER_PARTY_NAME LIKE '%Instrumenta%'

7. Proper JOINs:
   Use INNER JOIN for required relationships
   Use LEFT JOIN when line items might not exist

8. Performance considerations:
   - Filter on indexed columns (INVOICE_ID, dates)
   - Use WHERE before GROUP BY
   - Limit result sets with TOP when appropriate
"""):
    print("✅ Successfully trained Best practices")

else:
    print("❌ Failed to train Best practices")

# ================================================================
# SECTION 16: GEOGRAPHIC AND REGIONAL CONTEXT
# ================================================================

if vanna_manager.train(documentation="""
GEOGRAPHIC CONTEXT - REGION VÄSTERBOTTEN:

Region Västerbotten is located in northern Sweden (Norrland region).
It covers a large geographic area with both urban and rural communities.

Major Cities and Towns:
- Umeå: Largest city, regional capital (population ~130,000)
- Skellefteå: Second largest city (population ~70,000)
- Lycksele: Central town
- Other municipalities: Storuman, Vilhelmina, Åsele, Dorotea, Malå, Norsjö, 
  Bjurholm, Vindeln, Robertsfors, Vännäs, Sorsele, Nordmaling

Healthcare Facilities:
- Norrlands universitetssjukhus (NUS) in Umeå - Main university hospital
- Skellefteå lasarett
- Lycksele lasarett
- Numerous vårdcentraler (healthcare centers) across the region
- Specialist clinics and facilities

Common Location References in Data:
- VLL = Västerbottens Läns Landsting (former organization name)
- NUS = Norrlands universitetssjukhus
- Vårdcentral = Healthcare center/clinic
- Lasarett = Hospital
"""):
    print("✅ Successfully trained Geographic context")

else:
    print("❌ Failed to train Geographic context")

# ================================================================
# SECTION 17: SWEDISH LANGUAGE TERMS
# ================================================================

if vanna_manager.train(documentation="""
COMMON SWEDISH TERMS IN THE DATABASE:

ORGANIZATIONAL:
- AB = Aktiebolag (Limited company)
- Säte = Seat/Registered office
- Organisationsnummer = Organization/company registration number

HEALTHCARE:
- Vårdcentral = Healthcare center/clinic
- Lasarett = Hospital
- Region = Regional authority
- REF = Reference number

FINANCIAL:
- Moms = VAT (Value Added Tax)
- Ert/Vår referens = Your/Our reference
- Betalningsvillkor = Payment terms
- Förfallodag = Due date

ADDRESSES:
- Gata/Gatan = Street (Styrmansgatan = Styrman Street)
- Box = P.O. Box
- Väg/Vägen = Road (Lasarettsvägen = Lasarett Road)

GENERAL:
- Tillhandahållen = Provided/Delivered to
- Ordernummer = Order number
- Artikelnummer = Article number
- Leverans = Delivery
"""):
    print("✅ Successfully trained Swedish Language Terms")

else:
    print("❌ Failed to train Swedish Language Terms")

print("✅ Documentation Training Complete!")
print("📚 Trained on:")
print("   - Critical technical requirements")
print("   - Database purpose and context")
print("   - Date formats and handling")
print("   - Supplier party information")
print("   - Customer party (Region Västerbotten) details")
print("   - Delivery information")
print("   - Currency and monetary amounts")
print("   - Swedish tax (VAT) information")
print("   - Invoice types and codes")
print("   - Reference fields")
print("   - Invoice line item structure")
print("   - Item descriptions and identifiers")
print("   - Pricing information")
print("   - Document references")
print("   - Period-based invoicing")
print("   - ETL metadata")
print("   - Common query patterns")
print("   - Best practices")
print("   - Geographic context")
print("   - Swedish language terms")


# ================================================================
# PHASE 3: COMMON SQL QUERIES TRAINING
# Example SQL patterns for invoice database queries
# ================================================================

# ================================================================
# SECTION 1: BASIC SUPPLIER ANALYSIS QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2025-01-01' 
  AND ISSUE_DATE < '2026-01-01'
  AND DOCUMENT_CURRENCY_CODE = 'SEK'
GROUP BY SUPPLIER_PARTY_NAME
ORDER BY total_amount DESC
"""):
    print("✅ Successfully trained: Supplier spending analysis by year")
else:
    print("❌ Failed to train: Supplier spending analysis by year")

if vanna_manager.train(sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID,
    COUNT(*) AS invoice_count,
    MIN(ISSUE_DATE) AS first_invoice_date,
    MAX(ISSUE_DATE) AS last_invoice_date
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
HAVING COUNT(*) > 10
ORDER BY invoice_count DESC
"""):
    print("✅ Successfully trained: Top suppliers by invoice volume")
else:
    print("❌ Failed to train: Top suppliers by invoice volume")

if vanna_manager.train(sql="""
SELECT 
    SUPPLIER_PARTY_CITY,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS supplier_count,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_COUNTRY = 'SE'
GROUP BY SUPPLIER_PARTY_CITY
ORDER BY total_amount DESC
"""):
    print("✅ Successfully trained: Supplier analysis by city")
else:
    print("❌ Failed to train: Supplier analysis by city")

# ================================================================
# SECTION 2: TIME-BASED ANALYSIS QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    LEFT(ISSUE_DATE, 7) AS year_month,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2024-01-01'
GROUP BY LEFT(ISSUE_DATE, 7)
ORDER BY year_month
"""):
    print("✅ Successfully trained: Monthly invoice trends")
else:
    print("❌ Failed to train: Monthly invoice trends")

if vanna_manager.train(sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    DUE_DATE,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    DATEDIFF(DAY, CAST(ISSUE_DATE AS DATE), CAST(DUE_DATE AS DATE)) AS payment_days
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DUE_DATE < CONVERT(NVARCHAR(10), GETDATE(), 23)
  AND LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT > 0
ORDER BY DUE_DATE
"""):
    print("✅ Successfully trained: Overdue invoices query")
else:
    print("❌ Failed to train: Overdue invoices query")

if vanna_manager.train(sql="""
SELECT 
    DATEPART(YEAR, CAST(ISSUE_DATE AS DATE)) AS invoice_year,
    DATEPART(QUARTER, CAST(ISSUE_DATE AS DATE)) AS invoice_quarter,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY DATEPART(YEAR, CAST(ISSUE_DATE AS DATE)), 
         DATEPART(QUARTER, CAST(ISSUE_DATE AS DATE))
ORDER BY invoice_year, invoice_quarter
"""):
    print("✅ Successfully trained: Quarterly invoice analysis")
else:
    print("❌ Failed to train: Quarterly invoice analysis")

# ================================================================
# SECTION 3: INVOICE AND LINE ITEM JOIN QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.ISSUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    il.INVOICE_LINE_ID,
    il.ITEM_NAME,
    il.ITEM_DESCRIPTION,
    il.INVOICED_QUANTITY,
    il.PRICE_AMOUNT,
    il.INVOICED_LINE_EXTENSION_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE i.ISSUE_DATE >= '2025-01-01'
ORDER BY i.ISSUE_DATE DESC, i.INVOICE_ID, il.INVOICE_LINE_ID
"""):
    print("✅ Successfully trained: Invoice with line items detail")
else:
    print("❌ Failed to train: Invoice with line items detail")

if vanna_manager.train(sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME,
    COUNT(DISTINCT i.INVOICE_ID) AS invoice_count,
    COUNT(il.INVOICE_LINE_ID) AS total_line_items,
    AVG(CAST(line_count AS FLOAT)) AS avg_lines_per_invoice
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
INNER JOIN (
    SELECT INVOICE_ID, COUNT(*) AS line_count
    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
    GROUP BY INVOICE_ID
) lc ON i.INVOICE_ID = lc.INVOICE_ID
GROUP BY i.SUPPLIER_PARTY_NAME
ORDER BY invoice_count DESC
"""):
    print("✅ Successfully trained: Supplier invoice complexity analysis")
else:
    print("❌ Failed to train: Supplier invoice complexity analysis")

if vanna_manager.train(sql="""
SELECT TOP 100
    il.ITEM_NAME,
    il.ITEM_DESCRIPTION,
    COUNT(*) AS order_count,
    SUM(il.INVOICED_QUANTITY) AS total_quantity,
    AVG(il.PRICE_AMOUNT) AS avg_unit_price,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
WHERE il.ITEM_NAME IS NOT NULL 
  AND il.ITEM_NAME != ''
GROUP BY il.ITEM_NAME, il.ITEM_DESCRIPTION
ORDER BY order_count DESC
"""):
    print("✅ Successfully trained: Most frequently ordered items")
else:
    print("❌ Failed to train: Most frequently ordered items")

# ================================================================
# SECTION 4: CUSTOMER/DEPARTMENT ANALYSIS QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    CUSTOMER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE CUSTOMER_PARTY_NAME LIKE '%Region Västerbotten%'
GROUP BY CUSTOMER_PARTY_NAME
ORDER BY total_spent DESC
"""):
    print("✅ Successfully trained: Department spending analysis")
else:
    print("❌ Failed to train: Department spending analysis")

if vanna_manager.train(sql="""
SELECT 
    ACCOUNTING_COST,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ACCOUNTING_COST IS NOT NULL
  AND ACCOUNTING_COST != ''
GROUP BY ACCOUNTING_COST
ORDER BY total_amount DESC
"""):
    print("✅ Successfully trained: Cost center spending summary")
else:
    print("❌ Failed to train: Cost center spending summary")

if vanna_manager.train(sql="""
SELECT 
    DELIVERY_LOCATION_CITY_NAME,
    COUNT(*) AS delivery_count,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS supplier_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DELIVERY_LOCATION_CITY_NAME IS NOT NULL
  AND DELIVERY_LOCATION_CITY_NAME != ''
GROUP BY DELIVERY_LOCATION_CITY_NAME
ORDER BY total_amount DESC
"""):
    print("✅ Successfully trained: Deliveries by city analysis")
else:
    print("❌ Failed to train: Deliveries by city analysis")

# ================================================================
# SECTION 5: TAX AND FINANCIAL ANALYSIS QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    LEFT(ISSUE_DATE, 7) AS year_month,
    SUM(LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) AS amount_excl_tax,
    SUM(TAX_AMOUNT) AS total_tax,
    SUM(LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT) AS amount_incl_tax,
    AVG(CASE 
        WHEN LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT > 0 
        THEN (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) * 100 
        ELSE 0 
    END) AS avg_tax_rate_percent
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2024-01-01'
GROUP BY LEFT(ISSUE_DATE, 7)
ORDER BY year_month
"""):
    print("✅ Successfully trained: Monthly tax summary")
else:
    print("❌ Failed to train: Monthly tax summary")

if vanna_manager.train(sql="""
SELECT 
    INVOICE_TYPE_CODE,
    CASE 
        WHEN INVOICE_TYPE_CODE = '380' THEN 'Standard Invoice'
        WHEN INVOICE_TYPE_CODE = '381' THEN 'Credit Note'
        WHEN INVOICE_TYPE_CODE = '383' THEN 'Prepayment Invoice'
        ELSE 'Other'
    END AS invoice_type_description,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY INVOICE_TYPE_CODE
ORDER BY invoice_count DESC
"""):
    print("✅ Successfully trained: Invoice type breakdown")
else:
    print("❌ Failed to train: Invoice type breakdown")

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.SUPPLIER_PARTY_NAME,
    i.LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT,
    i.LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT,
    i.LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT,
    i.TAX_AMOUNT,
    i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    (i.LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT 
     - ISNULL(i.LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT, 0)
     + ISNULL(i.LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT, 0)
     + i.TAX_AMOUNT) AS calculated_total
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT > 0
"""):
    print("✅ Successfully trained: Invoice amount breakdown and validation")
else:
    print("❌ Failed to train: Invoice amount breakdown and validation")

# ================================================================
# SECTION 6: ITEM-LEVEL ANALYSIS QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    il.ITEM_TAXCAT_PERCENT,
    COUNT(*) AS line_item_count,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount_excl_tax,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT * (il.ITEM_TAXCAT_PERCENT / 100)) AS calculated_tax
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
WHERE il.ITEM_TAXCAT_PERCENT IS NOT NULL
GROUP BY il.ITEM_TAXCAT_PERCENT
ORDER BY il.ITEM_TAXCAT_PERCENT
"""):
    print("✅ Successfully trained: Tax rate distribution on line items")
else:
    print("❌ Failed to train: Tax rate distribution on line items")

if vanna_manager.train(sql="""
SELECT 
    il.INVOICED_QUANTITY_UNIT_CODE,
    COUNT(*) AS line_count,
    SUM(il.INVOICED_QUANTITY) AS total_quantity,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
WHERE il.INVOICED_QUANTITY_UNIT_CODE IS NOT NULL
GROUP BY il.INVOICED_QUANTITY_UNIT_CODE
ORDER BY line_count DESC
"""):
    print("✅ Successfully trained: Unit of measure analysis")
else:
    print("❌ Failed to train: Unit of measure analysis")

if vanna_manager.train(sql="""
SELECT TOP 50
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    COUNT(*) AS purchase_count,
    SUM(il.INVOICED_QUANTITY) AS total_quantity,
    AVG(il.PRICE_AMOUNT) AS avg_unit_price,
    MIN(il.PRICE_AMOUNT) AS min_price,
    MAX(il.PRICE_AMOUNT) AS max_price
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE il.ITEM_NAME IS NOT NULL
  AND il.PRICE_AMOUNT > 0
GROUP BY i.SUPPLIER_PARTY_NAME, il.ITEM_NAME
HAVING COUNT(*) > 5
ORDER BY purchase_count DESC
"""):
    print("✅ Successfully trained: Frequently purchased items with price tracking")
else:
    print("❌ Failed to train: Frequently purchased items with price tracking")

# ================================================================
# SECTION 7: SEARCH AND FILTER QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.ISSUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    i.NOTE
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.SUPPLIER_PARTY_NAME LIKE '%Instrumenta%'
  OR i.SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID = '5560466137'
ORDER BY i.ISSUE_DATE DESC
"""):
    print("✅ Successfully trained: Search invoices by supplier name or ID")
else:
    print("❌ Failed to train: Search invoices by supplier name or ID")

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.ISSUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    il.ITEM_DESCRIPTION,
    il.INVOICED_QUANTITY,
    il.INVOICED_LINE_EXTENSION_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE il.ITEM_NAME LIKE '%kateter%'
   OR il.ITEM_DESCRIPTION LIKE '%kateter%'
ORDER BY i.ISSUE_DATE DESC
"""):
    print("✅ Successfully trained: Search items by name or description")
else:
    print("❌ Failed to train: Search items by name or description")

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.ISSUE_DATE,
    i.DUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    i.ORDER_REFERENCE_ID
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.ORDER_REFERENCE_ID = '7051810'
ORDER BY i.ISSUE_DATE DESC
"""):
    print("✅ Successfully trained: Find invoices by order reference")
else:
    print("❌ Failed to train: Find invoices by order reference")

# ================================================================
# SECTION 8: PERIOD-BASED AND RECURRING SERVICE QUERIES
# ================================================================

if vanna_manager.train(sql="""
SELECT 
    i.INVOICE_ID,
    i.SUPPLIER_PARTY_NAME,
    i.PERIOD_START_DATE,
    i.PERIOD_END_DATE,
    DATEDIFF(DAY, CAST(i.PERIOD_START_DATE AS DATE), CAST(i.PERIOD_END_DATE AS DATE)) AS period_days,
    i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.PERIOD_START_DATE IS NOT NULL
  AND i.PERIOD_END_DATE IS NOT NULL
ORDER BY i.PERIOD_START_DATE DESC
"""):
    print("✅ Successfully trained: Invoices with billing periods (recurring services)")
else:
    print("❌ Failed to train: Invoices with billing periods (recurring services)")

if vanna_manager.train(sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME,
    LEFT(i.PERIOD_START_DATE, 7) AS billing_month,
    COUNT(*) AS invoice_count,
    SUM(i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.PERIOD_START_DATE >= '2024-01-01'
  AND i.PERIOD_END_DATE IS NOT NULL
GROUP BY i.SUPPLIER_PARTY_NAME, LEFT(i.PERIOD_START_DATE, 7)
ORDER BY billing_month DESC, total_amount DESC
"""):
    print("✅ Successfully trained: Monthly recurring service costs by supplier")
else:
    print("❌ Failed to train: Monthly recurring service costs by supplier")

print("\n" + "="*80)
print("✅ PHASE 3: COMMON SQL QUERIES TRAINING COMPLETE!")
print("="*80)
print("📊 Training Summary:")
print("   - Section 1: Basic Supplier Analysis (3 queries)")
print("   - Section 2: Time-Based Analysis (3 queries)")
print("   - Section 3: Invoice & Line Item Joins (3 queries)")
print("   - Section 4: Customer/Department Analysis (3 queries)")
print("   - Section 5: Tax & Financial Analysis (3 queries)")
print("   - Section 6: Item-Level Analysis (3 queries)")
print("   - Section 7: Search & Filter (3 queries)")
print("   - Section 8: Period-Based & Recurring Services (2 queries)")
print("   TOTAL: 23 SQL query patterns trained")
print("="*80)


# ================================================================
# PHASE 4: QUESTION-SQL PAIRS TRAINING
# Most important phase - maps natural language to SQL queries
# ================================================================

# ================================================================
# SECTION 1: BASIC COUNTING AND TOTALING QUESTIONS
# ================================================================

if vanna_manager.train(
    question="How many invoices were issued in 2025?",
    sql="""
SELECT COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'
"""
):
    print("✅ Successfully trained: How many invoices in 2025")
else:
    print("❌ Failed to train: How many invoices in 2025")

if vanna_manager.train(
    question="What is the total amount payable across all invoices?",
    sql="""
SELECT 
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_payable,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DOCUMENT_CURRENCY_CODE = 'SEK'
"""
):
    print("✅ Successfully trained: Total amount payable")
else:
    print("❌ Failed to train: Total amount payable")

if vanna_manager.train(
    question="How many invoices do we have from each supplier?",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY SUPPLIER_PARTY_NAME
ORDER BY invoice_count DESC
"""
):
    print("✅ Successfully trained: Invoice count by supplier")
else:
    print("❌ Failed to train: Invoice count by supplier")

if vanna_manager.train(
    question="What is the average invoice amount?",
    sql="""
SELECT 
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice_amount,
    MIN(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS min_amount,
    MAX(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS max_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT > 0
"""
):
    print("✅ Successfully trained: Average invoice amount")
else:
    print("❌ Failed to train: Average invoice amount")

# ================================================================
# SECTION 2: SUPPLIER-FOCUSED QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Which suppliers have sent the most invoices?",
    sql="""
SELECT TOP 10
    SUPPLIER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY SUPPLIER_PARTY_NAME
ORDER BY invoice_count DESC
"""
):
    print("✅ Successfully trained: Top suppliers by invoice count")
else:
    print("❌ Failed to train: Top suppliers by invoice count")

if vanna_manager.train(
    question="Who are our top 10 suppliers by total spending?",
    sql="""
SELECT TOP 10
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
ORDER BY total_spent DESC
"""
):
    print("✅ Successfully trained: Top suppliers by spending")
else:
    print("❌ Failed to train: Top suppliers by spending")

if vanna_manager.train(
    question="Show me all invoices from Instrumenta",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    DUE_DATE,
    SUPPLIER_PARTY_NAME,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    DOCUMENT_CURRENCY_CODE
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_NAME LIKE '%Instrumenta%'
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Invoices from specific supplier")
else:
    print("❌ Failed to train: Invoices from specific supplier")

if vanna_manager.train(
    question="Which suppliers are based in Stockholm?",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_STREET_NAME,
    SUPPLIER_PARTY_POSTAL_ZONE,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_CITY = 'Stockholm'
GROUP BY SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_STREET_NAME, SUPPLIER_PARTY_POSTAL_ZONE
ORDER BY invoice_count DESC
"""
):
    print("✅ Successfully trained: Suppliers by city")
else:
    print("❌ Failed to train: Suppliers by city")

if vanna_manager.train(
    question="What is the average invoice amount per supplier?",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice_amount,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY SUPPLIER_PARTY_NAME
HAVING COUNT(*) >= 5
ORDER BY avg_invoice_amount DESC
"""
):
    print("✅ Successfully trained: Average invoice amount per supplier")
else:
    print("❌ Failed to train: Average invoice amount per supplier")

# ================================================================
# SECTION 3: TIME-BASED QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Show me invoices from last month",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= CONVERT(NVARCHAR(10), DATEADD(MONTH, -1, DATEADD(DAY, 1-DAY(GETDATE()), GETDATE())), 23)
  AND ISSUE_DATE < CONVERT(NVARCHAR(10), DATEADD(DAY, 1-DAY(GETDATE()), GETDATE()), 23)
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Invoices from last month")
else:
    print("❌ Failed to train: Invoices from last month")

if vanna_manager.train(
    question="What is our monthly spending trend for 2025?",
    sql="""
SELECT 
    LEFT(ISSUE_DATE, 7) AS year_month,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS monthly_total
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'
GROUP BY LEFT(ISSUE_DATE, 7)
ORDER BY year_month
"""
):
    print("✅ Successfully trained: Monthly spending trend")
else:
    print("❌ Failed to train: Monthly spending trend")

if vanna_manager.train(
    question="Which invoices are overdue?",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    DUE_DATE,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    DATEDIFF(DAY, CAST(DUE_DATE AS DATE), GETDATE()) AS days_overdue
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DUE_DATE < CONVERT(NVARCHAR(10), GETDATE(), 23)
  AND LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT > 0
ORDER BY DUE_DATE
"""
):
    print("✅ Successfully trained: Overdue invoices")
else:
    print("❌ Failed to train: Overdue invoices")

if vanna_manager.train(
    question="Show me invoices due this month",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    DUE_DATE,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DUE_DATE >= CONVERT(NVARCHAR(10), DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0), 23)
  AND DUE_DATE < CONVERT(NVARCHAR(10), DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) + 1, 0), 23)
ORDER BY DUE_DATE
"""
):
    print("✅ Successfully trained: Invoices due this month")
else:
    print("❌ Failed to train: Invoices due this month")

if vanna_manager.train(
    question="What was our quarterly spending in 2024?",
    sql="""
SELECT 
    DATEPART(QUARTER, CAST(ISSUE_DATE AS DATE)) AS quarter,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS quarterly_total
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2024-01-01' AND ISSUE_DATE < '2025-01-01'
GROUP BY DATEPART(QUARTER, CAST(ISSUE_DATE AS DATE))
ORDER BY quarter
"""
):
    print("✅ Successfully trained: Quarterly spending")
else:
    print("❌ Failed to train: Quarterly spending")

# ================================================================
# SECTION 4: ITEM AND PRODUCT QUESTIONS
# ================================================================

if vanna_manager.train(
    question="What are the most frequently ordered items?",
    sql="""
SELECT TOP 20
    ITEM_NAME,
    COUNT(*) AS order_count,
    SUM(INVOICED_QUANTITY) AS total_quantity,
    AVG(PRICE_AMOUNT) AS avg_price
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
WHERE ITEM_NAME IS NOT NULL AND ITEM_NAME != ''
GROUP BY ITEM_NAME
ORDER BY order_count DESC
"""
):
    print("✅ Successfully trained: Most frequently ordered items")
else:
    print("❌ Failed to train: Most frequently ordered items")

if vanna_manager.train(
    question="Show me all invoices that include catheters",
    sql="""
SELECT 
    i.INVOICE_ID,
    i.ISSUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    il.ITEM_DESCRIPTION,
    il.INVOICED_QUANTITY,
    il.INVOICED_LINE_EXTENSION_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE il.ITEM_NAME LIKE '%kateter%'
   OR il.ITEM_DESCRIPTION LIKE '%kateter%'
   OR il.ITEM_NAME LIKE '%catheter%'
   OR il.ITEM_DESCRIPTION LIKE '%catheter%'
ORDER BY i.ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Search for specific medical items")
else:
    print("❌ Failed to train: Search for specific medical items")

if vanna_manager.train(
    question="What items did we purchase from a specific supplier?",
    sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    COUNT(*) AS purchase_count,
    SUM(il.INVOICED_QUANTITY) AS total_quantity,
    AVG(il.PRICE_AMOUNT) AS avg_unit_price,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE i.SUPPLIER_PARTY_NAME LIKE '%Medtronic%'
  AND il.ITEM_NAME IS NOT NULL
GROUP BY i.SUPPLIER_PARTY_NAME, il.ITEM_NAME
ORDER BY total_spent DESC
"""
):
    print("✅ Successfully trained: Items from specific supplier")
else:
    print("❌ Failed to train: Items from specific supplier")

if vanna_manager.train(
    question="Which items have the highest total spending?",
    sql="""
SELECT TOP 50
    ITEM_NAME,
    COUNT(*) AS order_count,
    SUM(INVOICED_QUANTITY) AS total_quantity,
    SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_spent,
    AVG(PRICE_AMOUNT) AS avg_unit_price
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
WHERE ITEM_NAME IS NOT NULL
GROUP BY ITEM_NAME
ORDER BY total_spent DESC
"""
):
    print("✅ Successfully trained: Highest spending items")
else:
    print("❌ Failed to train: Highest spending items")

if vanna_manager.train(
    question="Show me price history for a specific item",
    sql="""
SELECT 
    i.ISSUE_DATE,
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    il.PRICE_AMOUNT,
    il.INVOICED_QUANTITY,
    il.INVOICED_LINE_EXTENSION_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE il.ITEM_NAME LIKE '%SAX METZENBAUM%'
ORDER BY i.ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Price history for item")
else:
    print("❌ Failed to train: Price history for item")

# ================================================================
# SECTION 5: DEPARTMENT AND COST CENTER QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Which Region Västerbotten departments have the highest spending?",
    sql="""
SELECT TOP 10
    CUSTOMER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE CUSTOMER_PARTY_NAME LIKE '%Region Västerbotten%'
GROUP BY CUSTOMER_PARTY_NAME
ORDER BY total_spent DESC
"""
):
    print("✅ Successfully trained: Department spending ranking")
else:
    print("❌ Failed to train: Department spending ranking")

if vanna_manager.train(
    question="Show me spending by cost center",
    sql="""
SELECT 
    ACCOUNTING_COST,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ACCOUNTING_COST IS NOT NULL AND ACCOUNTING_COST != ''
GROUP BY ACCOUNTING_COST
ORDER BY total_amount DESC
"""
):
    print("✅ Successfully trained: Spending by cost center")
else:
    print("❌ Failed to train: Spending by cost center")

if vanna_manager.train(
    question="Which cities receive the most deliveries?",
    sql="""
SELECT 
    DELIVERY_LOCATION_CITY_NAME,
    COUNT(*) AS delivery_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_value
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DELIVERY_LOCATION_CITY_NAME IS NOT NULL
  AND DELIVERY_LOCATION_CITY_NAME != ''
GROUP BY DELIVERY_LOCATION_CITY_NAME
ORDER BY delivery_count DESC
"""
):
    print("✅ Successfully trained: Deliveries by city")
else:
    print("❌ Failed to train: Deliveries by city")

if vanna_manager.train(
    question="Show me all invoices for a specific department",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    CUSTOMER_PARTY_NAME,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE CUSTOMER_PARTY_NAME LIKE '%REF 1050103%'
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Invoices for specific department")
else:
    print("❌ Failed to train: Invoices for specific department")

# ================================================================
# SECTION 6: FINANCIAL AND TAX QUESTIONS
# ================================================================

if vanna_manager.train(
    question="What is the total tax amount paid?",
    sql="""
SELECT 
    SUM(TAX_AMOUNT) AS total_tax_paid,
    SUM(LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) AS total_excl_tax,
    SUM(LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT) AS total_incl_tax
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE TAX_AMOUNT IS NOT NULL
"""
):
    print("✅ Successfully trained: Total tax paid")
else:
    print("❌ Failed to train: Total tax paid")

if vanna_manager.train(
    question="Show me monthly tax totals for 2025",
    sql="""
SELECT 
    LEFT(ISSUE_DATE, 7) AS year_month,
    SUM(TAX_AMOUNT) AS monthly_tax,
    SUM(LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) AS amount_excl_tax,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2025-01-01' AND ISSUE_DATE < '2026-01-01'
GROUP BY LEFT(ISSUE_DATE, 7)
ORDER BY year_month
"""
):
    print("✅ Successfully trained: Monthly tax totals")
else:
    print("❌ Failed to train: Monthly tax totals")

if vanna_manager.train(
    question="How many credit notes do we have?",
    sql="""
SELECT 
    INVOICE_TYPE_CODE,
    COUNT(*) AS count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE INVOICE_TYPE_CODE = '381'
GROUP BY INVOICE_TYPE_CODE
"""
):
    print("✅ Successfully trained: Credit notes count")
else:
    print("❌ Failed to train: Credit notes count")

if vanna_manager.train(
    question="Show me breakdown of invoice types",
    sql="""
SELECT 
    INVOICE_TYPE_CODE,
    CASE 
        WHEN INVOICE_TYPE_CODE = '380' THEN 'Standard Invoice'
        WHEN INVOICE_TYPE_CODE = '381' THEN 'Credit Note'
        WHEN INVOICE_TYPE_CODE = '383' THEN 'Prepayment Invoice'
        ELSE 'Other'
    END AS invoice_type,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
GROUP BY INVOICE_TYPE_CODE
ORDER BY invoice_count DESC
"""
):
    print("✅ Successfully trained: Invoice type breakdown")
else:
    print("❌ Failed to train: Invoice type breakdown")

if vanna_manager.train(
    question="What is the average tax rate on our purchases?",
    sql="""
SELECT 
    AVG(CASE 
        WHEN LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT > 0 
        THEN (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) * 100 
        ELSE 0 
    END) AS avg_tax_rate_percent,
    MIN(CASE 
        WHEN LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT > 0 
        THEN (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) * 100 
        ELSE 0 
    END) AS min_tax_rate,
    MAX(CASE 
        WHEN LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT > 0 
        THEN (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT) * 100 
        ELSE 0 
    END) AS max_tax_rate
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT > 0
"""
):
    print("✅ Successfully trained: Average tax rate calculation")
else:
    print("❌ Failed to train: Average tax rate calculation")

# ================================================================
# SECTION 7: INVOICE DETAIL QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Show me details for a specific invoice",
    sql="""
SELECT 
    i.*
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
WHERE i.INVOICE_ID = 'INV-2025-12345'
"""
):
    print("✅ Successfully trained: Specific invoice details")
else:
    print("❌ Failed to train: Specific invoice details")

if vanna_manager.train(
    question="Show me all line items for a specific invoice",
    sql="""
SELECT 
    il.INVOICE_LINE_ID,
    il.ITEM_NAME,
    il.ITEM_DESCRIPTION,
    il.INVOICED_QUANTITY,
    il.INVOICED_QUANTITY_UNIT_CODE,
    il.PRICE_AMOUNT,
    il.INVOICED_LINE_EXTENSION_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
WHERE il.INVOICE_ID = 'INV-2025-12345'
ORDER BY il.INVOICE_LINE_ID
"""
):
    print("✅ Successfully trained: Line items for invoice")
else:
    print("❌ Failed to train: Line items for invoice")

if vanna_manager.train(
    question="Show me invoices with more than 10 line items",
    sql="""
SELECT 
    i.INVOICE_ID,
    i.SUPPLIER_PARTY_NAME,
    i.ISSUE_DATE,
    COUNT(il.INVOICE_LINE_ID) AS line_count,
    i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
GROUP BY i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, i.ISSUE_DATE, i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
HAVING COUNT(il.INVOICE_LINE_ID) > 10
ORDER BY line_count DESC
"""
):
    print("✅ Successfully trained: Complex invoices with many lines")
else:
    print("❌ Failed to train: Complex invoices with many lines")

if vanna_manager.train(
    question="What is the average number of line items per invoice?",
    sql="""
SELECT 
    AVG(CAST(line_count AS FLOAT)) AS avg_lines_per_invoice,
    MIN(line_count) AS min_lines,
    MAX(line_count) AS max_lines
FROM (
    SELECT 
        INVOICE_ID,
        COUNT(*) AS line_count
    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
    GROUP BY INVOICE_ID
) AS line_counts
"""
):
    print("✅ Successfully trained: Average lines per invoice")
else:
    print("❌ Failed to train: Average lines per invoice")

# ================================================================
# SECTION 8: SEARCH AND LOOKUP QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Find invoices by order reference number",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    ORDER_REFERENCE_ID,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ORDER_REFERENCE_ID = '12345'
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Find by order reference")
else:
    print("❌ Failed to train: Find by order reference")

if vanna_manager.train(
    question="Search for invoices with specific keywords in notes",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    NOTE,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE NOTE LIKE '%urgent%'
   OR NOTE LIKE '%special%'
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Search notes for keywords")
else:
    print("❌ Failed to train: Search notes for keywords")

if vanna_manager.train(
    question="Show me invoices above a certain amount",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    DOCUMENT_CURRENCY_CODE
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT > 100000
ORDER BY LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DESC
"""
):
    print("✅ Successfully trained: High-value invoices")
else:
    print("❌ Failed to train: High-value invoices")

if vanna_manager.train(
    question="Find invoices by supplier organization number",
    sql="""
SELECT 
    INVOICE_ID,
    ISSUE_DATE,
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID = '5560466137'
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Find by organization number")
else:
    print("❌ Failed to train: Find by organization number")

# ================================================================
# SECTION 9: COMPARISON AND ANALYSIS QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Compare spending between two suppliers",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_NAME IN ('Instrumenta AB', 'Medtronic AB')
GROUP BY SUPPLIER_PARTY_NAME
"""
):
    print("✅ Successfully trained: Compare suppliers")
else:
    print("❌ Failed to train: Compare suppliers")

if vanna_manager.train(
    question="Compare this year's spending to last year",
    sql="""
SELECT 
    CASE 
        WHEN ISSUE_DATE >= '2025-01-01' THEN '2025'
        WHEN ISSUE_DATE >= '2024-01-01' AND ISSUE_DATE < '2025-01-01' THEN '2024'
    END AS year,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2024-01-01'
GROUP BY CASE 
    WHEN ISSUE_DATE >= '2025-01-01' THEN '2025'
    WHEN ISSUE_DATE >= '2024-01-01' AND ISSUE_DATE < '2025-01-01' THEN '2024'
END
ORDER BY year
"""
):
    print("✅ Successfully trained: Year-over-year comparison")
else:
    print("❌ Failed to train: Year-over-year comparison")


if vanna_manager.train(
    question="Which supplier has the fastest payment terms?",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    AVG(DATEDIFF(DAY, CAST(ISSUE_DATE AS DATE), CAST(DUE_DATE AS DATE))) AS avg_payment_days,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DUE_DATE IS NOT NULL AND ISSUE_DATE IS NOT NULL
GROUP BY SUPPLIER_PARTY_NAME
HAVING COUNT(*) >= 5
ORDER BY avg_payment_days
"""
):
    print("✅ Successfully trained: Payment terms analysis")
else:
    print("❌ Failed to train: Payment terms analysis")

if vanna_manager.train(
    question="Show me spending trends by month for each supplier",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    LEFT(ISSUE_DATE, 7) AS year_month,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS monthly_total
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= '2024-01-01'
GROUP BY SUPPLIER_PARTY_NAME, LEFT(ISSUE_DATE, 7)
ORDER BY SUPPLIER_PARTY_NAME, year_month
"""
):
    print("✅ Successfully trained: Supplier spending trends by month")
else:
    print("❌ Failed to train: Supplier spending trends by month")

# ================================================================
# SECTION 10: RECURRING SERVICES AND SUBSCRIPTION QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Which invoices are for recurring services?",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    PERIOD_START_DATE,
    PERIOD_END_DATE,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE PERIOD_START_DATE IS NOT NULL
  AND PERIOD_END_DATE IS NOT NULL
ORDER BY PERIOD_START_DATE DESC
"""
):
    print("✅ Successfully trained: Recurring service invoices")
else:
    print("❌ Failed to train: Recurring service invoices")

if vanna_manager.train(
    question="What are our monthly recurring costs?",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    LEFT(PERIOD_START_DATE, 7) AS billing_month,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS monthly_cost
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE PERIOD_START_DATE IS NOT NULL
  AND PERIOD_END_DATE IS NOT NULL
  AND PERIOD_START_DATE >= '2024-01-01'
GROUP BY SUPPLIER_PARTY_NAME, LEFT(PERIOD_START_DATE, 7)
ORDER BY billing_month DESC, monthly_cost DESC
"""
):
    print("✅ Successfully trained: Monthly recurring costs breakdown")
else:
    print("❌ Failed to train: Monthly recurring costs breakdown")

if vanna_manager.train(
    question="Show me subscription services by supplier",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    COUNT(DISTINCT INVOICE_ID) AS subscription_invoice_count,
    MIN(PERIOD_START_DATE) AS first_billing_period,
    MAX(PERIOD_END_DATE) AS last_billing_period,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_subscription_cost
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE PERIOD_START_DATE IS NOT NULL
GROUP BY SUPPLIER_PARTY_NAME
ORDER BY total_subscription_cost DESC
"""
):
    print("✅ Successfully trained: Subscription services by supplier")
else:
    print("❌ Failed to train: Subscription services by supplier")

# ================================================================
# SECTION 11: ADVANCED FILTERING QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Show me invoices from suppliers in specific cities",
    sql="""
SELECT 
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_CITY,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_CITY IN ('Stockholm', 'Göteborg', 'Malmö')
GROUP BY SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_CITY
ORDER BY total_amount DESC
"""
):
    print("✅ Successfully trained: Filter by supplier cities")
else:
    print("❌ Failed to train: Filter by supplier cities")

if vanna_manager.train(
    question="Find invoices with discounts or allowances",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT > 0
ORDER BY LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DESC
"""
):
    print("✅ Successfully trained: Invoices with discounts")
else:
    print("❌ Failed to train: Invoices with discounts")

if vanna_manager.train(
    question="Show me invoices with additional charges",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT > 0
ORDER BY LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DESC
"""
):
    print("✅ Successfully trained: Invoices with additional charges")
else:
    print("❌ Failed to train: Invoices with additional charges")

if vanna_manager.train(
    question="Find invoices that reference a contract",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    CONTRACT_DOCUMENT_REFERENCE_ID,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE CONTRACT_DOCUMENT_REFERENCE_ID IS NOT NULL
  AND CONTRACT_DOCUMENT_REFERENCE_ID != ''
ORDER BY ISSUE_DATE DESC
"""
):
    print("✅ Successfully trained: Invoices with contract references")
else:
    print("❌ Failed to train: Invoices with contract references")

if vanna_manager.train(
    question="Show me invoices with prepaid amounts",
    sql="""
SELECT 
    INVOICE_ID,
    SUPPLIER_PARTY_NAME,
    ISSUE_DATE,
    LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT,
    LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT,
    (LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT / NULLIF(LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT, 0)) * 100 AS prepaid_percentage
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT > 0
ORDER BY LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DESC
"""
):
    print("✅ Successfully trained: Invoices with prepayments")
else:
    print("❌ Failed to train: Invoices with prepayments")

# ================================================================
# SECTION 12: UNIT OF MEASURE AND QUANTITY QUESTIONS
# ================================================================

if vanna_manager.train(
    question="What are the most common units of measure used?",
    sql="""
SELECT 
    INVOICED_QUANTITY_UNIT_CODE,
    COUNT(*) AS usage_count,
    SUM(INVOICED_QUANTITY) AS total_quantity,
    AVG(PRICE_AMOUNT) AS avg_price
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
WHERE INVOICED_QUANTITY_UNIT_CODE IS NOT NULL
GROUP BY INVOICED_QUANTITY_UNIT_CODE
ORDER BY usage_count DESC
"""
):
    print("✅ Successfully trained: Common units of measure")
else:
    print("❌ Failed to train: Common units of measure")

if vanna_manager.train(
    question="Show me items purchased by the hour",
    sql="""
SELECT 
    i.SUPPLIER_PARTY_NAME,
    il.ITEM_NAME,
    COUNT(*) AS purchase_count,
    SUM(il.INVOICED_QUANTITY) AS total_hours,
    AVG(il.PRICE_AMOUNT) AS avg_hourly_rate,
    SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_cost
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il 
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE il.INVOICED_QUANTITY_UNIT_CODE = 'HUR'
GROUP BY i.SUPPLIER_PARTY_NAME, il.ITEM_NAME
ORDER BY total_cost DESC
"""
):
    print("✅ Successfully trained: Hourly-based purchases")
else:
    print("❌ Failed to train: Hourly-based purchases")

if vanna_manager.train(
    question="What is the total quantity ordered for each item?",
    sql="""
SELECT 
    ITEM_NAME,
    INVOICED_QUANTITY_UNIT_CODE,
    COUNT(*) AS order_count,
    SUM(INVOICED_QUANTITY) AS total_quantity,
    SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_spent
FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]
WHERE ITEM_NAME IS NOT NULL
GROUP BY ITEM_NAME, INVOICED_QUANTITY_UNIT_CODE
ORDER BY total_quantity DESC
"""
):
    print("✅ Successfully trained: Total quantities by item")
else:
    print("❌ Failed to train: Total quantities by item")

# ================================================================
# SECTION 13: CONTACT AND COMMUNICATION QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Show me supplier contact information",
    sql="""
SELECT DISTINCT
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_CONTACT_NAME,
    SUPPLIER_PARTY_CONTACT_EMAIL,
    SUPPLIER_PARTY_CONTACT_PHONE,
    SUPPLIER_PARTY_STREET_NAME,
    SUPPLIER_PARTY_CITY,
    SUPPLIER_PARTY_POSTAL_ZONE
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_NAME IS NOT NULL
ORDER BY SUPPLIER_PARTY_NAME
"""
):
    print("✅ Successfully trained: Supplier contact information")
else:
    print("❌ Failed to train: Supplier contact information")

if vanna_manager.train(
    question="Which suppliers have email contacts?",
    sql="""
SELECT DISTINCT
    SUPPLIER_PARTY_NAME,
    SUPPLIER_PARTY_CONTACT_EMAIL,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE SUPPLIER_PARTY_CONTACT_EMAIL IS NOT NULL
  AND SUPPLIER_PARTY_CONTACT_EMAIL != ''
GROUP BY SUPPLIER_PARTY_NAME, SUPPLIER_PARTY_CONTACT_EMAIL
ORDER BY invoice_count DESC
"""
):
    print("✅ Successfully trained: Suppliers with email contacts")
else:
    print("❌ Failed to train: Suppliers with email contacts")

if vanna_manager.train(
    question="Show me customer contact details for our departments",
    sql="""
SELECT DISTINCT
    CUSTOMER_PARTY_NAME,
    CUSTOMER_PARTY_CONTACT_NAME,
    CUSTOMER_PARTY_CONTACT_EMAIL,
    CUSTOMER_PARTY_CONTACT_PHONE
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE CUSTOMER_PARTY_NAME LIKE '%Region Västerbotten%'
  AND CUSTOMER_PARTY_CONTACT_EMAIL IS NOT NULL
ORDER BY CUSTOMER_PARTY_NAME
"""
):
    print("✅ Successfully trained: Department contact details")
else:
    print("❌ Failed to train: Department contact details")

# ================================================================
# SECTION 14: DELIVERY AND LOGISTICS QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Which cities receive the most medical deliveries?",
    sql="""
SELECT 
    DELIVERY_LOCATION_CITY_NAME,
    COUNT(*) AS delivery_count,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS supplier_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_value
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DELIVERY_LOCATION_CITY_NAME IS NOT NULL
GROUP BY DELIVERY_LOCATION_CITY_NAME
ORDER BY delivery_count DESC
"""
):
    print("✅ Successfully trained: Deliveries by city")
else:
    print("❌ Failed to train: Deliveries by city")

if vanna_manager.train(
    question="Show me delivery locations in Umeå",
    sql="""
SELECT 
    DELIVERY_LOCATION_STREET_NAME,
    DELIVERY_LOCATION_ADDRESS_LINE,
    DELIVERY_PARTY_NAME,
    COUNT(*) AS delivery_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_value
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE DELIVERY_LOCATION_CITY_NAME = 'Umeå'
GROUP BY DELIVERY_LOCATION_STREET_NAME, DELIVERY_LOCATION_ADDRESS_LINE, DELIVERY_PARTY_NAME
ORDER BY delivery_count DESC
"""
):
    print("✅ Successfully trained: Delivery locations in specific city")
else:
    print("❌ Failed to train: Delivery locations in specific city")

if vanna_manager.train(
    question="What is the average delivery time from invoice date?",
    sql="""
SELECT 
    AVG(DATEDIFF(DAY, CAST(ISSUE_DATE AS DATE), CAST(ACTUAL_DELIVERY_DATE AS DATE))) AS avg_delivery_days,
    MIN(DATEDIFF(DAY, CAST(ISSUE_DATE AS DATE), CAST(ACTUAL_DELIVERY_DATE AS DATE))) AS min_delivery_days,
    MAX(DATEDIFF(DAY, CAST(ISSUE_DATE AS DATE), CAST(ACTUAL_DELIVERY_DATE AS DATE))) AS max_delivery_days,
    COUNT(*) AS invoice_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ACTUAL_DELIVERY_DATE IS NOT NULL
  AND ISSUE_DATE IS NOT NULL
  AND CAST(ACTUAL_DELIVERY_DATE AS DATE) >= CAST(ISSUE_DATE AS DATE)
"""
):
    print("✅ Successfully trained: Average delivery time")
else:
    print("❌ Failed to train: Average delivery time")

# ================================================================
# SECTION 15: SUMMARY AND DASHBOARD QUESTIONS
# ================================================================

if vanna_manager.train(
    question="Give me a summary of invoice activity",
    sql="""
SELECT 
    COUNT(*) AS total_invoices,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS unique_suppliers,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice_amount,
    MIN(ISSUE_DATE) AS earliest_invoice,
    MAX(ISSUE_DATE) AS latest_invoice
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
"""
):
    print("✅ Successfully trained: Overall invoice summary")
else:
    print("❌ Failed to train: Overall invoice summary")

if vanna_manager.train(
    question="Show me key metrics for this month",
    sql="""
SELECT 
    COUNT(*) AS invoice_count,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS supplier_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount,
    SUM(TAX_AMOUNT) AS total_tax
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= CONVERT(NVARCHAR(10), DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0), 23)
  AND ISSUE_DATE < CONVERT(NVARCHAR(10), DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) + 1, 0), 23)
"""
):
    print("✅ Successfully trained: Current month metrics")
else:
    print("❌ Failed to train: Current month metrics")

if vanna_manager.train(
    question="What are the top 5 expense categories by cost center?",
    sql="""
SELECT TOP 5
    ACCOUNTING_COST,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent,
    AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_invoice
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ACCOUNTING_COST IS NOT NULL
GROUP BY ACCOUNTING_COST
ORDER BY total_spent DESC
"""
):
    print("✅ Successfully trained: Top expense categories")
else:
    print("❌ Failed to train: Top expense categories")

if vanna_manager.train(
    question="Show me year-to-date spending summary",
    sql="""
SELECT 
    DATEPART(YEAR, CAST(ISSUE_DATE AS DATE)) AS year,
    COUNT(*) AS invoice_count,
    SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent,
    SUM(TAX_AMOUNT) AS total_tax,
    COUNT(DISTINCT SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID) AS supplier_count
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
WHERE ISSUE_DATE >= CONVERT(NVARCHAR(10), DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0), 23)
GROUP BY DATEPART(YEAR, CAST(ISSUE_DATE AS DATE))
"""
):
    print("✅ Successfully trained: Year-to-date summary")
else:
    print("❌ Failed to train: Year-to-date summary")

print("\n" + "="*80)
print("✅ PHASE 4: QUESTION-SQL PAIRS TRAINING COMPLETE!")
print("="*80)
print("📊 Training Summary:")
print("   - Section 1: Basic Counting & Totaling (4 Q&A pairs)")
print("   - Section 2: Supplier-Focused Questions (5 Q&A pairs)")
print("   - Section 3: Time-Based Questions (5 Q&A pairs)")
print("   - Section 4: Item & Product Questions (5 Q&A pairs)")
print("   - Section 5: Department & Cost Center (4 Q&A pairs)")
print("   - Section 6: Financial & Tax Questions (5 Q&A pairs)")
print("   - Section 7: Invoice Detail Questions (4 Q&A pairs)")
print("   - Section 8: Search & Lookup Questions (4 Q&A pairs)")
print("   - Section 9: Comparison & Analysis (3 Q&A pairs)")
print("   - Section 10: Recurring Services (3 Q&A pairs)")
print("   - Section 11: Advanced Filtering (5 Q&A pairs)")
print("   - Section 12: Unit of Measure & Quantity (3 Q&A pairs)")
print("   - Section 13: Contact & Communication (3 Q&A pairs)")
print("   - Section 14: Delivery & Logistics (3 Q&A pairs)")
print("   - Section 15: Summary & Dashboard (4 Q&A pairs)")
print("   TOTAL: 60 Question-SQL pairs trained")
print("="*80)
print("\n🎉 ALL TRAINING PHASES COMPLETE!")
print("="*80)
print("✅ Phase 1: DDL Training")
print("✅ Phase 2: Documentation Training (17 sections)")
print("✅ Phase 3: Common SQL Queries (23 queries)")
print("✅ Phase 4: Question-SQL Pairs (60 pairs)")
print("="*80)
print("\n🚀 Your Vanna model is now fully trained and ready to use!")
print("💡 Next steps:")
print("   1. Test with: vanna_manager.ask('Your question here')")
print("   2. Review results and add more training as needed")
print("   3. Use auto_train=True to continuously improve")
print("="*80)


# print("Training with Invoice documentation (first 5 rows)...")
# if vanna_manager.train(documentation=invoice_doc):
#     print("✅ Successfully trained Invoice documentation")

# else:
#     print("❌ Failed to train Invoice documentation")

# print("Training with Invoice_Line documentation (first 5 rows)...")
# if vanna_manager.train(documentation=invoice_line_doc):
#     print("✅ Successfully trained Invoice_Line documentation")

# else:
#     print("❌ Failed to train Invoice_Line documentation")

# print("Training with synonym handling instructions...")
# if vanna_manager.train(documentation=synonym_instructions):
#     print("✅ Successfully trained synonym handling instructions")

# else:
#     print("❌ Failed to train synonym handling instructions")

# # Train with question-SQL pairs
# print("Training with question-SQL pairs...")

# successful_pairs = 0
# total_pairs = len(training_pairs)

# for i, pair in enumerate(training_pairs, 1):
#     question = pair["question"]
#     sql = pair["sql"]
#     print(f"Training pair {i}/{total_pairs}: {question[:50]}...")
    
#     try:
#         if vanna_manager.train(question=question, sql=sql):
#             successful_pairs += 1
#             print(f"✅ Successfully trained pair {i}")
#         else:
#             print(f"❌ Failed to train pair {i}")
#     except Exception as e:
#         print(f"❌ Error training pair {i}: {e}")

# print(f"📊 Question-SQL training completed: {successful_pairs}/{total_pairs} pairs successful")

# Added a strict syntax command.
vanna_manager.train(documentation="#STRICT SYNTAX RULE: Your SQL should be on a single line with no line breaks. It should follow this exact syntax ```sql <command> ```")

# Add comprehensive plural/singular handling instructions
vanna_manager.train(documentation=get_comprehensive_search_instructions())

# Auto-train on startup if enabled
if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
    print("\n🔄 Auto-training enabled. Additional training can be done manually with vanna_manager.train(...).")

else:
    print("\n🔄 Auto-training is disabled. Use vanna_manager.train(...) to manually train the model.")


async def greet(name: Optional[str] = None) -> str:
    """Provide a friendly greeting to the user."""
    if not name or name.strip() == "":
        name = None

    hour = datetime.now().hour

    if 5 <= hour < 12:
        time_greeting = "Good morning"
    elif 12 <= hour < 17:
        time_greeting = "Good afternoon"
    elif 17 <= hour < 21:
        time_greeting = "Good evening"
    else:
        time_greeting = "Good evening"

    if name:
        response = f"[RESPONSE]: {time_greeting} {name}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, and payment tracking. What can I do for you?"

    else:
        response = f"[RESPONSE]: {time_greeting}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, and payment tracking. How can I assist you today?"

    return f"{response}\n\n[Success]"


# --------------------
# ENDPOINTS
# --------------------

@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name"""
    try:
        name = request.name if request.name and request.name.strip() else None
        message = await greet(name)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import re

def format_vanna_sql(vanna_sql):
    # Strip each line and join with a space
    sql_no_newlines = " ".join(line.strip() for line in vanna_sql.split('\n'))
    # Optionally collapse multiple spaces into one
    return re.sub(r'\s+', ' ', sql_no_newlines).strip()





def singular_to_plural(word):
    """
    Convert a single word from singular to plural using English pluralization rules.
    
    Args:
        word (str): The word to convert
        
    Returns:
        str: The plural form of the word
    """
    if len(word) < 2:
        return word
    
    word_lower = word.lower()
    
    # Handle irregular plurals
    irregular_singulars = {
        'child': 'children',
        'foot': 'feet',
        'goose': 'geese',
        'man': 'men',
        'woman': 'women',
        'tooth': 'teeth',
        'mouse': 'mice',
        'person': 'people',
        'ox': 'oxen',
        'deer': 'deer',
        'sheep': 'sheep',
        'fish': 'fish',
        'moose': 'moose',
        'series': 'series',
        'species': 'species',
        'datum': 'data',
        'medium': 'media',
        'criterion': 'criteria',
        'phenomenon': 'phenomena',
        'bacterium': 'bacteria',
        'alumnus': 'alumni',
        'fungus': 'fungi',
        'nucleus': 'nuclei',
        'cactus': 'cacti',
        'focus': 'foci',
        'radius': 'radii',
        'analysis': 'analyses',
        'basis': 'bases',
        'diagnosis': 'diagnoses',
        'oasis': 'oases',
        'thesis': 'theses',
        'crisis': 'crises',
        'axis': 'axes',
        'matrix': 'matrices',
        'vertex': 'vertices',
        'index': 'indices',
        'appendix': 'appendices'
    }
    
    if word_lower in irregular_singulars:
        # Preserve original case pattern
        plural = irregular_singulars[word_lower]
        if word.isupper():
            return plural.upper()
        elif word.istitle():
            return plural.capitalize()
        else:
            return plural
    
    # Handle regular singular to plural patterns
    
    # Words ending in 'y' preceded by a consonant -> 'ies' (e.g., company -> companies, battery -> batteries)
    if word_lower.endswith('y') and len(word) > 2 and word[-2].lower() not in 'aeiou':
        return word[:-1] + 'ies'
    
    # Words ending in 'f' or 'fe' -> 'ves' (e.g., knife -> knives, shelf -> shelves)
    if word_lower.endswith('f'):
        return word[:-1] + 'ves'
    elif word_lower.endswith('fe'):
        return word[:-2] + 'ves'
    
    # Words ending in 's', 'ss', 'sh', 'ch', 'x', 'z' -> add 'es'
    if word_lower.endswith(('s', 'ss', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    
    # Words ending in 'o' preceded by a consonant -> 'oes' (e.g., tomato -> tomatoes, hero -> heroes)
    if word_lower.endswith('o') and len(word) > 1 and word[-2].lower() not in 'aeiou':
        return word + 'es'
    
@app.post("/query_sql_database")
async def query_sql_database_endpoint(request: QueryDatabaseRequest):
    """Query invoice database with natural language"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Query: {request.query}")
            # print(f"[MCP DEBUG] Keywords: {request.keywords}")

        # TODO: QUERY HARDCODED TO [].
        #  BECAUSE KEYWORD HITTING FAILS FOR LARGE DBs
        keywords = []

        # provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        # TODO: WE CAN SET VANNA UP HERE!
        # sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)
        # sql_query = await query_engine.generate_sql(request.query, keywords, provider)
        # TODO: GENERATE SQL QUERY WITH VANNA

        # Enhance query to search for both singular and plural forms
        enhanced_query = normalize_for_comprehensive_search(request.query)
        
        # Add additional instructions for better SQL generation
        enhanced_query = enhanced_query + "\nMake sure you use Like and Lower Keywords to compare the values if needed, to get better results."
        
        sql_query = vanna_manager.generate_sql(enhanced_query)

        print(f"Vanna Generated SQL: {sql_query}")

        if sql_query:
            sql_query = format_vanna_sql(vanna_sql=sql_query)

        if not sql_query:
            return {"success": False, "error": "Failed to generate SQL", "original_query": request.query}

        results = await query_engine.execute_query(sql_query)

        return {
            "success": True,
            "sql_query": f"```sql {sql_query} ```",
            "results": results,
            "original_query": request.query,
            "record_count": len(results)
        }

    except Exception as e:
        return {"success": False, "error": str(e), "original_query": request.query}


@app.post("/query_sql_database_stream")
async def query_sql_database_stream_endpoint(request: QueryDatabaseRequest):
    """Stream invoice query results"""

    async def generate_response():
        try:
            provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

            sql_query = ""
            async for sql_update in query_engine.generate_sql_stream(request.query, request.keywords, provider):
                yield json.dumps(sql_update) + "\n"
                if sql_update.get("status") == "sql_complete":
                    sql_query = sql_update.get("sql_query", "")

            if not sql_query:
                yield json.dumps({"success": False, "error": "Failed to generate SQL"}) + "\n"
                return

            yield json.dumps({"status": "executing_query", "sql_query": f"```sql {sql_query} ```"}) + "\n"

            async for db_result in query_engine.execute_query_stream(sql_query):
                db_result["original_query"] = request.query
                yield json.dumps(db_result) + "\n"

        except Exception as e:
            yield json.dumps({"success": False, "error": str(e)}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = "unknown"
        db_type = "remote" if USE_REMOTE else "local"

        if USE_REMOTE:
            # Test remote SQL Server connectivity
            try:
                # Verify tables exist
                # result = await query_engine.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
                # invoice_count = result[0]['invoice_count'] if result else 0

                # lines_result =  await query_engine.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice_Line]")
                # line_count = lines_result[0]['line_count'] if lines_result else 0
                
                db_health = "connected"

                return {
                    "status": "healthy",
                    "service": "Invoice MCP Server",
                    "timestamp": datetime.now().isoformat(),
                    "database_type": db_type,
                    "database_connection": db_health,
                    "database_info": {
                        "server": config.SQL_SERVER_HOST,
                        "database": config.SQL_SERVER_DATABASE,
                        "auth": "Windows" if config.SQL_SERVER_USE_WINDOWS_AUTH else "SQL Server"
                    },
                    # "invoice_count": invoice_count,
                    # "line_count": line_count,
                    # "tables_initialized": invoice_count > 0 and line_count > 0,
                    "llm_providers": ["openai", "ollama", "mistral"],
                    "endpoints": {
                        "greet": "/greet",
                        "query_sql_database": "/query_sql_database",
                        "query_sql_database_stream": "/query_sql_database_stream",
                        "health": "/health"
                    }
                }
            except Exception as e:
                return {
                    "status": "error",
                    "service": "Invoice MCP Server",
                    "timestamp": datetime.now().isoformat(),
                    "database_type": db_type,
                    "database_connection": "error",
                    "database_info": {
                        "server": config.SQL_SERVER_HOST,
                        "database": config.SQL_SERVER_DATABASE
                    },
                    "error": str(e)
                }
        else:
            # Test local SQLite API connectivity
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{DATABASE_SERVER_URL}/health", timeout=5) as response:
                        if response.status == 200:
                            db_health = "connected"
                            health_data = await response.json()
                            return {
                                "status": "healthy",
                                "service": "Invoice MCP Server",
                                "timestamp": datetime.now().isoformat(),
                                "database_type": db_type,
                                "database_connection": db_health,
                                "database_server": DATABASE_SERVER_URL,
                                "invoice_count": health_data.get("invoice_count", 0),
                                "line_count": health_data.get("line_count", 0),
                                "tables_initialized": health_data.get("tables_initialized", False),
                                "llm_providers": ["openai", "ollama", "mistral"],
                                "endpoints": {
                                    "greet": "/greet",
                                    "query_sql_database": "/query_sql_database",
                                    "query_sql_database_stream": "/query_sql_database_stream",
                                    "health": "/health"
                                }
                            }
                        else:
                            db_health = "error"
            except:
                db_health = "disconnected"

            return {
                "status": "error" if db_health != "connected" else "healthy",
                "service": "Invoice MCP Server",
                "timestamp": datetime.now().isoformat(),
                "database_type": db_type,
                "database_connection": db_health,
                "database_server": DATABASE_SERVER_URL,
                "error": "Cannot connect to database server" if db_health != "connected" else None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list():
    """MCP Protocol: List available tools"""
    tools = [
        MCPTool(
            name="greet",
            description="TRIGGER: hello, hi, good morning, good afternoon, good evening, introduce yourself, who are you, start conversation | ACTION: Welcome user with time-based greeting | RETURNS: Personalized greeting with invoice management capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's name", "default": ""}
                },
                "required": ["name"]
            }
        ),
        MCPTool(
            name="query_sql_database",
            description="TRIGGER: SQL, SQL database, query database, invoice database, invoice queries, supplier information, customer data, payment tracking, financial reports, invoice amounts, due dates, supplier analysis, customer analysis, invoice lines, item details, tax information, payment terms | ACTION: Query invoice SQL database with AI-generated SQL from natural language | RETURNS: Structured invoice data with the generated SQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """Natural language question about the invoice database - leverages context awareness from previous chat
                                        **Example:**
                - User asks: "How many invoices does ABC Corp have?"
                - User then asks: "What's the IT department for that company?"
                - You should understand "that company" = "ABC Corp" and query accordingly"""
                    }
                },
                "required": ["query"]
            }
        )
    ]
    return MCPToolsListResponse(tools=tools)


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def mcp_tools_call(request: MCPToolCallRequest):
    """MCP Protocol: Call a specific tool"""
    try:
        tool_name = request.name
        arguments = request.arguments

        if DEBUG:
            print(f"[MCP] Calling tool: {tool_name} with args: {arguments}")

        if tool_name == "greet":
            raw_name = arguments.get("name", "")
            clean_name = raw_name if raw_name and raw_name.strip() else None
            greet_request = GreetRequest(name=clean_name)
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "query_sql_database":
            query_request = QueryDatabaseRequest(**arguments)
            result = await query_sql_database_endpoint(query_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        else:
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=f"Unknown tool: {tool_name}")],
                isError=True
            )

    except Exception as e:
        if DEBUG:
            print(f"[MCP] Error calling tool {request.name}: {e}")
        return MCPToolCallResponse(
            content=[MCPContent(type="text", text=f"Error calling tool {request.name}: {str(e)}")],
            isError=True
        )


@app.post("/")
async def mcp_streamable_http_endpoint(request: Request):
    """Streamable HTTP MCP protocol endpoint for MCPO compatibility"""
    try:
        body = await request.json()
        method = body.get("method")
        request_id = body.get("id")

        if DEBUG:
            print(f"[STREAMABLE HTTP] Received method: {method}")
            print(f"[STREAMABLE HTTP] Body: {body}")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "Invoice SQL MCP Server",
                        "version": "1.0.0",
                        "description": "MCP server with LLM-powered SQL generation for invoice management databases"
                    }
                }
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Initialize response: {response}")
            return response

        elif method == "notifications/initialized":
            if DEBUG:
                print(f"[STREAMABLE HTTP] Received initialized notification - connection established")
            return Response(status_code=204)

        elif method == "tools/list":
            tools_response = await mcp_tools_list()
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [tool.model_dump() for tool in tools_response.tools]}
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Tools list response: {response}")
            return response

        elif method == "tools/call":
            params = body.get("params", {})
            call_request = MCPToolCallRequest(**params)
            result = await mcp_tools_call(call_request)
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result.model_dump()
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Tools call response: {response}")
            return response

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            }

        else:
            if DEBUG:
                print(f"[STREAMABLE HTTP] Unknown method: {method}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    except Exception as e:
        if DEBUG:
            print(f"[STREAMABLE HTTP] Error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.post("/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint"""
    body = await request.json()
    print(f"[DEBUG] {json.dumps(body, indent=2)}")
    return {"received": body, "timestamp": datetime.now().isoformat()}


@app.get("/info")
async def server_info():
    """Server information endpoint"""
    return {
        "service": "Invoice SQL MCP Server",
        "version": "1.0.0",
        "description": "MCP Tools Server with LLM-powered natural language to SQL conversion for invoice database queries",
        "protocols": ["REST API", "MCP (Model Context Protocol)"],
        "database_mode": DATABASE_CHOICE,
        "database_info": DATABASE_SERVER_URL,

        "mcp_endpoints": {
            "streamable_http": "/",
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call"
        },

        "rest_endpoints": [
            "/greet",
            "/query_sql_database",
            "/query_sql_database_stream",
            "/health",
            "/info",
            "/debug"
        ],

        "features": [
            "LLM SQL Generation (Ollama, OpenAI, Mistral)",
            "Context-Aware Keyword Search",
            "Streaming Query Execution",
            "Invoice Domain Optimization",
            "MCP Protocol Support",
            "Async Database Operations",
            "Local SQLite + Remote SQL Server Support"
        ],

        "database": {
            "mode": DATABASE_CHOICE,
            "tables": query_engine.all_tables,
            "table_count": len(query_engine.all_tables)
        },

        "tools": [
            "greet",
            "query_sql_database"
        ],

        "docs": "/docs",
        "mcp_compatible": True,
        "debug_mode": DEBUG
    }


if __name__ == "__main__":
    print(f"Starting Invoice MCP Server on port 8009...")
    print(f"Database mode: {DATABASE_CHOICE}")
    print(f"Database info: {DATABASE_SERVER_URL}")

    if DEBUG:
        print("[DEBUG] Debug mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=8009)
