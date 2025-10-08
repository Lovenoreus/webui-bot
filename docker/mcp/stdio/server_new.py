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
    database_url=DATABASE_SERVER_URL if not USE_REMOTE else None,
    tables=INVOICE_TABLES,
    table_variations=INVOICE_TABLE_VARIATIONS,
    system_prompt=INVOICE_SYSTEM_PROMPT,
    debug=DEBUG,
    use_remote=USE_REMOTE
)


# Initialize Vanna Manager at module level
print("Initializing Vanna Manager...")
# Custom path
vanna_manager = VannaModelManager(chroma_path=config.CHROMA_DB_PATH)

# Initialize on module load
print("Initializing Vanna...")
vanna_manager.initialize_vanna()
print(f"✅ Vanna initialized with provider: {vanna_manager.current_provider}")

# Train with your schema
print("📚 Training Vanna with schema...")

# Define DDLs and documentation based on database mode
# if not USE_REMOTE:
#     print(f"Using remote database schema: [Nodinite].[ods]")
#
#     invoice_ddl = """
#     CREATE TABLE [Nodinite].[ods].[Invoice] (
#         INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
#         ISSUE_DATE NVARCHAR(10) NOT NULL,
#         SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
#         SUPPLIER_PARTY_NAME NVARCHAR(255),
#         SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),
#         SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
#         SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),
#         SUPPLIER_PARTY_CITY NVARCHAR(100),
#         SUPPLIER_PARTY_COUNTRY NVARCHAR(2),
#         SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
#         SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
#         SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100),
#         SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
#         SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
#         SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
#         SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
#         CUSTOMER_PARTY_ID NVARCHAR(50),
#         CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),
#         CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
#         CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
#         CUSTOMER_PARTY_NAME NVARCHAR(255),
#         CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),
#         CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),
#         CUSTOMER_PARTY_COUNTRY NVARCHAR(2),
#         CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
#         CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),
#         CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
#         CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
#         CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
#         DUE_DATE NVARCHAR(10),
#         DOCUMENT_CURRENCY_CODE NVARCHAR(3),
#         DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),
#         DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
#         DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),
#         DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),
#         DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
#         DELIVERY_LOCATION_COUNTRY NVARCHAR(2),
#         DELIVERY_PARTY_NAME NVARCHAR(255),
#         ACTUAL_DELIVERY_DATE NVARCHAR(10),
#         TAX_AMOUNT_CURRENCY NVARCHAR(3),
#         TAX_AMOUNT DECIMAL(18,2),
#         PERIOD_START_DATE NVARCHAR(10),
#         PERIOD_END_DATE NVARCHAR(10),
#         LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,2),
#         LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
#         LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,2),
#         BUYER_REFERENCE NVARCHAR(100),
#         PROJECT_REFERENCE_ID NVARCHAR(100),
#         INVOICE_TYPE_CODE NVARCHAR(10),
#         NOTE NVARCHAR(MAX),
#         TAX_POINT_DATE NVARCHAR(10),
#         ACCOUNTING_COST NVARCHAR(100),
#         ORDER_REFERENCE_ID NVARCHAR(100),
#         ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
#         PAYMENT_TERMS_NOTE NVARCHAR(MAX),
#         BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
#         BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(10),
#         CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),
#         DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
#         ETL_LOAD_TS NVARCHAR(30)
#     );
#     """
#
#     invoice_line_ddl = """
#     CREATE TABLE [Nodinite].[ods].[Invoice_Line] (
#         INVOICE_ID NVARCHAR(50) NOT NULL,
#         ISSUE_DATE NVARCHAR(10) NOT NULL,
#         SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
#         INVOICE_LINE_ID NVARCHAR(50) NOT NULL,
#         ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(50),
#         ACCOUNTING_COST NVARCHAR(100),
#         INVOICED_QUANTITY DECIMAL(18,4),
#         INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),
#         INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,2),
#         INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3),
#         INVOICE_PERIOD_START_DATE NVARCHAR(10),
#         INVOICE_PERIOD_END_DATE NVARCHAR(10),
#         INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),
#         INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(10),
#         INVOICE_LINE_NOTE NVARCHAR(MAX),
#         ITEM_DESCRIPTION NVARCHAR(MAX),
#         ITEM_NAME NVARCHAR(255),
#         ITEM_TAXCAT_ID NVARCHAR(10),
#         ITEM_TAXCAT_PERCENT DECIMAL(5,2),
#         ITEM_BUYERS_ID NVARCHAR(100),
#         ITEM_SELLERS_ITEM_ID NVARCHAR(100),
#         ITEM_STANDARD_ITEM_ID NVARCHAR(100),
#         ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),
#         ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(50),
#         PRICE_AMOUNT DECIMAL(18,2),
#         PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),
#         PRICE_BASE_QUANTITY DECIMAL(18,4),
#         PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),
#         PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,2),
#         PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),
#         ETL_LOAD_TS NVARCHAR(30),
#         PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
#         FOREIGN KEY (INVOICE_ID) REFERENCES [Nodinite].[ods].[Invoice](INVOICE_ID)
#     );
#     """
#
#     invoice_doc = """
#     # Invoice Table Documentation
#
#     ## Business Context
#     The Invoice table stores header-level information for invoices received from suppliers.
#     Each invoice has a unique INVOICE_ID and contains supplier, customer, delivery, and financial information.
#
#     ## Key Fields Explanation
#     - INVOICE_ID: Unique identifier for each invoice (e.g., '0000470081')
#     - ISSUE_DATE: Date the invoice was issued (format: YYYY-MM-DD)
#     - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the supplier (e.g., '5560466137' for Abbott Scandinavia)
#     - SUPPLIER_PARTY_NAME: Name of the supplier company
#     - CUSTOMER_PARTY_NAME: Name of the customer (often Swedish regions like 'Region Västerbotten')
#     - CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the customer
#     - DOCUMENT_CURRENCY_CODE: Currency used (typically 'SEK' for Swedish Krona)
#     - TAX_AMOUNT: Total VAT/tax amount on the invoice
#     - LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT: Total amount excluding tax
#     - LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT: Total excluding tax (can differ from line extension due to charges/allowances)
#     - LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT: Total including tax
#     - LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT: Final amount to be paid
#     - BUYER_REFERENCE: Customer's internal reference/cost center
#     - ORDER_REFERENCE_ID: Reference to purchase order
#     - INVOICE_TYPE_CODE: Type of invoice (e.g., '380' = standard commercial invoice)
#     - PAYMENT_TERMS_NOTE: Payment terms description (e.g., '30 | Dröjsmålsränta %' means 30 days payment terms with late payment interest)
#     - DELIVERY_LOCATION_*: Fields describing where goods/services were delivered
#
#     ## Sample Data (Real Examples from Abbott Scandinavia)
#     Invoice 0000470081 issued 2023-05-16:
#     - Supplier: Abbott Scandinavia (5560466137), Solna
#     - Customer: Region Västerbotten (2321000222)
#     - Delivery: Universitetssjukhuset, UMEÅ
#     - Currency: SEK
#     - Line Extension Amount: 173,333.59 SEK
#     - Tax Amount: 43,333.42 SEK
#     - Total Payable: 216,667.01 SEK
#     - Payment Terms: 30 days
#     - Note: 'CPR Statistik April 2023'
#
#     Invoice 0000470109 issued 2023-05-22:
#     - Supplier: Abbott Scandinavia (5560466137)
#     - Customer: Region Västerbotten (2321000222)
#     - Delivery: Godsmottagning, UMEÅ
#     - Total Payable: 13,750.00 SEK
#
#     ## Common Suppliers
#     - Abbott Scandinavia (5560466137): Medical supplies and equipment
#     - Visma Draftit AB: Software and IT services
#     - JA Hotel Karlskrona: Hospitality services
#
#     ## Common Customers
#     - Region Västerbotten (2321000222)
#     - Region Skåne
#     - Stockholms Stad
#     - Västra Götaland
#     """
#
#     invoice_line_doc = """
#     # Invoice_Line Table Documentation
#
#     ## Business Context
#     The Invoice_Line table stores individual line items for each invoice.
#     Each line represents a specific product or service being invoiced.
#     One invoice (INVOICE_ID) can have multiple lines (INVOICE_LINE_ID).
#
#     ## Key Fields Explanation
#     - INVOICE_ID: Links to the parent Invoice table
#     - INVOICE_LINE_ID: Unique line number within the invoice (e.g., '1', '2', '10', '11')
#     - ITEM_NAME: Description of the product/service (e.g., 'PP ALNTY I HAVAB IGM RGT' = medical test reagent)
#     - INVOICED_QUANTITY: Quantity of items (e.g., 26.000)
#     - INVOICED_QUANTITY_UNIT_CODE: Unit of measurement (e.g., 'EA' = Each/piece)
#     - PRICE_AMOUNT: Unit price per item
#     - INVOICED_LINE_EXTENSION_AMOUNT: Total line amount (quantity × price, before tax)
#     - ITEM_TAXCAT_ID: Tax category identifier (typically 'S' for standard VAT)
#     - ITEM_TAXCAT_PERCENT: VAT percentage (typically 25.000 for Sweden)
#     - ITEM_SELLERS_ITEM_ID: Supplier's product code (e.g., '2R2897', '7P8797')
#     - INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID: Currency (typically 'SEK')
#
#     ## Sample Data (Real Examples from Abbott Scandinavia Invoice 0000470081)
#
#     Line 1:
#     - Item: PP ALNTY I HAVAB IGM RGT (medical test)
#     - Quantity: 26 EA
#     - Unit Price: 66.29 SEK
#     - Line Total: 1,723.54 SEK
#     - Tax: 25%
#     - Seller Item ID: 2R2897
#
#     Line 10:
#     - Item: PP ALNTY I ANTI-HBC RGT
#     - Quantity: 443 EA
#     - Unit Price: 19.51 SEK
#     - Line Total: 8,642.93 SEK
#     - Tax: 25%
#     - Seller Item ID: 7P8797
#
#     Line 11:
#     - Item: PP ALNTY I ANTI HBS RGT
#     - Quantity: 170 EA
#     - Unit Price: 35.10 SEK
#     - Line Total: 5,967.00 SEK
#
#     Line 12:
#     - Item: PP ALNTY I ANTI-HCV RGT
#     - Quantity: 1,726 EA
#     - Unit Price: 31.37 SEK
#     - Line Total: 54,144.62 SEK
#
#     Line 13:
#     - Item: PP ALNTY I HIV COMBO RGT
#     - Quantity: 1,928 EA
#     - Unit Price: 18.49 SEK
#     - Line Total: 35,648.72 SEK
#
#     ## Common Product Types
#     - Medical test reagents (PP ALNTY I series)
#     - Hotel accommodation services
#     - Software licenses
#     - Training services
#     - Medical supplies
#
#     ## Important Notes
#     - Invoice line totals should sum to the invoice header LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
#     - Tax is calculated per line: INVOICED_LINE_EXTENSION_AMOUNT × (ITEM_TAXCAT_PERCENT / 100)
#     - NULL values in ORDER_LINE_REFERENCE_LINE_ID, ACCOUNTING_COST are common
#     """
#
# else:
print("Using local database schema")

invoice_ddl = """
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

# Train Vanna with DDL and documentation
print("\n📚 Training Vanna with DDL and documentation...")
print("Training with Invoice DDL...")

if vanna_manager.train(ddl=invoice_ddl):
    print("✅ Successfully trained Invoice DDL")

else:
    print("❌ Failed to train Invoice DDL")

print("Training with Invoice_Line DDL...")
if vanna_manager.train(ddl=invoice_line_ddl):
    print("✅ Successfully trained Invoice_Line DDL")

else:
    print("❌ Failed to train Invoice_Line DDL")

print("Training with Invoice documentation (first 5 rows)...")
if vanna_manager.train(documentation=invoice_doc):
    print("✅ Successfully trained Invoice documentation")

else:
    print("❌ Failed to train Invoice documentation")

print("Training with Invoice_Line documentation (first 5 rows)...")
if vanna_manager.train(documentation=invoice_line_doc):
    print("✅ Successfully trained Invoice_Line documentation")

else:
    print("❌ Failed to train Invoice_Line documentation")

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


@app.post("/query_sql_database")
async def query_sql_database_endpoint(request: QueryDatabaseRequest):
    """Query invoice database with natural language"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Query: {request.query}")
            print(f"[MCP DEBUG] Keywords: {request.keywords}")

        # TODO: QUERY HARDCODED TO [].
        #  BECAUSE KEYWORD HITTING FAILS FOR LARGE DBs
        keywords = []

        # provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        # TODO: WE CAN SET VANNA UP HERE!
        # sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)
        # sql_query = await query_engine.generate_sql(request.query, keywords, provider)
        # TODO: GENERATE SQL QUERY WITH VANNA
        sql_query = vanna_manager.generate_sql(request.query)

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
                test_query = "SELECT COUNT(*) as invoice_count FROM Invoice"
                result = await query_engine.execute_query(test_query)
                invoice_count = result[0]['invoice_count'] if result else 0

                lines_result = await query_engine.execute_query("SELECT COUNT(*) as line_count FROM Invoice_Line")
                line_count = lines_result[0]['line_count'] if lines_result else 0

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
                    "invoice_count": invoice_count,
                    "line_count": line_count,
                    "tables_initialized": invoice_count > 0 and line_count > 0,
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
                        "description": "Natural language question about the invoice database (invoices, suppliers, customers, payments, line items)"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extract keywords from the query"
                    }
                },
                "required": ["query", "keywords"]
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
