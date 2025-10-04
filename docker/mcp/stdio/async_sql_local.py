# -------------------- Built-in Libraries --------------------
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

# -------------------- External Libraries --------------------
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import aiosqlite
import pymssql
from pydantic import BaseModel

# -------------------- Load environment variables --------------------
load_dotenv(find_dotenv())


# Load configuration first
def load_config():
    """Load configuration from config.json if it exists"""
    config_path = "config.json"

    default_config = {
        "mcp": {
            "database_choice": "local",
            "database_path": "sqlite_invoices_full.db",
            "docker_database_path": "/app/database_data/sqlite_invoices_full.db",
            "connection_pool_size": 10
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f'Returned config: {config}')
                return config
        except Exception as e:
            print(f'Error loading config: {e}')
            return default_config

    print('Configuration file not found')
    return default_config


# Load config at module level
CONFIG = load_config()
MCP_CONFIG = CONFIG.get('mcp', {})

# Database configuration
DATABASE_CHOICE = MCP_CONFIG.get('database_choice', 'local')
USE_REMOTE = DATABASE_CHOICE == "remote"

SQL_SERVER_HOST = os.getenv("SQL_SERVER_HOST", "localhost\\SQLEXPRESS")
SQL_SERVER_DATABASE = os.getenv("SQL_SERVER_DATABASE", "InvoiceDB")
SQL_SERVER_DRIVER = os.getenv("SQL_SERVER_DRIVER", "ODBC Driver 17 for SQL Server")
SQL_SERVER_USE_WINDOWS_AUTH = os.getenv("SQL_SERVER_USE_WINDOWS_AUTH", "true").lower() == "true"

SQL_SERVER_USERNAME = os.getenv("SQL_SERVER_USERNAME", None)
SQL_SERVER_PASSWORD = os.getenv("SQL_SERVER_PASSWORD", None)

print(f"Database choice: {DATABASE_CHOICE}")
print(f"Using remote database: {USE_REMOTE}")
print(f"SQL_SERVER_HOST: {SQL_SERVER_HOST}")


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely traverse a nested dict using a list of keys.
    Returns default if any key is missing.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_database_path():
    """Determine the correct database path based on environment"""
    # Check if we're running in Docker by looking for docker-specific paths
    if os.path.exists("/app/database_data"):
        return MCP_CONFIG.get("docker_database_path", "/app/database_data/sqlite_invoices_full.db")
    else:
        return MCP_CONFIG.get("database_path", "sqlite_invoices_full.db")


class AsyncDatabaseServer:
    def __init__(self, db_path: str = None, pool_size: int = None):
        self.use_remote = USE_REMOTE

        if self.use_remote:
            # MSSQL configuration
            self.server = SQL_SERVER_HOST
            self.database = SQL_SERVER_DATABASE
            self.pool_size = pool_size or MCP_CONFIG.get("connection_pool_size", 10)

            # Create thread pool for blocking pymssql operations
            self.executor = ThreadPoolExecutor(max_workers=self.pool_size)

            # Handle authentication
            if SQL_SERVER_USE_WINDOWS_AUTH:
                self.username = None
                self.password = None
                print(f"Using MSSQL Server: {self.server}")
                print(f"Database: {self.database}")
                print(f"Auth: Windows Authentication")
            else:
                self.username = SQL_SERVER_USERNAME
                self.password = SQL_SERVER_PASSWORD
                print(f"Using MSSQL Server: {self.server}")
                print(f"Database: {self.database}")
                print(f"Auth: SQL Server Authentication")
        else:
            # SQLite configuration
            self.db_path = db_path or get_database_path()
            self.pool_size = pool_size or MCP_CONFIG.get("connection_pool_size", 10)
            self.executor = None
            print(f"Using SQLite")
            print(f"Database path: {self.db_path}")
            print(f"Database file exists: {os.path.exists(self.db_path)}")

        self.pool = asyncio.Queue(maxsize=self.pool_size)
        self.pool_initialized = False
        print(f"Connection pool size: {self.pool_size}")

    def is_database_initialized(self):
        """Check if database has been initialized using a flag file"""
        if self.use_remote:
            return True  # Remote DB should already exist
        flag_file = f"{self.db_path}.initialized"
        return os.path.exists(flag_file)

    def mark_database_initialized(self):
        """Mark database as initialized"""
        if self.use_remote:
            return  # No need for flag file with remote DB
        flag_file = f"{self.db_path}.initialized"
        with open(flag_file, 'w') as f:
            f.write(str(datetime.now()))

    def _create_connection_sync(self):
        """Synchronous connection creation for pymssql"""
        if SQL_SERVER_USE_WINDOWS_AUTH:
            # Windows Authentication
            return pymssql.connect(
                server=self.server,
                database=self.database,
                as_dict=True
            )
        else:
            # SQL Server Authentication
            return pymssql.connect(
                server=self.server,
                user=self.username,
                password=self.password,
                database=self.database,
                as_dict=True
            )

    async def _create_connection(self):
        """Create a new database connection with optimal settings"""
        if self.use_remote:
            # Create pymssql connection in executor
            loop = asyncio.get_running_loop()
            conn = await loop.run_in_executor(
                self.executor,
                self._create_connection_sync
            )
            return conn
        else:
            # Create SQLite connection
            db = await aiosqlite.connect(self.db_path)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            await db.execute("PRAGMA temp_store=memory")
            await db.execute("PRAGMA foreign_keys=ON")
            db.row_factory = aiosqlite.Row
            return db

    async def initialize_pool(self):
        """Initialize the connection pool - ALL CONNECTIONS IN PARALLEL!"""
        if self.pool_initialized:
            return

        print(f"Initializing connection pool with {self.pool_size} connections in parallel...")
        start_time = asyncio.get_event_loop().time()

        # Create ALL connections simultaneously
        connections = await asyncio.gather(
            *[self._create_connection() for _ in range(self.pool_size)]
        )

        # Add all connections to the pool
        for conn in connections:
            await self.pool.put(conn)

        self.pool_initialized = True
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"Connection pool initialized in {elapsed * 1000:.2f}ms!")
        print(f"   ({self.pool_size} connections ready)")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self.pool_initialized:
            await self.initialize_pool()

        conn = await self.pool.get()
        try:
            yield conn
        finally:
            await self.pool.put(conn)

    def _close_connection_sync(self, conn):
        """Synchronous connection close for pymssql"""
        conn.close()

    async def close_pool(self):
        """Close all connections in the pool - also in parallel!"""
        if not self.pool_initialized:
            return

        print("Closing connection pool...")
        connections = []

        # Collect all connections from pool
        while not self.pool.empty():
            connections.append(await self.pool.get())

        # Close all connections in parallel
        if self.use_remote:
            loop = asyncio.get_running_loop()
            await asyncio.gather(*[
                loop.run_in_executor(self.executor, self._close_connection_sync, conn)
                for conn in connections
            ])
            # Shutdown the executor
            self.executor.shutdown(wait=True)
        else:
            await asyncio.gather(*[conn.close() for conn in connections])

        self.pool_initialized = False
        print(f"Closed {len(connections)} connections")

    async def initialize_database(self):
        """Initialize database with invoice tables and sample data"""
        if self.use_remote:
            print(f"Using remote MSSQL database - skipping initialization")
            return

        print(f"Checking invoice database at: {self.db_path}")

        # Fast flag-based check
        if self.is_database_initialized():
            print("Database already initialized. Ready to use!")
            return

        print("Initializing new database...")

        # Ensure directory exists for docker path
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                print(f"Creating database directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)
                print(f"Directory created successfully. Exists: {os.path.exists(db_dir)}")
        except Exception as e:
            print(f"Error creating directory: {e}")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            await db.execute("PRAGMA temp_store=memory")
            await db.execute("PRAGMA foreign_keys=ON")

            print("Creating database tables...")

            # Drop any existing tables for clean slate
            drop_tables = [
                "DROP TABLE IF EXISTS Invoice_Line",
                "DROP TABLE IF EXISTS Invoice"
            ]

            for drop_sql in drop_tables:
                await db.execute(drop_sql)

            # Create Invoice table
            create_invoice_table = """
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
            )
            """

            # Create Invoice_Line table
            create_invoice_line_table = """
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
            )
            """

            await db.execute(create_invoice_table)
            await db.execute(create_invoice_line_table)

            # Create indexes for better performance
            indexes = [
                "CREATE INDEX idx_invoice_issue_date ON Invoice(ISSUE_DATE)",
                "CREATE INDEX idx_invoice_supplier ON Invoice(SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID)",
                "CREATE INDEX idx_invoice_customer ON Invoice(CUSTOMER_PARTY_ID)",
                "CREATE INDEX idx_invoice_line_invoice_id ON Invoice_Line(INVOICE_ID)",
                "CREATE INDEX idx_invoice_line_issue_date ON Invoice_Line(ISSUE_DATE)",
                "CREATE INDEX idx_invoice_line_supplier ON Invoice_Line(SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID)"
            ]

            for index_sql in indexes:
                await db.execute(index_sql)

            # Insert sample data
            print("Starting to insert invoice sample data...")
            await self.insert_sample_data(db)
            print("Invoice sample data insertion completed!")

            await db.commit()
            print("Invoice database initialized successfully!")

            # Verify data was inserted
            cursor = await db.execute("SELECT COUNT(*) FROM Invoice")
            count = await cursor.fetchone()
            print(f"Total invoices inserted: {count[0] if count else 0}")

            # Mark as initialized
            self.mark_database_initialized()
            print("Database initialization complete!")

    async def insert_sample_data(self, db):
        """Insert comprehensive invoice sample data"""
        # Generate realistic Swedish companies and data
        suppliers = [
            ('5592985237', 'JA Hotel Karlskrona', 'Borgmästaregatan 13', '37115', 'Karlskrona', 'info@jahotel.se',
             '045555560'),
            ('5565783957', 'Visma Draftit AB', 'Styrmansgatan 2', '21118', 'Malmo', 'support@vismadraftit.se',
             '+46101992350'),
            ('5560466137', 'Abbott Scandinavia', 'Hemvärnsgatan 9', '171 29', 'Solna', 'contact@abbott.se',
             '+46850123400'),
            ('5591234567', 'Nordic IT Solutions AB', 'Sveavägen 45', '111 34', 'Stockholm', 'info@nordicit.se',
             '+46812345678'),
            ('5598765432', 'Malmö Tech Services', 'Drottninggatan 22', '211 15', 'Malmö', 'support@malmotech.se',
             '+46401234567'),
            ('5587654321', 'Göteborg Consulting Group', 'Avenyn 12', '411 36', 'Göteborg', 'hello@gbgconsult.se',
             '+46312345678'),
            ('5576543210', 'Uppsala Innovation Labs', 'Kungsgatan 8', '753 21', 'Uppsala', 'lab@uppsalainnovation.se',
             '+46181234567'),
            ('5565432109', 'Linköping Software House', 'Storgatan 33', '582 23', 'Linköping', 'dev@linkopingsw.se',
             '+46131234567')
        ]

        customers = [
            ('7362321000224', 'Region Västerbotten', '2321000222', 'DEJA01', 'daniel.stromberg@regionvasterbotten.se'),
            ('7365432100001', 'Stockholms Stad', '2123456789', 'STOCK01', 'procurement@stockholm.se'),
            ('7367654321002', 'Region Skåne', '2987654321', 'SKANE01', 'inkop@skane.se'),
            ('7361234567003', 'Västra Götaland', '2456789012', 'VGR01', 'inköp@vgregion.se'),
            ('7368765432004', 'Region Uppsala', '2345678901', 'UPP01', 'ekonomi@regionuppsala.se')
        ]

        # Insert sample invoices
        current_date = datetime.now()
        invoice_data = []

        for i in range(50):  # Generate 50 invoices
            invoice_id = f"INV{str(i + 1).zfill(6)}"
            issue_date = (current_date - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            due_date = (datetime.strptime(issue_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

            supplier = random.choice(suppliers)
            customer = random.choice(customers)

            tax_amount = round(random.uniform(100, 5000), 2)
            line_ext_amount = round(random.uniform(1000, 25000), 2)
            tax_incl_amount = line_ext_amount + tax_amount

            etl_timestamp = (current_date + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S.000')

            invoice_data.append((
                invoice_id,  # INVOICE_ID
                issue_date,  # ISSUE_DATE
                supplier[0],  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
                supplier[1],  # SUPPLIER_PARTY_NAME
                supplier[2],  # SUPPLIER_PARTY_STREET_NAME
                None,  # SUPPLIER_PARTY_ADDITIONAL_STREET_NAME
                supplier[3],  # SUPPLIER_PARTY_POSTAL_ZONE
                supplier[4],  # SUPPLIER_PARTY_CITY
                'SE',  # SUPPLIER_PARTY_COUNTRY
                None,  # SUPPLIER_PARTY_ADDRESS_LINE
                supplier[1],  # SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME
                'Aktiebolag',  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM
                None,  # SUPPLIER_PARTY_CONTACT_NAME
                supplier[5],  # SUPPLIER_PARTY_CONTACT_EMAIL
                supplier[6],  # SUPPLIER_PARTY_CONTACT_PHONE
                supplier[0],  # SUPPLIER_PARTY_ENDPOINT_ID
                customer[0],  # CUSTOMER_PARTY_ID
                '0088',  # CUSTOMER_PARTY_ID_SCHEME_ID
                customer[0],  # CUSTOMER_PARTY_ENDPOINT_ID
                '0088',  # CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID
                customer[1],  # CUSTOMER_PARTY_NAME
                None,  # CUSTOMER_PARTY_STREET_NAME
                None,  # CUSTOMER_PARTY_POSTAL_ZONE
                'SE',  # CUSTOMER_PARTY_COUNTRY
                customer[1],  # CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME
                customer[2],  # CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID
                customer[3],  # CUSTOMER_PARTY_CONTACT_NAME
                customer[4],  # CUSTOMER_PARTY_CONTACT_EMAIL
                None,  # CUSTOMER_PARTY_CONTACT_PHONE
                due_date,  # DUE_DATE
                'SEK',  # DOCUMENT_CURRENCY_CODE
                None,  # DELIVERY_LOCATION_STREET_NAME
                None,  # DELIVERY_LOCATION_ADDITIONAL_STREET_NAME
                None,  # DELIVERY_LOCATION_CITY_NAME
                None,  # DELIVERY_LOCATION_POSTAL_ZONE
                None,  # DELIVERY_LOCATION_ADDRESS_LINE
                None,  # DELIVERY_LOCATION_COUNTRY
                None,  # DELIVERY_PARTY_NAME
                None,  # ACTUAL_DELIVERY_DATE
                'SEK',  # TAX_AMOUNT_CURRENCY
                tax_amount,  # TAX_AMOUNT
                None,  # PERIOD_START_DATE
                None,  # PERIOD_END_DATE
                'SEK',  # LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY
                line_ext_amount,  # LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
                'SEK',  # LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY
                line_ext_amount,  # LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT
                'SEK',  # LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY
                tax_incl_amount,  # LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT
                'SEK',  # LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY
                tax_incl_amount,  # LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
                None,  # LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY
                None,  # LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT
                None,  # LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY
                None,  # LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT
                'SEK',  # LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY
                0.0,  # LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT
                None,  # LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY
                None,  # LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT
                customer[3],  # BUYER_REFERENCE
                None,  # PROJECT_REFERENCE_ID
                '380',  # INVOICE_TYPE_CODE
                f'Invoice for {supplier[1]} services',  # NOTE
                None,  # TAX_POINT_DATE
                None,  # ACCOUNTING_COST
                f'PO-{str(i + 1).zfill(4)}',  # ORDER_REFERENCE_ID
                None,  # ORDER_REFERENCE_SALES_ORDER_ID
                'Net 30',  # PAYMENT_TERMS_NOTE
                None,  # BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID
                None,  # BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE
                None,  # CONTRACT_DOCUMENT_REFERENCE_ID
                None,  # DESPATCH_DOCUMENT_REFERENCE_ID
                etl_timestamp  # ETL_LOAD_TS
            ))

        invoice_columns = [
            'INVOICE_ID', 'ISSUE_DATE', 'SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID', 'SUPPLIER_PARTY_NAME',
            'SUPPLIER_PARTY_STREET_NAME', 'SUPPLIER_PARTY_ADDITIONAL_STREET_NAME', 'SUPPLIER_PARTY_POSTAL_ZONE',
            'SUPPLIER_PARTY_CITY', 'SUPPLIER_PARTY_COUNTRY', 'SUPPLIER_PARTY_ADDRESS_LINE',
            'SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME', 'SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM',
            'SUPPLIER_PARTY_CONTACT_NAME', 'SUPPLIER_PARTY_CONTACT_EMAIL', 'SUPPLIER_PARTY_CONTACT_PHONE',
            'SUPPLIER_PARTY_ENDPOINT_ID', 'CUSTOMER_PARTY_ID', 'CUSTOMER_PARTY_ID_SCHEME_ID',
            'CUSTOMER_PARTY_ENDPOINT_ID', 'CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID', 'CUSTOMER_PARTY_NAME',
            'CUSTOMER_PARTY_STREET_NAME', 'CUSTOMER_PARTY_POSTAL_ZONE', 'CUSTOMER_PARTY_COUNTRY',
            'CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME', 'CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID',
            'CUSTOMER_PARTY_CONTACT_NAME', 'CUSTOMER_PARTY_CONTACT_EMAIL', 'CUSTOMER_PARTY_CONTACT_PHONE',
            'DUE_DATE', 'DOCUMENT_CURRENCY_CODE', 'DELIVERY_LOCATION_STREET_NAME',
            'DELIVERY_LOCATION_ADDITIONAL_STREET_NAME', 'DELIVERY_LOCATION_CITY_NAME',
            'DELIVERY_LOCATION_POSTAL_ZONE', 'DELIVERY_LOCATION_ADDRESS_LINE', 'DELIVERY_LOCATION_COUNTRY',
            'DELIVERY_PARTY_NAME', 'ACTUAL_DELIVERY_DATE', 'TAX_AMOUNT_CURRENCY', 'TAX_AMOUNT',
            'PERIOD_START_DATE', 'PERIOD_END_DATE', 'LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT', 'LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT', 'LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT', 'LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT', 'LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT', 'LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT', 'LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT', 'LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY',
            'LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT', 'BUYER_REFERENCE', 'PROJECT_REFERENCE_ID',
            'INVOICE_TYPE_CODE', 'NOTE', 'TAX_POINT_DATE', 'ACCOUNTING_COST', 'ORDER_REFERENCE_ID',
            'ORDER_REFERENCE_SALES_ORDER_ID', 'PAYMENT_TERMS_NOTE', 'BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID',
            'BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE', 'CONTRACT_DOCUMENT_REFERENCE_ID',
            'DESPATCH_DOCUMENT_REFERENCE_ID', 'ETL_LOAD_TS'
        ]

        placeholders = ', '.join(['?' for _ in invoice_columns])
        insert_sql = f"INSERT INTO Invoice ({', '.join(invoice_columns)}) VALUES ({placeholders})"
        await db.executemany(insert_sql, invoice_data)

        # Insert invoice lines
        services = [
            ('IT Consulting', 'ITCONS', 'HR', 25.0),
            ('Software License', 'SWLIC', 'S', 25.0),
            ('Hotel Accommodation', 'HOTEL', 'S', 12.0),
            ('Training Services', 'TRAIN', 'S', 25.0),
            ('Medical Supplies', 'MEDSUP', 'S', 25.0),
            ('Office Equipment', 'OFFICE', 'S', 25.0),
            ('Maintenance Service', 'MAINT', 'S', 25.0),
            ('Consulting Fee', 'CONFEE', 'S', 25.0)
        ]

        invoice_line_data = []
        for i, invoice in enumerate(invoice_data):
            invoice_id = invoice[0]
            issue_date = invoice[1]
            supplier_id = invoice[2]

            # Generate 1-3 lines per invoice
            num_lines = random.randint(1, 3)
            for line_num in range(1, num_lines + 1):
                service = random.choice(services)
                quantity = round(random.uniform(1, 10), 3)
                unit_price = round(random.uniform(500, 3000), 2)
                line_amount = round(quantity * unit_price, 2)

                etl_timestamp = invoice[69]  # Use same ETL timestamp as invoice

                invoice_line_data.append((
                    invoice_id,  # INVOICE_ID
                    issue_date,  # ISSUE_DATE
                    supplier_id,  # SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID
                    str(line_num),  # INVOICE_LINE_ID
                    None,  # ORDER_LINE_REFERENCE_LINE_ID
                    None,  # ACCOUNTING_COST
                    quantity,  # INVOICED_QUANTITY
                    'EA',  # INVOICED_QUANTITY_UNIT_CODE
                    line_amount,  # INVOICED_LINE_EXTENSION_AMOUNT
                    'SEK',  # INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID
                    None,  # INVOICE_PERIOD_START_DATE
                    None,  # INVOICE_PERIOD_END_DATE
                    None,  # INVOICE_LINE_DOCUMENT_REFERENCE_ID
                    None,  # INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE
                    None,  # INVOICE_LINE_NOTE
                    None,  # ITEM_DESCRIPTION
                    service[0],  # ITEM_NAME
                    service[2],  # ITEM_TAXCAT_ID
                    service[3],  # ITEM_TAXCAT_PERCENT
                    None,  # ITEM_BUYERS_ID
                    service[1],  # ITEM_SELLERS_ITEM_ID
                    None,  # ITEM_STANDARD_ITEM_ID
                    None,  # ITEM_COMMODITYCLASS_CLASSIFICATION
                    None,  # ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID
                    unit_price,  # PRICE_AMOUNT
                    'SEK',  # PRICE_AMOUNT_CURRENCY_ID
                    None,  # PRICE_BASE_QUANTITY
                    None,  # PRICE_BASE_QUANTITY_UNIT_CODE
                    None,  # PRICE_ALLOWANCE_CHARGE_AMOUNT
                    None,  # PRICE_ALLOWANCE_CHARGE_INDICATOR
                    etl_timestamp  # ETL_LOAD_TS
                ))

        invoice_line_columns = [
            'INVOICE_ID', 'ISSUE_DATE', 'SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID', 'INVOICE_LINE_ID',
            'ORDER_LINE_REFERENCE_LINE_ID', 'ACCOUNTING_COST', 'INVOICED_QUANTITY',
            'INVOICED_QUANTITY_UNIT_CODE', 'INVOICED_LINE_EXTENSION_AMOUNT',
            'INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID', 'INVOICE_PERIOD_START_DATE',
            'INVOICE_PERIOD_END_DATE', 'INVOICE_LINE_DOCUMENT_REFERENCE_ID',
            'INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE', 'INVOICE_LINE_NOTE',
            'ITEM_DESCRIPTION', 'ITEM_NAME', 'ITEM_TAXCAT_ID', 'ITEM_TAXCAT_PERCENT',
            'ITEM_BUYERS_ID', 'ITEM_SELLERS_ITEM_ID', 'ITEM_STANDARD_ITEM_ID',
            'ITEM_COMMODITYCLASS_CLASSIFICATION', 'ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID',
            'PRICE_AMOUNT', 'PRICE_AMOUNT_CURRENCY_ID', 'PRICE_BASE_QUANTITY',
            'PRICE_BASE_QUANTITY_UNIT_CODE', 'PRICE_ALLOWANCE_CHARGE_AMOUNT',
            'PRICE_ALLOWANCE_CHARGE_INDICATOR', 'ETL_LOAD_TS'
        ]

        line_placeholders = ', '.join(['?' for _ in invoice_line_columns])
        line_insert_sql = f"INSERT INTO Invoice_Line ({', '.join(invoice_line_columns)}) VALUES ({line_placeholders})"
        await db.executemany(line_insert_sql, invoice_line_data)

    def _execute_query_sync(self, conn, query: str):
        """Synchronous query execution for pymssql"""
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute any SQL query and return results"""
        async with self.get_connection() as conn:
            if self.use_remote:
                # MSSQL query execution in thread pool
                loop = asyncio.get_running_loop()
                rows = await loop.run_in_executor(
                    self.executor,
                    self._execute_query_sync,
                    conn,
                    query
                )
                # pymssql with as_dict=True returns dictionaries already
                return rows
            else:
                # SQLite query execution
                async with conn.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]

    async def execute_query_stream(self, query: str):
        """Execute SQL query and yield results one row at a time"""
        async with self.get_connection() as conn:
            if self.use_remote:
                # MSSQL streaming - fetch all then yield
                # (true streaming would require cursor iteration in executor)
                loop = asyncio.get_running_loop()
                rows = await loop.run_in_executor(
                    self.executor,
                    self._execute_query_sync,
                    conn,
                    query
                )
                for row in rows:
                    yield row
            else:
                # SQLite streaming
                async with conn.execute(query) as cursor:
                    async for row in cursor:
                        yield dict(row)


# Initialize database server
db_server = AsyncDatabaseServer(
    pool_size=MCP_CONFIG.get("connection_pool_size", 10)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting invoice database initialization...")
    start_time = asyncio.get_event_loop().time()

    await db_server.initialize_database()
    await db_server.initialize_pool()

    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"✅ System ready in {elapsed * 1000:.2f}ms!")

    # Schema inspection for remote database
    if USE_REMOTE:
        try:
            print("\n" + "="*60)
            print("DATABASE SCHEMA INSPECTION")
            print("="*60)
            
            # Get all tables
            tables_query = """
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
            tables = await db_server.execute_query(tables_query)
            
            print(f"\nFound {len(tables)} tables in database '{SQL_SERVER_DATABASE}':\n")
            
            for table in tables:
                table_schema = table['TABLE_SCHEMA']
                table_name = table['TABLE_NAME']
                print(f"\nTable: [{table_schema}].[{table_name}]")
                
                # Get columns for this table
                columns_query = f"""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{table_schema}'
                  AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
                """
                columns = await db_server.execute_query(columns_query)
                
                for col in columns:
                    nullable = "NULL" if col['IS_NULLABLE'] == 'YES' else "NOT NULL"
                    max_len = f"({col['CHARACTER_MAXIMUM_LENGTH']})" if col['CHARACTER_MAXIMUM_LENGTH'] else ""
                    print(f"   - {col['COLUMN_NAME']}: {col['DATA_TYPE']}{max_len} {nullable}")
            
            print("\n" + "="*60)
            print("Schema inspection complete!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\nSchema inspection failed: {e}\n")

    # Verify database is accessible
    try:
        if USE_REMOTE:
            test_result = await db_server.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
            lines_result = await db_server.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice_Line]")
        else:
            test_result = await db_server.execute_query("SELECT 1 FROM Invoice LIMIT 1")
            lines_result = await db_server.execute_query("SELECT 1 FROM Invoice_Line LIMIT 1")

        if test_result and lines_result:
            print(f"Invoice Test: {test_result}")
            print(f"Invoice_Line Test: {lines_result}")
            print(f"✅ Database validated - tables ready!")
        else:
            print(f"⚠️ Database may be empty")
    except Exception as e:
        print(f"❌ Error checking database: {e}")

    yield  # App runs here

    # Shutdown
    print("🛑 Shutting down database connections...")
    await db_server.close_pool()
    print("✅ All connections closed gracefully!")


# FastAPI setup with lifespan
app = FastAPI(
    title="Invoice Database API",
    description="Database Server for Invoice Management",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def execute_query(request: QueryRequest):
    """Execute SQL query and return results"""
    try:
        results = await db_server.execute_query(request.query)
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/query_stream")
async def execute_query_stream(request: QueryRequest):
    """Stream SQL query results one row at a time"""

    async def generate_results():
        try:
            yield json.dumps({"type": "start", "query": request.query}) + "\n"

            count = 0
            async for row in db_server.execute_query_stream(request.query):
                count += 1
                result = {
                    "type": "row",
                    "data": row,
                    "index": count
                }
                yield json.dumps(result) + "\n"

            yield json.dumps({"type": "complete", "total_rows": count}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(
        generate_results(),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    try:
        # Verify tables exist
        if USE_REMOTE:
            invoice_exists = await db_server.execute_query("SELECT TOP 1 1 FROM Invoice")
            line_exists = await db_server.execute_query("SELECT TOP 1 1 FROM Invoice_Line")
        else:
            invoice_exists = await db_server.execute_query("SELECT 1 FROM Invoice LIMIT 1")
            line_exists = await db_server.execute_query("SELECT 1 FROM Invoice_Line LIMIT 1")

        return {
            "status": "healthy",
            "database_type": "MSSQL" if USE_REMOTE else "SQLite",
            "database_location": SQL_SERVER_HOST if USE_REMOTE else db_server.db_path,
            "pool_size": db_server.pool_size,
            "pool_available": db_server.pool.qsize(),
            "tables_initialized": bool(invoice_exists and line_exists)
        }
    except Exception as e:
        return {
            "status": "error",
            "database_type": "MSSQL" if USE_REMOTE else "SQLite",
            "pool_size": db_server.pool_size,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8762)
