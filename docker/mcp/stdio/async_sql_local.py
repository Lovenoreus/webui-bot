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
    print("🚀 Starting invoice database server...")
    start_time = asyncio.get_event_loop().time()

    # Initialize connection pool only (no database creation)
    await db_server.initialize_pool()

    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"✅ System ready in {elapsed * 1000:.2f}ms!")

    # Schema inspection for remote database
    if USE_REMOTE:
        try:
            print("\n" + "=" * 60)
            print("DATABASE SCHEMA INSPECTION")
            print("=" * 60)

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

            print("\n" + "=" * 60)
            print("Schema inspection complete!")
            print("=" * 60 + "\n")

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
            invoice_exists = await db_server.execute_query("SELECT TOP 1 1 FROM [Nodinite].[ods].[Invoice]")
            line_exists = await db_server.execute_query("SELECT TOP 1 1 FROM [Nodinite].[ods].[Invoice_Line]")
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
