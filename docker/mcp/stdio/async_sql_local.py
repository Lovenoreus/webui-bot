# -------------------- Built-in Libraries --------------------
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Optional
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
            "connection_pool_size": 10,
            "retry": {
                "max_attempts": 5,
                "initial_delay": 2.0,
                "backoff_multiplier": 2.0,
                "connection_timeout": 300
            }
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


async def retry_async(
    func: Callable,
    max_attempts: int = 5,
    delay: float = 2.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    timeout: float = 300.0
) -> any:
    """
    Retry an async function with exponential backoff
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt
        max_delay: Maximum delay between retries
        timeout: Timeout for each individual attempt
    """
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  Attempt {attempt}/{max_attempts}...")
            result = await asyncio.wait_for(func(), timeout=timeout)
            print(f"  ✅ Success on attempt {attempt}")
            return result
        except asyncio.TimeoutError:
            print(f"  ⏱️ Attempt {attempt} timed out after {timeout}s")
            if attempt == max_attempts:
                raise Exception(f"Failed after {max_attempts} attempts: Timeout")
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Attempt {attempt} failed: {error_msg[:150]}")
            
            if attempt == max_attempts:
                raise Exception(f"Failed after {max_attempts} attempts: {error_msg}")
        
        # Wait before next attempt (except on last attempt)
        if attempt < max_attempts:
            print(f"  ⏳ Waiting {current_delay:.1f}s before retry...")
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * backoff, max_delay)
    
    raise Exception(f"Failed after {max_attempts} attempts")


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
        
        # Load retry configuration
        retry_config = MCP_CONFIG.get("retry", {})
        self.max_retry_attempts = retry_config.get("max_attempts", 5)
        self.retry_delay = retry_config.get("initial_delay", 2.0)
        self.retry_backoff = retry_config.get("backoff_multiplier", 2.0)
        self.connection_timeout = retry_config.get("connection_timeout", 300.0)
        
        print(f"Retry configuration:")
        print(f"  Max attempts: {self.max_retry_attempts}")
        print(f"  Initial delay: {self.retry_delay}s")
        print(f"  Backoff multiplier: {self.retry_backoff}x")
        print(f"  Connection timeout: {self.connection_timeout}s")

    def _create_connection_sync(self):
        """Synchronous connection creation for pymssql with timeout"""
        if SQL_SERVER_USE_WINDOWS_AUTH:
            # Windows Authentication
            return pymssql.connect(
                server=self.server,
                database=self.database,
                timeout=int(self.connection_timeout),
                login_timeout=int(self.connection_timeout),
                as_dict=True
            )
            
        else:
            # SQL Server Authentication
            return pymssql.connect(
                server=self.server, 
                user=self.username,
                password=self.password,
                database=self.database,
                timeout=int(self.connection_timeout),
                login_timeout=int(self.connection_timeout),
                as_dict=True
            )

    async def _create_connection_with_retry(self):
        """Create connection with retry logic"""
        async def create_conn():
            if self.use_remote:
                loop = asyncio.get_running_loop()
                conn = await loop.run_in_executor(
                    self.executor,
                    self._create_connection_sync
                )
                return conn
            else:
                db = await aiosqlite.connect(self.db_path)
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=memory")
                await db.execute("PRAGMA foreign_keys=ON")
                db.row_factory = aiosqlite.Row
                return db
        
        return await retry_async(
            create_conn,
            max_attempts=self.max_retry_attempts,
            delay=self.retry_delay,
            backoff=self.retry_backoff,
            timeout=self.connection_timeout
        )

    async def _create_connection(self):
        """Create a new database connection with optimal settings and retry logic"""
        return await self._create_connection_with_retry()

    async def initialize_pool(self):
        """Initialize the connection pool with retry logic"""
        if self.pool_initialized:
            return

        print(f"Initializing connection pool with {self.pool_size} connections...")
        start_time = asyncio.get_event_loop().time()

        # Create connections with retry logic
        # Note: We create them sequentially to avoid overwhelming the server
        connections = []
        for i in range(self.pool_size):
            print(f"\nCreating connection {i+1}/{self.pool_size}...")
            try:
                conn = await self._create_connection()
                connections.append(conn)
                print(f"✅ Connection {i+1} ready")
            except Exception as e:
                print(f"❌ Failed to create connection {i+1}: {e}")
                # Clean up any successful connections
                for c in connections:
                    if self.use_remote:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(self.executor, self._close_connection_sync, c)
                    else:
                        await c.close()
                raise Exception(f"Failed to initialize pool: {e}")

        # Add all connections to the pool
        for conn in connections:
            await self.pool.put(conn)

        self.pool_initialized = True
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"\n✅ Connection pool initialized in {elapsed:.2f}s!")
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

    def _is_connection_error(self, error: Exception) -> bool:
        """Check if error is a connection-related error that should trigger retry"""
        error_str = str(error).lower()
        
        print(f"[Connection Error Check] Checking error string: {error_str[:200]}")
        
        connection_errors = [
            'unable to connect',
            'adaptive server is unavailable',
            'connection is closed',
            'connection was killed',
            'broken pipe',
            'connection reset',
            'timed out',
            'timeout',
            'network error',
            'communication link failure',
            'lost connection',
            'no such host',
            'host is down',
            '20009',  # DB-Lib error code
            '20003',  # Another common connection error
            '20006',  # Write to server failed
            'database is locked',  # SQLite error
            'disk i/o error',  # SQLite error
        ]
        
        for error_pattern in connection_errors:
            if error_pattern in error_str:
                print(f"[Connection Error Check] ✓ MATCH FOUND: '{error_pattern}'")
                return True
        
        print(f"[Connection Error Check] ✗ NO MATCH - Not a connection error")
        return False

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute any SQL query and return results with retry logic"""
        max_retries = 5
        retry_delay = 1.0
        max_backoff = 5.0
        
        print(f"[Query Execution] ⚡ STARTING QUERY EXECUTION WITH RETRY LOGIC")
        print(f"[Query Execution] Query: {query[:200]}...")
        print(f"[Query Execution] Max retries: {max_retries}, Initial delay: {retry_delay}s, Max backoff: {max_backoff}s")
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Query Execution] 🔄 Attempt {attempt}/{max_retries}")
                print(f"[Query Execution] Getting connection from pool (use_remote={self.use_remote})...")
                
                async with self.get_connection() as conn:
                    print(f"[Query Execution] ✓ Connection acquired")
                    
                    if self.use_remote:
                        print(f"[Query Execution] Executing MSSQL query in thread pool...")
                        # MSSQL query execution in thread pool
                        loop = asyncio.get_running_loop()
                        rows = await loop.run_in_executor(
                            self.executor,
                            self._execute_query_sync,
                            conn,
                            query
                        )
                        # pymssql with as_dict=True returns dictionaries already
                        print(f"[Query Execution] ✅ SUCCESS on attempt {attempt} - Returned {len(rows)} rows")
                        return rows
                    else:
                        print(f"[Query Execution] Executing SQLite query...")
                        # SQLite query execution
                        async with conn.execute(query) as cursor:
                            rows = await cursor.fetchall()
                            result = [dict(row) for row in rows]
                            print(f"[Query Execution] ✅ SUCCESS on attempt {attempt} - Returned {len(result)} rows")
                            return result
                            
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"[Query Execution] ❌ EXCEPTION CAUGHT on attempt {attempt}")
                print(f"[Query Execution] Exception type: {error_type}")
                print(f"[Query Execution] Exception message: {error_msg[:300]}")
                
                # Check if it's a connection error that should trigger retry
                is_conn_error = self._is_connection_error(e)
                print(f"[Query Execution] Is connection error? {is_conn_error}")
                
                if is_conn_error:
                    print(f"[Query Execution] ⚠️ CONNECTION ERROR DETECTED - Will retry")
                    
                    if attempt < max_retries:
                        current_backoff = min(retry_delay, max_backoff)
                        print(f"[Query Execution] ⏳ RETRYING in {current_backoff:.1f}s...")
                        print(f"[Query Execution]    (calculated backoff: {retry_delay:.1f}s, capped at {max_backoff}s)")
                        await asyncio.sleep(current_backoff)
                        retry_delay *= 2  # Exponential backoff
                        print(f"[Query Execution] Next backoff will be: {retry_delay:.1f}s")
                        continue
                    else:
                        print(f"[Query Execution] 💥 ALL {max_retries} RETRY ATTEMPTS EXHAUSTED")
                        raise Exception(f"Query failed after {max_retries} attempts: {error_msg}")
                else:
                    # Not a connection error - fail immediately
                    print(f"[Query Execution] 💥 NON-CONNECTION ERROR - FAILING IMMEDIATELY")
                    raise
            
            print(f"[Query Execution] ⚠️ End of attempt {attempt} (should not see this)")
        
        # Should never reach here
        print(f"[Query Execution] ⚠️ REACHED END OF RETRY LOOP (should never happen)")
        raise Exception(f"Query failed after {max_retries} attempts")

    async def execute_query_stream(self, query: str):
        """Execute SQL query and yield results one row at a time with retry logic"""
        max_retries = 5
        retry_delay = 1.0
        max_backoff = 5.0
        
        print(f"[Query Stream] ⚡ STARTING STREAMING QUERY WITH RETRY LOGIC")
        print(f"[Query Stream] Query: {query[:200]}...")
        print(f"[Query Stream] Max retries: {max_retries}, Initial delay: {retry_delay}s, Max backoff: {max_backoff}s")
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Query Stream] 🔄 Attempt {attempt}/{max_retries}")
                print(f"[Query Stream] Getting connection from pool (use_remote={self.use_remote})...")
                
                async with self.get_connection() as conn:
                    print(f"[Query Stream] ✓ Connection acquired")
                    
                    if self.use_remote:
                        print(f"[Query Stream] Executing MSSQL streaming query in thread pool...")
                        # MSSQL streaming - fetch all then yield
                        # (true streaming would require cursor iteration in executor)
                        loop = asyncio.get_running_loop()
                        rows = await loop.run_in_executor(
                            self.executor,
                            self._execute_query_sync,
                            conn,
                            query
                        )
                        print(f"[Query Stream] ✅ SUCCESS on attempt {attempt} - Streaming {len(rows)} rows")
                        for row in rows:
                            yield row
                        return  # Success - exit retry loop
                    else:
                        print(f"[Query Stream] Executing SQLite streaming query...")
                        # SQLite streaming
                        print(f"[Query Stream] ✅ SUCCESS on attempt {attempt} - Streaming results")
                        async with conn.execute(query) as cursor:
                            async for row in cursor:
                                yield dict(row)
                        return  # Success - exit retry loop
                        
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"[Query Stream] ❌ EXCEPTION CAUGHT on attempt {attempt}")
                print(f"[Query Stream] Exception type: {error_type}")
                print(f"[Query Stream] Exception message: {error_msg[:300]}")
                
                # Check if it's a connection error that should trigger retry
                is_conn_error = self._is_connection_error(e)
                print(f"[Query Stream] Is connection error? {is_conn_error}")
                
                if is_conn_error:
                    print(f"[Query Stream] ⚠️ CONNECTION ERROR DETECTED - Will retry")
                    
                    if attempt < max_retries:
                        current_backoff = min(retry_delay, max_backoff)
                        print(f"[Query Stream] ⏳ RETRYING in {current_backoff:.1f}s...")
                        print(f"[Query Stream]    (calculated backoff: {retry_delay:.1f}s, capped at {max_backoff}s)")
                        await asyncio.sleep(current_backoff)
                        retry_delay *= 2  # Exponential backoff
                        print(f"[Query Stream] Next backoff will be: {retry_delay:.1f}s")
                        continue
                    else:
                        print(f"[Query Stream] 💥 ALL {max_retries} RETRY ATTEMPTS EXHAUSTED")
                        raise Exception(f"Query stream failed after {max_retries} attempts: {error_msg}")
                else:
                    # Not a connection error - fail immediately
                    print(f"[Query Stream] 💥 NON-CONNECTION ERROR - FAILING IMMEDIATELY")
                    raise
            
            print(f"[Query Stream] ⚠️ End of attempt {attempt} (should not see this)")
        
        # Should never reach here
        print(f"[Query Stream] ⚠️ REACHED END OF RETRY LOOP (should never happen)")
        raise Exception(f"Query stream failed after {max_retries} attempts")


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
        # # Verify tables exist
        # if USE_REMOTE:
        #     invoice_exists = await db_server.execute_query("SELECT TOP 1 1 FROM [Nodinite].[ods].[Invoice]")
        #     line_exists = await db_server.execute_query("SELECT TOP 1 1 FROM [Nodinite].[ods].[Invoice_Line]")
        # else:
        #     invoice_exists = await db_server.execute_query("SELECT 1 FROM Invoice LIMIT 1")
        #     line_exists = await db_server.execute_query("SELECT 1 FROM Invoice_Line LIMIT 1")

        return {
            "status": "healthy",
            "database_type": "MSSQL" if USE_REMOTE else "SQLite",
            "database_location": SQL_SERVER_HOST if USE_REMOTE else db_server.db_path,
            "pool_size": db_server.pool_size,
            "pool_available": db_server.pool.qsize(),
            # "tables_initialized": bool(invoice_exists and line_exists)
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
