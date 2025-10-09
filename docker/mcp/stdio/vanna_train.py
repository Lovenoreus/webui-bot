# ====== SSL CERTIFICATE BYPASS - MUST BE FIRST ======
import ssl
import os
import warnings

# Create unverified SSL context globally
_original_create_default_context = ssl.create_default_context
def _create_unverified_context(*args, **kwargs):
    context = _original_create_default_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# Override SSL context creation
ssl.create_default_context = _create_unverified_context
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables FIRST
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OPENAI_VERIFY_SSL"] = "false"

# Import and patch requests before anything else uses it
import requests
import urllib3
import http.client

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Patch http.client for low-level SSL bypass
original_https_connection_init = http.client.HTTPSConnection.__init__
def patched_https_connection_init(self, *args, **kwargs):
    kwargs['context'] = ssl._create_unverified_context()
    return original_https_connection_init(self, *args, **kwargs)
http.client.HTTPSConnection.__init__ = patched_https_connection_init

# Patch all requests methods
_original_request = requests.request
def _patched_request(method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return _original_request(method, url, **kwargs)
requests.request = _patched_request

# Patch Session.request
_original_session_request = requests.Session.request
def _patched_session_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return _original_session_request(self, method, url, **kwargs)
requests.Session.request = _patched_session_request

# Final low-level socket SSL bypass
import socket
_original_create_connection = socket.create_connection
def _patched_create_connection(address, *args, **kwargs):
    # For HTTPS connections, disable SSL verification at socket level
    return _original_create_connection(address, *args, **kwargs)
socket.create_connection = _patched_create_connection

# ====== END SSL BYPASS ======

from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import logging
import builtins
from typing import Optional
import pymssql
import pandas as pd
import config

load_dotenv()

# Additional imports
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

# Override certifi to return empty path
certifi.where = lambda: ""

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Store original print
_original_print = builtins.print

# List of strings to suppress
SUPPRESS_PHRASES = [
    "SQL Prompt:",
    "Using model",
    "LLM Response:",
    "Extracted SQL:",
    "tokens (approx)"
]

def filtered_print(*args, **kwargs):
    """Custom print that filters out Vanna's verbose output"""
    text = ' '.join(str(arg) for arg in args)
    # Only suppress if it matches our phrases
    if any(phrase in text for phrase in SUPPRESS_PHRASES):
        return
    _original_print(*args, **kwargs)

# Replace built-in print
builtins.print = filtered_print

class VannaModelManager:
    """Manager class to handle switching between different LLM providers and databases for Vanna training"""
    
    def __init__(self):
        self.current_provider = self._get_active_provider()
        self.current_database = config.VANNA_ACTIVE_DATABASE
        self.vanna_client = None
        self.database_connection = None
    
    def _get_active_provider(self) -> str:
        """Determine which provider is currently active based on config"""
        if config.USE_VANNA_OPENAI:
            return "openai"
        elif config.USE_VANNA_OLLAMA:
            return "ollama"
        else:
            raise ValueError("No Vanna provider is enabled in config. Set either vanna.openai.enabled or vanna.ollama.enabled to true")
    
    def get_vanna_class(self, provider: str):
        """Get the appropriate Vanna class based on provider"""
        if provider == "openai":
            class MyVannaOpenAI(ChromaDB_VectorStore, OpenAI_Chat):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    OpenAI_Chat.__init__(self, config=config)
            return MyVannaOpenAI
        
        elif provider == "ollama":
            class MyVannaOllama(ChromaDB_VectorStore, Ollama):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    Ollama.__init__(self, config=config)
            return MyVannaOllama
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def initialize_vanna(self, provider: Optional[str] = None):
        """Initialize Vanna with specified provider or use config default"""
        target_provider = provider or self.current_provider
        
        if target_provider == "openai":
            self._init_openai_vanna()
        elif target_provider == "ollama":
            self._init_ollama_vanna()
        else:
            raise ValueError(f"Unsupported provider: {target_provider}")
        
        # Connect to the active database
        self.connect_to_database()
        
        print(f"Vanna initialized with provider: {target_provider} and database: {self.current_database}")
        return self.vanna_client
    
    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        VannaClass = self.get_vanna_class("openai")
        self.vanna_client = VannaClass(config={
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OPENAI_VERBOSE
        })
        
        # Post-initialization SSL bypass attempts
        try:
            # Try to patch any internal session objects
            if hasattr(self.vanna_client, '_client'):
                if hasattr(self.vanna_client._client, '_session'):
                    self.vanna_client._client._session.verify = False
                if hasattr(self.vanna_client._client, 'session'):
                    self.vanna_client._client.session.verify = False
        except Exception:
            pass
            
        self.current_provider = "openai"
    
    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        VannaClass = self.get_vanna_class("ollama")
        self.vanna_client = VannaClass(config={
            'model': config.VANNA_OLLAMA_MODEL,
            'base_url': config.VANNA_OLLAMA_BASE_URL,
            'allow_llm_to_see_data': config.VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OLLAMA_VERBOSE
        })
        self.current_provider = "ollama"
    
    def get_db_port(self, database_type: str) -> int:
        """Get the appropriate port for the database type"""
        db_port = os.getenv("DB_PORT")
        if db_port and db_port.isdigit():
            return int(db_port)
        return config.VANNA_DEFAULT_PORTS.get(database_type, 5432)
    
    def connect_to_database(self, database_type: Optional[str] = None):
        """Connect to the specified database or use the active one from config"""
        target_database = database_type or self.current_database
        
        if not self.vanna_client:
            raise ValueError("Vanna client must be initialized before connecting to database")
        
        try:
            if target_database == "postgresql":
                self._connect_to_postgresql()
            elif target_database == "mysql":
                self._connect_to_mysql()
            elif target_database == "mssql":
                self._connect_to_mssql()
            elif target_database == "sqlite":
                self._connect_to_sqlite()
            else:
                raise ValueError(f"Unsupported database type: {target_database}")
            
            self.current_database = target_database
            print(f"Connected to {target_database} database successfully")
            
        except Exception as e:
            print(f"Failed to connect to {target_database} database: {e}")
            raise
    
    def _connect_to_postgresql(self):
        """Connect to PostgreSQL database"""
        if not config.VANNA_POSTGRESQL_ENABLED:
            raise ValueError("PostgreSQL is not enabled in configuration")
        
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 package is required for PostgreSQL connection. Install with: pip install psycopg2-binary")
        
        port = self.get_db_port('postgresql')
        connection_string = f"postgresql://{config.VANNA_DB_USERNAME}:{config.VANNA_DB_PASSWORD}@{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}?sslmode={config.VANNA_POSTGRESQL_SSL_MODE}"
        self.vanna_client.connect_to_postgres(
            host=config.VANNA_DB_HOST,
            dbname=config.VANNA_DB_DATABASE,
            user=config.VANNA_DB_USERNAME,
            password=config.VANNA_DB_PASSWORD,
            port=port
        )
        self.database_connection = connection_string
    
    def _connect_to_mysql(self):
        """Connect to MySQL database"""
        if not config.VANNA_MYSQL_ENABLED:
            raise ValueError("MySQL is not enabled in configuration")
        
        try:
            import mysql.connector
        except ImportError:
            raise ImportError("mysql-connector-python package is required for MySQL connection. Install with: pip install mysql-connector-python")
        
        port = self.get_db_port('mysql')
        self.vanna_client.connect_to_mysql(
            host=config.VANNA_DB_HOST,
            database=config.VANNA_DB_DATABASE,
            user=config.VANNA_DB_USERNAME,
            password=config.VANNA_DB_PASSWORD,
            port=port
        )
        self.database_connection = f"mysql://{config.VANNA_DB_USERNAME}:***@{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}"
    
    def _connect_to_mssql(self):
        """Connect to Microsoft SQL Server database"""
        if not config.VANNA_MSSQL_ENABLED:
            raise ValueError("Microsoft SQL Server is not enabled in configuration")
        
        # Connect using pymssql and set up Vanna to use it
        try:
            connection = pymssql.connect(
                server=config.VANNA_DB_HOST,
                user=config.VANNA_DB_USERNAME,
                password=config.VANNA_DB_PASSWORD,
                database=config.VANNA_DB_DATABASE,
                as_dict=True
            )
            
            # Create a custom run_sql function that uses our pymssql connection
            def run_sql_pymssql(sql: str):
                sql = sql.replace("```sql", "").replace("```", "").strip()
                cursor = connection.cursor()
                try:
                    cursor.execute(sql)
                    if sql.strip().upper().startswith('SELECT'):
                        results = cursor.fetchall()
                        print("✅ EXECUTE SUCCESSFULLY")
                        if results:
                            try:
                                import pandas as pd
                                return pd.DataFrame(results)
                            except:
                                return results

                    
                except Exception as e:
                    print("ERORRRRRRRRRRRR: ", e)
                    return e
                finally:
                    cursor.close()
                
                #     # print("RESULTTTTTTTTTT",results)
                #     if results:
                #         # return results
                #     # Convert to pandas DataFrame if results exist
                #         try:
                #             import pandas as pd
                #             return pd.DataFrame(results)
                #         except:
                #             return results
                #     else:
                #         import pandas as pd
                #         return pd.DataFrame()
                # else:
                #     connection.commit()
                #     cursor.close()
                #     return None
            
            # Set the custom run_sql function on the vanna client
            self.vanna_client.run_sql = run_sql_pymssql
            
            # Store the pymssql connection for later use
            self.database_connection = f"mssql://{config.VANNA_DB_HOST}/{config.VANNA_DB_DATABASE}"
            print(f"Connected to MSSQL database using pymssql: {config.VANNA_DB_HOST}/{config.VANNA_DB_DATABASE}")
        except Exception as e:
            raise Exception(f"Failed to connect to MSSQL database with pymssql: {str(e)}")
    
    def _connect_to_sqlite(self):
        """Connect to SQLite database"""
        if not config.VANNA_SQLITE_ENABLED:
            raise ValueError("SQLite is not enabled in configuration")
        
        # Determine the full path to the SQLite database
        db_path = os.path.join(os.path.dirname(__file__), config.VANNA_SQLITE_DATABASE_PATH)
        self.vanna_client.connect_to_sqlite(db_path)
        self.database_connection = db_path
    
    def switch_database(self, database_type: str):
        """Switch to a different database type"""
        available_databases = {
            "postgresql": config.VANNA_POSTGRESQL_ENABLED,
            "mysql": config.VANNA_MYSQL_ENABLED,
            "mssql": config.VANNA_MSSQL_ENABLED,
            "sqlite": config.VANNA_SQLITE_ENABLED
        }
        
        if database_type not in available_databases:
            raise ValueError(f"Unsupported database type: {database_type}")
        
        if not available_databases[database_type]:
            raise ValueError(f"{database_type} is not enabled in configuration")
        
        print(f"Switching from {self.current_database} to {database_type}...")
        self.connect_to_database(database_type)
    
    def get_database_info(self) -> dict:
        """Get current database configuration info"""
        return {
            "current_database": self.current_database,
            "connection": self.database_connection,
            "available_databases": {
                "postgresql": config.VANNA_POSTGRESQL_ENABLED,
                "mysql": config.VANNA_MYSQL_ENABLED,
                "mssql": config.VANNA_MSSQL_ENABLED,
                "sqlite": config.VANNA_SQLITE_ENABLED
            }
        }
    
    def get_current_provider(self) -> str:
        """Get current active provider"""
        return self.current_provider

# Convenience functions for easy usage
def vanna_train(
    ddl: Optional[str] = None,
    documentation: Optional[str] = None,
    question: Optional[str] = None,
    sql: Optional[str] = None
) -> None:
    """
    Train Vanna with different types of data
    
    Args:
        ddl: Database schema DDL statements
        documentation: Documentation text
        question: Natural language question
        sql: Corresponding SQL query for the question
    """
    global vanna_manager
    
    # Ensure Vanna is initialized
    if not vanna_manager.vanna_client:
        vanna_manager.initialize_vanna()
    
    # Train based on provided data
    if ddl:
        vanna_manager.vanna_client.train(ddl=ddl)
        print(f"Trained DDL with {vanna_manager.current_provider}")
    
    if documentation:
        vanna_manager.vanna_client.train(documentation=documentation)
        print(f"Trained documentation with {vanna_manager.current_provider}")
    
    if question and sql:
        vanna_manager.vanna_client.train(question=question, sql=sql)
        print(f"Trained SQL pair with {vanna_manager.current_provider}")
    
    if not any([ddl, documentation, (question and sql)]):
        raise ValueError("Must provide at least one training data type")





def get_vanna_info() -> dict:
    """Get current Vanna configuration info"""
    return {
        "provider": vanna_manager.get_current_provider(),
        "model": config.VANNA_OPENAI_MODEL if vanna_manager.current_provider == "openai" else config.VANNA_OLLAMA_MODEL,
        "database": vanna_manager.current_database,
        "database_connection": vanna_manager.database_connection,
        "initialized": vanna_manager.vanna_client is not None,
        "available_databases": vanna_manager.get_database_info()["available_databases"]
    }

def switch_database_interactive():
    """Interactive database switching"""
    available_dbs = vanna_manager.get_database_info()["available_databases"]
    enabled_dbs = [db for db, enabled in available_dbs.items() if enabled]
    
    if len(enabled_dbs) <= 1:
        print("❌ Only one database is enabled. Enable more databases in config.json to switch.")
        return
    
    print(f"\n📊 Available databases:")
    for i, db in enumerate(enabled_dbs, 1):
        current_marker = " (current)" if db == vanna_manager.current_database else ""
        print(f"  {i}. {db.upper()}{current_marker}")
    
    try:
        choice = input(f"\nSelect database (1-{len(enabled_dbs)}) or 'cancel': ").strip()
        if choice.lower() == 'cancel':
            return
        
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(enabled_dbs):
            selected_db = enabled_dbs[choice_idx]
            if selected_db == vanna_manager.current_database:
                print(f"✅ Already connected to {selected_db.upper()}")
                return
            
            vanna_manager.switch_database(selected_db)
            print(f"✅ Successfully switched to {selected_db.upper()}")
        else:
            print("❌ Invalid selection")
    except (ValueError, IndexError):
        print("❌ Invalid input")
    except Exception as e:
        print(f"❌ Error switching database: {e}")

# Create global manager instance
vanna_manager = VannaModelManager()
vn = vanna_manager.initialize_vanna()

def get_database_schema_info():
    """Get database schema information based on current database type"""
    try:
        if vanna_manager.current_database == "sqlite":
            return vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
        elif vanna_manager.current_database == "postgresql":
            return vn.run_sql("""
                SELECT 'table' as type, 
                       'CREATE TABLE ' || t.table_schema || '.' || t.table_name || ' (' ||
                       string_agg(c.column_name || ' ' || c.data_type, ', ') || ')' as sql
                FROM information_schema.tables t
                JOIN information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
                WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                  AND t.table_type = 'BASE TABLE'
                GROUP BY t.table_schema, t.table_name
                ORDER BY t.table_schema, t.table_name
            """)
        elif vanna_manager.current_database == "mysql":
            return vn.run_sql("""
                SELECT 'table' as type,
                       CONCAT('CREATE TABLE ', TABLE_NAME, ' (',
                       GROUP_CONCAT(CONCAT(COLUMN_NAME, ' ', COLUMN_TYPE) SEPARATOR ', '),
                       ')') as sql
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                GROUP BY TABLE_NAME
            """)
        elif vanna_manager.current_database == "mssql":
            # Generate proper CREATE TABLE statements with correct schema names
            return vn.run_sql("""
                SELECT 'table' as type,
                       'CREATE TABLE [Nodinite].[ods].[' + t.TABLE_NAME + '] (' +
                       STUFF((
                           SELECT ', ' + COLUMN_NAME + ' ' + 
                                  CASE 
                                      WHEN DATA_TYPE = 'varchar' THEN 'NVARCHAR(' + CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
                                      WHEN DATA_TYPE = 'nvarchar' THEN 'NVARCHAR(' + CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
                                      WHEN DATA_TYPE = 'decimal' THEN 'DECIMAL(' + CAST(NUMERIC_PRECISION AS VARCHAR) + ',' + CAST(NUMERIC_SCALE AS VARCHAR) + ')'
                                      WHEN DATA_TYPE = 'int' THEN 'INT'
                                      WHEN DATA_TYPE = 'date' THEN 'DATE'
                                      WHEN DATA_TYPE = 'datetime' THEN 'DATETIME'
                                      ELSE UPPER(DATA_TYPE)
                                  END +
                                  CASE WHEN IS_NULLABLE = 'NO' THEN ' NOT NULL' ELSE '' END
                           FROM INFORMATION_SCHEMA.COLUMNS c2
                           WHERE c2.TABLE_NAME = t.TABLE_NAME AND c2.TABLE_SCHEMA = t.TABLE_SCHEMA
                           ORDER BY ORDINAL_POSITION
                           FOR XML PATH('')
                       ), 1, 2, '') + ');' as sql
                FROM INFORMATION_SCHEMA.TABLES t
                WHERE t.TABLE_TYPE = 'BASE TABLE' AND t.TABLE_SCHEMA = 'ods'
                GROUP BY t.TABLE_NAME, t.TABLE_SCHEMA
            """)
    except Exception as e:
        print(f"Error getting schema info: {e}")
        return None

def get_table_names():
    """Get table names based on current database type"""
    try:
        if vanna_manager.current_database == "sqlite":
            return vn.run_sql("SELECT name FROM sqlite_master WHERE type='table'")
        elif vanna_manager.current_database == "postgresql":
            return vn.run_sql("SELECT table_name as name FROM information_schema.tables WHERE table_schema = 'public'")
        elif vanna_manager.current_database == "mysql":
            return vn.run_sql("SELECT table_name as name FROM information_schema.tables WHERE table_schema = DATABASE()")
        elif vanna_manager.current_database == "mssql":
            return vn.run_sql("SELECT TABLE_NAME as name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'ods'")
    except Exception as e:
        print(f"Error getting table names: {e}")
        return None

def get_allowed_tables() -> list:
    """Get list of allowed tables for training from config"""
    if hasattr(config, 'VANNA_ALLOWED_TABLES'):
        allowed_tables = config.VANNA_ALLOWED_TABLES
        
        # If it's a string and equals 'all', return all tables
        if isinstance(allowed_tables, str) and allowed_tables.lower() == 'all':
            return 'all'
        # If it's a list, return the list
        elif isinstance(allowed_tables, list):
            return allowed_tables
        else:
            return 'all'
    return 'all'

def is_table_allowed_for_training(table_name: str) -> bool:
    """Check if a table is allowed for training"""
    allowed_tables = get_allowed_tables()
    
    # If all tables are allowed
    if allowed_tables == 'all':
        return True
    
    # If specific tables are listed
    if isinstance(allowed_tables, list):
        return table_name in allowed_tables
    
    return True

def get_filtered_ddl_statements(df_ddl):
    """Filter DDL statements to include only allowed tables"""
    if df_ddl is None or df_ddl.empty:
        return df_ddl
    
    allowed_tables = get_allowed_tables()
    
    # If all tables are allowed, return original DDL
    if allowed_tables == 'all':
        return df_ddl
    
    # Filter DDL statements for allowed tables only
    if isinstance(allowed_tables, list):
        filtered_ddl = []
        for ddl in df_ddl['sql'].to_list():
            # Extract table name from DDL
            table_name = None
            if 'CREATE TABLE' in ddl.upper():
                lines = ddl.split('\n')
                for line in lines:
                    if 'CREATE TABLE' in line.upper():
                        # Extract table name (handle different formats like [schema].[table] or just table)
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.upper() == 'TABLE' and i + 1 < len(parts):
                                full_table_name = parts[i + 1].strip('[]"')
                                # Get just the table name (last part after dots)
                                table_name = full_table_name.split('.')[-1]
                                break
                        break
            
            # Include DDL if table is allowed
            if table_name and table_name in allowed_tables:
                filtered_ddl.append(ddl)
                print(f"Including DDL for allowed table: {table_name}")
            elif table_name:
                print(f"Skipping DDL for table: {table_name} (not in allowed list)")
        
        # Create new DataFrame with filtered DDL
        import pandas as pd
        return pd.DataFrame({'sql': filtered_ddl}) if filtered_ddl else pd.DataFrame()
    
    return df_ddl

# Auto-train on startup if enabled
if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
    print(f"Attempting to get schema information for {vanna_manager.current_database} database...")
    df_ddl = get_database_schema_info()
    
    if df_ddl is not None and not df_ddl.empty:
        # Check if training data already exists
        existing_training_data = vn.get_training_data()
        if existing_training_data.empty or config.VANNA_TRAIN_ON_STARTUP:
            print(f"Training Vanna with {vanna_manager.current_provider} provider on {vanna_manager.current_database} database...")
            
            # Filter DDL statements based on allowed tables
            filtered_ddl = get_filtered_ddl_statements(df_ddl)
            
            if filtered_ddl is not None and not filtered_ddl.empty:
                print(f"Training on {len(filtered_ddl)} filtered DDL statements...")
                # Train on filtered DDL statements
                for ddl in filtered_ddl['sql'].to_list():
                    try:
                        vanna_train(ddl=ddl)
                    except Exception as e:
                        print(f"Error training DDL: {e}")
            else:
                print("No DDL statements found for allowed tables.")
            
            # Get list of all tables and filter based on allowed tables
            tables_df = get_table_names()
            if tables_df is not None and not tables_df.empty:
                allowed_tables = get_allowed_tables()
                print(f"Allowed tables for training: {allowed_tables}")
                
                # Filter tables and train only on allowed ones
                for table_name in tables_df['name']:
                    # Check if this table is allowed for training
                    if not is_table_allowed_for_training(table_name):
                        print(f"Skipping table '{table_name}' - not in allowed tables list")
                        continue
                    
                    print(f"Training documentation for allowed table: {table_name}")
                    
                    # Build sample query based on database type
                    if vanna_manager.current_database == "mssql":
                        sample_query = f"SELECT TOP 5 * FROM [Nodinite].[ods].[{table_name}]"
                    elif vanna_manager.current_database in ["postgresql", "mysql"]:
                        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                    else:  # sqlite
                        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                    
                    print(f"Using query: {sample_query}")
                    
                    try:
                        sample_df = vn.run_sql(sample_query)
                        training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                        vanna_train(documentation=training_text)
                        print(f"Successfully trained documentation for table: {table_name}")
                    except Exception as e:
                        print(f"Error training table data for {table_name}: {e}")
            
            print("Training completed!")
        else:
            print(f"Training data already exists for {vanna_manager.current_provider}. Skipping training.")
    else:
        print(f"Could not retrieve schema information from {vanna_manager.current_database} database.")
else:
    print("Auto-training is disabled. Use vanna_train() function to manually train the model.")

# # Auto-train on startup if enabled
# if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
#     print(f"Attempting to get schema information for {vanna_manager.current_database} database...")
#     df_ddl = get_database_schema_info()
    
#     if df_ddl is not None and not df_ddl.empty:
#         # Check if training data already exists
#         existing_training_data = vn.get_training_data()
#         if existing_training_data.empty or config.VANNA_TRAIN_ON_STARTUP:
#             print(f"Training Vanna with {vanna_manager.current_provider} provider on {vanna_manager.current_database} database...")
            
#             # First, add database-specific system instructions for MSSQL
#             if vanna_manager.current_database == "mssql":
#                 system_instructions = """
#                 CRITICAL SQL GENERATION RULES FOR MICROSOFT SQL SERVER:
                
#                 1. ALWAYS use full three-part table names: [Nodinite].[ods].[TableName]
#                 2. NEVER use simple table names like 'Invoice' - always use '[Nodinite].[ods].[Invoice]'
#                 3. Use T-SQL syntax: SELECT TOP N ... instead of LIMIT N
#                 4. For date operations use: CAST(column_name AS DATE)
#                 5. Database: Nodinite, Schema: ods
                
#                 CORRECT EXAMPLES:
#                 - SELECT TOP 100 * FROM [Nodinite].[ods].[Invoice]
#                 - SELECT DUE_DATE FROM [Nodinite].[ods].[Invoice] WHERE INVOICE_ID = '12345'
#                 - SELECT i.*, il.* FROM [Nodinite].[ods].[Invoice] i JOIN [Nodinite].[ods].[Invoice_Line] il ON i.INVOICE_ID = il.INVOICE_ID
                
#                 WRONG EXAMPLES:
#                 - SELECT * FROM Invoice (NEVER do this)
#                 - SELECT * FROM ods.Invoice (incomplete)
#                 - SELECT * FROM Invoice LIMIT 10 (wrong syntax)
#                 """
#                 try:
#                     vanna_train(documentation=system_instructions)
#                     print("Trained system instructions for MSSQL")
#                 except Exception as e:
#                     print(f"Error training system instructions: {e}")
            
#             # Train on DDL statements
#             for ddl in df_ddl['sql'].to_list():
#                 try:
#                     vanna_train(ddl=ddl)
#                 except Exception as e:
#                     print(f"Error training DDL: {e}")
            
#             # # Add specific SQL examples with correct table naming for MSSQL
#             # if vanna_manager.current_database == "mssql":
#             #     example_queries = [
#             #         {
#             #             "question": "Show me all invoices",
#             #             "sql": "SELECT TOP 100 * FROM [Nodinite].[ods].[Invoice]"
#             #         },
#             #         {
#             #             "question": "What is the due date for invoice 00000363?",
#             #             "sql": "SELECT DUE_DATE FROM [Nodinite].[ods].[Invoice] WHERE INVOICE_ID = '00000363'"
#             #         },
#             #         {
#             #             "question": "List all supplier names",
#             #             "sql": "SELECT DISTINCT SUPPLIER_PARTY_NAME FROM [Nodinite].[ods].[Invoice]"
#             #         },
#             #         {
#             #             "question": "Show invoice line items for a specific invoice",
#             #             "sql": "SELECT * FROM [Nodinite].[ods].[Invoice_Line] WHERE INVOICE_ID = '00000363'"
#             #         },
#             #         {
#             #             "question": "Count total number of invoices",
#             #             "sql": "SELECT COUNT(*) FROM [Nodinite].[ods].[Invoice]"
#             #         },
#             #         {
#             #             "question": "Get invoice details with line items",
#             #             "sql": "SELECT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, il.ITEM_NAME FROM [Nodinite].[ods].[Invoice] i JOIN [Nodinite].[ods].[Invoice_Line] il ON i.INVOICE_ID = il.INVOICE_ID"
#             #         }
#             #     ]
                
#             #     for example in example_queries:
#             #         try:
#             #             vanna_train(question=example["question"], sql=example["sql"])
#             #             print(f"Trained example: {example['question']}")
#             #         except Exception as e:
#             #             print(f"Error training example query: {e}")
            
#             # # Get list of all tables
#             # tables_df = get_table_names()
#             # if tables_df is not None and not tables_df.empty:
#             #     # For each table, get distinct examples
#             #     for table_name in tables_df['name']:
#             #         # Use appropriate LIMIT syntax based on database type
#             #         if vanna_manager.current_database == "mssql":
#             #             sample_query = f"SELECT TOP 5 * FROM [Nodinite].[ods].[{table_name}]"
#             #         else:
#             #             sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                    
#             #         try:
#             #             sample_df = vn.run_sql(sample_query)
#             #             training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
#             #             vanna_train(documentation=training_text)
#             #         except Exception as e:
#             #             print(f"Error training table data for {table_name}: {e}")
            
#             print("Training completed!")
#         else:
#             print(f"Training data already exists for {vanna_manager.current_provider}. Skipping training.")
#     else:
#         print(f"Could not retrieve schema information from {vanna_manager.current_database} database.")
# else:
#     print("Auto-training is disabled. Use vanna_train() function to manually train the model.")

# Restore original print for our output
builtins.print = _original_print

print(f"\n🤖 Vanna SQL Assistant initialized with {vanna_manager.current_provider} provider")
print(f"📊 Current model: {get_vanna_info()['model']}")
print(f"🗄️  Current database: {vanna_manager.current_database.upper()}")