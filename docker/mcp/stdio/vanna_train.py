from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
import logging
import warnings
import builtins
from typing import Optional
import pymssql
import pandas as pd
import config

load_dotenv()

# Disable SSL certificate verification globally
import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http.client
import certifi

# Monkey patch http.client to disable SSL verification
original_https_connection_init = http.client.HTTPSConnection.__init__
def patched_https_connection_init(self, *args, **kwargs):
    kwargs['context'] = ssl._create_unverified_context()
    return original_https_connection_init(self, *args, **kwargs)
http.client.HTTPSConnection.__init__ = patched_https_connection_init

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables to disable SSL verification
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OPENAI_CA_BUNDLE"] = ""

# Monkey patch requests to disable SSL verification globally
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Also patch requests.request directly
original_requests_request = requests.request
def patched_requests_request(method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_requests_request(method, url, **kwargs)
requests.request = patched_requests_request

# Patch requests.get, post, etc.
for method in ['get', 'post', 'put', 'delete', 'head', 'options', 'patch']:
    original_method = getattr(requests, method)
    def make_patched_method(orig_method):
        def patched_method(url, **kwargs):
            kwargs.setdefault('verify', False)
            return orig_method(url, **kwargs)
        return patched_method
    setattr(requests, method, make_patched_method(original_method))

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
        
        # Force SSL bypass for OpenAI
        import ssl
        import urllib3
        import certifi
        
        # Multiple approaches to disable SSL
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Override certifi to return empty string
        original_where = certifi.where
        certifi.where = lambda: ""
        
        # Set OpenAI-specific environment variables
        os.environ["OPENAI_VERIFY_SSL"] = "false"
        
        try:
            # Try to import and patch OpenAI client directly
            import openai
            if hasattr(openai, '_client'):
                # Patch the OpenAI client's session
                if hasattr(openai._client, '_session'):
                    openai._client._session.verify = False
        except:
            pass
        
        VannaClass = self.get_vanna_class("openai")
        self.vanna_client = VannaClass(config={
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OPENAI_VERBOSE
        })
        
        # Additional post-initialization SSL bypass
        try:
            if hasattr(self.vanna_client, '_client') and hasattr(self.vanna_client._client, '_session'):
                self.vanna_client._client._session.verify = False
        except:
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
                cursor = connection.cursor()
                cursor.execute(sql)
                if sql.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    cursor.close()
                    # Convert to pandas DataFrame if results exist
                    if results:
                        import pandas as pd
                        return pd.DataFrame(results)
                    else:
                        import pandas as pd
                        return pd.DataFrame()
                else:
                    connection.commit()
                    cursor.close()
                    return None
            
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
            # Use a simpler approach to avoid STRING_AGG limitations and ambiguous column names
            return vn.run_sql("""
                SELECT 'table' as type,
                       'Table: ' + t.TABLE_NAME + ' - Columns: ' + 
                       CAST(COUNT(*) AS VARCHAR(10)) + ' columns including: ' +
                       STUFF((
                           SELECT TOP 10 ', ' + COLUMN_NAME + ' (' + DATA_TYPE + ')'
                           FROM INFORMATION_SCHEMA.COLUMNS c2
                           WHERE c2.TABLE_NAME = t.TABLE_NAME AND c2.TABLE_SCHEMA = t.TABLE_SCHEMA
                           ORDER BY ORDINAL_POSITION
                           FOR XML PATH('')
                       ), 1, 2, '') as sql
                FROM INFORMATION_SCHEMA.TABLES t
                JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
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

# Auto-train on startup if enabled
if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
    print(f"Attempting to get schema information for {vanna_manager.current_database} database...")
    df_ddl = get_database_schema_info()
    
    if df_ddl is not None and not df_ddl.empty:
        # Check if training data already exists
        existing_training_data = vn.get_training_data()
        if existing_training_data.empty or config.VANNA_TRAIN_ON_STARTUP:
            print(f"Training Vanna with {vanna_manager.current_provider} provider on {vanna_manager.current_database} database...")
            
            # Train on DDL statements
            for ddl in df_ddl['sql'].to_list():
                try:
                    vanna_train(ddl=ddl)
                except Exception as e:
                    print(f"Error training DDL: {e}")
            
            # Get list of all tables
            tables_df = get_table_names()
            if tables_df is not None and not tables_df.empty:
                # For each table, get distinct examples
                for table_name in tables_df['name']:
                    # Use appropriate LIMIT syntax based on database type
                    if vanna_manager.current_database == "mssql":
                        sample_query = f"SELECT TOP 5 * FROM [Nodinite].[ods].[{table_name}]"
                    else:
                        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                    
                    try:
                        sample_df = vn.run_sql(sample_query)
                        training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                        vanna_train(documentation=training_text)
                    except Exception as e:
                        print(f"Error training table data for {table_name}: {e}")
            
            print("Training completed!")
        else:
            print(f"Training data already exists for {vanna_manager.current_provider}. Skipping training.")
    else:
        print(f"Could not retrieve schema information from {vanna_manager.current_database} database.")
else:
    print("Auto-training is disabled. Use vanna_train() function to manually train the model.")

# Restore original print for our output
builtins.print = _original_print

print(f"\n🤖 Vanna SQL Assistant initialized with {vanna_manager.current_provider} provider")
print(f"📊 Current model: {get_vanna_info()['model']}")
print(f"🗄️  Current database: {vanna_manager.current_database.upper()}")