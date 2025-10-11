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
        print("ollama_host: ",config.VANNA_OLLAMA_BASE_URL)
        self.vanna_client = VannaClass(config={
            'model': config.VANNA_OLLAMA_MODEL,
            'ollama_host':config.VANNA_OLLAMA_BASE_URL, #"http://125.209.124.155:11434",#"http://localhost:11434",#config.VANNA_OLLAMA_BASE_URL,
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
    vanna_client,
    current_provider: str,
    ddl: Optional[str] = None,
    documentation: Optional[str] = None,
    question: Optional[str] = None,
    sql: Optional[str] = None
) -> None:
    """
    Train Vanna with different types of data
    
    Args:
        vanna_client: Vanna client instance
        current_provider: Current provider name
        ddl: Database schema DDL statements
        documentation: Documentation text
        question: Natural language question
        sql: Corresponding SQL query for the question
    """
    if not vanna_client:
        raise ValueError("Vanna client must be provided")
    
    # Train based on provided data
    if ddl:
        vanna_client.train(ddl=ddl)
        print(f"Trained DDL with {current_provider}")
    
    if documentation:
        vanna_client.train(documentation=documentation)
        print(f"Trained documentation with {current_provider}")
    
    if question and sql:
        vanna_client.train(question=question, sql=sql)
        print(f"Trained SQL pair with {current_provider}")
    
    
    if not any([ddl, documentation, (question and sql)]):
        raise ValueError("Must provide at least one training data type")

def get_vanna_info(vanna_manager) -> dict:
    """Get current Vanna configuration info"""
    if not vanna_manager:
        return {"error": "VannaModelManager not provided"}
        
    return {
        "provider": vanna_manager.get_current_provider(),
        "model": config.VANNA_OPENAI_MODEL if vanna_manager.current_provider == "openai" else config.VANNA_OLLAMA_MODEL,
        "database": vanna_manager.current_database,
        "database_connection": vanna_manager.database_connection,
        "initialized": vanna_manager.vanna_client is not None,
        "available_databases": vanna_manager.get_database_info()["available_databases"]
    }


def get_database_schema_info(vanna_client, current_database):
    """Get database schema information based on current database type"""
    try:
        if current_database == "sqlite":
            return vanna_client.run_sql("""
                SELECT type, 
                       CASE 
                           WHEN type = 'table' THEN SUBSTR(sql, INSTR(sql, 'TABLE ') + 6, 
                                                          CASE 
                                                              WHEN INSTR(SUBSTR(sql, INSTR(sql, 'TABLE ') + 6), ' ') > 0 
                                                              THEN INSTR(SUBSTR(sql, INSTR(sql, 'TABLE ') + 6), ' ') - 1
                                                              ELSE LENGTH(SUBSTR(sql, INSTR(sql, 'TABLE ') + 6))
                                                          END)
                           ELSE name
                       END as table_name,
                       sql 
                FROM sqlite_master 
                WHERE sql is not null
            """)
        elif current_database == "postgresql":
            return vanna_client.run_sql("""
                SELECT 'table' as type,
                       t.table_name,
                       'CREATE TABLE ' || t.table_schema || '.' || t.table_name || ' (' ||
                       string_agg(c.column_name || ' ' || c.data_type, ', ') || ')' as sql
                FROM information_schema.tables t
                JOIN information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
                WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                  AND t.table_type = 'BASE TABLE'
                GROUP BY t.table_schema, t.table_name
                ORDER BY t.table_schema, t.table_name
            """)
        elif current_database == "mysql":
            return vanna_client.run_sql("""
                SELECT 'table' as type,
                       TABLE_NAME as table_name,
                       CONCAT('CREATE TABLE ', TABLE_NAME, ' (',
                       GROUP_CONCAT(CONCAT(COLUMN_NAME, ' ', COLUMN_TYPE) SEPARATOR ', '),
                       ')') as sql
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                GROUP BY TABLE_NAME
            """)
        elif current_database == "mssql":
            # Generate proper CREATE TABLE statements with correct schema names
            return vanna_client.run_sql("""
                SELECT 'table' as type,
                       t.TABLE_NAME as table_name,
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

def get_table_names(vanna_client, current_database):
    """Get table names based on current database type"""
    try:
        if current_database == "sqlite":
            return vanna_client.run_sql("SELECT name FROM sqlite_master WHERE type='table'")
        elif current_database == "postgresql":
            return vanna_client.run_sql("SELECT table_name as name FROM information_schema.tables WHERE table_schema = 'public'")
        elif current_database == "mysql":
            return vanna_client.run_sql("SELECT table_name as name FROM information_schema.tables WHERE table_schema = DATABASE()")
        elif current_database == "mssql":
            return vanna_client.run_sql("SELECT TABLE_NAME as name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'ods'")
    except Exception as e:
        print(f"Error getting table names: {e}")
        return None

# Restore original print for our output
builtins.print = _original_print