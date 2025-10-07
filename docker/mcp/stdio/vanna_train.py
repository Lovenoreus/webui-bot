from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
import logging
import warnings
import builtins
from typing import Optional
import config
import ssl
import urllib3
import time

load_dotenv()

# Configure SSL handling based on config settings
def apply_comprehensive_ssl_bypass():
    """Apply SSL bypass at all levels to prevent certificate verification errors"""
    
    if not config.VANNA_SSL_BYPASS_ERRORS:
        return
        
    print("[VANNA DEBUG] Applying comprehensive SSL bypass for all API calls...")
    
    try:
        # 1. Global SSL context bypass
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # 2. Environment variables for SSL bypass
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        os.environ['SSL_CERT_DIR'] = ''
        
        # 3. Disable urllib3 warnings
        if config.VANNA_SSL_DISABLE_WARNINGS:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')
            warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
        
    except Exception as e:
        print(f"[VANNA DEBUG] Warning: Could not apply basic SSL bypass: {e}")
    
    # 4. Patch requests at multiple levels
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.ssl_ import create_urllib3_context
        
        # Patch the requests module directly
        original_request = requests.request
        def patched_request(*args, **kwargs):
            kwargs.setdefault('verify', False)
            kwargs.setdefault('timeout', 30)  # Add timeout to prevent hanging
            return original_request(*args, **kwargs)
        requests.request = patched_request
        
        # Patch session requests
        original_session_request = requests.Session.request
        def patched_session_request(self, *args, **kwargs):
            kwargs.setdefault('verify', False)
            kwargs.setdefault('timeout', 30)
            return original_session_request(self, *args, **kwargs)
        requests.Session.request = patched_session_request
        
        # Create custom SSL adapter
        class SSLBypassAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                try:
                    context = create_urllib3_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    kwargs['ssl_context'] = context
                except Exception:
                    pass  # Fallback to default if SSL context creation fails
                return super().init_poolmanager(*args, **kwargs)
        
        # Patch default session creation
        original_session_init = requests.Session.__init__
        def patched_session_init(self, *args, **kwargs):
            original_session_init(self, *args, **kwargs)
            try:
                self.mount('https://', SSLBypassAdapter())
                self.mount('http://', HTTPAdapter())
                self.verify = False
            except Exception:
                pass  # Fallback to default if mounting fails
        requests.Session.__init__ = patched_session_init
        
    except (ImportError, Exception) as e:
        print(f"[VANNA DEBUG] Warning: Could not patch requests library: {e}")
    
    # 5. Try to patch httpx if available (used by some newer libraries)
    try:
        import httpx
        
        # Patch httpx Client to disable SSL verification
        original_httpx_client_init = httpx.Client.__init__
        def patched_httpx_client_init(self, *args, **kwargs):
            kwargs.setdefault('verify', False)
            kwargs.setdefault('timeout', 30.0)
            return original_httpx_client_init(self, *args, **kwargs)
        httpx.Client.__init__ = patched_httpx_client_init
        
        # Patch httpx AsyncClient if available
        if hasattr(httpx, 'AsyncClient'):
            original_httpx_async_client_init = httpx.AsyncClient.__init__
            def patched_httpx_async_client_init(self, *args, **kwargs):
                kwargs.setdefault('verify', False)
                kwargs.setdefault('timeout', 30.0)
                return original_httpx_async_client_init(self, *args, **kwargs)
            httpx.AsyncClient.__init__ = patched_httpx_async_client_init
            
    except (ImportError, AttributeError, Exception) as e:
        # httpx might not be available or might have different attributes
        print(f"[VANNA DEBUG] Info: httpx patching skipped: {e}")
    
    print("[VANNA DEBUG] ✅ Comprehensive SSL bypass applied successfully")

# Apply SSL bypass immediately
apply_comprehensive_ssl_bypass()

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
    
    def _patch_ssl_for_api_calls(self):
        """Re-apply SSL patches before training (some libraries reset them)"""
        apply_comprehensive_ssl_bypass()
    
    def initialize_vanna(self, provider: Optional[str] = None):
        """Initialize Vanna with specified provider or use config default"""
        target_provider = provider or self.current_provider
        
        # Patch SSL verification for API calls
        self._patch_ssl_for_api_calls()
        
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
        
        # Apply comprehensive SSL bypass before initializing OpenAI client
        apply_comprehensive_ssl_bypass()
        
        VannaClass = self.get_vanna_class("openai")
        self.vanna_client = VannaClass(config={
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OPENAI_VERBOSE
        })
        self.current_provider = "openai"
    
    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        # Apply comprehensive SSL bypass before initializing Ollama client
        apply_comprehensive_ssl_bypass()
        
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
    
    def _handle_ssl_connection_error(self, error_msg: str, database_type: str) -> bool:
        """
        Check if error is SSL certificate related and should be handled with fallback
        
        Args:
            error_msg: Error message from database connection
            database_type: Type of database (postgresql, mysql, mssql)
            
        Returns:
            bool: True if this is a certificate error that should be retried with SSL fallback
        """
        ssl_error_indicators = [
            "certificate verify failed",
            "SSL",
            "certificate",
            "self-signed certificate",
            "certificate chain",
            "TLS",
            "ssl",
            "CERTIFICATE_VERIFY_FAILED"
        ]
        
        error_lower = error_msg.lower()
        is_ssl_error = any(indicator.lower() in error_lower for indicator in ssl_error_indicators)
        
        if is_ssl_error:
            print(f"[VANNA DEBUG] Detected SSL certificate error for {database_type}: {error_msg}")
            return True
        
        return False
    
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
        
        # Handle SSL configuration with fallback for certificate issues
        ssl_mode = config.VANNA_POSTGRESQL_SSL_MODE
        connection_string = f"postgresql://{config.VANNA_DB_USERNAME}:{config.VANNA_DB_PASSWORD}@{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}?sslmode={ssl_mode}"
        
        try:
            self.vanna_client.connect_to_postgres(
                host=config.VANNA_DB_HOST,
                dbname=config.VANNA_DB_DATABASE,
                user=config.VANNA_DB_USERNAME,
                password=config.VANNA_DB_PASSWORD,
                port=port,
                sslmode=ssl_mode
            )
        except Exception as e:
            if self._handle_ssl_connection_error(str(e), "postgresql"):
                # Retry with fallback SSL mode (typically 'require' which bypasses certificate verification)
                fallback_ssl_mode = config.VANNA_POSTGRESQL_SSL_FALLBACK_MODE
                connection_string = f"postgresql://{config.VANNA_DB_USERNAME}:{config.VANNA_DB_PASSWORD}@{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}?sslmode={fallback_ssl_mode}"
                self.vanna_client.connect_to_postgres(
                    host=config.VANNA_DB_HOST,
                    dbname=config.VANNA_DB_DATABASE,
                    user=config.VANNA_DB_USERNAME,
                    password=config.VANNA_DB_PASSWORD,
                    port=port,
                    sslmode=fallback_ssl_mode
                )
            else:
                raise
        
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
        
        # MySQL SSL configuration - use config values or fallback for certificate issues
        ssl_verify_cert = config.VANNA_MYSQL_SSL_VERIFY_CERT
        ssl_verify_identity = config.VANNA_MYSQL_SSL_VERIFY_IDENTITY
        ssl_disabled = config.VANNA_MYSQL_SSL_DISABLED
        
        try:
            self.vanna_client.connect_to_mysql(
                host=config.VANNA_DB_HOST,
                database=config.VANNA_DB_DATABASE,
                user=config.VANNA_DB_USERNAME,
                password=config.VANNA_DB_PASSWORD,
                port=port,
                ssl_verify_cert=ssl_verify_cert,
                ssl_verify_identity=ssl_verify_identity,
                ssl_disabled=ssl_disabled
            )
        except Exception as e:
            if self._handle_ssl_connection_error(str(e), "mysql"):
                # Retry with SSL but without certificate verification
                self.vanna_client.connect_to_mysql(
                    host=config.VANNA_DB_HOST,
                    database=config.VANNA_DB_DATABASE,
                    user=config.VANNA_DB_USERNAME,
                    password=config.VANNA_DB_PASSWORD,
                    port=port,
                    ssl_verify_cert=False,
                    ssl_verify_identity=False,
                    ssl_disabled=False  # Still use SSL but without verification
                )
            else:
                raise
        
        self.database_connection = f"mysql://{config.VANNA_DB_USERNAME}:***@{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}"
    
    def _connect_to_mssql(self):
        """Connect to Microsoft SQL Server database"""
        if not config.VANNA_MSSQL_ENABLED:
            raise ValueError("Microsoft SQL Server is not enabled in configuration")
        
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc package is required for SQL Server connection. Install with: pip install pyodbc")
        
        # Build connection string for SQL Server
        port = self.get_db_port('mssql')
        
        # Build connection string with SSL configuration from config
        ssl_options = ""
        if config.VANNA_MSSQL_TRUST_SERVER_CERTIFICATE:
            ssl_options += "TrustServerCertificate=yes;"
        if config.VANNA_MSSQL_ENCRYPT:
            ssl_options += "Encrypt=yes;"
        
        if config.VANNA_MSSQL_TRUSTED_CONNECTION:
            connection_string = f"Driver={{{config.VANNA_MSSQL_DRIVER}}};Server={config.VANNA_DB_HOST},{port};Database={config.VANNA_DB_DATABASE};Trusted_Connection=yes;{ssl_options}"
        else:
            connection_string = f"Driver={{{config.VANNA_MSSQL_DRIVER}}};Server={config.VANNA_DB_HOST},{port};Database={config.VANNA_DB_DATABASE};UID={config.VANNA_DB_USERNAME};PWD={config.VANNA_DB_PASSWORD};{ssl_options}"
        
        try:
            self.vanna_client.connect_to_mssql(odbc_conn_str=connection_string)
        except Exception as e:
            if self._handle_ssl_connection_error(str(e), "mssql"):
                # Retry with TrustServerCertificate=yes to bypass certificate verification
                if config.VANNA_MSSQL_TRUSTED_CONNECTION:
                    connection_string_ssl_bypass = f"Driver={{{config.VANNA_MSSQL_DRIVER}}};Server={config.VANNA_DB_HOST},{port};Database={config.VANNA_DB_DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;Encrypt=yes;"
                else:
                    connection_string_ssl_bypass = f"Driver={{{config.VANNA_MSSQL_DRIVER}}};Server={config.VANNA_DB_HOST},{port};Database={config.VANNA_DB_DATABASE};UID={config.VANNA_DB_USERNAME};PWD={config.VANNA_DB_PASSWORD};TrustServerCertificate=yes;Encrypt=yes;"
                
                self.vanna_client.connect_to_mssql(odbc_conn_str=connection_string_ssl_bypass)
                connection_string = connection_string_ssl_bypass
            else:
                raise
        
        self.database_connection = f"mssql://{config.VANNA_DB_HOST}:{port}/{config.VANNA_DB_DATABASE}"
    
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
    
    def test_database_connection(self, database_type: Optional[str] = None) -> dict:
        """
        Test database connectivity and return detailed results
        
        Args:
            database_type: Database type to test, or None for current active database
            
        Returns:
            dict: Connection test results with status, message, and SSL info
        """
        target_database = database_type or self.current_database
        result = {
            "database_type": target_database,
            "success": False,
            "message": "",
            "ssl_used": False,
            "ssl_fallback_used": False,
            "connection_string": ""
        }
        
        try:
            # Temporarily store current database
            original_database = self.current_database
            
            # Test connection
            self.connect_to_database(target_database)
            
            result["success"] = True
            result["message"] = f"Successfully connected to {target_database}"
            result["connection_string"] = self.database_connection
            
            # Check if SSL is being used based on connection string
            if target_database == "postgresql":
                result["ssl_used"] = "sslmode=disable" not in self.database_connection
                result["ssl_fallback_used"] = "sslmode=require" in self.database_connection
            elif target_database == "mysql":
                result["ssl_used"] = "ssl_disabled=true" not in str(self.database_connection)
            elif target_database == "mssql":
                result["ssl_used"] = "Encrypt=yes" in self.database_connection
                result["ssl_fallback_used"] = "TrustServerCertificate=yes" in self.database_connection
            
            # Restore original database
            if original_database != target_database:
                self.connect_to_database(original_database)
                
        except Exception as e:
            result["success"] = False
            result["message"] = f"Failed to connect to {target_database}: {str(e)}"
            
            # Check if it's an SSL-related error
            if self._handle_ssl_connection_error(str(e), target_database):
                result["message"] += " (SSL certificate error detected)"
        
        return result
    
    def get_ssl_configuration_summary(self) -> dict:
        """Get summary of SSL configuration for all database types"""
        return {
            "postgresql": {
                "ssl_mode": config.VANNA_POSTGRESQL_SSL_MODE,
                "ssl_fallback_mode": config.VANNA_POSTGRESQL_SSL_FALLBACK_MODE,
                "ssl_verify_cert": config.VANNA_POSTGRESQL_SSL_VERIFY_CERT
            },
            "mysql": {
                "ssl_verify_cert": config.VANNA_MYSQL_SSL_VERIFY_CERT,
                "ssl_verify_identity": config.VANNA_MYSQL_SSL_VERIFY_IDENTITY,
                "ssl_disabled": config.VANNA_MYSQL_SSL_DISABLED
            },
            "mssql": {
                "trust_server_certificate": config.VANNA_MSSQL_TRUST_SERVER_CERTIFICATE,
                "encrypt": config.VANNA_MSSQL_ENCRYPT,
                "driver": config.VANNA_MSSQL_DRIVER
            },
            "sqlite": {
                "note": "SQLite uses local files and does not require SSL configuration"
            }
        }

# Convenience functions for easy usage
def vanna_train(
    ddl: Optional[str] = None,
    documentation: Optional[str] = None,
    question: Optional[str] = None,
    sql: Optional[str] = None
) -> bool:
    """
    Train Vanna with different types of data with SSL error handling
    
    Args:
        ddl: Database schema DDL statements
        documentation: Documentation text
        question: Natural language question
        sql: Corresponding SQL query for the question
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    global vanna_manager
    
    # Ensure Vanna is initialized
    if not vanna_manager.vanna_client:
        vanna_manager.initialize_vanna()
    
    def _safe_train(train_func, data, data_type):
        """Safely execute training with SSL error handling and retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Re-apply SSL bypass before each attempt
                if attempt > 0:
                    print(f"[VANNA DEBUG] Retry attempt {attempt + 1} for {data_type} training")
                    apply_comprehensive_ssl_bypass()
                    time.sleep(1)  # Brief delay between retries
                
                train_func(data)
                print(f"✅ Trained {data_type} with {vanna_manager.current_provider}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(term in error_msg for term in ["certificate verify failed", "ssl", "certificate", "handshake"]):
                    print(f"[VANNA DEBUG] SSL error during {data_type} training (attempt {attempt + 1}): {str(e)}")
                    
                    if attempt == max_retries - 1:
                        print(f"❌ Failed to train {data_type} due to SSL certificate error after {max_retries} attempts")
                        return False
                    else:
                        print(f"🔄 Retrying {data_type} training...")
                        continue
                else:
                    print(f"❌ Error training {data_type}: {e}")
                    return False
        
        return False
    
    success = True
    
    # Train based on provided data
    if ddl:
        if not _safe_train(lambda x: vanna_manager.vanna_client.train(ddl=x), ddl, "DDL"):
            success = False
    
    if documentation:
        if not _safe_train(lambda x: vanna_manager.vanna_client.train(documentation=x), documentation, "documentation"):
            success = False
    
    if question and sql:
        if not _safe_train(lambda x: vanna_manager.vanna_client.train(question=x[0], sql=x[1]), (question, sql), "SQL pair"):
            success = False
    
    if not any([ddl, documentation, (question and sql)]):
        raise ValueError("Must provide at least one training data type")
    
    return success





def get_vanna_info() -> dict:
    """Get current Vanna configuration info"""
    return {
        "provider": vanna_manager.get_current_provider(),
        "model": config.VANNA_OPENAI_MODEL if vanna_manager.current_provider == "openai" else config.VANNA_OLLAMA_MODEL,
        "database": vanna_manager.current_database,
        "database_connection": vanna_manager.database_connection,
        "initialized": vanna_manager.vanna_client is not None,
        "available_databases": vanna_manager.get_database_info()["available_databases"],
        "ssl_configuration": vanna_manager.get_ssl_configuration_summary()
    }

def test_all_database_connections() -> dict:
    """Test connectivity to all enabled databases"""
    available_databases = vanna_manager.get_database_info()["available_databases"]
    enabled_databases = [db for db, enabled in available_databases.items() if enabled]
    
    results = {}
    for db_type in enabled_databases:
        results[db_type] = vanna_manager.test_database_connection(db_type)
    
    return results

def diagnose_ssl_issues() -> dict:
    """Diagnose SSL configuration and potential issues"""
    diagnosis = {
        "ssl_configuration": vanna_manager.get_ssl_configuration_summary(),
        "connection_tests": test_all_database_connections(),
        "recommendations": []
    }
    
    # Analyze results and provide recommendations
    for db_type, test_result in diagnosis["connection_tests"].items():
        if not test_result["success"]:
            if "certificate" in test_result["message"].lower() or "ssl" in test_result["message"].lower():
                diagnosis["recommendations"].append(
                    f"{db_type.upper()}: SSL certificate issue detected. "
                    f"Consider updating SSL configuration or using SSL bypass options."
                )
            else:
                diagnosis["recommendations"].append(
                    f"{db_type.upper()}: Connection failed. Check database credentials and network connectivity."
                )
        elif test_result["ssl_fallback_used"]:
            diagnosis["recommendations"].append(
                f"{db_type.upper()}: Using SSL fallback mode. "
                f"Consider installing proper SSL certificates for better security."
            )
    
    return diagnosis

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
            return vn.run_sql("""
                SELECT 'table' as type,
                       'CREATE TABLE ' + t.TABLE_NAME + ' (' +
                       STRING_AGG(c.COLUMN_NAME + ' ' + c.DATA_TYPE, ', ') + ')' as sql
                FROM INFORMATION_SCHEMA.TABLES t
                JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_TYPE = 'BASE TABLE'
                GROUP BY t.TABLE_NAME
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
            return vn.run_sql("SELECT TABLE_NAME as name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
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
            
            # Re-apply SSL bypass before training starts
            apply_comprehensive_ssl_bypass()
            
            successful_trainings = 0
            total_trainings = 0
            
            # Train on DDL statements
            for ddl in df_ddl['sql'].to_list():
                total_trainings += 1
                try:
                    print(f"Adding ddl: {ddl[:100]}...")
                    if vanna_train(ddl=ddl):
                        successful_trainings += 1
                except Exception as e:
                    print(f"Error training DDL: {e}")
            
            # Get list of all tables
            tables_df = get_table_names()
            if tables_df is not None and not tables_df.empty:
                # For each table, get distinct examples
                for table_name in tables_df['name']:
                    total_trainings += 1
                    sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                    try:
                        sample_df = vn.run_sql(sample_query)
                        training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                        print("Adding documentation....")
                        if vanna_train(documentation=training_text):
                            successful_trainings += 1
                    except Exception as e:
                        print(f"Error training table data for {table_name}: {e}")
            
            print(f"Training completed! Successfully trained {successful_trainings}/{total_trainings} items.")
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

# Show SSL configuration summary
ssl_config = vanna_manager.get_ssl_configuration_summary()
current_db_ssl = ssl_config.get(vanna_manager.current_database, {})
if current_db_ssl and vanna_manager.current_database != "sqlite":
    print(f"🔒 SSL Configuration: {current_db_ssl}")

print("\n💡 Available commands:")
print("  • Ask any SQL question")
print("  • Type 'info' to see current configuration")
print("  • Type 'train' to manually train with sample data")
print("  • Type 'switch' to switch between databases")
print("  • Type 'databases' to see available databases")
print("  • Type 'ssl' to diagnose SSL configuration")
print("  • Type 'test' to test database connections")
print("  • Type 'exit' to quit")
print("  • To switch providers, edit config.json and restart\n")

# while True:
#     user_q = input("❓ Question (or command): ").strip()
    
#     if user_q.lower() == 'exit':
#         break
#     elif user_q.lower() == 'info':
#         info = get_vanna_info()
#         print(f"\n📋 Current Configuration:")
#         print(f"  Provider: {info['provider']}")
#         print(f"  Model: {info['model']}")
#         print(f"  Database: {info['database'].upper()}")
#         print(f"  Connection: {info['database_connection']}")
#         print(f"  Initialized: {info['initialized']}")
#         print(f"  Available Databases: {', '.join([db.upper() for db, enabled in info['available_databases'].items() if enabled])}")
#         print()
#         continue

#     elif user_q.lower() == 'databases':
#         info = get_vanna_info()
#         print(f"\n🗄️  Database Status:")
#         for db_name, enabled in info['available_databases'].items():
#             status = "✅ ENABLED" if enabled else "❌ DISABLED"
#             current_marker = " (current)" if db_name == info['database'] else ""
#             print(f"  {db_name.upper()}: {status}{current_marker}")
#         print()
#         continue

#     elif user_q.lower() == 'switch':
#         switch_database_interactive()
#         print()
#         continue

#     elif user_q.lower() == 'train':
#         print("🔄 Starting manual training...")
#         try:
#             df_ddl = get_database_schema_info()
#             if df_ddl is not None and not df_ddl.empty:
#                 for ddl in df_ddl['sql'].to_list()[:3]:  # Train on first 3 DDL statements
#                     vanna_train(ddl=ddl)
#                 print("✅ Manual training completed!")
#             else:
#                 print("❌ Could not retrieve schema information for training")
#         except Exception as e:
#             print(f"❌ Training error: {e}")
#         print()
#         continue

#     # Use filtered print during ask
#     builtins.print = filtered_print
#     try:
#         allow_llm_to_see_data = config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA if vanna_manager.current_provider == "openai" else config.VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA
#         sql, df, fig = vn.ask(question=user_q, print_results=False, allow_llm_to_see_data=allow_llm_to_see_data, visualize=False)
#         builtins.print = _original_print

#         print(f"\n🔍 Generated SQL ({vanna_manager.current_provider} + {vanna_manager.current_database.upper()}):")
#         print(sql)
#         print("\n📊 Results:")
#         print(df)
#         print()
#     except Exception as e:
#         builtins.print = _original_print
#         print(f"❌ Error: {e}")
#         print()
# from vanna.openai import OpenAI_Chat
# from vanna.qdrant import Qdrant_VectorStore
# # from qdrant_client import QdrantClient
# from vanna.chromadb import ChromaDB_VectorStore
# from dotenv import load_dotenv
# import os
# load_dotenv()




# class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
#     def __init__(self, config=None):
#         # Qdrant_VectorStore.__init__(self, config=config)
#         ChromaDB_VectorStore.__init__(self, config=config)
#         OpenAI_Chat.__init__(self, config=config)Which suppliers have sent the highest total invoice amounts?

# import logging
# import warnings
# import os

# # Suppress tokenizer warnings
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Suppress all warnings
# warnings.filterwarnings('ignore')

# # Configure logging to only show errors
# logging.basicConfig(level=logging.ERROR)

# vn = MyVanna(config={
#     'api_key': os.getenv("OPENAI_API_KEY"),
#     'model': "gpt-4o-mini",
#     'allow_llm_to_see_data': True,
#     'verbose': False  # Disable verbose logging
# })


# # vn.connect_to_sqlite('compacted.db')
# # Determine the full path to compacted.db (same folder as this script)
# db_path = os.path.join(os.path.dirname(__file__), "compacted.db")

# # Connect to your local SQLite database
# vn.connect_to_sqlite(db_path)



# df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
# # print(df_ddl)

# # Check if training data already exists
# existing_training_data = vn.get_training_data()
# if existing_training_data.empty:
#     print("No existing training data found. Starting training...")
#     # Train on DDL statements
#     for ddl in df_ddl['sql'].to_list():
#         vn.train(ddl=ddl)
#     print("DDL training completed.")
# else:
#     print("Training data already exists. Skipping DDL training.")

# # Get list of all tables
# tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
# tables_df = vn.run_sql(tables_query)

# # For each table, get distinct examples
# if existing_training_data.empty:
#     print("Starting table data training...")
#     for table_name in tables_df['name']:
#         # Get sample of distinct records from each table
#         sample_query = f"""
#         SELECT DISTINCT * FROM {table_name} 
#         LIMIT 5
#         """
#         try:
#             sample_df = vn.run_sql(sample_query)
            
#             # Create a training example showing the data pattern
#             training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
#             vn.train(documentation=training_text)
            
#             print(f"Added training examples for table: {table_name}")
#         except Exception as e:
#             print(f"Error processing table {table_name}: {e}")
#     print("Table data training completed.")
# else:
#     print("Training data already exists. Skipping table data training.")

# # The following are methods for adding training data. Make sure you modify the examples to match your database.

# # DDL statements are powerful because they specify table names, colume names, types, and potentially relationships
# # vn.train(ddl="""
# #     CREATE TABLE IF NOT EXISTS my-table (
# #         id INT PRIMARY KEY,
# #         name VARCHAR(100),
# #         age INT
# #     )
# # """)

# # Sometimes you may want to add documentation about your business terminology or definitions.
# # vn.train(documentation="Our business defines OTIF score as the percentage of orders that are delivered on time and in full")

# # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
# # vn.train(sql="SELECT * FROM my-table WHERE name = 'John Doe'")


# # At any time you can inspect what training data the package is able to reference
# training_data = vn.get_training_data()
# # print("Training Data:",training_data)

# # You can remove training data if there's obsolete/incorrect information. 
# # vn.remove_training_data(id='1-ddl')


# ## Now you can ask questions about your database and the package will use the training data to help generate SQL queries.
# # while True:
# #     question = input("Enter your question (or 'exit' to quit): ")
# #     if question.lower() == 'exit':
# #         break
# #     # answer = vn.ask(question)
# #     # print("Answer:", answer)
# #     question = "Show me total sales by region for last quarter."

# # # Generate SQL
# # sql = vn.generate_sql(question)
# # print("Generated SQL:")
# # print(sql)

# # # Execute it
# # df = run_sql(sql)
# # print("Results:")
# # print(df)

# while True:
#     user_q = input("Question (or 'exit'): ").strip()
#     if user_q.lower() == 'exit':
#         break

#     # Ask, but don’t worry about figure
#     sql, df, fig = vn.ask(question=user_q, print_results=False, allow_llm_to_see_data=True)
#     # vn.ask returns: (sql, df, fig, followups) as per docs :contentReference[oaicite:0]{index=0}

#     print("Generated SQL:")
#     print(sql)
#     print("\nAnswer (results):")
#     print(df)
