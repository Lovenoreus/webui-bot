import os
import json
import argparse
from tqdm import tqdm

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.schema import Document
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
import config

# Load configuration from JSON file
def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Could not load config.json: {e}")
        return {}

# Load the configuration
app_config = load_config()
print(f"App version: {app_config.get('app', {}).get('version', 'Unknown')}")

# Determine cosmic database collection name
cosmic_config = app_config.get('cosmic_database', {})
collection_base_name = cosmic_config.get('collection_name', 'cosmic_documents_group2')
embedding_model = app_config.get('openai', {}).get('embeddings_model_name', 'text-embedding-3-large')
cosmic_collection_name = f"{collection_base_name}-{embedding_model}"
print(f"Using cosmic database collection name: {cosmic_collection_name}")

# Check database type
vector_db_config = cosmic_config.get('vector_database', {})
use_pgvector = vector_db_config.get('use_pgvector', True)
use_qdrant = vector_db_config.get('use_qdrant', False)

if use_pgvector:
    print("Cosmic database will use: PostgreSQL/pgvector")
elif use_qdrant:
    print("Cosmic database will use: Qdrant")
else:
    print("No vector database configured")

# Check if RAG is enabled in features
features = app_config.get('features', {})
rag_enabled = features.get('cosmic_agent', False)
print(f"RAG configuration: enabled={rag_enabled}")

# Enhanced configuration classes with multi-provider support
class PostgresSettings:
    pg_host = os.getenv("PG_HOST", "localhost")  # Use "postgres" when running in Docker
    pg_port = os.getenv("PG_PORT", "5432")
    pg_database = os.getenv("PG_DATABASE", "vectordb")
    pg_username = os.getenv("PG_USERNAME", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "password")

class OllamaSettings:
    llm_model = os.getenv("OLLAMA_LLM_MODEL")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    direct_url = os.getenv("OLLAMA_BASE_URL")  

class OpenAISettings:
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

class Settings:
    def __init__(self):
        self.postgres = PostgresSettings()
        self.ollama = OllamaSettings()
        self.openai = OpenAISettings()
        
        # Check configuration from config.py if available
        try:
            self.use_ollama = getattr(config, 'USE_OLLAMA', False)
            self.use_openai = getattr(config, 'USE_OPENAI', False)
            self.use_mistral = getattr(config, 'USE_MISTRAL', False)
        except:
            # Fallback to environment variables
            self.use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
            self.use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
            self.use_mistral = os.getenv("USE_MISTRAL", "false").lower() == "true"

# Initialize settings
settings = Settings()
print(f"✅ Configuration loaded:")
print(f"   - Database: {settings.postgres.pg_host}:{settings.postgres.pg_port}/{settings.postgres.pg_database}")
print(f"   - Ollama: {settings.ollama.direct_url}")
print(f"   - LLM Model: {settings.ollama.llm_model}")
print(f"   - Embedding Model: {settings.ollama.embedding_model}")
print(f"   - Provider settings: Ollama={settings.use_ollama}, OpenAI={settings.use_openai}, Mistral={settings.use_mistral}")

# Build connection string
CONNECTION_STR = (
    f"postgresql+psycopg2://"
    f"{quote_plus(settings.postgres.pg_username)}:"
    f"{quote_plus(settings.postgres.pg_password)}@"
    f"{settings.postgres.pg_host}:"
    f"{settings.postgres.pg_port}/"
    f"{settings.postgres.pg_database}"
)

# Initialize models based on configuration with error handling
llm = None
embeddings = None

def initialize_models():
    """Initialize LLM and embedding models based on configuration"""
    global llm, embeddings
    
    print("🚀 Initializing AI models...")
    print(f"📊 Provider status: OpenAI={settings.use_openai}, Ollama={settings.use_ollama}, Mistral={settings.use_mistral}")
    
    # Initialize LLM - prioritize based on configuration
    if settings.use_openai and settings.openai.api_key:
        try:
            print(f"🔧 Initializing OpenAI LLM with model: {settings.openai.llm_model}")
            print(f"🔗 Base URL: {settings.openai.api_base_url}")
            llm = ChatOpenAI(
                model=settings.openai.llm_model,
                api_key=settings.openai.api_key,
                base_url=settings.openai.api_base_url,
                temperature=0.0
            )
            print(f"✅ OpenAI LLM initialized: {settings.openai.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing OpenAI LLM: {e}")
            
    elif settings.use_ollama:
        try:
            print(f"🔧 Initializing Ollama LLM with model: {settings.ollama.llm_model}")
            print(f"🔗 Base URL: {settings.ollama.direct_url}")
            # Test Ollama connectivity first
            import requests
            try:
                response = requests.get(f"{settings.ollama.direct_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    available_models = [model['name'] for model in response.json().get('models', [])]
                    print(f"📋 Available Ollama models: {available_models}")
                    if settings.ollama.llm_model not in available_models:
                        print(f"⚠️  Warning: Model '{settings.ollama.llm_model}' not found in available models")
                else:
                    print(f"⚠️  Warning: Ollama API returned status {response.status_code}")
            except requests.exceptions.RequestException as req_e:
                print(f"⚠️  Warning: Could not connect to Ollama API: {req_e}")
                print(f"💡 Make sure Ollama is running at: {settings.ollama.direct_url}")
            
            llm = OllamaLLM(
                model=settings.ollama.llm_model,
                base_url=settings.ollama.direct_url,
                temperature=0.0
            )
            print(f"✅ Ollama LLM initialized: {settings.ollama.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing Ollama LLM: {e}")
    
    # Initialize embeddings - prioritize based on configuration
    if settings.use_openai and settings.openai.api_key:
        try:
            print(f"🔧 Initializing OpenAI embeddings with model: {settings.openai.embedding_model}")
            print(f"🔗 Base URL: {settings.openai.api_base_url}")
            # Test API key validity
            test_headers = {"Authorization": f"Bearer {settings.openai.api_key}"}
            import requests
            try:
                test_response = requests.get(f"{settings.openai.api_base_url}/models", 
                                           headers=test_headers, timeout=10)
                if test_response.status_code == 200:
                    print("✅ OpenAI API key validation successful")
                else:
                    print(f"⚠️  Warning: OpenAI API returned status {test_response.status_code}")
            except requests.exceptions.RequestException as req_e:
                print(f"⚠️  Warning: Could not validate OpenAI API key: {req_e}")
            
            embeddings = OpenAIEmbeddings(
                model=settings.openai.embedding_model,
                api_key=settings.openai.api_key,
                # base_url=settings.openai.api_base_url
                # Removed timeout and max_retries as they are not valid arguments
            )
            print(f"✅ OpenAI embeddings initialized: {settings.openai.embedding_model}")
        except Exception as e:
            print(f"❌ Error initializing OpenAI embeddings: {e}")
            # Fallback to Ollama if available
            if settings.use_ollama:
                print("🔄 Falling back to Ollama embeddings...")
                try:
                    embeddings = OllamaEmbeddings(
                        model=settings.ollama.embedding_model, 
                        base_url=settings.ollama.direct_url
                    )
                    print(f"✅ Ollama embeddings initialized as fallback: {settings.ollama.embedding_model}")
                except Exception as fallback_e:
                    print(f"❌ Error initializing Ollama embeddings fallback: {fallback_e}")
                    
    elif settings.use_ollama:
        try:
            print(f"🔧 Initializing Ollama embeddings with model: {settings.ollama.embedding_model}")
            print(f"🔗 Base URL: {settings.ollama.direct_url}")
            # Test embedding model availability
            import requests
            try:
                response = requests.get(f"{settings.ollama.direct_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    available_models = [model['name'] for model in response.json().get('models', [])]
                    if settings.ollama.embedding_model not in available_models:
                        print(f"⚠️  Warning: Embedding model '{settings.ollama.embedding_model}' not found")
                        print(f"📋 Available models: {available_models}")
                        print(f"💡 You may need to pull the model: ollama pull {settings.ollama.embedding_model}")
            except requests.exceptions.RequestException:
                pass  # Already warned above
                
            embeddings = OllamaEmbeddings(
                model=settings.ollama.embedding_model, 
                base_url=settings.ollama.direct_url
            )
            print(f"✅ Ollama embeddings initialized: {settings.ollama.embedding_model}")
        except Exception as e:
            print(f"❌ Error initializing Ollama embeddings: {e}")
            print(f"💡 Troubleshooting steps:")
            print(f"   1. Check if Ollama is running: curl {settings.ollama.direct_url}/api/tags")
            print(f"   2. Pull the embedding model: ollama pull {settings.ollama.embedding_model}")
            print(f"   3. Check network connectivity to {settings.ollama.direct_url}")
    
    # Verify at least one model is initialized
    if llm is None:
        print("❌ No LLM model could be initialized")
        print("💡 Check your configuration in config.json or environment variables")
        exit(1)
    if embeddings is None:
        print("❌ No embedding model could be initialized")
        print("💡 Check your configuration and service availability")
        exit(1)
        
    print("🎉 Model initialization completed successfully!")

# Initialize models
initialize_models()


def check_collection_exists(collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Check if a collection exists in the PGVector database"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=settings.postgres.pg_host,
            port=settings.postgres.pg_port,
            database=settings.postgres.pg_database,
            user=settings.postgres.pg_username,
            password=settings.postgres.pg_password
        )
        cursor = conn.cursor()
        
        # Check if the langchain_pg_collection table exists and has our collection
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_collection'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            cursor.close()
            conn.close()
            return False, "PGVector tables not initialized"
        
        cursor.execute("""
            SELECT COUNT(*) FROM langchain_pg_collection 
            WHERE name = %s;
        """, (collection_name,))
        collection_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        if collection_count > 0:
            return True, f"Collection '{collection_name}' exists with documents"
        else:
            return False, f"Collection '{collection_name}' does not exist"
            
    except Exception as e:
        return False, f"Error checking collection: {e}"


def log_collection_status(collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Log the status of a collection during startup"""
    exists, message = check_collection_exists(collection_name)
    
    if exists:
        print(f"✅ Collection Status: {message}")
    else:
        print(f"⚠️ Collection Status: {message}")
        print(f"💡 To initialize this collection, run:")
        print(f"   python rag2.py --embed <path_to_json_files> --collection {collection_name}")
        
        # Check provider-specific advice
        if settings.use_openai and settings.openai.api_key:
            print(f"🔧 Using OpenAI embeddings: {settings.openai.embedding_model}")
        elif settings.use_ollama:
            print(f"🔧 Using Ollama embeddings: {settings.ollama.embedding_model}")
            print(f"   Make sure Ollama is running at: {settings.ollama.direct_url}")
        else:
            print("❌ No embedding provider configured!")

# Check default collection at startup
log_collection_status()


def diagnose_connection_issues():
    """Diagnose potential connection issues with embedding providers"""
    print("\n🔍 Running connection diagnostics...")
    
    # Check Ollama connectivity
    if settings.use_ollama:
        print(f"\n📡 Testing Ollama connection to {settings.ollama.direct_url}")
        try:
            import requests
            response = requests.get(f"{settings.ollama.direct_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✅ Ollama is accessible with {len(models)} models")
                model_names = [model['name'] for model in models]
                if settings.ollama.embedding_model in model_names:
                    print(f"✅ Embedding model '{settings.ollama.embedding_model}' is available")
                else:
                    print(f"❌ Embedding model '{settings.ollama.embedding_model}' not found")
                    print(f"📋 Available models: {model_names}")
                    print(f"💡 Run: ollama pull {settings.ollama.embedding_model}")
            else:
                print(f"❌ Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {settings.ollama.direct_url}")
            print("💡 Make sure Ollama is running and accessible")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
    
    # Check OpenAI connectivity
    if settings.use_openai and settings.openai.api_key:
        print(f"\n📡 Testing OpenAI connection to {settings.openai.api_base_url}")
        try:
            import requests
            headers = {"Authorization": f"Bearer {settings.openai.api_key}"}
            response = requests.get(f"{settings.openai.api_base_url}/models", 
                                  headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ OpenAI API is accessible")
                models = response.json().get('data', [])
                model_ids = [model['id'] for model in models]
                if settings.openai.embedding_model in model_ids:
                    print(f"✅ Embedding model '{settings.openai.embedding_model}' is available")
                else:
                    print(f"⚠️  Embedding model '{settings.openai.embedding_model}' not found in model list")
                    print("💡 This might still work - OpenAI doesn't always list all models")
            else:
                print(f"❌ OpenAI API returned status {response.status_code}")
                if response.status_code == 401:
                    print("💡 Check your OPENAI_API_KEY")
                elif response.status_code == 429:
                    print("💡 Rate limited - try again later")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to OpenAI at {settings.openai.api_base_url}")
            print("💡 Check internet connectivity and API base URL")
        except Exception as e:
            print(f"❌ OpenAI connection error: {e}")
    
    # Check PostgreSQL connectivity
    print(f"\n📡 Testing PostgreSQL connection to {settings.postgres.pg_host}:{settings.postgres.pg_port}")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=settings.postgres.pg_host,
            port=settings.postgres.pg_port,
            database=settings.postgres.pg_database,
            user=settings.postgres.pg_username,
            password=settings.postgres.pg_password,
            connect_timeout=5
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL is accessible: {version}")
        
        # Check for pgvector extension
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        has_vector = cursor.fetchone()[0]
        if has_vector:
            print("✅ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
            print("💡 Install with: CREATE EXTENSION vector;")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        print("💡 Check database credentials and connectivity")
    
    print("\n" + "="*50)


def load_json_file(file_path):
    """Load and parse JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_json_as_documents(file_path, chunk_size=1000, overlap=200):
    """
    Load JSON file and convert to LangChain documents.
    Handles both single JSON objects and arrays of JSON objects.
    """
    json_data = load_json_file(file_path)
    if not json_data:
        return []
    
    documents = []
    
    # Extract better document metadata
    document_name = os.path.basename(file_path)
    
    # Handle array of JSON objects (like your big_chunks.json)
    if isinstance(json_data, list):
        for idx, item in tqdm(enumerate(json_data), total=len(json_data), desc=f"Processing {os.path.basename(file_path)}", unit="item", leave=False):
            if isinstance(item, dict) and 'text' in item:
                # Handle objects with 'text' field
                text = item['text']
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
                
                # Extract metadata from the item if available
                section = item.get('section', item.get('heading', 'unknown'))
                page = item.get('page', item.get('page_number', 'unknown'))
                context = item.get('context', item.get('summary', ''))
                
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source_file": document_name,  # Use the document name instead of full path
                        "source": file_path,  # Keep original path as backup
                        "item_index": idx,
                        "chunk_index": ci,
                        "section": section,
                        "page": page,
                        "context_summary": context,
                        "original_item": item  # Store the original JSON object
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            elif isinstance(item, str):
                # Handle array of strings
                chunks = chunk_text(item, chunk_size=chunk_size, overlap=overlap) or [item]
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source_file": document_name,  # Use the document name instead of full path
                        "source": file_path,  # Keep original path as backup
                        "item_index": idx,
                        "chunk_index": ci,
                        "section": "Text Fragment",
                        "page": "unknown"
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            else:
                # Handle other object types - convert to string
                text = json.dumps(item, ensure_ascii=False, indent=2)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
                
                # Try to extract metadata from the item if it's a dictionary
                section = "unknown"
                page = "unknown"
                if isinstance(item, dict):
                    section = item.get('section', item.get('heading', item.get('title', 'unknown')))
                    page = item.get('page', item.get('page_number', 'unknown'))
                
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source_file": document_name,  # Use the document name instead of full path
                        "source": file_path,  # Keep original path as backup
                        "item_index": idx,
                        "chunk_index": ci,
                        "section": section,
                        "page": page,
                        "original_item": item
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
    
    # Handle single JSON object
    elif isinstance(json_data, dict):
        if 'text' in json_data:
            text = json_data['text']
        else:
            text = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # Extract metadata from the JSON if available
        section = json_data.get('section', json_data.get('heading', json_data.get('title', 'unknown')))
        page = json_data.get('page', json_data.get('page_number', 'unknown'))
        context = json_data.get('context', json_data.get('summary', ''))
        
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
        for ci, chunk in enumerate(chunks):
            metadata = {
                "source_file": document_name,  # Use the document name instead of full path
                "source": file_path,  # Keep original path as backup
                "chunk_index": ci,
                "section": section,
                "page": page,
                "context_summary": context,
                "original_item": json_data
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    # Handle string data
    elif isinstance(json_data, str):
        chunks = chunk_text(json_data, chunk_size=chunk_size, overlap=overlap) or [json_data]
        for ci, chunk in enumerate(chunks):
            metadata = {
                "source_file": document_name,  # Use the document name instead of full path
                "source": file_path,  # Keep original path as backup
                "chunk_index": ci,
                "section": "Text Content",
                "page": "unknown"
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents


def load_jsons_as_documents(dir_path, limit=100, chunk_size=1000, overlap=200):
    """Load multiple JSON files from a directory"""
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(".json")]
    files = sorted(files)[:limit]
    documents = []
    
    # Add progress bar for loading multiple files
    for file_idx, file_path in tqdm(enumerate(files), total=len(files), desc="Loading JSON files", unit="file"):
        file_documents = load_json_as_documents(file_path, chunk_size, overlap)
        # Update metadata to include file index
        for doc in file_documents:
            doc.metadata["file_index"] = file_idx
        documents.extend(file_documents)
    
    return documents


def get_vectorstorage(docs, embeddings, collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Create vector storage from documents"""
    return PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)


def save_to_pgvector(documents, embeddings, collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Save documents to PGVector database with progress tracking"""
    import time
    
    try:
        print(f"🚀 Generating embeddings for {len(documents)} documents...")
        print(f"🔧 Using embedding provider: {type(embeddings).__name__}")
        
        # Test embedding connection first
        try:
            print("🔍 Testing embedding connection...")
            test_embedding = embeddings.embed_documents(["test connection"])
            print(f"✅ Embedding connection successful! Dimensions: {len(test_embedding[0])}")
        except Exception as test_e:
            print(f"❌ Embedding connection test failed: {test_e}")
            print("🔄 Attempting to reinitialize embeddings...")
            
            # Try to reinitialize embeddings
            global settings
            if settings.use_openai and settings.openai.api_key:
                try:
                    print("🔄 Reinitializing OpenAI embeddings...")
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(
                        model=settings.openai.embedding_model,
                        api_key=settings.openai.api_key,
                        base_url=settings.openai.api_base_url
                    )
                    # Test again
                    test_embedding = embeddings.embed_documents(["test connection"])
                    print(f"✅ OpenAI embeddings reinitialized successfully!")
                except Exception as openai_e:
                    print(f"❌ OpenAI reinitialize failed: {openai_e}")
                    raise
                    
            elif settings.use_ollama:
                try:
                    print("🔄 Reinitializing Ollama embeddings...")
                    from langchain_ollama.embeddings import OllamaEmbeddings
                    embeddings = OllamaEmbeddings(
                        model=settings.ollama.embedding_model,
                        base_url=settings.ollama.direct_url
                    )
                    # Test again
                    test_embedding = embeddings.embed_documents(["test connection"])
                    print(f"✅ Ollama embeddings reinitialized successfully!")
                except Exception as ollama_e:
                    print(f"❌ Ollama reinitialize failed: {ollama_e}")
                    print(f"💡 Make sure Ollama is running at: {settings.ollama.direct_url}")
                    print(f"💡 Check if model '{settings.ollama.embedding_model}' is available")
                    raise
            else:
                print("❌ No embedding provider available for reinitialization")
                raise test_e
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings with progress bar
        embedded_texts = []
        batch_size = 3  # Even smaller batch size for better reliability
        max_retries = 5  # More retries
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            batch_success = False
            
            # Try batch processing with retries
            for retry in range(max_retries):
                try:
                    # Add a progressive delay between batches
                    if i > 0:
                        time.sleep(min(2 + (i // 10), 5))  # Progressive delay, max 5 seconds
                    
                    # Generate embeddings for this batch
                    batch_embeddings = embeddings.embed_documents(batch_texts)
                    embedded_texts.extend(batch_embeddings)
                    batch_success = True
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"❌ Error generating embeddings for batch {i//batch_size + 1}, retry {retry + 1}: {e}")
                    
                    # Different retry strategies based on error type
                    if "connection" in error_msg or "timeout" in error_msg:
                        wait_time = min(5 * (2 ** retry), 30)  # Exponential backoff, max 30 seconds
                        print(f"🔄 Connection issue detected, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    elif "rate limit" in error_msg or "quota" in error_msg:
                        wait_time = min(10 * (2 ** retry), 60)  # Longer wait for rate limits
                        print(f"🔄 Rate limit detected, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(2 ** retry)  # Standard exponential backoff
                    
            # If batch failed, try individual documents
            if not batch_success:
                print(f"🔄 Falling back to individual processing for batch {i//batch_size + 1}")
                for j, text in enumerate(batch_texts):
                    individual_success = False
                    for retry in range(max_retries):
                        try:
                            time.sleep(0.5)  # Small delay between individual requests
                            embedding = embeddings.embed_documents([text])
                            embedded_texts.extend(embedding)
                            individual_success = True
                            break
                        except Exception as single_error:
                            print(f"❌ Error embedding document {i + j + 1}, retry {retry + 1}: {single_error}")
                            if retry < max_retries - 1:
                                time.sleep(1)
                    
                    # Use zero vector as fallback if all retries failed
                    if not individual_success:
                        print(f"⚠️  Using zero vector for document {i + j + 1}")
                        dimensions = 1536  # Default for OpenAI models
                        if settings.use_ollama and "nomic" in settings.ollama.embedding_model.lower():
                            dimensions = 768  # nomic-embed-text uses 768 dimensions
                        elif settings.use_openai and "text-embedding-3-large" in settings.openai.embedding_model:
                            dimensions = 3072  # text-embedding-3-large uses 3072 dimensions
                        embedded_texts.append([0.0] * dimensions)
        
        print(f"✅ Generated {len(embedded_texts)} embeddings")
        print(f"💾 Storing documents in PGVector database...")
        
        # Create vectorstore with pre-computed embeddings
        vectorstore = PGVector.from_embeddings(
            text_embeddings=list(zip(texts, embedded_texts)),
            embedding=embeddings,
            metadatas=[doc.metadata for doc in documents],
            collection_name=collection_name,
            connection_string=CONNECTION_STR,
            use_jsonb=True,
        )
        
        print(f"✅ Saved {len(documents)} documents to collection '{collection_name}'")
        return vectorstore
    except Exception as e:
        print(f"❌ Error saving to PGVector: {e}")
        raise


def embedd_and_store_json_documents(json_path, limit=100, chunk_size=1000, overlap=200, collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Main function to embed and store JSON documents"""
    print(f"📂 Processing: {json_path}")
    
    if os.path.isfile(json_path):
        # Single file
        print(f"📄 Loading JSON file...")
        docs = load_json_as_documents(json_path, chunk_size, overlap)
        print(f"✅ Loaded single JSON file with {len(docs)} document chunks")
    elif os.path.isdir(json_path):
        # Directory of JSON files
        print(f"📁 Loading JSON files from directory...")
        docs = load_jsons_as_documents(json_path, limit, chunk_size, overlap)
        print(f"✅ Loaded directory with {len(docs)} document chunks from multiple files")
    else:
        print(f"❌ Path {json_path} is neither a file nor a directory")
        return
    
    if not docs:
        print("❌ No documents found to process")
        return
    
    # Show document statistics
    print(f"\n📊 Document Statistics:")
    print(f"   Total document chunks: {len(docs)}")
    print(f"   Average chunk length: {sum(len(doc.page_content) for doc in docs) // len(docs)} characters")
    print(f"   Collection name: {collection_name}")
    
    save_to_pgvector(docs, embeddings, collection_name)
    print(f"\n🎉 Successfully processed {len(docs)} documents!")


# Create Document Parsing Function to String
def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_similar_documents(user_prompt, top_k=3, collection_name=config.COSMIC_DATABASE_COLLECTION_NAME):
    """Retrieve similar documents and generate answer"""
    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # Create the Prompt Template
    prompt_template = """Use the context provided to answer
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry. Answer in Swedish.

    context: {context}

    question: {query}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Query the RAG Chain
    answer = rag_chain.invoke(user_prompt)
    print("\nAnswer:: ", answer)
    return answer


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="RAG system for JSON documents using PostgreSQL with pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed JSON file with default settings
  python rag2.py --embed /path/to/file.json
  
  # Embed with custom chunk size and overlap
  python rag2.py --embed /path/to/file.json --chunk-size 1500 --overlap 300
  
  # Query the embedded documents
  python rag2.py --query "What is CDS FH Referral?" --top-k 3
  
  # Embed and then query
  python rag2.py --embed /path/to/file.json --query "Your question here"
  
  # Use custom collection name
  python rag2.py --embed /path/to/file.json --collection my_collection
        """
    )
    # Main action arguments
    parser.add_argument(
        "--embed", 
        type=str, 
        help="Path to JSON file or directory to embed and store"
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query string to search for similar documents"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Size of text chunks (default: 1000)"
    )
    
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=200, 
        help="Overlap between chunks (default: 200)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of similar documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--collection", 
        type=str, 
        default=config.COSMIC_DATABASE_COLLECTION_NAME, 
        help="Collection name in vector database (default: json_documents_collection)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Maximum number of files to process from directory (default: 100)"
    )
    
    # Language and output options
    parser.add_argument(
        "--language", 
        type=str, 
        default="Swedish", 
        choices=["Swedish", "English", "German", "French", "Spanish"],
        help="Language for the response (default: Swedish)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--diagnose", 
        action="store_true", 
        help="Run connection diagnostics and exit"
    )
    
    args = parser.parse_args()
    
    # Handle diagnostics mode
    if args.diagnose:
        diagnose_connection_issues()
        return
    
    # Validate arguments
    if not args.embed and not args.query:
        parser.error("You must specify either --embed or --query (or both)")
    
    if args.embed and not os.path.exists(args.embed):
        parser.error(f"Path does not exist: {args.embed}")
    
    # Set verbose mode
    if args.verbose:
        print(f"Settings: {settings}")
        print(f"Connection string: {CONNECTION_STR}")
    
    # Embedding phase
    if args.embed:
        print(f"Embedding documents from: {args.embed}")
        print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
        print(f"Collection: {args.collection}")
        
        # Run diagnostics if verbose mode is enabled
        if args.verbose:
            diagnose_connection_issues()
        
        embedd_and_store_json_documents(
            json_path=args.embed,
            limit=args.limit,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            collection_name=args.collection
        )
        
        print("✅ Embedding completed successfully!")
    
    # Query phase
    if args.query:
        print(f"\n🔍 Searching for: '{args.query}'")
        print(f"Top K: {args.top_k}, Collection: {args.collection}")
        
        # Check collection status before querying
        log_collection_status(args.collection)
        
        # Update prompt template based on language
        language_prompts = {
            "Swedish": "Answer in Swedish",
            "English": "Answer in English", 
            "German": "Answer in German",
            "French": "Answer in French",
            "Spanish": "Answer in Spanish"
        }
        
        # Temporarily modify the retrieve function to use custom language
        answer = retrieve_similar_documents_with_language(
            user_prompt=args.query,
            top_k=args.top_k,
            collection_name=args.collection,
            language=language_prompts[args.language]
        )
        
        print(f"\n📋 Final Answer:\n{answer}")


def retrieve_similar_documents_with_language(user_prompt, top_k=3, collection_name=config.COSMIC_DATABASE_COLLECTION_NAME, language="Answer in Swedish"):
    """Retrieve similar documents and generate answer with custom language"""
    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # Create the Prompt Template with custom language
    prompt_template = f"""Use the context provided to answer
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry. {language}.

    context: {{context}}

    question: {{query}}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Query the RAG Chain
    answer = rag_chain.invoke(user_prompt)
    return answer


if __name__ == "__main__":
    main()
