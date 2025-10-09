from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv, find_dotenv
from typing import Optional
from pathlib import Path
import os
import shutil
import config
import chromadb.utils.embedding_functions as embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

load_dotenv(find_dotenv())

# Set environment variables FIRST
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OPENAI_VERIFY_SSL"] = "false"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
class VannaModelManager:
    """Manager class to handle Vanna with LLM providers for SQL generation"""

    def __init__(self, chroma_path: Optional[str] = None, clear_existing: bool = False):
        """
        Initialize VannaModelManager

        Args:
            chroma_path: Path where ChromaDB data will be stored.
                        Defaults to './chroma_db' if not specified.
            clear_existing: If True, clears existing ChromaDB data before initialization.
                           Defaults to False.
        """
        self.current_provider = self._get_active_provider()
        self.vanna_client = None

        # Set up ChromaDB storage path
        self.chroma_path = chroma_path or getattr(config, 'CHROMA_DB_PATH', './chroma_db')

        # Clear existing data if requested
        if clear_existing:
            self._clear_chroma_directory()

        self._setup_storage_directory()
        self._pre_download_onnx_model()

    def _clear_chroma_directory(self):
        """Clear all files and subdirectories in the ChromaDB path"""
        if os.path.exists(self.chroma_path):
            try:
                print(f"[VANNA DEBUG] 🗑️  Clearing existing ChromaDB data at: {os.path.abspath(self.chroma_path)}")

                # Remove the entire directory and its contents
                shutil.rmtree(self.chroma_path)

                print(f"[VANNA DEBUG] ✅ ChromaDB directory cleared successfully")

            except Exception as e:
                print(f"[VANNA DEBUG] ❌ Failed to clear ChromaDB directory: {e}")
                raise
        else:
            print(f"[VANNA DEBUG] ℹ️  ChromaDB path does not exist, nothing to clear")

    def _setup_storage_directory(self):
        """Create ChromaDB storage directory if it doesn't exist"""
        try:
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            print(f"[VANNA DEBUG] ChromaDB storage path: {os.path.abspath(self.chroma_path)}")

        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Failed to create storage directory: {e}")
            raise

    def _get_active_provider(self) -> str:
        """Determine which provider is currently active based on config"""
        if config.USE_VANNA_OPENAI:
            return "openai"

        elif config.USE_VANNA_OLLAMA:
            return "ollama"

        else:
            raise ValueError(
                "No Vanna provider is enabled in config. Set either vanna.openai.enabled or vanna.ollama.enabled to true")

    # def _pre_download_onnx_model(self):
    #     """Pre-download the ONNX model for ChromaDB to prevent timeouts during training"""
    #     try:
    #         print("[VANNA DEBUG] Pre-downloading ONNX model for ChromaDB...")
    #         embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
    #         embedding_function._download_model_if_not_exists()
    #         print("[VANNA DEBUG] ✅ ONNX model pre-downloaded successfully")

    #     except Exception as e:
    #         print(f"[VANNA DEBUG] ❌ Failed to pre-download ONNX model: {e}")
    
    def _pre_download_onnx_model(self):
        """Pre-download the ONNX model for ChromaDB to prevent timeouts during training"""
        try:
            # Define persistent model cache directory (mounted volume)
            model_dir = Path("/home/appuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx")
            model_path = model_dir / "model.onnx"
            print(f"This is the model path i am checking at yo: {model_path}")
            # Ensure directory exists
            model_dir.mkdir(parents=True, exist_ok=True)

            # Check if model already exists
            if model_path.exists():
                print(f"[VANNA DEBUG] ✅ Found existing ONNX model at {model_path}")
                os.environ["CHROMA_CACHE_DIR"] = str(model_dir)  # Make ChromaDB use this path
                
            else:
                print(f"[VANNA DEBUG] ONNX model not found. Starting download to {model_dir}...")

                # Redirect ChromaDB to use /app/docker/onnx_models as cache
                os.environ["CHROMA_CACHE_DIR"] = str(model_dir)

                # Initialize embedding function and download model
                embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
                embedding_function._download_model_if_not_exists()

                print(f"[VANNA DEBUG] ✅ ONNX model downloaded successfully to {model_dir}")

        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Failed to pre-download ONNX model: {e}")

    # def _pre_download_onnx_model(self):
    #     """Pre-download the ONNX model for ChromaDB to prevent timeouts during training"""
    #     try:
    #         # Define persistent model cache directory (mounted volume)
    #         model_dir = Path("/home/appuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx")
    #         model_path = model_dir / "model.onnx"

    #         # Ensure directory exists
    #         model_dir.mkdir(parents=True, exist_ok=True)

    #         # Check if model already exists
    #         if model_path.exists():
    #             print(f"[VANNA DEBUG] ✅ Found existing ONNX model at {model_path}")
    #             return

    #         print(f"[VANNA DEBUG] ONNX model not found. Starting download to {model_dir}...")

    #         # Set environment variables (must be strings, not Path objects)
    #         os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(model_dir.parent.parent.parent)  # /home/appuser/.cache/chroma
            
    #         # Initialize embedding function and download model
    #         embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
            
    #         # Force download by actually calling the model
    #         test_embedding = embedding_function(["test"])

    #         # Verify the model was downloaded
    #         if model_path.exists():
    #             print(f"[VANNA DEBUG] ✅ ONNX model downloaded successfully to {model_path}")
    #         else:
    #             print(f"[VANNA DEBUG] ⚠️ Model initialized but not found at expected path: {model_path}")

    #     except Exception as e:
    #         print(f"[VANNA DEBUG] ❌ Failed to pre-download ONNX model: {e}")
    #         import traceback
    #         traceback.print_exc()
            
    def get_vanna_class(self, provider: str):
        """Get the appropriate Vanna class based on provider"""
        if provider == "openai":
            class MyVannaOpenAI(ChromaDB_VectorStore, OpenAI_Chat):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    OpenAI_Chat.__init__(self, config=config)

                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def train(self, **kwargs):
                    return super().train(**kwargs)

            return MyVannaOpenAI

        elif provider == "ollama":
            class MyVannaOllama(ChromaDB_VectorStore, Ollama):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    Ollama.__init__(self, config=config)

                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def train(self, **kwargs):
                    return super().train(**kwargs)

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

        print(f"[VANNA DEBUG] Vanna initialized with provider: {target_provider}")

        return self.vanna_client

    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        VannaClass = self.get_vanna_class("openai")

        client_config = {
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OPENAI_VERBOSE,
            'http_client': httpx.Client(timeout=60.0),  # Increased timeout
            'path': self.chroma_path  # ChromaDB storage path
        }

        self.vanna_client = VannaClass(config=client_config)

        self.current_provider = "openai"

    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        VannaClass = self.get_vanna_class("ollama")

        self.vanna_client = VannaClass(config={
            'model': config.VANNA_OLLAMA_MODEL,
            'base_url': config.VANNA_OLLAMA_BASE_URL,
            'allow_llm_to_see_data': config.VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OLLAMA_VERBOSE,
            'path': self.chroma_path  # ChromaDB storage path
        })

        self.current_provider = "ollama"

    def train(
            self,
            ddl: Optional[str] = None,
            documentation: Optional[str] = None,
            question: Optional[str] = None,
            sql: Optional[str] = None
    ) -> bool:
        """Train Vanna with different types of data"""

        if not self.vanna_client:
            self.initialize_vanna()

        def _safe_train(train_func, data, data_type):
            try:
                train_func(data)
                print(f"[VANNA DEBUG] ✅ Trained {data_type} with {self.current_provider}")

                return True

            except Exception as e:
                print(f"[VANNA DEBUG] ❌ Error training {data_type}: {e}")

                return False

        success = True

        if ddl:
            if not _safe_train(lambda x: self.vanna_client.train(ddl=x), ddl, "DDL"):
                success = False

        if documentation:
            if not _safe_train(lambda x: self.vanna_client.train(documentation=x), documentation, "documentation"):
                success = False

        if question and sql:
            if not _safe_train(lambda x: self.vanna_client.train(question=x[0], sql=x[1]), (question, sql), "SQL pair"):
                success = False

        if not any([ddl, documentation, (question and sql)]):
            raise ValueError("Must provide at least one training data type")

        return success

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_sql(self, query: str) -> str:
        """Generate SQL from a natural language query"""
        if not self.vanna_client:
            raise ValueError("Vanna client must be initialized before generating SQL")
        try:
            sql = self.vanna_client.generate_sql(query)
            if not sql:
                print("[VANNA DEBUG] Warning: Generated SQL is empty")
            return sql
        except Exception as e:
            print(f"[VANNA DEBUG] Error generating SQL: {e}")
            return ""

    def get_current_provider(self) -> str:
        """Get current active provider"""
        return self.current_provider

    def get_storage_path(self) -> str:
        """Get the ChromaDB storage path"""
        return os.path.abspath(self.chroma_path)

    def get_info(self) -> dict:
        """Get current Vanna configuration info"""
        return {
            "provider": self.current_provider,
            "model": config.VANNA_OPENAI_MODEL if self.current_provider == "openai" else config.VANNA_OLLAMA_MODEL,
            "initialized": self.vanna_client is not None,
            "chroma_path": self.get_storage_path()
        }

    def clear_training_data(self, reinitialize: bool = False):
        """
        Clear all training data from ChromaDB

        Args:
            reinitialize: If True, reinitialize the Vanna client after clearing.
                         Defaults to True.
        """
        try:
            print(f"[VANNA DEBUG] 🗑️  Clearing training data from {self.chroma_path}")

            if os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path)
                print(f"[VANNA DEBUG] ✅ Cleared training data successfully")
            else:
                print(f"[VANNA DEBUG] ℹ️  No training data found at {self.chroma_path}")

            # Recreate directory
            self._setup_storage_directory()

            # Reinitialize if client was already initialized and reinitialize=True
            if self.vanna_client and reinitialize:
                print(f"[VANNA DEBUG] 🔄 Reinitializing Vanna client...")
                self.initialize_vanna()

        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Error clearing training data: {e}")
            raise

    def reset_and_retrain(self):
        """
        Convenience method to clear all data and prepare for fresh training.
        This is useful when you want to start training from scratch.
        """
        print(f"[VANNA DEBUG] 🔄 Resetting Vanna - clearing all training data...")
        self.clear_training_data(reinitialize=True)

        print(f"[VANNA DEBUG] ✅ Ready for fresh training!")