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
import re
import traceback

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
        if config.USE_OPENAI:
            return "openai"
        elif config.USE_OLLAMA:
            return "ollama"
        else:
            raise ValueError(
                "No Vanna provider is enabled in config. Set either vanna.openai.enabled or vanna.ollama.enabled to true")

    def _pre_download_onnx_model(self):
        """Pre-download the ONNX model for ChromaDB to prevent timeouts during training"""
        try:
            # Define persistent model cache directory (mounted volume)
            model_dir = Path("/home/appuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx")
            model_path = model_dir / "model.onnx"
            
            # Ensure directory exists
            model_dir.mkdir(parents=True, exist_ok=True)

            # Check if model already exists
            if model_path.exists():
                print(f"[VANNA DEBUG] ✅ Found existing ONNX model at {model_path}")
                os.environ["CHROMA_CACHE_DIR"] = str(model_dir.parent.parent.parent)
            else:
                print(f"[VANNA DEBUG] ONNX model not found. Starting download to {model_dir}...")
                
                # Redirect ChromaDB to use persistent cache
                os.environ["CHROMA_CACHE_DIR"] = str(model_dir.parent.parent.parent)

                # Initialize embedding function and download model
                embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
                embedding_function._download_model_if_not_exists()

                print(f"[VANNA DEBUG] ✅ ONNX model downloaded successfully to {model_dir}")

        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Failed to pre-download ONNX model: {e}")
            traceback.print_exc()

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

    def _patch_extract_sql(self):
        """
        Patch Vanna's extract_sql method to:
        1. Fix SQL truncation bugs
        2. Remove SQL comments (-- and /* */)
        3. Handle various SQL formatting styles
        """
        original_extract = self.vanna_client.extract_sql
        
        def _clean_sql(sql_text: str) -> str:
            """Remove SQL comments and clean up the query"""
            if not sql_text:
                return sql_text
                
            # Remove single-line comments (-- ...)
            sql_no_line_comments = re.sub(r'--.*?(?=\n|$)', '', sql_text)
            
            # Remove block comments (/* ... */)
            sql_no_block_comments = re.sub(r'/\*.*?\*/', '', sql_no_line_comments, flags=re.DOTALL)
            
            # Collapse multiple whitespaces and newlines into single spaces
            sql_clean = re.sub(r'\s+', ' ', sql_no_block_comments)
            
            return sql_clean.strip()
        
        def patched_extract_sql(llm_response: str) -> str:
            """
            Fixed SQL extraction that:
            - Doesn't truncate queries
            - Removes comments
            - Handles multiple formatting styles
            """
            print(f"[VANNA DEBUG] 📥 Raw LLM Response:\n{llm_response[:500]}...")
            
            sql = None
            
            # Try to extract from ```sql code blocks
            if "```sql" in llm_response.lower():
                pattern = r'```sql\s*(.*?)\s*```'
                matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
                if matches:
                    sql = matches[0].strip()
                    print(f"[VANNA DEBUG] ✅ Extracted SQL from ```sql block")
            
            # Try generic code blocks
            elif "```" in llm_response:
                pattern = r'```\s*(.*?)\s*```'
                matches = re.findall(pattern, llm_response, re.DOTALL)
                if matches:
                    for match in matches:
                        if any(keyword in match.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                            sql = match.strip()
                            print(f"[VANNA DEBUG] ✅ Extracted SQL from generic code block")
                            break
            
            # Handle case where response starts with "sql" or "SQL" without backticks
            elif llm_response.strip().lower().startswith('sql'):
                sql = llm_response.strip()[3:].strip()
                print(f"[VANNA DEBUG] ✅ Extracted SQL by removing 'sql' prefix")
            
            # Check if the response directly contains SQL keywords (no formatting at all)
            elif any(keyword in llm_response.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                sql = llm_response.strip()
                print(f"[VANNA DEBUG] ✅ Using raw response as SQL")
            
            # Fallback to original method
            else:
                try:
                    sql = original_extract(llm_response)
                    print(f"[VANNA DEBUG] ⚠️ Used original extract_sql")
                except Exception as e:
                    print(f"[VANNA DEBUG] ❌ Original extract_sql failed: {e}")
                    sql = llm_response.strip()
                    print(f"[VANNA DEBUG] ⚠️ Returning cleaned raw response")
            
            # Clean the SQL (remove comments and extra whitespace)
            if sql:
                original_sql = sql
                sql = _clean_sql(sql)
                if sql != original_sql:
                    print(f"[VANNA DEBUG] 🧹 Cleaned SQL (removed comments)")
                print(f"[VANNA DEBUG] 📝 Final SQL: {sql[:200]}...")
            else:
                print(f"[VANNA DEBUG] ⚠️ No SQL extracted from response")
            
            return sql
        
        # Replace the method
        self.vanna_client.extract_sql = patched_extract_sql
        print("[VANNA DEBUG] ✅ Patched extract_sql method")

    def _patch_generate_sql(self):
        """
        HYBRID APPROACH: Patch both submit_prompt and generate_sql to:
        1. Capture raw LLM responses before Vanna processes them
        2. Extract SQL immediately from LLM response
        3. Let Vanna's context retrieval and prompt building work properly
        4. Block all execution attempts
        5. Always return extracted SQL, never error messages
        """
        
        # Store original methods
        original_submit_prompt = self.vanna_client.submit_prompt
        original_generate_sql = self.vanna_client.generate_sql
        
        # Storage for captured SQL from LLM response
        captured_sql = {'value': None}
        
        def patched_submit_prompt(prompt, **kwargs):
            """
            Intercept LLM response and extract SQL immediately
            before Vanna's internal logic can replace it with error messages
            """
            print(f"[VANNA DEBUG] 🤖 Submitting prompt to LLM...")
            
            try:
                # Get raw LLM response
                llm_response = original_submit_prompt(prompt, **kwargs)
                
                print(f"[VANNA DEBUG] 📥 Raw LLM Response received ({len(llm_response)} chars)")
                print(f"LLM Response: {llm_response}")
                
                # Extract SQL immediately using our patched extract_sql
                sql = self.vanna_client.extract_sql(llm_response)
                
                # Store it so generate_sql can return it
                if sql:
                    captured_sql['value'] = sql
                    print(f"[VANNA DEBUG] ✅ SQL captured from LLM response ({len(sql)} chars)")
                else:
                    print(f"[VANNA DEBUG] ⚠️ No SQL found in LLM response")
                
                return llm_response
                
            except Exception as e:
                print(f"[VANNA DEBUG] ❌ Error in patched_submit_prompt: {e}")
                traceback.print_exc()
                raise
        
        def patched_generate_sql(question: str, allow_llm_to_see_data: bool = False) -> str:
            """
            Generate SQL and return captured value from LLM, never error messages
            Lets Vanna do context retrieval but intercepts the result
            """
            print(f"[VANNA DEBUG] 🔍 Generating SQL for question: '{question}'")
            
            # Reset captured SQL
            captured_sql['value'] = None
            
            # Block execution completely
            original_run_sql_is_set = self.vanna_client.run_sql_is_set
            original_run_sql = getattr(self.vanna_client, 'run_sql', None)
            
            def mock_run_sql(*args, **kwargs):
                """Mock that blocks any execution attempts"""
                print("[VANNA DEBUG] 🚫 Blocked intermediate SQL execution attempt")
                return None
            
            # Replace run_sql temporarily
            self.vanna_client.run_sql_is_set = False
            self.vanna_client.run_sql = mock_run_sql
            
            try:
                # Call original generate_sql
                # This will trigger our patched_submit_prompt which captures SQL
                # Force allow_llm_to_see_data to False to prevent execution logic
                result = original_generate_sql(question=question, allow_llm_to_see_data=False)
                
                # Priority 1: Return captured SQL from LLM response
                if captured_sql['value']:
                    print(f"[VANNA DEBUG] ✅ Returning captured SQL from LLM ({len(captured_sql['value'])} chars)")
                    return captured_sql['value']
                
                # Priority 2: Check if result itself is valid SQL
                if result and isinstance(result, str):
                    # Filter out error messages
                    if result.startswith("Error") or result.startswith("The LLM") or "not allowed" in result.lower():
                        print(f"[VANNA DEBUG] ❌ Result is error message: {result[:100]}")
                        return None
                    
                    # Verify it looks like SQL
                    if any(kw in result.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'ALTER', 'DROP']):
                        print(f"[VANNA DEBUG] ✅ Result is valid SQL, returning it")
                        return result
                
                print(f"[VANNA DEBUG] ⚠️ No valid SQL generated")
                return None
                
            except Exception as e:
                print(f"[VANNA DEBUG] ❌ Error in patched_generate_sql: {e}")
                traceback.print_exc()
                return None
                
            finally:
                # Restore original methods
                self.vanna_client.run_sql_is_set = original_run_sql_is_set
                if original_run_sql:
                    self.vanna_client.run_sql = original_run_sql
        
        # Apply both patches
        self.vanna_client.submit_prompt = patched_submit_prompt
        self.vanna_client.generate_sql = patched_generate_sql
        
        print("[VANNA DEBUG] ✅ Patched both submit_prompt and generate_sql (HYBRID approach)")

    def initialize_vanna(self, provider: Optional[str] = None):
        """Initialize Vanna with specified provider or use config default"""
        target_provider = provider or self.current_provider

        if target_provider == "openai":
            self._init_openai_vanna()
        elif target_provider == "ollama":
            self._init_ollama_vanna()
        else:
            raise ValueError(f"Unsupported provider: {target_provider}")

        print(f"[VANNA DEBUG] ✅ Vanna initialized with provider: {target_provider}")
        return self.vanna_client

    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        print(f"[VANNA DEBUG] 🔧 Initializing OpenAI Vanna...")
        print(f"[VANNA DEBUG] Model: {config.VANNA_OPENAI_MODEL}")
        print(f"[VANNA DEBUG] ChromaDB Path: {self.chroma_path}")

        VannaClass = self.get_vanna_class("openai")

        client_config = {
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'temperature': 0.0,
            'allow_llm_to_see_data': False,  # CRITICAL: Never allow data access
            'verbose': config.VANNA_OPENAI_VERBOSE,
            'http_client': httpx.Client(timeout=120.0),  # Increased timeout
            'path': self.chroma_path
        }

        self.vanna_client = VannaClass(config=client_config)
        
        # Apply all patches AFTER initialization
        self._patch_extract_sql()
        self._patch_generate_sql()
        
        self.current_provider = "openai"
        
        print(f"[VANNA DEBUG] ✅ OpenAI Vanna initialized successfully")

    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        print(f"[VANNA DEBUG] 🔧 Initializing Ollama Vanna...")
        print(f"[VANNA DEBUG] Model: {config.VANNA_OLLAMA_MODEL}")
        print(f"[VANNA DEBUG] Host: {config.VANNA_OLLAMA_BASE_URL}")
        print(f"[VANNA DEBUG] ChromaDB Path: {self.chroma_path}")

        VannaClass = self.get_vanna_class("ollama")

        self.vanna_client = VannaClass(config={
            'model': config.VANNA_OLLAMA_MODEL,
            'temperature': 0.0,
            'ollama_host': config.VANNA_OLLAMA_BASE_URL,
            'allow_llm_to_see_data': False,  # CRITICAL: Never allow data access
            'verbose': config.VANNA_OLLAMA_VERBOSE,
            'path': self.chroma_path
        })
        
        # Apply all patches AFTER initialization
        self._patch_extract_sql()
        self._patch_generate_sql()
        
        self.current_provider = "ollama"
        
        print(f"[VANNA DEBUG] ✅ Ollama Vanna initialized successfully")

    def train(
        self,
        ddl: Optional[str] = None,
        documentation: Optional[str] = None,
        question: Optional[str] = None,
        sql: Optional[str] = None
    ) -> bool:
        """
        Train Vanna with different types of data
        
        Args:
            ddl: Database schema definition (CREATE TABLE statements, etc.)
            documentation: Business logic or domain documentation
            question: Natural language question (must be paired with sql)
            sql: SQL query (can be standalone or paired with question)
            
        Returns:
            bool: True if all training succeeded, False otherwise
        """
        if not self.vanna_client:
            print(f"[VANNA DEBUG] Vanna client not initialized, initializing now...")
            self.initialize_vanna()

        def _safe_train(train_func, data, data_type):
            """Helper to safely train with error handling"""
            try:
                train_func(data)
                print(f"[VANNA DEBUG] ✅ Successfully trained {data_type}")
                return True
            except Exception as e:
                print(f"[VANNA DEBUG] ❌ Error training {data_type}: {e}")
                traceback.print_exc()
                return False

        success = True

        if ddl:
            print(f"[VANNA DEBUG] 📊 Training DDL ({len(ddl)} chars)...")
            if not _safe_train(lambda x: self.vanna_client.train(ddl=x), ddl, "DDL"):
                success = False

        if documentation:
            print(f"[VANNA DEBUG] 📖 Training documentation ({len(documentation)} chars)...")
            if not _safe_train(lambda x: self.vanna_client.train(documentation=x), documentation, "documentation"):
                success = False

        # Handle question-SQL pairs
        if question and sql:
            print(f"[VANNA DEBUG] 💬 Training Q&A pair: '{question[:50]}...'")
            if not _safe_train(lambda x: self.vanna_client.train(question=x[0], sql=x[1]), (question, sql), "Q&A pair"):
                success = False
        # Handle SQL-only training
        elif sql and not question:
            print(f"[VANNA DEBUG] 📝 Training SQL query ({len(sql)} chars)...")
            if not _safe_train(lambda x: self.vanna_client.train(sql=x), sql, "SQL"):
                success = False

        if not any([ddl, documentation, sql, (question and sql)]):
            raise ValueError("Must provide at least one training data type")

        return success

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_sql(self, query: str) -> Optional[str]:
        """
        Generate SQL from a natural language query
        
        Args:
            query: Natural language question
            
        Returns:
            str: Generated SQL query, or None if generation failed
        """
        if not self.vanna_client:
            raise ValueError("Vanna client must be initialized before generating SQL")

        print(f"[VANNA DEBUG] " + "="*80)
        print(f"[VANNA DEBUG] 🚀 Starting SQL generation")
        print(f"[VANNA DEBUG] Question: '{query}'")
        print(f"[VANNA DEBUG] " + "="*80)

        try:
            # Generate SQL (patched version handles everything with hybrid approach)
            sql = self.vanna_client.generate_sql(query, allow_llm_to_see_data=False)

            if not sql:
                print(f"[VANNA DEBUG] ⚠️ Warning: SQL generation returned None/empty")
                return None

            # Double-check it's not an error message
            if isinstance(sql, str) and (sql.startswith("Error") or "not allowed" in sql.lower()[:100]):
                print(f"[VANNA DEBUG] ❌ Error message returned instead of SQL: {sql[:100]}")
                return None

            print(f"[VANNA DEBUG] " + "="*80)
            print(f"[VANNA DEBUG] ✅ SQL GENERATION SUCCESSFUL")
            print(f"[VANNA DEBUG] Generated SQL Length: {len(sql)} characters")
            print(f"[VANNA DEBUG] Full SQL: {sql}")
            print(f"[VANNA DEBUG] " + "="*80)

            return sql

        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Error generating SQL: {e}")
            traceback.print_exc()
            return None

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

    def clear_training_data(self, reinitialize: bool = True):
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
            traceback.print_exc()
            raise

    def reset_and_retrain(self):
        """
        Convenience method to clear all data and prepare for fresh training.
        Useful when you want to start training from scratch.
        """
        print(f"[VANNA DEBUG] 🔄 Resetting Vanna - clearing all training data...")
        self.clear_training_data(reinitialize=True)
        print(f"[VANNA DEBUG] ✅ Ready for fresh training!")

    def get_training_data_summary(self) -> dict:
        """
        Get a summary of the training data stored in ChromaDB
        
        Returns:
            dict: Summary of training data including counts and examples
        """
        if not self.vanna_client:
            return {"error": "Vanna client not initialized"}
        
        try:
            # Get training data from ChromaDB
            training_data = self.vanna_client.get_training_data()
            
            summary = {
                "total_entries": len(training_data) if training_data else 0,
                "ddl_count": 0,
                "documentation_count": 0,
                "sql_count": 0,
                "question_sql_pairs": 0
            }
            
            if training_data:
                for entry in training_data:
                    if 'ddl' in entry:
                        summary["ddl_count"] += 1
                    if 'documentation' in entry:
                        summary["documentation_count"] += 1
                    if 'sql' in entry and 'question' in entry:
                        summary["question_sql_pairs"] += 1
                    elif 'sql' in entry:
                        summary["sql_count"] += 1
            
            print(f"[VANNA DEBUG] 📊 Training Data Summary:")
            print(f"[VANNA DEBUG]    Total entries: {summary['total_entries']}")
            print(f"[VANNA DEBUG]    DDL: {summary['ddl_count']}")
            print(f"[VANNA DEBUG]    Documentation: {summary['documentation_count']}")
            print(f"[VANNA DEBUG]    SQL: {summary['sql_count']}")
            print(f"[VANNA DEBUG]    Q&A pairs: {summary['question_sql_pairs']}")
            
            return summary
            
        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Error getting training data summary: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def get_similar_training_data(self, question: str, n: int = 5) -> dict:
        """
        Get similar training data for a given question (useful for debugging)
        
        Args:
            question: The question to find similar training data for
            n: Number of similar items to retrieve
            
        Returns:
            dict: Similar DDL, documentation, and Q&A pairs
        """
        if not self.vanna_client:
            return {"error": "Vanna client not initialized"}
        
        try:
            print(f"[VANNA DEBUG] 🔍 Finding similar training data for: '{question}'")
            
            similar_qna = self.vanna_client.get_similar_question_sql(question, n=n)
            related_ddl = self.vanna_client.get_related_ddl(question, n=n)
            related_docs = self.vanna_client.get_related_documentation(question, n=n)
            
            result = {
                "question": question,
                "similar_qna_pairs": similar_qna,
                "related_ddl": related_ddl,
                "related_documentation": related_docs
            }
            
            print(f"[VANNA DEBUG] ✓ Found {len(similar_qna)} similar Q&A pairs")
            print(f"[VANNA DEBUG] ✓ Found {len(related_ddl)} related DDL statements")
            print(f"[VANNA DEBUG] ✓ Found {len(related_docs)} related documentation entries")
            
            return result
            
        except Exception as e:
            print(f"[VANNA DEBUG] ❌ Error getting similar training data: {e}")
            traceback.print_exc()
            return {"error": str(e)}


# ## What This Hybrid Approach Does

# ### **1. Intercepts at Two Levels**
# - **`submit_prompt` level**: Captures raw LLM response immediately
# - **`generate_sql` level**: Wraps entire generation process with safeguards

# ### **2. SQL Capture Flow**
# ```
# User Question
#     ↓
# generate_sql() called
#     ↓
# Vanna builds context (DDL, docs, similar Q&A)
#     ↓
# submit_prompt() called with full prompt
#     ↓
# LLM generates SQL → CAPTURED HERE immediately
#     ↓
# Vanna tries to process (we block execution)
#     ↓
# Return captured SQL (never error messages)
