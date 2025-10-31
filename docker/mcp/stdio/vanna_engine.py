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
import time
from contextlib import contextmanager
import json
from datetime import datetime
from pathlib import Path as FilePath

load_dotenv(find_dotenv())

# Set environment variables FIRST
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OPENAI_VERIFY_SSL"] = "false"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"


# ============================================================================
# TIMING UTILITIES
# ============================================================================

@contextmanager
def timer(operation_name: str, timings_dict: dict = None):
    """
    Context manager for timing operations
    
    Args:
        operation_name: Name of the operation being timed
        timings_dict: Optional dictionary to store timing results
    
    Usage:
        with timer("SQL Generation", timings):
            # code to time
            pass
    """
    start_time = time.perf_counter()
    print(f"[TIMING] ⏱️  Starting: {operation_name}")
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[TIMING] ✅ Completed: {operation_name} - {elapsed:.4f}s ({elapsed*1000:.2f}ms)")
        
        if timings_dict is not None:
            timings_dict[operation_name] = {
                "duration_seconds": round(elapsed, 4),
                "duration_ms": round(elapsed * 1000, 2)
            }


class TimingTracker:
    """
    Class to track and report timing information for various operations
    """
    def __init__(self, json_log_path: str = "vanna_timing.json"):
        self.timings = {}
        self.start_time = None
        self.json_log_path = json_log_path
        self.current_operation_metadata = {}  # Store metadata for current operation
        
    def start(self):
        """Start tracking total time"""
        self.start_time = time.perf_counter()
        self.timings = {}
        self.current_operation_metadata = {}
        
    def record(self, operation_name: str, duration: float):
        """Record a timing manually"""
        self.timings[operation_name] = {
            "duration_seconds": round(duration, 4),
            "duration_ms": round(duration * 1000, 2)
        }
        
    def set_metadata(self, **kwargs):
        """Set metadata for the current operation (provider, model, etc.)"""
        self.current_operation_metadata.update(kwargs)
        
    def log_to_json(self, operation: str, elapsed_time: float, status: str = "success", **extra_metadata):
        """
        Log timing data to JSON file in the required format
        
        Args:
            operation: Name of the operation (e.g., "initialize_vanna", "generate_sql")
            elapsed_time: Time taken in seconds
            status: Operation status ("success" or "error")
            **extra_metadata: Additional metadata to include in the log
        """
        try:
            # Combine stored metadata with extra metadata
            metadata = {**self.current_operation_metadata, **extra_metadata}
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "vanna_engine",
                "operation": operation,
                "elapsed_time_seconds": round(elapsed_time, 4),
                "status": status
            }
            
            # Add all metadata fields
            log_entry.update(metadata)
            
            # Read existing logs
            log_file = FilePath(self.json_log_path)
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            else:
                logs = []
            
            # Append new entry
            logs.append(log_entry)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            print(f"[TIMING] 💾 Logged to {self.json_log_path}: {operation} ({elapsed_time:.4f}s)")
            
        except Exception as e:
            print(f"[TIMING] ⚠️  Failed to log to JSON: {e}")
        
    def get_total_time(self) -> float:
        """Get total elapsed time since start()"""
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time
        
    def get_summary(self) -> dict:
        """Get a summary of all timings"""
        total_time = self.get_total_time()
        return {
            "total_time_seconds": round(total_time, 4),
            "total_time_ms": round(total_time * 1000, 2),
            "operations": self.timings,
            "operation_count": len(self.timings)
        }
        
    def print_summary(self):
        """Print a formatted summary of all timings"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("[TIMING SUMMARY] 📊 Performance Report")
        print("="*80)
        print(f"Total Time: {summary['total_time_seconds']:.4f}s ({summary['total_time_ms']:.2f}ms)")
        print(f"Operations Tracked: {summary['operation_count']}")
        print("-"*80)
        
        if self.timings:
            # Sort by duration (longest first)
            sorted_ops = sorted(
                self.timings.items(), 
                key=lambda x: x[1]['duration_seconds'], 
                reverse=True
            )
            
            for op_name, timing in sorted_ops:
                percentage = (timing['duration_seconds'] / summary['total_time_seconds'] * 100) if summary['total_time_seconds'] > 0 else 0
                print(f"  • {op_name:.<50} {timing['duration_seconds']:>8.4f}s ({percentage:>5.1f}%)")
        
        print("="*80 + "\n")

# ============================================================================


class VannaModelManager:
    """Manager class to handle Vanna with LLM providers for SQL generation"""

    def __init__(self, chroma_path: Optional[str] = None, clear_existing: bool = False, timing_log_path: str = "vanna_timing.json"):
        """
        Initialize VannaModelManager

        Args:
            chroma_path: Path where ChromaDB data will be stored.
                        Defaults to './chroma_db' if not specified.
            clear_existing: If True, clears existing ChromaDB data before initialization.
                           Defaults to False.
            timing_log_path: Path to JSON file for logging timing data.
                           Defaults to 'vanna_timing.json'.
        """
        self.current_provider = self._get_active_provider()
        self.vanna_client = None
        self.timing_tracker = TimingTracker(json_log_path=timing_log_path)  # Initialize timing tracker with log path

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
            llm_start = time.perf_counter()
            
            try:
                # Get raw LLM response
                llm_response = original_submit_prompt(prompt, **kwargs)
                llm_elapsed = time.perf_counter() - llm_start
                self.timing_tracker.record("LLM API Call", llm_elapsed)
                
                print(f"[VANNA DEBUG] 📥 Raw LLM Response received ({len(llm_response)} chars)")
                print(f"LLM Response: {llm_response}")
                
                # Extract SQL immediately using our patched extract_sql
                extract_start = time.perf_counter()
                sql = self.vanna_client.extract_sql(llm_response)
                extract_elapsed = time.perf_counter() - extract_start
                self.timing_tracker.record("SQL Extraction from LLM Response", extract_elapsed)
                
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
                context_start = time.perf_counter()
                
                # Call original generate_sql
                # This will trigger our patched_submit_prompt which captures SQL
                # Force allow_llm_to_see_data to False to prevent execution logic
                result = original_generate_sql(question=question, allow_llm_to_see_data=False)
                
                context_elapsed = time.perf_counter() - context_start
                # This includes context retrieval + LLM call (LLM timing is captured separately in patched_submit_prompt)
                retrieval_time = context_elapsed - self.timing_tracker.timings.get("LLM API Call", {}).get("duration_seconds", 0)
                if retrieval_time > 0:
                    self.timing_tracker.record("Context Retrieval & Prompt Building", retrieval_time)
                
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
        init_start = time.perf_counter()
        
        target_provider = provider or self.current_provider

        try:
            if target_provider == "openai":
                self._init_openai_vanna()
            elif target_provider == "ollama":
                self._init_ollama_vanna()
            else:
                raise ValueError(f"Unsupported provider: {target_provider}")

            init_elapsed = time.perf_counter() - init_start
            
            # Set metadata for JSON logging
            model = config.VANNA_OPENAI_MODEL if target_provider == "openai" else config.VANNA_OLLAMA_MODEL
            self.timing_tracker.set_metadata(
                provider=target_provider,
                model=model,
                chroma_path=os.path.abspath(self.chroma_path)
            )
            
            # Log to JSON file
            self.timing_tracker.log_to_json(
                operation="initialize_vanna",
                elapsed_time=init_elapsed,
                status="success"
            )
            
            # Also record in timing tracker for summary
            self.timing_tracker.record("Initialize Vanna", init_elapsed)
            
            print(f"[VANNA DEBUG] ✅ Vanna initialized with provider: {target_provider}")
            return self.vanna_client
            
        except Exception as e:
            init_elapsed = time.perf_counter() - init_start
            
            # Log error to JSON
            self.timing_tracker.log_to_json(
                operation="initialize_vanna",
                elapsed_time=init_elapsed,
                status="error",
                error=str(e)
            )
            raise

    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        start_time = time.perf_counter()
        
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
        
        elapsed = time.perf_counter() - start_time
        self.timing_tracker.record("OpenAI Client Setup", elapsed)
        
        print(f"[VANNA DEBUG] ✅ OpenAI Vanna initialized successfully")

    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        start_time = time.perf_counter()
        
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
        
        elapsed = time.perf_counter() - start_time
        self.timing_tracker.record("Ollama Client Setup", elapsed)
        
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
        train_start = time.perf_counter()
        
        if not self.vanna_client:
            print(f"[VANNA DEBUG] Vanna client not initialized, initializing now...")
            self.initialize_vanna()

        def _safe_train(train_func, data, data_type):
            """Helper to safely train with error handling"""
            item_start = time.perf_counter()
            try:
                train_func(data)
                item_elapsed = time.perf_counter() - item_start
                print(f"[VANNA DEBUG] ✅ Successfully trained {data_type} ({item_elapsed:.3f}s)")
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

        train_elapsed = time.perf_counter() - train_start
        self.timing_tracker.record("Training Data Ingestion", train_elapsed)
        
        # Determine what was trained
        data_types = []
        if ddl:
            data_types.append("ddl")
        if documentation:
            data_types.append("documentation")
        if question and sql:
            data_types.append("question_sql_pair")
        elif sql:
            data_types.append("sql")
        
        # Log to JSON
        self.timing_tracker.log_to_json(
            operation="train",
            elapsed_time=train_elapsed,
            status="success" if success else "partial_failure",
            data_types=",".join(data_types),
            item_count=len(data_types)
        )

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
        # Start timing for the entire SQL generation process
        total_start = time.perf_counter()
        self.timing_tracker.start()  # Reset timing tracker for this query
        
        if not self.vanna_client:
            raise ValueError("Vanna client must be initialized before generating SQL")

        print(f"[VANNA DEBUG] " + "="*80)
        print(f"[VANNA DEBUG] 🚀 Starting SQL generation")
        print(f"[VANNA DEBUG] Question: '{query}'")
        print(f"[VANNA DEBUG] " + "="*80)

        try:
            # Generate SQL (patched version handles everything with hybrid approach)
            # The timing for LLM call will be captured in the patched methods
            gen_start = time.perf_counter()
            sql = self.vanna_client.generate_sql(query, allow_llm_to_see_data=False)
            gen_elapsed = time.perf_counter() - gen_start
            self.timing_tracker.record("Vanna generate_sql() call", gen_elapsed)

            if not sql:
                total_elapsed = time.perf_counter() - total_start
                
                # Log failure to JSON
                self.timing_tracker.log_to_json(
                    operation="generate_sql",
                    elapsed_time=total_elapsed,
                    status="error",
                    error="SQL generation returned None/empty",
                    question=query
                )
                
                print(f"[VANNA DEBUG] ⚠️ Warning: SQL generation returned None/empty")
                return None

            # Double-check it's not an error message
            if isinstance(sql, str) and (sql.startswith("Error") or "not allowed" in sql.lower()[:100]):
                total_elapsed = time.perf_counter() - total_start
                
                # Log error to JSON
                self.timing_tracker.log_to_json(
                    operation="generate_sql",
                    elapsed_time=total_elapsed,
                    status="error",
                    error="Error message returned instead of SQL",
                    question=query
                )
                
                print(f"[VANNA DEBUG] ❌ Error message returned instead of SQL: {sql[:100]}")
                return None

            total_elapsed = time.perf_counter() - total_start
            self.timing_tracker.record("Total SQL Generation", total_elapsed)

            # Get timing breakdown
            llm_time = self.timing_tracker.timings.get("LLM API Call", {}).get("duration_seconds", 0)
            context_time = self.timing_tracker.timings.get("Context Retrieval & Prompt Building", {}).get("duration_seconds", 0)
            extraction_time = self.timing_tracker.timings.get("SQL Extraction from LLM Response", {}).get("duration_seconds", 0)
            
            # Log success to JSON with detailed breakdown
            self.timing_tracker.log_to_json(
                operation="generate_sql",
                elapsed_time=total_elapsed,
                status="success",
                question=query,
                sql_length=len(sql),
                llm_time_seconds=llm_time,
                context_retrieval_seconds=context_time,
                sql_extraction_seconds=extraction_time
            )

            print(f"[VANNA DEBUG] " + "="*80)
            print(f"[VANNA DEBUG] ✅ SQL GENERATION SUCCESSFUL")
            print(f"[VANNA DEBUG] Generated SQL Length: {len(sql)} characters")
            print(f"[VANNA DEBUG] Full SQL: {sql}")
            print(f"[VANNA DEBUG] " + "="*80)
            
            # Print timing summary
            self.timing_tracker.print_summary()

            return sql

        except Exception as e:
            total_elapsed = time.perf_counter() - total_start
            
            # Log exception to JSON
            self.timing_tracker.log_to_json(
                operation="generate_sql",
                elapsed_time=total_elapsed,
                status="error",
                error=str(e),
                question=query
            )
            
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

    def get_timing_summary(self) -> dict:
        """
        Get a summary of timing information from the last operation
        
        Returns:
            dict: Dictionary containing timing information with structure:
                  {
                      "total_time_seconds": float,
                      "total_time_ms": float,
                      "operations": {
                          "operation_name": {
                              "duration_seconds": float,
                              "duration_ms": float
                          },
                          ...
                      },
                      "operation_count": int
                  }
        """
        return self.timing_tracker.get_summary()
    
    def print_timing_summary(self):
        """
        Print a formatted summary of timing information from the last operation
        """
        self.timing_tracker.print_summary()
    
    def reset_timing(self):
        """
        Reset the timing tracker
        """
        self.timing_tracker = TimingTracker()
        print("[VANNA DEBUG] ⏱️  Timing tracker reset")


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