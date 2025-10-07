from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
import logging
import warnings
import builtins
from typing import Optional
from config import (
    OPENAI_API_KEY, OLLAMA_BASE_URL,
    USE_VANNA_OPENAI, USE_VANNA_OLLAMA,
    VANNA_OPENAI_MODEL, VANNA_OLLAMA_MODEL,
    VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA, VANNA_OPENAI_VERBOSE,
    VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA, VANNA_OLLAMA_VERBOSE,
    VANNA_AUTO_TRAIN, VANNA_TRAIN_ON_STARTUP,
    get_nested, config
)

load_dotenv()

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
    """Manager class to handle switching between different LLM providers for Vanna training"""
    
    def __init__(self):
        self.current_provider = self._get_active_provider()
        self.vanna_client = None
    
    def _get_active_provider(self) -> str:
        """Determine which provider is currently active based on config"""
        if USE_VANNA_OPENAI:
            return "openai"
        elif USE_VANNA_OLLAMA:
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
        
        print(f"Vanna initialized with provider: {target_provider}")
        return self.vanna_client
    
    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        VannaClass = self.get_vanna_class("openai")
        self.vanna_client = VannaClass(config={
            'api_key': OPENAI_API_KEY,
            'model': VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': VANNA_OPENAI_VERBOSE
        })
        self.current_provider = "openai"
    
    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        VannaClass = self.get_vanna_class("ollama")
        self.vanna_client = VannaClass(config={
            'model': VANNA_OLLAMA_MODEL,
            'base_url': OLLAMA_BASE_URL,
            'allow_llm_to_see_data': VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA,
            'verbose': VANNA_OLLAMA_VERBOSE
        })
        self.current_provider = "ollama"
    

    
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
        "model": VANNA_OPENAI_MODEL if vanna_manager.current_provider == "openai" else VANNA_OLLAMA_MODEL,
        "initialized": vanna_manager.vanna_client is not None
    }

# Create global manager instance
vanna_manager = VannaModelManager()
vn = vanna_manager.initialize_vanna()

# Determine the full path to compacted.db
db_path = os.path.join(os.path.dirname(__file__), "compacted.db")
vn.connect_to_sqlite(db_path)

# Auto-train on startup if enabled
if VANNA_AUTO_TRAIN or VANNA_TRAIN_ON_STARTUP:
    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
    
    # Check if training data already exists
    existing_training_data = vn.get_training_data()
    if existing_training_data.empty or VANNA_TRAIN_ON_STARTUP:
        print(f"Training Vanna with {vanna_manager.current_provider} provider...")
        
        # Train on DDL statements
        for ddl in df_ddl['sql'].to_list():
            vanna_train(ddl=ddl)
        
        # Get list of all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables_df = vn.run_sql(tables_query)

        # For each table, get distinct examples
        for table_name in tables_df['name']:
            sample_query = f"SELECT DISTINCT * FROM {table_name} LIMIT 5"
            try:
                sample_df = vn.run_sql(sample_query)
                training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                vanna_train(documentation=training_text)
            except Exception:
                pass
        
        print("Training completed!")
    else:
        print(f"Training data already exists for {vanna_manager.current_provider}. Skipping training.")
else:
    print("Auto-training is disabled. Use vanna_train() function to manually train the model.")

# Restore original print for our output
builtins.print = _original_print

print(f"\n🤖 Vanna SQL Assistant initialized with {vanna_manager.current_provider} provider")
print(f"📊 Current model: {get_vanna_info()['model']}")
print("\n💡 Available commands:")
print("  • Ask any SQL question")
print("  • Type 'info' to see current configuration")
print("  • Type 'train' to manually train with sample data")
print("  • Type 'exit' to quit")
print("  • To switch providers, edit config.json and restart\n")

while True:
    user_q = input("❓ Question (or command): ").strip()
    
    if user_q.lower() == 'exit':
        break
    elif user_q.lower() == 'info':
        info = get_vanna_info()
        print(f"\n📋 Current Configuration:")
        print(f"  Provider: {info['provider']}")
        print(f"  Model: {info['model']}")
        print(f"  Initialized: {info['initialized']}")
        print()
        continue

    elif user_q.lower() == 'train':
        print("🔄 Starting manual training...")
        try:
            df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
            for ddl in df_ddl['sql'].to_list()[:3]:  # Train on first 3 DDL statements
                vanna_train(ddl=ddl)
            print("✅ Manual training completed!")
        except Exception as e:
            print(f"❌ Training error: {e}")
        print()
        continue

    # Use filtered print during ask
    builtins.print = filtered_print
    try:
        allow_llm_to_see_data = VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA if vanna_manager.current_provider == "openai" else VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA
        sql, df, fig = vn.ask(question=user_q, print_results=False, allow_llm_to_see_data=allow_llm_to_see_data, visualize=False)
        builtins.print = _original_print

        print(f"\n🔍 Generated SQL ({vanna_manager.current_provider}):")
        print(sql)
        print("\n📊 Results:")
        print(df)
        print()
    except Exception as e:
        builtins.print = _original_print
        print(f"❌ Error: {e}")
        print()
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
