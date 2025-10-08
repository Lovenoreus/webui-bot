'''
Authors: Praveen Kehelella | Hammad Faheem
Description: Vanna integration with the MCP server
'''
import os
import logging
import warnings
import builtins
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv,find_dotenv
import config

def setup_vanna():
    load_dotenv(find_dotenv())

    DEBUG=config.DEBUG
    # -------------------- Vanna Setup --------------------
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

    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    # Initialize Vanna instance
    vanna_instance = MyVanna(config={
        'api_key': os.getenv("OPENAI_API_KEY"),
        'model': "gpt-4o-mini",
        'allow_llm_to_see_data': True,
        'verbose': False
    })

    # Determine the full path to compacted.db
    db_path = os.path.join(os.path.dirname(__file__), "compacted.db")

    # Connect Vanna to SQLite database
    try:
        vanna_instance.connect_to_sqlite(db_path)
        
        # Check if training data already exists and train if needed
        existing_training_data = vanna_instance.get_training_data()
        if existing_training_data.empty:
            if DEBUG:
                print("[Vanna] No existing training data found. Starting training...")
            
            # Train on DDL statements
            df_ddl = vanna_instance.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
            for ddl in df_ddl['sql'].to_list():
                vanna_instance.train(ddl=ddl)
            
            # Get list of all tables and train on sample data
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_df = vanna_instance.run_sql(tables_query)
            
            for table_name in tables_df['name']:
                sample_query = f"SELECT DISTINCT * FROM {table_name} LIMIT 5"
                try:
                    sample_df = vanna_instance.run_sql(sample_query)
                    training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                    vanna_instance.train(documentation=training_text)
                except Exception:
                    pass
            
            if DEBUG:
                print("[Vanna] Training completed.")
        else:
            if DEBUG:
                print("[Vanna] Training data already exists. Skipping training.")
                
    except Exception as e:
        if DEBUG:
            print(f"[Vanna] Error initializing Vanna: {e}")
        vanna_instance = None
        
    return vanna_instance