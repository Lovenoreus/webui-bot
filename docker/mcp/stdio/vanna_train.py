from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv
import os
import logging
import warnings
import builtins

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

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={
    'api_key': os.getenv("OPENAI_API_KEY"),
    'model': "gpt-4o-mini",
    'allow_llm_to_see_data': True,
    'verbose': False
})

# Determine the full path to compacted.db
db_path = os.path.join(os.path.dirname(__file__), "compacted.db")
vn.connect_to_sqlite(db_path)

df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")

# Check if training data already exists
existing_training_data = vn.get_training_data()
if existing_training_data.empty:
    # Train on DDL statements
    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)
    
    # Get list of all tables
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables_df = vn.run_sql(tables_query)

    # For each table, get distinct examples
    for table_name in tables_df['name']:
        sample_query = f"SELECT DISTINCT * FROM {table_name} LIMIT 5"
        try:
            sample_df = vn.run_sql(sample_query)
            training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
            vn.train(documentation=training_text)
        except Exception:
            pass

# Restore original print for our output
builtins.print = _original_print

while True:
    user_q = input("Question (or 'exit'): ").strip()
    if user_q.lower() == 'exit':
        break

    # Use filtered print during ask
    builtins.print = filtered_print
    sql, df, fig = vn.ask(question=user_q, print_results=False, allow_llm_to_see_data=True, visualize=False)
    builtins.print = _original_print

    print("\nGenerated SQL:")
    print(sql)
    print("\nAnswer:")
    print(df)
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
