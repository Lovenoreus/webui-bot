# vanna_query_engine.py
"""
Vanna Query Engine Wrapper - Integrates trained Vanna model with existing QueryEngine interface
Provides the same interface as QueryEngine but uses Vanna for SQL generation and execution
"""

import asyncio
import json
from typing import List, Dict, Optional, AsyncGenerator

try:
    from vanna_train import VannaModelManager, vanna_train, get_database_schema_info, get_table_names, get_vanna_info
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from vanna_train import VannaModelManager, vanna_train, get_database_schema_info, get_table_names, get_vanna_info
import config
import pandas as pd


class VannaQueryEngine:
    """Vanna-powered query engine that matches the existing QueryEngine interface"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize Vanna Query Engine
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.vanna_manager = None
        self.vanna_client = None
        
        # Initialize Vanna in a separate method to handle async properly
        self._initialize_vanna()
    
    def _initialize_vanna(self):
        """Initialize Vanna manager and client"""
        try:
            self.vanna_manager = VannaModelManager()
            self.vanna_client = self.vanna_manager.initialize_vanna()
            
            if self.debug:
                print(f"[VANNA DEBUG] Initialized with provider: {self.vanna_manager.current_provider}")
                print(f"[VANNA DEBUG] Database: {self.vanna_manager.current_database}")
            
            # Perform training if configured
            self._perform_training_if_needed(selected_tables = config.VANNA_ALLOWED_TABLES)
            
            # Print initialization success message
            vanna_info = get_vanna_info(self.vanna_manager)
            print(f"\n🤖 Vanna SQL Assistant initialized with {self.vanna_manager.current_provider} provider")
            print(f"📊 Current model: {vanna_info['model']}")
            print(f"🗄️  Current database: {self.vanna_manager.current_database.upper()}")
                
        except Exception as e:
            print(f"[VANNA ERROR] Failed to initialize Vanna: {e}")
            raise
    
    async def generate_sql(self, user_question: str, keywords: List[str], provider: str = "openai") -> str:
        """
        Generate SQL query from natural language using Vanna
        
        Args:
            user_question: Natural language question
            keywords: Keywords for filtering (ignored by Vanna, kept for interface compatibility)
            provider: Provider preference (ignored, uses Vanna's configured provider)
            
        Returns:
            Generated SQL query
        """
        try:
            if not self.vanna_client:
                raise Exception("Vanna client not initialized")
            
            if self.debug:
                print(f"[VANNA DEBUG] Generating SQL for: {user_question}")
            
            # Use Vanna's ask method to get SQL
            # Run in thread pool since Vanna is synchronous
            loop = asyncio.get_event_loop()
            
            def vanna_ask():
                try:
                    allow_llm_to_see_data = (
                        config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA 
                        if self.vanna_manager.current_provider == "openai" 
                        else config.VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA
                    )
                    
                    sql = self.vanna_client.generate_sql(
                        question=user_question,
                        allow_llm_to_see_data=allow_llm_to_see_data,
                    )
                    return sql
                except Exception as e:
                    if self.debug:
                        print(f"[VANNA ERROR] SQL generation failed: {e}")
                    return ""
            
            sql_query = await loop.run_in_executor(None, vanna_ask)
            
            if self.debug:
                print(f"[VANNA DEBUG] Generated SQL: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            if self.debug:
                print(f"[VANNA ERROR] generate_sql failed: {e}")
            return ""
    
    async def generate_sql_stream(self, user_question: str, keywords: List[str], provider: str = "openai") -> AsyncGenerator[Dict, None]:
        """
        Generate SQL query with streaming updates (simulated for Vanna compatibility)
        
        Args:
            user_question: Natural language question
            keywords: Keywords for filtering (ignored)
            provider: Provider preference (ignored)
            
        Yields:
            Dict updates about SQL generation progress
        """
        try:
            # Simulate streaming for interface compatibility
            yield {"status": "analyzing_query", "message": "Analyzing your question with Vanna..."}
            
            yield {"status": "generating_sql", "message": "Generating SQL query using trained model..."}
            
            # Generate SQL
            sql_query = await self.generate_sql(user_question, keywords, provider)
            
            if sql_query:
                yield {
                    "status": "sql_complete",
                    "sql_query": sql_query,
                    "message": f"SQL generated successfully using {self.vanna_manager.current_provider}"
                }
            else:
                yield {
                    "status": "error",
                    "message": "Failed to generate SQL query with Vanna"
                }
                
        except Exception as e:
            yield {
                "status": "error",
                "message": f"Vanna SQL generation error: {str(e)}"
            }
    
    async def execute_query(self, sql_query: str) -> List[Dict]:
        """
        Execute SQL query using Vanna's database connection
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            List of result dictionaries
        """
        try:
            if not self.vanna_client:
                raise Exception("Vanna client not initialized")
            
            if self.debug:
                print(f"[VANNA DEBUG] Executing SQL: {sql_query}")
            
            # Use Vanna's run_sql method
            loop = asyncio.get_event_loop()
            
            def vanna_execute():
                try:
                    result = self.vanna_client.run_sql(sql_query)
                    # if df is not None and not df.empty:
                    if type(result) == pd.DataFrame:
                        # Convert DataFrame to list of dictionaries
                        return result.to_dict('records')
                    elif type(result) == list or type(result) == dict:
                        return result
                    else:
                        return []
                except Exception as e:
                    if self.debug:
                        print(f"[VANNA ERROR] SQL execution failed: {e}")
                    raise
            
            results = await loop.run_in_executor(None, vanna_execute)
            
            if self.debug:
                print(f"[VANNA DEBUG] Query returned {len(results)} records")
            
            return results
            
        except Exception as e:
            if self.debug:
                print(f"[VANNA ERROR] execute_query failed: {e}")
            return []
    
    async def execute_query_stream(self, sql_query: str) -> AsyncGenerator[Dict, None]:
        """
        Execute SQL query with streaming results
        
        Args:
            sql_query: SQL query to execute
            
        Yields:
            Dict updates about query execution progress
        """
        try:
            yield {"status": "connecting_database", "message": "Connecting to database..."}
            
            results = await self.execute_query(sql_query)
            
            yield {
                "status": "query_complete",
                "results": results,
                "record_count": len(results),
                "message": f"Query executed successfully, returned {len(results)} records"
            }
            
        except Exception as e:
            yield {
                "status": "error",
                "message": f"Database execution error: {str(e)}"
            }
    
    def _perform_training_if_needed(self, selected_tables: Optional[List[str]] = config.VANNA_ALLOWED_TABLES):    
        """Perform auto-training on startup if enabled"""
        if not (config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP):
            print("Auto-training is disabled. Use vanna_train() function to manually train the model.")
            return
            
        print(f"Attempting to get schema information for {self.vanna_manager.current_database} database...")
        df_ddl = get_database_schema_info(self.vanna_client, self.vanna_manager.current_database)
        
        # Handle different types of selected_tables parameter
        train_all_tables = False
        allowed_tables_list = []
        
        if selected_tables == "all" or selected_tables is None:
            train_all_tables = True
            print("Training on ALL tables as selected_tables is 'all' or None")
        elif isinstance(selected_tables, list):
            train_all_tables = False
            allowed_tables_list = selected_tables
            print(f"Training on selected tables: {allowed_tables_list}")
        else:
            print(f"Warning: Invalid selected_tables type: {type(selected_tables)}. Training on all tables.")
            train_all_tables = True
        
        print("Schema information retrieved:")
        print("ddllllllll:", df_ddl["table_name"] if df_ddl is not None else "No DataFrame")
        print("Type of df_ddl:", type(df_ddl))
        if df_ddl is not None and not df_ddl.empty:
            # Check if training data already exists
            existing_training_data = self.vanna_client.get_training_data()
            if existing_training_data.empty or config.VANNA_TRAIN_ON_STARTUP:
                print(f"Training Vanna with {self.vanna_manager.current_provider} provider on {self.vanna_manager.current_database} database...")
                
                # Train on DDL statements with selected_columns
                for ddl, table_name in zip(df_ddl['sql'].to_list(), df_ddl['table_name'].to_list()):
                    table_name = table_name.strip('"')
                    if not train_all_tables and table_name not in allowed_tables_list:
                        print(f"Skipping training for table {table_name} as it's not in selected_tables")
                        continue
                    try:
                        vanna_train(
                            vanna_client=self.vanna_client,
                            current_provider=self.vanna_manager.current_provider,
                            ddl=ddl
                        )
                    except Exception as e:
                        print(f"Error training DDL: {e}")
                
                # Get list of all tables
                # tables_df = get_table_names(self.vanna_client, self.vanna_manager.current_database)
                if df_ddl is not None and not df_ddl.empty:
                    # For each table, get distinct examples
                    print("df_ddl['table_name'].to_list(): ",df_ddl['table_name'].to_list())
                    for table_name in df_ddl['table_name'].to_list():
                        table_name = table_name.strip('"')
                        # print("selected_tables:", selected_tables, "type:", type(selected_tables))
                        if not train_all_tables and table_name not in allowed_tables_list:
                            print(f"Skipping training for table {table_name} as it's not in selected_tables")
                            continue
                        # if mssql
                        if self.vanna_manager.current_database.lower() == "mssql":
                            sample_query = f"SELECT TOP 5 * FROM [Nodinite].[dbo].[{table_name}]"
                            print(f"DOCUMENTATION TABLE: {table_name} \nWith query {sample_query}")
                            
                            try:
                                sample_df = self.vanna_client.run_sql(sample_query)
                                training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                                vanna_train(
                                    vanna_client=self.vanna_client,
                                    current_provider=self.vanna_manager.current_provider,
                                    documentation=training_text
                                )
                            except Exception as e:
                                print(f"Error training table data for {table_name}: {e}")
                        elif self.vanna_manager.current_database.lower() == "sqlite":
                            sample_query = f'SELECT * FROM "{table_name}" LIMIT 5;'
                            print(f"DOCUMENTATION TABLE: {table_name} \nWith query {sample_query}")
                            
                            try:
                                sample_df = self.vanna_client.run_sql(sample_query)
                                training_text = f"Table '{table_name}' contains records like:\n{sample_df.to_string()}"
                                vanna_train(
                                    vanna_client=self.vanna_client,
                                    current_provider=self.vanna_manager.current_provider,
                                    documentation=training_text
                                )
                            except Exception as e:
                                print(f"Error training table data for {table_name}: {e}")
                    
                    # training on plan examples
                    if self.vanna_manager.current_database.lower() != "sqlite":
                        if train_all_tables:
                            # Query all tables in the database
                            query = """
                            SELECT 
                                *
                            FROM INFORMATION_SCHEMA.COLUMNS
                            """
                        else:
                            # Query only specific tables
                            table_list = ', '.join([f'"{t}"' for t in allowed_tables_list])
                            query = f"""
                            SELECT 
                                *
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_NAME IN ({table_list})
                            """

                    df_information_schema = self.vanna_client.run_sql(query)
                    plan = self.vanna_client.get_training_plan_generic(df_information_schema)
                    vanna_train(
                        vanna_client=self.vanna_client,
                        current_provider=self.vanna_manager.current_provider,
                        plan = plan
                    )

                    # train on question-SQL pairs only for mssql
                    if self.vanna_manager.current_database.lower() == "postgresql" or self.vanna_manager.current_database.lower() == "mssql":
                        question_sql_pairs = json.load(open("traning_ques_sql.json", "r"))
                        for pair in question_sql_pairs:
                            try:
                                vanna_train(
                                    vanna_client=self.vanna_client,
                                    current_provider=self.vanna_manager.current_provider,
                                    # question_sql=(pair['question'], pair['sql'])
                                    question_sql=pair
                                )
                            except Exception as e:
                                print(f"Error training question-SQL pair: {e}")

                
                print("Training completed!")
            else:
                print(f"Training data already exists for {self.vanna_manager.current_provider}. Skipping training.")
        else:
            print(f"Could not retrieve schema information from {self.vanna_manager.current_database} database.")

    

    def get_database_info(self) -> Dict:
        """Get information about the current Vanna configuration"""
        if self.vanna_manager:
            return {
                "engine": "vanna",
                "provider": self.vanna_manager.current_provider,
                "database": self.vanna_manager.current_database,
                "connection": self.vanna_manager.database_connection,
                "model": (
                    config.VANNA_OPENAI_MODEL 
                    if self.vanna_manager.current_provider == "openai" 
                    else config.VANNA_OLLAMA_MODEL
                )
            }
        else:
            return {"engine": "vanna", "status": "not_initialized"}
    
    def get_vanna_info(self) -> Dict:
        """Get current Vanna configuration info"""
        return get_vanna_info(self.vanna_manager)