# vanna_query_engine.py
"""
Vanna Query Engine Wrapper - Integrates trained Vanna model with existing QueryEngine interface
Provides the same interface as QueryEngine but uses Vanna for SQL generation and execution
"""

import asyncio
import json
from typing import List, Dict, Optional, AsyncGenerator

try:
    from vanna_train import VannaModelManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from vanna_train import VannaModelManager
import config


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
                    df = self.vanna_client.run_sql(sql_query)
                    if df is not None and not df.empty:
                        # Convert DataFrame to list of dictionaries
                        return df.to_dict('records')
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