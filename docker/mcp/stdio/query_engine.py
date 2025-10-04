# query_engine.py
"""
Query Engine - SQL Database Query Generator with LLM Assistance
Handles natural language to SQL conversion for any database domain
Supports both local SQLite (via API) and remote SQL Server (via API)
"""

# -------------------- Built-in Libraries --------------------
import json
import os
import re
from typing import List

# -------------------- External Libraries --------------------
import aiohttp
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- User-defined Modules --------------------
import config


class QueryEngine:
    """Natural language to SQL query engine - domain agnostic"""

    def __init__(
            self,
            database_url: str,
            tables: List[str],
            system_prompt: str,
            debug: bool = False
    ):
        """
        Initialize the query engine with domain-specific configuration.

        Args:
            database_url: URL of the database API server
            tables: List of table names in the database
            system_prompt: LLM system prompt with schema and instructions
            debug: Enable debug logging
        """
        self.database_url = database_url
        self.all_tables = tables
        self.system_prompt = system_prompt
        self.debug = debug

    def extract_sql_from_json(self, llm_output: str) -> str:
        """Extract SQL from LLM response - handles JSON, markdown, or plain SQL"""
        # 1. Try JSON
        try:
            data = json.loads(llm_output)

            if 'query' in data:
                return data['query'].strip()

        except:
            pass

        # 2. Try markdown SQL block
        match = re.search(r'```sql\s*(.*?)\s*```', llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # 3. Try any code block
        match = re.search(r'```\s*(.*?)\s*```', llm_output, re.DOTALL)

        if match:
            return match.group(1).replace('sql\n', '').strip()

        # 4. Find SELECT statement
        match = re.search(r'(SELECT\s+.*?)(?:;|\n\n|\Z)', llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return ""

    async def execute_query(self, query: str):
        """Execute query via database API server"""
        try:
            if self.debug:
                print(f"[QueryEngine] Executing query: {query}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.database_url}/query",
                        json={"query": query},
                        timeout=300
                ) as response:
                    result = await response.json()

                    if self.debug:
                        print(f"[QueryEngine] Query result: {result}")

                    if result.get("success"):
                        return result.get("data", [])

                    else:
                        if self.debug:
                            print(f"[QueryEngine] Database error: {result.get('error')}")

                        return []

        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Database connection error: {e}")

            return []

    async def execute_query_stream(self, query: str):
        """Stream database results via API"""
        try:
            if self.debug:
                print(f"[QueryEngine] Starting streaming query: {query}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.database_url}/query_stream",
                        json={"query": query},
                        timeout=300
                ) as response:

                    if response.status != 200:
                        if self.debug:
                            print(f"[QueryEngine] Database connection failed, status: {response.status}")

                        yield {"success": False, "error": "Database connection failed"}

                        return

                    results = []

                    async for line in response.content:
                        if line.strip():
                            try:
                                data = json.loads(line.decode())

                                if self.debug:
                                    print(f"[QueryEngine] Streaming data: {data}")

                                if data["type"] == "start":
                                    yield {
                                        "success": True,
                                        "sql_query": f"```sql {data['query']} ```",
                                        "streaming": True,
                                        "status": "started"
                                    }

                                elif data["type"] == "row":
                                    results.append(data["data"])

                                    yield {
                                        "success": True,
                                        "type": "row",
                                        "data": data["data"],
                                        "index": data["index"],
                                        "running_total": len(results)
                                    }

                                elif data["type"] == "complete":
                                    yield {
                                        "success": True,
                                        "type": "complete",
                                        "results": results,
                                        "record_count": data["total_rows"],
                                        "status": "finished"
                                    }

                                elif data["type"] == "error":
                                    if self.debug:
                                        print(f"[QueryEngine] Database error in stream: {data['error']}")

                                    yield {"success": False, "error": data["error"]}

                            except json.JSONDecodeError:
                                if self.debug:
                                    print(f"[QueryEngine] Failed to decode JSON line: {line}")

                                continue

        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Stream execution error: {e}")

            yield {"success": False, "error": f"Database connection error: {str(e)}"}

    async def generate_sql_stream(self, user_question: str, provider: str = "ollama"):
        """Generate SQL from natural language question - streaming"""

        if self.debug:
            print(f"[QueryEngine] Starting SQL generation for: {user_question}")
            print(f"[QueryEngine] Provider: {provider}")

        yield {"status": "generating_sql", "message": "Generating SQL query..."}

        # Set up LLM
        if provider == "ollama":
            llm = ChatOllama(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                stream=True,
                base_url=config.OLLAMA_BASE_URL
            )
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            llm = ChatOpenAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                streaming=True,
                api_key=api_key
            )
        elif provider == "mistral":
            api_key = os.environ.get("MISTRAL_API_KEY")
            llm = ChatMistralAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                streaming=True,
                mistral_api_key=api_key,
                endpoint=config.MISTRAL_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_question)
        ]

        yield {"status": "streaming_sql", "message": "Streaming SQL generation..."}

        # Stream the LLM response
        accumulated_content = ""

        if self.debug:
            print(f"[QueryEngine] Starting LLM streaming...")

        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_content += chunk.content
                if self.debug:
                    print(f"[QueryEngine] LLM chunk: {chunk.content}")

                yield {
                    "status": "sql_streaming",
                    "partial_content": chunk.content,
                    "accumulated_content": accumulated_content
                }

        # Parse final SQL
        if self.debug:
            print(f"[QueryEngine] Final accumulated content: {accumulated_content}")

        sql_query = self.extract_sql_from_json(accumulated_content)

        if not sql_query:
            if self.debug:
                print(f"[QueryEngine] Failed to extract SQL from: {accumulated_content}")
            yield {"status": "error", "message": "Failed to extract SQL from LLM response"}
            return

        # Clean up SQL - ensure single line
        sql_query = ' '.join(sql_query.replace('\\n', ' ').replace('\n', ' ').split())

        if self.debug:
            print(f"[QueryEngine] Final cleaned SQL: {sql_query}")

        yield {"status": "sql_complete", "sql_query": sql_query}

    async def generate_sql(self, user_question: str, provider: str = "ollama") -> str:
        """Non-streaming version - convenience method"""
        sql_query = ""

        async for update in self.generate_sql_stream(user_question, provider):
            if update.get("status") == "sql_complete":
                sql_query = update.get("sql_query", "")
                break

        return sql_query
