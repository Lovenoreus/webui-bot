# query_engine.py
"""
Query Engine - SQL Database Query Generator with LLM Assistance
Handles natural language to SQL conversion for any database domain
Supports both local SQLite (via API) and remote SQL Server (direct connection)
"""

import json
import asyncio
import os
from typing import List, Dict, Optional
import re

import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

import config


class QueryEngine:
    """Natural language to SQL query engine - domain agnostic"""

    def __init__(
            self,
            database_url: str,
            tables: List[str],
            table_variations: Dict[str, List[str]],
            system_prompt: str,
            debug: bool = False,
            use_remote: bool = False
    ):
        """
        Initialize the query engine with domain-specific configuration.

        Args:
            database_url: URL of the database server (for local) or connection info (for remote)
            tables: List of table names in the database
            table_variations: Dict mapping table names to keyword variations to filter
            system_prompt: LLM system prompt with schema and instructions
            debug: Enable debug logging
            use_remote: Use remote SQL Server instead of local SQLite API
        """
        self.database_url = database_url
        self.all_tables = tables
        self.table_variations = table_variations
        self.system_prompt = system_prompt
        self.debug = debug
        self.use_remote = use_remote

        # Initialize remote SQL Server connection if needed
        if self.use_remote:
            self._init_remote_connection()
        else:
            self.engine = None
            self.async_session = None

    def _init_remote_connection(self):
        """Initialize async SQL Server connection"""
        server = config.SQL_SERVER_HOST
        database = config.SQL_SERVER_DATABASE
        driver = config.SQL_SERVER_DRIVER

        if config.SQL_SERVER_USE_WINDOWS_AUTH:
            connection_string = (
                f"mssql+aioodbc://@{server}/{database}"
                f"?driver={driver.replace(' ', '+')}"
                "&trusted_connection=yes"
            )

        else:
            username = config.SQL_SERVER_USERNAME
            password = config.SQL_SERVER_PASSWORD
            connection_string = (
                f"mssql+aioodbc://{username}:{password}@{server}/{database}"
                f"?driver={driver.replace(' ', '+')}"
            )

        if self.debug:
            print(f"[QueryEngine] Initializing remote SQL Server connection")
            print(f"[QueryEngine] Server: {server}")
            print(f"[QueryEngine] Database: {database}")
            print(f"[QueryEngine] Auth: {'Windows' if config.SQL_SERVER_USE_WINDOWS_AUTH else 'SQL Server'}")

        self.engine = create_async_engine(
            connection_string,
            echo=self.debug,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=0
        )

        self.async_session = sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )

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
        """Run a query on the database - routes to local or remote"""
        if self.use_remote:
            return await self._execute_query_remote(query)
        else:
            return await self._execute_query_local(query)

    async def _execute_query_local(self, query: str):
        """Execute query via local SQLite API server"""
        try:
            if self.debug:
                print(f"[QueryEngine] Executing LOCAL query: {query}")

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

    async def _execute_query_remote(self, query: str):
        """Execute query directly on remote SQL Server"""
        try:
            if self.debug:
                print(f"[QueryEngine] Executing REMOTE query: {query}")

            async with self.async_session() as session:
                result = await session.execute(text(query))

                # Fetch all rows and convert to dict
                rows = result.fetchall()
                columns = result.keys()

                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert datetime and decimal to string for JSON serialization
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif hasattr(value, '__float__'):
                            value = float(value)
                        row_dict[col] = value
                    data.append(row_dict)

                if self.debug:
                    print(f"[QueryEngine] Remote query returned {len(data)} rows")

                return data

        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Remote database error: {e}")
            return []

    async def execute_query_stream(self, query: str):
        """Stream database results - routes to local or remote"""
        if self.use_remote:
            async for result in self._execute_query_stream_remote(query):
                yield result
        else:
            async for result in self._execute_query_stream_local(query):
                yield result

    async def _execute_query_stream_local(self, query: str):
        """Stream results from local SQLite API"""
        try:
            if self.debug:
                print(f"[QueryEngine] Starting LOCAL streaming query: {query}")

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

    async def _execute_query_stream_remote(self, query: str):
        """Stream results from remote SQL Server"""
        try:
            if self.debug:
                print(f"[QueryEngine] Starting REMOTE streaming query: {query}")

            yield {
                "success": True,
                "sql_query": f"```sql {query} ```",
                "streaming": True,
                "status": "started"
            }

            async with self.async_session() as session:
                result = await session.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()

                results = []
                for idx, row in enumerate(rows, 1):
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert datetime and decimal to string for JSON serialization
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif hasattr(value, '__float__'):
                            value = float(value)
                        row_dict[col] = value

                    results.append(row_dict)

                    yield {
                        "success": True,
                        "type": "row",
                        "data": row_dict,
                        "index": idx,
                        "running_total": len(results)
                    }

                yield {
                    "success": True,
                    "type": "complete",
                    "results": results,
                    "record_count": len(results),
                    "status": "finished"
                }

        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Remote stream error: {e}")
            yield {"success": False, "error": f"Remote database error: {str(e)}"}

    def filter_keywords_for_table(self, keywords: List[str], table_name: str) -> List[str]:
        """Filter out keywords that match the table name to avoid false matches"""
        table_name_lower = table_name.lower()
        filtered_keywords = []

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Skip if keyword matches table name exactly
            if keyword_lower == table_name_lower:
                continue

            # Skip if keyword is part of table name or vice versa
            if keyword_lower in table_name_lower or table_name_lower in keyword_lower:
                continue

            # Check against table variations
            skip_keyword = False
            for table_var, variations in self.table_variations.items():
                if table_name_lower.startswith(table_var) and keyword_lower in variations:
                    skip_keyword = True
                    break

            if not skip_keyword:
                filtered_keywords.append(keyword)

        if self.debug:
            print(f"[QueryEngine] Filtered keywords for {table_name}: {keywords} -> {filtered_keywords}")

        return filtered_keywords

    async def get_table_string_columns(self, table_name: str) -> List[str]:
        """Get all string/text columns from a table dynamically"""
        try:
            if self.use_remote:
                # SQL Server schema query
                schema_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}'
                """
                schema_results = await self.execute_query(schema_query)

                string_columns = []
                for column_info in schema_results:
                    column_name = column_info['COLUMN_NAME']
                    column_type = column_info['DATA_TYPE'].upper()

                    # Check if it's a string/text type in SQL Server
                    if any(text_type in column_type for text_type in ['VARCHAR', 'NVARCHAR', 'TEXT', 'CHAR', 'NCHAR']):
                        string_columns.append(column_name)
            else:
                # SQLite schema query
                schema_query = f"PRAGMA table_info({table_name})"
                schema_results = await self.execute_query(schema_query)

                string_columns = []
                for column_info in schema_results:
                    column_name = column_info['name']
                    column_type = column_info['type'].upper()

                    # Check if it's a string/text type
                    if any(text_type in column_type for text_type in ['VARCHAR', 'TEXT', 'CHAR']):
                        string_columns.append(column_name)

            if self.debug:
                print(f"[QueryEngine] String columns for {table_name}: {string_columns}")

            return string_columns
        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Error getting string columns for {table_name}: {e}")
            return []

    async def search_table_for_keywords(self, table_name: str, keywords: List[str]) -> List[Dict]:
        """Search a single table for keyword matches in all string columns"""
        filtered_keywords = self.filter_keywords_for_table(keywords, table_name)

        if not filtered_keywords:
            if self.debug:
                print(f"[QueryEngine] No relevant keywords for {table_name} after filtering")
            return []

        if self.debug:
            print(f"[QueryEngine] Searching {table_name} for keywords: {filtered_keywords}")

        string_columns = await self.get_table_string_columns(table_name)
        if not string_columns:
            if self.debug:
                print(f"[QueryEngine] No string columns found in {table_name}")
            return []

        # Build WHERE clause for keyword matching
        where_conditions = []
        for keyword in filtered_keywords:
            keyword_conditions = []
            for column in string_columns:
                keyword_conditions.append(f"LOWER({column}) LIKE LOWER('%{keyword}%')")
            if keyword_conditions:
                where_conditions.append(f"({' OR '.join(keyword_conditions)})")

        if not where_conditions:
            return []

        query = f"""
        SELECT {', '.join(string_columns)}
        FROM {table_name} 
        WHERE {' OR '.join(where_conditions)}
        """

        try:
            results = await self.execute_query(query)
            if self.debug:
                print(f"[QueryEngine] Found {len(results)} matches in {table_name}")
                if results:
                    print(f"[QueryEngine] Sample match: {results[0]}")
            return results
        except Exception as e:
            if self.debug:
                print(f"[QueryEngine] Error searching {table_name}: {e}")
            return []

    async def search_all_tables_async(self, keywords: List[str]) -> List[Dict]:
        """Search all tables for keyword matches and return structured results"""
        if self.debug:
            print(f"[QueryEngine] Starting async search for keywords: {keywords}")

        search_tasks = []
        for table_name in self.all_tables:
            task = self.search_table_for_keywords(table_name, keywords)
            search_tasks.append((table_name, task))

        all_results = []
        results = await asyncio.gather(*[task for _, task in search_tasks], return_exceptions=True)

        for i, (table_name, _) in enumerate(search_tasks):
            if isinstance(results[i], Exception):
                if self.debug:
                    print(f"[QueryEngine] Error searching {table_name}: {results[i]}")
                continue

            table_results = results[i]
            if table_results:
                for row in table_results:
                    all_results.append({
                        'table': table_name,
                        'row': row,
                        'matched_keywords': self.filter_keywords_for_table(keywords, table_name)
                    })

        if self.debug:
            print(f"[QueryEngine] Total matches found across all tables: {len(all_results)}")

        return all_results

    def generate_hit_results(self, matches: List[Dict], keywords: List[str]) -> List[str]:
        """Generate hit result strings showing what was found where"""
        hit_results = []

        for match in matches:
            table = match['table']
            row = match['row']
            matched_keywords = match['matched_keywords']

            for column, value in row.items():
                if not value:
                    continue

                value_str = str(value).lower()
                found_keywords = []

                for keyword in matched_keywords:
                    if keyword.lower() in value_str:
                        found_keywords.append(keyword)

                if found_keywords:
                    hit_text = f"In {table} table found '{value}' in column {column}"
                    if len(found_keywords) > 1:
                        hit_text += f" (matched keywords: {', '.join(found_keywords)})"
                    hit_results.append(hit_text)

        if self.debug:
            print(f"[QueryEngine] Generated hit results:")
            for hit in hit_results:
                print(f"[QueryEngine]   - {hit}")

        return hit_results

    async def generate_sql_stream(self, user_question: str, keywords: List[str], provider: str = "ollama"):
        """Generate SQL using keyword search context - streaming with single LLM call"""

        if self.debug:
            print(f"[QueryEngine] Starting SQL generation for: {user_question}")
            print(f"[QueryEngine] Keywords: {keywords}")
            print(f"[QueryEngine] Provider: {provider}")
            print(f"[QueryEngine] Database mode: {'REMOTE' if self.use_remote else 'LOCAL'}")

        yield {"status": "generating_sql", "message": "Searching database for context..."}

        # Search all tables for keyword matches
        matches = await self.search_all_tables_async(keywords)
        hit_results = self.generate_hit_results(matches, keywords)

        if self.debug:
            print(f"[QueryEngine] Generated {len(hit_results)} hit results")

        # Add context from search results to system prompt
        seen = set()
        unique_hits = []
        for hit in hit_results:
            if hit not in seen:
                unique_hits.append(hit)
                seen.add(hit)

        if unique_hits:
            hint_text = f"\n\nDATA CONTEXT FROM SEARCH:\n"
            for hit in unique_hits[:10]:
                hint_text += f"- {hit}\n"
            enhanced_prompt = self.system_prompt + hint_text

            if self.debug:
                print(f"[QueryEngine] Added {len(unique_hits)} context hints to prompt")
        else:
            enhanced_prompt = self.system_prompt
            if self.debug:
                print(f"[QueryEngine] No context hits found, using base prompt only")

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
            SystemMessage(content=enhanced_prompt),
            HumanMessage(content=user_question)
        ]

        yield {"status": "streaming_sql", "message": "Generating SQL query..."}

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

    async def generate_sql(self, user_question: str, keywords: List[str], provider: str = "ollama") -> str:
        """Non-streaming version - convenience method"""
        sql_query = ""
        async for update in self.generate_sql_stream(user_question, keywords, provider):
            if update.get("status") == "sql_complete":
                sql_query = update.get("sql_query", "")
                break
        return sql_query
