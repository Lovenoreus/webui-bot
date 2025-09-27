import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional, List, Any, Union, Literal

import aiohttp
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
# Reuse your existing class/models

from vector_mistral_tool import hospital_support_questions_tool
from create_jira_ticket import create_jira_ticket

from active_directory import ActiveDirectory
from active_directory import CreateUserPayload, PasswordProfile
from vector_database_tools import cosmic_database_tool

# RAG imports - temporarily commented out to focus on cosmic_database_tool
# from rag_tools import retrieve_documents_tool, get_rag_tool_schema, RAGRetrieveRequest

# ++++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY PYDANTIC MODELS START
# ++++++++++++++++++++++++++++++++

# Simplified models for specific endpoints (without action field)
class UserUpdates(BaseModel):
    updates: Dict[str, Any] = Field(..., description="Fields to update")

class RoleAddMember(BaseModel):
    user_id: str = Field(..., description="User ID to add to role")

class GroupMemberRequest(BaseModel):
    user_id: str = Field(..., description="User ID")

class GroupOwnerRequest(BaseModel):
    user_id: str = Field(..., description="User ID")

class GroupUpdates(BaseModel):
    updates: Dict[str, Any] = Field(..., description="Fields to update")

class RoleInstantiation(BaseModel):
    roleTemplateId: str = Field(..., description="Role template ID to instantiate")

class CreateUserRequest(BaseModel):
    action: Literal["create_user"]
    user: Dict[str, Any] = Field(..., description="Graph API user payload")

class CreateGroupRequest(BaseModel):
    action: Literal["create_group"]
    display_name: str = Field(..., description="Display name for the group")
    mail_nickname: str = Field(..., description="Mail nickname for the group")
    description: Optional[str] = Field(None, description="Group description")
    group_type: Optional[str] = Field("security", description="Type of group (security, unified)")
    visibility: Optional[str] = Field(None, description="Group visibility")
    membership_rule: Optional[str] = Field(None, description="Dynamic membership rule")
    owners: Optional[List[str]] = Field(None, description="List of owner user IDs")
    members: Optional[List[str]] = Field(None, description="List of member user IDs")

# ++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY PYDANTIC MODELS END
# ++++++++++++++++++++++++++++++


# MCP Protocol Models
class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPToolsListResponse(BaseModel):
    tools: List[MCPTool]


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


class MCPContent(BaseModel):
    type: Literal["text"]
    text: str


class MCPToolCallResponse(BaseModel):
    content: List[MCPContent]
    isError: Optional[bool] = False


class MCPServerInfo(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    author: Optional[str] = None
    homepage: Optional[str] = None
    capabilities: Dict[str, bool]


class ThreadTicketRequest(BaseModel):
    thread_id: str
# Existing models currently used in the code
class KnownProblemsRequest(BaseModel):
    query: str
    ticket_id: Optional[str] = None  # Add optional ticket_id field

class CreateJiraRequest(BaseModel):
    thread_id: str
    conversation_topic: str
    description: str
    location: str
    queue: str
    priority: str
    department: str
    name: str
    category: str


ad = ActiveDirectory()

ADMIN_API_KEY = os.getenv("MCP_ADMIN_API_KEY")  # set to enable header guard


load_dotenv()

# Debug flag
DEBUG = True

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

app = FastAPI(title="MCP Server API", description="Standalone MCP Tools Server with LLM SQL Generation")

# Add CORS middleware - this must be added immediately after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URLs: ["http://localhost:3000", "http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request models
class GreetRequest(BaseModel):
    name: str


class QueryDatabaseRequest(BaseModel):
    query: str
    keywords: List[str]


class WeatherRequest(BaseModel):
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    units: Optional[str] = None


class CosmicDatabaseRequest(BaseModel):
    query: str = Field(..., description="The search query to find relevant information in the cosmic database")

class HospitalSupportQuestionsRequest(BaseModel):
    query: str = Field(..., description="The support question to search for in the hospital support knowledge base")


# RAG request model is now imported from rag_tools


def extract_sql_from_json(llm_output: str) -> str:
    """Extract SQL from LLM JSON response"""
    try:
        # Try to parse as JSON first
        data = json.loads(llm_output)
        return data.get('query', '')
    except:
        # Try regex extraction for malformed JSON
        pattern = r'"query"\s*:\s*"([^"]*)"'
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


def get_database_server_url():
    """Determine the correct database server URL based on environment"""
    if os.environ.get('DOCKER_CONTAINER') or os.path.exists('/.dockerenv'):
        return config.MCP_DOCKER_DATABASE_SERVER_URL
    else:
        return config.MCP_DATABASE_SERVER_URL


DATABASE_SERVER_URL = get_database_server_url()

if DEBUG:
    print(f"[MCP DEBUG] Database server URL: {DATABASE_SERVER_URL}")

# RAG models are now initialized in the rag_tools module


async def execute_query(query: str):
    """Run a query on the FastAPI database server"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Executing query: {query}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{DATABASE_SERVER_URL}/query",
                    json={"query": query},
                    timeout=300
            ) as response:
                result = await response.json()

                if DEBUG:
                    print(f"[MCP DEBUG] Query result: {result}")

                if result.get("success"):
                    return result.get("data", [])
                else:
                    if DEBUG:
                        print(f"[MCP DEBUG] Database error: {result.get('error')}")
                    return []
    except Exception as e:
        if DEBUG:
            print(f"[MCP DEBUG] Database connection error: {e}")
        return []


async def execute_query_stream(query: str):
    """Stream database results as they arrive"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Starting streaming query: {query}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{DATABASE_SERVER_URL}/query_stream",
                    json={"query": query},
                    timeout=300
            ) as response:

                if response.status != 200:
                    if DEBUG:
                        print(f"[MCP DEBUG] Database connection failed, status: {response.status}")
                    yield {"success": False, "error": "Database connection failed"}
                    return

                results = []
                async for line in response.content:
                    if line.strip():
                        try:
                            data = json.loads(line.decode())

                            if DEBUG:
                                print(f"[MCP DEBUG] Streaming data: {data}")

                            if data["type"] == "start":
                                yield {
                                    "success": True,
                                    "sql_query": data["query"],
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
                                if DEBUG:
                                    print(f"[MCP DEBUG] Database error in stream: {data['error']}")
                                yield {"success": False, "error": data["error"]}

                        except json.JSONDecodeError:
                            if DEBUG:
                                print(f"[MCP DEBUG] Failed to decode JSON line: {line}")
                            continue

    except Exception as e:
        if DEBUG:
            print(f"[MCP DEBUG] Stream execution error: {e}")
        yield {"success": False, "error": f"Database connection error: {str(e)}"}


class DatabaseKeywordHints:
    def __init__(self):
        self.all_tables = ['Users', 'Departments', 'Roles', 'Permissions', 'Teams', 'User_Roles', 'User_Teams',
                           'Role_Permissions']

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

            # Special cases for common table variations
            table_variations = {
                'users': ['user', 'employee', 'person', 'people'],
                'departments': ['department', 'dept'],
                'roles': ['role'],
                'permissions': ['permission', 'perm'],
                'teams': ['team']
            }

            skip_keyword = False
            for table_var, variations in table_variations.items():
                if table_name_lower.startswith(table_var) and keyword_lower in variations:
                    skip_keyword = True
                    break

            if not skip_keyword:
                filtered_keywords.append(keyword)

        if DEBUG:
            print(f"[MCP DEBUG] Filtered keywords for {table_name}: {keywords} -> {filtered_keywords}")

        return filtered_keywords

    async def get_table_string_columns(self, table_name: str) -> List[str]:
        """Get all string/text columns from a table dynamically"""
        try:
            schema_query = f"PRAGMA table_info({table_name})"
            schema_results = await execute_query(schema_query)

            string_columns = []
            for column_info in schema_results:
                column_name = column_info['name']
                column_type = column_info['type'].upper()

                # Check if it's a string/text type
                if any(text_type in column_type for text_type in ['VARCHAR', 'TEXT', 'CHAR']):
                    string_columns.append(column_name)

            if DEBUG:
                print(f"[MCP DEBUG] String columns for {table_name}: {string_columns}")

            return string_columns
        except Exception as e:
            if DEBUG:
                print(f"[MCP DEBUG] Error getting string columns for {table_name}: {e}")
            return []

    async def search_table_for_keywords(self, table_name: str, keywords: List[str]) -> List[Dict]:
        """Search a single table for keyword matches in all string columns"""
        # Filter keywords to avoid table name matches
        filtered_keywords = self.filter_keywords_for_table(keywords, table_name)

        if not filtered_keywords:
            if DEBUG:
                print(f"[MCP DEBUG] No relevant keywords for {table_name} after filtering")
            return []

        if DEBUG:
            print(f"[MCP DEBUG] Searching {table_name} for keywords: {filtered_keywords}")

        # Get string columns dynamically
        string_columns = await self.get_table_string_columns(table_name)
        if not string_columns:
            if DEBUG:
                print(f"[MCP DEBUG] No string columns found in {table_name}")
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

        # Construct the search query
        query = f"""
        SELECT {', '.join(string_columns)}
        FROM {table_name} 
        WHERE {' OR '.join(where_conditions)}
        """

        try:
            results = await execute_query(query)
            if DEBUG:
                print(f"[MCP DEBUG] Found {len(results)} matches in {table_name}")
                if results:
                    print(f"[MCP DEBUG] Sample match: {results[0]}")
            return results
        except Exception as e:
            if DEBUG:
                print(f"[MCP DEBUG] Error searching {table_name}: {e}")
            return []

    async def search_all_tables_async(self, keywords: List[str]) -> List[Dict]:
        """Search all tables for keyword matches and return structured results"""
        if DEBUG:
            print(f"[MCP DEBUG] Starting async search for keywords: {keywords}")

        # Create async tasks for each table search
        search_tasks = []
        for table_name in self.all_tables:
            task = self.search_table_for_keywords(table_name, keywords)
            search_tasks.append((table_name, task))

        # Execute all searches concurrently
        all_results = []
        results = await asyncio.gather(*[task for _, task in search_tasks], return_exceptions=True)

        # Process results
        for i, (table_name, _) in enumerate(search_tasks):
            if isinstance(results[i], Exception):
                if DEBUG:
                    print(f"[MCP DEBUG] Error searching {table_name}: {results[i]}")
                continue

            table_results = results[i]
            if table_results:
                # Add table context to results
                for row in table_results:
                    all_results.append({
                        'table': table_name,
                        'row': row,
                        'matched_keywords': self.filter_keywords_for_table(keywords, table_name)
                    })

        if DEBUG:
            print(f"[MCP DEBUG] Total matches found across all tables: {len(all_results)}")

        return all_results

    def generate_hit_results(self, matches: List[Dict], keywords: List[str]) -> List[str]:
        """Generate hit result strings showing what was found where"""
        hit_results = []

        for match in matches:
            table = match['table']
            row = match['row']
            matched_keywords = match['matched_keywords']

            # Find which columns had matches
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

        if DEBUG:
            print(f"[MCP DEBUG] Generated hit results:")
            for hit in hit_results:
                print(f"[MCP DEBUG]   - {hit}")

        return hit_results


# Initialize the hint generator
hint_generator = DatabaseKeywordHints()


async def get_sql_query_stream(user_question: str, keywords: List[str], provider: str = "ollama"):
    """Generate SQL using keyword search context - SINGLE LLM CALL"""

    if DEBUG:
        print(f"[MCP DEBUG] Starting SQL generation for: {user_question}")
        print(f"[MCP DEBUG] Keywords: {keywords}")
        print(f"[MCP DEBUG] Provider: {provider}")

    yield {"status": "generating_sql", "message": "Searching database for context..."}

    # STEP 1: Search all tables for keyword matches
    matches = await hint_generator.search_all_tables_async(keywords)

    # STEP 2: Generate hit results showing what was found
    hit_results = hint_generator.generate_hit_results(matches, keywords)

    if DEBUG:
        print(f"[MCP DEBUG] Generated {len(hit_results)} hit results")

    # STEP 3: Build system prompt with enhanced SQL generation rules
    base_system_prompt = """You are an expert SQL generator for SQLite database queries.

DATABASE SCHEMA:
CREATE TABLE Departments (department_id INTEGER PRIMARY KEY AUTOINCREMENT, department_name VARCHAR(100), manager_id INTEGER NULL);
CREATE TABLE Users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR(100), email VARCHAR(100), department_id INTEGER, hire_date DATE, FOREIGN KEY (department_id) REFERENCES Departments(department_id));
CREATE TABLE Roles (role_id INTEGER PRIMARY KEY AUTOINCREMENT, role_name VARCHAR(100), description TEXT);
CREATE TABLE Permissions (permission_id INTEGER PRIMARY KEY AUTOINCREMENT, permission_name VARCHAR(100), description TEXT);
CREATE TABLE Role_Permissions (role_id INTEGER, permission_id INTEGER, PRIMARY KEY (role_id, permission_id), FOREIGN KEY (role_id) REFERENCES Roles(role_id), FOREIGN KEY (permission_id) REFERENCES Permissions(permission_id));
CREATE TABLE Teams (team_id INTEGER PRIMARY KEY AUTOINCREMENT, team_name VARCHAR(100), department_id INTEGER, FOREIGN KEY (department_id) REFERENCES Departments(department_id));
CREATE TABLE User_Roles (user_id INTEGER, role_id INTEGER, assigned_date DATE, PRIMARY KEY (user_id, role_id), FOREIGN KEY (user_id) REFERENCES Users(user_id), FOREIGN KEY (role_id) REFERENCES Roles(role_id));
CREATE TABLE User_Teams (user_id INTEGER, team_id INTEGER, PRIMARY KEY (user_id, team_id), FOREIGN KEY (user_id) REFERENCES Users(user_id), FOREIGN KEY (team_id) REFERENCES Teams(team_id));

CRITICAL SQL GENERATION RULES:
1. SINGLE LINE ONLY - Generate SQL as one continuous line with no newlines, line breaks, or formatting.
2. MATCH THE QUERY EXACTLY - Understand what the user is asking for and generate SQL that returns exactly that data.
3. USE PROPER JOINS - For complex queries involving multiple tables, use appropriate JOIN statements.
4. USE DISTINCT when needed to avoid duplicate rows.
5. For text searches, use LIKE with % wildcards: WHERE column LIKE '%searchterm%'
6. Always use table aliases for clarity in joins.

COMMON QUERY PATTERNS:
- "users and permissions" = JOIN Users -> User_Roles -> Role_Permissions -> Permissions
- "users and their complete permissions" = JOIN Users -> User_Roles -> Role_Permissions -> Permissions  
- "users in department" = JOIN Users -> Departments
- "users in team" = JOIN Users -> User_Teams -> Teams
- "roles and permissions" = JOIN Roles -> Role_Permissions -> Permissions

EXAMPLES:
Query: "List all users and their permissions"
SQL: SELECT DISTINCT u.name, p.permission_name FROM Users u JOIN User_Roles ur ON u.user_id = ur.user_id JOIN Role_Permissions rp ON ur.role_id = rp.role_id JOIN Permissions p ON rp.permission_id = p.permission_id ORDER BY u.name, p.permission_name;

Query: "List all users and each of their complete permissions"  
SQL: SELECT DISTINCT u.name AS username, p.permission_name FROM Users u JOIN User_Roles ur ON u.user_id = ur.user_id JOIN Role_Permissions rp ON ur.role_id = rp.role_id JOIN Permissions p ON rp.permission_id = p.permission_id ORDER BY u.name, p.permission_name;

Query: "Show users in Engineering department"
SQL: SELECT u.name, u.email FROM Users u JOIN Departments d ON u.department_id = d.department_id WHERE d.department_name LIKE '%Engineering%';

Query: "Count users by department"
SQL: SELECT d.department_name, COUNT(u.user_id) as user_count FROM Departments d LEFT JOIN Users u ON d.department_id = u.department_id GROUP BY d.department_name;

Return ONLY valid single-line SQL in JSON format: {"query": "SELECT..."}"""

    # STEP 4: Add data context from search results
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
        system_prompt = base_system_prompt + hint_text

        if DEBUG:
            print(f"[MCP DEBUG] Added {len(unique_hits)} context hints to prompt")
    else:
        system_prompt = base_system_prompt
        if DEBUG:
            print(f"[MCP DEBUG] No context hits found, using base prompt only")

    # STEP 5: Set up LLM
    if provider == "ollama":
        llm = ChatOllama(model=config.MCP_AGENT_MODEL_NAME, temperature=0, stream=True, base_url=config.OLLAMA_BASE_URL)

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

        llm = ChatOpenAI(model=config.MCP_AGENT_MODEL_NAME, temperature=0, streaming=True, api_key=api_key)

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
        raise ValueError("Unsupported provider")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]

    yield {"status": "streaming_sql", "message": "Generating SQL query..."}

    # STEP 6: Stream the LLM response (SINGLE CALL)
    accumulated_content = ""

    if DEBUG:
        print(f"[MCP DEBUG] Starting LLM streaming with context-enhanced prompt...")

    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content') and chunk.content:
            accumulated_content += chunk.content
            if DEBUG:
                print(f"[MCP DEBUG] LLM chunk: {chunk.content}")

            yield {
                "status": "sql_streaming",
                "partial_content": chunk.content,
                "accumulated_content": accumulated_content
            }

    # STEP 7: Parse final SQL
    if DEBUG:
        print(f"[MCP DEBUG] Final accumulated content: {accumulated_content}")

    sql_query = extract_sql_from_json(accumulated_content)

    if not sql_query:
        if DEBUG:
            print(f"[MCP DEBUG] Failed to extract SQL from: {accumulated_content}")
        yield {"status": "error", "message": "Failed to extract SQL from LLM response"}
        return

    # Clean up SQL - ensure single line
    sql_query = ' '.join(sql_query.replace('\\n', ' ').replace('\n', ' ').split())

    if DEBUG:
        print(f"[MCP DEBUG] Final cleaned SQL: {sql_query}")

    yield {"status": "sql_complete", "sql_query": sql_query}


async def get_sql_query(user_question: str, keywords: List[str], provider: str = "ollama") -> str:
    """Non-streaming version for backward compatibility"""
    sql_query = ""
    async for update in get_sql_query_stream(user_question, keywords, provider):
        if update.get("status") == "sql_complete":
            sql_query = update.get("sql_query", "")
            break
    return sql_query


@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name"""
    try:
        message = f"Hello, {request.name}!"
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_database")
async def query_database_endpoint(request: QueryDatabaseRequest):
    """Query database with natural language using LLM SQL generation"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Received DB request: {request.query}")
            print(f"[MCP DEBUG] Keywords: {request.keywords}")

        if config.MCP_PROVIDER_OLLAMA:
            provider = "ollama"

        elif config.MCP_PROVIDER_OPENAI:
            provider = "openai"

        elif config.MCP_PROVIDER_MISTRAL:
            provider = "mistral"

        else:
            raise ValueError("Unsupported provider")

        if DEBUG:
            print(f'[MCP DEBUG] Using provider: {provider}')

        # Generate SQL using LLM with both query and keywords
        sql_query = await get_sql_query(request.query, request.keywords, provider)

        # Remove all \n characters
        sql_query = sql_query.replace('\\n', ' ').replace('\n', ' ')

        if DEBUG:
            print(f"[MCP DEBUG] Generated SQL query: {sql_query}")

        if not sql_query:
            return {
                "success": False,
                "error": "Failed to generate SQL",
                "original_query": request.query
            }

        # Execute the SQL query
        results = await execute_query(sql_query)

        if DEBUG:
            print(f"[MCP DEBUG] Query results: {results}")

        return {
            "success": True,
            "sql_query": sql_query,
            "results": results,
            "original_query": request.query,
            "record_count": len(results)
        }

    except Exception as e:
        if DEBUG:
            print(f"[MCP DEBUG] Error in query_database: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_query": request.query
        }


@app.post("/query_database_stream")
async def query_database_stream_endpoint(request: QueryDatabaseRequest):
    """Stream database query results with SQL generation"""

    async def generate_response():
        try:
            if DEBUG:
                print(f"[MCP DEBUG] Received streaming request: {request.query}")
                print(f"[MCP DEBUG] Keywords: {request.keywords}")

            if config.MCP_PROVIDER_OLLAMA:
                provider = "ollama"

            elif config.MCP_PROVIDER_OPENAI:
                provider = "openai"

            elif config.MCP_PROVIDER_MISTRAL:
                provider = "mistral"

            else:
                raise ValueError("Unsupported provider")

            # Stream SQL generation with keyword context
            sql_query = ""

            async for sql_update in get_sql_query_stream(request.query, request.keywords, provider):
                if DEBUG:
                    print(f"[MCP DEBUG] SQL update: {sql_update}")
                yield json.dumps(sql_update) + "\n"
                if sql_update.get("status") == "sql_complete":
                    sql_query = sql_update.get("sql_query", "")

            if not sql_query:
                if DEBUG:
                    print("[MCP DEBUG] No SQL query generated!")
                yield json.dumps({
                    "success": False,
                    "error": "Failed to generate SQL"
                }) + "\n"
                return

            # Stream database execution
            if DEBUG:
                print(f"[MCP DEBUG] Executing SQL: {sql_query}")

            yield json.dumps({
                "status": "executing_query",
                "message": "Executing SQL query...",
                "sql_query": sql_query
            }) + "\n"

            # Stream results from database
            async for db_result in execute_query_stream(sql_query):
                db_result["original_query"] = request.query
                if DEBUG:
                    print(f"[MCP DEBUG] Database result: {db_result}")
                yield json.dumps(db_result) + "\n"

        except Exception as e:
            if DEBUG:
                print(f"[MCP DEBUG] Error in streaming endpoint: {e}")
            yield json.dumps({
                "success": False,
                "error": str(e),
                "original_query": request.query
            }) + "\n"

    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson"
    )


@app.post("/weather")
async def weather_endpoint(request: WeatherRequest):
    """Get current weather using OpenWeather API"""
    try:
        if not OPENWEATHER_API_KEY:
            raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")

        units = request.units or "metric"
        base = "https://api.openweathermap.org/data/2.5/weather"
        params = {"appid": OPENWEATHER_API_KEY, "units": units}

        if request.city:
            params["q"] = request.city
        elif request.lat is not None and request.lon is not None:
            params["lat"] = request.lat
            params["lon"] = request.lon
        else:
            raise HTTPException(status_code=400, detail="Provide either 'city' or ('lat' and 'lon')")

        response = requests.get(base, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        main = data.get("main", {})
        wind = data.get("wind", {})
        weather0 = (data.get("weather") or [{}])[0]
        sysinfo = data.get("sys", {})

        return {
            "source": "openweather",
            "units": units,
            "coord": data.get("coord", {}),
            "location": {
                "name": data.get("name"),
                "country": sysinfo.get("country"),
            },
            "current": {
                "temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": wind.get("speed"),
                "wind_deg": wind.get("deg"),
                "condition": weather0.get("main"),
                "description": weather0.get("description"),
            },
        }

    except requests.HTTPError as e:
        try:
            payload = response.json()
        except Exception:
            payload = {"message": str(e)}
        raise HTTPException(status_code=400, detail=f"OpenWeather error: {payload}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qdrant/cosmic_database_tool")
async def cosmic_database_endpoint(request: CosmicDatabaseRequest):
    """Search the cosmic database using vector similarity search"""
    try:
        if DEBUG:
            print(f"[COSMIC DB] Searching cosmic database for query: {request.query}")
        
        result = await cosmic_database_tool(request.query)
        return {
            "success": True,
            "query": request.query,
            "result": result
        }
    
    except Exception as e:
        if DEBUG:
            print(f"[COSMIC DB] Error: {e}")
        return {
            "success": False,
            "query": request.query,
            "error": str(e)
        }


# RAG endpoint is now handled by the rag_tools module through MCP tools


# # ++++++++++++++++++++++++++++++++
# # TICKET ENDPOINTS START
# # ++++++++++++++++++++++++++++++++


# @app.post("/ticket/create_jira")
# async def create_jira_endpoint(request: CreateJiraRequest):
#     """
#     Create a hospital support ticket for technical, equipment, software, or facility issues.

#     This endpoint collects and validates all required ticket fields, including:
#       - conversation_topic: A brief summary of the issue (ticket title)
#       - description: Detailed description of the problem
#       - location: Where the issue is occurring (e.g., room, department)
#       - queue: The hospital support queue to route the ticket (must be a valid queue)
#       - priority: Urgency level (e.g., Critical, High, Normal)
#       - department: Department responsible or affected
#       - name: Name of the person reporting the issue
#       - category: Type of issue (e.g., hardware, software, facility)

#     The endpoint ensures all information is complete and the queue is valid before creating the ticket. On success, it returns a confirmation and ticket reference for follow-up. If required fields are missing or invalid, it returns an error with details.

#     Returns:
#         success (bool): True if the ticket was created successfully, False otherwise.
#         result (dict): Ticket creation confirmation and reference details if successful.
#         error (str, optional): Error message if ticket creation failed.
#     """
#     try:
#         payload = request.model_dump()
#         thread_id = payload.get("thread_id")

#         # Keep asyncio.to_thread for synchronous create_jira_ticket function
#         result = await asyncio.to_thread(create_jira_ticket, **payload)

#         # Mark the ticket as completed after successful JIRA creation
#         # if thread_id:
#         #     await thread_ticket_manager.complete_ticket(thread_id)
#         #     # Or even better, add a new status like "jira_created"
#         #     # await thread_ticket_manager.update_field(thread_id, "status", "jira_created")

#         #     if DEBUG:
#         #         print(f"[CREATE_JIRA] Marked ticket as completed for thread: {thread_id}")

#         return {"success": True, "result": result}
#     except Exception as e:
#         return {"success": False, "error": str(e)}


# @app.post("/ticket/hospital_support_questions_tool")
# async def hospital_support_questions_endpoint(request: HospitalSupportQuestionsRequest):
#     """
#     Identify support protocols, diagnostic questions, and routing information for hospital technical issues.

#     This endpoint is used by a hospital technical support assistant to:
#     - Analyze user problem reports (equipment, software, facility issues)
#     - Return structured protocols, diagnostic questions, and recommended queue/department for ticket routing
#     - Enforce strict workflow: ALL protocol diagnostic questions must be asked and answered sequentially before ticket creation
#     - Ensure queue selection is always from the allowed QUEUE_CHOICES list
#     - Never mention tool names, agent names, or internal components to the user
#     - Return responses in the following format:
#         - success (bool): Whether the tool executed successfully
#         - message (str): Human-readable confirmation or error message
#         - response (dict): Protocols, diagnostic questions, and routing info

#     Usage rules:
#     - Call this endpoint ONCE per new issue report
#     - Pass the user's query exactly as written
#     - Use the returned protocol to guide all diagnostic questioning (one question at a time)
#     - Do NOT proceed to ticket creation until all protocol questions are answered
#     - Always validate queue selection against QUEUE_CHOICES before creating a ticket
#     - Never ask for information already provided in conversation history or stored fields
#     - Never offer to skip protocol questions
#     - Be friendly, conversational, and empathetic in all user interactions
#     """
#     try:
#         if DEBUG:
#             print(f"[HOSPITAL SUPPORT] Searching hospital support for query: {request.query}")
#         result = await hospital_support_questions_tool(request.query)
#         return {
#             "success": True,
#             "query": request.query,
#             "result": result
#         }
#     except Exception as e:
#         if DEBUG:
#             print(f"[HOSPITAL SUPPORT] Error: {e}")
#         return {
#             "success": False,
#             "query": request.query,
#             "error": str(e)
#         }


# # ++++++++++++++++++++++++++++++++
# # TICKET ENDPOINTS END
# # ++++++++++++++++++++++++++++++++



# +++++++++++++++++++++++++++
# TICKET AGENT ENDPOINTS START
# +++++++++++++++++++++++++++

# Thread-based ticket storage - each thread has its own ticket data
thread_tickets = {}


class ThreadTicketManager:
    def __init__(self):
        self._tickets = {}  # thread_id -> ticket_data
        self._lock = asyncio.Lock()

    async def get_or_create_ticket(self, thread_id: str) -> Dict[str, Any]:
        """Get existing ticket or create new one for thread"""
        async with self._lock:
            if thread_id not in self._tickets:
                self._tickets[thread_id] = {
                    "thread_id": thread_id,
                    "description": "",
                    "location": "",
                    "priority": "",
                    "category": "",
                    "queue": "",
                    "department": "",
                    "name": "",
                    "conversation_topic": "",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "status": "draft",
                    "is_active": True,
                    "field_entries": []
                }
                if DEBUG:
                    print(f"[THREAD_TICKET] Created new ticket for thread: {thread_id}")

            return self._tickets[thread_id]

    async def has_active_ticket(self, thread_id: str) -> tuple[bool, Dict[str, Any]]:
        """Check if thread has an active ticket"""
        async with self._lock:
            if thread_id in self._tickets:
                ticket = self._tickets[thread_id]
                is_active = ticket.get("is_active", False) and ticket.get("status") != "completed"
                return is_active, ticket
            return False, {}

    async def complete_ticket(self, thread_id: str) -> Dict[str, Any]:
        """Mark ticket as completed and ready for creation"""
        async with self._lock:
            if thread_id in self._tickets:
                self._tickets[thread_id]["status"] = "completed"
                self._tickets[thread_id]["completed_at"] = datetime.now().isoformat()
                self._tickets[thread_id]["is_active"] = False
                if DEBUG:
                    print(f"[THREAD_TICKET] Completed ticket for thread: {thread_id}")
                return self._tickets[thread_id]
            return {}

    async def update_field(self, thread_id: str, field_name: str, field_value: str) -> Dict[str, Any]:
        """Update a field for specific thread"""
        ticket = await self.get_or_create_ticket(thread_id)

        async with self._lock:
            old_value = ticket.get(field_name, "")
            ticket[field_name] = field_value
            ticket["last_updated"] = datetime.now().isoformat()

            # Add to field entries log
            ticket["field_entries"].append({
                "timestamp": datetime.now().isoformat(),
                "field": field_name,
                "old_value": old_value,
                "new_value": field_value,
                "action": "update"
            })

            if DEBUG:
                print(f"[THREAD_TICKET] Updated {field_name} for thread {thread_id}: '{old_value}' -> '{field_value}'")

            return ticket

    async def append_to_description(self, thread_id: str, text: str) -> Dict[str, Any]:
        """Append text to description for specific thread"""
        ticket = await self.get_or_create_ticket(thread_id)

        async with self._lock:
            old_description = ticket["description"]

            if ticket["description"]:
                ticket["description"] += f"\n\n{text}"
            else:
                ticket["description"] = text

            ticket["last_updated"] = datetime.now().isoformat()

            # Add to field entries log
            ticket["field_entries"].append({
                "timestamp": datetime.now().isoformat(),
                "field": "description",
                "old_value": old_description,
                "new_value": ticket["description"],
                "action": "append",
                "appended_text": text
            })

            if DEBUG:
                print(f"[THREAD_TICKET] Appended to description for thread {thread_id}")

            return ticket

    async def is_complete(self, thread_id: str) -> tuple[bool, list]:
        """Check if ticket is complete for specific thread"""
        ticket = await self.get_or_create_ticket(thread_id)

        required_fields = ["description", "category", "priority"]
        missing = []

        for field in required_fields:
            value = ticket.get(field, "").strip()
            if not value:
                missing.append(field)

        is_complete_status = len(missing) == 0

        if DEBUG:
            print(f"[THREAD_TICKET] Thread {thread_id} complete: {is_complete_status}, missing: {missing}")

        return is_complete_status, missing


# Global thread ticket manager
thread_ticket_manager = ThreadTicketManager()


@app.post("/ticket/known_problems")
async def known_problems_endpoint(request: KnownProblemsRequest):
    """
    Get known problems based on a query. Handle both new tickets and existing ticket updates.
    If ticket_id is provided, this is an existing ticket - return continuation message.
    If ticket_id is None, this is a new issue - process with Qdrant and LLM.
    """
    try:
        # Get thread_id from request
        thread_id = getattr(request, 'thread_id', None) or request.dict().get('thread_id', 'default')

        if DEBUG:
            print(f"[KNOWN_PROBLEMS] Processing query for thread: {thread_id}")

        # This is a new ticket - proceed with full Qdrant + LLM flow
        if DEBUG:
            print(f"[API DEBUG] New ticket - Querying Qdrant for: {request.query}")

        # Step 1: Query Qdrant for hospital support protocols
        qdrant_result = await hospital_support_questions_tool(request.query)

        if DEBUG:
            print(f"[API DEBUG] Qdrant result: {qdrant_result}")

        # Step 2: Select LLM provider based on configuration
        if config.MCP_PROVIDER_OLLAMA:
            provider = "ollama"
            llm = ChatOllama(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                base_url=config.OLLAMA_BASE_URL
            )
        elif config.MCP_PROVIDER_OPENAI:
            provider = "openai"
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
            llm = ChatOpenAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                api_key=api_key
            )
        elif config.MCP_PROVIDER_MISTRAL:
            provider = "mistral"
            api_key = os.environ.get("MISTRAL_API_KEY")
            llm = ChatMistralAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0,
                mistral_api_key=api_key,
                endpoint=config.MISTRAL_BASE_URL
            )
        else:
            raise ValueError("Unsupported provider")

        if DEBUG:
            print(f"[API DEBUG] Using LLM provider: {provider}")

        QUEUE_CHOICES = [
            'Technical Support',  # General technical issues, equipment problems
            'Servicedesk',  # First-line support, basic user assistance
            '2nd line',  # Escalated technical issues requiring specialist knowledge
            'Cambio JIRA',  # Cambio system-specific issues
            'Cosmic',  # Cosmic system-related problems
            'Billing Payments',  # Payment processing, billing inquiries
            'Account Management',  # User account issues, access management
            'Product Inquiries',  # Questions about products/services
            'Feature Requests',  # New feature suggestions
            'Bug Reports',  # Software bugs and defects
            'Security Department',  # Security incidents, access violations
            'Compliance Legal',  # Regulatory compliance, legal matters
            'Service Outages',  # System downtime, service interruptions
            'Onboarding Setup',  # New user setup, initial configuration
            'API Integration',  # API-related issues and integrations
            'Data Migration',  # Data transfer and migration issues
            'Accessibility',  # Accessibility compliance and support
            'Training Education',  # Training requests, educational support
            'General Inquiries',  # Non-specific questions
            'Permissions Access',  # Permission changes, access requests
            'Management Department',  # Management-level issues
            'Maintenance Department',  # Facility maintenance, equipment servicing
            'Logistics Department',  # Supply chain, logistics issues
            'IT Department',  # IT infrastructure, network issues
        ]

        # Step 3: Prepare LLM prompt for analyzing Qdrant response
        system_prompt = """
        You are an expert hospital support system analyzer. Your job is to process user queries and provide complete ticket information.

        INPUT:
        - User Query: {user_query}
        - Available Protocols: {qdrant_response}
        - Available Queues: {queue_choices}

        TASK:
        You will receive multiple hospital support protocols from the database. You must ALWAYS return a successful response by either:
        1. SELECTING the most relevant protocol from the available options, OR
        2. GENERATING your own response if none of the protocols are relevant enough

        EVALUATION CRITERIA:
        - Compare user query keywords with protocol keywords
        - Match query intent with protocol issue_category and description
        - Consider match_score from database (higher is better)
        - Look for alignment between query type and protocol clinical_domain

        RESPONSE RULES:
        1. ALWAYS return success: true
        2. If you SELECT a protocol, use its exact values for all fields
        3. If you GENERATE, create appropriate values based on hospital operations knowledge
        4. All questions must be specific, actionable, and relevant to hospital support
        5. Queue must be one of the provided available queues: {queue_choices}
        6. Priority levels: "low", "medium", "high", "critical"

        OUTPUT FORMAT (JSON ONLY - NO MARKDOWN, NO EXPLANATIONS):

        For SELECTED protocol:
        {{
          "success": true,
          "source": "protocol",
          "questions": [protocol's questions_to_ask array],
          "issue_category": "protocol's issue_category",
          "queue": "protocol's queue", 
          "priority": "protocol's urgency_level",
          "ticket_id": "will be generated separately"
        }}

        For GENERATED response:
        {{
          "success": true,
          "source": "generated",
          "questions": ["question1", "question2", "question3"],
          "issue_category": "appropriate category for the issue",
          "queue": "appropriate queue from available options",
          "priority": "low|medium|high|critical",
          "ticket_id": "will be generated separately"
        }}

        CRITICAL: Respond with valid JSON only. No additional text, no markdown blocks, no explanations.
        """

        messages = [
            SystemMessage(content=system_prompt.format(
                user_query=request.query,
                qdrant_response=str(qdrant_result),
                queue_choices=str(QUEUE_CHOICES)
            ))
        ]

        # Step 4: Call LLM to analyze Qdrant response
        if DEBUG:
            print(f"[API DEBUG] Calling LLM to analyze Qdrant response for new ticket")

        llm_response = llm.invoke(messages)

        # Step 5: Process LLM response
        try:
            response_content = llm_response.content.strip()

            if DEBUG:
                print(f"[API DEBUG] Raw LLM response: {response_content}")

            # Handle potential code block formatting
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()

            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "").strip()

            # Try to parse as JSON
            try:
                llm_output = json.loads(response_content)

                if DEBUG:
                    print(f"[API DEBUG] Parsed JSON successfully: {llm_output}")

                # Generate ticket ID for new tickets (both success and clarification cases)
                if llm_output.get("success") or llm_output.get("questions"):
                    ticket_id = str(uuid.uuid4())

                    llm_output["ticket_id"] = ticket_id
                    llm_output["is_new_ticket"] = True

                    # Create ticket in thread manager
                    await thread_ticket_manager.get_or_create_ticket(thread_id)

                    if DEBUG:
                        print(f"[API DEBUG] Generated new ticket ID: {ticket_id}")

                return llm_output

            except json.JSONDecodeError as json_err:
                if DEBUG:
                    print(f"[API DEBUG] JSON parse failed: {json_err}")

                # Try to extract JSON pattern from response
                import re
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, response_content, re.DOTALL)

                if match:
                    try:
                        llm_output = json.loads(match.group(0))

                        # Generate ticket ID for extracted JSON too
                        if llm_output.get("success") or llm_output.get("questions"):
                            ticket_id = str(uuid.uuid4())
                            llm_output["ticket_id"] = ticket_id
                            llm_output["is_new_ticket"] = True

                            # Create ticket in thread manager
                            await thread_ticket_manager.get_or_create_ticket(thread_id)

                        if DEBUG:
                            print(f"[API DEBUG] Extracted JSON pattern successfully: {llm_output}")
                        return llm_output
                    except json.JSONDecodeError:
                        pass

                # If all JSON parsing fails, return error
                return {
                    "success": False,
                    "error": f"Could not parse LLM response as JSON: {json_err}",
                    "raw_response": response_content,
                    "trigger_fallback": True
                }

        except Exception as e:
            if DEBUG:
                print(f"[API DEBUG] General error processing LLM response: {e}")
            return {
                "success": False,
                "error": f"Error processing LLM response: {str(e)}",
                "trigger_fallback": True
            }

    except Exception as e:
        if DEBUG:
            print(f"[API DEBUG] Known problems error: {e}")

        return {
            "success": False,
            "error": str(e),
            "trigger_fallback": True
        }


@app.post("/ticket/check_active")
async def check_active_ticket_endpoint(request: ThreadTicketRequest):
    """
    Check if a thread has an active ticket.

    Returns information about active tickets to prevent multiple concurrent tickets
    and provide context about ongoing ticket creation process.
    """
    try:
        thread_id = request.thread_id

        if DEBUG:
            print(f"[CHECK_ACTIVE] Checking active ticket for thread: {thread_id}")

        has_active, ticket_data = await thread_ticket_manager.has_active_ticket(thread_id)

        if has_active:
            # Check if ticket is ready for completion
            is_complete, missing_fields = await thread_ticket_manager.is_complete(thread_id)

            response = {
                "success": True,
                "has_active_ticket": True,
                "ticket_status": ticket_data.get("status", "draft"),
                "is_complete": is_complete,
                "missing_fields": missing_fields,
                "created_at": ticket_data.get("created_at"),
                "last_updated": ticket_data.get("last_updated"),
                "field_count": len(ticket_data.get("field_entries", [])),
                "message": "Active ticket found. Continue with this ticket or complete it to create a new one."
            }

            if is_complete:
                response["message"] = "Ticket is complete and ready for JIRA creation."
                response["ready_for_creation"] = True
            else:
                response["message"] = f"Active ticket in progress. Missing fields: {', '.join(missing_fields)}"
                response["ready_for_creation"] = False

            if DEBUG:
                print(
                    f"[CHECK_ACTIVE] Active ticket found - Status: {response['ticket_status']}, Complete: {is_complete}")

            return response
        else:
            if DEBUG:
                print(f"[CHECK_ACTIVE] No active ticket found for thread: {thread_id}")

            return {
                "success": True,
                "has_active_ticket": False,
                "message": "No active ticket found. Ready to create new ticket.",
                "ready_for_new_ticket": True
            }

    except Exception as e:
        if DEBUG:
            print(f"[CHECK_ACTIVE] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/ticket/complete")
async def complete_ticket_endpoint(request: ThreadTicketRequest):
    """Mark a ticket as completed and ready for JIRA creation"""
    try:
        thread_id = request.thread_id

        if DEBUG:
            print(f"[COMPLETE_TICKET] Completing ticket for thread: {thread_id}")

        # Check if ticket exists and is complete
        is_complete, missing_fields = await thread_ticket_manager.is_complete(thread_id)

        if not is_complete:
            return {
                "success": False,
                "error": f"Cannot complete ticket. Missing required fields: {', '.join(missing_fields)}",
                "missing_fields": missing_fields
            }

        # Mark as completed
        completed_ticket = await thread_ticket_manager.complete_ticket(thread_id)

        if DEBUG:
            print(f"[COMPLETE_TICKET] Ticket completed successfully for thread: {thread_id}")

        return {
            "success": True,
            "message": "Ticket marked as completed and ready for JIRA creation",
            "completed_at": completed_ticket.get("completed_at"),
            "ready_for_jira": True
        }

    except Exception as e:
        if DEBUG:
            print(f"[COMPLETE_TICKET] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ======================================
# TICKET FIELD MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ticket/{thread_id}")
async def get_ticket_endpoint(thread_id: str):
    """Get current ticket state for a thread"""
    try:
        if DEBUG:
            print(f"[TICKET] Getting ticket for thread: {thread_id}")

        ticket = await thread_ticket_manager.get_or_create_ticket(thread_id)
        is_complete, missing = await thread_ticket_manager.is_complete(thread_id)

        return {
            "success": True,
            "action": "get_ticket",
            "thread_id": thread_id,
            "ticket_data": ticket,
            "is_complete": is_complete,
            "missing_fields": missing,
            "total_entries": len(ticket.get("field_entries", [])),
            "message": f"Retrieved ticket data for thread {thread_id}"
        }

    except Exception as e:
        if DEBUG:
            print(f"[TICKET] Error: {e}")
        return {"success": False, "action": "get_ticket", "error": str(e)}


@app.patch("/ticket/{thread_id}/fields")
async def update_ticket_fields_endpoint(thread_id: str, request: dict = Body(...)):
    """Update one or more fields in a ticket"""
    try:
        fields_data = request.get('fields', {})

        if DEBUG:
            print(f"[TICKET] Updating fields for thread {thread_id}: {list(fields_data.keys())}")

        if not fields_data:
            return {"success": False, "error": "fields dictionary required"}

        # Validate field names
        ALLOWED_FIELDS = {
            "description", "conversation_topic", "category", "queue",
            "priority", "department", "name", "location"
        }

        invalid_fields = [f for f in fields_data.keys() if f not in ALLOWED_FIELDS]
        if invalid_fields:
            return {
                "success": False,
                "error": f"Invalid fields: {invalid_fields}. Allowed: {list(ALLOWED_FIELDS)}"
            }

        # Update each field
        updated_fields = {}
        for field_name, field_value in fields_data.items():
            if field_value and str(field_value).strip():
                await thread_ticket_manager.update_field(thread_id, field_name, str(field_value))
                updated_fields[field_name] = field_value

        # Get current state after updates
        ticket = await thread_ticket_manager.get_or_create_ticket(thread_id)
        is_complete, missing = await thread_ticket_manager.is_complete(thread_id)

        return {
            "success": True,
            "action": "update_fields",
            "thread_id": thread_id,
            "updated_fields": updated_fields,
            "ticket_data": ticket,
            "is_complete": is_complete,
            "missing_fields": missing,
            "message": f"Updated {len(updated_fields)} fields for thread {thread_id}"
        }

    except Exception as e:
        if DEBUG:
            print(f"[TICKET] Error: {e}")
        return {"success": False, "action": "update_fields", "error": str(e)}


@app.post("/ticket/{thread_id}/description/append")
async def append_ticket_description_endpoint(thread_id: str, request: dict = Body(...)):
    """Append text to ticket description"""
    try:
        text = request.get('text', request.get('content', ''))

        if DEBUG:
            print(f"[TICKET] Appending to description for thread {thread_id}")

        if not text:
            return {"success": False, "error": "text content required"}

        # Append to description
        await thread_ticket_manager.append_to_description(thread_id, text)

        # Get current state after append
        ticket = await thread_ticket_manager.get_or_create_ticket(thread_id)
        is_complete, missing = await thread_ticket_manager.is_complete(thread_id)

        return {
            "success": True,
            "action": "append_description",
            "thread_id": thread_id,
            "appended_text": text,
            "ticket_data": ticket,
            "is_complete": is_complete,
            "missing_fields": missing,
            "message": f"Appended text to description for thread {thread_id}"
        }

    except Exception as e:
        if DEBUG:
            print(f"[TICKET] Error: {e}")
        return {"success": False, "action": "append_description", "error": str(e)}


@app.get("/ticket/{thread_id}/status")
async def check_ticket_completeness_endpoint(thread_id: str):
    """Check if ticket is complete and ready for creation"""
    try:
        if DEBUG:
            print(f"[TICKET] Checking completeness for thread {thread_id}")

        ticket = await thread_ticket_manager.get_or_create_ticket(thread_id)
        is_complete, missing = await thread_ticket_manager.is_complete(thread_id)

        return {
            "success": True,
            "action": "check_completeness",
            "thread_id": thread_id,
            "is_complete": is_complete,
            "missing_fields": missing,
            "total_entries": len(ticket.get("field_entries", [])),
            "status": ticket.get("status", "draft"),
            "message": "Complete" if is_complete else f"Missing: {', '.join(missing)}"
        }

    except Exception as e:
        if DEBUG:
            print(f"[TICKET] Error: {e}")
        return {"success": False, "action": "check_completeness", "error": str(e)}


@app.get("/ticket/{thread_id}/history")
async def get_ticket_history_endpoint(thread_id: str):
    """Get the field update history for a ticket"""
    try:
        if DEBUG:
            print(f"[TICKET] Getting history for thread {thread_id}")

        ticket = await thread_ticket_manager.get_or_create_ticket(thread_id)
        field_entries = ticket.get("field_entries", [])

        return {
            "success": True,
            "action": "get_history",
            "thread_id": thread_id,
            "field_entries": field_entries,
            "total_entries": len(field_entries),
            "created_at": ticket.get("created_at"),
            "last_updated": ticket.get("last_updated"),
            "message": f"Retrieved {len(field_entries)} field entries for thread {thread_id}"
        }

    except Exception as e:
        if DEBUG:
            print(f"[TICKET] Error: {e}")
        return {"success": False, "action": "get_history", "error": str(e)}


@app.post("/ticket/create_jira")
async def create_jira_endpoint(request: CreateJiraRequest):
    """Create a JIRA ticket"""
    try:
        payload = request.model_dump()
        thread_id = payload.get("thread_id")

        # Keep asyncio.to_thread for synchronous create_jira_ticket function
        result = await asyncio.to_thread(create_jira_ticket, **payload)

        # Mark the ticket as completed after successful JIRA creation
        if thread_id:
            await thread_ticket_manager.complete_ticket(thread_id)
            # Or even better, add a new status like "jira_created"
            # await thread_ticket_manager.update_field(thread_id, "status", "jira_created")

            if DEBUG:
                print(f"[CREATE_JIRA] Marked ticket as completed for thread: {thread_id}")

        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


# +++++++++++++++++++++++++++
# TICKET AGENT ENDPOINTS END
# +++++++++++++++++++++++++++



# ++++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY ENDPOINTS START
# ++++++++++++++++++++++++++++++++





# ------------------------------
# AZURE AD: Single operations endpoint
# ------------------------------

def _require_fields(action: str, body: dict, fields: list[str]):
    missing = [f for f in fields if not str(body.get(f, "")).strip()]
    if missing:
        return {
            "success": False,
            "error": f"Missing required field(s) for '{action}': {', '.join(missing)}",
            "missing_fields": missing
        }
    return None



# ======================================
# USER MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ad/users")
async def list_users_endpoint():
    """List all users in the directory"""
    try:
        if DEBUG:
            print("[AD_USERS] Listing all users")
        
        data = ad.list_users()
        return {"success": True, "action": "list_users", "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "list_users", "error": str(e)}


@app.post("/ad/users")
async def create_user_endpoint(request: CreateUserRequest):
    """Create a new user in the directory"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Creating user")
        
        user_payload = request.user
        # Auto-generate userPrincipalName from displayName (strip spaces and lowercase)
        if "displayName" in user_payload:
            clean_name = user_payload["displayName"].replace(" ", "").lower()
            user_payload["userPrincipalName"] = f"{clean_name}@lovenoreusgmail.onmicrosoft.com"
            # Also set mailNickname if not provided
            if "mailNickname" not in user_payload:
                user_payload["mailNickname"] = clean_name
        
        data = ad.create_user(user_payload)
        return {"success": True, "action": "create_user", "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_USERS] Validation Error: {ve}")
        return {"success": False, "action": "create_user", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "create_user", "error": str(e)}


@app.patch("/ad/users/{user_id}")
async def update_user_endpoint(user_id: str, request: UserUpdates):
    """Update an existing user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Updating user: {user_id}")
        
        data = ad.update_user(user_id, request.updates)
        return {"success": True, "action": "update_user", "user_id": user_id, "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_USERS] Validation Error: {ve}")
        return {"success": False, "action": "update_user", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "update_user", "error": str(e)}


@app.delete("/ad/users/{user_id}")
async def delete_user_endpoint(user_id: str):
    """Delete a user from the directory"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Deleting user: {user_id}")
        
        data = ad.delete_user(user_id)
        return {"success": True, "action": "delete_user", "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "delete_user", "error": str(e)}


@app.get("/ad/users/{user_id}/roles")
async def get_user_roles_endpoint(user_id: str):
    """Get roles assigned to a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting roles for user: {user_id}")
        
        data = ad.get_user_roles(user_id)
        return {"success": True, "action": "get_user_roles", "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_roles", "error": str(e)}


@app.get("/ad/users/{user_id}/groups")
async def get_user_groups_endpoint(user_id: str, transitive: bool = False):
    """Get groups for a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting groups for user: {user_id}, transitive: {transitive}")
        
        data = ad.get_user_groups(user_id, transitive=transitive)
        return {"success": True, "action": "get_user_groups", "user_id": user_id, "transitive": transitive, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_groups", "error": str(e)}


@app.get("/ad/users/{user_id}/owned-groups")
async def get_user_owned_groups_endpoint(user_id: str):
    """Get groups owned by a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting owned groups for user: {user_id}")
        
        data = ad.get_user_owned_groups(user_id)
        return {"success": True, "action": "get_user_owned_groups", "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_owned_groups", "error": str(e)}


@app.get("/ad/users-with-groups")
async def list_users_with_groups_endpoint(
    include_transitive: bool = False,
    include_owned: bool = True,
    select: str = "id,displayName,userPrincipalName"
):
    """List users with their group information"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Listing users with groups - transitive: {include_transitive}, owned: {include_owned}")
        
        data = ad.list_users_with_groups(
            include_transitive=include_transitive,
            include_owned=include_owned,
            select=select
        )
        return {"success": True, "action": "list_users_with_groups", "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "list_users_with_groups", "error": str(e)}


# ======================================
# ROLE MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ad/roles")
async def list_roles_endpoint():
    """List all directory roles"""
    try:
        if DEBUG:
            print("[AD_ROLES] Listing all roles")
        
        data = ad.list_roles()
        return {"success": True, "action": "list_roles", "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "list_roles", "error": str(e)}


@app.post("/ad/roles/{role_id}/members")
async def add_user_to_role_endpoint(role_id: str, request: RoleAddMember):
    """Add a user to a role"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Adding user {request.user_id} to role {role_id}")
        
        data = ad.add_user_to_role(request.user_id, role_id)
        return {"success": True, "action": "add_to_role", "role_id": role_id, "user_id": request.user_id, "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_ROLES] Validation Error: {ve}")
        return {"success": False, "action": "add_to_role", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "add_to_role", "error": str(e)}


@app.delete("/ad/roles/{role_id}/members/{user_id}")
async def remove_user_from_role_endpoint(role_id: str, user_id: str):
    """Remove a user from a role"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Removing user {user_id} from role {role_id}")
        
        data = ad.remove_user_from_role(user_id, role_id)
        return {"success": True, "action": "remove_from_role", "role_id": role_id, "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "remove_from_role", "error": str(e)}


@app.post("/ad/roles/instantiate")
async def instantiate_role_endpoint(request: RoleInstantiation):
    """Instantiate a directory role from template"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Instantiating role from template: {request.roleTemplateId}")
        
        data = ad.instantiate_directory_role(request.roleTemplateId)
        return {"success": True, "action": "instantiate_role", "roleTemplateId": request.roleTemplateId, "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_ROLES] Validation Error: {ve}")
        return {"success": False, "action": "instantiate_role", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "instantiate_role", "error": str(e)}


# ======================================
# GROUP MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ad/groups")
async def list_groups_endpoint(
    security_only: bool = False,
    unified_only: bool = False,
    select: str = "id,displayName,mailNickname,mail,securityEnabled,groupTypes"
):
    """List all groups in the directory"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Listing groups - security_only: {security_only}, unified_only: {unified_only}")
        
        data = ad.list_groups(
            security_only=security_only,
            unified_only=unified_only,
            select=select
        )
        return {"success": True, "action": "list_groups", "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "list_groups", "error": str(e)}


@app.post("/ad/groups")
async def create_group_endpoint(request: CreateGroupRequest):
    """Create a new group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Creating group: {request.display_name}")
        
        params = {
            "display_name": request.display_name,
            "mail_nickname": request.mail_nickname,
            "description": request.description,
            "group_type": request.group_type,
            "visibility": request.visibility,
            "membership_rule": request.membership_rule,
            "owners": request.owners,
            "members": request.members
        }
        data = ad.create_group(**{k: v for k, v in params.items() if v is not None})
        return {"success": True, "action": "create_group", "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "create_group", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "create_group", "error": str(e)}


@app.get("/ad/groups/{group_id}/members")
async def get_group_members_endpoint(group_id: str):
    """Get members of a specific group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Getting members for group: {group_id}")
        
        data = ad.get_group_members(group_id)
        return {"success": True, "action": "get_group_members", "group_id": group_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "get_group_members", "error": str(e)}


@app.get("/ad/groups/{group_id}/owners")
async def get_group_owners_endpoint(group_id: str):
    """Get owners of a specific group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Getting owners for group: {group_id}")
        
        data = ad.get_group_owners(group_id)
        return {"success": True, "action": "get_group_owners", "group_id": group_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "get_group_owners", "error": str(e)}


@app.post("/ad/groups/{group_id}/members")
async def add_group_member_endpoint(group_id: str, request: GroupMemberRequest):
    """Add a user to a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Adding user {request.user_id} to group {group_id}")
        
        token = ad.get_access_token()
        body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{request.user_id}"}
        data = ad.graph_api_request("POST", f"groups/{group_id}/members/$ref", token, data=body)
        group = ad.get_user_groups(request.user_id)
        return {"success": True, "action": "add_group_member", "group_id": group_id, "user_id": request.user_id, "data": data, "group": group.get('data', [])}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "add_group_member", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "add_group_member", "error": str(e)}


@app.delete("/ad/groups/{group_id}/members/{user_id}")
async def remove_group_member_endpoint(group_id: str, user_id: str):
    """Remove a user from a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Removing user {user_id} from group {group_id}")
        
        token = ad.get_access_token()
        endpoint = f"groups/{group_id}/members/{user_id}/$ref"
        data = ad.graph_api_request("DELETE", endpoint, token)
        return {"success": True, "action": "remove_group_member", "group_id": group_id, "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "remove_group_member", "error": str(e)}


@app.post("/ad/groups/{group_id}/owners")
async def add_group_owner_endpoint(group_id: str, request: GroupOwnerRequest):
    """Add an owner to a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Adding owner {request.user_id} to group {group_id}")
        
        token = ad.get_access_token()
        body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{request.user_id}"}
        data = ad.graph_api_request("POST", f"groups/{group_id}/owners/$ref", token, data=body)
        return {"success": True, "action": "add_group_owner", "group_id": group_id, "user_id": request.user_id, "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "add_group_owner", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "add_group_owner", "error": str(e)}


@app.delete("/ad/groups/{group_id}/owners/{user_id}")
async def remove_group_owner_endpoint(group_id: str, user_id: str):
    """Remove an owner from a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Removing owner {user_id} from group {group_id}")
        
        token = ad.get_access_token()
        endpoint = f"groups/{group_id}/owners/{user_id}/$ref"
        data = ad.graph_api_request("DELETE", endpoint, token)
        return {"success": True, "action": "remove_group_owner", "group_id": group_id, "user_id": user_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "remove_group_owner", "error": str(e)}


@app.patch("/ad/groups/{group_id}")
async def update_group_endpoint(group_id: str, request: GroupUpdates):
    """Update an existing group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Updating group: {group_id}")
        
        token = ad.get_access_token()
        data = ad.graph_api_request("PATCH", f"groups/{group_id}", token, data=request.updates)
        return {"success": True, "action": "update_group", "group_id": group_id, "data": data}
    
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "update_group", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "update_group", "error": str(e)}


@app.delete("/ad/groups/{group_id}")
async def delete_group_endpoint(group_id: str):
    """Delete a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Deleting group: {group_id}")
        
        token = ad.get_access_token()
        data = ad.graph_api_request("DELETE", f"groups/{group_id}", token)
        return {"success": True, "action": "delete_group", "group_id": group_id, "data": data}
    
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "delete_group", "error": str(e)}




# ++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY ENDPOINTS END
# ++++++++++++++++++++++++++++++


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connectivity
        db_health = "unknown"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{DATABASE_SERVER_URL}/health", timeout=5) as response:
                    if response.status == 200:
                        db_health = "connected"
                    else:
                        db_health = "error"
        except:
            db_health = "disconnected"

        return {
            "status": "healthy",
            "service": "MCP Server",
            "timestamp": datetime.now().isoformat(),
            "database_connection": db_health,
            "weather_api_configured": bool(OPENWEATHER_API_KEY),
            "llm_providers": ["openai", "ollama", "mistral"],
            "endpoints": {
                "greet": "/greet",
                "query_database": "/query_database",
                "query_database_stream": "/query_database_stream",
                "weather": "/weather",
                "health": "/health"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list():
    """MCP Protocol: List available tools"""
    tools = [
        MCPTool(
            name="greet",
            description="Provide a friendly greeting to the user with appropriate time-based salutation (Good morning/afternoon/evening). Use when user says hello, hi, or similar greetings, or to start conversations politely.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "User's name if provided. If None, gives general greeting."
                    }
                },
                "required": []
            }
        ),
        MCPTool(
            name="query_database",
            description="Execute natural language queries against the database using AI-powered SQL generation. Keywords are automatically generated using OpenAI model for optimal database searching. Use for questions about users, posts, database data, statistics, counts, summaries. Database contains hospital-related tables with current operational data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about the database content"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to help with query context (automatically generated if not provided)"
                    }
                },
                "required": ["query", "keywords"]
            }
        ),
        MCPTool(
            name="get_current_weather",
            description="Get current weather information for a specific location using OpenWeatherMap API. Use when user asks about current weather conditions, temperature, atmospheric conditions, or mentions weather. Defaults to Karachi if no location provided.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York', 'Tokyo', 'Karachi')"
                    },
                    "lat": {"type": "number", "description": "Latitude coordinate (use with lon)"},
                    "lon": {"type": "number", "description": "Longitude coordinate (use with lat)"},
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "kelvin"],
                        "description": "Temperature units - metric (Celsius), imperial (Fahrenheit), kelvin",
                        "default": "metric"
                    }
                },
                "anyOf": [
                    {"required": ["city"]},
                    {"required": ["lat", "lon"]}
                ]
            }
        ),
        MCPTool(
            name="ad_list_users",
            description="List all users in Azure Active Directory. Returns user information including ID, display name, email, and other user properties.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        MCPTool(
            name="ad_create_user",
            description="Create a new user in Azure Active Directory. Auto-generates userPrincipalName and mailNickname from displayName if not provided.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "description": "Microsoft Graph user payload with user properties",
                        "properties": {
                            "displayName": {"type": "string", "description": "User's display name"},
                            "mailNickname": {"type": "string", "description": "Mail nickname (auto-generated if not provided)"},
                            "userPrincipalName": {"type": "string", "description": "User principal name (auto-generated if not provided)"},
                            "passwordProfile": {
                                "type": "object",
                                "properties": {
                                    "password": {"type": "string", "description": "Temporary password"},
                                    "forceChangePasswordNextSignIn": {"type": "boolean", "description": "Force password change on next sign-in"}
                                }
                            },
                            "accountEnabled": {"type": "boolean", "description": "Whether account is enabled"}
                        },
                        "required": ["displayName"]
                    }
                },
                "required": ["user"]
            }
        ),
        MCPTool(
            name="ad_update_user",
            description="Update an existing user's properties in Azure Active Directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID to update"},
                    "updates": {"type": "object", "description": "Properties to update"}
                },
                "required": ["user_id", "updates"]
            }
        ),
        MCPTool(
            name="ad_delete_user",
            description="Delete a user from Azure Active Directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID to delete"}
                },
                "required": ["user_id"]
            }
        ),
        MCPTool(
            name="ad_get_user_roles",
            description="Get all directory roles assigned to a specific user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID to get roles for"}
                },
                "required": ["user_id"]
            }
        ),
        MCPTool(
            name="ad_get_user_groups",
            description="Get groups that a user is a member of, with optional transitive membership.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID to get groups for"},
                    "transitive": {"type": "boolean", "description": "Include transitive group memberships", "default": False}
                },
                "required": ["user_id"]
            }
        ),
        MCPTool(
            name="ad_list_roles",
            description="List all directory roles in Azure Active Directory.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        MCPTool(
            name="ad_add_user_to_role",
            description="Add a user to a directory role.",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_id": {"type": "string", "description": "Role ID to add user to"},
                    "user_id": {"type": "string", "description": "User ID to add to role"}
                },
                "required": ["role_id", "user_id"]
            }
        ),
        MCPTool(
            name="ad_remove_user_from_role",
            description="Remove a user from a directory role.",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_id": {"type": "string", "description": "Role ID to remove user from"},
                    "user_id": {"type": "string", "description": "User ID to remove from role"}
                },
                "required": ["role_id", "user_id"]
            }
        ),
        MCPTool(
            name="ad_list_groups",
            description="List all groups in Azure Active Directory with filtering options.",
            inputSchema={
                "type": "object",
                "properties": {
                    "security_only": {"type": "boolean", "description": "List only security groups", "default": False},
                    "unified_only": {"type": "boolean", "description": "List only unified groups", "default": False},
                    "select": {"type": "string", "description": "Fields to select", "default": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
                },
                "required": []
            }
        ),
        MCPTool(
            name="ad_create_group",
            description="Create a new group in Azure Active Directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "display_name": {"type": "string", "description": "Group display name"},
                    "mail_nickname": {"type": "string", "description": "Group mail nickname"},
                    "description": {"type": "string", "description": "Group description"},
                    "group_type": {"type": "string", "enum": ["security", "unified"], "description": "Group type", "default": "security"},
                    "visibility": {"type": "string", "description": "Group visibility"},
                    "owners": {"type": "array", "items": {"type": "string"}, "description": "List of owner user IDs"},
                    "members": {"type": "array", "items": {"type": "string"}, "description": "List of member user IDs"}
                },
                "required": ["display_name", "mail_nickname"]
            }
        ),
        MCPTool(
            name="ad_add_group_member",
            description="Add a user to a group as a member.",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Group ID to add member to"},
                    "user_id": {"type": "string", "description": "User ID to add to group"}
                },
                "required": ["group_id", "user_id"]
            }
        ),
        MCPTool(
            name="ad_remove_group_member",
            description="Remove a user from a group.",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Group ID to remove member from"},
                    "user_id": {"type": "string", "description": "User ID to remove from group"}
                },
                "required": ["group_id", "user_id"]
            }
        ),
        MCPTool(
            name="ad_get_group_members",
            description="Get all members of a specific group.",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Group ID to get members for"}
                },
                "required": ["group_id"]
            }
        ),
        MCPTool(
            name="search_cosmic_database",
            description="Search the cosmic database using vector similarity search. Use this tool when you need to find information from medical documentation, procedures, policies, or any content stored in the cosmic database. This tool uses advanced multi-strategy retrieval with metadata filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information in the cosmic database"
                    }
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="hospital_support_questions_tool",
            description="You are a friendly technical support assistant for a hospital environment. This tool analyzes user problem reports (equipment, software, facility issues), returns structured support protocols, diagnostic questions, and recommended queue/department for ticket routing. It enforces strict workflow: ALL protocol diagnostic questions must be asked and answered sequentially before ticket creation. Queue selection is always from the allowed QUEUE_CHOICES list. Never mention tool names, agent names, or internal components to the user. Always check conversation history and stored fields before asking questions. Use the returned protocol to guide all diagnostic questioning (one question at a time). Do NOT proceed to ticket creation until all protocol questions are answered. Always validate queue selection against QUEUE_CHOICES before creating a ticket. Be friendly, conversational, and empathetic in all user interactions. Returns: success (bool), message (str), response (dict with protocols, diagnostic questions, and routing info).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The support question to search for in the hospital support knowledge base"
                    }
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="create_jira_ticket",
            description="Create a JIRA ticket for a support request, incident, or task. Use this tool to log issues, feature requests, or support needs in the JIRA system. Requires thread_id, conversation_topic, description, location, queue, priority, department, name, and category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "Thread or conversation ID for the ticket context"},
                    "conversation_topic": {"type": "string", "description": "Topic or subject of the conversation"},
                    "description": {"type": "string", "description": "Detailed description of the issue or request"},
                    "location": {"type": "string", "description": "Location related to the ticket (e.g., department, room)"},
                    "queue": {"type": "string", "description": "JIRA queue or project name"},
                    "priority": {"type": "string", "description": "Priority of the ticket (e.g., High, Medium, Low)"},
                    "department": {"type": "string", "description": "Department related to the ticket"},
                    "name": {"type": "string", "description": "Name of the requester or subject"},
                    "category": {"type": "string", "description": "Category of the ticket (e.g., Incident, Request, Task)"}
                },
                "required": ["thread_id", "conversation_topic", "description", "location", "queue", "priority", "department", "name", "category"]
            }
        ),
        MCPTool(**get_rag_tool_schema()),
    ]

    return MCPToolsListResponse(tools=tools)


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def mcp_tools_call(request: MCPToolCallRequest):
    """MCP Protocol: Call a specific tool"""
    try:
        tool_name = request.name
        arguments = request.arguments

        if DEBUG:
            print(f"[MCP] Calling tool: {tool_name} with args: {arguments}")

        # Add thread_id to arguments if not present
        if "thread_id" not in arguments:
            arguments["thread_id"] = "default"

        if tool_name == "greet":
            greet_request = GreetRequest(name=arguments.get("name"))
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "query_database":
            query_request = QueryDatabaseRequest(**arguments)
            result = await query_database_endpoint(query_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )
           
            
        elif tool_name == "search_cosmic_database":
            query = arguments.get("query", "")
            if not query:
                return MCPToolCallResponse(
                    content=[MCPContent(type="text", text="Error: Query parameter is required")],
                    isError=True
                )
            
            result = await cosmic_database_tool(query)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=result)]
            )

        
        elif tool_name == "get_current_weather":
            weather_request = WeatherRequest(
                **{k: v for k, v in arguments.items() if k in ["city", "lat", "lon", "units"]})
            result = await weather_endpoint(weather_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )
        
        elif tool_name == "ad_list_users":
            result = await list_users_endpoint()
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_create_user":
            create_request = CreateUserRequest(action="create_user", user=arguments["user"])
            result = await create_user_endpoint(create_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_update_user":
            user_updates = UserUpdates(updates=arguments["updates"])
            result = await update_user_endpoint(arguments["user_id"], user_updates)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_delete_user":
            result = await delete_user_endpoint(arguments["user_id"])
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_roles":
            result = await get_user_roles_endpoint(arguments["user_id"])
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_groups":
            transitive = arguments.get("transitive", False)
            result = await get_user_groups_endpoint(arguments["user_id"], transitive)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_list_roles":
            result = await list_roles_endpoint()
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_user_to_role":
            role_member = RoleAddMember(user_id=arguments["user_id"])
            result = await add_user_to_role_endpoint(arguments["role_id"], role_member)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_user_from_role":
            result = await remove_user_from_role_endpoint(arguments["role_id"], arguments["user_id"])
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_list_groups":
            security_only = arguments.get("security_only", False)
            unified_only = arguments.get("unified_only", False)
            select = arguments.get("select", "id,displayName,mailNickname,mail,securityEnabled,groupTypes")
            result = await list_groups_endpoint(security_only, unified_only, select)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_create_group":
            create_group_request = CreateGroupRequest(
                action="create_group",
                display_name=arguments["display_name"],
                mail_nickname=arguments["mail_nickname"],
                description=arguments.get("description"),
                group_type=arguments.get("group_type", "security"),
                visibility=arguments.get("visibility"),
                owners=arguments.get("owners"),
                members=arguments.get("members")
            )
            result = await create_group_endpoint(create_group_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_group_member":
            group_member = GroupMemberRequest(user_id=arguments["user_id"])
            result = await add_group_member_endpoint(arguments["group_id"], group_member)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_group_member":
            result = await remove_group_member_endpoint(arguments["group_id"], arguments["user_id"])
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_group_members":
            result = await get_group_members_endpoint(arguments["group_id"])
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "retrieve_documents":
            # Use the new rag_tools module
            result = await retrieve_documents_tool(
                query=arguments.get("query"),
                top_k=arguments.get("top_k", 3),
                collection_name=arguments.get("collection_name", "json_documents_collection"),
                language=arguments.get("language", "Answer in English")
            )
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=result)]
            )

        else:
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=f"Unknown tool: {tool_name}")],
                isError=True
            )

    except Exception as e:
        if DEBUG:
            print(f"[MCP] Error calling tool {request.name}: {e}")
        return MCPToolCallResponse(
            content=[MCPContent(type="text", text=f"Error calling tool {request.name}: {str(e)}")],
            isError=True
        )


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "MCP Server with LLM SQL Generation",
        "version": "1.0.0",
        "description": "Standalone MCP Tools Server using LLM for natural language to SQL conversion",
        "protocols": ["REST API", "MCP (Model Context Protocol)"],
        "mcp_endpoints": {
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call",
            "server_info": "/mcp/server/info"
        },
        "rest_endpoints": [
            "/greet",
            "/query_database",
            "/query_database_stream",
            "/weather",
            "/ad/users",
            "/ad/roles", 
            "/ad/groups",
            "/health"
        ],
        "features": [
            "LLM SQL Generation",
            "OpenWeather Integration",
            "Async Database Queries",
            "Streaming Support",
            "Active Directory Operations",
            "MCP Protocol Support",
            "Vector Database Integration"
        ],
        "tools": [
            "greet",
            "query_database",
            "get_current_weather",
            "ad_list_users",
            "ad_create_user",
            "ad_update_user",
            "ad_delete_user",
            "ad_get_user_roles",
            "ad_get_user_groups",
            "ad_list_roles",
            "ad_add_user_to_role",
            "ad_remove_user_from_role",
            "ad_list_groups",
            "ad_create_group",
            "ad_add_group_member",
            "ad_remove_group_member",
            "ad_get_group_members",
            "search_cosmic_database",
            "hospital_support_questions_tool",
            "rag_retrieve_documents"
        ],
        "docs": "/docs",
        "mcp_compatible": True
    }


if __name__ == "__main__":
    print("Starting MCP Server on port 8009...")
    print(f"Database URL: {DATABASE_SERVER_URL}")
    print(f"Weather API: {'Configured' if OPENWEATHER_API_KEY else 'Not configured'}")
    if DEBUG:
        print("[MCP DEBUG] Debug mode enabled - detailed logging active")
    uvicorn.run(app, host="0.0.0.0", port=8009)

# python -c "import requests; print(requests.post('http://mcp_server:8009/mcp/tools/call', json={'name': 'get_current_weather', 'arguments': {'city': 'Yaoundé', 'units': 'metric'}}).json()['content'][0]['text'])"