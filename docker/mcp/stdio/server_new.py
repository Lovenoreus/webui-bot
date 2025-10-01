# -------------------- Built-in Libraries --------------------
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Any, Literal
import os

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- User-defined Libraries --------------------
import config

# -------------------- User-defined Modules --------------------


# SQL Models
class QueryDatabaseRequest(BaseModel):
    query: str
    keywords: List[str]


# Request models
class GreetRequest(BaseModel):
    name: Optional[str]


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


load_dotenv()

# Debug flag
DEBUG = True

app = FastAPI(title="Invoice MCP Server API", description="Standalone MCP Tools Server with LLM SQL Generation for Invoice Management")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

async def greet(name: Optional[str] = None) -> str:
    """
    Provide a friendly greeting to the user with appropriate time-based salutation.
    """
    if not name or name.strip() == "":
        name = None

    hour = datetime.now().hour

    if 5 <= hour < 12:
        time_greeting = "Good morning"
    elif 12 <= hour < 17:
        time_greeting = "Good afternoon"
    elif 17 <= hour < 21:
        time_greeting = "Good evening"
    else:
        time_greeting = "Good evening"

    if name:
        response = f"[RESPONSE]: {time_greeting} {name}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, payment tracking, and financial reporting. What can I do for you?"
    else:
        response = f"[RESPONSE]: {time_greeting}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, payment tracking, and financial reporting. How can I assist you today?"

    return f"{response}\n\n[Success]"

# --------------------
# SQL START
# --------------------
def extract_sql_from_json(llm_output: str) -> str:
    """Extract SQL from LLM response - handles JSON, markdown, or plain SQL"""
    import re

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


def get_database_server_url():
    """Determine the correct database server URL based on environment"""
    if os.environ.get('DOCKER_CONTAINER') or os.path.exists('/.dockerenv'):
        return config.MCP_DOCKER_DATABASE_SERVER_URL
    else:
        return config.MCP_DATABASE_SERVER_URL


DATABASE_SERVER_URL = get_database_server_url()

if DEBUG:
    print(f"[MCP DEBUG] Database server URL: {DATABASE_SERVER_URL}")


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


class InvoiceKeywordHints:
    def __init__(self):
        self.all_tables = ['Invoice', 'Invoice_Line']

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

            # Special cases for invoice table variations
            table_variations = {
                'invoice': ['invoice', 'bill', 'receipt', 'payment'],
                'invoice_line': ['line', 'item', 'product', 'service']
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
hint_generator = InvoiceKeywordHints()


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

    # STEP 3: Build system prompt with enhanced SQL generation rules for invoices
    base_system_prompt = """You are a helpful SQL query assistant for an invoice management database. The database contains the following tables and structure:

    ## Database Schema

    ### Core Tables

    **Invoice**
    ```sql
    CREATE TABLE Invoice (
        INVOICE_ID TEXT NOT NULL PRIMARY KEY,
        ISSUE_DATE TEXT NOT NULL,
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
        SUPPLIER_PARTY_NAME TEXT,
        SUPPLIER_PARTY_STREET_NAME TEXT,
        SUPPLIER_PARTY_ADDITIONAL_STREET_NAME TEXT,
        SUPPLIER_PARTY_POSTAL_ZONE TEXT,
        SUPPLIER_PARTY_CITY TEXT,
        SUPPLIER_PARTY_COUNTRY TEXT,
        SUPPLIER_PARTY_ADDRESS_LINE TEXT,
        SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM TEXT,
        SUPPLIER_PARTY_CONTACT_NAME TEXT,
        SUPPLIER_PARTY_CONTACT_EMAIL TEXT,
        SUPPLIER_PARTY_CONTACT_PHONE TEXT,
        SUPPLIER_PARTY_ENDPOINT_ID TEXT,
        CUSTOMER_PARTY_ID TEXT,
        CUSTOMER_PARTY_ID_SCHEME_ID TEXT,
        CUSTOMER_PARTY_ENDPOINT_ID TEXT,
        CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID TEXT,
        CUSTOMER_PARTY_NAME TEXT,
        CUSTOMER_PARTY_STREET_NAME TEXT,
        CUSTOMER_PARTY_POSTAL_ZONE TEXT,
        CUSTOMER_PARTY_COUNTRY TEXT,
        CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
        CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT,
        CUSTOMER_PARTY_CONTACT_NAME TEXT,
        CUSTOMER_PARTY_CONTACT_EMAIL TEXT,
        CUSTOMER_PARTY_CONTACT_PHONE TEXT,
        DUE_DATE TEXT,
        DOCUMENT_CURRENCY_CODE TEXT,
        DELIVERY_LOCATION_STREET_NAME TEXT,
        DELIVERY_LOCATION_ADDITIONAL_STREET_NAME TEXT,
        DELIVERY_LOCATION_CITY_NAME TEXT,
        DELIVERY_LOCATION_POSTAL_ZONE TEXT,
        DELIVERY_LOCATION_ADDRESS_LINE TEXT,
        DELIVERY_LOCATION_COUNTRY TEXT,
        DELIVERY_PARTY_NAME TEXT,
        ACTUAL_DELIVERY_DATE TEXT,
        TAX_AMOUNT_CURRENCY TEXT,
        TAX_AMOUNT REAL,
        PERIOD_START_DATE TEXT,
        PERIOD_END_DATE TEXT,
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT REAL,
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY TEXT,
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT REAL,
        BUYER_REFERENCE TEXT,
        PROJECT_REFERENCE_ID TEXT,
        INVOICE_TYPE_CODE TEXT,
        NOTE TEXT,
        TAX_POINT_DATE TEXT,
        ACCOUNTING_COST TEXT,
        ORDER_REFERENCE_ID TEXT,
        ORDER_REFERENCE_SALES_ORDER_ID TEXT,
        PAYMENT_TERMS_NOTE TEXT,
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID TEXT,
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE TEXT,
        CONTRACT_DOCUMENT_REFERENCE_ID TEXT,
        DESPATCH_DOCUMENT_REFERENCE_ID TEXT,
        ETL_LOAD_TS TEXT
    );
    ```

    **Invoice_Line**
    ```sql
    CREATE TABLE Invoice_Line (
        INVOICE_ID TEXT NOT NULL,
        ISSUE_DATE TEXT NOT NULL,
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
        INVOICE_LINE_ID TEXT NOT NULL,
        ORDER_LINE_REFERENCE_LINE_ID TEXT,
        ACCOUNTING_COST TEXT,
        INVOICED_QUANTITY REAL,
        INVOICED_QUANTITY_UNIT_CODE TEXT,
        INVOICED_LINE_EXTENSION_AMOUNT REAL,
        INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID TEXT,
        INVOICE_PERIOD_START_DATE TEXT,
        INVOICE_PERIOD_END_DATE TEXT,
        INVOICE_LINE_DOCUMENT_REFERENCE_ID TEXT,
        INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE TEXT,
        INVOICE_LINE_NOTE TEXT,
        ITEM_DESCRIPTION TEXT,
        ITEM_NAME TEXT,
        ITEM_TAXCAT_ID TEXT,
        ITEM_TAXCAT_PERCENT REAL,
        ITEM_BUYERS_ID TEXT,
        ITEM_SELLERS_ITEM_ID TEXT,
        ITEM_STANDARD_ITEM_ID TEXT,
        ITEM_COMMODITYCLASS_CLASSIFICATION TEXT,
        ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID TEXT,
        PRICE_AMOUNT REAL,
        PRICE_AMOUNT_CURRENCY_ID TEXT,
        PRICE_BASE_QUANTITY REAL,
        PRICE_BASE_QUANTITY_UNIT_CODE TEXT,
        PRICE_ALLOWANCE_CHARGE_AMOUNT REAL,
        PRICE_ALLOWANCE_CHARGE_INDICATOR TEXT,
        ETL_LOAD_TS TEXT,
        PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
        FOREIGN KEY (INVOICE_ID) REFERENCES Invoice(INVOICE_ID)
    );
    ```

    ## Sample Data Context

    ### Common Swedish Suppliers
    - JA Hotel Karlskrona, Visma Draftit AB, Abbott Scandinavia, Nordic IT Solutions AB

    ### Common Customers  
    - Region Västerbotten, Stockholms Stad, Region Skåne, Västra Götaland

    ### Common Services/Items
    - IT Consulting, Software License, Hotel Accommodation, Training Services, Medical Supplies, Office Equipment

    ### Currency
    - All amounts in SEK (Swedish Krona)

    ## Key Relationships
    - **One-to-Many**: Invoice → Invoice_Line (one invoice can have multiple line items)
    - **Join Key**: INVOICE_ID

    ## Common Query Patterns

    ### Financial Queries
    ```sql
    -- Total invoice amounts by supplier
    SELECT SUPPLIER_PARTY_NAME, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) as total_amount
    FROM Invoice 
    GROUP BY SUPPLIER_PARTY_NAME;

    -- Invoice details with line items
    SELECT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, il.ITEM_NAME, il.INVOICED_QUANTITY, il.PRICE_AMOUNT
    FROM Invoice i
    JOIN Invoice_Line il ON i.INVOICE_ID = il.INVOICE_ID
    WHERE i.INVOICE_ID = ?;

    -- Overdue invoices
    SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
    FROM Invoice 
    WHERE DUE_DATE < date('now');
    ```

    ### Reporting Queries
    ```sql
    -- Monthly invoice summary
    SELECT strftime('%Y-%m', ISSUE_DATE) as month, 
           COUNT(*) as invoice_count,
           SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) as total_amount
    FROM Invoice 
    GROUP BY strftime('%Y-%m', ISSUE_DATE);

    -- Top suppliers by volume
    SELECT SUPPLIER_PARTY_NAME, COUNT(*) as invoice_count, 
           SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) as total_spent
    FROM Invoice 
    GROUP BY SUPPLIER_PARTY_NAME 
    ORDER BY total_spent DESC;
    ```

    ## Instructions
    1. **Always use proper JOINs** to connect Invoice and Invoice_Line tables
    2. **Use table aliases** for readability (i for Invoice, il for Invoice_Line)
    3. **Include relevant financial columns** for business reporting
    4. **Consider date filters** using SQLite date functions
    5. **Handle NULL values** appropriately in conditions
    6. **Use REAL for monetary calculations**
    7. **Include proper ORDER BY** for meaningful result sorting
    8. **Group by supplier/customer** for aggregation queries

    When users ask about invoices, generate efficient SQL queries using this schema. Focus on financial reporting, supplier analysis, payment tracking, and business intelligence needs.

    ## CRITICAL OUTPUT FORMAT
    You must respond with ONLY a valid SQL query. No explanations, no markdown, no code blocks.
    Return only the raw SQL statement that can be executed directly.

    Example response format:
    SELECT * FROM Invoice WHERE SUPPLIER_PARTY_COUNTRY = 'SE'

    Do not wrap in ```sql blocks. Do not add explanations. Just the SQL query.
    """

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


# --------------------
# SQL STOP
# --------------------


# --------------------
# ENDPOINTS START
# --------------------

@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name with time-based salutation"""
    try:
        name = request.name if request.name and request.name.strip() else None
        message = await greet(name)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_sql_database")
async def query_sql_database_endpoint(request: QueryDatabaseRequest):
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
            "sql_query": f"```sql {sql_query} ```",
            "results": results,
            "original_query": request.query,
            "record_count": len(results)
        }

    except Exception as e:
        if DEBUG:
            print(f"[MCP DEBUG] Error in query_sql_database: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_query": request.query
        }


@app.post("/query_sql_database_stream")
async def query_sql_database_stream_endpoint(request: QueryDatabaseRequest):
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
                "sql_query": f"```sql {sql_query} ```"
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

# --------------------
# ENDPOINTS STOP
# --------------------


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
            "service": "Invoice MCP Server",
            "timestamp": datetime.now().isoformat(),
            "database_connection": db_health,
            "llm_providers": ["openai", "ollama", "mistral"],
            "endpoints": {
                "greet": "/greet",
                "query_sql_database": "/query_sql_database",
                "query_sql_database_stream": "/query_sql_database_stream",
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
            description="TRIGGER: hello, hi, good morning, good afternoon, good evening, introduce yourself, who are you, start conversation | ACTION: Welcome user with time-based greeting | RETURNS: Personalized greeting with invoice management capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's name", "default": ""}
                },
                "required": ["name"]
            }
        ),
        MCPTool(
            name="query_sql_database",
            description="TRIGGER: SQL, SQL database, query database, invoice database, invoice queries, supplier information, customer data, payment tracking, financial reports, invoice amounts, due dates, supplier analysis, customer analysis, invoice lines, item details, tax information, payment terms | ACTION: Query invoice SQL database with AI-generated SQL from natural language | RETURNS: Structured invoice data with the generated SQL query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question about the invoice database (invoices, suppliers, customers, payments, line items)"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key invoice/business terms from the query: supplier names, customer names, invoice IDs, cities, amounts, currencies, item names, payment terms, dates"
                    }
                },
                "required": ["query", "keywords"]
            }
        )
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

        if tool_name == "greet":
            raw_name = arguments.get("name", "")
            clean_name = raw_name if raw_name and raw_name.strip() else None
            greet_request = GreetRequest(name=clean_name)
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "query_sql_database":
            query_request = QueryDatabaseRequest(**arguments)
            result = await query_sql_database_endpoint(query_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
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


@app.post("/")
async def mcp_streamable_http_endpoint(request: Request):
    """Streamable HTTP MCP protocol endpoint for MCPO compatibility"""
    try:
        body = await request.json()
        method = body.get("method")
        request_id = body.get("id")

        if DEBUG:
            print(f"[STREAMABLE HTTP] Received method: {method}")
            print(f"[STREAMABLE HTTP] Body: {body}")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "Invoice SQL MCP Server",
                        "version": "1.0.0",
                        "description": "MCP server with LLM-powered SQL generation for invoice management databases"
                    }
                }
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Initialize response: {response}")
            return response

        elif method == "notifications/initialized":
            if DEBUG:
                print(f"[STREAMABLE HTTP] Received initialized notification - connection established")
            return Response(status_code=204)

        elif method == "tools/list":
            tools_response = await mcp_tools_list()
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [tool.dict() for tool in tools_response.tools]}
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Tools list response: {response}")
            return response

        elif method == "tools/call":
            params = body.get("params", {})
            call_request = MCPToolCallRequest(**params)
            result = await mcp_tools_call(call_request)
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result.dict()
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Tools call response: {response}")
            return response

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            }

        else:
            if DEBUG:
                print(f"[STREAMABLE HTTP] Unknown method: {method}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    except Exception as e:
        if DEBUG:
            print(f"[STREAMABLE HTTP] Error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.post("/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint"""
    body = await request.json()
    print(f"[DEBUG] {json.dumps(body, indent=2)}")
    return {"received": body, "timestamp": datetime.now().isoformat()}


@app.get("/info")
async def server_info():
    """Server information endpoint"""
    return {
        "service": "Invoice SQL MCP Server",
        "version": "1.0.0",
        "description": "MCP Tools Server with LLM-powered natural language to SQL conversion for invoice database queries",
        "protocols": ["REST API", "MCP (Model Context Protocol)"],

        "mcp_endpoints": {
            "streamable_http": "/",
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call"
        },

        "rest_endpoints": [
            "/greet",
            "/query_sql_database",
            "/query_sql_database_stream",
            "/health",
            "/info",
            "/debug"
        ],

        "features": [
            "LLM SQL Generation (Ollama, OpenAI, Mistral)",
            "Context-Aware Keyword Search",
            "Streaming Query Execution",
            "Invoice Domain Optimization",
            "MCP Protocol Support",
            "Async Database Operations"
        ],

        "database": {
            "url": DATABASE_SERVER_URL,
            "tables": hint_generator.all_tables,
            "table_count": len(hint_generator.all_tables)
        },

        "tools": [
            "greet",
            "query_sql_database"
        ],

        "docs": "/docs",
        "mcp_compatible": True,
        "debug_mode": DEBUG
    }


if __name__ == "__main__":
    print(f"Starting Invoice MCP Server on port 8009...")
    print(f"Database: {DATABASE_SERVER_URL}")

    if DEBUG:
        print("[DEBUG] Debug mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=8009)