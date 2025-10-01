# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Optional
import os

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- User-defined Modules --------------------
import config
from query_engine import QueryEngine
from models import (
    QueryDatabaseRequest,
    GreetRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
)

load_dotenv()

# Debug flag
DEBUG = True

app = FastAPI(
    title="Invoice SQL MCP Server", 
    description="MCP Tools Server with LLM SQL Generation for Invoice Management"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def get_database_server_url():
    """Determine the correct database server URL based on environment"""
    if os.environ.get('DOCKER_CONTAINER') or os.path.exists('/.dockerenv'):
        return config.MCP_DOCKER_DATABASE_SERVER_URL
    else:
        return config.MCP_DATABASE_SERVER_URL


DATABASE_SERVER_URL = get_database_server_url()

if DEBUG:
    print(f"[MCP DEBUG] Database server URL: {DATABASE_SERVER_URL}")


# Invoice-specific configuration
INVOICE_TABLES = [
    'Invoice',
    'Invoice_Line'
]

INVOICE_TABLE_VARIATIONS = {
    'invoice': ['invoice', 'bill', 'receipt', 'payment'],
    'invoice_line': ['line', 'item', 'product', 'service']
}

INVOICE_SYSTEM_PROMPT = """You are a helpful SQL query assistant for an invoice management database. The database contains the following tables and structure:

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

# Initialize QueryEngine with invoice configuration
query_engine = QueryEngine(
    database_url=DATABASE_SERVER_URL,
    tables=INVOICE_TABLES,
    table_variations=INVOICE_TABLE_VARIATIONS,
    system_prompt=INVOICE_SYSTEM_PROMPT,
    debug=DEBUG
)


async def greet(name: Optional[str] = None) -> str:
    """Provide a friendly greeting to the user."""
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
        response = f"[RESPONSE]: {time_greeting} {name}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, and payment tracking. What can I do for you?"
    else:
        response = f"[RESPONSE]: {time_greeting}! I'm your Invoice Management assistant. I can help with invoice queries, supplier information, customer data, and payment tracking. How can I assist you today?"

    return f"{response}\n\n[Success]"


# --------------------
# ENDPOINTS
# --------------------

@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name"""
    try:
        name = request.name if request.name and request.name.strip() else None
        message = await greet(name)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_sql_database")
async def query_sql_database_endpoint(request: QueryDatabaseRequest):
    """Query invoice database with natural language"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Query: {request.query}")
            print(f"[MCP DEBUG] Keywords: {request.keywords}")

        provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)

        if not sql_query:
            return {"success": False, "error": "Failed to generate SQL", "original_query": request.query}

        results = await query_engine.execute_query(sql_query)

        return {
            "success": True,
            "sql_query": f"```sql {sql_query} ```",
            "results": results,
            "original_query": request.query,
            "record_count": len(results)
        }

    except Exception as e:
        return {"success": False, "error": str(e), "original_query": request.query}


@app.post("/query_sql_database_stream")
async def query_sql_database_stream_endpoint(request: QueryDatabaseRequest):
    """Stream invoice query results"""

    async def generate_response():
        try:
            provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

            sql_query = ""
            async for sql_update in query_engine.generate_sql_stream(request.query, request.keywords, provider):
                yield json.dumps(sql_update) + "\n"
                if sql_update.get("status") == "sql_complete":
                    sql_query = sql_update.get("sql_query", "")

            if not sql_query:
                yield json.dumps({"success": False, "error": "Failed to generate SQL"}) + "\n"
                return

            yield json.dumps({"status": "executing_query", "sql_query": f"```sql {sql_query} ```"}) + "\n"

            async for db_result in query_engine.execute_query_stream(sql_query):
                db_result["original_query"] = request.query
                yield json.dumps(db_result) + "\n"

        except Exception as e:
            yield json.dumps({"success": False, "error": str(e)}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


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
            "tables": query_engine.all_tables,
            "table_count": len(query_engine.all_tables)
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
