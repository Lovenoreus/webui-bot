# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Optional
import os

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- User-defined Modules --------------------
import config
from query_engine import QueryEngine
from instructions import SQLITE_INVOICE_PROMPT, SQLSERVER_INVOICE_PROMPT
from models import (
    QueryDatabaseRequest,
    GreetRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
)

# -------------------- Vanna Integration --------------------
import logging
import warnings
import builtins
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

load_dotenv(find_dotenv())

# Debug flag
DEBUG = True

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


# Database configuration
DATABASE_CHOICE = config.DATABASE_CHOICE
USE_REMOTE = DATABASE_CHOICE == "remote"

if USE_REMOTE:
    DATABASE_SERVER_URL = f"SQL Server: {config.SQL_SERVER_HOST}/{config.SQL_SERVER_DATABASE}"
else:
    DATABASE_SERVER_URL = get_database_server_url()

if DEBUG:
    print(f"[MCP DEBUG] Database mode: {DATABASE_CHOICE}")
    print(f"[MCP DEBUG] Database info: {DATABASE_SERVER_URL}")

# Invoice-specific configuration
INVOICE_TABLES = [
    'Invoice',
    'Invoice_Line'
]

INVOICE_TABLE_VARIATIONS = {
    'invoice': ['invoice', 'bill', 'receipt', 'payment'],
    'invoice_line': ['line', 'item', 'product', 'service']
}


if USE_REMOTE:
    INVOICE_SYSTEM_PROMPT = SQLSERVER_INVOICE_PROMPT

else:
    INVOICE_SYSTEM_PROMPT = SQLITE_INVOICE_PROMPT

if DEBUG:
    print(f"[MCP DEBUG] Database mode: {DATABASE_CHOICE}")
    print(f"[MCP DEBUG] Using prompt: {'SQL Server' if USE_REMOTE else 'SQLite'}")
    print(f"[MCP DEBUG] Database info: {DATABASE_SERVER_URL}")

# Initialize QueryEngine with invoice configuration
query_engine = QueryEngine(
    database_url=DATABASE_SERVER_URL if not USE_REMOTE else None,
    tables=INVOICE_TABLES,
    table_variations=INVOICE_TABLE_VARIATIONS,
    system_prompt=INVOICE_SYSTEM_PROMPT,
    debug=DEBUG,
    use_remote=USE_REMOTE
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

        # TODO: QUERY HARDCODED TO [].
        #  BECAUSE KEYWORD HITTING FAILS FOR LARGE DBs
        keywords = []

        provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        # sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)
        sql_query = await query_engine.generate_sql(request.query, keywords, provider)

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


@app.post("/query_vanna_database")
async def query_vanna_database_endpoint(request: QueryDatabaseRequest):
    """Query invoice database using Vanna AI with natural language"""
    try:
        if not vanna_instance:
            return {"success": False, "error": "Vanna instance not available", "original_query": request.query}
            
        if DEBUG:
            print(f"[Vanna DEBUG] Query: {request.query}")

        # Use filtered print during Vanna operations
        builtins.print = filtered_print
        
        # Use Vanna's ask method which handles SQL generation and execution
        sql, df, fig = vanna_instance.ask(
            question=request.query, 
            print_results=False, 
            allow_llm_to_see_data=True, 
            visualize=False
        )
        
        # Restore original print
        builtins.print = _original_print
        
        if sql is None or df is None:
            return {"success": False, "error": "Failed to generate or execute SQL", "original_query": request.query}

        # Convert DataFrame to list of dictionaries for JSON serialization
        if hasattr(df, 'to_dict'):
            results = df.to_dict('records')
        else:
            results = df if isinstance(df, list) else []

        return {
            "success": True,
            "sql_query": f"```sql {sql} ```",
            "results": results,
            "original_query": request.query,
            "record_count": len(results),
            "method": "vanna"
        }

    except Exception as e:
        # Restore original print in case of error
        builtins.print = _original_print
        return {"success": False, "error": str(e), "original_query": request.query}


@app.post("/query_vanna_database_stream")
async def query_vanna_database_stream_endpoint(request: QueryDatabaseRequest):
    """Stream Vanna query results"""
    
    async def generate_response():
        try:
            if not vanna_instance:
                yield json.dumps({"success": False, "error": "Vanna instance not available"}) + "\n"
                return
                
            yield json.dumps({"status": "generating_sql", "message": "Vanna is analyzing your question..."}) + "\n"
            
            # Use filtered print during Vanna operations
            builtins.print = filtered_print
            
            # Use Vanna's ask method
            sql, df, fig = vanna_instance.ask(
                question=request.query, 
                print_results=False, 
                allow_llm_to_see_data=True, 
                visualize=False
            )
            
            # Restore original print
            builtins.print = _original_print
            
            if sql is None or df is None:
                yield json.dumps({"success": False, "error": "Failed to generate or execute SQL"}) + "\n"
                return

            yield json.dumps({"status": "executing_query", "sql_query": f"```sql {sql} ```"}) + "\n"

            # Convert DataFrame to list of dictionaries for JSON serialization
            if hasattr(df, 'to_dict'):
                results = df.to_dict('records')
            else:
                results = df if isinstance(df, list) else []

            # Stream results row by row
            for idx, row in enumerate(results, 1):
                yield json.dumps({
                    "success": True,
                    "type": "row",
                    "data": row,
                    "index": idx,
                    "running_total": len(results)
                }) + "\n"

            # Final completion message
            yield json.dumps({
                "success": True,
                "type": "complete",
                "results": results,
                "record_count": len(results),
                "status": "finished",
                "method": "vanna"
            }) + "\n"

        except Exception as e:
            # Restore original print in case of error
            builtins.print = _original_print
            yield json.dumps({"success": False, "error": str(e)}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = "unknown"
        db_type = "remote" if USE_REMOTE else "local"

        if USE_REMOTE:
            # Test remote SQL Server connectivity
            try:
                test_query = "SELECT COUNT(*) as invoice_count FROM Invoice"
                result = await query_engine.execute_query(test_query)
                invoice_count = result[0]['invoice_count'] if result else 0

                lines_result = await query_engine.execute_query("SELECT COUNT(*) as line_count FROM Invoice_Line")
                line_count = lines_result[0]['line_count'] if lines_result else 0

                db_health = "connected"

                return {
                    "status": "healthy",
                    "service": "Invoice MCP Server",
                    "timestamp": datetime.now().isoformat(),
                    "database_type": db_type,
                    "database_connection": db_health,
                    "database_info": {
                        "server": config.SQL_SERVER_HOST,
                        "database": config.SQL_SERVER_DATABASE,
                        "auth": "Windows" if config.SQL_SERVER_USE_WINDOWS_AUTH else "SQL Server"
                    },
                    "invoice_count": invoice_count,
                    "line_count": line_count,
                    "tables_initialized": invoice_count > 0 and line_count > 0,
                    "llm_providers": ["openai", "ollama", "mistral"],
                    "vanna_status": "available" if vanna_instance else "unavailable",
                    "endpoints": {
                        "greet": "/greet",
                        "query_sql_database": "/query_sql_database",
                        "query_sql_database_stream": "/query_sql_database_stream",
                        "query_vanna_database": "/query_vanna_database",
                        "query_vanna_database_stream": "/query_vanna_database_stream",
                        "health": "/health"
                    }
                }
            except Exception as e:
                return {
                    "status": "error",
                    "service": "Invoice MCP Server",
                    "timestamp": datetime.now().isoformat(),
                    "database_type": db_type,
                    "database_connection": "error",
                    "database_info": {
                        "server": config.SQL_SERVER_HOST,
                        "database": config.SQL_SERVER_DATABASE
                    },
                    "error": str(e)
                }
        else:
            # Test local SQLite API connectivity
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{DATABASE_SERVER_URL}/health", timeout=5) as response:
                        if response.status == 200:
                            db_health = "connected"
                            health_data = await response.json()
                            return {
                                "status": "healthy",
                                "service": "Invoice MCP Server",
                                "timestamp": datetime.now().isoformat(),
                                "database_type": db_type,
                                "database_connection": db_health,
                                "database_server": DATABASE_SERVER_URL,
                                "invoice_count": health_data.get("invoice_count", 0),
                                "line_count": health_data.get("line_count", 0),
                                "tables_initialized": health_data.get("tables_initialized", False),
                                "llm_providers": ["openai", "ollama", "mistral"],
                                "vanna_status": "available" if vanna_instance else "unavailable",
                                "endpoints": {
                                    "greet": "/greet",
                                    "query_sql_database": "/query_sql_database",
                                    "query_sql_database_stream": "/query_sql_database_stream",
                                    "query_vanna_database": "/query_vanna_database",
                                    "query_vanna_database_stream": "/query_vanna_database_stream",
                                    "health": "/health"
                                }
                            }
                        else:
                            db_health = "error"
            except:
                db_health = "disconnected"

            return {
                "status": "error" if db_health != "connected" else "healthy",
                "service": "Invoice MCP Server",
                "timestamp": datetime.now().isoformat(),
                "database_type": db_type,
                "database_connection": db_health,
                "database_server": DATABASE_SERVER_URL,
                "error": "Cannot connect to database server" if db_health != "connected" else None
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
                        "description": "Extract keywords from the query"
                    }
                },
                "required": ["query", "keywords"]
            }
        ),
        MCPTool(
            name="query_vanna_database",
            description="TRIGGER: Vanna AI, AI-powered SQL, intelligent database query, smart SQL generation, vector search SQL, trained SQL model, advanced database analysis | ACTION: Query invoice database using Vanna AI with vector search and trained context | RETURNS: Structured invoice data with AI-generated SQL using Vanna's trained knowledge base",
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
                        "description": "Extract keywords from the query (used for compatibility but Vanna uses vector search)"
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

        elif tool_name == "query_vanna_database":
            query_request = QueryDatabaseRequest(**arguments)
            result = await query_vanna_database_endpoint(query_request)
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
        "database_mode": DATABASE_CHOICE,
        "database_info": DATABASE_SERVER_URL,

        "mcp_endpoints": {
            "streamable_http": "/",
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call"
        },

        "rest_endpoints": [
            "/greet",
            "/query_sql_database",
            "/query_sql_database_stream",
            "/query_vanna_database",
            "/query_vanna_database_stream",
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
            "Async Database Operations",
            "Local SQLite + Remote SQL Server Support",
            "Vanna AI Integration with Vector Search"
        ],

        "database": {
            "mode": DATABASE_CHOICE,
            "tables": query_engine.all_tables,
            "table_count": len(query_engine.all_tables)
        },

        "tools": [
            "greet",
            "query_sql_database",
            "query_vanna_database"
        ],

        "docs": "/docs",
        "mcp_compatible": True,
        "debug_mode": DEBUG,
        "vanna": {
            "status": "available" if vanna_instance else "unavailable",
            "database_path": db_path,
            "training_completed": vanna_instance is not None
        }
    }


if __name__ == "__main__":
    print(f"Starting Invoice MCP Server on port 8009...")
    print(f"Database mode: {DATABASE_CHOICE}")
    print(f"Database info: {DATABASE_SERVER_URL}")

    if DEBUG:
        print("[DEBUG] Debug mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=8009)
