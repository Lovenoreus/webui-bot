# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Optional
import os
import re

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- User-defined Modules --------------------
import config
# from vanna_engine import initialize_vanna_engine, vanna_manager, generate_sql
from vanna_engine import VannaModelManager
from models import (
    QueryDatabaseRequest,
    GreetRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
)
from vanna_train_remote_db import train_for_remote_db
from vanna_train_local_db import train_for_local_db
from query_rephraser import rephrase_query

load_dotenv(find_dotenv())

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


# Database configuration
DATABASE_CHOICE = config.DATABASE_CHOICE
USE_REMOTE = DATABASE_CHOICE == "remote"

if USE_REMOTE:
    DATABASE_SERVER_URL = f"SQL Server: {config.SQL_SERVER_HOST}/{config.SQL_SERVER_DATABASE}"

else:
    DATABASE_SERVER_URL = get_database_server_url()

# For docker database server
DATABASE_SERVER_ENDPOINT = "http://database_server:8762"

if DEBUG:
    print(f"[MCP DEBUG] Database mode: {DATABASE_CHOICE}")
    print(f"[MCP DEBUG] Using prompt: {'SQL Server' if USE_REMOTE else 'SQLite'}")
    print(f"[MCP DEBUG] Database info: {DATABASE_SERVER_URL}")

# Initialize Vanna Manager at module level
print("Initializing Vanna Manager...")

# Custom path
vanna_manager = VannaModelManager(
    chroma_path=config.CHROMA_DB_PATH,
    clear_existing=True  # Clear the path before setup.
)

# Initialize on module load
print("Initializing Vanna...")
vanna_manager.initialize_vanna()

print(f"✅ Vanna initialized with provider: {vanna_manager.current_provider}")

# Train with your schema
print("📚 Training Vanna with schema...")

print(f"Remote choice for Vanna is: {USE_REMOTE}")

# Get the training data.
# invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, synonym_instructions, training_pairs = get_vanna_training(remote=USE_REMOTE)


# MANUAL TRAINING. TO SEND TO TRAINING.
# CRITICAL: Database dialect and naming conventions
# if vanna_manager.train(ddl="""
# -- Database: Nodinite
# -- Schema: dbo
# -- Dialect: T-SQL (Microsoft SQL Server)
# -- CRITICAL REQUIREMENT: ALL table references MUST use three-part names: [Nodinite].[dbo].[TableName]
# """):
#     print("✅ Successfully trained Schema DDL")

start_time = datetime.now()

# Train depending on the database choice.
if USE_REMOTE:
    vanna_manager = train_for_remote_db(vanna_manager)

else:
    vanna_manager = train_for_local_db(vanna_manager)

print(f"🕒 Completed remote DB specific training in:", datetime.now() - start_time)

# Added a strict syntax command.
vanna_manager.train(
    documentation="#STRICT SYNTAX RULE: Your SQL should be on a single line with no line breaks. It should follow this exact syntax ```sql <command> ```. Do not add comments to the code you generate")

# Add comprehensive plural/singular handling instructions
# vanna_manager.train(documentation=get_comprehensive_search_instructions())

# Auto-train on startup if enabled
if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
    print("\n🔄 Auto-training enabled. Additional training can be done manually with vanna_manager.train(...).")

else:
    print("\n🔄 Auto-training is disabled. Use vanna_manager.train(...) to manually train the model.")


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


# Helper function
def format_vanna_sql(vanna_sql: str) -> str:
    """
    Clean and format SQL:
    - Remove line and block comments
    - Remove extra whitespace and newlines
    - Return clean single-line T-SQL (for MSSQL)
    """

    # Remove -- single-line comments
    sql_no_line_comments = re.sub(r'--.*?(?=\n|$)', '', vanna_sql)

    # Remove /* block */ comments
    sql_no_block_comments = re.sub(r'/\*.*?\*/', '', sql_no_line_comments, flags=re.DOTALL)

    # Remove newlines and collapse multiple spaces
    sql_clean = re.sub(r'\s+', ' ', sql_no_block_comments)

    return sql_clean.strip()


# Helper function
async def execute_query(self, query: str):
    """Execute query via database API server (works for both local and remote)"""
    try:
        if self.debug:
            print(f"[QueryEngine] Executing query via API: {query}")
            print(f"[QueryEngine] Database Endpoint URL: {DATABASE_SERVER_ENDPOINT}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{DATABASE_SERVER_ENDPOINT}/query",
                    json={"query": query},
                    timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                result = await response.json()

                if self.debug:
                    print(f"[QueryEngine] Query result success: {result.get('success')}")
                    if result.get('success'):
                        print(f"[QueryEngine] Returned {len(result.get('data', []))} rows")

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


# Helper function
async def execute_query_stream(self, query: str):
    """Stream database results via API server (works for both local and remote)"""
    try:
        if self.debug:
            print(f"[QueryEngine] Starting streaming query via API: {query}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{DATABASE_SERVER_ENDPOINT}/query_stream",
                    json={"query": query},
                    timeout=aiohttp.ClientTimeout(total=300)
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
                                print(f"[QueryEngine] Streaming data type: {data.get('type')}")

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


@app.post("/query_sql_database")
async def query_sql_database_endpoint(request: QueryDatabaseRequest):
    """Query invoice database with natural language"""
    try:
        if DEBUG:
            print(f"[MCP DEBUG] Query: {request.query}")
            # print(f"[MCP DEBUG] Keywords: {request.keywords}")

        # TODO: QUERY HARDCODED TO [].
        #  BECAUSE KEYWORD HITTING FAILS FOR LARGE DBs
        keywords = []

        # provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        # TODO: WE CAN SET VANNA UP HERE!
        # sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)
        # sql_query = await query_engine.generate_sql(request.query, keywords, provider)
        # TODO: GENERATE SQL QUERY WITH VANNA

        # Enhance query to search for both singular and plural forms
        # enhanced_query = normalize_for_comprehensive_search(request.query)

        # Add additional instructions for better SQL generation
        # enhanced_query = enhanced_query + "\nMake sure you use Like and Lower Keywords to compare the values if needed, to get better results."
        rephrased_query = rephrase_query(request.query)

        sql_query = vanna_manager.generate_sql(rephrased_query)
        # sql_query = vanna_manager.generate_sql(request.query)

        print(f"Vanna Generated SQL: {sql_query}")

        if sql_query:
            sql_query = format_vanna_sql(vanna_sql=sql_query)

        if not sql_query:
            # return {"success": False, "error": "Failed to generate SQL", "original_query": request.query}
            return {"success": False, "error": "Failed to generate SQL", "original_query": request.query,
                    "rephrased_query": rephrased_query}

        results = await execute_query(sql_query)

        return {
            "success": True,
            "sql_query": f"```sql {sql_query} ```",
            "results": results,
            "original_query": request.query,
            "rephrased_query": rephrased_query,
            "record_count": len(results)
        }

    except Exception as e:
        return {"success": False, "error": str(e), "original_query": request.query}


# TODO: NEEDS UPDATE TO VANNA INSTEAD OF QUERY ENGINE
# @app.post("/query_sql_database_stream")
# async def query_sql_database_stream_endpoint(request: QueryDatabaseRequest):
#     """Stream invoice query results"""
#
#     async def generate_response():
#         try:
#             provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"
#
#             sql_query = ""
#             async for sql_update in query_engine.generate_sql_stream(request.query, request.keywords, provider):
#                 yield json.dumps(sql_update) + "\n"
#                 if sql_update.get("status") == "sql_complete":
#                     sql_query = sql_update.get("sql_query", "")
#
#             if not sql_query:
#                 yield json.dumps({"success": False, "error": "Failed to generate SQL"}) + "\n"
#                 return
#
#             yield json.dumps({"status": "executing_query", "sql_query": f"```sql {sql_query} ```"}) + "\n"
#
#             async for db_result in execute_query_stream(sql_query):
#                 db_result["original_query"] = request.query
#                 yield json.dumps(db_result) + "\n"
#
#         except Exception as e:
#             yield json.dumps({"success": False, "error": str(e)}) + "\n"
#
#     return StreamingResponse(generate_response(), media_type="application/x-ndjson")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = "unknown"
        db_type = "remote" if USE_REMOTE else "local"

        if USE_REMOTE:
            # Test remote SQL Server connectivity
            try:
                # Verify tables exist
                # result = await query_engine.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice]")
                # invoice_count = result[0]['invoice_count'] if result else 0

                # lines_result =  await query_engine.execute_query("SELECT TOP 1 * FROM [Nodinite].[ods].[Invoice_Line]")
                # line_count = lines_result[0]['line_count'] if lines_result else 0

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
                    # "invoice_count": invoice_count,
                    # "line_count": line_count,
                    # "tables_initialized": invoice_count > 0 and line_count > 0,
                    "llm_providers": ["openai", "ollama", "mistral"],
                    "endpoints": {
                        "greet": "/greet",
                        "query_sql_database": "/query_sql_database",
                        "query_sql_database_stream": "/query_sql_database_stream",
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
                                "endpoints": {
                                    "greet": "/greet",
                                    "query_sql_database": "/query_sql_database",
                                    "query_sql_database_stream": "/query_sql_database_stream",
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
            description="""TRIGGER: SQL, SQL database, query database, invoice database, invoice queries, supplier information, customer data, payment tracking, financial reports, invoice amounts, due dates, supplier analysis, customer analysis, invoice lines, item details, tax information, payment terms

        SCOPE: Use this tool for invoice, payment, supplier, purchase, or expense-related questions. Handles financial transaction data including amounts, dates, items, vendors, and deliveries.

        ACTION: Converts natural language questions to SQL and queries the invoice database

        RETURNS: Structured invoice data with the generated SQL query""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """Natural language question about the invoice database - leverages context awareness from previous chat
                                        **Example:**
                - User asks: "How many invoices does ABC Corp have?"
                - User then asks: "What's the IT department for that company?"
                - You should understand "that company" = "ABC Corp" and query accordingly"""
                    }
                },
                "required": ["query"]
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
                "result": {"tools": [tool.model_dump() for tool in tools_response.tools]}
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
                "result": result.model_dump()
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
            "Local SQLite + Remote SQL Server Support"
        ],

        "database": {
            "mode": DATABASE_CHOICE
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
    print(f"Database mode: {DATABASE_CHOICE}")
    print(f"Database info: {DATABASE_SERVER_URL}")

    if DEBUG:
        print("[DEBUG] Debug mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=8009)
