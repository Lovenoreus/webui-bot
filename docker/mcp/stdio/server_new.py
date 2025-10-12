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
# from vanna_engine import initialize_vanna_engine, vanna_manager, generate_sql
from vanna_engine import VannaModelManager
from training import get_vanna_training, get_vanna_question_sql_pairs
from models import (
    QueryDatabaseRequest,
    GreetRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
)

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
    database_url="http://database_server:8762",
    tables=INVOICE_TABLES,
    table_variations=INVOICE_TABLE_VARIATIONS,
    system_prompt=INVOICE_SYSTEM_PROMPT,
    debug=DEBUG,
    use_remote=USE_REMOTE
)


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
invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, synonym_instructions, training_pairs = get_vanna_training(remote=USE_REMOTE)

# Train Vanna with DDL and documentation
print("\n📚 Training Vanna with DDL and documentation...")
print("Training with Invoice DDL...")

if vanna_manager.train(ddl=invoice_ddl):
    print("✅ Successfully trained Invoice DDL")

else:
    print("❌ Failed to train Invoice DDL")

print("Training with Invoice_Line DDL...")
if vanna_manager.train(ddl=invoice_line_ddl):
    print("✅ Successfully trained Invoice_Line DDL")

else:
    print("❌ Failed to train Invoice_Line DDL")

print("Training with Invoice documentation (first 5 rows)...")
if vanna_manager.train(documentation=invoice_doc):
    print("✅ Successfully trained Invoice documentation")

else:
    print("❌ Failed to train Invoice documentation")

print("Training with Invoice_Line documentation (first 5 rows)...")
if vanna_manager.train(documentation=invoice_line_doc):
    print("✅ Successfully trained Invoice_Line documentation")

else:
    print("❌ Failed to train Invoice_Line documentation")

print("Training with synonym handling instructions...")
if vanna_manager.train(documentation=synonym_instructions):
    print("✅ Successfully trained synonym handling instructions")

else:
    print("❌ Failed to train synonym handling instructions")

# Train with question-SQL pairs
print("Training with question-SQL pairs...")
successful_pairs = 0
total_pairs = len(training_pairs)

for i, pair in enumerate(training_pairs, 1):
    question = pair["question"]
    sql = pair["sql"]
    print(f"Training pair {i}/{total_pairs}: {question[:50]}...")
    
    try:
        if vanna_manager.train(question=question, sql=sql):
            successful_pairs += 1
            print(f"✅ Successfully trained pair {i}")
        else:
            print(f"❌ Failed to train pair {i}")
    except Exception as e:
        print(f"❌ Error training pair {i}: {e}")

print(f"📊 Question-SQL training completed: {successful_pairs}/{total_pairs} pairs successful")

# Added a strict syntax command.
vanna_manager.train(documentation="#STRICT SYNTAX RULE: Your SQL should be on a single line with no line breaks. It should follow this exact syntax ```sql <command> ```")

# Add plural/singular handling instructions
singular_plural_instructions = """
# CRITICAL PLURAL/SINGULAR HANDLING FOR ITEM SEARCHES

## Key Rule: Always use SINGULAR forms in LIKE conditions for item names

## Core Principle:
When users ask about items in plural form, convert to singular in your LIKE conditions.
This ensures broader matching of data that may contain both singular and plural forms.

## Conversion Rules:
1. Remove 's' from regular plurals: 'computers' → 'computer'
2. Handle 'ies' endings: 'companies' → 'company', 'batteries' → 'battery' 
3. Handle 'es' endings: 'boxes' → 'box', 'glasses' → 'glass'
4. Handle 'ves' endings: 'knives' → 'knife', 'shelves' → 'shelf'
5. Handle compound words: 'swimming trunks' → 'swimming trunk'

## Why This Works:
- Searching for 'screw' matches: "Screw", "Screws", "Hex Screw", "Wood Screws", "Screw Set"
- Searching for 'computer' matches: "Computer", "Computers", "Computer Mouse", "Gaming Computer"
- Searching for 'swimming trunk' matches: "Swimming Trunk", "Swimming Trunks", "Men's Swimming Trunk"

## SQL Pattern:
User query: "What companies sold us swimming trunks?"
SQL: WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunk%')

User query: "How many screws did we buy?"  
SQL: WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%screw%')

User query: "Show me all batteries purchased"
SQL: WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%battery%')

## Implementation Strategy:
- Automatically convert ANY plural word in user query to singular form for LIKE conditions
- Use linguistic rules (remove 's', handle 'ies'→'y', 'es'→'', 'ves'→'f', etc.)
- This approach works for ANY item name, not just predefined ones
- Focus on the root/base form for maximum data coverage
"""

vanna_manager.train(documentation=singular_plural_instructions)

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


import re

def format_vanna_sql(vanna_sql):
    # Strip each line and join with a space
    sql_no_newlines = " ".join(line.strip() for line in vanna_sql.split('\n'))
    # Optionally collapse multiple spaces into one
    return re.sub(r'\s+', ' ', sql_no_newlines).strip()


def pluralize_to_singular(word):
    """
    Convert a single word from plural to singular using English pluralization rules.
    
    Args:
        word (str): The word to convert
        
    Returns:
        str: The singular form of the word
    """
    if len(word) < 2:
        return word
    
    word_lower = word.lower()
    
    # Handle irregular plurals
    irregular_plurals = {
        'children': 'child',
        'feet': 'foot',
        'geese': 'goose',
        'men': 'man',
        'women': 'woman',
        'teeth': 'tooth',
        'mice': 'mouse',
        'people': 'person',
        'oxen': 'ox',
        'deer': 'deer',
        'sheep': 'sheep',
        'fish': 'fish',
        'moose': 'moose',
        'series': 'series',
        'species': 'species',
        'data': 'datum',
        'media': 'medium',
        'criteria': 'criterion',
        'phenomena': 'phenomenon',
        'bacteria': 'bacterium',
        'alumni': 'alumnus',
        'fungi': 'fungus',
        'nuclei': 'nucleus',
        'cacti': 'cactus',
        'foci': 'focus',
        'radii': 'radius',
        'analyses': 'analysis',
        'bases': 'basis',
        'diagnoses': 'diagnosis',
        'oases': 'oasis',
        'theses': 'thesis',
        'crises': 'crisis',
        'axes': 'axis',
        'matrices': 'matrix',
        'vertices': 'vertex',
        'indices': 'index',
        'appendices': 'appendix'
    }
    
    if word_lower in irregular_plurals:
        # Preserve original case pattern
        singular = irregular_plurals[word_lower]
        if word.isupper():
            return singular.upper()
        elif word.istitle():
            return singular.capitalize()
        else:
            return singular
    
    # Handle regular plural patterns
    
    # Words ending in 'ies' -> 'y' (e.g., companies -> company, batteries -> battery)
    if word_lower.endswith('ies') and len(word) > 3:
        base = word[:-3] + 'y'
        return base
    
    # Words ending in 'ves' -> 'f' or 'fe' (e.g., knives -> knife, shelves -> shelf)
    if word_lower.endswith('ves') and len(word) > 3:
        if word_lower.endswith('ives'):
            # knives -> knife, lives -> life
            base = word[:-4] + 'ife'
        else:
            # shelves -> shelf, calves -> calf
            base = word[:-3] + 'f'
        return base
    
    # Words ending in 'ses' -> 's' (e.g., glasses -> glass, classes -> class)
    if word_lower.endswith('ses') and len(word) > 3:
        # Special case for 'chases', 'purchases', 'releases', etc.
        if word_lower.endswith('chases') or word_lower.endswith('eases'):
            return word[:-1]  # Remove just the 's'
        return word[:-2]
    
    # Words ending in 'xes' -> 'x' (e.g., boxes -> box, fixes -> fix)
    if word_lower.endswith('xes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in 'zes' -> 'z' (e.g., prizes -> prize)
    if word_lower.endswith('zes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in 'shes' -> 'sh' (e.g., dishes -> dish, brushes -> brush)
    if word_lower.endswith('shes') and len(word) > 4:
        return word[:-2]
    
    # Words ending in 'ches' -> 'ch' (e.g., watches -> watch, beaches -> beach)
    if word_lower.endswith('ches') and len(word) > 4:
        return word[:-2]
    
    # Words ending in 'oes' -> 'o' (e.g., tomatoes -> tomato, heroes -> hero)
    if word_lower.endswith('oes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in just 's' (most common case)
    if word_lower.endswith('s') and len(word) > 1:
        # Don't remove 's' from words that are naturally singular and end in 's'
        # (e.g., glass, pass, mass, etc.)
        potential_singular = word[:-1]
        
        # Simple heuristic: if removing 's' creates a very short word, it might be incorrect
        if len(potential_singular) < 2:
            return word
        
        # Special cases where the word naturally ends in 's' when singular
        if word_lower in ['glass', 'mass', 'pass', 'class', 'bass', 'grass', 'brass', 'cross']:
            return word
            
        # For most regular plurals, just remove the 's'
        return potential_singular
    
    # If no plural pattern matches, return the original word
    return word


def normalize_plural_to_singular(query):
    """
    Dynamically normalize plural forms to singular forms for better matching.
    This uses linguistic rules to handle any plural word, not just a static list.
    """
    import re
    
    # Split query into words, preserving spaces and punctuation
    words = re.findall(r'\b\w+\b|\W+', query)
    
    normalized_words = []
    query_changed = False
    
    for word in words:
        if re.match(r'\b\w+\b', word):  # It's a word
            singular_word = pluralize_to_singular(word)
            if singular_word.lower() != word.lower():
                query_changed = True
            normalized_words.append(singular_word)
        else:  # It's whitespace or punctuation
            normalized_words.append(word)
    
    normalized_query = ''.join(normalized_words)
    
    # If the query was changed, append instruction
    if query_changed:
        return f"{normalized_query}\n\nNOTE: When searching for items, use singular forms in LIKE conditions to match both singular and plural forms in the data. For example: use 'swimming trunk' to find both 'Swimming Trunk' and 'Swimming Trunks', use 'screw' to find 'Screw', 'Screws', etc."
    
    return query


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

        # provider = "ollama" if config.MCP_PROVIDER_OLLAMA else "openai" if config.MCP_PROVIDER_OPENAI else "mistral"

        # TODO: WE CAN SET VANNA UP HERE!
        # sql_query = await query_engine.generate_sql(request.query, request.keywords, provider)
        # sql_query = await query_engine.generate_sql(request.query, keywords, provider)
        # TODO: GENERATE SQL QUERY WITH VANNA

        # Normalize plural forms to singular for better matching
        normalized_query = normalize_plural_to_singular(request.query)
        
        # Add additional instructions for better SQL generation
        enhanced_query = normalized_query + "\nMake sure you use Like and Lower Keywords to compare the values if needed, to get better results."
        
        sql_query = vanna_manager.generate_sql(enhanced_query)
        print(f"Vanna Generated SQL: {sql_query}")

        if sql_query:
            sql_query = format_vanna_sql(vanna_sql=sql_query)

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
            description="TRIGGER: SQL, SQL database, query database, invoice database, invoice queries, supplier information, customer data, payment tracking, financial reports, invoice amounts, due dates, supplier analysis, customer analysis, invoice lines, item details, tax information, payment terms | ACTION: Query invoice SQL database with AI-generated SQL from natural language | RETURNS: Structured invoice data with the generated SQL query",
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
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extract keywords from the query"
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
            "mode": DATABASE_CHOICE,
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
    print(f"Database mode: {DATABASE_CHOICE}")
    print(f"Database info: {DATABASE_SERVER_URL}")

    if DEBUG:
        print("[DEBUG] Debug mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=8009)
