# -------------------- Built-in Libraries --------------------
from typing import Dict, Optional, List, Any, Literal

# -------------------- External Libraries --------------------
from pydantic import BaseModel, Field, ValidationError


# -------------------- SQL MODELS --------------------
class QueryDatabaseRequest(BaseModel):
    query: str
    # keywords: List[str]


# -------------------- GreetRequest MODELS --------------------
class GreetRequest(BaseModel):
    name: Optional[str]


# -------------------- MCP PROTOCOL MODELS --------------------
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
