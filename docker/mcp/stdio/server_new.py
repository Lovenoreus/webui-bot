# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Dict, Optional, List, Any, Literal

# -------------------- External Libraries --------------------
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# -------------------- FastMCP Import --------------------
from fastmcp import FastMCP
from fastmcp.exceptions import McpError
from fastmcp import Client

# -------------------- User-defined Modules --------------------
from active_directory import FastActiveDirectory


# ++++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY PYDANTIC MODELS START
# ++++++++++++++++++++++++++++++++

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


class BatchUserIdentifiersRequest(BaseModel):
    identifiers: List[str] = Field(..., description="List of user IDs, emails, or display names")


class GreetRequest(BaseModel):
    name: Optional[str]


# ++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY PYDANTIC MODELS END
# ++++++++++++++++++++++++++++++


load_dotenv()

# Debug flag
DEBUG = True

# ++++++++++++++++++++++++++++++++
# FASTMCP SERVER SETUP
# ++++++++++++++++++++++++++++++++

# Create FastMCP server instance for tool management
mcp_server = FastMCP("MCP Server with LLM SQL Generation")


# ++++++++++++++++++++++++++++++++
# FASTMCP TOOL DEFINITIONS START
# ++++++++++++++++++++++++++++++++

@mcp_server.tool()
async def greet(name: str = "") -> str:
    """
    Provide a friendly greeting to the user with appropriate time-based salutation.

    Args:
        name: User's name (optional)
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
        response = f"[RESPONSE]: {time_greeting} {name}! I'm your Cosmic hospital assistant. I can help with policies, user management, database queries, weather information, and more. What can I do for you?"
    else:
        response = f"[RESPONSE]: {time_greeting}! I'm your Cosmic hospital assistant. I can help with policies, user management, database queries, weather information, and more. How can I assist you today?"

    return f"{response}\n\n[Success]"


@mcp_server.tool()
async def ad_list_users() -> dict:
    """
    List all Azure Active Directory user accounts.

    Returns:
        Dictionary containing success status and list of all AD users
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.list_users()
        return {"success": True, "action": "list_users", "data": data}


@mcp_server.tool()
async def ad_create_user(user: dict) -> dict:
    """
    Create new Azure Active Directory user account.

    Args:
        user: User payload with displayName (required) and optional properties

    Returns:
        Dictionary with created user details
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        user_payload = user.copy()
        if "displayName" in user_payload:
            clean_name = user_payload["displayName"].replace(" ", "").lower()
            user_payload["userPrincipalName"] = f"{clean_name}@lovenoreusgmail.onmicrosoft.com"
            if "mailNickname" not in user_payload:
                user_payload["mailNickname"] = clean_name
        data = await ad.create_user(user_payload)
        return {"success": True, "action": "create_user", "data": data}


@mcp_server.tool()
async def ad_update_user(user_id: str, updates: dict) -> dict:
    """
    Update existing Azure Active Directory user properties.

    Args:
        user_id: User identifier (GUID, email, or display name)
        updates: Dictionary of properties to update

    Returns:
        Dictionary with update confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.update_user_smart(user_id, updates)
        return {"success": True, "action": "update_user", "user_id": user_id, "data": data}


@mcp_server.tool()
async def ad_delete_user(user_id: str) -> dict:
    """
    Delete user account from Azure Active Directory.

    Args:
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with deletion confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.delete_user_smart(user_id)
        return {"success": True, "action": "delete_user", "user_id": user_id, "data": data}


@mcp_server.tool()
async def ad_get_user_roles(user_id: str) -> dict:
    """
    Get Azure AD user's assigned directory roles.

    Args:
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with list of assigned roles
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.get_user_roles_smart(user_id)
        return {"success": True, "action": "get_user_roles", "user_id": user_id, "data": data}


@mcp_server.tool()
async def ad_get_user_groups(user_id: str, transitive: bool = False) -> dict:
    """
    Get user's Azure Active Directory group memberships.

    Args:
        user_id: User identifier (GUID, email, or display name)
        transitive: Include transitive group memberships

    Returns:
        Dictionary with list of groups
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.get_user_groups_smart(user_id, transitive=transitive)
        return {"success": True, "action": "get_user_groups", "user_id": user_id, "transitive": transitive,
                "data": data}


@mcp_server.tool()
async def ad_list_roles() -> dict:
    """
    List all Azure Active Directory roles.

    Returns:
        Dictionary with list of all directory roles
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.list_roles()
        return {"success": True, "action": "list_roles", "data": data}


@mcp_server.tool()
async def ad_add_user_to_role(role_id: str, user_id: str) -> dict:
    """
    Assign Azure Active Directory role to user.

    Args:
        role_id: Role ID (must be actual GUID)
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with role assignment confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.add_user_to_role_smart(user_id, role_id)
        return {"success": True, "action": "add_to_role", "role_id": role_id, "user_id": user_id, "data": data}


@mcp_server.tool()
async def ad_remove_user_from_role(role_id: str, user_id: str) -> dict:
    """
    Remove Azure Active Directory role from user.

    Args:
        role_id: Role ID (must be actual GUID)
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with role removal confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.remove_user_from_role_smart(user_id, role_id)
        return {"success": True, "action": "remove_from_role", "role_id": role_id, "user_id": user_id, "data": data}


@mcp_server.tool()
async def ad_list_groups(
        security_only: bool = False,
        unified_only: bool = False,
        select: str = "id,displayName,mailNickname,mail,securityEnabled,groupTypes"
) -> dict:
    """
    List all Azure Active Directory groups.

    Args:
        security_only: List only security groups
        unified_only: List only unified groups
        select: Fields to select

    Returns:
        Dictionary with list of groups
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.list_groups(security_only=security_only, unified_only=unified_only, select=select)
        return {"success": True, "action": "list_groups", "data": data}


@mcp_server.tool()
async def ad_create_group(
        display_name: str,
        mail_nickname: str,
        description: str = None,
        group_type: str = "security",
        visibility: str = None,
        owners: list = None,
        members: list = None
) -> dict:
    """
    Create new Azure Active Directory group.

    Args:
        display_name: Group display name
        mail_nickname: Group mail nickname
        description: Group description (optional)
        group_type: Type of group (security or unified)
        visibility: Group visibility (optional)
        owners: List of owner user IDs (optional)
        members: List of member user IDs (optional)

    Returns:
        Dictionary with created group details
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        params = {
            "display_name": display_name,
            "mail_nickname": mail_nickname,
            "description": description,
            "group_type": group_type,
            "visibility": visibility,
            "owners": owners,
            "members": members
        }
        data = await ad.create_group(**{k: v for k, v in params.items() if v is not None})
        return {"success": True, "action": "create_group", "data": data}


@mcp_server.tool()
async def ad_add_group_member(group_id: str, user_id: str) -> dict:
    """
    Add user to Azure Active Directory group.

    Args:
        group_id: Group ID (must be actual GUID)
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with membership confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        user_resolved_id = await ad.resolve_user(user_id)
        token = await ad.get_access_token()
        body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{user_resolved_id}"}
        data = await ad.graph_api_request("POST", f"groups/{group_id}/members/$ref", token, data=body)
        group = await ad.get_user_groups(user_resolved_id)
        return {"success": True, "action": "add_group_member", "group_id": group_id, "user_id": user_id, "data": data,
                "group": group}


@mcp_server.tool()
async def ad_remove_group_member(group_id: str, user_id: str) -> dict:
    """
    Remove user from Azure Active Directory group.

    Args:
        group_id: Group ID (must be actual GUID)
        user_id: User identifier (GUID, email, or display name)

    Returns:
        Dictionary with removal confirmation
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        user_resolved_id = await ad.resolve_user(user_id)
        token = await ad.get_access_token()
        endpoint = f"groups/{group_id}/members/{user_resolved_id}/$ref"
        data = await ad.graph_api_request("DELETE", endpoint, token)
        return {"success": True, "action": "remove_group_member", "group_id": group_id, "user_id": user_id,
                "data": data}


@mcp_server.tool()
async def ad_get_group_members(group_id: str) -> dict:
    """
    Get Azure Active Directory group member list.

    Args:
        group_id: Group ID (must be actual GUID)

    Returns:
        Dictionary with list of group members
    """
    async with FastActiveDirectory(max_concurrent=20) as ad:
        data = await ad.get_group_members(group_id)
        return {"success": True, "action": "get_group_members", "group_id": group_id, "data": data}


# ++++++++++++++++++++++++++++++++
# FASTMCP TOOL DEFINITIONS END
# ++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++
# FASTAPI BRIDGE APPLICATION
# ++++++++++++++++++++++++++++++++

app = FastAPI(title="MCP Bridge Server", description="Bridge between MCPO/OpenWebUI and FastMCP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global in-memory client reference
_mcp_client = None


async def get_mcp_client():
    """Get in-memory FastMCP client - creates on first use"""
    global _mcp_client
    if _mcp_client is None:
        # Correct: Pass FastMCP server instance directly to Client
        _mcp_client = Client(mcp_server)
        await _mcp_client.__aenter__()
        if DEBUG:
            print("[CLIENT] In-memory FastMCP client initialized")
    return _mcp_client


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    if DEBUG:
        print("[STARTUP] Initializing FastMCP bridge...")
        print(f"[STARTUP] Registered tools: {list(mcp_server._tool_manager._tools.keys())}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.__aexit__(None, None, None)
        if DEBUG:
            print("[SHUTDOWN] FastMCP client closed")


# Dependency to get FastActiveDirectory instance
async def get_ad():
    """Dependency that provides FastActiveDirectory instance"""
    async with FastActiveDirectory(max_concurrent=20) as ad:
        yield ad


# ++++++++++++++++++++++++++++++++
# MCP PROTOCOL BRIDGE ENDPOINT (FOR MCPO/OPENWEBUI)
# ++++++++++++++++++++++++++++++++

@app.post("/")
async def mcp_streamable_http_endpoint(request: Request):
    """Streamable HTTP MCP protocol endpoint - bridges to FastMCP"""
    try:
        body = await request.json()
        method = body.get("method")
        request_id = body.get("id")

        if DEBUG:
            print(f"[BRIDGE] Received method: {method}")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "MCP Server with LLM SQL Generation",
                        "version": "1.0.0"
                    }
                }
            }
            return response

        elif method == "notifications/initialized":
            if DEBUG:
                print("[BRIDGE] Connection established")
            return Response(status_code=204)


        elif method == "tools/list":

            # Get tools from FastMCP

            client = await get_mcp_client()

            tools_result = await client.list_tools()

            if DEBUG:
                print(f"[BRIDGE] tools_result type: {type(tools_result)}")

                print(f"[BRIDGE] tools_result: {tools_result}")

            # Handle both list and object with .tools

            tools = tools_result if isinstance(tools_result, list) else tools_result.tools

            tools_list = []

            for tool in tools:
                tools_list.append({

                    "name": tool.name,

                    "description": tool.description or "",

                    "inputSchema": tool.inputSchema

                })

            response = {

                "jsonrpc": "2.0",

                "id": request_id,

                "result": {"tools": tools_list}

            }

            if DEBUG:
                print(f"[BRIDGE] Returning {len(tools_list)} tools")

            return response

        elif method == "tools/call":
            # Forward tool call to FastMCP
            params = body.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if DEBUG:
                print(f"[BRIDGE] Calling tool: {tool_name} with args: {arguments}")

            try:
                client = await get_mcp_client()
                result = await client.call_tool(tool_name, arguments)

                # Extract result from FastMCP response
                if hasattr(result, 'content') and result.content:
                    # FastMCP returns McpToolResponse with content list
                    result_text = result.content[0].text if result.content else str(result)
                else:
                    result_text = str(result)

                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}]
                    }
                }

                if DEBUG:
                    print(f"[BRIDGE] Tool call successful")

                return response

            except McpError as e:
                if DEBUG:
                    print(f"[BRIDGE] MCP Error: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": str(e)}
                }
            except Exception as e:
                if DEBUG:
                    print(f"[BRIDGE] Execution error: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": str(e)}
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

    except Exception as e:
        if DEBUG:
            print(f"[BRIDGE] Error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {"code": -32000, "message": str(e)}
        }


# ++++++++++++++++++++++++++++++++
# REST API ENDPOINTS START
# ++++++++++++++++++++++++++++++++

@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name with time-based salutation"""
    try:
        name = request.name if request.name and request.name.strip() else ""
        result = await greet(name)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# USER MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ad/users")
async def list_users_endpoint(ad: FastActiveDirectory = Depends(get_ad)):
    """List all users in the directory"""
    try:
        if DEBUG:
            print("[AD_USERS] Listing all users")
        data = await ad.list_users()
        return {"success": True, "action": "list_users", "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "list_users", "error": str(e)}


@app.post("/ad/users")
async def create_user_endpoint(request: CreateUserRequest, ad: FastActiveDirectory = Depends(get_ad)):
    """Create a new user in the directory"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Creating user")
        user_payload = request.user
        if "displayName" in user_payload:
            clean_name = user_payload["displayName"].replace(" ", "").lower()
            user_payload["userPrincipalName"] = f"{clean_name}@lovenoreusgmail.onmicrosoft.com"
            if "mailNickname" not in user_payload:
                user_payload["mailNickname"] = clean_name
        data = await ad.create_user(user_payload)
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
async def update_user_endpoint(user_id: str, request: UserUpdates, ad: FastActiveDirectory = Depends(get_ad)):
    """Update an existing user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Updating user: {user_id}")
        data = await ad.update_user_smart(user_id, request.updates)
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
async def delete_user_endpoint(user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Delete a user from the directory"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Deleting user: {user_id}")
        data = await ad.delete_user_smart(user_id)
        return {"success": True, "action": "delete_user", "user_id": user_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "delete_user", "error": str(e)}


@app.get("/ad/users/{user_id}/roles")
async def get_user_roles_endpoint(user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Get roles assigned to a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting roles for user: {user_id}")
        data = await ad.get_user_roles_smart(user_id)
        return {"success": True, "action": "get_user_roles", "user_id": user_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_roles", "error": str(e)}


@app.get("/ad/users/{user_id}/groups")
async def get_user_groups_endpoint(user_id: str, transitive: bool = False, ad: FastActiveDirectory = Depends(get_ad)):
    """Get groups for a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting groups for user: {user_id}, transitive: {transitive}")
        data = await ad.get_user_groups_smart(user_id, transitive=transitive)
        return {"success": True, "action": "get_user_groups", "user_id": user_id, "transitive": transitive,
                "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_groups", "error": str(e)}


@app.get("/ad/users/{user_id}/owned-groups")
async def get_user_owned_groups_endpoint(user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Get groups owned by a specific user"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Getting owned groups for user: {user_id}")
        user_resolved_id = await ad.resolve_user(user_id)
        data = await ad.get_user_owned_groups(user_resolved_id)
        return {"success": True, "action": "get_user_owned_groups", "user_id": user_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "get_user_owned_groups", "error": str(e)}


@app.get("/ad/users-with-groups")
async def list_users_with_groups_endpoint(
        include_transitive: bool = False,
        include_owned: bool = True,
        select: str = "id,displayName,userPrincipalName",
        ad: FastActiveDirectory = Depends(get_ad)
):
    """List users with their group information"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Listing users with groups - transitive: {include_transitive}, owned: {include_owned}")
        data = await ad.list_users_with_groups(
            include_transitive=include_transitive,
            include_owned=include_owned,
            select=select
        )
        return {"success": True, "action": "list_users_with_groups", "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Error: {e}")
        return {"success": False, "action": "list_users_with_groups", "error": str(e)}


@app.post("/ad/users/batch/groups")
async def batch_get_user_groups_endpoint(
        request: BatchUserIdentifiersRequest,
        transitive: bool = False,
        ad: FastActiveDirectory = Depends(get_ad)
):
    """Get groups for MULTIPLE users concurrently"""
    try:
        if DEBUG:
            print(f"[AD_USERS] Batch getting groups for {len(request.identifiers)} users")
        import time
        start = time.time()
        results = await ad.batch_get_user_groups(request.identifiers, transitive=transitive)
        elapsed = time.time() - start

        formatted_results = []
        for identifier, groups in zip(request.identifiers, results):
            if isinstance(groups, Exception):
                formatted_results.append({
                    "identifier": identifier,
                    "success": False,
                    "error": str(groups)
                })
            else:
                formatted_results.append({
                    "identifier": identifier,
                    "success": True,
                    "groups": groups,
                    "count": len(groups)
                })

        return {
            "success": True,
            "action": "batch_get_user_groups",
            "total_users": len(request.identifiers),
            "elapsed_seconds": round(elapsed, 2),
            "users_per_second": round(len(request.identifiers) / elapsed, 1),
            "transitive": transitive,
            "results": formatted_results
        }
    except Exception as e:
        if DEBUG:
            print(f"[AD_USERS] Batch Error: {e}")
        return {"success": False, "action": "batch_get_user_groups", "error": str(e)}


# ======================================
# ROLE MANAGEMENT ENDPOINTS
# ======================================

@app.get("/ad/roles")
async def list_roles_endpoint(ad: FastActiveDirectory = Depends(get_ad)):
    """List all directory roles"""
    try:
        if DEBUG:
            print("[AD_ROLES] Listing all roles")
        data = await ad.list_roles()
        return {"success": True, "action": "list_roles", "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "list_roles", "error": str(e)}


@app.post("/ad/roles/{role_id}/members")
async def add_user_to_role_endpoint(role_id: str, request: RoleAddMember, ad: FastActiveDirectory = Depends(get_ad)):
    """Add a user to a role"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Adding user {request.user_id} to role {role_id}")
        data = await ad.add_user_to_role_smart(request.user_id, role_id)
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
async def remove_user_from_role_endpoint(role_id: str, user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Remove a user from a role"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Removing user {user_id} from role {role_id}")
        data = await ad.remove_user_from_role_smart(user_id, role_id)
        return {"success": True, "action": "remove_from_role", "role_id": role_id, "user_id": user_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_ROLES] Error: {e}")
        return {"success": False, "action": "remove_from_role", "error": str(e)}


@app.post("/ad/roles/instantiate")
async def instantiate_role_endpoint(request: RoleInstantiation, ad: FastActiveDirectory = Depends(get_ad)):
    """Instantiate a directory role from template"""
    try:
        if DEBUG:
            print(f"[AD_ROLES] Instantiating role from template: {request.roleTemplateId}")
        data = await ad.instantiate_directory_role(request.roleTemplateId)
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
        select: str = "id,displayName,mailNickname,mail,securityEnabled,groupTypes",
        ad: FastActiveDirectory = Depends(get_ad)
):
    """List all groups in the directory"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Listing groups - security_only: {security_only}, unified_only: {unified_only}")
        data = await ad.list_groups(
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
async def create_group_endpoint(request: CreateGroupRequest, ad: FastActiveDirectory = Depends(get_ad)):
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
        data = await ad.create_group(**{k: v for k, v in params.items() if v is not None})
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
async def get_group_members_endpoint(group_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Get members of a specific group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Getting members for group: {group_id}")
        data = await ad.get_group_members(group_id)
        return {"success": True, "action": "get_group_members", "group_id": group_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "get_group_members", "error": str(e)}


@app.get("/ad/groups/{group_id}/owners")
async def get_group_owners_endpoint(group_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Get owners of a specific group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Getting owners for group: {group_id}")
        data = await ad.get_group_owners(group_id)
        return {"success": True, "action": "get_group_owners", "group_id": group_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "get_group_owners", "error": str(e)}


@app.post("/ad/groups/{group_id}/members")
async def add_group_member_endpoint(group_id: str, request: GroupMemberRequest,
                                    ad: FastActiveDirectory = Depends(get_ad)):
    """Add a user to a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Adding user {request.user_id} to group {group_id}")
        user_resolved_id = await ad.resolve_user(request.user_id)
        token = await ad.get_access_token()
        body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{user_resolved_id}"}
        data = await ad.graph_api_request("POST", f"groups/{group_id}/members/$ref", token, data=body)
        group = await ad.get_user_groups(user_resolved_id)
        return {"success": True, "action": "add_group_member", "group_id": group_id, "user_id": request.user_id,
                "data": data, "group": group}
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "add_group_member", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "add_group_member", "error": str(e)}


@app.delete("/ad/groups/{group_id}/members/{user_id}")
async def remove_group_member_endpoint(group_id: str, user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Remove a user from a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Removing user {user_id} from group {group_id}")
        user_resolved_id = await ad.resolve_user(user_id)
        token = await ad.get_access_token()
        endpoint = f"groups/{group_id}/members/{user_resolved_id}/$ref"
        data = await ad.graph_api_request("DELETE", endpoint, token)
        return {"success": True, "action": "remove_group_member", "group_id": group_id, "user_id": user_id,
                "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "remove_group_member", "error": str(e)}


@app.post("/ad/groups/{group_id}/owners")
async def add_group_owner_endpoint(group_id: str, request: GroupOwnerRequest,
                                   ad: FastActiveDirectory = Depends(get_ad)):
    """Add an owner to a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Adding owner {request.user_id} to group {group_id}")
        user_resolved_id = await ad.resolve_user(request.user_id)
        token = await ad.get_access_token()
        body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{user_resolved_id}"}
        data = await ad.graph_api_request("POST", f"groups/{group_id}/owners/$ref", token, data=body)
        return {"success": True, "action": "add_group_owner", "group_id": group_id, "user_id": request.user_id,
                "data": data}
    except ValidationError as ve:
        if DEBUG:
            print(f"[AD_GROUPS] Validation Error: {ve}")
        return {"success": False, "action": "add_group_owner", "error": f"Input validation failed: {str(ve)}"}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "add_group_owner", "error": str(e)}


@app.delete("/ad/groups/{group_id}/owners/{user_id}")
async def remove_group_owner_endpoint(group_id: str, user_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Remove an owner from a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Removing owner {user_id} from group {group_id}")
        user_resolved_id = await ad.resolve_user(user_id)
        token = await ad.get_access_token()
        endpoint = f"groups/{group_id}/owners/{user_resolved_id}/$ref"
        data = await ad.graph_api_request("DELETE", endpoint, token)
        return {"success": True, "action": "remove_group_owner", "group_id": group_id, "user_id": user_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "remove_group_owner", "error": str(e)}


@app.patch("/ad/groups/{group_id}")
async def update_group_endpoint(group_id: str, request: GroupUpdates, ad: FastActiveDirectory = Depends(get_ad)):
    """Update an existing group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Updating group: {group_id}")
        token = await ad.get_access_token()
        data = await ad.graph_api_request("PATCH", f"groups/{group_id}", token, data=request.updates)
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
async def delete_group_endpoint(group_id: str, ad: FastActiveDirectory = Depends(get_ad)):
    """Delete a group"""
    try:
        if DEBUG:
            print(f"[AD_GROUPS] Deleting group: {group_id}")
        token = await ad.get_access_token()
        data = await ad.graph_api_request("DELETE", f"groups/{group_id}", token)
        return {"success": True, "action": "delete_group", "group_id": group_id, "data": data}
    except Exception as e:
        if DEBUG:
            print(f"[AD_GROUPS] Error: {e}")
        return {"success": False, "action": "delete_group", "error": str(e)}


# ++++++++++++++++++++++++++++++++
# REST API ENDPOINTS END
# ++++++++++++++++++++++++++++++++


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "MCP Bridge Server",
            "timestamp": datetime.now().isoformat(),
            "fastmcp_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/info")
async def server_info():
    """Server information endpoint"""
    try:
        client = await get_mcp_client()
        tools_result = await client.list_tools()

        return {
            "service": "MCP Bridge Server",
            "version": "1.0.0",
            "description": "FastAPI bridge to FastMCP server for MCPO/OpenWebUI compatibility",
            "protocols": ["REST API", "MCP (Model Context Protocol)"],
            "mcp_endpoints": {
                "streamable_http": "/",
                "tools_list": "POST / with method='tools/list'",
                "tools_call": "POST / with method='tools/call'"
            },
            "rest_endpoints": [
                "/greet",
                "/ad/users",
                "/ad/roles",
                "/ad/groups",
                "/health",
                "/info"
            ],
            "features": [
                "FastMCP Integration",
                "In-Memory Transport",
                "Active Directory Operations",
                "MCP Protocol Support",
                "Automatic Schema Generation",
                "Type Validation"
            ],
            "tools_count": len(tools_result.tools),
            "tools": [t.name for t in tools_result.tools],
            "fastmcp_enabled": True,
            "mcp_compatible": True
        }
    except Exception as e:
        if DEBUG:
            print(f"[INFO] Error: {e}")
        return {
            "service": "MCP Bridge Server",
            "version": "1.0.0",
            "error": str(e)
        }


@app.post("/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint to see raw requests"""
    body = await request.json()

    print(f"[DEBUG] Raw request: {body}")

    return {"received": body}


if __name__ == "__main__":
    print("=" * 60)
    print("Starting MCP Bridge Server on port 8009...")
    print("=" * 60)
    print(f"FastMCP Server: {mcp_server.name}")
    print(f"Registered Tools: {len(mcp_server._tool_manager._tools)}")
    for tool_name in mcp_server._tool_manager._tools.keys():
        print(f"  - {tool_name}")
    print("=" * 60)
    print("Bridge Architecture:")
    print("  MCPO/OpenWebUI → FastAPI (port 8009) → FastMCP (in-memory)")
    print("=" * 60)

    if DEBUG:
        print("[DEBUG] Debug mode enabled - detailed logging active")

    uvicorn.run(app, host="0.0.0.0", port=8009)
