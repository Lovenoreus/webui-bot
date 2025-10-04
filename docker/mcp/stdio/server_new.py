# -------------------- Built-in Libraries --------------------
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Any, Literal

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

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


load_dotenv(find_dotenv())

# Debug flag
DEBUG = True


def format_mcp_response(result: dict, tool_name: str) -> MCPToolCallResponse:
    """
    Format Active Directory results for MCP tool responses with proper error handling
    """
    # If result has success field (our enhanced format)
    if isinstance(result, dict) and "success" in result:
        if result["success"]:
            # Success case - return the result as JSON
            content_text = json.dumps(result, indent=2)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=content_text)],
                isError=False
            )
        else:
            # Check if this is a friendly message or technical error
            if "message" in result and "error" not in result:
                # Friendly message - treat as informational, not an error
                content_text = json.dumps(result, indent=2)
                return MCPToolCallResponse(
                    content=[MCPContent(type="text", text=content_text)],
                    isError=False  # Don't mark as error - it's a helpful message
                )
            else:
                # Technical error case
                content_text = json.dumps(result, indent=2)
                return MCPToolCallResponse(
                    content=[MCPContent(type="text", text=content_text)],
                    isError=True
                )
    
    # Legacy format (no success field) - assume success if no exception
    else:
        content_text = json.dumps(result, indent=2)
        return MCPToolCallResponse(
            content=[MCPContent(type="text", text=content_text)],
            isError=False
        )

app = FastAPI(title="MCP Server API", description="Standalone MCP Tools Server with LLM SQL Generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Dependency to get FastActiveDirectory instance
async def get_ad():
    """Dependency that provides FastActiveDirectory instance"""
    async with FastActiveDirectory(max_concurrent=20) as ad:
        yield ad


# Request models
class GreetRequest(BaseModel):
    name: Optional[str]


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
        response = f"[RESPONSE]: {time_greeting} {name}! I'm your Cosmic hospital assistant. I can help with policies, user management, database queries, weather information, and more. What can I do for you?"
    else:
        response = f"[RESPONSE]: {time_greeting}! I'm your Cosmic hospital assistant. I can help with policies, user management, database queries, weather information, and more. How can I assist you today?"

    return f"{response}\n\n[Success]"


@app.post("/greet")
async def greet_endpoint(request: GreetRequest):
    """Greet a user by name with time-based salutation"""
    try:
        name = request.name if request.name and request.name.strip() else None
        message = await greet(name)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ++++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY ENDPOINTS START
# ++++++++++++++++++++++++++++++++

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


# ++++++++++++++++++++++++++++++
# ACTIVE DIRECTORY ENDPOINTS END
# ++++++++++++++++++++++++++++++


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "MCP Server",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "greet": "/greet",
                "health": "/health"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list():
    """MCP Protocol: List available tools"""
    tools = [
        # CONVERSATIONAL
        MCPTool(
            name="greet",
            description="TRIGGER: hello, hi, good morning, good afternoon, good evening, introduce yourself, who are you, start conversation | ACTION: Welcome user with time-based greeting | RETURNS: Personalized greeting with hospital assistant capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's name", "default": ""}
                },
                "required": ["name"]
            }
        ),

        # AZURE ACTIVE DIRECTORY - USER MANAGEMENT
        MCPTool(
            name="ad_list_users",
            description="TRIGGER: list AD users, active directory users, show all users, Azure AD users, directory users, AD accounts | ACTION: List all Azure Active Directory user accounts | RETURNS: Complete AD user directory with IDs, names, emails from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        MCPTool(
            name="ad_create_user",
            description="TRIGGER: create AD user, add Azure AD user, new AD account, register AD user, add employee to directory | ACTION: Create new Azure Active Directory user account with automatic password generation | RETURNS: Created AD user details with temporary password and login instructions",
            inputSchema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "description": "Azure AD user payload - only displayName is required, other fields are auto-generated",
                        "properties": {
                            "displayName": {"type": "string", "description": "User's full display name (REQUIRED)"},
                            "mailNickname": {"type": "string", "description": "Mail nickname (auto-generated from displayName if not provided)"},
                            "userPrincipalName": {"type": "string", "description": "User login email (auto-generated as displayname@domain if not provided)"},
                            "passwordProfile": {
                                "type": "object",
                                "description": "Password settings (secure password auto-generated if not provided)",
                                "properties": {
                                    "password": {"type": ["string", "null"], "description": "Custom password (secure password auto-generated if null or not specified)"},
                                    "forceChangePasswordNextSignIn": {"type": "boolean", "description": "Force password change on first login (defaults to true)"}
                                }
                            },
                            "accountEnabled": {"type": "boolean", "description": "Whether account is enabled (defaults to true)"}
                        },
                        "required": ["displayName"]
                    }
                },
                "required": ["user"]
            }
        ),

        MCPTool(
            name="ad_update_user",
            description="TRIGGER: update AD user, modify Azure AD user, change AD user details, edit directory user, AD user updates | ACTION: Update existing Azure Active Directory user properties | INSTRUCTION: Accepts flexible user identification - use name ('John Marks'), email ('john@company.com'), or GUID. System auto-resolves to actual Azure AD user ID | RETURNS: Updated AD user information from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address (userPrincipalName like 'john@company.com'), or display name (like 'John Marks'). Smart resolution automatically finds the correct user."
                    },
                    "updates": {"type": "object", "description": "AD user properties to update"}
                },
                "required": ["user_identifier", "updates"]
            }
        ),

        MCPTool(
            name="ad_delete_user",
            description="TRIGGER: delete AD user, remove Azure AD user, deactivate directory user, remove AD account | ACTION: Delete user account from Azure Active Directory | INSTRUCTION: Accepts flexible user identification - use name ('John Marks'), email ('john@company.com'), or GUID. System auto-resolves to actual Azure AD user ID | RETURNS: AD account deletion confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address (userPrincipalName like 'john@company.com'), or display name (like 'John Marks'). Smart resolution automatically finds the correct user."
                    }
                },
                "required": ["user_identifier"]
            }
        ),

        MCPTool(
            name="ad_get_user_roles",
            description="TRIGGER: AD user roles, Azure AD user permissions, directory user roles, what AD roles does user have, check AD access | ACTION: Get Azure AD user's assigned directory roles | INSTRUCTION: Accepts flexible user identification - use name, email, or GUID. System auto-resolves | RETURNS: List of AD roles assigned to specific user from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    }
                },
                "required": ["user_identifier"]
            }
        ),

        MCPTool(
            name="ad_get_user_groups",
            description="TRIGGER: AD user groups, Azure AD user memberships, directory user groups, what AD groups is user in, check AD group membership | ACTION: Get user's Azure Active Directory group memberships | INSTRUCTION: Accepts flexible user identification - use name, email, or GUID. System auto-resolves | RETURNS: List of AD groups user belongs to in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    },
                    "transitive": {"type": "boolean", "description": "Include transitive AD group memberships", "default": False}
                },
                "required": ["user_identifier", "transitive"]
            }
        ),

        MCPTool(
            name="ad_get_user_full_profile",
            description="TRIGGER: user profile, full AD user info, complete user details, user information, comprehensive user data | ACTION: Get complete user profile with groups, roles, and owned groups in single call | INSTRUCTION: Accepts flexible user identification - use name, email, or GUID | RETURNS: Comprehensive user data including basic info, groups, roles, and owned groups",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    }
                },
                "required": ["user_identifier"]
            }
        ),

        MCPTool(
            name="ad_search_users",
            description="TRIGGER: find user, search AD users, lookup user, search for user, find employee | ACTION: Fuzzy search across display name, email, and userPrincipalName | RETURNS: List of matching users with details",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query - partial name, email, etc."},
                    "limit": {"type": "integer", "description": "Maximum results to return", "default": 10}
                },
                "required": ["query", "limit"]
            }
        ),

        # AZURE ACTIVE DIRECTORY - ROLE MANAGEMENT
        MCPTool(
            name="ad_list_roles",
            description="TRIGGER: list AD roles, show Azure AD roles, available directory roles, all AD roles, Azure role directory | ACTION: List all Azure Active Directory roles | RETURNS: Complete list of AD directory roles from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        MCPTool(
            name="ad_add_user_to_role",
            description="TRIGGER: assign AD role, add user to AD role, give Azure AD role, grant directory role, AD role assignment | ACTION: Assign Azure Active Directory role to user | INSTRUCTION: Accepts flexible identifiers for BOTH user and role - use names, emails, or GUIDs. System auto-resolves both | RETURNS: AD role assignment confirmation in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    },
                    "role_identifier": {
                        "type": "string",
                        "description": "Flexible role identifier - accepts Azure AD role GUID or role display name (like 'Global Administrator'). Smart resolution automatically finds the correct role."
                    }
                },
                "required": ["user_identifier", "role_identifier"]
            }
        ),

        MCPTool(
            name="ad_remove_user_from_role",
            description="TRIGGER: remove AD role, unassign Azure AD role, revoke directory role, take away AD role | ACTION: Remove Azure Active Directory role from user | INSTRUCTION: Accepts flexible identifiers for BOTH user and role - use names, emails, or GUIDs. System auto-resolves both | RETURNS: AD role removal confirmation from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    },
                    "role_identifier": {
                        "type": "string",
                        "description": "Flexible role identifier - accepts Azure AD role GUID or role display name (like 'Global Administrator'). Smart resolution automatically finds the correct role."
                    }
                },
                "required": ["user_identifier", "role_identifier", ]
            }
        ),

        MCPTool(
            name="ad_batch_add_users_to_role",
            description="TRIGGER: assign role to multiple users, bulk role assignment, add many users to role | ACTION: Assign Azure AD role to multiple users concurrently | INSTRUCTION: Accepts flexible identifiers for users and role | RETURNS: Batch operation results with success/failure counts",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers (names, emails, or GUIDs)"
                    },
                    "role_identifier": {
                        "type": "string",
                        "description": "Role identifier (GUID or display name like 'User Administrator')"
                    },
                    "ignore_errors": {"type": "boolean", "description": "Continue on errors", "default": True}
                },
                "required": ["user_identifiers", "role_identifier", "ignore_errors"]
            }
        ),

        MCPTool(
            name="ad_batch_remove_users_from_role",
            description="TRIGGER: remove role from multiple users, bulk role removal, revoke role from many users | ACTION: Remove Azure AD role from multiple users concurrently | RETURNS: Batch operation results with success/failure counts",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers (names, emails, or GUIDs)"
                    },
                    "role_identifier": {
                        "type": "string",
                        "description": "Role identifier (GUID or display name)"
                    },
                    "ignore_errors": {"type": "boolean", "description": "Continue on errors", "default": True}
                },
                "required": ["user_identifiers", "role_identifier", "ignore_errors"]
            }
        ),

        # AZURE ACTIVE DIRECTORY - GROUP MANAGEMENT
        MCPTool(
            name="ad_list_groups",
            description="TRIGGER: list AD groups, show Azure AD groups, all directory groups, AD group directory, available Azure groups | ACTION: List all Azure Active Directory groups | RETURNS: Complete list of AD security and distribution groups from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "security_only": {"type": "boolean", "description": "List only AD security groups", "default": False},
                    "unified_only": {"type": "boolean", "description": "List only AD unified groups", "default": False},
                    "select": {"type": "string", "description": "AD fields to select", "default": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
                },
                "required": ["select", "security_only", "unified_only"]
            }
        ),

        MCPTool(
            name="ad_create_group",
            description="TRIGGER: create AD group, add new Azure AD group, new directory group, make AD group | ACTION: Create new Azure Active Directory group | RETURNS: Created AD group details with ID from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "display_name": {"type": "string", "description": "AD group display name"},
                    "mail_nickname": {"type": "string", "description": "AD group mail nickname"},
                    "description": {"type": ["string", "null"], "description": "AD group description (optional)"},
                    "group_type": {"type": "string", "enum": ["security", "m365", "dynamic-security", "dynamic-m365"], "description": "Azure AD group type", "default": "security"},
                    "visibility": {"type": ["string", "null"], "enum": ["Private", "Public"], "description": "AD group visibility (for M365 groups, optional)"},
                    "membership_rule": {"type": ["string", "null"], "description": "Dynamic membership rule (for dynamic groups, optional)"},
                    "owners": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List of owner identifiers (names, emails, or GUIDs, optional)"},
                    "members": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List of member identifiers (names, emails, or GUIDs, optional)"}
                },
                "required": ["display_name", "mail_nickname"]
            }
        ),

        MCPTool(
            name="ad_add_group_member",
            description="TRIGGER: add to AD group, add member to Azure AD group, add user to directory group, join AD group | ACTION: Add user to Azure Active Directory group | INSTRUCTION: Accepts flexible identifiers for BOTH user and group - use names, emails, or GUIDs. System auto-resolves both | RETURNS: AD group membership confirmation in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name. Smart resolution automatically finds the correct group."
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_remove_group_member",
            description="TRIGGER: remove from AD group, remove member from Azure AD group, leave directory group, kick from AD group | ACTION: Remove user from Azure Active Directory group | INSTRUCTION: Accepts flexible identifiers for BOTH user and group - use names, emails, or GUIDs. System auto-resolves both | RETURNS: AD group removal confirmation from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name. Smart resolution automatically finds the correct user."
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name. Smart resolution automatically finds the correct group."
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_get_group_members",
            description="TRIGGER: AD group members, Azure AD group members, who is in directory group, show AD group members, list AD group members | ACTION: Get Azure Active Directory group member list | INSTRUCTION: Accepts flexible group identification - use name, email, mail nickname, or GUID | RETURNS: All members of specified AD group from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name. Smart resolution automatically finds the correct group."
                    }
                },
                "required": ["group_identifier"]
            }
        ),

        MCPTool(
            name="ad_get_group_owners",
            description="TRIGGER: AD group owners, Azure AD group owners, who owns group, group administrators | ACTION: Get Azure Active Directory group owner list | INSTRUCTION: Accepts flexible group identification - use name, email, mail nickname, or GUID | RETURNS: All owners of specified AD group from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name. Smart resolution automatically finds the correct group."
                    }
                },
                "required": ["group_identifier"]
            }
        ),

        MCPTool(
            name="ad_add_group_owner",
            description="TRIGGER: add group owner, make group admin, add group administrator | ACTION: Add owner to Azure Active Directory group | INSTRUCTION: Accepts flexible identifiers for both user and group | RETURNS: Group ownership assignment confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name."
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name."
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_remove_group_owner",
            description="TRIGGER: remove group owner, remove group admin, revoke group administrator | ACTION: Remove owner from Azure Active Directory group | INSTRUCTION: Accepts flexible identifiers for both user and group | RETURNS: Group ownership removal confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier - accepts Azure AD GUID, email address, or display name."
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name."
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_update_group",
            description="TRIGGER: update AD group, modify group details, change group properties, edit AD group | ACTION: Update Azure Active Directory group properties | INSTRUCTION: Accepts flexible group identification | RETURNS: Updated group information",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name."
                    },
                    "updates": {"type": "object", "description": "Group properties to update"}
                },
                "required": ["group_identifier", "updates"]
            }
        ),

        MCPTool(
            name="ad_delete_group",
            description="TRIGGER: delete AD group, remove group, delete directory group | ACTION: Delete Azure Active Directory group | INSTRUCTION: Accepts flexible group identification | RETURNS: Group deletion confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name."
                    }
                },
                "required": ["group_identifier"]
            }
        ),

        MCPTool(
            name="ad_get_group_full_details",
            description="TRIGGER: group details, full group info, complete group data, group information | ACTION: Get comprehensive group information including members and owners | INSTRUCTION: Accepts flexible group identification | RETURNS: Complete group data with members and owners in single call",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier - accepts Azure AD GUID, group email, mail nickname, or display name."
                    }
                },
                "required": ["group_identifier"]
            }
        ),

        MCPTool(
            name="ad_search_groups",
            description="TRIGGER: find group, search AD groups, lookup group, search for group | ACTION: Fuzzy search across group display name, mail nickname, and email | RETURNS: List of matching groups with details",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query - partial name, email, etc."},
                    "limit": {"type": "integer", "description": "Maximum results to return", "default": 10}
                },
                "required": ["query", "limit"]
            }
        ),

        MCPTool(
            name="ad_batch_add_users_to_group",
            description="TRIGGER: add multiple users to group, bulk add to group, add many members | ACTION: Add multiple users to a group concurrently | INSTRUCTION: Accepts flexible identifiers for users and group | RETURNS: Batch operation results with success/failure counts",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers (names, emails, or GUIDs)"
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Group identifier (GUID, email, mail nickname, or display name)"
                    },
                    "ignore_errors": {"type": "boolean", "description": "Continue on errors", "default": True}
                },
                "required": ["user_identifiers", "group_identifier", "ignore_errors"]
            }
        ),

        MCPTool(
            name="ad_batch_remove_users_from_group",
            description="TRIGGER: remove multiple users from group, bulk remove from group, remove many members | ACTION: Remove multiple users from a group concurrently | RETURNS: Batch operation results with success/failure counts",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers (names, emails, or GUIDs)"
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Group identifier (GUID, email, mail nickname, or display name)"
                    },
                    "ignore_errors": {"type": "boolean", "description": "Continue on errors", "default": True}
                },
                "required": ["user_identifiers", "group_identifier", "ignore_errors"]
            }
        ),

        MCPTool(
            name="ad_check_user_membership",
            description="TRIGGER: is user in group, check group membership, verify membership, user member of group | ACTION: Check if user is member of specific group | INSTRUCTION: Accepts flexible identifiers for both user and group | RETURNS: Boolean membership status",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier"
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier"
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_check_user_ownership",
            description="TRIGGER: is user group owner, check group ownership, verify owner, user owns group | ACTION: Check if user is owner of specific group | INSTRUCTION: Accepts flexible identifiers for both user and group | RETURNS: Boolean ownership status",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifier": {
                        "type": "string",
                        "description": "Flexible user identifier"
                    },
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier"
                    }
                },
                "required": ["user_identifier", "group_identifier"]
            }
        ),

        MCPTool(
            name="ad_sync_group_members",
            description="TRIGGER: sync group members, synchronize group, match group membership, set group members | ACTION: Synchronize group membership to match desired list (adds missing, removes extra) | INSTRUCTION: Accepts flexible identifiers | RETURNS: Summary of changes (added count, removed count, errors)",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_identifier": {
                        "type": "string",
                        "description": "Flexible group identifier"
                    },
                    "desired_user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers that should be in the group"
                    }
                },
                "required": ["group_identifier", "desired_user_identifiers"]
            }
        ),

        MCPTool(
            name="ad_batch_get_user_groups",
            description="TRIGGER: get groups for multiple users, bulk user groups, groups for many users | ACTION: Get group memberships for multiple users concurrently | RETURNS: List of group memberships for each user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user identifiers (names, emails, or GUIDs)"
                    },
                    "transitive": {"type": "boolean", "description": "Include transitive memberships", "default": False}
                },
                "required": ["user_identifiers", "transitive"]
            }
        ),

        MCPTool(
            name="ad_list_users_with_groups",
            description="TRIGGER: all users with their groups, users and groups, complete user group mapping | ACTION: List all users with their group memberships, owned groups | RETURNS: Complete user directory with group information for each user",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_transitive": {"type": "boolean", "description": "Include transitive group memberships", "default": False},
                    "include_owned": {"type": "boolean", "description": "Include groups owned by user", "default": True},
                    "select": {"type": "string", "description": "User fields to select", "default": "id,displayName,userPrincipalName"}
                },
                "required": ["include_transitive", "include_owned", "select"]
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

        if "thread_id" not in arguments:
            arguments["thread_id"] = "default"

        # ==================== CONVERSATIONAL ====================
        if tool_name == "greet":
            raw_name = arguments.get("name") if arguments else None
            clean_name = raw_name if raw_name and raw_name.strip() else None
            greet_request = GreetRequest(name=clean_name)
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        # ==================== USER MANAGEMENT ====================
        elif tool_name == "ad_list_users":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                data = await ad.list_users()
                result = {
                    "success": True,
                    "action": "list_users",
                    "message": f"✅ Found {len(data)} users in the directory",
                    "data": data,
                    "count": len(data)
                }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_create_user":
            import secrets
            import string
            
            user_data = arguments["user"].copy()
            
            # Ensure required fields are present
            if "displayName" not in user_data:
                result = {
                    "success": False,
                    "action": "create_user",
                    "message": "❌ Display name is required to create a user",
                    "error": "Missing required field: displayName"
                }
                return format_mcp_response(result, tool_name)
            
            # Auto-generate userPrincipalName if not provided OR correct domain if wrong domain used
            if "userPrincipalName" not in user_data:
                clean_name = user_data["displayName"].replace(" ", "").lower()
                user_data["userPrincipalName"] = f"{clean_name}@lovenoreusgmail.onmicrosoft.com"
            else:
                # Ensure correct domain is used - override if different domain provided
                current_upn = user_data["userPrincipalName"]
                if "@" in current_upn and not current_upn.endswith("@lovenoreusgmail.onmicrosoft.com"):
                    username_part = current_upn.split("@")[0]
                    user_data["userPrincipalName"] = f"{username_part}@lovenoreusgmail.onmicrosoft.com"
                    # Add a note about domain correction
                    user_data["_domain_corrected"] = True
            
            # Auto-generate mailNickname if not provided
            if "mailNickname" not in user_data:
                clean_name = user_data["displayName"].replace(" ", "").lower()
                user_data["mailNickname"] = clean_name
            
            # Set accountEnabled to True by default if not specified
            if "accountEnabled" not in user_data:
                user_data["accountEnabled"] = True
            
            # Generate secure password if not provided or if password is null/empty
            password_profile = user_data.get("passwordProfile", {})
            password = password_profile.get("password") if password_profile else None
            
            if not password or password is None or str(password).strip() == "" or str(password).lower() == "null":
                # Generate a secure 12-character password
                alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                secure_password = ''.join(secrets.choice(alphabet) for i in range(12))
                
                user_data["passwordProfile"] = {
                    "password": secure_password,
                    "forceChangePasswordNextSignIn": True
                }
            
            try:
                create_request = CreateUserRequest(action="create_user", user=user_data)
                async with FastActiveDirectory(max_concurrent=20) as ad:
                    raw_result = await create_user_endpoint(create_request, ad)
                
                # Check if the operation was successful
                if raw_result.get("success"):
                    # Success - enhance the response with helpful information
                    message = f"✅ Successfully created user '{user_data['displayName']}'"
                    if user_data.get("_domain_corrected"):
                        message += " (domain corrected to organization domain)"
                    
                    result = {
                        "success": True,
                        "action": "create_user",
                        "message": message,
                        "user_data": {
                            "displayName": user_data["displayName"],
                            "userPrincipalName": user_data["userPrincipalName"],
                            "mailNickname": user_data["mailNickname"],
                            "accountEnabled": user_data["accountEnabled"],
                            "temporaryPassword": user_data["passwordProfile"]["password"] if user_data["passwordProfile"]["forceChangePasswordNextSignIn"] else "*** (password set by user)"
                        },
                        "data": raw_result.get("data"),
                        "next_steps": [
                            "The user should sign in and change their password on first login",
                            f"User can sign in at: https://login.microsoftonline.com with email: {user_data['userPrincipalName']}"
                        ]
                    }
                else:
                    # Handle errors from the endpoint
                    error_message = raw_result.get("error", "Unknown error")
                    
                    # Check for specific error conditions
                    if "already exists" in error_message.lower() or "same value for property userPrincipalName already exists" in error_message:
                        result = {
                            "success": False,
                            "action": "create_user",
                            "message": f"❌ A user with email '{user_data.get('userPrincipalName', 'unknown')}' already exists",
                            "suggestion": f"Try creating the user with a different email address, or check if user '{user_data['displayName']}' is already in the system",
                            "user_data": {
                                "attempted_displayName": user_data["displayName"],
                                "attempted_userPrincipalName": user_data["userPrincipalName"]
                            },
                            "error": "User already exists"
                        }
                    elif "password must be specified" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "create_user", 
                            "message": f"❌ Password is required to create user '{user_data['displayName']}'",
                            "suggestion": "The system should have auto-generated a password. Please try again.",
                            "error": "Missing password"
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "create_user",
                            "message": f"❌ Failed to create user '{user_data['displayName']}'",
                            "suggestion": "Please check the user details and try again with different values",
                            "user_data": {
                                "attempted_displayName": user_data["displayName"],
                                "attempted_userPrincipalName": user_data["userPrincipalName"]
                            },
                            "error": error_message
                        }
            except Exception as e:
                error_message = str(e)
                if "already exists" in error_message.lower():
                    result = {
                        "success": False,
                        "action": "create_user",
                        "message": f"❌ A user with email '{user_data.get('userPrincipalName', 'unknown')}' already exists",
                        "error": "User already exists"
                    }
                else:
                    result = {
                        "success": False,
                        "action": "create_user", 
                        "message": f"❌ Failed to create user '{user_data['displayName']}'. {error_message}",
                        "error": error_message
                    }
            
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_update_user":
            user_identifier = arguments["user_identifier"]
            user_updates = UserUpdates(updates=arguments["updates"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve user first to get actual details
                    user_id = await ad.resolve_user(user_identifier)
                    
                    # Perform the update
                    raw_result = await ad.update_user_smart(user_identifier, user_updates.updates)
                    
                    # Success - Graph API returns empty response for successful updates
                    result = {
                        "success": True,
                        "action": "update_user",
                        "message": f"✅ Successfully updated user '{user_identifier}'",
                        "user_identifier": user_identifier,
                        "user_id": user_id,
                        "updated_fields": list(user_updates.updates.keys())
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "update_user",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "update_user",
                            "message": f"❌ Failed to update user '{user_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_delete_user":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve user first to get actual details
                    user_id = await ad.resolve_user(user_identifier)
                    
                    # Perform the delete
                    raw_result = await ad.delete_user_smart(user_identifier)
                    
                    # Success - Graph API returns empty response for successful deletes
                    result = {
                        "success": True,
                        "action": "delete_user",
                        "message": f"✅ Successfully deleted user '{user_identifier}'",
                        "user_identifier": user_identifier,
                        "user_id": user_id,
                        "warning": "This action cannot be undone. The user account has been permanently removed."
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "delete_user",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "delete_user",
                            "message": f"❌ Failed to delete user '{user_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_user_roles":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    user_id = await ad.resolve_user(user_identifier)
                    data = await ad.get_user_roles(user_id)
                    result = {
                        "success": True,
                        "action": "get_user_roles",
                        "message": f"✅ Found {len(data)} roles for user '{user_identifier}'",
                        "user_identifier": user_identifier,
                        "user_id": user_id,
                        "data": data,
                        "count": len(data)
                    }
                except Exception as e:
                    if "No user found" in str(e):
                        result = {
                            "success": False,
                            "action": "get_user_roles",
                            "message": f"Please check the username '{user_identifier}' - I couldn't find anyone with that name. You might want to try their full name or email address.",
                            "user_identifier": user_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "get_user_roles",
                            "message": f"There was an issue retrieving roles for '{user_identifier}'. Please try again.",
                            "user_identifier": user_identifier,
                            "error": str(e)
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_user_groups":
            user_identifier = arguments["user_identifier"]
            transitive = arguments.get("transitive", False)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    user_id = await ad.resolve_user(user_identifier)
                    data = await ad.get_user_groups(user_id, transitive=transitive)
                    result = {
                        "success": True,
                        "action": "get_user_groups",
                        "message": f"✅ Found {len(data)} groups for user '{user_identifier}'",
                        "user_identifier": user_identifier,
                        "user_id": user_id,
                        "transitive": transitive,
                        "data": data,
                        "count": len(data)
                    }
                except Exception as e:
                    if "No user found" in str(e):
                        result = {
                            "success": False,
                            "action": "get_user_groups",
                            "message": f"Please check the username '{user_identifier}' - I couldn't find anyone with that name. You might want to try their full name or email address.",
                            "user_identifier": user_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "get_user_groups",
                            "message": f"There was an issue retrieving groups for '{user_identifier}'. Please try again.",
                            "user_identifier": user_identifier,
                            "error": str(e)
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_user_full_profile":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    user_id = await ad.resolve_user(user_identifier)
                    data = await ad.get_user_full_profile(user_id)
                    result = {
                        "success": True,
                        "action": "get_user_full_profile",
                        "message": f"✅ Retrieved full profile for user '{user_identifier}'",
                        "user_identifier": user_identifier,
                        "user_id": user_id,
                        "data": data
                    }
                except Exception as e:
                    if "No user found" in str(e):
                        result = {
                            "success": False,
                            "action": "get_user_full_profile",
                            "message": f"Please check the username '{user_identifier}' - I couldn't find anyone with that name. You might want to try their full name or email address.",
                            "user_identifier": user_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "get_user_full_profile",
                            "message": f"There was an issue retrieving profile for '{user_identifier}'. Please try again.",
                            "user_identifier": user_identifier,
                            "error": str(e)
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_search_users":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                data = await ad.search_users_fuzzy(query, limit=limit)
                result = {
                    "success": True,
                    "action": "search_users",
                    "message": f"✅ Found {len(data)} users matching '{query}'",
                    "query": query,
                    "limit": limit,
                    "data": data,
                    "count": len(data)
                }
            return format_mcp_response(result, tool_name)

        # ==================== ROLE MANAGEMENT ====================
        elif tool_name == "ad_list_roles":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                data = await ad.list_roles()
                result = {
                    "success": True,
                    "action": "list_roles",
                    "message": f"✅ Found {len(data)} roles in the directory",
                    "data": data,
                    "count": len(data)
                }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_add_user_to_role":
            user_identifier = arguments["user_identifier"]
            role_identifier = arguments["role_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    role_id = await ad.resolve_role(role_identifier)
                    
                    # Perform the add operation
                    raw_result = await ad.add_user_to_role_smart(user_identifier, role_identifier)
                    
                    # Success - Graph API returns empty response for successful adds
                    result = {
                        "success": True,
                        "action": "add_user_to_role",
                        "message": f"✅ Successfully added '{user_identifier}' to role '{role_identifier}'",
                        "user_identifier": user_identifier,
                        "role_identifier": role_identifier,
                        "user_id": user_id,
                        "role_id": role_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_user_to_role",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    elif "No role found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_user_to_role",
                            "message": f"❌ Could not find role '{role_identifier}'. Please check the role name.",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    elif "already exists" in error_message.lower() or "already assigned" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "add_user_to_role",
                            "message": f"ℹ️ User '{user_identifier}' already has role '{role_identifier}'",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "add_user_to_role",
                            "message": f"❌ Failed to add '{user_identifier}' to role '{role_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_remove_user_from_role":
            user_identifier = arguments["user_identifier"]
            role_identifier = arguments["role_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    role_id = await ad.resolve_role(role_identifier)
                    
                    # Perform the remove operation
                    raw_result = await ad.remove_user_from_role_smart(user_identifier, role_identifier)
                    
                    # Success - Graph API returns empty response for successful removes
                    result = {
                        "success": True,
                        "action": "remove_user_from_role",
                        "message": f"✅ Successfully removed '{user_identifier}' from role '{role_identifier}'",
                        "user_identifier": user_identifier,
                        "role_identifier": role_identifier,
                        "user_id": user_id,
                        "role_id": role_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_user_from_role",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    elif "No role found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_user_from_role",
                            "message": f"❌ Could not find role '{role_identifier}'. Please check the role name.",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    elif "does not exist" in error_message.lower() or "not assigned" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "remove_user_from_role",
                            "message": f"ℹ️ User '{user_identifier}' does not have role '{role_identifier}'",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "remove_user_from_role",
                            "message": f"❌ Failed to remove '{user_identifier}' from role '{role_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "role_identifier": role_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_batch_add_users_to_role":
            user_identifiers = arguments["user_identifiers"]
            role_identifier = arguments["role_identifier"]
            ignore_errors = arguments.get("ignore_errors", True)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.batch_add_users_to_role(
                    user_identifiers,
                    role_identifier,
                    ignore_errors=ignore_errors
                )
            # Format results with success/failure counts
            success_count = len([r for r in result if not isinstance(r, Exception)])
            error_count = len([r for r in result if isinstance(r, Exception)])
            formatted_result = {
                "success": error_count == 0,
                "action": "batch_add_users_to_role",
                "message": f"✅ Batch operation completed: {success_count} successful, {error_count} failed" if error_count == 0 else f"⚠️ Batch operation completed with issues: {success_count} successful, {error_count} failed",
                "success_count": success_count,
                "error_count": error_count,
                "total_users": len(user_identifiers),
                "role_identifier": role_identifier,
                "results": [str(r) if isinstance(r, Exception) else "✅ Added successfully" for r in result]
            }
            return format_mcp_response(formatted_result, tool_name)

        elif tool_name == "ad_batch_remove_users_from_role":
            user_identifiers = arguments["user_identifiers"]
            role_identifier = arguments["role_identifier"]
            ignore_errors = arguments.get("ignore_errors", True)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.batch_remove_users_from_role(
                    user_identifiers,
                    role_identifier,
                    ignore_errors=ignore_errors
                )
            # Format results
            success_count = len([r for r in result if not isinstance(r, Exception)])
            error_count = len([r for r in result if isinstance(r, Exception)])
            formatted_result = {
                "success": error_count == 0,
                "action": "batch_remove_users_from_role",
                "message": f"✅ Batch operation completed: {success_count} successful, {error_count} failed" if error_count == 0 else f"⚠️ Batch operation completed with issues: {success_count} successful, {error_count} failed",
                "success_count": success_count,
                "error_count": error_count,
                "total_users": len(user_identifiers),
                "role_identifier": role_identifier,
                "results": [str(r) if isinstance(r, Exception) else "✅ Removed successfully" for r in result]
            }
            return format_mcp_response(formatted_result, tool_name)

        # ==================== GROUP MANAGEMENT ====================
        elif tool_name == "ad_list_groups":
            security_only = arguments.get("security_only", False)
            unified_only = arguments.get("unified_only", False)
            limit = arguments.get("limit", 50)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                data = await ad.list_groups(security_only=security_only, unified_only=unified_only)
                # Apply limit if specified
                if limit and len(data) > limit:
                    data = data[:limit]
                result = {
                    "success": True,
                    "action": "list_groups",
                    "message": f"✅ Found {len(data)} groups in the directory",
                    "security_only": security_only,
                    "unified_only": unified_only,
                    "data": data,
                    "count": len(data)
                }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_create_group":
            try:
                # Handle null values by converting them to None or appropriate defaults
                description = arguments.get("description")
                if description is None or str(description).lower() == "null":
                    description = None
                    
                visibility = arguments.get("visibility")
                if visibility is None or str(visibility).lower() == "null":
                    visibility = None
                    
                membership_rule = arguments.get("membership_rule")
                if membership_rule is None or str(membership_rule).lower() == "null":
                    membership_rule = None
                    
                owners = arguments.get("owners")
                if owners is None or str(owners).lower() == "null":
                    owners = None
                elif isinstance(owners, list) and len(owners) == 0:
                    owners = None
                    
                members = arguments.get("members")
                if members is None or str(members).lower() == "null":
                    members = None
                elif isinstance(members, list) and len(members) == 0:
                    members = None
                
                create_group_request = CreateGroupRequest(
                    action="create_group",
                    display_name=arguments["display_name"],
                    mail_nickname=arguments["mail_nickname"],
                    description=description,
                    group_type=arguments.get("group_type", "security"),
                    visibility=visibility,
                    membership_rule=membership_rule,
                    owners=owners,
                    members=members
                )
                async with FastActiveDirectory(max_concurrent=20) as ad:
                    raw_result = await create_group_endpoint(create_group_request, ad)
                
                # Enhance the response with helpful information
                if raw_result.get("success"):
                    group_data = raw_result.get("data", {})
                    if isinstance(group_data, dict) and "group" in group_data:
                        group_info = group_data["group"]
                        result = {
                            "success": True,
                            "action": "create_group",
                            "message": f"✅ Successfully created group '{arguments['display_name']}'",
                            "group_data": {
                                "displayName": arguments["display_name"],
                                "mailNickname": arguments["mail_nickname"],
                                "groupType": arguments.get("group_type", "security"),
                                "description": arguments.get("description", "No description provided"),
                                "id": group_info.get("id"),
                                "mail": group_info.get("mail")
                            },
                            "data": raw_result.get("data"),
                            "next_steps": [
                                "You can now add members and owners to this group",
                                "Use ad_add_group_member to add users to the group",
                                "Use ad_add_group_owner to add group administrators"
                            ]
                        }
                    else:
                        result = raw_result
                else:
                    result = raw_result
            except ValidationError as ve:
                result = {
                    "success": False,
                    "action": "create_group",
                    "message": f"❌ Invalid group parameters provided for '{arguments.get('display_name', 'unknown')}'",
                    "suggestion": "Please check that required fields (display_name, mail_nickname) are provided and optional fields have valid values",
                    "error": f"Validation error: {str(ve)}"
                }
            except Exception as e:
                error_message = str(e)
                if "already exists" in error_message.lower():
                    result = {
                        "success": False,
                        "action": "create_group",
                        "message": f"❌ A group with name '{arguments.get('display_name', 'unknown')}' or mail nickname '{arguments.get('mail_nickname', 'unknown')}' already exists",
                        "suggestion": "Try using a different display name or mail nickname",
                        "error": "Group already exists"
                    }
                else:
                    result = {
                        "success": False,
                        "action": "create_group",
                        "message": f"❌ Failed to create group '{arguments.get('display_name', 'unknown')}'. {error_message}",
                        "error": error_message
                    }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_add_group_member":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the add operation
                    raw_result = await ad.add_user_to_group_smart(user_identifier, group_identifier)
                    
                    # Success - Graph API returns empty response for successful adds
                    result = {
                        "success": True,
                        "action": "add_group_member",
                        "message": f"✅ Successfully added '{user_identifier}' to group '{group_identifier}'",
                        "user_identifier": user_identifier,
                        "group_identifier": group_identifier,
                        "user_id": user_id,
                        "group_id": group_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_group_member",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_group_member",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "already exists" in error_message.lower() or "already a member" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "add_group_member",
                            "message": f"ℹ️ User '{user_identifier}' is already a member of group '{group_identifier}'",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "add_group_member",
                            "message": f"❌ Failed to add '{user_identifier}' to group '{group_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_remove_group_member":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the remove operation
                    raw_result = await ad.remove_user_from_group_smart(user_identifier, group_identifier)
                    
                    # Success - Graph API returns empty response for successful removes
                    result = {
                        "success": True,
                        "action": "remove_group_member",
                        "message": f"✅ Successfully removed '{user_identifier}' from group '{group_identifier}'",
                        "user_identifier": user_identifier,
                        "group_identifier": group_identifier,
                        "user_id": user_id,
                        "group_id": group_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_group_member",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_group_member",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "does not exist" in error_message.lower() or "not a member" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "remove_group_member",
                            "message": f"ℹ️ User '{user_identifier}' is not a member of group '{group_identifier}'",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "remove_group_member",
                            "message": f"❌ Failed to remove '{user_identifier}' from group '{group_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_group_members":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    group_id = await ad.resolve_group(group_identifier)
                    data = await ad.get_group_members(group_id)
                    result = {
                        "success": True,
                        "action": "get_group_members",
                        "message": f"✅ Found {len(data)} members in group '{group_identifier}'",
                        "group_identifier": group_identifier,
                        "group_id": group_id,
                        "data": data,
                        "count": len(data)
                    }
                except Exception as e:
                    if "No group found" in str(e):
                        result = {
                            "success": False,
                            "action": "get_group_members",
                            "message": f"Please check the group name '{group_identifier}' - I couldn't find a group with that name. You might want to try the full group name or mail nickname.",
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "get_group_members",
                            "message": f"There was an issue retrieving members for group '{group_identifier}'. Please try again.",
                            "group_identifier": group_identifier,
                            "error": str(e)
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_group_owners":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    group_id = await ad.resolve_group(group_identifier)
                    data = await ad.get_group_owners(group_id)
                    result = {
                        "success": True,
                        "action": "get_group_owners",
                        "message": f"✅ Found {len(data)} owners for group '{group_identifier}'",
                        "group_identifier": group_identifier,
                        "group_id": group_id,
                        "data": data,
                        "count": len(data)
                    }
                except Exception as e:
                    if "No group found" in str(e):
                        result = {
                            "success": False,
                            "action": "get_group_owners",
                            "message": f"Please check the group name '{group_identifier}' - I couldn't find a group with that name. You might want to try the full group name or mail nickname.",
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "get_group_owners",
                            "message": f"There was an issue retrieving owners for group '{group_identifier}'. Please try again.",
                            "group_identifier": group_identifier,
                            "error": str(e)
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_add_group_owner":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the add owner operation
                    raw_result = await ad.add_owner_to_group_smart(user_identifier, group_identifier)
                    
                    # Success - Graph API returns empty response for successful adds
                    result = {
                        "success": True,
                        "action": "add_group_owner",
                        "message": f"✅ Successfully added '{user_identifier}' as owner of group '{group_identifier}'",
                        "user_identifier": user_identifier,
                        "group_identifier": group_identifier,
                        "user_id": user_id,
                        "group_id": group_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_group_owner",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "add_group_owner",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "already exists" in error_message.lower() or "already an owner" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "add_group_owner",
                            "message": f"ℹ️ User '{user_identifier}' is already an owner of group '{group_identifier}'",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "add_group_owner",
                            "message": f"❌ Failed to add '{user_identifier}' as owner of group '{group_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_remove_group_owner":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve identifiers first to get actual names/IDs
                    user_id = await ad.resolve_user(user_identifier)
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the remove owner operation
                    raw_result = await ad.remove_owner_from_group_smart(user_identifier, group_identifier)
                    
                    # Success - Graph API returns empty response for successful removes
                    result = {
                        "success": True,
                        "action": "remove_group_owner",
                        "message": f"✅ Successfully removed '{user_identifier}' as owner from group '{group_identifier}'",
                        "user_identifier": user_identifier,
                        "group_identifier": group_identifier,
                        "user_id": user_id,
                        "group_id": group_id
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No user found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_group_owner",
                            "message": f"❌ Could not find user '{user_identifier}'. Please check the username, email, or display name.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "remove_group_owner",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    elif "does not exist" in error_message.lower() or "not an owner" in error_message.lower():
                        result = {
                            "success": False,
                            "action": "remove_group_owner",
                            "message": f"ℹ️ User '{user_identifier}' is not an owner of group '{group_identifier}'",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "remove_group_owner",
                            "message": f"❌ Failed to remove '{user_identifier}' as owner from group '{group_identifier}'. {error_message}",
                            "user_identifier": user_identifier,
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_update_group":
            group_identifier = arguments["group_identifier"]
            updates = arguments["updates"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve group first to get actual details
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the update
                    raw_result = await ad.update_group_smart(group_identifier, updates)
                    
                    # Success - Graph API returns empty response for successful updates
                    result = {
                        "success": True,
                        "action": "update_group",
                        "message": f"✅ Successfully updated group '{group_identifier}'",
                        "group_identifier": group_identifier,
                        "group_id": group_id,
                        "updated_fields": list(updates.keys())
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "update_group",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "update_group",
                            "message": f"❌ Failed to update group '{group_identifier}'. {error_message}",
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_delete_group":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                try:
                    # Resolve group first to get actual details
                    group_id = await ad.resolve_group(group_identifier)
                    
                    # Perform the delete
                    raw_result = await ad.delete_group_smart(group_identifier)
                    
                    # Success - Graph API returns empty response for successful deletes
                    result = {
                        "success": True,
                        "action": "delete_group",
                        "message": f"✅ Successfully deleted group '{group_identifier}'",
                        "group_identifier": group_identifier,
                        "group_id": group_id,
                        "warning": "This action cannot be undone. The group and all its settings have been permanently removed."
                    }
                except Exception as e:
                    error_message = str(e)
                    if "No group found" in error_message:
                        result = {
                            "success": False,
                            "action": "delete_group",
                            "message": f"❌ Could not find group '{group_identifier}'. Please check the group name or mail nickname.",
                            "group_identifier": group_identifier
                        }
                    else:
                        result = {
                            "success": False,
                            "action": "delete_group",
                            "message": f"❌ Failed to delete group '{group_identifier}'. {error_message}",
                            "group_identifier": group_identifier,
                            "error": error_message
                        }
            return format_mcp_response(result, tool_name)

        elif tool_name == "ad_get_group_full_details":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_group_full_details(group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_search_groups":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.search_groups_fuzzy(query, limit=limit)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_batch_add_users_to_group":
            user_identifiers = arguments["user_identifiers"]
            group_identifier = arguments["group_identifier"]
            ignore_errors = arguments.get("ignore_errors", True)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.batch_add_users_to_group(
                    user_identifiers,
                    group_identifier,
                    ignore_errors=ignore_errors
                )
            # Format results
            success_count = len([r for r in result if not isinstance(r, Exception)])
            error_count = len([r for r in result if isinstance(r, Exception)])
            formatted_result = {
                "success_count": success_count,
                "error_count": error_count,
                "results": [str(r) if isinstance(r, Exception) else "Success" for r in result]
            }
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(formatted_result, indent=2))]
            )

        elif tool_name == "ad_batch_remove_users_from_group":
            user_identifiers = arguments["user_identifiers"]
            group_identifier = arguments["group_identifier"]
            ignore_errors = arguments.get("ignore_errors", True)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.batch_remove_users_from_group(
                    user_identifiers,
                    group_identifier,
                    ignore_errors=ignore_errors
                )
            # Format results
            success_count = len([r for r in result if not isinstance(r, Exception)])
            error_count = len([r for r in result if isinstance(r, Exception)])
            formatted_result = {
                "success_count": success_count,
                "error_count": error_count,
                "results": [str(r) if isinstance(r, Exception) else "Success" for r in result]
            }
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(formatted_result, indent=2))]
            )

        elif tool_name == "ad_check_user_membership":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.check_user_membership(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps({"is_member": result}, indent=2))]
            )

        elif tool_name == "ad_check_user_ownership":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.check_user_ownership(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps({"is_owner": result}, indent=2))]
            )

        elif tool_name == "ad_sync_group_members":
            group_identifier = arguments["group_identifier"]
            desired_user_identifiers = arguments["desired_user_identifiers"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.sync_group_members(group_identifier, desired_user_identifiers)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_batch_get_user_groups":
            user_identifiers = arguments["user_identifiers"]
            transitive = arguments.get("transitive", False)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.batch_get_user_groups(user_identifiers, transitive=transitive)
            # Format results
            formatted_result = []
            for i, groups in enumerate(result):
                formatted_result.append({
                    "user_identifier": user_identifiers[i],
                    "groups": groups if not isinstance(groups, Exception) else str(groups),
                    "success": not isinstance(groups, Exception)
                })
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(formatted_result, indent=2))]
            )

        elif tool_name == "ad_list_users_with_groups":
            include_transitive = arguments.get("include_transitive", False)
            include_owned = arguments.get("include_owned", True)
            select = arguments.get("select", "id,displayName,userPrincipalName")
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.list_users_with_groups(
                    include_transitive=include_transitive,
                    include_owned=include_owned,
                    select=select
                )
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
            import traceback
            traceback.print_exc()
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
                        "name": "MCP Server with LLM SQL Generation",
                        "version": "1.0.0"
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
    """Debug endpoint to see raw requests"""
    body = await request.json()
    print(f"[DEBUG] Raw request: {body}")
    return {"received": body}


@app.get("/info")
async def server_info():
    """Server information endpoint"""
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
            "/ad/users",
            "/ad/roles",
            "/ad/groups",
            "/health"
        ],
        "features": [
            "Streaming Support",
            "Active Directory Operations",
            "MCP Protocol Support",
            "Vector Database Integration"
        ],
        "tools": [
            "greet",
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
            "search_cosmic_database"
        ],
        "docs": "/docs",
        "mcp_compatible": True
    }


if __name__ == "__main__":
    print("Starting MCP Server on port 8009...")

    if DEBUG:
        print("[MCP DEBUG] Debug mode enabled - detailed logging active")

    uvicorn.run(app, host="0.0.0.0", port=8009)