# -------------------- Built-in Libraries --------------------
import json
import asyncio
import uuid
import re
import os
from datetime import datetime
from typing import Dict, Optional, List, Any, Literal

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

# -------------------- User-defined Modules --------------------
from active_directory import FastActiveDirectory
from vector_mistral_tool import hospital_support_questions_tool
from create_jira_ticket import create_jira_ticket


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


# ++++++++++++++++++++++++++++++++
# TICKET SYSTEM PYDANTIC MODELS START
# ++++++++++++++++++++++++++++++++

class ThreadTicketRequest(BaseModel):
    thread_id: str

class KnownProblemsRequest(BaseModel):
    query: str
    ticket_id: Optional[str] = None

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

class HospitalSupportQuestionsRequest(BaseModel):
    query: str = Field(..., description="The support question to search for in the hospital support knowledge base")

# ++++++++++++++++++++++++++++++
# TICKET SYSTEM PYDANTIC MODELS END
# ++++++++++++++++++++++++++++++


load_dotenv()

# Debug flag
DEBUG = True

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


# +++++++++++++++++++++++++++
# TICKET AGENT SYSTEM START
# +++++++++++++++++++++++++++

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

        # This is a new ticket - proceed with full hospital support tool flow
        if DEBUG:
            print(f"[API DEBUG] New ticket - Querying hospital support for: {request.query}")

        # Step 1: Query hospital support protocols
        qdrant_result = await hospital_support_questions_tool(request.query)

        if DEBUG:
            print(f"[API DEBUG] Hospital support result: {qdrant_result}")

        # Step 2: Process the result and generate ticket ID
        if qdrant_result:
            ticket_id = str(uuid.uuid4())
            
            # Create ticket in thread manager
            await thread_ticket_manager.get_or_create_ticket(thread_id)

            if DEBUG:
                print(f"[API DEBUG] Generated new ticket ID: {ticket_id}")

            return {
                "success": True,
                "result": qdrant_result,
                "ticket_id": ticket_id,
                "is_new_ticket": True,
                "thread_id": thread_id
            }
        else:
            return {
                "success": False,
                "error": "No hospital support information found",
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
    """Check if a thread has an active ticket."""
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
                print(f"[CHECK_ACTIVE] Active ticket found - Status: {response['ticket_status']}, Complete: {is_complete}")

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

            if DEBUG:
                print(f"[CREATE_JIRA] Marked ticket as completed for thread: {thread_id}")

        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

# +++++++++++++++++++++++++++
# TICKET AGENT SYSTEM END
# +++++++++++++++++++++++++++


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
            description="TRIGGER: create AD user, add Azure AD user, new AD account, register AD user, add employee to directory | ACTION: Create new Azure Active Directory user account | RETURNS: Created AD user details with generated ID and tenant information",
            inputSchema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "description": "Azure AD user payload with user properties",
                        "properties": {
                            "displayName": {"type": "string", "description": "User's display name in AD"},
                            "mailNickname": {"type": "string", "description": "Mail nickname (auto-generated if not provided)"},
                            "userPrincipalName": {"type": "string", "description": "User principal name (auto-generated if not provided)"},
                            "passwordProfile": {
                                "type": "object",
                                "properties": {
                                    "password": {"type": "string", "description": "Temporary password"},
                                    "forceChangePasswordNextSignIn": {"type": "boolean", "description": "Force password change on next sign-in"}
                                }
                            },
                            "accountEnabled": {"type": "boolean", "description": "Whether AD account is enabled"}
                        },
                        "required": ["displayName"]
                    }
                },
                "required": ["user"]
            }
        ),

        MCPTool(
            name="ad_update_user",
            description="TRIGGER: update AD user, modify Azure AD user, change AD user details, edit directory user, AD user updates | ACTION: Update existing Azure Active Directory user properties | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: Updated AD user information from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    },
                    "updates": {"type": "object", "description": "AD user properties to update"}
                },
                "required": ["user_id", "updates"]
            }
        ),

        MCPTool(
            name="ad_delete_user",
            description="TRIGGER: delete AD user, remove Azure AD user, deactivate directory user, remove AD account | ACTION: Delete user account from Azure Active Directory | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: AD account deletion confirmation",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["user_id"]
            }
        ),

        MCPTool(
            name="ad_get_user_roles",
            description="TRIGGER: AD user roles, Azure AD user permissions, directory user roles, what AD roles does user have, check AD access | ACTION: Get Azure AD user's assigned directory roles | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: List of AD roles assigned to specific user from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["user_id"]
            }
        ),

        MCPTool(
            name="ad_get_user_groups",
            description="TRIGGER: AD user groups, Azure AD user memberships, directory user groups, what AD groups is user in, check AD group membership | ACTION: Get user's Azure Active Directory group memberships | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: List of AD groups user belongs to in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    },
                    "transitive": {"type": "boolean", "description": "Include transitive AD group memberships", "default": False}
                },
                "required": ["user_id"]
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
            description="TRIGGER: assign AD role, add user to AD role, give Azure AD role, grant directory role, AD role assignment | ACTION: Assign Azure Active Directory role to user | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: AD role assignment confirmation in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_id": {"type": "string", "description": "Azure AD role ID (must be actual GUID) to assign"},
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["role_id", "user_id"]
            }
        ),

        MCPTool(
            name="ad_remove_user_from_role",
            description="TRIGGER: remove AD role, unassign Azure AD role, revoke directory role, take away AD role | ACTION: Remove Azure Active Directory role from user | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: AD role removal confirmation from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_id": {"type": "string", "description": "Azure AD role ID (must be actual GUID) to remove"},
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["role_id", "user_id"]
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
                "required": ["security_only", "unified_only", "select"]
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
                    "description": {"type": "string", "description": "AD group description"},
                    "group_type": {"type": "string", "enum": ["security", "unified"], "description": "Azure AD group type", "default": "security"},
                    "visibility": {"type": "string", "description": "AD group visibility"},
                    "owners": {"type": "array", "items": {"type": "string"}, "description": "List of AD owner user IDs"},
                    "members": {"type": "array", "items": {"type": "string"}, "description": "List of AD member user IDs"}
                },
                "required": ["display_name", "mail_nickname"]
            }
        ),

        MCPTool(
            name="ad_add_group_member",
            description="TRIGGER: add to AD group, add member to Azure AD group, add user to directory group, join AD group | ACTION: Add user to Azure Active Directory group | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: AD group membership confirmation in Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Azure AD group ID (must be actual GUID) to add member to"},
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["group_id", "user_id"]
            }
        ),

        MCPTool(
            name="ad_remove_group_member",
            description="TRIGGER: remove from AD group, remove member from Azure AD group, leave directory group, kick from AD group | ACTION: Remove user from Azure Active Directory group | INSTRUCTION: If you know the user's name, email, or display name, use that as user_id. The system will automatically resolve it to the actual Azure AD user ID. You do NOT need the actual GUID - just provide whatever identifier you have (name like 'John Marks', email like 'john@company.com', or the actual user ID) | RETURNS: AD group removal confirmation from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Azure AD group ID (must be actual GUID) to remove member from"},
                    "user_id": {
                        "type": "string",
                        "description": "User identifier - can be Azure AD user GUID, email address (userPrincipalName), or display name (e.g., 'John Marks'). The system automatically resolves any format to the actual user ID."
                    }
                },
                "required": ["group_id", "user_id"]
            }
        ),

        MCPTool(
            name="ad_get_group_members",
            description="TRIGGER: AD group members, Azure AD group members, who is in directory group, show AD group members, list AD group members | ACTION: Get Azure Active Directory group member list | RETURNS: All members of specified AD group from Azure tenant",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "Azure AD group ID to get members for"}
                },
                "required": ["group_id"]
            }
        ),

        # TICKET SYSTEM TOOLS
        MCPTool(
            name="hospital_support_questions_tool",
            description="TRIGGER: technical support, equipment issues, software problems, facility maintenance, IT help, system down, printer not working, network issues, computer problems, medical equipment malfunction | ACTION: Analyze hospital support queries and return diagnostic protocols | RETURNS: Structured support protocols, diagnostic questions, and routing information for hospital technical issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The support question or problem description to analyze"
                    }
                },
                "required": ["query"]
            }
        ),

        MCPTool(
            name="ticket_known_problems",
            description="TRIGGER: report issue, create ticket, problem with, need help with, having trouble, technical issue, equipment failure, software bug | ACTION: Search for known problems and initiate ticket creation process | RETURNS: Relevant solutions or starts ticket creation workflow with diagnostic questions",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the problem or issue to search for"
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID for ticket management",
                        "default": "default"
                    }
                },
                "required": ["query"]
            }
        ),

        MCPTool(
            name="ticket_check_active",
            description="TRIGGER: check ticket status, active ticket, ticket in progress, current ticket | ACTION: Check if thread has an active ticket in progress | RETURNS: Active ticket status and completion information",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID to check for active tickets"
                    }
                },
                "required": ["thread_id"]
            }
        ),

        MCPTool(
            name="ticket_update_fields",
            description="TRIGGER: update ticket, add information, set priority, specify location, add details | ACTION: Update ticket fields with collected information | RETURNS: Updated ticket status and completeness check",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID of the ticket to update"
                    },
                    "fields": {
                        "type": "object",
                        "description": "Dictionary of field names and values to update",
                        "properties": {
                            "description": {"type": "string"},
                            "conversation_topic": {"type": "string"},
                            "category": {"type": "string"},
                            "queue": {"type": "string"},
                            "priority": {"type": "string"},
                            "department": {"type": "string"},
                            "name": {"type": "string"},
                            "location": {"type": "string"}
                        }
                    }
                },
                "required": ["thread_id", "fields"]
            }
        ),

        MCPTool(
            name="create_jira_ticket",
            description="TRIGGER: create ticket, submit ticket, finalize ticket, ready to create | ACTION: Create final JIRA ticket with all collected information | RETURNS: JIRA ticket creation confirmation and ticket reference",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "Thread ID for ticket context"},
                    "conversation_topic": {"type": "string", "description": "Brief summary/title of the issue"},
                    "description": {"type": "string", "description": "Detailed description of the problem"},
                    "location": {"type": "string", "description": "Location where issue occurs"},
                    "queue": {"type": "string", "description": "Support queue/department to route ticket"},
                    "priority": {"type": "string", "description": "Priority level (Critical/High/Normal/Low)"},
                    "department": {"type": "string", "description": "Department affected or responsible"},
                    "name": {"type": "string", "description": "Name of person reporting issue"},
                    "category": {"type": "string", "description": "Issue category (hardware/software/facility/etc)"}
                },
                "required": ["thread_id", "conversation_topic", "description", "location", "queue", "priority", "department", "name", "category"]
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

        if tool_name == "greet":
            raw_name = arguments.get("name") if arguments else None
            clean_name = raw_name if raw_name and raw_name.strip() else None
            greet_request = GreetRequest(name=clean_name)
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_list_users":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await list_users_endpoint(ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_create_user":
            create_request = CreateUserRequest(action="create_user", user=arguments["user"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await create_user_endpoint(create_request, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_update_user":
            user_updates = UserUpdates(updates=arguments["updates"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await update_user_endpoint(arguments["user_id"], user_updates, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_delete_user":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await delete_user_endpoint(arguments["user_id"], ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_roles":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await get_user_roles_endpoint(arguments["user_id"], ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_groups":
            transitive = arguments.get("transitive", False)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await get_user_groups_endpoint(arguments["user_id"], transitive, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_list_roles":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await list_roles_endpoint(ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_user_to_role":
            role_member = RoleAddMember(user_id=arguments["user_id"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await add_user_to_role_endpoint(arguments["role_id"], role_member, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_user_from_role":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await remove_user_from_role_endpoint(arguments["role_id"], arguments["user_id"], ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_list_groups":
            security_only = arguments.get("security_only", False)
            unified_only = arguments.get("unified_only", False)
            select = arguments.get("select", "id,displayName,mailNickname,mail,securityEnabled,groupTypes")
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await list_groups_endpoint(security_only, unified_only, select, ad)
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
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await create_group_endpoint(create_group_request, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_group_member":
            group_member = GroupMemberRequest(user_id=arguments["user_id"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await add_group_member_endpoint(arguments["group_id"], group_member, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_group_member":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await remove_group_member_endpoint(arguments["group_id"], arguments["user_id"], ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_group_members":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await get_group_members_endpoint(arguments["group_id"], ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "hospital_support_questions_tool":
            hospital_request = HospitalSupportQuestionsRequest(query=arguments["query"])
            result = await hospital_support_questions_tool(hospital_request.query)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps({"success": True, "result": result}, indent=2))]
            )

        elif tool_name == "ticket_known_problems":
            # Add thread_id if not present
            if "thread_id" not in arguments:
                arguments["thread_id"] = "default"
            
            known_problems_request = KnownProblemsRequest(**arguments)
            result = await known_problems_endpoint(known_problems_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ticket_check_active":
            check_request = ThreadTicketRequest(thread_id=arguments["thread_id"])
            result = await check_active_ticket_endpoint(check_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ticket_update_fields":
            thread_id = arguments["thread_id"]
            fields_data = {"fields": arguments["fields"]}
            result = await update_ticket_fields_endpoint(thread_id, fields_data)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "create_jira_ticket":
            create_jira_request = CreateJiraRequest(**arguments)
            result = await create_jira_endpoint(create_jira_request)
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