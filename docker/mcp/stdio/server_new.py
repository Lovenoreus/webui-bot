# -------------------- Built-in Libraries --------------------
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import os

# -------------------- External Libraries --------------------
import aiohttp
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    Body,
    Request,
    Response
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- User-defined Modules --------------------
from active_directory import FastActiveDirectory
from models import (
    UserUpdates,
    RoleAddMember,
    GroupMemberRequest,
    GroupOwnerRequest,
    GroupUpdates,
    RoleInstantiation,
    CreateUserRequest,
    CreateGroupRequest,
    BatchUserIdentifiersRequest,
    MCPTool,
    MCPToolsListResponse,
    MCPToolCallRequest,
    MCPContent,
    MCPToolCallResponse,
    GreetRequest,
    # Ticket-related models
    TicketStatus,
    TicketPriority,
    TicketCategory,
    CreateTicketRequest,
    UpdateTicketRequest,
    SubmitTicketRequest,
    SubmitTicketResponse,
    EscalatorInfo,
    TicketResponse,
    InitializeTicketRequest,
    # MCPServerInfo
)
from vector_mistral_tool import hospital_support_questions_tool
from create_jira_ticket import create_jira_ticket
import config

load_dotenv(find_dotenv())

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


# Dependency to get FastActiveDirectory instance
async def get_ad():
    """Dependency that provides FastActiveDirectory instance"""
    async with FastActiveDirectory(max_concurrent=20) as ad:
        yield ad


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


# ==================== JSON STORAGE ====================

TICKETS_FILE = "tickets_data.json"


class JSONStorage:
    """Thread-safe JSON file storage"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._lock = asyncio.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create empty JSON file if it doesn't exist"""
        if not self.filepath.exists():
            self.filepath.write_text(json.dumps({
                "tickets": {},
                "thread_index": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }, indent=2))
            if DEBUG:
                print(f"[STORAGE] Created new storage file: {self.filepath}")

    async def load(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        async with self._lock:
            try:
                data = json.loads(self.filepath.read_text())
                return data
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(f"[STORAGE] Error loading JSON: {e}")
                return {
                    "tickets": {},
                    "thread_index": {},
                    "metadata": {"created_at": datetime.now().isoformat()}
                }

    async def save(self, data: Dict[str, Any]):
        """Save data to JSON file"""
        async with self._lock:
            try:
                temp_file = self.filepath.with_suffix('.tmp')
                temp_file.write_text(json.dumps(data, indent=2))
                temp_file.replace(self.filepath)

                if DEBUG:
                    print(f"[STORAGE] Saved data to {self.filepath}")
            except Exception as e:
                if DEBUG:
                    print(f"[STORAGE] Error saving JSON: {e}")
                raise

    async def backup(self):
        """Create a backup of the current data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.filepath.with_name(f"{self.filepath.stem}_backup_{timestamp}.json")

        async with self._lock:
            if self.filepath.exists():
                backup_file.write_text(self.filepath.read_text())
                if DEBUG:
                    print(f"[STORAGE] Created backup: {backup_file}")


# Global storage instance
storage = JSONStorage(TICKETS_FILE)


# ==================== TICKET MANAGER ====================

class TicketManager:
    """
    JSON-based ticket manager with file persistence
    """

    # List of fields that **must** be present and non-empty in a ticket
    REQUIRED_FIELDS = ["description", "category", "priority"]

    # Set of all fields that are permitted in the ticket data
    # These are the fields that are recognized and allowed
    ALLOWED_FIELDS = {
        "conversation_topic", "description", "location", "queue",
        "priority", "department", "category", "reporter_name"
    }

    VALID_PRIORITIES = [
        # TicketPriority.CRITICAL,
        TicketPriority.HIGH,
        TicketPriority.MEDIUM,
        TicketPriority.LOW
    ]

    # VALID_CATEGORIES = [
    #     TicketCategory.HARDWARE,
    #     TicketCategory.SOFTWARE,
    #     TicketCategory.FACILITY,
    #     TicketCategory.NETWORK,
    #     TicketCategory.MEDICAL_EQUIPMENT,
    #     TicketCategory.OTHER
    # ]

    def __init__(self, storage: JSONStorage):
        self.storage = storage

    async def create_ticket(
            self,
            query: str,
            conversation_id: str,
            ticket_id: str,
            systems: Optional[str] = None,
            reporter_name: Optional[str] = "Mathew Pattel",
            reporter_email: Optional[str] = "mathewp@gmail.com",
            knowledge_base_result: Optional[str] = None,
            category: Optional[str] = None,
            priority: Optional[str] = None,
            queue: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new ticket"""
        data = await self.storage.load()
        now = datetime.now().isoformat()

        ticket = {
            "ticket_id": ticket_id,
            "systems": systems,
            "conversation_id": conversation_id,
            "status": TicketStatus.DRAFT,
            "created_at": now,
            "updated_at": now,
            "reporter_name": reporter_name,
            "reporter_email": reporter_email,
            "jira_key": None,

            # Ticket fields
            "conversation_topic": None,
            "description": query,
            "location": None,
            "queue": queue if queue else "queue",
            "priority": priority,
            "department": None,
            "category": category,

            # Metadata
            "knowledge_base_result": knowledge_base_result,
            "history": []
        }

        # Store ticket
        data["tickets"][ticket_id] = ticket

        # Update thread index
        if conversation_id not in data["thread_index"]:
            data["thread_index"][conversation_id] = []

        data["thread_index"][conversation_id].append(ticket_id)

        await self.storage.save(data)

        if DEBUG:
            print(f"[TICKET] Created ticket {ticket_id} for thread {conversation_id}")

        return self._enrich_ticket(ticket)

    async def get_ticket(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get ticket by ID"""
        data = await self.storage.load()
        ticket = data["tickets"].get(ticket_id)

        if not ticket:
            return None

        return self._enrich_ticket(ticket)

    async def get_active_ticket_for_thread(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent active (non-completed, non-cancelled) ticket for a given conversation/thread"""

        # Load all ticket data from storage (likely a file or DB abstraction)
        data = await self.storage.load()

        # Retrieve the list of ticket IDs associated with this conversation
        ticket_ids = data["thread_index"].get(conversation_id, [])

        # Iterate over ticket IDs in reverse (most recent first)
        for ticket_id in reversed(ticket_ids):
            # Get the actual ticket data for this ticket ID
            ticket = data["tickets"].get(ticket_id)

            # If ticket exists and is not completed or cancelled, return the enriched ticket
            if ticket and ticket["status"] not in [TicketStatus.SUBMITTED, TicketStatus.COMPLETED, TicketStatus.CANCELLED]:
                return self._enrich_ticket(ticket)

        # If no active ticket is found, return None
        return None

    async def list_tickets_for_thread(self, conversation_id: str) -> List[Dict[str, Any]]:
        """List all tickets for a thread"""
        data = await self.storage.load()
        ticket_ids = data["thread_index"].get(conversation_id, [])

        tickets = []
        for ticket_id in reversed(ticket_ids):
            ticket = data["tickets"].get(ticket_id)
            if ticket:
                tickets.append(self._enrich_ticket(ticket))

        return tickets

    async def update_field(
            self,
            ticket_id: str,
            field_name: str,
            field_value: Any
    ) -> Dict[str, Any]:
        """Update a single field"""
        if field_name not in self.ALLOWED_FIELDS:
            raise ValueError(f"Field '{field_name}' not allowed. Allowed: {self.ALLOWED_FIELDS}")

        # Validate priority
        if field_name == "priority" and field_value not in self.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority. Must be one of: {self.VALID_PRIORITIES}")

        # Validate category
        # if field_name == "category" and field_value not in self.VALID_CATEGORIES:
        #     raise ValueError(f"Invalid category. Must be one of: {self.VALID_CATEGORIES}")

        data = await self.storage.load()
        ticket = data["tickets"].get(ticket_id)

        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        old_value = ticket.get(field_name)
        now = datetime.now().isoformat()

        # Update field
        ticket[field_name] = field_value
        ticket["updated_at"] = now

        # Auto-route queue based on category
        if field_name == "category" and field_value:
            ticket["queue"] = self._route_by_category(field_value)
            ticket["history"].append({
                "timestamp": now,
                "field_name": "queue",
                "old_value": None,
                "new_value": ticket["queue"],
                "action": "auto_route"
            })

        # Log history
        ticket["history"].append({
            "timestamp": now,
            "field_name": field_name,
            "old_value": old_value,
            "new_value": field_value,
            "action": "continue"
        })

        await self.storage.save(data)

        if DEBUG:
            print(f"[TICKET] Updated {field_name} for {ticket_id}: '{old_value}' -> '{field_value}'")

        return self._enrich_ticket(ticket)

    async def update_fields(
            self,
            ticket_id: str,
            fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update multiple fields at once"""
        invalid_fields = [f for f in fields.keys() if f not in self.ALLOWED_FIELDS]
        if invalid_fields:
            raise ValueError(f"Invalid fields: {invalid_fields}. Allowed: {self.ALLOWED_FIELDS}")

        # Validate priority
        if "priority" in fields and fields["priority"] not in self.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority. Must be one of: {self.VALID_PRIORITIES}")

        # Validate category
        # if "category" in fields and fields["category"] not in self.VALID_CATEGORIES:
        #     raise ValueError(f"Invalid category. Must be one of: {self.VALID_CATEGORIES}")

        data = await self.storage.load()
        ticket = data["tickets"].get(ticket_id)

        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        now = datetime.now().isoformat()

        # Update all fields
        for field_name, field_value in fields.items():
            old_value = ticket.get(field_name)
            ticket[field_name] = field_value

            # Auto-route queue based on category
            if field_name == "category" and field_value:
                ticket["queue"] = self._route_by_category(field_value)
                ticket["history"].append({
                    "timestamp": now,
                    "field_name": "queue",
                    "old_value": None,
                    "new_value": ticket["queue"],
                    "action": "auto_route"
                })

            # Log history
            ticket["history"].append({
                "timestamp": now,
                "field_name": field_name,
                "old_value": old_value,
                "new_value": field_value,
                "action": "continue"
            })

        ticket["updated_at"] = now

        await self.storage.save(data)

        if DEBUG:
            print(f"[TICKET] Updated {len(fields)} fields for {ticket_id}")

        return self._enrich_ticket(ticket)

    async def update_status(
            self,
            ticket_id: str,
            new_status: str,
            jira_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update ticket status"""
        data = await self.storage.load()
        ticket = data["tickets"].get(ticket_id)

        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")

        old_status = ticket["status"]
        now = datetime.now().isoformat()

        # Update status
        ticket["status"] = new_status
        ticket["updated_at"] = now

        if jira_key:
            ticket["jira_key"] = jira_key

        # Log history
        ticket["history"].append({
            "timestamp": now,
            "field_name": "status",
            "old_value": old_status,
            "new_value": new_status,
            "action": "status_change"
        })

        await self.storage.save(data)

        if DEBUG:
            print(f"[TICKET] Status changed for {ticket_id}: {old_status} -> {new_status}")

        return self._enrich_ticket(ticket)

    async def search_tickets(
            self,
            status: Optional[str] = None,
            priority: Optional[str] = None,
            category: Optional[str] = None,
            jira_key: Optional[str] = None,
            limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search tickets with filters"""
        data = await self.storage.load()
        tickets = []

        for ticket in data["tickets"].values():
            if status and ticket["status"] != status:
                continue
            if priority and ticket["priority"] != priority:
                continue
            if category and ticket["category"] != category:
                continue
            if jira_key and ticket["jira_key"] != jira_key:
                continue

            tickets.append(self._enrich_ticket(ticket))

        # Sort by priority first, then created_at
        priority_order = {
            # TicketPriority.CRITICAL: 0,
            TicketPriority.HIGH: 1,
            TicketPriority.MEDIUM: 2,
            TicketPriority.LOW: 3,
            None: 4
        }

        tickets.sort(key=lambda t: (
            priority_order.get(t.get("priority"), 4),
            t["created_at"]
        ), reverse=True)

        return tickets[:limit]

    def _route_by_category(self, category: str) -> str:
        """Auto-route ticket to appropriate queue based on category"""
        routing_map = {
            TicketCategory.HARDWARE: "Hardware Support",
            TicketCategory.SOFTWARE: "IT Support",
            TicketCategory.FACILITY: "Facilities Management",
            TicketCategory.NETWORK: "Network Operations",
            TicketCategory.MEDICAL_EQUIPMENT: "Biomedical Engineering",
            TicketCategory.OTHER: "General Support"
        }
        return routing_map.get(category, "General Support")

    def _calculate_sla_deadline(self, ticket: Dict[str, Any]) -> Optional[str]:
        """Calculate SLA deadline based on priority"""
        from datetime import timedelta

        sla_hours = {
            # TicketPriority.CRITICAL: 2,
            TicketPriority.HIGH: 8,
            TicketPriority.MEDIUM: 24,
            TicketPriority.LOW: 72
        }

        priority = ticket.get("priority")
        if not priority or priority not in sla_hours:
            return None

        created = datetime.fromisoformat(ticket["created_at"])
        deadline = created + timedelta(hours=sla_hours[priority])

        return deadline.isoformat()

    def _enrich_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Add computed fields to ticket"""
        is_complete, missing = self._check_completeness(ticket)

        return {
            **ticket,
            "is_complete": is_complete,
            "missing_fields": missing,
            "sla_deadline": self._calculate_sla_deadline(ticket),
            "fields": {
                "conversation_topic": ticket.get("conversation_topic"),
                "description": ticket.get("description"),
                "location": ticket.get("location"),
                "queue": ticket.get("queue"),
                "priority": ticket.get("priority"),
                "department": ticket.get("department"),
                "category": ticket.get("category"),
                "reporter_name": ticket.get("reporter_name")
            }
        }

    def _check_completeness(self, ticket: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check if ticket has all required fields"""

        missing = []  # List to keep track of fields that are missing or empty

        # Iterate over all fields that are required for the ticket
        for field in self.REQUIRED_FIELDS:
            value = ticket.get(field)  # Get the value for the current field from the ticket

            # Check if the value is missing or empty (including strings with only whitespace)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)

        # Return a tuple:
        # - First element: True if no fields are missing, else False
        # - Second element: List of missing field names
        return len(missing) == 0, missing


# Global ticket manager
ticket_manager = TicketManager(storage)


# ==================== API ENDPOINTS ====================
@app.post("/tickets/initialize")
async def initialize_ticket_endpoint(request: InitializeTicketRequest):
    """
    Open a ticket with knowledge base search, LLM analysis, and intelligent routing.
    Returns comprehensive contextual information for natural conversation flow.
    """
    try:
        print(request.system)
        systems = request.systems

    except:
        print('No systems!')
        systems = None

    try:
        # TODO: CHANGE BACK FROM STATIC.
        conversation_id = request.conversation_id

        if DEBUG:
            print(f"[INITIALIZE_TICKET] Processing query for thread: {conversation_id}")

        # Step 1: Check for existing active ticket in this thread
        active_ticket = await ticket_manager.get_active_ticket_for_thread(conversation_id)

        if active_ticket:
            if DEBUG:
                print(f"[INITIALIZE_TICKET] Found existing ticket: {active_ticket['ticket_id']}")

            # Parse the knowledge_base_result from the stored ticket
            kb_data = {}
            if active_ticket.get("knowledge_base_result"):
                try:
                    kb_data = json.loads(active_ticket["knowledge_base_result"])
                except:
                    kb_data = {}

            # Extract data from knowledge base result (if it exists)
            analysis = kb_data.get("analysis", {}) if isinstance(kb_data.get("analysis"), dict) else {}
            guidance = kb_data.get("guidance", {}) if isinstance(kb_data.get("guidance"), dict) else {}
            suggestions = kb_data.get("suggestions", {}) if isinstance(kb_data.get("suggestions"), dict) else {}
            metadata = kb_data.get("metadata", {}) if isinstance(kb_data.get("metadata"), dict) else {}

            # For flat structure (new format)
            if not analysis and not guidance:
                # Data is already flat
                analysis = kb_data
                guidance = kb_data
                suggestions = kb_data
                metadata = kb_data

            # DYNAMICALLY CALCULATE MISSING FIELDS
            required_fields = ["description", "category", "priority"]
            missing_fields = []

            for field in required_fields:
                value = active_ticket.get(field)

                # Check if field is missing or empty (including whitespace-only strings)
                if not value or (isinstance(value, str) and not value.strip()):
                    missing_fields.append(field)

            if DEBUG:
                print(f"[INITIALIZE_TICKET] Dynamically calculated missing fields: {missing_fields}")

            return {
                "success": True,
                "has_existing_ticket": True,
                "ticket_id": active_ticket["ticket_id"],
                "conversation_id": conversation_id,
                "is_new_ticket": False,
                "status": active_ticket.get("status", "draft"),

                # ACTUAL DATA FROM JSON - NO GUESSING
                "description": active_ticket.get("description", None),
                "category": active_ticket.get("category", None),
                "priority": active_ticket.get("priority", None),
                "queue": active_ticket.get("queue", None),
                "location": active_ticket.get("location", None),
                "department": active_ticket.get("department", None),
                "conversation_topic": active_ticket.get("conversation_topic", None),
                "reporter_name": active_ticket.get("reporter_name", None),
                "reporter_email": active_ticket.get("reporter_email", None),

                # KNOWLEDGE BASE FIELDS - From stored data with fallbacks
                "source": analysis.get("source", "unknown"),
                "protocol_id": analysis.get("protocol_id", None),
                "confidence_score": analysis.get("confidence_score", 0.0),
                "has_known_solution": analysis.get("has_known_solution", False),
                "solution_available": analysis.get("solution_available", False),
                "issue_interpretation": analysis.get("issue_interpretation",
                                                     active_ticket.get("description", "Issue details unavailable")),
                "troubleshooting_steps": kb_data.get("troubleshooting_steps", None),
                "similar_issues_count": metadata.get("similar_issues_found", 0),

                # GUIDANCE FIELDS - From stored data with fallbacks
                "message": guidance.get("message",
                                        f"Continuing with ticket {active_ticket['ticket_id']}. Please provide any additional information needed."),
                "reasoning": guidance.get("reasoning", "Collecting additional information to complete the ticket"),
                "next_step": guidance.get("next_step", "collect_info"),

                # DIAGNOSTIC QUESTIONS - From stored data with fallbacks
                "must_ask_diagnostic_questions": kb_data.get("must_ask_diagnostic_questions", [
                    "Would you like to add more details to your existing ticket?",
                    "Is this a different issue that needs a separate ticket?",
                    "Would you like to review what information we've collected so far?"
                ]),

                # TICKET STATUS - Based on actual JSON data with DYNAMIC missing fields
                "ticket_status": {
                    "ticket_id": active_ticket["ticket_id"],
                    "status": active_ticket.get("status", "draft"),
                    "is_complete": active_ticket.get("is_complete", False),
                    "missing_fields": missing_fields,  # DYNAMICALLY CALCULATED
                    "fields_filled": {k: v for k, v in {
                        "description": active_ticket.get("description"),
                        "category": active_ticket.get("category"),
                        "priority": active_ticket.get("priority"),
                        "queue": active_ticket.get("queue"),
                        "location": active_ticket.get("location"),
                        "department": active_ticket.get("department"),
                        "conversation_topic": active_ticket.get("conversation_topic"),
                        "reporter_name": active_ticket.get("reporter_name")
                    }.items() if v is not None}
                },

                # METADATA - From stored data with fallbacks
                "estimated_resolution_time": metadata.get("estimated_resolution_time", None),
                "escalation_recommended": metadata.get("escalation_recommended", False),

                "should_use_ticket_continue": True
            }

        # Step 2: Query knowledge base for similar issues and solutions
        if DEBUG:
            print(f"[INITIALIZE_TICKET] Querying knowledge base for: {request.query}")

        kb_result = await hospital_support_questions_tool(request.query)

        if DEBUG:
            print(f"[INITIALIZE_TICKET] Knowledge base returned {len(kb_result) if kb_result else 0} results")
            print(f"[INITIALIZE_TICKET] Knowledge base results {kb_result}")

        # Step 3: Select and configure LLM provider
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
                raise ValueError("OPENAI_API_KEY environment variable is required")
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
            raise ValueError("No valid LLM provider configured")

        if DEBUG:
            print(f"[INITIALIZE_TICKET] Using LLM provider: {provider}")

        # Available queues for routing
        QUEUE_CHOICES = [
            'Technical Support', 'Servicedesk', '2nd line', 'Cambio JIRA', 'Cosmic',
            'Billing Payments', 'Account Management', 'Product Inquiries', 'Feature Requests',
            'Bug Reports', 'Security Department', 'Compliance Legal', 'Service Outages',
            'Onboarding Setup', 'API Integration', 'Data Migration', 'Accessibility',
            'Training Education', 'General Inquiries', 'Permissions Access',
            'Management Department', 'Maintenance Department', 'Logistics Department',
            'IT Department'
        ]

        # Step 4: Prepare enhanced LLM prompt for intelligent analysis with FLAT structure
        system_prompt = """
            You are an expert hospital support system analyzer. Analyze user queries and provide intelligent ticket initialization with actionable guidance.
            
            INPUT:
            - User Query: {user_query}
            - Knowledge Base Results: {qdrant_response}
            - Available Queues: {queue_choices}
            
            ANALYSIS TASK:
            1. Evaluate knowledge base protocols for relevance to user query
            2. Determine if there's a known issue available
               - If there is, get all the information about the known issue
               - If there isn't, generate yours
            3. Suggest appropriate routing (category, queue, priority)
            4. Provide clear guidance on next steps
            
            EVALUATION CRITERIA:
            - Match query keywords with protocol keywords and descriptions
            - Consider protocol confidence/match scores
            - Assess clinical domain and issue category alignment
            - Evaluate if troubleshooting steps are available
            
            RESPONSE STRUCTURE (FLAT - NO NESTED OBJECTS):
            {{
              "success": true,
              "source": "protocol" | "generated",
              "protocol_id": "ID from knowledge base or null",
              "confidence_score": 0.0 to 1.0,
              "has_known_solution": boolean,
              "solution_available": boolean,
              "issue_interpretation": "your understanding of the user's issue",
              "message": "clear, helpful message to user about what was found",
              "next_step": "try_solution" | "follow_troubleshooting" | "collect_info" | "immediate_escalation",
              "reasoning": "brief explanation of why this next step",
              "must_ask_diagnostic_questions": [
                "specific, actionable question 1",
                "specific, actionable question 2", 
                "specific, actionable question 3",
                "specific, actionable question 4",
                "specific, actionable question 5",
                ...
              ],
              "troubleshooting_steps": [
                "step 1",
                "step 2"
              ] or null,
              "category": "Hardware" | "Software" | "Network" | "Facility" | "Medical Equipment" | "Other",
              "queue": "queue name from available list",
              "priority": "High" | "Medium" | "Low",
              "estimated_resolution_time": "time estimate or null",
              "similar_issues_found": number,
              "escalation_recommended": boolean
            }}
            
            PRIORITY GUIDELINES:
            - High: Major functionality broken, significant operational impact, System down, patient care directly impacted, safety issue
            - Medium: Standard issues, workarounds available
            - Low: Minor inconveniences, feature requests
            
            CATEGORY GUIDELINES:
            - Hardware: Physical equipment, devices, peripherals
            - Software: Applications, systems, programs
            - Network: Connectivity, internet, infrastructure
            - Medical Equipment: Clinical devices, diagnostic tools
            - Facility: Building, rooms, physical environment
            - Other: Doesn't fit above categories
            
            CRITICAL: Return ONLY valid JSON with a FLAT structure. No nested objects for "analysis", "guidance", "suggestions", or "metadata". All fields should be at the top level of the JSON object. No markdown formatting, no code blocks, no explanations outside the JSON structure.
        """

        messages = [
            SystemMessage(content=system_prompt.format(
                user_query=request.query,
                qdrant_response=json.dumps(kb_result) if kb_result else "No results found",
                queue_choices=", ".join(QUEUE_CHOICES)
            ))
        ]

        # Step 5: Call LLM for intelligent analysis
        if DEBUG:
            print(f"[INITIALIZE_TICKET] Calling LLM to analyze query and knowledge base")

        llm_response = llm.invoke(messages)
        response_content = llm_response.content.strip()

        if DEBUG:
            print(f"[INITIALIZE_TICKET] Raw LLM response length: {len(response_content)} chars")

        # Step 6: Parse and validate LLM response
        llm_analysis = None

        try:
            # Clean markdown formatting if present
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()

            elif response_content.startswith("```"):
                response_content = response_content.replace("```", "").strip()

            # Parse JSON
            llm_analysis = json.loads(response_content)

            if DEBUG:
                print(f"[INITIALIZE_TICKET] Successfully parsed LLM analysis")
                print(f"[LLM RESPONSE] {llm_analysis}")

        except json.JSONDecodeError as json_err:
            if DEBUG:
                print(f"[INITIALIZE_TICKET] JSON parse failed, attempting extraction: {json_err}")

            # Try to extract JSON pattern
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_content, re.DOTALL)

            if match:
                try:
                    llm_analysis = json.loads(match.group(0))
                    if DEBUG:
                        print(f"[INITIALIZE_TICKET] Extracted JSON successfully")
                except json.JSONDecodeError:
                    if DEBUG:
                        print(f"[INITIALIZE_TICKET] Extraction failed, using fallback")

        # Step 7: Create ticket with analysis results
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

        # Extract fields from LLM analysis to pass to create_ticket
        category = None
        priority = None
        queue = None

        if llm_analysis and llm_analysis.get("success"):
            category = llm_analysis.get("category")
            priority = llm_analysis.get("priority")
            queue = llm_analysis.get("queue")

        ticket = await ticket_manager.create_ticket(
            query=request.query,
            systems=systems,
            conversation_id=conversation_id,
            ticket_id=ticket_id,
            reporter_name=request.reporter_name,
            reporter_email=request.reporter_email,
            knowledge_base_result=json.dumps(llm_analysis) if llm_analysis else json.dumps(kb_result),
            category=category,
            priority=priority,
            queue=queue
        )

        print(
            f'This is the knowledge_base_result: {json.dumps(llm_analysis) if llm_analysis else json.dumps(kb_result)}')

        if DEBUG:
            print(f"[INITIALIZE_TICKET] Created ticket: {ticket_id}")
            print(f"[INITIALIZE_TICKET] Pre-populated with category={category}, priority={priority}, queue={queue}")

        # Step 7.5: DYNAMICALLY CALCULATE MISSING FIELDS for new ticket
        required_fields = ["description", "category", "priority"]
        new_ticket_missing_fields = []

        # Get the enriched ticket data to check completeness
        enriched_ticket = await ticket_manager.get_ticket(ticket_id)

        for field in required_fields:
            value = enriched_ticket.get(field)
            # Check if field is missing or empty (including whitespace-only strings)
            if not value or (isinstance(value, str) and not value.strip()):
                new_ticket_missing_fields.append(field)

        if DEBUG:
            print(
                f"[INITIALIZE_TICKET] Dynamically calculated missing fields for new ticket: {new_ticket_missing_fields}")

        # Step 8: Build comprehensive response with FLAT structure
        if llm_analysis and llm_analysis.get("success"):
            print(f'[Success Analysis]: LLM Response: {llm_analysis}')

            response = {
                "success": True,
                "ticket_id": ticket_id,
                "conversation_id": conversation_id,
                "is_new_ticket": True,
                "data_collection_started": True,

                # KNOWLEDGE BASE FIELDS - Direct from LLM (FLAT)
                "source": llm_analysis.get("source", "unknown"),
                "protocol_id": llm_analysis.get("protocol_id", None),
                "confidence_score": llm_analysis.get("confidence_score", 0.0),
                "has_known_solution": llm_analysis.get("has_known_solution", False),
                "solution_available": llm_analysis.get("solution_available", False),
                "issue_interpretation": llm_analysis.get("issue_interpretation", request.query),
                "troubleshooting_steps": llm_analysis.get("troubleshooting_steps", None),
                "similar_issues_count": llm_analysis.get("similar_issues_found", 0),

                # ROUTING FIELDS - Direct from LLM (FLAT, NO "suggestions" wrapper)
                "category": llm_analysis.get("category", None),
                "queue": llm_analysis.get("queue", None),
                "priority": llm_analysis.get("priority", None),

                # CONVERSATION GUIDANCE - Direct from LLM (FLAT)
                "message": f"""Your ticket with ID: {ticket_id} is now OPEN with status 'active'. 
                Save the ticket id because it is very important. You will need it when submitting the ticket. 
                Now, move immediately to collecting information. Make sure to complete active ticket""",
                "reasoning": llm_analysis.get("reasoning", "Need more information to proceed"),
                "next_step": llm_analysis.get("next_step", "collect_info"),

                # DIAGNOSTIC QUESTIONS - Direct from LLM
                "must_ask_diagnostic_questions": llm_analysis.get("must_ask_diagnostic_questions", []),

                # METADATA - Direct from LLM (FLAT)
                "estimated_resolution_time": llm_analysis.get("estimated_resolution_time", None),
                "escalation_recommended": llm_analysis.get("escalation_recommended", False),

                # TICKET STATUS with DYNAMIC missing fields
                "ticket_status": {
                    "status": "active",
                    "is_complete": False,
                    "missing_fields": new_ticket_missing_fields,  # DYNAMICALLY CALCULATED
                },

                "should_use_ticket_continue": True
            }

            print(f"This is the original response: {response}")

        else:
            # Fallback response when LLM analysis fails
            if DEBUG:
                print(f"[INITIALIZE_TICKET] Using fallback response structure")

            response = {
                "success": True,
                "ticket_id": ticket_id,
                "conversation_id": conversation_id,
                "is_new_ticket": True,

                # KNOWLEDGE BASE FIELDS - Fallback values
                "source": "fallback",
                "protocol_id": None,
                "confidence_score": 0.0,
                "has_known_solution": False,
                "solution_available": False,
                "issue_interpretation": request.query,
                "troubleshooting_steps": None,
                "similar_issues_count": 0,

                # ROUTING FIELDS - No suggestions in fallback
                "category": None,
                "queue": None,
                "priority": None,

                # CONVERSATION GUIDANCE - Fallback messages
                "message": f"Support Ticket {ticket_id} initialized. I'll help you provide the information needed to resolve this issue.",
                "next_step": "collect_info",
                "reasoning": "Gathering information to understand the issue",

                # DIAGNOSTIC QUESTIONS - Generic fallback
                "must_ask_diagnostic_questions": [
                    "What type of issue are you experiencing? (Hardware, Software, Network, Facility, Medical Equipment)",
                    "How urgent is this issue?",
                    "Can you describe what's happening in more detail?"
                ],

                # METADATA - Fallback values
                "estimated_resolution_time": None,
                "escalation_recommended": False,

                # TICKET STATUS with DYNAMIC missing fields
                "ticket_status": {
                    "status": "active",
                    "is_complete": False,
                    "missing_fields": new_ticket_missing_fields,  # DYNAMICALLY CALCULATED
                },

                "should_use_ticket_continue": True
            }

            print(f"This is the fallback response: {response}")

        return response

    except Exception as e:
        if DEBUG:
            print(f"[INITIALIZE_TICKET] Error: {e}")
            import traceback
            traceback.print_exc()

        # Return error response
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initialize ticket. Please try again or contact support.",
            "trigger_fallback": True
        }


# ==================== TICKET OPERATION ====================
@app.post("/tickets/{ticket_id}/submit", response_model=SubmitTicketResponse)
async def submit_ticket_endpoint(
        ticket_id: str,
        request: SubmitTicketRequest,
        ad: FastActiveDirectory = Depends(get_ad)
) -> SubmitTicketResponse:
    """
    Submit ticket to JIRA with permission check and escalation.

    Flow:
    1. Validates ticket completeness
    2. Checks if user has permission to create tickets (member of "Ticket Creators" group)
    3. If no permission: finds appropriate escalator from shared groups
    4. Creates JIRA ticket (with escalation info if applicable)
    5. Returns appropriate message to user

    Args:
        ticket_id: Unique ticket identifier
        request: Ticket submission details
        ad: Active Directory client (injected dependency)

    Returns:
        SubmitTicketResponse with ticket info, JIRA result, and escalation details
    """
    try:
        if DEBUG:
            print(f"[DEBUG] Received request to submit ticket: {ticket_id}")
            print(f"[DEBUG] Request payload: {request.model_dump()}")

        # ==================== GET AND VALIDATE TICKET ====================
        ticket = await ticket_manager.get_ticket(ticket_id)
        if DEBUG:
            print(f"[DEBUG] Fetched ticket: {ticket}")

        if not ticket:
            if DEBUG:
                print(f"[DEBUG] Ticket not found: {ticket_id}")
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

        # Check if already submitted
        if ticket['status'] in [TicketStatus.SUBMITTED, TicketStatus.COMPLETED]:
            if DEBUG:
                print(f"[DEBUG] Ticket already in status: {ticket['status']}")

            return SubmitTicketResponse(
                success=False,
                ticket=TicketResponse(**ticket),
                jira_result={"error": f"Ticket already {ticket['status']}"},
                ticket_escalator=None,
                escalated=False,
                message=f"Ticket already {ticket['status']}"
            )

        # Update all fields from request
        if DEBUG:
            print(f"[DEBUG] Updating ticket fields for {ticket_id}")
        await ticket_manager.update_fields(ticket_id, request.model_dump())

        # Re-fetch updated ticket
        ticket = await ticket_manager.get_ticket(ticket_id)
        if DEBUG:
            print(f"[DEBUG] Ticket after update: {ticket}")

        if not ticket['is_complete']:
            if DEBUG:
                print(f"[DEBUG] Ticket incomplete: missing fields {ticket['missing_fields']}")
            raise HTTPException(
                status_code=400,
                detail=f"Ticket incomplete. Missing: {', '.join(ticket['missing_fields'])}"
            )

        # ==================== PERMISSION CHECK & ESCALATION LOGIC ====================
        ticket_escalator = None
        escalation_message = None
        escalated_to_name = ""
        escalated_to_email = ""
        user_identifier = request.reporter_name

        if DEBUG:
            print(f"[DEBUG] ==================== PERMISSION CHECK START ====================")
            print(f"[DEBUG] Checking permissions for user: {user_identifier}")

        try:
            # Check if user belongs to "Ticket Creators" group
            is_ticket_creator = await ad.check_user_membership(
                user_identifier=user_identifier,
                group_identifier="Ticket Creator" # The Ticket Creator Group
            )

            if DEBUG:
                print(f"[DEBUG] User '{user_identifier}' is ticket creator: {is_ticket_creator}")

            if not is_ticket_creator:
                if DEBUG:
                    print(f"[DEBUG] ⚠️  User lacks permission - initiating escalation process")

                # Step 1: Get user's groups
                if DEBUG:
                    print(f"[DEBUG] Fetching user's group memberships...")

                user_groups = await ad.get_user_groups_smart(
                    user_identifier=user_identifier,
                    transitive=False
                )

                if DEBUG:
                    print(f"[DEBUG] User belongs to {len(user_groups)} groups:")
                    for g in user_groups[:5]:  # Log first 5 groups
                        print(f"[DEBUG]   - {g.get('displayName')} (ID: {g.get('id')})")

                # Step 2: Get members of "Ticket Creators" group
                if DEBUG:
                    print(f"[DEBUG] Fetching members of 'Ticket Creators' group...")

                ticket_creators = await ad.get_group_members_smart(
                    group_identifier="Ticket Creator"
                )

                if DEBUG:
                    print(f"[DEBUG] Found {len(ticket_creators)} ticket creators:")

                    for tc in ticket_creators[:5]:  # Log first 5
                        print(f"[DEBUG]   - {tc.get('displayName')} ({tc.get('userPrincipalName')})")

                if not ticket_creators:
                    error_msg = "No ticket creators found in the system. Please contact IT support."

                    if DEBUG:
                        print(f"[ERROR] {error_msg}")

                    raise HTTPException(status_code=500, detail=error_msg)

                # Step 3: Find escalator - someone who is in Ticket Creators AND shares a group with the user
                if DEBUG:
                    print(f"[DEBUG] Finding best escalator (someone who shares groups with user)...")

                user_group_ids = {g['id'] for g in user_groups}
                potential_escalators = []

                for creator in ticket_creators:
                    if DEBUG:
                        print(f"[DEBUG] Checking creator: {creator.get('displayName')}")

                    try:
                        creator_groups = await ad.get_user_groups_smart(
                            user_identifier=creator['id'],
                            transitive=False
                        )

                        creator_group_ids = {g['id'] for g in creator_groups}

                        # Check for common groups
                        common_groups = user_group_ids.intersection(creator_group_ids)

                        if common_groups:
                            potential_escalators.append({
                                'user': creator,
                                'common_groups_count': len(common_groups),
                                'common_groups': common_groups
                            })

                            if DEBUG:
                                print(f"[DEBUG]   ✓ {creator.get('displayName')} shares {len(common_groups)} group(s)")

                        else:
                            if DEBUG:
                                print(f"[DEBUG]   ✗ {creator.get('displayName')} shares no groups")

                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG]   ✗ Error checking {creator.get('displayName')}: {e}")
                        continue

                if DEBUG:
                    print(f"[DEBUG] Found {len(potential_escalators)} potential escalators with shared groups")

                # Step 4: Select the best escalator
                if potential_escalators:
                    # Sort by most common groups (better match = more shared context)
                    potential_escalators.sort(
                        key=lambda x: x['common_groups_count'],
                        reverse=True
                    )

                    escalator = potential_escalators[0]['user']

                    if DEBUG:
                        print(f"[DEBUG] ✓ Selected escalator: {escalator.get('displayName')} "
                              f"(shares {potential_escalators[0]['common_groups_count']} groups)")

                else:
                    # Fallback: just use first ticket creator
                    escalator = ticket_creators[0]

                    if DEBUG:
                        print(f"[DEBUG] ⚠️  No shared groups found. Using fallback: {escalator.get('displayName')}")

                # Build escalator info
                ticket_escalator = EscalatorInfo(
                    id=escalator['id'],
                    displayName=escalator.get('displayName', 'Unknown'),
                    userPrincipalName=escalator.get('userPrincipalName', ''),
                    email=escalator.get('userPrincipalName', '')
                )

                escalated_to_name = ticket_escalator.displayName
                escalated_to_email = ticket_escalator.email

                escalation_message = (
                    f"Your ticket has been successfully sent to {ticket_escalator.displayName} "
                    f"({ticket_escalator.email}), who is responsible for creating tickets. "
                    f"Don't worry - they will review your ticket and create it on your behalf. "
                    f"You will receive additional information once the ticket is processed."
                )

                if DEBUG:
                    print(f"[DEBUG] ✓ Escalation complete:")
                    print(f"[DEBUG]   Escalator: {ticket_escalator.displayName}")
                    print(f"[DEBUG]   Email: {ticket_escalator.email}")
                    print(f"[DEBUG] ==================== PERMISSION CHECK END ====================")

            else:
                if DEBUG:
                    print(f"[DEBUG] ✓ User has permission to create tickets directly")
                    print(f"[DEBUG] ==================== PERMISSION CHECK END ====================")

        except HTTPException:
            # Re-raise HTTP exceptions (like 500 for no ticket creators)
            raise

        except Exception as e:
            if DEBUG:
                print(f"[ERROR] ==================== PERMISSION CHECK FAILED ====================")
                print(f"[ERROR] Permission check error: {e}")
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Continuing with ticket creation (degraded mode)")
                print(f"[ERROR] ==================== PERMISSION CHECK END ====================")

            # Continue with ticket creation even if permission check fails
            # This ensures system resilience - better to create ticket than fail completely

        # ==================== CREATE JIRA TICKET ====================
        if DEBUG:
            print(f"[DEBUG] ==================== JIRA CREATION START ====================")

        jira_payload = {
            "conversation_id": ticket['conversation_id'],
            "conversation_topic": request.conversation_topic,
            "description": request.description,
            "location": request.location,
            "queue": request.queue,
            "priority": request.priority,
            "department": request.department,
            "name": request.reporter_name,
            "category": request.category,
            "escalated_to": escalated_to_name,
            "escalated_to_email": escalated_to_email
        }

        if DEBUG:
            print(f"[DEBUG] JIRA Payload:")
            print(f"[DEBUG]   Topic: {jira_payload['conversation_topic']}")
            print(f"[DEBUG]   Location: {jira_payload['location']}")
            print(f"[DEBUG]   Queue: {jira_payload['queue']}")
            print(f"[DEBUG]   Priority: {jira_payload['priority']}")
            print(f"[DEBUG]   Category: {jira_payload['category']}")
            print(f"[DEBUG]   Reporter: {jira_payload['name']}")
            if escalated_to_name:
                print(f"[DEBUG]   ⚠️  ESCALATED TO: {escalated_to_name} ({escalated_to_email})")

        jira_result = await asyncio.to_thread(create_jira_ticket, **jira_payload)

        if DEBUG:
            print(f"[DEBUG] JIRA Result: {jira_result}")
            print(f"[DEBUG] ==================== JIRA CREATION END ====================")

        # Extract JIRA key
        jira_key = None
        if isinstance(jira_result, dict):
            jira_key = jira_result.get('key') or jira_result.get('jira_key')

        if DEBUG:
            print(f"[DEBUG] Extracted JIRA key: {jira_key}")

        # Check if JIRA creation was successful
        if isinstance(jira_result, dict) and not jira_result.get('success', True):
            if DEBUG:
                print(f"[ERROR] JIRA creation failed: {jira_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create JIRA ticket: {jira_result.get('error', 'Unknown error')}"
            )

        # ==================== UPDATE TICKET STATUS ====================
        if DEBUG:
            print(f"[DEBUG] Updating ticket status to SUBMITTED")

        ticket = await ticket_manager.update_status(
            ticket_id,
            TicketStatus.SUBMITTED,
            jira_key=jira_key
        )

        if DEBUG:
            print(f"[DEBUG] Final ticket data: {ticket}")

        # ==================== BUILD RESPONSE ====================
        response_message = escalation_message if escalation_message else "Ticket successfully created in JIRA"

        if DEBUG:
            print(f"[DEBUG] ==================== RESPONSE SUMMARY ====================")
            print(f"[DEBUG] Success: True")
            print(f"[DEBUG] Ticket ID: {ticket_id}")
            print(f"[DEBUG] JIRA Key: {jira_key}")
            print(f"[DEBUG] Escalated: {bool(ticket_escalator)}")
            if ticket_escalator:
                print(f"[DEBUG] Escalator: {ticket_escalator.displayName} ({ticket_escalator.email})")
            print(f"[DEBUG] Message: {response_message}")
            print(f"[DEBUG] ==================== REQUEST COMPLETE ====================")

        return SubmitTicketResponse(
            success=True,
            ticket=TicketResponse(**ticket),
            jira_result=jira_result,
            ticket_escalator=ticket_escalator,
            escalated=bool(ticket_escalator),
            message=response_message
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        if DEBUG:
            print(f"[ERROR] ==================== FATAL ERROR ====================")
            print(f"[ERROR] Exception while submitting ticket: {e}")
            import traceback
            traceback.print_exc()
            print(f"[ERROR] ==================== ERROR END ====================")

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tickets/{ticket_id}/cancel")
async def cancel_ticket_endpoint(ticket_id: str):
    """Cancel a ticket"""
    try:
        ticket = await ticket_manager.update_status(ticket_id, TicketStatus.CANCELLED)
        return TicketResponse(**ticket)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        if DEBUG:
            print(f"[API] Error cancelling ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MCP TOOLS ====================

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
                    "name": {"type": "string", "description": "User's name", "default": ""},
                    "chat_id": {"type": "string", "description": "Conversation ID", "default": ""}
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
                            "mailNickname": {"type": "string",
                                             "description": "Mail nickname (auto-generated if not provided)"},
                            "userPrincipalName": {"type": "string",
                                                  "description": "User principal name (auto-generated if not provided)"},
                            "passwordProfile": {
                                "type": "object",
                                "properties": {
                                    "password": {"type": "string", "description": "Temporary password"},
                                    "forceChangePasswordNextSignIn": {"type": "boolean",
                                                                      "description": "Force password change on next sign-in"}
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
                    "transitive": {"type": "boolean", "description": "Include transitive AD group memberships",
                                   "default": False}
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
                    "security_only": {"type": "boolean", "description": "List only AD security groups",
                                      "default": False},
                    "unified_only": {"type": "boolean", "description": "List only AD unified groups", "default": False},
                    "select": {"type": "string", "description": "AD fields to select",
                               "default": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
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
                    "description": {"type": "string", "description": "AD group description"},
                    "group_type": {"type": "string", "enum": ["security", "m365", "dynamic-security", "dynamic-m365"],
                                   "description": "Azure AD group type", "default": "security"},
                    "visibility": {"type": "string", "enum": ["Private", "Public"],
                                   "description": "AD group visibility (for M365 groups)"},
                    "membership_rule": {"type": "string",
                                        "description": "Dynamic membership rule (for dynamic groups)"},
                    "owners": {"type": "array", "items": {"type": "string"},
                               "description": "List of owner identifiers (names, emails, or GUIDs)"},
                    "members": {"type": "array", "items": {"type": "string"},
                                "description": "List of member identifiers (names, emails, or GUIDs)"}
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
                    "include_transitive": {"type": "boolean", "description": "Include transitive group memberships",
                                           "default": False},
                    "include_owned": {"type": "boolean", "description": "Include groups owned by user",
                                      "default": True},
                    "select": {"type": "string", "description": "User fields to select",
                               "default": "id,displayName,userPrincipalName"}
                },
                "required": ["include_transitive", "include_owned", "select"]
            }
        ),

        # ==================== TICKET SYSTEM TOOLS ====================
        MCPTool(
            name="ticket_open",
            description=(
                "SCOPE: Use this tool when a user first reports a NEW problem, issue, or incident. Handles initial problem intake for any type of issue (technical, facility, equipment, or service-related).\n\n"
                "TRIGGER: new issue, report problem, something broken, not working, need help, create ticket, open ticket, start ticket\n\n"
                "ACTION: Initialize a new ticket when an issue is reported for the FIRST time. "
                "Creates a new ticket record and returns comprehensive ticket information with guidance, "
                "progress tracking, diagnostic question tracking, and next steps.\n\n"
                "INSTRUCTION: The issue must be completely unique and unrelated to other existing issues.\n\n"
                "SYSTEM ROUTING OPTIONS:\n"
                "- Technical: For IT/computer/software/network equipment issues. Examples: desktop computers, laptops, printers, phones, software applications, network connectivity, servers\n"
                "- Service Desk: For administrative tasks, account access, password resets, general inquiries, and unclear routing situations (DEFAULT)\n"
                "- Facilities: For BUILDING INFRASTRUCTURE only - walls, floors, ceilings, doors, windows, HVAC, plumbing, electrical wiring, lighting, locks, painting, structural repairs. NOT for medical or office equipment\n"
                "- Incident Report: For CRITICAL MEDICAL EQUIPMENT failures or major operational disruptions that impact patient care delivery. Examples: MRI scanners, CT scanners, X-ray machines, ventilators, surgical equipment, lab analyzers, patient monitoring systems. Use when equipment failure disrupts hospital operations or patient treatment"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the problem/issue (REQUIRED for initialization)"
                    },
                    "systems": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Technical", "Service Desk", "Facilities", "Incident Report"]
                        },
                        "description": (
                            "One or more systems to route this ticket to:\n"
                            "• Technical - IT equipment only: computers, laptops, printers, phones, software, networks, servers\n"
                            "• Service Desk - Administrative tasks, account issues, password resets, general inquiries (DEFAULT when unclear)\n"
                            "• Facilities - BUILDING INFRASTRUCTURE ONLY: walls, floors, HVAC, plumbing, electrical wiring, doors, windows, locks. NOT for equipment or devices\n"
                            "• Incident Report - CRITICAL MEDICAL EQUIPMENT failures affecting patient care: MRI/CT/X-ray machines, ventilators, surgical equipment, lab analyzers, patient monitors, dialysis machines"
                        ),
                        "default": ["Service Desk"],
                        "minItems": 1
                    },
                    "reporter_name": {
                        "type": "string",
                        "description": "Name of person reporting issue (optional)",
                        "default": "Jean Gray"
                    },
                    "reporter_email": {
                        "type": "string",
                        "description": "Email of person reporting issue (optional)",
                        "default": "jeangray@lovenoreusgmail.onmicrosoft.com"
                    },
                    "chat_id": {
                        "type": "string",
                        "description":
                            "Conversation ID",
                        "default": ""
                    }
                },
                "required": ["query", "systems"]
            }
        ),

        MCPTool(
            name="ticket_submit",
            description=(
                "SCOPE: Use this tool ONLY after the user has explicitly confirmed they want to submit the ticket. "
                "This tool performs the final submission of a complete ticket to JIRA after user verification and approval.\n\n"
                "TRIGGER: User explicitly confirms submission after being asked. Examples of confirmation phrases:\n"
                "  - 'yes' or 'yes submit'\n"
                "  - 'submit it' or 'submit the ticket'\n"
                "  - 'go ahead' or 'proceed'\n"
                "  - 'confirm' or 'confirmed'\n"
                "  - 'please submit'\n"
                "DO NOT trigger on: ticket being complete, all questions answered, or ticket being 'ready'. "
                "Wait for explicit user confirmation.\n\n"
                "ACTION: Submit the completed ticket to JIRA with all gathered information.\n\n"
                "INSTRUCTION: This tool should only be called after:\n"
                "  1. All required information has been collected\n"
                "  2. User has been shown a complete summary of the ticket information\n"
                "  3. User has been asked 'Would you like me to submit this ticket?'\n"
                "  4. User has explicitly confirmed they want to submit\n"
                "Validates all required fields (description, category, priority, etc.) are filled before submission. "
                "Cannot submit tickets that are already submitted or completed.\n\n"
                "RETURNS: JIRA ticket key (for tracking in JIRA), submission confirmation, and final ticket status"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket ID to submit to JIRA"
                    },
                    "conversation_topic": {
                        "type": "string",
                        "description": "Brief title/summary for JIRA ticket"
                    },
                    "description": {
                        "type": "string",
                        "description": "Complete detailed description for JIRA"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location for JIRA ticket",
                        "default": "Room 100000"
                    },
                    "queue": {
                        "type": "string",
                        "description": "Support queue for JIRA routing",
                        "default": "Tech Support"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["High", "Normal", "Low"],
                        "description": "Priority level for JIRA (If not provided, generate base on reported issue)",
                        "default": "Medium"
                    },
                    "department": {
                        "type": "string",
                        "description": "Department for JIRA ticket",
                        "default": "Neurology"
                    },
                    "reporter_name": {
                        "type": "string",
                        "description": "Reporter name for JIRA",
                        "default": "Blade"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Hardware", "Software", "Facility", "Network", "Medical Equipment", "Other"],
                        "description": "Category for JIRA ticket",
                        "default": "Other"
                    },
                    "user_submitting": {
                        "type": "boolean",
                        "description": "User explicitly stating or agreeing that the ticket should be submitted. If not True, do not (NEVER) attempt to submit ticket.",
                        "default": False,
                    },
                    "chat_id": {
                        "type": "string",
                        "description":
                        "Conversation ID",
                        "default": ""
                    }
                },
                "required": ["ticket_id", "conversation_topic", "description", "location", "queue", "priority",
                             "department", "reporter_name", "category", "user_submitting"]
            }
        ),

        MCPTool(
            name="ticket_cancel",
            description=(
                "SCOPE: Use this tool when a user wants to stop or remove an existing ticket. Handles ticket cancellation for resolved, duplicate, or unwanted issues.\n\n"
                "TRIGGER: cancel ticket, close ticket, abort ticket, discard ticket, delete ticket, remove ticket, nevermind, false alarm\n\n"
                "ACTION: Cancel an existing ticket (changes status to cancelled).\n\n"
                "RETURNS: Cancelled ticket confirmation with updated status"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket ID to cancel"
                    },
                    "chat_id": {
                        "type": "string",
                        "description":
                            "Conversation ID",
                        "default": ""
                    }
                },
                "required": ["ticket_id"]
            }
        )
    ]

    return MCPToolsListResponse(tools=tools)


def process_questions_generic(answered_questions: dict[str, str], must_ask_questions: list[str]) -> dict:
    """
    Filters must-ask questions to identify which ones remain unanswered.

    Args:
        answered_questions: Dict mapping questions to their answers.
        must_ask_questions: List of all required diagnostic questions.

    Returns:
        Diagnostic tracking dictionary including answered and remaining questions.
    """
    answered_set = set(answered_questions.keys())
    remaining_questions = [q for q in must_ask_questions if q not in answered_set]

    return {
        "questions_asked": list(answered_set),
        "questions_answered": answered_questions,
        "questions_remaining": remaining_questions,
        "auto_extracted_info": {}  # You can expand this later if needed
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Removed all database checks and related info

        return {
            "status": "healthy",
            "service": "Invoice MCP Server",
            "timestamp": datetime.now().isoformat(),
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


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def mcp_tools_call(request: MCPToolCallRequest):
    """MCP Protocol: Call a specific tool"""
    try:
        tool_name = request.name
        arguments = request.arguments

        if DEBUG:
            print(f"[MCP] Calling tool: {tool_name} with args: {arguments}")

        # ==================== CONVERSATIONAL ====================
        if tool_name == "greet":
            raw_name = arguments.get("name") if arguments else None
            clean_name = raw_name if raw_name and raw_name.strip() else None
            greet_request = GreetRequest(name=clean_name)
            result = await greet_endpoint(greet_request)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        # ==================== AD TOOLS ====================
        # ==================== USER MANAGEMENT ====================
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
            user_identifier = arguments["user_identifier"]
            user_updates = UserUpdates(updates=arguments["updates"])
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.update_user_smart(user_identifier, user_updates.updates)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_delete_user":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.delete_user_smart(user_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_roles":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_user_roles_smart(user_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_groups":
            user_identifier = arguments["user_identifier"]
            transitive = arguments.get("transitive", False)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_user_groups_smart(user_identifier, transitive=transitive)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_user_full_profile":
            user_identifier = arguments["user_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_user_full_profile(user_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_search_users":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.search_users_fuzzy(query, limit=limit)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        # ==================== ROLE MANAGEMENT ====================
        elif tool_name == "ad_list_roles":
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await list_roles_endpoint(ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_user_to_role":
            user_identifier = arguments["user_identifier"]
            role_identifier = arguments["role_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.add_user_to_role_smart(user_identifier, role_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_user_from_role":
            user_identifier = arguments["user_identifier"]
            role_identifier = arguments["role_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.remove_user_from_role_smart(user_identifier, role_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

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
                "success_count": success_count,
                "error_count": error_count,
                "results": [str(r) if isinstance(r, Exception) else "Success" for r in result]
            }
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(formatted_result, indent=2))]
            )

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
                "success_count": success_count,
                "error_count": error_count,
                "results": [str(r) if isinstance(r, Exception) else "Success" for r in result]
            }
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(formatted_result, indent=2))]
            )

        # ==================== GROUP MANAGEMENT ====================
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
                membership_rule=arguments.get("membership_rule"),
                owners=arguments.get("owners"),
                members=arguments.get("members")
            )
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await create_group_endpoint(create_group_request, ad)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_group_member":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.add_user_to_group_smart(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_group_member":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.remove_user_from_group_smart(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_group_members":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_group_members_smart(group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_get_group_owners":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.get_group_owners_smart(group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_add_group_owner":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.add_owner_to_group_smart(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_remove_group_owner":
            user_identifier = arguments["user_identifier"]
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.remove_owner_from_group_smart(user_identifier, group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_update_group":
            group_identifier = arguments["group_identifier"]
            updates = arguments["updates"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.update_group_smart(group_identifier, updates)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif tool_name == "ad_delete_group":
            group_identifier = arguments["group_identifier"]
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await ad.delete_group_smart(group_identifier)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
            )

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


        # ==================== TICKET SYSTEM TOOLS ====================
        elif tool_name == "ticket_open":
            try:
                conversation_id = arguments.get("chat_id", "")
                query = arguments.get("query", "")

                print(f'The conversation id: {conversation_id}')

                if not conversation_id:
                    return MCPToolCallResponse(
                        content=[MCPContent(
                            type="text",
                            text=json.dumps({
                                "success": False,
                                "error": "conversation_id is required",
                                "message": "Please provide the conversation_id to track this ticket",
                                "required_fields": ["conversation_id", "query"]
                            }, indent=2)
                        )],
                        isError=True
                    )

                if not query:
                    return MCPToolCallResponse(
                        content=[MCPContent(
                            type="text",
                            text=json.dumps({
                                "success": False,
                                "error": "query is required for initialization",
                                "message": "Please provide a description of the issue to initialize a ticket",
                                "example": {
                                    "conversation_id": "00001",
                                    "query": "Printer in room 305 is not working"
                                }
                            }, indent=2)
                        )],
                        isError=True
                    )

                if DEBUG:
                    print(f"[MCP] ticket_open: Initializing ticket with query: {query[:50]}...")

                reporter_name = arguments.get("name", "")

                print(f"Reported with name: {reporter_name}")

                if not reporter_name:
                    reporter_name =arguments.get("reporter_name", "")

                    print(f"Reported with reporter_name: {reporter_name}")

                init_request = InitializeTicketRequest(
                    query=query,
                    conversation_id=conversation_id,
                    reporter_name=reporter_name,
                    reporter_email=arguments.get("reporter_email", "")
                )

                result = await initialize_ticket_endpoint(init_request)

                # TODO: GET SYSTEM AND SEARCH FOR INCIDENT RESPONSE.
                try:
                    systems = arguments.get("systems", [])
                    print(systems)

                except:
                    print('No systems!')
                    systems = []

                if "Incident Report" in systems:
                    # TODO: INCIDENT QUESTIONS.
                    incident_questions = [
                        "Were any of the following factors involved? (Medical device, Assistive device, Radiation/Hospital physics, IT, Telephony, Materials/other equipment, Medication)",
                        "Date and time of the event",
                        "What injury occurred or could have occurred?",
                        "What happened or could have happened? (Do not include any sensitive or confidential information, such as personal identity numbers.)",
                        "Was a patient affected? (Search for the person here.)",
                        "Why did the event occur? Describe the likely cause of the incident.",
                        "What actions do you think should be taken? Please suggest measures to prevent recurrence of the incident.",
                        "What actions have been taken? Describe any immediate actions taken directly in connection with the incident."
                    ]

                    # Process diagnostic tracking for initialization
                    must_ask_questions = result.get("must_ask_diagnostic_questions", []).extend(incident_questions)

                else:
                    must_ask_questions = result.get("must_ask_diagnostic_questions", [])

                result.update({
                    "should_use_ticket_continue": True,
                    "must_ask_diagnostic_questions": must_ask_questions
                })

                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False)
                    )],
                    isError=not result.get("success", False)
                )

            except ValidationError as ve:
                if DEBUG:
                    print(f"[MCP] Validation error for ticket_open: {ve}")
                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": "Validation failed",
                            "details": str(ve),
                            "message": "Invalid field values provided"
                        }, indent=2)
                    )],
                    isError=True
                )

            except Exception as e:
                if DEBUG:
                    print(f"[MCP] Error in ticket_open: {e}")
                    import traceback
                    traceback.print_exc()
                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": str(e),
                            "message": "An unexpected error occurred"
                        }, indent=2)
                    )],
                    isError=True
                )


        elif tool_name == "ticket_submit":
            print(arguments.get("user_submitting"))

            if not arguments.get("user_submitting", False):
                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": str(e),
                            "message": "User does not want to submit ticket. Continue asking all questions or ask what user wants"
                        }, indent=2)
                    )],
                    isError=True
                )

            ticket_id = arguments["ticket_id"]
            submit_data = {k: v for k, v in arguments.items() if k != "ticket_id" and k != "user_submitting"}
            submit_request = SubmitTicketRequest(**submit_data)

            # Create AD instance and pass it to the endpoint - same pattern as ad_create_user
            async with FastActiveDirectory(max_concurrent=20) as ad:
                result = await submit_ticket_endpoint(ticket_id, submit_request, ad)

            # Convert result to dict if needed
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result

            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result_dict, indent=2))],
                isError=not result_dict.get("success", False)

            )

        elif tool_name == "ticket_cancel":
            ticket_id = arguments["ticket_id"]
            result = await cancel_ticket_endpoint(ticket_id)
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
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
        chat_id = body.get("params", {}).get("chat_id")
        if chat_id:
            print(f"Received chat_id: {chat_id}")

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
                        "name": "Hospital MCP Server with Ticketing System",
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
                "result": {"tools": [tool.model_dump() for tool in tools_response.tools]}
            }
            if DEBUG:
                print(f"[STREAMABLE HTTP] Tools list response with {len(tools_response.tools)} tools")
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
                print(f"[STREAMABLE HTTP] Tools call response for {call_request.name}")
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
            import traceback
            traceback.print_exc()
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.get("/info")
async def server_info():
    """Server information endpoint"""
    data = await storage.load()

    return {
        "service": "Hospital MCP Server with Ticket System",
        "version": "1.0.0",
        "description": "Standalone MCP Tools Server with Azure AD management and hospital ticketing system integrated with JIRA",
        "protocols": ["REST API", "MCP (Model Context Protocol)", "Streamable HTTP"],

        "mcp_endpoints": {
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call",
            "streamable_http": "/"
        },

        "rest_endpoints": {
            "general": [
                "/health",
                "/info",
                "/docs"
            ],
            "greet": [
                "/greet"
            ],
            "active_directory": [
                "/ad/users",
                "/ad/users/{user_id}",
                "/ad/users/{user_id}/roles",
                "/ad/users/{user_id}/groups",
                "/ad/users/{user_id}/owned-groups",
                "/ad/users-with-groups",
                "/ad/users/batch/groups",
                "/ad/roles",
                "/ad/roles/{role_id}/members",
                "/ad/roles/{role_id}/members/{user_id}",
                "/ad/roles/instantiate",
                "/ad/groups",
                "/ad/groups/{group_id}",
                "/ad/groups/{group_id}/members",
                "/ad/groups/{group_id}/owners"
            ],
            "tickets": [
                "/tickets/create",
                "/tickets/{ticket_id}",
                "/tickets/{ticket_id}/submit",
                "/tickets/{ticket_id}/cancel",
                "/tickets/search",
                "/tickets/stats",
                "/threads/{conversation_id}/tickets",
                "/threads/{conversation_id}/active-ticket",
                "/storage/backup"
            ]
        },

        "features": [
            "MCP Protocol Support",
            "Streamable HTTP Transport",
            "Azure Active Directory Management",
            "Hospital Ticket System with JIRA Integration",
            "Qdrant Vector Database Knowledge Base",
            "Priority-based SLA Tracking",
            "Auto-routing by Category",
            "JSON Persistent Storage",
            "Complete Audit History",
            "Batch Operations Support",
            "Smart User/Group/Role Resolution"
        ],

        "tools": {
            "conversational": [
                "greet"
            ],
            "active_directory_users": [
                "ad_list_users",
                "ad_create_user",
                "ad_update_user",
                "ad_delete_user",
                "ad_get_user_roles",
                "ad_get_user_groups",
                "ad_get_user_full_profile",
                "ad_search_users"
            ],
            "active_directory_roles": [
                "ad_list_roles",
                "ad_add_user_to_role",
                "ad_remove_user_from_role",
                "ad_batch_add_users_to_role",
                "ad_batch_remove_users_from_role"
            ],
            "active_directory_groups": [
                "ad_list_groups",
                "ad_create_group",
                "ad_add_group_member",
                "ad_remove_group_member",
                "ad_get_group_members",
                "ad_get_group_owners"
            ],
            "ticket_system": [
                "ticket_open",
                "ticket_submit",
                "ticket_cancel"
            ]
        },

        "ticket_system": {
            "storage": {
                "type": "JSON",
                "file": str(storage.filepath),
                "total_tickets": len(data.get("tickets", {})),
                "total_threads": len(data.get("thread_index", {}))
            },
            "validation": {
                "required_fields": TicketManager.REQUIRED_FIELDS,
                "allowed_fields": list(TicketManager.ALLOWED_FIELDS)
            },
            "priorities": {
                "values": TicketManager.VALID_PRIORITIES,
                "sla": {
                    "High": "8 hours",
                    "Normal": "24 hours",
                    "Low": "72 hours"
                }
            },
            "categories": {
                # "values": TicketManager.VALID_CATEGORIES,
                "routing": {
                    "Hardware": "Hardware Support",
                    "Software": "IT Support",
                    "Facility": "Facilities Management",
                    "Network": "Network Operations",
                    "Medical Equipment": "Biomedical Engineering",
                    "Other": "General Support"
                }
            },
            "statuses": [
                "active",
                "pending",
                "submitted",
                "completed",
                "cancelled"
            ],
            "integrations": [
                "JIRA Ticketing System",
                "Qdrant Knowledge Base"
            ]
        },

        "active_directory": {
            "tenant": "lovenoreusgmail.onmicrosoft.com",
            "features": [
                "User Management",
                "Role Assignment",
                "Group Management",
                "Batch Operations",
                "Smart Resolution (Name/Email/GUID)"
            ]
        },

        "docs": "/docs",
        "mcp_compatible": True,
        "debug_mode": DEBUG
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Hospital MCP Server with Ticket System")
    print("=" * 60)
    print(f"Port: 8009")
    print(f"Debug Mode: {DEBUG}")
    print(f"Ticket Storage: {TICKETS_FILE}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8009)
