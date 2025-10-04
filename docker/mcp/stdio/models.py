# -------------------- Built-in Libraries --------------------
from typing import Dict, Optional, List, Any, Literal

# -------------------- External Libraries --------------------
from pydantic import BaseModel, Field, ValidationError


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


# Request models
class GreetRequest(BaseModel):
    name: Optional[str]

