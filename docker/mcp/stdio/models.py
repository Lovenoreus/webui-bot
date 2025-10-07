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



# ==================== TICKET CREATION MODELS ====================

class TicketStatus:
    DRAFT = "draft"
    PENDING = "pending"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TicketPriority:
    # CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TicketCategory:
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    FACILITY = "Facility"
    NETWORK = "Network"
    MEDICAL_EQUIPMENT = "Medical Equipment"
    OTHER = "Other"


class CreateTicketRequest(BaseModel):
    query: str = Field(..., description="Initial problem description")
    conversation_id: str = Field(..., description="Conversation thread ID")
    reporter_name: Optional[str] = Field(None, description="Name of person reporting")
    reporter_email: Optional[str] = Field(None, description="Email of reporter")


class UpdateTicketRequest(BaseModel):
    fields: Dict[str, Any] = Field(..., description="Fields to update")


class SubmitTicketRequest(BaseModel):
    conversation_topic: str = Field(..., description="Brief summary/title")
    description: str = Field(..., description="Detailed description")
    location: str = Field(..., description="Location where issue occurs")
    queue: str = Field(..., description="Support queue/department")
    priority: str = Field(..., description="Priority level")
    department: str = Field(..., description="Department affected")
    reporter_name: str = Field(..., description="Reporter name")
    category: str = Field(..., description="Issue category")


class TicketResponse(BaseModel):
    ticket_id: str
    conversation_id: str
    status: str
    created_at: str
    updated_at: str
    jira_key: Optional[str] = None
    fields: Dict[str, Any]
    is_complete: bool
    missing_fields: List[str]
    history: List[Dict[str, Any]] = []


class InitializeTicketRequest(BaseModel):
    """Request model for ticket initialization with knowledge base search"""
    query: str = Field(
        ...,
        description="Description of the problem or issue to create ticket for",
        min_length=3,
        max_length=2000
    )
    conversation_id: str = Field(
        ...,
        description="Conversation thread ID for tracking this ticket session"
    )
    reporter_name: Optional[str] = Field(
        None,
        description="Name of person reporting the issue"
    )
    reporter_email: Optional[str] = Field(
        None,
        description="Email address of person reporting the issue"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Printer in room 305 is not responding",
                "conversation_id": "conv_12345",
                "reporter_name": "John Doe",
                "reporter_email": "john.doe@hospital.com"
            }
        }