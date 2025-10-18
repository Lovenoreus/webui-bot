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


class EscalatorInfo(BaseModel):
    """Information about the person the ticket was escalated to"""
    id: str = Field(..., description="User ID (GUID)")
    displayName: str = Field(..., description="Full name of the escalator")
    userPrincipalName: str = Field(..., description="User Principal Name (email format)")
    email: str = Field(..., description="Email address")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "displayName": "Jane Smith",
                "userPrincipalName": "jane.smith@hospital.com",
                "email": "jane.smith@hospital.com"
            }
        }


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


class SubmitTicketResponse(BaseModel):
    """Response model for ticket submission with escalation support"""
    success: bool = Field(..., description="Whether the submission was successful")
    ticket: TicketResponse = Field(..., description="Complete ticket information")
    jira_result: Dict[str, Any] = Field(..., description="JIRA API response")
    ticket_escalator: Optional[EscalatorInfo] = Field(
        None,
        description="Information about escalator if ticket was escalated (null if not escalated)"
    )
    escalated: bool = Field(
        False,
        description="Whether the ticket was escalated due to insufficient permissions"
    )
    message: str = Field(
        ...,
        description="User-facing message about ticket status and next steps"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Normal Ticket Creation",
                    "value": {
                        "success": True,
                        "ticket": {
                            "ticket_id": "ticket_12345",
                            "conversation_id": "conv_67890",
                            "status": "submitted",
                            "created_at": "2025-10-18T10:30:00Z",
                            "updated_at": "2025-10-18T10:30:00Z",
                            "jira_key": "HEAL-456",
                            "fields": {
                                "conversation_topic": "Printer Issue",
                                "description": "Printer not responding",
                                "location": "Room 305",
                                "queue": "IT Support",
                                "priority": "Medium",
                                "department": "Radiology",
                                "reporter_name": "John Doe",
                                "category": "Hardware"
                            },
                            "is_complete": True,
                            "missing_fields": [],
                            "history": []
                        },
                        "jira_result": {
                            "success": True,
                            "key": "HEAL-456",
                            "jira_key": "HEAL-456",
                            "message": "✅ Ticket HEAL-456 has been created successfully!",
                            "escalated": False,
                            "escalated_to": None
                        },
                        "ticket_escalator": None,
                        "escalated": False,
                        "message": "Ticket successfully created in JIRA"
                    }
                },
                {
                    "name": "Escalated Ticket Creation",
                    "value": {
                        "success": True,
                        "ticket": {
                            "ticket_id": "ticket_12346",
                            "conversation_id": "conv_67891",
                            "status": "submitted",
                            "created_at": "2025-10-18T10:35:00Z",
                            "updated_at": "2025-10-18T10:35:00Z",
                            "jira_key": "HEAL-457",
                            "fields": {
                                "conversation_topic": "Network Outage",
                                "description": "Unable to access network resources",
                                "location": "Building A",
                                "queue": "Network Team",
                                "priority": "High",
                                "department": "Emergency",
                                "reporter_name": "Sarah Johnson",
                                "category": "Network"
                            },
                            "is_complete": True,
                            "missing_fields": [],
                            "history": []
                        },
                        "jira_result": {
                            "success": True,
                            "key": "HEAL-457",
                            "jira_key": "HEAL-457",
                            "message": "✅ Ticket HEAL-457 has been created successfully!",
                            "escalated": True,
                            "escalated_to": "Jane Smith"
                        },
                        "ticket_escalator": {
                            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                            "displayName": "Jane Smith",
                            "userPrincipalName": "jane.smith@hospital.com",
                            "email": "jane.smith@hospital.com"
                        },
                        "escalated": True,
                        "message": "Your ticket has been successfully sent to Jane Smith (jane.smith@hospital.com), who is responsible for creating tickets. Don't worry - they will review your ticket and create it on your behalf. You will receive additional information once the ticket is processed."
                    }
                }
            ]
        }


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


# ==================== ADDITIONAL HELPER MODELS ====================

class TicketFieldUpdate(BaseModel):
    """Model for individual field updates during ticket creation flow"""
    field_name: str = Field(..., description="Name of the field to update")
    field_value: Any = Field(..., description="Value to set for the field")


class TicketValidationResponse(BaseModel):
    """Response model for ticket validation"""
    is_valid: bool = Field(..., description="Whether the ticket has all required fields")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors")


class TicketSearchRequest(BaseModel):
    """Request model for searching tickets"""
    conversation_id: Optional[str] = Field(None, description="Filter by conversation ID")
    status: Optional[str] = Field(None, description="Filter by status")
    reporter_name: Optional[str] = Field(None, description="Filter by reporter name")
    jira_key: Optional[str] = Field(None, description="Filter by JIRA key")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class BulkTicketResponse(BaseModel):
    """Response model for bulk ticket operations"""
    total: int = Field(..., description="Total number of tickets")
    tickets: List[TicketResponse] = Field(..., description="List of tickets")
    has_more: bool = Field(..., description="Whether there are more results")