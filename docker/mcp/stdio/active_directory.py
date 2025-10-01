# from config import CLIENT_ID, TENANT_ID, CLIENT_SECRET, AUTHORITY, SCOPE
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

from db_init import save_user_to_db, get_user_from_db_by_name, upsert_user, view_users, remove_user_fields

# Load environment variables
load_dotenv()

# Get Azure credentials from environment variables
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID") 
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

CLIENT_ID = AZURE_CLIENT_ID
TENANT_ID = AZURE_TENANT_ID
CLIENT_SECRET = AZURE_CLIENT_SECRET
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]


class PasswordProfile(BaseModel):
    password: str = Field(..., description="A strong initial password for the user.")
    forceChangePasswordNextSignIn: bool = Field(True, description="Require user to change password on next sign-in.")

class CreateUserPayload(BaseModel):
    accountEnabled: bool = Field(..., description="Whether the account is enabled.")
    displayName: str = Field(..., description="Full name of the user.")
    mailNickname: str = Field(..., description="Alias used to generate the email address.")
    userPrincipalName: str = Field(..., description="The user's email/login in format 'user@domain.com'.")
    passwordProfile: PasswordProfile = Field(..., description="Initial password configuration.")

#
def list_users_tool(input=None):
    """
    Tool: List all users in Azure Active Directory.

    Args:
        input (None): No input required.

    Returns:
        dict: A dictionary of users from the tenant.
    """
    # return ad.list_users()
    return view_users()

def assign_user_to_default_department_tool(first_name: str, last_name: str, departments: List[str]):
    """
    Tool: Assign a user to a department(s)..

    This tool:
    - Updates the user's 'department' field
    - Adds the user to the "<department> Rooms Entry" group


    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        departments (List[str], optional): Department name (default is "Cancer Centrum").

    Returns:
        dict: Result of department update and group assignment.

    Example:
        assign_user_to_default_department_tool("David", "Mike", "Staff", ["Cancer Centrum"] )
        assign_user_to_default_department_tool("Jane", "Smith", "Admin", ["Medicals", "Cancer Centrum"] )
    """


    upsert_user(first_name, last_name, department=departments)

    if departments:
        # return ad.assign_user_to_department_and_group(display_name, department)
        return f"{first_name} {last_name} has been assigned to the {departments} department."
    else:
        return f"{first_name} {last_name} has been updated."


def assign_permission_to_user(first_name: str, last_name: str, permissions: List[str]):
    """
    Tool: Assign/add permissions to a user.

    This tool:
    - Assigns, updates or adds permissions to a user in the active directory.

    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        permissions (List(str)): Permissions to assign to the user.

    Returns:
        dict: Result of department update and group assignment.

    Example:
        assign_user_to_default_department_tool("David", "Mike", "Staff", ["Medical Access"])
        assign_user_to_default_department_tool("Jane", "Smith", "Admin", ["Full Access", "Read Only"])
    """

    upsert_user(first_name, last_name, permissions=permissions)

    if permissions:
        # return ad.assign_user_to_department_and_group(display_name, department)
        return f"{first_name} {last_name} has been updated with permissions {permissions}"
    else:
        return f"{first_name} {last_name} has been updated."


def assign_group_to_user(first_name: str, last_name: str,  groups: List[str]):
    """
    Tool: Assign/add a user to a group.

    This tool:
    - Assigns, updates or adds a user to an active directory group.



    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        groups (List[str]): Groups to assign to user to.

    Returns:
        dict: Result of department update and group assignment.

    Example:
        assign_user_to_default_department_tool("David", "Mike", ["Cancer Centrum Rooms Entry"])
        assign_user_to_default_department_tool("Jane", "Smith", "Admin", ["Cancer Centrum Changing Rooms", "Cancer Centrum Rooms Entry"])
    """

    upsert_user(first_name, last_name, groups=groups)

    if groups:
        # return ad.assign_user_to_department_and_group(display_name, department)
        return f"{first_name} {last_name} has been added to group {groups}"
    else:
        return f"{first_name} {last_name} has been updated."

def remove_departments_from_user(first_name: str, last_name: str, departments: List[str]):
    """
    Tool: Remove one or more permissions from a user.

    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        departments (List[str]): Departments to remove.

    Returns:
        str: Result of removal.
    """
    return remove_user_fields(first_name, last_name, remove_department=departments)

def remove_permissions_from_user(first_name: str, last_name: str, permissions: List[str]):
    """
    Tool: Remove one or more permissions from a user.

    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        permissions (List[str]): Permissions to remove.

    Returns:
        str: Result of removal.
    """
    return remove_user_fields(first_name, last_name, permissions=permissions)

def remove_groups_from_user(first_name: str, last_name: str, groups: List[str]):
    """
    Tool: Remove one or more groups from a user.

    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.
        groups (List[str]): Groups to remove.

    Returns:
        str: Result of removal.
    """
    return remove_user_fields(first_name, last_name, groups=groups)

def check_user_department_tool(first_name: str, last_name: str):
    """
    Tool: Check a user's department and their corresponding AD group.

    This tool:
    - Check the user's 'department' field


    Args:
        first_name (str): First name of the user.
        last_name (str): Last name of the user.


    Returns:
        dict: Result of department update and group assignment.

    Example:
        check_user_department_tool("David Mike")
        check_user_department_tool("Jane Smith")
    """

    user = get_user_from_db_by_name(first_name, last_name)
    if not user:
        return f"User is not in active directory."

    return user


# from config import CLIENT_ID, TENANT_ID, CLIENT_SECRET, AUTHORITY, SCOPE
import msal
import requests
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any


class PasswordProfile(BaseModel):
    password: str = Field(..., description="A strong initial password for the user.")
    forceChangePasswordNextSignIn: bool = Field(True, description="Require user to change password on next sign-in.")

class CreateUserPayload(BaseModel):
    accountEnabled: bool = Field(..., description="Whether the account is enabled.")
    displayName: str = Field(..., description="Full name of the user.")
    mailNickname: str = Field(..., description="Alias used to generate the email address.")
    userPrincipalName: str = Field(..., description="The user's email/login in format 'user@domain.com'.")
    passwordProfile: PasswordProfile = Field(..., description="Initial password configuration.")

class ActiveDirectory:
    """
    A utility class for interacting with Microsoft Azure Active Directory via the Microsoft Graph API.

    This class handles authentication using client credentials (MSAL) and provides high-level methods
    to perform directory-related operations such as listing users and roles, managing user-role assignments,
    and creating/updating/deleting user accounts.

    Features:
    - Obtain an access token using client credentials.
    - Make authenticated requests to Microsoft Graph API.
    - List all users and directory roles.
    - Get roles assigned to a specific user.
    - Add or remove users from directory roles.
    - Instantiate a directory role from a template.
    - Create, update, or delete user accounts.

    Requirements:
    - Azure AD App registration with necessary permissions.
    - `CLIENT_ID`, `TENANT_ID`, `CLIENT_SECRET`, `AUTHORITY`, and `SCOPE` configured in `config.py`.

    Example usage:
        ad = ActiveDirectory()
        users = ad.list_users()
        ad.create_user({...})
    """

    def __init__(self):
        pass

    def get_access_token(self):
        app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET
        )
        result = app.acquire_token_for_client(scopes=SCOPE)
        if "access_token" in result:
            return result["access_token"]
        raise Exception(f"Auth failed: {result.get('error_description')}")

    def graph_api_request(self, method, endpoint, token, data=None, params=None):
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        url = f"https://graph.microsoft.com/v1.0/{endpoint}"
        response = requests.request(method, url, headers=headers, json=data, params=params)
        if not response.ok:
            raise Exception(f"Graph API error: {response.status_code} - {response.text}")
        return response.json() if response.content else {}

    def list_users(self):
        token = self.get_access_token()
        return self.graph_api_request("GET", "users", token)

    def list_roles(self):
        token = self.get_access_token()
        return self.graph_api_request("GET", "directoryRoles", token)

    def get_user_roles(self, user_id):
        token = self.get_access_token()
        return self.graph_api_request("GET", f"users/{user_id}/memberOf", token)

    def add_user_to_role(self, user_id, role_id):
        token = self.get_access_token()
        data = {
            "@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{user_id}"
        }
        return self.graph_api_request("POST", f"directoryRoles/{role_id}/members/$ref", token, data=data)

    def remove_user_from_role(self, user_id, role_id):
        token = self.get_access_token()
        endpoint = f"directoryRoles/{role_id}/members/{user_id}/$ref"
        return self.graph_api_request("DELETE", endpoint, token)

    def instantiate_directory_role(self, role_template_id):
        token = self.get_access_token()
        data = {
            "roleTemplateId": role_template_id
        }
        return self.graph_api_request("POST", "directoryRoles", token, data=data)

    def create_user(self, user_data):
        token = self.get_access_token()
        return self.graph_api_request("POST", "users", token, data=user_data)

    def update_user(self, user_id, updates: dict):
        token = self.get_access_token()
        return self.graph_api_request("PATCH", f"users/{user_id}", token, data=updates)

    def delete_user(self, user_id):
        token = self.get_access_token()
        return self.graph_api_request("DELETE", f"users/{user_id}", token)

    def create_group(
            self,
            *,
            display_name: str,
            mail_nickname: str,
            description: Optional[str] = None,
            group_type: str = "security",  # "security" | "m365" | "dynamic-security" | "dynamic-m365"
            visibility: Optional[str] = None,  # "Private" | "Public" (M365 groups only)
            membership_rule: Optional[str] = None,  # required for dynamic groups
            owners: Optional[List[str]] = None,  # list of directoryObject IDs
            members: Optional[List[str]] = None  # list of directoryObject IDs
    ) -> Dict:
        """
        Create an Azure AD group and optionally attach owners/members.

        Returns:
            {
              "group": {...created group object...},
              "owner_results": [{"id": "...", "ok": True/False, "error": "...?"}, ...],
              "member_results": [{"id": "...", "ok": True/False, "error": "...?"}, ...]
            }

        Required Graph (Application) permissions:
          - Create group: Group.ReadWrite.All
          - Add owners/members: GroupMember.ReadWrite.All (or Directory.ReadWrite.All)
        Notes:
          - Use tokens requested with scope: ["https://graph.microsoft.com/.default"]
          - mailNickname must be unique within the tenant.
          - For dynamic groups, provide `membership_rule` and we'll enable processing.
        """
        token = self.get_access_token()

        # Build base payload
        payload: Dict = {
            "displayName": display_name,
            "mailNickname": mail_nickname
        }
        if description:
            payload["description"] = description

        gtype = group_type.lower().strip()

        if gtype == "security":
            payload.update({
                "securityEnabled": True,
                "mailEnabled": False,
                "groupTypes": []
            })

        elif gtype == "m365":
            payload.update({
                "securityEnabled": False,
                "mailEnabled": True,
                "groupTypes": ["Unified"]
            })
            if visibility in {"Private", "Public"}:
                payload["visibility"] = visibility

        elif gtype == "dynamic-security":
            if not membership_rule:
                raise ValueError("membership_rule is required for dynamic-security groups")
            payload.update({
                "securityEnabled": True,
                "mailEnabled": False,
                "groupTypes": ["DynamicMembership"],
                "membershipRule": membership_rule,
                "membershipRuleProcessingState": "On"
            })

        elif gtype == "dynamic-m365":
            if not membership_rule:
                raise ValueError("membership_rule is required for dynamic-m365 groups")
            payload.update({
                "securityEnabled": False,
                "mailEnabled": True,
                "groupTypes": ["Unified", "DynamicMembership"],
                "membershipRule": membership_rule,
                "membershipRuleProcessingState": "On"
            })
            if visibility in {"Private", "Public"}:
                payload["visibility"] = visibility

        else:
            raise ValueError("group_type must be one of: 'security', 'm365', 'dynamic-security', 'dynamic-m365'")

        # 1) Create the group
        group = self.graph_api_request("POST", "groups", token, data=payload)
        group_id = group.get("id")
        results = {"group": group, "owner_results": [], "member_results": []}

        # 2) Optionally add owners
        for owner_id in owners or []:
            try:
                body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{owner_id}"}
                self.graph_api_request("POST", f"groups/{group_id}/owners/$ref", token, data=body)
                results["owner_results"].append({"id": owner_id, "ok": True})
            except Exception as e:
                results["owner_results"].append({"id": owner_id, "ok": False, "error": str(e)})

        # 3) Optionally add members
        for member_id in members or []:
            try:
                body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{member_id}"}
                self.graph_api_request("POST", f"groups/{group_id}/members/$ref", token, data=body)
                results["member_results"].append({"id": member_id, "ok": True})
            except Exception as e:
                results["member_results"].append({"id": member_id, "ok": False, "error": str(e)})

        return results

    # inside ActiveDirectory
    def _paged_get(self, endpoint: str, token: str, params: dict | None = None) -> list[dict]:
        """
        GET a collection from Graph and follow @odata.nextLink to return a flat list.
        endpoint: e.g., "users", "groups", "users/{id}/memberOf/microsoft.graph.group"
        """
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"https://graph.microsoft.com/v1.0/{endpoint}"
        items: list[dict] = []
        while True:
            resp = requests.get(url, headers=headers, params=params if url.endswith(endpoint) else None, timeout=30)
            if not resp.ok:
                raise Exception(f"Graph API error: {resp.status_code} - {resp.text}")
            data = resp.json()
            items.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            if not next_link:
                break
            url = next_link
            params = None  # nextLink already includes query params
        return items

    # inside ActiveDirectory
    def list_groups(
            self,
            security_only: bool = False,
            unified_only: bool = False,
            select: str | None = "id,displayName,mailNickname,mail,securityEnabled,groupTypes",
    ) -> list[dict]:
        """
        List groups. Set one of:
          - security_only=True  -> classic security groups (securityEnabled true, groupTypes [])
          - unified_only=True   -> Microsoft 365 groups (groupTypes contains 'Unified')
        """
        if security_only and unified_only:
            raise ValueError("Choose either security_only or unified_only, not both.")
        token = self.get_access_token()

        # Build filter
        _filter = None
        if security_only:
            # securityEnabled eq true AND NOT Unified
            _filter = "securityEnabled eq true and not(groupTypes/any(t:t eq 'Unified'))"
        elif unified_only:
            _filter = "groupTypes/any(t:t eq 'Unified')"

        params = {}
        if select:
            params["$select"] = select
        if _filter:
            params["$filter"] = _filter

        return self._paged_get("groups", token, params=params)

    # inside ActiveDirectory
    def get_user_groups(self, user_id: str, transitive: bool = False) -> list[dict]:
        """
        Return ONLY group objects the user is a member of.
        Uses type-cast segment to groups to avoid roles/other directoryObjects.
        """
        token = self.get_access_token()
        segment = "transitiveMemberOf" if transitive else "memberOf"
        endpoint = f"users/{user_id}/{segment}/microsoft.graph.group"
        # Select a concise set of fields
        params = {"$select": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
        return self._paged_get(endpoint, token, params=params)

    # inside ActiveDirectory
    def get_user_owned_groups(self, user_id: str) -> list[dict]:
        """
        Return groups where the user is an owner.
        """
        token = self.get_access_token()
        endpoint = f"users/{user_id}/ownedObjects/microsoft.graph.group"
        params = {"$select": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
        return self._paged_get(endpoint, token, params=params)

    # inside ActiveDirectory
    def list_users_with_groups(
            self,
            include_transitive: bool = False,
            include_owned: bool = True,
            select: str | None = "id,displayName,userPrincipalName",
    ) -> list[dict]:
        """
        For each user, attach 'groups' (direct), optional 'transitive_groups', and optional 'owned_groups'.
        NOTE: This makes multiple Graph calls; consider batching if your directory is large.
        """
        token = self.get_access_token()
        users = self._paged_get("users", token, params={"$select": select} if select else None)
        enriched: list[dict] = []

        for u in users:
            uid = u["id"]
            user_entry = dict(u)  # shallow copy

            # direct groups
            user_entry["groups"] = self.get_user_groups(uid, transitive=False)

            # transitive groups (optional)
            if include_transitive:
                # user_entry["transitive_groups"] = self.get_user_groups(uid, transitive=True)
                user_entry["transitive_groups"] = []

            # owned groups (optional)
            if include_owned:
                # user_entry["owned_groups"] = self.get_user_owned_groups(uid)
                user_entry["owned_groups"] = []

            enriched.append(user_entry)

        return enriched

    # inside ActiveDirectory
    def get_group_owners(self, group_id: str) -> list[dict]:
        token = self.get_access_token()
        endpoint = f"groups/{group_id}/owners"
        params = {"$select": "id,displayName,userPrincipalName"}
        return self._paged_get(endpoint, token, params=params)

    def get_group_members(self, group_id: str) -> list[dict]:
        token = self.get_access_token()
        endpoint = f"groups/{group_id}/members"
        params = {"$select": "id,displayName,userPrincipalName"}
        return self._paged_get(endpoint, token, params=params)

ad = ActiveDirectory()


def list_users_tool(input=None):
    """
    Tool: List all users in Azure Active Directory.

    Args:
        input (None): No input required.

    Returns:
        dict: A dictionary of users from the tenant.
    """
    return ad.list_users()

# print(list_users_tool())

# def create_user_tool(user_data: dict):
#     """
#     Tool: Create a new user in Azure Active Directory.
#
#     Args:
#         user_data (dict): User object payload matching Microsoft Graph user creation schema.
#
#     Returns:
#         dict: The created user object.
#     """
#     return ad.create_user(user_data)

def create_user_tool(user_data: CreateUserPayload) -> dict:
    """
    Tool: Create a new user in Azure Active Directory.

    Args:
        user_data (CreateUserPayload): The user creation payload.

    Returns:
        dict: The created user object.
    """
    return ad.create_user(user_data.dict())

def delete_user_tool(user_id: str):
    """
    Tool: Delete an existing user by ID.

    Args:
        user_id (str): The unique ID of the user to delete.

    Returns:
        dict: Result of the deletion request.
    """
    return ad.delete_user(user_id)

def get_user_roles_tool(user_id: str):
    """
    Tool: Get roles and group memberships for a specific user.

    Args:
        user_id (str): The unique Azure AD user ID.

    Returns:
        dict: A dictionary of directory roles and groups the user belongs to.
    """
    return ad.get_user_roles(user_id)

def remove_user_from_role_tool(data: dict):
    """
    Tool: Remove a user from a specific Azure AD directory role.

    Args:
        data (dict): Dictionary with:
            - user_id (str): The ID of the user to remove.
            - role_id (str): The ID of the role to remove from.

    Returns:
        dict: Result of the removal operation.
    """
    user_id = data.get("user_id")
    role_id = data.get("role_id")

    if not user_id or not role_id:
        raise ValueError("Both 'user_id' and 'role_id' must be provided.")

    return ad.remove_user_from_role(user_id, role_id)

def instantiate_directory_role_tool(role_template_id: str):
    """
    Tool: Activate a directory role based on its template ID.

    Args:
        role_template_id (str): The ID of the role template to instantiate.

    Returns:
        dict: The activated directory role object.
    """
    return ad.instantiate_directory_role(role_template_id)

def list_roles_tool(input=None):
    """
    Tool: List all activated directory roles in the Azure AD tenant.

    Args:
        input (None): No input required.

    Returns:
        dict: A dictionary of active directory roles.
    """
    return ad.list_roles()

def add_user_to_role_tool(data: dict):
    """
    Tool: Assign a user to a specific Azure AD directory role.

    Args:
        data (dict): Dictionary with:
            - user_id (str): The ID of the user to add.
            - role_id (str): The ID of the role to assign.

    Returns:
        dict: Result of the assignment operation.
    """
    user_id = data.get("user_id")
    role_id = data.get("role_id")

    if not user_id or not role_id:
        raise ValueError("Both 'user_id' and 'role_id' must be provided.")

    return ad.add_user_to_role(user_id, role_id)
