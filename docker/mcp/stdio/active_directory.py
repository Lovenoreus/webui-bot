# -------------------- Built-in Libraries --------------------
import os
import asyncio
from typing import Dict, Optional, List, Union
from functools import lru_cache

# -------------------- External Libraries --------------------
from dotenv import load_dotenv
import msal
import httpx
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
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


class FastActiveDirectory:
    """
    Ultra-fast async Active Directory client with ALL original operations.

    Features:
    - HTTP/2 support for multiplexing
    - Connection pooling and keep-alive
    - Token caching
    - Concurrent batch operations
    - Automatic retries with exponential backoff
    - All original ActiveDirectory methods
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Args:
            max_concurrent: Maximum concurrent requests (default: 10)
        """
        self._client: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None
        self._token_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    async def __aenter__(self):
        """Initialize async HTTP client with optimal settings"""
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0
        )

        timeout = httpx.Timeout(30.0, connect=10.0)

        self._client = httpx.AsyncClient(
            http2=True,  # HTTP/2 for request multiplexing
            limits=limits,
            timeout=timeout,
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        if self._client:
            await self._client.aclose()

    @lru_cache(maxsize=1)
    def _get_msal_app(self):
        """Cached MSAL app instance"""
        return msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET
        )

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get cached token or fetch new one.
        Thread-safe with async lock.
        """
        async with self._token_lock:
            if self._token and not force_refresh:
                return self._token

            # Run MSAL in thread pool (it's sync)
            app = self._get_msal_app()
            result = await asyncio.to_thread(
                app.acquire_token_for_client,
                scopes=SCOPE
            )

            if "access_token" in result:
                self._token = result["access_token"]
                return self._token

            raise Exception(f"Auth failed: {result.get('error_description')}")

    async def graph_api_request(
            self,
            method: str,
            endpoint: str,
            token: str,
            data: Optional[dict] = None,
            params: Optional[dict] = None
    ) -> dict:
        """
        Original method signature - kept for backward compatibility.
        """
        return await self._request(method, endpoint, data=data, params=params)

    async def _request(
            self,
            method: str,
            endpoint: str,
            data: Optional[dict] = None,
            params: Optional[dict] = None,
            retry_count: int = 3
    ) -> dict:
        """
        Make Graph API request with automatic retries and rate limiting.
        """
        async with self._semaphore:  # Rate limiting
            token = await self.get_access_token()
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'ConsistencyLevel': 'eventual'  # Better performance for queries
            }

            url = f"https://graph.microsoft.com/v1.0/{endpoint}"

            for attempt in range(retry_count):
                try:
                    response = await self._client.request(
                        method,
                        url,
                        headers=headers,
                        json=data,
                        params=params
                    )

                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        await asyncio.sleep(retry_after)
                        continue

                    # Handle token expiration
                    if response.status_code == 401:
                        token = await self.get_access_token(force_refresh=True)
                        headers['Authorization'] = f'Bearer {token}'
                        continue

                    response.raise_for_status()
                    return response.json() if response.content else {}

                except httpx.HTTPStatusError as e:
                    if attempt == retry_count - 1:
                        raise Exception(f"Graph API error: {e.response.status_code} - {e.response.text}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

                except httpx.RequestError as e:
                    if attempt == retry_count - 1:
                        raise Exception(f"Request failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)

    async def _paged_get(
            self,
            endpoint: str,
            token: str,
            params: Optional[dict] = None
    ) -> List[dict]:
        """
        Original signature - fetch all pages.
        """
        items = []
        next_link = None
        url = endpoint

        while True:
            if next_link:
                # Extract just the path from nextLink
                next_link_url = next_link.replace('https://graph.microsoft.com/v1.0/', '')
                data = await self._request("GET", next_link_url)
            else:
                data = await self._request("GET", url, params=params)

            items.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")

            if not next_link:
                break

        return items

    # ==================== ORIGINAL USER OPERATIONS ====================

    async def list_users(self) -> dict:
        """Original: List all users"""
        token = await self.get_access_token()
        return await self.graph_api_request("GET", "users", token)

    async def create_user(self, user_data: dict) -> dict:
        """Original: Create a new user"""
        token = await self.get_access_token()
        return await self.graph_api_request("POST", "users", token, data=user_data)

    async def update_user(self, user_id: str, updates: dict) -> dict:
        """Original: Update user properties"""
        token = await self.get_access_token()
        return await self.graph_api_request("PATCH", f"users/{user_id}", token, data=updates)

    async def delete_user(self, user_id: str) -> dict:
        """Original: Delete a user"""
        token = await self.get_access_token()
        return await self.graph_api_request("DELETE", f"users/{user_id}", token)

    # ==================== ORIGINAL ROLE OPERATIONS ====================

    async def list_roles(self) -> dict:
        """Original: List all directory roles"""
        token = await self.get_access_token()
        return await self.graph_api_request("GET", "directoryRoles", token)

    async def get_user_roles(self, user_id: str) -> dict:
        """Original: Get roles for a user"""
        token = await self.get_access_token()
        return await self.graph_api_request("GET", f"users/{user_id}/memberOf", token)

    async def add_user_to_role(self, user_id: str, role_id: str) -> dict:
        """Original: Add user to a role"""
        token = await self.get_access_token()
        data = {
            "@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{user_id}"
        }
        return await self.graph_api_request("POST", f"directoryRoles/{role_id}/members/$ref", token, data=data)

    async def remove_user_from_role(self, user_id: str, role_id: str) -> dict:
        """Original: Remove user from a role"""
        token = await self.get_access_token()
        endpoint = f"directoryRoles/{role_id}/members/{user_id}/$ref"
        return await self.graph_api_request("DELETE", endpoint, token)

    async def instantiate_directory_role(self, role_template_id: str) -> dict:
        """Original: Instantiate a directory role from template"""
        token = await self.get_access_token()
        data = {
            "roleTemplateId": role_template_id
        }
        return await self.graph_api_request("POST", "directoryRoles", token, data=data)

    # ==================== ORIGINAL GROUP OPERATIONS ====================

    async def create_group(
            self,
            *,
            display_name: str,
            mail_nickname: str,
            description: Optional[str] = None,
            group_type: str = "security",
            visibility: Optional[str] = None,
            membership_rule: Optional[str] = None,
            owners: Optional[List[str]] = None,
            members: Optional[List[str]] = None
    ) -> Dict:
        """
        Original: Create an Azure AD group and optionally attach owners/members.
        """
        token = await self.get_access_token()

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
        group = await self.graph_api_request("POST", "groups", token, data=payload)
        group_id = group.get("id")
        results = {"group": group, "owner_results": [], "member_results": []}

        # 2) Add owners concurrently
        async def add_owner(owner_id: str):
            try:
                body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{owner_id}"}
                await self.graph_api_request("POST", f"groups/{group_id}/owners/$ref", token, data=body)
                return {"id": owner_id, "ok": True}
            except Exception as e:
                return {"id": owner_id, "ok": False, "error": str(e)}

        # 3) Add members concurrently
        async def add_member(member_id: str):
            try:
                body = {"@odata.id": f"https://graph.microsoft.com/v1.0/directoryObjects/{member_id}"}
                await self.graph_api_request("POST", f"groups/{group_id}/members/$ref", token, data=body)
                return {"id": member_id, "ok": True}
            except Exception as e:
                return {"id": member_id, "ok": False, "error": str(e)}

        if owners:
            results["owner_results"] = await asyncio.gather(*[add_owner(o) for o in owners])

        if members:
            results["member_results"] = await asyncio.gather(*[add_member(m) for m in members])

        return results

    async def list_groups(
            self,
            security_only: bool = False,
            unified_only: bool = False,
            select: str | None = "id,displayName,mailNickname,mail,securityEnabled,groupTypes",
    ) -> list[dict]:
        """
        Original: List groups with optional filtering.
        """
        if security_only and unified_only:
            raise ValueError("Choose either security_only or unified_only, not both.")

        token = await self.get_access_token()

        # Build filter
        _filter = None
        if security_only:
            _filter = "securityEnabled eq true and not(groupTypes/any(t:t eq 'Unified'))"
        elif unified_only:
            _filter = "groupTypes/any(t:t eq 'Unified')"

        params = {}
        if select:
            params["$select"] = select
        if _filter:
            params["$filter"] = _filter

        return await self._paged_get("groups", token, params=params)

    async def get_user_groups(self, user_id: str, transitive: bool = False) -> list[dict]:
        """
        Original: Return ONLY group objects the user is a member of.
        """
        token = await self.get_access_token()
        segment = "transitiveMemberOf" if transitive else "memberOf"
        endpoint = f"users/{user_id}/{segment}/microsoft.graph.group"
        params = {"$select": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
        return await self._paged_get(endpoint, token, params=params)

    async def get_user_owned_groups(self, user_id: str) -> list[dict]:
        """
        Original: Return groups where the user is an owner.
        """
        token = await self.get_access_token()
        endpoint = f"users/{user_id}/ownedObjects/microsoft.graph.group"
        params = {"$select": "id,displayName,mailNickname,mail,securityEnabled,groupTypes"}
        return await self._paged_get(endpoint, token, params=params)

    async def list_users_with_groups(
            self,
            include_transitive: bool = False,
            include_owned: bool = True,
            select: str | None = "id,displayName,userPrincipalName",
    ) -> list[dict]:
        """
        Original: For each user, attach 'groups' (direct), optional 'transitive_groups', and optional 'owned_groups'.
        NOTE: This makes multiple Graph calls; consider batching if your directory is large.
        """
        token = await self.get_access_token()
        users_data = await self._paged_get("users", token, params={"$select": select} if select else None)

        enriched: list[dict] = []

        # Process users concurrently in batches
        async def enrich_user(u: dict):
            uid = u["id"]
            user_entry = dict(u)

            # Fetch all data concurrently for this user
            tasks = [self.get_user_groups(uid, transitive=False)]

            if include_transitive:
                tasks.append(self.get_user_groups(uid, transitive=True))
            else:
                tasks.append(asyncio.sleep(0))  # Dummy task

            if include_owned:
                tasks.append(self.get_user_owned_groups(uid))
            else:
                tasks.append(asyncio.sleep(0))  # Dummy task

            results = await asyncio.gather(*tasks, return_exceptions=True)

            user_entry["groups"] = results[0] if not isinstance(results[0], Exception) else []

            if include_transitive:
                user_entry["transitive_groups"] = results[1] if not isinstance(results[1], Exception) else []

            if include_owned:
                user_entry["owned_groups"] = results[2] if not isinstance(results[2], Exception) else []

            return user_entry

        # Process all users concurrently
        enriched = await asyncio.gather(*[enrich_user(u) for u in users_data])

        return enriched

    async def get_group_owners(self, group_id: str) -> list[dict]:
        """Original: Get owners of a group"""
        token = await self.get_access_token()
        endpoint = f"groups/{group_id}/owners"
        params = {"$select": "id,displayName,userPrincipalName"}
        return await self._paged_get(endpoint, token, params=params)

    async def get_group_members(self, group_id: str) -> list[dict]:
        """Original: Get members of a group"""
        token = await self.get_access_token()
        endpoint = f"groups/{group_id}/members"
        params = {"$select": "id,displayName,userPrincipalName"}
        return await self._paged_get(endpoint, token, params=params)

    # ==================== NEW: SMART USER RESOLUTION ====================

    async def resolve_user(self, identifier: str) -> str:
        """
        NEW: Fast user resolution with multiple strategies.
        Returns user ID (GUID).

        Supports:
        - User ID (GUID) - returns as-is
        - Email (userPrincipalName) - direct lookup
        - Display name - searches and resolves
        """
        # Strategy 1: Already a GUID
        if len(identifier) == 36 and identifier.count('-') == 4:
            return identifier

        # Strategy 2: Email - direct lookup (fastest)
        if '@' in identifier:
            try:
                user = await self._request("GET", f"users/{identifier}")
                return user.get("id")
            except:
                pass

        # Strategy 3: Search by displayName
        params = {
            "$filter": f"startsWith(displayName, '{identifier}')",
            "$select": "id,displayName,userPrincipalName",
            "$top": 10,
            "$count": "true"
        }

        result = await self._request("GET", "users", params=params)
        users = result.get("value", [])

        if not users:
            raise Exception(f"❌ No user found: '{identifier}'")

        # Exact match (case-insensitive)
        exact = [u for u in users if u.get("displayName", "").lower() == identifier.lower()]
        if len(exact) == 1:
            return exact[0]["id"]

        if len(users) == 1:
            return users[0]["id"]

        # Multiple matches
        candidates = "\n".join([
            f"  • {u.get('displayName')} <{u.get('userPrincipalName')}>"
            for u in users[:5]
        ])
        raise Exception(
            f"❌ Multiple users found for '{identifier}':\n{candidates}\n\n"
            f"💡 Use full name or email address"
        )

    # ==================== NEW: ENHANCED BATCH OPERATIONS ====================

    async def batch_resolve_users(
            self,
            identifiers: List[str],
            ignore_errors: bool = True
    ) -> List[Union[str, Exception]]:
        """
        NEW: Resolve multiple users concurrently.
        Up to 10x faster than sequential!
        """
        tasks = [self.resolve_user(ident) for ident in identifiers]
        results = await asyncio.gather(*tasks, return_exceptions=ignore_errors)
        return results

    async def batch_get_user_groups(
            self,
            user_identifiers: List[str],
            transitive: bool = False
    ) -> List[Union[List[dict], Exception]]:
        """
        NEW: Get groups for multiple users concurrently.
        BLAZING FAST! 🔥
        """
        # First resolve all users concurrently
        user_ids = await self.batch_resolve_users(user_identifiers, ignore_errors=True)

        # Then get groups concurrently
        async def get_groups_safe(uid):
            if isinstance(uid, Exception):
                return uid
            try:
                return await self.get_user_groups(uid, transitive=transitive)
            except Exception as e:
                return e

        tasks = [get_groups_safe(uid) for uid in user_ids]
        return await asyncio.gather(*tasks)

    async def search_users_fuzzy(
            self,
            query: str,
            limit: int = 10
    ) -> List[dict]:
        """
        NEW: Fuzzy search for users across multiple fields.
        Searches displayName, userPrincipalName, and mail.
        """
        filter_parts = [
            f"startsWith(displayName, '{query}')",
            f"startsWith(userPrincipalName, '{query}')",
            f"startsWith(mail, '{query}')"
        ]

        params = {
            "$filter": " or ".join(filter_parts),
            "$select": "id,displayName,userPrincipalName,mail,jobTitle,department",
            "$top": limit,
            "$count": "true"
        }

        result = await self._request("GET", "users", params=params)
        return result.get("value", [])

    # ==================== NEW: WRAPPER METHODS WITH SMART RESOLUTION ====================

    async def get_user_groups_smart(self, user_identifier: str, transitive: bool = False) -> list[dict]:
        """
        NEW: Get user groups with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.get_user_groups(user_id, transitive=transitive)

    async def get_user_roles_smart(self, user_identifier: str) -> dict:
        """
        NEW: Get user roles with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.get_user_roles(user_id)

    async def add_user_to_role_smart(self, user_identifier: str, role_id: str) -> dict:
        """
        NEW: Add user to role with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.add_user_to_role(user_id, role_id)

    async def remove_user_from_role_smart(self, user_identifier: str, role_id: str) -> dict:
        """
        NEW: Remove user from role with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.remove_user_from_role(user_id, role_id)

    async def update_user_smart(self, user_identifier: str, updates: dict) -> dict:
        """
        NEW: Update user with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.update_user(user_id, updates)

    async def delete_user_smart(self, user_identifier: str) -> dict:
        """
        NEW: Delete user with smart resolution (accepts ID, email, or name)
        """
        user_id = await self.resolve_user(user_identifier)
        return await self.delete_user(user_id)