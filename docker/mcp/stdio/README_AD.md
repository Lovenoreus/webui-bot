# Active Directory MCP Server

A high-performance Model Context Protocol (MCP) server for managing Azure Active Directory operations. This server provides comprehensive AD management capabilities through both REST APIs and MCP tool calls.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure AD tenant with appropriate permissions
- Azure App Registration with necessary API permissions

### 1. Azure Setup

#### Create Azure App Registration
1. Go to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Click "New registration"
3. Set name (e.g., "AD MCP Server") and register
4. Note the **Application (client) ID** and **Directory (tenant) ID**

#### Configure API Permissions
1. Go to your app → API permissions → Add a permission
2. Select **Microsoft Graph** → **Application permissions**
3. Add these permissions:
   - `User.Read.All` - Read all users
   - `User.ReadWrite.All` - Create/modify users
   - `Group.Read.All` - Read all groups  
   - `Group.ReadWrite.All` - Create/modify groups
   - `Directory.Read.All` - Read directory data
   - `RoleManagement.ReadWrite.Directory` - Manage directory roles

4. **Grant admin consent** for your organization

#### Create Client Secret
1. Go to Certificates & secrets → New client secret
2. Set description and expiry → Add
3. **Copy the secret value immediately** (you won't see it again)

### 2. Environment Configuration

Create a `.env` file in the `stdio/` directory:

```env
# Azure AD Configuration
AZURE_CLIENT_ID=your-application-client-id
AZURE_TENANT_ID=your-directory-tenant-id  
AZURE_CLIENT_SECRET=your-client-secret-value
```

### 3. Installation & Setup

```bash
# Install dependencies
cd docker/mcp/stdio
pip install -r ../requirements.mcp.txt

# Run the server
python server_new.py
```

The server will start on `http://localhost:8000`

### 4. Verify Setup

Test the connection:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ad/users
```

## 📋 Available Operations

### 👥 User Management
- **List Users** - Get all users in directory
- **Create User** - Create new AD users with password policies
- **Update User** - Modify user properties (displayName, jobTitle, etc.)
- **Delete User** - Remove users from directory
- **Search Users** - Fuzzy search across multiple user fields
- **Get User Profile** - Complete user info with groups and roles

### 🔐 Role Management  
- **List Roles** - Get all directory roles
- **Add/Remove User to Role** - Assign administrative roles
- **Get Role Members** - List users with specific roles
- **Batch Role Operations** - Bulk role assignments
- **Instantiate Roles** - Activate role templates

### 👥 Group Management
- **List Groups** - Get all groups (Security/M365/Dynamic)
- **Create Groups** - Create Security, M365, or Dynamic groups
- **Add/Remove Members** - Manage group membership
- **Add/Remove Owners** - Manage group ownership
- **Get Group Details** - Complete group information
- **Batch Operations** - Bulk membership changes

## 🛠️ Usage Examples

### Using REST API

```bash
# List all users
curl http://localhost:8000/ad/users

# Create a user
curl -X POST http://localhost:8000/ad/users \
  -H "Content-Type: application/json" \
  -d '{
    "accountEnabled": true,
    "displayName": "John Doe", 
    "mailNickname": "johndoe",
    "userPrincipalName": "johndoe@yourdomain.com"
  }'

# Search users
curl "http://localhost:8000/ad/users?search=john&limit=10"
```

### Using MCP Tools

The server exposes 25+ MCP tools with names prefixed `ad_*`:

```python
# Example MCP tool calls
{
  "tool": "ad_list_users",
  "arguments": {}
}

{
  "tool": "ad_create_user", 
  "arguments": {
    "user": {
      "displayName": "Jane Smith",
      "mailNickname": "janesmith", 
      "userPrincipalName": "janesmith@yourdomain.com"
    }
  }
}

{
  "tool": "ad_add_user_to_role",
  "arguments": {
    "user_identifier": "janesmith@yourdomain.com",
    "role_identifier": "Global Administrator"
  }
}
```

## 🎯 Key Features

### Smart Resolution
- **Flexible Identifiers** - Use ID, email, or display name for users/groups/roles
- **Intelligent Matching** - Automatic exact and partial name matching
- **Error Prevention** - Clear disambiguation for multiple matches

### High Performance  
- **Async/Concurrent** - HTTP/2 with connection pooling
- **Batch Operations** - Process multiple items simultaneously
- **Token Caching** - Automatic token refresh and reuse
- **Rate Limiting** - Built-in throttling for API limits

### Developer Friendly
- **Comprehensive Logging** - Detailed operation tracking
- **Error Handling** - Graceful failures with helpful messages  
- **Type Safety** - Full Pydantic model validation
- **REST + MCP** - Dual interface for maximum compatibility

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   REST Client    │    │   FastAPI       │
│                 │◄──►│                  │◄──►│   Server        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ FastActiveDir   │
                                                │   (async)       │
                                                └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Microsoft Graph │
                                                │      API        │
                                                └─────────────────┘
```

## 📁 Project Structure

```
docker/mcp/stdio/
├── server_new.py          # FastAPI server with REST endpoints
├── active_directory.py    # High-performance AD client
├── models.py             # Pydantic data models
├── config.py             # Configuration management
├── README_AD.md          # This documentation
└── .env                  # Environment variables (create this)
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_CLIENT_ID` | ✅ | Application (client) ID from Azure |
| `AZURE_TENANT_ID` | ✅ | Directory (tenant) ID from Azure |
| `AZURE_CLIENT_SECRET` | ✅ | Client secret value from Azure |

### Server Configuration

The server runs on port 8000 by default. Key settings in `server_new.py`:

- **CORS** - Enabled for web clients
- **Concurrent Limit** - 20 simultaneous requests
- **Timeout** - 30s for API calls
- **HTTP/2** - Enabled for performance

## 🚨 Security & Permissions

### Required Azure Permissions

| Permission | Scope | Purpose |
|------------|-------|---------|
| `User.Read.All` | Application | List and read user profiles |
| `User.ReadWrite.All` | Application | Create, update, delete users |
| `Group.Read.All` | Application | List and read group information |
| `Group.ReadWrite.All` | Application | Create, update, delete groups |
| `Directory.Read.All` | Application | Read directory structure |
| `RoleManagement.ReadWrite.Directory` | Application | Manage admin roles |

### Security Best Practices

- **Principle of Least Privilege** - Only grant necessary permissions
- **Client Secret Rotation** - Regularly rotate secrets  
- **Network Security** - Use HTTPS in production
- **Audit Logging** - Monitor all AD operations
- **Access Control** - Restrict server access to authorized users

## 🔍 Troubleshooting

### Common Issues

**Authentication Errors**
```
❌ Missing required Azure AD credentials
```
- Verify `.env` file exists with correct values
- Check Azure app registration is complete
- Ensure client secret hasn't expired

**Permission Errors** 
```
❌ Insufficient privileges to complete the operation
```
- Verify all required permissions are granted
- Ensure admin consent is provided
- Check if role requires activation

**Network/API Errors**
```
❌ Request failed: Connection timeout
```
- Check internet connectivity
- Verify Azure service health
- Review rate limiting

### Debug Mode

Enable detailed logging by setting `DEBUG = True` in `server_new.py`:

```python
DEBUG = True  # Set to True for verbose logging
```

## 📈 Performance Tips

- **Batch Operations** - Use batch endpoints for multiple users/groups
- **Concurrent Requests** - Server handles up to 20 simultaneous operations  
- **Caching** - Tokens are cached automatically
- **Filtering** - Use search parameters to limit result sets
- **Paging** - Large results are automatically paginated




---

## 🎯 Next Steps

1. **Set up Azure App Registration** with required permissions
2. **Configure environment variables** in `.env` file  
3. **Install dependencies** and run the server
4. **Test connection** with health endpoint
5. **Start managing** your Active Directory! 

Happy AD automation! 🚀