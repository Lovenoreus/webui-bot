# 🩺 OMNIGATE: Healthbot Agentic – Complete Setup Guide

This is the first project of **Omnigate** - an advanced healthcare support system powered by AI.

**SYSTEM ARCHITECTURE**: `MCP-Orchestrated Multi-Agent Architecture with Hybrid Tool Distribution`

## 🏗️ Architecture Overview

The Healthbot Agentic system is a sophisticated multi-service architecture that combines:

- **MCP (Model Context Protocol) Server**: Core orchestration and tool management
- **Vector Database (Qdrant)**: Healthcare knowledge base with semantic search
- **SQLite Database Server**: Structured data storage and querying
- **Open WebUI**: Modern chat interface for user interactions
- **Azure Active Directory Integration**: Enterprise user and group management
- **Multi-LLM Support**: OpenAI, Mistral, and Ollama providers

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- API Keys (see Environment Setup below)

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone git@github.com:Lovenoreus/Healthbot-agentic.git
cd Healthbot-agentic

# Pull to ensure you have the latest
git pull
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following required variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Mistral Configuration (Primary)
MISTRAL_API_KEY=your_mistral_api_key_here

# OpenWeather API (Optional)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Azure Active Directory (Pre-configured)
AZURE_CLIENT_ID=8453aaa6-f125-4c28-9c3e-ff636d833539
AZURE_TENANT_ID=cfbb6550-a2fc-4705-9adc-c5711634d8a8
AZURE_CLIENT_SECRET=cky8Q~iBJ3mhQxKk03SXC.lTD0wx_b2uT6Q2dc6e

# Database Configuration (Auto-configured)
SQLITE_SERVER_URL=http://database_server:8762
DOCKER_CONTAINER=true
```

### 3. Launch the System

```bash
# Clean up any existing containers (optional)
docker system prune -f

# Start all services
docker compose up --build
```

### 4. Access the Application

Once all services are running, access the system at:

- **Chat Interface**: http://localhost:3000 (Open WebUI)
- **MCP Server API**: http://localhost:8009 (FastAPI Documentation: `/docs`)
- **Database Server**: http://localhost:8762
- **Qdrant Vector DB**: http://localhost:6333

## 🛠️ System Components

### Core Services

| Service | Port | Description | Technology     |
|---------|------|-------------|----------------|
| **open-webui** | 3000 | Modern chat interface | Svelte/Node.js |
| **mcp_server** | 8009 | MCP protocol server & tool orchestration | FastAPI/Python |
| **database_server** | 8762 | SQLite database with async API | FastAPI/SQLite |
| **qdrant** | 6333 | Vector database for semantic search | Qdrant         |

### Key Features

#### 🤖 Multi-Agent Architecture
- **MCP Protocol Integration**: Standardized tool calling and orchestration
- **Hybrid Tool Distribution**: Tools distributed across multiple specialized agents
- **Async Processing**: Non-blocking operations for real-time responses

#### 🔍 Advanced Search Capabilities
- **Vector Search**: Mistral embeddings for semantic healthcare knowledge retrieval
- **Multi-Strategy Retrieval**: High precision, medium precision, and broad fallback search
- **Healthcare Knowledge Base**: Pre-indexed medical documentation and support protocols

#### 🏥 Healthcare-Specific Tools
- **Medical Equipment Support**: IV pumps, monitors, diagnostic equipment troubleshooting
- **Environmental Monitoring**: HVAC, structural issues, safety protocols
- **Knowledge Management**: COSMIC healthcare system documentation
- **Support Ticket Integration**: Automated issue classification and routing

#### 👥 Enterprise Integration
- **Azure Active Directory**: Complete user lifecycle management
- **Group Management**: Security and unified groups with dynamic membership
- **Role-Based Access**: Directory role assignments and permissions
- **SSO Support**: SITHS card integration for healthcare environments

## 🔧 Available Tools & Endpoints

### MCP Tools (Model Context Protocol)

The system provides the following tools accessible via the MCP protocol:

#### General Tools
- `greet`: Friendly greeting with time-based salutation
- `query_database`: Natural language to SQL conversion with AI-powered query generation
- `get_current_weather`: OpenWeatherMap integration for current conditions

#### Active Directory Management
- `ad_list_users`: List all users in Azure AD
- `ad_create_user`: Create new users with auto-generated credentials
- `ad_update_user`: Update user properties
- `ad_delete_user`: Remove users from directory
- `ad_get_user_roles`: Get assigned roles for specific users
- `ad_get_user_groups`: Get group memberships (with transitive options)
- `ad_list_roles`: List all directory roles
- `ad_add_user_to_role`: Assign users to roles
- `ad_remove_user_from_role`: Remove role assignments
- `ad_list_groups`: List all groups with filtering options
- `ad_create_group`: Create security or unified groups
- `ad_add_group_member`: Add users to groups
- `ad_remove_group_member`: Remove group members
- `ad_get_group_members`: List group membership

### REST API Endpoints

#### Database Operations
- `POST /query_database`: Execute natural language database queries
- `POST /query_database_stream`: Streaming database query responses
- `GET /health`: System health check with component status

#### Weather Integration
- `POST /weather`: Get current weather by city or coordinates

#### Active Directory Operations
- `GET /ad/users`: List all users
- `POST /ad/users`: Create new user
- `PATCH /ad/users/{user_id}`: Update user
- `DELETE /ad/users/{user_id}`: Delete user
- `GET /ad/users/{user_id}/roles`: Get user roles
- `GET /ad/users/{user_id}/groups`: Get user groups
- `GET /ad/roles`: List all roles
- `POST /ad/roles/{role_id}/members`: Add user to role
- `DELETE /ad/roles/{role_id}/members/{user_id}`: Remove user from role
- `GET /ad/groups`: List all groups
- `POST /ad/groups`: Create new group
- `GET /ad/groups/{group_id}/members`: Get group members
- `POST /ad/groups/{group_id}/members`: Add group member
- `DELETE /ad/groups/{group_id}/members/{user_id}`: Remove group member

## 🔌 Adding New MCP Tools

The system is designed for easy extensibility. Follow these steps to add new functionality:

### Step 1: Add MCP Server Endpoint

Navigate to `mcp/stdio/server_new.py` and add your endpoint:

```python
@app.post("/your_endpoint")
async def your_endpoint_function(request: YourRequest):
    """Your endpoint description"""
    try:
        # Your endpoint logic here
        result = process_your_request(request)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Requirements:**
- All endpoints must be async
- Follow the established error handling pattern
- Return structured JSON responses

### Step 2: Add MCP Tool Definition

In the same file, add your tool to the `mcp_tools_list()` function:

```python
MCPTool(
    name="your_tool_name",
    description="Clear description of what your tool does and when to use it",
    inputSchema={
        "type": "object",
        "properties": {
            "parameter1": {
                "type": "string",
                "description": "Description of parameter1"
            }
        },
        "required": ["parameter1"]
    }
)
```

### Step 3: Add Tool Call Handler

Add the tool handling logic in `mcp_tools_call()` function:

```python
elif tool_name == "your_tool_name":
    your_request = YourRequest(**arguments)
    result = await your_endpoint_function(your_request)
    return MCPToolCallResponse(
        content=[MCPContent(type="text", text=json.dumps(result, indent=2))]
    )
```

### Step 4: Add the tool to the root endpoint ('/')
Add the tool to the `root()` function:

**🎉 Congratulations!** You've successfully added a new MCP tool to the system.

## 📊 Database Schema

The system uses SQLite with the following key tables:

- **users**: User management and profiles
- **posts**: Content and interaction tracking
- **company_database**: Company structure metrics

The database supports:
- Async operations via aiosqlite
- Natural language querying via LLM-powered SQL generation
- Streaming responses for large datasets
- Health monitoring and connection status

## 🔍 Vector Database Collections

### Healthcare Knowledge Collections
- **hospital_support_questions_mistral_embeddings**: Support protocols and procedures
- **cosmic_database_collection**: COSMIC healthcare system documentation
- **medical_equipment_protocols**: Equipment troubleshooting and maintenance

### Search Strategies
- **High Precision**: Score threshold 0.7+ with metadata filtering
- **Medium Precision**: Score threshold 0.5+ for broader matches
- **Broad Fallback**: Score threshold 0.3+ for comprehensive coverage

## 🔧 Configuration Files

### Main Configuration (`config.py`)
- LLM provider selection (OpenAI, Mistral, Ollama)
- Database connection strings
- Vector database settings
- Debug and logging configuration

### Docker Configuration (`docker-compose.yml`)
- Multi-service orchestration
- Environment variable injection
- Network configuration and service dependencies
- Volume management for persistent storage

### Requirements
- **MCP Server**: `requirements.mcp.txt` - FastAPI, LangChain, vector clients
- **Database Server**: `requirements.database.txt` - Async SQLite, data processing

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check your .env file contains all required keys
cat .env | grep -E "(OPENAI|MISTRAL|AZURE)"
```

#### 2. Docker Container Issues
```bash
# Clean up and rebuild
docker compose down
docker system prune -f
docker compose up --build
```

#### 3. Database Connection Issues
- Verify database_server is running on port 8762
- Check Docker network connectivity between services

#### 4. Vector Search Not Working
- Ensure Qdrant is running on port 6333
- Verify MISTRAL_API_KEY is valid for embeddings
- Check collection initialization status

### Health Checks

Visit `http://localhost:8009/health` to check:
- MCP Server status
- Database connectivity
- API key configuration
- LLM provider availability

### Logs and Debugging

```bash
# View service logs
docker compose logs mcp_server
docker compose logs database_server
docker compose logs qdrant
docker compose logs open-webui

# Enable debug mode
export DEBUG=true
```

## 🚀 Advanced Usage

### Custom LLM Providers

The system supports multiple LLM providers. Configure in `config.py`:

```python
# Provider selection
MCP_PROVIDER_OPENAI = True
MCP_PROVIDER_MISTRAL = False  
MCP_PROVIDER_OLLAMA = False
```

### Vector Database Customization

Add new collections for domain-specific knowledge:

```python
# In qdrant_vector_store/
python create_mistral_embeddings_known_questions.py
```

### Healthcare Domain Integration

The system includes pre-built integrations for:
- **Medical Equipment**: IV pumps, monitors, diagnostic tools
- **Environmental Systems**: HVAC, structural monitoring
- **Clinical Workflows**: EWS scoring, patient prioritization
- **Documentation**: COSMIC system manuals and procedures

## 📈 Performance Optimization

### Vector Search Optimization
- Use specific keywords in queries for better retrieval
- Leverage metadata filtering for targeted searches
- Adjust score thresholds based on use case requirements

### Database Query Optimization
- Use natural language queries for complex joins
- Leverage streaming responses for large datasets
- Monitor query performance via health endpoints

### Resource Management
- Monitor container resource usage
- Scale services independently using Docker Compose
- Optimize vector collection sizes based on usage patterns

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow the MCP tool addition process for new functionality
4. Test thoroughly with health checks
5. Submit pull request with detailed description

### Code Standards
- Follow async/await patterns for all I/O operations
- Include comprehensive error handling
- Add detailed docstrings for all functions
- Use type hints for better code maintainability

## 📞 Support

For technical issues or questions:

- **Primary Developer**: Praveen Kehelella
- **System Architecture**: MCP-Orchestrated Multi-Agent System
- **Documentation**: This README and inline code documentation
- **Health Monitoring**: Built-in health checks and logging

---

**Built with ❤️ for Healthcare Innovation by the Omnigate Team**
