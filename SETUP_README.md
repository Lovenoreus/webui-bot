# WebUI Bot - Setup Guide 🚀

This guide provides comprehensive setup instructions for the WebUI Bot project, an enhanced version of Open WebUI with Model Context Protocol (MCP) integration, Azure Active Directory authentication, and database server capabilities.

## 📋 Prerequisites

Before setting up the project, ensure you have the following installed:

- **Git** - For cloning the repository
- **Docker** - Version 20.10 or higher
- **Docker Compose** - Version 2.0 or higher
- **Node.js** - Version 18+ (for frontend development)
- **Python** - Version 3.11+ (for local development)

## 📥 Repository Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Lovenoreus/webui-bot
cd webui-bot
```

### 2. Environment Configuration (Required)

This project requires **ONE** `.env` file in the root directory containing all environment variables:

#### Single `.env` File Setup (Required)

Create a `.env` file in the **root directory** of the project:

```bash
# Create .env file in root directory
cp .env.example .env
```

Edit the root `.env` file with the following configuration:

```env
# Ollama URL for the backend to connect
OLLAMA_BASE_URL='http://localhost:11434'

# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Azure Active Directory Configuration (Required for MCP Server)
AZURE_CLIENT_ID=your_azure_client_id_here
AZURE_TENANT_ID=your_azure_tenant_id_here
AZURE_CLIENT_SECRET=your_azure_client_secret_here

# Additional API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Optional configurations
OPENAI_API_BASE_URL=''
CORS_ALLOW_ORIGIN='*'
FORWARDED_ALLOW_IPS='*'
SCARF_NO_ANALYTICS=true
DO_NOT_TRACK=true
ANONYMIZED_TELEMETRY=false
```

### Environment Variables Explanation

All environment variables are now consolidated in the single root `.env` file:

| Variable | Description | Required | Where to Get |
|----------|-------------|----------|--------------|
| `OPENAI_API_KEY` | OpenAI API key for both main application and MCP server | **Yes** | OpenAI Platform → API Keys |
| `AZURE_CLIENT_ID` | Azure AD application client ID for MCP authentication | **Yes** | Azure Portal → App Registrations |
| `AZURE_TENANT_ID` | Azure AD tenant ID for your organization | **Yes** | Azure Portal → Azure Active Directory |
| `AZURE_CLIENT_SECRET` | Azure AD application secret key | **Yes** | Azure Portal → App Registrations → Certificates & secrets |
| `MISTRAL_API_KEY` | Mistral AI API key for MCP server | **Yes** | Mistral AI Platform |
| `OPENWEATHER_API_KEY` | OpenWeather API key for weather functionalities | **Yes** | OpenWeatherMap → API Keys |
| `OLLAMA_BASE_URL` | Ollama server URL | No | Local Ollama installation |

## 🐳 Docker Setup

This project uses a multi-service Docker architecture with the following services:

- **open-webui**: Main web interface service (Port 3000)
- **mcp_server**: Model Context Protocol server (Port 8009)
- **database_server**: Database service (Port 8762)

### How Environment Variables are Used

The `docker-compose.yaml` file is configured to read all environment variables from the single root `.env` file:

#### Main Application (open-webui service):
```yaml
environment:
  - 'OPENAI_API_KEY=${OPENAI_API_KEY}'  # Reads from root .env file
```

#### MCP Server (mcp_server service):
```yaml
environment:
  - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
  - AZURE_TENANT_ID=${AZURE_TENANT_ID}
  - AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET}
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - MISTRAL_API_KEY=${MISTRAL_API_KEY}
  - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
```

All environment variables are now passed directly from the root `.env` file to the respective services.

### Quick Start with Docker Compose (Recommended)

1. **Ensure the .env file is properly configured** (see Environment Configuration section above)

2. **Start all services**:
```bash
docker-compose up -d
```

3. **View logs**:
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f open-webui
docker-compose logs -f mcp_server
docker-compose logs -f database_server
```

4. **Stop all services**:
```bash
docker-compose down
```

5. **Stop and remove volumes** (WARNING: This will delete all data):
```bash
docker-compose down -v
```

### Individual Service Management

If you need to build and run services individually:

```bash
# Build the main application
docker build -t webui-bot .

# Build MCP server
docker build -f docker/DockerFIle.mcp -t mcp-server ./docker

# Build database server
docker build -f docker/DockerFile.database -t database-server ./docker

# Run individual services with environment variables from .env file
docker run -d -p 3000:8080 --env-file .env webui-bot
docker run -d -p 8009:8009 --env-file .env -v $(pwd):/app mcp-server
docker run -d -p 8762:8762 database-server
```

## 🌐 Access Points

After successful startup, you can access the following services:

| Service | URL | Description |
|---------|-----|-------------|
| **Main Application** | http://localhost:3000 | Primary WebUI interface |
| **MCP Server** | http://localhost:8009 | Model Context Protocol server |
| **Database Server** | http://localhost:8762 | Database management interface |

**Happy coding! 🎉**