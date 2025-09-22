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

This project requires **TWO** separate `.env` files in different locations:

#### File 1: Root Directory `.env` (Required)

Create a `.env` file in the **root directory** of the project:

```bash
# Create .env file in root directory
cp .env.example .env
```

Edit the root `.env` file with the following configuration:

```env

# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional configurations

OLLAMA_BASE_URL=''
OPENAI_API_BASE_URL=''
CORS_ALLOW_ORIGIN='*'
FORWARDED_ALLOW_IPS='*'
SCARF_NO_ANALYTICS=true
DO_NOT_TRACK=true
ANONYMIZED_TELEMETRY=false
```

#### File 2: MCP Server `.env` (Required)

Create a `.env` file in the **docker/mcp/stdio/** directory:

```bash
# The directory already exists, create/edit the .env file
touch docker/mcp/stdio/.env
```

Edit the MCP `.env` file with the following configuration:

```env
# Azure Active Directory Configuration (Required)
AZURE_CLIENT_ID=your_azure_client_id_here
AZURE_TENANT_ID=your_azure_tenant_id_here
AZURE_CLIENT_SECRET=your_azure_client_secret_here

# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### Environment Variables Explanation

#### Root `.env` File Variables:

| Variable | Description | Required | Where to Get |
|----------|-------------|----------|--------------|
| `OPENAI_API_KEY` | OpenAI API key for main application | **Yes** | OpenAI Platform → API Keys |
| `OLLAMA_BASE_URL` | Ollama server URL | No | Local Ollama installation |

#### MCP `.env` File Variables:

| Variable | Description | Required | Where to Get |
|----------|-------------|----------|--------------|
| `AZURE_CLIENT_ID` | Azure AD application client ID | **Yes** | Azure Portal → App Registrations |
| `AZURE_TENANT_ID` | Azure AD tenant ID | **Yes** | Azure Portal → Azure Active Directory |
| `AZURE_CLIENT_SECRET` | Azure AD application secret | **Yes** | Azure Portal → App Registrations → Certificates & secrets |
| `OPENAI_API_KEY` | OpenAI API key for MCP server | **Yes** | OpenAI Platform → API Keys |
| `MISTRAL_API_KEY` | Mistral AI API key | **Yes** | Mistral AI Platform |
| `OPENWEATHER_API_KEY` | OpenWeather API key | **Yes** | OpenWeatherMap → API Keys |

## 🐳 Docker Setup

This project uses a multi-service Docker architecture with the following services:

- **open-webui**: Main web interface service (Port 3000)
- **mcp_server**: Model Context Protocol server (Port 8009)
- **database_server**: Database service (Port 8762)

### How Environment Variables are Used

The `docker-compose.yaml` file is configured to read environment variables as follows:

#### Main Application (open-webui service):
```yaml
environment:
  - 'OPENAI_API_KEY=${OPENAI_API_KEY}'  # Reads from root .env file
```

#### MCP Server (mcp_server service):
```yaml
volumes:
  - .:/app  # Mounts entire project, including docker/mcp/stdio/.env
```

The MCP server reads its environment variables from `docker/mcp/stdio/.env` when the container starts.

### Quick Start with Docker Compose (Recommended)

1. **Ensure both .env files are properly configured** (see Environment Configuration section above)

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

# Run individual services
docker run -d -p 3000:8080 --env-file .env webui-bot
docker run -d -p 8009:8009 -v $(pwd):/app mcp-server
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