#!/bin/bash
echo "Shutting down all docker containers"
docker compose down 

# --- Detect host OS even if Bash reports 'Linux' ---
if [[ -n "$WINDIR" || -n "$SystemRoot" ]]; then
    echo "Host: Windows"
elif [[ "$(uname -r)" =~ "microsoft" ]]; then
    echo "Host: Windows (WSL)"
else
    echo "Host: Linux"
fi

# Default: use certs enabled
USE_CERTS_FLAG=true

# Parse optional cert flag
for arg in "$@"; do
  if [ "$arg" == "--no-certs" ]; then
    USE_CERTS_FLAG=false
  fi
done

# Function for application startup
start_app() {
  echo "Starting application..."
  docker compose up
}

# Track which services to build
BUILD_MCP=false
BUILD_DB=false
BUILD_WEBUI=false

# Parse main build targets
for arg in "$@"; do
  case "$arg" in
    mcp)
      BUILD_MCP=true
      ;;
    db)
      BUILD_DB=true
      ;;
    webui)
      BUILD_WEBUI=true
      ;;
  esac
done

# If no targets given, show usage
if ! $BUILD_MCP && ! $BUILD_DB && ! $BUILD_WEBUI; then
  echo "Usage:"
  echo "bash run_webui.sh mcp [--no-certs]       >> build only mcp_server"
  echo "bash run_webui.sh db [--no-certs]        >> build only database_server"
  echo "bash run_webui.sh webui [--no-certs]     >> build only web UI"
  echo "bash run_webui.sh mcp db [--no-certs]    >> build mcp + database servers"
  echo "bash run_webui.sh webui mcp db [--no-certs] >> build everything"
  echo "bash run_webui.sh --run                  >> re-run all containers"
  echo "bash run_webui.sh --down                 >> shut down all containers"
  exit 1
fi

# Run builds based on flags
if $BUILD_MCP; then
  echo "Building mcp_server (USE_CERTS=$USE_CERTS_FLAG)..."
  docker compose build --build-arg USE_CERTS=$USE_CERTS_FLAG mcp_server
fi

if $BUILD_DB; then
  echo "Building database_server (USE_CERTS=$USE_CERTS_FLAG)..."
  docker compose build --build-arg USE_CERTS=$USE_CERTS_FLAG database_server
fi

if $BUILD_WEBUI; then
  echo "Building open-webui (USE_CERTS=$USE_CERTS_FLAG)..."
  docker compose build --build-arg USE_CERTS=$USE_CERTS_FLAG open-webui
fi

# Start app after builds
start_app
