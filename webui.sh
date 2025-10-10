#!/bin/bash

# --- Detect host OS even if Bash reports 'Linux' ---
if [[ -n "$WINDIR" || -n "$SystemRoot" ]]; then
    echo "Host: Windows"
elif [[ "$(uname -r)" =~ "microsoft" ]]; then
    echo "Host: Windows (WSL)"
else
    echo "Host: Linux"
fi

echo "Shutting down all docker containers"
docker compose down 

# Function for application startup
start_app() {
  echo "Starting application..."
  docker compose up
}

# Check for flags
if [ "$1" == "--mcp" ]; then
  echo "Building mcp_server..."
  docker compose build mcp_server
  start_app

elif [ "$1" == "--db" ]; then
  echo "Building database_server..."
  docker compose build database_server
  start_app

elif [ "$1" == "--all" ]; then
  echo "Building everything..."
  docker compose build
  start_app

elif [ "$1" == "--run" ]; then
  echo "Re-running all containers..."
  docker compose up 


elif [ "$1" == "--down" ]; then
  echo "Shutting down all containers"
  docker compose down 

else
  echo "  bash run_webui.sh --mcp     Build only mcp_server"
  echo "  bash run_webui.sh --db      Build only database_server"
  echo "  bash run_webui.sh --all     Build all services"
  echo "  bash run_webui.sh --usage   Show this help message"
  echo "  bash run_webui.sh --run     Re-run all containers"
  echo "  bash run_webui.sh           Shutdown all containers"
fi
