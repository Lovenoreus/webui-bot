#!/bin/bash

# Function to attach to a container
attach_container() {
  local name=$1
  echo "Attaching to container: $name"
  docker exec -it "$name" /bin/bash
}

# Check the flag
case "$1" in
  mcp)
    attach_container "webui-bot-mcp_server-1"
    ;;
  db)
    attach_container "webui-bot-database_server-1"
    ;;
  webui)
    attach_container "open-webui"
    ;;
  *)
    echo "Invalid or missing argument."
    echo "Usage:"
    echo "  bash attach.sh mcp     # Attach to MCP server"
    echo "  bash attach.sh db      # Attach to Database server"
    echo "  bash attach.sh webui   # Attach to Web UI"
    exit 1
    ;;
esac
