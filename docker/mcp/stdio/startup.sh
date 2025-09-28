#!/bin/bash
# Docker startup script for RAG system with collection checking

set -e

echo "🚀 Starting RAG System with Collection Checking..."
echo "=================================================="

# Set working directory
cd /app

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if we're in Docker environment
if [ -f /.dockerenv ]; then
    log "✅ Running in Docker environment"
    export IN_DOCKER=true
else
    log "⚠️  Not running in Docker environment"
    export IN_DOCKER=false
fi

# Wait a bit for other services to start
log "⏳ Waiting for services to initialize..."
sleep 5

# Check Python environment
log "🐍 Checking Python environment..."
python --version
pip list | grep -E "(langchain|psycopg2|openai|ollama)" || true

# Run collection status check
log "🔍 Running collection status check..."
python docker/mcp/stdio/check_collections.py

CHECK_EXIT_CODE=$?

if [ $CHECK_EXIT_CODE -eq 0 ]; then
    log "✅ Collection check passed - Collections are available"
elif [ $CHECK_EXIT_CODE -eq 1 ]; then
    log "⚠️  Collection check completed - No collections found (this is normal for first run)"
else
    log "❌ Collection check failed with unexpected error"
fi

# Display configuration summary
log "📊 Configuration Summary:"
echo "   - PG_HOST: ${PG_HOST:-localhost}"
echo "   - PG_PORT: ${PG_PORT:-5432}"  
echo "   - PG_DATABASE: ${PG_DATABASE:-vectordb}"
echo "   - USE_OLLAMA: ${USE_OLLAMA:-true}"
echo "   - USE_OPENAI: ${USE_OPENAI:-false}"
echo "   - OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://localhost:11434}"

# Check if OpenAI API key is set (without exposing it)
if [ -n "${OPENAI_API_KEY}" ]; then
    echo "   - OPENAI_API_KEY: ✅ Set"
else
    echo "   - OPENAI_API_KEY: ❌ Not set"
fi

log "🎯 RAG system startup completed"
log "📖 Usage examples:"
echo "   # Embed documents:"
echo "   python rag2.py --embed /path/to/documents --collection my_collection"
echo ""
echo "   # Query documents:"  
echo "   python rag2.py --query 'Your question here' --collection my_collection"
echo ""
echo "   # Health check collections:"
echo "   python check_collections.py"

echo "=================================================="
log "✅ Startup script completed successfully"

# Keep container running if this is the main process
if [ "$1" = "--keep-alive" ]; then
    log "🔄 Keeping container alive..."
    tail -f /dev/null
fi
