#!/bin/bash

# Vanna SQL Test Runner
# This script runs the Vanna SQL test suite

echo "🧪 Vanna SQL Test Suite Runner"
echo "================================"
echo "Features: CSV export, result comparison, performance plotting"
echo ""

# Check if MCP server is running
echo "🔍 Checking if MCP server is running..."
if curl -s http://localhost:8009/health > /dev/null; then
    echo "✅ MCP server is running"
else
    echo "❌ MCP server is not running at http://localhost:8009"
    echo "Please start the MCP server first:"
    echo "  docker-compose up mcp_server"
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   The test will use simple string comparison for grading"
    echo "   Set OPENAI_API_KEY for more accurate grading with GPT-4o-mini"
fi

# Run the test
echo ""
echo "🚀 Starting Vanna SQL tests..."
echo ""

python test_vanna_sql.py

echo ""
echo "✅ Test completed!"
echo "📊 Results saved to test_results/ directory:"
echo "   - CSV file with detailed results"
echo "   - JSON file with full test data"
echo "   - Performance plots (PNG files)"
echo "   - Comparison with previous runs"