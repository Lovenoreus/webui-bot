#!/bin/bash
# Example script for setting up RAG system in HEALTH environment

set -e

echo "🏥 Setting up RAG System for HEALTH Environment"
echo "================================================"

# Set environment name
export ENVIRONMENT_NAME="HEALTH"
export HEALTH_DOCUMENTS_PATH="/app/health_documents"
export HEALTH_COLLECTION_NAME="health_documents_collection"

# Configuration for health environment
export PG_HOST="postgres"
export PG_DATABASE="vectordb_health"
export USE_OLLAMA="true"
export USE_OPENAI="true"
export OLLAMA_BASE_URL="http://ollama:11434"

echo "Environment: $ENVIRONMENT_NAME"
echo "Collection: $HEALTH_COLLECTION_NAME"
echo "Documents Path: $HEALTH_DOCUMENTS_PATH"

# Check if documents directory exists
if [ ! -d "$HEALTH_DOCUMENTS_PATH" ]; then
    echo "⚠️  Creating health documents directory: $HEALTH_DOCUMENTS_PATH"
    mkdir -p "$HEALTH_DOCUMENTS_PATH"
    
    # Create example health document
    cat > "$HEALTH_DOCUMENTS_PATH/sample_health_data.json" << 'EOF'
[
    {
        "text": "CDS FH Referral is a clinical decision support system for Familial Hypercholesterolemia referrals. It helps healthcare providers identify patients who may benefit from specialized lipid clinic referrals.",
        "type": "clinical_guidance",
        "category": "cardiology",
        "source": "clinical_guidelines_2024"
    },
    {
        "text": "Patient assessment for FH includes family history review, lipid profile analysis, and clinical examination for xanthomas. LDL-C levels above 190 mg/dL in adults warrant FH evaluation.",
        "type": "clinical_protocol",
        "category": "lipid_management",
        "source": "clinical_protocols_2024"
    }
]
EOF
    echo "✅ Created sample health document"
fi

# Check system status
echo "🔍 Checking system status..."
python check_collections.py

# Check if collection exists
echo "🔍 Checking if $HEALTH_COLLECTION_NAME exists..."
COLLECTION_EXISTS=$(python -c "
from rag2 import check_collection_exists
exists, message = check_collection_exists('$HEALTH_COLLECTION_NAME')
print('true' if exists else 'false')
" 2>/dev/null || echo "false")

if [ "$COLLECTION_EXISTS" = "false" ]; then
    echo "📥 Embedding health documents into $HEALTH_COLLECTION_NAME..."
    python rag2.py --embed "$HEALTH_DOCUMENTS_PATH" --collection "$HEALTH_COLLECTION_NAME" --chunk-size 1000 --overlap 200
    echo "✅ Health documents embedded successfully"
else
    echo "✅ Collection $HEALTH_COLLECTION_NAME already exists"
fi

# Test query functionality
echo "🔍 Testing query functionality..."
echo "Query: 'What is CDS FH Referral?'"
python rag2.py --query "What is CDS FH Referral?" --collection "$HEALTH_COLLECTION_NAME" --language English --top-k 3

echo ""
echo "🎯 HEALTH Environment Setup Complete!"
echo "================================================"
echo "Next steps:"
echo "1. Add your health documents to: $HEALTH_DOCUMENTS_PATH"
echo "2. Re-run embedding: python rag2.py --embed $HEALTH_DOCUMENTS_PATH --collection $HEALTH_COLLECTION_NAME"
echo "3. Query documents: python rag2.py --query 'your question' --collection $HEALTH_COLLECTION_NAME"
echo ""
echo "Configuration Summary:"
echo "- Environment: $ENVIRONMENT_NAME"
echo "- Collection: $HEALTH_COLLECTION_NAME"
echo "- Database: $PG_HOST/$PG_DATABASE"
echo "- Providers: Ollama=$USE_OLLAMA, OpenAI=$USE_OPENAI"
