#!/usr/bin/env python3
"""
Collection status checker for RAG system during Docker startup.
This script checks if required collections exist and logs appropriate messages.
"""

import os
import sys
import time
import config
from rag2 import Settings, check_collection_exists, log_collection_status

def check_multiple_collections():
    """Check status of multiple collections based on configuration"""
    settings = Settings()
    
    print("🔍 Checking collection status during startup...")
    print(f"📊 Configuration Summary:")
    print(f"   - Database: {settings.postgres.pg_host}:{settings.postgres.pg_port}/{settings.postgres.pg_database}")
    print(f"   - Provider settings: Ollama={settings.use_ollama}, OpenAI={settings.use_openai}, Mistral={settings.use_mistral}")
    
    # List of default collections to check
    collections_to_check = [
        "json_documents_collection",
        "cosmic_documents_collection", 
        "health_documents_collection"
    ]
    
    # Add collection from config if available
    try:
        if hasattr(config, 'COSMIC_DATABASE_COLLECTION_NAME'):
            collections_to_check.append(config.COSMIC_DATABASE_COLLECTION_NAME)
    except:
        pass
    
    # Check each collection
    collections_found = 0
    collections_missing = 0
    
    for collection_name in collections_to_check:
        try:
            exists, message = check_collection_exists(collection_name)
            if exists:
                print(f"✅ Collection '{collection_name}': {message}")
                collections_found += 1
            else:
                print(f"⚠️  Collection '{collection_name}': {message}")
                collections_missing += 1
                
                # Provide initialization guidance
                print(f"💡 To initialize '{collection_name}', run:")
                print(f"   python rag2.py --embed <path_to_json_files> --collection {collection_name}")
                
        except Exception as e:
            print(f"❌ Error checking collection '{collection_name}': {e}")
            collections_missing += 1
    
    # Summary
    print(f"\n📋 Collection Status Summary:")
    print(f"   - Collections found: {collections_found}")
    print(f"   - Collections missing: {collections_missing}")
    print(f"   - Total checked: {len(collections_to_check)}")
    
    # Provider-specific guidance
    print(f"\n🔧 Embedding Provider Status:")
    if settings.use_openai and settings.openai.api_key:
        print(f"   - OpenAI: ✅ Configured ({settings.openai.embedding_model})")
    elif settings.use_openai:
        print(f"   - OpenAI: ⚠️ Enabled but no API key found")
    
    if settings.use_ollama:
        print(f"   - Ollama: ✅ Configured ({settings.ollama.embedding_model})")
        print(f"     URL: {settings.ollama.direct_url}")
    
    if settings.use_mistral:
        print(f"   - Mistral: ✅ Configured")
    
    if not any([settings.use_ollama, settings.use_openai, settings.use_mistral]):
        print(f"   - ❌ No embedding providers configured!")
        return False
    
    return collections_found > 0

def wait_for_database(max_retries=30, retry_delay=2):
    """Wait for PostgreSQL database to be ready"""
    settings = Settings()
    
    print(f"⏳ Waiting for PostgreSQL database at {settings.postgres.pg_host}:{settings.postgres.pg_port}...")
    
    for attempt in range(max_retries):
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=settings.postgres.pg_host,
                port=settings.postgres.pg_port,
                database=settings.postgres.pg_database,
                user=settings.postgres.pg_username,
                password=settings.postgres.pg_password,
                connect_timeout=5
            )
            conn.close()
            print(f"✅ Database connection successful after {attempt + 1} attempts")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Database not ready (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"❌ Database failed to become ready after {max_retries} attempts: {e}")
                return False
    
    return False

def main():
    """Main function for collection checking during startup"""
    print("🚀 RAG Collection Status Checker - Docker Startup")
    print("=" * 50)
    
    # Wait for database to be ready
    if not wait_for_database():
        print("❌ Database is not available. Exiting.")
        sys.exit(1)
    
    # Check collections
    collections_available = check_multiple_collections()
    
    print("\n" + "=" * 50)
    if collections_available:
        print("✅ Startup check completed - Some collections are available")
        print("🎯 RAG system is ready for queries")
    else:
        print("⚠️  Startup check completed - No collections found")
        print("📝 You'll need to embed documents before using the RAG system")
    
    print("🔗 For embedding documents, use:")
    print("   docker exec -it <container_name> python /app/rag2.py --embed <path_to_files>")
    
    return 0 if collections_available else 1

if __name__ == "__main__":
    try:
        print(f"This should not run!!!")
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Collection check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during collection check: {e}")
        sys.exit(1)
