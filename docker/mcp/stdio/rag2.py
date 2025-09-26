import os
import json
import argparse
from tqdm import tqdm

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus

# Simple configuration classes - no external dependencies
class PostgresSettings:
    pg_host = os.getenv("PG_HOST", "localhost")  # Use "postgres" when running in Docker
    pg_port = os.getenv("PG_PORT", "5432")
    pg_database = os.getenv("PG_DATABASE", "vectordb")
    pg_username = os.getenv("PG_USERNAME", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "password")

class OllamaSettings:
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    direct_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Use "http://ollama:11434" in Docker

class Settings:
    def __init__(self):
        self.postgres = PostgresSettings()
        self.ollama = OllamaSettings()

# Initialize settings
settings = Settings()
print(f"✅ Configuration loaded:")
print(f"   - Database: {settings.postgres.pg_host}:{settings.postgres.pg_port}/{settings.postgres.pg_database}")
print(f"   - Ollama: {settings.ollama.direct_url}")
print(f"   - LLM Model: {settings.ollama.llm_model}")
print(f"   - Embedding Model: {settings.ollama.embedding_model}")

# Build connection string
CONNECTION_STR = (
    f"postgresql+psycopg2://"
    f"{quote_plus(settings.postgres.pg_username)}:"
    f"{quote_plus(settings.postgres.pg_password)}@"
    f"{settings.postgres.pg_host}:"
    f"{settings.postgres.pg_port}/"
    f"{settings.postgres.pg_database}"
)

# Initialize Ollama models with error handling
try:
    llm = OllamaLLM(
        model=settings.ollama.llm_model,
        base_url=settings.ollama.direct_url,
        temperature=0.0
    )
    embeddings = OllamaEmbeddings(
        model=settings.ollama.embedding_model, 
        base_url=settings.ollama.direct_url
    )
    print(f"✅ Ollama models initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Ollama models: {e}")
    print("Please ensure Ollama is running and accessible")
    exit(1)


def load_json_file(file_path):
    """Load and parse JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_json_as_documents(file_path, chunk_size=1000, overlap=200):
    """
    Load JSON file and convert to LangChain documents.
    Handles both single JSON objects and arrays of JSON objects.
    """
    json_data = load_json_file(file_path)
    if not json_data:
        return []
    
    documents = []
    
    # Handle array of JSON objects (like your big_chunks.json)
    if isinstance(json_data, list):
        for idx, item in tqdm(enumerate(json_data), total=len(json_data), desc=f"Processing {os.path.basename(file_path)}", unit="item", leave=False):
            if isinstance(item, dict) and 'text' in item:
                # Handle objects with 'text' field
                text = item['text']
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source": file_path,
                        "item_index": idx,
                        "chunk_index": ci,
                        "original_item": item  # Store the original JSON object
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            elif isinstance(item, str):
                # Handle array of strings
                chunks = chunk_text(item, chunk_size=chunk_size, overlap=overlap) or [item]
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source": file_path,
                        "item_index": idx,
                        "chunk_index": ci
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            else:
                # Handle other object types - convert to string
                text = json.dumps(item, ensure_ascii=False, indent=2)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
                for ci, chunk in enumerate(chunks):
                    metadata = {
                        "source": file_path,
                        "item_index": idx,
                        "chunk_index": ci,
                        "original_item": item
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
    
    # Handle single JSON object
    elif isinstance(json_data, dict):
        if 'text' in json_data:
            text = json_data['text']
        else:
            text = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap) or [text]
        for ci, chunk in enumerate(chunks):
            metadata = {
                "source": file_path,
                "chunk_index": ci,
                "original_item": json_data
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    # Handle string data
    elif isinstance(json_data, str):
        chunks = chunk_text(json_data, chunk_size=chunk_size, overlap=overlap) or [json_data]
        for ci, chunk in enumerate(chunks):
            metadata = {
                "source": file_path,
                "chunk_index": ci
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents


def load_jsons_as_documents(dir_path, limit=100, chunk_size=1000, overlap=200):
    """Load multiple JSON files from a directory"""
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(".json")]
    files = sorted(files)[:limit]
    documents = []
    
    # Add progress bar for loading multiple files
    for file_idx, file_path in tqdm(enumerate(files), total=len(files), desc="Loading JSON files", unit="file"):
        file_documents = load_json_as_documents(file_path, chunk_size, overlap)
        # Update metadata to include file index
        for doc in file_documents:
            doc.metadata["file_index"] = file_idx
        documents.extend(file_documents)
    
    return documents


def get_vectorstorage(docs, embeddings, collection_name="json_documents_collection"):
    """Create vector storage from documents"""
    return PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)


def save_to_pgvector(documents, embeddings, collection_name="json_documents_collection"):
    """Save documents to PGVector database with progress tracking"""
    try:
        print(f"🚀 Generating embeddings for {len(documents)} documents...")
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings with progress bar
        embedded_texts = []
        batch_size = 10  # Process embeddings in batches to show progress
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            try:
                # Generate embeddings for this batch
                batch_embeddings = embeddings.embed_documents(batch_texts)
                embedded_texts.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback: generate embeddings one by one for this batch
                for text in batch_texts:
                    try:
                        embedding = embeddings.embed_documents([text])
                        embedded_texts.extend(embedding)
                    except Exception as single_error:
                        print(f"❌ Error embedding single document: {single_error}")
                        # Use zero vector as fallback
                        embedded_texts.append([0.0] * 1536)  # Assuming 1536 dimensions for nomic-embed-text
        
        print(f"✅ Generated {len(embedded_texts)} embeddings")
        print(f"💾 Storing documents in PGVector database...")
        
        # Create vectorstore with pre-computed embeddings
        vectorstore = PGVector.from_embeddings(
            text_embeddings=list(zip(texts, embedded_texts)),
            embedding=embeddings,
            metadatas=[doc.metadata for doc in documents],
            collection_name=collection_name,
            connection_string=CONNECTION_STR,
            use_jsonb=True,
        )
        
        print(f"✅ Saved {len(documents)} documents to collection '{collection_name}'")
        return vectorstore
    except Exception as e:
        print(f"❌ Error saving to PGVector: {e}")
        raise


def embedd_and_store_json_documents(json_path, limit=100, chunk_size=1000, overlap=200, collection_name="json_documents_collection"):
    """Main function to embed and store JSON documents"""
    print(f"📂 Processing: {json_path}")
    
    if os.path.isfile(json_path):
        # Single file
        print(f"📄 Loading JSON file...")
        docs = load_json_as_documents(json_path, chunk_size, overlap)
        print(f"✅ Loaded single JSON file with {len(docs)} document chunks")
    elif os.path.isdir(json_path):
        # Directory of JSON files
        print(f"📁 Loading JSON files from directory...")
        docs = load_jsons_as_documents(json_path, limit, chunk_size, overlap)
        print(f"✅ Loaded directory with {len(docs)} document chunks from multiple files")
    else:
        print(f"❌ Path {json_path} is neither a file nor a directory")
        return
    
    if not docs:
        print("❌ No documents found to process")
        return
    
    # Show document statistics
    print(f"\n📊 Document Statistics:")
    print(f"   Total document chunks: {len(docs)}")
    print(f"   Average chunk length: {sum(len(doc.page_content) for doc in docs) // len(docs)} characters")
    print(f"   Collection name: {collection_name}")
    
    save_to_pgvector(docs, embeddings, collection_name)
    print(f"\n🎉 Successfully processed {len(docs)} documents!")


# Create Document Parsing Function to String
def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_similar_documents(user_prompt, top_k=3, collection_name="json_documents_collection"):
    """Retrieve similar documents and generate answer"""
    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # Create the Prompt Template
    prompt_template = """Use the context provided to answer
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry. Answer in Swedish.

    context: {context}

    question: {query}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Query the RAG Chain
    answer = rag_chain.invoke(user_prompt)
    print("\nAnswer:: ", answer)
    return answer


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="RAG system for JSON documents using PostgreSQL with pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed JSON file with default settings
  python rag2.py --embed /path/to/file.json
  
  # Embed with custom chunk size and overlap
  python rag2.py --embed /path/to/file.json --chunk-size 1500 --overlap 300
  
  # Query the embedded documents
  python rag2.py --query "What is CDS FH Referral?" --top-k 3
  
  # Embed and then query
  python rag2.py --embed /path/to/file.json --query "Your question here"
  
  # Use custom collection name
  python rag2.py --embed /path/to/file.json --collection my_collection
        """
    )
    
    # Main action arguments
    parser.add_argument(
        "--embed", 
        type=str, 
        help="Path to JSON file or directory to embed and store"
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query string to search for similar documents"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Size of text chunks (default: 1000)"
    )
    
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=200, 
        help="Overlap between chunks (default: 200)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of similar documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--collection", 
        type=str, 
        default="json_documents_collection", 
        help="Collection name in vector database (default: json_documents_collection)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Maximum number of files to process from directory (default: 100)"
    )
    
    # Language and output options
    parser.add_argument(
        "--language", 
        type=str, 
        default="Swedish", 
        choices=["Swedish", "English", "German", "French", "Spanish"],
        help="Language for the response (default: Swedish)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.embed and not args.query:
        parser.error("You must specify either --embed or --query (or both)")
    
    if args.embed and not os.path.exists(args.embed):
        parser.error(f"Path does not exist: {args.embed}")
    
    # Set verbose mode
    if args.verbose:
        print(f"Settings: {settings}")
        print(f"Connection string: {CONNECTION_STR}")
    
    # Embedding phase
    if args.embed:
        print(f"Embedding documents from: {args.embed}")
        print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
        print(f"Collection: {args.collection}")
        
        embedd_and_store_json_documents(
            json_path=args.embed,
            limit=args.limit,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            collection_name=args.collection
        )
        
        print("✅ Embedding completed successfully!")
    
    # Query phase
    if args.query:
        print(f"\n🔍 Searching for: '{args.query}'")
        print(f"Top K: {args.top_k}, Collection: {args.collection}")
        
        # Update prompt template based on language
        language_prompts = {
            "Swedish": "Answer in Swedish",
            "English": "Answer in English", 
            "German": "Answer in German",
            "French": "Answer in French",
            "Spanish": "Answer in Spanish"
        }
        
        # Temporarily modify the retrieve function to use custom language
        answer = retrieve_similar_documents_with_language(
            user_prompt=args.query,
            top_k=args.top_k,
            collection_name=args.collection,
            language=language_prompts[args.language]
        )
        
        print(f"\n📋 Final Answer:\n{answer}")


def retrieve_similar_documents_with_language(user_prompt, top_k=3, collection_name="json_documents_collection", language="Answer in Swedish"):
    """Retrieve similar documents and generate answer with custom language"""
    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STR)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # Create the Prompt Template with custom language
    prompt_template = f"""Use the context provided to answer
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry. {language}.

    context: {{context}}

    question: {{query}}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the RAG Chain
    rag_chain = (
        {{"context": retriever | format_docs, "query": RunnablePassthrough()}}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Query the RAG Chain
    answer = rag_chain.invoke(user_prompt)
    return answer


if __name__ == "__main__":
    main()
