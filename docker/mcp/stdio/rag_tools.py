"""
RAG (Retrieval-Augmented Generation) Tools for PostgreSQL with pgvector and Ollama
Following the vectodef _initialize_rag_models():
    Initialize RAG LLM and embedding models based on config.py settings
    global rag_llm, rag_embeddings
    
    # Get RAG configuration (lazy initialization)
    config = _get_rag_config()
    
    try:
        # Initialize LLM based on configured provider
        if config.llm_provider == "ollama":ase tools pattern for consistency
Integrated with config.py for centralized configuration management
"""

import os
import asyncio
import traceback
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

import config

class RAGRetrieveRequest(BaseModel):
    """Pydantic model for RAG retrieve requests"""
    query: str = Field(..., description="The query to search for")
    top_k: int = Field(default=3, description="Number of similar documents to retrieve")
    collection_name: str = Field(default="rag_documents_collection", description="Vector collection name")
    language: str = Field(default="Answer in English", description="Response language")


# RAG Configuration using config.py settings
class RAGConfig:
    """RAG configuration using centralized config system"""
    
    def __init__(self):
        # RAG configuration (now optional, used alongside cosmic database conditional approach)
        # Note: RAG functionality is available but may use the cosmic database conditional backend
        
        # PostgreSQL settings from config.py (with environment variable fallbacks)
        self.pg_host = config.RAG_POSTGRES_HOST
        self.pg_port = config.RAG_POSTGRES_PORT
        self.pg_database = config.RAG_POSTGRES_DATABASE
        self.pg_username = config.RAG_POSTGRES_USERNAME
        self.pg_password = config.RAG_POSTGRES_PASSWORD
        
        # Use embedding model from config.py based on enabled provider
        self.embedding_model_name = config.EMBEDDINGS_MODEL_NAME
        
        # LLM settings based on config.py provider selection
        if config.USE_OLLAMA:
            self.llm_provider = "ollama"
            self.llm_model_name = config.AGENT_MODEL_NAME
            self.ollama_base_url = config.OLLAMA_BASE_URL
        elif config.USE_OPENAI:
            self.llm_provider = "openai"
            self.llm_model_name = config.AGENT_MODEL_NAME
            self.openai_api_key = config.OPENAI_API_KEY
        elif config.USE_MISTRAL:
            self.llm_provider = "mistral"
            self.llm_model_name = config.MISTRAL_MODEL_NAME
            self.mistral_api_key = config.MISTRAL_API_KEY
            self.mistral_base_url = config.MISTRAL_BASE_URL
        else:
            raise ValueError("No LLM provider enabled in config. Enable OLLAMA, OPENAI, or MISTRAL.")
        
        # RAG-specific settings from config.py
        self.default_collection_name = config.RAG_DEFAULT_COLLECTION_NAME
        self.default_top_k = config.RAG_DEFAULT_TOP_K
        self.default_language = config.RAG_DEFAULT_LANGUAGE
        
        # Build PostgreSQL connection string
        self.connection_string = (
            f"postgresql+psycopg2://"
            f"{quote_plus(self.pg_username)}:"
            f"{quote_plus(self.pg_password)}@"
            f"{self.pg_host}:"
            f"{self.pg_port}/"
            f"{self.pg_database}"
        )
        
        if config.DEBUG:
            print(f"✅ RAG Config initialized:")
            print(f"   - LLM Provider: {self.llm_provider}")
            print(f"   - LLM Model: {self.llm_model_name}")
            print(f"   - Embedding Model: {self.embedding_model_name}")
            print(f"   - PostgreSQL: {self.pg_host}:{self.pg_port}/{self.pg_database}")
            print(f"   - Default Collection: {self.default_collection_name}")
            print(f"   - Default Top K: {self.default_top_k}")
            print(f"   - Default Language: {self.default_language}")


# Initialize RAG configuration lazily
rag_config = None
rag_llm = None
rag_embeddings = None

def _get_rag_config():
    """Lazy initialization of RAG configuration"""
    global rag_config, rag_llm, rag_embeddings
    if rag_config is None:
        rag_config = RAGConfig()
    return rag_config

def initialize_rag_models():
    """Initialize RAG LLM and embedding models based on config.py settings"""
    global rag_llm, rag_embeddings
    
    try:
        # Initialize LLM based on configured provider
        if rag_config.llm_provider == "ollama":
            rag_llm = OllamaLLM(
                model=rag_config.llm_model_name,
                base_url=rag_config.ollama_base_url,
                temperature=0.0
            )
            # Initialize embeddings for Ollama
            rag_embeddings = OllamaEmbeddings(
                model=rag_config.embedding_model_name,
                base_url=rag_config.ollama_base_url
            )
            
        elif rag_config.llm_provider == "openai":
            rag_llm = ChatOpenAI(
                model=rag_config.llm_model_name,
                api_key=rag_config.openai_api_key,
                temperature=0.0
            )
            # Initialize embeddings for OpenAI
            rag_embeddings = OpenAIEmbeddings(
                model=rag_config.embedding_model_name,
                api_key=rag_config.openai_api_key
            )
            
        elif rag_config.llm_provider == "mistral":
            # For Mistral, we'll use the configured model from config.py
            try:
                from langchain_mistralai import ChatMistralAI
                rag_llm = ChatMistralAI(
                    model=rag_config.llm_model_name,
                    api_key=rag_config.mistral_api_key,
                    endpoint=rag_config.mistral_base_url,
                    temperature=0.0
                )
            except ImportError:
                # Fallback to using the existing AGENT_MODEL from config
                rag_llm = config.AGENT_MODEL
                
            # For embeddings, use OpenAI as fallback since Mistral doesn't have embeddings API
            if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
                rag_embeddings = OpenAIEmbeddings(
                    model=rag_config.embedding_model_name,
                    api_key=config.OPENAI_API_KEY
                )
            else:
                raise ValueError("Mistral provider requires OpenAI API key for embeddings. Please set OPENAI_API_KEY.")
        
        if config.DEBUG:
            print(f"✅ RAG LLM initialized: {rag_config.llm_model_name} ({rag_config.llm_provider})")
            print(f"✅ RAG Embeddings initialized: {rag_config.embedding_model_name}")
            print(f"✅ RAG Connection string: {rag_config.connection_string}")
            
        return True
        
    except Exception as e:
        if config.DEBUG:
            print(f"❌ Failed to initialize RAG models: {e}")
            print(f"   Provider: {rag_config.llm_provider}")
            print(f"   LLM Model: {rag_config.llm_model_name}")
            print(f"   Embedding Model: {rag_config.embedding_model_name}")
        return False


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the RAG chain"""
    return "\n\n".join(doc.page_content for doc in docs)


async def retrieve_documents_tool(query: str, top_k: int = None, 
                                  collection_name: str = None,
                                  language: str = None) -> str:
    """
    Retrieve similar documents and generate answer using RAG with PostgreSQL and configurable LLM providers.
    
    This tool uses the existing RAG infrastructure with PostgreSQL/pgvector for document storage
    and supports multiple LLM providers (Ollama, OpenAI, Mistral) based on config.py settings.
    All model names, collection names, and other settings are read from the centralized config system.

    Args:
        query (str): The query to search for in the document collection
        top_k (int): Number of similar documents to retrieve (uses config default if None)
        collection_name (str): Name of the vector collection to search (uses config default if None)
        language (str): Response language instruction (uses config default if None)

    Returns:
        str: Generated answer with source information, or error message
    """
    
    # Use config defaults if not provided
    top_k = top_k or rag_config.default_top_k
    collection_name = collection_name or rag_config.default_collection_name
    language = language or rag_config.default_language
    
    if config.DEBUG:
        print(f"RAG retrieval started for query: {query}")
        print(f"Provider: {rag_config.llm_provider}")
        print(f"Collection: {collection_name}, Top K: {top_k}, Language: {language}")
        print(f"LLM Model: {rag_config.llm_model_name}")
        print(f"Embedding Model: {rag_config.embedding_model_name}")

    try:
        # Initialize models if not already done
        if not rag_llm or not rag_embeddings:
            if not initialize_rag_models():
                return f"❌ RAG models not initialized. Please check {rag_config.llm_provider} configuration."

        # Create vector store connection
        vectorstore = PGVector(
            embedding_function=rag_embeddings,
            collection_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
            connection_string=rag_config.connection_string
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": top_k}
        )
        
        # Create the Prompt Template
        prompt_template = f"""Use the context provided to answer the user's question below. 
If you do not know the answer based on the context provided, tell the user that you do 
not know the answer to their question based on the context provided and that you are sorry. 
{language}.

Context: {{context}}

Question: {{query}}

Answer: """

        # Create Prompt Instance from template
        custom_rag_prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "query": RunnablePassthrough()}
            | custom_rag_prompt
            | rag_llm
            | StrOutputParser()
        )

        # Query the RAG Chain
        answer = rag_chain.invoke(query)
        
        # Also get the source documents for transparency
        source_docs = retriever.invoke(query)
        
        # Build comprehensive response
        response_parts = [f"**Answer:** {answer}"]
        
        if source_docs:
            response_parts.append(f"\n**Sources ({len(source_docs)} documents):**")
            for i, doc in enumerate(source_docs, 1):
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                metadata_info = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items() if v]) if doc.metadata else "No metadata"
                response_parts.append(f"\n{i}. {content_preview}")
                response_parts.append(f"   Metadata: {metadata_info}")

        final_response = "\n".join(response_parts)

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"RAG retrieval completed successfully")
            print(f"Answer length: {len(answer)} characters")
            print(f"Source documents: {len(source_docs)}")

        return final_response

    except Exception as e:
        error_msg = f"❌ Error in RAG retrieval: {str(e)}"
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"RAG ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        return error_msg


def get_rag_tool_schema() -> Dict[str, Any]:
    """
    Get the MCP tool schema for the retrieve_documents tool.
    This follows the same pattern as other tools in the vector database tools.
    """
    return {
        "name": "retrieve_documents",
        "description": f"Retrieve similar documents and generate answer using RAG with PostgreSQL and {rag_config.llm_provider.upper()}. Use this when you need to search through document collections and provide contextual answers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the document collection"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar documents to retrieve",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                },
                "collection_name": {
                    "type": "string", 
                    "description": "Name of the vector collection to search",
                    "default": rag_config.default_collection_name
                },
                "language": {
                    "type": "string",
                    "description": "Response language instruction",
                    "default": "Answer in English"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }


# Initialize RAG models on module import
initialize_rag_models()

if config.DEBUG:
    print("✅ RAG tools module loaded successfully")
    print(f"   - Provider: {rag_config.llm_provider}")
    print(f"   - LLM Model: {rag_config.llm_model_name}")
    print(f"   - Embedding Model: {rag_config.embedding_model_name}")
    print(f"   - PostgreSQL: {rag_config.pg_host}:{rag_config.pg_port}/{rag_config.pg_database}")
    print(f"   - Default Collection: {rag_config.default_collection_name}")
