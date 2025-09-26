import sys
import openai
from dotenv import load_dotenv
import requests
import config
from typing import List, Optional, Tuple, Generator
import json
import os
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage


import httpx
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from langchain_openai import OpenAIEmbeddings
import traceback

load_dotenv()


async def async_embed_with_fallback(
        query: str,
        ollama_model: str = None,
        ollama_base_url: str = None,
        openai_embedder: Optional[OpenAIEmbeddings] = None,
        timeout: float = 10.0
) -> List[float]:
    """Async embedding with Ollama primary and OpenAI fallback"""

    # Try Ollama first if configured
    if config.USE_OLLAMA and ollama_base_url and ollama_model:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{ollama_base_url}/api/embeddings",
                    json={"model": ollama_model, "prompt": query}
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding')
                    if embedding and len(embedding) > 0:
                        if config.DEBUG:
                            print(f"Ollama embedding success: {len(embedding)} dimensions")
                        return embedding

        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            if config.DEBUG:
                print(f"Ollama embedding failed: {type(e).__name__} - {e}")
        except Exception as e:
            if config.DEBUG:
                print(f"Ollama unexpected error: {e}")

    # Fallback to OpenAI
    if config.USE_OPENAI and openai_embedder:
        try:
            if config.DEBUG:
                print("Falling back to OpenAI embedding")
            embedding = await asyncio.to_thread(openai_embedder.embed_query, query)
            if embedding and len(embedding) > 0:
                if config.DEBUG:
                    print(f"OpenAI embedding success: {len(embedding)} dimensions")
                return embedding
        except Exception as e:
            if config.DEBUG:
                print(f"OpenAI embedding failed: {e}")

    return []


def intelligent_metadata_filter(query: str) -> Optional[rest.Filter]:
    """Build filters based on query intent for medical documentation"""
    try:
        query_lower = query.lower()
        should_conditions = []

        intent_patterns = {
            "definition": {
                "indicators": ["what is", "define", "definition", "explain", "meaning", "describe"],
                "content_type": ["narrative", "definition"],
            },
            "process": {
                "indicators": ["how to", "process", "procedure", "steps", "workflow", "method"],
                "content_type": ["list", "procedure", "process"],
                "hierarchical_tags": ["workflow", "process", "procedure"]
            },
            "benefits": {
                "indicators": ["benefits", "advantages", "pros", "strengths", "value"],
                "hierarchical_tags": ["benefits", "documentation benefits", "advantages"],
            },
            "compliance": {
                "indicators": ["compliance", "hipaa", "gdpr", "standards", "regulatory", "audit"],
                "clinical_domain": ["compliance", "data security", "regulatory"],
            },
            "audit": {
                "indicators": ["audit", "trail", "log", "tracking", "history", "record"],
                "keywords": ["audit trail", "audit", "log", "tracking"],
                "hierarchical_tags": ["audit overview", "audit definitions", "audit trail"],
            },
            "technical": {
                "indicators": ["system", "software", "technical", "implementation", "architecture"],
                "content_type": ["technical", "system"],
                "clinical_domain": ["system", "technical"]
            },
            "clinical": {
                "indicators": ["clinical", "medical", "patient", "care", "treatment"],
                "clinical_domain": ["clinical", "medical"],
                "content_type": ["clinical", "medical"]
            },
            "documentation": {
                "indicators": ["documentation", "document", "record", "chart", "note"],
                "hierarchical_tags": ["documentation", "medical documentation"],
                "content_type": ["documentation"]
            }
        }

        # Apply intent-based filtering
        for intent, config_pattern in intent_patterns.items():
            if any(indicator in query_lower for indicator in config_pattern["indicators"]):
                # Content type filtering
                if "content_type" in config_pattern:
                    should_conditions.append(
                        rest.FieldCondition(
                            key="metadata.content_type",
                            match=rest.MatchAny(any=config_pattern["content_type"])
                        )
                    )

                # Hierarchical tags filtering
                if "hierarchical_tags" in config_pattern:
                    should_conditions.append(
                        rest.FieldCondition(
                            key="metadata.hierarchical_tags",
                            match=rest.MatchAny(any=config_pattern["hierarchical_tags"])
                        )
                    )

                # Clinical domain filtering
                if "clinical_domain" in config_pattern:
                    should_conditions.append(
                        rest.FieldCondition(
                            key="metadata.clinical_domain",
                            match=rest.MatchAny(any=config_pattern["clinical_domain"])
                        )
                    )

                # Keywords filtering
                if "keywords" in config_pattern:
                    should_conditions.append(
                        rest.FieldCondition(
                            key="metadata.keywords",
                            match=rest.MatchAny(any=config_pattern["keywords"])
                        )
                    )
                break

        # Always include broad keyword matching
        query_words = [w for w in query_lower.split() if len(w) > 2]
        if query_words:
            should_conditions.append(
                rest.FieldCondition(
                    key="metadata.keywords",
                    match=rest.MatchAny(any=query_words)
                )
            )
            should_conditions.append(
                rest.FieldCondition(
                    key="metadata.context_summary",
                    match=rest.MatchText(text=" ".join(query_words))
                )
            )

        return rest.Filter(should=should_conditions) if should_conditions else None

    except Exception as e:
        if config.DEBUG:
            print(f"Metadata filter error: {e}")
        return None


async def llm_intelligent_metadata_filter(query: str) -> Optional[rest.Filter]:
    """Use LLM to generate intelligent metadata filters based on query intent"""

    prompt = f"""You are a metadata filter generator for a medical documentation search system.

Analyze this query and determine the most relevant metadata filters:

Query: "{query}"

CRITICAL LANGUAGE INSTRUCTIONS:
- The query can be in Swedish OR English
- Your metadata values MUST match the language of the stored documents
- Swedish documents have Swedish metadata (keywords: ["familjär hyperkolesterolemi"], clinical_domain: ["Kardiologi"])  
- English documents have English metadata (keywords: ["familial hypercholesterolemia"], clinical_domain: ["Cardiology"])
- Generate metadata values in the SAME language as the query to ensure proper matching

Available metadata fields:
- content_type: ["instruction", "narrative", "definition", "list", "procedure", "process", "technical", "clinical", "medical", "documentation"]
- hierarchical_tags: Swedish docs: ["Beslutsstöd", "Journalintegration", "Remissprocess"] | English docs: ["Decision support", "Journal integration", "Referral process"]
- clinical_domain: Swedish docs: ["Kardiologi", "Endokrinologi", "Lipidmetabolism", "Primärvården"] | English docs: ["Cardiology", "Endocrinology", "Lipid metabolism", "Primary care"]
- keywords: Generate in same language as query

Now analyze: "{query}"
Return ONLY valid JSON, no other text.
"""

    try:
        # Use your existing provider logic
        if config.MCP_PROVIDER_OLLAMA:
            llm = ChatOllama(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                base_url=config.OLLAMA_BASE_URL
            )
        elif config.MCP_PROVIDER_OPENAI:
            llm = ChatOpenAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        elif config.MCP_PROVIDER_MISTRAL:
            llm = ChatMistralAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                streaming=False,
                mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
                endpoint=config.MISTRAL_BASE_URL
            )
        else:
            if config.DEBUG:
                print("No LLM provider configured for metadata filtering")
            return None

        messages = [
            SystemMessage(
                content="You are a metadata filter generator. Generate metadata values in the same language as the query. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]

        if config.DEBUG:
            print(f"Generating metadata filter using provider for query: {query}")

        response = await llm.ainvoke(messages)
        response_content = response.content.strip()

        if config.DEBUG:
            print(f"LLM metadata response: {response_content}")

        metadata_config = json.loads(response_content)

        if config.DEBUG:
            print(f"Parsed metadata config: {metadata_config}")

        return build_qdrant_filters_from_config(metadata_config)

    except json.JSONDecodeError as e:
        if config.DEBUG:
            print(f"JSON parsing failed for metadata filter: {e}")
            print(f"Response was: {response_content}")
        return None
    except Exception as e:
        if config.DEBUG:
            print(f"LLM metadata filtering failed: {e}")
        return None


def build_qdrant_filters_from_config(metadata_config: dict) -> Optional[rest.Filter]:
    """Convert LLM-generated metadata config to Qdrant filters"""
    should_conditions = []

    for field_name, values in metadata_config.items():
        if not values:
            continue

        field_key = f"metadata.{field_name}"

        if isinstance(values, list) and values:
            should_conditions.append(
                rest.FieldCondition(
                    key=field_key,
                    match=rest.MatchAny(any=values)
                )
            )
        elif isinstance(values, str) and values.strip():
            should_conditions.append(
                rest.FieldCondition(
                    key=field_key,
                    match=rest.MatchText(text=values)
                )
            )

    if config.DEBUG and should_conditions:
        print(f"Built {len(should_conditions)} metadata filter conditions")

    return rest.Filter(should=should_conditions) if should_conditions else None


async def enhanced_multi_strategy_retrieval(
        query: str,
        collection_name: str,
        qdrant_client: AsyncQdrantClient,
        openai_embedder: Optional[OpenAIEmbeddings] = None,
        k: int = None,
        min_score: float = 0.6
) -> List:
    """Fully async multi-strategy retrieval with zero sync operations"""

    # Use config defaults if not provided
    k = k or config.QDRANT_RESULT_LIMIT

    if config.DEBUG:
        print(f"Starting async multi-strategy retrieval for: '{query}'")

    try:
        # Generate embedding with fallback
        query_embedding = await async_embed_with_fallback(
            query=query,
            ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
            ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
            openai_embedder=openai_embedder,
            timeout=10.0
        )

        if not query_embedding or len(query_embedding) == 0:
            if config.DEBUG:
                print("No embedding generated, aborting retrieval")
            return []

        if config.DEBUG:
            print(f"Using embedding with {len(query_embedding)} dimensions")

        # Define three async search strategies
        async def high_precision_search():
            try:
                # metadata_filter = intelligent_metadata_filter(query)
                metadata_filter = await llm_intelligent_metadata_filter(query)

                response = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    query_filter=metadata_filter,
                    limit=k,
                    score_threshold=0.7
                )
                return [(hit, "high_precision") for hit in response.points]

            except Exception as e:
                if config.DEBUG:
                    print(f"High precision search failed: {e}")
                return []

        async def medium_precision_search():
            try:
                response = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=k,
                    score_threshold=0.5
                )
                return [(hit, "medium_precision") for hit in response.points]
            except Exception as e:
                if config.DEBUG:
                    print(f"Medium precision search failed: {e}")
                return []

        async def broad_fallback_search():
            try:
                response = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=k,
                    score_threshold=0.3
                )
                return [(hit, "broad_fallback") for hit in response.points]
            except Exception as e:
                if config.DEBUG:
                    print(f"Broad fallback search failed: {e}")
                return []

        # Execute all strategies in parallel
        if config.DEBUG:
            print("Executing 3 search strategies in parallel")

        results = await asyncio.gather(
            high_precision_search(),
            medium_precision_search(),
            broad_fallback_search(),
            return_exceptions=True
        )

        # Process results with first success termination
        all_hits = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if config.DEBUG:
                    print(f"Strategy {i} failed with exception: {result}")
                continue

            if result:
                all_hits.extend(result)

        # Deduplicate and sort
        seen_ids = set()
        unique_hits = []
        for hit, strategy in all_hits:
            hit_id = getattr(hit, 'id', id(hit))
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                unique_hits.append(hit)

        # Sort by score and apply minimum score filter
        unique_hits.sort(key=lambda x: x.score, reverse=True)
        filtered_hits = [hit for hit in unique_hits if hit.score >= min_score]
        final_hits = filtered_hits[:k]

        if config.DEBUG:
            print(f"Retrieved {len(final_hits)} hits after dedup and filtering (min_score: {min_score})")

        return final_hits

    except Exception as e:
        if config.DEBUG:
            print(f"Multi-strategy retrieval error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        return []


async def cosmic_database_tool(query: str) -> str:
    """
    Enhanced async cosmic database search with multi-strategy retrieval.

    Args:
        query (str): The query to search for in the cosmic database.

    Returns:
        str: The content of relevant documents found, or a message indicating no information was found.
    """

    if config.DEBUG:
        print(f"Searching cosmic database for query: {query}")

    try:
        # Initialize async Qdrant client
        qdrant_client = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )

        # Initialize OpenAI embedder if using OpenAI
        openai_embedder = None
        if config.USE_OPENAI:
            openai_embedder = OpenAIEmbeddings(
                model=config.EMBEDDINGS_MODEL_NAME,
                openai_api_key=config.OPENAI_API_KEY
            )

        if config.DEBUG:
            print(f"Using collection: {config.COSMIC_DATABASE_COLLECTION_NAME}")
            print(f"Result limit: {config.QDRANT_RESULT_LIMIT}")
            print(f"Qdrant host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

        # Perform enhanced multi-strategy retrieval
        results = await enhanced_multi_strategy_retrieval(
            query=query,
            collection_name=config.COSMIC_DATABASE_COLLECTION_NAME,
            qdrant_client=qdrant_client,
            openai_embedder=openai_embedder,
            k=config.QDRANT_RESULT_LIMIT,
            min_score=0.6
        )

        if config.DEBUG:
            print(f"Found {len(results)} results in cosmic database")

        # Process results and build response
        if results:
            content_list = []
            sources = []

            for result in results:
                if hasattr(result, 'payload') and result.payload:
                    # The content is stored in 'text' field, not 'content'
                    content = result.payload.get("text", "").strip()

                    # Get metadata - some fields might be at root level of payload
                    source_file = result.payload.get("source_file", "unknown")
                    section = result.payload.get("section", "unknown")
                    page = result.payload.get("page", "unknown")
                    context_summary = result.payload.get("context_summary", "")

                    if content:
                        # Track sources
                        if source_file not in sources:
                            sources.append(source_file)

                        # Build enhanced content with metadata
                        score = result.score

                        enhanced_content = f"[Source: {source_file} | Section: {section} | Page: {page} | Score: {score:.3f}]\n{content}"

                        if context_summary:
                            enhanced_content += f"\n[Context: {context_summary}]"

                        content_list.append(enhanced_content)

            if config.DEBUG:
                print(f"Processing {len(content_list)} content chunks from {len(sources)} sources")

                for i, content in enumerate(content_list[:3]):  # Show first 3
                    print(f"Chunk {i + 1} preview: {content[:200]}...")

                print(f"Sources: {sources}")

            # Join all content with clear separators
            if content_list:
                final_content = "\n\n" + "=" * 50 + "\n\n".join(content_list)
                return final_content
            else:
                return "No content could be extracted from the search results."

    except Exception as e:
        error_msg = f"Error searching cosmic database: {e}"
        if config.DEBUG:
            print(f"ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        return f"An error occurred while searching the cosmic database: {str(e)}"

    finally:
        # Clean up async client
        try:
            if 'qdrant_client' in locals():
                await qdrant_client.close()
        except Exception as e:
            if config.DEBUG:
                print(f"Error closing Qdrant client: {e}")
