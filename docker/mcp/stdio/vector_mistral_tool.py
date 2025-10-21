import sys
import openai
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
import os
from typing import List
import requests
import config
from typing import List, Generator, Optional
import httpx
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
import traceback
import json
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI

load_dotenv(find_dotenv())

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Mistral Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"


async def async_embed_with_openai(query: str, model_name: str = "text-embedding-3-large", timeout: float = 10.0) -> \
List[float]:
    """Generate embedding using OpenAI API"""
    if not OPENAI_API_KEY or not openai_client:
        print("❌ OPENAI_API_KEY not found in environment")
        return []

    try:
        response = openai_client.embeddings.create(
            model=model_name,
            input=query,
            timeout=timeout
        )
        embedding = response.data[0].embedding

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"OpenAI embedding success: {len(embedding)} dimensions")
        return embedding

    except Exception as e:
        print(f"❌ OpenAI embedding failed: {e}")
        return []


async def async_embed_with_mistral(query: str, timeout: float = 10.0) -> List[float]:
    """Generate embedding using Mistral API"""
    if not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not found in environment")
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    payload = {
        "model": "mistral-embed",
        "input": [query]
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(MISTRAL_EMBED_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]

            if hasattr(config, 'DEBUG') and config.DEBUG:
                print(f"Mistral embedding success: {len(embedding)} dimensions")
            return embedding

    except httpx.TimeoutException as e:
        print(f"❌ Mistral API timeout: {e}")
    except httpx.HTTPError as e:
        print(f"❌ Mistral API HTTP error: {e}")
    except Exception as e:
        print(f"❌ Mistral embedding failed: {e}")

    return []


async def get_embedding(query: str) -> List[float]:
    """Get embedding based on configured provider"""
    if hasattr(config, 'USE_OPENAI') and config.USE_OPENAI:
        model_name = getattr(config, 'EMBEDDINGS_MODEL_NAME', 'text-embedding-3-large')
        return await async_embed_with_openai(query, model_name=model_name)
    
    elif hasattr(config, 'USE_MISTRAL') and config.USE_MISTRAL:
        return await async_embed_with_mistral(query)
    
    elif hasattr(config, 'USE_OLLAMA') and config.USE_OLLAMA:
        # Ollama doesn't have a standard async embedding endpoint
        # You'd need to implement this based on your Ollama setup
        print("❌ Ollama embeddings not yet implemented")
        return []
    
    else:
        print("❌ No embedding provider configured")
        return []


def intelligent_metadata_filter(query: str) -> Optional[rest.Filter]:
    """Build filters based on query intent for hospital support questions"""
    try:
        query_lower = query.lower()
        should_conditions = []

        # Equipment and issue type patterns for hospital support
        issue_patterns = {
            "mri": {
                "indicators": ["mri", "scanner", "magnetic resonance", "imaging"],
                "queue": "Radiology and Maintenance Departments",
                "keywords": ["MRI scanner", "MRI", "scanner", "imaging", "radiology"]
            },
            "software": {
                "indicators": ["software", "application", "program", "install", "crash", "error"],
                "queue": "IT Department",
                "keywords": ["software", "installation", "crashes", "error messages"]
            },
            "facilities": {
                "indicators": ["leak", "faucet", "toilet", "light", "door", "fixture", "plumbing"],
                "queue": "Maintenance Department",
                "keywords": ["facilities", "fixtures", "faucets", "toilets", "lighting"]
            },
            "hvac": {
                "indicators": ["temperature", "hot", "cold", "air conditioning", "heating", "hvac"],
                "queue": "Maintenance Department",
                "keywords": ["HVAC", "temperature", "heating", "cooling", "climate"]
            },
            "safety": {
                "indicators": ["safety", "security", "emergency", "lock", "alarm", "hazard"],
                "queue": "Security or Maintenance Department",
                "keywords": ["safety", "security", "emergency", "locks"]
            },
            "supply": {
                "indicators": ["supply", "shortage", "missing", "stock", "ppe", "supplies"],
                "queue": "Logistics Department",
                "keywords": ["supply", "shortage", "logistics", "PPE", "medical supplies"]
            },
            "lab": {
                "indicators": ["lab", "laboratory", "analyzer", "centrifuge", "microscope", "blood"],
                "queue": "Biomedical Engineering and Laboratory",
                "keywords": ["laboratory", "lab equipment", "analyzers", "centrifuges"]
            },
            "patient_care": {
                "indicators": ["nurse call", "patient monitor", "call button", "bedside", "telemetry"],
                "queue": "Biomedical Engineering",
                "keywords": ["nurse call", "patient monitoring", "call buttons", "bedside"]
            }
        }

        # Check for specific issue types
        for issue_type, pattern in issue_patterns.items():
            if any(indicator in query_lower for indicator in pattern["indicators"]):
                should_conditions.append(
                    rest.FieldCondition(
                        key="queue",
                        match=rest.MatchValue(value=pattern["queue"])
                    )
                )
                should_conditions.append(
                    rest.FieldCondition(
                        key="keywords",
                        match=rest.MatchAny(any=pattern["keywords"])
                    )
                )
                break

        # Always include broad keyword matching
        query_words = [w for w in query_lower.split() if len(w) > 2]
        if query_words:
            should_conditions.append(
                rest.FieldCondition(
                    key="keywords",
                    match=rest.MatchAny(any=query_words)
                )
            )
            should_conditions.append(
                rest.FieldCondition(
                    key="description",
                    match=rest.MatchText(text=" ".join(query_words))
                )
            )

        return rest.Filter(should=should_conditions) if should_conditions else None

    except Exception as e:
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Metadata filter error: {e}")
        return None


async def llm_intelligent_metadata_filter_hospital(query: str) -> Optional[rest.Filter]:
    """Use LLM to generate intelligent metadata filters for hospital support queries"""

    prompt = f"""You are a metadata filter generator for a hospital support system that handles technical issues, equipment problems, and facility maintenance.

    Analyze this support query and determine the most relevant metadata filters:

    Query: "{query}"

    Available metadata fields:
    - queue: ["Technical Support", "Servicedesk", "2nd line", "Cambio JIRA", "Cosmic", "Billing Payments", "Account Management", "Product Inquiries", "Feature Requests", "Bug Reports", "Security Department", "Compliance Legal", "Service Outages", "Onboarding Setup", "API Integration", "Data Migration", "Accessibility", "Training Education", "General Inquiries", "Permissions Access", "Management Department", "Maintenance Department", "Logistics Department", "IT Department"]
    - issue_category: ["Hardware", "Software", "Facility", "Network", "Medical Equipment", "Other"]
    - urgency_level: ["High", "Medium", "Low"]
    - keywords: Generate relevant technical terms and equipment names based on the query

    Examples:
    - Query: "MRI scanner not working" → {{"queue": ["Technical Support"], "issue_category": ["Medical Equipment"], "keywords": ["MRI", "scanner", "imaging", "radiology"], "urgency_level": ["High"]}}
    - Query: "software keeps crashing" → {{"queue": ["IT Department", "Technical Support"], "issue_category": ["Software"], "keywords": ["software", "crashes", "error", "application"], "urgency_level": ["Medium", "High"]}}
    - Query: "toilet is leaking" → {{"queue": ["Maintenance Department"], "issue_category": ["Facility"], "keywords": ["leak", "toilet", "plumbing", "facilities"], "urgency_level": ["Medium"]}}

    Now analyze: "{query}"

    Return ONLY valid JSON with the metadata filters, no other text. Include only fields that are relevant to the query.
    """

    try:
        # Use your existing provider logic
        if hasattr(config, 'MCP_PROVIDER_OLLAMA') and config.MCP_PROVIDER_OLLAMA:
            llm = ChatOllama(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                base_url=config.OLLAMA_BASE_URL
            )
        elif hasattr(config, 'MCP_PROVIDER_OPENAI') and config.MCP_PROVIDER_OPENAI:
            llm = ChatOpenAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        elif hasattr(config, 'MCP_PROVIDER_MISTRAL') and config.MCP_PROVIDER_MISTRAL:
            llm = ChatMistralAI(
                model=config.MCP_AGENT_MODEL_NAME,
                temperature=0.1,
                streaming=False,
                mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
                endpoint=getattr(config, 'MISTRAL_BASE_URL', 'https://api.mistral.ai')
            )
        else:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print("No LLM provider configured for metadata filtering, falling back to rule-based")
            return intelligent_metadata_filter(query)

        messages = [
            SystemMessage(
                content="You are a metadata filter generator for hospital support. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Generating hospital support metadata filter for query: {query}")

        response = await llm.ainvoke(messages)
        response_content = response.content.strip()

        # Clean up response if it has markdown code blocks
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()
        elif response_content.startswith("```"):
            response_content = response_content.replace("```", "").strip()

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"LLM metadata response: {response_content}")

        metadata_config = json.loads(response_content)

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Parsed metadata config: {metadata_config}")

        return build_qdrant_filters_from_config_hospital(metadata_config)

    except json.JSONDecodeError as e:
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"JSON parsing failed for metadata filter: {e}")
            print(f"Response was: {response_content}")
        # Fallback to rule-based
        return intelligent_metadata_filter(query)
    except Exception as e:
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"LLM metadata filtering failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        # Fallback to rule-based
        return intelligent_metadata_filter(query)


def build_qdrant_filters_from_config_hospital(metadata_config: dict) -> Optional[rest.Filter]:
    """Convert LLM-generated metadata config to Qdrant filters for hospital support"""
    should_conditions = []

    for field_name, values in metadata_config.items():
        if not values:
            continue

        field_key = field_name if field_name in ['queue', 'issue_category', 'urgency_level'] else field_name

        if isinstance(values, list) and values:
            if field_name == 'queue' and len(values) == 1:
                # Use exact match for single queue value
                should_conditions.append(
                    rest.FieldCondition(
                        key=field_key,
                        match=rest.MatchValue(value=values[0])
                    )
                )
            else:
                # Use MatchAny for multiple values or keywords
                should_conditions.append(
                    rest.FieldCondition(
                        key=field_key,
                        match=rest.MatchAny(any=values)
                    )
                )
        elif isinstance(values, str) and values.strip():
            if field_name == 'description':
                should_conditions.append(
                    rest.FieldCondition(
                        key=field_key,
                        match=rest.MatchText(text=values)
                    )
                )
            else:
                should_conditions.append(
                    rest.FieldCondition(
                        key=field_key,
                        match=rest.MatchValue(value=values)
                    )
                )

    if hasattr(config, 'DEBUG') and config.DEBUG and should_conditions:
        print(f"Built {len(should_conditions)} metadata filter conditions for hospital support")

    return rest.Filter(should=should_conditions) if should_conditions else None


async def enhanced_multi_strategy_retrieval(
        query: str,
        collection_name: str,
        qdrant_client: AsyncQdrantClient,
        k: int = 5,
        min_score: float = 0.6,
        use_llm_filter: bool = True
) -> List:
    """Fully async multi-strategy retrieval using configured embedding provider with optional LLM filtering"""

    if hasattr(config, 'DEBUG') and config.DEBUG:
        print(f"Starting multi-strategy retrieval for: '{query}'")

    try:
        # Generate embedding with configured provider
        query_embedding = await get_embedding(query)

        if not query_embedding or len(query_embedding) == 0:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print("No embedding generated, aborting retrieval")
            return []

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Using embedding with {len(query_embedding)} dimensions")

        # Define three async search strategies
        async def high_precision_search():
            try:
                # Use LLM filter or fallback to rule-based
                if use_llm_filter:
                    metadata_filter = await llm_intelligent_metadata_filter_hospital(query)
                    
                else:
                    metadata_filter = intelligent_metadata_filter(query)

                response = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    query_filter=metadata_filter,
                    limit=k,
                    score_threshold=0.7
                )
                return [(hit, "high_precision") for hit in response.points]
            except Exception as e:
                if hasattr(config, 'DEBUG') and config.DEBUG:
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
                if hasattr(config, 'DEBUG') and config.DEBUG:
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
                if hasattr(config, 'DEBUG') and config.DEBUG:
                    print(f"Broad fallback search failed: {e}")
                return []

        # Execute all strategies in parallel
        if hasattr(config, 'DEBUG') and config.DEBUG:
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
                if hasattr(config, 'DEBUG') and config.DEBUG:
                    print(f"Strategy {i} failed with exception: {result}")
                continue

            if result:
                # Check for early success - if we get enough high-quality results
                high_quality = [hit for hit, _ in result if hit.score > 0.8]
                if len(high_quality) >= k // 2:
                    if hasattr(config, 'DEBUG') and config.DEBUG:
                        print(f"Early success from strategy {i} with {len(high_quality)} high-quality hits")
                    all_hits = result
                    break
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

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Retrieved {len(final_hits)} hits after dedup and filtering (min_score: {min_score})")

        return final_hits

    except Exception as e:
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Multi-strategy retrieval error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        return []


async def hospital_support_questions_tool(query: str, use_llm_filter: bool = True) -> str:
    """
    Search hospital support questions database for troubleshooting protocols.

    Args:
        query (str): The issue or problem to search for
        use_llm_filter (bool): Whether to use LLM for metadata filtering (default: True)

    Returns:
        str: Support protocol with questions to ask users
    """

    if hasattr(config, 'DEBUG') and config.DEBUG:
        print(f"Searching hospital support questions for: {query}")
        print(f"Using LLM filter: {use_llm_filter}")

    # Determine collection name based on embedding provider
    if hasattr(config, 'USE_OPENAI') and config.USE_OPENAI:
        collection_name = "hospital_support_questions_openai_embeddings"

    elif hasattr(config, 'USE_MISTRAL') and config.USE_MISTRAL:
        collection_name = "hospital_support_questions_mistral_embeddings"

    elif hasattr(config, 'USE_OLLAMA') and config.USE_OLLAMA:
        collection_name = "hospital_support_questions_ollama_embeddings"

    else:
        collection_name = "hospital_support_questions_mistral_embeddings"  # default fallback

    print(f'Using Hospital Collection: {collection_name}')
    try:
        qdrant_client = AsyncQdrantClient(
            host=getattr(config, 'QDRANT_HOST', 'localhost'),
            port=getattr(config, 'QDRANT_PORT', 6333)
        )

        results = await enhanced_multi_strategy_retrieval(
            query=query,
            collection_name=collection_name,
            qdrant_client=qdrant_client,
            k=3,
            min_score=0.5,
            use_llm_filter=use_llm_filter
        )

        if not results:
            return "No matching support protocols found for your issue. Please provide more details about the problem you're experiencing."

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Found {len(results)} support protocols")

        formatted_results = []

        for result in results:
            if hasattr(result, 'payload') and result.payload:
                payload = result.payload
                score = result.score

                response_part = f"""
ISSUE CATEGORY: {payload.get('issue_category', 'Unknown Issue')}
DEPARTMENT: {payload.get('queue', 'Unknown Department')}
URGENCY LEVEL: {payload.get('urgency_level', 'Unknown').upper()}
MATCH SCORE: {score:.3f}

DESCRIPTION: 
{payload.get('description', 'No description available')}

QUESTIONS TO ASK THE USER:"""

                questions = payload.get('questions_to_ask', [])
                if questions:
                    for i, question in enumerate(questions, 1):
                        response_part += f"\n{i}. {question}"
                else:
                    response_part += "\nNo specific questions available for this issue type."

                keywords = payload.get('keywords', [])
                if keywords:
                    response_part += f"\n\nRELATED KEYWORDS: {', '.join(keywords[:10])}"

                formatted_results.append(response_part)

        final_response = "HOSPITAL SUPPORT PROTOCOLS FOUND:\n"
        final_response += "\n" + "=" * 80 + "\n".join(formatted_results)

        return final_response

    except Exception as e:
        error_msg = f"Error searching hospital support questions: {str(e)}"
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        return error_msg

    finally:
        try:
            if 'qdrant_client' in locals():
                await qdrant_client.close()

        except Exception as e:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print(f"Error closing Qdrant client: {e}")


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

        if config.DEBUG:
            print(f"Using collection: {config.COSMIC_DATABASE_COLLECTION_NAME}")
            print(f"Result limit: {config.QDRANT_RESULT_LIMIT}")
            print(f"Qdrant host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

        # Perform enhanced multi-strategy retrieval
        results = await enhanced_multi_strategy_retrieval(
            query=query,
            collection_name=config.COSMIC_DATABASE_COLLECTION_NAME,
            qdrant_client=qdrant_client,
            k=config.QDRANT_RESULT_LIMIT,
            min_score=0.6,
            use_llm_filter=False  # Cosmic database doesn't use LLM filters
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
