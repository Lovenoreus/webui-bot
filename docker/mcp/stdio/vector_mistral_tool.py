import sys
import openai
from qdrant_client import QdrantClient
from dotenv import load_dotenv
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

load_dotenv()

# Mistral Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"



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
                # Add queue filter
                should_conditions.append(
                    rest.FieldCondition(
                        key="queue",
                        match=rest.MatchValue(value=pattern["queue"])
                    )
                )

                # Add keyword filter for this issue type
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

            # Also search in descriptions
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


async def enhanced_multi_strategy_retrieval(
        query: str,
        collection_name: str,
        qdrant_client: AsyncQdrantClient,
        k: int = 5,
        min_score: float = 0.6
) -> List:
    """Fully async multi-strategy retrieval using Mistral embeddings"""

    if hasattr(config, 'DEBUG') and config.DEBUG:
        print(f"Starting multi-strategy retrieval for: '{query}'")

    try:
        # Generate embedding with Mistral
        query_embedding = await async_embed_with_mistral(query)

        if not query_embedding or len(query_embedding) == 0:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print("No embedding generated, aborting retrieval")
            return []

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Using Mistral embedding with {len(query_embedding)} dimensions")

        # Define three async search strategies
        async def high_precision_search():
            try:
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


async def hospital_support_questions_tool(query: str) -> str:
    """
    Search hospital support questions database for troubleshooting protocols.

    Args:
        query (str): The issue or problem to search for

    Returns:
        str: Support protocol with questions to ask users
    """

    if hasattr(config, 'DEBUG') and config.DEBUG:
        print(f"Searching hospital support questions for: {query}")

    try:
        # Initialize async Qdrant client
        qdrant_client = AsyncQdrantClient(
            host=getattr(config, 'QDRANT_HOST', 'localhost'),
            port=getattr(config, 'QDRANT_PORT', 6333)
        )

        # Perform enhanced multi-strategy retrieval
        results = await enhanced_multi_strategy_retrieval(
            query=query,
            collection_name="hospital_support_questions_mistral_embeddings",
            qdrant_client=qdrant_client,
            k=3,
            min_score=0.5
        )

        if not results:
            return "No matching support protocols found for your issue. Please provide more details about the problem you're experiencing."

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Found {len(results)} support protocols")

        # Format results for support staff
        formatted_results = []

        for result in results:
            if hasattr(result, 'payload') and result.payload:
                payload = result.payload
                score = result.score

                # Build the response format
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

                # Add relevant keywords for reference
                keywords = payload.get('keywords', [])
                if keywords:
                    response_part += f"\n\nRELATED KEYWORDS: {', '.join(keywords[:10])}"  # Limit to first 10

                formatted_results.append(response_part)

        # Combine all results
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
        # Clean up async client
        try:
            if 'qdrant_client' in locals():
                await qdrant_client.close()
        except Exception as e:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print(f"Error closing Qdrant client: {e}")


async def cosmic_database_tool(query: str) -> str:
    """
    Enhanced async cosmic database search with multi-strategy retrieval.
    Note: This maintains the original function but now uses Mistral embeddings.

    Args:
        query (str): The query to search for in the cosmic database.

    Returns:
        str: The content of relevant documents found, or a message indicating no information was found.
    """

    if hasattr(config, 'DEBUG') and config.DEBUG:
        print(f"Searching cosmic database for query: {query}")

    try:
        # Initialize async Qdrant client
        qdrant_client = AsyncQdrantClient(
            host=getattr(config, 'QDRANT_HOST', 'localhost'),
            port=getattr(config, 'QDRANT_PORT', 6333)
        )

        if hasattr(config, 'DEBUG') and config.DEBUG:
            cosmic_collection = getattr(config, 'COSMIC_DATABASE_COLLECTION_NAME',
                                        'cosmic_documents_clean-jeffh-intfloat-multilingual-e5-large-q8_0')
            result_limit = getattr(config, 'QDRANT_RESULT_LIMIT', 5)
            print(f"Using collection: {cosmic_collection}")
            print(f"Result limit: {result_limit}")

        # Perform enhanced multi-strategy retrieval
        results = await enhanced_multi_strategy_retrieval(
            query=query,
            collection_name=getattr(config, 'COSMIC_DATABASE_COLLECTION_NAME',
                                    'cosmic_documents_clean-jeffh-intfloat-multilingual-e5-large-q8_0'),
            qdrant_client=qdrant_client,
            k=getattr(config, 'QDRANT_RESULT_LIMIT', 5),
            min_score=0.6
        )

        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"Found {len(results)} results in cosmic database")

        # Process results and build response
        if results:
            content_list = []
            sources = []

            for result in results:
                if hasattr(result, 'payload') and result.payload:
                    # The content is stored in 'text' field
                    content = result.payload.get("text", "").strip()

                    # Get metadata
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

            if hasattr(config, 'DEBUG') and config.DEBUG:
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
        else:
            return "No relevant information found in the cosmic database."

    except Exception as e:
        error_msg = f"Error searching cosmic database: {e}"
        if hasattr(config, 'DEBUG') and config.DEBUG:
            print(f"ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        return f"An error occurred while searching the cosmic database: {str(e)}"

    finally:
        # Clean up async client
        try:
            if 'qdrant_client' in locals():
                await qdrant_client.close()
        except Exception as e:
            if hasattr(config, 'DEBUG') and config.DEBUG:
                print(f"Error closing Qdrant client: {e}")