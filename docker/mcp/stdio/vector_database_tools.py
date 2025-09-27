# import sys
# import openai
# from qdrant_client import QdrantClient
# from dotenv import load_dotenv
# import os
# from typing import List
# import requests
# import config
# from typing import List, Generator, Tuple


# # load_dotenv()

# def get_known_problems_tool(query: str) -> str:
#     """Called when the you have to find known problems related to the query."""

#     print(f"Retrieving known problems for query: {query}")
#     # results = kp_retriever.get_relevant_documents(query, k=2)
#     # print(results)
#     # results = [result.page_content for result in results]
#     # return results if results else "No known problems found."
#     return 'Got known problems'


# def get_who_am_i_tool(query: str) -> str:
#     """Called when the user wants to know information about the bot's identity and capabilities.

#     Parameters:
#     query (str): The query to search for in the 'Who Am I' database.
#     Returns:
#     str: The content of the relevant document found, or a message indicating no information was found.

#     """

#     print(f"Retrieving 'Who Am I' information for query: {query}")
#     # print(query)
#     # results = wmi_retriever.get_relevant_documents(query, k=2)
#     # results = [result.page_content for result in results]
#     # return results if results else "No known problems found."
#     return 'Got Who Am I'



# def check_conversation_completion(numIssueReported: int, numTicketsCreated: int) -> bool:
#     """
#     When to Call: Call this tool when ticket is successfully created or when the user explicitly states that the issue is resolved.
#     Check if the conversation is complete based on reported issues and created tickets.
#     Args:
#         numIssueReported (int): Number of issues reported by the user.
#         numTicketsCreated (int): Number of tickets created by the agent.
#     Returns:
#         bool: True if the conversation is complete, False otherwise."""
#     print(f"Checking conversation completion: {numIssueReported} issues reported, {numTicketsCreated} tickets created.")
#     if numIssueReported == numTicketsCreated:
#         return True
#     else:
#         return False


# def embed_query_using_ollama_embedding_model(query: str, model_name: str, ollama_url: str):
#     """Generate embedding using Nomic model via Ollama"""
#     payload = {
#         "model": model_name,
#         "prompt": query
#     }

#     try:
#         # ollama_url = "http://vs2153.vll.se:11434"
#         # ollama_url = "http://localhost:11434"
#         response = requests.post(
#             f"{ollama_url}/api/embeddings",
#             json=payload,
#             timeout=30
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result["embedding"]

#     except requests.exceptions.RequestException as e:
#         print(f"❌ Ollama request failed: {e}")
#         raise
#     except KeyError as e:
#         print(f"❌ Unexpected response format: {e}")
#         print(f"Response: {response.text}")
#         raise
#     except Exception as e:
#         print(f"❌ Nomic embedding failed: {e}")
#         raise


# import httpx
# import asyncio
# from typing import List, Optional, Tuple
# from qdrant_client import AsyncQdrantClient
# from qdrant_client.http import models as rest
# from langchain_openai import OpenAIEmbeddings
# import traceback
# from dotenv import load_dotenv

# load_dotenv()


# async def async_embed_with_fallback(
#         query: str,
#         ollama_model: str = None,
#         ollama_base_url: str = None,
#         openai_embedder: Optional[OpenAIEmbeddings] = None,
#         timeout: float = 10.0
# ) -> List[float]:
#     """Async embedding with Ollama primary and OpenAI fallback"""

#     # Try Ollama first if configured
#     if config.USE_OLLAMA and ollama_base_url and ollama_model:
#         try:
#             async with httpx.AsyncClient(timeout=timeout) as client:
#                 response = await client.post(
#                     f"{ollama_base_url}/api/embeddings",
#                     json={"model": ollama_model, "prompt": query}
#                 )

#                 if response.status_code == 200:
#                     result = response.json()
#                     embedding = result.get('embedding')
#                     if embedding and len(embedding) > 0:
#                         if config.DEBUG:
#                             print(f"Ollama embedding success: {len(embedding)} dimensions")
#                         return embedding

#         except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
#             if config.DEBUG:
#                 print(f"Ollama embedding failed: {type(e).__name__} - {e}")
#         except Exception as e:
#             if config.DEBUG:
#                 print(f"Ollama unexpected error: {e}")

#     # Fallback to OpenAI
#     if config.USE_OPENAI and openai_embedder:
#         try:
#             if config.DEBUG:
#                 print("Falling back to OpenAI embedding")
#             embedding = await asyncio.to_thread(openai_embedder.embed_query, query)
#             if embedding and len(embedding) > 0:
#                 if config.DEBUG:
#                     print(f"OpenAI embedding success: {len(embedding)} dimensions")
#                 return embedding
#         except Exception as e:
#             if config.DEBUG:
#                 print(f"OpenAI embedding failed: {e}")

#     return []


# def intelligent_metadata_filter(query: str, strictness: str = "medium") -> dict:
#     """
#     Generate intelligent metadata filters based on query analysis with configurable strictness
    
#     Args:
#         query: The search query string
#         strictness: Filter strictness level - "high", "medium", or "low"
    
#     Returns:
#         Qdrant filter dictionary
#     """
#     import re
#     from datetime import datetime, timedelta
#     from typing import List, Dict, Any
    
#     # Initialize base filter
#     filters = {"must": [], "should": [], "must_not": []}
    
#     # Extract various metadata hints from query
#     query_lower = query.lower()
    
#     # Document type detection
#     doc_types = {
#         "pdf": ["pdf", "document", "paper", "report"],
#         "email": ["email", "message", "correspondence", "mail"],
#         "code": ["code", "function", "class", "script", "programming"],
#         "article": ["article", "blog", "post", "news"],
#         "manual": ["manual", "guide", "documentation", "docs", "readme"],
#         "presentation": ["presentation", "slides", "ppt", "slideshow"]
#     }
    
#     # Time-based keywords
#     time_keywords = {
#         "recent": timedelta(days=7),
#         "last week": timedelta(days=7),
#         "last month": timedelta(days=30),
#         "last year": timedelta(days=365),
#         "today": timedelta(days=1),
#         "yesterday": timedelta(days=2),
#         "this year": timedelta(days=365)
#     }
    
#     # Category/topic detection
#     categories = {
#         "technical": ["api", "database", "server", "config", "setup", "installation"],
#         "business": ["revenue", "sales", "profit", "strategy", "meeting", "client"],
#         "legal": ["contract", "agreement", "terms", "policy", "compliance"],
#         "hr": ["employee", "hiring", "performance", "training", "onboarding"],
#         "marketing": ["campaign", "brand", "social", "advertisement", "promotion"]
#     }
    
#     # Priority/importance indicators
#     priority_keywords = {
#         "high": ["urgent", "critical", "important", "asap", "priority", "emergency"],
#         "medium": ["soon", "needed", "required", "necessary"],
#         "low": ["later", "eventually", "someday", "optional"]
#     }
    
#     def add_filter_condition(condition: Dict[str, Any], filter_type: str = "must"):
#         """Helper to add conditions based on strictness"""
#         if strictness == "high" and filter_type in ["must", "must_not"]:
#             filters[filter_type].append(condition)
#         elif strictness == "medium":
#             filters[filter_type].append(condition)
#         elif strictness == "low" and filter_type != "must_not":
#             # In low strictness, convert some "must" to "should" for flexibility
#             target_type = "should" if filter_type == "must" else filter_type
#             filters[target_type].append(condition)
    
#     # Document type filtering
#     detected_types = []
#     for doc_type, keywords in doc_types.items():
#         if any(keyword in query_lower for keyword in keywords):
#             detected_types.append(doc_type)
    
#     if detected_types:
#         if strictness == "high":
#             # Must match exactly one of the detected types
#             add_filter_condition({
#                 "key": "document_type",
#                 "match": {"any": detected_types}
#             }, "must")
#         elif strictness == "medium":
#             # Should prefer detected types but allow others
#             add_filter_condition({
#                 "key": "document_type", 
#                 "match": {"any": detected_types}
#             }, "should")
#         # Low strictness: no document type restriction
    
#     # Time-based filtering
#     current_time = datetime.now()
#     time_filter_applied = False
    
#     for time_phrase, delta in time_keywords.items():
#         if time_phrase in query_lower:
#             cutoff_time = current_time - delta
#             time_condition = {
#                 "key": "timestamp",
#                 "range": {"gte": cutoff_time.isoformat()}
#             }
            
#             if strictness == "high":
#                 add_filter_condition(time_condition, "must")
#             else:
#                 add_filter_condition(time_condition, "should")
            
#             time_filter_applied = True
#             break
    
#     # Category/topic filtering
#     detected_categories = []
#     for category, keywords in categories.items():
#         if any(keyword in query_lower for keyword in keywords):
#             detected_categories.append(category)
    
#     if detected_categories:
#         category_condition = {
#             "key": "category",
#             "match": {"any": detected_categories}
#         }
        
#         if strictness == "high":
#             add_filter_condition(category_condition, "must")
#         else:
#             add_filter_condition(category_condition, "should")
    
#     # Priority filtering
#     detected_priority = None
#     for priority, keywords in priority_keywords.items():
#         if any(keyword in query_lower for keyword in keywords):
#             detected_priority = priority
#             break
    
#     if detected_priority:
#         priority_condition = {
#             "key": "priority",
#             "match": {"value": detected_priority}
#         }
        
#         if strictness == "high" and detected_priority == "high":
#             add_filter_condition(priority_condition, "must")
#         else:
#             add_filter_condition(priority_condition, "should")
    
#     # Language detection (basic)
#     if any(word in query_lower for word in ["python", "javascript", "java", "sql"]):
#         lang_condition = {
#             "key": "programming_language",
#             "match": {"any": ["python", "javascript", "java", "sql"]}
#         }
#         add_filter_condition(lang_condition, "should")
    
#     # Author/source filtering (if mentioned)
#     author_match = re.search(r'by (\w+)|from (\w+)|author:(\w+)', query_lower)
#     if author_match:
#         author = author_match.group(1) or author_match.group(2) or author_match.group(3)
#         author_condition = {
#             "key": "author",
#             "match": {"value": author}
#         }
#         add_filter_condition(author_condition, "must")
    
#     # File extension filtering
#     ext_match = re.search(r'\.(\w+)', query)
#     if ext_match:
#         extension = ext_match.group(1).lower()
#         ext_condition = {
#             "key": "file_extension",
#             "match": {"value": extension}
#         }
#         add_filter_condition(ext_condition, "must")
    
#     # Strictness-specific adjustments
#     if strictness == "high":
#         # Add more restrictive conditions
#         if not time_filter_applied:
#             # Prefer recent documents in high strictness
#             recent_condition = {
#                 "key": "timestamp",
#                 "range": {"gte": (current_time - timedelta(days=90)).isoformat()}
#             }
#             add_filter_condition(recent_condition, "should")
        
#         # Exclude low-quality content
#         filters["must_not"].append({
#             "key": "quality_score",
#             "range": {"lt": 0.3}
#         })
    
#     elif strictness == "low":
#         # Be more permissive - convert some must to should
#         if len(filters["must"]) > 2:
#             # Move some must conditions to should for flexibility
#             while len(filters["must"]) > 2:
#                 condition = filters["must"].pop()
#                 filters["should"].append(condition)
    
#     # Clean up empty filter sections
#     final_filter = {}
#     for key, value in filters.items():
#         if value:
#             final_filter[key] = value
    
#     # Return empty dict if no filters (let vector similarity dominate)
#     if not final_filter:
#         return {}
    
#     # Ensure we have at least a basic structure
#     if not any(final_filter.values()):
#         return {}
    
#     return final_filter



# collection_list = ["cosmic_documents_group1-text-embedding-3-large", "cosmic_documents_group2-text-embedding-3-large"]


# async def collection_worker(
#         query: str,
#         collection_name: str,
#         qdrant_client: AsyncQdrantClient,
#         openai_embedder: Optional[OpenAIEmbeddings] = None,
#         k: int = None,
#         min_score: float = 0.6
# ) -> Tuple[str, List]:
#     """
#     Worker function to search a single collection with multi-strategy retrieval.
    
#     Args:
#         query: The search query
#         collection_name: Name of the collection to search
#         qdrant_client: Async Qdrant client
#         openai_embedder: OpenAI embedder instance
#         k: Number of results to return
#         min_score: Minimum score threshold
        
#     Returns:
#         Tuple of (collection_name, results_list)
#     """
    
#     if config.DEBUG:
#         print(f"Worker starting search in collection: {collection_name}")
    
#     try:
#         # Use the existing enhanced_multi_strategy_retrieval function
#         results = await enhanced_multi_strategy_retrieval(
#             query=query,
#             collection_name=collection_name,
#             qdrant_client=qdrant_client,
#             openai_embedder=openai_embedder,
#             k=k,
#             min_score=min_score
#         )
        
#         if config.DEBUG:
#             print(f"Worker found {len(results)} results in collection: {collection_name}")
        
#         # Only return results if we found something
#         if results:
#             return (collection_name, results)
#         else:
#             return (collection_name, [])
            
#     except Exception as e:
#         if config.DEBUG:
#             print(f"Worker error in collection {collection_name}: {e}")
#         return (collection_name, [])


# async def parallel_multi_collection_retrieval(
#         query: str,
#         collection_names: List[str],
#         qdrant_client: AsyncQdrantClient,
#         openai_embedder: Optional[OpenAIEmbeddings] = None,
#         k: int = None,
#         min_score: float = 0.6
# ) -> List:
#     """
#     Search multiple collections in parallel using workers.
    
#     Args:
#         query: The search query
#         collection_names: List of collection names to search
#         qdrant_client: Async Qdrant client
#         openai_embedder: OpenAI embedder instance
#         k: Number of results per collection
#         min_score: Minimum score threshold
        
#     Returns:
#         Combined and deduplicated results from all collections
#     """
    
#     if config.DEBUG:
#         print(f"Starting parallel search across {len(collection_names)} collections")
    
#     try:
#         # Create worker tasks for all collections
#         worker_tasks = [
#             collection_worker(
#                 query=query,
#                 collection_name=collection_name,
#                 qdrant_client=qdrant_client,
#                 openai_embedder=openai_embedder,
#                 k=k,
#                 min_score=min_score
#             )
#             for collection_name in collection_names
#         ]
        
#         if config.DEBUG:
#             print(f"Created {len(worker_tasks)} worker tasks")
        
#         # Execute all workers in parallel
#         worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
#         # Process results from all workers
#         all_hits = []
#         successful_collections = []
        
#         for result in worker_results:
#             if isinstance(result, Exception):
#                 if config.DEBUG:
#                     print(f"Worker failed with exception: {result}")
#                 continue
                
#             collection_name, hits = result
            
#             if hits:  # Only include collections that found results
#                 successful_collections.append(collection_name)
#                 # Add collection info to each hit for tracking
#                 for hit in hits:
#                     if hasattr(hit, 'payload'):
#                         hit.payload = hit.payload or {}
#                         hit.payload['_source_collection'] = collection_name
#                     all_hits.append(hit)
        
#         if config.DEBUG:
#             print(f"Successfully retrieved results from {len(successful_collections)} collections: {successful_collections}")
#             print(f"Total hits before deduplication: {len(all_hits)}")
        
#         # Deduplicate results across collections
#         seen_content = set()
#         unique_hits = []
        
#         # Sort by score first to prioritize higher scoring results
#         all_hits.sort(key=lambda x: x.score, reverse=True)
        
#         for hit in all_hits:
#             if hasattr(hit, 'payload') and hit.payload:
#                 # Use text content for deduplication
#                 content = hit.payload.get("text", "").strip()
                
#                 # Create a simple hash for deduplication (first 200 chars)
#                 content_hash = hash(content[:200]) if content else hash(str(hit.id))
                
#                 if content_hash not in seen_content and content:
#                     seen_content.add(content_hash)
#                     unique_hits.append(hit)
        
#         # Final filtering and limiting
#         final_hits = unique_hits[:k] if k else unique_hits
        
#         if config.DEBUG:
#             collection_counts = {}
#             for hit in final_hits:
#                 collection = getattr(hit, 'payload', {}).get('_source_collection', 'unknown')
#                 collection_counts[collection] = collection_counts.get(collection, 0) + 1
            
#             print(f"Final results: {len(final_hits)} unique hits")
#             print(f"Collection distribution: {collection_counts}")
        
#         return final_hits
        
#     except Exception as e:
#         if config.DEBUG:
#             print(f"Parallel multi-collection retrieval error: {e}")
#             print(f"Traceback: {traceback.format_exc()}")
#         return []






# async def enhanced_multi_strategy_retrieval(
#         query: str,
#         collection_name: str,
#         qdrant_client: AsyncQdrantClient,
#         openai_embedder: Optional[OpenAIEmbeddings] = None,
#         k: int = None,
#         min_score: float = 0.6
# ) -> List:
#     """Fully async multi-strategy retrieval with intelligent metadata filtering across all strategies"""

#     # Use config defaults if not provided
#     k = k or config.QDRANT_RESULT_LIMIT

#     if config.DEBUG:
#         print(f"Starting async multi-strategy retrieval for: '{query}'")

#     try:
#         # Generate embedding with fallback
#         query_embedding = await async_embed_with_fallback(
#             query=query,
#             ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
#             ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
#             openai_embedder=openai_embedder,
#             timeout=10.0
#         )

#         if not query_embedding or len(query_embedding) == 0:
#             if config.DEBUG:
#                 print("No embedding generated, aborting retrieval")
#             return []

#         if config.DEBUG:
#             print(f"Using embedding with {len(query_embedding)} dimensions")

#         # Generate metadata filters with different strictness levels
#         strict_metadata_filter = intelligent_metadata_filter(query, strictness="high")
#         moderate_metadata_filter = intelligent_metadata_filter(query, strictness="medium") 
#         loose_metadata_filter = intelligent_metadata_filter(query, strictness="low")

#         if config.DEBUG:
#             print("Generated metadata filters for all strategies")

#         # Define three async search strategies with progressive metadata filtering
#         async def high_precision_search():
#             try:
#                 response = await qdrant_client.query_points(
#                     collection_name=collection_name,
#                     query=query_embedding,
#                     query_filter=strict_metadata_filter,
#                     limit=k,
#                     score_threshold=0.7
#                 )
#                 return [(hit, "high_precision") for hit in response.points]
#             except Exception as e:
#                 if config.DEBUG:
#                     print(f"High precision search failed: {e}")
#                 return []

#         async def medium_precision_search():
#             try:
#                 response = await qdrant_client.query_points(
#                     collection_name=collection_name,
#                     query=query_embedding,
#                     query_filter=moderate_metadata_filter,
#                     limit=k,
#                     score_threshold=0.5
#                 )
#                 return [(hit, "medium_precision") for hit in response.points]
#             except Exception as e:
#                 if config.DEBUG:
#                     print(f"Medium precision search failed: {e}")
#                 return []

#         async def broad_fallback_search():
#             try:
#                 response = await qdrant_client.query_points(
#                     collection_name=collection_name,
#                     query=query_embedding,
#                     query_filter=loose_metadata_filter,
#                     limit=k,
#                     score_threshold=0.3
#                 )
#                 return [(hit, "broad_fallback") for hit in response.points]
#             except Exception as e:
#                 if config.DEBUG:
#                     print(f"Broad fallback search failed: {e}")
#                 return []

#         # Execute all strategies in parallel
#         if config.DEBUG:
#             print("Executing 3 search strategies in parallel with metadata filtering")

#         results = await asyncio.gather(
#             high_precision_search(),
#             medium_precision_search(),
#             broad_fallback_search(),
#             return_exceptions=True
#         )

#         # Process results with enhanced early success detection
#         all_hits = []
#         strategy_names = ["high_precision", "medium_precision", "broad_fallback"]
        
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 if config.DEBUG:
#                     print(f"Strategy {strategy_names[i]} failed with exception: {result}")
#                 continue

#             if result:
#                 # Enhanced early success logic - prioritize metadata-filtered results
#                 high_quality = [hit for hit, _ in result if hit.score > 0.8]
#                 metadata_matched = [hit for hit, _ in result if hit.score > 0.6]  # Lower threshold for metadata matches
                
#                 # Early termination if we have sufficient high-quality, metadata-filtered results
#                 if len(high_quality) >= k // 2 and i == 0:  # High precision strategy
#                     if config.DEBUG:
#                         print(f"Early success from {strategy_names[i]} with {len(high_quality)} high-quality hits")
#                     all_hits = result
#                     break
#                 elif len(metadata_matched) >= k and i <= 1:  # High or medium precision
#                     if config.DEBUG:
#                         print(f"Early success from {strategy_names[i]} with {len(metadata_matched)} metadata-matched hits")
#                     all_hits = result
#                     break
                
#                 all_hits.extend(result)

#         # Deduplicate with strategy preference (higher precision strategies preferred)
#         seen_ids = set()
#         unique_hits = []
#         strategy_priority = {"high_precision": 3, "medium_precision": 2, "broad_fallback": 1}
        
#         # Sort by strategy priority first, then by score
#         all_hits.sort(key=lambda x: (strategy_priority.get(x[1], 0), x[0].score), reverse=True)
        
#         for hit, strategy in all_hits:
#             hit_id = getattr(hit, 'id', id(hit))
#             if hit_id not in seen_ids:
#                 seen_ids.add(hit_id)
#                 # Add strategy info to hit for debugging
#                 if hasattr(hit, 'payload'):
#                     hit.payload = hit.payload or {}
#                     hit.payload['_retrieval_strategy'] = strategy
#                 unique_hits.append(hit)

#         # Sort by score and apply minimum score filter
#         unique_hits.sort(key=lambda x: x.score, reverse=True)
#         filtered_hits = [hit for hit in unique_hits if hit.score >= min_score]
#         final_hits = filtered_hits[:k]

#         if config.DEBUG:
#             strategy_counts = {}
#             for hit in final_hits:
#                 strategy = getattr(hit, 'payload', {}).get('_retrieval_strategy', 'unknown')
#                 strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
#             print(f"Retrieved {len(final_hits)} hits after dedup and filtering (min_score: {min_score})")
#             print(f"Strategy breakdown: {strategy_counts}")

#         return final_hits

#     except Exception as e:
#         if config.DEBUG:
#             print(f"Multi-strategy retrieval error: {e}")
#             print(f"Traceback: {traceback.format_exc()}")
#         return []


# async def cosmic_database_tool(query: str) -> str:
#     """
#     Enhanced async cosmic database search with parallel multi-collection retrieval.

#     Args:
#         query (str): The query to search for in the cosmic database.

#     Returns:
#         str: The content of relevant documents found, or a message indicating no information was found.
#     """

#     if config.DEBUG:
#         print(f"Searching cosmic database across multiple collections for query: {query}")

#     try:
#         # Initialize async Qdrant client
#         qdrant_client = AsyncQdrantClient(
#             host=config.QDRANT_HOST,
#             port=config.QDRANT_PORT
#         )

#         # Initialize OpenAI embedder if using OpenAI
#         openai_embedder = None
#         if config.USE_OPENAI:
#             openai_embedder = OpenAIEmbeddings(
#                 model=config.EMBEDDINGS_MODEL_NAME,
#                 openai_api_key=config.OPENAI_API_KEY
#             )

#         if config.DEBUG:
#             print(f"Using collections: {collection_list}")
#             print(f"Result limit per collection: {config.QDRANT_RESULT_LIMIT}")
#             print(f"Qdrant host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

#         # Perform parallel multi-collection retrieval
#         results = await parallel_multi_collection_retrieval(
#             query=query,
#             collection_names=collection_list,
#             qdrant_client=qdrant_client,
#             openai_embedder=openai_embedder,
#             k=config.QDRANT_RESULT_LIMIT,
#             min_score=0.6
#         )

#         if config.DEBUG:
#             print(f"Found {len(results)} total results across all collections")

#         # Process results and build response
#         if results:
#             content_list = []
#             sources = []

#             for result in results:
#                 if hasattr(result, 'payload') and result.payload:
#                     # The content is stored in 'text' field, not 'content'
#                     content = result.payload.get("text", "").strip()

#                     # Get metadata - some fields might be at root level of payload
#                     source_file = result.payload.get("source_file", "unknown")
#                     section = result.payload.get("section", "unknown")
#                     page = result.payload.get("page", "unknown")
#                     context_summary = result.payload.get("context_summary", "")
#                     source_collection = result.payload.get("_source_collection", "unknown")

#                     if content:
#                         # Track sources
#                         if source_file not in sources:
#                             sources.append(source_file)

#                         # Build enhanced content with metadata including collection info
#                         score = result.score

#                         enhanced_content = f"[Collection: {source_collection} | Source: {source_file} | Section: {section} | Page: {page} | Score: {score:.3f}]\n{content}"

#                         if context_summary:
#                             enhanced_content += f"\n[Context: {context_summary}]"

#                         content_list.append(enhanced_content)

#             if config.DEBUG:
#                 print(f"Processing {len(content_list)} content chunks from {len(sources)} sources")

#                 for i, content in enumerate(content_list[:3]):  # Show first 3
#                     print(f"Chunk {i + 1} preview: {content[:200]}...")

#                 print(f"Sources: {sources}")

#             # Join all content with clear separators
#             if content_list:
#                 final_content = "\n\n" + "=" * 50 + "\n\n".join(content_list)
#                 return final_content
#             else:
#                 return "No content could be extracted from the search results."
#         else:
#             return "No relevant information found in any of the cosmic database collections."

#     except Exception as e:
#         error_msg = f"Error searching cosmic database: {e}"
#         if config.DEBUG:
#             print(f"ERROR: {error_msg}")
#             print(f"Traceback: {traceback.format_exc()}")
#         return f"An error occurred while searching the cosmic database: {str(e)}"

#     finally:
#         # Clean up async client
#         try:
#             if 'qdrant_client' in locals():
#                 await qdrant_client.close()
#         except Exception as e:
#             if config.DEBUG:
#                 print(f"Error closing Qdrant client: {e}")



import sys
import openai
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from typing import List
import requests
import config
from typing import List, Generator
from langchain.docstore.document import Document


# load_dotenv()

def get_known_problems_tool(query: str) -> str:
    """Called when the you have to find known problems related to the query."""

    print(f"Retrieving known problems for query: {query}")
    # results = kp_retriever.get_relevant_documents(query, k=2)
    # print(results)
    # results = [result.page_content for result in results]
    # return results if results else "No known problems found."
    return 'Got known problems'


def get_who_am_i_tool(query: str) -> str:
    """Called when the user wants to know information about the bot's identity and capabilities.

    Parameters:
    query (str): The query to search for in the 'Who Am I' database.
    Returns:
    str: The content of the relevant document found, or a message indicating no information was found.

    """

    print(f"Retrieving 'Who Am I' information for query: {query}")
    # print(query)
    # results = wmi_retriever.get_relevant_documents(query, k=2)
    # results = [result.page_content for result in results]
    # return results if results else "No known problems found."
    return 'Got Who Am I'



def check_conversation_completion(numIssueReported: int, numTicketsCreated: int) -> bool:
    """
    When to Call: Call this tool when ticket is successfully created or when the user explicitly states that the issue is resolved.
    Check if the conversation is complete based on reported issues and created tickets.
    Args:
        numIssueReported (int): Number of issues reported by the user.
        numTicketsCreated (int): Number of tickets created by the agent.
    Returns:
        bool: True if the conversation is complete, False otherwise."""
    print(f"Checking conversation completion: {numIssueReported} issues reported, {numTicketsCreated} tickets created.")
    if numIssueReported == numTicketsCreated:
        return True
    else:
        return False


def embed_query_using_ollama_embedding_model(query: str, model_name: str, ollama_url: str):
    """Generate embedding using Nomic model via Ollama"""
    payload = {
        "model": model_name,
        "prompt": query
    }

    try:
        # ollama_url = "http://vs2153.vll.se:11434"
        # ollama_url = "http://localhost:11434"
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]

    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama request failed: {e}")
        raise
    except KeyError as e:
        print(f"❌ Unexpected response format: {e}")
        print(f"Response: {response.text}")
        raise
    except Exception as e:
        print(f"❌ Nomic embedding failed: {e}")
        raise


import httpx
import asyncio
from typing import List, Optional, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from langchain_openai import OpenAIEmbeddings
import traceback
from dotenv import load_dotenv

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
                metadata_filter = intelligent_metadata_filter(query)
                response = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    query_filter=metadata_filter,
                    limit=k,
                    score_threshold=0.3
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
                # Check for early success - if we get enough high-quality results
                high_quality = [hit for hit, _ in result if hit.score > 0.8]
                if len(high_quality) >= k // 2:
                    if config.DEBUG:
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
    Enhanced async cosmic database search with conditional vector database backend.
    
    Uses PostgreSQL/pgvector when config.COSMIC_DATABASE_USE_PGVECTOR is True,
    otherwise uses Qdrant. The collection name and embedding model are read from config.
    This allows for flexible deployment while maintaining the same tool interface.

    Args:
        query (str): The query to search for in the cosmic database.

    Returns:
        str: The content of relevant documents found, or a message indicating no information was found.
    """

    if config.DEBUG:
        vector_db_type = "PostgreSQL/pgvector" if config.COSMIC_DATABASE_USE_PGVECTOR else "Qdrant"
        print(f"Searching cosmic database for query: {query}")
        print(f"Using vector database: {vector_db_type}")
        print(f"Collection: {config.COSMIC_DATABASE_COLLECTION_NAME}")
        print(f"Embedding model: {config.EMBEDDINGS_MODEL_NAME}")

    try:
        # Branch based on configured vector database
        if config.COSMIC_DATABASE_USE_PGVECTOR:
            return await _cosmic_database_pgvector_search(query)
        else:
            return await _cosmic_database_qdrant_search(query)

    except Exception as e:
        error_msg = f"Error searching cosmic database: {e}"
        if config.DEBUG:
            print(f"ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        return f"An error occurred while searching the cosmic database: {str(e)}"


async def _cosmic_database_qdrant_search(query: str) -> str:
    """Search cosmic database using Qdrant backend"""
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
            print(f"Qdrant search - Host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
            print(f"Result limit: {config.QDRANT_RESULT_LIMIT}")

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
            print(f"Qdrant search found {len(results)} results")

        return _process_qdrant_results(results)

    finally:
        # Clean up async client
        try:
            if 'qdrant_client' in locals():
                await qdrant_client.close()
        except Exception as e:
            if config.DEBUG:
                print(f"Error closing Qdrant client: {e}")


async def _cosmic_database_pgvector_search(query: str) -> str:
    """Search cosmic database using PostgreSQL/pgvector backend"""
    from urllib.parse import quote_plus
    from langchain_community.vectorstores import PGVector
    from langchain_community.vectorstores.pgvector import DistanceStrategy
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document
    
    try:
        # Build PostgreSQL connection string
        connection_string = (
            f"postgresql+psycopg2://"
            f"{quote_plus(config.PGVECTOR_USERNAME)}:"
            f"{quote_plus(config.PGVECTOR_PASSWORD)}@"
            f"{config.PGVECTOR_HOST}:"
            f"{config.PGVECTOR_PORT}/"
            f"{config.PGVECTOR_DATABASE}"
        )
        
        # Initialize embeddings based on configured provider
        embeddings = None
        if config.USE_OLLAMA:
            embeddings = OllamaEmbeddings(
                model=config.EMBEDDINGS_MODEL_NAME,
                base_url=config.OLLAMA_BASE_URL
            )
        elif config.USE_OPENAI:
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDINGS_MODEL_NAME,
                api_key=config.OPENAI_API_KEY
            )
        elif config.USE_MISTRAL:
            # For Mistral, fallback to OpenAI embeddings
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDINGS_MODEL_NAME,
                api_key=config.OPENAI_API_KEY
            )
        else:
            raise ValueError("No embedding provider configured. Enable OLLAMA, OPENAI, or MISTRAL in config.")

        if config.DEBUG:
            print(f"PGVector search - Connection: {config.PGVECTOR_HOST}:{config.PGVECTOR_PORT}")
            print(f"Database: {config.PGVECTOR_DATABASE}")
            print(f"Embedding provider: {config.USE_OLLAMA and 'Ollama' or config.USE_OPENAI and 'OpenAI' or 'Mistral'}")

        # Create vector store connection
        vectorstore = PGVector(
            embedding_function=embeddings,
            collection_name=config.COSMIC_DATABASE_COLLECTION_NAME,
            connection_string=connection_string,
            distance_strategy=DistanceStrategy.COSINE
        )
        
        # Perform similarity search
        docs = vectorstore.similarity_search_with_score(
            query, 
            k=config.QDRANT_RESULT_LIMIT  # Use same limit as Qdrant for consistency
        )

        if config.DEBUG:
            print(f"PGVector search found {len(docs)} results")

        return _process_pgvector_results(docs)

    except Exception as e:
        if config.DEBUG:
            print(f"PGVector search error: {e}")
        raise


def _process_qdrant_results(results) -> str:
    """Process Qdrant search results"""
    if not results:
        return "No relevant information found in the cosmic database."

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

    # Join all content with clear separators
    if content_list:
        final_content = "\n\n" + "=" * 50 + "\n\n".join(content_list)
        return final_content
    else:
        return "No content could be extracted from the search results."


def _process_pgvector_results(docs) -> str:
    """Process PostgreSQL/pgvector search results"""
    if not docs:
        return "No relevant information found in the cosmic database."

    content_list = []
    sources = []

    for doc, score in docs:
        if isinstance(doc, Document):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}

            # Get metadata fields
            source_file = metadata.get("source_file", metadata.get("source", "unknown"))
            section = metadata.get("section", "unknown")
            page = metadata.get("page", "unknown")
            context_summary = metadata.get("context_summary", "")

            if content:
                # Track sources
                if source_file not in sources:
                    sources.append(source_file)

                # Build enhanced content with metadata (lower score is better for cosine distance)
                enhanced_content = f"[Source: {source_file} | Section: {section} | Page: {page} | Distance: {score:.3f}]\n{content}"

                if context_summary:
                    enhanced_content += f"\n[Context: {context_summary}]"

                content_list.append(enhanced_content)

    if config.DEBUG:
        print(f"Processing {len(content_list)} content chunks from {len(sources)} sources")

    # Join all content with clear separators
    if content_list:
        final_content = "\n\n" + "=" * 50 + "\n\n".join(content_list)
        return final_content
    else:
        return "No content could be extracted from the search results."