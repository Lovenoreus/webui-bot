#!/usr/bin/env python3
import os
import sys
import json
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# -------------------
# Config
# -------------------
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "hospital_support_questions_openai_embeddings"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL_NAME = "text-embedding-3-large"  # or "text-embedding-3-large" or "text-embedding-ada-002"

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------
# Embedding Function
# -------------------
def embed_with_openai(text: str, model_name: str) -> list:
    try:
        response = openai_client.embeddings.create(
            model=model_name,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed: {e}")
        # Fallback dimensions based on model
        fallback_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        dim = fallback_dims.get(model_name, 1536)
        return [0.0] * dim


# -------------------
# Main Logic
# -------------------
def main(json_path: str):
    # Load chunks
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {json_path}")

    # Init Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create / recreate collection
    print("Getting sample embedding to determine vector dimension...")
    sample_embedding = embed_with_openai(chunks[0]["text"], EMBEDDINGS_MODEL_NAME)
    vector_dim = len(sample_embedding)
    print(f"Vector dimension: {vector_dim}")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' with dim={vector_dim}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # Process and upload with rate limiting
    points = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding & Preparing")):
        emb = embed_with_openai(chunk["text"], EMBEDDINGS_MODEL_NAME)
        point = PointStruct(
            id=i,
            vector=emb,
            payload=chunk  # store the whole chunk JSON as metadata
        )
        points.append(point)

        # Rate limiting - small delay between API calls
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)  # 500ms delay every 10 requests

    print("Uploading to Qdrant...")
    for i in tqdm(range(0, len(points), 100), desc="Qdrant Upload"):
        batch = points[i:i + 100]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"✅ Finished: {len(points)} chunks uploaded to '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_json_to_qdrant_openai.py path/to/chunks.json")
        sys.exit(1)
    main(sys.argv[1])