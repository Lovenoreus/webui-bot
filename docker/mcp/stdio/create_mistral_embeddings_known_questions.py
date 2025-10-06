#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()


# -------------------
# Config
# -------------------
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "hospital_support_questions_mistral_embeddings"

# Mistral AI Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBEDDINGS_MODEL_NAME = "mistral-embed"

if not MISTRAL_API_KEY:
    print("Error: MISTRAL_API_KEY environment variable not set")
    sys.exit(1)


# -------------------
# Embedding Function
# -------------------
def embed_with_mistral(text: str, model_name: str) -> list:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    payload = {
        "model": model_name,
        "input": [text]
    }

    try:
        response = requests.post(MISTRAL_EMBED_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return [0.0] * 1024  # fallback zero-vector (Mistral embed is 1024 dim)


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
    sample_embedding = embed_with_mistral(chunks[0]["text"], EMBEDDINGS_MODEL_NAME)
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
        emb = embed_with_mistral(chunk["text"], EMBEDDINGS_MODEL_NAME)
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
        print("Usage: python load_json_to_qdrant_mistral.py path/to/chunks.json")
        sys.exit(1)
    main(sys.argv[1])
