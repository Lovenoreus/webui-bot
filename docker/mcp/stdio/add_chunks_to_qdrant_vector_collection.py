#!/usr/bin/env python3
import os
import sys
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import config

# -------------------
# Config
# -------------------
QDRANT_HOST = 'localhost'
QDRANT_PORT = config.QDRANT_PORT
COLLECTION_NAME = config.COSMIC_DATABASE_COLLECTION_NAME

EMBEDDINGS_MODEL_NAME = config.EMBEDDINGS_MODEL_NAME

# Load environment variables (make sure OPENAI_API_KEY is in .env one level above)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------
# Embedding Function
# -------------------
def embed_with_openai(text: str) -> list:
    try:
        response = client_openai.embeddings.create(
            input=text,
            model=EMBEDDINGS_MODEL_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed: {e}")
        return [0.0] * 3072   # fallback zero-vector, dim of text-embedding-3-large


# -------------------
# Main Logic
# -------------------
def main(json_path: str):
    # Load chunks
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {json_path}")

    # Init Qdrant client
    client_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create / recreate collection
    sample_embedding = embed_with_openai(chunks[0]["text"])
    vector_dim = len(sample_embedding)

    if client_qdrant.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client_qdrant.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' with dim={vector_dim}")
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # Process and upload
    points = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding & Preparing")):
        emb = embed_with_openai(chunk["text"])
        point = PointStruct(
            id=i,
            vector=emb,
            payload=chunk  # store the whole chunk JSON as metadata
        )
        points.append(point)

    print("Uploading to Qdrant...")
    for i in tqdm(range(0, len(points), 100), desc="Qdrant Upload"):
        batch = points[i:i+100]
        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"✅ Finished: {len(points)} chunks uploaded to '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_json_to_qdrant.py path/to/chunks.json")
        sys.exit(1)
    main(sys.argv[1])
