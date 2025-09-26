#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import config

# -------------------
# Config
# -------------------

# Toggle here: "openai" or "ollama"
if config.USE_OPENAI:
    EMBEDDING_PROVIDER = "openai"
elif config.USE_OLLAMA:
    EMBEDDING_PROVIDER = "ollama"

QDRANT_HOST = config.QDRANT_HOST
QDRANT_PORT = config.QDRANT_PORT
COLLECTION_NAME = config.COSMIC_DATABASE_COLLECTION_NAME

# Models
OPENAI_MODEL = config.EMBEDDINGS_MODEL_NAME  # e.g. "text-embedding-3-large"
OLLAMA_MODEL = config.EMBEDDINGS_MODEL_NAME            # must exist in ollama /api/tags

OLLAMA_HOST = config.OLLAMA_HOST
OLLAMA_PORT = config.OLLAMA_PORT
# -------------------
# Setup
# -------------------
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------
# Embedding Functions
# -------------------
def embed_with_openai(text: str) -> list:
    try:
        response = client_openai.embeddings.create(
            input=text,
            model=OPENAI_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI embedding failed: {e}")
        return [0.0] * 3072   # fallback vector, matches text-embedding-3-large dim


def embed_with_ollama(text: str) -> list:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    print(f"url: {url}, model: {OLLAMA_MODEL}")
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()
        return result["embedding"]
    except Exception as e:
        print(f"Ollama embedding failed: {e}")
        return [0.0] * 768   # fallback vector, matches nomic-embed-text dim


def get_embedding(text: str) -> list:
    if EMBEDDING_PROVIDER == "openai":
        return embed_with_openai(text)
    elif EMBEDDING_PROVIDER == "ollama":
        return embed_with_ollama(text)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


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

    # Get a sample embedding to determine vector dimension
    sample_embedding = get_embedding(chunks[0]["text"])
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
        emb = get_embedding(chunk["text"])
        point = PointStruct(
            id=i,
            vector=emb,
            payload=chunk  # store full chunk JSON as metadata
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
