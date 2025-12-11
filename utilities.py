# utilities.py (Qdrant version)
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# from database import CLIENT_NAME
from database import COLLECTION_NAME, qdrant_client_raw

load_dotenv()

# These env vars are kept for compatibility
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")

# Qdrant does not use db_name, but your other code imports it
from database import db_name


def flush_db(collection_name=COLLECTION_NAME):
    """
    Qdrant does NOT require or support explicit flush.
    But to keep compatibility with your existing code,
    this function will simply validate that the collection exists.
    """
    collections = qdrant_client_raw.get_collections().collections
    existing = [c.name for c in collections]

    if collection_name not in existing:
        print(f"✗ Collection '{collection_name}' does NOT exist in Qdrant.")
        return

    print(f"✓ Qdrant collection '{collection_name}' is active (no flush needed).")


def get_collection_details(collection_name, limit=5):
    """
    Returns:
        - schema (vector size & distance)
        - number of points
        - first N records (text + keywords)
    """
    # Ensure collection exists
    collections = qdrant_client_raw.get_collections().collections
    existing = [c.name for c in collections]
    if collection_name not in existing:
        return {"error": f"Collection '{collection_name}' does NOT exist in Qdrant"}

    # Get collection info
    info = qdrant_client_raw.get_collection(collection_name)

    # Count points
    try:
        count = qdrant_client_raw.count(collection_name).count
    except Exception:
        count = "Unknown"

    # Scroll (retrieve first N records)
    points, _ = qdrant_client_raw.scroll(
        collection_name=collection_name,
        limit=limit,
        with_vectors=False,
        with_payload=True,
    )

    # Extract text + keywords if present
    results = []
    for p in points:
        payload = p.payload or {}
        results.append(
            {
                "text": payload.get("text"),
                "keywords": payload.get("keywords"),
                "id": p.id,
            }
        )

    return {
        "schema": {
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance,
        },
        "name": collection_name,
        "number": count,
        "limits": results,
    }
