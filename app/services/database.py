from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.models import Distance

load_dotenv()

q_host = os.getenv("QDRANT_HOST", "localhost")
q_port = int(os.getenv("QDRANT_PORT", 6333))
q_user = os.getenv("QDRANT_USER_NAME")
q_password = os.getenv("QDRANT_PASSWORD")

# Collection and DB settings
COLLECTION_NAME = "LangChainCollection"

# Embedding model (keeps what you had)
EMBED_MODEL_NAME = "mxbai-embed-large"
hf_embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

# Create Qdrant client
# If you use an HTTP API key or cloud, adjust QdrantClient(...) args accordingly.
qdrant_client_raw = QdrantClient(url=f"http://{q_host}:{q_port}")


def _ensure_collection(collection_name: str):
    """
    Ensure a Qdrant collection exists with the right vector size and distance metric.
    We probe the embedding model at runtime to determine vector dimensionality.
    """
    # If collection already exists, do nothing
    collections = qdrant_client_raw.get_collections().collections
    existing_names = [c.name for c in collections]
    if collection_name in existing_names:
        print(f"Qdrant collection '{collection_name}' already exists.")
        return

    # Determine vector size from embeddings by probing one query
    try:
        probe = hf_embeddings.embed_query("probe")
        vector_size = len(probe)
    except Exception as e:
        # Fallback: if embedding probe fails, raise a clear error
        raise RuntimeError(
            "Failed to compute embedding dimension from the embedding model. "
            "Make sure Ollama is running and the EMBED_MODEL_NAME is correct."
        ) from e

    # Create collection with Euclidean (L2) distance to match your previous L2 metric
    print(
        f"Creating Qdrant collection '{collection_name}' with vector size {vector_size}."
    )
    qdrant_client_raw.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,  # Changed from EUCLID
        ),
    )
    print(f"Qdrant collection '{collection_name}' created.")


def delete_collection(collection_name: str):
    existing = [c.name for c in qdrant_client_raw.get_collections().collections]
    if collection_name in existing:
        qdrant_client_raw.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


# Ensure collection exists
try:
    _ensure_collection(COLLECTION_NAME)
except Exception as exc:
    print(f"Error while ensuring Qdrant collection: {exc}")
    raise

# LangChain Qdrant vector store wrapper
vector_store = QdrantVectorStore(
    client=qdrant_client_raw,
    collection_name=COLLECTION_NAME,
    embedding=hf_embeddings,
    distance=Distance.COSINE,
)
